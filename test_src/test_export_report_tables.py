# -*- coding: utf-8 -*-
"""报告表格导出的回归测试。

重点不是"表格好不好看"，而是两条不能塌的：
① node 桩必须能跑通报告页那种「主体 IIFE + window.xxx 往外挂符号 + 懒加载 tab」的结构；
② 抽出来的每一格必须同时带可见文本和 tooltip（列口径全在 tooltip 里）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from export_report_tables import _TableParser, export


NODE = shutil.which("node")


PAGE = """<!DOCTYPE html>
<html><head><script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script></head>
<body>
<div id="st"></div><div id="rk300Tbl"></div>
<script>
(function(){
const rows = [{n:1, x:'90.5/-54.7', note:'accepted'}, {n:2, x:'—', note:'rejected'}];
document.getElementById('st').innerHTML = '<span>Ball 3D: <span class="v">473</span></span>';
function tableHtml(){
  return '<table><thead><tr><th title="第几抛">#</th>'
    + '<th title="RK@300ms 的 x/z，单位 cm">RK x/z(cm)</th><th>备注</th></tr></thead><tbody>'
    + rows.map(r=>'<tr><td>'+r.n+'</td><td title="同锚：臂最后更新HT">'+r.x+'</td><td>'+r.note+'</td></tr>').join('')
    + '</tbody></table>';
}
const buildPlots = [];
buildPlots[5] = () => { document.getElementById('rk300Tbl').innerHTML = tableHtml(); };
const built = new Set();
function ensurePlot(i){ if(built.has(i)) return; const b=buildPlots[i]; if(typeof b!=='function') return; b(); built.add(i); }
window.ensurePlot = ensurePlot;
window.__rkTimeMap = {scale:1, bias:-11.7505};
window.__dbgAlign = {
  auto:()=>({scale:1,bias:-11.7505,err:0.018,n:166,flights:7,requiredFlights:2,
             windowSource:'pose',margin:null,anchors:14}),
  clockBridge:()=>({bias:null,mad:null,n:0}),
  poseLock:()=>({bias:-11.8197,err:0.0049,n:196,span:0.6,usable:true}),
  clockAnchor:()=>({scale:1,bias:-11.854,anchors:14,mad:0.0457}),
};
})();
function sw(i){
  ensurePlot(i);
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  window.dispatchEvent(new Event('resize'));
}
sw(0);
</script>
</body></html>
"""


def test_table_parser_keeps_text_and_tooltip():
    parser = _TableParser()
    parser.feed(
        '<table><thead><tr><th title="第几抛">#</th><th>x/z</th></tr></thead>'
        '<tbody><tr><td>1</td><td title="同锚：臂最后更新HT">90.5/-54.7</td></tr></tbody></table>'
    )
    assert len(parser.tables) == 1
    table = parser.tables[0]
    assert table["headers"] == ["#", "x/z"]
    assert table["header_titles"][0] == "第几抛"
    assert table["rows"] == [["1", "90.5/-54.7"]]
    assert table["titles"][0][1] == "同锚：臂最后更新HT"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_export_runs_the_page_and_writes_both_products(tmp_path: Path):
    html_path = tmp_path / "tracker_fake.html"
    html_path.write_text(PAGE, encoding="utf-8")

    md_path, json_path = export(html_path)
    assert md_path.exists() and json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["script_error"] is None, payload["script_error"]
    assert payload["tab_errors"] == []
    # 懒加载的表（只有 sw(5) 才会建）必须被逼出来
    table = payload["sections"]["rk300Tbl"]["tables"][0]
    assert table["headers"] == ["#", "RK x/z(cm)", "备注"]
    assert table["rows"][0] == ["1", "90.5/-54.7", "accepted"]
    assert table["header_titles"][1].startswith("RK@300ms")

    text = md_path.read_text(encoding="utf-8")
    assert "✓ 对齐可信" in text
    assert "PC t = RK t + -11.7505 s（scale 固定为 1）" in text
    assert "| 1 | 90.5/-54.7 | accepted |" in text
    assert "Ball 3D: 473" in text


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_export_flags_untrustworthy_alignment(tmp_path: Path):
    bad = PAGE.replace(
        "err:0.018,n:166,flights:7,requiredFlights:2,\n             windowSource:'pose'",
        "err:0.289,n:269,flights:7,requiredFlights:3,\n             windowSource:'scan'",
    )
    assert bad != PAGE
    html_path = tmp_path / "tracker_bad.html"
    html_path.write_text(bad, encoding="utf-8")
    md_path, _ = export(html_path)
    assert "⚠ 对齐不可信" in md_path.read_text(encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_export_flags_legacy_non_unit_scale(tmp_path: Path):
    old = PAGE.replace("{scale:1, bias:-11.7505}", "{scale:0.9992, bias:-11.7505}")
    assert old != PAGE
    html_path = tmp_path / "tracker_old_scale.html"
    html_path.write_text(old, encoding="utf-8")
    md_path, _ = export(html_path)
    assert "旧版报告使用非 1 scale" in md_path.read_text(encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_export_does_not_guess_on_pre_0812_reports(tmp_path: Path):
    """旧版 HTML 的 __dbgAlign 字段名不同（off 而非 bias），不能硬套新口径下结论。"""
    old = PAGE.replace(
        "auto:()=>({scale:1,bias:-11.7505,err:0.018,n:166,flights:7,requiredFlights:2,\n"
        "             windowSource:'pose',margin:null,anchors:14})",
        "auto:()=>({off:2.7796,err:0.0141,n:69,flights:3})",
    )
    assert old != PAGE
    html_path = tmp_path / "tracker_old.html"
    html_path.write_text(old, encoding="utf-8")
    md_path, _ = export(html_path)
    text = md_path.read_text(encoding="utf-8")
    assert "结论: 未知" in text
    assert "旧版报告代码" in text
