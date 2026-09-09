# -*- coding: utf-8 -*-
"""把生成好的报告 HTML 里的表格导出成给人/AI 直接读的文本 + JSON。

动机：报告页是 15~50MB 的单文件 HTML，表格全是运行时 JS 生成的——想看几个数就得起
浏览器/HTTP 服务。这里用 `report_page_snapshot.js` 在 node 里把**同一份页面脚本**跑
一遍（最小 DOM 桩），把每个 innerHTML 快照下来再转成表格，所以数值和口径不可能与页面分叉。

产物（与 HTML 同目录同前缀）：
  <run>_tables.md    —— 对齐结论 + 概览 + 每张表（纯文本，可直接 Read/grep）
  <run>_tables.json  —— 同样内容的结构化版，外加每格的 tooltip（列口径都在里面）

单独用：
  python test_src/export_report_tables.py --html tracker_output/<run>/<run>.html
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path


_SNAPSHOT_JS = Path(__file__).resolve().parent / "report_page_snapshot.js"

# 页面里各容器的中文名；没列到的按 id 原样输出
_SECTION_TITLES = {
    "st": "概览",
    "rk300Tbl": "北极星表（Pre300HT / FinalHT 预测 / PC 真值 / 臂执行）",
    "armEv": "Arm Accepted（臂受理与触球）",
    "rkInfo": "RK Car Move 轴说明",
    "rkSigInfo": "RK Signals 轴说明",
    "mvNote": "RK Car Move 备注",
}


class _TableParser(HTMLParser):
    """把 innerHTML 片段里的 <table> 抽成 {headers, rows, titles}。

    每格同时保留可见文本和最近一层 title 属性——报告把「这一列到底是什么口径、
    为什么是 —」全写在 tooltip 里，那是这张表最值钱的部分。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self._table: dict | None = None
        self._row: list[str] | None = None
        self._row_titles: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_title: list[str] = []
        self._title_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_map = dict(attrs)
        title = attrs_map.get("title")
        if tag == "table":
            self._table = {"headers": [], "rows": [], "titles": [], "header_titles": []}
        elif tag == "tr" and self._table is not None:
            self._row, self._row_titles = [], []
        elif tag in ("td", "th") and self._table is not None:
            self._cell, self._cell_title = [], []
            self._title_stack = []
        if title:
            self._title_stack.append(title)
            if self._cell is not None:
                self._cell_title.append(title)
        if tag == "br" and self._cell is not None:
            self._cell.append(" / ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._table is not None and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            title = " || ".join(dict.fromkeys(self._cell_title))
            if self._row is not None:
                self._row.append(text)
                self._row_titles.append(title)
            self._cell = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            if not self._table["headers"] and not self._table["rows"]:
                self._table["headers"] = self._row
                self._table["header_titles"] = self._row_titles
            else:
                self._table["rows"].append(self._row)
                self._table["titles"].append(self._row_titles)
            self._row, self._row_titles = None, None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _strip_tags(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", " / ", fragment)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", html_mod.unescape(text)).strip()


def _plain_lines(fragment: str) -> list[str]:
    """没有 <table> 的片段（概览/横幅/说明）：按 <div>/<span> 边界拆成行。"""
    text = re.sub(r"</(div|p|li|tr)>", "\n", fragment)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</span>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    out = []
    for line in html_mod.unescape(text).splitlines():
        line = re.sub(r"\s+", " ", line).strip(" |").strip()
        if line:
            out.append(line)
    return out


def _md_table(table: dict) -> list[str]:
    headers = [h.replace("|", "/") or " " for h in table["headers"]]
    if not headers:
        return []
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in table["rows"]:
        cells = [(c or "—").replace("|", "/") for c in row]
        cells += [" "] * (len(headers) - len(cells))
        lines.append("| " + " | ".join(cells[:len(headers)]) + " |")
    return lines


def snapshot(html_path: Path, *, node: str | None = None) -> dict:
    """跑一遍页面脚本，拿回 {order, captured, align, error, tabErrors}。"""
    node = node or shutil.which("node")
    if not node:
        raise RuntimeError("node 不在 PATH 上，无法导出表格")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "snapshot.json"
        proc = subprocess.run(
            [node, "--max-old-space-size=8192", str(_SNAPSHOT_JS), str(html_path), str(out)],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"report_page_snapshot.js 失败: {proc.stderr[-500:]}")
        return json.loads(out.read_text(encoding="utf-8"))


def _align_lines(align: dict) -> list[str]:
    auto = align.get("auto") or {}
    time_map = align.get("timeMap") or {"scale": auto.get("scale"), "bias": auto.get("bias")}
    bridge = align.get("clockBridge") or {}
    pose = align.get("poseLock") or {}
    anchor = align.get("clockAnchor") or {}
    fmt = lambda v, n=4: ("n/a" if not isinstance(v, (int, float)) else f"{v:.{n}f}")
    if "windowSource" not in auto:
        # 2026-08-12 之前的报告代码生成的 HTML：字段名都不一样，别硬套新口径下结论
        return [
            "- 结论: 未知——这份 HTML 由旧版报告代码生成，缺少对齐质量字段。"
            "重新跑一次 `generate_curve3_html.py --input <run>.json` 即可拿到完整判定。",
            f"- 旧版字段原样: {json.dumps(auto, ensure_ascii=False)}",
        ]
    non_unit_scale = (
        isinstance(time_map.get("scale"), (int, float))
        and abs(time_map["scale"] - 1.0) > 1e-12
    )
    lines = [
        f"- 时间映射: PC t = RK t + {fmt(time_map.get('bias'))} s（scale 固定为 1）",
        f"- 粗定位来源: {auto.get('windowSource')}"
        f"（时钟桥 bias={fmt(bridge.get('bias'))}/n={bridge.get('n')}"
        f"；位姿形状锁 bias={fmt(pose.get('bias'))}/残差={fmt(pose.get('err'))}/usable={pose.get('usable')}"
        f"；精确值锚 {anchor.get('anchors')} 条 bias={fmt(anchor.get('bias'))}）",
        f"- z 形状精锁: err={fmt(auto.get('err'))} m / {auto.get('n')} 点 / {auto.get('flights')} 抛"
        f"（需 ≤0.08m / ≥{30 if auto.get('windowSource') == 'scan' else 15} 点 /"
        f" ≥{auto.get('requiredFlights')} 抛）",
    ]
    if auto.get("margin") is not None:
        lines.append(f"- 全场扫描混叠余量 margin={fmt(auto.get('margin'), 2)}×（需 ≥1.35）")
    bad = (
        non_unit_scale
        or
        auto.get("bias") is None or auto.get("err") is None or auto.get("err") > 0.08
        or (auto.get("n") or 0) < (30 if auto.get("windowSource") == "scan" else 15)
        or (auto.get("flights") or 0) < (auto.get("requiredFlights") or 3)
        or (auto.get("windowSource") == "scan" and auto.get("margin") is not None
            and auto["margin"] < 1.35)
    )
    if non_unit_scale:
        lines.insert(0, "- 结论: ⚠ 旧版报告使用非 1 scale，必须重新生成后再用")
    else:
        lines.insert(0, "- 结论: " + ("⚠ 对齐不可信，本文件里所有跨轴数值都不要用"
                                     if bad else "✓ 对齐可信"))
    return lines


def export(html_path: Path, *, node: str | None = None) -> tuple[Path, Path]:
    snap = snapshot(html_path, node=node)
    base = html_path.with_suffix("")
    md_path = Path(str(base) + "_tables.md")
    json_path = Path(str(base) + "_tables.json")

    sections: dict[str, dict] = {}
    md: list[str] = [f"# {html_path.stem} 报告数据表", "",
                     "> 由 export_report_tables.py 从同名 HTML 直接快照生成（跑的是页面自己的 JS，"
                     "数值与页面严格一致）。列口径见 `_tables.json` 里每格的 title。", ""]
    md += ["## PC↔RK 对齐", ""] + _align_lines(snap.get("align") or {}) + [""]
    if snap.get("error"):
        md += ["> ⚠ 页面脚本执行报错，下面的表可能不全：",
               "> `" + snap["error"].split("\n")[0] + "`", ""]

    for element_id in snap.get("order", []):
        fragment = snap["captured"][element_id]
        parser = _TableParser()
        parser.feed(fragment)
        title = _SECTION_TITLES.get(element_id, element_id)
        section: dict = {"id": element_id, "title": title, "tables": parser.tables}
        md.append(f"## {title}")
        md.append("")
        if parser.tables:
            # 表前的横幅/说明文字（对齐警告等）也要留着
            head = _plain_lines(fragment.split("<table", 1)[0])
            if head:
                md += head + [""]
            for table in parser.tables:
                md += _md_table(table) + [""]
        else:
            lines = _plain_lines(fragment)
            section["lines"] = lines
            md += lines + [""]
        sections[element_id] = section

    md_path.write_text("\n".join(md), encoding="utf-8")
    json_path.write_text(json.dumps({
        "source_html": html_path.name,
        "align": snap.get("align"),
        "arm_contract": snap.get("armContract"),
        "sections": sections,
        "script_error": snap.get("error"),
        "tab_errors": snap.get("tabErrors"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return md_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True, help="已生成的报告 HTML")
    args = parser.parse_args()
    md_path, json_path = export(args.html)
    print(f"tables saved: {md_path}")
    print(f"tables saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
