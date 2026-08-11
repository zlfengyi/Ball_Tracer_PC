# -*- coding: utf-8 -*-
"""回归测试：报告端车位姿取值（carAt）在单 tag 退化帧上的降级行为。

单 tag 帧只有 x/y、yaw 为 null（车上一块 tag 被臂座挡住时的常态，见
src/car_localizer.py 的退化路径）。位置插值必须照用这些行，yaw 插值只能用
带 yaw 的行——不能让一个 null yaw 把好好的位置一起废掉（那就等于退回改动前
整帧作废的行为，PC 真值列还是空）。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
NODE = shutil.which("node")


def _car_at_core_js() -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        r"// \[\[car-at-core-begin\]\].*?\n(.*)// \[\[car-at-core-end\]\]",
        text,
        re.S,
    )
    assert match, "generate_curve3_html.py 缺 [[car-at-core-begin/end]] 标记"
    return match.group(1)


def _run(tmp_path: Path, rows: list[dict], calls: list[float]) -> list[dict | None]:
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const lerp=(a,b,f)=>a+(b-a)*f;\n"
        "const interpRow=(rows,t,maxGap)=>{\n"
        "  if(!rows.length) return null;\n"
        "  let lo=0,hi=rows.length;\n"
        "  while(lo<hi){const mid=(lo+hi)>>1; if(rows[mid].t<t) lo=mid+1; else hi=mid;}\n"
        "  if(lo<=0||lo>=rows.length) return null;\n"
        "  const a=rows[lo-1],b=rows[lo];\n"
        "  if(!(t>=a.t&&t<=b.t)||b.t-a.t>maxGap) return null;\n"
        "  return {a,b,f:(t-a.t)/Math.max(1e-9,b.t-a.t)};\n"
        "};\n"
        f"const pcCarRows={json.dumps(rows)};\n"
        "const pcCarYawRows=pcCarRows.filter(p=>isNum(p.yaw));\n"
        f"{_car_at_core_js()}\n"
        f"console.log(JSON.stringify({json.dumps(calls)}.map(t=>carAt(t))));\n"
    )
    script = tmp_path / "car_at_harness.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _row(t, x, y, yaw=None, single=False):
    return {"t": t, "x": x, "y": y, "z": 0.0, "yaw": yaw, "single": single}


# 双 tag → 一段单 tag → 双 tag：中间三帧只有位置
ROWS = [
    _row(0.00, 0.00, 2.00, yaw=0.10),
    _row(0.07, 0.07, 2.00, yaw=0.10),
    _row(0.14, 0.14, 2.00, single=True),
    _row(0.21, 0.21, 2.00, single=True),
    _row(0.28, 0.28, 2.00, single=True),
    _row(0.35, 0.35, 2.00, yaw=0.20),
]


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_position_survives_single_tag_rows(tmp_path):
    (got,) = _run(tmp_path, ROWS, [0.175])
    assert got is not None
    assert got["x"] == pytest.approx(0.175, abs=1e-9)     # 位置照插
    assert got["y"] == pytest.approx(2.0, abs=1e-9)
    assert got["ga"] == pytest.approx(0.035, abs=1e-9)    # 夹住的是相邻两条单 tag 行
    assert got["single"] is True                          # 并且如实标出来


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_yaw_interpolates_across_the_single_tag_stretch(tmp_path):
    """yaw 从 0.07 直插到 0.35（跨 0.28s ≤ 0.5s 上限），中间的 null 行跳过。"""
    (got,) = _run(tmp_path, ROWS, [0.21])
    assert got is not None
    assert got["yaw"] == pytest.approx(0.10 + 0.10 * (0.21 - 0.07) / 0.28, abs=1e-9)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_yaw_is_null_when_the_yaw_gap_exceeds_the_cap(tmp_path):
    """单 tag 段长到两侧 yaw 行间隔 >0.5s：位置仍给，yaw 报 null——只让依赖
    yaw 的列缺失，世界轴的 PC 真值列照常出。"""
    rows = [
        _row(0.0, 0.0, 2.0, yaw=0.1),
        _row(0.2, 0.2, 2.0, single=True),
        _row(0.4, 0.4, 2.0, single=True),
        _row(0.6, 0.6, 2.0, single=True),
        _row(0.8, 0.8, 2.0, yaw=0.3),      # 与 t=0.0 相隔 0.8s > 0.5s
    ]
    (got,) = _run(tmp_path, rows, [0.3])
    assert got is not None
    assert got["x"] == pytest.approx(0.3, abs=1e-9)
    assert got["yaw"] is None
    assert got["single"] is True


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_position_gap_cap_still_applies(tmp_path):
    """位置侧的 0.5s 夹住上限没被放宽：空洞照旧整格拒。"""
    rows = [_row(0.0, 0.0, 2.0, yaw=0.1), _row(0.7, 0.7, 2.0, yaw=0.1)]
    assert _run(tmp_path, rows, [0.35]) == [None]


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_dual_tag_only_rows_are_unchanged(tmp_path):
    """全是双 tag 行时结果与改动前逐位一致（老场次报告不能变）。"""
    rows = [_row(0.0, 0.0, 2.0, yaw=0.1), _row(0.1, 0.2, 2.4, yaw=0.3)]
    (got,) = _run(tmp_path, rows, [0.05])
    assert got["x"] == pytest.approx(0.1, abs=1e-12)
    assert got["y"] == pytest.approx(2.2, abs=1e-12)
    assert got["yaw"] == pytest.approx(0.2, abs=1e-12)
    assert got["single"] is False
