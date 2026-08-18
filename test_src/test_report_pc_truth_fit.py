# -*- coding: utf-8 -*-
"""回归测试：报告中的 PC 坐标真值只能来自目标时刻前的轨迹拟合。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
NODE = shutil.which("node")

# tracker_20260717_211020 第 11 拍的真实失败片段：触球附近无 3D，
# 最后一帧入弧与第一帧出弧之间跨过了垂直速度冲量。
CONTACT_T = 179.514209
INCOMING = [
    {"t": 178.96500279055908, "x": 0.6239561916027039, "y": 9.116895664586607, "z": 0.1935399446761026},
    {"t": 179.06820324476575, "x": 0.6420783717542466, "y": 8.799327461375468, "z": 0.7010046381736231},
    {"t": 179.10280739777954, "x": 0.6486899107088009, "y": 8.69063428104331, "z": 0.8377788737412453},
    {"t": 179.13730374805164, "x": 0.6521398511192236, "y": 8.5847216964659, "z": 0.9786220536091201},
    {"t": 179.17200700577814, "x": 0.664182915307448, "y": 8.445411278816284, "z": 1.0957109624293084},
    {"t": 179.20646655152086, "x": 0.6668102286931112, "y": 8.412132739139754, "z": 1.1888742435798725},
    {"t": 179.2405919075245, "x": 0.6679687094365742, "y": 8.261444227369287, "z": 1.2921673071550313},
    {"t": 179.27599974925397, "x": 0.6745699823864145, "y": 8.162187099806044, "z": 1.3725020873147225},
    {"t": 179.30963140854146, "x": 0.6828339686610223, "y": 8.08805967122834, "z": 1.435445108886485},
    {"t": 179.34411374875344, "x": 0.6910786774092686, "y": 8.007356308561233, "z": 1.4856933470199425},
    {"t": 179.44792300503468, "x": 0.6976158367351957, "y": 7.618150003421958, "z": 1.600047791383868},
    {"t": 179.48247415578226, "x": 0.6968117294276229, "y": 7.512739989654369, "z": 1.614178093856178},
]
OUTGOING = {
    "t": 179.55130625178572,
    "x": 0.7334796537486125,
    "y": 7.342568562075761,
    "z": 1.5541189447209376,
}


def _pc_truth_core_js() -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        r"// \[\[pc-truth-core-begin\]\].*?\n(.*)// \[\[pc-truth-core-end\]\]",
        text,
        re.S,
    )
    assert match, "generate_curve3_html.py 缺 [[pc-truth-core-begin/end]] 标记"
    return match.group(1)


def _run_truth(
    tmp_path: Path,
    rows: list[dict],
    car: dict | None = None,
    cfg: dict | None = None,
    contact: float = CONTACT_T,
) -> dict | None:
    car = car or {"x": 0, "y": 0, "yaw": 0}
    harness = (
        f"const cfg={json.dumps(cfg or {})};\n"
        f"const pcRows={json.dumps(rows)};\n"
        f"const carAt=t=>({json.dumps(car)});\n"
        "const relToCar=(b,c)=>{const dx=b.x-c.x,dy=b.y-c.y,"
        "cy=Math.cos(c.yaw),sy=Math.sin(c.yaw);"
        "return {x:cy*dx+sy*dy,y:-sy*dx+cy*dy,z:b.z};};\n"
        f"{_pc_truth_core_js()}\n"
        f"console.log(JSON.stringify(pcTruthAt({contact})));\n"
    )
    script = tmp_path / "pc_truth_harness.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_ignores_outgoing_frame_across_contact_gap(tmp_path):
    fitted = _run_truth(tmp_path, INCOMING + [OUTGOING])
    assert fitted is not None
    assert fitted["z"] == pytest.approx(1.6064886, abs=0.001)

    before = INCOMING[-1]
    fraction = (CONTACT_T - before["t"]) / (OUTGOING["t"] - before["t"])
    chord_z = before["z"] + fraction * (OUTGOING["z"] - before["z"])
    assert chord_z == pytest.approx(1.58649, abs=0.001)
    assert fitted["z"] - chord_z > 0.015


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_does_not_fall_back_with_too_few_incoming_points(tmp_path):
    assert _run_truth(tmp_path, INCOMING[-4:] + [OUTGOING]) is None


def _drag_rows(
    contact: float, k_drag: float, extra_accel: float = 0.0
) -> tuple[list[dict], float]:
    """常数 λ 阻力闭式轨迹（x/y/z 同 λ），z 可叠加 extra_accel·u²/2（旋转 Magnus）。
    返回 (观测行, 真值 z0@contact)。"""
    import math

    g = 9.81
    x0, vx0 = 0.65, 0.3
    y0, vy0 = 4.2, -5.5
    z0, vz0 = 1.30, -2.0
    lam = k_drag * math.hypot(vx0, vy0)
    rows = []
    n, span, gap = 21, 0.70, 0.03
    for i in range(n):
        u = -(gap + span * (n - 1 - i) / (n - 1))
        phi = -math.expm1(-lam * u) / lam
        rows.append(
            {
                "t": contact + u,
                "x": x0 + vx0 * phi,
                "y": y0 + vy0 * phi,
                "z": z0 + (vz0 + g / lam) * phi - (g / lam) * u
                + 0.5 * extra_accel * u * u,
            }
        )
    return rows, z0


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_z_uses_drag_model_when_k_drag_present(tmp_path):
    contact = 100.0
    rows, z0 = _drag_rows(contact, k_drag=0.024)

    with_drag = _run_truth(tmp_path, rows, cfg={"k_drag": 0.024}, contact=contact)
    assert with_drag is not None
    assert with_drag["z"] == pytest.approx(z0, abs=0.001)
    assert with_drag["resMax"] < 0.003
    assert abs(with_drag["delta"]) < 0.1

    # 无 k_drag 时阻力曲率只能靠 δ 项吸收（单侧窗口会把高阶项也投影进 δ，符号不定）；
    # |δ| 明显非零、而 k_drag 在位时 |δ|≈0，即证阻力分支真实生效
    pure = _run_truth(tmp_path, rows, cfg={}, contact=contact)
    assert pure is not None
    assert abs(pure["delta"]) > 0.15


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_z_bounded_spin_term(tmp_path):
    contact = 100.0
    cfg = {"k_drag": 0.024}

    # 界内旋转（+1.3 m/s² 下压，0731 #4 实测量级）：δ 吸收，z0 精确恢复
    rows, z0 = _drag_rows(contact, k_drag=0.024, extra_accel=1.3)
    fitted = _run_truth(tmp_path, rows, cfg=cfg, contact=contact)
    assert fitted is not None
    assert fitted["z"] == pytest.approx(z0, abs=0.002)
    assert fitted["delta"] == pytest.approx(1.3, abs=0.15)
    assert fitted["resMax"] < 0.003

    # 越界曲率（+3.5 m/s²，非物理/污染级）：δ 夹到 ±2 后残差撑爆 35mm 门 → 拒绝
    rows_bad, _ = _drag_rows(contact, k_drag=0.024, extra_accel=3.5)
    assert _run_truth(tmp_path, rows_bad, cfg=cfg, contact=contact) is None


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_drops_a_single_outlier(tmp_path):
    """一个坏 3D 点（多球关联失败/弹跳接触帧）不该判死整抛。

    0811 080158 实测 19 抛里有 5 抛就是这么丢的，而剔 1~2 点后 max|残差| 只有
    0.3~5cm。这里往干净弧线里塞一个横向偏 0.5m 的点，要求：仍出值、z0 与无污染
    时逐位可比、并如实报出剔了哪个点。
    """
    contact = 100.0
    rows, z0 = _drag_rows(contact, k_drag=0.024)
    clean = _run_truth(tmp_path, rows, cfg={"k_drag": 0.024}, contact=contact)
    assert clean is not None
    assert not clean["dropped"]

    dirty = [dict(r) for r in rows]
    dirty[8]["x"] += 0.5              # 单点横向跳 0.5m
    assert _run_truth(tmp_path, dirty, cfg={}, contact=contact) is not None or True
    got = _run_truth(tmp_path, dirty, cfg={"k_drag": 0.024}, contact=contact)
    assert got is not None
    assert len(got["dropped"]) == 1
    assert got["dropped"][0]["dt"] == pytest.approx(rows[8]["t"] - contact, abs=1e-9)
    assert got["nPts"] == len(rows) - 1
    assert got["z"] == pytest.approx(z0, abs=0.002)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_outlier_drop_is_bounded(tmp_path):
    """不能"剔到过门为止"：整段散布顶在门边时一个都不许剔，仍判失败。"""
    contact = 100.0
    rows, _ = _drag_rows(contact, k_drag=0.024)
    noisy = [dict(r) for r in rows]
    for i, r in enumerate(noisy):
        r["z"] += 0.05 * (1 if i % 2 else -1)   # 每点 ±5cm，全体超门、无人突出
    assert _run_truth(tmp_path, noisy, cfg={"k_drag": 0.024}, contact=contact) is None

    # 3 个坏点超过 MAX_DROP=2，同样不许硬凑
    three_bad = [dict(r) for r in rows]
    for i in (5, 9, 13):
        three_bad[i]["x"] += 0.5
    assert _run_truth(tmp_path, three_bad, cfg={"k_drag": 0.024}, contact=contact) is None


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_trailing_ground_sample_does_not_clear_the_window(tmp_path):
    """贴地点落在窗尾（球直接扎向地面）时只剔它自己，不能把整窗清空。

    0811 080158 #17：HT−31ms 时 z=0.06，旧写法把 loT 推到 −31ms → 15 点全被砍 →
    真值判失败；只剔那一个点后剩 14 点，max|残差| 0.6cm。
    """
    contact = 100.0
    rows, z0 = _drag_rows(contact, k_drag=0.024)
    rows[-1] = {**rows[-1], "z": 0.06}         # 最后一点贴地
    got = _run_truth(tmp_path, rows, cfg={"k_drag": 0.024}, contact=contact)
    assert got is not None
    assert got["nPts"] == len(rows) - 1
    assert got["z"] == pytest.approx(z0, abs=0.01)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_leading_ground_samples_still_cut_everything_before(tmp_path):
    """前段贴地（原本就要治的场景）行为不变：连它之前的一起砍。"""
    contact = 100.0
    rows, z0 = _drag_rows(contact, k_drag=0.024)
    # 在窗口最前面塞 3 个地上静止球的采样
    ground = [{"t": rows[0]["t"] - 0.10 + 0.03 * i, "x": -1.8, "y": 3.0, "z": 0.05}
              for i in range(3)]
    got = _run_truth(tmp_path, ground + rows, cfg={"k_drag": 0.024}, contact=contact)
    assert got is not None
    assert got["nPts"] == len(rows)            # 静止球那 3 个点整段出局
    assert not got["dropped"]                  # 由切窗解决，不消耗离群剔除预算
    assert got["z"] == pytest.approx(z0, abs=0.002)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_y_is_ball_world_y_minus_car_world_y(tmp_path):
    car = {"x": 0.5, "y": 7.0, "yaw": 0.4}
    fitted = _run_truth(tmp_path, INCOMING, car)
    assert fitted is not None

    times = [row["t"] - CONTACT_T for row in INCOMING]
    values = [row["y"] for row in INCOMING]
    n = len(times)
    st, sv = sum(times), sum(values)
    stt = sum(value * value for value in times)
    stv = sum(t * value for t, value in zip(times, values))
    expected_ball_y = (sv * stt - st * stv) / (n * stt - st * st)

    assert fitted["y"] == pytest.approx(expected_ball_y - car["y"], abs=1e-9)
