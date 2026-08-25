# -*- coding: utf-8 -*-
"""回归测试：All-in-One 表 "PC回球 yaw/俯仰" 列的 [[pc-return-core]] 统计。

合成一次抛球+回球（含重力、观测噪声、触球遮挡断档），验证：
  1. 干净回球给出正确来/回球三维速度、yaw/俯仰/速率，且对锚点 ±80ms 偏差鲁棒；
  2. 挥空（y 不反向）不认定回球；
  3. 贴地静止球观测（z<0.12）不污染结果；
  4. 出弧断档后检测器接上别的球（>12m/s 跳变）被截断，方向仍来自真实首段。
出弧三道防污染门（0813_083521 抛28 定案）：
  5. 断档后隐含速度 <20m/s 的野点（跳变门放行）被轨迹一致性门切掉，方向仍来自真弧；
  6. 深度噪声乱拟合（残差爆表）拒绝出数，不给假方向；
  7. 段首在地面反弹之后的段（下压球+遮挡）拒绝倒推，不给假仰角。
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
NODE = shutil.which("node")

T_HIT = 100.0
HIT_POS = (0.6, 0.4, 0.5)
V_IN = (-0.05, -4.5, -1.0)
V_OUT = (0.8, 5.5, 1.2)
G = 9.81


def _synth_rows(
    *,
    with_return: bool = True,
    ground_clutter: bool = False,
    swap_ball_after: float | None = None,
    occlusion_gap: tuple[float, float] | None = None,
    alien_after_gap: tuple[float, float, tuple[float, float, float]] | None = None,
    y_scramble: float = 0.0,
    out_cadence: float = 0.01,
    out_until: float = 0.45,
    noise: float = 0.002,
    seed: int = 7,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    def add(t: float, x: float, y: float, z: float) -> None:
        rows.append({
            "t": round(t, 5),
            "x": round(x + rng.gauss(0.0, noise), 5),
            "y": round(y + rng.gauss(0.0, noise), 5),
            "z": round(z + rng.gauss(0.0, noise), 5),
        })

    # 入弧：触球前 0.5s，触球邻域 [−30,+40]ms 遮挡断档
    t = T_HIT - 0.5
    while t < T_HIT - 0.03:
        u = t - T_HIT
        add(
            t,
            HIT_POS[0] + V_IN[0] * u,
            HIT_POS[1] + V_IN[1] * u,
            HIT_POS[2] + V_IN[2] * u - 0.5 * G * u * u,
        )
        t += 0.01

    t = T_HIT + 0.04
    frame = 0
    while t < T_HIT + out_until:
        u = t - T_HIT
        if occlusion_gap is not None and occlusion_gap[0] <= u <= occlusion_gap[1]:
            t += out_cadence
            continue
        if alien_after_gap is not None and u >= alien_after_gap[0]:
            # 真回球被遮挡断档，检测器接上场上另一颗静止球（断档 <150ms、隐含速度 <20m/s：
            # 两道旧门都放行，只有轨迹一致性门能切）
            if u < alien_after_gap[1]:
                t += out_cadence
                continue
            add(t, *alien_after_gap[2])
            t += out_cadence
            frame += 1
            continue
        if with_return:
            x = HIT_POS[0] + V_OUT[0] * u
            y = HIT_POS[1] + V_OUT[1] * u
            z = HIT_POS[2] + V_OUT[2] * u - 0.5 * G * u * u
        else:  # 挥空：球按原速穿过
            x = HIT_POS[0] + V_IN[0] * u
            y = HIT_POS[1] + V_IN[1] * u
            z = max(0.03, HIT_POS[2] + V_IN[2] * u - 0.5 * G * u * u)
        if swap_ball_after is not None and u > swap_ball_after:
            x, y, z = 5.0 + 0.2 * u, 8.0, 1.0  # 检测器接上场上另一颗球
        if y_scramble:
            y += y_scramble if frame % 2 else -y_scramble  # 深度三角测量逐帧摆动
        add(t, x, y, z)
        t += out_cadence
        frame += 1

    if ground_clutter:  # 场上贴地静止球
        t = T_HIT - 0.5
        while t < T_HIT + 0.45:
            add(t, 1.5, 1.0, 0.05)
            t += 0.01

    rows.sort(key=lambda r: r["t"])
    return rows


def _pc_return_core_js() -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        r"// \[\[pc-return-core-begin\]\].*?\n(.*)// \[\[pc-return-core-end\]\]",
        text,
        re.S,
    )
    assert match, "generate_curve3_html.py 缺 [[pc-return-core-begin/end]] 标记"
    return match.group(1)


def _solve3_js() -> str:
    """从 [[pc-truth-core]] 里取真身 solve3，避免测试桩与线上实现分叉。"""
    text = SRC.read_text(encoding="utf-8")
    match = re.search(r"(const solve3=.*?\n};)", text, re.S)
    assert match, "generate_curve3_html.py 缺 solve3 定义"
    return match.group(1)


def _run_expr(tmp_path: Path, rows: list[dict], expr: str):
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        f"{_solve3_js()}\n"
        f"const pcRows={json.dumps(rows)};\n"
        f"{_pc_return_core_js()}\n"
        f"console.log(JSON.stringify({expr}));\n"
    )
    script = tmp_path / "pc_return_harness.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _run_return(tmp_path: Path, rows: list[dict], anchor: float) -> dict | None:
    return _run_expr(tmp_path, rows, f"pcReturnAt({anchor})")


YAW_TRUTH = math.degrees(math.atan2(V_OUT[0], V_OUT[1]))
PITCH_TRUTH = math.degrees(math.atan2(V_OUT[2], math.hypot(V_OUT[0], V_OUT[1])))
SPEED_TRUTH = math.hypot(*V_OUT)


@pytest.mark.skipif(NODE is None, reason="需要 node")
@pytest.mark.parametrize("anchor_offset", [0.0, -0.08, 0.08])
def test_clean_return_direction(tmp_path: Path, anchor_offset: float) -> None:
    rows = _synth_rows()
    ret = _run_return(tmp_path, rows, T_HIT + anchor_offset)
    assert ret is not None
    assert abs(ret["tHit"] - T_HIT) < 0.02
    assert abs(ret["yaw"] - YAW_TRUTH) < 2.0
    assert abs(ret["pitch"] - PITCH_TRUTH) < 2.5
    assert abs(ret["speed"] - SPEED_TRUTH) < 0.4
    assert ret["n"] >= 5
    incoming = ret["incoming"]
    assert incoming is not None
    assert incoming["vx"] == pytest.approx(V_IN[0], abs=0.25)
    assert incoming["vy"] == pytest.approx(V_IN[1], abs=0.25)
    assert incoming["vz"] == pytest.approx(V_IN[2], abs=0.35)
    assert incoming["n"] >= 5
    assert incoming["maxRes"] <= 0.12


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_miss_gives_no_return(tmp_path: Path) -> None:
    rows = _synth_rows(with_return=False)
    assert _run_return(tmp_path, rows, T_HIT) is None


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_short_out_arc_keeps_hit_time_without_return(tmp_path: Path) -> None:
    """出弧只有 ~40ms（5 点）：回球速度段门槛（≥5 点且跨度 ≥60ms）拒绝 → pcReturnAt=null，
    但来回交点本身成立 → pcHitTimeNear 仍给触球时刻（回球列的交点内核）。"""
    rows = _synth_rows(out_until=0.09)
    out = _run_expr(
        tmp_path, rows,
        f"({{ret:pcReturnAt({T_HIT}),cross:pcHitTimeNear({T_HIT})}})",
    )
    assert out["ret"] is None
    assert out["cross"] is not None
    assert abs(out["cross"] - T_HIT) < 0.02


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_miss_gives_no_hit_time_either(tmp_path: Path) -> None:
    """挥空（y 不反向）：交点也不应存在，不产出假触球时刻。"""
    rows = _synth_rows(with_return=False)
    assert _run_expr(tmp_path, rows, f"pcHitTimeNear({T_HIT})") is None


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_ground_clutter_filtered(tmp_path: Path) -> None:
    rows = _synth_rows(ground_clutter=True)
    ret = _run_return(tmp_path, rows, T_HIT)
    assert ret is not None
    assert abs(ret["yaw"] - YAW_TRUTH) < 2.0
    assert abs(ret["speed"] - SPEED_TRUTH) < 0.4


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_occlusion_gap_after_hit_still_detected(tmp_path: Path) -> None:
    """触球后被拍/臂遮挡断档 200ms（真实 #3 场景）：仍应用遮挡后的真弧给出方向。"""
    rows = _synth_rows(occlusion_gap=(0.05, 0.25))
    ret = _run_return(tmp_path, rows, T_HIT)
    assert ret is not None
    assert abs(ret["yaw"] - YAW_TRUTH) < 2.5
    assert abs(ret["pitch"] - PITCH_TRUTH) < 3.5
    assert abs(ret["speed"] - SPEED_TRUTH) < 0.6


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_ball_swap_jump_truncates_segment(tmp_path: Path) -> None:
    rows = _synth_rows(swap_ball_after=0.15)
    ret = _run_return(tmp_path, rows, T_HIT)
    assert ret is not None
    # 首段被 12m/s 跳变门截断：只用换球前的观测，方向仍来自真实回球
    assert ret["n"] <= 16
    assert abs(ret["yaw"] - YAW_TRUTH) < 3.0
    assert abs(ret["speed"] - SPEED_TRUTH) < 0.6


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_gap_bridged_alien_under_speed_gate_split_off(tmp_path: Path) -> None:
    """0813_083521 抛28 场景：真出弧（上升期）断档 140ms 后接上另一颗更低的静止球，
    隐含跳变速度 ≈11m/s（<20 门）、断档 <150ms、野点不构成 vz 降转升（bounceCut 帮不上）
    ——旧防线全放行，靠轨迹一致性门切段。方向必须来自真弧，速度不得被段末野点杠杆拉爆。"""
    rows = _synth_rows(alien_after_gap=(0.17, 0.30, (2.3, 0.6, 0.4)))
    ret = _run_return(tmp_path, rows, T_HIT)
    assert ret is not None
    assert abs(ret["yaw"] - YAW_TRUTH) < 2.5
    assert abs(ret["pitch"] - PITCH_TRUTH) < 3.0
    assert abs(ret["speed"] - SPEED_TRUTH) < 0.6
    assert ret["n"] <= 14  # 只有真弧（野点子段被 vy 门拒绝）


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_scrambled_depth_noise_rejected(tmp_path: Path) -> None:
    """0813_083521 抛3 场景：出弧 y 逐帧 ±0.18m 摆动（帧间隐含速度 <20m/s、局部
    外推偏差 <0.5m：分段门都不切），拟合残差爆表 → 残差门拒绝出数，不给假方向。"""
    rows = _synth_rows(y_scramble=0.18, out_cadence=0.034)
    out = _run_expr(
        tmp_path, rows,
        f"({{ret:pcReturnAt({T_HIT}),cross:pcHitTimeNear({T_HIT})}})",
    )
    assert out["cross"] is not None  # 交点仍在（乱的是散布不是均值斜率）
    assert out["ret"] is None


@pytest.mark.skipif(NODE is None, reason="需要 node")
def test_post_bounce_tail_not_extrapolated_across_ground(tmp_path: Path) -> None:
    """0812_091825 抛13 场景：球被下压打进地面，触球→反弹段被遮挡，可见出弧全在
    反弹之后。把反弹后的弧倒推回触球时刻会给出假仰角（+70°级），段首前拟合 z
    触地检查必须拒绝出数。"""
    rng = random.Random(11)
    rows: list[dict] = []

    def add(t: float, x: float, y: float, z: float) -> None:
        rows.append({
            "t": round(t, 5),
            "x": round(x + rng.gauss(0.0, 0.002), 5),
            "y": round(y + rng.gauss(0.0, 0.002), 5),
            "z": round(z + rng.gauss(0.0, 0.002), 5),
        })

    t = T_HIT - 0.5
    while t < T_HIT - 0.03:  # 入弧同 _synth_rows
        u = t - T_HIT
        add(t, HIT_POS[0] + V_IN[0] * u, HIT_POS[1] + V_IN[1] * u,
            HIT_POS[2] + V_IN[2] * u - 0.5 * G * u * u)
        t += 0.01
    # 下压出球 (0.3, 6.0, −3.5)，z=0.5 → u≈0.122 触地反弹；反弹前全被遮挡
    u_b = 0.122
    bx = HIT_POS[0] + 0.3 * u_b
    by = HIT_POS[1] + 6.0 * u_b
    t = T_HIT + u_b + 0.036
    while t < T_HIT + 0.45:
        u = t - T_HIT - u_b
        add(t, bx + 0.2 * u, by + 3.6 * u, 3.5 * u - 0.5 * G * u * u)
        t += 0.01
    rows.sort(key=lambda r: r["t"])

    out = _run_expr(
        tmp_path, rows,
        f"({{ret:pcReturnAt({T_HIT}),cross:pcHitTimeNear({T_HIT})}})",
    )
    assert out["cross"] is not None  # 来回交点成立（确认 None 出自段拒绝而非找不到触球）
    assert out["ret"] is None
