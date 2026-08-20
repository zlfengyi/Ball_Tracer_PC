# -*- coding: utf-8 -*-
"""Regression tests for the report-side PC/RK z-axis alignment."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
NODE = shutil.which("node")


def _align_core_js() -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        r"// \[\[align-core-begin\]\].*?\n(.*)// \[\[align-core-end\]\]",
        text,
        re.S,
    )
    assert match, "generate_curve3_html.py is missing the align-core markers"
    return match.group(1)


def _synth(
    bias: float,
    *,
    scale: float = 1.0,
    z_bias: float = 0.08,
    sparse_pc: bool = False,
    noise_amplitude: float = 0.01,
    exact_pose_every: int | None = 1,
    pose_unrelated: bool = False,
):
    """合成一场。

    ``exact_pose_every`` 控制 RK 侧位姿有多少行与 PC 值**逐位相同**（=精确值锚的来源）：
    1 表示全部相同（老场景），N 表示每 N 行才留一条，None 表示一条都没有——真机上
    bot_state 多数时候发的是 KF 传播过的位姿，实测 806 行里只有 33 行逐位相同。
    ``pose_unrelated`` 让 RK 位姿变成一条与 PC 无关的信号，用来验证形状锁会自己弃权。
    """
    starts = [10.0, 24.5, 40.5, 55.0, 71.0, 85.5]
    obs = []
    car = []
    rk = []
    row_index = 0

    def rk_pose(pose, t):
        nonlocal row_index
        row_index += 1
        if pose_unrelated:
            return {
                "x": round(0.7 * math.sin(t * 0.11), 4),
                "y": round(3.0 + 0.5 * math.cos(t * 0.07), 4),
                "yaw": round(0.2 * math.sin(t * 0.05), 4),
            }
        if exact_pose_every is not None and row_index % exact_pose_every == 0:
            return pose
        # KF 传播后的等价位姿：值不再逐位相同，但形状仍是同一条曲线（mm 级差异）
        drift = 0.0012
        return {
            "x": round(pose["x"] + drift, 6),
            "y": round(pose["y"] - drift, 6),
            "yaw": round(pose["yaw"] + drift * 0.5, 6),
        }

    for index, start in enumerate(starts):
        duration = 1.08 + 0.04 * (index % 3)
        amplitude = 0.9 + 0.12 * index
        sample_count = round(duration * 30) + 1
        for sample in range(sample_count):
            elapsed = sample / 30.0
            t = start + elapsed
            phase = min(elapsed / duration, 1.0)
            z = 0.25 + amplitude * 4.0 * phase * (1.0 - phase)
            if not sparse_pc or sample % 9 not in (3, 4):
                obs.append(
                    {"rel_s": round(t, 6), "x": 0.0, "y": 0.0, "z": round(z, 6)}
                )
            pose = {
                "elapsed_s": round(t, 6),
                "x": round(index * 0.03 + sample * 0.0001, 4),
                "y": round(5.0 - index * 0.02 - sample * 0.0002, 4),
                "yaw": round(index * 0.01 + sample * 0.0001, 4),
            }
            car.append(pose)
            rk.append(
                (
                    round((t - bias) / scale, 6),
                    round(
                        z + z_bias + noise_amplitude * math.sin(sample * 1.7),
                        6,
                    ),
                    rk_pose(pose, t),
                )
            )

    for sample in range(200):
        t = starts[-1] + 3.0 + sample / 30.0
        pose = {
            "elapsed_s": round(t, 6),
            "x": round(0.3 + sample * 0.0001, 4),
            "y": round(4.7 - sample * 0.0002, 4),
            "yaw": round(0.08 + sample * 0.0001, 4),
        }
        car.append(pose)
        rk.append(
            (
                round((t - bias) / scale, 6),
                0.25 + z_bias,
                rk_pose(pose, t),
            )
        )

    rk.sort()
    return obs, car, {
        "world": {
            "t": [t for t, _, _ in rk],
            "y": {
                "z": [z for _, z, _ in rk],
                "bot_x": [pose["x"] for _, _, pose in rk],
                "bot_y": [pose["y"] for _, _, pose in rk],
                "bot_yaw": [pose["yaw"] for _, _, pose in rk],
            },
        }
    }


def _run_estimate(
    obs,
    car,
    rk_data,
    tmp_path: Path,
    expr: str = "estimateTimeMap()",
    *,
    cfg: dict | None = None,
) -> dict:
    """在 node 里跑 align-core。

    桩必须覆盖 align-core 声明的全部外部依赖（obs/car/RK/cfg/t0/isNum/relTime）；
    ``t0`` 与 ``RK.t0`` 都取 0，于是时钟桥的 ``pc_minus_rk`` 就直接等于 rel 轴 bias。
    """
    rk_data = {**rk_data, "t0": 0}
    harness = (
        "const isNum = v => typeof v==='number' && isFinite(v);\n"
        f"const obs = {json.dumps(obs)};\n"
        f"const car = {json.dumps(car)};\n"
        "const relTime = v => v;\n"
        "const t0 = 0;\n"
        f"const cfg = {json.dumps(cfg or {})};\n"
        f"const RK = {json.dumps(rk_data)};\n"
        f"{_align_core_js()}\n"
        f"console.log(JSON.stringify({expr}));\n"
    )
    script = tmp_path / "align_harness.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_all_cross_axis_rendering_uses_offset_only_time_map():
    source = SRC.read_text(encoding="utf-8")
    assert "const rkToPc = t => isNum(Number(t)) ? Number(t)+rkBias : null;" in source
    assert "const shifted = xs => xs.map(rkToPc);" in source
    assert "const ctPc=rkToPc(th.ref300T);" in source
    assert "const htPc=rkToPc(th.ref300Ht);" in source
    assert "armTcpRows.map(s=>rkToPc(s.t))" in source
    assert "rkScale" not in source
    assert "driftPpm" not in source
    assert "scale*row.t" not in _align_core_js()
    assert "rkOffset" not in source
    assert "__rkOffset" not in source


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("bias", [75.0, -9.0])
def test_estimate_time_map_recovers_bias(tmp_path, bias):
    obs, car, rk_data = _synth(bias)
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["err"] is not None and best["err"] < 0.05, best
    assert abs(best["scale"] - 1.0) <= 1e-5, best
    assert abs(best["bias"] - bias) <= 0.01, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_time_map_ignores_z_bias_and_pc_gaps(tmp_path):
    obs, car, rk_data = _synth(-8.823, z_bias=0.35, sparse_pc=True)
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["err"] is not None and best["err"] < 0.05, best
    assert abs(best["bias"] + 8.823) <= 0.012, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_time_map_sub_millisecond_refinement(tmp_path):
    obs, car, rk_data = _synth(-8.8222, noise_amplitude=0.0)
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert abs(best["bias"] + 8.8222) <= 0.0005, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_time_map_never_fits_clock_drift(tmp_path):
    obs, car, rk_data = _synth(
        -8.8222,
        scale=1.0004,
        noise_amplitude=0.0,
    )
    anchor = _run_estimate(obs, car, rk_data, tmp_path, expr="clockAnchor")
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert anchor["scale"] == 1, anchor
    assert best["scale"] == 1, best
    assert best["err"] is not None and best["err"] < 0.05, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pose_pipeline_age_cannot_create_a_clock_scale(tmp_path):
    """共同 pose 值的两端时刻混有视觉/网络/状态保持年龄，斜率不得进入时间映射。"""
    bias = -9.0693
    obs, car, rk_data = _synth(bias, noise_amplitude=0.0)
    for row in car:
        row["elapsed_s"] = round(row["elapsed_s"] - 0.0008 * row["elapsed_s"], 6)

    anchor = _run_estimate(obs, car, rk_data, tmp_path, expr="clockAnchor")
    assert anchor["anchors"] >= 20, anchor
    assert anchor["scale"] == 1, anchor

    cfg = {"rk_clock_bridge": {"pc_minus_rk": bias + 0.020, "mad": 0.002, "n": 5000}}
    best = _run_estimate(obs, car, rk_data, tmp_path, cfg=cfg)
    assert best["scale"] == 1, best
    assert abs(best["bias"] - bias) <= 0.01, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_time_map_weights_flights_not_frames(tmp_path):
    true_bias = -8.8222
    obs, car, rk_data = _synth(true_bias, noise_amplitude=0.0)
    outlier_bias = true_bias + 0.030
    for sample in range(481):
        elapsed = sample / 120.0
        t = 110.0 + elapsed
        phase = elapsed / 4.0
        z = 0.25 + 1.4 * 4.0 * phase * (1.0 - phase)
        obs.append({"rel_s": round(t, 6), "x": 0.0, "y": 0.0, "z": round(z, 6)})
        rk_data["world"]["t"].append(round(t - outlier_bias, 6))
        rk_data["world"]["y"]["z"].append(round(z + 0.2, 6))
        rk_data["world"]["y"]["bot_x"].append(None)
        rk_data["world"]["y"]["bot_y"].append(None)
        rk_data["world"]["y"]["bot_yaw"].append(None)

    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert abs(best["bias"] - true_bias) <= 0.003, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_time_map_survives_contaminated_pose_anchors(tmp_path):
    """0809 场：多数共享位姿锚配错，把 offMad 顶到秒级，8×offMad 反而全放行。

    错锚会把 bias 搜索窗带离真解；锚阶段必须靠硬上限过滤，只交出常数 bias 收窄窗口，
    绝不能再从这种管线年龄里拟合 scale。
    """
    bias = -10.11
    obs, car, rk_data = _synth(bias, noise_amplitude=0.0)
    # 2/3 的锚配错（本场 83 条里只有 30 条是真的），错量以真值为中心双向散开几秒到
    # 二十几秒——直接扰动 PC 侧锚时刻即可等价表达这种配对错误。
    for index, pose in enumerate(car):
        if index % 3 == 0:
            continue
        pose["elapsed_s"] = round(pose["elapsed_s"] + ((index * 37) % 41 - 20) * 1.1, 6)

    anchor = _run_estimate(obs, car, rk_data, tmp_path, expr="clockAnchor")
    assert anchor["bias"] is not None, anchor
    assert anchor["scale"] == 1, anchor
    assert abs(anchor["bias"] - bias) <= 0.05, anchor
    assert anchor["anchors"] >= 20, anchor

    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["err"] is not None and best["err"] < 0.05, best
    assert abs(best["bias"] - bias) <= 0.01, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_clock_bridge_beats_every_inferred_source(tmp_path):
    """录制时量出来的 PC↔RK 时钟桥是唯一**不依赖本场有没有球**的粗定位手段，优先级最高。

    tracker 订阅 /bot_state 记 median(perf 收到 − 载荷 RK 时刻)，写进 config.rk_clock_bridge。
    这里把位姿也打乱（形状锁失效、精确值锚归零），验证光靠时钟桥仍能把 bias 锁到亚厘秒。
    """
    bias = -11.7505
    obs, car, rk_data = _synth(
        bias, noise_amplitude=0.0, exact_pose_every=None, pose_unrelated=True
    )
    cfg = {"rk_clock_bridge": {"pc_minus_rk": bias + 0.031, "mad": 0.004, "n": 4200}}

    bridge = _run_estimate(obs, car, rk_data, tmp_path, expr="clockBridge", cfg=cfg)
    assert bridge["bias"] is not None and abs(bridge["bias"] - bias) <= 0.05, bridge

    best = _run_estimate(obs, car, rk_data, tmp_path, cfg=cfg)
    assert best["windowSource"] == "bridge", best
    assert abs(best["bias"] - bias) <= 0.01, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_clock_bridge_ignored_when_jittery_or_absent(tmp_path):
    """桥抖动过大（那条 topic 当场断断续续）或场次太老没有该字段时必须弃用，
    退回位姿形状锁——绝不能拿一个脏桥把窗定歪。"""
    bias = -11.7505
    obs, car, rk_data = _synth(bias, noise_amplitude=0.0)

    jittery = {"rk_clock_bridge": {"pc_minus_rk": bias, "mad": 0.9, "n": 4200}}
    assert _run_estimate(obs, car, rk_data, tmp_path, expr="clockBridge", cfg=jittery)["bias"] is None
    assert _run_estimate(obs, car, rk_data, tmp_path, cfg=jittery)["windowSource"] != "bridge"

    thin = {"rk_clock_bridge": {"pc_minus_rk": bias, "mad": 0.001, "n": 3}}
    assert _run_estimate(obs, car, rk_data, tmp_path, expr="clockBridge", cfg=thin)["bias"] is None

    absent = _run_estimate(obs, car, rk_data, tmp_path, expr="clockBridge")
    assert absent["bias"] is None and absent["n"] == 0, absent


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pose_shape_lock_works_when_exact_anchors_vanish(tmp_path):
    """精确值锚归零时，位姿**形状**锁必须独自把搜索窗收住。

    0812_052638 场的教训：精确值锚天生稀疏（bot_state 发的是传播过的位姿），场次间
    从 87 条到 0 条乱跳，不能当唯一的粗定位手段。形状锁不要求数值相等，用的是全部
    位姿行，且小车位姿全场不重复 → 免疫抛球周期混叠。
    """
    bias = -11.7505
    obs, car, rk_data = _synth(bias, noise_amplitude=0.0, exact_pose_every=None)

    anchor = _run_estimate(obs, car, rk_data, tmp_path, expr="clockAnchor")
    assert anchor["bias"] is None and anchor["anchors"] < 8, anchor

    lock = _run_estimate(obs, car, rk_data, tmp_path, expr="poseLock")
    assert lock["usable"], lock
    # 形状锁只负责粗定位：真机上它比 z 精锁早 0.1~0.29s（/pc_car_loc→bot_state 管线
    # 延迟），窗开 ±1.0s 就够。精度由后面的 z 精锁负责，见最后一条断言。
    assert abs(lock["bias"] - bias) <= 0.5, lock

    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["windowSource"] == "pose", best
    assert abs(best["bias"] - bias) <= 0.01, best
    assert best["err"] is not None and best["err"] < 0.05, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pose_shape_lock_abstains_when_the_two_streams_are_different_signals(tmp_path):
    """老场次 RK 自算位姿 / 小车全程静止时，两条位姿不同源——形状锁必须自己弃权，
    把窗让回全场扫描，而不是拿一个 0.5~6s 的错窗把 z 精锁带偏。"""
    bias = 7.0
    obs, car, rk_data = _synth(bias, noise_amplitude=0.0, pose_unrelated=True)

    lock = _run_estimate(obs, car, rk_data, tmp_path, expr="poseLock")
    assert not lock["usable"], lock

    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["windowSource"] == "scan", best
    assert abs(best["bias"] - bias) <= 0.01, best
    # 全场扫描路径必须自报混叠余量（冠军 vs 2s 外次优），页面用它当质量门
    assert best["margin"] is None or best["margin"] > 1.35, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_clock_anchor_keeps_bias_when_anchors_are_few_but_tight(tmp_path):
    """锚少 ≠ 锚不准。

    0812_052638 场只有 14 条锚（MAD 45ms，够把窗收到 ±0.75s），旧代码的 `<20` 硬门
    把 bias 连同锚一起丢掉 → 搜索退回全场 ±130s → 锁到错的一抛（z 形状误差 0.289m）。
    位姿值锚无论多少都只做常数 bias 粗定位，scale 永远为 1。
    """
    bias = -11.75
    obs, car, rk_data = _synth(bias, noise_amplitude=0.0, exact_pose_every=30)

    anchor = _run_estimate(obs, car, rk_data, tmp_path, expr="clockAnchor")
    assert 8 <= anchor["anchors"] < 20, anchor
    assert anchor["bias"] is not None, anchor
    assert abs(anchor["bias"] - bias) <= 0.05, anchor
    assert anchor["scale"] == 1, anchor


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_time_map_degenerate_returns_null_err(tmp_path):
    obs, car, rk_data = _synth(5.0)
    rk_data["world"]["t"] = rk_data["world"]["t"][:20]
    for values in rk_data["world"]["y"].values():
        del values[20:]
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["bias"] is None
    assert best["err"] is None
    assert best["n"] == 0
