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
):
    starts = [10.0, 24.5, 40.5, 55.0, 71.0, 85.5]
    obs = []
    car = []
    rk = []
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
                    pose,
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
                pose,
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


def _run_estimate(obs, car, rk_data, tmp_path: Path) -> dict:
    harness = (
        "const isNum = v => typeof v==='number' && isFinite(v);\n"
        f"const obs = {json.dumps(obs)};\n"
        f"const car = {json.dumps(car)};\n"
        "const relTime = v => v;\n"
        f"const RK = {json.dumps(rk_data)};\n"
        f"{_align_core_js()}\n"
        "console.log(JSON.stringify(estimateTimeMap()));\n"
    )
    script = tmp_path / "align_harness.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_all_cross_axis_rendering_uses_affine_time_map():
    source = SRC.read_text(encoding="utf-8")
    assert "const rkToPc = t =>" in source
    assert "const shifted = xs => xs.map(rkToPc);" in source
    assert "const ctPc=rkToPc(th.ref300T);" in source
    assert "const htPc=rkToPc(th.ref300Ht);" in source
    assert "armTcpRows.map(s=>rkToPc(s.t))" in source
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
def test_estimate_time_map_recovers_clock_drift(tmp_path):
    obs, car, rk_data = _synth(
        -8.8222,
        scale=1.0004,
        noise_amplitude=0.0,
    )
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert abs(best["scale"] - 1.0004) <= 2e-5, best
    assert abs(best["bias"] + 8.8222) <= 0.002, best


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
def test_estimate_time_map_degenerate_returns_null_err(tmp_path):
    obs, car, rk_data = _synth(5.0)
    rk_data["world"]["t"] = rk_data["world"]["t"][:20]
    for values in rk_data["world"]["y"].values():
        del values[20:]
    best = _run_estimate(obs, car, rk_data, tmp_path)
    assert best["bias"] is None
    assert best["err"] is None
    assert best["n"] == 0
