import math
import sys
from pathlib import Path

import pytest


TEST_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_SRC))
try:
    import extract_rk_tracking_bag as extractor
finally:
    sys.path.pop(0)


def _world(t, x, y, z):
    return ("/ball_world_topic", 0, {"t": t, "x": x, "y": y, "z": z})


def _pre_point(t):
    dt = t - 1.0
    return _world(
        t,
        1.0 + dt,
        5.0 - 6.0 * dt,
        0.033 - 5.0 * dt - 0.5 * 9.8 * dt * dt,
    )


def _post_point(t):
    dt = t - 1.0
    lam, g = 0.12, 10.0
    phi = -math.expm1(-lam * dt) / lam
    return _world(
        t,
        1.0 + 0.6 * phi,
        5.0 - 3.6 * phi,
        0.033 + (4.0 + g / lam) * phi - (g / lam) * dt,
    )


def test_quantized_bounce_replay_only_marks_first_accepted_s1_row():
    rows = [_world(0.3, 0.3, 9.2, 1.0)]
    rows.extend(_pre_point(t) for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    rows.append(
        (
            "/predict_hit_pos",
            0,
            {
                "ct": 0.9,
                "stage": 0,
                "n_points": 6,
                "online_n": 10,
                "az_eff": 10.0,
            },
        )
    )
    rows.append(_world(1.02, 1.02, 4.88, 0.04))  # runtime landing-skip point
    rows.extend(_post_point(t) for t in (1.1, 1.2, 1.3, 1.4))
    rejected_index = len(rows)
    rows.append(
        (
            "/predict_hit_pos",
            0,
            {
                "ct": 1.4,
                "stage": 1,
                "n_points": 10,
                "n_bounce_fit": 4,
                "online_n": 10,
                "bounce_rej": "post_count",
                "az_eff": 10.0,
            },
        )
    )
    rows.extend(_post_point(t) for t in (1.5, 1.6))
    accepted_index = len(rows)
    rows.append(
        (
            "/predict_hit_pos",
            0,
            {
                "ct": 1.6,
                "stage": 1,
                "n_points": 12,
                "n_bounce_fit": 6,
                "online_n": 11,
                "bounce_rej": "ok",
                "az_eff": 12.0,  # S1 payload telemetry is not the transition-locked g
            },
        )
    )
    duplicate_index = len(rows)
    rows.append(("/predict_hit_pos", 0, dict(rows[-1][2])))

    announce = {
        "bot_center": {
            "params": {
                "ground_z": 0.0,
                "s0_drag_k": 0.0,
                "stage1_drag_k": 0.0216,
                "stage1_drag_lambda": 0.12,
                "s1_az_blend": 1.0,
            }
        }
    }
    replay = extractor._replay_bounce_measurements(rows, announce)

    assert rejected_index not in replay
    assert duplicate_index not in replay
    assert set(replay) == {accepted_index}
    cor, cxy, closure_ms = replay[accepted_index]
    assert cor == pytest.approx(0.8, abs=1e-9)
    assert cxy == pytest.approx(0.6, abs=1e-9)
    assert closure_ms == pytest.approx(0.0, abs=1e-8)


def test_extractor_exports_replay_fields_and_non_direct_semantics():
    source = (TEST_SRC / "extract_rk_tracking_bag.py").read_text(encoding="utf-8")
    assert "cor_meas_replay=cor_meas" in source
    assert "cxy_meas_replay=cxy_meas" in source
    assert "cor_meas_closure_ms=closure_ms" in source
    assert "not direct RK BounceSample/history" in source
