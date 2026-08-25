from __future__ import annotations

import pytest

from src.racket_contact import StableRacketContactSolver


def _outgoing_point(t: float, *, contact_t: float = 10.0) -> dict:
    dt = t - contact_t
    return {
        "t": t,
        "x": 0.20 + 1.4 * dt,
        "y": 18.0 - 12.0 * dt,
        "z": 1.05 + 3.0 * dt - 4.9 * dt * dt,
    }


def test_requires_stable_six_seven_eight_point_prefixes():
    solver = StableRacketContactSolver()
    result = None
    for index in range(8):
        result = solver.add(_outgoing_point(10.10 + index * 0.02))
        if index < 7:
            assert result is None

    assert result is not None and result.valid
    assert result.contact_anchor_t_rk == pytest.approx(10.0, abs=1e-10)
    assert result.contact_anchor_world_m == pytest.approx((0.20, 18.0, 1.05), abs=1e-10)
    assert result.contact_model == "rk_ball_z_crossing_fixed_height"
    assert len(result.prefix_anchor_t_rk) == 3
    assert result.prefix_spread_s == pytest.approx(0.0, abs=1e-10)
    assert result.ball_fit_rms_m == pytest.approx(0.0, abs=1e-10)
    assert result.first_observation_lead_s == pytest.approx(0.10)
    assert result.approach_speed_mps == pytest.approx(12.0)
    assert result.trajectory is not None
    assert result.trajectory.position_at(10.0) == pytest.approx((0.20, 18.0, 1.05))
    assert solver.add(_outgoing_point(10.28)) is None


def test_unstable_prefix_fails_closed_and_emits_no_contact_time():
    solver = StableRacketContactSolver(max_prefix_spread_s=0.005)
    result = None
    for index in range(8):
        point = _outgoing_point(20.10 + index * 0.02, contact_t=20.0)
        if index == 7:
            point["z"] += 0.35
        result = solver.add(point)
    if result is None:
        result = solver.finish()

    assert result is not None and not result.valid
    assert result.failure_reason in {
        "unstable_ball_contact_time",
        "ball_contact_fit_residual",
    }
    assert result.contact_anchor_t_rk is None
    assert result.contact_anchor_world_m is None


def test_non_approaching_points_do_not_form_a_contact():
    solver = StableRacketContactSolver()
    result = None
    for index in range(12):
        t = 30.0 + index * 0.02
        result = solver.add({"t": t, "x": 0.0, "y": 4.0 + index * 0.1, "z": 1.0})
    assert result is None
