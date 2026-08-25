from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from src.racket_contact import RacketContactEstimate, StableRacketContactSolver


_RK_ROOT = Path(r"D:\Ball_Tracer_PC\tracker_output")

_REAL_DOWN = [
    (
        [
            (0.0, -0.6864, 17.8240, 1.4023),
            (0.033370, -0.6863, 17.6284, 1.3469),
            (0.066807, -0.6883, 17.5952, 1.2657),
            (0.100104, -0.6683, 17.4510, 1.2890),
            (0.133238, -0.6293, 17.0737, 1.4181),
            (0.166392, -0.5954, 16.9254, 1.5562),
            (0.199539, -0.5099, 16.2797, 1.6427),
            (0.232666, -0.4607, 15.6922, 1.7215),
        ],
        -0.0740131556,
    ),
    (
        [
            (0.0, -0.1789, 16.5833, 1.2979),
            (0.033444, -0.1846, 16.5576, 1.2198),
            (0.200169, -0.1662, 16.2219, 1.1393),
            (0.233272, -0.1238, 16.0524, 1.3028),
            (0.266356, -0.0832, 15.5325, 1.4376),
            (0.299427, -0.0466, 15.1265, 1.5735),
            (0.332536, -0.0173, 14.7522, 1.6959),
            (0.365605, 0.0169, 14.5465, 1.8250),
        ],
        -0.0297250773,
    ),
    (
        [
            (0.0, -1.4002, 18.9363, 1.1887),
            (0.166577, -1.2466, 18.7611, 1.2651),
            (0.199732, -1.1722, 18.4490, 1.3961),
            (0.232915, -1.0690, 18.0309, 1.5068),
            (0.266066, -0.9974, 17.7502, 1.6227),
            (0.299245, -0.8978, 17.3805, 1.7117),
            (0.332486, -0.8040, 16.8087, 1.7815),
            (0.365604, -0.7407, 16.6876, 1.8848),
        ],
        0.0068430289,
    ),
]

_CORRUPT_DOWN = [
    (0.0, 3.3311, 11.8780, 1.0714),
    (0.033334, 3.3214, 11.8444, 1.0679),
    (0.101314, 0.9529, 6.7873, 0.4893),
    (0.171285, 1.0031, 6.2367, 0.0539),
    (0.203655, 1.0090, 6.0590, 0.2040),
    (0.235733, 1.0141, 5.8841, 0.3672),
    (0.267756, 1.0309, 5.7346, 0.5179),
    (0.299656, 1.0339, 5.5583, 0.6571),
    (0.331428, 1.0461, 5.3983, 0.7863),
    (0.363055, 1.0565, 5.2395, 0.9051),
]


def _solve(points) -> RacketContactEstimate:
    solver = StableRacketContactSolver()
    result = None
    for t_rk, x_m, y_m, z_m in points:
        candidate = solver.add({"t": t_rk, "x": x_m, "y": y_m, "z": z_m})
        if candidate is not None:
            result = candidate
    final_candidate = solver.finish()
    if final_candidate is not None:
        result = final_candidate
    assert result is not None
    return result


@pytest.mark.parametrize(("points", "anchor_t_rk"), _REAL_DOWN)
def test_three_visible_down_events_are_physical_consensus(points, anchor_t_rk):
    result = _solve(points)

    assert result.valid
    assert result.acceptance_mode == "physical_consensus"
    assert result.contact_anchor_t_rk == pytest.approx(anchor_t_rk, abs=2e-6)
    assert result.prefix_spread_s <= 0.060
    assert result.contact_point_spread_m <= 0.15
    assert result.ball_fit_rms_m <= 0.40


def test_corrupt_down_identity_jump_is_still_rejected():
    result = _solve(_CORRUPT_DOWN)

    assert not result.valid
    assert result.failure_reason == "invalid_ball_contact_prefix"
    assert result.contact_anchor_t_rk is None
    assert result.window_shift == 0


def test_contact_cooldown_uses_physical_anchor_and_suppresses_late_duplicate():
    template = _solve(_REAL_DOWN[0][0])
    first = replace(template, contact_anchor_t_rk=10.0)
    duplicate = replace(template, contact_anchor_t_rk=9.98)
    later = replace(template, contact_anchor_t_rk=12.0)
    solver = StableRacketContactSolver(cooldown_s=1.5)

    assert solver._finalize(first, 10.2) is first
    assert solver._finalize(duplicate, 11.8) is None
    assert solver._finalize(later, 12.2) is later


def test_pending_rejection_emits_once_at_eof():
    solver = StableRacketContactSolver()
    emitted = [
        solver.add({"t": t, "x": x, "y": y, "z": z})
        for t, x, y, z in _CORRUPT_DOWN[2:]
    ]

    assert all(result is None for result in emitted)
    result = solver.finish()
    assert result is not None and not result.valid
    assert result.window_shift == 0
    assert solver.finish() is None


def test_pending_rejection_emits_once_at_leg_boundary_without_crossing_it():
    solver = StableRacketContactSolver()
    for t, x, y, z in _CORRUPT_DOWN[2:]:
        assert solver.add({"t": t, "x": x, "y": y, "z": z}) is None

    result = solver.add({"t": 1.0, "x": 0.0, "y": 18.0, "z": 1.0})
    assert result is not None and not result.valid
    assert result.window_shift == 0
    assert solver.finish() is None


def _session_contacts(short_session: str) -> list[RacketContactEstimate]:
    path = (
        _RK_ROOT
        / f"tracker_20260822_{short_session}"
        / f"tracker_20260822_{short_session}_rk_tracking.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    world = data["world"]
    values = world["y"]
    solver = StableRacketContactSolver()
    results = []
    for relative_t, x_m, y_m, z_m in zip(
        world["t"], values["x"], values["y"], values["z"]
    ):
        result = solver.add(
            {
                "t": float(data["t0"]) + float(relative_t),
                "x": x_m,
                "y": y_m,
                "z": z_m,
            }
        )
        if result is not None:
            results.append(result)
    final_result = solver.finish()
    if final_result is not None:
        results.append(final_result)
    return results


def test_three_session_retry_replay_keeps_88_of_95_and_exact29_keeps_27_anchors():
    expected = {
        "145905": Counter(
            strict=35,
            physical_consensus=6,
            invalid_ball_contact_prefix=2,
        ),
        "113456": Counter(
            strict=18,
            physical_consensus=5,
            invalid_ball_contact_prefix=1,
            contact_outside_reach_volume=2,
        ),
        "124829": Counter(
            strict=21,
            physical_consensus=3,
            invalid_ball_contact_prefix=2,
        ),
    }
    exact_indices = {
        "145905": [0, 2, 3, 4, 6, 8, 10, 15, 16, 31],
        "113456": [0, 4, 5, 7, 8, 9, 11, 15, 21],
        "124829": [0, 1, 2, 8, 10, 12, 17, 18, 19, 20],
    }

    total = accepted = exact_accepted = 0
    for session, counts in expected.items():
        contacts = _session_contacts(session)
        actual = Counter(
            item.acceptance_mode if item.valid else item.failure_reason
            for item in contacts
        )
        assert actual == counts
        total += len(contacts)
        accepted += sum(item.valid for item in contacts)
        exact_accepted += sum(contacts[index].valid for index in exact_indices[session])

    assert (total, accepted, exact_accepted) == (95, 88, 27)


def test_external_three_session_retry_replay_keeps_40_of_45_after_anchor_dedup():
    expected = {
        "112835": Counter(strict=15),
        "094941": Counter(
            strict=10,
            invalid_ball_contact_prefix=2,
            contact_outside_reach_volume=1,
        ),
        "071624": Counter(
            strict=15,
            unstable_ball_contact_time=1,
            invalid_ball_contact_prefix=1,
        ),
    }

    contacts = {
        session: _session_contacts(session)
        for session in expected
    }
    assert {
        session: Counter(
            item.acceptance_mode if item.valid else item.failure_reason
            for item in rows
        )
        for session, rows in contacts.items()
    } == expected
    assert sum(len(rows) for rows in contacts.values()) == 45
    assert sum(item.valid for rows in contacts.values() for item in rows) == 40

    retry_shifts = Counter(
        item.window_shift
        for rows in contacts.values()
        for item in rows
        if item.valid and item.window_shift > 0
    )
    assert retry_shifts == Counter({1: 4, 4: 1})
