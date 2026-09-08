from __future__ import annotations

import math
import queue
from pathlib import Path

import pytest

from src.online_time_alignment import OnlineThrowTimeAligner, _Point, _upper_median
from src.run_tracker import DirectRos2Sink
from src.ros2_support import ROS2_RELIABLE_TOPICS


ROOT = Path(__file__).resolve().parents[1]


def test_report_median_and_bounded_interpolation_contract_are_exact():
    assert _upper_median([1.0, 3.0, 5.0, 7.0]) == 5.0
    points = (
        _Point(t=1.00, x=0.0, y=0.0, z=1.0),
        _Point(t=1.04, x=0.0, y=0.0, z=2.0),
        _Point(t=1.08, x=0.0, y=0.0, z=3.0),
    )
    times = [point.t for point in points]

    assert OnlineThrowTimeAligner._interp_z(points, times, 1.00) is None
    assert OnlineThrowTimeAligner._interp_z(points, times, 1.04) == 2.0
    assert OnlineThrowTimeAligner._interp_z(points, times, 1.08) == 3.0


def _ball_z(physical_t: float) -> float:
    u = physical_t - 0.65
    if u <= 0.0:
        return 0.04 + 2.8 * (-u) + u * u
    return 0.04 + 3.0 * u - 4.9 * u * u


def _throw_events(
    *,
    rk_start: float,
    pc_minus_rk: float,
    pc_z_bias: float = 0.08,
    pc_only_from: float = 0.0,
    pc_stop: float = 1.0,
    rk_stop: float = 1.0,
    pc_spike_at: float | None = None,
    y_start: float = 12.2,
    y_speed: float = 10.0,
    pc_arrival_delay: float = 0.020,
    rk_arrival_delay: float = 0.050,
) -> list[tuple[float, str, dict]]:
    events: list[tuple[float, str, dict]] = []
    for index in range(31):
        physical_t = index / 30.0
        if not (pc_only_from <= physical_t <= pc_stop):
            continue
        z = _ball_z(physical_t) + pc_z_bias
        if pc_spike_at is not None and abs(physical_t - pc_spike_at) < 0.018:
            z += 0.9
        t = rk_start + physical_t + pc_minus_rk
        events.append((
            t + pc_arrival_delay,
            "pc",
            {
                "t": t,
                "x": 0.2 + 0.3 * physical_t,
                "y": y_start - y_speed * physical_t,
                "z": z,
            },
        ))
    for index in range(51):
        physical_t = index / 50.0
        if physical_t > rk_stop:
            continue
        t = rk_start + physical_t
        events.append((
            t + pc_minus_rk + rk_arrival_delay,
            "rk",
            {
                "t": t,
                "x": 0.2 + 0.3 * physical_t,
                "y": y_start - y_speed * physical_t,
                "z": _ball_z(physical_t),
            },
        ))
    return sorted(events, key=lambda row: row[0])


def _two_bounce_z(physical_t: float) -> float:
    first_bounce = 0.60
    second_bounce = first_bounce + 2.30 / 3.80
    if physical_t <= first_bounce:
        u = first_bounce - physical_t
        return 0.05 + 2.80 * u + u * u
    if physical_t <= second_bounce:
        u = physical_t - first_bounce
        return 0.05 + 2.30 * u - 3.80 * u * u
    u = physical_t - second_bounce
    return 0.05 + 1.80 * u - 4.90 * u * u


def _two_bounce_events(
    *,
    rk_start: float,
    pc_minus_rk: float,
    y_start: float,
    y_speed: float,
) -> list[tuple[float, str, dict]]:
    events = []
    for source, hz, delay, z_bias in (
        ("pc", 30, 0.020, 0.08),
        ("rk", 50, 0.050, 0.00),
    ):
        for index in range(int(1.55 * hz) + 1):
            physical_t = index / hz
            rk_t = rk_start + physical_t
            payload_t = rk_t + (pc_minus_rk if source == "pc" else 0.0)
            payload = {
                "t": payload_t,
                "x": 0.2 + 0.25 * physical_t,
                "y": y_start - y_speed * physical_t,
                "z": _two_bounce_z(physical_t) + z_bias,
            }
            events.append((
                rk_t + pc_minus_rk + delay,
                source,
                payload,
            ))
    return sorted(events, key=lambda row: row[0])


def _feed(
    aligner: OnlineThrowTimeAligner,
    events: list[tuple[float, str, dict]],
) -> list[dict]:
    results = []
    for recv_pc, source, payload in events:
        if source == "pc":
            result = aligner.add_pc(**payload, arrival_pc=recv_pc)
        else:
            result = aligner.add_rk(payload, arrival_pc=recv_pc)
        if result is not None:
            results.append(result)
    return results


def test_complete_throw_updates_absolute_offset_once_then_next_throw_replaces_it():
    aligner = OnlineThrowTimeAligner()
    first_offset = 4.6234

    first = _feed(
        aligner,
        _throw_events(rk_start=200.0, pc_minus_rk=first_offset),
    )

    assert len(first) == 1
    assert first[0]["accepted"] is True
    assert first[0]["pc_minus_rk"] == pytest.approx(first_offset, abs=0.001)
    assert first[0]["baseline_source"] == "paired_bounce"
    assert aligner.snapshot()["updates"] == 1

    # More near-side points and a complete late replay of the same timestamps
    # must not re-arm the already consumed physical throw.
    tail = []
    for index in range(1, 8):
        physical_t = 1.0 + index / 30.0
        rk_t = 200.0 + physical_t
        tail.extend([
            (
                rk_t + first_offset + 0.02,
                "pc",
                {
                    "t": rk_t + first_offset,
                    "x": 0.5,
                    "y": 6.0 - physical_t,
                    "z": 0.5,
                },
            ),
            (
                rk_t + first_offset + 0.05,
                "rk",
                {"t": rk_t, "x": 0.5, "y": 6.0 - physical_t, "z": 0.42},
            ),
        ])
    assert _feed(aligner, sorted(tail)) == []
    assert _feed(
        aligner,
        _throw_events(rk_start=200.0, pc_minus_rk=first_offset),
    ) == []
    assert aligner.snapshot()["updates"] == 1


    second_offset = first_offset + 0.006
    second = _feed(
        aligner,
        _throw_events(rk_start=204.0, pc_minus_rk=second_offset),
    )

    assert len(second) == 1
    assert second[0]["accepted"] is True
    assert second[0]["baseline_source"] == "previous_throw"
    assert second[0]["pc_minus_rk"] == pytest.approx(second_offset, abs=0.001)
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 2
    assert snapshot["attempts"] == 2
    assert snapshot["pc_minus_rk_s"] == pytest.approx(second_offset, abs=0.001)


def test_rk_clock_rollback_invalidates_output_until_next_complete_throw():
    aligner = OnlineThrowTimeAligner()
    first = _feed(
        aligner,
        _throw_events(rk_start=200.0, pc_minus_rk=4.625),
    )
    assert len(first) == 1 and first[0]["accepted"] is True

    sink = DirectRos2Sink.__new__(DirectRos2Sink)
    sink._time_aligner = aligner
    sink._alignment_queue = None
    sink._world_ball_pub = object()
    sink._car_pub = object()
    sink._world_ball_count = 0
    sink._car_count = 0
    published = []
    sink._publish = lambda publisher, payload: published.append((publisher, payload))
    car = {"topic": "car_loc", "t": 206.0, "x": 1.0, "y": 2.0, "z": 0.0}
    ball = {"t": 206.1, "x": 0.1, "y": 7.0, "z": 1.2}
    sink.publish_car_loc(car)
    sink.add_pc_ball(ball)
    assert len(published) == 2

    before = aligner.snapshot()
    rk_clock = aligner._source_clock["rk"]
    assert aligner.add_rk(
        {"t": 10.0, "x": 0.2, "y": 12.2, "z": _ball_z(0.0)},
        arrival_pc=rk_clock.last_arrival_pc + 1.0,
    ) is None
    # One late/corrupt packet is not enough to invalidate a good offset.
    assert aligner.current_offset() == pytest.approx(4.625, abs=0.001)
    assert aligner.add_rk(
        {"t": 10.02, "x": 0.2, "y": 12.0, "z": _ball_z(0.02)},
        arrival_pc=rk_clock.last_arrival_pc + 1.02,
    ) is None
    invalidated = aligner.snapshot()
    assert invalidated["pc_minus_rk_s"] is None
    assert invalidated["updated_pc"] is None
    assert invalidated["updates"] == before["updates"]
    assert invalidated["attempts"] == before["attempts"]
    assert invalidated["history"] == before["history"]

    sink.publish_car_loc(car)
    sink.add_pc_ball(ball)
    assert len(published) == 2

    rebuilt = _feed(
        aligner,
        _throw_events(rk_start=11.0, pc_minus_rk=197.0),
    )
    assert len(rebuilt) == 1 and rebuilt[0]["accepted"] is True
    assert rebuilt[0]["baseline_source"] == "paired_bounce"
    assert aligner.current_offset() == pytest.approx(197.0, abs=0.001)
    assert aligner.snapshot()["updates"] == 2

    resumed_car = {**car, "t": 209.5}
    resumed_ball = {**ball, "t": 209.6}
    sink.publish_car_loc(resumed_car)
    sink.add_pc_ball(resumed_ball)
    assert len(published) == 4
    assert published[-2][1]["rk_timestamp"] == pytest.approx(12.5, abs=0.001)
    assert published[-1][1]["rk_timestamp"] == pytest.approx(12.6, abs=0.001)


def test_throw_first_seen_just_beyond_8m_is_eligible_without_12m_evidence():
    aligner = OnlineThrowTimeAligner()
    true_offset = 4.25

    results = _feed(
        aligner,
        _throw_events(
            rk_start=250.0,
            pc_minus_rk=true_offset,
            y_start=8.4,
            y_speed=3.0,
        ),
    )

    assert len(results) == 1 and results[0]["accepted"] is True
    assert results[0]["pc_minus_rk"] == pytest.approx(true_offset, abs=0.001)
    assert aligner.snapshot()["incoming_cross_y_m"] == 8.0


def test_next_clean_throw_replaces_offset_within_30ms_continuity_gate():
    aligner = OnlineThrowTimeAligner()
    initial_offset = 4.0
    _feed(
        aligner,
        _throw_events(rk_start=270.0, pc_minus_rk=initial_offset),
    )
    changed_offset = initial_offset + 0.025

    results = _feed(
        aligner,
        _throw_events(rk_start=274.0, pc_minus_rk=changed_offset),
    )

    assert len(results) == 1 and results[0]["accepted"] is True
    assert results[0]["pc_minus_rk"] == pytest.approx(changed_offset, abs=0.001)
    assert aligner.snapshot()["updates"] == 2


def test_existing_offset_rejects_40ms_z_phase_wrong_track_with_one_metre_xy_bias():
    aligner = OnlineThrowTimeAligner()
    initial_offset = 4.0
    first = _feed(
        aligner,
        _throw_events(rk_start=285.0, pc_minus_rk=initial_offset),
    )
    assert len(first) == 1 and first[0]["accepted"] is True

    rk_start = 289.0
    events = []
    for arrival_pc, source, payload in _throw_events(
        rk_start=rk_start,
        pc_minus_rk=initial_offset,
    ):
        if source == "pc":
            physical_t = payload["t"] - rk_start - initial_offset
            payload = {
                **payload,
                "x": payload["x"] + 1.0,
                "z": _ball_z(physical_t - 0.040) + 0.08,
            }
        events.append((arrival_pc, source, payload))
    results = _feed(aligner, events)
    if results:
        assert len(results) == 1
        rejection = results[0]
    else:
        candidate = aligner._candidate
        assert candidate is not None
        rejection = aligner.expire(
            now_arrival_pc=candidate.last_joint_arrival_pc + 0.351,
        )

    assert rejection is not None
    assert rejection["accepted"] is False
    assert rejection["reason"] == "offset_jump"
    assert rejection["features"]["offset_jump_ms"] == pytest.approx(
        40.2, abs=1.0
    )
    assert rejection["features"]["xy_shift"] == pytest.approx(1.09, abs=0.05)
    assert rejection["features"]["xy_res_median"] < 0.05
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 1
    assert snapshot["attempts"] == 2
    assert snapshot["pc_minus_rk_s"] == pytest.approx(initial_offset, abs=0.001)
    assert snapshot["reason_counts"] == {"offset_jump": 1}


@pytest.mark.parametrize(
    "events",
    [
        # PC starts after the 8 m crossing: a nice-looking bounce is still not
        # a complete user-defined stage 0 on both sides.
        _throw_events(
            rk_start=300.0,
            pc_minus_rk=5.0,
            pc_only_from=0.85,
        ),
        # Both sides stop at the bounce and never prove stage 1.
        _throw_events(
            rk_start=320.0,
            pc_minus_rk=5.0,
            pc_stop=0.67,
            rk_stop=0.67,
        ),
    ],
)
def test_incomplete_throws_do_not_attempt_or_create_an_offset(events):
    aligner = OnlineThrowTimeAligner()

    _feed(aligner, events)

    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 0
    assert snapshot["attempts"] == 0
    assert snapshot["pc_minus_rk_s"] is None


def test_expire_emits_one_terminal_for_a_committed_incomplete_final_throw():
    aligner = OnlineThrowTimeAligner()
    assert _feed(
        aligner,
        _throw_events(
            rk_start=350.0,
            pc_minus_rk=4.0,
            pc_stop=0.78,
            rk_stop=0.78,
        ),
    ) == []
    candidate = aligner._candidate
    assert candidate is not None and candidate.confirmed is True
    assert any(index is not None for index in candidate.bounce_index.values())

    result = aligner.expire(
        now_arrival_pc=candidate.last_joint_arrival_pc + 0.351,
    )

    assert result is not None
    assert result["accepted"] is False
    assert result["reason"] == "stage1_incomplete"
    snapshot = aligner.snapshot()
    assert snapshot["attempts"] == 1
    assert snapshot["rejected"] == 1
    assert snapshot["reason_counts"] == {"stage1_incomplete": 1}
    assert snapshot["history"] == [result]
    assert aligner.expire(now_arrival_pc=999.0) is None
    assert aligner.snapshot()["history"] == [result]


def test_expire_can_fit_six_clean_stage1_points_without_relaxing_fit_gates():
    aligner = OnlineThrowTimeAligner()
    events = []
    rk_start = 370.0
    pc_minus_rk = 4.0
    for arrival_pc, source, payload in _throw_events(
        rk_start=rk_start,
        pc_minus_rk=pc_minus_rk,
        pc_stop=1.0,
        rk_stop=0.75,
    ):
        physical_t = payload["t"] - rk_start - (
            pc_minus_rk if source == "pc" else 0.0
        )
        # A small clean incoming feature gives the unchanged z-profile fit a
        # unique minimum even though RK ends with only six Stage1 points.
        z_feature = 0.10 * math.exp(-((physical_t - 0.38) / 0.06) ** 2)
        events.append((arrival_pc, source, {**payload, "z": payload["z"] + z_feature}))

    assert _feed(aligner, events) == []
    candidate = aligner._candidate
    assert candidate is not None and candidate.confirmed is True
    active = aligner.snapshot()["active_throw_features"]
    assert active["pc_state"] == "complete"
    assert active["rk_state"] == "stage1_ready"
    assert active["rk_post_points"] == 6

    result = aligner.expire(
        now_arrival_pc=candidate.last_joint_arrival_pc + 0.351,
    )

    assert result is not None and result["accepted"] is True
    assert result["pc_minus_rk"] == pytest.approx(pc_minus_rk, abs=0.001)
    assert result["usable"] is True
    assert result["features"]["rk_post_points"] == 6
    assert result["features"]["xy_n"] >= 8


def test_spatial_conflict_finishes_complete_and_ready_candidate_before_identity_loss():
    aligner = OnlineThrowTimeAligner()
    rk_start = 375.0
    pc_minus_rk = 4.0
    events = []
    for arrival_pc, source, payload in _throw_events(
        rk_start=rk_start,
        pc_minus_rk=pc_minus_rk,
        pc_stop=1.0,
        rk_stop=0.75,
    ):
        physical_t = payload["t"] - rk_start - (
            pc_minus_rk if source == "pc" else 0.0
        )
        z_feature = 0.10 * math.exp(-((physical_t - 0.38) / 0.06) ** 2)
        events.append((arrival_pc, source, {**payload, "z": payload["z"] + z_feature}))

    assert _feed(aligner, events) == []
    active = aligner.snapshot()["active_throw_features"]
    assert active["pc_state"] == "complete"
    assert active["rk_state"] == "stage1_ready"

    pc_clock = aligner._source_clock["pc"]
    rk_clock = aligner._source_clock["rk"]
    rival_arrival = pc_clock.last_arrival_pc + 0.005
    rival_dt = rival_arrival - rk_clock.last_arrival_pc
    rival = {
        "t": rk_clock.last_t + rival_dt,
        "x": rk_clock.last_point.x + 4.0,
        "y": rk_clock.last_point.y - 0.2,
        "z": rk_clock.last_point.z,
    }
    result = aligner.add_rk(rival, arrival_pc=rival_arrival)

    assert result is not None and result["accepted"] is True
    assert result["pc_minus_rk"] == pytest.approx(pc_minus_rk, abs=0.001)
    assert result["features"]["pc_state"] == "complete"
    assert result["features"]["rk_state"] == "stage1_ready"
    assert result["features"]["spatial_reject_count"] == 0
    assert result["features"]["max_spatial_gap_m"] < 3.3
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 1
    assert snapshot["rejected"] == 0
    assert snapshot["reason_counts"] == {}


def test_expire_silently_discards_an_unconfirmed_provisional_seed():
    aligner = OnlineThrowTimeAligner()
    events = []
    for index in range(4):
        arrival = 380.0 + 0.04 * index
        events.extend([
            (
                arrival,
                "pc",
                {
                    "t": 380.0 + 0.04 * index,
                    "x": 0.2,
                    "y": 9.0,
                    "z": 1.0,
                },
            ),
            (
                arrival + 0.02,
                "rk",
                {
                    "t": 375.0 + 0.04 * index,
                    "x": 0.2,
                    "y": 9.0,
                    "z": 1.0,
                },
            ),
        ])
    assert _feed(aligner, events) == []
    candidate = aligner._candidate
    assert candidate is not None and candidate.confirmed is False

    assert aligner.expire(
        now_arrival_pc=candidate.last_joint_arrival_pc + 0.351,
    ) is None
    assert aligner.expire(now_arrival_pc=999.0) is None
    snapshot = aligner.snapshot()
    assert snapshot["attempts"] == 0
    assert snapshot["rejected"] == 0
    assert snapshot["reason_counts"] == {}
    assert snapshot["history"] == []
    assert snapshot["active_throw_features"] is None


def test_one_sided_or_dirty_throw_cannot_pollute_previous_offset_or_next_pair():
    aligner = OnlineThrowTimeAligner()
    initial_offset = 3.8
    _feed(
        aligner,
        _throw_events(rk_start=400.0, pc_minus_rk=initial_offset),
    )
    assert aligner.snapshot()["updates"] == 1

    # PC contains a large but finite wrong-ball jump.  It may reach the fitter,
    # but must be consumed once and rejected without changing the offset.
    dirty_results = _feed(
        aligner,
        _throw_events(
            rk_start=404.0,
            pc_minus_rk=initial_offset + 0.004,
            pc_spike_at=0.50,
        ),
    )
    dirty_rejections = [
        result for result in dirty_results if result["accepted"] is False
    ]
    assert len(dirty_rejections) == 1
    rejection = dirty_rejections[0]
    assert isinstance(rejection["reason"], str) and rejection["reason"]
    assert isinstance(rejection["features"], dict) and rejection["features"]
    after_dirty = aligner.snapshot()
    assert after_dirty["updates"] == 1
    assert after_dirty["attempts"] == 2
    assert after_dirty["rejected"] == 1
    assert after_dirty["pc_minus_rk_s"] == pytest.approx(initial_offset, abs=0.001)
    assert after_dirty["reason_counts"] == {rejection["reason"]: 1}

    # The stale one-sided RK flight expires; it must not cross-pair with the
    # next physical throw.
    final_offset = initial_offset + 0.009
    results = _feed(
        aligner,
        _throw_events(rk_start=408.0, pc_minus_rk=final_offset),
    )
    assert len(results) == 1 and results[0]["accepted"] is True
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 2
    assert snapshot["attempts"] == 3
    assert snapshot["pc_minus_rk_s"] == pytest.approx(final_offset, abs=0.001)


def test_correlated_single_point_false_bounce_is_rejected():
    rk_start = 450.0
    offset = 3.0
    events = _throw_events(rk_start=rk_start, pc_minus_rk=offset)
    corrupted = []
    for arrival_pc, source, payload in events:
        physical_t = payload["t"] - rk_start - (offset if source == "pc" else 0.0)
        z = 1.20 - 0.60 * physical_t
        if abs(physical_t - 2.0 / 3.0) < 0.018:
            z = 0.03
        corrupted.append((
            arrival_pc,
            source,
            {**payload, "z": z + (0.08 if source == "pc" else 0.0)},
        ))
    aligner = OnlineThrowTimeAligner()

    results = _feed(aligner, corrupted)

    assert len(results) == 1
    rejection = results[0]
    assert rejection["accepted"] is False
    assert rejection["reason"] == "bounce_invalid"
    assert rejection["features"]["rk_state"] == "bounce_invalid"
    assert rejection["features"]["pc_state"] == "first_bounce_incomplete"
    snapshot = aligner.snapshot()
    assert snapshot["attempts"] == 1
    assert snapshot["rejected"] == 1
    assert snapshot["reason_counts"] == {"bounce_invalid": 1}
    assert snapshot["pc_complete_flights"] == 0
    assert snapshot["rk_complete_flights"] == 0
    assert snapshot["updates"] == 0


def test_point_level_arrival_only_identifies_the_throw_and_never_enters_offset():
    true_offset = 6.125
    events = _throw_events(
        rk_start=500.0,
        pc_minus_rk=true_offset,
        pc_arrival_delay=0.005,
        rk_arrival_delay=0.095,
    )
    aligner = OnlineThrowTimeAligner()

    results = _feed(aligner, events)

    assert len(results) == 1 and results[0]["accepted"] is True
    assert results[0]["baseline_source"] == "paired_bounce"
    assert results[0]["pc_minus_rk"] == pytest.approx(true_offset, abs=0.001)
    assert 90.0 < results[0]["features"]["max_arrival_gap_ms"] <= 100.0


def test_dynamic_arrival_spatial_gate_keeps_fast_same_ball_then_xy_fit_confirms():
    aligner = OnlineThrowTimeAligner()

    results = _feed(
        aligner,
        _throw_events(
            rk_start=510.0,
            pc_minus_rk=4.0,
            y_start=20.0,
            y_speed=25.0,
            pc_arrival_delay=0.0,
            rk_arrival_delay=0.095,
        ),
    )

    assert len(results) == 1 and results[0]["accepted"] is True
    features = results[0]["features"]
    assert 2.50 < features["max_spatial_gap_m"] < 3.30
    assert features["xy_shift"] < 0.01
    assert features["xy_res_median"] < 0.01
    assert features["xy_res_p90"] < 0.01


def test_arrival_separated_tracks_cannot_be_fitted_as_one_throw():
    aligner = OnlineThrowTimeAligner()

    results = _feed(
        aligner,
        _throw_events(
            rk_start=520.0,
            pc_minus_rk=5.0,
            pc_arrival_delay=0.005,
            rk_arrival_delay=0.185,
        ),
    )

    assert not any(result["accepted"] for result in results)
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 0
    assert snapshot["pc_minus_rk_s"] is None


def test_first_bounce_past_the_floor_still_locks_onto_bounce_one():
    aligner = OnlineThrowTimeAligner()

    # Bounce #1 is at y=8.5 m, past the 8 m floor, so the ball never crosses the
    # floor while airborne -- the gate line moves to 8.5 + margin = 9.5 m, which
    # the track (starting at 10 m) does cross.  Bounce #2 sits at y~6.99 m;
    # selecting it would fabricate a valid throw, so pin the bounce we used.
    results = _feed(
        aligner,
        _two_bounce_events(
            rk_start=540.0,
            pc_minus_rk=4.0,
            y_start=10.0,
            y_speed=2.5,
        ),
    )

    assert len(results) == 1
    accepted = results[0]
    assert accepted["accepted"] is True
    assert accepted["pc_minus_rk"] == pytest.approx(4.0, abs=0.001)
    features = accepted["features"]
    assert features["pc_cross_y"] == pytest.approx(9.5, abs=0.001)
    assert features["rk_cross_y"] == pytest.approx(9.5, abs=0.001)
    assert features["pc_bounce_y"] == pytest.approx(8.5, abs=0.05)
    assert features["rk_bounce_y"] == pytest.approx(8.5, abs=0.05)
    assert aligner.snapshot()["updates"] == 1


def test_first_bounce_past_the_floor_without_enough_descent_still_rejects():
    aligner = OnlineThrowTimeAligner()

    # Same bounce #1 depth (y=8.5 m), but the track only starts at 9.4 m, so the
    # ball is never seen coming down through the 9.5 m line.  Without that
    # descent this side cannot claim it watched the same incoming flight, and
    # falling through to bounce #2 stays forbidden.
    results = _feed(
        aligner,
        _two_bounce_events(
            rk_start=540.0,
            pc_minus_rk=4.0,
            y_start=9.40,
            y_speed=1.5,
        ),
    )

    assert len(results) == 1
    rejection = results[0]
    assert rejection["accepted"] is False
    assert rejection["reason"] == "bounce_before_cross"
    assert rejection["features"]["pc_state"] == "bounce_before_cross"
    assert rejection["features"]["pc_bounce_y"] > 8.0
    assert rejection["features"]["pc_cross_y"] == pytest.approx(9.5, abs=0.001)
    assert rejection["features"]["pc_crossed_8"] is False
    assert aligner.current_offset() is None
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 0
    assert snapshot["attempts"] == 1
    assert snapshot["reason_counts"] == {"bounce_before_cross": 1}


def test_first_offset_does_not_cross_pair_an_old_one_sided_throw():
    aligner = OnlineThrowTimeAligner()
    true_offset = 2.75
    pc_only = [
        event for event in _throw_events(
            rk_start=600.0,
            pc_minus_rk=true_offset,
        )
        if event[1] == "pc"
    ]
    assert _feed(aligner, pc_only) == []

    results = _feed(
        aligner,
        _throw_events(rk_start=604.0, pc_minus_rk=true_offset),
    )

    assert len(results) == 1 and results[0]["accepted"] is True
    assert results[0]["pc_flight"] == 1
    assert results[0]["rk_flight"] == 1
    assert results[0]["pc_minus_rk"] == pytest.approx(true_offset, abs=0.001)


def test_arrival_close_but_spatially_different_tracks_fail_closed_then_recover():
    aligner = OnlineThrowTimeAligner()
    true_offset = 2.5
    events = _throw_events(rk_start=640.0, pc_minus_rk=true_offset)
    dirty = []
    for arrival_pc, source, payload in events:
        if source == "rk":
            payload = {**payload, "x": payload["x"] + 4.0}
        dirty.append((arrival_pc, source, payload))

    dirty_results = _feed(aligner, dirty)

    assert not any(result["accepted"] for result in dirty_results)
    assert aligner.snapshot()["updates"] == 0
    clean_results = _feed(
        aligner,
        _throw_events(rk_start=644.0, pc_minus_rk=true_offset),
    )
    assert len([r for r in clean_results if r["accepted"]]) == 1
    assert aligner.current_offset() == pytest.approx(true_offset, abs=0.001)


def test_three_consecutive_spatial_conflicts_kill_confirmed_track_and_keep_offset():
    aligner = OnlineThrowTimeAligner()
    previous_offset = 2.5
    clean = _feed(
        aligner,
        _throw_events(rk_start=660.0, pc_minus_rk=previous_offset),
    )
    assert len(clean) == 1 and clean[0]["accepted"] is True

    # Feed the next throw only until a real joint incoming prefix is confirmed.
    for arrival_pc, source, payload in _throw_events(
        rk_start=664.0,
        pc_minus_rk=previous_offset,
    ):
        result = (
            aligner.add_pc(**payload, arrival_pc=arrival_pc)
            if source == "pc"
            else aligner.add_rk(payload, arrival_pc=arrival_pc)
        )
        assert result is None
        if aligner._candidate is not None and aligner._candidate.confirmed:
            break
    candidate = aligner._candidate
    assert candidate is not None and candidate.confirmed is True

    pc_clock = aligner._source_clock["pc"]
    rk_clock = aligner._source_clock["rk"]
    base_arrival = max(
        pc_clock.last_arrival_pc,
        rk_clock.last_arrival_pc,
        candidate.last_joint_arrival_pc,
    ) + 0.11
    pc_dt = base_arrival - pc_clock.last_arrival_pc
    rk_arrival = base_arrival + 0.005
    rk_dt = rk_arrival - rk_clock.last_arrival_pc
    first_pc = {
        "t": pc_clock.last_t + pc_dt,
        "x": pc_clock.last_point.x + 0.3 * pc_dt,
        "y": pc_clock.last_point.y - 10.0 * pc_dt,
        "z": pc_clock.last_point.z - 1.0 * pc_dt,
    }
    first_rk = {
        "t": rk_clock.last_t + rk_dt,
        "x": rk_clock.last_point.x + 4.0,
        "y": rk_clock.last_point.y - 10.0 * rk_dt,
        "z": rk_clock.last_point.z - 1.0 * rk_dt,
    }
    assert aligner.add_pc(**first_pc, arrival_pc=base_arrival) is None
    assert aligner.add_rk(first_rk, arrival_pc=rk_arrival) is None

    second_pc = {
        "t": first_pc["t"] + 0.04,
        "x": first_pc["x"] + 0.012,
        "y": first_pc["y"] - 0.4,
        "z": first_pc["z"] - 0.04,
    }
    assert aligner.add_pc(
        **second_pc,
        arrival_pc=base_arrival + 0.04,
    ) is None
    second_rk = {
        "t": first_rk["t"] + 0.04,
        "x": first_rk["x"] + 0.012,
        "y": first_rk["y"] - 0.4,
        "z": first_rk["z"] - 0.04,
    }
    rejection = aligner.add_rk(
        second_rk,
        arrival_pc=rk_arrival + 0.04,
    )

    assert rejection is not None
    assert rejection["accepted"] is False
    assert rejection["reason"] == "joint_identity_lost"
    assert rejection["features"]["spatial_reject_streak"] == 3
    snapshot = aligner.snapshot()
    assert snapshot["updates"] == 1
    assert snapshot["pc_minus_rk_s"] == pytest.approx(previous_offset, abs=0.001)
    assert snapshot["reason_counts"] == {"joint_identity_lost": 1}


# tracker_20260901_084648: the car parked 1.33 m from the config init pose,
# bot_state had no motion and (deadlocked) no /pc_car_loc corrections, so a
# constant XY translation sat on every RK ball point and the raw-distance
# gate rejected every throw of the session by centimetres.  A constant
# translation is pose belief error, not wrong-ball evidence.
@pytest.mark.parametrize("bias", [1.36, 2.0])
def test_constant_xy_translation_is_pose_belief_error_and_still_aligns(bias):
    aligner = OnlineThrowTimeAligner()
    events = []
    for arrival_pc, source, payload in _throw_events(
        rk_start=680.0,
        pc_minus_rk=3.5,
    ):
        if source == "pc":
            payload = {**payload, "x": payload["x"] + bias}
        events.append((arrival_pc, source, payload))

    results = _feed(aligner, events)

    assert len(results) == 1
    accepted = results[0]
    assert accepted["accepted"] is True
    assert accepted["features"]["xy_n"] >= 8
    assert accepted["features"]["xy_shift"] == pytest.approx(bias, abs=0.05)
    assert accepted["features"]["xy_res_median"] < 0.05
    assert aligner.current_offset() == pytest.approx(3.5, abs=0.005)
    assert aligner.snapshot()["reason_counts"] == {}


def test_time_varying_xy_skew_is_a_wrong_track_and_still_rejects():
    aligner = OnlineThrowTimeAligner()
    events = []
    for arrival_pc, source, payload in _throw_events(
        rk_start=700.0,
        pc_minus_rk=2.5,
    ):
        if source == "pc":
            physical_t = payload["t"] - 700.0 - 2.5
            payload = {**payload, "x": payload["x"] + 2.0 * physical_t}
        events.append((arrival_pc, source, payload))

    results = _feed(aligner, events)

    assert len(results) == 1
    rejection = results[0]
    assert rejection["accepted"] is False
    assert rejection["reason"] == "spatial_residual"
    assert rejection["features"]["xy_n"] >= 8
    assert rejection["features"]["xy_res_median"] > 0.30
    assert aligner.current_offset() is None
    assert aligner.snapshot()["reason_counts"] == {"spatial_residual": 1}


def test_invalid_payloads_and_non_finite_values_are_ignored():
    aligner = OnlineThrowTimeAligner()

    assert aligner.add_rk({}, arrival_pc=1.0) is None
    assert aligner.add_rk(
        {"t": 1.0, "x": 0.0, "y": 13.0, "z": math.nan},
        arrival_pc=1.1,
    ) is None
    assert aligner.add_pc(
        t=1.0, x=0.0, y=13.0, z=math.inf, arrival_pc=1.1
    ) is None
    snapshot = aligner.snapshot()
    assert snapshot["attempts"] == 0
    assert snapshot["updates"] == 0


def test_future_timestamp_outlier_cannot_poison_the_next_complete_throw():
    aligner = OnlineThrowTimeAligner()
    true_offset = 3.25
    first_pc = next(
        event for event in _throw_events(
            rk_start=700.0,
            pc_minus_rk=true_offset,
        )
        if event[1] == "pc"
    )
    recv_pc, _, payload = first_pc
    aligner.add_pc(**payload, arrival_pc=recv_pc)
    # Coordinates are plausible, but the payload clock jumps 10,000 seconds
    # during one frame.  It must not advance the collector high-water mark.
    aligner.add_pc(
        t=payload["t"] + 10_000.0,
        x=payload["x"],
        y=payload["y"],
        z=payload["z"],
        arrival_pc=recv_pc + 0.03,
    )

    results = _feed(
        aligner,
        _throw_events(rk_start=704.0, pc_minus_rk=true_offset),
    )

    assert len(results) == 1 and results[0]["accepted"] is True
    assert results[0]["pc_minus_rk"] == pytest.approx(true_offset, abs=0.001)


def test_new_launcher_is_a_thin_opt_in_and_old_launcher_stays_default_off():
    wrapper = (ROOT / "run_tracker_with_predict.ps1").read_text(encoding="utf-8")
    launcher = (ROOT / "run_tracker.ps1").read_text(encoding="utf-8")
    runtime = (ROOT / "src" / "run_tracker.py").read_text(encoding="utf-8")

    assert 'Join-Path $PSScriptRoot "run_tracker.ps1"' in wrapper
    assert "@args -EnableRkTimeAlign" in wrapper
    assert "[switch]$EnableRkTimeAlign" in launcher
    assert 'if ($EnableRkTimeAlign) {' in launcher
    assert '$args += "--online-time-align"' in launcher
    assert 'action="store_true"' in runtime
    assert '"rk_time_alignment": _ros2_sink.time_alignment()' in runtime
    assert "from src.online_time_alignment" not in "\n".join(
        runtime.splitlines()[:150]
    )
    assert "if args.online_time_align:\n                        _ros2_sink.add_pc_ball" in runtime
    assert "} if args.online_time_align else {})," in runtime


def test_default_direct_sink_preserves_legacy_car_payload_and_close_order():
    sink = DirectRos2Sink.__new__(DirectRos2Sink)
    sink._time_aligner = None
    sink._car_pub = object()
    sink._car_count = 0
    published = []
    sink._publish = lambda publisher, payload: published.append((publisher, payload))
    sink._with_rk_timestamp = lambda _payload: pytest.fail(
        "default publish path must not enter RK timestamp conversion"
    )
    payload = {
        "topic": "car_loc",
        "t": 100.25,
        "x": 1.0,
        "y": 2.0,
        "z": 0.0,
        "yaw": 0.1,
        "yaw_valid": True,
        "tag_id": 0,
    }

    sink.publish_car_loc(payload)

    assert published == [(sink._car_pub, payload)]
    assert published[0][1] is payload
    assert "rk_timestamp" not in payload

    calls = []

    class _Stop:
        def set(self):
            calls.append("stop")

    class _Thread:
        def join(self, timeout):
            calls.append(("join", timeout))

    class _Executor:
        def shutdown(self):
            calls.append("executor_shutdown")

        def remove_node(self, node):
            calls.append(("remove_node", node))

    class _Node:
        def destroy_node(self):
            calls.append("destroy_node")

    class _Rclpy:
        @staticmethod
        def ok():
            calls.append("rclpy_ok")
            return True

        @staticmethod
        def shutdown():
            calls.append("rclpy_shutdown")

    node = _Node()
    sink._spin_stop = _Stop()
    sink._spin_thread = _Thread()
    sink._executor = _Executor()
    sink._node = node
    sink._rclpy = _Rclpy()

    sink.close()

    assert calls == [
        "stop",
        ("join", 2.0),
        "executor_shutdown",
        ("remove_node", node),
        "destroy_node",
        "rclpy_ok",
        "rclpy_shutdown",
    ]


class _OffsetStub:
    def __init__(self, value: float | None) -> None:
        self.value = value

    def current_offset(self) -> float | None:
        return self.value


class _AlignmentResultStub:
    def __init__(self) -> None:
        self.results = [
            {
                "accepted": True,
                "update": 2,
                "pc_minus_rk": 4.625,
                "delta_ms": 1.2,
                "err": 0.003,
                "n": 12,
            },
            {
                "accepted": False,
                "attempt": 3,
                "reason": "shape_residual",
            },
            {
                "accepted": True,
                "update": 3,
                "pc_minus_rk": 4.630,
                "delta_ms": 5.0,
                "err": 0.004,
                "n": 11,
            },
        ]

    def add_pc(self, **_kwargs) -> dict:
        return self.results.pop(0)


def test_each_accepted_alignment_publishes_one_reliable_offset_event(monkeypatch):
    sink = DirectRos2Sink.__new__(DirectRos2Sink)
    sink._alignment_queue = queue.Queue()
    sink._time_aligner = _AlignmentResultStub()
    sink._alignment_error = None
    sink._time_offset_pub = object()
    published = []
    sink._publish = lambda publisher, payload: published.append((publisher, payload))
    publish_times = iter((100.25, 101.25))
    monkeypatch.setattr(
        "src.run_tracker.time.perf_counter",
        lambda: next(publish_times),
    )
    payload = {"t": 1.0, "x": 0.0, "y": 8.0, "z": 1.0}
    sink._alignment_queue.put(("pc", payload, 1.1))
    sink._alignment_queue.put(("pc", payload, 1.2))
    sink._alignment_queue.put(("pc", payload, 1.3))
    sink._alignment_queue.put(None)

    sink._alignment_loop()

    assert "/pc_rk_time_offset" in ROS2_RELIABLE_TOPICS
    assert published == [
        (
            sink._time_offset_pub,
            {
                "topic": "pc_rk_time_offset",
                "update": 2,
                "pc_minus_rk_s": 4.625,
                "pc_timestamp": 100.25,
                "rk_timestamp": 95.625,
            },
        ),
        (
            sink._time_offset_pub,
            {
                "topic": "pc_rk_time_offset",
                "update": 3,
                "pc_minus_rk_s": 4.63,
                "pc_timestamp": 101.25,
                "rk_timestamp": 96.62,
            },
        ),
    ]
    assert all(
        event["pc_timestamp"]
        == pytest.approx(event["rk_timestamp"] + event["pc_minus_rk_s"])
        for _, event in published
    )


class _OverflowRecoveryStub:
    def __init__(self) -> None:
        self.calls = []

    def add_pc(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "accepted": True,
            "update": 1,
            "pc_minus_rk": 4.625,
            "delta_ms": 0.5,
            "err": 0.003,
            "n": 12,
        }


def test_oversized_payload_number_is_skipped_without_killing_worker(monkeypatch):
    aligner = OnlineThrowTimeAligner()
    huge = 10**10000
    assert aligner.add_pc(
        t=huge,
        x=0.0,
        y=8.0,
        z=1.0,
        arrival_pc=1.0,
    ) is None
    assert aligner.add_rk(
        {"t": huge, "x": 0.0, "y": 8.0, "z": 1.0},
        arrival_pc=1.0,
    ) is None

    sink = DirectRos2Sink.__new__(DirectRos2Sink)
    sink._alignment_queue = queue.Queue()
    sink._time_aligner = _OverflowRecoveryStub()
    sink._alignment_error = None
    sink._time_offset_pub = object()
    published = []
    sink._publish = lambda publisher, payload: published.append((publisher, payload))
    sink._alignment_queue.put((
        "pc",
        {"t": huge, "x": 0.0, "y": 8.0, "z": 1.0},
        1.1,
    ))
    sink._alignment_queue.put((
        "pc",
        {"t": 1.0, "x": 0.0, "y": 8.0, "z": 1.0},
        1.2,
    ))
    sink._alignment_queue.put(None)
    monkeypatch.setattr("src.run_tracker.time.perf_counter", lambda: 50.25)

    sink._alignment_loop()

    assert sink._alignment_error is None
    assert len(sink._time_aligner.calls) == 1
    assert sink._time_aligner.calls[0]["t"] == 1.0
    assert len(published) == 1
    assert published[0][1]["pc_minus_rk_s"] == 4.625


class _TimeoutThenStopQueue:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, timeout=None):
        assert timeout == pytest.approx(0.1)
        self.calls += 1
        if self.calls == 1:
            raise queue.Empty
        return None


class _ExpiryResultStub:
    def __init__(self) -> None:
        self.expire_times = []

    def expire(self, *, now_arrival_pc: float) -> dict:
        self.expire_times.append(now_arrival_pc)
        return {
            "accepted": True,
            "update": 1,
            "pc_minus_rk": 4.625,
            "delta_ms": 0.5,
            "err": 0.003,
            "n": 12,
        }


def test_idle_alignment_worker_expires_throw_and_publishes_same_result(monkeypatch):
    sink = DirectRos2Sink.__new__(DirectRos2Sink)
    sink._alignment_queue = _TimeoutThenStopQueue()
    sink._time_aligner = _ExpiryResultStub()
    sink._alignment_error = None
    sink._time_offset_pub = object()
    published = []
    sink._publish = lambda publisher, payload: published.append((publisher, payload))
    perf_times = iter((50.0, 50.25))
    monkeypatch.setattr(
        "src.run_tracker.time.perf_counter",
        lambda: next(perf_times),
    )

    sink._alignment_loop()

    assert sink._time_aligner.expire_times == [50.0]
    assert published == [
        (
            sink._time_offset_pub,
            {
                "topic": "pc_rk_time_offset",
                "update": 1,
                "pc_minus_rk_s": 4.625,
                "pc_timestamp": 50.25,
                "rk_timestamp": 45.625,
            },
        )
    ]


def test_world_topic_is_gated_and_car_gains_payload_timestamp_conversion():
    sink = DirectRos2Sink.__new__(DirectRos2Sink)
    sink._time_aligner = _OffsetStub(None)
    sink._alignment_queue = None
    sink._world_ball_pub = object()
    sink._car_pub = object()
    sink._world_ball_count = 0
    sink._car_count = 0
    published = []
    sink._publish = lambda publisher, payload: published.append((publisher, payload))

    car = {"topic": "car_loc", "t": 100.25, "x": 1.0, "y": 2.0, "z": 0.0}
    ball = {"t": 100.50, "x": 0.1, "y": 7.0, "z": 1.2}
    sink.publish_car_loc(car)
    sink.add_pc_ball(ball)
    assert published == []

    sink._time_aligner.value = 4.625
    sink.publish_car_loc(car)
    sink.add_pc_ball(ball)

    assert len(published) == 2
    assert published[0][1]["rk_timestamp"] == pytest.approx(95.625)
    assert published[1][1]["topic"] == "world_ball_loc"
    assert published[1][1]["rk_timestamp"] == pytest.approx(95.875)
    assert published[0][1]["t"] == car["t"]
    assert published[1][1]["t"] == ball["t"]
