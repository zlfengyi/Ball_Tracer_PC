"""Production contract tests for ``racket_impact/v3``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.racket_contact import RacketContactEstimate
from src.racket_impact import RacketBBoxObservationRecord, RacketImpactMeasurement
from test_src.measure_racket_impact import (
    ContactJob,
    ExactVideoFrame,
    _providers,
    exact_video_frames,
    exposure_center_offset_s,
    iter_rk_world_payloads,
    measure_video_jobs,
    parse_args,
    run,
    serialize_contact_estimate,
    serialize_measurement,
    validate_clock_bridge,
)


def _contact() -> RacketContactEstimate:
    return RacketContactEstimate(
        valid=True,
        failure_reason="",
        acceptance_mode="physical_consensus",
        contact_anchor_t_rk=10.0,
        contact_anchor_world_m=(0.2, 18.0, 1.05),
        contact_model="rk_ball_z_crossing_fixed_height",
        contact_height_m=1.05,
        prefix_anchor_t_rk=(9.99, 10.0, 10.01),
        prefix_spread_s=0.02,
        contact_point_spread_m=0.08,
        ball_fit_rms_m=0.20,
        first_observation_lead_s=0.10,
        approach_speed_mps=12.0,
        n_points=8,
        window_shift=0,
        trajectory=None,
    )


def test_default_providers_load_cuda_and_skip_incompatible_tensorrt(monkeypatch):
    monkeypatch.setattr(
        "test_src.measure_racket_impact.ort.get_available_providers",
        lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert _providers(None) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_v3_contact_record_separates_anchor_from_unevaluated_vision():
    record = serialize_contact_estimate(_contact(), 4, 1000.0)

    assert record["contact_anchor_status"] == "accepted"
    assert record["vision_evaluated"] is False
    assert record["status"] == "rejected"
    assert record["vz_world_mps"] is None
    assert record["acceptance_mode"] == "physical_consensus"
    assert record["prefix_spread_s"] == pytest.approx(0.02)
    assert record["contact_point_spread_m"] == pytest.approx(0.08)
    assert record["contact_window_shift"] == 0
    assert record["contact_anchor_t_pc"] == pytest.approx(1010.0)


def test_rk_world_uses_only_current_t0_plus_nested_xyz_contract():
    current = {
        "t0": 100.0,
        "world": {
            "t": [0.1, 0.2],
            "y": {
                "x": [1.0, 1.1],
                "y": [18.0, 17.8],
                "z": [1.2, 1.3],
            },
        },
    }

    assert list(iter_rk_world_payloads(current)) == [
        {"t": 100.1, "x": 1.0, "y": 18.0, "z": 1.2},
        {"t": 100.2, "x": 1.1, "y": 17.8, "z": 1.3},
    ]
    with pytest.raises(ValueError, match="rk_tracking.world must"):
        list(iter_rk_world_payloads({
            "t0": 100.0,
            "world": {"t": [0.1], "x": [1.0], "y": [18.0], "z": [1.2]},
        }))


def test_clock_bridge_and_exact_exposure_center_contracts():
    bridge, failure = validate_clock_bridge({
        "rk_clock_bridge": {
            "source": "/bot_state",
            "pc_minus_rk": 12.3,
            "mad": 0.005,
            "n": 20,
        }
    })
    assert failure == ""
    assert bridge == {
        "source": "/bot_state",
        "pc_minus_rk": 12.3,
        "mad_s": 0.005,
        "n": 20,
    }
    assert validate_clock_bridge({
        "rk_clock_bridge": {"pc_minus_rk": 1.0, "mad": 0.001, "n": 19}
    }) == (None, "rk_clock_bridge_insufficient_samples")

    config = {
        "serials": ["a", "b"],
        "camera_settings": {
            "a": {"exposure_us": 4000.0},
            "b": {"exposure_us": 8000.0},
        },
        "video_frame_mapping_exact": True,
    }
    offset = exposure_center_offset_s(config)
    base = {
        "config": config,
        "frames": [{
            "exposure_pc": 50.0,
            "video_frame_idx": 7,
            "video_mapping_exact": True,
        }],
    }
    assert offset == pytest.approx(0.003)
    assert exact_video_frames(base, offset) == [ExactVideoFrame(7, 50.003)]

    base["frames"][0]["video_mapping_exact"] = False
    with pytest.raises(ValueError, match="non-exact"):
        exact_video_frames(base, offset)


def test_video_jobs_preserve_exact_frame_id_and_fail_closed_vz(tmp_path):
    class Capture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def read(self):
            image = np.full((4, 4, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, image

        def release(self):
            pass

    class Estimator:
        prepared = []

        def prepare_frame(self, video_frame_idx, exposure_center_pc, panels):
            self.prepared.append(
                (video_frame_idx, exposure_center_pc, panels["a"])
            )
            return exposure_center_pc

        def measure(self, frames, contact_anchor_pc):
            return RacketImpactMeasurement(
                accepted=True,
                reason="accepted",
                contact_anchor_pc=contact_anchor_pc,
                window_start_pc=9.8,
                window_end_pc=9.99,
                observation_semantics=(
                    "racket_head_bbox_geometric_center_native_pixel"
                ),
                velocity_semantics=(
                    "racket_head_bbox_center_world_velocity_proxy"
                ),
                raw_bbox_observations=(),
                bundle_diagnostics={
                    "accepted": True,
                    "reason": "accepted",
                    "bbox_center_vz_world_mps": 2.5,
                    "inlier_observation_indices": [],
                },
                bbox_center_vz_world_mps=2.5,
                n_input_frames=len(frames),
                n_contact_window_frames=len(frames),
            )

    record = {"status": "rejected", "failure_reason": "", "vz_world_mps": None}
    job = ContactJob(
        record=record,
        contact_anchor_pc=10.0,
        frames=[ExactVideoFrame(1, 9.90), ExactVideoFrame(3, 9.96)],
    )
    estimator = Estimator()

    measure_video_jobs(
        tmp_path / "fake.mp4",
        ["a", "b", "c"],
        estimator,
        [job],
        capture_factory=Capture,
        panel_extractor=lambda image, _serials: {"a": int(image[0, 0, 0])},
    )

    assert estimator.prepared == [(1, 9.90, 1), (3, 9.96, 3)]
    assert record["vision_evaluated"] is True
    assert record["status"] == "accepted"
    assert record["vz_world_mps"] == pytest.approx(2.5)
    assert record["measurement"]["bbox_center_vz_world_mps"] == pytest.approx(2.5)


def test_measurement_marks_actual_bundle_inlier_bbox_indices():
    raw = tuple(
        RacketBBoxObservationRecord(
            video_frame_idx=100 + index,
            exposure_center_pc=20.0 + index * 0.01,
            time_to_anchor_s=-0.1 + index * 0.01,
            serial=f"cam_{index}",
            candidate_rank=1,
            bbox_confidence=0.9,
            bbox_native_xyxy=(10.0, 20.0, 30.0, 40.0),
            bbox_center_native_xy=(20.0, 30.0),
        )
        for index in range(3)
    )
    measurement = RacketImpactMeasurement(
        accepted=True,
        reason="accepted",
        contact_anchor_pc=20.2,
        window_start_pc=20.02,
        window_end_pc=20.19,
        observation_semantics="racket_head_bbox_geometric_center_native_pixel",
        velocity_semantics="racket_head_bbox_center_world_velocity_proxy",
        raw_bbox_observations=raw,
        bundle_diagnostics={
            "accepted": True,
            "reason": "accepted",
            "inlier_observation_indices": [0, 2],
            "bbox_center_vz_world_mps": -1.2,
        },
        bbox_center_vz_world_mps=-1.2,
    )

    serialized = serialize_measurement(measurement)

    assert serialized["bbox_center_vz_world_mps"] == pytest.approx(-1.2)
    assert [item["bundle_inlier"] for item in serialized["raw_bbox_observations"]] == [
        True,
        False,
        True,
    ]


def _outgoing_point(t: float, contact_t: float = 10.0) -> dict:
    dt = t - contact_t
    return {
        "t": t,
        "x": 0.2 + dt,
        "y": 18.0 - 12.0 * dt,
        "z": 1.05 + 3.0 * dt - 4.9 * dt * dt,
    }


def test_invalid_clock_emits_v3_and_keeps_contact_and_bundle_margins_distinct(tmp_path):
    input_path = tmp_path / "tracker.json"
    rk_path = tmp_path / "tracker_rk_tracking.json"
    video_path = tmp_path / "tracker.mp4"
    output_path = tmp_path / "impact.json"
    points = [_outgoing_point(10.10 + 0.02 * index) for index in range(8)]
    base = {
        "config": {
            "serials": ["a", "b", "c"],
            "camera_settings": {
                serial: {"exposure_us": 4000.0} for serial in ("a", "b", "c")
            },
            "rk_clock_bridge": {"pc_minus_rk": 1000.0, "mad": 0.001, "n": 19},
            "racket_impact": {
                "player_box_world_m": {
                    "x": [-3.0, 3.0],
                    "y": [13.5, 20.5],
                    "z": [0.0, 3.2],
                }
            },
        },
        "frames": [],
    }
    rk = {
        "t0": 9.0,
        "world": {
            "t": [point["t"] - 9.0 for point in points],
            "y": {
                axis: [point[axis] for point in points]
                for axis in ("x", "y", "z")
            },
        },
    }
    input_path.write_text(json.dumps(base), encoding="utf-8")
    rk_path.write_text(json.dumps(rk), encoding="utf-8")
    video_path.write_bytes(b"")

    output = run(
        parse_args(["--input", str(input_path), "--output", str(output_path)])
    )

    assert output["schema"] == "racket_impact/v3"
    assert output["frame_time_semantics"] == (
        "mosaic_group_mean_exposure_center_pc_perf_counter"
    )
    assert output["config"]["contact_reach_margin_m"] == pytest.approx(0.25)
    assert output["config"]["bundle_player_world_margin_m"] == pytest.approx(0.60)
    expected_bundle_world_volume = [
        [-3.60, 3.60],
        [12.90, 21.10],
        [-0.60, 3.80],
    ]
    for actual, expected in zip(
        output["config"]["bundle_world_volume_m"],
        expected_bundle_world_volume,
    ):
        assert actual == pytest.approx(expected)
    record = output["racket_impact"][0]
    assert record["contact_anchor_status"] == "accepted"
    assert record["vision_evaluated"] is False
    assert record["failure_reason"] == "rk_clock_bridge_insufficient_samples"
    assert record["vz_world_mps"] is None


def test_cli_contains_bbox_bundle_contract_and_no_pose_keypoint_flags():
    args = parse_args(["--input", "tracker.json"])

    assert args.bbox_confidence_min == pytest.approx(0.05)
    assert args.bbox_nms_iou == pytest.approx(0.45)
    assert args.player_roi_padding_px == 40
    assert args.bundle_min_abs_vz_mps == pytest.approx(0.30)
    assert args.bundle_player_world_margin_m == pytest.approx(0.60)
    assert args.contact_reach_margin_m == pytest.approx(0.25)
    assert args.window_before_contact_s == pytest.approx(0.125)
    assert args.window_after_contact_s == pytest.approx(0.035)
    assert not hasattr(args, "preimpact_window_s")
    assert not hasattr(args, "preimpact_guard_s")
    assert not hasattr(args, "pose_model")
    assert not hasattr(args, "face_keypoint_score_min")
