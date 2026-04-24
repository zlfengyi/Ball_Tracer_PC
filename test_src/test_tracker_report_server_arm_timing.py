from __future__ import annotations

import json
import pytest

from test_src.tracker_report_server import (
    RunArtifacts,
    TrackerReportHandler,
    TrackerReportServer,
)


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_arm_report_uses_joint_stamp_pc_ns(tmp_path):
    stem = "tracker_20260403_120000"
    tracker_json = tmp_path / f"{stem}.json"
    pc_logger_json = tmp_path / f"{stem}_pc_logger.json"

    _write_json(
        tracker_json,
        {
            "config": {
                "first_frame_exposure_pc": 100.0,
                "distance_unit": "m",
            },
            "summary": {},
            "frames": [],
            "observations": [],
            "predictions": [],
            "car_locs": [],
            "racket_observations": [],
        },
    )
    _write_json(
        pc_logger_json,
        {
            "schema": "pc_event_logger_v2",
            "saved_at_perf_s": 101.0,
            "last_save_reason": "shutdown",
            "joint_state_layout": {
                "joint_names": ["joint_1"],
                "sample_fields": [
                    "stamp_pc_ns",
                    "receipt_stamp_pc_ns",
                    "position",
                    "velocity",
                    "effort",
                ],
            },
            "joint_states_matrix": [
                [100_500_000_000, 123_456_789, [1.0], [0.1], [0.2]],
            ],
            "mit_command_layout": {
                "joint_names": [],
                "frame_fields": [
                    "stamp_pc_ns",
                    "send_index",
                    "request_id",
                    "sequence",
                    "profile_mode",
                    "execution_t_sec",
                    "is_final",
                    "commands",
                ],
                "command_fields": [
                    "motor_id",
                    "position_rad",
                    "velocity_rad_s",
                    "torque_ff_nm",
                    "computed_torque_ff_nm",
                    "kp",
                    "kd",
                    "is_hold",
                ],
            },
            "mit_command_frames_matrix": [],
            "hit_events": [],
            "control_events": [],
            "stats": {
                "joint_state_count": 1,
                "mit_command_count": 0,
                "hit_event_count": 0,
                "control_event_count": 0,
            },
        },
    )

    run = RunArtifacts(
        stem=stem,
        tracker_json=tracker_json,
        pc_logger_json=pc_logger_json,
        all_files=[tracker_json, pc_logger_json],
    )
    server = TrackerReportServer(
        ("127.0.0.1", 0),
        TrackerReportHandler,
        tracker_output_dirs=[tmp_path],
        poe_config_path=tmp_path / "missing_poe_config.json",
    )
    try:
        report = server.build_arm_report(run)
    finally:
        server.server_close()

    assert report["joint_actual"][0]["t"] == [0.5]
    assert report["joint_actual"][0]["position"] == [1.0]
    assert report["arm_summary"]["segment_start_text"] == "100.500000s"
    assert report["missing_pc_timestamps"][0]["expected_field"] == "stamp_pc_ns"


def test_build_arm_report_includes_time_sync_offset_rows(tmp_path):
    stem = "tracker_20260403_120100"
    tracker_json = tmp_path / f"{stem}.json"
    pc_logger_json = tmp_path / f"{stem}_pc_logger.json"

    _write_json(
        tracker_json,
        {
            "config": {
                "first_frame_exposure_pc": 100.0,
                "distance_unit": "m",
            },
            "summary": {},
            "frames": [],
            "observations": [],
            "predictions": [],
            "car_locs": [],
            "racket_observations": [],
        },
    )
    _write_json(
        pc_logger_json,
        {
            "schema": "pc_event_logger_v2",
            "saved_at_perf_s": 101.0,
            "last_save_reason": "shutdown",
            "joint_state_layout": {
                "joint_names": [],
                "sample_fields": [
                    "stamp_pc_ns",
                    "receipt_stamp_pc_ns",
                    "position",
                    "velocity",
                    "effort",
                ],
            },
            "joint_states_matrix": [],
            "mit_command_layout": {
                "joint_names": [],
                "frame_fields": [
                    "stamp_pc_ns",
                    "send_index",
                    "request_id",
                    "sequence",
                    "profile_mode",
                    "execution_t_sec",
                    "is_final",
                    "commands",
                ],
                "command_fields": [
                    "motor_id",
                    "position_rad",
                    "velocity_rad_s",
                    "torque_ff_nm",
                    "computed_torque_ff_nm",
                    "kp",
                    "kd",
                    "is_hold",
                ],
            },
            "mit_command_frames_matrix": [],
            "hit_events": [],
            "control_events": [],
            "time_sync_offset_events": [
                {
                    "tag": "arm-main",
                    "source_id": "rk-arm",
                    "publish_reason": "periodic",
                    "stamp_pc_ns": 100_800_000_000,
                    "current_offset_sec": 0.0125,
                    "latest_accepted_offset_sec": 0.0124,
                    "latest_offset_median_sec": 0.0123,
                    "latest_rtt_sec": 0.0032,
                    "rtt_p95_sec": 0.0041,
                    "period_sample_count": 8,
                    "period_accepted_count": 7,
                    "clock_domain": "pc",
                }
            ],
            "stats": {
                "joint_state_count": 0,
                "mit_command_count": 0,
                "hit_event_count": 0,
                "control_event_count": 0,
                "time_sync_offset_count": 1,
            },
        },
    )

    run = RunArtifacts(
        stem=stem,
        tracker_json=tracker_json,
        pc_logger_json=pc_logger_json,
        all_files=[tracker_json, pc_logger_json],
    )
    server = TrackerReportServer(
        ("127.0.0.1", 0),
        TrackerReportHandler,
        tracker_output_dirs=[tmp_path],
        poe_config_path=tmp_path / "missing_poe_config.json",
    )
    try:
        report = server.build_arm_report(run)
    finally:
        server.server_close()

    assert len(report["time_sync_offsets"]) == 1
    assert report["time_sync_offsets"][0]["rel_s"] == pytest.approx(0.8)
    assert report["time_sync_offsets"][0]["current_offset_ms"] == pytest.approx(12.5)
    assert report["time_sync_offsets"][0]["latest_rtt_ms"] == pytest.approx(3.2)
    assert report["time_sync_summary"]["available"] is True
    assert report["time_sync_summary"]["latest_rtt_p95_ms"] == pytest.approx(4.1)
