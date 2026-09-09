"""Capture one synchronized 18F four-camera frame at 40 wide V04 IK poses."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

import capture_v04_sweet_spot_map as sweet

if str(sweet.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(sweet.PROJECT_ROOT))


JOINT_ENVELOPE_RATE = 0.90
JOINT_ENVELOPE_MARGIN_RAD = math.radians(10.0)
MIN_REACH_MARGIN_M = 0.10
MIN_RACKET_GROUND_M = 0.54
MIN_LINK_GROUND_M = 0.85

# Top to bottom.  Five x values per z layer make the Cartesian spacing close
# to square while tapering only where the V04 analytic workspace requires it.
WIDE_ROWS: tuple[tuple[float, tuple[float, ...]], ...] = (
    (1.4000000000000000, (0.8000, 0.8675, 0.9350, 1.0025, 1.0700)),
    (1.3142857142857143, (0.8000, 0.8775, 0.9550, 1.0325, 1.1100)),
    (1.2285714285714286, (0.8000, 0.8900, 0.9800, 1.0700, 1.1600)),
    (1.1428571428571428, (0.8000, 0.8975, 0.9950, 1.0925, 1.1900)),
    (1.0571428571428572, (0.8000, 0.9000, 1.0000, 1.1000, 1.2000)),
    (0.9714285714285714, (0.8300, 0.9225, 1.0150, 1.1075, 1.2000)),
    (0.8857142857142857, (0.9600, 1.0200, 1.0800, 1.1400, 1.2000)),
    (0.8000000000000000, (1.0700, 1.1025, 1.1350, 1.1675, 1.2000)),
)


def _plan() -> tuple[list[sweet.PlanPoint], dict[str, Any], Any, dict[str, Any]]:
    kin, config = sweet._load_v04_kinematics()
    limits = config["joint_limits"]
    lower = JOINT_ENVELOPE_RATE * np.radians(
        np.asarray(limits["lower_deg"], dtype=np.float64)
    )
    upper = JOINT_ENVELOPE_RATE * np.radians(
        np.asarray(limits["upper_deg"], dtype=np.float64)
    )
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    points: list[sweet.PlanPoint] = []

    for grid_z, (z_model_m, row_x) in enumerate(WIDE_ROWS):
        ordered_x = tuple(reversed(row_x)) if grid_z % 2 == 0 else row_x
        for grid_x, x_m in enumerate(ordered_x):
            candidates: list[tuple[float, float, float, np.ndarray, float]] = []
            for phi_deg in range(-30, 31):
                phi_rad = math.radians(phi_deg)
                try:
                    q5 = kin.ik_hit(
                        float(x_m),
                        float(z_model_m),
                        racket_angle=phi_rad,
                        elbow="up",
                    )
                except ValueError:
                    continue
                q = np.concatenate((np.asarray(q5, dtype=np.float64), [0.0]))
                joint_margin = float(np.min(np.minimum(q - lower, upper - q)))
                reach_margin = sweet._reach_margin(
                    kin, float(x_m), float(z_model_m), phi_rad
                )
                if (
                    joint_margin >= JOINT_ENVELOPE_MARGIN_RAD
                    and reach_margin >= MIN_REACH_MARGIN_M
                ):
                    candidates.append(
                        (abs(float(phi_deg)), -reach_margin, phi_rad, q, reach_margin)
                    )
            if not candidates:
                raise sweet.SafetyError(
                    f"wide point x={x_m:.4f}, z={z_model_m:.4f} lacks safe V04 IK"
                )
            _, _, phi_rad, q, reach_margin = min(candidates)
            fk = kin.fk_hit(q)
            tcp = np.asarray(fk["tcp"], dtype=np.float64)
            if np.linalg.norm(tcp[[0, 2]] - [x_m, z_model_m]) > 1.0e-9:
                raise sweet.SafetyError("wide-plan IK/FK round trip failed")
            if np.max(np.abs(q[[0, 4, 5]])) > 1.0e-12:
                raise sweet.SafetyError("wide plan violated J1/J5/J6=0")
            points.append(
                sweet.PlanPoint(
                    index=len(points),
                    grid_x=grid_x,
                    grid_z=grid_z,
                    x_m=float(x_m),
                    z_model_m=float(z_model_m),
                    z_ground_m=float(z_model_m - z_offset),
                    phi_deg=math.degrees(phi_rad),
                    q_command_rad=tuple(float(value) for value in q),
                    fk_tcp_model_m=tuple(float(value) for value in tcp),
                    reach_margin_m=float(reach_margin),
                )
            )

    if len(points) != 40:
        raise sweet.SafetyError(f"wide plan has {len(points)} points, expected 40")
    q_deg = np.degrees(np.asarray([point.q_command_rad for point in points]))
    summary = {
        "count": 40,
        "requested_x_range_m": [0.8, 1.2],
        "requested_z_model_range_m": [0.8, 1.4],
        "z_layers_model_m": [row[0] for row in WIDE_ROWS],
        "joint_envelope_rate": JOINT_ENVELOPE_RATE,
        "joint_envelope_margin_deg": math.degrees(JOINT_ENVELOPE_MARGIN_RAD),
        "minimum_reach_margin_m": min(point.reach_margin_m for point in points),
        "q_min_deg": q_deg.min(axis=0).tolist(),
        "q_max_deg": q_deg.max(axis=0).tolist(),
    }
    return points, summary, kin, config


def _validate_move(
    start_q: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
    kin: Any,
    config: dict[str, Any],
) -> dict[str, float]:
    limits = config["joint_limits"]
    lower = JOINT_ENVELOPE_RATE * np.radians(
        np.asarray(limits["lower_deg"], dtype=np.float64)
    )
    upper = JOINT_ENVELOPE_RATE * np.radians(
        np.asarray(limits["upper_deg"], dtype=np.float64)
    )
    path_q, path_v = sweet._cubic_samples(
        np.asarray(start_q, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.asarray(target_q, dtype=np.float64),
        duration_s,
    )
    if np.any(path_q < lower - 1.0e-12) or np.any(path_q > upper + 1.0e-12):
        raise sweet.SafetyError("wide direct path exceeded the 90% V04 envelope")
    if (
        float(np.max(np.abs(path_q[:, [0, 4, 5]])))
        > sweet.FIXED_JOINT_TOL_RAD
    ):
        raise sweet.SafetyError("wide direct path moved J1/J5/J6 outside zero tolerance")
    max_speed = float(np.max(np.abs(path_v)))
    if max_speed > sweet.MOVE_PATH_MAX_SPEED_RAD_S:
        raise sweet.SafetyError(f"wide direct path speed {max_speed:.4f}rad/s is unsafe")

    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    min_racket_ground = math.inf
    min_link_ground = math.inf
    for q in path_q:
        planar_q = q.copy()
        planar_q[[0, 4, 5]] = 0.0
        fk = kin.fk_hit(planar_q)
        min_racket_ground = min(
            min_racket_ground,
            float(fk["tcp"][2]) - z_offset - sweet.RACKET_BOUND_RADIUS_M,
        )
        min_link_ground = min(
            min_link_ground,
            float(fk["p2_xz"][1]) - z_offset,
            float(fk["p3_xz"][1]) - z_offset,
            float(fk["p4_xz"][1]) - z_offset,
        )
    if min_racket_ground < MIN_RACKET_GROUND_M:
        raise sweet.SafetyError(
            f"wide direct path racket bound is only {min_racket_ground:.3f}m above ground"
        )
    if min_link_ground < MIN_LINK_GROUND_M:
        raise sweet.SafetyError(
            f"wide direct path link origin is only {min_link_ground:.3f}m above ground"
        )
    return {
        "max_speed_rad_s": max_speed,
        "min_racket_ground_m": min_racket_ground,
        "min_link_ground_m": min_link_ground,
    }


def _fresh_joint(monitor: Any) -> dict[str, Any]:
    deadline = time.perf_counter() + 8.0
    while time.perf_counter() < deadline:
        sample = monitor.latest_joint()
        if (
            sample is not None
            and time.perf_counter() - float(sample["received_perf"])
            <= sweet.JOINT_STATE_MAX_AGE_S
        ):
            return sample
        time.sleep(0.05)
    raise sweet.SafetyError("timed out waiting for fresh /joint_states")


def _move(monitor: Any, target_q: np.ndarray, kin: Any, config: dict[str, Any]) -> float:
    sample = _fresh_joint(monitor)
    start_q = np.asarray(sample["q"], dtype=np.float64)
    if np.max(np.abs(start_q[[0, 4, 5]])) > sweet.FIXED_JOINT_TOL_RAD:
        raise sweet.SafetyError("J1/J5/J6 are not at zero before wide move")
    duration_s = sweet._move_duration(start_q, target_q)
    _validate_move(start_q, target_q, duration_s, kin, config)
    sweet.assert_runtime_graph(monitor)
    monitor.direct.move(
        start_q,
        np.zeros(6, dtype=np.float64),
        target_q,
        duration_s,
        guard=lambda: sweet.assert_runtime_graph(monitor),
    )
    time.sleep(1.5)
    actual = np.asarray(_fresh_joint(monitor)["q"], dtype=np.float64)
    error = float(np.max(np.abs(actual - target_q)))
    if error > sweet.TARGET_TOL_RAD:
        raise sweet.SafetyError(
            f"wide target error {math.degrees(error):.3f}deg exceeds 0.5deg"
        )
    return duration_s


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def run() -> Path:
    points, summary, kin, config = _plan()
    output = (
        sweet.PROJECT_ROOT
        / "arm_controller_data"
        / f"v04_raw_wide_40_{time.perf_counter_ns()}"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(
        json.dumps(
            {"summary": summary, "points": [point.json() for point in points]},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    feedforward = sweet.load_v04_feedforward()
    tau_limit = np.asarray(config["tuning"]["tau_limit_nm"], dtype=np.float64)
    monitor: Any | None = None
    motion_started = False
    returned_zero = False
    captured = 0
    try:
        monitor = sweet.start_ros_monitor(feedforward, tau_limit)
        initial = np.asarray(_fresh_joint(monitor)["q"], dtype=np.float64)
        if np.max(np.abs(initial)) > math.radians(1.0):
            raise sweet.SafetyError(
                "wide session must start at zero; actual_deg="
                + str(np.round(np.degrees(initial), 3).tolist())
            )
        sweet.assert_runtime_graph(monitor)

        from src import SyncCapture, frame_to_numpy

        with SyncCapture.from_config(str(sweet.CAMERA_CONFIG_PATH)) as cap:
            serials = list(cap.sync_serials)
            if len(serials) != 4:
                raise sweet.SafetyError(f"expected four synchronized cameras, got {serials}")
            time.sleep(2.0)
            for point in points:
                target_q = np.asarray(point.q_command_rad, dtype=np.float64)
                motion_started = True
                duration_s = _move(monitor, target_q, kin, config)
                frames = cap.get_frames(timeout_s=2.0)
                if frames is None or set(frames) != set(serials):
                    raise sweet.SafetyError("18F synchronized four-camera capture timed out")
                if any(frame.lost_packet != 0 for frame in frames.values()):
                    raise sweet.SafetyError("18F frame reported lost packets")
                exposures = [float(frame.exposure_start_pc) for frame in frames.values()]
                spread_ms = 1000.0 * (max(exposures) - min(exposures))
                if spread_ms > 10.0:
                    raise sweet.SafetyError(
                        f"18F four-camera exposure spread is {spread_ms:.3f}ms"
                    )

                point_dir = output / f"point_{point.index:02d}"
                point_dir.mkdir()
                files: dict[str, str] = {}
                for serial, frame in frames.items():
                    image = frame_to_numpy(frame, rotate_180=False)
                    if image.shape[:2] != (1536, 2048):
                        raise sweet.SafetyError("18F frame is not 2048x1536")
                    camera_dir = point_dir / serial
                    camera_dir.mkdir()
                    path = camera_dir / "0001.png"
                    if not cv2.imwrite(str(path), image):
                        raise sweet.SafetyError(f"failed to save {path}")
                    files[serial] = str(path.relative_to(output))

                exposure_perf = float(np.mean(exposures))
                sample = monitor.nearest_joint(exposure_perf)
                actual_q = np.asarray(sample["q"], dtype=np.float64)
                if np.max(np.abs(actual_q - target_q)) > sweet.TARGET_TOL_RAD:
                    raise sweet.SafetyError("arm left target during camera exposure")
                record = {
                    "index": point.index,
                    "planned": point.json(),
                    "move_duration_s": duration_s,
                    "exposure_perf": exposure_perf,
                    "sync_spread_ms": spread_ms,
                    "frame_num": {
                        serial: int(frame.frame_num) for serial, frame in frames.items()
                    },
                    "q_measured_rad": actual_q.tolist(),
                    "q_measured_deg": np.degrees(actual_q).tolist(),
                    "files": files,
                }
                _append_jsonl(output / "records.jsonl", record)
                captured += 1
                print(
                    f"CAPTURED {captured:02d}/40 x={point.x_m:.4f} "
                    f"z={point.z_model_m:.4f} q_deg="
                    f"{np.round(np.degrees(target_q), 2).tolist()}",
                    flush=True,
                )
    finally:
        if monitor is not None and motion_started:
            try:
                _move(monitor, np.zeros(6, dtype=np.float64), kin, config)
                returned_zero = True
                print("RETURNED_ZERO", flush=True)
            except BaseException as exc:
                print(f"RETURN_ZERO_FAILED: {exc}", file=sys.stderr, flush=True)
                raise
        if monitor is not None:
            try:
                monitor.close()
            finally:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()

    (output / "session.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "captured": captured,
                "returned_zero": returned_zero,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if captured != 40 or not returned_zero:
        raise sweet.SafetyError(
            f"wide session incomplete: captured={captured}, returned_zero={returned_zero}"
        )
    print(f"ALL_40_CAPTURED {output}", flush=True)
    return output


if __name__ == "__main__":
    os.environ.setdefault(
        "MVS_MVIMPORT_DIR",
        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
    )
    try:
        run()
    except BaseException as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
