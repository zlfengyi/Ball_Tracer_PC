"""Run the fixed V04 40-point sweet-spot capture on the live 18F rig."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any

import numpy as np

import capture_v04_sweet_spot_map as sweet


FIXED_CAR = SimpleNamespace(
    x=-0.009924,
    y=3.495573,
    yaw=0.001873,
)
LOCKED_MARKER_OFFSET_TOOL_M = np.asarray(
    [-0.09424879, 0.05204759, 0.09125745], dtype=np.float64
)


def _wait_for_joint(monitor: Any, timeout_s: float = 8.0) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        sample = monitor.latest_joint()
        if sample is not None:
            age = time.perf_counter() - float(sample["received_perf"])
            if age <= sweet.JOINT_STATE_MAX_AGE_S:
                return sample
        time.sleep(0.05)
    raise sweet.SafetyError("timed out waiting for fresh /joint_states")


def _wait_position_settled(
    monitor: Any,
    target_q: np.ndarray,
    timeout_s: float,
    command_started_perf: float,
) -> np.ndarray:
    """Settle from position error and position span; raw velocities are ignored."""
    target_q = np.asarray(target_q, dtype=np.float64)
    deadline = time.perf_counter() + timeout_s
    samples: deque[dict[str, Any]] = deque()
    last_header = -1
    next_guard = -math.inf
    latest_q: np.ndarray | None = None

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if now >= next_guard:
            sweet._command_guard(monitor, command_started_perf)
            next_guard = now + 0.25
        sample = monitor.latest_joint()
        if sample is None or now - float(sample["received_perf"]) > sweet.JOINT_STATE_MAX_AGE_S:
            raise sweet.SafetyError("/joint_states became stale")
        q = np.asarray(sample["q"], dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise sweet.SafetyError("invalid joint position sample")
        if np.max(np.abs(q[[0, 4, 5]])) > sweet.FIXED_JOINT_TOL_RAD:
            raise sweet.SafetyError("J1/J5/J6 left the zero tolerance")
        latest_q = q

        header = int(sample["header_stamp_ns"])
        if header < last_header:
            raise sweet.SafetyError("/joint_states header moved backwards")
        if header > last_header:
            samples.append({"received_perf": float(sample["received_perf"]), "q": q})
            last_header = header
        while samples and samples[0]["received_perf"] < now - sweet.SETTLE_WINDOW_S:
            samples.popleft()

        if len(samples) >= sweet.SETTLE_MIN_DISTINCT_HEADERS:
            observed_s = float(samples[-1]["received_perf"] - samples[0]["received_perf"])
            positions = np.asarray([item["q"] for item in samples], dtype=np.float64)
            if (
                observed_s >= sweet.SETTLE_MIN_OBSERVED_S
                and float(np.max(np.abs(q - target_q))) <= sweet.TARGET_TOL_RAD
                and float(np.max(np.ptp(positions, axis=0))) <= sweet.SETTLE_SPAN_RAD
            ):
                return q.copy()
        time.sleep(0.05)

    error_deg = math.inf
    if latest_q is not None:
        error_deg = math.degrees(float(np.max(np.abs(latest_q - target_q))))
    raise sweet.SafetyError(
        f"arm did not settle by position; latest error_deg={error_deg:.3f}"
    )


def _move_validated(
    monitor: Any,
    target_q: np.ndarray,
    kin: Any,
    config: dict[str, Any],
    command_started_perf: float,
) -> float:
    sweet._command_guard(monitor, command_started_perf)
    latest = _wait_for_joint(monitor)
    start_q = np.asarray(latest["q"], dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    start_v = np.zeros(6, dtype=np.float64)
    duration_s = sweet._move_duration(start_q, target_q)
    sweet.validate_direct_cubic(
        start_q,
        start_v,
        target_q,
        duration_s,
        kin,
        config,
        include_start_velocity_envelope=False,
    )
    monitor.direct.move(
        start_q,
        start_v,
        target_q,
        duration_s,
        guard=lambda: sweet._command_guard(monitor, command_started_perf),
    )
    _wait_position_settled(
        monitor,
        target_q,
        timeout_s=duration_s + 10.0,
        command_started_perf=command_started_perf,
    )
    return duration_s


def _nearest_anchor_candidate(
    image: np.ndarray,
    anchor_uv: tuple[float, float],
) -> tuple[list[sweet.MarkerCandidate], int, float]:
    candidates = sweet.find_marker_candidates(image, anchor_uv)
    if not candidates:
        return [], 0, math.inf
    anchor = np.asarray(anchor_uv, dtype=np.float64)
    nearest = min(
        candidates,
        key=lambda item: float(np.linalg.norm(np.asarray(item.uv) - anchor)),
    )
    distance = float(np.linalg.norm(np.asarray(nearest.uv) - anchor))
    return [nearest], len(candidates), distance


def _measure_fixed_burst(
    groups: list[dict[str, Any]],
    serials: list[str],
    cameras: dict[str, sweet.CameraModel],
    monitor: Any,
    kin: Any,
    z_offset: float,
    target_q: np.ndarray,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    for group in groups:
        sample = monitor.nearest_joint(float(group["exposure_perf"]))
        q = np.asarray(sample["q"], dtype=np.float64)
        if np.max(np.abs(q[[0, 4, 5]])) > sweet.FIXED_JOINT_TOL_RAD:
            raise sweet.SafetyError("fixed joint left zero during exposure")
        if np.max(np.abs(q - target_q)) > sweet.TARGET_TOL_RAD:
            raise sweet.SafetyError("arm left the requested target during exposure")

        expected = sweet._expected_sweet_world_mm(
            q,
            FIXED_CAR,
            kin,
            z_offset,
            LOCKED_MARKER_OFFSET_TOOL_M,
        )
        anchors = {
            serial: tuple(
                float(value) for value in sweet.project_raw(cameras[serial], expected)
            )
            for serial in serials
        }
        selected: dict[str, list[sweet.MarkerCandidate]] = {}
        candidate_counts: dict[str, int] = {}
        anchor_distances: dict[str, float] = {}
        for serial in serials:
            chosen, count, distance = _nearest_anchor_candidate(
                group["images"][serial], anchors[serial]
            )
            selected[serial] = chosen
            candidate_counts[serial] = count
            anchor_distances[serial] = distance

        frame_record: dict[str, Any] = {
            "burst_index": int(group["burst_index"]),
            "frame_num": group["frame_num"],
            "exposure_perf": float(group["exposure_perf"]),
            "sync_spread_ms": float(group["sync_spread_ms"]),
            "q_measured_rad": q.tolist(),
            "candidate_counts": candidate_counts,
            "nearest_anchor_distance_px": anchor_distances,
            "anchor_uv": anchors,
            "accepted": False,
        }
        fit = sweet.solve_marker_4cam(
            selected,
            cameras,
            expected,
            sweet.MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM,
        )
        if fit is None:
            frame_record["failure"] = "nearest black dot failed reprojection/LOO/heldout gates"
        else:
            frame_record.update(
                {
                    "accepted": True,
                    "pixels": fit.pixels,
                    "sweet_world_mm": fit.point.xyz_mm.tolist(),
                    "reproj_px": fit.point.radial_errors_px,
                    "reproj_rms_px": fit.point.rms_px,
                    "reproj_max_px": fit.point.max_px,
                    "loo_delta_mm": fit.loo_delta_mm,
                    "loo_heldout_px": fit.loo_heldout_px,
                    "expected_distance_mm": fit.expected_distance_mm,
                }
            )
            accepted.append({"fit": fit, "q": q, "group": group})
        frame_records.append(frame_record)

    if len(accepted) < sweet.BURST_MIN_GOOD:
        raise sweet.SafetyError(
            f"only {len(accepted)}/{sweet.BURST_COUNT} synchronized bursts passed"
        )
    points = np.asarray([item["fit"].point.xyz_mm for item in accepted])
    pairwise = np.linalg.norm(
        points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2
    )
    spread_mm = float(np.max(pairwise))
    if spread_mm >= sweet.BURST_MAX_SPREAD_MM:
        raise sweet.SafetyError(
            f"black-dot burst 3D spread {spread_mm:.3f}mm exceeds gate"
        )
    representative = accepted[int(np.argmin(np.sum(pairwise, axis=1)))]
    representative_fit = representative["fit"]
    representative_q = representative["q"]
    sweet_world_mm = representative_fit.point.xyz_mm
    return {
        "capture_perf": float(representative["group"]["exposure_perf"]),
        "representative_burst_index": int(representative["group"]["burst_index"]),
        "q_measured_rad": representative_q.tolist(),
        "q_measured_deg": np.degrees(representative_q).tolist(),
        "sweet_world_m": (sweet_world_mm / 1000.0).tolist(),
        "sweet_car_m": sweet._sweet_in_car_m(sweet_world_mm, FIXED_CAR),
        "burst_good": len(accepted),
        "burst_spread_mm": spread_mm,
        "worst_reproj_px": max(item["fit"].point.max_px for item in accepted),
        "worst_loo_mm": max(
            max(item["fit"].loo_delta_mm.values()) for item in accepted
        ),
        "worst_heldout_px": max(
            max(item["fit"].loo_heldout_px.values()) for item in accepted
        ),
        "frames": frame_records,
    }


def run() -> Path:
    plan, summary, kin, config = sweet.generate_plan()
    serials, cameras = sweet.load_camera_models()
    feedforward = sweet.load_v04_feedforward()
    tau_limit_nm = np.asarray(config["tuning"]["tau_limit_nm"], dtype=np.float64)
    output = (
        sweet.PROJECT_ROOT
        / "arm_controller_data"
        / f"v04_live_40_{time.perf_counter_ns()}"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(
        json.dumps(
            {"summary": summary, "points": [point.json() for point in plan]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monitor: Any | None = None
    command_started_perf: float | None = None
    motion_started = False
    results: list[dict[str, Any]] = []
    primary_error: BaseException | None = None
    return_error: BaseException | None = None
    try:
        monitor = sweet.start_ros_monitor(feedforward, tau_limit_nm)
        _wait_for_joint(monitor)
        sweet.assert_runtime_graph(monitor)
        command_started_perf = time.perf_counter()
        first_target = np.asarray(plan[0].q_command_rad, dtype=np.float64)
        _wait_position_settled(
            monitor,
            first_target,
            timeout_s=5.0,
            command_started_perf=command_started_perf,
        )

        from src import SyncCapture

        z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
        with SyncCapture.from_config(str(sweet.CAMERA_CONFIG_PATH)) as cap:
            if cap.sync_serials != serials:
                raise sweet.SafetyError(
                    f"unexpected synchronized cameras: {cap.sync_serials}"
                )
            time.sleep(2.0)
            for point in plan:
                target_q = np.asarray(point.q_command_rad, dtype=np.float64)
                motion_started = True
                duration_s = _move_validated(
                    monitor, target_q, kin, config, command_started_perf
                )
                groups = sweet.capture_synced_burst(cap, serials)
                measured = _measure_fixed_burst(
                    groups,
                    serials,
                    cameras,
                    monitor,
                    kin,
                    z_offset,
                    target_q,
                )
                record = {
                    "index": point.index,
                    "planned": point.json(),
                    "move_duration_s": duration_s,
                    **measured,
                }
                sweet._append_jsonl(output / "results.jsonl", record)
                results.append(record)
                sweet._write_csv(output / "joints_to_sweet_spot.csv", results)
                print(
                    f"[{point.index + 1:02d}/40] q_deg="
                    f"{np.round(record['q_measured_deg'], 3).tolist()} sweet_world_m="
                    f"{np.round(record['sweet_world_m'], 5).tolist()} "
                    f"bursts={record['burst_good']}/4",
                    flush=True,
                )
        if len(results) != 40:
            raise sweet.SafetyError(f"session ended with {len(results)}/40 accepted points")
    except BaseException as exc:
        primary_error = exc
    finally:
        if monitor is not None and motion_started:
            try:
                if command_started_perf is None:
                    raise sweet.SafetyError("direct command session start time is missing")
                _move_validated(
                    monitor,
                    np.zeros(6, dtype=np.float64),
                    kin,
                    config,
                    command_started_perf,
                )
            except BaseException as exc:
                return_error = exc
                print(
                    f"EMERGENCY: validated direct return to zero failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if monitor is not None:
            try:
                monitor.close()
            finally:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()

    session = {
        "status": "complete" if primary_error is None and return_error is None else "failed",
        "result_count": len(results),
        "fixed_car": {"x_m": FIXED_CAR.x, "y_m": FIXED_CAR.y, "yaw_rad": FIXED_CAR.yaw},
        "marker_offset_tool_m": LOCKED_MARKER_OFFSET_TOOL_M.tolist(),
        "returned_to_zero": motion_started and return_error is None,
    }
    if primary_error is not None:
        session["error"] = f"{type(primary_error).__name__}: {primary_error}"
    if return_error is not None:
        session["return_to_zero_error"] = f"{type(return_error).__name__}: {return_error}"
    (output / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if return_error is not None:
        raise sweet.SafetyError(
            "session is unsafe because validated direct return to zero failed"
        ) from return_error
    if primary_error is not None:
        raise primary_error
    return output


def main() -> int:
    try:
        output = run()
        print(f"Complete: {output}")
        return 0
    except (sweet.SafetyError, FileNotFoundError) as exc:
        print(f"SAFETY STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
