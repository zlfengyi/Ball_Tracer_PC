# -*- coding: utf-8 -*-
"""Measure the V04 fixed black racket-centre marker around report final HTs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


FRAME_RADIUS = 4
MARKER_ROI_RADIUS_PX = 250
MARKER_CANDIDATES_PER_CAMERA = 12
MARKER_ASSOCIATION_PX = 20.0
MARKER_MAX_REPROJ_PX = 2.5
MARKER_MAX_LOO_MM = 10.0
MARKER_MAX_HELDOUT_PX = 6.0
# Automatic post-run measurement has no human overlay review. 110 mm retains the
# validated marker in recent sessions while rejecting the known racket-rim alias.
MARKER_MAX_EXPECTED_DISTANCE_MM = 110.0
ARM_STATE_MAX_GAP_S = 0.1
CAR_LOC_MAX_GAP_S = 0.5


@dataclass(frozen=True)
class CameraModel:
    serial: str
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    rvec: np.ndarray
    P: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class MarkerCandidate:
    uv: tuple[float, float]
    score: float
    area: int
    bbox_xywh: tuple[int, int, int, int]


@dataclass(frozen=True)
class PointFit:
    xyz_mm: np.ndarray
    rms_px: float
    max_px: float


@dataclass(frozen=True)
class MarkerFit:
    point: PointFit
    pixels: dict[str, tuple[float, float]]
    loo_delta_mm: dict[str, float]
    loo_heldout_px: dict[str, float]
    expected_distance_mm: float


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _session_path(raw: Any, tracker_path: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (tracker_path.parent / path).resolve()


def _load_cameras(
    calibration_path: Path, serials: list[str]
) -> dict[str, CameraModel]:
    raw = _load_json(calibration_path).get("cameras")
    if not isinstance(raw, dict) or set(raw) != set(serials) or len(serials) != 4:
        raise ValueError("session must use exactly the four calibrated 18F cameras")
    cameras: dict[str, CameraModel] = {}
    for serial in serials:
        item = raw[serial]
        K = np.asarray(item["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(item["D"], dtype=np.float64).reshape(-1)
        R = np.asarray(item["R_world"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(item["t_world"], dtype=np.float64).reshape(3, 1)
        image_size = tuple(int(v) for v in item["image_size"])
        if D.shape != (5,) or image_size != (2048, 1536):
            raise ValueError(f"unexpected calibration shape for {serial}")
        if not all(np.all(np.isfinite(v)) for v in (K, D, R, t)):
            raise ValueError(f"nonfinite camera calibration for {serial}")
        cameras[serial] = CameraModel(
            serial=serial,
            K=K,
            D=D,
            R=R,
            t=t,
            rvec=cv2.Rodrigues(R)[0],
            P=K @ np.hstack([R, t]),
            image_size=image_size,
        )
    return cameras


def _project_raw(camera: CameraModel, xyz_mm: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(xyz_mm, dtype=np.float64).reshape(1, 3),
        camera.rvec,
        camera.t,
        camera.K,
        camera.D,
    )
    return projected.reshape(2)


def _dlt(
    observations: dict[str, tuple[float, float]],
    cameras: dict[str, CameraModel],
) -> np.ndarray:
    rows = []
    for serial, uv in observations.items():
        camera = cameras[serial]
        undistorted = cv2.undistortPoints(
            np.asarray([[uv]], dtype=np.float64), camera.K, camera.D, P=camera.K
        ).reshape(2)
        u, v = undistorted
        rows.extend([u * camera.P[2] - camera.P[0], v * camera.P[2] - camera.P[1]])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        raise ValueError("degenerate marker triangulation")
    return homogeneous[:3] / homogeneous[3]


def _triangulate_refined(
    observations: dict[str, tuple[float, float]],
    cameras: dict[str, CameraModel],
) -> PointFit:
    initial = _dlt(observations, cameras)

    def residual(xyz: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                _project_raw(cameras[serial], xyz) - np.asarray(uv)
                for serial, uv in observations.items()
            ]
        )

    optimized = least_squares(
        residual,
        initial,
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=100,
    )
    xyz = np.asarray(optimized.x, dtype=np.float64)
    if not optimized.success or not np.all(np.isfinite(xyz)):
        raise ValueError("marker refinement failed")
    errors = []
    for serial, uv in observations.items():
        camera = cameras[serial]
        depth = float((camera.R @ xyz.reshape(3, 1) + camera.t)[2, 0])
        if depth <= 0.0:
            raise ValueError("marker is behind a camera")
        errors.append(float(np.linalg.norm(_project_raw(camera, xyz) - np.asarray(uv))))
    values = np.asarray(errors, dtype=np.float64)
    return PointFit(
        xyz_mm=xyz,
        rms_px=float(np.sqrt(np.mean(values * values))),
        max_px=float(np.max(values)),
    )


def _component_candidates(
    image: np.ndarray,
    anchor_uv: tuple[float, float],
    *,
    radius: int,
    thresholds: list[float],
    anchor_limit_px: float | None,
    result_limit: int,
    prefer_near: bool = False,
    dedupe_px: float = 5.0,
) -> list[MarkerCandidate]:
    height, width = image.shape[:2]
    anchor_x, anchor_y = anchor_uv
    if not np.all(np.isfinite([anchor_x, anchor_y])):
        return []
    x0 = max(0, int(math.floor(anchor_x - radius)))
    y0 = max(0, int(math.floor(anchor_y - radius)))
    x1 = min(width, int(math.ceil(anchor_x + radius)))
    y1 = min(height, int(math.ceil(anchor_y + radius)))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = float(np.median(gray))
    ys, xs = np.indices(gray.shape)
    found: list[MarkerCandidate] = []
    for threshold in thresholds:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (gray < threshold).astype(np.uint8), 8
        )
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if not (8 <= area <= 1200 and 3 <= w <= 50 and 3 <= h <= 50):
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            fill = area / float(w * h)
            if aspect > 2.4 or fill < 0.30:
                continue
            component = labels == label
            weights = np.where(component, np.maximum(background - gray, 1.0), 0.0)
            mass = float(weights.sum())
            if mass <= 0.0:
                continue
            u = x0 + float((weights * xs).sum() / mass)
            v = y0 + float((weights * ys).sum() / mass)
            distance = math.hypot(u - anchor_x, v - anchor_y)
            if anchor_limit_px is not None and distance > anchor_limit_px:
                continue
            contrast = float(np.median(background - gray[component]))
            scale = 25.0 if anchor_limit_px is not None else 140.0
            found.append(
                MarkerCandidate(
                    uv=(u, v),
                    score=(
                        max(contrast, 1.0)
                        * fill
                        * min(area, 250)
                        / (1.0 + (distance / scale) ** 2)
                    ),
                    area=area,
                    bbox_xywh=(x0 + x, y0 + y, w, h),
                )
            )
    found.sort(
        key=(
            (lambda item: (math.hypot(item.uv[0] - anchor_x, item.uv[1] - anchor_y), -item.score))
            if prefer_near
            else (lambda item: -item.score)
        )
    )
    deduped: list[MarkerCandidate] = []
    for item in found:
        if all(
            np.linalg.norm(np.subtract(item.uv, old.uv)) > dedupe_px
            for old in deduped
        ):
            deduped.append(item)
        if len(deduped) == result_limit:
            break
    return deduped


def _marker_candidates(
    image: np.ndarray, anchor_uv: tuple[float, float]
) -> list[MarkerCandidate]:
    if not np.all(np.isfinite(anchor_uv)):
        return []
    height, width = image.shape[:2]
    if (
        anchor_uv[0] < -55
        or anchor_uv[0] >= width + 55
        or anchor_uv[1] < -55
        or anchor_uv[1] >= height + 55
    ):
        return []
    x0 = max(0, int(anchor_uv[0]) - 55)
    y0 = max(0, int(anchor_uv[1]) - 55)
    x1 = min(width, int(anchor_uv[0]) + 56)
    y1 = min(height, int(anchor_uv[1]) + 56)
    local_gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    adaptive = (
        list(np.percentile(local_gray, [5, 10, 15, 20, 25, 30]))
        if local_gray.size
        else []
    )
    local = _component_candidates(
        image,
        anchor_uv,
        radius=55,
        thresholds=sorted(
            set(float(v) for v in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56] + adaptive)
        ),
        anchor_limit_px=45.0,
        result_limit=12,
        prefer_near=True,
        dedupe_px=3.0,
    )
    broad = _component_candidates(
        image,
        anchor_uv,
        radius=MARKER_ROI_RADIUS_PX,
        thresholds=[8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
        anchor_limit_px=None,
        result_limit=32,
    )
    combined = [*local, *broad]
    combined.sort(
        key=lambda item: float(
            np.linalg.norm(np.asarray(item.uv) - np.asarray(anchor_uv))
        )
    )
    deduped: list[MarkerCandidate] = []
    for item in combined:
        if all(np.linalg.norm(np.subtract(item.uv, old.uv)) > 3.0 for old in deduped):
            deduped.append(item)
        if len(deduped) == MARKER_CANDIDATES_PER_CAMERA:
            break
    return deduped


def _solve_marker_4cam(
    candidates: dict[str, list[MarkerCandidate]],
    cameras: dict[str, CameraModel],
    expected_xyz_mm: np.ndarray,
) -> MarkerFit | None:
    serials = list(cameras)
    if any(not candidates.get(serial) for serial in serials):
        return None
    proposed: dict[tuple[int, ...], float] = {}
    for left_index, left_serial in enumerate(serials):
        for right_serial in serials[left_index + 1 :]:
            for left in candidates[left_serial]:
                for right in candidates[right_serial]:
                    try:
                        seed = _dlt(
                            {left_serial: left.uv, right_serial: right.uv}, cameras
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if (
                        float(np.linalg.norm(seed - expected_xyz_mm))
                        > 1.5 * MARKER_MAX_EXPECTED_DISTANCE_MM
                    ):
                        continue
                    choice: list[int] = []
                    total_distance = 0.0
                    for serial in serials:
                        prediction = _project_raw(cameras[serial], seed)
                        distances = [
                            float(np.linalg.norm(prediction - np.asarray(item.uv)))
                            for item in candidates[serial]
                        ]
                        best_index = int(np.argmin(distances))
                        if distances[best_index] > MARKER_ASSOCIATION_PX:
                            break
                        choice.append(best_index)
                        total_distance += distances[best_index]
                    if len(choice) == len(serials):
                        key = tuple(choice)
                        proposed[key] = min(proposed.get(key, math.inf), total_distance)

    best: tuple[tuple[float, float, float, float], MarkerFit] | None = None
    for choice, _ in sorted(proposed.items(), key=lambda item: item[1])[:32]:
        pixels = {
            serial: candidates[serial][candidate_index].uv
            for serial, candidate_index in zip(serials, choice)
        }
        try:
            fit = _triangulate_refined(pixels, cameras)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fit.max_px > MARKER_MAX_REPROJ_PX:
            continue
        expected_distance = float(np.linalg.norm(fit.xyz_mm - expected_xyz_mm))
        if expected_distance > MARKER_MAX_EXPECTED_DISTANCE_MM:
            continue
        loo_delta: dict[str, float] = {}
        heldout: dict[str, float] = {}
        try:
            for dropped in serials:
                fit3 = _triangulate_refined(
                    {serial: uv for serial, uv in pixels.items() if serial != dropped},
                    cameras,
                )
                loo_delta[dropped] = float(np.linalg.norm(fit3.xyz_mm - fit.xyz_mm))
                heldout[dropped] = float(
                    np.linalg.norm(
                        _project_raw(cameras[dropped], fit3.xyz_mm) - pixels[dropped]
                    )
                )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if max(loo_delta.values()) >= MARKER_MAX_LOO_MM:
            continue
        if max(heldout.values()) > MARKER_MAX_HELDOUT_PX:
            continue
        score_sum = sum(
            candidates[serial][candidate_index].score
            for serial, candidate_index in zip(serials, choice)
        )
        marker = MarkerFit(
            point=fit,
            pixels=pixels,
            loo_delta_mm=loo_delta,
            loo_heldout_px=heldout,
            expected_distance_mm=expected_distance,
        )
        rank = (expected_distance, max(loo_delta.values()), fit.rms_px, -score_sum)
        if best is None or rank < best[0]:
            best = (rank, marker)
    return None if best is None else best[1]


def _interpolate(
    rows: list[dict],
    target: float,
    time_key: str,
    value_keys: tuple[str, ...],
    max_gap_s: float,
) -> dict | None:
    times = np.asarray([float(row[time_key]) for row in rows], dtype=np.float64)
    right = int(np.searchsorted(times, target))
    if right <= 0 or right >= len(rows):
        return None
    before, after = rows[right - 1], rows[right]
    span = float(after[time_key]) - float(before[time_key])
    if not (0.0 < span <= max_gap_s):
        return None
    fraction = (target - float(before[time_key])) / span
    result: dict[str, Any] = {"gap_s": span}
    for key in value_keys:
        a = np.asarray(before[key], dtype=np.float64)
        b = np.asarray(after[key], dtype=np.float64)
        value = a + fraction * (b - a)
        if not np.all(np.isfinite(value)):
            return None
        result[key] = float(value) if value.ndim == 0 else value
    return result


def _expected_world_mm(
    pc_elapsed: float,
    arm_states: list[dict],
    car_locs: list[dict],
    rk_t0: float,
    rk_to_pc_bias: float,
    arm_z_offset_m: float,
) -> np.ndarray | None:
    rk_absolute = rk_t0 + pc_elapsed - rk_to_pc_bias
    state = _interpolate(
        arm_states, rk_absolute, "t", ("tcp",), ARM_STATE_MAX_GAP_S
    )
    car = _interpolate(
        car_locs, pc_elapsed, "elapsed_s", ("x", "y", "z", "yaw"), CAR_LOC_MAX_GAP_S
    )
    if state is None or car is None:
        return None
    local = np.asarray(state["tcp"], dtype=np.float64)
    cosine, sine = math.cos(car["yaw"]), math.sin(car["yaw"])
    relative_world = np.asarray(
        [
            cosine * local[0] - sine * local[1],
            sine * local[0] + cosine * local[1],
            local[2] - arm_z_offset_m,
        ],
        dtype=np.float64,
    )
    return 1000.0 * (
        np.asarray([car["x"], car["y"], car["z"]], dtype=np.float64)
        + relative_world
    )


def _grid_panels(
    image: np.ndarray, serials: list[str], cameras: dict[str, CameraModel]
) -> dict[str, np.ndarray]:
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError(f"unexpected grid video shape {image.shape[:2]}")
    half_height, half_width = image.shape[0] // 2, image.shape[1] // 2
    panels: dict[str, np.ndarray] = {}
    for index, serial in enumerate(serials):
        x = (index % 2) * half_width
        y = (index // 2) * half_height
        panel = image[y : y + half_height, x : x + half_width]
        panels[serial] = cv2.resize(
            panel, cameras[serial].image_size, interpolation=cv2.INTER_LINEAR
        )
    return panels


def _measurement_context(
    tracker_path: Path,
    arm_path: Path,
    rk_path: Path,
    tables_path: Path,
) -> tuple[dict, dict, dict, list[str], float, float, float, list[dict]]:
    tracker = _load_json(tracker_path)
    arm = _load_json(arm_path)
    rk = _load_json(rk_path)
    tables = _load_json(tables_path)
    config = tracker.get("config") or {}
    if str(arm.get("car", "")).lower() != "v04" or "v04" not in Path(
        str(config.get("car_config_path", ""))
    ).stem.lower():
        raise ValueError("fixed black-marker HT measurement is supported only for V04")
    video_output = config.get("video_output") or {}
    if (
        video_output.get("layout") != "grid"
        or int(video_output.get("grid_cols", 0)) != 2
        or int(video_output.get("grid_rows", 0)) != 2
        or config.get("video_frame_mapping_exact") is not True
    ):
        raise ValueError("fixed black-marker HT measurement requires exact 2x2 grid video")
    serials = list(video_output.get("serial_order") or config.get("serials") or [])
    if len(serials) != 4 or len(set(serials)) != 4:
        raise ValueError("fixed black-marker HT measurement requires four unique cameras")
    settings = config.get("camera_settings") or {}
    exposures = [float(settings[serial]["exposure_us"]) for serial in serials]
    if not exposures or not all(math.isfinite(v) and v > 0.0 for v in exposures):
        raise ValueError("missing finite camera exposure")
    if max(exposures) - min(exposures) > 1e-6:
        raise ValueError("four cameras must use the same exposure time")
    exposure_center_offset_s = 0.5e-6 * exposures[0]

    if tables.get("script_error") or tables.get("tab_errors"):
        raise ValueError("report page snapshot did not complete cleanly")
    align = tables.get("align") or {}
    auto = align.get("auto") or {}
    required_points = 30 if auto.get("windowSource") == "scan" else 15
    alignment_trusted = (
        auto.get("windowSource") is not None
        and _finite(auto.get("bias"))
        and _finite(auto.get("err"))
        and float(auto["err"]) <= 0.08
        and int(auto.get("n") or 0) >= required_points
        and int(auto.get("flights") or 0) >= int(auto.get("requiredFlights") or 3)
        and not (
            auto.get("windowSource") == "scan"
            and _finite(auto.get("margin"))
            and float(auto["margin"]) < 1.35
        )
    )
    if not alignment_trusted:
        raise ValueError("report PC/RK alignment is not trusted")
    time_map = align.get("timeMap") or {}
    if not _finite(time_map.get("scale")) or abs(float(time_map["scale"]) - 1.0) > 1e-12:
        raise ValueError("report time map must use scale=1")
    if not _finite(time_map.get("bias")):
        raise ValueError("report time map has no finite bias")
    baseline_bias = float(time_map["bias"])
    contract = tables.get("arm_contract") or {}
    if contract.get("schema") != "arm_final_ht/v4":
        raise ValueError("report tables lack arm_final_ht/v4 contract")
    phase_policy = contract.get("zPhasePolicy") or {}
    if (
        phase_policy.get("appliesTo") != "all_pc_sampling"
        or phase_policy.get("rkUse") != "global_baseline"
        or not _finite(phase_policy.get("maxAbsOffsetMs"))
        or abs(float(phase_policy["maxAbsOffsetMs"]) - 100.0) > 1e-9
    ):
        raise ValueError("report arm contract has the wrong zPhase policy")
    max_offset_s = 0.001 * float(phase_policy["maxAbsOffsetMs"])
    rk_t0 = float(rk["t0"])
    if not _finite(contract.get("rkT0")) or abs(float(contract["rkT0"]) - rk_t0) > 1e-6:
        raise ValueError("report arm contract and RK sidecar use different t0")
    calibration = contract.get("calibration") or {}
    if not _finite(calibration.get("zOff")):
        raise ValueError("report arm contract has no calibrated z offset")
    z_offset = float(calibration["zOff"])
    targets = []
    for row in contract.get("rows", []):
        phase = row.get("zPhase") or {}
        if (
            row.get("accepted") is not True
            or row.get("finalMismatch") is True
            or not _finite(row.get("finalHtRkAbs"))
            or not _finite(row.get("finalHtPcBaselineElapsed"))
            or phase.get("usable") is not True
            or not _finite(phase.get("deltaS"))
            or not _finite(row.get("finalHtPcSampleElapsed"))
        ):
            continue
        delta_s = float(phase["deltaS"])
        if abs(delta_s) > max_offset_s + 1e-12:
            raise ValueError("visual HT zPhase exceeds the contract limit")
        expected_baseline = float(row["finalHtRkAbs"]) - rk_t0 + baseline_bias
        if abs(float(row["finalHtPcBaselineElapsed"]) - expected_baseline) > 1e-6:
            raise ValueError("visual HT baseline does not match the global PC/RK map")
        expected_sample = expected_baseline + delta_s
        if abs(float(row["finalHtPcSampleElapsed"]) - expected_sample) > 1e-6:
            raise ValueError("visual HT sample does not match baseline plus zPhase")
        targets.append(
            {
                "report_row": int(row["reportRow"]),
                "final_ht_pc_elapsed_s": expected_sample,
                "rk_to_pc_bias_s": baseline_bias + delta_s,
            }
        )
    targets.sort(key=lambda row: row["final_ht_pc_elapsed_s"])
    if not targets:
        raise ValueError("report arm contract has no accepted final HT with usable zPhase")
    return tracker, arm, rk, serials, exposure_center_offset_s, z_offset, targets


def measure(
    tracker_path: Path,
    video_path: Path,
    arm_path: Path,
    rk_path: Path,
    tables_path: Path,
) -> dict:
    started = time.perf_counter()
    (
        tracker,
        arm,
        rk,
        serials,
        exposure_center_offset_s,
        z_offset,
        targets,
    ) = _measurement_context(tracker_path, arm_path, rk_path, tables_path)
    config = tracker["config"]
    calibration_path = _session_path(config["calib_config_path"], tracker_path)
    cameras = _load_cameras(calibration_path, serials)
    tracker_t0 = float(config["first_frame_exposure_pc"])
    frames = sorted(
        [
            row
            for row in tracker.get("frames", [])
            if isinstance(row, dict)
            and isinstance(row.get("video_frame_idx"), int)
            and _finite(row.get("exposure_pc"))
        ],
        key=lambda row: int(row["video_frame_idx"]),
    )
    arm_states = sorted(
        [row for row in arm.get("states", []) if _finite(row.get("t")) and row.get("tcp")],
        key=lambda row: float(row["t"]),
    )
    car_locs = sorted(
        [row for row in tracker.get("car_locs", []) if _finite(row.get("elapsed_s"))],
        key=lambda row: float(row["elapsed_s"]),
    )
    if not frames or not arm_states or not car_locs:
        raise ValueError("session lacks video frames, arm states, or car locations")

    windows: list[tuple[dict, list[dict]]] = []
    for target in targets:
        ht_absolute_pc = tracker_t0 + target["final_ht_pc_elapsed_s"]
        center = min(
            range(len(frames)),
            key=lambda index: abs(
                float(frames[index]["exposure_pc"])
                + exposure_center_offset_s
                - ht_absolute_pc
            ),
        )
        windows.append(
            (
                target,
                frames[
                    max(0, center - FRAME_RADIUS) :
                    min(len(frames), center + FRAME_RADIUS + 1)
                ],
            )
        )

    print(
        f"[racket] scan {len(targets)} report rows / "
        f"{sum(len(items) for _, items in windows)} HT-neighbour frames",
        flush=True,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    observations: list[dict] = []
    attempted = 0
    try:
        for target, target_frames in windows:
            first_video_index = int(target_frames[0]["video_frame_idx"])
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, first_video_index):
                raise RuntimeError(f"could not seek to video frame {first_video_index}")
            actual_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if abs(actual_position - first_video_index) > 0.5:
                raise RuntimeError(
                    f"video seek landed at {actual_position}, expected {first_video_index}"
                )
            video_position = first_video_index
            row_observations_before = len(observations)
            for frame in target_frames:
                video_index = int(frame["video_frame_idx"])
                while video_position < video_index:
                    if not capture.grab():
                        raise RuntimeError(
                            f"could not decode through video frame {video_position}"
                        )
                    video_position += 1
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"could not read video frame {video_index}")
                video_position += 1
                panels = _grid_panels(image, serials, cameras)
                attempted += 1
                center_abs_pc = float(frame["exposure_pc"]) + exposure_center_offset_s
                center_elapsed = center_abs_pc - tracker_t0
                expected_mm = _expected_world_mm(
                    center_elapsed,
                    arm_states,
                    car_locs,
                    float(rk["t0"]),
                    target["rk_to_pc_bias_s"],
                    z_offset,
                )
                if expected_mm is None:
                    continue
                anchors = {
                    serial: _project_raw(cameras[serial], expected_mm) for serial in serials
                }
                candidates = {
                    serial: _marker_candidates(panels[serial], tuple(anchors[serial]))
                    for serial in serials
                }
                fit = _solve_marker_4cam(candidates, cameras, expected_mm)
                if fit is None:
                    continue
                dt_ms = 1000.0 * (
                    center_elapsed - target["final_ht_pc_elapsed_s"]
                )
                observations.append(
                    {
                        "x": float(fit.point.xyz_mm[0] / 1000.0),
                        "y": float(fit.point.xyz_mm[1] / 1000.0),
                        "z": float(fit.point.xyz_mm[2] / 1000.0),
                        "t": center_abs_pc,
                        "elapsed_s": center_elapsed,
                        "frame_idx": int(frame["idx"]),
                        "video_frame_idx": video_index,
                        "report_row": target["report_row"],
                        "final_ht_pc_elapsed_s": target["final_ht_pc_elapsed_s"],
                        "dt_ht_ms": dt_ms,
                        "ht_side": "before" if dt_ms < 0.0 else "after",
                        "n_cam": 4,
                        "reproj_err": fit.point.rms_px,
                        "reproj_max_px": fit.point.max_px,
                        "loo_max_mm": max(fit.loo_delta_mm.values()),
                        "heldout_max_px": max(fit.loo_heldout_px.values()),
                        "expected_distance_mm": fit.expected_distance_mm,
                        "black_marker": True,
                    }
                )
            print(
                f"[racket] row {target['report_row']}: "
                f"{len(observations) - row_observations_before}/{len(target_frames)} frames",
                flush=True,
            )
    finally:
        capture.release()

    observations.sort(key=lambda row: (row["t"], row["report_row"]))
    rows_with_observations = {row["report_row"] for row in observations}
    elapsed = time.perf_counter() - started
    print(
        f"[racket] fixed black marker: {len(observations)}/{attempted} frames, "
        f"{len(rows_with_observations)}/{len(targets)} report rows, {elapsed:.1f}s"
    )
    return {
        "config": {
            "measurement": "V04 fixed black marker center",
            "timing": (
                "four-camera exposure center: exposure_pc + "
                f"{exposure_center_offset_s * 1000.0:.3f} ms"
            ),
            "selection": (
                f"+/-{FRAME_RADIUS} video frames around each report raw final HT "
                "+ per-throw zPhase; "
                f"four cameras; max reprojection {MARKER_MAX_REPROJ_PX:g} px; "
                f"leave-one-out < {MARKER_MAX_LOO_MM:g} mm; held-out <= "
                f"{MARKER_MAX_HELDOUT_PX:g} px; FK search distance <= "
                f"{MARKER_MAX_EXPECTED_DISTANCE_MM:g} mm"
            ),
            "coordinate": (
                "four-camera marker world position; report subtracts visual car "
                "center on field world axes"
            ),
        },
        "summary": {
            "racket_observations_3d": len(observations),
            "racket_frames_processed": attempted,
            "racket_min_cams": 4,
            "report_rows_scanned": len(targets),
            "report_rows_with_observations": len(rows_with_observations),
        },
        "racket_observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="tracker session JSON")
    parser.add_argument("--video", type=Path, required=True, help="2x2 grid session video")
    parser.add_argument("--arm-json", type=Path, required=True)
    parser.add_argument("--rk-tracking-json", type=Path, required=True)
    parser.add_argument("--tables-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(
        args.input,
        args.video,
        args.arm_json,
        args.rk_tracking_json,
        args.tables_json,
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[racket] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
