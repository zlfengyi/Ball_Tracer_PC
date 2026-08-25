"""Robust event-level 3-D motion fit from native-pixel racket bbox centres.

Calibration translations are millimetres.  The public position and velocity
fields are metres and metres/second.  They describe the detector bbox centre,
not a calibrated racket-face point.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


_OBSERVATION_SEMANTICS = "racket_head_bbox_geometric_center"


@dataclass(frozen=True)
class BBoxObservation:
    frame_id: int
    serial: str
    exposure_center_s: float
    center_xy: tuple[float, float]
    bbox_confidence: float


@dataclass(frozen=True)
class BundleGates:
    world_volume_m: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    bbox_confidence_min: float
    window_before_contact_s: float = 0.125
    window_after_contact_s: float = 0.035
    reprojection_error_px: float = 8.0
    min_supported_frames: int = 3
    min_fit_span_s: float = 0.055
    min_abs_vz_mps: float = 0.30
    max_speed_mps: float = 35.0
    max_hypotheses: int = 4096


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    rvec: np.ndarray


def load_cameras(calibration_path: str | Path) -> dict[str, CameraCalibration]:
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    cameras = {}
    for serial, values in calibration["cameras"].items():
        rotation = np.asarray(values["R_world"], dtype=np.float64).reshape(3, 3)
        camera = CameraCalibration(
            K=np.asarray(values["K"], dtype=np.float64).reshape(3, 3),
            D=np.asarray(values["D"], dtype=np.float64).ravel(),
            R=rotation,
            t=np.asarray(values["t_world"], dtype=np.float64).reshape(3, 1),
            rvec=cv2.Rodrigues(rotation)[0],
        )
        arrays = (camera.K, camera.D, camera.R, camera.t, camera.rvec)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError(f"camera calibration is nonfinite: {serial}")
        cameras[str(serial)] = camera
    if not cameras:
        raise ValueError("camera calibration is empty")
    return cameras


def project_world_m(
    cameras: dict[str, CameraCalibration],
    serial: str,
    point_world_m: tuple[float, float, float],
) -> tuple[float, float]:
    point_mm = 1000.0 * np.asarray(point_world_m, dtype=np.float64)
    if point_mm.shape != (3,) or not np.all(np.isfinite(point_mm)):
        raise ValueError("world point must contain three finite metres")
    camera = cameras[serial]
    projected, _ = cv2.projectPoints(
        point_mm.reshape(1, 3),
        camera.rvec,
        camera.t,
        camera.K,
        camera.D,
    )
    u, v = projected.reshape(2)
    return float(u), float(v)


def _linear_system(
    observations: list[BBoxObservation],
    contact_time_s: float,
    cameras: dict[str, CameraCalibration],
) -> tuple[np.ndarray, np.ndarray]:
    rows, rhs = [], []
    for observation in observations:
        camera = cameras[observation.serial]
        normalized = cv2.undistortPoints(
            np.asarray([[observation.center_xy]], dtype=np.float64),
            camera.K,
            camera.D,
        )[0, 0]
        tau = observation.exposure_center_s - contact_time_s
        for image_value, row_index in zip(normalized, (0, 1)):
            spatial = image_value * camera.R[2] - camera.R[row_index]
            scale = float(np.linalg.norm(spatial))
            rows.append(np.r_[spatial, tau * spatial] / scale)
            rhs.append(
                (
                    float(camera.t[row_index, 0])
                    - image_value * float(camera.t[2, 0])
                )
                / scale
            )
    return np.asarray(rows), np.asarray(rhs)


def _solve(
    A: np.ndarray,
    b: np.ndarray,
    observation_indices: np.ndarray,
) -> np.ndarray | None:
    equation_indices = np.ravel(
        np.column_stack((2 * observation_indices, 2 * observation_indices + 1))
    )
    solution, _, rank, _ = np.linalg.lstsq(
        A[equation_indices],
        b[equation_indices],
        rcond=None,
    )
    if rank < 6 or not np.all(np.isfinite(solution)):
        return None
    return solution


def _residuals(
    solution: np.ndarray,
    observations: list[BBoxObservation],
    contact_time_s: float,
    cameras: dict[str, CameraCalibration],
) -> np.ndarray:
    taus = np.asarray(
        [item.exposure_center_s - contact_time_s for item in observations],
        dtype=np.float64,
    )
    points = solution[:3] + taus[:, None] * solution[3:]
    residuals = np.full(len(observations), np.inf, dtype=np.float64)
    serials = np.asarray([item.serial for item in observations])
    for serial in set(serials):
        indices = np.flatnonzero(serials == serial)
        camera = cameras[serial]
        projected, _ = cv2.projectPoints(
            points[indices],
            camera.rvec,
            camera.t,
            camera.K,
            camera.D,
        )
        measured = np.asarray(
            [observations[index].center_xy for index in indices],
            dtype=np.float64,
        )
        residuals[indices] = np.linalg.norm(
            projected.reshape(-1, 2) - measured,
            axis=1,
        )
    return residuals


def _supported_frames(
    observations: list[BBoxObservation],
    inliers: np.ndarray,
) -> list[int]:
    cameras_by_frame: dict[int, set[str]] = {}
    for observation, is_inlier in zip(observations, inliers):
        if is_inlier:
            cameras_by_frame.setdefault(observation.frame_id, set()).add(
                observation.serial
            )
    return sorted(
        frame_id
        for frame_id, serials in cameras_by_frame.items()
        if len(serials) >= 2
    )


def _one_candidate_per_camera_frame(
    observations: list[BBoxObservation],
    residuals: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Select at most one geometric candidate for each camera/frame cell."""

    selected = np.zeros(len(observations), dtype=bool)
    best_by_cell: dict[tuple[int, str], int] = {}
    for index, (observation, residual) in enumerate(zip(observations, residuals)):
        if not math.isfinite(float(residual)) or residual > threshold_px:
            continue
        cell = (observation.frame_id, observation.serial)
        previous = best_by_cell.get(cell)
        if previous is None or (
            residual,
            -observation.bbox_confidence,
            index,
        ) < (
            residuals[previous],
            -observations[previous].bbox_confidence,
            previous,
        ):
            best_by_cell[cell] = index
    selected[list(best_by_cell.values())] = True
    return selected


def _frame_span_s(
    observations: list[BBoxObservation],
    frame_ids: list[int],
) -> float:
    times = {
        item.frame_id: item.exposure_center_s
        for item in observations
        if item.frame_id in frame_ids
    }
    return max(times.values()) - min(times.values()) if len(times) >= 2 else 0.0


def _hypotheses(
    observations: list[BBoxObservation],
    limit: int,
) -> list[tuple[int, ...]]:
    eligible = [
        indices
        for indices in combinations(range(len(observations)), 4)
        if len({observations[index].frame_id for index in indices}) >= 3
        and len({observations[index].serial for index in indices}) >= 2
        and len(
            {
                (observations[index].frame_id, observations[index].serial)
                for index in indices
            }
        )
        == 4
    ]
    if len(eligible) <= limit:
        return eligible
    selected = np.random.default_rng(0).choice(
        len(eligible),
        size=limit,
        replace=False,
    )
    return [eligible[int(index)] for index in sorted(selected)]


def _confidently_opposite(
    fitted_vz_mps: float,
    leave_one_frame_vz_mps: list[float],
    min_abs_vz_mps: float,
) -> bool:
    if fitted_vz_mps > 0.0:
        return any(value <= -min_abs_vz_mps for value in leave_one_frame_vz_mps)
    return any(value >= min_abs_vz_mps for value in leave_one_frame_vz_mps)


def fit_bbox_bundle(
    observations: list[BBoxObservation],
    contact_time_s: float,
    cameras: dict[str, CameraCalibration],
    gates: BundleGates,
) -> dict:
    """Fit one contact-centred event and return an explicit acceptance decision."""

    if not math.isfinite(contact_time_s):
        raise ValueError("contact_time_s must be finite")

    eligible_pairs = []
    for input_index, item in enumerate(observations):
        tau = item.exposure_center_s - contact_time_s
        values = (*item.center_xy, item.exposure_center_s, item.bbox_confidence)
        if (
            item.serial in cameras
            and all(math.isfinite(value) for value in values)
            and item.bbox_confidence >= gates.bbox_confidence_min
            and -gates.window_before_contact_s
            <= tau
            <= gates.window_after_contact_s
        ):
            eligible_pairs.append((input_index, item))
    filtered = [item for _, item in eligible_pairs]

    base = {
        "accepted": False,
        "reason": "insufficient_bbox_observations",
        "observation_semantics": _OBSERVATION_SEMANTICS,
        "input_observations": len(observations),
        "eligible_observations": len(filtered),
    }
    if (
        len(filtered) < 6
        or len({item.frame_id for item in filtered}) < gates.min_supported_frames
        or len({item.serial for item in filtered}) < 2
    ):
        return base

    A, b = _linear_system(filtered, contact_time_s, cameras)
    best = None
    for hypothesis in _hypotheses(filtered, gates.max_hypotheses):
        solution = _solve(A, b, np.asarray(hypothesis))
        if solution is None:
            continue
        previous_indices: tuple[int, ...] | None = None
        stable_unique_assignment = False
        for _ in range(4):
            residuals = _residuals(solution, filtered, contact_time_s, cameras)
            inliers = _one_candidate_per_camera_frame(
                filtered,
                residuals,
                gates.reprojection_error_px,
            )
            frame_ids = _supported_frames(filtered, inliers)
            span_s = _frame_span_s(filtered, frame_ids)
            if (
                len(frame_ids) < gates.min_supported_frames
                or span_s < gates.min_fit_span_s
            ):
                break
            current_indices = tuple(int(index) for index in np.flatnonzero(inliers))
            if current_indices == previous_indices:
                stable_unique_assignment = True
                break
            previous_indices = current_indices
            solution = _solve(A, b, np.asarray(current_indices))
            if solution is None:
                break
        if not stable_unique_assignment:
            continue
        final_inliers = inliers
        rank = (
            len(frame_ids),
            span_s,
            -float(np.median(residuals[final_inliers])),
            -float(np.max(residuals[final_inliers])),
        )
        if best is None or rank > best[0]:
            best = (rank, solution, residuals, final_inliers, frame_ids, span_s)

    if best is None:
        return base | {"reason": "no_bundle_inlier_model"}

    _, solution, residuals, inliers, frame_ids, span_s = best
    position_m = solution[:3] / 1000.0
    velocity_mps = solution[3:] / 1000.0
    inlier_filtered_indices = np.flatnonzero(inliers)
    input_inlier_indices = [
        eligible_pairs[int(index)][0] for index in inlier_filtered_indices
    ]
    detail = base | {
        "bbox_center_position_world_m": position_m.tolist(),
        "bbox_center_velocity_world_mps": velocity_mps.tolist(),
        "bbox_center_vz_world_mps": float(velocity_mps[2]),
        "supported_frames": frame_ids,
        "fit_span_s": float(span_s),
        "inlier_observations": len(input_inlier_indices),
        "inlier_observation_indices": input_inlier_indices,
        "mean_reprojection_error_px": float(np.mean(residuals[inliers])),
        "max_reprojection_error_px": float(np.max(residuals[inliers])),
    }

    leave_one_frame_vz_mps = []
    for omitted_frame in frame_ids:
        keep = np.asarray(
            [
                is_inlier and item.frame_id != omitted_frame
                for item, is_inlier in zip(filtered, inliers)
            ]
        )
        leave_one_out = _solve(A, b, np.flatnonzero(keep))
        if leave_one_out is None:
            return detail | {
                "reason": "leave_one_frame_sign_instability",
                "leave_one_frame_bbox_center_vz_mps": None,
            }
        leave_one_frame_vz_mps.append(float(leave_one_out[5] / 1000.0))
    detail["leave_one_frame_bbox_center_vz_mps"] = leave_one_frame_vz_mps

    in_world = all(
        lower <= value <= upper
        for value, (lower, upper) in zip(position_m, gates.world_volume_m)
    )
    if not in_world or float(np.linalg.norm(velocity_mps)) > gates.max_speed_mps:
        return detail | {"reason": "bundle_world_or_speed_gate"}

    vz_mps = float(velocity_mps[2])
    if abs(vz_mps) < gates.min_abs_vz_mps:
        return detail | {"reason": "weak_or_implausible_vz"}
    if _confidently_opposite(
        vz_mps,
        leave_one_frame_vz_mps,
        gates.min_abs_vz_mps,
    ):
        return detail | {"reason": "leave_one_frame_sign_instability"}
    return detail | {"accepted": True, "reason": "accepted"}


__all__ = [
    "BBoxObservation",
    "BundleGates",
    "CameraCalibration",
    "fit_bbox_bundle",
    "load_cameras",
    "project_world_m",
]
