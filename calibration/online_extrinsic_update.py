# -*- coding: utf-8 -*-
"""Safely update one 18-floor camera pose from live tennis-court imagery.

Painted court lines generate a candidate while K/D stay fixed.  Because a
planar scene cannot distinguish every intrinsic change from a pose change,
production promotion additionally requires an independent flying-ball
leave-one-camera-out validation from a tracker JSON recording.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.four_camera_calib_common import PROJECT_ROOT


MAX_REFERENCE_SAMPLE_RANGE_PX = 1.5
MAX_PROFILE_DEVIATION_PX = 12.0
MAX_JACOBIAN_CONDITION = 200.0
MAX_ACCEPTED_ROTATION_DEG = 0.50
MAX_ACCEPTED_POSITION_MM = 30.0
MIN_RMS_IMPROVEMENT_PX = 0.20
MAX_REMAINING_RMS_RATIO = 0.70
MAX_LINE_RMS_INCREASE_PX = 0.15
MIN_SUPPORTING_LINE_IMPROVEMENT_PX = 0.10
MIN_FINAL_LINE_SUPPORT_RATIO = 0.75
MAX_REDETECTED_RMS_EXCESS_PX = 0.25
MIN_AIRBORNE_HOLDOUT_SAMPLES = 20
MAX_CANDIDATE_MEDIAN_RATIO = 0.80
MIN_CANDIDATE_BETTER_FRACTION = 0.65
MAX_ANCHOR_REPROJ_ERROR_PX = 5.0
AIRBORNE_Z_M = 0.25
SELECTION_MAX_REPROJ_ERROR_PX = 15.0
DEFAULT_CALIB_PATH = PROJECT_ROOT / "src" / "config" / "four_camera_calib_18.json"
DEFAULT_CAMERA_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "camera_18.json"


# The registered 18-floor controls define the singles width and near service
# line. The far service coordinate follows the same full-court layout and its
# per-camera projection bias is learned from known-good reference frames.
COURT_LINES_MM = {
    "left_singles": (
        np.array([-4115.0, 0.0, 0.0]),
        np.array([-4115.0, 23770.0, 0.0]),
        "longitudinal",
    ),
    "center_service": (
        np.array([0.0, 5480.0, 0.0]),
        np.array([0.0, 18290.0, 0.0]),
        "longitudinal",
    ),
    "right_singles": (
        np.array([4115.0, 0.0, 0.0]),
        np.array([4115.0, 23770.0, 0.0]),
        "longitudinal",
    ),
    "near_service": (
        np.array([-4115.0, 5480.0, 0.0]),
        np.array([4115.0, 5480.0, 0.0]),
        "transverse",
    ),
    "far_service": (
        np.array([-4115.0, 18290.0, 0.0]),
        np.array([4115.0, 18290.0, 0.0]),
        "transverse",
    ),
}


@dataclass(frozen=True)
class LineObservation:
    line_name: str
    orientation: str
    world_mm: np.ndarray
    image_px: np.ndarray
    normal: np.ndarray
    offset_px: float
    prominence: float


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class TriangulatedBall:
    xyz_m: np.ndarray
    cameras_used: tuple[str, ...]
    pixels: dict[str, tuple[float, float]]
    confidence: float
    reprojection_error_px: float


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t.reshape(3, 1)).ravel()


def _pose_delta(
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    refined_rvec: np.ndarray,
    refined_tvec: np.ndarray,
) -> tuple[float, float]:
    initial_R, _ = cv2.Rodrigues(initial_rvec)
    refined_R, _ = cv2.Rodrigues(refined_rvec)
    delta_rvec, _ = cv2.Rodrigues(refined_R @ initial_R.T)
    rotation_deg = float(np.linalg.norm(delta_rvec) * 180.0 / np.pi)
    position_mm = float(
        np.linalg.norm(
            _camera_center(refined_R, refined_tvec)
            - _camera_center(initial_R, initial_tvec)
        )
    )
    return rotation_deg, position_mm


def _project(
    world_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image, _ = cv2.projectPoints(world_mm, rvec, tvec, K, D)
    R, _ = cv2.Rodrigues(rvec)
    depth = (R @ world_mm.T + tvec.reshape(3, 1))[2]
    return image.reshape(-1, 2), depth


def _line_world_samples(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alpha = np.linspace(0.04, 0.96, count, dtype=np.float64)
    return start[None, :] + alpha[:, None] * (end - start)[None, :]


def _whiteness_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("court-line refinement requires a BGR image")
    whiteness = np.min(image.astype(np.float32), axis=2)
    return cv2.GaussianBlur(whiteness, (3, 3), 0.0)


def _validate_image_sizes(images: list[np.ndarray], camera: dict) -> None:
    expected_width, expected_height = map(int, camera["image_size"])
    for image in images:
        if image.shape[:2] != (expected_height, expected_width):
            raise ValueError(
                "court-line image size does not match calibrated image_size: "
                f"got {image.shape[1]}x{image.shape[0]}, "
                f"expected {expected_width}x{expected_height}"
            )


def _profile_offset(
    whiteness: np.ndarray,
    point: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    search_radius_px: int,
    min_prominence: float,
) -> tuple[float, float] | None:
    offsets = np.arange(-search_radius_px, search_radius_px + 1, dtype=np.float32)
    along = np.arange(-3, 4, dtype=np.float32)
    coordinates = (
        point[None, None, :]
        + offsets[:, None, None] * normal[None, None, :]
        + along[None, :, None] * tangent[None, None, :]
    )
    profile = cv2.remap(
        whiteness,
        coordinates[:, :, 0].astype(np.float32),
        coordinates[:, :, 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).mean(axis=1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 5), 0.0).ravel()

    baseline = float(np.percentile(profile, 25.0))
    peak_index = int(np.argmax(profile))
    prominence = float(profile[peak_index] - baseline)
    if prominence < min_prominence:
        return None

    threshold = baseline + 0.65 * prominence
    left = peak_index
    right = peak_index
    while left > 0 and profile[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(profile) and profile[right + 1] >= threshold:
        right += 1

    weights = np.maximum(profile[left:right + 1] - baseline, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return None
    offset = float(np.sum(offsets[left:right + 1] * weights) / np.sum(weights))
    if abs(offset) >= search_radius_px - 2:
        return None
    return offset, prominence


def detect_line_observations(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    samples_per_line: int = 100,
    search_radius_px: int = 25,
    min_prominence: float = 12.0,
    min_points_per_line: int = 20,
) -> list[LineObservation]:
    """Find painted-line centers near their current calibrated projections."""
    whiteness = _whiteness_image(image)
    height, width = whiteness.shape
    observations: list[LineObservation] = []

    for line_name, (start, end, orientation) in COURT_LINES_MM.items():
        world = _line_world_samples(start, end, samples_per_line)
        projected, depth = _project(world, rvec, tvec, K, D)
        tangents = np.gradient(projected, axis=0)
        tangent_norm = np.linalg.norm(tangents, axis=1)
        valid_tangent = tangent_norm > 1e-6
        tangents[valid_tangent] /= tangent_norm[valid_tangent, None]
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        candidates: list[LineObservation] = []
        margin = search_radius_px + 5
        for world_point, point, tangent, normal, point_depth, tangent_ok in zip(
            world, projected, tangents, normals, depth, valid_tangent
        ):
            if (
                not tangent_ok
                or point_depth <= 0.0
                or point[0] < margin
                or point[0] >= width - margin
                or point[1] < margin
                or point[1] >= height - margin
            ):
                continue
            detected = _profile_offset(
                whiteness,
                point,
                tangent,
                normal,
                search_radius_px,
                min_prominence,
            )
            if detected is None:
                continue
            offset, prominence = detected
            candidates.append(
                LineObservation(
                    line_name=line_name,
                    orientation=orientation,
                    world_mm=world_point,
                    image_px=point + offset * normal,
                    normal=normal,
                    offset_px=offset,
                    prominence=prominence,
                )
            )

        if len(candidates) < min_points_per_line:
            continue
        offsets = np.asarray([item.offset_px for item in candidates])
        median = float(np.median(offsets))
        mad = float(np.median(np.abs(offsets - median)))
        tolerance = max(3.0, 4.0 * 1.4826 * mad)
        retained = [
            item for item in candidates if abs(item.offset_px - median) <= tolerance
        ]
        if len(retained) >= min_points_per_line:
            observations.extend(retained)

    return observations


def _validate_observability(observations: list[LineObservation]) -> None:
    lines_by_orientation = {
        orientation: {
            item.line_name
            for item in observations
            if item.orientation == orientation
        }
        for orientation in ("longitudinal", "transverse")
    }
    if any(len(lines) < 2 for lines in lines_by_orientation.values()):
        raise RuntimeError(
            "Court lines are not observable: need two distinct lines in each direction"
        )


def _line_rms(observations: list[LineObservation]) -> float:
    offsets = np.asarray([item.offset_px for item in observations], dtype=np.float64)
    return float(np.sqrt(np.mean(offsets * offsets)))


def build_line_reference(
    images: list[np.ndarray],
    camera: dict,
) -> tuple[dict[str, dict[str, list[float]]], dict]:
    """Measure the stable per-line projection bias of a known-good pose."""
    if len(images) < 3:
        raise ValueError("at least three court-line reference images are required")
    _validate_image_sizes(images, camera)
    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera["D"], dtype=np.float64).ravel()
    R = np.asarray(camera["R_world"], dtype=np.float64).reshape(3, 3)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.asarray(camera["t_world"], dtype=np.float64).reshape(3)

    by_line: dict[str, dict[float, list[float]]] = {}
    observed_frames: dict[str, int] = {}
    for image in images:
        observations = detect_line_observations(image, K, D, rvec, tvec)
        for line_name in {item.line_name for item in observations}:
            observed_frames[line_name] = observed_frames.get(line_name, 0) + 1
        for item in observations:
            start, end, _ = COURT_LINES_MM[item.line_name]
            direction = end - start
            alpha = float(
                np.dot(item.world_mm - start, direction) / np.dot(direction, direction)
            )
            by_line.setdefault(item.line_name, {}).setdefault(
                round(alpha, 8), []
            ).append(item.offset_px)

    required_frames = len(images) // 2 + 1
    profiles = {}
    for line_name, samples in by_line.items():
        if observed_frames.get(line_name, 0) < required_frames:
            continue
        stable = [
            (alpha, offsets)
            for alpha, offsets in sorted(samples.items())
            if len(offsets) >= required_frames
            and max(offsets) - min(offsets) <= MAX_REFERENCE_SAMPLE_RANGE_PX
        ]
        if len(stable) < 20:
            continue
        profiles[line_name] = {
            "alpha": [item[0] for item in stable],
            "offset_px": [float(np.median(item[1])) for item in stable],
        }
    lines_by_orientation = {
        orientation: {
            line_name
            for line_name in profiles
            if COURT_LINES_MM[line_name][2] == orientation
        }
        for orientation in ("longitudinal", "transverse")
    }
    if any(len(lines) < 2 for lines in lines_by_orientation.values()):
        raise RuntimeError(
            "Court-line reference is incomplete: need two stable lines in each direction"
        )

    temporal_mad = {}
    line_medians = {}
    for line_name, profile in profiles.items():
        deviations = []
        for alpha, reference_offset in zip(profile["alpha"], profile["offset_px"]):
            deviations.extend(
                abs(value - reference_offset)
                for value in by_line[line_name][alpha]
            )
        temporal_mad[line_name] = float(np.median(deviations))
        line_medians[line_name] = float(np.median(profile["offset_px"]))
    return profiles, {
        "frames": len(images),
        "line_bias_median_px": line_medians,
        "temporal_mad_px": temporal_mad,
        "profiles": profiles,
    }


def _apply_line_reference(
    observations: list[LineObservation],
    line_reference: dict[str, dict[str, list[float]]],
    max_deviation_px: float = MAX_PROFILE_DEVIATION_PX,
) -> list[LineObservation]:
    corrected = []
    for item in observations:
        if item.line_name not in line_reference:
            continue
        start, end, _ = COURT_LINES_MM[item.line_name]
        direction = end - start
        alpha = float(
            np.dot(item.world_mm - start, direction) / np.dot(direction, direction)
        )
        profile = line_reference[item.line_name]
        bias = float(np.interp(alpha, profile["alpha"], profile["offset_px"]))
        corrected_offset = item.offset_px - bias
        if abs(corrected_offset) > max_deviation_px:
            continue
        corrected.append(
            LineObservation(
                line_name=item.line_name,
                orientation=item.orientation,
                world_mm=item.world_mm,
                image_px=item.image_px - bias * item.normal,
                normal=item.normal,
                offset_px=corrected_offset,
                prominence=item.prominence,
            )
        )
    return corrected


def _detect_images(
    images: list[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    line_reference: dict[str, dict[str, list[float]]],
) -> list[LineObservation]:
    by_sample: dict[tuple[str, float], list[LineObservation]] = {}
    for image in images:
        detected = detect_line_observations(image, K, D, rvec, tvec)
        for item in _apply_line_reference(detected, line_reference):
            start, end, _ = COURT_LINES_MM[item.line_name]
            direction = end - start
            alpha = float(
                np.dot(item.world_mm - start, direction) / np.dot(direction, direction)
            )
            by_sample.setdefault((item.line_name, round(alpha, 8)), []).append(item)

    required_frames = len(images) // 2 + 1
    observations = []
    for items in by_sample.values():
        if len(items) < required_frames:
            continue
        template = items[0]
        offset = float(np.median([item.offset_px for item in items]))
        projected = template.image_px - template.offset_px * template.normal
        observations.append(
            LineObservation(
                line_name=template.line_name,
                orientation=template.orientation,
                world_mm=template.world_mm,
                image_px=projected + offset * template.normal,
                normal=template.normal,
                offset_px=offset,
                prominence=float(np.median([item.prominence for item in items])),
            )
        )

    retained = []
    for line_name in {item.line_name for item in observations}:
        line_items = [item for item in observations if item.line_name == line_name]
        start, end, _ = COURT_LINES_MM[line_name]
        direction = end - start
        alpha = np.asarray(
            [
                np.dot(item.world_mm - start, direction)
                / np.dot(direction, direction)
                for item in line_items
            ]
        )
        offsets = np.asarray([item.offset_px for item in line_items])
        alpha_span = float(np.ptp(alpha))
        if alpha_span <= 1e-8:
            continue
        normalized_alpha = (alpha - np.mean(alpha)) / alpha_span
        coefficients = np.polyfit(normalized_alpha, offsets, deg=2)
        residuals = offsets - np.polyval(coefficients, normalized_alpha)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        tolerance = max(0.75, 4.0 * 1.4826 * mad)
        consistent = [
            item
            for item, residual in zip(line_items, residuals)
            if abs(residual - median) <= tolerance
        ]
        if len(consistent) >= 20:
            retained.extend(consistent)
    return retained


def refine_camera_pose(
    images: list[np.ndarray],
    camera: dict,
    line_reference: dict[str, dict[str, list[float]]],
    *,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict, list[LineObservation]]:
    """Refine one world-to-camera pose while keeping K/D fixed."""
    if len(images) < 3:
        raise ValueError("at least three target images are required")
    _validate_image_sizes(images, camera)
    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera["D"], dtype=np.float64).ravel()
    initial_R = np.asarray(camera["R_world"], dtype=np.float64).reshape(3, 3)
    initial_rvec, _ = cv2.Rodrigues(initial_R)
    initial_rvec = initial_rvec.ravel()
    initial_tvec = np.asarray(camera["t_world"], dtype=np.float64).reshape(3)
    initial = np.concatenate((initial_rvec, initial_tvec))
    current = initial.copy()

    initial_observations = _detect_images(
        images, K, D, current[:3], current[3:], line_reference
    )
    _validate_observability(initial_observations)
    evaluation_world = np.asarray(
        [item.world_mm for item in initial_observations]
    )
    evaluation_image = np.asarray(
        [item.image_px for item in initial_observations]
    )
    evaluation_normals = np.asarray(
        [item.normal for item in initial_observations]
    )

    def evaluation_residuals(params: np.ndarray) -> np.ndarray:
        projected, _ = _project(
            evaluation_world, params[:3], params[3:], K, D
        )
        return np.sum(
            (projected - evaluation_image) * evaluation_normals,
            axis=1,
        )

    def rms(residuals: np.ndarray) -> float:
        return float(np.sqrt(np.mean(residuals * residuals)))

    initial_residuals = evaluation_residuals(initial)
    initial_rms = rms(initial_residuals)
    _, projection_jacobian = cv2.projectPoints(
        evaluation_world, initial[:3], initial[3:], K, D
    )
    pose_jacobian = projection_jacobian[:, :6].reshape(-1, 2, 6)
    normal_jacobian = np.einsum(
        "ni,nij->nj", evaluation_normals, pose_jacobian
    )
    scaled_jacobian = normal_jacobian * np.array(
        [0.005, 0.005, 0.005, 20.0, 20.0, 20.0]
    )
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    jacobian_rank = int(np.linalg.matrix_rank(scaled_jacobian))
    jacobian_condition = float(singular_values[0] / singular_values[-1])
    if jacobian_rank < 6 or jacobian_condition > MAX_JACOBIAN_CONDITION:
        raise RuntimeError(
            "Court-line pose is ill-conditioned: "
            f"rank={jacobian_rank}, condition={jacobian_condition:.1f}"
        )

    hit_bounds = False
    for _ in range(iterations):
        observations = _detect_images(
            images, K, D, current[:3], current[3:], line_reference
        )
        _validate_observability(observations)
        world = np.asarray([item.world_mm for item in observations])
        observed = np.asarray([item.image_px for item in observations])
        normals = np.asarray([item.normal for item in observations])

        def residuals(params: np.ndarray) -> np.ndarray:
            projected, _ = _project(world, params[:3], params[3:], K, D)
            line_error = np.sum((projected - observed) * normals, axis=1)
            pose_prior = np.concatenate(
                (
                    (params[:3] - initial[:3]) / np.deg2rad(0.35),
                    (params[3:] - initial[3:]) / 75.0,
                )
            )
            return np.concatenate((line_error, pose_prior))

        bounds = (
            initial + np.array([-0.02, -0.02, -0.02, -200.0, -200.0, -200.0]),
            initial + np.array([0.02, 0.02, 0.02, 200.0, 200.0, 200.0]),
        )
        result = least_squares(
            residuals,
            current,
            bounds=bounds,
            x_scale=np.array([0.005, 0.005, 0.005, 20.0, 20.0, 20.0]),
            loss="huber",
            f_scale=2.0,
            max_nfev=200,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Court-line pose refinement failed: {result.message}")
        hit_bounds = hit_bounds or bool(np.any(result.active_mask))
        current = result.x

    final_observations = _detect_images(
        images, K, D, current[:3], current[3:], line_reference
    )
    _validate_observability(final_observations)
    redetected_rms = _line_rms(final_observations)
    initial_line_counts = {
        line_name: sum(
            item.line_name == line_name for item in initial_observations
        )
        for line_name in {item.line_name for item in initial_observations}
    }
    final_line_counts = {
        line_name: sum(item.line_name == line_name for item in final_observations)
        for line_name in {item.line_name for item in final_observations}
    }
    common_lines = set(final_line_counts) & set(initial_line_counts)
    common_lines_by_orientation = {
        orientation: {
            line_name
            for line_name in common_lines
            if COURT_LINES_MM[line_name][2] == orientation
        }
        for orientation in ("longitudinal", "transverse")
    }
    stable_final_geometry = all(
        len(lines) >= 2 for lines in common_lines_by_orientation.values()
    )
    final_line_support_ratio = (
        min(
            final_line_counts[line_name] / initial_line_counts[line_name]
            for line_name in common_lines
        )
        if stable_final_geometry
        else 0.0
    )
    candidate_residuals = evaluation_residuals(current)
    candidate_rms = rms(candidate_residuals)
    evaluation_line_names = np.asarray(
        [item.line_name for item in initial_observations]
    )
    initial_line_rms = {}
    candidate_line_rms = {}
    for line_name in sorted(set(evaluation_line_names)):
        mask = evaluation_line_names == line_name
        initial_line_rms[line_name] = rms(initial_residuals[mask])
        candidate_line_rms[line_name] = rms(candidate_residuals[mask])
    line_improvements = {
        line_name: initial_line_rms[line_name] - candidate_line_rms[line_name]
        for line_name in initial_line_rms
    }
    supporting_lines = {
        line_name
        for line_name, improvement in line_improvements.items()
        if improvement >= MIN_SUPPORTING_LINE_IMPROVEMENT_PX
    }
    supporting_orientations = {
        COURT_LINES_MM[line_name][2] for line_name in supporting_lines
    }
    max_line_rms_increase = max(
        candidate_line_rms[line_name] - initial_line_rms[line_name]
        for line_name in initial_line_rms
    )
    candidate_rotation_deg, candidate_position_mm = _pose_delta(
        initial[:3], initial[3:], current[:3], current[3:]
    )
    accepted = (
        not hit_bounds
        and candidate_rotation_deg <= MAX_ACCEPTED_ROTATION_DEG
        and candidate_position_mm <= MAX_ACCEPTED_POSITION_MM
        and initial_rms - candidate_rms >= MIN_RMS_IMPROVEMENT_PX
        and candidate_rms <= initial_rms * MAX_REMAINING_RMS_RATIO
        and max_line_rms_increase <= MAX_LINE_RMS_INCREASE_PX
        and len(supporting_lines) >= 3
        and supporting_orientations == {"longitudinal", "transverse"}
        and stable_final_geometry
        and final_line_support_ratio >= MIN_FINAL_LINE_SUPPORT_RATIO
        and redetected_rms <= candidate_rms + MAX_REDETECTED_RMS_EXCESS_PX
    )
    if not accepted:
        current = initial.copy()
        final_observations = initial_observations
        final_rms = initial_rms
    else:
        final_rms = candidate_rms
    rotation_deg, position_mm = _pose_delta(
        initial[:3], initial[3:], current[:3], current[3:]
    )
    line_names = sorted({item.line_name for item in final_observations})
    diagnostics = {
        "initial_rms_px": initial_rms,
        "final_rms_px": final_rms,
        "candidate_rms_px": candidate_rms,
        "redetected_rms_px": redetected_rms,
        "initial_line_rms_px": initial_line_rms,
        "candidate_line_rms_px": candidate_line_rms,
        "supporting_lines": sorted(supporting_lines),
        "max_line_rms_increase_px": max_line_rms_increase,
        "initial_line_counts": initial_line_counts,
        "redetected_line_counts": final_line_counts,
        "final_line_support_ratio": final_line_support_ratio,
        "observations": len(final_observations),
        "lines": line_names,
        "rotation_change_deg": rotation_deg,
        "position_change_mm": position_mm,
        "candidate_rotation_change_deg": candidate_rotation_deg,
        "candidate_position_change_mm": candidate_position_mm,
        "jacobian_rank": jacobian_rank,
        "jacobian_condition": jacobian_condition,
        "hit_bounds": hit_bounds,
        "accepted": accepted,
    }
    refined_R, _ = cv2.Rodrigues(current[:3])
    return refined_R, current[3:].reshape(3, 1), diagnostics, final_observations


def update_relative_extrinsics(
    calib: dict,
    changed_serial: str | None = None,
) -> dict[str, list[str]]:
    """Refresh derived fields, touching only values affected by one pose change."""
    reference = calib["reference_serial"]
    ref_camera = calib["cameras"][reference]
    ref_R = np.asarray(ref_camera["R_world"], dtype=np.float64).reshape(3, 3)
    ref_t = np.asarray(ref_camera["t_world"], dtype=np.float64).reshape(3, 1)

    if changed_serial is None:
        position_serials = set(calib["cameras"])
        relative_serials = set(calib["cameras"])
    else:
        if changed_serial not in calib["cameras"]:
            raise KeyError(f"Unknown camera serial: {changed_serial}")
        position_serials = {changed_serial}
        relative_serials = (
            set(calib["cameras"])
            if changed_serial == reference
            else {changed_serial}
        )

    changed_fields: dict[str, list[str]] = {}
    for serial in position_serials:
        camera = calib["cameras"][serial]
        R = np.asarray(camera["R_world"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(camera["t_world"], dtype=np.float64).reshape(3, 1)
        camera["pos_world"] = _camera_center(R, t).reshape(3, 1).tolist()
        changed_fields.setdefault(serial, []).append("pos_world")

    for serial in relative_serials:
        camera = calib["cameras"][serial]
        if serial == reference:
            relative_R = np.eye(3, dtype=np.float64)
            relative_t = np.zeros((3, 1), dtype=np.float64)
        else:
            R = np.asarray(camera["R_world"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(camera["t_world"], dtype=np.float64).reshape(3, 1)
            relative_R = R @ ref_R.T
            relative_t = t - relative_R @ ref_t
        camera["R_ref_to_camera"] = relative_R.tolist()
        camera["t_ref_to_camera"] = relative_t.tolist()
        changed_fields.setdefault(serial, []).extend(
            ("R_ref_to_camera", "t_ref_to_camera")
        )
    return changed_fields


def _draw_overlay(
    image: np.ndarray,
    camera: dict,
    refined_R: np.ndarray,
    refined_t: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    K = np.asarray(camera["K"], dtype=np.float64)
    D = np.asarray(camera["D"], dtype=np.float64)
    initial_R = np.asarray(camera["R_world"], dtype=np.float64)
    initial_t = np.asarray(camera["t_world"], dtype=np.float64)
    initial_rvec, _ = cv2.Rodrigues(initial_R)
    refined_rvec, _ = cv2.Rodrigues(refined_R)

    for start, end, _ in COURT_LINES_MM.values():
        world = _line_world_samples(start, end, 160)
        initial_points, _ = _project(world, initial_rvec, initial_t, K, D)
        refined_points, _ = _project(world, refined_rvec, refined_t, K, D)
        cv2.polylines(
            overlay,
            [np.rint(initial_points).astype(np.int32)],
            False,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.polylines(
            overlay,
            [np.rint(refined_points).astype(np.int32)],
            False,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _read_calibration(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    calibration = json.loads(text)
    if _serialize_like_source(calibration, text) != text:
        raise ValueError(
            "Calibration JSON formatting is not round-trip stable; refusing a noisy rewrite"
        )
    return calibration, text, raw


def _json_source_style(source: str) -> tuple[int, str, bool]:
    if "\r\n" in source:
        without_crlf = source.replace("\r\n", "")
        if "\r" in without_crlf or "\n" in without_crlf:
            raise ValueError("Calibration JSON has mixed line endings")
        newline = "\r\n"
    else:
        if "\r" in source:
            raise ValueError("Calibration JSON has unsupported line endings")
        newline = "\n"
    normalized = source.replace("\r\n", "\n")
    indent = 0
    for line in normalized.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if stripped.startswith('"'):
            indent = len(line) - len(stripped)
            break
    if indent <= 0:
        raise ValueError("Could not infer calibration JSON indentation")
    return indent, newline, source.endswith(newline)


def _serialize_like_source(value: dict[str, Any], source: str) -> str:
    indent, newline, trailing_newline = _json_source_style(source)
    rendered = json.dumps(value, indent=indent, ensure_ascii=False)
    if newline != "\n":
        rendered = rendered.replace("\n", newline)
    if trailing_newline:
        rendered += newline
    return rendered


def _atomic_write_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if path.exists() and not replace:
            raise FileExistsError(f"Output appeared during write: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


def _validate_rotation(R: np.ndarray, serial: str, field: str) -> None:
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError(f"{serial}: invalid {field}")
    orthogonality = float(np.max(np.abs(R.T @ R - np.eye(3))))
    determinant = float(np.linalg.det(R))
    if orthogonality > 1e-8 or abs(determinant - 1.0) > 1e-8:
        raise ValueError(
            f"{serial}: {field} is not a proper rotation "
            f"(orthogonality={orthogonality:.3e}, det={determinant:.12f})"
        )


def validate_calibration_geometry(calibration: dict[str, Any]) -> None:
    reference = calibration.get("reference_serial")
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict) or reference not in cameras:
        raise ValueError("Calibration has an invalid reference camera")
    ref_R = np.asarray(cameras[reference]["R_world"], dtype=np.float64)
    ref_t = np.asarray(cameras[reference]["t_world"], dtype=np.float64).reshape(3, 1)
    _validate_rotation(ref_R, reference, "R_world")

    for serial, camera in cameras.items():
        K = np.asarray(camera["K"], dtype=np.float64)
        D = np.asarray(camera["D"], dtype=np.float64)
        R = np.asarray(camera["R_world"], dtype=np.float64)
        t = np.asarray(camera["t_world"], dtype=np.float64).reshape(3, 1)
        position = np.asarray(camera["pos_world"], dtype=np.float64).reshape(3, 1)
        relative_R = np.asarray(camera["R_ref_to_camera"], dtype=np.float64)
        relative_t = np.asarray(
            camera["t_ref_to_camera"], dtype=np.float64
        ).reshape(3, 1)
        valid_intrinsics = (
            K.shape == (3, 3)
            and D.size >= 4
            and np.all(np.isfinite(K))
            and np.all(np.isfinite(D))
        )
        if not valid_intrinsics:
            raise ValueError(f"{serial}: invalid K/D")
        if list(map(int, camera["image_size"])) != list(camera["image_size"]):
            raise ValueError(f"{serial}: invalid image_size")
        _validate_rotation(R, serial, "R_world")
        _validate_rotation(relative_R, serial, "R_ref_to_camera")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(relative_t)):
            raise ValueError(f"{serial}: non-finite translation")
        expected_position = -R.T @ t
        expected_relative_R = R @ ref_R.T
        expected_relative_t = t - expected_relative_R @ ref_t
        if not np.allclose(position, expected_position, atol=1e-6, rtol=0.0):
            raise ValueError(f"{serial}: pos_world is inconsistent")
        if not np.allclose(relative_R, expected_relative_R, atol=1e-10, rtol=0.0):
            raise ValueError(f"{serial}: R_ref_to_camera is inconsistent")
        if not np.allclose(relative_t, expected_relative_t, atol=1e-6, rtol=0.0):
            raise ValueError(f"{serial}: t_ref_to_camera is inconsistent")


def validate_candidate_changes(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    target_serial: str,
) -> dict[str, list[str]]:
    if baseline.keys() != candidate.keys():
        raise ValueError("Candidate changes top-level calibration keys")
    if baseline["reference_serial"] != candidate["reference_serial"]:
        raise ValueError("Candidate changes reference_serial")
    if set(baseline["cameras"]) != set(candidate["cameras"]):
        raise ValueError("Candidate changes the camera set")

    reference = baseline["reference_serial"]
    allowed: dict[str, set[str]] = {
        serial: set() for serial in baseline["cameras"]
    }
    allowed[target_serial].update(
        ("R_world", "t_world", "pos_world", "R_ref_to_camera", "t_ref_to_camera")
    )
    if target_serial == reference:
        for fields in allowed.values():
            fields.update(("R_ref_to_camera", "t_ref_to_camera"))

    changed: dict[str, list[str]] = {}
    for top_level_key in baseline:
        if top_level_key != "cameras" and baseline[top_level_key] != candidate[top_level_key]:
            raise ValueError(f"Candidate changes top-level field {top_level_key}")
    for serial in baseline["cameras"]:
        old = baseline["cameras"][serial]
        new = candidate["cameras"][serial]
        if old.keys() != new.keys():
            raise ValueError(f"{serial}: candidate changes camera keys")
        for field in old:
            if old[field] == new[field]:
                continue
            if field not in allowed[serial]:
                raise ValueError(f"{serial}: candidate illegally changes {field}")
            changed.setdefault(serial, []).append(field)
    return changed


def _validate_frame_stems(stems: list[str], option: str) -> None:
    if len(stems) < 3:
        raise ValueError(f"{option} requires at least three frames")
    if len(set(stems)) != len(stems):
        raise ValueError(f"{option} must be unique")
    if any(not stem or Path(stem).name != stem for stem in stems):
        raise ValueError(f"{option} must contain plain frame stems")


def _load_frames(
    root: Path,
    serial: str,
    frame_stems: list[str],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    images: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for frame_stem in frame_stems:
        image_path = root / serial / f"{frame_stem}.png"
        payload = image_path.read_bytes()
        digest = _sha256_bytes(payload)
        if digest in hashes:
            raise ValueError(f"Duplicate image content is not an independent frame: {image_path}")
        hashes.add(digest)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unreadable camera image: {image_path}")
        images.append(image)
        records.append(
            {
                "stem": frame_stem,
                "path": str(image_path.resolve()),
                "sha256": digest,
            }
        )
    return images, records


def _configure_mvs_import() -> None:
    if os.name != "nt" or "MVS_MVIMPORT_DIR" in os.environ:
        return
    default_mvs = Path(
        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
    )
    if default_mvs.is_dir():
        os.environ["MVS_MVIMPORT_DIR"] = str(default_mvs)


def capture_live_images(
    output_dir: Path,
    camera_config_path: Path,
    calibration: dict[str, Any],
    *,
    count: int,
    duration_s: float,
    warmup_s: float,
    exposure_us: float,
    gain_db: float,
) -> tuple[Path, list[str], dict[str, Any]]:
    """Capture synchronized current frames. The SDK is imported only in live mode."""
    if count < 3 or duration_s <= 0.0 or warmup_s < 0.0:
        raise ValueError("Live capture needs count>=3, duration>0, and warmup>=0")
    _configure_mvs_import()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src import SyncCapture, frame_to_numpy

    capture_dir = output_dir / "capture"
    capture_dir.mkdir()
    expected_serials = list(calibration["cameras"])
    for serial in expected_serials:
        (capture_dir / serial).mkdir()

    overrides: dict[str, float] = {}
    if exposure_us > 0.0:
        overrides["exposure_us"] = exposure_us
    if gain_db >= 0.0:
        overrides["gain_db"] = gain_db

    frame_stems: list[str] = []
    settings: dict[str, Any] = {}
    started = time.perf_counter()
    with SyncCapture.from_config(str(camera_config_path), **overrides) as capture:
        if set(capture.sync_serials) != set(expected_serials):
            raise RuntimeError(
                "Live camera set differs from calibration: "
                f"got {capture.sync_serials}, expected {expected_serials}"
            )
        warmup_deadline = time.perf_counter() + warmup_s
        while time.perf_counter() < warmup_deadline:
            capture.get_frames(timeout_s=0.2)

        interval_s = duration_s / count
        capture_started = time.perf_counter()
        next_capture = capture_started
        hard_deadline = capture_started + duration_s + 10.0
        while len(frame_stems) < count:
            if time.perf_counter() > hard_deadline:
                raise TimeoutError(
                    f"Live capture timed out after {len(frame_stems)}/{count} frames"
                )
            frames = capture.get_frames(timeout_s=0.5)
            now = time.perf_counter()
            if frames is None or now < next_capture:
                continue
            if set(frames) != set(expected_serials):
                continue
            stem = f"{len(frame_stems) + 1:04d}"
            for serial, frame in frames.items():
                image = frame_to_numpy(frame)
                width, height = map(int, calibration["cameras"][serial]["image_size"])
                if image.shape[:2] != (height, width):
                    raise ValueError(
                        f"{serial}: live image is {image.shape[1]}x{image.shape[0]}, "
                        f"calibration expects {width}x{height}"
                    )
                image_path = capture_dir / serial / f"{stem}.png"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Could not save live image: {image_path}")
            frame_stems.append(stem)
            next_capture = capture_started + len(frame_stems) * interval_s
        settings = capture.camera_settings()

    camera_config = _load_json(camera_config_path)
    manifest = {
        "method": "synchronized_live_capture",
        "created_perf_counter_s": started,
        "camera_config": {
            "path": str(camera_config_path.resolve()),
            "sha256": _sha256_path(camera_config_path),
            "snapshot": camera_config,
            "overrides": overrides,
            "effective_settings": settings,
        },
        "frame_stems": frame_stems,
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "serials": expected_serials,
    }
    _write_report(capture_dir / "session.json", manifest)
    return capture_dir, frame_stems, manifest


class OfflineBallLocalizer:
    """Small calibration-only localizer with no camera SDK or detector imports."""

    def __init__(self, calibration: dict[str, Any]):
        self.serials = list(calibration["cameras"])
        self.K: dict[str, np.ndarray] = {}
        self.D: dict[str, np.ndarray] = {}
        self.P: dict[str, np.ndarray] = {}
        for serial, camera in calibration["cameras"].items():
            K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
            D = np.asarray(camera["D"], dtype=np.float64).ravel()
            R = np.asarray(camera["R_world"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(camera["t_world"], dtype=np.float64).reshape(3, 1)
            self.K[serial] = K
            self.D[serial] = D
            self.P[serial] = K @ np.column_stack((R, t))

    def triangulate(
        self, detections: dict[str, BallDetection]
    ) -> TriangulatedBall:
        serials = tuple(detections)
        if len(serials) < 2:
            raise ValueError("Triangulation requires at least two cameras")
        undistorted: dict[str, np.ndarray] = {}
        for serial, detection in detections.items():
            point = np.asarray([[[detection.x, detection.y]]], dtype=np.float64)
            undistorted[serial] = cv2.undistortPoints(
                point, self.K[serial], self.D[serial], P=self.K[serial]
            )[0, 0]

        rows = []
        for serial in serials:
            u, v = undistorted[serial]
            P = self.P[serial]
            rows.extend((u * P[2] - P[0], v * P[2] - P[1]))
        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        homogeneous = vh[-1]
        if abs(float(homogeneous[3])) < 1e-12:
            xyz_mm = np.full(3, np.nan)
        else:
            xyz_mm = homogeneous[:3] / homogeneous[3]

        errors = []
        pixels = {}
        xyz_h = np.append(xyz_mm, 1.0)
        for serial in serials:
            projected = self.P[serial] @ xyz_h
            if abs(float(projected[2])) < 1e-12:
                error = math.inf
            else:
                projected = projected[:2] / projected[2]
                error = float(np.linalg.norm(projected - undistorted[serial]))
            errors.append(error)
            detection = detections[serial]
            pixels[serial] = (detection.x, detection.y)
        return TriangulatedBall(
            xyz_m=xyz_mm / 1000.0,
            cameras_used=serials,
            pixels=pixels,
            confidence=min(detections[serial].confidence for serial in serials),
            reprojection_error_px=float(np.mean(errors)),
        )

    def select_and_triangulate(
        self,
        candidates: dict[str, list[BallDetection]],
        *,
        min_cameras: int = 2,
        max_reproj_error_px: float = SELECTION_MAX_REPROJ_ERROR_PX,
        max_per_camera: int = 1,
    ) -> TriangulatedBall | None:
        usable = {
            serial: sorted(
                candidates.get(serial, ()),
                key=lambda detection: detection.confidence,
                reverse=True,
            )[:max_per_camera]
            for serial in self.serials
            if candidates.get(serial)
        }
        serials = list(usable)
        for camera_count in range(len(serials), min_cameras - 1, -1):
            best: TriangulatedBall | None = None
            for subset in combinations(serials, camera_count):
                for selected in product(*(usable[serial] for serial in subset)):
                    result = self.triangulate(dict(zip(subset, selected)))
                    if not np.isfinite(result.reprojection_error_px):
                        continue
                    if result.reprojection_error_px > max_reproj_error_px:
                        continue
                    if best is None or (
                        result.reprojection_error_px,
                        -result.confidence,
                    ) < (best.reprojection_error_px, -best.confidence):
                        best = result
            if best is not None:
                return best
        return None


def _ball_candidates(frame: dict[str, Any]) -> dict[str, list[BallDetection]]:
    result: dict[str, list[BallDetection]] = {}
    for serial, items in (frame.get("detections") or {}).items():
        detections = [
            BallDetection(
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=float(item.get("conf", item.get("confidence", 0.0))),
            )
            for item in items
            if item.get("label", "tennis_ball") == "tennis_ball"
        ]
        if detections:
            result[serial] = detections
    return result


def _selected_detections(
    candidates: dict[str, list[BallDetection]],
    selected: TriangulatedBall,
) -> dict[str, BallDetection]:
    return {
        serial: min(
            candidates[serial],
            key=lambda detection: (
                (detection.x - pixel[0]) ** 2 + (detection.y - pixel[1]) ** 2
            ),
        )
        for serial, pixel in selected.pixels.items()
    }


def _project_raw(
    calibration: dict[str, Any], serial: str, xyz_m: np.ndarray
) -> np.ndarray:
    camera = calibration["cameras"][serial]
    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera["D"], dtype=np.float64).ravel()
    R = np.asarray(camera["R_world"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(camera["t_world"], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(
        np.asarray(xyz_m, dtype=np.float64).reshape(1, 1, 3) * 1000.0,
        rvec,
        t,
        K,
        D,
    )
    return projected.reshape(2)


def _raw_reprojection_rms(
    calibration: dict[str, Any],
    xyz_m: np.ndarray,
    detections: dict[str, BallDetection],
) -> float:
    errors = [
        float(
            np.linalg.norm(
                _project_raw(calibration, serial, xyz_m)
                - np.asarray((detection.x, detection.y), dtype=np.float64)
            )
        )
        for serial, detection in detections.items()
    ]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not finite.size:
        return {"n": 0, "mean": None, "median": None, "p95": None, "rms": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }


def evaluate_holdout_decision(
    baseline_errors: list[float], candidate_errors: list[float]
) -> dict[str, Any]:
    if len(baseline_errors) != len(candidate_errors):
        raise ValueError("Holdout samples must be paired")
    paired = [
        (old, new)
        for old, new in zip(baseline_errors, candidate_errors)
        if np.isfinite(old) and np.isfinite(new)
    ]
    old_values = [item[0] for item in paired]
    new_values = [item[1] for item in paired]
    old_stats = _stats(old_values)
    new_stats = _stats(new_values)
    better_fraction = (
        float(np.mean(np.asarray(new_values) < np.asarray(old_values)))
        if paired
        else None
    )
    old_median = old_stats["median"]
    new_median = new_stats["median"]
    old_p95 = old_stats["p95"]
    new_p95 = new_stats["p95"]
    supported = bool(
        len(paired) >= MIN_AIRBORNE_HOLDOUT_SAMPLES
        and old_median is not None
        and new_median is not None
        and old_p95 is not None
        and new_p95 is not None
        and better_fraction is not None
        and new_median <= MAX_CANDIDATE_MEDIAN_RATIO * old_median
        and new_p95 <= old_p95
        and better_fraction >= MIN_CANDIDATE_BETTER_FRACTION
    )
    return {
        "baseline_px": old_stats,
        "candidate_px": new_stats,
        "candidate_better_fraction": better_fraction,
        "supported": supported,
        "thresholds": {
            "minimum_airborne_samples": MIN_AIRBORNE_HOLDOUT_SAMPLES,
            "maximum_candidate_median_ratio": MAX_CANDIDATE_MEDIAN_RATIO,
            "candidate_p95_must_not_increase": True,
            "minimum_candidate_better_fraction": MIN_CANDIDATE_BETTER_FRACTION,
        },
    }


def _plausible_court_point(point: np.ndarray) -> bool:
    x, y, z = point
    return -7.0 <= x <= 7.0 and -2.0 <= y <= 26.0 and -0.25 <= z <= 8.0


def validate_with_flying_ball(
    tracker_data: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    target_serial: str,
) -> dict[str, Any]:
    """Validate the changed camera against 3D points from unchanged cameras."""
    changes = validate_candidate_changes(baseline, candidate, target_serial)
    if target_serial not in changes or not {
        "R_world", "t_world"
    }.issubset(changes[target_serial]):
        raise ValueError("Candidate does not contain the requested pose update")
    tracker_serials = set((tracker_data.get("config") or {}).get("serials", ()))
    if tracker_serials and tracker_serials != set(baseline["cameras"]):
        raise ValueError("Tracker recording camera set differs from calibration")

    baseline_localizer = OfflineBallLocalizer(baseline)
    baseline_errors: list[float] = []
    candidate_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    reconstructed = 0
    stored_3d = 0
    camera_set_matches = 0

    for frame in tracker_data.get("frames", []):
        stored = frame.get("ball3d")
        if stored is not None:
            stored_3d += 1
        candidates = _ball_candidates(frame)
        selected_result = baseline_localizer.select_and_triangulate(candidates)
        if selected_result is None:
            continue
        reconstructed += 1
        selected = _selected_detections(candidates, selected_result)
        if stored is not None and set(stored.get("cameras", ())) == set(selected):
            camera_set_matches += 1
        if target_serial not in selected or len(selected) < 3:
            continue

        anchors = {
            serial: detection
            for serial, detection in selected.items()
            if serial != target_serial
        }
        if len(anchors) < 2:
            continue
        anchor_result = baseline_localizer.triangulate(anchors)
        anchor_point = anchor_result.xyz_m
        anchor_rms = _raw_reprojection_rms(baseline, anchor_point, anchors)
        if anchor_rms > MAX_ANCHOR_REPROJ_ERROR_PX:
            continue
        if not _plausible_court_point(anchor_point):
            continue
        measured = np.asarray(
            (selected[target_serial].x, selected[target_serial].y),
            dtype=np.float64,
        )
        old_error = float(
            np.linalg.norm(_project_raw(baseline, target_serial, anchor_point) - measured)
        )
        new_error = float(
            np.linalg.norm(_project_raw(candidate, target_serial, anchor_point) - measured)
        )
        airborne = bool(anchor_point[2] >= AIRBORNE_Z_M)
        rows.append(
            {
                "frame_idx": int(frame.get("idx", -1)),
                "state": str(frame.get("state", "")),
                "anchor_cameras": list(anchors),
                "anchor_xyz_m": anchor_point.tolist(),
                "anchor_reproj_rms_px": anchor_rms,
                "baseline_error_px": old_error,
                "candidate_error_px": new_error,
                "airborne": airborne,
            }
        )
        if airborne:
            baseline_errors.append(old_error)
            candidate_errors.append(new_error)

    decision = evaluate_holdout_decision(baseline_errors, candidate_errors)
    return {
        "method": "fixed_association_unchanged_camera_flying_ball_holdout_v1",
        "target_serial": target_serial,
        "selection": {
            "max_reprojection_error_px": SELECTION_MAX_REPROJ_ERROR_PX,
            "max_per_camera": 1,
            "association_calibration": "baseline",
        },
        "holdout": {
            "anchor_max_reprojection_error_px": MAX_ANCHOR_REPROJ_ERROR_PX,
            "airborne_z_m": AIRBORNE_Z_M,
            **decision,
        },
        "reconstruction": {
            "stored_3d_frames": stored_3d,
            "reconstructed_3d_frames": reconstructed,
            "stored_camera_set_matches": camera_set_matches,
            "stored_camera_set_match_fraction": (
                camera_set_matches / stored_3d if stored_3d else None
            ),
        },
        "rows": rows,
    }


def promote_candidate(
    calibration_path: Path,
    expected_baseline_sha256: str,
    candidate_payload: bytes,
    backup_path: Path,
) -> None:
    """Atomically promote after verifying that the baseline did not change."""
    current_payload = calibration_path.read_bytes()
    if _sha256_bytes(current_payload) != expected_baseline_sha256:
        raise RuntimeError("Calibration changed during validation; refusing promotion")
    _atomic_write_bytes(backup_path, current_payload)
    if _sha256_path(backup_path) != expected_baseline_sha256:
        raise RuntimeError("Calibration backup verification failed")

    temporary = calibration_path.with_name(
        f".{calibration_path.name}.{os.getpid()}.online-update.tmp"
    )
    if temporary.exists():
        raise FileExistsError(f"Promotion temporary file exists: {temporary}")
    try:
        with temporary.open("xb") as output:
            output.write(candidate_payload)
            output.flush()
            os.fsync(output.fileno())
        shutil.copystat(calibration_path, temporary)
        if _sha256_path(calibration_path) != expected_baseline_sha256:
            raise RuntimeError("Calibration changed immediately before promotion")
        os.replace(temporary, calibration_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if calibration_path.read_bytes() != candidate_payload:
        raise RuntimeError("Promoted calibration verification failed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and independently validate one 18-floor camera pose update. "
            "Production is changed only with --apply and a passing tracker holdout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Typical workflow:
  1. Run with --capture-live (or --images/--frames) to create a candidate.
  2. Record flying-ball tracker JSON while production still uses the baseline.
  3. Rerun with the same images plus --tracker-json; inspect report and overlay.
  4. Add --apply only when promotion is intended.

The reference session must match the input calibration. After a successful
promotion, use newly approved current frames as the next reference session.
Use a new --output-dir for every run; existing output directories are refused.""",
    )
    parser.add_argument(
        "--reference-images",
        required=True,
        help="Known-good synchronized session matching the current calibration.",
    )
    parser.add_argument(
        "--reference-frames",
        nargs="+",
        required=True,
        help="At least three independent known-good frame stems.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--images", help="Existing current synchronized session.")
    target.add_argument(
        "--capture-live",
        action="store_true",
        help="Capture current synchronized images directly from the cameras.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        help="Current frame stems; required with --images and forbidden with --capture-live.",
    )
    parser.add_argument("--serial", required=True, help="Single camera serial to refine.")
    parser.add_argument(
        "--calib",
        default=str(DEFAULT_CALIB_PATH),
        help="Current 18-floor production calibration.",
    )
    parser.add_argument(
        "--camera-config",
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help="18-floor camera configuration used only by --capture-live.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for candidate, report, overlays, capture, and backup.",
    )
    parser.add_argument(
        "--tracker-json",
        help="Baseline tracker recording used for independent flying-ball validation.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically promote only after the independent validation passes.",
    )
    parser.add_argument("--capture-count", type=int, default=7)
    parser.add_argument("--capture-duration", type=float, default=7.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--exposure-us", type=float, default=-1.0)
    parser.add_argument("--gain-db", type=float, default=-1.0)
    return parser


def _format_stat(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    _validate_frame_stems(args.reference_frames, "--reference-frames")
    if args.capture_live:
        if args.frames:
            raise ValueError("--frames is forbidden with --capture-live")
    else:
        if not args.frames:
            raise ValueError("--frames is required with --images")
        _validate_frame_stems(args.frames, "--frames")
    if args.apply and not args.tracker_json:
        raise ValueError("--apply requires --tracker-json")

    calibration_path = _resolve_path(args.calib).resolve()
    camera_config_path = _resolve_path(args.camera_config).resolve()
    reference_dir = _resolve_path(args.reference_images).resolve()
    output_dir = _resolve_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / "four_camera_calib_18_candidate.json"
    report_path = output_dir / "online_extrinsic_update_report.json"
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir()

    baseline, source_text, baseline_payload = _read_calibration(calibration_path)
    baseline_sha256 = _sha256_bytes(baseline_payload)
    validate_calibration_geometry(baseline)
    if args.serial not in baseline["cameras"]:
        raise ValueError(f"Unknown --serial {args.serial}")

    capture_manifest = None
    if args.capture_live:
        images_dir, frame_stems, capture_manifest = capture_live_images(
            output_dir,
            camera_config_path,
            baseline,
            count=args.capture_count,
            duration_s=args.capture_duration,
            warmup_s=args.warmup,
            exposure_us=args.exposure_us,
            gain_db=args.gain_db,
        )
    else:
        images_dir = _resolve_path(args.images).resolve()
        frame_stems = list(args.frames)
    if reference_dir == images_dir and set(args.reference_frames) & set(frame_stems):
        raise ValueError("Reference and current frame sets must not overlap")

    camera = baseline["cameras"][args.serial]
    reference_images, reference_records = _load_frames(
        reference_dir, args.serial, list(args.reference_frames)
    )
    target_images, target_records = _load_frames(
        images_dir, args.serial, frame_stems
    )
    reference_hashes = {item["sha256"] for item in reference_records}
    target_hashes = {item["sha256"] for item in target_records}
    if reference_hashes & target_hashes:
        raise ValueError("Reference and current images contain identical frame content")

    line_reference, reference_metrics = build_line_reference(reference_images, camera)
    refined_R, refined_t, refinement_metrics, _ = refine_camera_pose(
        target_images, camera, line_reference
    )
    candidate = copy.deepcopy(baseline)
    if refinement_metrics["accepted"]:
        candidate["cameras"][args.serial]["R_world"] = refined_R.tolist()
        candidate["cameras"][args.serial]["t_world"] = refined_t.tolist()
        update_relative_extrinsics(candidate, args.serial)
    validate_calibration_geometry(candidate)
    changed_fields = validate_candidate_changes(baseline, candidate, args.serial)

    overlay_path = overlays_dir / f"{args.serial}.jpg"
    if not cv2.imwrite(
        str(overlay_path),
        _draw_overlay(target_images[0], camera, refined_R, refined_t),
    ):
        raise RuntimeError(f"Could not write overlay: {overlay_path}")

    candidate_text = _serialize_like_source(candidate, source_text)
    candidate_payload = candidate_text.encode("utf-8")
    _atomic_write_bytes(candidate_path, candidate_payload)
    report: dict[str, Any] = {
        "method": "18f_court_lines_plus_flying_ball_holdout_v1",
        "status": (
            "candidate_requires_independent_validation"
            if refinement_metrics["accepted"]
            else "court_line_candidate_rejected"
        ),
        "assumptions": [
            "Camera focus, zoom, intrinsic K/D, image resolution, and ROI are unchanged.",
            "Reference frames are independently reviewed and match the input calibration.",
            "Court-line geometry is the registered 18-floor singles court.",
        ],
        "inputs": {
            "calibration": str(calibration_path),
            "calibration_sha256": baseline_sha256,
            "target_serial": args.serial,
            "reference_images": str(reference_dir),
            "reference_frames": reference_records,
            "current_images": str(images_dir),
            "current_frames": target_records,
            "capture": capture_manifest,
        },
        "court_line": {
            "reference": reference_metrics,
            "refinement": refinement_metrics,
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256_bytes(candidate_payload),
            "changed_fields": changed_fields,
            "overlay": str(overlay_path),
        },
        "flying_ball_validation": None,
        "applied": False,
    }

    print(
        f"{args.serial}: {refinement_metrics['initial_rms_px']:.3f}px -> "
        f"{refinement_metrics['final_rms_px']:.3f}px, "
        f"rotation={refinement_metrics['rotation_change_deg']:.3f}deg, "
        f"position={refinement_metrics['position_change_mm']:.1f}mm, "
        f"accepted={refinement_metrics['accepted']}"
    )
    if not refinement_metrics["accepted"]:
        report["elapsed_s"] = time.perf_counter() - started
        _write_report(report_path, report)
        print("No court-line update passed the safety gates.")
        print(f"Report: {report_path}")
        return 2

    independent_supported = False
    if args.tracker_json:
        tracker_path = _resolve_path(args.tracker_json).resolve()
        flying_ball = validate_with_flying_ball(
            _load_json(tracker_path), baseline, candidate, args.serial
        )
        flying_ball["tracker_json"] = str(tracker_path)
        flying_ball["tracker_json_sha256"] = _sha256_path(tracker_path)
        report["flying_ball_validation"] = flying_ball
        holdout = flying_ball["holdout"]
        independent_supported = bool(holdout["supported"])
        print(
            "Flying-ball holdout: "
            f"n={holdout['baseline_px']['n']}, "
            f"median {_format_stat(holdout['baseline_px']['median'])} -> "
            f"{_format_stat(holdout['candidate_px']['median'])}px, "
            f"p95 {_format_stat(holdout['baseline_px']['p95'])} -> "
            f"{_format_stat(holdout['candidate_px']['p95'])}px, "
            f"candidate better={_format_stat(holdout['candidate_better_fraction'])}, "
            f"supported={independent_supported}"
        )
        report["status"] = (
            "independently_validated_candidate"
            if independent_supported
            else "independent_validation_rejected"
        )

    if args.apply:
        if not independent_supported:
            report["status"] = "apply_refused_by_independent_validation"
            report["elapsed_s"] = time.perf_counter() - started
            _write_report(report_path, report)
            print("Apply refused: independent flying-ball validation did not pass.")
            print(f"Candidate: {candidate_path}")
            print(f"Report: {report_path}")
            return 3
        backup_path = output_dir / f"{calibration_path.stem}_before_apply.json"
        try:
            promote_candidate(
                calibration_path,
                baseline_sha256,
                candidate_payload,
                backup_path,
            )
        except (FileExistsError, RuntimeError) as error:
            report["status"] = "apply_failed"
            report["apply_error"] = str(error)
            report["elapsed_s"] = time.perf_counter() - started
            _write_report(report_path, report)
            raise
        report["applied"] = True
        report["status"] = "applied_after_independent_validation"
        report["backup"] = {
            "path": str(backup_path),
            "sha256": _sha256_path(backup_path),
        }
        print(f"Applied calibration atomically: {calibration_path}")

    report["elapsed_s"] = time.perf_counter() - started
    _write_report(report_path, report)
    print(f"Candidate: {candidate_path}")
    print(f"Report: {report_path}")
    if args.tracker_json and not independent_supported:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
