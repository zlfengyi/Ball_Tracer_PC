# -*- coding: utf-8 -*-
"""Register the four-camera rig to a flat ground checkerboard."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibration.detect_corners_v2 import (
    MIN_PARITY_CONFIDENCE,
    canonicalize_corner_order,
    detect_corners,
)
from calibration.four_camera_calib_common import (
    DEFAULT_CALIB_CANDIDATE_PATH,
    PROJECT_ROOT,
)

_SQUARE_COLS = 12
_SQUARE_ROWS = 9
_INNER_COLS = _SQUARE_COLS - 1
_INNER_ROWS = _SQUARE_ROWS - 1
_SQUARE_SIZE_MM = 45.0
_MIN_STABLE_FRAMES = 3
_MAX_FRAME_STABILITY_RMS_PX = 0.25
_MAX_SINGLE_VIEW_RMS_PX = 0.5
_MAX_GROUND_RMS_PX = 1.0
_MAX_GROUND_VIEW_RMS_PX = 1.25
_MAX_GROUND_P95_PX = 2.0
_MAX_CONTROL_RMS_PX = 15.0
_MAX_CONTROL_ERROR_PX = 20.0
_MAX_CONTROL_BASELINE_RELATIVE_ERROR = 0.01


def make_ground_checkerboard_points(outer_x_m: float, outer_y_m: float) -> np.ndarray:
    """Return row-major inner corners in mm: columns +x, rows -y, z=0."""
    outer_x_mm = float(outer_x_m) * 1000.0
    outer_y_mm = float(outer_y_m) * 1000.0
    return np.array(
        [
            [
                outer_x_mm + (col + 1) * _SQUARE_SIZE_MM,
                outer_y_mm - (row + 1) * _SQUARE_SIZE_MM,
                0.0,
            ]
            for row in range(_INNER_ROWS)
            for col in range(_INNER_COLS)
        ],
        dtype=np.float64,
    )


def reproj_rms(world: np.ndarray, image: np.ndarray, rvec: np.ndarray,
               tvec: np.ndarray, K: np.ndarray, D: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(world, rvec, tvec, K, D)
    delta = projected.reshape(-1, 2) - image
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def aggregate_stable_corners(frames: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Median repeated, parity-canonical corner observations."""
    if len(frames) < _MIN_STABLE_FRAMES:
        raise RuntimeError(
            f"Need at least {_MIN_STABLE_FRAMES} stable checkerboard frames, got {len(frames)}"
        )
    stack = np.stack(frames).astype(np.float64)
    median = np.median(stack, axis=0)
    frame_rms = np.sqrt(np.mean(np.sum((stack - median) ** 2, axis=2), axis=1))
    worst = float(np.max(frame_rms))
    if worst > _MAX_FRAME_STABILITY_RMS_PX:
        raise RuntimeError(
            f"Checkerboard moved between captures: {worst:.3f}px > "
            f"{_MAX_FRAME_STABILITY_RMS_PX:.3f}px"
        )
    return median, worst


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _validate_session_board(session_dir: Path) -> None:
    metadata_path = session_dir / "session.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing session metadata: {metadata_path}")
    with open(metadata_path, encoding="utf-8") as inp:
        board = json.load(inp).get("board", {})

    square_dims = sorted([board.get("square_cols"), board.get("square_rows")])
    inner_dims = sorted([board.get("inner_cols"), board.get("inner_rows")])
    if square_dims != [_SQUARE_ROWS, _SQUARE_COLS]:
        raise RuntimeError(
            f"{session_dir.name}: expected {_SQUARE_ROWS}x{_SQUARE_COLS} squares, "
            f"got {board.get('square_rows')}x{board.get('square_cols')}"
        )
    if inner_dims != [_INNER_ROWS, _INNER_COLS]:
        raise RuntimeError(
            f"{session_dir.name}: expected {_INNER_ROWS}x{_INNER_COLS} inner corners"
        )
    if float(board.get("square_size_mm", 0.0)) != _SQUARE_SIZE_MM:
        raise RuntimeError(
            f"{session_dir.name}: expected {_SQUARE_SIZE_MM:g}mm squares, "
            f"got {board.get('square_size_mm')!r}"
        )


def _detect_session(
    session_dir: Path,
    serials: list[str],
    world_mm: np.ndarray,
) -> tuple[list[dict], dict]:
    pattern = (_INNER_COLS, _INNER_ROWS)
    observations = []
    session_diagnostics = {"path": _relative_path(session_dir), "cameras": {}}

    for serial in serials:
        image_paths = sorted((session_dir / serial).glob("*.png"))
        frames = []
        for image_path in image_paths:
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Unreadable image: {image_path}")
            corners, _ = detect_corners(gray, pattern)
            if corners is None:
                continue
            canonical, parity, _ = canonicalize_corner_order(
                gray, corners, _INNER_COLS, _INNER_ROWS
            )
            if parity < MIN_PARITY_CONFIDENCE:
                continue
            frames.append(canonical.reshape(-1, 2))

        if not frames:
            continue
        if len(frames) < _MIN_STABLE_FRAMES:
            raise RuntimeError(
                f"{session_dir.name}/{serial}: only {len(frames)} reliable detections"
            )
        image, stability_rms = aggregate_stable_corners(frames)
        observations.append(
            {
                "session": session_dir.name,
                "serial": serial,
                "world_mm": world_mm,
                "image": image,
            }
        )
        session_diagnostics["cameras"][serial] = {
            "detected_frames": len(frames),
            "captured_frames": len(image_paths),
            "max_frame_stability_rms_px": stability_rms,
        }

    if len(observations) < 2:
        raise RuntimeError(
            f"{session_dir.name}: checkerboard must be reliably visible in at least two cameras"
        )
    return observations, session_diagnostics


def _load_ground_controls(
    session_dir: Path, frame: str, serials: list[str]
) -> tuple[list[dict], dict[str, str]]:
    controls = []
    annotation_paths = {}
    for serial in serials:
        path = session_dir / serial / f"{frame}_annotations.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing ground control annotations: {path}")
        with open(path, encoding="utf-8") as inp:
            data = json.load(inp)
        entries = [data[key] for key in sorted(data, key=int)]
        world_m = np.asarray([entry[0] for entry in entries], dtype=np.float64)
        image = np.asarray([entry[1] for entry in entries], dtype=np.float64)
        if (world_m.ndim != 2 or world_m.shape[1:] != (3,)
                or image.shape != (len(world_m), 2) or len(world_m) == 0
                or not np.all(np.isfinite(world_m))
                or not np.all(np.isfinite(image))
                or not np.all(world_m[:, 2] == 0.0)):
            raise RuntimeError(f"Invalid ground control annotations: {path}")
        controls.append(
            {
                "serial": serial,
                "world_mm": world_m * 1000.0,
                "image": image,
            }
        )
        annotation_paths[serial] = _relative_path(path)

    unique_points = np.unique(
        np.concatenate([item["world_mm"] for item in controls]), axis=0
    )
    if len(unique_points) < 2:
        raise RuntimeError("Ground registration requires at least two control points")
    return controls, annotation_paths


def _camera_arrays(calib: dict) -> dict[str, dict[str, np.ndarray]]:
    arrays = {}
    for serial, camera in calib["cameras"].items():
        arrays[serial] = {
            "K": np.asarray(camera["K"], dtype=np.float64).reshape(3, 3),
            "D": np.asarray(camera["D"], dtype=np.float64).ravel(),
            "R_relative": np.asarray(
                camera["R_ref_to_camera"], dtype=np.float64
            ).reshape(3, 3),
            "t_relative": np.asarray(
                camera["t_ref_to_camera"], dtype=np.float64
            ).reshape(3, 1),
        }
    return arrays


def _calibrate_control_scale(calib: dict, controls: list[dict]) -> dict:
    cameras = _camera_arrays(calib)
    previous = (
        calib.get("diagnostics", {})
        .get("ground_registration", {})
        .get("ground_controls", {})
        .get("baseline_scale_calibration", {})
    )
    grouped: dict[tuple[float, float, float], list[tuple[str, np.ndarray]]] = {}
    for control in controls:
        for world_mm, pixel in zip(control["world_mm"], control["image"]):
            key = tuple(float(value) for value in world_mm)
            grouped.setdefault(key, []).append((control["serial"], pixel))

    triangulated = []
    for world_mm, observations in grouped.items():
        if len(observations) < 2:
            raise RuntimeError(
                f"Ground control point {world_mm} is visible in fewer than two cameras"
            )

        dlt_rows = []
        views = []
        for serial, pixel in observations:
            camera = cameras[serial]
            normalized = cv2.undistortPoints(
                np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
                camera["K"],
                camera["D"],
            ).reshape(2)
            projection = np.concatenate(
                [camera["R_relative"], camera["t_relative"]], axis=1
            )
            dlt_rows.extend(
                [
                    normalized[0] * projection[2] - projection[0],
                    normalized[1] * projection[2] - projection[1],
                ]
            )
            rvec, _ = cv2.Rodrigues(camera["R_relative"])
            views.append((serial, np.asarray(pixel), camera, rvec))

        _, _, vt = np.linalg.svd(np.asarray(dlt_rows, dtype=np.float64))
        homogeneous = vt[-1]
        if abs(float(homogeneous[3])) < 1e-12:
            raise RuntimeError(f"Ground control triangulation failed: {world_mm}")
        initial = homogeneous[:3] / homogeneous[3]

        def residuals(point: np.ndarray) -> np.ndarray:
            values = []
            for _, pixel, camera, rvec in views:
                projected, _ = cv2.projectPoints(
                    point.reshape(1, 3),
                    rvec,
                    camera["t_relative"],
                    camera["K"],
                    camera["D"],
                )
                values.append(projected.reshape(2) - pixel)
            return np.concatenate(values)

        result = least_squares(residuals, initial, method="trf", max_nfev=500)
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Ground control triangulation failed: {world_mm}")
        for serial, _, camera, _ in views:
            depth = float(
                (camera["R_relative"] @ result.x.reshape(3, 1)
                 + camera["t_relative"])[2, 0]
            )
            if depth <= 0.0:
                raise RuntimeError(
                    f"Triangulated ground control is behind camera {serial}: {world_mm}"
                )

        pixel_errors = np.linalg.norm(residuals(result.x).reshape(-1, 2), axis=1)
        triangulated.append(
            {
                "world_mm": np.asarray(world_mm, dtype=np.float64),
                "reference_mm": result.x,
                "cameras": [serial for serial, _, _, _ in views],
                "reprojection_rms_px": float(
                    np.sqrt(np.mean(pixel_errors * pixel_errors))
                ),
            }
        )

    first, second = max(
        itertools.combinations(triangulated, 2),
        key=lambda pair: np.linalg.norm(pair[1]["world_mm"] - pair[0]["world_mm"]),
    )
    expected_mm = float(np.linalg.norm(second["world_mm"] - first["world_mm"]))
    measured_mm = float(np.linalg.norm(second["reference_mm"] - first["reference_mm"]))
    error_before_scale_mm = measured_mm - expected_mm
    relative_error = abs(error_before_scale_mm) / expected_mm
    if relative_error > _MAX_CONTROL_BASELINE_RELATIVE_ERROR:
        raise RuntimeError(
            "Ground control triangulated distance failed: "
            f"measured={measured_mm:.1f}mm, expected={expected_mm:.1f}mm, "
            f"error={error_before_scale_mm:+.1f}mm "
            f"({100.0 * relative_error:.3f}%)"
        )

    scale = expected_mm / measured_mm
    for camera in calib["cameras"].values():
        translation = np.asarray(
            camera["t_ref_to_camera"], dtype=np.float64
        ).reshape(3, 1)
        camera["t_ref_to_camera"] = (translation * scale).tolist()
    for pair in (
        calib.get("diagnostics", {})
        .get("epipolar_residual", {})
        .get("pairs", {})
        .values()
    ):
        pair["baseline_mm"] = float(pair["baseline_mm"]) * scale

    corrected_mm = measured_mm * scale
    initial_mm = float(previous.get("initial_triangulated_mm", measured_mm))
    cumulative_scale = float(previous.get("cumulative_scale_factor", 1.0)) * scale
    return {
        "expected_mm": expected_mm,
        "initial_triangulated_mm": initial_mm,
        "initial_error_mm": initial_mm - expected_mm,
        "input_triangulated_mm": measured_mm,
        "input_error_mm": error_before_scale_mm,
        "input_relative_error_pct": 100.0 * relative_error,
        "applied_scale_factor": scale,
        "cumulative_scale_factor": cumulative_scale,
        "triangulated_mm": corrected_mm,
        "error_mm": corrected_mm - expected_mm,
        "max_initial_relative_error_pct": (
            100.0 * _MAX_CONTROL_BASELINE_RELATIVE_ERROR
        ),
        "points": [
            {
                "world_m": (item["world_mm"] / 1000.0).tolist(),
                "triangulated_reference_mm": (
                    item["reference_mm"] * scale
                ).tolist(),
                "cameras": item["cameras"],
                "reprojection_rms_px": item["reprojection_rms_px"],
            }
            for item in (first, second)
        ],
    }


def _camera_pose(reference_pose: np.ndarray, camera: dict[str, np.ndarray]
                 ) -> tuple[np.ndarray, np.ndarray]:
    R_reference, _ = cv2.Rodrigues(reference_pose[:3])
    t_reference = reference_pose[3:].reshape(3, 1)
    R = camera["R_relative"] @ R_reference
    t = camera["R_relative"] @ t_reference + camera["t_relative"]
    return R, t


def _oriented_image(observation: dict, flipped_sessions: dict[str, bool]) -> np.ndarray:
    image = observation["image"]
    return image[::-1] if flipped_sessions[observation["session"]] else image


def _project(
    reference_pose: np.ndarray,
    observation: dict,
    cameras: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    camera = cameras[observation["serial"]]
    R, t = _camera_pose(reference_pose, camera)
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(
        observation["world_mm"], rvec, t, camera["K"], camera["D"]
    )
    depths = (R @ observation["world_mm"].T + t)[2]
    return projected.reshape(-1, 2), depths


def _residuals(
    reference_pose: np.ndarray,
    observations: list[dict],
    cameras: dict[str, dict[str, np.ndarray]],
    flipped_sessions: dict[str, bool],
    controls: list[dict],
) -> np.ndarray:
    residuals = []
    for observation in observations:
        projected, _ = _project(reference_pose, observation, cameras)
        residuals.append(
            (projected - _oriented_image(observation, flipped_sessions)).ravel()
        )
    for control in controls:
        projected, _ = _project(reference_pose, control, cameras)
        residuals.append((projected - control["image"]).ravel())
    return np.concatenate(residuals)


def _initial_reference_pose(
    observation: dict,
    camera: dict[str, np.ndarray],
    flipped_sessions: dict[str, bool],
) -> np.ndarray:
    image = _oriented_image(observation, flipped_sessions)
    ok, rvec, t = cv2.solvePnP(
        observation["world_mm"], image, camera["K"], camera["D"],
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError(
            f"solvePnP failed: {observation['session']}/{observation['serial']}"
        )
    view_rms = reproj_rms(
        observation["world_mm"], image, rvec, t, camera["K"], camera["D"]
    )
    if view_rms > _MAX_SINGLE_VIEW_RMS_PX:
        raise RuntimeError(
            f"Single-view checkerboard RMS too high for "
            f"{observation['session']}/{observation['serial']}: {view_rms:.3f}px"
        )

    R_camera, _ = cv2.Rodrigues(rvec)
    R_reference = camera["R_relative"].T @ R_camera
    t_reference = camera["R_relative"].T @ (t - camera["t_relative"])
    rvec_reference, _ = cv2.Rodrigues(R_reference)
    return np.concatenate([rvec_reference.ravel(), t_reference.ravel()])


def _solve_orientation(
    observations: list[dict],
    cameras: dict[str, dict[str, np.ndarray]],
    flipped_sessions: dict[str, bool],
    controls: list[dict],
) -> tuple[float, np.ndarray]:
    best = None
    for observation in observations:
        x0 = _initial_reference_pose(
            observation, cameras[observation["serial"]], flipped_sessions
        )
        result = least_squares(
            _residuals,
            x0,
            args=(observations, cameras, flipped_sessions, controls),
            method="trf",
            max_nfev=2000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        point_errors = _residuals(
            result.x, observations, cameras, flipped_sessions, []
        ).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(point_errors * point_errors, axis=1))))
        if best is None or rms < best[0]:
            best = (rms, result.x)
    if best is None:
        raise RuntimeError(f"Ground optimization failed for orientation {flipped_sessions}")
    return best


def solve_ground_pose(
    calib: dict,
    observations: list[dict],
    session_names: list[str],
    controls: list[dict],
) -> tuple[np.ndarray, dict[str, bool], list[tuple[float, dict[str, bool]]]]:
    cameras = _camera_arrays(calib)
    candidates = []
    for flags in itertools.product((False, True), repeat=len(session_names)):
        flips = dict(zip(session_names, flags))
        rms, pose = _solve_orientation(observations, cameras, flips, controls)
        candidates.append((rms, flips, pose))

    candidates.sort(key=lambda item: item[0])
    passing = [item for item in candidates if item[0] <= _MAX_GROUND_RMS_PX]
    if len(passing) != 1:
        scores = ", ".join(
            f"{item[1]}={item[0]:.3f}px" for item in candidates
        )
        raise RuntimeError(f"Ground corner orientation is not uniquely valid: {scores}")
    selected = passing[0]
    scores = [(item[0], item[1]) for item in candidates]
    return selected[2], selected[1], scores


def _validate_and_apply(
    calib: dict,
    reference_pose: np.ndarray,
    observations: list[dict],
    flipped_sessions: dict[str, bool],
    controls: list[dict],
) -> dict:
    cameras = _camera_arrays(calib)
    all_errors = []
    camera_errors: dict[str, list[np.ndarray]] = {}
    observation_metrics = {}

    for observation in observations:
        serial = observation["serial"]
        projected, depths = _project(reference_pose, observation, cameras)
        if np.any(depths <= 0.0):
            raise RuntimeError(
                f"Ground checkerboard has non-positive depth: "
                f"{observation['session']}/{serial}"
            )
        delta = projected - _oriented_image(
            observation, flipped_sessions
        )
        errors = np.linalg.norm(delta, axis=1)
        rms = float(np.sqrt(np.mean(errors * errors)))
        p95 = float(np.percentile(errors, 95))
        if rms > _MAX_GROUND_VIEW_RMS_PX or p95 > _MAX_GROUND_P95_PX:
            raise RuntimeError(
                f"Ground reprojection failed for {observation['session']}/{serial}: "
                f"RMS={rms:.3f}px, P95={p95:.3f}px"
            )
        key = f"{observation['session']}/{serial}"
        observation_metrics[key] = {
            "rms_px": rms,
            "p95_px": p95,
            "max_px": float(np.max(errors)),
        }
        all_errors.append(errors)
        camera_errors.setdefault(serial, []).append(errors)

    combined = np.concatenate(all_errors)
    total_rms = float(np.sqrt(np.mean(combined * combined)))
    total_p95 = float(np.percentile(combined, 95))
    if total_rms > _MAX_GROUND_RMS_PX or total_p95 > _MAX_GROUND_P95_PX:
        raise RuntimeError(
            f"Ground reprojection failed: RMS={total_rms:.3f}px, P95={total_p95:.3f}px"
        )

    control_errors = []
    control_metrics = {}
    for control in controls:
        projected, depths = _project(reference_pose, control, cameras)
        if np.any(depths <= 0.0):
            raise RuntimeError(
                f"Ground control point has non-positive depth: {control['serial']}"
            )
        delta = projected - control["image"]
        errors = np.linalg.norm(delta, axis=1)
        control_errors.append(errors)
        for index, error in enumerate(errors):
            key = f"{control['serial']}/{index + 1}"
            control_metrics[key] = {
                "world_m": (control["world_mm"][index] / 1000.0).tolist(),
                "observed_pixel": control["image"][index].tolist(),
                "projected_pixel": projected[index].tolist(),
                "error_px": float(error),
            }

    control_combined = np.concatenate(control_errors)
    control_rms = float(np.sqrt(np.mean(control_combined * control_combined)))
    control_max = float(np.max(control_combined))
    if control_rms > _MAX_CONTROL_RMS_PX or control_max > _MAX_CONTROL_ERROR_PX:
        raise RuntimeError(
            f"Ground control reprojection failed: RMS={control_rms:.3f}px, "
            f"max={control_max:.3f}px"
        )

    camera_metrics = {}
    camera_positions = {}
    for serial, camera in cameras.items():
        R, t = _camera_pose(reference_pose, camera)
        position = (-R.T @ t).ravel()
        if not np.all(np.isfinite(position)) or position[2] <= 0.0:
            raise RuntimeError(f"Invalid world pose for camera {serial}: {position.tolist()}")
        calib["cameras"][serial]["R_world"] = R.tolist()
        calib["cameras"][serial]["t_world"] = t.reshape(3, 1).tolist()
        calib["cameras"][serial]["pos_world"] = position.reshape(3, 1).tolist()
        camera_positions[serial] = position

        if serial in camera_errors:
            errors = np.concatenate(camera_errors[serial])
            camera_metrics[serial] = {
                "rms_px": float(np.sqrt(np.mean(errors * errors))),
                "p95_px": float(np.percentile(errors, 95)),
            }

    return {
        "total_rms_px": total_rms,
        "total_p95_px": total_p95,
        "per_camera": camera_metrics,
        "per_observation": observation_metrics,
        "controls": {
            "rms_px": control_rms,
            "p95_px": float(np.percentile(control_combined, 95)),
            "max_px": control_max,
            "per_point": control_metrics,
        },
        "camera_positions": camera_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register four cameras from two flat boards and ground controls."
    )
    parser.add_argument(
        "--board",
        action="append",
        nargs=3,
        required=True,
        metavar=("SESSION", "OUTER_X_M", "OUTER_Y_M"),
        help="Session path and the first square's outer top-left world coordinate.",
    )
    parser.add_argument(
        "--ground-frame",
        nargs=2,
        required=True,
        metavar=("SESSION", "FRAME"),
        help="Session path and frame stem containing ground control annotations.",
    )
    parser.add_argument(
        "--calib", default=str(DEFAULT_CALIB_CANDIDATE_PATH),
        help="Input calibration candidate.",
    )
    parser.add_argument(
        "--output", default=None, help="Output path; defaults to overwriting --calib."
    )
    args = parser.parse_args()

    if len(args.board) != 2:
        raise RuntimeError("Ground registration requires exactly two board sessions")

    calib_path = _resolve_path(args.calib)
    output_path = calib_path if args.output is None else _resolve_path(args.output)
    with open(calib_path, encoding="utf-8") as inp:
        calib = json.load(inp)

    serials = list(calib["cameras"])
    observations = []
    session_names = []
    session_diagnostics = []

    for raw_session, raw_x, raw_y in args.board:
        session_dir = _resolve_path(raw_session)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Ground checkerboard session not found: {session_dir}")
        if session_dir.name in session_names:
            raise RuntimeError(f"Duplicate board session: {session_dir.name}")
        try:
            outer_x_m = float(raw_x)
            outer_y_m = float(raw_y)
        except ValueError as exc:
            raise RuntimeError(f"Invalid board coordinate: {raw_x}, {raw_y}") from exc
        if not np.isfinite([outer_x_m, outer_y_m]).all():
            raise RuntimeError(f"Non-finite board coordinate: {raw_x}, {raw_y}")

        _validate_session_board(session_dir)
        world_mm = make_ground_checkerboard_points(outer_x_m, outer_y_m)
        detected, diagnostics = _detect_session(session_dir, serials, world_mm)
        diagnostics.update(
            {
                "outer_square_top_left_m": [outer_x_m, outer_y_m, 0.0],
                "first_inner_corner_m": [
                    float(world_mm[0, 0] / 1000.0),
                    float(world_mm[0, 1] / 1000.0),
                    0.0,
                ],
            }
        )
        observations.extend(detected)
        session_names.append(session_dir.name)
        session_diagnostics.append(diagnostics)

    control_session = _resolve_path(args.ground_frame[0])
    control_frame = args.ground_frame[1]
    controls, control_annotation_paths = _load_ground_controls(
        control_session, control_frame, serials
    )
    control_baseline = _calibrate_control_scale(calib, controls)
    reference_pose, flipped_sessions, orientation_scores = solve_ground_pose(
        calib, observations, session_names, controls
    )
    metrics = _validate_and_apply(
        calib, reference_pose, observations, flipped_sessions, controls
    )

    registered_perf_counter_s = time.perf_counter()
    diagnostics = calib.setdefault("diagnostics", {})
    diagnostics["ground_reproj_error"] = metrics["total_rms_px"]
    diagnostics["ground_registration"] = {
        "method": "two_flat_checkerboards_plus_ground_controls",
        "registered_perf_counter_s": registered_perf_counter_s,
        "reference_serial": calib["reference_serial"],
        "board": {
            "square_cols": _SQUARE_COLS,
            "square_rows": _SQUARE_ROWS,
            "inner_cols": _INNER_COLS,
            "inner_rows": _INNER_ROWS,
            "square_size_mm": _SQUARE_SIZE_MM,
        },
        "sessions": session_diagnostics,
        "reversed_from_parity_canonical": flipped_sessions,
        "orientation_candidate_rms_px": [
            {"reversed": flips, "rms_px": rms}
            for rms, flips in orientation_scores
        ],
        "num_observed_cameras": len({item["serial"] for item in observations}),
        "num_registered_cameras": len(serials),
        "num_ground_corners": sum(len(item["world_mm"]) for item in observations),
        "total_rms_px": metrics["total_rms_px"],
        "total_p95_px": metrics["total_p95_px"],
        "per_camera": metrics["per_camera"],
        "per_observation": metrics["per_observation"],
        "ground_controls": {
            "session": _relative_path(control_session),
            "frame": control_frame,
            "annotation_files": control_annotation_paths,
            "baseline_scale_calibration": control_baseline,
            **metrics["controls"],
        },
    }
    calib.pop("config_written_at", None)
    calib["config_written_perf_counter_s"] = registered_perf_counter_s

    sources = calib.setdefault("sources", {})
    sources["ground_registration"] = {
        "method": "two_flat_checkerboards_plus_ground_controls",
        "input_calib": _relative_path(calib_path),
        "sessions": [item["path"] for item in session_diagnostics],
        "ground_control_annotations": control_annotation_paths,
        "relative_translation_scale_factor": control_baseline[
            "cumulative_scale_factor"
        ],
        "registered_perf_counter_s": registered_perf_counter_s,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(calib, out, indent=4, ensure_ascii=False)

    print("Ground registration passed")
    print(
        f"  checkerboard: RMS={metrics['total_rms_px']:.3f}px, "
        f"P95={metrics['total_p95_px']:.3f}px"
    )
    print(
        f"  ground controls: RMS={metrics['controls']['rms_px']:.3f}px, "
        f"P95={metrics['controls']['p95_px']:.3f}px, "
        f"max={metrics['controls']['max_px']:.3f}px"
    )
    print(
        f"  triangulated baseline before scale: "
        f"{control_baseline['input_triangulated_mm']:.1f}mm "
        f"(expected {control_baseline['expected_mm']:.1f}mm, "
        f"error {control_baseline['input_error_mm']:+.1f}mm)"
    )
    print(
        f"  relative translation scale: "
        f"applied={control_baseline['applied_scale_factor']:.9f}, "
        f"cumulative={control_baseline['cumulative_scale_factor']:.9f}"
    )
    print(f"  selected corner reversal: {flipped_sessions}")
    for serial, position in metrics["camera_positions"].items():
        print(
            f"  {serial}: position=({position[0]:.1f}, {position[1]:.1f}, "
            f"{position[2]:.1f})mm"
        )
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()
