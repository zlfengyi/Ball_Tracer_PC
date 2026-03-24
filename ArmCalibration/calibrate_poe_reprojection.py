# -*- coding: utf-8 -*-
"""15.4: calibrate a position-only POE model from direct 2D reprojection."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ArmCalibration.common import ARM_DATA_ROOT, load_json, rel_or_abs, save_json


DEFAULT_CAMERA_CALIB_PATH = PROJECT_ROOT / "src" / "config" / "four_camera_calib.json"
DEFAULT_EXPORT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "arm_poe_racket_center.json"
APRILTAG_TO_CAR_BASE_OFFSET_M = [0.06, 0.10, -0.34]
APRILTAG_TO_CAR_BASE_OFFSET_MM = [value * 1000.0 for value in APRILTAG_TO_CAR_BASE_OFFSET_M]


@dataclass(frozen=True)
class CameraModel:
    serial: str
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray
    rvec: np.ndarray


@dataclass(frozen=True)
class CameraObservation:
    serial: str
    uv: np.ndarray


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    q4: np.ndarray
    q5: float
    observations: tuple[CameraObservation, ...]


@dataclass(frozen=True)
class TriangulatedSample:
    sample: SampleRecord
    point_world: np.ndarray
    mean_reproj_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="15.4 calibrate a position-only POE model using direct 2D reprojection.",
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session directory path or ArmCalibration/data/<session_name>.",
    )
    parser.add_argument(
        "--camera-calib",
        type=str,
        default=str(DEFAULT_CAMERA_CALIB_PATH),
        help="Path to four_camera_calib.json.",
    )
    parser.add_argument(
        "--init-reproj-threshold",
        type=float,
        default=15.0,
        help="Keep triangulated samples below this reprojection error for initialization only.",
    )
    parser.add_argument(
        "--loss-scale-px",
        type=float,
        default=5.0,
        help="Soft-L1 scale for the final 2D reprojection optimization.",
    )
    parser.add_argument(
        "--outlier-threshold-px",
        type=float,
        default=20.0,
        help="Residual threshold used only for reporting outliers.",
    )
    parser.add_argument(
        "--max-nfev-structured",
        type=int,
        default=1200,
        help="Max function evaluations for each structured 3D initialization seed.",
    )
    parser.add_argument(
        "--max-nfev-generic-3d",
        type=int,
        default=3000,
        help="Max function evaluations for generic 3D initialization.",
    )
    parser.add_argument(
        "--max-nfev-2d",
        type=int,
        default=3000,
        help="Max function evaluations for the final direct 2D optimization.",
    )
    parser.add_argument(
        "--result-name",
        type=str,
        default="poe_reprojection_result.json",
        help="Output JSON filename inside the session directory.",
    )
    parser.add_argument(
        "--base-origin-mode",
        type=str,
        choices=("joint1_joint2_closest", "joint1_ground_intersection"),
        default="joint1_ground_intersection",
        help="How to define the exported base origin from the fitted joint_1 axis.",
    )
    parser.add_argument(
        "--ground-plane-z",
        type=float,
        default=0.0,
        help="World Z of the ground plane. The ground registration in this repo uses z=0.",
    )
    parser.add_argument(
        "--export-config",
        type=str,
        default=str(DEFAULT_EXPORT_CONFIG_PATH),
        help="Path to the exported POE config JSON for downstream use.",
    )
    return parser.parse_args()


def resolve_session_dir(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.exists():
        return candidate.resolve()
    candidate = ARM_DATA_ROOT / raw
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Session directory not found: {raw}")


def load_camera_models(path: Path) -> dict[str, CameraModel]:
    payload = load_json(path)
    cameras: dict[str, CameraModel] = {}
    for serial, cd in payload["cameras"].items():
        K = np.array(cd["K"], dtype=np.float64).reshape(3, 3)
        D = np.array(cd["D"], dtype=np.float64).reshape(-1)
        R = np.array(cd["R_world"], dtype=np.float64).reshape(3, 3)
        t = np.array(cd["t_world"], dtype=np.float64).reshape(3, 1)
        P = K @ np.hstack([R, t])
        rvec, _ = cv2.Rodrigues(R)
        cameras[serial] = CameraModel(
            serial=serial,
            K=K,
            D=D,
            R=R,
            t=t,
            P=P,
            rvec=rvec,
        )
    return cameras


def collect_dataset(session_dir: Path) -> tuple[list[SampleRecord], dict[str, int]]:
    sample_dirs = sorted(
        entry
        for entry in session_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("sample_")
    )
    samples: list[SampleRecord] = []
    stats = {
        "sample_dir_count": len(sample_dirs),
        "samples_with_joint_state": 0,
        "samples_missing_joint_state": 0,
        "samples_with_1plus_cameras": 0,
        "samples_with_2plus_cameras": 0,
        "accepted_image_observations": 0,
    }

    for sample_dir in sample_dirs:
        payload = load_json(sample_dir / "sample.json")
        joint_state = payload.get("joint_state")
        if not isinstance(joint_state, dict):
            stats["samples_missing_joint_state"] += 1
            continue

        stats["samples_with_joint_state"] += 1
        q = np.array(joint_state["position"], dtype=np.float64)
        observations: list[CameraObservation] = []
        cameras = payload.get("racket_pose", {}).get("cameras", {})
        for serial, cam_obs in cameras.items():
            if not cam_obs.get("accepted"):
                continue
            center_xy = cam_obs.get("center_xy")
            if center_xy is None:
                continue
            observations.append(
                CameraObservation(serial=serial, uv=np.array(center_xy, dtype=np.float64))
            )

        if len(observations) < 1:
            continue

        stats["samples_with_1plus_cameras"] += 1
        if len(observations) >= 2:
            stats["samples_with_2plus_cameras"] += 1
        stats["accepted_image_observations"] += len(observations)
        samples.append(
            SampleRecord(
                sample_id=sample_dir.name,
                q4=q[:4].astype(np.float64),
                q5=float(q[4]),
                observations=tuple(observations),
            )
        )

    return samples, stats


def undistort_pixel(camera: CameraModel, uv: np.ndarray) -> np.ndarray:
    pts = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, camera.K, camera.D, P=camera.K)
    return undist[0, 0]


def triangulate_sample(sample: SampleRecord, cameras: dict[str, CameraModel]) -> TriangulatedSample:
    A_rows: list[np.ndarray] = []
    undistorted: dict[str, np.ndarray] = {}
    for obs in sample.observations:
        cam = cameras[obs.serial]
        uv = undistort_pixel(cam, obs.uv)
        undistorted[obs.serial] = uv
        A_rows.append(uv[0] * cam.P[2] - cam.P[0])
        A_rows.append(uv[1] * cam.P[2] - cam.P[1])
    A = np.array(A_rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    homogeneous = vt[-1]
    point_world = homogeneous[:3] / homogeneous[3]

    reproj: list[float] = []
    ph = np.append(point_world, 1.0)
    for obs in sample.observations:
        cam = cameras[obs.serial]
        proj = cam.P @ ph
        uv_pred = proj[:2] / proj[2]
        reproj.append(float(np.linalg.norm(uv_pred - obs.uv)))

    return TriangulatedSample(
        sample=sample,
        point_world=point_world.astype(np.float64),
        mean_reproj_px=float(np.mean(reproj)),
    )


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def forward_structured_base(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    q1, q2, q3, q4 = q
    l1, l2, l3, l4 = lengths

    R1 = rot_z(float(q1))
    p1 = np.array([0.0, 0.0, l1], dtype=np.float64)

    R2 = R1 @ rot_y(float(q2))
    p2 = p1 + R2 @ np.array([l2, 0.0, 0.0], dtype=np.float64)

    R3 = R2 @ rot_y(float(q3))
    p3 = p2 + R3 @ np.array([l3, 0.0, 0.0], dtype=np.float64)

    R4 = R3 @ rot_y(float(q4))
    p4 = p3 + R4 @ np.array([l4, 0.0, 0.0], dtype=np.float64)
    return p4


def fit_structured_init(samples: list[TriangulatedSample], max_nfev: int) -> np.ndarray:
    q = np.array([item.sample.q4 for item in samples], dtype=np.float64)
    points = np.array([item.point_world for item in samples], dtype=np.float64)
    centroid = points.mean(axis=0)

    init = np.array(
        [
            1.2,
            -0.2,
            2.3,
            centroid[0],
            centroid[1] - 600.0,
            max(0.0, float(points[:, 2].min()) - 100.0),
            300.0,
            450.0,
            400.0,
            250.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    lower = np.array(
        [
            -math.pi,
            -math.pi,
            -math.pi,
            -3000.0,
            -3000.0,
            -500.0,
            50.0,
            50.0,
            50.0,
            20.0,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            math.pi,
            math.pi,
            math.pi,
            3000.0,
            5000.0,
            3000.0,
            1500.0,
            1500.0,
            1500.0,
            1000.0,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
        ],
        dtype=np.float64,
    )

    def residual(params: np.ndarray) -> np.ndarray:
        rvec = params[:3]
        translation = params[3:6]
        lengths = params[6:10]
        offsets = params[10:14]
        rotation, _ = cv2.Rodrigues(rvec)
        pred = np.array(
            [rotation @ forward_structured_base(angles + offsets, lengths) + translation for angles in q],
            dtype=np.float64,
        )
        return (pred - points).reshape(-1)

    seeds: list[np.ndarray] = []
    for rz in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
        for ry in (0.0, 0.6):
            seed = init.copy()
            seed[1] = ry
            seed[2] = rz
            seeds.append(seed)

    best: least_squares | None = None
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=50.0,
            max_nfev=max_nfev,
        )
        if best is None or result.cost < best.cost:
            best = result

    if best is None:
        raise RuntimeError("failed to compute structured initialization")
    return best.x.astype(np.float64)


def structured_to_generic_world(params: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(params[:3])
    translation = params[3:6]
    l1, l2, l3, l4 = params[6:10]
    offsets = params[10:14]

    base_axes = [
        (np.array([0.0, 0.0, 1.0], dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        (np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, l1], dtype=np.float64)),
        (np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([l2, 0.0, l1], dtype=np.float64)),
        (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([l2 + l3, 0.0, l1], dtype=np.float64),
        ),
    ]
    home_point_base = np.array([l2 + l3 + l4, 0.0, l1], dtype=np.float64)

    generic: list[float] = []
    for axis_dir_base, axis_point_base in base_axes:
        axis_dir_world = rotation @ axis_dir_base
        axis_point_world = rotation @ axis_point_base + translation
        generic.extend(axis_dir_world.tolist())
        generic.extend(axis_point_world.tolist())
    generic.extend((rotation @ home_point_base + translation).tolist())
    generic.extend(offsets.tolist())
    return np.array(generic, dtype=np.float64)


def exp_revolute(axis_direction: np.ndarray, axis_point: np.ndarray, angle: float) -> np.ndarray:
    direction = axis_direction / np.linalg.norm(axis_direction)
    R, _ = cv2.Rodrigues((direction * angle).reshape(3, 1))
    t = (np.eye(3, dtype=np.float64) - R) @ axis_point
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def normalize_direction(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def forward_generic_point(params: np.ndarray, q4: np.ndarray) -> np.ndarray:
    point_home = params[24:27]
    offsets = params[27:31]
    T = np.eye(4, dtype=np.float64)
    idx = 0
    for joint_idx in range(4):
        axis_direction = normalize_direction(params[idx : idx + 3])
        idx += 3
        axis_point = params[idx : idx + 3]
        idx += 3
        T = T @ exp_revolute(axis_direction, axis_point, float(q4[joint_idx] + offsets[joint_idx]))
    ph = T @ np.array([point_home[0], point_home[1], point_home[2], 1.0], dtype=np.float64)
    return ph[:3]


def fit_generic_3d_init(
    init_params: np.ndarray,
    samples: list[TriangulatedSample],
    max_nfev: int,
) -> np.ndarray:
    q = np.array([item.sample.q4 for item in samples], dtype=np.float64)
    points = np.array([item.point_world for item in samples], dtype=np.float64)

    lower: list[float] = []
    upper: list[float] = []
    for _ in range(4):
        lower.extend([-2.0, -2.0, -2.0, -4000.0, -4000.0, -1000.0])
        upper.extend([2.0, 2.0, 2.0, 4000.0, 5000.0, 4000.0])
    lower.extend([-4000.0, -4000.0, -1000.0, -math.pi, -math.pi, -math.pi, -math.pi])
    upper.extend([4000.0, 5000.0, 4000.0, math.pi, math.pi, math.pi, math.pi])

    def residual(params: np.ndarray) -> np.ndarray:
        pred = np.array([forward_generic_point(params, angles) for angles in q], dtype=np.float64)
        return (pred - points).reshape(-1)

    result = least_squares(
        residual,
        init_params,
        bounds=(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)),
        loss="soft_l1",
        f_scale=50.0,
        max_nfev=max_nfev,
    )
    return result.x.astype(np.float64)


def project_point(camera: CameraModel, point_world: np.ndarray) -> np.ndarray:
    img, _ = cv2.projectPoints(
        point_world.reshape(1, 1, 3),
        camera.rvec,
        camera.t,
        camera.K,
        camera.D,
    )
    return img.reshape(2)


def reprojection_residual_vector(
    params: np.ndarray,
    samples: Iterable[SampleRecord],
    cameras: dict[str, CameraModel],
) -> np.ndarray:
    values: list[float] = []
    for sample in samples:
        point_world = forward_generic_point(params, sample.q4)
        for obs in sample.observations:
            uv_pred = project_point(cameras[obs.serial], point_world)
            values.append(float(uv_pred[0] - obs.uv[0]))
            values.append(float(uv_pred[1] - obs.uv[1]))
    return np.array(values, dtype=np.float64)


def fit_direct_reprojection(
    init_params: np.ndarray,
    samples: list[SampleRecord],
    cameras: dict[str, CameraModel],
    loss_scale_px: float,
    max_nfev: int,
) -> np.ndarray:
    lower: list[float] = []
    upper: list[float] = []
    for _ in range(4):
        lower.extend([-2.0, -2.0, -2.0, -4000.0, -4000.0, -1000.0])
        upper.extend([2.0, 2.0, 2.0, 4000.0, 5000.0, 4000.0])
    lower.extend([-4000.0, -4000.0, -1000.0, -math.pi, -math.pi, -math.pi, -math.pi])
    upper.extend([4000.0, 5000.0, 4000.0, math.pi, math.pi, math.pi, math.pi])

    result = least_squares(
        lambda params: reprojection_residual_vector(params, samples, cameras),
        init_params,
        bounds=(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)),
        loss="soft_l1",
        f_scale=loss_scale_px,
        max_nfev=max_nfev,
    )
    return result.x.astype(np.float64)


def closest_point_on_line_to_line(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray:
    d1 = normalize_direction(d1)
    d2 = normalize_direction(d2)
    r = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, r))
    e = float(np.dot(d2, r))
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return p1
    t1 = (b * e - c * d) / denom
    return p1 + t1 * d1


def point_on_line_at_world_z(
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    world_z: float,
) -> np.ndarray:
    axis_direction = normalize_direction(axis_direction)
    if abs(float(axis_direction[2])) < 1e-9:
        raise RuntimeError("joint_1 axis is too parallel to the ground plane to define a unique ground intersection")
    scale = (float(world_z) - float(axis_point[2])) / float(axis_direction[2])
    return axis_point + scale * axis_direction


def line_to_line_distance(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> float:
    d1 = normalize_direction(d1)
    d2 = normalize_direction(d2)
    normal = np.cross(d1, d2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return float(np.linalg.norm(np.cross(p2 - p1, d1)))
    return float(abs(np.dot(p2 - p1, normal / normal_norm)))


def build_model_dict(params: np.ndarray) -> dict:
    axes_world: list[dict] = []
    for joint_idx in range(4):
        offset = joint_idx * 6
        axis_direction = normalize_direction(params[offset : offset + 3])
        axis_point = params[offset + 3 : offset + 6]
        screw_v = -np.cross(axis_direction, axis_point)
        axes_world.append(
            {
                "joint_index": joint_idx + 1,
                "axis_direction_world": axis_direction.tolist(),
                "axis_point_world": axis_point.tolist(),
                "screw_axis_world": axis_direction.tolist() + screw_v.tolist(),
            }
        )

    return {
        "position_joint_count": 4,
        "joint_5_position_invariant_assumption": True,
        "joint_angle_offsets_rad": params[27:31].tolist(),
        "home_point_world": params[24:27].tolist(),
        "space_axes_world": axes_world,
    }


def derive_base_frame(
    params: np.ndarray,
    base_origin_mode: str,
    ground_plane_z: float,
) -> dict:
    axis1_dir = normalize_direction(params[0:3])
    axis1_point = params[3:6]
    axis2_dir = normalize_direction(params[6:9])
    axis2_point = params[9:12]
    axis3_dir = normalize_direction(params[12:15])

    if base_origin_mode == "joint1_ground_intersection":
        origin = point_on_line_at_world_z(axis1_point, axis1_dir, ground_plane_z)
        origin_definition = (
            f"intersection of joint_1 axis with world ground plane z={ground_plane_z:.3f}"
        )
    else:
        origin = closest_point_on_line_to_line(axis1_point, axis1_dir, axis2_point, axis2_dir)
        origin_definition = "closest point on joint_1 axis to joint_2 axis"

    z_axis = axis1_dir
    y_seed = axis2_dir - float(np.dot(axis2_dir, z_axis)) * z_axis
    if float(np.linalg.norm(y_seed)) < 1e-6:
        y_seed = axis3_dir - float(np.dot(axis3_dir, z_axis)) * z_axis
    y_axis = normalize_direction(y_seed)
    x_axis = normalize_direction(np.cross(y_axis, z_axis))
    y_axis = normalize_direction(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])

    axes_base: list[dict] = []
    for joint_idx in range(4):
        offset = joint_idx * 6
        axis_direction_world = normalize_direction(params[offset : offset + 3])
        axis_point_world = params[offset + 3 : offset + 6]
        axis_direction_base = rotation.T @ axis_direction_world
        axis_point_base = rotation.T @ (axis_point_world - origin)
        screw_v_base = -np.cross(axis_direction_base, axis_point_base)
        axes_base.append(
            {
                "joint_index": joint_idx + 1,
                "axis_direction_base": axis_direction_base.tolist(),
                "axis_point_base": axis_point_base.tolist(),
                "screw_axis_base": axis_direction_base.tolist() + screw_v_base.tolist(),
            }
        )

    home_point_base = rotation.T @ (params[24:27] - origin)
    link_distances = []
    for joint_idx in range(3):
        start = joint_idx * 6
        end = (joint_idx + 1) * 6
        distance = line_to_line_distance(
            params[start + 3 : start + 6],
            params[start : start + 3],
            params[end + 3 : end + 6],
            params[end : end + 3],
        )
        link_distances.append(
            {
                "pair": f"joint_{joint_idx + 1}_to_joint_{joint_idx + 2}",
                "shortest_axis_distance_mm": distance,
            }
        )

    return {
        "origin_definition": origin_definition,
        "base_origin_mode": base_origin_mode,
        "ground_plane_world_z": float(ground_plane_z),
        "R_base_in_world": rotation.tolist(),
        "t_base_in_world": origin.tolist(),
        "home_point_base": home_point_base.tolist(),
        "space_axes_base": axes_base,
        "consecutive_axis_distances_mm": link_distances,
    }


def summarize_reprojection(
    params: np.ndarray,
    samples: list[SampleRecord],
    cameras: dict[str, CameraModel],
    outlier_threshold_px: float,
) -> dict:
    per_obs_errors: list[dict] = []
    for sample in samples:
        point_world = forward_generic_point(params, sample.q4)
        for obs in sample.observations:
            uv_pred = project_point(cameras[obs.serial], point_world)
            residual = uv_pred - obs.uv
            per_obs_errors.append(
                {
                    "sample_id": sample.sample_id,
                    "camera_serial": obs.serial,
                    "residual_uv": residual.tolist(),
                    "residual_px": float(np.linalg.norm(residual)),
                    "q4": sample.q4.tolist(),
                    "q5": sample.q5,
                }
            )

    magnitudes = np.array([item["residual_px"] for item in per_obs_errors], dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(magnitudes)))) if len(magnitudes) else 0.0
    per_camera: dict[str, list[float]] = {}
    for item in per_obs_errors:
        per_camera.setdefault(item["camera_serial"], []).append(item["residual_px"])

    per_camera_summary = {}
    for serial, values in per_camera.items():
        arr = np.array(values, dtype=np.float64)
        per_camera_summary[serial] = {
            "observation_count": int(arr.size),
            "mean_px": float(arr.mean()),
            "median_px": float(np.median(arr)),
            "p95_px": float(np.percentile(arr, 95)),
            "max_px": float(arr.max()),
        }

    q5_values = np.array([item["q5"] for item in per_obs_errors], dtype=np.float64)
    q5_corr = float(np.corrcoef(magnitudes, q5_values)[0, 1]) if len(magnitudes) >= 2 else 0.0

    q5_bins: dict[int, list[float]] = {}
    for item in per_obs_errors:
        bucket = int(math.floor(item["q5"] / 0.1))
        q5_bins.setdefault(bucket, []).append(item["residual_px"])
    q5_bin_summary = []
    for bucket in sorted(q5_bins):
        arr = np.array(q5_bins[bucket], dtype=np.float64)
        q5_bin_summary.append(
            {
                "bin_index": bucket,
                "range_rad": [bucket * 0.1, (bucket + 1) * 0.1],
                "observation_count": int(arr.size),
                "mean_px": float(arr.mean()),
                "median_px": float(np.median(arr)),
            }
        )

    outliers = sorted(per_obs_errors, key=lambda item: item["residual_px"], reverse=True)
    outliers_over_threshold = [item for item in outliers if item["residual_px"] > outlier_threshold_px]

    return {
        "observation_count": len(per_obs_errors),
        "rms_px": rms,
        "mean_px": float(magnitudes.mean()),
        "median_px": float(np.median(magnitudes)),
        "p95_px": float(np.percentile(magnitudes, 95)),
        "max_px": float(magnitudes.max()),
        "count_over_10px": int(np.count_nonzero(magnitudes > 10.0)),
        "count_over_20px": int(np.count_nonzero(magnitudes > 20.0)),
        "count_over_50px": int(np.count_nonzero(magnitudes > 50.0)),
        "count_over_threshold_px": int(np.count_nonzero(magnitudes > outlier_threshold_px)),
        "per_camera": per_camera_summary,
        "joint_5": {
            "range_rad": [float(q5_values.min()), float(q5_values.max())],
            "range_width_rad": float(q5_values.max() - q5_values.min()),
            "discrete_0p1rad_bin_count": int(len(q5_bins)),
            "residual_abs_corr_with_q5": q5_corr,
            "residual_by_q5_bin": q5_bin_summary,
            "conclusion": (
                "Current position-only POE fit ignores joint_5. "
                "If residual does not track q5 and the fit is already low, "
                "joint_5 behaves like an orientation-only axis for the racket center."
            ),
        },
        "top_outliers": outliers[:20],
        "outliers_over_threshold": outliers_over_threshold[:50],
    }


def build_export_config(
    result: dict,
    export_path: Path,
    camera_calib_path: Path,
) -> dict:
    base = result["derived_base_in_world"]
    model = result["model"]
    reprojection = result["reprojection_fit"]

    exported_axes = []
    for axis in base["space_axes_base"]:
        omega = np.array(axis["axis_direction_base"], dtype=np.float64)
        q_point = np.array(axis["axis_point_base"], dtype=np.float64)
        v = -np.cross(omega, q_point)
        exported_axes.append(
            {
                "joint_index": axis["joint_index"],
                "omega": omega.tolist(),
                "q_mm": q_point.tolist(),
                "v_mm": v.tolist(),
                "screw_axis_base": omega.tolist() + v.tolist(),
            }
        )

    return {
        "config_role": "arm_poe_racket_center_position",
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "source_session": result["session_dir"],
        "source_result": rel_or_abs(export_path.parent / "poe_reprojection_result.json"),
        "camera_calibration": rel_or_abs(camera_calib_path),
        "measurement_target": {
            "name": "racket_center",
            "definition": "geometric center of racket keypoints 0-3",
            "joint_5_position_invariant_assumption": True,
            "joint_5_note": (
                "This config is position-only. joint_5 is treated as an orientation-only axis "
                "for the racket center and is not included in the translational POE chain."
            ),
        },
        "frames": {
            "world": {
                "definition": "ground-registered world frame from four_camera_calib.json",
                "ground_plane_world_z": base["ground_plane_world_z"],
            },
            "base": {
                "definition": (
                    "Base origin is the intersection of joint_1 axis with the world ground plane. "
                    "base z-axis is aligned with joint_1 axis. "
                    "base y-axis is the projection of joint_2 axis onto the plane orthogonal to base z. "
                    "base x-axis completes the right-handed frame."
                ),
                "origin_definition": base["origin_definition"],
                "physical_meaning": (
                    "This base origin is also the car chassis center used as p_car in step 15.6."
                ),
            },
        },
        "validation_manual": {
            "step_15_status": "completed",
            "checked_at": "2026-03-22",
            "summary": (
                "On-site manual cross-checks indicate the overall discrepancy between manual "
                "measurement, vision, and POE is at the few-centimeter level."
            ),
            "z_axis_note": "The z-axis accuracy is at the centimeter level.",
            "source_note": "These notes are the final manual validation notes recorded in DEV.md.",
        },
        "vehicle_reference": {
            "car_base_definition": (
                "car chassis center in world coordinates. In this project it is the same physical point "
                "as the robotic arm base used by the exported POE model."
            ),
            "apriltag_center_measurement": (
                "p_apriltag is the multi-camera triangulated world position of the detected AprilTag center."
            ),
            "p_car_definition": "p_car = p_apriltag + apriltag_center_to_car_base_offset_world",
            "apriltag_center_to_car_base_offset_m": APRILTAG_TO_CAR_BASE_OFFSET_M,
            "apriltag_center_to_car_base_offset_mm": APRILTAG_TO_CAR_BASE_OFFSET_MM,
            "offset_axis_convention": (
                "The offset is applied directly in the world frame axes, exactly as "
                "p_car = (x + 0.06, y + 0.10, z - 0.34) when p_apriltag = (x, y, z) in meters."
            ),
        },
        "T_base_in_world": {
            "R": base["R_base_in_world"],
            "t_mm": base["t_base_in_world"],
        },
        "poe_model_position_only": {
            "joint_count": model["position_joint_count"],
            "joint_angle_offsets_rad": model["joint_angle_offsets_rad"],
            "space_axes_base": exported_axes,
            "home_point_base_mm": base["home_point_base"],
        },
        "diagnostics": {
            "dataset": result["dataset"],
            "reprojection": {
                "observation_count": reprojection["observation_count"],
                "mean_px": reprojection["mean_px"],
                "median_px": reprojection["median_px"],
                "p95_px": reprojection["p95_px"],
                "max_px": reprojection["max_px"],
                "count_over_10px": reprojection["count_over_10px"],
                "count_over_20px": reprojection["count_over_20px"],
                "count_over_50px": reprojection["count_over_50px"],
                "per_camera": reprojection["per_camera"],
            },
            "joint_5_analysis": reprojection["joint_5"],
            "axis_distances_mm": base["consecutive_axis_distances_mm"],
            "top_outliers": reprojection["top_outliers"][:10],
        },
    }


def main() -> None:
    args = parse_args()
    session_dir = resolve_session_dir(args.session)
    camera_calib_path = Path(args.camera_calib).resolve()
    cameras = load_camera_models(camera_calib_path)
    samples, dataset_stats = collect_dataset(session_dir)
    if not samples:
        raise RuntimeError("No valid samples with accepted cameras and complete joint_state metadata.")
    triangulation_source_samples = [sample for sample in samples if len(sample.observations) >= 2]
    if not triangulation_source_samples:
        raise RuntimeError("No valid samples with 2+ accepted cameras for initialization.")

    print("=== ArmCalibration 15.4 POE Reprojection Calibration ===")
    print(f"Session: {session_dir}")
    print(f"Camera calib: {camera_calib_path}")
    print(f"Samples used for 2D fit: {len(samples)}")
    print(f"Accepted image observations: {dataset_stats['accepted_image_observations']}")
    print(f"Samples with 2+ cameras for init: {len(triangulation_source_samples)}")

    triangulated = [triangulate_sample(sample, cameras) for sample in triangulation_source_samples]
    triangulated_inliers = [
        item for item in triangulated if item.mean_reproj_px <= args.init_reproj_threshold
    ]
    if len(triangulated_inliers) < 20:
        raise RuntimeError(
            f"Too few triangulated initialization samples below {args.init_reproj_threshold:.1f}px: "
            f"{len(triangulated_inliers)}"
        )

    print(
        f"Init triangulation inliers: {len(triangulated_inliers)} / {len(triangulated)} "
        f"(threshold {args.init_reproj_threshold:.1f}px)"
    )

    structured_init = fit_structured_init(triangulated_inliers, args.max_nfev_structured)
    generic_init = structured_to_generic_world(structured_init)
    generic_3d_init = fit_generic_3d_init(generic_init, triangulated_inliers, args.max_nfev_generic_3d)
    final_params = fit_direct_reprojection(
        generic_3d_init,
        samples,
        cameras,
        args.loss_scale_px,
        args.max_nfev_2d,
    )

    reprojection_summary = summarize_reprojection(
        final_params,
        samples,
        cameras,
        args.outlier_threshold_px,
    )
    base_frame = derive_base_frame(final_params, args.base_origin_mode, args.ground_plane_z)
    model = build_model_dict(final_params)

    result = {
        "session_dir": rel_or_abs(session_dir),
        "camera_calibration": rel_or_abs(camera_calib_path),
        "dataset": dataset_stats,
        "initialization": {
            "triangulated_sample_count": len(triangulated),
            "triangulated_inlier_count": len(triangulated_inliers),
            "triangulated_inlier_threshold_px": args.init_reproj_threshold,
            "triangulated_inlier_mean_reproj_px": float(
                np.mean([item.mean_reproj_px for item in triangulated_inliers])
            ),
            "triangulated_inlier_p95_reproj_px": float(
                np.percentile([item.mean_reproj_px for item in triangulated_inliers], 95)
            ),
            "structured_init_params": structured_init.tolist(),
            "generic_3d_init_params": generic_3d_init.tolist(),
        },
        "reprojection_fit": reprojection_summary,
        "model": model,
        "derived_base_in_world": base_frame,
    }

    result_path = session_dir / args.result_name
    save_json(result_path, result)
    export_config_path = Path(args.export_config).resolve()
    export_config = build_export_config(result, result_path, camera_calib_path)
    save_json(export_config_path, export_config)

    print("\nCalibration finished.")
    print(f"Result JSON: {result_path}")
    print(f"Export config: {export_config_path}")
    print(
        "Reprojection summary: "
        f"mean={reprojection_summary['mean_px']:.3f}px, "
        f"median={reprojection_summary['median_px']:.3f}px, "
        f"p95={reprojection_summary['p95_px']:.3f}px, "
        f"max={reprojection_summary['max_px']:.3f}px"
    )


if __name__ == "__main__":
    main()
