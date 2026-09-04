"""V04 planar-arm sweet-spot map: 40 safe IK poses -> 18F four-camera 3D.

The default mode is offline-only and never imports ROS or opens a camera.
Hardware motion is available only with ``--execute`` and is additionally
blocked while the V04 source still labels its joint limits as unmeasured.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import itertools
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares


# 18F calibration images are not rotated. Set this before importing ball_grabber.
os.environ["BALL_TRACER_CAMERA_REVERSE_180"] = "0"
os.environ["BALL_TRACER_CAMERA_REVERSE_X"] = "0"
os.environ["BALL_TRACER_CAMERA_REVERSE_Y"] = "0"
os.environ["BALL_TRACER_SOFTWARE_ROTATE_180"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V04_SOURCE_ROOT = Path(r"D:\arm_controller-unify")
V04_KINEMATICS_PATH = (
    V04_SOURCE_ROOT / "src" / "arm_controller" / "compact_arm_kinematics.py"
)
V04_CONFIG_PATH = (
    V04_SOURCE_ROOT
    / "cpp"
    / "arm_controller_cpp"
    / "config"
    / "cars"
    / "v04.yaml"
)
CAMERA_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "camera_18.json"
CALIBRATION_PATH = PROJECT_ROOT / "src" / "config" / "four_camera_calib_18.json"
VEHICLE_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "vehicle_v04.json"
V04_PYTHON_ROOT = V04_SOURCE_ROOT / "src"
V04_PIXI_ENV = Path(r"C:\dev\ros2_jazzy\.pixi\envs\default")
_DLL_DIRECTORY_HANDLES: list[Any] = []

CAR = "v04"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
PLAN_X_VALUES_M = np.linspace(0.76, 0.98, 8)
PLAN_Z_VALUES_MODEL_M = np.linspace(1.28, 1.43, 5)
PLANNING_LIMIT_RATE = 0.65
PATH_LIMIT_RATE = 0.70
PHI_MAX_RAD = math.radians(15.0)
PHI_STEP_RAD = math.radians(1.0)
MIN_REACH_MARGIN_M = 0.09
MIN_RACKET_BOUND_GROUND_M = 0.95
RACKET_BOUND_RADIUS_M = 0.40
PATH_SAMPLES = 301

FIXED_JOINT_TOL_RAD = math.radians(1.0)
TARGET_TOL_RAD = math.radians(0.5)
SETTLE_SPAN_RAD = math.radians(0.08)
SETTLE_WINDOW_S = 0.8
SETTLE_MIN_OBSERVED_S = 0.6
JOINT_STATE_MAX_AGE_S = 0.2
SETTLE_MIN_DISTINCT_HEADERS = 8
MOVE_MAX_PEAK_SPEED_RAD_S = 0.18
MOVE_START_SPEED_ACCEPT_RAD_S = 0.01
MOVE_START_SPEED_VALIDATED_RAD_S = 0.02
MOVE_PATH_MAX_SPEED_RAD_S = 0.22
MOVE_MIN_DURATION_S = 4.0
MOVE_MAX_DURATION_S = 30.0
DIRECT_CONTROL_HZ = 100.0
DIRECT_CONTROL_DT_S = 1.0 / DIRECT_CONTROL_HZ
DIRECT_ECHO_TIMEOUT_S = 0.5
DIRECT_ECHO_ATOL = 1.0e-7

BURST_COUNT = 4
BURST_GAP_S = 0.20
MARKER_ROI_RADIUS_PX = 250
MARKER_CANDIDATES_PER_CAMERA = 8
MARKER_ASSOCIATION_PX = 10.0
MARKER_MAX_REPROJ_PX = 2.5
MARKER_MAX_LOO_MM = 10.0
MARKER_MAX_HELDOUT_PX = 6.0
MARKER_INITIAL_MAX_EXPECTED_DISTANCE_MM = 100.0
MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM = 40.0
BURST_MIN_GOOD = 3
BURST_MAX_SPREAD_MM = 10.0
CAR_MAX_REPROJ_PX = 4.0


class SafetyError(RuntimeError):
    """A failed invariant that must stop the hardware sequence."""


@dataclass(frozen=True)
class PlanPoint:
    index: int
    grid_x: int
    grid_z: int
    x_m: float
    z_model_m: float
    z_ground_m: float
    phi_deg: float
    q_command_rad: tuple[float, float, float, float, float, float]
    fk_tcp_model_m: tuple[float, float, float]
    reach_margin_m: float

    def json(self) -> dict[str, Any]:
        out = asdict(self)
        out["q_command_deg"] = [math.degrees(v) for v in self.q_command_rad]
        return out


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
    radial_errors_px: dict[str, float]
    rms_px: float
    max_px: float
    depths_mm: dict[str, float]


@dataclass(frozen=True)
class MarkerFit:
    point: PointFit
    pixels: dict[str, tuple[float, float]]
    loo_delta_mm: dict[str, float]
    loo_heldout_px: dict[str, float]
    expected_distance_mm: float


def _load_v04_kinematics() -> tuple[Any, dict[str, Any]]:
    if not V04_KINEMATICS_PATH.is_file() or not V04_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"V04 source is required at {V04_SOURCE_ROOT}; "
            "the V03-only c++version checkout must not be used"
        )
    spec = importlib.util.spec_from_file_location(
        "v04_compact_arm_kinematics", V04_KINEMATICS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V04_KINEMATICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.use_car(CAR)
    config = yaml.safe_load(V04_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("car") != CAR:
        raise SafetyError(f"expected car={CAR}, got {config.get('car')!r}")
    return module, config


def _reach_margin(kin: Any, x_m: float, z_m: float, phi_rad: float) -> float:
    tool_x, tool_z = (float(v) for v in kin.P4_TO_TCP_XZ_AT_ZERO)
    rel_x = (
        x_m
        - (math.cos(phi_rad) * tool_x - math.sin(phi_rad) * tool_z)
        - float(kin.P2_XZ[0])
    )
    rel_z = (
        z_m
        - (math.sin(phi_rad) * tool_x + math.cos(phi_rad) * tool_z)
        - float(kin.P2_XZ[1])
    )
    distance = math.hypot(rel_x, rel_z)
    return min(
        distance - abs(float(kin.L23) - float(kin.L34)),
        float(kin.L23) + float(kin.L34) - distance,
    )


def generate_plan() -> tuple[list[PlanPoint], dict[str, Any], Any, dict[str, Any]]:
    """Generate and validate the one supported 8x5 V04 calibration plan."""
    kin, config = _load_v04_kinematics()
    limits = config["joint_limits"]
    lower = np.radians(np.asarray(limits["lower_deg"], dtype=np.float64))
    upper = np.radians(np.asarray(limits["upper_deg"], dtype=np.float64))
    if lower.shape != (6,) or upper.shape != (6,) or np.any(lower >= upper):
        raise SafetyError("invalid V04 joint_limits")
    plan_lower = lower * PLANNING_LIMIT_RATE
    plan_upper = upper * PLANNING_LIMIT_RATE
    path_lower = lower * PATH_LIMIT_RATE
    path_upper = upper * PATH_LIMIT_RATE
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])

    grid: dict[tuple[int, int], PlanPoint] = {}
    for grid_z, z_m in enumerate(PLAN_Z_VALUES_MODEL_M):
        for grid_x, x_m in enumerate(PLAN_X_VALUES_M):
            solved = kin.min_abs_racket_angle(
                float(x_m),
                float(z_m),
                elbow="up",
                phi_max=PHI_MAX_RAD,
                step=PHI_STEP_RAD,
                lower=plan_lower[:5],
                upper=plan_upper[:5],
            )
            if solved is None:
                raise SafetyError(
                    f"plan point ({x_m:.4f}, {z_m:.4f}) has no V04 IK solution "
                    f"inside {PLANNING_LIMIT_RATE:.0%} limits"
                )
            phi_rad, q5 = solved
            q6 = np.concatenate([np.asarray(q5, dtype=np.float64), [0.0]])
            if np.max(np.abs(q6[[0, 4, 5]])) > 1e-12:
                raise SafetyError("plan violated J1/J5/J6=0")
            if np.any(q6 < plan_lower - 1e-12) or np.any(q6 > plan_upper + 1e-12):
                raise SafetyError("IK result exceeded the planning envelope")

            fk = kin.fk_hit(q6)
            tcp = np.asarray(fk["tcp"], dtype=np.float64)
            if np.linalg.norm(tcp[[0, 2]] - [x_m, z_m]) > 1e-9:
                raise SafetyError("V04 IK/FK round-trip failed")
            reach_margin = _reach_margin(kin, float(x_m), float(z_m), float(phi_rad))
            if reach_margin < MIN_REACH_MARGIN_M:
                raise SafetyError(
                    f"point ({x_m:.4f}, {z_m:.4f}) reach margin "
                    f"{reach_margin:.4f}m < {MIN_REACH_MARGIN_M:.4f}m"
                )
            grid[(grid_x, grid_z)] = PlanPoint(
                index=-1,
                grid_x=grid_x,
                grid_z=grid_z,
                x_m=float(x_m),
                z_model_m=float(z_m),
                z_ground_m=float(z_m - z_offset),
                phi_deg=math.degrees(float(phi_rad)),
                q_command_rad=tuple(float(v) for v in q6),
                fk_tcp_model_m=tuple(float(v) for v in tcp),
                reach_margin_m=float(reach_margin),
            )

    # Start at the point closest to zero, then snake through adjacent grid cells.
    ordered: list[PlanPoint] = []
    for row, grid_z in enumerate(range(len(PLAN_Z_VALUES_MODEL_M) - 1, -1, -1)):
        xs: Iterable[int] = range(len(PLAN_X_VALUES_M) - 1, -1, -1)
        if row % 2:
            xs = range(len(PLAN_X_VALUES_M))
        for grid_x in xs:
            item = grid[(grid_x, grid_z)]
            ordered.append(PlanPoint(**{**asdict(item), "index": len(ordered)}))

    if len(ordered) != 40:
        raise SafetyError(f"expected exactly 40 plan points, got {len(ordered)}")

    path_q = [np.zeros(6)] + [np.asarray(p.q_command_rad) for p in ordered] + [np.zeros(6)]
    min_tcp_model = math.inf
    min_link_model = math.inf
    max_step_deg = 0.0
    for start_q, target_q in zip(path_q, path_q[1:]):
        max_step_deg = max(
            max_step_deg, math.degrees(float(np.max(np.abs(target_q - start_q))))
        )
        for alpha in np.linspace(0.0, 1.0, PATH_SAMPLES):
            q = start_q + float(alpha) * (target_q - start_q)
            if np.any(q < path_lower - 1e-12) or np.any(q > path_upper + 1e-12):
                raise SafetyError("interpolated path exceeded the planning envelope")
            fk = kin.fk_hit(q)
            min_tcp_model = min(min_tcp_model, float(fk["tcp"][2]))
            min_link_model = min(
                min_link_model,
                float(fk["p2_xz"][1]),
                float(fk["p3_xz"][1]),
                float(fk["p4_xz"][1]),
            )

    min_racket_bound_ground = min_tcp_model - z_offset - RACKET_BOUND_RADIUS_M
    if min_racket_bound_ground < MIN_RACKET_BOUND_GROUND_M:
        raise SafetyError(
            f"path racket bound reaches {min_racket_bound_ground:.3f}m ground height"
        )
    q_deg = np.degrees(np.asarray([p.q_command_rad for p in ordered]))
    summary = {
        "count": len(ordered),
        "x_values_m": [float(v) for v in PLAN_X_VALUES_M],
        "z_model_values_m": [float(v) for v in PLAN_Z_VALUES_MODEL_M],
        "z_ground_values_m": [float(v - z_offset) for v in PLAN_Z_VALUES_MODEL_M],
        "planning_limit_rate": PLANNING_LIMIT_RATE,
        "path_limit_rate": PATH_LIMIT_RATE,
        "q_min_deg": q_deg.min(axis=0).tolist(),
        "q_max_deg": q_deg.max(axis=0).tolist(),
        "max_internal_step_deg": max(
            math.degrees(
                float(
                    np.max(
                        np.abs(
                            np.diff(
                                np.asarray([p.q_command_rad for p in ordered]), axis=0
                            )
                        )
                    )
                )
            ),
            0.0,
        ),
        "max_zero_transition_deg": max_step_deg,
        "min_reach_margin_m": min(p.reach_margin_m for p in ordered),
        "min_link_model_z_m": min_link_model,
        "min_tcp_model_z_m": min_tcp_model,
        "min_racket_bound_ground_m": min_racket_bound_ground,
    }
    return ordered, summary, kin, config


def assert_measured_v04_limits() -> None:
    """Do not run hardware while the V04 source declares inherited limits."""
    source = V04_CONFIG_PATH.read_text(encoding="utf-8")
    unmeasured_markers = ("待填实测值", "从 v0.3 继承", "没重新量过行程")
    found = [marker for marker in unmeasured_markers if marker in source]
    if found:
        raise SafetyError(
            "V04 execution is locked: v04.yaml explicitly says its joint limits are "
            "inherited from V03 and not measured on V04. Measure and update the real "
            "V04 limits first; analytic reachability is not a hardware safety proof."
        )


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(V04_SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_local_v04_source_clean() -> str:
    relative = [
        str(V04_KINEMATICS_PATH.relative_to(V04_SOURCE_ROOT)),
        str(V04_CONFIG_PATH.relative_to(V04_SOURCE_ROOT)),
        r"cpp\arm_controller_cpp\src\control\controller.hpp",
        r"cpp\arm_controller_cpp\src\control\controller.cpp",
        r"cpp\arm_controller_cpp\src\main.cpp",
        r"src\arm_controller\arm_controller.py",
        r"src\arm_controller\config.py",
        r"src\arm_controller\pinocchio_feedforward.py",
    ]
    changed = _git_output("status", "--porcelain", "--", *relative)
    if changed:
        raise SafetyError(f"V04 IK/config source has local changes:\n{changed}")
    main_source = (V04_SOURCE_ROOT / r"cpp\arm_controller_cpp\src\main.cpp").read_text(
        encoding="utf-8"
    )
    if "controller_->adopt_panel_command(adopted_q)" not in main_source:
        raise SafetyError("local V04 controller lacks panel-stream adopt support")
    return _git_output("rev-parse", "HEAD")


def _cubic_samples(
    start_q: np.ndarray,
    start_v: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < duration_s <= MOVE_MAX_DURATION_S:
        raise SafetyError(f"invalid direct move duration {duration_s}")
    start_q = np.asarray(start_q, dtype=np.float64)
    start_v = np.asarray(start_v, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    if any(array.shape != (6,) for array in (start_q, start_v, target_q)):
        raise SafetyError("direct cubic requires six joints")
    if not all(np.all(np.isfinite(array)) for array in (start_q, start_v, target_q)):
        raise SafetyError("direct cubic contains non-finite values")
    t = np.linspace(0.0, duration_s, PATH_SAMPLES, dtype=np.float64)[:, None]
    a2 = 3.0 * (target_q - start_q) / duration_s**2 - 2.0 * start_v / duration_s
    a3 = -2.0 * (target_q - start_q) / duration_s**3 + start_v / duration_s**2
    q = start_q + start_v * t + a2 * t**2 + a3 * t**3
    v = start_v + 2.0 * a2 * t + 3.0 * a3 * t**2
    return q, v


def validate_direct_cubic(
    start_q: np.ndarray,
    start_v: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
    kin: Any,
    config: dict[str, Any],
    *,
    include_start_velocity_envelope: bool,
) -> None:
    """Validate the exact cubic published directly by this PC."""
    limits = config["joint_limits"]
    lower = PATH_LIMIT_RATE * np.radians(np.asarray(limits["lower_deg"], dtype=float))
    upper = PATH_LIMIT_RATE * np.radians(np.asarray(limits["upper_deg"], dtype=float))
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    velocity_cases = [np.asarray(start_v, dtype=np.float64)]
    if include_start_velocity_envelope:
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            bounded = np.asarray(start_v, dtype=np.float64).copy()
            bounded[[1, 2, 3]] = MOVE_START_SPEED_VALIDATED_RAD_S * np.asarray(signs)
            velocity_cases.append(bounded)

    for velocity_case in velocity_cases:
        path_q, path_v = _cubic_samples(start_q, velocity_case, target_q, duration_s)
        if np.any(path_q < lower - 1e-12) or np.any(path_q > upper + 1e-12):
            raise SafetyError("direct cubic exceeds the 70% V04 joint envelope")
        if float(np.max(np.abs(path_q[:, [0, 4, 5]]))) > FIXED_JOINT_TOL_RAD:
            raise SafetyError("direct cubic moves J1/J5/J6 outside the zero tolerance")
        max_speed = float(np.max(np.abs(path_v)))
        if max_speed > MOVE_PATH_MAX_SPEED_RAD_S:
            raise SafetyError(
                f"direct cubic reaches {max_speed:.3f}rad/s, above the path gate"
            )
        for q in path_q:
            planar_q = q.copy()
            planar_q[[0, 4, 5]] = 0.0
            fk = kin.fk_hit(planar_q)
            racket_bound_ground = (
                float(fk["tcp"][2]) - z_offset - RACKET_BOUND_RADIUS_M
            )
            if racket_bound_ground < MIN_RACKET_BOUND_GROUND_M:
                raise SafetyError(
                    "direct cubic violates the conservative racket/ground bound"
                )


def validate_direct_start(
    ros_sample: dict[str, Any],
    target_q: np.ndarray,
    duration_s: float,
    kin: Any,
    config: dict[str, Any],
    *,
    require_stopped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ros_q = np.asarray(ros_sample["q"], dtype=np.float64)
    ros_v = np.asarray(ros_sample["v"], dtype=np.float64)
    if ros_q.shape != (6,) or ros_v.shape != (6,):
        raise SafetyError("direct move requires a complete six-joint ROS sample")
    if not np.all(np.isfinite(ros_q)) or not np.all(np.isfinite(ros_v)):
        raise SafetyError("direct move ROS sample is non-finite")
    age = time.perf_counter() - float(ros_sample["received_perf"])
    if age > JOINT_STATE_MAX_AGE_S:
        raise SafetyError(f"/joint_states is stale by {age:.3f}s")
    if require_stopped and float(np.max(np.abs(ros_v))) > MOVE_START_SPEED_ACCEPT_RAD_S:
        raise SafetyError("refusing to replan before the arm is stopped")
    validate_direct_cubic(
        ros_q,
        ros_v,
        target_q,
        duration_s,
        kin,
        config,
        include_start_velocity_envelope=require_stopped,
    )
    return ros_q, ros_v


def load_camera_models() -> tuple[list[str], dict[str, CameraModel]]:
    camera_cfg = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
    serials = [camera_cfg["master_serial"], *camera_cfg["slave_serials"]]
    if camera_cfg.get("trigger_mode") != "action" or len(serials) != 4:
        raise SafetyError("camera_18.json must define four action-trigger cameras")
    raw = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["cameras"]
    if set(raw) != set(serials):
        raise SafetyError("18F camera serials do not match four_camera_calib_18.json")
    models: dict[str, CameraModel] = {}
    for serial in serials:
        item = raw[serial]
        K = np.asarray(item["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(item["D"], dtype=np.float64).reshape(-1)
        R = np.asarray(item["R_world"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(item["t_world"], dtype=np.float64).reshape(3, 1)
        image_size = tuple(int(v) for v in item["image_size"])
        if D.shape != (5,) or image_size != (2048, 1536):
            raise SafetyError(f"unexpected calibration shape for {serial}")
        if not all(np.all(np.isfinite(v)) for v in (K, D, R, t)):
            raise SafetyError(f"non-finite calibration for {serial}")
        if abs(float(np.linalg.det(R)) - 1.0) > 1e-5:
            raise SafetyError(f"invalid R_world for {serial}")
        models[serial] = CameraModel(
            serial=serial,
            K=K,
            D=D,
            R=R,
            t=t,
            rvec=cv2.Rodrigues(R)[0],
            P=K @ np.hstack([R, t]),
            image_size=image_size,
        )
    return serials, models


def project_raw(camera: CameraModel, xyz_mm: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(xyz_mm, dtype=np.float64).reshape(1, 3),
        camera.rvec,
        camera.t,
        camera.K,
        camera.D,
    )
    return projected.reshape(2)


def _dlt(obs_raw: dict[str, tuple[float, float]], cameras: dict[str, CameraModel]) -> np.ndarray:
    rows = []
    for serial, uv in obs_raw.items():
        camera = cameras[serial]
        undistorted = cv2.undistortPoints(
            np.asarray([[uv]], dtype=np.float64), camera.K, camera.D, P=camera.K
        ).reshape(2)
        u, v = undistorted
        rows.extend([u * camera.P[2] - camera.P[0], v * camera.P[2] - camera.P[1]])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        raise ValueError("degenerate DLT")
    return homogeneous[:3] / homogeneous[3]


def triangulate_refined(
    obs_raw: dict[str, tuple[float, float]], cameras: dict[str, CameraModel]
) -> PointFit:
    if len(obs_raw) < 2:
        raise ValueError("triangulation requires at least two cameras")
    initial = _dlt(obs_raw, cameras)

    def residual(xyz: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [project_raw(cameras[serial], xyz) - np.asarray(uv) for serial, uv in obs_raw.items()]
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
        raise ValueError("raw-pixel triangulation refinement failed")
    errors: dict[str, float] = {}
    depths: dict[str, float] = {}
    for serial, uv in obs_raw.items():
        camera = cameras[serial]
        errors[serial] = float(np.linalg.norm(project_raw(camera, xyz) - np.asarray(uv)))
        depths[serial] = float((camera.R @ xyz.reshape(3, 1) + camera.t)[2, 0])
    if min(depths.values()) <= 0.0:
        raise ValueError("triangulated point is behind a camera")
    radial = np.asarray(list(errors.values()), dtype=np.float64)
    return PointFit(
        xyz_mm=xyz,
        radial_errors_px=errors,
        rms_px=float(np.sqrt(np.mean(radial * radial))),
        max_px=float(np.max(radial)),
        depths_mm=depths,
    )


def find_marker_candidates(
    image: np.ndarray, anchor_uv: tuple[float, float]
) -> list[MarkerCandidate]:
    """Find compact black components near the current-pose projected sweet spot."""
    height, width = image.shape[:2]
    anchor_x, anchor_y = anchor_uv
    radius = MARKER_ROI_RADIUS_PX
    x0 = max(0, int(math.floor(anchor_x - radius)))
    y0 = max(0, int(math.floor(anchor_y - radius)))
    x1 = min(width, int(math.ceil(anchor_x + radius)))
    y1 = min(height, int(math.ceil(anchor_y + radius)))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = float(np.median(gray))
    candidates: list[MarkerCandidate] = []
    ys, xs = np.indices(gray.shape)
    # The 18F views have very different backgrounds (median gray 24..48 in the
    # reference frame). A global adaptive threshold can become negative in the
    # bright/heterogeneous 0414 view, while a single high threshold connects most
    # of a dark view. Scan the black range and let four-view geometry select one.
    for threshold in (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0):
        mask = (gray < threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
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
            contrast = float(np.median(background - gray[component]))
            anchor_distance = math.hypot(u - anchor_x, v - anchor_y)
            score = (
                max(contrast, 1.0)
                * fill
                * min(area, 250)
                / (1.0 + (anchor_distance / 140.0) ** 2)
            )
            candidates.append(
                MarkerCandidate(
                    uv=(u, v),
                    score=float(score),
                    area=area,
                    bbox_xywh=(x0 + x, y0 + y, w, h),
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    deduped: list[MarkerCandidate] = []
    for candidate in candidates:
        if all(np.linalg.norm(np.subtract(candidate.uv, old.uv)) > 5.0 for old in deduped):
            deduped.append(candidate)
        if len(deduped) == MARKER_CANDIDATES_PER_CAMERA:
            break
    return deduped


def solve_marker_4cam(
    candidates: dict[str, list[MarkerCandidate]],
    cameras: dict[str, CameraModel],
    expected_xyz_mm: np.ndarray,
    max_expected_distance_mm: float = MARKER_INITIAL_MAX_EXPECTED_DISTANCE_MM,
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
                        > 1.5 * max_expected_distance_mm
                    ):
                        continue
                    choice: list[int] = []
                    total_distance = 0.0
                    valid = True
                    for serial in serials:
                        prediction = project_raw(cameras[serial], seed)
                        distances = [
                            float(np.linalg.norm(prediction - np.asarray(item.uv)))
                            for item in candidates[serial]
                        ]
                        best_index = int(np.argmin(distances))
                        if distances[best_index] > MARKER_ASSOCIATION_PX:
                            valid = False
                            break
                        choice.append(best_index)
                        total_distance += distances[best_index]
                    if valid:
                        key = tuple(choice)
                        proposed[key] = min(proposed.get(key, math.inf), total_distance)

    best: tuple[tuple[float, float, float, float], MarkerFit] | None = None
    for choice, _ in sorted(proposed.items(), key=lambda item: item[1])[:32]:
        pixels = {
            serial: candidates[serial][candidate_index].uv
            for serial, candidate_index in zip(serials, choice)
        }
        try:
            fit = triangulate_refined(pixels, cameras)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fit.max_px > MARKER_MAX_REPROJ_PX:
            continue
        expected_distance = float(np.linalg.norm(fit.xyz_mm - expected_xyz_mm))
        if expected_distance > max_expected_distance_mm:
            continue
        loo_delta: dict[str, float] = {}
        heldout: dict[str, float] = {}
        failed = False
        for dropped in serials:
            try:
                fit3 = triangulate_refined(
                    {serial: uv for serial, uv in pixels.items() if serial != dropped}, cameras
                )
            except (ValueError, np.linalg.LinAlgError):
                failed = True
                break
            loo_delta[dropped] = float(np.linalg.norm(fit3.xyz_mm - fit.xyz_mm))
            heldout[dropped] = float(
                np.linalg.norm(project_raw(cameras[dropped], fit3.xyz_mm) - pixels[dropped])
            )
        if failed or max(loo_delta.values()) >= MARKER_MAX_LOO_MM:
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
        # Semantic identity comes first: among geometrically valid black objects,
        # prefer the one closest to the current-pose racket prediction.
        rank = (expected_distance, max(loo_delta.values()), fit.rms_px, -score_sum)
        if best is None or rank < best[0]:
            best = (rank, marker)
    return None if best is None else best[1]


def _direct_cubic_state(
    start_q: np.ndarray,
    start_v: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_q = np.asarray(start_q, dtype=np.float64)
    start_v = np.asarray(start_v, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    if any(array.shape != (6,) for array in (start_q, start_v, target_q)):
        raise SafetyError("direct cubic requires six joints")
    if not all(np.all(np.isfinite(array)) for array in (start_q, start_v, target_q)):
        raise SafetyError("direct cubic contains non-finite values")
    if not 0.0 < duration_s <= MOVE_MAX_DURATION_S:
        raise SafetyError(f"invalid direct move duration {duration_s}")
    t = min(max(float(elapsed_s), 0.0), float(duration_s))
    a2 = 3.0 * (target_q - start_q) / duration_s**2 - 2.0 * start_v / duration_s
    a3 = -2.0 * (target_q - start_q) / duration_s**3 + start_v / duration_s**2
    q = start_q + start_v * t + a2 * t**2 + a3 * t**3
    v = start_v + 2.0 * a2 * t + 3.0 * a3 * t**2
    a = 2.0 * a2 + 6.0 * a3 * t
    return q, v, a


def load_v04_feedforward() -> Any:
    pixi_site = V04_PIXI_ENV / "Lib" / "site-packages"
    if pixi_site.is_dir() and str(pixi_site) not in sys.path:
        sys.path.append(str(pixi_site))
        if hasattr(os, "add_dll_directory"):
            for dll_dir in (V04_PIXI_ENV, V04_PIXI_ENV / "Library" / "bin"):
                if dll_dir.is_dir():
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    python_root = str(V04_PYTHON_ROOT.resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    try:
        arm_module = importlib.import_module("arm_controller.arm_controller")
        config_module = importlib.import_module("arm_controller.config")
        package_root = (V04_PYTHON_ROOT / "arm_controller").resolve()
        for module in (arm_module, config_module):
            if not Path(module.__file__).resolve().is_relative_to(package_root):
                raise SafetyError(f"loaded arm_controller from unexpected path: {module.__file__}")
        if float(config_module.CONTROL_HZ) != DIRECT_CONTROL_HZ:
            raise SafetyError(
                f"V04 feedforward control rate is {config_module.CONTROL_HZ}, "
                f"expected {DIRECT_CONTROL_HZ}Hz"
            )
        if not bool(config_module.USE_FEEDFORWARD):
            raise SafetyError("V04 feedforward is disabled")
        feedforward = arm_module.ArmController._load_feedforward(
            True, str(config_module.FEEDFORWARD_MODEL_PATH)
        )
        if feedforward is None:
            raise SafetyError("V04 feedforward loader returned None")
        zero = np.zeros(6, dtype=np.float64)
        probe = np.asarray(
            feedforward.predict(zero, zero, zero, measured_velocities=zero), dtype=np.float64
        )
        if probe.shape != (6,) or not np.all(np.isfinite(probe)):
            raise SafetyError("V04 feedforward probe is not a finite six-joint vector")
        return feedforward
    except SafetyError:
        raise
    except Exception as exc:
        raise SafetyError(f"failed to load the local V04 feedforward: {exc}") from exc


class DirectPanelPublisher:
    """Publish one validated six-axis cubic directly to the controller at 100 Hz."""

    def __init__(
        self,
        publish_frame: Any,
        latest_joint: Any,
        echo_matches: Any,
        feedforward: Any,
        tau_limit_nm: np.ndarray,
        clock: Any = time.perf_counter,
        sleep: Any = time.sleep,
    ) -> None:
        self._publish_frame = publish_frame
        self._latest_joint = latest_joint
        self._echo_matches = echo_matches
        self._feedforward = feedforward
        self._clock = clock
        self._sleep = sleep
        self._tau_limit = np.asarray(tau_limit_nm, dtype=np.float64)
        if self._tau_limit.shape != (6,) or not np.all(np.isfinite(self._tau_limit)):
            raise SafetyError("V04 torque limits must be a finite six-joint vector")
        if np.any(self._tau_limit <= 0.0):
            raise SafetyError("V04 torque limits must be positive")
        self._first_echo_confirmed = False

    def _effort(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray, measured_v: np.ndarray
    ) -> np.ndarray:
        try:
            effort = np.asarray(
                self._feedforward.predict(q, v, a, measured_velocities=measured_v),
                dtype=np.float64,
            )
        except Exception as exc:
            raise SafetyError(f"V04 feedforward prediction failed: {exc}") from exc
        if effort.shape != (6,) or not np.all(np.isfinite(effort)):
            raise SafetyError("V04 feedforward produced a non-finite six-joint effort")
        return np.clip(effort, -self._tau_limit, self._tau_limit)

    def move(
        self,
        start_q: np.ndarray,
        start_v: np.ndarray,
        target_q: np.ndarray,
        duration_s: float,
        guard: Any | None = None,
    ) -> None:
        started = self._clock()
        next_publish = started
        next_guard = started
        pending_echoes: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=64
        )
        while True:
            now = self._clock()
            if now < next_publish:
                self._sleep(next_publish - now)
            now = self._clock()
            if guard is not None and now >= next_guard:
                guard()
                next_guard = now + 0.25
            elapsed = min(now - started, duration_s)
            q, v, a = _direct_cubic_state(start_q, start_v, target_q, duration_s, elapsed)
            latest = self._latest_joint()
            if latest is None or now - float(latest["received_perf"]) > JOINT_STATE_MAX_AGE_S:
                raise SafetyError("/joint_states became stale during direct move")
            measured_v = np.asarray(latest["v"], dtype=np.float64)
            effort = self._effort(q, v, a, measured_v)
            sent_perf = self._clock()
            self._publish_frame(q, v, effort, sent_perf)
            if not self._first_echo_confirmed and elapsed >= DIRECT_CONTROL_DT_S:
                pending_echoes.append((sent_perf, q.copy(), v.copy(), effort.copy()))
                if any(self._echo_matches(*candidate) for candidate in pending_echoes):
                    self._first_echo_confirmed = True
                elif sent_perf - started >= DIRECT_ECHO_TIMEOUT_S:
                    raise SafetyError(
                        "controller did not echo a direct panel frame; "
                        "panel-stream adopt/priority build is not active"
                    )
            if elapsed >= duration_s:
                return
            next_publish += DIRECT_CONTROL_DT_S
            if next_publish <= now:
                next_publish = now + DIRECT_CONTROL_DT_S


def start_ros_monitor(feedforward: Any, tau_limit_nm: np.ndarray) -> Any:
    """Import ROS lazily so the default offline plan needs no ROS environment."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    reliable = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    transient = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__("v04_sweet_spot_capture")
            self._lock = threading.Lock()
            self._joint_history: deque[dict[str, Any]] = deque(maxlen=5000)
            self._status_history: deque[tuple[float, str]] = deque(maxlen=500)
            self._command_echo_history: deque[dict[str, Any]] = deque(maxlen=500)
            self._runtime_config: dict[str, Any] | None = None
            self.create_subscription(JointState, "/joint_states", self._on_joint, reliable)
            self.create_subscription(String, "/tennis/status", self._on_status, reliable)
            self.create_subscription(String, "/tennis/config", self._on_config, transient)
            self.create_subscription(
                JointTrajectory, "/tennis/motor_command", self._on_command_echo, reliable
            )
            self._panel_stream_pub = self.create_publisher(
                JointTrajectory, "/tennis/panel_stream", reliable
            )
            self.direct = DirectPanelPublisher(
                self._publish_direct_frame,
                self.latest_joint,
                self.echo_matches,
                feedforward,
                tau_limit_nm,
            )
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

        def _spin(self) -> None:
            while rclpy.ok() and not self._stop.is_set():
                self._executor.spin_once(timeout_sec=0.05)

        def _on_joint(self, msg: JointState) -> None:
            indices = {name: index for index, name in enumerate(msg.name)}
            if any(
                name not in indices
                or indices[name] >= len(msg.position)
                or indices[name] >= len(msg.velocity)
                for name in JOINT_NAMES
            ):
                return
            q = np.asarray([msg.position[indices[name]] for name in JOINT_NAMES], dtype=float)
            v = np.asarray([msg.velocity[indices[name]] for name in JOINT_NAMES], dtype=float)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                return
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            sample = {
                "received_perf": time.perf_counter(),
                "header_stamp_ns": stamp_ns,
                "q": q,
                "v": v,
            }
            with self._lock:
                self._joint_history.append(sample)

        def _on_status(self, msg: String) -> None:
            with self._lock:
                self._status_history.append((time.perf_counter(), str(msg.data)))

        def _on_command_echo(self, msg: JointTrajectory) -> None:
            if not msg.points or set(msg.joint_names) != set(JOINT_NAMES):
                return
            point = msg.points[0]
            if any(
                len(values) != len(JOINT_NAMES)
                for values in (point.positions, point.velocities, point.effort)
            ):
                return
            indices = {name: index for index, name in enumerate(msg.joint_names)}
            q = np.asarray([point.positions[indices[name]] for name in JOINT_NAMES], dtype=float)
            v = np.asarray([point.velocities[indices[name]] for name in JOINT_NAMES], dtype=float)
            effort = np.asarray([point.effort[indices[name]] for name in JOINT_NAMES], dtype=float)
            if not all(np.all(np.isfinite(values)) for values in (q, v, effort)):
                return
            with self._lock:
                self._command_echo_history.append(
                    {
                        "received_perf": time.perf_counter(),
                        "q": q,
                        "v": v,
                        "effort": effort,
                    }
                )

        def _publish_direct_frame(
            self,
            q: np.ndarray,
            v: np.ndarray,
            effort: np.ndarray,
            sent_perf: float,
        ) -> None:
            msg = JointTrajectory()
            msg.header.stamp.sec = int(sent_perf)
            msg.header.stamp.nanosec = int((sent_perf - int(sent_perf)) * 1.0e9)
            msg.joint_names = list(JOINT_NAMES)
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in q]
            point.velocities = [float(value) for value in v]
            point.effort = [float(value) for value in effort]
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(DIRECT_CONTROL_DT_S * 1.0e9)
            msg.points = [point]
            self._panel_stream_pub.publish(msg)

        def echo_matches(
            self,
            sent_perf: float,
            q: np.ndarray,
            v: np.ndarray,
            effort: np.ndarray,
        ) -> bool:
            with self._lock:
                rows = list(self._command_echo_history)
            return any(
                row["received_perf"] >= sent_perf
                and np.allclose(row["q"], q, rtol=0.0, atol=DIRECT_ECHO_ATOL)
                and np.allclose(row["v"], v, rtol=0.0, atol=DIRECT_ECHO_ATOL)
                and np.allclose(row["effort"], effort, rtol=0.0, atol=DIRECT_ECHO_ATOL)
                for row in rows
            )

        def _on_config(self, msg: String) -> None:
            try:
                config = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            with self._lock:
                self._runtime_config = config

        def latest_joint(self) -> dict[str, Any] | None:
            with self._lock:
                if not self._joint_history:
                    return None
                sample = self._joint_history[-1]
                return {**sample, "q": sample["q"].copy(), "v": sample["v"].copy()}

        def nearest_joint(self, exposure_perf: float, max_delta_s: float = 0.2) -> dict[str, Any]:
            with self._lock:
                if not self._joint_history:
                    raise SafetyError("no /joint_states at camera exposure")
                sample = min(
                    self._joint_history,
                    key=lambda item: abs(item["received_perf"] - exposure_perf),
                )
                copied = {
                    **sample,
                    "q": sample["q"].copy(),
                    "v": sample["v"].copy(),
                }
            delta = abs(float(copied["received_perf"]) - exposure_perf)
            if delta > max_delta_s:
                raise SafetyError(f"nearest joint state is {delta:.3f}s from camera exposure")
            return copied

        def runtime_config(self) -> dict[str, Any] | None:
            with self._lock:
                return None if self._runtime_config is None else dict(self._runtime_config)

        def wait_for_runtime(self, timeout_s: float = 8.0) -> tuple[dict[str, Any], dict[str, Any]]:
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                config = self.runtime_config()
                joint = self.latest_joint()
                if config is not None and joint is not None:
                    return config, joint
                time.sleep(0.05)
            raise SafetyError("timed out waiting for /tennis/config and /joint_states")

        def wait_status(self, sent_after: float, accepted_prefix: str, timeout_s: float) -> str:
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                with self._lock:
                    rows = list(self._status_history)
                for received, text in rows:
                    if received < sent_after:
                        continue
                    if text.startswith("reject") or text.startswith("error"):
                        raise SafetyError(f"controller: {text}")
                    if text.startswith(accepted_prefix):
                        return text
                time.sleep(0.05)
            raise SafetyError(f"no controller ack matching {accepted_prefix!r}")

        def wait_settled(
            self,
            target_q: np.ndarray,
            timeout_s: float,
            command_started_perf: float | None,
        ) -> np.ndarray:
            deadline = time.perf_counter() + timeout_s
            next_guard = -math.inf
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                if command_started_perf is not None and now >= next_guard:
                    _command_guard(self, command_started_perf)
                    next_guard = now + 0.25
                latest = self.latest_joint()
                if latest is None or now - float(latest["received_perf"]) > JOINT_STATE_MAX_AGE_S:
                    raise SafetyError("/joint_states became stale")
                q = np.asarray(latest["q"])
                v = np.asarray(latest["v"])
                if np.max(np.abs(q[[0, 4, 5]])) > FIXED_JOINT_TOL_RAD:
                    raise SafetyError(
                        "J1/J5/J6 left zero: "
                        + str(np.round(np.degrees(q[[0, 4, 5]]), 3).tolist())
                    )
                if command_started_perf is not None:
                    self.assert_no_external_command(command_started_perf)
                with self._lock:
                    window = [
                        item
                        for item in self._joint_history
                        if item["received_perf"] >= now - SETTLE_WINDOW_S
                    ]
                if len(window) < SETTLE_MIN_DISTINCT_HEADERS:
                    time.sleep(0.05)
                    continue
                observed_s = float(window[-1]["received_perf"] - window[0]["received_perf"])
                if observed_s < SETTLE_MIN_OBSERVED_S:
                    time.sleep(0.05)
                    continue
                header_values = [int(item["header_stamp_ns"]) for item in window]
                headers = set(header_values)
                if (
                    len(headers) < SETTLE_MIN_DISTINCT_HEADERS
                    or header_values[-1] <= header_values[0]
                    or any(right < left for left, right in zip(header_values, header_values[1:]))
                ):
                    time.sleep(0.05)
                    continue
                positions = np.asarray([item["q"] for item in window])
                velocities = np.asarray([item["v"] for item in window])
                span = np.ptp(positions, axis=0)
                target_error = np.max(np.abs(q - target_q))
                median_speed = float(np.max(np.median(np.abs(velocities), axis=0)))
                if (
                    target_error <= TARGET_TOL_RAD
                    and float(np.max(span)) <= SETTLE_SPAN_RAD
                    and float(np.max(np.abs(v))) <= MOVE_START_SPEED_ACCEPT_RAD_S
                    and median_speed <= MOVE_START_SPEED_ACCEPT_RAD_S
                ):
                    return q.copy()
                time.sleep(0.05)
            raise SafetyError(
                "arm did not settle at target; latest error_deg="
                f"{np.max(np.abs(np.degrees(q - target_q))):.3f}"
            )

        def publishers(self, topic: str) -> list[str]:
            return [info.node_name for info in self.get_publishers_info_by_topic(topic)]

        def assert_no_external_command(self, since_perf: float) -> None:
            with self._lock:
                rows = list(self._status_history)
            unexpected = [
                text
                for received, text in rows
                if received >= since_perf
                and (text.startswith("accepted hit") or text.startswith("accepted arm_command"))
            ]
            if unexpected:
                raise SafetyError(f"competing controller command during calibration: {unexpected[-1]}")

        def close(self) -> None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._executor.remove_node(self)
            self._executor.shutdown(timeout_sec=1.0)
            self.destroy_node()

    rclpy.init()
    return Monitor()


def assert_runtime_graph(monitor: Any) -> None:
    deadline = time.perf_counter() + 8.0
    expected = {
        "/joint_states": ["arm_controller_cpp"],
        "/tennis/panel_stream": [monitor.get_name()],
        "/tennis/arm_command": [],
    }
    while time.perf_counter() < deadline:
        actual = {topic: monitor.publishers(topic) for topic in expected}
        if all(actual[topic] == nodes for topic, nodes in expected.items()):
            break
        time.sleep(0.1)
    else:
        raise SafetyError(f"unexpected ROS command graph: {actual}")
    hit_publishers = monitor.publishers("/predict_hit_pos")
    if hit_publishers:
        raise SafetyError(f"competing /predict_hit_pos publishers: {hit_publishers}")


def assert_runtime_identity(runtime: dict[str, Any], local_commit: str) -> None:
    if runtime.get("node") != "arm_controller_cpp":
        raise SafetyError(f"unexpected controller node config: {runtime}")
    if runtime.get("car") != CAR or runtime.get("mode") != "active":
        raise SafetyError(
            f"controller must announce car={CAR}, mode=active; got "
            f"car={runtime.get('car')}, mode={runtime.get('mode')}"
        )
    runtime_git = str(runtime.get("git", ""))
    if "dirty" in runtime_git or local_commit[:7] not in runtime_git:
        raise SafetyError(
            f"controller git {runtime_git!r} does not match clean local {local_commit[:12]}"
        )


def _move_duration(current_q: np.ndarray, target_q: np.ndarray) -> float:
    max_delta = float(np.max(np.abs(target_q - current_q)))
    duration = max(MOVE_MIN_DURATION_S, 1.5 * max_delta / MOVE_MAX_PEAK_SPEED_RAD_S)
    if duration > MOVE_MAX_DURATION_S:
        raise SafetyError(f"required safe move duration {duration:.2f}s exceeds panel limit")
    return duration


def _car_pose(images: dict[str, np.ndarray], localizer: Any) -> Any:
    grouped: dict[int, dict[str, Any]] = {}
    for serial, image in images.items():
        for detection in localizer.detect(image):
            if detection.tag_id in localizer.tag_ids:
                grouped.setdefault(detection.tag_id, {})[serial] = detection
    if not any(len(per_camera) >= 2 for per_camera in grouped.values()):
        raise SafetyError("car tags are not triangulatable in the synchronized frame")
    result = localizer.estimate_car_pose(grouped, t=time.perf_counter())
    if (
        result is None
        or result.yaw is None
        or not result.yaw_valid
        or result.reprojection_error > CAR_MAX_REPROJ_PX
    ):
        raise SafetyError("V04 car pose is unavailable or fails its quality gate")
    return result


def _expected_sweet_world_mm(
    q: np.ndarray,
    car: Any,
    kin: Any,
    z_offset: float,
    marker_offset_tool_m: np.ndarray | None = None,
) -> np.ndarray:
    fk = kin.fk_hit(q)
    tcp = np.asarray(fk["tcp"], dtype=np.float64)
    if marker_offset_tool_m is not None:
        offset = np.asarray(marker_offset_tool_m, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise SafetyError("invalid locked marker offset")
        phi = float(fk["racket_angle"])
        cp = math.cos(phi)
        sp = math.sin(phi)
        tcp = tcp + np.asarray(
            [cp * offset[0] - sp * offset[2], offset[1], sp * offset[0] + cp * offset[2]]
        )
    c = math.cos(float(car.yaw))
    s = math.sin(float(car.yaw))
    world_x = float(car.x) + c * tcp[0] - s * tcp[1]
    world_y = float(car.y) + s * tcp[0] + c * tcp[1]
    world_z = tcp[2] - z_offset
    return 1000.0 * np.asarray([world_x, world_y, world_z])


def marker_offset_tool_m(
    q: np.ndarray,
    sweet_car_m: np.ndarray,
    kin: Any,
    z_offset: float,
) -> np.ndarray:
    fk = kin.fk_hit(q)
    expected_car = np.asarray(fk["tcp"], dtype=np.float64)
    expected_car[2] -= z_offset
    delta = np.asarray(sweet_car_m, dtype=np.float64) - expected_car
    phi = float(fk["racket_angle"])
    cp = math.cos(phi)
    sp = math.sin(phi)
    offset = np.asarray(
        [cp * delta[0] + sp * delta[2], delta[1], -sp * delta[0] + cp * delta[2]]
    )
    if float(np.linalg.norm(offset)) * 1000.0 > MARKER_INITIAL_MAX_EXPECTED_DISTANCE_MM:
        raise SafetyError("confirmed black dot is too far from the V04 FK sweet point")
    return offset


def _sweet_in_car_m(world_mm: np.ndarray, car: Any) -> list[float]:
    dx = world_mm[0] / 1000.0 - float(car.x)
    dy = world_mm[1] / 1000.0 - float(car.y)
    c = math.cos(float(car.yaw))
    s = math.sin(float(car.yaw))
    return [c * dx + s * dy, -s * dx + c * dy, world_mm[2] / 1000.0]


def capture_synced_burst(cap: Any, serials: list[str]) -> list[dict[str, Any]]:
    from src import frame_to_numpy

    groups: list[dict[str, Any]] = []
    for burst_index in range(BURST_COUNT):
        frames = cap.get_frames(timeout_s=2.0)
        if frames is None or set(frames) != set(serials):
            raise SafetyError("18F synchronized four-camera capture timed out")
        if any(frame.lost_packet != 0 for frame in frames.values()):
            raise SafetyError("camera frame reported lost packets")
        exposures = [float(frame.exposure_start_pc) for frame in frames.values()]
        spread_ms = 1000.0 * (max(exposures) - min(exposures))
        if spread_ms > 10.0:
            raise SafetyError(f"four-camera exposure spread is {spread_ms:.3f}ms")
        images = {
            serial: frame_to_numpy(frame, rotate_180=False).copy()
            for serial, frame in frames.items()
        }
        if any(image.shape[:2] != (1536, 2048) for image in images.values()):
            raise SafetyError("18F capture is not full-resolution 2048x1536")
        groups.append(
            {
                "burst_index": burst_index,
                "images": images,
                "frame_num": {serial: int(frame.frame_num) for serial, frame in frames.items()},
                "exposure_perf": float(np.mean(exposures)),
                "sync_spread_ms": spread_ms,
            }
        )
        if burst_index + 1 < BURST_COUNT:
            time.sleep(BURST_GAP_S)
    return groups


def measure_burst(
    groups: list[dict[str, Any]],
    point_dir: Path,
    serials: list[str],
    cameras: dict[str, CameraModel],
    monitor: Any,
    car_localizer: Any,
    kin: Any,
    z_offset: float,
    target_q: np.ndarray,
    marker_offset_tool: np.ndarray | None,
    write_review: bool,
) -> dict[str, Any]:
    point_dir.mkdir(parents=True, exist_ok=False)
    accepted: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    for group in groups:
        frame_index = int(group["burst_index"])
        files: dict[str, str] = {}
        for serial, image in group["images"].items():
            path = point_dir / f"burst_{frame_index}_{serial}.png"
            if not cv2.imwrite(str(path), image):
                raise SafetyError(f"failed to save {path}")
            files[serial] = str(path.relative_to(point_dir.parent.parent))
        sample = monitor.nearest_joint(float(group["exposure_perf"]))
        q = np.asarray(sample["q"], dtype=np.float64)
        v = np.asarray(sample["v"], dtype=np.float64)
        if np.max(np.abs(q[[0, 4, 5]])) > FIXED_JOINT_TOL_RAD:
            raise SafetyError("fixed joint left zero during exposure")
        if np.max(np.abs(q - target_q)) > TARGET_TOL_RAD:
            raise SafetyError("arm left the requested target during exposure")
        if np.max(np.abs(v)) > MOVE_START_SPEED_ACCEPT_RAD_S:
            raise SafetyError("arm was moving during exposure")
        frame_record: dict[str, Any] = {
            "burst_index": frame_index,
            "frame_num": group["frame_num"],
            "exposure_perf": group["exposure_perf"],
            "sync_spread_ms": group["sync_spread_ms"],
            "q_measured_rad": q.tolist(),
            "v_measured_rad_s": v.tolist(),
            "files": files,
            "accepted": False,
        }
        try:
            car = _car_pose(group["images"], car_localizer)
            expected = _expected_sweet_world_mm(
                q, car, kin, z_offset, marker_offset_tool
            )
            anchors = {
                serial: tuple(float(v) for v in project_raw(cameras[serial], expected))
                for serial in serials
            }
            candidates = {
                serial: find_marker_candidates(group["images"][serial], anchors[serial])
                for serial in serials
            }
            max_expected = (
                MARKER_INITIAL_MAX_EXPECTED_DISTANCE_MM
                if marker_offset_tool is None
                else MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM
            )
            fit = solve_marker_4cam(candidates, cameras, expected, max_expected)
            if fit is None:
                raise SafetyError("black dot failed four-camera geometry gates")
            frame_record.update(
                {
                    "accepted": True,
                    "candidate_counts": {serial: len(candidates[serial]) for serial in serials},
                    "anchor_uv": anchors,
                    "pixels": fit.pixels,
                    "sweet_world_mm": fit.point.xyz_mm.tolist(),
                    "reproj_px": fit.point.radial_errors_px,
                    "reproj_rms_px": fit.point.rms_px,
                    "reproj_max_px": fit.point.max_px,
                    "loo_delta_mm": fit.loo_delta_mm,
                    "loo_heldout_px": fit.loo_heldout_px,
                    "expected_distance_mm": fit.expected_distance_mm,
                    "car": {
                        "x_m": float(car.x),
                        "y_m": float(car.y),
                        "yaw_rad": float(car.yaw),
                        "yaw_deg": math.degrees(float(car.yaw)),
                        "reproj_px": float(car.reprojection_error),
                        "tag_ids": list(car.tag_ids),
                        "cameras_used": list(car.cameras_used),
                    },
                }
            )
            accepted.append(
                {
                    "fit": fit,
                    "q": q,
                    "car": car,
                    "group": group,
                    "anchors": anchors,
                }
            )
        except SafetyError as exc:
            frame_record["failure"] = str(exc)
        frame_records.append(frame_record)

    if len(accepted) < BURST_MIN_GOOD:
        raise SafetyError(f"only {len(accepted)}/{BURST_COUNT} burst frames passed")
    points = np.asarray([item["fit"].point.xyz_mm for item in accepted])
    spread = max(
        float(np.linalg.norm(points[i] - points[j]))
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )
    if spread >= BURST_MAX_SPREAD_MM:
        raise SafetyError(f"black-dot burst 3D spread {spread:.3f}mm exceeds gate")
    pairwise = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    representative_index = int(np.argmin(np.sum(pairwise, axis=1)))
    representative = accepted[representative_index]
    representative_group = representative["group"]
    representative_fit = representative["fit"]
    representative_q = representative["q"]
    representative_car = representative["car"]

    review_image: str | None = None
    if write_review:
        tiles: list[np.ndarray] = []
        for serial in serials:
            tile = representative_group["images"][serial].copy()
            anchor = tuple(
                int(round(value)) for value in representative["anchors"][serial]
            )
            marker = tuple(
                int(round(value)) for value in representative_fit.pixels[serial]
            )
            cv2.drawMarker(tile, anchor, (0, 255, 255), cv2.MARKER_CROSS, 36, 3)
            cv2.circle(tile, marker, 28, (0, 0, 255), 4)
            cv2.putText(
                tile,
                serial,
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            tiles.append(cv2.resize(tile, (1024, 768)))
        montage = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        review_path = point_dir / "first_dot_review.jpg"
        if not cv2.imwrite(str(review_path), montage):
            raise SafetyError(f"failed to save {review_path}")
        review_image = str(review_path.resolve())

    sweet_world_mm = representative_fit.point.xyz_mm
    return {
        "capture_perf": float(representative_group["exposure_perf"]),
        "representative_burst_index": int(representative_group["burst_index"]),
        "q_measured_rad": representative_q.tolist(),
        "q_measured_deg": np.degrees(representative_q).tolist(),
        "sweet_world_m": (sweet_world_mm / 1000.0).tolist(),
        "sweet_car_m": _sweet_in_car_m(sweet_world_mm, representative_car),
        "burst_good": len(accepted),
        "burst_spread_mm": spread,
        "worst_reproj_px": max(item["fit"].point.max_px for item in accepted),
        "worst_loo_mm": max(
            max(item["fit"].loo_delta_mm.values()) for item in accepted
        ),
        "car": {
            "x_m": float(representative_car.x),
            "y_m": float(representative_car.y),
            "yaw_deg": math.degrees(float(representative_car.yaw)),
            "reproj_px": float(representative_car.reprojection_error),
        },
        "frames": frame_records,
        "review_image": review_image,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        *[f"j{i}_deg" for i in range(1, 7)],
        "sweet_world_x_m",
        "sweet_world_y_m",
        "sweet_world_z_m",
        "sweet_car_x_m",
        "sweet_car_y_m",
        "sweet_car_z_m",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {"index": result["index"]}
            for joint_index, value in enumerate(result["q_measured_deg"], start=1):
                row[f"j{joint_index}_deg"] = f"{float(value):.6f}"
            for prefix, values in (
                ("sweet_world", result["sweet_world_m"]),
                ("sweet_car", result["sweet_car_m"]),
            ):
                for axis, value in zip("xyz", values):
                    row[f"{prefix}_{axis}_m"] = f"{float(value):.6f}"
            writer.writerow(row)


def _write_session(
    output: Path,
    status: str,
    runtime: dict[str, Any] | None,
    summary: dict[str, Any],
    result_count: int,
    *,
    error: BaseException | None = None,
    return_to_zero_error: BaseException | None = None,
    marker_offset_tool: np.ndarray | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "car": CAR,
        "runtime_config": runtime,
        "plan_summary": summary,
        "result_count": result_count,
        "marker_offset_tool_m": (
            None if marker_offset_tool is None else marker_offset_tool.tolist()
        ),
    }
    if error is not None:
        payload["error"] = f"{type(error).__name__}: {error}"
    if return_to_zero_error is not None:
        payload["return_to_zero_error"] = (
            f"{type(return_to_zero_error).__name__}: {return_to_zero_error}"
        )
    (output / "session.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _confirm_first_dot(review_image: str) -> None:
    review_path = Path(review_image)
    if not review_path.is_file() or cv2.imread(str(review_path)) is None:
        raise SafetyError("first black-dot review image is missing or unreadable")
    print(
        "FIRST-POINT SAFETY REVIEW: open the montage below. The red circle in all "
        "four views must be the racket black sweet-spot dot; yellow is the FK anchor.\n"
        f"{review_image}",
        flush=True,
    )
    try:
        answer = input("Type CONFIRM to lock this physical dot for the remaining 39 points: ")
    except EOFError as exc:
        raise SafetyError("first black-dot identity was not interactively confirmed") from exc
    if answer.strip() != "CONFIRM":
        raise SafetyError("operator rejected or did not confirm the first black dot")


def _command_guard(monitor: Any, command_started_perf: float) -> None:
    assert_runtime_graph(monitor)
    monitor.assert_no_external_command(command_started_perf)


def _return_to_zero(
    monitor: Any,
    command_started_perf: float,
    kin: Any,
    config: dict[str, Any],
) -> None:
    _command_guard(monitor, command_started_perf)
    latest = monitor.latest_joint()
    if latest is None:
        raise SafetyError("cannot return to zero without /joint_states")
    target_q = np.zeros(6, dtype=np.float64)
    duration = _move_duration(np.asarray(latest["q"]), target_q)
    start_q, start_v = validate_direct_start(
        latest,
        target_q,
        duration,
        kin,
        config,
        require_stopped=False,
    )
    monitor.direct.move(
        start_q,
        start_v,
        target_q,
        duration,
        guard=lambda: _command_guard(monitor, command_started_perf),
    )
    monitor.wait_settled(
        target_q,
        timeout_s=duration + 10.0,
        command_started_perf=command_started_perf,
    )


def execute_plan(
    plan: list[PlanPoint], summary: dict[str, Any], kin: Any, config: dict[str, Any]
) -> Path:
    assert_measured_v04_limits()
    local_commit = assert_local_v04_source_clean()
    serials, cameras = load_camera_models()
    feedforward = load_v04_feedforward()
    tau_limit_nm = np.asarray(config["tuning"]["tau_limit_nm"], dtype=np.float64)
    output = (
        PROJECT_ROOT
        / "arm_controller_data"
        / f"v04_sweet_spot_{time.perf_counter_ns()}"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "plan.json").write_text(
        json.dumps(
            {"summary": summary, "points": [point.json() for point in plan]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    monitor: Any | None = None
    runtime: dict[str, Any] | None = None
    command_started_perf: float | None = None
    motion_may_have_started = False
    primary_error: BaseException | None = None
    return_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    marker_offset_tool: np.ndarray | None = None
    try:
        monitor = start_ros_monitor(feedforward, tau_limit_nm)
        runtime, _ = monitor.wait_for_runtime()
        assert_runtime_graph(monitor)
        assert_runtime_identity(runtime, local_commit)
        command_started_perf = time.perf_counter()

        # Never issue an unvalidated current->zero preset. The operator must start
        # with the real arm already settled at zero and all fixed axes at zero.
        monitor.wait_settled(
            np.zeros(6),
            timeout_s=5.0,
            command_started_perf=command_started_perf,
        )

        from src import SyncCapture
        from src.car_localizer import CarLocalizer

        car_localizer = CarLocalizer(
            calib_config_path=str(CALIBRATION_PATH),
            vehicle_config_path=str(VEHICLE_CONFIG_PATH),
        )
        z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
        with SyncCapture.from_config(str(CAMERA_CONFIG_PATH)) as cap:
            if cap.sync_serials != serials:
                raise SafetyError(f"unexpected synchronized cameras: {cap.sync_serials}")
            time.sleep(2.0)
            for point in plan:
                assert_runtime_graph(monitor)
                monitor.assert_no_external_command(command_started_perf)
                latest = monitor.latest_joint()
                if latest is None:
                    raise SafetyError("lost /joint_states before move")
                target_q = np.asarray(point.q_command_rad, dtype=np.float64)
                duration = _move_duration(np.asarray(latest["q"]), target_q)
                start_q, start_v = validate_direct_start(
                    latest,
                    target_q,
                    duration,
                    kin,
                    config,
                    require_stopped=True,
                )
                motion_may_have_started = True
                monitor.direct.move(
                    start_q,
                    start_v,
                    target_q,
                    duration,
                    guard=lambda: _command_guard(monitor, command_started_perf),
                )
                monitor.wait_settled(
                    target_q,
                    timeout_s=duration + 10.0,
                    command_started_perf=command_started_perf,
                )
                assert_runtime_graph(monitor)
                monitor.assert_no_external_command(command_started_perf)
                groups = capture_synced_burst(cap, serials)
                measured = measure_burst(
                    groups,
                    output / "frames" / f"point_{point.index:02d}",
                    serials,
                    cameras,
                    monitor,
                    car_localizer,
                    kin,
                    z_offset,
                    target_q,
                    marker_offset_tool,
                    write_review=point.index == 0,
                )
                if marker_offset_tool is None:
                    review_image = measured.get("review_image")
                    if not isinstance(review_image, str):
                        raise SafetyError("first black-dot review image was not created")
                    _confirm_first_dot(review_image)
                    marker_offset_tool = marker_offset_tool_m(
                        np.asarray(measured["q_measured_rad"], dtype=np.float64),
                        np.asarray(measured["sweet_car_m"], dtype=np.float64),
                        kin,
                        z_offset,
                    )
                    monitor.wait_settled(
                        target_q,
                        timeout_s=5.0,
                        command_started_perf=command_started_perf,
                    )
                record = {
                    "index": point.index,
                    "planned": point.json(),
                    **measured,
                }
                _append_jsonl(output / "results.jsonl", record)
                results.append(record)
                print(
                    f"[{point.index + 1:02d}/40] q_deg="
                    f"{np.round(record['q_measured_deg'], 3).tolist()} sweet_world_m="
                    f"{np.round(record['sweet_world_m'], 5).tolist()}",
                    flush=True,
                )

        if len(results) != 40:
            raise SafetyError(f"session ended with {len(results)}/40 accepted points")
        _write_csv(output / "joints_to_sweet_spot.csv", results)
    except BaseException as exc:
        primary_error = exc
    finally:
        if monitor is not None and motion_may_have_started:
            try:
                if command_started_perf is None:
                    raise SafetyError("direct command session start time is missing")
                _return_to_zero(
                    monitor,
                    command_started_perf,
                    kin,
                    config,
                )
            except BaseException as exc:
                return_error = exc
                print(
                    "EMERGENCY: automatic return to zero failed. Keep the area clear "
                    f"and use the hardware stop/manual recovery: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if monitor is not None:
            try:
                monitor.close()
            except BaseException as exc:
                cleanup_error = exc
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc

    if return_error is not None:
        _write_session(
            output,
            "unsafe",
            runtime,
            summary,
            len(results),
            error=primary_error,
            return_to_zero_error=return_error,
            marker_offset_tool=marker_offset_tool,
        )
        raise SafetyError(
            "session is UNSAFE because automatic return to zero failed; "
            "hardware stop/manual recovery is required"
        ) from return_error
    if primary_error is not None:
        _write_session(
            output,
            "failed",
            runtime,
            summary,
            len(results),
            error=primary_error,
            marker_offset_tool=marker_offset_tool,
        )
        raise primary_error
    if cleanup_error is not None:
        _write_session(
            output,
            "failed",
            runtime,
            summary,
            len(results),
            error=cleanup_error,
            marker_offset_tool=marker_offset_tool,
        )
        raise SafetyError("arm returned to zero, but ROS monitor cleanup failed") from cleanup_error

    _write_session(
        output,
        "complete",
        runtime,
        summary,
        len(results),
        marker_offset_tool=marker_offset_tool,
    )
    return output


def print_plan(plan: list[PlanPoint], summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("idx,x_m,z_model_m,z_ground_m,phi_deg,j1,j2,j3,j4,j5,j6")
    for point in plan:
        q_deg = [math.degrees(v) for v in point.q_command_rad]
        values = [
            str(point.index),
            f"{point.x_m:.6f}",
            f"{point.z_model_m:.6f}",
            f"{point.z_ground_m:.6f}",
            f"{point.phi_deg:.3f}",
            *[f"{value:.6f}" for value in q_deg],
        ]
        print(",".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V04 40-point J2-J4 to 18F four-camera sweet-spot map"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the already-validated plan on the live V04 arm and cameras",
    )
    args = parser.parse_args()
    try:
        plan, summary, kin, config = generate_plan()
        print_plan(plan, summary)
        if not args.execute:
            print("OFFLINE ONLY: no ROS command was sent and no camera was opened.")
            return 0
        output = execute_plan(plan, summary, kin, config)
        print(f"Complete: {output}")
        return 0
    except (SafetyError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"SAFETY STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
