#!/usr/bin/env python3
"""Replay one tracker session's FinalHT targets as static poses and snapshot 18F.

The arm is driven with `inspect <x> <z_model>` (smooth 1.0s move, no swing), held
still, and one synchronized four-camera frame set is stored per pose together with
the joint state at exposure.  `analyze_v04_ht_replay.py` then compares full FK
against the camera-side racket localisation offline.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG = PROJECT_ROOT / "src" / "config" / "camera_18.json"
TRACKER_OUTPUT = PROJECT_ROOT / "tracker_output"
OUTPUT_ROOT = PROJECT_ROOT / "arm_controller_data"

V04_SOURCE_ROOT = Path("D:/tennis-man/arm_controller")  # RK 同步的标准 checkout（唯一真值）
V04_KINEMATICS_PATH = V04_SOURCE_ROOT / "src" / "arm_controller" / "compact_arm_kinematics.py"
V04_CONFIG_PATH = V04_SOURCE_ROOT / "cpp" / "arm_controller_cpp" / "config" / "cars" / "v04.yaml"
# The controller-side face table decides what `inspect` accepts; use the checkout that is
# synced to the RK (tennis-man submodule), not the analysis checkout above.
V04_FACE_TABLE_PATH = V04_SOURCE_ROOT / "cpp" / "arm_controller_cpp" / "assets" / "v04" / "face_table.bin"
CAR = "v04"

ARM_COMMAND_TOPIC = "/tennis/arm_command"
STATUS_TOPIC = "/tennis/status"
JOINT_STATE_TOPIC = "/joint_states"
HIT_TOPIC = "/predict_hit_pos"
PANEL_STREAM_TOPIC = "/tennis/panel_stream"
NODE_NAME = "v04_ht_replay_capture"

STATUS_TIMEOUT_S = 3.0
JOINT_MAX_AGE_S = 0.5
SETTLE_WINDOW_S = 0.6
SETTLE_MIN_SAMPLES = 8
SETTLE_SPAN_RAD = math.radians(0.08)
# The controller solves `inspect` off the baked face table (25mm trilinear grid),
# so the settled TCP is allowed to miss the exact-IK target by a few millimetres.
FK_TARGET_TOL_M = 0.015
CAMERA_SYNC_MAX_MS = 10.0
# The swing face pitch the controller bakes into `hit_pose` (cfg::kSwingFacePitchRad).
SWING_FACE_PITCH_DEG = 24.0
JOINT_ALLOW_RATE = 0.9


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    label: str
    source: str
    x_m: float
    z_model_m: float
    q_ref_deg: tuple[float, ...]
    loft_deg: float
    rel_x_m: float | None = None
    rel_z_m: float | None = None
    ht_rk_abs_s: float | None = None
    ct_rk_abs_s: float | None = None


@dataclass(frozen=True)
class Trial:
    index: int
    trial_id: str
    phase: str
    target: Target


def load_kinematics() -> tuple[Any, dict[str, Any]]:
    import importlib.util

    import yaml

    if not V04_KINEMATICS_PATH.is_file() or not V04_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"V04 source is required at {V04_SOURCE_ROOT}")
    spec = importlib.util.spec_from_file_location("v04_compact_arm_kinematics", V04_KINEMATICS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V04_KINEMATICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.use_car(CAR)
    config = yaml.safe_load(V04_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("car") != CAR:
        raise ExperimentError(f"expected car={CAR}, got {config.get('car')!r}")
    return module, config


def read_final_ht_targets(session: str) -> list[dict[str, float]]:
    """Pull each throw's FinalHT /predict_hit_pos payload out of a tracker session."""
    base = TRACKER_OUTPUT / session / session
    tables = json.loads(Path(f"{base}_tables.json").read_text(encoding="utf-8"))
    rk = json.loads(Path(f"{base}_rk_tracking.json").read_text(encoding="utf-8"))
    t0 = float(rk["t0"])
    times = rk["pred"]["t"]
    payload = rk["pred"]["y"]
    rows = []
    for row in tables["arm_contract"]["rows"]:
        ct_rel = float(row["finalCtRkAbs"]) - t0
        index = min(range(len(times)), key=lambda k: abs(float(times[k]) - ct_rel))
        if abs(float(times[index]) - ct_rel) > 0.002:
            raise ExperimentError(
                f"throw {row['reportRow']}: no /predict_hit_pos within 2ms of the accepted ct"
            )
        ht_rel = float(payload["ht_rel"][index])
        if abs((ht_rel + t0) - float(row["finalHtRkAbs"])) > 0.002:
            raise ExperimentError(
                f"throw {row['reportRow']}: matched message ht disagrees with the report"
            )
        rows.append(
            {
                "throw": int(row["reportRow"]),
                "rel_x_m": float(payload["rel_x"][index]),
                "rel_z_m": float(payload["rel_z"][index]),
                "ht_rk_abs_s": float(row["finalHtRkAbs"]),
                "ct_rk_abs_s": float(row["finalCtRkAbs"]),
                "stage": int(payload["stage"][index]),
                "n_bounce_fit": int(payload["n_bounce_fit"][index]),
            }
        )
    if not rows:
        raise ExperimentError(f"{session}: no FinalHT rows")
    return rows


def solve_reference_pose(kin: Any, config: dict[str, Any], x_m: float, z_model_m: float):
    """Exact-IK reference pose for the commanded point, mirroring build_face_table's ladder."""
    import numpy as np

    limits = config["joint_limits"]
    lower = np.radians(np.asarray(limits["lower_deg"], dtype=np.float64)) * JOINT_ALLOW_RATE
    upper = np.radians(np.asarray(limits["upper_deg"], dtype=np.float64)) * JOINT_ALLOW_RATE
    pitch = math.radians(SWING_FACE_PITCH_DEG)
    step = math.radians(0.5)
    n_phi = int(round(math.radians(60.0) / step))
    for index in range(n_phi + 1):
        for phi in ((0.0,) if index == 0 else (index * step, -index * step)):
            try:
                seed = kin.ik_hit(x_m, z_model_m, racket_angle=phi, elbow="up")
            except ValueError:
                continue
            if any(seed[a] < lower[a] or seed[a] > upper[a] for a in (1, 2, 3)):
                continue
            try:
                q = kin.ik_hit_face(
                    x_m, z_model_m, pitch=pitch, yaw=0.0, tilt=phi, seed=seed, elbow="up"
                )
            except ValueError:
                continue
            if bool(np.all(q >= lower) and np.all(q <= upper)):
                return q, phi
    raise ExperimentError(
        f"({x_m:.4f}, {z_model_m:.4f}) has no in-limit yaw-free pose "
        f"at pitch {SWING_FACE_PITCH_DEG:g}deg"
    )


def build_targets(session: str) -> tuple[list[Target], dict[str, Any]]:
    import numpy as np

    kin, config = load_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    targets: list[Target] = []
    for row in read_final_ht_targets(session):
        # `inspect` carries 4 decimals, so quantise here: the commanded point, the
        # reference pose and the offline target must all be the same number.
        x_m = round(row["rel_x_m"], 4)
        z_model_m = round(row["rel_z_m"] + z_offset, 4)
        q, loft = solve_reference_pose(kin, config, x_m, z_model_m)
        tcp = np.asarray(kin.fk(q)["tcp"], dtype=np.float64)
        residual = math.hypot(float(tcp[0]) - x_m, float(tcp[2]) - z_model_m)
        if residual > 1e-4:
            raise ExperimentError(
                f"throw {row['throw']}: reference IK does not reproduce the target"
            )
        targets.append(
            Target(
                label=f"throw{row['throw']}",
                source="session_final_ht",
                rel_x_m=row["rel_x_m"],
                rel_z_m=row["rel_z_m"],
                x_m=x_m,
                z_model_m=z_model_m,
                ht_rk_abs_s=row["ht_rk_abs_s"],
                ct_rk_abs_s=row["ct_rk_abs_s"],
                q_ref_deg=tuple(float(v) for v in np.degrees(q)),
                loft_deg=math.degrees(loft),
            )
        )
    meta = {
        "plan": "session_final_ht",
        "session": session,
        "car": CAR,
        "hit_pos_z_offset_m": z_offset,
        "swing_face_pitch_deg": SWING_FACE_PITCH_DEG,
        "kinematics_source": str(V04_KINEMATICS_PATH),
    }
    return targets, meta


def _calibration_regressor_block(kin: Any, q, np_mod):
    """One pose's rows of the joint-zero regressor: [dJ/dq1..5 | R_link6 | car dx,dy]."""
    block = np_mod.zeros((3, 10))
    step = 1e-6
    for j in range(5):
        plus, minus = q.copy(), q.copy()
        plus[j] += step
        minus[j] -= step
        block[:, j] = (kin.fk(plus)["tcp"] - kin.fk(minus)["tcp"]) / (2.0 * step)
    block[:, 5:8] = kin.fk(q)["link_transforms"][kin.JOINTS[-1]["child"]][:3, :3]
    block[0, 8] = 1.0
    block[1, 9] = 1.0
    # 1deg of joint against 1mm of offset, so the SVD compares like with like
    return block * np_mod.array([math.radians(1.0)] * 5 + [1e-3] * 5)


def load_face_table(path: Path = V04_FACE_TABLE_PATH) -> dict[str, Any]:
    """Parse tools/export_cpp_assets.py's FTB1 binary exactly as control/face_table.cpp does."""
    import hashlib
    import struct

    import numpy as np

    raw = path.read_bytes()
    if raw[:4] != b"FTB1":
        raise ExperimentError(f"face table bad magic: {path}")
    nx, nz, npitch = struct.unpack_from("<3I", raw, 4)
    x0, z0, pitch0, grid, pitch_grid = struct.unpack_from("<5d", raw, 16)
    cells = nx * nz * npitch
    ok = np.frombuffer(raw, dtype=np.uint8, count=cells, offset=56).reshape(nx, nz, npitch)
    q = np.frombuffer(raw, dtype="<f8", count=cells * 6, offset=56 + cells).reshape(nx, nz, npitch, 6)
    return {
        "path": str(path),
        "md5": hashlib.md5(raw).hexdigest(),
        "nx": nx, "nz": nz, "np": npitch,
        "x0": x0, "z0": z0, "pitch0": pitch0, "grid": grid, "pitch_grid": pitch_grid,
        "ok": ok, "q": q,
    }


def face_table_pose(table: dict[str, Any], x_m: float, z_model_m: float, pitch_rad: float):
    """Trilinear q at (x, z, pitch) or None where the controller would reject the cell."""
    import math as _math

    import numpy as np

    fx = (x_m - table["x0"]) / table["grid"]
    fz = (z_model_m - table["z0"]) / table["grid"]
    fp = (pitch_rad - table["pitch0"]) / table["pitch_grid"]
    if fx < 0 or fz < 0 or fp < 0:
        return None
    if fx > table["nx"] - 1 or fz > table["nz"] - 1 or fp > table["np"] - 1:
        return None
    i = min(int(_math.floor(fx)), table["nx"] - 2)
    k = min(int(_math.floor(fz)), table["nz"] - 2)
    j = min(int(_math.floor(fp)), table["np"] - 2)
    ok = table["ok"]
    for di in (0, 1):
        for dk in (0, 1):
            for dj in (0, 1):
                if not ok[i + di, k + dk, j + dj]:
                    return None
    wx, wz, wp = fx - i, fz - k, fp - j
    out = np.zeros(6)
    for di in (0, 1):
        for dk in (0, 1):
            for dj in (0, 1):
                w = (wx if di else 1 - wx) * (wz if dk else 1 - wz) * (wp if dj else 1 - wp)
                out += w * table["q"][i + di, k + dk, j + dj]
    return out


def build_sweep_targets(
    x_range: tuple[float, float],
    z_ground_range: tuple[float, float],
    grid_m: float,
    count: int,
) -> tuple[list[Target], dict[str, Any]]:
    """Greedy D-optimal pose set for identifying the joint zeros.

    joint6 is deliberately absent from the regressor: the model puts the TCP on the
    joint6 axis, so rotating it does not move the point at all and no amount of
    position data can pin its zero (that needs the racket face orientation).
    """
    import numpy as np

    kin, config = load_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    table = load_face_table()
    pitch = math.radians(SWING_FACE_PITCH_DEG)
    pool: list[dict[str, Any]] = []
    for z_ground in np.arange(z_ground_range[0], z_ground_range[1] + 1e-9, grid_m):
        for x in np.arange(x_range[0], x_range[1] + 1e-9, grid_m):
            x_m = round(float(x), 4)
            z_model_m = round(float(z_ground) + z_offset, 4)
            q = face_table_pose(table, x_m, z_model_m, pitch)
            if q is None:
                continue
            loft = -q[1] - q[2] + q[3]
            pool.append(
                {
                    "x_m": x_m,
                    "z_model_m": z_model_m,
                    "z_ground_m": round(float(z_ground), 4),
                    "q": q,
                    "loft": loft,
                    "block": _calibration_regressor_block(kin, q, np),
                }
            )
    if len(pool) < count:
        raise ExperimentError(f"only {len(pool)} reachable poses in the requested envelope")

    chosen: list[int] = []
    for _ in range(count):
        best_value, best_index = -1.0, None
        for index in range(len(pool)):
            if index in chosen:
                continue
            stacked = np.vstack([pool[k]["block"] for k in chosen + [index]])
            singular = np.linalg.svd(stacked, compute_uv=False)
            value = singular[-1] if len(chosen) + 1 >= 4 else singular[0]
            if value > best_value:
                best_value, best_index = value, index
        chosen.append(best_index)

    stacked = np.vstack([pool[k]["block"] for k in chosen])
    singular = np.linalg.svd(stacked, compute_uv=False)
    sigma_deg_mm = np.sqrt(np.diag(np.linalg.inv(stacked.T @ stacked))) * 1e-3
    targets = [
        Target(
            label=f"S{order + 1:02d}",
            source="joint_zero_sweep",
            x_m=pool[k]["x_m"],
            z_model_m=pool[k]["z_model_m"],
            q_ref_deg=tuple(float(v) for v in np.degrees(pool[k]["q"])),
            loft_deg=math.degrees(pool[k]["loft"]),
        )
        for order, k in enumerate(chosen)
    ]
    meta = {
        "plan": "joint_zero_sweep",
        "car": CAR,
        "hit_pos_z_offset_m": z_offset,
        "swing_face_pitch_deg": SWING_FACE_PITCH_DEG,
        "kinematics_source": str(V04_KINEMATICS_PATH),
        "face_table": {"path": table["path"], "md5": table["md5"]},
        "envelope": {
            "x_m": list(x_range),
            "z_ground_m": list(z_ground_range),
            "grid_m": grid_m,
            "reachable_pool": len(pool),
        },
        "design": {
            "condition_number": float(singular[0] / singular[-1]),
            "smallest_singular_mm": float(singular[-1] * 1000.0),
            "sigma_at_1mm_noise": dict(
                zip(
                    ["dq1_deg", "dq2_deg", "dq3_deg", "dq4_deg", "dq5_deg",
                     "tool_x_mm", "tool_y_mm", "tool_z_mm", "car_x_mm", "car_y_mm"],
                    [float(v) for v in sigma_deg_mm],
                )
            ),
        },
    }
    return targets, meta


def build_plan(targets: list[Target], repeats: int) -> tuple[Trial, ...]:
    """Alternate the visit order so pose hysteresis shows up as a pass-to-pass split."""
    trials: list[Trial] = []
    for pass_index in range(repeats):
        forward = pass_index % 2 == 0
        ordered = targets if forward else list(reversed(targets))
        phase = f"pass{pass_index + 1}_{'fwd' if forward else 'rev'}"
        for target in ordered:
            trials.append(Trial(len(trials), f"HTR{len(trials) + 1:03d}", phase, target))
    return tuple(trials)


def inspect_command(target: Target) -> str:
    return f"inspect {target.x_m:.4f} {target.z_model_m:.4f}"


def parse_inspect_status(text: str, target: Target) -> dict[str, Any]:
    prefix = "accepted arm_command inspect "
    if not text.startswith(prefix):
        raise ExperimentError(f"unexpected inspect status: {text}")
    fields: dict[str, str] = {}
    for token in text[len(prefix):].split():
        key, separator, value = token.partition("=")
        if not separator or key in fields:
            raise ExperimentError(f"malformed inspect status: {text}")
        fields[key] = value
    if set(fields) != {"x", "z", "duration", "t"}:
        raise ExperimentError(f"unexpected inspect fields: {text}")
    values = {key: float(value) for key, value in fields.items()}
    expected = {"x": target.x_m, "z": target.z_model_m, "duration": 1.0}
    for key, wanted in expected.items():
        tolerance = 5.1e-5 if key != "duration" else 5.1e-4
        if not math.isfinite(values[key]) or abs(values[key] - wanted) > tolerance:
            raise ExperimentError(f"accepted {key}={values[key]:g}, expected {wanted:g}")
    if not math.isfinite(values["t"]):
        raise ExperimentError("accepted status time is not finite")
    return {**values, "raw_status": text}


def parse_preset_status(text: str, command: str) -> dict[str, Any]:
    prefix = f"accepted arm_command {command} duration="
    if not text.startswith(prefix):
        raise ExperimentError(f"unexpected {command} status: {text}")
    duration_text, separator, status_t_text = text[len(prefix):].partition(" t=")
    if not separator:
        raise ExperimentError(f"malformed {command} status: {text}")
    duration = float(duration_text)
    status_t = float(status_t_text)
    expected = 1.0 if command == "inspect ready" else 8.0
    if not math.isfinite(duration) or abs(duration - expected) > 5.1e-4:
        raise ExperimentError(f"accepted {command} duration={duration:g}, expected {expected:g}")
    if not math.isfinite(status_t):
        raise ExperimentError("accepted status time is not finite")
    return {"command": command, "duration": duration, "t": status_t, "raw_status": text}


def print_plan(targets: list[Target], plan: tuple[Trial, ...], meta: dict[str, Any]) -> None:
    print(
        f"V04 static capture [{meta['plan']}]: {len(targets)} targets, {len(plan)} poses"
        + (f"  session={meta['session']}" if "session" in meta else "")
    )
    print(
        f"z_model = z_ground + hit_pos_z_offset_m ({meta['hit_pos_z_offset_m']:+.9f}m); "
        f"face pitch {meta['swing_face_pitch_deg']:g}deg"
    )
    if "design" in meta:
        design = meta["design"]
        sigma = design["sigma_at_1mm_noise"]
        print(
            f"design: cond={design['condition_number']:.0f} "
            f"s_min={design['smallest_singular_mm']:.3f}mm  "
            "1sigma@1mm noise: "
            + " ".join(f"{k.replace('_deg', '')}={v:.2f}deg" for k, v in sigma.items() if k.endswith("_deg"))
        )
    for target in targets:
        ground_z = target.z_model_m - meta["hit_pos_z_offset_m"]
        origin = (
            f"rel=({target.rel_x_m:.4f}, {target.rel_z_m:.4f})m -> "
            if target.rel_x_m is not None
            else ""
        )
        print(
            f"  {target.label:>7} {origin}inspect x={target.x_m:.4f} "
            f"z_model={target.z_model_m:.4f} (ground z={ground_z:.4f}) "
            f"loft={target.loft_deg:+.2f}deg "
            f"q_ref_deg=[{', '.join(f'{v:.2f}' for v in target.q_ref_deg)}]"
        )
    for trial in plan:
        print(
            f"{trial.trial_id} {trial.phase:>10} {trial.target.label:>7} "
            f"{inspect_command(trial.target)}"
        )


def execute(targets: list[Target], plan: tuple[Trial, ...], meta: dict[str, Any]) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import cv2
    import numpy as np
    # On this Windows ROS2 build, Torch preloads the MSVC/CUDA runtime DLLs that
    # rmw_cyclonedds_cpp also needs; the tracker uses the same import order.
    import torch  # noqa: F401
    from src.ros2_support import ensure_ros2_environment

    ensure_ros2_environment()
    import rclpy
    from rcl_interfaces.msg import ParameterType
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    from src import SyncCapture, frame_to_numpy

    kin, _config = load_kinematics()
    reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__(NODE_NAME)
            self._lock = threading.Lock()
            self._joints: deque[dict[str, Any]] = deque(maxlen=5000)
            self._statuses: deque[tuple[float, str]] = deque(maxlen=500)
            self.arm_pub = self.create_publisher(String, ARM_COMMAND_TOPIC, reliable)
            self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint, reliable)
            self.create_subscription(String, STATUS_TOPIC, self._on_status, reliable)
            self._controller_params = AsyncParameterClient(self, "/arm_controller_cpp")
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
            names = [f"joint{index}" for index in range(1, 7)]
            if any(name not in indices for name in names):
                return
            try:
                q = np.asarray([msg.position[indices[name]] for name in names], dtype=float)
                v = np.asarray([msg.velocity[indices[name]] for name in names], dtype=float)
                effort = np.asarray([msg.effort[indices[name]] for name in names], dtype=float)
            except (IndexError, TypeError, ValueError):
                return
            if not all(np.all(np.isfinite(values)) for values in (q, v, effort)):
                return
            with self._lock:
                self._joints.append(
                    {
                        "received_perf": time.perf_counter(),
                        "header_stamp_ns": (
                            int(msg.header.stamp.sec) * 1_000_000_000
                            + int(msg.header.stamp.nanosec)
                        ),
                        "q": q,
                        "v": v,
                        "effort": effort,
                    }
                )

        def _on_status(self, msg: String) -> None:
            with self._lock:
                self._statuses.append((time.perf_counter(), str(msg.data).strip()))

        def latest_joint(self) -> dict[str, Any] | None:
            with self._lock:
                if not self._joints:
                    return None
                row = self._joints[-1]
                return {
                    **row,
                    "q": row["q"].copy(),
                    "v": row["v"].copy(),
                    "effort": row["effort"].copy(),
                }

        @staticmethod
        def _endpoint_name(endpoint: Any) -> str:
            namespace = str(endpoint.node_namespace).rstrip("/")
            return f"{namespace}/{endpoint.node_name}" if namespace else f"/{endpoint.node_name}"

        def _graph(self) -> dict[str, list[str]]:
            publishers = lambda topic: sorted(  # noqa: E731
                self._endpoint_name(endpoint)
                for endpoint in self.get_publishers_info_by_topic(topic)
            )
            subscribers = lambda topic: sorted(  # noqa: E731
                self._endpoint_name(endpoint)
                for endpoint in self.get_subscriptions_info_by_topic(topic)
            )
            return {
                "arm_command_publishers": publishers(ARM_COMMAND_TOPIC),
                "arm_command_subscribers": subscribers(ARM_COMMAND_TOPIC),
                "hit_publishers": publishers(HIT_TOPIC),
                "panel_stream_publishers": publishers(PANEL_STREAM_TOPIC),
                "joint_state_publishers": publishers(JOINT_STATE_TOPIC),
                "status_publishers": publishers(STATUS_TOPIC),
            }

        @staticmethod
        def _safe_graph(graph: dict[str, list[str]]) -> bool:
            """Nothing but this script may be able to move the arm while it holds a pose."""
            return (
                graph["arm_command_publishers"] == ["/arm_cpp_ready_session", f"/{NODE_NAME}"]
                and "/arm_controller_cpp" in graph["arm_command_subscribers"]
                and not graph["hit_publishers"]
                and not graph["panel_stream_publishers"]
                and graph["joint_state_publishers"] == ["/arm_controller_cpp"]
                and graph["status_publishers"] == ["/arm_controller_cpp"]
            )

        def _read_controller_runtime(self, timeout_s: float) -> dict[str, Any]:
            if not self._controller_params.wait_for_services(timeout_sec=timeout_s):
                raise ExperimentError("arm_controller_cpp parameter services are unavailable")
            future = self._controller_params.get_parameters(["mode", "config_path", "assets_dir"])
            deadline = time.perf_counter() + timeout_s
            while not future.done() and time.perf_counter() < deadline:
                time.sleep(0.02)
            if not future.done():
                future.cancel()
                raise ExperimentError("timed out reading arm_controller_cpp parameters")
            response = future.result()
            if response is None or len(response.values) != 3:
                raise ExperimentError("invalid arm_controller_cpp parameter response")
            if any(value.type != ParameterType.PARAMETER_STRING for value in response.values):
                raise ExperimentError("arm_controller_cpp identity parameters are not strings")
            mode, config_path, assets_dir = (
                value.string_value.replace("\\", "/") for value in response.values
            )
            if mode != "active":
                raise ExperimentError(f"arm_controller_cpp mode is {mode!r}, expected 'active'")
            if not config_path.endswith("/config/cars/v04.yaml"):
                raise ExperimentError(f"controller is not using V04 config: {config_path}")
            if not assets_dir.endswith("/assets/v04"):
                raise ExperimentError(f"controller is not using V04 assets: {assets_dir}")
            return {
                "node": "arm_controller_cpp",
                "car": CAR,
                "mode": mode,
                "config_path": config_path,
                "assets_dir": assets_dir,
            }

        def wait_ready(self, timeout_s: float = 10.0) -> dict[str, Any]:
            deadline = time.perf_counter() + timeout_s
            graph: dict[str, list[str]] = {}
            while time.perf_counter() < deadline:
                latest = self.latest_joint()
                graph = self._graph()
                if (
                    latest is not None
                    and time.perf_counter() - float(latest["received_perf"]) <= JOINT_MAX_AGE_S
                    and self._safe_graph(graph)
                ):
                    runtime = self._read_controller_runtime(
                        max(0.1, deadline - time.perf_counter())
                    )
                    return {**runtime, "graph": graph}
                time.sleep(0.05)
            raise ExperimentError(
                f"timed out waiting for the V04 capture graph and fresh joints: {graph}"
            )

        def assert_command_graph(self) -> None:
            graph = self._graph()
            if not self._safe_graph(graph):
                raise ExperimentError(f"unsafe ROS command graph: {graph}")

        def command(self, text: str, parser: Any) -> tuple[float, dict[str, Any]]:
            self.assert_command_graph()
            sent_at = time.perf_counter()
            msg = String()
            msg.data = text
            self.arm_pub.publish(msg)
            deadline = sent_at + STATUS_TIMEOUT_S
            while time.perf_counter() < deadline:
                with self._lock:
                    statuses = list(self._statuses)
                for received_at, status in statuses:
                    if received_at < sent_at:
                        continue
                    if status.startswith("reject ") or status.startswith("error "):
                        raise ExperimentError(f"controller: {status}")
                    if status.startswith("accepted arm_command "):
                        return sent_at, parser(status)
                time.sleep(0.02)
            raise ExperimentError(f"no controller acceptance for {text!r}")

        def wait_settled(
            self, target_xz: tuple[float, float] | None, timeout_s: float = 8.0
        ) -> dict[str, Any]:
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                with self._lock:
                    rows = [
                        row
                        for row in self._joints
                        if float(row["received_perf"]) >= now - SETTLE_WINDOW_S
                    ]
                if rows and now - float(rows[-1]["received_perf"]) > JOINT_MAX_AGE_S:
                    raise ExperimentError("/joint_states became stale")
                if len(rows) >= SETTLE_MIN_SAMPLES:
                    observed = float(rows[-1]["received_perf"] - rows[0]["received_perf"])
                    headers = [int(row["header_stamp_ns"]) for row in rows]
                    positions = np.asarray([row["q"] for row in rows])
                    latest = rows[-1]
                    span = float(np.max(np.ptp(positions, axis=0)))
                    tcp = np.asarray(kin.fk(latest["q"])["tcp"], dtype=np.float64)
                    fk_error = (
                        0.0
                        if target_xz is None
                        else math.hypot(
                            float(tcp[0]) - target_xz[0], float(tcp[2]) - target_xz[1]
                        )
                    )
                    if (
                        observed >= SETTLE_WINDOW_S - 0.1
                        and len(set(headers)) >= SETTLE_MIN_SAMPLES
                        and headers[-1] > headers[0]
                        and span <= SETTLE_SPAN_RAD
                        and fk_error <= FK_TARGET_TOL_M
                    ):
                        return {
                            "received_perf": float(latest["received_perf"]),
                            "header_stamp_ns": int(latest["header_stamp_ns"]),
                            "q_rad": latest["q"].tolist(),
                            "v_rad_s": latest["v"].tolist(),
                            "effort": latest["effort"].tolist(),
                            "fk_tcp_model_m": tcp.tolist(),
                            "fk_target_error_mm": 1000.0 * fk_error,
                            "position_span_deg": math.degrees(span),
                            "observed_s": observed,
                        }
                time.sleep(0.03)
            raise ExperimentError("arm did not settle at the requested pose")

        def nearest_joint(self, exposure_perf: float) -> dict[str, Any]:
            with self._lock:
                if not self._joints:
                    raise ExperimentError("no joint state near camera exposure")
                row = min(
                    self._joints,
                    key=lambda item: abs(float(item["received_perf"]) - exposure_perf),
                )
            delta = abs(float(row["received_perf"]) - exposure_perf)
            if delta > 0.2:
                raise ExperimentError(
                    f"nearest joint state is {delta:.3f}s from camera exposure"
                )
            return {
                "received_perf": float(row["received_perf"]),
                "header_stamp_ns": int(row["header_stamp_ns"]),
                "q_rad": row["q"].tolist(),
                "q_deg": np.degrees(row["q"]).tolist(),
                "v_rad_s": row["v"].tolist(),
                "effort": row["effort"].tolist(),
                "exposure_delta_ms": 1000.0 * delta,
            }

        def close(self) -> None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._executor.remove_node(self)
            self._executor.shutdown(timeout_sec=1.0)
            self.destroy_node()

    output = OUTPUT_ROOT / f"v04_ht_replay_{time.perf_counter_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(
        json.dumps(
            {
                "schema": "v04_ht_replay/v1",
                "camera_config": str(CAMERA_CONFIG),
                **meta,
                "targets": [target.__dict__ for target in targets],
                "trials": [
                    {
                        "trial_id": trial.trial_id,
                        "phase": trial.phase,
                        "point": trial.target.label,
                        "command": inspect_command(trial.target),
                    }
                    for trial in plan
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    monitor: Monitor | None = None
    motion_started = False
    result_count = 0
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    runtime: dict[str, Any] | None = None
    try:
        monitor = Monitor()
        runtime = monitor.wait_ready()
        camera_cfg = json.loads(CAMERA_CONFIG.read_text(encoding="utf-8"))
        expected_serials = [camera_cfg["master_serial"], *camera_cfg["slave_serials"]]
        with SyncCapture.from_config(str(CAMERA_CONFIG)) as cap:
            if list(cap.sync_serials) != expected_serials:
                raise ExperimentError(
                    f"unexpected synchronized cameras: {list(cap.sync_serials)}"
                )
            time.sleep(2.0)
            motion_started = True
            _, ready = monitor.command(
                "inspect ready", lambda status: parse_preset_status(status, "inspect ready")
            )
            time.sleep(float(ready["duration"]) + 0.2)
            monitor.wait_settled(None, timeout_s=5.0)

            results_path = output / "results.jsonl"
            for trial in plan:
                target = trial.target
                command = inspect_command(target)
                sent_at, accepted = monitor.command(
                    command, lambda status, t=target: parse_inspect_status(status, t)
                )
                settled = monitor.wait_settled((target.x_m, target.z_model_m))
                frames = None
                exposures: list[float] = []
                for _attempt in range(5):
                    candidate = cap.get_frames(timeout_s=2.0)
                    if candidate is None or set(candidate) != set(expected_serials):
                        raise ExperimentError("synchronized four-camera capture timed out")
                    candidate_exposures = [
                        float(frame.exposure_start_pc) for frame in candidate.values()
                    ]
                    if min(candidate_exposures) >= float(settled["received_perf"]):
                        frames = candidate
                        exposures = candidate_exposures
                        break
                if frames is None:
                    raise ExperimentError("camera did not deliver an exposure after settling")
                if any(int(frame.lost_packet) != 0 for frame in frames.values()):
                    raise ExperimentError("camera frame reported lost packets")
                sync_spread_ms = 1000.0 * (max(exposures) - min(exposures))
                if sync_spread_ms > CAMERA_SYNC_MAX_MS:
                    raise ExperimentError(
                        f"four-camera exposure spread is {sync_spread_ms:.3f}ms"
                    )
                exposure_perf = float(sum(exposures) / len(exposures))
                joint_at_exposure = monitor.nearest_joint(exposure_perf)
                exposure_tcp = np.asarray(
                    kin.fk(np.asarray(joint_at_exposure["q_rad"], dtype=np.float64))["tcp"],
                    dtype=np.float64,
                )
                exposure_error = math.hypot(
                    float(exposure_tcp[0]) - target.x_m,
                    float(exposure_tcp[2]) - target.z_model_m,
                )
                if exposure_error > FK_TARGET_TOL_M:
                    raise ExperimentError("arm left the target before camera exposure")

                point_dir = output / f"{trial.index + 1:02d}_{trial.trial_id}_{target.label}"
                point_dir.mkdir()
                files: dict[str, str] = {}
                frame_numbers: dict[str, int] = {}
                for serial in expected_serials:
                    frame = frames[serial]
                    image = frame_to_numpy(frame, rotate_180=False)
                    if image.shape[:2] != (1536, 2048):
                        raise ExperimentError(
                            f"{serial} frame is {image.shape[:2]}, expected 1536x2048"
                        )
                    path = point_dir / f"{serial}.png"
                    if not cv2.imwrite(str(path), image):
                        raise ExperimentError(f"failed to save {path}")
                    files[serial] = str(path.relative_to(output))
                    frame_numbers[serial] = int(frame.frame_num)

                record = {
                    "schema": "v04_ht_replay/v1",
                    "trial_id": trial.trial_id,
                    "index": trial.index,
                    "phase": trial.phase,
                    "point": target.label,
                    "point_source": target.source,
                    "target": {
                        "rel_x_m": target.rel_x_m,
                        "rel_z_m": target.rel_z_m,
                        "x_m": target.x_m,
                        "z_model_m": target.z_model_m,
                        "z_ground_m": target.z_model_m - meta["hit_pos_z_offset_m"],
                        "pitch_deg": SWING_FACE_PITCH_DEG,
                        "loft_deg": target.loft_deg,
                        "q_ref_deg": list(target.q_ref_deg),
                        "ht_rk_abs_s": target.ht_rk_abs_s,
                    },
                    "command": command,
                    "command_sent_perf": sent_at,
                    "accepted": accepted,
                    "settled": settled,
                    "exposure_perf": exposure_perf,
                    "exposure_fk_target_error_mm": 1000.0 * exposure_error,
                    "sync_spread_ms": sync_spread_ms,
                    "frame_num": frame_numbers,
                    "joint_at_exposure": joint_at_exposure,
                    "files": files,
                }
                with results_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                result_count += 1
                print(
                    f"CAPTURED {result_count:02d}/{len(plan)} {trial.trial_id} "
                    f"{target.label} x={target.x_m:.4f} z={target.z_model_m:.4f} "
                    f"fk_err={settled['fk_target_error_mm']:.2f}mm sync={sync_spread_ms:.3f}ms",
                    flush=True,
                )
    except BaseException as exc:  # noqa: BLE001
        primary_error = exc
    finally:
        if monitor is not None and motion_started and rclpy.ok():
            try:
                _, droop = monitor.command(
                    "droop", lambda status: parse_preset_status(status, "droop")
                )
                time.sleep(float(droop["duration"]) + 0.2)
                monitor.wait_settled(None, timeout_s=5.0)
                print("DROOP_COMPLETE", flush=True)
            except BaseException as exc:  # noqa: BLE001
                cleanup_error = exc
                print(f"DROOP_FAILED: {exc}", file=sys.stderr, flush=True)
        if monitor is not None:
            monitor.close()
        if rclpy.ok():
            rclpy.shutdown()

    session_record = {
        "schema": "v04_ht_replay/v1",
        "status": (
            "complete"
            if primary_error is None and cleanup_error is None and result_count == len(plan)
            else "failed"
        ),
        "result_count": result_count,
        "expected_count": len(plan),
        "runtime_config": runtime,
        "droop_complete": motion_started and cleanup_error is None,
        **meta,
    }
    if primary_error is not None:
        session_record["error"] = f"{type(primary_error).__name__}: {primary_error}"
    if cleanup_error is not None:
        session_record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
    (output / "session.json").write_text(
        json.dumps(session_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if cleanup_error is not None:
        raise ExperimentError(f"droop cleanup failed: {cleanup_error}") from cleanup_error
    if primary_error is not None:
        raise primary_error
    if result_count != len(plan):
        raise ExperimentError(f"captured {result_count}/{len(plan)} trials")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drive V04 through static poses and snapshot the 18F rig at each one"
    )
    parser.add_argument(
        "--plan",
        choices=("session", "sweep"),
        default="session",
        help="session = replay a tracker session's FinalHT targets; "
        "sweep = D-optimal pose set for identifying the joint zeros",
    )
    parser.add_argument("--session", default="tracker_20260904_063357")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="passes over the target list, alternating direction (default: session=2, sweep=1)",
    )
    parser.add_argument("--sweep-count", type=int, default=30)
    parser.add_argument("--sweep-x", type=float, nargs=2, default=(0.75, 1.20), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--sweep-z-ground",
        type=float,
        nargs=2,
        default=(1.10, 1.50),
        metavar=("MIN", "MAX"),
        help="ground-referenced hit height; the default stays inside the envelope "
        "the HT replay already exercised",
    )
    parser.add_argument("--sweep-grid", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    repeats = args.repeats if args.repeats is not None else (1 if args.plan == "sweep" else 2)
    if repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.plan == "sweep":
        targets, meta = build_sweep_targets(
            tuple(args.sweep_x), tuple(args.sweep_z_ground), args.sweep_grid, args.sweep_count
        )
    else:
        targets, meta = build_targets(args.session)
    plan = build_plan(targets, repeats)
    print_plan(targets, plan, meta)
    if not args.execute:
        print("DRY RUN: no ROS command was sent and no camera was opened.")
        return 0
    try:
        output = execute(targets, plan, meta)
    except (ExperimentError, FileNotFoundError) as exc:
        print(f"EXPERIMENT STOPPED: {exc}", file=sys.stderr)
        return 2
    print(f"COMPLETE: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
