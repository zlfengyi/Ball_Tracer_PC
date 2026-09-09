#!/usr/bin/env python3
"""Move V04 through a fixed local grid and save one synchronized 18F image set per pose."""
from __future__ import annotations

import argparse
from collections import Counter, deque
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
OUTPUT_ROOT = PROJECT_ROOT / "arm_controller_data"

ARM_COMMAND_TOPIC = "/tennis/arm_command"
STATUS_TOPIC = "/tennis/status"
JOINT_STATE_TOPIC = "/joint_states"
HIT_TOPIC = "/predict_hit_pos"
PANEL_STREAM_TOPIC = "/tennis/panel_stream"

STATUS_TIMEOUT_S = 3.0
JOINT_MAX_AGE_S = 0.5
SETTLE_WINDOW_S = 0.6
SETTLE_MIN_SAMPLES = 8
SETTLE_SPAN_RAD = math.radians(0.08)
TARGET_TOL_RAD = math.radians(1.0)
CAMERA_SYNC_MAX_MS = 10.0


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Point:
    point_id: str
    x_m: float
    z_model_m: float
    z_ground_m: float
    q_ref_deg: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Trial:
    index: int
    trial_id: str
    phase: str
    point: Point


POINTS = {
    "C0": Point(
        "C0", 0.8245, 0.9253, 1.095650644,
        (-20.0902, -1.4248, 107.0873, 70.5504, 29.7632, 8.7556),
    ),
    "X1": Point(
        "X1", 0.8500, 0.9253, 1.095650644,
        (-18.2917, 1.7658, 108.2920, 78.0048, 27.8797, 11.1026),
    ),
    "X2": Point(
        "X2", 0.8750, 0.9253, 1.095650644,
        (-16.5509, 5.1671, 109.0855, 85.1134, 25.9073, 13.2110),
    ),
    "Cross": Point(
        "Cross", 0.8500, 0.9753, 1.145650644,
        (-16.9208, -0.1885, 108.8939, 81.9262, 25.6990, 14.0815),
    ),
    "Z1": Point(
        "Z1", 0.8245, 0.9753, 1.145650644,
        (-18.6201, -3.5729, 109.7273, 77.6863, 27.4795, 12.6208),
    ),
}

PLAN_SPEC = (
    ("C0", "anchor_start"),
    ("X1", "forward_1"),
    ("X2", "forward_1"),
    ("Cross", "forward_1"),
    ("Z1", "forward_1"),
    ("C0", "forward_1_end"),
    ("Z1", "reverse"),
    ("Cross", "reverse"),
    ("X2", "reverse"),
    ("X1", "reverse"),
    ("C0", "reverse_end"),
    ("X1", "forward_2"),
    ("X2", "forward_2"),
    ("Cross", "forward_2"),
    ("Z1", "forward_2"),
    ("C0", "anchor_end"),
)


def build_plan() -> tuple[Trial, ...]:
    return tuple(
        Trial(index, f"VTP{index + 1:03d}", phase, POINTS[point_id])
        for index, (point_id, phase) in enumerate(PLAN_SPEC)
    )


def inspect_command(point: Point) -> str:
    return f"inspect {point.x_m:.4f} {point.z_model_m:.4f}"


def parse_inspect_status(text: str, point: Point) -> dict[str, Any]:
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
    expected = {"x": point.x_m, "z": point.z_model_m, "duration": 1.0}
    for key, target in expected.items():
        tolerance = 5.1e-5 if key != "duration" else 5.1e-4
        if not math.isfinite(values[key]) or abs(values[key] - target) > tolerance:
            raise ExperimentError(
                f"accepted {key}={values[key]:g}, expected {target:g}"
            )
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
        raise ExperimentError(
            f"accepted {command} duration={duration:g}, expected {expected:g}"
        )
    if not math.isfinite(status_t):
        raise ExperimentError("accepted status time is not finite")
    return {
        "command": command,
        "duration": duration,
        "t": status_t,
        "raw_status": text,
    }


def print_plan(plan: tuple[Trial, ...]) -> None:
    counts = Counter(trial.point.point_id for trial in plan)
    print(
        "V04 visual TCP static capture: "
        f"{len(plan)} poses, counts={dict(counts)}"
    )
    for trial in plan:
        point = trial.point
        print(
            f"{trial.trial_id} {trial.phase:>13} {point.point_id:>5} "
            f"x={point.x_m:.4f} z_model={point.z_model_m:.4f}"
        )


def execute(plan: tuple[Trial, ...]) -> Path:
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

    reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__("v04_visual_tcp_static_capture")
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
            sample = {
                "received_perf": time.perf_counter(),
                "header_stamp_ns": (
                    int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec)
                ),
                "q": q,
                "v": v,
                "effort": effort,
            }
            with self._lock:
                self._joints.append(sample)

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
            return (
                graph["arm_command_publishers"]
                == ["/arm_cpp_ready_session", "/v04_visual_tcp_static_capture"]
                and "/arm_controller_cpp" in graph["arm_command_subscribers"]
                and not graph["hit_publishers"]
                and not graph["panel_stream_publishers"]
                and graph["joint_state_publishers"] == ["/arm_controller_cpp"]
                and graph["status_publishers"] == ["/arm_controller_cpp"]
            )

        def _read_controller_runtime(self, timeout_s: float) -> dict[str, Any]:
            if not self._controller_params.wait_for_services(timeout_sec=timeout_s):
                raise ExperimentError("arm_controller_cpp parameter services are unavailable")
            future = self._controller_params.get_parameters(
                ["mode", "config_path", "assets_dir"]
            )
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
                "car": "v04",
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
                "timed out waiting for the V04 capture graph and fresh joints: "
                f"{graph}"
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
            self, target_q: Any | None, timeout_s: float = 8.0
        ) -> dict[str, Any]:
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                with self._lock:
                    rows = [
                        row for row in self._joints
                        if float(row["received_perf"]) >= now - SETTLE_WINDOW_S
                    ]
                if rows and now - float(rows[-1]["received_perf"]) > JOINT_MAX_AGE_S:
                    raise ExperimentError("/joint_states became stale")
                if len(rows) >= SETTLE_MIN_SAMPLES:
                    observed = float(rows[-1]["received_perf"] - rows[0]["received_perf"])
                    headers = [int(row["header_stamp_ns"]) for row in rows]
                    positions = np.asarray([row["q"] for row in rows])
                    latest = rows[-1]
                    target_error = (
                        0.0
                        if target_q is None
                        else float(np.max(np.abs(latest["q"] - target_q)))
                    )
                    if (
                        observed >= SETTLE_WINDOW_S - 0.1
                        and len(set(headers)) >= SETTLE_MIN_SAMPLES
                        and headers[-1] > headers[0]
                        and float(np.max(np.ptp(positions, axis=0))) <= SETTLE_SPAN_RAD
                        and target_error <= TARGET_TOL_RAD
                    ):
                        return {
                            "received_perf": float(latest["received_perf"]),
                            "header_stamp_ns": int(latest["header_stamp_ns"]),
                            "q_rad": latest["q"].tolist(),
                            "v_rad_s": latest["v"].tolist(),
                            "effort": latest["effort"].tolist(),
                            "target_error_deg": math.degrees(target_error),
                            "position_span_deg": math.degrees(
                                float(np.max(np.ptp(positions, axis=0)))
                            ),
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

    output = OUTPUT_ROOT / f"v04_visual_tcp_probe_{time.perf_counter_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(
        json.dumps(
            {
                "schema": "v04_visual_tcp_static/v1",
                "camera_config": str(CAMERA_CONFIG),
                "trials": [
                    {
                        "trial_id": trial.trial_id,
                        "phase": trial.phase,
                        "point": trial.point.__dict__,
                        "command": inspect_command(trial.point),
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
                point = trial.point
                command = inspect_command(point)
                sent_at, accepted = monitor.command(
                    command, lambda status, p=point: parse_inspect_status(status, p)
                )
                target_q = np.radians(np.asarray(point.q_ref_deg, dtype=np.float64))
                settled = monitor.wait_settled(target_q)
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
                exposure_q = np.asarray(joint_at_exposure["q_rad"], dtype=np.float64)
                if float(np.max(np.abs(exposure_q - target_q))) > TARGET_TOL_RAD:
                    raise ExperimentError("arm left the target before camera exposure")

                point_dir = output / f"{trial.index + 1:02d}_{trial.trial_id}_{point.point_id}"
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
                    "schema": "v04_visual_tcp_static/v1",
                    "trial_id": trial.trial_id,
                    "index": trial.index,
                    "phase": trial.phase,
                    "point_id": point.point_id,
                    "target": {
                        "x_m": point.x_m,
                        "z_model_m": point.z_model_m,
                        "z_ground_m": point.z_ground_m,
                        "pitch_deg": 24.0,
                        "q_ref_deg": list(point.q_ref_deg),
                    },
                    "command": command,
                    "command_sent_perf": sent_at,
                    "accepted": accepted,
                    "settled": settled,
                    "exposure_perf": exposure_perf,
                    "sync_spread_ms": sync_spread_ms,
                    "frame_num": frame_numbers,
                    "joint_at_exposure": joint_at_exposure,
                    "files": files,
                }
                with results_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                result_count += 1
                print(
                    f"CAPTURED {result_count:02d}/{len(plan)} {trial.trial_id} "
                    f"{point.point_id} x={point.x_m:.4f} z={point.z_model_m:.4f} "
                    f"sync={sync_spread_ms:.3f}ms",
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

    session = {
        "schema": "v04_visual_tcp_static/v1",
        "status": (
            "complete"
            if primary_error is None and cleanup_error is None and result_count == len(plan)
            else "failed"
        ),
        "result_count": result_count,
        "expected_count": len(plan),
        "runtime_config": runtime,
        "droop_complete": motion_started and cleanup_error is None,
    }
    if primary_error is not None:
        session["error"] = f"{type(primary_error).__name__}: {primary_error}"
    if cleanup_error is not None:
        session["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
    (output / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
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
        description="V04 local visual/FK calibration: ROS inspect then 18F snapshot"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan()
    print_plan(plan)
    if not args.execute:
        print("DRY RUN: no ROS command was sent and no camera was opened.")
        return 0
    try:
        output = execute(plan)
    except (ExperimentError, FileNotFoundError) as exc:
        print(f"EXPERIMENT STOPPED: {exc}", file=sys.stderr)
        return 2
    print(f"COMPLETE: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
