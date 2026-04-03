# -*- coding: utf-8 -*-
"""15.2: capture synchronized four-camera color images and latest JointState snapshots."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ArmCalibration.common import (
    ARM_DATA_ROOT,
    DEFAULT_CAMERA_CONFIG_PATH,
    auto_session_dir,
    load_camera_config,
    rel_or_abs,
    sanitize_session_name,
    save_json,
)
from src import SyncCapture, frame_to_numpy


class JointStateCollector(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("arm_calibration_joint_state_collector")
        self._topic = topic
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=2000)
        self._first_msg = threading.Event()
        self._stop = threading.Event()
        self.create_subscription(JointState, topic, self._on_msg, qos_profile_sensor_data)
        self._thread = threading.Thread(target=self._spin_loop, name="JointStateCollector", daemon=True)
        self._thread.start()

    def _spin_loop(self) -> None:
        while rclpy.ok() and not self._stop.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)

    def _on_msg(self, msg: JointState) -> None:
        stamp = msg.header.stamp
        snapshot = {
            "topic": self._topic,
            "header_stamp": {
                "sec": int(stamp.sec),
                "nanosec": int(stamp.nanosec),
            },
            "received_time_pc": float(time.perf_counter()),
            "name": list(msg.name),
            "position": [float(v) for v in msg.position],
            "velocity": [float(v) for v in msg.velocity],
            "effort": [float(v) for v in msg.effort],
        }
        with self._lock:
            self._history.append(snapshot)
        self._first_msg.set()

    def wait_for_first_message(self, timeout_s: float) -> bool:
        return self._first_msg.wait(timeout_s)

    def latest_snapshot(
        self,
        *,
        max_age_s: float,
        stable_window_s: float,
        stable_tolerance_rad: float,
    ) -> dict | None:
        now = time.perf_counter()
        with self._lock:
            if not self._history:
                return None
            latest = deepcopy(self._history[-1])
            history = list(self._history)

        age_s = now - float(latest["received_time_pc"])
        if age_s > max_age_s:
            return None

        recent = [
            entry for entry in history
            if float(latest["received_time_pc"]) - float(entry["received_time_pc"]) <= stable_window_s
        ]
        position_span: list[float] = []
        is_stable = True
        if len(recent) >= 2 and latest["position"]:
            positions = np.array([entry["position"] for entry in recent], dtype=np.float64)
            span = positions.max(axis=0) - positions.min(axis=0)
            position_span = [float(v) for v in span]
            is_stable = bool(float(span.max()) <= stable_tolerance_rad)

        latest["age_s"] = float(age_s)
        latest["stable_window_s"] = float(stable_window_s)
        latest["stable_tolerance_rad"] = float(stable_tolerance_rad)
        latest["recent_message_count"] = len(recent)
        latest["position_span_rad"] = position_span
        latest["is_stable"] = is_stable
        return latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _build_session_manifest(session_dir: Path, args: argparse.Namespace, overrides: dict) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_name": session_dir.name,
        "session_dir": rel_or_abs(session_dir),
        "step": "15.2",
        "capture_mode": "manual" if args.manual else "auto",
        "target_samples": int(args.count),
        "duration_s": None if args.duration <= 0 else float(args.duration),
        "interval_s": float(args.interval),
        "joint_topic": args.joint_topic,
        "joint_model": "sensor_msgs/msg/JointState",
        "ros2_standard_messages": {
            "joint_state": "sensor_msgs/msg/JointState",
            "image": "sensor_msgs/msg/Image",
            "camera_info": "sensor_msgs/msg/CameraInfo",
        },
        "camera_config": {
            "path": rel_or_abs(Path(args.camera_config)),
            "snapshot": load_camera_config(args.camera_config),
            "overrides": overrides,
        },
        "filters": {
            "max_joint_age_s": float(args.max_joint_age),
            "require_stable_joints": bool(args.require_stable_joints),
            "stable_window_s": float(args.stable_window),
            "stable_tolerance_rad": float(args.stable_tolerance),
        },
        "results": {
            "captured_samples": 0,
            "rejected_attempts": 0,
            "sample_ids": [],
        },
    }


def _frame_metadata(frame) -> dict:
    return {
        "width": int(frame.width),
        "height": int(frame.height),
        "frame_num": int(frame.frame_num),
        "pixel_type": int(frame.pixel_type),
        "dev_timestamp": int(frame.dev_timestamp),
        "host_timestamp": int(frame.host_timestamp),
        "exposure_time_us": float(frame.exposure_time),
        "lost_packet": int(frame.lost_packet),
        "arrival_perf": float(frame.arrival_perf),
        "exposure_start_pc": float(frame.exposure_start_pc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="15.2 capture synchronized four-camera color images and latest JointState.")
    parser.add_argument("--count", type=int, default=30, help="Number of samples to capture.")
    parser.add_argument("--duration", type=float, default=0.0, help="Total auto-capture duration in seconds. If >0, interval = duration / count.")
    parser.add_argument("--manual", action="store_true", help="Press Enter for each sample instead of timed capture.")
    parser.add_argument("--interval", type=float, default=2.0, help="Auto-capture interval in seconds.")
    parser.add_argument("--session", type=str, default="", help="Explicit session directory name.")
    parser.add_argument("--label", type=str, default="15_2_capture", help="Session label when auto-naming.")
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG_PATH))
    parser.add_argument("--joint-topic", type=str, default="/joint_states")
    parser.add_argument("--max-joint-age", type=float, default=0.3)
    parser.add_argument("--require-stable-joints", action="store_true", help="Reject a sample if recent joint positions are still moving.")
    parser.add_argument("--stable-window", type=float, default=0.25)
    parser.add_argument("--stable-tolerance", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--exposure", type=float, default=-1.0)
    parser.add_argument("--gain", type=float, default=-1.0)
    parser.add_argument("--pixel-format", type=str, default="BayerRG8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if not args.manual and args.duration > 0:
        args.interval = float(args.duration) / float(args.count)
    if not args.manual and args.interval <= 0:
        raise SystemExit("--interval must be > 0")

    ARM_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if args.session:
        session_name = sanitize_session_name(args.session) or args.session
        session_dir = ARM_DATA_ROOT / session_name
    else:
        session_dir = auto_session_dir(ARM_DATA_ROOT, args.label)
    session_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.exposure > 0:
        overrides["exposure_us"] = args.exposure
    if args.gain >= 0:
        overrides["gain_db"] = args.gain
    if args.pixel_format.strip():
        overrides["pixel_format"] = args.pixel_format.strip()

    manifest = _build_session_manifest(session_dir, args, overrides)
    save_json(session_dir / "session.json", manifest)

    print("=== ArmCalibration Capture ===")
    print(f"  Session: {rel_or_abs(session_dir)}")
    print(f"  Mode: {'manual' if args.manual else 'auto'}")
    print(f"  Capture target: {args.count}")
    print(f"  Joint topic: {args.joint_topic}")
    print(f"  Pixel format: {overrides.get('pixel_format', '(camera default)')}")
    if not args.manual:
        print(f"  Interval: {args.interval:.3f}s")

    rclpy.init()
    collector = JointStateCollector(args.joint_topic)
    try:
        if not collector.wait_for_first_message(timeout_s=5.0):
            print("ERROR: did not receive /joint_states within 5 seconds.")
            return 1

        with SyncCapture.from_config(args.camera_config, **overrides) as cap:
            print(f"  Sync cameras: {cap.sync_serials}")
            print(f"  Warmup: {args.warmup:.1f}s")
            time.sleep(args.warmup)

            accepted = 0
            rejected = 0
            next_capture_time = time.perf_counter()

            while accepted < args.count:
                if args.manual:
                    input(f"\nPose the arm, then press Enter to capture sample {accepted + 1}/{args.count}...")
                else:
                    sleep_s = next_capture_time - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    next_capture_time += args.interval

                frames = cap.get_frames(timeout_s=2.0)
                if frames is None:
                    rejected += 1
                    print("  Rejected: timed out waiting for synchronized frames.")
                    continue

                joint_snapshot = collector.latest_snapshot(
                    max_age_s=args.max_joint_age,
                    stable_window_s=args.stable_window,
                    stable_tolerance_rad=args.stable_tolerance,
                )
                if joint_snapshot is None:
                    rejected += 1
                    print("  Rejected: latest JointState is too old.")
                    continue
                if args.require_stable_joints and not joint_snapshot["is_stable"]:
                    rejected += 1
                    print(f"  Rejected: joints not stable enough. span={joint_snapshot['position_span_rad']}")
                    continue

                images = {}
                for serial, frame in frames.items():
                    image = frame_to_numpy(frame)
                    images[serial] = image

                accepted += 1
                sample_id = f"sample_{accepted:04d}"
                sample_dir = session_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                image_paths = {}
                frame_group_meta = {}
                image_shapes = {}
                for serial, frame in frames.items():
                    image_path = sample_dir / f"{serial}.png"
                    cv2.imwrite(str(image_path), images[serial])
                    image_paths[serial] = image_path.name
                    frame_group_meta[serial] = _frame_metadata(frame)
                    image_shapes[serial] = list(images[serial].shape)

                sample_payload = {
                    "sample_id": sample_id,
                    "captured_at": datetime.now().isoformat(timespec="seconds"),
                    "session_dir": rel_or_abs(session_dir),
                    "images": image_paths,
                    "image_shapes": image_shapes,
                    "frame_group": frame_group_meta,
                    "joint_state": joint_snapshot,
                    "accepted_for_capture": True,
                }
                save_json(sample_dir / "sample.json", sample_payload)

                manifest["results"]["captured_samples"] = accepted
                manifest["results"]["rejected_attempts"] = rejected
                manifest["results"]["sample_ids"].append(sample_id)
                save_json(session_dir / "session.json", manifest)

                print(
                    f"  Captured {accepted}/{args.count}: {sample_id} "
                    f"(joint_age={joint_snapshot['age_s']:.3f}s)"
                )

    finally:
        collector.close()
        collector.destroy_node()
        rclpy.shutdown()

    print("\nCapture finished.")
    print(f"  Session saved to: {rel_or_abs(session_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
