# -*- coding: utf-8 -*-
"""15.6 live monitor: POE FK + car/racket/ball multi-camera localization."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ArmCalibration.common import DEFAULT_CAMERA_CONFIG_PATH, rel_or_abs
from src import BallLocalizer, CarDetection, CarLocalizer, SyncCapture, frame_to_numpy
from src.arm_poe import ArmPoePositionModel
from src.racket_localizer import RacketDetection, RacketLocalizer


DEFAULT_CAMERA_CALIB_PATH = project_root / "src" / "config" / "four_camera_calib.json"
DEFAULT_POE_CONFIG_PATH = project_root / "src" / "config" / "arm_poe_racket_center.json"
DEFAULT_RACKET_MODEL_PATH = project_root / "yolo_model" / "racket.onnx"
DEFAULT_RACKET_POSE_MODEL_PATH = project_root / "yolo_model" / "racket_pose.onnx"


def provider_list_from_mode(mode: str) -> Optional[list[str]]:
    if mode == "cpu":
        return ["CPUExecutionProvider"]
    if mode == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


class JointStateCollector(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("arm_live_monitor_joint_state_collector")
        self._topic = topic
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=2000)
        self._first_msg = threading.Event()
        self._stop = threading.Event()
        self.create_subscription(JointState, topic, self._on_msg, qos_profile_sensor_data)
        self._thread = threading.Thread(target=self._spin_loop, name="LiveJointStateCollector", daemon=True)
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
            "received_time_pc": float(time.time()),
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

    def latest_snapshot(self, *, max_age_s: float) -> Optional[dict]:
        now = time.time()
        with self._lock:
            if not self._history:
                return None
            latest = deepcopy(self._history[-1])
        latest["age_s"] = float(now - float(latest["received_time_pc"]))
        if latest["age_s"] > max_age_s:
            return None
        return latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="15.6 live POE + car/racket/ball monitor at 1 Hz.",
    )
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG_PATH))
    parser.add_argument("--camera-calib", type=str, default=str(DEFAULT_CAMERA_CALIB_PATH))
    parser.add_argument("--poe-config", type=str, default=str(DEFAULT_POE_CONFIG_PATH))
    parser.add_argument("--joint-topic", type=str, default="/joint_states")
    parser.add_argument("--max-joint-age", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=1.5)
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-prints", type=int, default=0, help="Stop after N print cycles. 0 means run forever.")
    parser.add_argument("--pixel-format", type=str, default="BayerRG8")
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--racket-keypoint-threshold", type=float, default=40.0)
    parser.add_argument("--ball-conf", type=float, default=0.25)
    parser.add_argument("--racket-model", type=str, default=str(DEFAULT_RACKET_MODEL_PATH))
    parser.add_argument("--racket-pose-model", type=str, default=str(DEFAULT_RACKET_POSE_MODEL_PATH))
    parser.add_argument(
        "--racket-bbox-provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution provider preference for racket bbox detection.",
    )
    parser.add_argument(
        "--racket-pose-provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution provider preference for racket keypoint recognition.",
    )
    parser.add_argument("--no-ball", action="store_true", help="Disable ball location output.")
    return parser.parse_args()


def choose_best_tag(
    detections_by_camera: dict[str, list[CarDetection]],
) -> tuple[Optional[int], dict[str, CarDetection]]:
    tag_cameras: dict[int, dict[str, CarDetection]] = {}
    for serial, detections in detections_by_camera.items():
        for det in detections:
            tag_cameras.setdefault(det.tag_id, {})[serial] = det

    best_tag: Optional[int] = None
    best_dets: dict[str, CarDetection] = {}
    for tag_id, matched in tag_cameras.items():
        if len(matched) >= 2 and len(matched) > len(best_dets):
            best_tag = tag_id
            best_dets = matched
    return best_tag, best_dets


def format_vec(vec) -> str:
    return f"({vec[0]:.1f}, {vec[1]:.1f}, {vec[2]:.1f})"


def format_uv(uv) -> str:
    return f"({uv[0]:.1f}, {uv[1]:.1f})"


def format_joint_values(values) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


def emphasize(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[1;96m{text}\033[0m"


def summarize_car_observations(
    serials: list[str],
    detections_by_camera: dict[str, list[CarDetection]],
    best_tag: Optional[int],
) -> str:
    parts: list[str] = []
    for serial in serials:
        detections = detections_by_camera.get(serial, [])
        if not detections:
            parts.append(f"{serial[-4:]}=none")
            continue
        if best_tag is not None:
            chosen = next((det for det in detections if det.tag_id == best_tag), detections[0])
        else:
            chosen = detections[0]
        parts.append(f"{serial[-4:]}=tag{chosen.tag_id}@({chosen.cx:.0f},{chosen.cy:.0f})")
    return "  ".join(parts)


def summarize_racket_observations(
    serials: list[str],
    detections_by_camera: dict[str, RacketDetection],
) -> str:
    parts: list[str] = []
    for serial in serials:
        det = detections_by_camera.get(serial)
        if det is None or not det.detected:
            parts.append(f"{serial[-4:]}=none")
            continue
        if det.accepted and det.center_xy is not None:
            parts.append(
                f"{serial[-4:]}={format_uv(det.center_xy)}/min={det.face_keypoint_score_min:.1f}"
            )
        else:
            parts.append(f"{serial[-4:]}=reject({det.failure_reason})")
    return "  ".join(parts)


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be > 0")

    overrides = {}
    if args.pixel_format.strip():
        overrides["pixel_format"] = args.pixel_format.strip()

    poe_model = ArmPoePositionModel(config_path=args.poe_config)
    car_localizer = CarLocalizer(calib_config_path=args.camera_calib)
    racket_localizer = RacketLocalizer(
        calib_config_path=args.camera_calib,
        racket_model_path=args.racket_model,
        pose_model_path=args.racket_pose_model,
        bbox_conf=args.bbox_conf,
        keypoint_score_threshold=args.racket_keypoint_threshold,
        min_valid_keypoints=4,
        bbox_onnx_providers=provider_list_from_mode(args.racket_bbox_provider),
        pose_providers=provider_list_from_mode(args.racket_pose_provider),
    )
    ball_localizer = None if args.no_ball else BallLocalizer(calib_config_path=args.camera_calib)

    print("=== ArmCalibration 15.6 Live Monitor ===")
    print(f"Camera config: {rel_or_abs(Path(args.camera_config))}")
    print(f"Camera calib:  {rel_or_abs(Path(args.camera_calib))}")
    print(f"POE config:    {rel_or_abs(Path(args.poe_config))}")
    print(f"Joint topic:   {args.joint_topic}")
    print(f"Pixel format:  {overrides.get('pixel_format', '(camera default)')}")
    print(f"Racket bbox provider: {args.racket_bbox_provider}")
    print(f"Racket pose provider: {args.racket_pose_provider}")
    print(f"Loop rate:     {args.rate_hz:.2f} Hz")
    print(
        "AprilTag->car/base offset (world axes): "
        f"{format_vec(poe_model.apriltag_to_car_base_offset_mm)} mm"
    )

    rclpy.init()
    collector = JointStateCollector(args.joint_topic)
    try:
        if not collector.wait_for_first_message(timeout_s=5.0):
            print("ERROR: did not receive JointState within 5 seconds.")
            return 1

        with SyncCapture.from_config(args.camera_config, **overrides) as cap:
            print(f"Sync cameras:  {cap.sync_serials}")
            time.sleep(args.warmup)

            loop_period_s = 1.0 / args.rate_hz
            next_tick = time.monotonic()
            printed = 0

            while True:
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_tick += loop_period_s

                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frames = cap.get_frames(timeout_s=args.timeout)
                if frames is None:
                    print(f"\n[{stamp}]")
                    print("frames: unavailable (synchronized capture timeout)")
                    continue

                images = {serial: frame_to_numpy(frame) for serial, frame in frames.items()}
                serials = list(images.keys())
                joint_snapshot = collector.latest_snapshot(max_age_s=args.max_joint_age)
                poe_summary = "unavailable"
                vision_summary = "unavailable"

                print(f"\n[{stamp}]")
                if joint_snapshot is None:
                    print(f"detail: joint_state unavailable (older than {args.max_joint_age:.2f}s)")
                else:
                    try:
                        joint_vector = poe_model.extract_joint_vector_from_snapshot(joint_snapshot)
                        fk = poe_model.forward(joint_vector)
                        delta_world_from_base = fk.point_world_mm - poe_model.t_base_in_world_mm
                        poe_summary = f"{format_vec(delta_world_from_base)} mm"
                        print(
                            "detail: joint_state "
                            f"age={joint_snapshot['age_s']:.3f}s  "
                            f"q={format_joint_values(joint_snapshot['position'])}"
                        )
                        print(
                            "detail: poe "
                            f"p_racket_rel_base_in_base={format_vec(fk.point_base_mm)} mm  "
                            f"p_racket_world(by poe)={format_vec(fk.point_world_mm)} mm  "
                            f"p_racket_rel_base_in_world(by poe)={format_vec(delta_world_from_base)} mm"
                        )
                    except Exception as exc:
                        print(f"detail: joint_state failed to evaluate POE ({exc})")

                detections_by_camera = {serial: car_localizer.detect(image) for serial, image in images.items()}
                best_tag, best_tag_dets = choose_best_tag(detections_by_camera)
                print(f"detail: car_obs {summarize_car_observations(serials, detections_by_camera, best_tag)}")
                car_loc = None
                car_world_mm = None
                if best_tag is not None:
                    car_loc = car_localizer.triangulate(best_tag_dets, t=time.perf_counter())
                    car_world_mm = (
                        float(car_loc.x + poe_model.apriltag_to_car_base_offset_mm[0]),
                        float(car_loc.y + poe_model.apriltag_to_car_base_offset_mm[1]),
                        float(car_loc.z + poe_model.apriltag_to_car_base_offset_mm[2]),
                    )
                    print(
                        "detail: p_apriltag "
                        f"tag={car_loc.tag_id}  {format_vec((car_loc.x, car_loc.y, car_loc.z))} mm  "
                        f"yaw={math.degrees(car_loc.yaw):.1f} deg  "
                        f"reproj={car_loc.reprojection_error:.2f}px  "
                        f"cams={'+'.join(sn[-4:] for sn in car_loc.cameras_used)}"
                    )
                    print(
                        "detail: p_car "
                        f"{format_vec(car_world_mm)} mm  "
                        f"(p_apriltag + {format_vec(poe_model.apriltag_to_car_base_offset_mm)} mm)"
                    )
                else:
                    print("detail: p_apriltag unavailable (<2 cameras with the same AprilTag)")
                    print("detail: p_car unavailable")

                racket_dets, racket_loc = racket_localizer.locate(images)
                print(f"detail: racket_obs {summarize_racket_observations(serials, racket_dets)}")
                if racket_loc is not None:
                    print(
                        "detail: p_racket_world(by vision) "
                        f"{format_vec((racket_loc.x, racket_loc.y, racket_loc.z))} mm  "
                        f"reproj={racket_loc.reprojection_error:.2f}px  "
                        f"face_min={racket_loc.face_keypoint_score_min:.1f}  "
                        f"cams={'+'.join(sn[-4:] for sn in racket_loc.cameras_used)}"
                    )
                else:
                    print("detail: p_racket_world(by vision) unavailable (<2 cameras passed racket 0-3 keypoint threshold)")

                if racket_loc is not None and car_world_mm is not None:
                    delta = (
                        racket_loc.x - car_world_mm[0],
                        racket_loc.y - car_world_mm[1],
                        racket_loc.z - car_world_mm[2],
                    )
                    vision_summary = f"{format_vec(delta)} mm"
                else:
                    vision_summary = "unavailable"

                print(emphasize("=" * 96))
                print(emphasize(f"KEY  p_racket_rel_base_in_world(by poe):    {poe_summary}"))
                print(emphasize(f"KEY  p_racket_rel_base_in_world(by vision): {vision_summary}"))
                print(emphasize("=" * 96))

                if ball_localizer is not None:
                    ball = ball_localizer.locate(images, conf=args.ball_conf)
                    if ball is not None:
                        print(
                            "detail: p_ball "
                            f"{format_vec((ball.x, ball.y, ball.z))} mm  "
                            f"reproj={ball.reprojection_error:.2f}px  "
                            f"cams={'+'.join(sn[-4:] for sn in ball.cameras_used)}"
                        )
                    else:
                        print("detail: p_ball unavailable")

                printed += 1
                if args.max_prints > 0 and printed >= args.max_prints:
                    break

    finally:
        collector.close()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
