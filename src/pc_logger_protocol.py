# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .ros2_support import ensure_ros2_environment, make_reliable_qos

JOINT_STATE_TOPIC = "/joint_states"
MIT_COMMAND_TOPIC = "/arm_logger/mit_command"
HIT_EVENT_TOPIC = "/arm_logger/hit_event"
LOGGER_CONTROL_TOPIC = "/arm_logger/control"
PC_CAR_LOC_TOPIC = "/pc_car_loc"
PREDICT_HIT_POS_TOPIC = "/predict_hit_pos"
TIME_SYNC_OFFSET_TOPIC = "/time_sync/offset"


def _path_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve())


def json_string_message(payload: dict):
    ensure_ros2_environment()
    from std_msgs.msg import String

    return String(
        data=json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    )


def build_logger_control_payload(
    command: str,
    *,
    source: str = "tracker",
    reason: str | None = None,
    command_id: str | None = None,
    run_id: str | None = None,
    group_id: str | None = None,
    target_path: str | Path | None = None,
    tracker_output_dir: str | Path | None = None,
    tracker_json_path: str | Path | None = None,
    tracker_video_path: str | Path | None = None,
    extra: dict | None = None,
) -> dict:
    normalized_command = str(command).strip().lower()
    payload = {
        "schema": "pc_logger_control_v1",
        "command": normalized_command,
        "command_id": command_id or f"{normalized_command}-{uuid.uuid4().hex}",
        "source": str(source),
        "stamp_pc_ns": time.perf_counter_ns(),
    }
    if reason:
        payload["reason"] = str(reason)
    if run_id:
        payload["run_id"] = str(run_id)
    if group_id:
        payload["group_id"] = str(group_id)
    if target_path is not None:
        payload["target_path"] = _path_text(target_path)
    if tracker_output_dir is not None:
        payload["tracker_output_dir"] = _path_text(tracker_output_dir)
    if tracker_json_path is not None:
        payload["tracker_json_path"] = _path_text(tracker_json_path)
    if tracker_video_path is not None:
        payload["tracker_video_path"] = _path_text(tracker_video_path)
    if extra:
        payload.update(dict(extra))
    return payload


class JsonTopicPublisher:
    def __init__(
        self,
        topic: str,
        *,
        node=None,
        node_name: str = "pc_json_topic_publisher",
        qos_profile=None,
        stamp_provider_ns=None,
    ):
        self._owns_context = False
        self._external_node = node is not None
        self._rclpy = None
        self._node = node
        self._publisher = None
        self._stamp_provider_ns = stamp_provider_ns
        if self._node is None:
            ensure_ros2_environment()
            import rclpy
            from std_msgs.msg import String

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_context = True
            self._node = rclpy.create_node(node_name)
            self._msg_type = String
        else:
            ensure_ros2_environment()
            from std_msgs.msg import String

            self._msg_type = String
        if qos_profile is None:
            qos_profile = make_reliable_qos(depth=100)
        self._publisher = self._node.create_publisher(
            self._msg_type,
            topic,
            qos_profile,
        )

    @property
    def node(self):
        return self._node

    def now_ns(self) -> int:
        if self._stamp_provider_ns is not None:
            return int(self._stamp_provider_ns())
        return time.perf_counter_ns()

    def publish(self, payload: dict) -> None:
        if self._publisher is None:
            return
        message_payload = dict(payload)
        message_payload.setdefault("stamp_pc_ns", self.now_ns())
        self._publisher.publish(json_string_message(message_payload))

    def close(self) -> None:
        if self._external_node:
            return
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
