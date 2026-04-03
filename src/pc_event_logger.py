# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pc_logger_protocol import (
    HIT_EVENT_TOPIC,
    JOINT_STATE_TOPIC,
    LOGGER_CONTROL_TOPIC,
    MIT_COMMAND_TOPIC,
    PC_CAR_LOC_TOPIC,
    PREDICT_HIT_POS_TOPIC,
    TIME_SYNC_OFFSET_TOPIC,
)
from src.ros2_support import (
    ensure_ros2_environment,
    make_best_effort_qos,
    make_reliable_qos,
)

ensure_ros2_environment()

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


@dataclass(frozen=True)
class PcEventLoggerConfig:
    target_path: Path
    ready_file: Path | None
    run_id: str | None
    group_id: str | None
    tracker_output_dir: Path | None
    tracker_json_path: Path | None
    tracker_video_path: Path | None
    idle_record_period_sec: float
    high_rate_tail_sec: float
    min_save_interval_sec: float
    post_hit_save_delay_sec: float
    tick_period_sec: float


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _item_primary_stamp_ns(item: dict, *, kind: str) -> int | None:
    if kind == "joint_state":
        return _safe_int(item.get("stamp_pc_ns"))
    if kind in {
        "mit_command",
        "hit_event",
        "control",
        "pc_car_loc",
        "predict_hit_pos",
        "time_sync_offset",
    }:
        return _safe_int(item.get("stamp_pc_ns"))
    return None


def _item_scheduled_hit_pc_ns(item: dict) -> int | None:
    return _safe_int(item.get("scheduled_hit_time_pc_ns"))


class PcEventLoggerNode(Node):
    def __init__(self, *, config: PcEventLoggerConfig):
        super().__init__("pc_event_logger")
        self.config = config
        self._queue: queue.SimpleQueue[dict | None] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._shutdown_requested = threading.Event()
        self._worker = threading.Thread(target=self._worker_main, daemon=True)
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ready_written = False

        self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._handle_joint_state,
            make_best_effort_qos(depth=100),
        )
        self.create_subscription(
            String,
            MIT_COMMAND_TOPIC,
            self._handle_mit_command,
            make_best_effort_qos(depth=100),
        )
        self.create_subscription(
            String,
            HIT_EVENT_TOPIC,
            self._handle_hit_event,
            make_best_effort_qos(depth=100),
        )
        self.create_subscription(
            String,
            LOGGER_CONTROL_TOPIC,
            self._handle_control,
            make_reliable_qos(depth=20),
        )
        self.create_subscription(
            String,
            PC_CAR_LOC_TOPIC,
            self._handle_pc_car_loc,
            make_best_effort_qos(depth=100),
        )
        self.create_subscription(
            String,
            PREDICT_HIT_POS_TOPIC,
            self._handle_predict_hit_pos,
            make_reliable_qos(depth=100),
        )
        self.create_subscription(
            String,
            TIME_SYNC_OFFSET_TOPIC,
            self._handle_time_sync_offset,
            make_best_effort_qos(depth=100),
        )

        self._worker.start()
        self._ticker.start()
        self._write_ready_file()
        self.get_logger().info(
            "pc_event_logger started: joint_state=%s mit=%s hit=%s control=%s car=%s predict=%s time_sync_offset=%s target=%s"
            % (
                JOINT_STATE_TOPIC,
                MIT_COMMAND_TOPIC,
                HIT_EVENT_TOPIC,
                LOGGER_CONTROL_TOPIC,
                PC_CAR_LOC_TOPIC,
                PREDICT_HIT_POS_TOPIC,
                TIME_SYNC_OFFSET_TOPIC,
                str(self.config.target_path),
            )
        )

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def destroy_node(self):
        self._stop_event.set()
        self._queue.put({"kind": "__tick__", "reason": "shutdown"})
        self._queue.put(None)
        self._worker.join(timeout=5.0)
        self._ticker.join(timeout=2.0)
        self._remove_ready_file()
        return super().destroy_node()

    def _write_ready_file(self) -> None:
        if self.config.ready_file is None or self._ready_written:
            return
        self.config.ready_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.ready_file.write_text(
            json.dumps(
                {
                    "schema": "pc_event_logger_ready_v1",
                    "stamp_pc_ns": time.perf_counter_ns(),
                    "target_path": str(self.config.target_path.resolve()),
                },
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        self._ready_written = True

    def _remove_ready_file(self) -> None:
        if self.config.ready_file is None:
            return
        try:
            self.config.ready_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _tick_loop(self):
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.config.tick_period_sec):
                break
            self._queue.put({"kind": "__tick__", "reason": "tick"})

    def _handle_joint_state(self, message: JointState):
        receipt_stamp_pc_ns = time.perf_counter_ns()
        stamp_pc_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        if stamp_pc_ns <= 0:
            stamp_pc_ns = None
        self._queue.put(
            {
                "kind": "joint_state",
                "stamp_pc_ns": stamp_pc_ns,
                "receipt_stamp_pc_ns": receipt_stamp_pc_ns,
                "name": [str(name) for name in message.name],
                "position": [float(value) for value in message.position],
                "velocity": [float(value) for value in message.velocity],
                "effort": [float(value) for value in message.effort],
            }
        )

    def _handle_mit_command(self, message: String):
        payload = self._parse_json_topic_payload(
            message=message,
            topic=MIT_COMMAND_TOPIC,
            kind="mit_command",
        )
        if payload is not None:
            self._queue.put(payload)

    def _handle_hit_event(self, message: String):
        payload = self._parse_json_topic_payload(
            message=message,
            topic=HIT_EVENT_TOPIC,
            kind="hit_event",
        )
        if payload is not None:
            self._queue.put(payload)

    def _handle_control(self, message: String):
        payload = self._parse_json_topic_payload(
            message=message,
            topic=LOGGER_CONTROL_TOPIC,
            kind="control",
        )
        if payload is None:
            return
        self._queue.put(payload)
        if str(payload.get("command", "")).strip().lower() == "shutdown":
            self._shutdown_requested.set()

    def _handle_pc_car_loc(self, message: String):
        payload = self._parse_json_topic_payload(
            message=message,
            topic=PC_CAR_LOC_TOPIC,
            kind="pc_car_loc",
        )
        if payload is not None:
            self._queue.put(payload)

    def _handle_predict_hit_pos(self, message: String):
        payload = self._parse_json_topic_payload(
            message=message,
            topic=PREDICT_HIT_POS_TOPIC,
            kind="predict_hit_pos",
        )
        if payload is not None:
            self._queue.put(payload)

    def _handle_time_sync_offset(self, message: String):
        payload = self._parse_json_topic_payload(
            message=message,
            topic=TIME_SYNC_OFFSET_TOPIC,
            kind="time_sync_offset",
        )
        if payload is not None:
            self._queue.put(payload)

    def _parse_json_topic_payload(
        self,
        *,
        message: String,
        topic: str,
        kind: str,
    ) -> dict | None:
        receipt_stamp_pc_ns = time.perf_counter_ns()
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning(f"ignored invalid {topic} JSON payload")
            return None
        if not isinstance(payload, dict):
            self.get_logger().warning(f"ignored non-object {topic} payload")
            return None
        result = dict(payload)
        result["kind"] = kind
        result.setdefault("stamp_pc_ns", receipt_stamp_pc_ns)
        return result

    def _new_log_path(self) -> Path:
        stamp = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime())
            + f"_{time.time_ns() % 1_000_000_000:09d}"
        )
        return self.config.target_path.with_name(f"pc_event_logger_{stamp}.json")

    def _metadata_from_config(self, *, path: Path) -> dict:
        return {
            "run_id": self.config.run_id,
            "group_id": self.config.group_id,
            "tracker_output_dir": (
                str(self.config.tracker_output_dir.resolve())
                if self.config.tracker_output_dir is not None
                else None
            ),
            "tracker_json_path": (
                str(self.config.tracker_json_path.resolve())
                if self.config.tracker_json_path is not None
                else None
            ),
            "tracker_video_path": (
                str(self.config.tracker_video_path.resolve())
                if self.config.tracker_video_path is not None
                else None
            ),
            "active_file": str(path.resolve()),
        }

    def _metadata_from_control(self, payload: dict, *, path: Path) -> dict:
        metadata = self._metadata_from_config(path=path)
        if payload.get("run_id") is not None:
            metadata["run_id"] = str(payload.get("run_id"))
        if payload.get("group_id") is not None:
            metadata["group_id"] = str(payload.get("group_id"))
        if payload.get("tracker_output_dir") is not None:
            metadata["tracker_output_dir"] = str(
                Path(payload.get("tracker_output_dir")).resolve()
            )
        if payload.get("tracker_json_path") is not None:
            metadata["tracker_json_path"] = str(
                Path(payload.get("tracker_json_path")).resolve()
            )
        if payload.get("tracker_video_path") is not None:
            metadata["tracker_video_path"] = str(
                Path(payload.get("tracker_video_path")).resolve()
            )
        metadata["active_file"] = str(path.resolve())
        return metadata

    def _new_active_log(self, *, path: Path, metadata: dict | None = None) -> dict:
        metadata = dict(metadata or self._metadata_from_config(path=path))
        return {
            "schema": "pc_event_logger_v2",
            "saved_at_perf_s": None,
            "config": {
                "idle_record_period_sec": self.config.idle_record_period_sec,
                "high_rate_tail_sec": self.config.high_rate_tail_sec,
                "min_save_interval_sec": self.config.min_save_interval_sec,
                "post_hit_save_delay_sec": self.config.post_hit_save_delay_sec,
                "tick_period_sec": self.config.tick_period_sec,
            },
            "topics": {
                "joint_state_topic": JOINT_STATE_TOPIC,
                "mit_command_topic": MIT_COMMAND_TOPIC,
                "hit_event_topic": HIT_EVENT_TOPIC,
                "control_topic": LOGGER_CONTROL_TOPIC,
                "pc_car_loc_topic": PC_CAR_LOC_TOPIC,
                "predict_hit_pos_topic": PREDICT_HIT_POS_TOPIC,
                "time_sync_offset_topic": TIME_SYNC_OFFSET_TOPIC,
            },
            **metadata,
            "segment_start_ns": None,
            "segment_end_ns": None,
            "joint_states": [],
            "mit_command_frames": [],
            "hit_events": [],
            "control_events": [],
            "pc_car_loc_events": [],
            "predict_hit_pos_events": [],
            "time_sync_offset_events": [],
            "stats": {
                "joint_state_count": 0,
                "mit_command_count": 0,
                "hit_event_count": 0,
                "control_event_count": 0,
                "pc_car_loc_count": 0,
                "predict_hit_pos_count": 0,
                "time_sync_offset_count": 0,
            },
        }

    @staticmethod
    def _joint_sort_key(name: str):
        if str(name).startswith("joint_"):
            try:
                return int(str(name).split("_", 1)[1])
            except ValueError:
                return 999
        return 999

    def _update_segment_bounds(self, segment: dict, stamp_ns: int | None):
        if stamp_ns is None:
            return
        stamp_ns = int(stamp_ns)
        if segment["segment_start_ns"] is None or stamp_ns < int(
            segment["segment_start_ns"]
        ):
            segment["segment_start_ns"] = stamp_ns
        if segment["segment_end_ns"] is None or stamp_ns > int(
            segment["segment_end_ns"]
        ):
            segment["segment_end_ns"] = stamp_ns

    def _record_segment_item(self, segment: dict, item: dict):
        kind = str(item.get("kind", ""))
        stamp_ns = _item_primary_stamp_ns(item, kind=kind)
        self._update_segment_bounds(segment, stamp_ns)
        if kind == "joint_state":
            segment["joint_states"].append(
                {k: v for k, v in item.items() if k != "kind"}
            )
            segment["stats"]["joint_state_count"] += 1
        elif kind == "mit_command":
            segment["mit_command_frames"].append(
                {k: v for k, v in item.items() if k != "kind"}
            )
            segment["stats"]["mit_command_count"] += 1
        elif kind == "hit_event":
            segment["hit_events"].append({k: v for k, v in item.items() if k != "kind"})
            segment["stats"]["hit_event_count"] += 1
        elif kind == "control":
            segment["control_events"].append(
                {k: v for k, v in item.items() if k != "kind"}
            )
            segment["stats"]["control_event_count"] += 1
        elif kind == "pc_car_loc":
            segment["pc_car_loc_events"].append(
                {k: v for k, v in item.items() if k != "kind"}
            )
            segment["stats"]["pc_car_loc_count"] += 1
        elif kind == "predict_hit_pos":
            segment["predict_hit_pos_events"].append(
                {k: v for k, v in item.items() if k != "kind"}
            )
            segment["stats"]["predict_hit_pos_count"] += 1
        elif kind == "time_sync_offset":
            segment["time_sync_offset_events"].append(
                {k: v for k, v in item.items() if k != "kind"}
            )
            segment["stats"]["time_sync_offset_count"] += 1

    def _has_payload(self, segment: dict) -> bool:
        total_items = (
            int(segment["stats"]["joint_state_count"])
            + int(segment["stats"]["mit_command_count"])
            + int(segment["stats"]["hit_event_count"])
            + int(segment["stats"]["control_event_count"])
            + int(segment["stats"]["pc_car_loc_count"])
            + int(segment["stats"]["predict_hit_pos_count"])
            + int(segment["stats"]["time_sync_offset_count"])
        )
        return total_items > 0

    def _compact_segment(
        self,
        segment: dict,
        *,
        reason: str,
        save_time_perf_ns: int,
    ) -> dict:
        joint_state_names: list[str] = []
        seen_joint_state_names: set[str] = set()
        for sample in segment.get("joint_states", []):
            for name in sample.get("name", []):
                joint_name = str(name)
                if joint_name in seen_joint_state_names:
                    continue
                seen_joint_state_names.add(joint_name)
                joint_state_names.append(joint_name)
        joint_state_names = sorted(joint_state_names, key=self._joint_sort_key)

        joint_state_matrix: list[list] = []
        for sample in segment.get("joint_states", []):
            name_to_index = {
                str(name): index for index, name in enumerate(sample.get("name", []))
            }
            positions = []
            velocities = []
            efforts = []
            raw_positions = list(sample.get("position", []))
            raw_velocities = list(sample.get("velocity", []))
            raw_efforts = list(sample.get("effort", []))
            for joint_name in joint_state_names:
                index = name_to_index.get(joint_name)
                positions.append(
                    None
                    if index is None or index >= len(raw_positions)
                    else float(raw_positions[index])
                )
                velocities.append(
                    None
                    if index is None or index >= len(raw_velocities)
                    else float(raw_velocities[index])
                )
                efforts.append(
                    None
                    if index is None or index >= len(raw_efforts)
                    else float(raw_efforts[index])
                )
            joint_state_matrix.append(
                [
                    _safe_int(sample.get("stamp_pc_ns")),
                    _safe_int(sample.get("receipt_stamp_pc_ns")),
                    positions,
                    velocities,
                    efforts,
                ]
            )

        mit_joint_names: list[str] = []
        seen_mit_joint_names: set[str] = set()
        for frame in segment.get("mit_command_frames", []):
            for command in frame.get("commands", []):
                joint_name = str(command.get("joint_name"))
                if joint_name in seen_mit_joint_names:
                    continue
                seen_mit_joint_names.add(joint_name)
                mit_joint_names.append(joint_name)
        mit_joint_names = sorted(mit_joint_names, key=self._joint_sort_key)

        mit_command_frames_matrix: list[list] = []
        for frame in segment.get("mit_command_frames", []):
            command_by_joint = {
                str(command.get("joint_name")): command
                for command in frame.get("commands", [])
            }
            commands_matrix = []
            for joint_name in mit_joint_names:
                command = command_by_joint.get(joint_name)
                if command is None:
                    commands_matrix.append(
                        [None, None, None, None, None, None, None, None]
                    )
                    continue
                commands_matrix.append(
                    [
                        int(command.get("motor_id", 0)),
                        None
                        if command.get("position_rad") is None
                        else float(command.get("position_rad")),
                        None
                        if command.get("velocity_rad_s") is None
                        else float(command.get("velocity_rad_s")),
                        None
                        if command.get("torque_ff_nm") is None
                        else float(command.get("torque_ff_nm")),
                        None
                        if command.get("computed_torque_ff_nm") is None
                        else float(command.get("computed_torque_ff_nm")),
                        None if command.get("kp") is None else float(command.get("kp")),
                        None if command.get("kd") is None else float(command.get("kd")),
                        bool(command.get("is_hold", False)),
                    ]
                )
            mit_command_frames_matrix.append(
                [
                    _safe_int(frame.get("stamp_pc_ns")),
                    int(frame.get("send_index", 0)),
                    None
                    if frame.get("request_id") is None
                    else str(frame.get("request_id")),
                    str(frame.get("sequence", "")),
                    str(frame.get("profile_mode", "")),
                    float(frame.get("execution_t_sec", 0.0)),
                    bool(frame.get("is_final", False)),
                    commands_matrix,
                ]
            )

        return {
            "schema": "pc_event_logger_v2",
            "saved_at_perf_s": float(save_time_perf_ns) / 1e9,
            "active_file": str(Path(segment["active_file"]).resolve()),
            "run_id": segment.get("run_id"),
            "group_id": segment.get("group_id"),
            "tracker_output_dir": segment.get("tracker_output_dir"),
            "tracker_json_path": segment.get("tracker_json_path"),
            "tracker_video_path": segment.get("tracker_video_path"),
            "segment_start_ns": segment.get("segment_start_ns"),
            "segment_end_ns": segment.get("segment_end_ns"),
            "last_save_reason": str(reason),
            "last_save_pc_ns": int(save_time_perf_ns),
            "config": dict(segment.get("config", {})),
            "topics": dict(segment.get("topics", {})),
            "joint_state_layout": {
                "joint_names": joint_state_names,
                "sample_fields": [
                    "stamp_pc_ns",
                    "receipt_stamp_pc_ns",
                    "position",
                    "velocity",
                    "effort",
                ],
            },
            "joint_states_matrix": joint_state_matrix,
            "mit_command_layout": {
                "joint_names": mit_joint_names,
                "frame_fields": [
                    "stamp_pc_ns",
                    "send_index",
                    "request_id",
                    "sequence",
                    "profile_mode",
                    "execution_t_sec",
                    "is_final",
                    "commands",
                ],
                "command_fields": [
                    "motor_id",
                    "position_rad",
                    "velocity_rad_s",
                    "torque_ff_nm",
                    "computed_torque_ff_nm",
                    "kp",
                    "kd",
                    "is_hold",
                ],
            },
            "mit_command_frames_matrix": mit_command_frames_matrix,
            "hit_events": list(segment.get("hit_events", [])),
            "control_events": list(segment.get("control_events", [])),
            "pc_car_loc_events": list(segment.get("pc_car_loc_events", [])),
            "predict_hit_pos_events": list(
                segment.get("predict_hit_pos_events", [])
            ),
            "time_sync_offset_events": list(
                segment.get("time_sync_offset_events", [])
            ),
            "stats": dict(segment.get("stats", {})),
        }

    def _write_active_log(
        self,
        segment: dict,
        path: Path,
        *,
        reason: str,
        save_time_perf_ns: int,
    ):
        if not self._has_payload(segment):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        compact_segment = self._compact_segment(
            segment,
            reason=reason,
            save_time_perf_ns=save_time_perf_ns,
        )
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(compact_segment, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)
        self.get_logger().info(
            "saved pc_event_logger to %s (reason=%s joint_states=%d mit=%d hit=%d control=%d car=%d predict=%d time_sync_offset=%d)"
            % (
                str(path),
                str(reason),
                int(segment["stats"]["joint_state_count"]),
                int(segment["stats"]["mit_command_count"]),
                int(segment["stats"]["hit_event_count"]),
                int(segment["stats"]["control_event_count"]),
                int(segment["stats"]["pc_car_loc_count"]),
                int(segment["stats"]["predict_hit_pos_count"]),
                int(segment["stats"]["time_sync_offset_count"]),
            )
        )

    def _worker_main(self):
        idle_period_ns = int(self.config.idle_record_period_sec * 1_000_000_000.0)
        high_rate_tail_ns = int(self.config.high_rate_tail_sec * 1_000_000_000.0)
        min_save_interval_ns = int(self.config.min_save_interval_sec * 1_000_000_000.0)
        post_hit_save_delay_ns = int(
            self.config.post_hit_save_delay_sec * 1_000_000_000.0
        )
        high_rate_until_ns = -1
        last_idle_record_ns = {"joint_state": None, "mit_command": None}
        active_path = self.config.target_path.resolve()
        metadata = self._metadata_from_config(path=active_path)
        segment = self._new_active_log(path=active_path, metadata=metadata)
        last_save_clock_ns: int | None = None
        pending_save_after_ns: int | None = None
        handled_command_ids: set[str] = set()

        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except Exception:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                break

            if item.get("kind") == "__tick__":
                now_ns = time.perf_counter_ns()
                if (
                    pending_save_after_ns is not None
                    and now_ns >= int(pending_save_after_ns)
                    and (
                        last_save_clock_ns is None
                        or (now_ns - int(last_save_clock_ns)) >= min_save_interval_ns
                    )
                ):
                    self._write_active_log(
                        segment,
                        active_path,
                        reason="scheduled_post_hit",
                        save_time_perf_ns=time.perf_counter_ns(),
                    )
                    last_save_clock_ns = now_ns
                    pending_save_after_ns = None
                continue

            kind = str(item.get("kind", ""))
            stamp_ns = _item_primary_stamp_ns(item, kind=kind)
            clock_ns = stamp_ns if stamp_ns is not None else time.perf_counter_ns()

            if kind == "control":
                command_id = item.get("command_id")
                if command_id is not None:
                    command_id = str(command_id)
                    if command_id in handled_command_ids:
                        continue
                    handled_command_ids.add(command_id)

                command = str(item.get("command", "")).strip().lower()
                if command == "new_file":
                    now_ns = clock_ns
                    had_payload_before_control = self._has_payload(segment)
                    next_path_text = item.get("target_path")
                    active_path = (
                        Path(next_path_text).resolve()
                        if next_path_text
                        else self._new_log_path().resolve()
                    )
                    metadata = self._metadata_from_control(item, path=active_path)
                    if had_payload_before_control:
                        self._write_active_log(
                            segment,
                            Path(segment["active_file"]),
                            reason="manual_new_file",
                            save_time_perf_ns=time.perf_counter_ns(),
                        )
                        last_save_clock_ns = now_ns
                    segment = self._new_active_log(path=active_path, metadata=metadata)
                    self._record_segment_item(segment, item)
                    pending_save_after_ns = None
                    last_idle_record_ns = {"joint_state": None, "mit_command": None}
                    high_rate_until_ns = -1
                    continue

                self._record_segment_item(segment, item)
                if command == "save_now":
                    self._write_active_log(
                        segment,
                        active_path,
                        reason="manual_save_now",
                        save_time_perf_ns=time.perf_counter_ns(),
                    )
                    last_save_clock_ns = clock_ns
                    pending_save_after_ns = None
                    continue

                if command == "shutdown":
                    self._write_active_log(
                        segment,
                        active_path,
                        reason="control_shutdown",
                        save_time_perf_ns=time.perf_counter_ns(),
                    )
                    last_save_clock_ns = clock_ns
                    pending_save_after_ns = None
                    continue

                continue

            if kind == "hit_event":
                scheduled_hit_time_ns = _item_scheduled_hit_pc_ns(item)
                if scheduled_hit_time_ns is None:
                    scheduled_hit_time_ns = clock_ns
                high_rate_until_ns = max(
                    high_rate_until_ns,
                    int(scheduled_hit_time_ns) + high_rate_tail_ns,
                )
                self._record_segment_item(segment, item)
                event_name = str(item.get("event", ""))
                if event_name in {
                    "execution_complete",
                    "ready_execution_complete",
                    "command_failed",
                    "execution_end",
                }:
                    target_save_ns = clock_ns + post_hit_save_delay_ns
                    pending_save_after_ns = (
                        target_save_ns
                        if pending_save_after_ns is None
                        else max(int(pending_save_after_ns), target_save_ns)
                    )
                continue

            if kind in {"pc_car_loc", "predict_hit_pos", "time_sync_offset"}:
                self._record_segment_item(segment, item)
                continue

            if clock_ns <= high_rate_until_ns:
                self._record_segment_item(segment, item)
                continue

            last_recorded = last_idle_record_ns.get(kind)
            if last_recorded is None or (clock_ns - int(last_recorded)) >= idle_period_ns:
                self._record_segment_item(segment, item)
                last_idle_record_ns[kind] = clock_ns

        self._write_active_log(
            segment,
            active_path,
            reason="shutdown",
            save_time_perf_ns=time.perf_counter_ns(),
        )


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Independent PC logger for arm ROS2 topics."
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        required=True,
        help="Path to the grouped pc logger JSON artifact.",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="Optional file created once the logger subscriptions are ready.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--group-id", type=str, default=None)
    parser.add_argument("--tracker-output-dir", type=Path, default=None)
    parser.add_argument("--tracker-json-path", type=Path, default=None)
    parser.add_argument("--tracker-video-path", type=Path, default=None)
    parser.add_argument("--idle-record-period-sec", type=float, default=1.0)
    parser.add_argument("--high-rate-tail-sec", type=float, default=1.0)
    parser.add_argument("--min-save-interval-sec", type=float, default=20.0)
    parser.add_argument("--post-hit-save-delay-sec", type=float, default=2.0)
    parser.add_argument("--tick-period-sec", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    config = PcEventLoggerConfig(
        target_path=args.target_path,
        ready_file=args.ready_file,
        run_id=args.run_id,
        group_id=args.group_id,
        tracker_output_dir=args.tracker_output_dir,
        tracker_json_path=args.tracker_json_path,
        tracker_video_path=args.tracker_video_path,
        idle_record_period_sec=float(args.idle_record_period_sec),
        high_rate_tail_sec=float(args.high_rate_tail_sec),
        min_save_interval_sec=float(args.min_save_interval_sec),
        post_hit_save_delay_sec=float(args.post_hit_save_delay_sec),
        tick_period_sec=float(args.tick_period_sec),
    )

    rclpy.init(args=None)
    node = PcEventLoggerNode(config=config)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.shutdown_requested:
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
