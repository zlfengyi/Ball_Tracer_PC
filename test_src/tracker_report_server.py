# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote, unquote, urlparse


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.arm_poe import ArmPoePositionModel
from test_src.generate_curve3_html import HTML_TEMPLATE, _merge_racket_json


DEFAULT_TRACKER_OUTPUT_DIR = _ROOT / "tracker_output"
DEFAULT_POE_CONFIG_PATH = _ROOT / "src" / "config" / "arm_poe_racket_center.json"
_RUN_RE = re.compile(r"^(tracker_\d{8}_\d{6})(.*)$")
_MAX_ARM_POINTS = 4000


@dataclass
class RunArtifacts:
    stem: str
    tracker_json: Path | None = None
    tracker_html: Path | None = None
    tracker_video: Path | None = None
    pc_logger_json: Path | None = None
    pc_logger_ready: Path | None = None
    extra_jsons: list[Path] = field(default_factory=list)
    extra_htmls: list[Path] = field(default_factory=list)
    extra_videos: list[Path] = field(default_factory=list)
    all_files: list[Path] = field(default_factory=list)

    def latest_mtime_ns(self) -> int:
        latest = 0
        for path in self.all_files:
            try:
                latest = max(latest, int(path.stat().st_mtime_ns))
            except FileNotFoundError:
                continue
        return latest

    def related_files(self) -> list[Path]:
        paths: list[Path] = []
        for item in (
            self.tracker_json,
            self.tracker_html,
            self.tracker_video,
            self.pc_logger_json,
            self.pc_logger_ready,
        ):
            if item is not None:
                paths.append(item)
        paths.extend(sorted(self.extra_jsons))
        paths.extend(sorted(self.extra_htmls))
        paths.extend(sorted(self.extra_videos))
        return paths


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _json_script(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _has_pc_timestamp_keys(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    for key in item.keys():
        if str(key).endswith("_pc_ns"):
            return True
    return False


def _field_index(fields: list[Any], name: str) -> int | None:
    for idx, field_name in enumerate(fields):
        if str(field_name) == str(name):
            return idx
    return None


def _row_field(row: Any, fields: list[Any], name: str) -> Any:
    if not isinstance(row, list):
        return None
    idx = _field_index(fields, name)
    if idx is None or idx >= len(row):
        return None
    return row[idx]


def _downsample_rows(rows: list[Any], limit: int = _MAX_ARM_POINTS) -> list[Any]:
    if len(rows) <= limit:
        return rows
    step = max(1, math.ceil(len(rows) / float(limit)))
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def _format_perf_time(perf_s: float | None) -> str:
    if perf_s is None:
        return "-"
    return f"{float(perf_s):.6f}s"


def _sec_to_ms(value: Any) -> float | None:
    sec = _safe_float(value)
    if sec is None:
        return None
    return float(sec) * 1000.0


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tracker_t0_perf_s(data: dict | None) -> float | None:
    if not isinstance(data, dict):
        return None
    cfg = data.get("config") or {}
    first_frame = _safe_float(cfg.get("first_frame_exposure_pc"))
    if first_frame is not None:
        return first_frame
    frames = data.get("frames") or []
    for frame in frames:
        if isinstance(frame, dict):
            exposure_pc = _safe_float(frame.get("exposure_pc"))
            if exposure_pc is not None:
                return exposure_pc
    for items, key in (
        (data.get("observations") or [], "t"),
        (data.get("racket_observations") or [], "t"),
        (data.get("car_locs") or [], "t"),
        (data.get("predictions") or [], "ct"),
    ):
        for item in items:
            if isinstance(item, dict):
                value = _safe_float(item.get(key))
                if value is not None:
                    return value
    return None


def _extract_tracker_racket_series(
    tracker_data: dict | None,
    *,
    tracker_t0_s: float | None,
) -> dict[str, list[float]]:
    result = {
        "t": [],
        "x": [],
        "y": [],
        "z": [],
    }
    if not isinstance(tracker_data, dict):
        return result
    cfg = tracker_data.get("config") or {}
    distance_scale = 1.0 if str(cfg.get("distance_unit", "m")) == "m" else 0.001
    racket_obs = tracker_data.get("racket_observations") or []
    if racket_obs:
        for item in racket_obs:
            if not isinstance(item, dict):
                continue
            t_s = _safe_float(item.get("t"))
            x = _safe_float(item.get("x"))
            y = _safe_float(item.get("y"))
            z = _safe_float(item.get("z"))
            if None in (t_s, x, y, z):
                continue
            rel_s = (float(t_s) - float(tracker_t0_s)) if tracker_t0_s is not None else 0.0
            result["t"].append(rel_s)
            result["x"].append(float(x) * distance_scale)
            result["y"].append(float(y) * distance_scale)
            result["z"].append(float(z) * distance_scale)
        return result

    frames = tracker_data.get("frames") or []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        racket3d = frame.get("racket3d")
        if not isinstance(racket3d, dict):
            continue
        exposure_pc = _safe_float(frame.get("exposure_pc"))
        x = _safe_float(racket3d.get("x"))
        y = _safe_float(racket3d.get("y"))
        z = _safe_float(racket3d.get("z"))
        if None in (exposure_pc, x, y, z):
            continue
        rel_s = (float(exposure_pc) - float(tracker_t0_s)) if tracker_t0_s is not None else 0.0
        result["t"].append(rel_s)
        result["x"].append(float(x) * distance_scale)
        result["y"].append(float(y) * distance_scale)
        result["z"].append(float(z) * distance_scale)
    return result


def _safe_child(root: Path, name: str) -> Path | None:
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _scan_runs(root: Path) -> list[RunArtifacts]:
    by_stem: dict[str, RunArtifacts] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = _RUN_RE.match(path.name)
        if match is None:
            continue
        stem = match.group(1)
        suffix = match.group(2)
        record = by_stem.setdefault(stem, RunArtifacts(stem=stem))
        record.all_files.append(path)
        if suffix == ".json":
            record.tracker_json = path
        elif suffix == ".html":
            record.tracker_html = path
        elif suffix == ".avi":
            record.tracker_video = path
        elif suffix == "_pc_logger.json":
            record.pc_logger_json = path
        elif suffix == "_pc_logger.ready":
            record.pc_logger_ready = path
        elif path.suffix.lower() == ".json":
            record.extra_jsons.append(path)
        elif path.suffix.lower() == ".html":
            record.extra_htmls.append(path)
        elif path.suffix.lower() in {".avi", ".mp4"}:
            record.extra_videos.append(path)
    runs = [run for run in by_stem.values() if run.tracker_json is not None or run.tracker_html is not None]
    return sorted(runs, key=lambda item: item.latest_mtime_ns(), reverse=True)


class TrackerReportServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        tracker_output_dir: Path,
        poe_config_path: Path,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.tracker_output_dir = Path(tracker_output_dir).resolve()
        self.poe_config_path = Path(poe_config_path).resolve()
        self._cache_lock = Lock()
        self._json_cache: dict[tuple[str, int, int], dict] = {}
        self._poe_model: ArmPoePositionModel | None = None
        self._poe_error: str | None = None
        try:
            self._poe_model = ArmPoePositionModel(config_path=self.poe_config_path)
        except Exception as exc:
            self._poe_error = str(exc)

    def file_signature(self, path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))

    def load_json(self, path: Path) -> dict:
        signature = self.file_signature(path)
        with self._cache_lock:
            cached = self._json_cache.get(signature)
        if cached is not None:
            return cached
        payload = _read_json(path)
        with self._cache_lock:
            self._json_cache = {signature: payload}
        return payload

    def list_runs(self) -> list[RunArtifacts]:
        return _scan_runs(self.tracker_output_dir)

    def get_run(self, stem: str) -> RunArtifacts | None:
        for run in self.list_runs():
            if run.stem == stem:
                return run
        return None

    def choose_tracker_payload(self, run: RunArtifacts) -> tuple[dict | None, str, list[Path]]:
        if run.tracker_json is None and run.extra_jsons:
            preferred = sorted(run.extra_jsons, key=lambda item: item.stat().st_mtime_ns, reverse=True)[0]
            return self.load_json(preferred), preferred.name, [preferred]

        base_payload = self.load_json(run.tracker_json) if run.tracker_json is not None else None

        exact_with_racket = None
        exact_racket = None
        variant_with_racket: list[Path] = []
        variant_racket: list[Path] = []
        for path in run.extra_jsons:
            if path.name == f"{run.stem}_with_racket.json":
                exact_with_racket = path
            elif path.name.endswith("_with_racket.json"):
                variant_with_racket.append(path)
            elif path.name == f"{run.stem}_racket.json":
                exact_racket = path
            elif path.name.endswith("_racket.json") and not path.name.endswith("_with_racket.json"):
                variant_racket.append(path)

        with_candidates = [
            item
            for item in [
                exact_with_racket,
                *sorted(variant_with_racket, key=lambda item: item.stat().st_mtime_ns, reverse=True),
            ]
            if item is not None
        ]
        if with_candidates:
            selected = with_candidates[0]
            return self.load_json(selected), selected.name, [selected]

        racket_candidates = [
            item
            for item in [
                exact_racket,
                *sorted(variant_racket, key=lambda item: item.stat().st_mtime_ns, reverse=True),
            ]
            if item is not None
        ]
        if base_payload is not None and racket_candidates:
            selected = racket_candidates[0]
            merged = _merge_racket_json(base_payload, self.load_json(selected), str(selected))
            return merged, f"{run.tracker_json.name} + {selected.name}", [run.tracker_json, selected]

        if base_payload is not None and run.tracker_json is not None:
            return base_payload, run.tracker_json.name, [run.tracker_json]
        return None, "no tracker json", []

    def build_arm_report(self, run: RunArtifacts) -> dict:
        tracker_payload, tracker_source_label, tracker_source_paths = self.choose_tracker_payload(run)
        tracker_t0_s = _tracker_t0_perf_s(tracker_payload)
        tracker_summary = (tracker_payload or {}).get("summary") or {}
        tracker_config = (tracker_payload or {}).get("config") or {}

        arm_summary = {
            "available": run.pc_logger_json is not None,
            "message": None,
            "poe_config_path": str(self.poe_config_path),
            "poe_available": self._poe_model is not None,
            "poe_error": self._poe_error,
            "segment_start_text": "-",
            "segment_end_text": "-",
        }
        report = {
            "stem": run.stem,
            "tracker_t0_perf_s": tracker_t0_s,
            "tracker_source_label": tracker_source_label,
            "tracker_source_files": [path.name for path in tracker_source_paths],
            "tracker_summary": {
                "actual_fps": tracker_summary.get("actual_fps"),
                "total_frames": tracker_summary.get("total_frames"),
                "observations_3d": tracker_summary.get("observations_3d"),
                "predictions": tracker_summary.get("predictions"),
                "car_locs": tracker_summary.get("car_locs"),
                "state_transitions": tracker_summary.get("state_transitions"),
                "latency_ms_avg": tracker_summary.get("latency_ms_avg"),
            },
            "tracker_start_text": _format_perf_time(tracker_t0_s),
            "tracker_distance_unit": tracker_config.get("distance_unit"),
            "arm_summary": arm_summary,
            "joint_actual": [],
            "joint_command": [],
            "joint_torque": [],
            "racket_actual": {"t": [], "x": [], "y": [], "z": []},
            "racket_command": {"t": [], "x": [], "y": [], "z": []},
            "racket_vision": _extract_tracker_racket_series(tracker_payload, tracker_t0_s=tracker_t0_s),
            "hit_events": [],
            "control_events": [],
            "time_sync_offsets": [],
            "time_sync_summary": {"available": False, "event_count": 0},
            "request_rows": [],
            "missing_pc_timestamps": [],
        }
        if run.pc_logger_json is None:
            arm_summary["message"] = "No *_pc_logger.json found for this tracker run."
            return report

        payload = self.load_json(run.pc_logger_json)
        stats = payload.get("stats") or {}
        joint_state_layout = payload.get("joint_state_layout") or {}
        mit_command_layout = payload.get("mit_command_layout") or {}
        joint_names = list(joint_state_layout.get("joint_names") or [])
        mit_joint_names = list(mit_command_layout.get("joint_names") or [])
        joint_sample_fields = list(
            joint_state_layout.get("sample_fields")
            or ["stamp_pc_ns", "receipt_stamp_pc_ns", "position", "velocity", "effort"]
        )
        mit_frame_fields = list(
            mit_command_layout.get("frame_fields")
            or [
                "stamp_ns",
                "send_index",
                "request_id",
                "sequence",
                "profile_mode",
                "execution_t_sec",
                "is_final",
                "commands",
            ]
        )
        mit_command_fields = list(
            mit_command_layout.get("command_fields")
            or [
                "motor_id",
                "position_rad",
                "velocity_rad_s",
                "torque_ff_nm",
                "computed_torque_ff_nm",
                "kp",
                "kd",
                "is_hold",
            ]
        )
        joint_rows = list(payload.get("joint_states_matrix") or [])
        mit_rows = list(payload.get("mit_command_frames_matrix") or [])
        hit_events = list(payload.get("hit_events") or [])
        control_events = list(payload.get("control_events") or [])
        time_sync_offset_events = list(payload.get("time_sync_offset_events") or [])

        strict_pc_timestamp_mode = (
            str(payload.get("schema", "")) == "pc_event_logger_v2"
            or "stamp_pc_ns" in {str(field) for field in joint_sample_fields}
            or "stamp_pc_ns" in {str(field) for field in mit_frame_fields}
            or any(_has_pc_timestamp_keys(item) for item in hit_events)
            or any(_has_pc_timestamp_keys(item) for item in control_events)
            or any(_has_pc_timestamp_keys(item) for item in time_sync_offset_events)
        )

        def ns_to_perf_s(stamp_ns: int | None) -> float | None:
            if stamp_ns is None:
                return None
            return float(stamp_ns) / 1e9

        def non_joint_stamp_ns(item: dict) -> int | None:
            stamp_pc_ns = _safe_int(item.get("stamp_pc_ns"))
            if stamp_pc_ns is not None:
                return stamp_pc_ns
            if not strict_pc_timestamp_mode:
                return _safe_int(item.get("stamp_ns"))
            return None

        def scheduled_hit_stamp_ns(item: dict) -> int | None:
            stamp_pc_ns = _safe_int(item.get("scheduled_hit_time_pc_ns"))
            if stamp_pc_ns is not None:
                return stamp_pc_ns
            if not strict_pc_timestamp_mode:
                return _safe_int(item.get("scheduled_hit_time_ns"))
            return None

        missing_pc_timestamps: list[dict[str, Any]] = []

        def add_missing_pc_entry(
            *,
            topic: str,
            expected_field: str,
            missing_count: int,
            total_count: int,
            used_field: str,
            note: str,
        ) -> None:
            missing_pc_timestamps.append(
                {
                    "topic": str(topic),
                    "expected_field": str(expected_field),
                    "used_field": str(used_field),
                    "missing_count": int(max(missing_count, 0)),
                    "total_count": int(max(total_count, 0)),
                    "note": str(note),
                }
            )

        effective_stamp_ns: list[int] = []
        arm_summary.update(
            {
                "schema": payload.get("schema"),
                "saved_at_perf_s": payload.get("saved_at_perf_s"),
                "last_save_reason": payload.get("last_save_reason"),
                "joint_state_count": stats.get("joint_state_count", 0),
                "mit_command_count": stats.get("mit_command_count", 0),
                "hit_event_count": stats.get("hit_event_count", 0),
                "control_event_count": stats.get("control_event_count", 0),
                "time_sync_offset_count": stats.get("time_sync_offset_count", 0),
                "joint_names": joint_names,
                "mit_joint_names": mit_joint_names,
                "run_id": payload.get("run_id"),
                "group_id": payload.get("group_id"),
                "strict_pc_timestamp_mode": strict_pc_timestamp_mode,
            }
        )

        actual_rows: list[dict[str, Any]] = []
        joint_stamp_field = (
            "stamp_pc_ns"
            if _field_index(joint_sample_fields, "stamp_pc_ns") is not None
            else "stamp_ns"
        )
        missing_joint_stamp_count = 0
        for row in joint_rows:
            if not isinstance(row, list):
                continue
            stamp_ns = _safe_int(_row_field(row, joint_sample_fields, joint_stamp_field))
            perf_s = ns_to_perf_s(stamp_ns)
            if stamp_ns is None:
                missing_joint_stamp_count += 1
            if stamp_ns is not None:
                effective_stamp_ns.append(stamp_ns)
            if perf_s is None:
                continue
            positions_raw = _row_field(row, joint_sample_fields, "position")
            velocities_raw = _row_field(row, joint_sample_fields, "velocity")
            efforts_raw = _row_field(row, joint_sample_fields, "effort")
            positions = list(positions_raw) if isinstance(positions_raw, list) else []
            velocities = list(velocities_raw) if isinstance(velocities_raw, list) else []
            efforts = list(efforts_raw) if isinstance(efforts_raw, list) else []
            rel_s = (perf_s - tracker_t0_s) if tracker_t0_s is not None else 0.0
            actual_rows.append(
                {
                    "perf_s": perf_s,
                    "rel_s": rel_s,
                    "positions": positions,
                    "velocities": velocities,
                    "efforts": efforts,
                }
            )
        add_missing_pc_entry(
            topic="joint_states",
            expected_field=joint_stamp_field,
            missing_count=missing_joint_stamp_count,
            total_count=len(joint_rows),
            used_field=joint_stamp_field,
            note=(
                "Uses JointState header.stamp mapped onto the PC perf axis as stamp_pc_ns; receipt_stamp_pc_ns is preserved only for transport diagnostics."
                if joint_stamp_field == "stamp_pc_ns"
                else "Legacy logger file detected; using stamp_ns until pc timestamp fields are present."
            ),
        )

        command_rows: list[dict[str, Any]] = []
        mit_stamp_field = "stamp_pc_ns" if strict_pc_timestamp_mode else (
            "stamp_pc_ns"
            if _field_index(mit_frame_fields, "stamp_pc_ns") is not None
            else "stamp_ns"
        )
        missing_mit_stamp_count = 0
        command_position_idx = _field_index(mit_command_fields, "position_rad")
        command_torque_idx = _field_index(mit_command_fields, "torque_ff_nm")
        command_computed_torque_idx = _field_index(
            mit_command_fields,
            "computed_torque_ff_nm",
        )
        for row in mit_rows:
            if not isinstance(row, list):
                continue
            stamp_ns = _safe_int(_row_field(row, mit_frame_fields, mit_stamp_field))
            if stamp_ns is None:
                missing_mit_stamp_count += 1
            else:
                effective_stamp_ns.append(stamp_ns)
            perf_s = ns_to_perf_s(stamp_ns)
            if perf_s is None:
                continue
            commands_raw = _row_field(row, mit_frame_fields, "commands")
            commands = commands_raw if isinstance(commands_raw, list) else []
            positions: list[float | None] = []
            torque_ff: list[float | None] = []
            computed_torque_ff: list[float | None] = []
            for command in commands:
                if not isinstance(command, list):
                    positions.append(None)
                    torque_ff.append(None)
                    computed_torque_ff.append(None)
                    continue
                positions.append(
                    _safe_float(command[command_position_idx])
                    if command_position_idx is not None and command_position_idx < len(command)
                    else None
                )
                torque_ff.append(
                    _safe_float(command[command_torque_idx])
                    if command_torque_idx is not None and command_torque_idx < len(command)
                    else None
                )
                computed_torque_ff.append(
                    _safe_float(command[command_computed_torque_idx])
                    if command_computed_torque_idx is not None
                    and command_computed_torque_idx < len(command)
                    else None
                )
            rel_s = (perf_s - tracker_t0_s) if tracker_t0_s is not None else 0.0
            command_rows.append(
                {
                    "perf_s": perf_s,
                    "rel_s": rel_s,
                    "positions": positions,
                    "torque_ff": torque_ff,
                    "computed_torque_ff": computed_torque_ff,
                    "request_id": _row_field(row, mit_frame_fields, "request_id"),
                    "sequence": _row_field(row, mit_frame_fields, "sequence"),
                    "profile_mode": _row_field(row, mit_frame_fields, "profile_mode"),
                    "is_final": bool(_row_field(row, mit_frame_fields, "is_final")),
                }
            )
        add_missing_pc_entry(
            topic="mit_command",
            expected_field="stamp_pc_ns",
            missing_count=missing_mit_stamp_count,
            total_count=len(mit_rows),
            used_field=mit_stamp_field,
            note=(
                "MIT frames without stamp_pc_ns are skipped in plots."
                if strict_pc_timestamp_mode
                else "Legacy logger file detected; using stamp_ns until renamed pc fields are present."
            ),
        )

        joint_actual_rows = _downsample_rows(actual_rows)
        joint_command_rows = _downsample_rows(command_rows)

        actual_by_joint: list[dict[str, Any]] = []
        for idx, joint_name in enumerate(joint_names):
            actual_by_joint.append(
                {
                    "name": str(joint_name),
                    "t": [float(item["rel_s"]) for item in joint_actual_rows if idx < len(item["positions"])],
                    "position": [
                        _safe_float(item["positions"][idx])
                        for item in joint_actual_rows
                        if idx < len(item["positions"])
                    ],
                    "velocity": [
                        _safe_float(item["velocities"][idx]) if idx < len(item["velocities"]) else None
                        for item in joint_actual_rows
                        if idx < len(item["positions"])
                    ],
                    "effort": [
                        _safe_float(item["efforts"][idx]) if idx < len(item["efforts"]) else None
                        for item in joint_actual_rows
                        if idx < len(item["positions"])
                    ],
                }
            )
        report["joint_actual"] = actual_by_joint

        command_by_joint: list[dict[str, Any]] = []
        for idx, joint_name in enumerate(mit_joint_names):
            command_by_joint.append(
                {
                    "name": str(joint_name),
                    "t": [float(item["rel_s"]) for item in joint_command_rows if idx < len(item["positions"])],
                    "position": [
                        _safe_float(item["positions"][idx])
                        for item in joint_command_rows
                        if idx < len(item["positions"])
                    ],
                }
            )
        report["joint_command"] = command_by_joint

        torque_by_joint: list[dict[str, Any]] = []
        for idx, joint_name in enumerate(joint_names):
            torque_by_joint.append(
                {
                    "name": str(joint_name),
                    "t_actual": [float(item["rel_s"]) for item in joint_actual_rows if idx < len(item["positions"])],
                    "joint_state_effort": [
                        _safe_float(item["efforts"][idx]) if idx < len(item["efforts"]) else None
                        for item in joint_actual_rows
                        if idx < len(item["positions"])
                    ],
                    "t_command": [float(item["rel_s"]) for item in joint_command_rows if idx < len(item["positions"])],
                    "torque_ff_nm": [
                        _safe_float(item["torque_ff"][idx]) if idx < len(item["torque_ff"]) else None
                        for item in joint_command_rows
                        if idx < len(item["positions"])
                    ],
                    "computed_torque_ff_nm": [
                        _safe_float(item["computed_torque_ff"][idx]) if idx < len(item["computed_torque_ff"]) else None
                        for item in joint_command_rows
                        if idx < len(item["positions"])
                    ],
                }
            )
        report["joint_torque"] = torque_by_joint

        if self._poe_model is not None:
            expected = list(self._poe_model.expected_joint_names)
            expected_count = len(expected)
            t_base_in_world_mm = self._poe_model.t_base_in_world_mm
            if joint_names[:expected_count] == expected:
                for item in joint_actual_rows:
                    positions = item["positions"]
                    if len(positions) < expected_count:
                        continue
                    try:
                        fk = self._poe_model.forward([float(value) for value in positions[:expected_count]])
                    except Exception:
                        continue
                    delta_world_from_base_mm = fk.point_world_mm - t_base_in_world_mm
                    report["racket_actual"]["t"].append(float(item["rel_s"]))
                    report["racket_actual"]["x"].append(float(delta_world_from_base_mm[0]) / 1000.0)
                    report["racket_actual"]["y"].append(float(delta_world_from_base_mm[1]) / 1000.0)
                    report["racket_actual"]["z"].append(float(delta_world_from_base_mm[2]) / 1000.0)

            if mit_joint_names[: len(expected)] == expected:
                for item in joint_command_rows:
                    positions = item["positions"]
                    if len(positions) < len(expected):
                        continue
                    if any(value is None for value in positions[: len(expected)]):
                        continue
                    try:
                        fk = self._poe_model.forward([float(value) for value in positions[: len(expected)]])
                    except Exception:
                        continue
                    delta_world_from_base_mm = fk.point_world_mm - t_base_in_world_mm
                    report["racket_command"]["t"].append(float(item["rel_s"]))
                    report["racket_command"]["x"].append(float(delta_world_from_base_mm[0]) / 1000.0)
                    report["racket_command"]["y"].append(float(delta_world_from_base_mm[1]) / 1000.0)
                    report["racket_command"]["z"].append(float(delta_world_from_base_mm[2]) / 1000.0)

        request_map: dict[str, dict[str, Any]] = {}
        missing_hit_stamp_count = 0
        missing_hit_scheduled_count = 0
        for item in hit_events:
            if not isinstance(item, dict):
                continue
            event_stamp_ns = non_joint_stamp_ns(item)
            scheduled_ns = scheduled_hit_stamp_ns(item)
            if event_stamp_ns is None:
                missing_hit_stamp_count += 1
            else:
                effective_stamp_ns.append(event_stamp_ns)
            if scheduled_ns is None:
                missing_hit_scheduled_count += 1
            perf_s = ns_to_perf_s(event_stamp_ns)
            scheduled_perf_s = ns_to_perf_s(scheduled_ns)
            rel_s = (perf_s - tracker_t0_s) if perf_s is not None and tracker_t0_s is not None else None
            scheduled_rel_s = (
                (scheduled_perf_s - tracker_t0_s)
                if scheduled_perf_s is not None and tracker_t0_s is not None
                else None
            )
            event_name = str(item.get("event", ""))
            request_id = str(item.get("request_id", "")) or "(none)"
            row = {
                "event": event_name,
                "request_id": request_id,
                "stamp_text": _format_perf_time(perf_s),
                "rel_s": rel_s,
                "scheduled_rel_s": scheduled_rel_s,
                "scheduled_hit_time_text": _format_perf_time(scheduled_perf_s),
                "hit_x_m": _safe_float(item.get("hit_x_m")),
                "hit_z_m": _safe_float(item.get("hit_z_m")),
                "mode": item.get("mode"),
                "source": item.get("source"),
                "detail": item.get("validation_detail"),
            }
            report["hit_events"].append(row)
            agg = request_map.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "source": item.get("source"),
                    "mode": item.get("mode"),
                    "hit_x_m": _safe_float(item.get("hit_x_m")),
                    "hit_z_m": _safe_float(item.get("hit_z_m")),
                    "events": [],
                    "first_rel_s": rel_s,
                    "scheduled_rel_s": scheduled_rel_s,
                },
            )
            agg["events"].append(event_name)
            if agg.get("first_rel_s") is None and rel_s is not None:
                agg["first_rel_s"] = rel_s
            if agg.get("scheduled_rel_s") is None and scheduled_rel_s is not None:
                agg["scheduled_rel_s"] = scheduled_rel_s
        add_missing_pc_entry(
            topic="hit_event",
            expected_field="stamp_pc_ns",
            missing_count=missing_hit_stamp_count,
            total_count=len(hit_events),
            used_field="stamp_pc_ns" if strict_pc_timestamp_mode else "stamp_pc_ns / stamp_ns",
            note="Hit events without stamp_pc_ns remain in tables but do not get a perf-counter timeline mapping.",
        )
        add_missing_pc_entry(
            topic="hit_event",
            expected_field="scheduled_hit_time_pc_ns",
            missing_count=missing_hit_scheduled_count,
            total_count=len(hit_events),
            used_field=(
                "scheduled_hit_time_pc_ns"
                if strict_pc_timestamp_mode
                else "scheduled_hit_time_pc_ns / scheduled_hit_time_ns"
            ),
            note="Missing scheduled_hit_time_pc_ns only affects the scheduled-hit marker and grouped scheduled time column.",
        )

        missing_control_stamp_count = 0
        for item in control_events:
            if not isinstance(item, dict):
                continue
            event_stamp_ns = non_joint_stamp_ns(item)
            if event_stamp_ns is None:
                missing_control_stamp_count += 1
            else:
                effective_stamp_ns.append(event_stamp_ns)
            perf_s = ns_to_perf_s(event_stamp_ns)
            rel_s = (perf_s - tracker_t0_s) if perf_s is not None and tracker_t0_s is not None else None
            report["control_events"].append(
                {
                    "command": item.get("command"),
                    "source": item.get("source"),
                    "reason": item.get("reason"),
                    "request_id": item.get("request_id"),
                    "stamp_text": _format_perf_time(perf_s),
                    "rel_s": rel_s,
                }
            )
        add_missing_pc_entry(
            topic="control",
            expected_field="stamp_pc_ns",
            missing_count=missing_control_stamp_count,
            total_count=len(control_events),
            used_field="stamp_pc_ns" if strict_pc_timestamp_mode else "stamp_pc_ns / stamp_ns",
            note="Logger control messages without stamp_pc_ns remain in the table but are not aligned onto the tracker timeline.",
        )

        missing_time_sync_stamp_count = 0
        for item in time_sync_offset_events:
            if not isinstance(item, dict):
                continue
            event_stamp_ns = _safe_int(item.get("stamp_pc_ns"))
            if event_stamp_ns is None:
                missing_time_sync_stamp_count += 1
            else:
                effective_stamp_ns.append(event_stamp_ns)
            perf_s = ns_to_perf_s(event_stamp_ns)
            rel_s = (perf_s - tracker_t0_s) if perf_s is not None and tracker_t0_s is not None else None
            report["time_sync_offsets"].append(
                {
                    "tag": item.get("tag"),
                    "source_id": item.get("source_id"),
                    "publish_reason": item.get("publish_reason"),
                    "clock_domain": item.get("clock_domain"),
                    "stamp_text": _format_perf_time(perf_s),
                    "perf_s": perf_s,
                    "rel_s": rel_s,
                    "report_period_sec": _safe_float(item.get("report_period_sec")),
                    "latest_seq": _safe_int(item.get("latest_seq")),
                    "offset_window_count": _safe_int(item.get("offset_window_count")),
                    "period_sample_count": _safe_int(item.get("period_sample_count")),
                    "period_accepted_count": _safe_int(item.get("period_accepted_count")),
                    "period_rejected_count": _safe_int(item.get("period_rejected_count")),
                    "total_sample_count": _safe_int(item.get("total_sample_count")),
                    "total_accepted_count": _safe_int(item.get("total_accepted_count")),
                    "total_rejected_count": _safe_int(item.get("total_rejected_count")),
                    "current_offset_ms": _sec_to_ms(item.get("current_offset_sec")),
                    "latest_accepted_offset_ms": _sec_to_ms(item.get("latest_accepted_offset_sec")),
                    "latest_offset_median_ms": _sec_to_ms(item.get("latest_offset_median_sec")),
                    "latest_rtt_ms": _sec_to_ms(item.get("latest_rtt_sec")),
                    "rtt_mean_ms": _sec_to_ms(item.get("rtt_mean_sec")),
                    "rtt_median_ms": _sec_to_ms(item.get("rtt_median_sec")),
                    "rtt_p95_ms": _sec_to_ms(item.get("rtt_p95_sec")),
                    "rtt_p99_ms": _sec_to_ms(item.get("rtt_p99_sec")),
                    "rtt_max_ms": _sec_to_ms(item.get("rtt_max_sec")),
                }
            )
        add_missing_pc_entry(
            topic="time_sync/offset",
            expected_field="stamp_pc_ns",
            missing_count=missing_time_sync_stamp_count,
            total_count=len(time_sync_offset_events),
            used_field="stamp_pc_ns",
            note="Time-sync offset snapshots without stamp_pc_ns remain in the table but are not aligned onto the tracker timeline.",
        )

        report["time_sync_offsets"] = sorted(
            report["time_sync_offsets"],
            key=lambda item: (
                float("inf") if item.get("perf_s") is None else float(item.get("perf_s")),
                str(item.get("tag", "")),
            ),
        )
        if report["time_sync_offsets"]:
            latest_time_sync = report["time_sync_offsets"][-1]
            report["time_sync_summary"] = {
                "available": True,
                "event_count": len(report["time_sync_offsets"]),
                "latest_tag": latest_time_sync.get("tag"),
                "latest_source_id": latest_time_sync.get("source_id"),
                "latest_publish_reason": latest_time_sync.get("publish_reason"),
                "latest_stamp_text": latest_time_sync.get("stamp_text"),
                "latest_rel_s": latest_time_sync.get("rel_s"),
                "latest_current_offset_ms": latest_time_sync.get("current_offset_ms"),
                "latest_accepted_offset_ms": latest_time_sync.get("latest_accepted_offset_ms"),
                "latest_offset_median_ms": latest_time_sync.get("latest_offset_median_ms"),
                "latest_rtt_ms": latest_time_sync.get("latest_rtt_ms"),
                "latest_rtt_p95_ms": latest_time_sync.get("rtt_p95_ms"),
                "latest_rtt_p99_ms": latest_time_sync.get("rtt_p99_ms"),
                "latest_period_accepted_count": latest_time_sync.get("period_accepted_count"),
                "latest_total_accepted_count": latest_time_sync.get("total_accepted_count"),
            }

        segment_start_ns = min(effective_stamp_ns) if effective_stamp_ns else _safe_int(payload.get("segment_start_ns"))
        segment_end_ns = max(effective_stamp_ns) if effective_stamp_ns else _safe_int(payload.get("segment_end_ns"))
        arm_summary.update(
            {
                "segment_start_text": _format_perf_time(ns_to_perf_s(segment_start_ns)),
                "segment_end_text": _format_perf_time(ns_to_perf_s(segment_end_ns)),
                "missing_pc_timestamp_rows": sum(
                    int(item.get("missing_count", 0))
                    for item in missing_pc_timestamps
                ),
            }
        )
        report["missing_pc_timestamps"] = missing_pc_timestamps

        report["request_rows"] = sorted(
            (
                {
                    **item,
                    "events_text": " -> ".join(item["events"]),
                }
                for item in request_map.values()
            ),
            key=lambda item: (
                float("inf") if item.get("first_rel_s") is None else float(item.get("first_rel_s")),
                str(item.get("request_id", "")),
            ),
        )
        return report


class TrackerReportHandler(BaseHTTPRequestHandler):
    server: TrackerReportServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/":
            self._serve_index()
            return
        if route.startswith("/run/"):
            self._serve_run(unquote(route.removeprefix("/run/")))
            return
        if route.startswith("/tracker-view/"):
            self._serve_tracker_view(unquote(route.removeprefix("/tracker-view/")))
            return
        if route.startswith("/artifact/"):
            self._serve_artifact(unquote(route.removeprefix("/artifact/")))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_html(self, body: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path, *, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 128)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _artifact_url(self, path: Path) -> str:
        return f"/artifact/{quote(path.name)}"

    def _run_rows_html(self, runs: list[RunArtifacts]) -> str:
        rows: list[str] = []
        for run in runs:
            files = run.related_files()
            file_links = " ".join(
                f'<a class="tag" href="{self._artifact_url(path)}">{html.escape(path.name)}</a>'
                for path in files[:8]
            )
            if len(files) > 8:
                file_links += f' <span class="muted">+{len(files) - 8} more</span>'
            summary_bits = []
            if run.pc_logger_json is not None:
                summary_bits.append("pc_logger")
            if any(path.name.endswith("_with_racket.json") for path in run.extra_jsons):
                summary_bits.append("with_racket")
            elif any(path.name.endswith("_racket.json") for path in run.extra_jsons):
                summary_bits.append("racket")
            if run.tracker_video is not None or run.extra_videos:
                summary_bits.append("video")
            badges = (
                " ".join(f'<span class="pill">{html.escape(bit)}</span>' for bit in summary_bits)
                or '<span class="muted">base tracker</span>'
            )
            rows.append(
                "<tr>"
                f'<td><a href="/run/{quote(run.stem)}"><strong>{html.escape(run.stem)}</strong></a><div class="row-meta">{badges}</div></td>'
                f"<td>{file_links}</td>"
                f"<td>{len(files)}</td>"
                "</tr>"
            )
        if not rows:
            return '<tr><td colspan="3">No tracker runs found.</td></tr>'
        return "".join(rows)

    def _serve_index(self) -> None:
        runs = self.server.list_runs()
        body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tracker Unified Report Server</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --paper: rgba(255,255,255,0.88);
      --ink: #1a2233;
      --muted: #61708a;
      --line: #d9d0c2;
      --accent: #bd4f2b;
      --accent-2: #1d6f8a;
      --pill: #f6dfcf;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", "Segoe UI", sans-serif; background:
      radial-gradient(circle at top left, rgba(29,111,138,0.14), transparent 28%),
      radial-gradient(circle at top right, rgba(189,79,43,0.18), transparent 24%),
      linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .page {{ max-width: 1440px; margin: 0 auto; padding: 28px 22px 48px; }}
    .hero {{ background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,247,240,0.92)); border: 1px solid rgba(217,208,194,0.95); border-radius: 26px; padding: 24px 26px; box-shadow: 0 20px 60px rgba(26,34,51,0.08); margin-bottom: 18px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 30px; }}
    .muted {{ color: var(--muted); }}
    .card {{ background: var(--paper); border: 1px solid rgba(217,208,194,0.95); border-radius: 22px; padding: 20px 22px; box-shadow: 0 16px 42px rgba(26,34,51,0.08); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 14px 10px; border-top: 1px solid rgba(217,208,194,0.95); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .tag {{ display: inline-block; background: rgba(255,255,255,0.7); border: 1px solid rgba(217,208,194,0.95); border-radius: 999px; padding: 4px 10px; margin: 0 6px 6px 0; font-size: 12px; }}
    .pill {{ display: inline-block; border-radius: 999px; background: var(--pill); color: #8f3d1f; padding: 3px 10px; margin-right: 6px; font-size: 12px; }}
    .row-meta {{ margin-top: 8px; }}
    code {{ background: rgba(255,255,255,0.78); border: 1px solid rgba(217,208,194,0.95); border-radius: 8px; padding: 1px 6px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Tracker Unified Report Server</h1>
      <p class="muted">选择一个 tracker run，页面会自动聚合同名的 base tracker JSON/HTML、pc logger JSON，以及后续标注生成的 <code>*_racket.json</code> / <code>*_with_racket.json</code> 等伴随文件。</p>
      <p class="muted">tracker_output: <code>{html.escape(str(self.server.tracker_output_dir))}</code></p>
      <p class="muted">POE config: <code>{html.escape(str(self.server.poe_config_path))}</code></p>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Run</th><th>Artifacts</th><th>Count</th></tr></thead>
        <tbody>{self._run_rows_html(runs)}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""
        self._send_html(body)

    def _serve_run(self, stem: str) -> None:
        run = self.server.get_run(stem)
        if run is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Tracker run not found")
            return

        arm_report = self.server.build_arm_report(run)
        file_rows = []
        for path in run.related_files():
            file_rows.append(
                "<tr>"
                f"<td>{html.escape(path.name)}</td>"
                f"<td>{html.escape(path.suffix.lower() or '(none)')}</td>"
                f'<td><a href="{self._artifact_url(path)}">open</a></td>'
                "</tr>"
            )
        files_table = "".join(file_rows) or '<tr><td colspan="3">No files.</td></tr>'

        body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(stem)} Unified Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>
    :root {{
      --bg: #111720;
      --paper: rgba(20,28,40,0.96);
      --paper-2: rgba(17,23,32,0.88);
      --ink: #f2f5f7;
      --muted: #90a0b8;
      --line: rgba(97,112,138,0.32);
      --accent: #ff8a5c;
      --accent-2: #5cd0ff;
      --accent-3: #a0ffb5;
      --warn: #ffc857;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", "Segoe UI", sans-serif; background:
      radial-gradient(circle at top left, rgba(92,208,255,0.12), transparent 22%),
      radial-gradient(circle at top right, rgba(255,138,92,0.16), transparent 18%),
      linear-gradient(180deg, #0f151e 0%, #111720 100%);
      color: var(--ink);
    }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .page {{ max-width: 1680px; margin: 0 auto; padding: 22px 22px 42px; }}
    .topbar {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 18px; }}
    .titlebox {{ background: linear-gradient(135deg, rgba(20,28,40,0.96), rgba(31,42,59,0.94)); border: 1px solid rgba(97,112,138,0.28); border-radius: 24px; padding: 20px 22px; flex: 1; box-shadow: 0 24px 60px rgba(0,0,0,0.28); }}
    .titlebox h1 {{ margin: 0 0 10px; font-size: 30px; letter-spacing: 0.02em; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-bottom: 18px; }}
    .card {{ background: linear-gradient(180deg, rgba(20,28,40,0.96), rgba(16,23,33,0.96)); border: 1px solid rgba(97,112,138,0.28); border-radius: 24px; padding: 18px 18px 20px; box-shadow: 0 18px 54px rgba(0,0,0,0.22); }}
    .card h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .statgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }}
    .stat {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(97,112,138,0.22); border-radius: 16px; padding: 12px 14px; }}
    .stat .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat .v {{ margin-top: 6px; font-size: 20px; font-weight: 700; }}
    iframe {{ width: 100%; height: 1470px; border: none; border-radius: 18px; background: #0d1118; }}
    .plot-shell {{ margin-top: 8px; }}
    .plot-toolbar {{ display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
    .plot-toolbar-label {{ color: var(--muted); font-size: 12px; }}
    .plot-zoom-btn {{ appearance: none; border: 1px solid rgba(97,112,138,0.30); background: rgba(255,255,255,0.04); color: var(--ink); border-radius: 999px; padding: 5px 12px; font: inherit; font-size: 12px; cursor: pointer; transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease; }}
    .plot-zoom-btn:hover {{ background: rgba(255,255,255,0.08); border-color: rgba(92,208,255,0.44); transform: translateY(-1px); }}
    .plot-zoom-btn.on {{ background: rgba(92,208,255,0.16); border-color: rgba(92,208,255,0.64); color: #c7f2ff; }}
    .plot-scale-readout {{ color: var(--muted); font-size: 12px; min-width: 44px; text-align: right; }}
    .trace-controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; }}
    .trace-toggle {{ appearance: none; display: inline-flex; align-items: center; gap: 8px; border: 1px solid rgba(97,112,138,0.30); background: rgba(255,255,255,0.04); color: var(--ink); border-radius: 999px; padding: 5px 12px; font: inherit; font-size: 12px; cursor: pointer; transition: background 0.18s ease, border-color 0.18s ease, opacity 0.18s ease, transform 0.18s ease; }}
    .trace-toggle:hover {{ background: rgba(255,255,255,0.08); border-color: rgba(255,138,92,0.48); transform: translateY(-1px); }}
    .trace-toggle.off {{ opacity: 0.48; }}
    .trace-swatch {{ width: 10px; height: 10px; border-radius: 999px; flex: 0 0 10px; box-shadow: 0 0 0 1px rgba(255,255,255,0.15); }}
    .plot-scroll {{ overflow: hidden; padding-bottom: 6px; border-radius: 18px; transition: box-shadow 0.18s ease, outline-color 0.18s ease; }}
    .plot-shell.active .plot-scroll {{ box-shadow: 0 0 0 1px rgba(92,208,255,0.55), 0 0 0 4px rgba(92,208,255,0.10); }}
    .plot {{ width: 100%; min-width: 100%; min-height: 780px; }}
    .section-title {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
    .tiny {{ font-size: 12px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-top: 1px solid rgba(97,112,138,0.22); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .subgrid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .empty {{ color: var(--muted); padding: 8px 0 4px; }}
    code {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(97,112,138,0.22); border-radius: 8px; padding: 1px 6px; }}
    @media (max-width: 1200px) {{
      .subgrid {{ grid-template-columns: 1fr; }}
      iframe {{ height: 1140px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div class="titlebox">
        <div class="tiny"><a href="/">Back to index</a></div>
        <h1>{html.escape(stem)}</h1>
        <div class="muted">上方继续显示 tracker / curve3 主视图；下方叠加 arm logger 时间轴、POE racket center、MIT command 与 hit event。</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="section-title">
          <h2>Run Summary</h2>
          <div class="tiny">tracker t0 perf: <code>{html.escape(str(arm_report["tracker_start_text"]))}</code></div>
        </div>
        <div id="summary-grid" class="statgrid"></div>
        <div style="margin-top:16px" class="tiny">POE config: <code>{html.escape(str(self.server.poe_config_path))}</code></div>
      </div>
      <div class="card">
        <div class="section-title">
          <h2>Missing PC Timestamps</h2>
          <div class="tiny">missing records stay in raw tables, but are not forced onto the tracker timeline</div>
        </div>
        <div id="missing-pc-table"></div>
      </div>
      <div class="card">
        <div class="section-title">
          <h2>Tracker Viewer</h2>
          <div class="tiny">source: <code>{html.escape(arm_report["tracker_source_label"])}</code></div>
        </div>
        <iframe src="/tracker-view/{quote(stem)}" loading="eager"></iframe>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="section-title">
        <h2>Joint Position Timeline</h2>
        <div class="tiny">points only: actual = circle | MIT command = diamond | vertical markers: hit / scheduled hit</div>
      </div>
      <div class="plot-shell">
        <div id="joint-trace-controls" class="trace-controls"></div>
        <div class="plot-toolbar">
          <span class="plot-toolbar-label">X zoom / click plot + wheel</span>
          <button type="button" class="plot-zoom-btn" data-plot="joint-plot" data-action="out">X-</button>
          <button type="button" class="plot-zoom-btn on" data-plot="joint-plot" data-action="reset">Reset</button>
          <button type="button" class="plot-zoom-btn" data-plot="joint-plot" data-action="in">X+</button>
          <span id="joint-plot-scale-readout" class="plot-scale-readout">1.00x</span>
        </div>
        <div class="plot-scroll">
          <div id="joint-plot" class="plot"></div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="section-title">
        <h2>Joint Torque Timeline</h2>
        <div class="tiny">points only: effort = circle | torque_ff = diamond | computed_torque = x</div>
      </div>
      <div class="plot-shell">
        <div id="torque-trace-controls" class="trace-controls"></div>
        <div class="plot-toolbar">
          <span class="plot-toolbar-label">X zoom / click plot + wheel</span>
          <button type="button" class="plot-zoom-btn" data-plot="torque-plot" data-action="out">X-</button>
          <button type="button" class="plot-zoom-btn on" data-plot="torque-plot" data-action="reset">Reset</button>
          <button type="button" class="plot-zoom-btn" data-plot="torque-plot" data-action="in">X+</button>
          <span id="torque-plot-scale-readout" class="plot-scale-readout">1.00x</span>
        </div>
        <div class="plot-scroll">
          <div id="torque-plot" class="plot"></div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="section-title">
        <h2>Racket Center Timeline</h2>
        <div class="tiny">points only: POE actual = circle | POE cmd = diamond | vision racket = small dot</div>
      </div>
      <div class="plot-shell">
        <div id="racket-trace-controls" class="trace-controls"></div>
        <div class="plot-toolbar">
          <span class="plot-toolbar-label">X zoom / click plot + wheel</span>
          <button type="button" class="plot-zoom-btn" data-plot="racket-plot" data-action="out">X-</button>
          <button type="button" class="plot-zoom-btn on" data-plot="racket-plot" data-action="reset">Reset</button>
          <button type="button" class="plot-zoom-btn" data-plot="racket-plot" data-action="in">X+</button>
          <span id="racket-plot-scale-readout" class="plot-scale-readout">1.00x</span>
        </div>
        <div class="plot-scroll">
          <div id="racket-plot" class="plot"></div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="section-title">
        <h2>Time Sync Offset / RTT</h2>
        <div class="tiny">source topic: <code>/time_sync/offset</code> from arm_controller bundled sync worker</div>
      </div>
      <div class="plot-shell">
        <div id="time-sync-trace-controls" class="trace-controls"></div>
        <div class="plot-toolbar">
          <span class="plot-toolbar-label">X zoom / click plot + wheel</span>
          <button type="button" class="plot-zoom-btn" data-plot="time-sync-plot" data-action="out">X-</button>
          <button type="button" class="plot-zoom-btn on" data-plot="time-sync-plot" data-action="reset">Reset</button>
          <button type="button" class="plot-zoom-btn" data-plot="time-sync-plot" data-action="in">X+</button>
          <span id="time-sync-plot-scale-readout" class="plot-scale-readout">1.00x</span>
        </div>
        <div class="plot-scroll">
          <div id="time-sync-plot" class="plot"></div>
        </div>
      </div>
      <div class="section-title" style="margin-top:18px">
        <h2>Time Sync Offset Snapshots</h2>
        <div class="tiny">latest offset / RTT statistics reported by arm_controller</div>
      </div>
      <div id="time-sync-table"></div>
    </div>

    <div class="subgrid">
      <div class="card">
        <div class="section-title">
          <h2>Hit Request Summary</h2>
          <div class="tiny">grouped by request_id</div>
        </div>
        <div id="request-table"></div>
      </div>
      <div class="card">
        <div class="section-title">
          <h2>Artifacts</h2>
          <div class="tiny">{len(run.related_files())} files</div>
        </div>
        <table>
          <thead><tr><th>Name</th><th>Type</th><th>Open</th></tr></thead>
          <tbody>{files_table}</tbody>
        </table>
      </div>
    </div>

    <div class="subgrid" style="margin-top:18px">
      <div class="card">
        <div class="section-title">
          <h2>Hit Events</h2>
          <div class="tiny">raw arm logger events</div>
        </div>
        <div id="hit-table"></div>
      </div>
      <div class="card">
        <div class="section-title">
          <h2>Logger Control Events</h2>
          <div class="tiny">tracker -> pc_logger control channel</div>
        </div>
        <div id="control-table"></div>
      </div>
    </div>
  </div>

  <script>
    const ARM = {_json_script(arm_report)};

    const fmtNum = (value, digits = 3) => (typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-');
    const fmtTime = value => (typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) + ' s' : '-');
    const colors = ['#5cd0ff', '#ff8a5c', '#a0ffb5', '#ffc857', '#d7a8ff', '#f472b6'];

    const X_ZOOM_STEP = 1.35;
    const X_ZOOM_MAX = 200.0;
    let activePlotId = null;
    const plotZoomState = {{}};

    function statCard(key, value) {{
      return `<div class="stat"><div class="k">${{key}}</div><div class="v">${{value}}</div></div>`;
    }}

    function getPlotElement(plotId) {{
      return document.getElementById(plotId);
    }}

    function setActivePlot(plotId) {{
      activePlotId = plotId;
      document.querySelectorAll('.plot-shell').forEach(shell => {{
        const plot = shell.querySelector('.plot');
        shell.classList.toggle('active', !!plot && plot.id === plotId);
      }});
    }}

    function collectNumericXValues(plot) {{
      const values = [];
      (plot?.data || []).forEach(trace => {{
        (trace?.x || []).forEach(value => {{
          if (typeof value === 'number' && Number.isFinite(value)) {{
            values.push(value);
          }}
        }});
      }});
      return values;
    }}

    function getFullXRange(plotId) {{
      const cached = plotZoomState[plotId]?.fullRange;
      if (cached) {{
        return [...cached];
      }}
      const plot = getPlotElement(plotId);
      if (!plot) {{
        return null;
      }}
      let range = null;
      const axis = plot._fullLayout && plot._fullLayout.xaxis;
      if (axis && Array.isArray(axis.range) && axis.range.length === 2) {{
        const a = Number(axis.range[0]);
        const b = Number(axis.range[1]);
        if (Number.isFinite(a) && Number.isFinite(b) && b > a) {{
          range = [a, b];
        }}
      }}
      if (!range) {{
        const xs = collectNumericXValues(plot);
        if (!xs.length) {{
          return null;
        }}
        range = [Math.min(...xs), Math.max(...xs)];
      }}
      plotZoomState[plotId] = {{
        ...(plotZoomState[plotId] || {{}}),
        fullRange: range,
      }};
      return [...range];
    }}

    function getCurrentXRange(plotId) {{
      const plot = getPlotElement(plotId);
      if (!plot) {{
        return null;
      }}
      const axis = plot._fullLayout && plot._fullLayout.xaxis;
      if (axis && Array.isArray(axis.range) && axis.range.length === 2) {{
        const a = Number(axis.range[0]);
        const b = Number(axis.range[1]);
        if (Number.isFinite(a) && Number.isFinite(b) && b > a) {{
          return [a, b];
        }}
      }}
      return getFullXRange(plotId);
    }}

    function clampXRange(range, fullRange) {{
      const fullMin = fullRange[0];
      const fullMax = fullRange[1];
      const fullSpan = fullMax - fullMin;
      if (!(fullSpan > 0)) {{
        return [fullMin, fullMax];
      }}
      let start = Number(range[0]);
      let end = Number(range[1]);
      let span = end - start;
      const minSpan = Math.max(fullSpan / X_ZOOM_MAX, 1e-6);
      if (!(span > 0)) {{
        span = minSpan;
      }}
      span = Math.max(minSpan, Math.min(fullSpan, span));
      const center = (start + end) / 2;
      start = center - span / 2;
      end = center + span / 2;
      if (start < fullMin) {{
        end += fullMin - start;
        start = fullMin;
      }}
      if (end > fullMax) {{
        start -= end - fullMax;
        end = fullMax;
      }}
      if (start < fullMin) {{
        start = fullMin;
      }}
      if (end > fullMax) {{
        end = fullMax;
      }}
      return [start, end];
    }}

    function updateXZoomUi(plotId) {{
      const fullRange = getFullXRange(plotId);
      const currentRange = getCurrentXRange(plotId);
      const readout = document.getElementById(`${{plotId}}-scale-readout`);
      if (!fullRange || !currentRange) {{
        if (readout) {{
          readout.textContent = 'n/a';
        }}
        return;
      }}
      const fullSpan = fullRange[1] - fullRange[0];
      const currentSpan = Math.max(1e-9, currentRange[1] - currentRange[0]);
      const zoomFactor = Math.max(1.0, fullSpan / currentSpan);
      if (readout) {{
        readout.textContent = `${{zoomFactor.toFixed(2)}}x`;
      }}
      document.querySelectorAll(`.plot-zoom-btn[data-plot="${{plotId}}"]`).forEach(btn => {{
        btn.classList.toggle('on', btn.dataset.action === 'reset' && Math.abs(zoomFactor - 1.0) < 1e-3);
      }});
    }}

    function relayoutXRange(plotId, range) {{
      return Plotly.relayout(plotId, {{
        'xaxis.range': range,
        'xaxis.autorange': false,
      }}).then(() => updateXZoomUi(plotId));
    }}

    function getMouseXValue(plotId, event) {{
      const plot = getPlotElement(plotId);
      const axis = plot?._fullLayout?.xaxis;
      const currentRange = getCurrentXRange(plotId);
      if (!plot || !axis || !currentRange) {{
        return null;
      }}
      const rect = plot.getBoundingClientRect();
      const axisOffset = Number(axis._offset);
      const axisLength = Number(axis._length);
      if (!Number.isFinite(axisOffset) || !Number.isFinite(axisLength) || !(axisLength > 0)) {{
        return null;
      }}
      const rawPixel = Number(event.clientX) - rect.left - axisOffset;
      const pixel = Math.max(0, Math.min(axisLength, rawPixel));
      if (typeof axis.p2l === 'function') {{
        const converted = Number(axis.p2l(pixel));
        if (Number.isFinite(converted)) {{
          return converted;
        }}
      }}
      const ratio = pixel / axisLength;
      return currentRange[0] + ratio * (currentRange[1] - currentRange[0]);
    }}

    function zoomPlotX(plotId, spanFactor, centerX = null) {{
      const fullRange = getFullXRange(plotId);
      const currentRange = getCurrentXRange(plotId);
      if (!fullRange || !currentRange) {{
        return Promise.resolve();
      }}
      const center = (
        typeof centerX === 'number' && Number.isFinite(centerX)
          ? centerX
          : (currentRange[0] + currentRange[1]) / 2
      );
      const nextRange = clampXRange(
        [center - ((currentRange[1] - currentRange[0]) * spanFactor) / 2, center + ((currentRange[1] - currentRange[0]) * spanFactor) / 2],
        fullRange,
      );
      return relayoutXRange(plotId, nextRange);
    }}

    function resetPlotX(plotId) {{
      const fullRange = getFullXRange(plotId);
      if (!fullRange) {{
        return Promise.resolve();
      }}
      return relayoutXRange(plotId, fullRange);
    }}

    function wireXZoom(plotId) {{
      const plot = getPlotElement(plotId);
      if (!plot) {{
        return;
      }}
      getFullXRange(plotId);
      updateXZoomUi(plotId);
      plot.addEventListener('pointerdown', () => setActivePlot(plotId));
      plot.addEventListener(
        'wheel',
        event => {{
          if (activePlotId !== plotId) {{
            return;
          }}
          if (event.ctrlKey || event.metaKey) {{
            return;
          }}
          event.preventDefault();
          const spanFactor = event.deltaY < 0 ? (1 / X_ZOOM_STEP) : X_ZOOM_STEP;
          zoomPlotX(plotId, spanFactor, getMouseXValue(plotId, event));
        }},
        {{ passive: false }},
      );
      plot.on('plotly_relayout', () => updateXZoomUi(plotId));
    }}

    function initPlotZoomButtons() {{
      document.querySelectorAll('.plot-zoom-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const plotId = btn.dataset.plot;
          setActivePlot(plotId);
          const action = btn.dataset.action;
          if (action === 'in') {{
            zoomPlotX(plotId, 1 / X_ZOOM_STEP);
          }} else if (action === 'out') {{
            zoomPlotX(plotId, X_ZOOM_STEP);
          }} else {{
            resetPlotX(plotId);
          }}
        }});
      }});
    }}

    function isTraceVisible(trace) {{
      return trace && trace.visible !== 'legendonly' && trace.visible !== false;
    }}

    function traceColor(trace) {{
      if (!trace) return '#c7d2e1';
      if (trace.line && typeof trace.line.color === 'string') return trace.line.color;
      if (trace.marker && typeof trace.marker.color === 'string') return trace.marker.color;
      return '#c7d2e1';
    }}

    function renderTraceControls(plotId, controlsId) {{
      const plot = document.getElementById(plotId);
      const controls = document.getElementById(controlsId);
      if (!plot || !controls || !plot.data) {{
        return;
      }}
      const buttons = plot.data.map((trace, idx) => {{
        const visible = isTraceVisible(trace);
        const name = trace && trace.name ? trace.name : `trace ${{idx + 1}}`;
        const swatch = traceColor(trace);
        return `<button type="button" class="trace-toggle${{visible ? '' : ' off'}}" data-plot="${{plotId}}" data-index="${{idx}}" aria-pressed="${{visible ? 'true' : 'false'}}"><span class="trace-swatch" style="background:${{swatch}}"></span><span>${{escapeHtml(name)}}</span></button>`;
      }});
      controls.innerHTML = buttons.join('');
      controls.querySelectorAll('.trace-toggle').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const index = Number(btn.dataset.index);
          const trace = plot.data[index];
          const nextVisible = !isTraceVisible(trace);
          Plotly.restyle(plotId, {{visible: nextVisible ? true : 'legendonly'}}, [index]).then(() => {{
            renderTraceControls(plotId, controlsId);
          }});
        }});
      }});
    }}

    function wireTraceControls(plotId, controlsId) {{
      const plot = document.getElementById(plotId);
      if (!plot) {{
        return;
      }}
      renderTraceControls(plotId, controlsId);
      plot.on('plotly_restyle', () => renderTraceControls(plotId, controlsId));
    }}

    function renderSummary() {{
      const arm = ARM.arm_summary || {{}};
      const tracker = ARM.tracker_summary || {{}};
      const sync = ARM.time_sync_summary || {{}};
      const cards = [
        statCard('Tracker FPS', fmtNum(tracker.actual_fps, 1)),
        statCard('Tracker Frames', tracker.total_frames ?? '-'),
        statCard('Ball 3D', tracker.observations_3d ?? '-'),
        statCard('Predictions', tracker.predictions ?? '-'),
        statCard('Joint States', arm.joint_state_count ?? '-'),
        statCard('MIT Frames', arm.mit_command_count ?? '-'),
        statCard('Hit Events', arm.hit_event_count ?? '-'),
        statCard('Control Events', arm.control_event_count ?? '-'),
        statCard('Missing PC Fields', arm.missing_pc_timestamp_rows ?? 0),
        statCard('Logger Start Perf', arm.segment_start_text ?? '-'),
        statCard('Logger End Perf', arm.segment_end_text ?? '-'),
        statCard('POE', arm.poe_available ? 'ready' : 'unavailable'),
        statCard('Tracker Source', ARM.tracker_source_label || '-'),
      ];
      if (sync.available) {{
        cards.push(statCard('Sync Reports', sync.event_count ?? '-'));
        if (typeof sync.latest_current_offset_ms === 'number' && Number.isFinite(sync.latest_current_offset_ms)) {{
          cards.push(statCard('Sync Offset', fmtNum(sync.latest_current_offset_ms, 3) + ' ms'));
        }}
        if (typeof sync.latest_rtt_p95_ms === 'number' && Number.isFinite(sync.latest_rtt_p95_ms)) {{
          cards.push(statCard('Sync RTT P95', fmtNum(sync.latest_rtt_p95_ms, 3) + ' ms'));
        }} else if (typeof sync.latest_rtt_ms === 'number' && Number.isFinite(sync.latest_rtt_ms)) {{
          cards.push(statCard('Sync RTT', fmtNum(sync.latest_rtt_ms, 3) + ' ms'));
        }}
      }}
      if (!arm.available && arm.message) {{
        cards.push(statCard('Arm Status', arm.message));
      }}
      document.getElementById('summary-grid').innerHTML = cards.join('');
    }}

    function buildEventShapes() {{
      const shapes = [];
      for (const ev of ARM.hit_events || []) {{
        if (typeof ev.rel_s === 'number' && Number.isFinite(ev.rel_s)) {{
          shapes.push({{
            type: 'line',
            x0: ev.rel_s,
            x1: ev.rel_s,
            yref: 'paper',
            y0: 0,
            y1: 1,
            line: {{ color: 'rgba(255,138,92,0.55)', width: 1.2, dash: 'dash' }},
          }});
        }}
        if (typeof ev.scheduled_rel_s === 'number' && Number.isFinite(ev.scheduled_rel_s)) {{
          shapes.push({{
            type: 'line',
            x0: ev.scheduled_rel_s,
            x1: ev.scheduled_rel_s,
            yref: 'paper',
            y0: 0,
            y1: 1,
            line: {{ color: 'rgba(92,208,255,0.42)', width: 1.0, dash: 'dot' }},
          }});
        }}
      }}
      return shapes;
    }}

    function baseLayout(title) {{
      return {{
        title: {{ text: title, font: {{ color: '#f2f5f7', size: 16 }} }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(255,255,255,0.03)',
        font: {{ color: '#dfe7f2', size: 12 }},
        legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0 }},
        margin: {{ l: 56, r: 28, t: 54, b: 48 }},
        xaxis: {{ title: 't - tracker_t0 (s)', gridcolor: 'rgba(97,112,138,0.16)', zerolinecolor: 'rgba(97,112,138,0.16)' }},
        yaxis: {{ gridcolor: 'rgba(97,112,138,0.16)', zerolinecolor: 'rgba(97,112,138,0.16)' }},
        shapes: buildEventShapes(),
      }};
    }}

    function renderJointPlot() {{
      const traces = [];
      (ARM.joint_actual || []).forEach((series, idx) => {{
        if (!(series.t || []).length) {{
          return;
        }}
        traces.push({{
          x: series.t || [],
          y: series.position || [],
          mode: 'markers',
          name: `${{series.name}} actual`,
          marker: {{ color: colors[idx % colors.length], size: 5, symbol: 'circle' }},
        }});
      }});
      (ARM.joint_command || []).forEach((series, idx) => {{
        if (!(series.t || []).length) {{
          return;
        }}
        traces.push({{
          x: series.t || [],
          y: series.position || [],
          mode: 'markers',
          name: `${{series.name}} cmd`,
          marker: {{ color: colors[idx % colors.length], size: 6, symbol: 'diamond-open' }},
          opacity: 0.8,
        }});
      }});
      if (!traces.length) {{
        document.getElementById('joint-plot').innerHTML = '<div class="empty">No joint or MIT command data available for this run.</div>';
        return;
      }}
      Plotly.newPlot('joint-plot', traces, baseLayout('Joint Position / Command'), {{responsive: true, displaylogo: false}})
        .then(() => {{
          wireTraceControls('joint-plot', 'joint-trace-controls');
          wireXZoom('joint-plot');
        }});
    }}

    function renderTorquePlot() {{
      const traces = [];
      (ARM.joint_torque || []).forEach((series, idx) => {{
        const color = colors[idx % colors.length];
        const effort = series.joint_state_effort || [];
        const torqueFf = series.torque_ff_nm || [];
        const computed = series.computed_torque_ff_nm || [];
        if (effort.some(v => typeof v === 'number' && Number.isFinite(v))) {{
          traces.push({{
            x: series.t_actual || [],
            y: effort,
            mode: 'markers',
            name: `${{series.name}} joint_state effort`,
            marker: {{ color, size: 5, symbol: 'circle' }},
          }});
        }}
        if (torqueFf.some(v => typeof v === 'number' && Number.isFinite(v))) {{
          traces.push({{
            x: series.t_command || [],
            y: torqueFf,
            mode: 'markers',
            name: `${{series.name}} mit torque_ff`,
            marker: {{ color, size: 6, symbol: 'diamond-open' }},
            opacity: 0.88,
          }});
        }}
        if (computed.some(v => typeof v === 'number' && Number.isFinite(v))) {{
          traces.push({{
            x: series.t_command || [],
            y: computed,
            mode: 'markers',
            name: `${{series.name}} mit computed_torque`,
            marker: {{ color, size: 7, symbol: 'x' }},
            opacity: 0.78,
          }});
        }}
      }});
      if (!traces.length) {{
        document.getElementById('torque-plot').innerHTML = '<div class="empty">No joint torque data available for this run.</div>';
        return;
      }}
      const layout = baseLayout('Joint Torque / Effort');
      layout.yaxis.title = 'torque (Nm)';
      Plotly.newPlot('torque-plot', traces, layout, {{responsive: true, displaylogo: false}})
        .then(() => {{
          wireTraceControls('torque-plot', 'torque-trace-controls');
          wireXZoom('torque-plot');
        }});
    }}

    function renderRacketPlot() {{
      const traces = [];
      const actual = ARM.racket_actual || {{}};
      const command = ARM.racket_command || {{}};
      const vision = ARM.racket_vision || {{}};
      [['x', '#5cd0ff'], ['y', '#a0ffb5'], ['z', '#ff8a5c']].forEach(([axis, color]) => {{
        if ((actual.t || []).length) {{
          traces.push({{
            x: actual.t || [],
            y: actual[axis] || [],
            mode: 'markers',
            name: `poe actual ${{axis}}`,
            marker: {{ color, size: 5, symbol: 'circle' }},
          }});
        }}
        if ((command.t || []).length) {{
          traces.push({{
            x: command.t || [],
            y: command[axis] || [],
            mode: 'markers',
            name: `poe cmd ${{axis}}`,
            marker: {{ color, size: 6, symbol: 'diamond-open' }},
            opacity: 0.82,
          }});
        }}
        if ((vision.t || []).length) {{
          traces.push({{
            x: vision.t || [],
            y: vision[axis] || [],
            mode: 'markers',
            name: `vision racket ${{axis}}`,
            marker: {{ color, size: 5, opacity: 0.42 }},
          }});
        }}
      }});
      if (!traces.length) {{
        document.getElementById('racket-plot').innerHTML = '<div class="empty">No POE or vision racket data available for this run.</div>';
        return;
      }}
      const layout = baseLayout('Racket Position Relative to Base, Expressed in World Frame (m)');
      layout.yaxis.title = 'position (m)';
      Plotly.newPlot('racket-plot', traces, layout, {{responsive: true, displaylogo: false}})
        .then(() => {{
          wireTraceControls('racket-plot', 'racket-trace-controls');
          wireXZoom('racket-plot');
        }});
    }}

    function renderTimeSyncPlot() {{
      const rows = (ARM.time_sync_offsets || []).filter(
        row => typeof row.rel_s === 'number' && Number.isFinite(row.rel_s)
      );
      if (!rows.length) {{
        document.getElementById('time-sync-plot').innerHTML = '<div class="empty">No /time_sync/offset snapshots found for this run.</div>';
        return;
      }}
      const traces = [
        {{
          x: rows.map(row => row.rel_s),
          y: rows.map(row => row.current_offset_ms),
          mode: 'markers',
          name: 'current offset',
          marker: {{ color: '#5cd0ff', size: 6, symbol: 'circle' }},
          hovertemplate: 't=%{{x:.3f}}s<br>offset=%{{y:.3f}} ms<extra>current offset</extra>',
        }},
        {{
          x: rows.map(row => row.rel_s),
          y: rows.map(row => row.latest_accepted_offset_ms),
          mode: 'markers',
          name: 'accepted offset',
          marker: {{ color: '#a0ffb5', size: 6, symbol: 'diamond-open' }},
          hovertemplate: 't=%{{x:.3f}}s<br>accepted=%{{y:.3f}} ms<extra>accepted offset</extra>',
        }},
        {{
          x: rows.map(row => row.rel_s),
          y: rows.map(row => row.latest_offset_median_ms),
          mode: 'markers',
          name: 'offset median',
          marker: {{ color: '#ffc857', size: 6, symbol: 'square' }},
          hovertemplate: 't=%{{x:.3f}}s<br>median=%{{y:.3f}} ms<extra>offset median</extra>',
          visible: 'legendonly',
        }},
        {{
          x: rows.map(row => row.rel_s),
          y: rows.map(row => row.latest_rtt_ms),
          mode: 'markers',
          name: 'latest RTT',
          marker: {{ color: '#ff8a5c', size: 6, symbol: 'x' }},
          yaxis: 'y2',
          hovertemplate: 't=%{{x:.3f}}s<br>rtt=%{{y:.3f}} ms<extra>latest RTT</extra>',
        }},
        {{
          x: rows.map(row => row.rel_s),
          y: rows.map(row => row.rtt_p95_ms),
          mode: 'markers',
          name: 'RTT p95',
          marker: {{ color: '#f472b6', size: 6, symbol: 'triangle-up' }},
          yaxis: 'y2',
          hovertemplate: 't=%{{x:.3f}}s<br>rtt p95=%{{y:.3f}} ms<extra>RTT p95</extra>',
        }},
        {{
          x: rows.map(row => row.rel_s),
          y: rows.map(row => row.rtt_p99_ms),
          mode: 'markers',
          name: 'RTT p99',
          marker: {{ color: '#d7a8ff', size: 6, symbol: 'triangle-down' }},
          yaxis: 'y2',
          hovertemplate: 't=%{{x:.3f}}s<br>rtt p99=%{{y:.3f}} ms<extra>RTT p99</extra>',
          visible: 'legendonly',
        }},
      ];
      const layout = baseLayout('Time Sync Offset / RTT');
      layout.yaxis.title = 'offset (ms)';
      layout.yaxis2 = {{
        title: 'rtt (ms)',
        overlaying: 'y',
        side: 'right',
        gridcolor: 'rgba(97,112,138,0.16)',
        zerolinecolor: 'rgba(97,112,138,0.16)',
      }};
      Plotly.newPlot('time-sync-plot', traces, layout, {{responsive: true, displaylogo: false}})
        .then(() => {{
          wireTraceControls('time-sync-plot', 'time-sync-trace-controls');
          wireXZoom('time-sync-plot');
        }});
    }}

    function escapeHtml(text) {{
      return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function renderTable(containerId, rows, columns, emptyText) {{
      const el = document.getElementById(containerId);
      if (!rows.length) {{
        el.innerHTML = `<div class="empty">${{escapeHtml(emptyText)}}</div>`;
        return;
      }}
      const head = columns.map(col => `<th>${{escapeHtml(col.label)}}</th>`).join('');
      const body = rows.map(row => (
        '<tr>' + columns.map(col => `<td>${{escapeHtml(col.render(row))}}</td>`).join('') + '</tr>'
      )).join('');
      el.innerHTML = `<table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table>`;
    }}

    initPlotZoomButtons();
    setActivePlot('joint-plot');
    renderSummary();
    renderJointPlot();
    renderTorquePlot();
    renderRacketPlot();
    renderTimeSyncPlot();
    renderTable(
      'missing-pc-table',
      ARM.missing_pc_timestamps || [],
      [
        {{ label: 'topic', render: row => row.topic || '' }},
        {{ label: 'expected field', render: row => row.expected_field || '' }},
        {{ label: 'used field', render: row => row.used_field || '' }},
        {{ label: 'missing / total', render: row => `${{row.missing_count ?? 0}} / ${{row.total_count ?? 0}}` }},
        {{ label: 'note', render: row => row.note || '' }},
      ],
      'No arm timing metadata found.'
    );
    renderTable(
      'request-table',
      ARM.request_rows || [],
      [
        {{ label: 'request_id', render: row => row.request_id }},
        {{ label: 't', render: row => fmtTime(row.first_rel_s) }},
        {{ label: 'scheduled', render: row => fmtTime(row.scheduled_rel_s) }},
        {{ label: 'hit_x', render: row => fmtNum(row.hit_x_m, 3) }},
        {{ label: 'hit_z', render: row => fmtNum(row.hit_z_m, 3) }},
        {{ label: 'mode', render: row => row.mode || '' }},
        {{ label: 'events', render: row => row.events_text || '' }},
      ],
      'No grouped hit requests found.'
    );
    renderTable(
      'time-sync-table',
      ARM.time_sync_offsets || [],
      [
        {{ label: 'tag', render: row => row.tag || '' }},
        {{ label: 'source', render: row => row.source_id || '' }},
        {{ label: 'reason', render: row => row.publish_reason || '' }},
        {{ label: 'perf', render: row => row.stamp_text || '' }},
        {{ label: 'rel', render: row => fmtTime(row.rel_s) }},
        {{ label: 'offset', render: row => typeof row.current_offset_ms === 'number' ? fmtNum(row.current_offset_ms, 3) + ' ms' : '-' }},
        {{ label: 'accepted', render: row => typeof row.latest_accepted_offset_ms === 'number' ? fmtNum(row.latest_accepted_offset_ms, 3) + ' ms' : '-' }},
        {{ label: 'rtt', render: row => typeof row.latest_rtt_ms === 'number' ? fmtNum(row.latest_rtt_ms, 3) + ' ms' : '-' }},
        {{ label: 'rtt p95', render: row => typeof row.rtt_p95_ms === 'number' ? fmtNum(row.rtt_p95_ms, 3) + ' ms' : '-' }},
        {{ label: 'accepted/total', render: row => `${{row.period_accepted_count ?? 0}} / ${{row.period_sample_count ?? 0}}` }},
      ],
      'No time-sync offset snapshots found.'
    );
    renderTable(
      'hit-table',
      ARM.hit_events || [],
      [
        {{ label: 'event', render: row => row.event }},
        {{ label: 'request_id', render: row => row.request_id }},
        {{ label: 'perf', render: row => row.stamp_text }},
        {{ label: 'rel', render: row => fmtTime(row.rel_s) }},
        {{ label: 'scheduled', render: row => fmtTime(row.scheduled_rel_s) }},
        {{ label: 'hit(x,z)', render: row => `${{fmtNum(row.hit_x_m, 3)}}, ${{fmtNum(row.hit_z_m, 3)}}` }},
      ],
      'No hit events found.'
    );
    renderTable(
      'control-table',
      ARM.control_events || [],
      [
        {{ label: 'command', render: row => row.command || '' }},
        {{ label: 'source', render: row => row.source || '' }},
        {{ label: 'reason', render: row => row.reason || '' }},
        {{ label: 'perf', render: row => row.stamp_text }},
        {{ label: 'rel', render: row => fmtTime(row.rel_s) }},
      ],
      'No control events found.'
    );
  </script>
</body>
</html>"""
        self._send_html(body)

    def _serve_tracker_view(self, stem: str) -> None:
        run = self.server.get_run(stem)
        if run is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Tracker run not found")
            return
        payload, _, _ = self.server.choose_tracker_payload(run)
        if payload is not None:
            body = HTML_TEMPLATE.replace("%%DATA_JSON%%", json.dumps(payload, ensure_ascii=False))
            self._send_html(body)
            return
        if run.tracker_html is not None and run.tracker_html.exists():
            self._send_file(run.tracker_html, content_type="text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "No tracker html/json source found")

    def _serve_artifact(self, name: str) -> None:
        path = _safe_child(self.server.tracker_output_dir, name)
        if path is None or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        suffix = path.suffix.lower()
        if suffix == ".json":
            self._send_file(path, content_type="application/json; charset=utf-8")
            return
        if suffix == ".html":
            self._send_file(path, content_type="text/html; charset=utf-8")
            return
        if suffix == ".avi":
            self._send_file(path, content_type="video/x-msvideo")
            return
        if suffix == ".mp4":
            self._send_file(path, content_type="video/mp4")
            return
        self._send_file(path, content_type="application/octet-stream")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve unified tracker + pc_logger reports from tracker_output.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tracker-output-dir", type=Path, default=DEFAULT_TRACKER_OUTPUT_DIR)
    parser.add_argument("--poe-config", type=Path, default=DEFAULT_POE_CONFIG_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    server = TrackerReportServer(
        (str(args.host), int(args.port)),
        TrackerReportHandler,
        tracker_output_dir=Path(args.tracker_output_dir),
        poe_config_path=Path(args.poe_config),
    )
    print(f"Tracker unified report server listening on http://{args.host}:{args.port}")
    print(f"tracker_output: {Path(args.tracker_output_dir).resolve()}")
    print(f"POE config: {Path(args.poe_config).resolve()}")
    if server._poe_error:
        print(f"POE unavailable: {server._poe_error}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
