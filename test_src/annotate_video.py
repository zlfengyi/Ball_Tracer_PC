# -*- coding: utf-8 -*-
"""
离线标注脚本：读取拼接视频 + JSON 数据，生成带标注的视频。

在原有球框/球 3D/curve3 标注之外，还会在离线阶段调用 ArmCalibration 同款的
`yolo_model/racket.onnx + yolo_model/racket_pose.onnx`，只使用关键点 0-3 的几何中心
做球拍 2D/3D 定位，并将结果补充回 JSON。

用法：
  python test_src/annotate_video.py --input tracker_output/tracker_20260311_193455.json
  python test_src/annotate_video.py --input tracker_output/tracker_20260311_193455.json ^
      --output tracker_output/tracker_20260311_193455_annotated.avi
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ball_detector import BallDetection
from src.ball_localizer import Ball3D
from src.racket_localizer import RacketDetection, RacketLoc, RacketLocalizer

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.1
FONT_THICKNESS = 2
BOX_COLORS = [
    (0, 255, 0),       # 绿色 - 1号相机
    (0, 165, 255),     # 橙色 - 2号相机
    (255, 100, 100),   # 蓝色 - 3号相机
    (255, 0, 255),     # 紫色 - 4号相机
]
RACKET_BOX_COLOR = (255, 0, 255)
STATIONARY_BOX_COLOR = (180, 180, 180)
TEXT_COLOR = (255, 255, 255)
TEXT_3D_COLOR = (0, 255, 255)
TEXT_RACKET_3D_COLOR = (255, 0, 255)
STATE_COLORS = {
    "idle":         (128, 128, 128),
    "tracking_s0":  (255, 200, 0),
    "in_landing":   (0, 165, 255),
    "tracking_s1":  (0, 255, 0),
    "done":         (0, 0, 255),
}

_DEFAULT_RACKET_MODEL = _PROJECT_ROOT / "yolo_model" / "racket.onnx"
_DEFAULT_RACKET_POSE_MODEL = _PROJECT_ROOT / "yolo_model" / "racket_pose.onnx"
_DEFAULT_TRACKER_CONFIG = _PROJECT_ROOT / "src" / "config" / "tracker.json"
_TRACKER_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def guess_tracker_video_path(json_path: Path, data: dict) -> Path:
    config = data.get("config", {}) if isinstance(data, dict) else {}
    video_output = config.get("video_output", {}) if isinstance(config, dict) else {}

    for key in ("artifact_path", "path"):
        candidate = video_output.get(key) if isinstance(video_output, dict) else None
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (json_path.parent / candidate_path).resolve()
        if candidate_path.exists():
            return candidate_path

    for suffix in _TRACKER_VIDEO_SUFFIXES:
        candidate_path = json_path.with_suffix(suffix)
        if candidate_path.exists():
            return candidate_path

    return json_path.with_suffix(".avi")


def draw_badge(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    font_scale: float = 1.2,
    thickness: int = 3,
) -> int:
    """Draw a high-contrast text badge and return the box right edge."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, FONT, font_scale, thickness
    )
    pad_x = 12
    pad_y = 10
    box_tl = (x, y)
    box_br = (x + text_w + pad_x * 2, y + text_h + baseline + pad_y * 2)
    cv2.rectangle(img, box_tl, box_br, (0, 0, 0), -1)
    cv2.rectangle(img, box_tl, box_br, color, 2)
    cv2.putText(
        img,
        text,
        (x + pad_x, y + pad_y + text_h),
        FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return box_br[0]


def _format_xyz_m(x: float, y: float, z: float) -> str:
    return f"({x:.3f}, {y:.3f}, {z:.3f}) m"


def _scale_xyz_entry(entry: dict, scale: float) -> None:
    for key in ("x", "y", "z"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            entry[key] = float(value) * scale


def normalize_tracker_json_to_m(data: dict) -> None:
    cfg = data.setdefault("config", {})
    if cfg.get("distance_unit") == "m":
        return

    scale = 1.0 / 1000.0
    if isinstance(cfg.get("ideal_hit_z"), (int, float)):
        cfg["ideal_hit_z"] = float(cfg["ideal_hit_z"]) * scale

    for seq_name in ("observations", "predictions", "car_locs", "racket_observations"):
        for entry in data.get(seq_name, []):
            if isinstance(entry, dict):
                _scale_xyz_entry(entry, scale)

    for frame_data in data.get("frames", []):
        if not isinstance(frame_data, dict):
            continue
        for key in ("ball3d", "prediction", "car_loc", "racket3d"):
            entry = frame_data.get(key)
            if isinstance(entry, dict):
                _scale_xyz_entry(entry, scale)

    cfg["distance_unit"] = "m"


@dataclass
class RacketPipeline:
    localizer: RacketLocalizer
    pose_model_path: str
    keypoint_score_threshold: float
    min_face_valid_keypoints: int


def sync_video_frame_metadata(
    data: dict,
    frame_mapping: list[int],
    has_exact_mapping: bool,
) -> None:
    """Attach video-frame indices onto JSON frame entries for exact frame linkage."""
    frames_data = data.get("frames", [])
    for frame_data in frames_data:
        if isinstance(frame_data, dict):
            frame_data.pop("video_frame_idx", None)
            frame_data.pop("video_mapping_exact", None)

    for video_frame_idx, json_frame_idx in enumerate(frame_mapping):
        if 0 <= json_frame_idx < len(frames_data):
            frame_data = frames_data[json_frame_idx]
            if isinstance(frame_data, dict):
                frame_data["video_frame_idx"] = video_frame_idx
                frame_data["video_mapping_exact"] = bool(has_exact_mapping)

    cfg = data.setdefault("config", {})
    summary = data.setdefault("summary", {})
    cfg["video_frame_mapping_exact"] = bool(has_exact_mapping)
    summary["video_frame_mapping_exact"] = bool(has_exact_mapping)
    summary["video_frames_mapped"] = len(frame_mapping)


def build_video_frame_mapping(data: dict, total_video_frames: int) -> tuple[list[int], bool]:
    """
    返回“视频第 i 帧 -> JSON frames[j]”的映射。
    新版 run_tracker 会在 JSON 中写入 video_frame_indices；
    旧版 JSON 没有这个字段时，只能退化为 1:1 顺序映射。
    """
    frames_data = data["frames"]
    mapping = data.get("video_frame_indices")
    if mapping is None:
        fallback = list(range(min(total_video_frames, len(frames_data))))
        return fallback, False

    valid_mapping = [
        int(idx) for idx in mapping
        if isinstance(idx, int) and 0 <= idx < len(frames_data)
    ]
    return valid_mapping, True


def grid_dimensions(n_panels: int, cols: int = 2) -> tuple[int, int]:
    cols = max(1, min(cols, n_panels))
    rows = max(1, math.ceil(n_panels / cols))
    return cols, rows


def infer_stitched_grid(n_panels: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    if n_panels <= 2:
        return grid_dimensions(n_panels, cols=n_panels)
    if frame_h > 0 and frame_w / frame_h >= 2.5:
        return grid_dimensions(n_panels, cols=n_panels)
    return grid_dimensions(n_panels, cols=2)


def grid_slot(
    index: int,
    panel_w: int,
    panel_h: int,
    *,
    cols: int = 2,
) -> tuple[int, int]:
    col = index % cols
    row = index // cols
    return col * panel_w, row * panel_h


def split_stitched_panels(
    img: np.ndarray,
    serials: list[str],
) -> tuple[dict[str, np.ndarray], int, int]:
    """将拼接视频帧按 2x2 row-major 相机顺序拆成 panel。"""
    h, w = img.shape[:2]
    n_cams = len(serials)
    cols, rows = infer_stitched_grid(n_cams, w, h)
    panel_w = w // cols
    panel_h = h // rows
    panels: dict[str, np.ndarray] = {}
    for i, sn in enumerate(serials):
        x1, y1 = grid_slot(i, panel_w, panel_h, cols=cols)
        x2 = w if (i % cols) == cols - 1 else x1 + panel_w
        y2 = h if (i // cols) == rows - 1 else y1 + panel_h
        panels[sn] = img[y1:y2, x1:x2]
    return panels, panel_w, panel_h


def load_tracker_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_panel_timestamp(frame_data: dict, frame_idx: int, fps: float) -> float:
    """优先使用 JSON 中的 exposure_pc，缺失时再退化到按帧时间。"""
    exposure_pc = frame_data.get("exposure_pc")
    if isinstance(exposure_pc, (int, float)):
        return float(exposure_pc)
    if fps <= 0:
        return float(frame_idx)
    return frame_idx / fps


def extract_time_reference(data: dict) -> float | None:
    """Return the tracker time used as t=0 for HTML/video overlays."""
    cfg = data.get("config", {})
    first_frame_exposure_pc = cfg.get("first_frame_exposure_pc")
    if isinstance(first_frame_exposure_pc, (int, float)):
        return float(first_frame_exposure_pc)

    frames = data.get("frames", [])
    if frames:
        exposure_pc = frames[0].get("exposure_pc")
        if isinstance(exposure_pc, (int, float)):
            return float(exposure_pc)

    candidates: list[float] = []
    for items, key in (
        (data.get("observations", []), "t"),
        (data.get("car_locs", []), "t"),
        (data.get("predictions", []), "ct"),
    ):
        for item in items:
            value = item.get(key)
            if isinstance(value, (int, float)):
                candidates.append(float(value))

    return min(candidates) if candidates else None


def build_relative_frame_time_s(
    frame_data: dict,
    frame_idx: int,
    fps: float,
    time_reference: float | None,
) -> float:
    """Frame time shown in annotated video; aligns with HTML t-axis."""
    timestamp = build_panel_timestamp(frame_data, frame_idx, fps)
    if time_reference is None:
        return float(timestamp)
    return max(0.0, float(timestamp) - float(time_reference))


def convert_racket_loc_mm_to_m(loc: RacketLoc) -> RacketLoc:
    return RacketLoc(
        x=float(loc.x) / 1000.0,
        y=float(loc.y) / 1000.0,
        z=float(loc.z) / 1000.0,
        cameras_used=list(loc.cameras_used),
        pixels=dict(loc.pixels),
        reprojection_error=float(loc.reprojection_error),
        face_keypoint_score_min=float(loc.face_keypoint_score_min),
    )


def scale_panel_to_full(panel: np.ndarray) -> np.ndarray:
    """把半分辨率 panel 拉回原始坐标系大小，便于复用在线分片逻辑。"""
    h, w = panel.shape[:2]
    return cv2.resize(panel, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)


def init_racket_pipeline(
    first_frame: np.ndarray,
    serials: list[str],
    racket_model_path: str | Path,
    conf_threshold: float,
    pose_model_path: str | Path,
    keypoint_score_threshold: float,
    min_face_valid_keypoints: int,
) -> RacketPipeline:
    """Initialize the ArmCalibration racket-center pipeline for offline annotation."""
    racket_model_path = Path(racket_model_path)
    if not racket_model_path.exists():
        raise FileNotFoundError(f"找不到球拍模型: {racket_model_path}")
    pose_model_path = Path(pose_model_path)
    if not pose_model_path.exists():
        raise FileNotFoundError(f"找不到球拍关键点模型: {pose_model_path}")

    localizer = RacketLocalizer(
        racket_model_path=racket_model_path,
        pose_model_path=pose_model_path,
        bbox_conf=conf_threshold,
        keypoint_score_threshold=keypoint_score_threshold,
        min_valid_keypoints=min_face_valid_keypoints,
    )

    panels_half, _, _ = split_stitched_panels(first_frame, serials)
    panels_full = {
        sn: scale_panel_to_full(panel)
        for sn, panel in panels_half.items()
    }
    try:
        localizer.locate(panels_full)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "球拍 ONNX 推理缺少依赖，请安装 onnx / onnxruntime 后重试。"
        ) from e

    return RacketPipeline(
        localizer=localizer,
        pose_model_path=str(pose_model_path),
        keypoint_score_threshold=float(keypoint_score_threshold),
        min_face_valid_keypoints=int(min_face_valid_keypoints),
    )


def detect_racket_frame(
    img: np.ndarray,
    serials: list[str],
    pipeline: RacketPipeline,
) -> tuple[dict[str, RacketDetection], Optional[RacketLoc]]:
    """Run the ArmCalibration racket detector on one stitched frame."""
    panels_half, _, _ = split_stitched_panels(img, serials)
    panels_full = {
        sn: scale_panel_to_full(panel)
        for sn, panel in panels_half.items()
    }
    detections, loc = pipeline.localizer.locate(panels_full)
    if loc is not None:
        loc = convert_racket_loc_mm_to_m(loc)
    return detections, loc


def serialize_racket_detection(
    det: RacketDetection,
    *,
    keypoint_score_threshold: float,
) -> dict:
    payload = {
        "detected": bool(det.detected),
        "accepted": bool(det.accepted),
        "failure_reason": det.failure_reason,
    }
    if not det.detected:
        return payload

    if det.bbox_xyxy is not None:
        x1, y1, x2, y2 = det.bbox_xyxy
        payload["bbox"] = {
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "confidence": round(float(det.bbox_confidence), 4),
        }
        payload["x1"] = round(float(x1))
        payload["y1"] = round(float(y1))
        payload["x2"] = round(float(x2))
        payload["y2"] = round(float(y2))
        payload["conf"] = round(float(det.bbox_confidence), 3)

    if det.center_xy is not None:
        payload["x"] = round(float(det.center_xy[0]))
        payload["y"] = round(float(det.center_xy[1]))
        payload["center_xy"] = [
            round(float(det.center_xy[0]), 2),
            round(float(det.center_xy[1]), 2),
        ]

    if det.keypoints_xy is not None and len(det.keypoints_xy) > 0:
        keypoints_xy = np.asarray(det.keypoints_xy, dtype=np.float64)
        score_arr = np.asarray(det.keypoint_scores, dtype=np.float64)
        valid_mask = score_arr >= float(keypoint_score_threshold)
        payload["keypoints"] = [
            {
                "id": int(idx),
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "score": round(float(score_arr[idx]), 3),
                "valid": bool(valid_mask[idx]),
                "used_for_center": bool(idx in (0, 1, 2, 3)),
            }
            for idx, point in enumerate(keypoints_xy)
        ]
        payload["all_keypoints_center_xy"] = [
            round(float(keypoints_xy[:, 0].mean()), 2),
            round(float(keypoints_xy[:, 1].mean()), 2),
        ]
        payload["keypoint_score_mean"] = round(float(score_arr.mean()), 3)
        payload["keypoint_score_min"] = round(float(score_arr.min()), 3)
        payload["keypoint_score_max"] = round(float(score_arr.max()), 3)
        payload["valid_keypoint_count"] = int(np.sum(valid_mask))

    payload["center_keypoint_ids"] = [0, 1, 2, 3]
    payload["face_keypoint_score_min"] = round(float(det.face_keypoint_score_min), 3)
    payload["face_valid_keypoint_count"] = int(det.face_valid_keypoint_count)
    return payload


def serialize_racket_detections(
    detections: dict[str, RacketDetection],
    *,
    keypoint_score_threshold: float,
) -> dict[str, list[dict]]:
    serialized: dict[str, list[dict]] = {}
    for sn, det in detections.items():
        if not det.detected:
            continue
        serialized[sn] = [
            serialize_racket_detection(
                det,
                keypoint_score_threshold=keypoint_score_threshold,
            )
        ]
    return serialized


def serialize_3d(obj3d: Ball3D) -> dict:
    return {
        "x": round(obj3d.x, 4),
        "y": round(obj3d.y, 4),
        "z": round(obj3d.z, 4),
        "reproj": round(obj3d.reprojection_error, 1),
        "conf": round(obj3d.confidence, 3),
        "cameras": obj3d.cameras_used,
    }


def serialize_racket_3d(obj3d: RacketLoc) -> dict:
    return {
        "x": round(obj3d.x, 4),
        "y": round(obj3d.y, 4),
        "z": round(obj3d.z, 4),
        "reproj": round(obj3d.reprojection_error, 1),
        "conf": round(obj3d.face_keypoint_score_min, 3),
        "face_min": round(obj3d.face_keypoint_score_min, 3),
        "cameras": obj3d.cameras_used,
        "pixels": {
            sn: [round(float(px), 2), round(float(py), 2)]
            for sn, (px, py) in obj3d.pixels.items()
        },
    }


def apply_racket_results(
    frame_data: dict,
    detections: dict[str, RacketDetection],
    racket3d: Optional[RacketLoc],
    *,
    keypoint_score_threshold: float,
) -> None:
    """把当前帧的球拍结果写回 JSON frame entry。"""
    frame_data.pop("racket_detections", None)
    frame_data.pop("racket3d", None)

    serialized_dets = serialize_racket_detections(
        detections,
        keypoint_score_threshold=keypoint_score_threshold,
    )
    if serialized_dets:
        frame_data["racket_detections"] = serialized_dets

    if racket3d is not None:
        frame_data["racket3d"] = serialize_racket_3d(racket3d)


def clear_racket_results(data: dict) -> None:
    """清理旧的球拍结果，避免重复运行时留下脏数据。"""
    for frame_data in data.get("frames", []):
        frame_data.pop("racket_detections", None)
        frame_data.pop("racket3d", None)
    data.pop("racket_observations", None)

    summary = data.get("summary")
    if isinstance(summary, dict):
        summary.pop("racket_observations_3d", None)
        summary.pop("racket_frames_processed", None)


def build_racket_json_payload(
    data: dict,
    *,
    source_json_path: Path,
    source_video_path: Path,
) -> dict:
    """Build a racket-only JSON payload that stays frame-aligned with the saved video."""
    cfg = data.get("config", {})
    summary = data.get("summary", {})
    racket_frames: list[dict] = []

    for frame_data in data.get("frames", []):
        if not isinstance(frame_data, dict):
            continue
        frame_payload: dict = {}
        for key in (
            "idx",
            "video_frame_idx",
            "video_mapping_exact",
            "exposure_pc",
            "elapsed_s",
        ):
            if key in frame_data:
                frame_payload[key] = frame_data[key]

        for key in ("racket_detections", "racket3d"):
            if key in frame_data:
                frame_payload[key] = frame_data[key]

        if frame_payload:
            racket_frames.append(frame_payload)

    return {
        "config": {
            "source_tracker_json": str(source_json_path),
            "source_video_path": str(source_video_path),
            "distance_unit": cfg.get("distance_unit", "m"),
            "serials": cfg.get("serials", []),
            "first_frame_exposure_pc": cfg.get("first_frame_exposure_pc"),
            "video_frame_mapping_exact": cfg.get("video_frame_mapping_exact"),
            "racket_model_path": cfg.get("racket_model_path"),
            "racket_pose_model_path": cfg.get("racket_pose_model_path"),
            "racket_conf_threshold": cfg.get("racket_conf_threshold"),
            "racket_keypoint_score_threshold": cfg.get("racket_keypoint_score_threshold"),
            "racket_min_face_valid_keypoints": cfg.get("racket_min_face_valid_keypoints"),
        },
        "summary": {
            "video_frame_mapping_exact": summary.get(
                "video_frame_mapping_exact",
                cfg.get("video_frame_mapping_exact"),
            ),
            "video_frames_mapped": summary.get("video_frames_mapped"),
            "racket_observations_3d": summary.get(
                "racket_observations_3d",
                len(data.get("racket_observations", [])),
            ),
            "racket_frames_processed": summary.get("racket_frames_processed"),
        },
        "frames": racket_frames,
        "racket_observations": data.get("racket_observations", []),
    }


def draw_scaled_detections(
    out: np.ndarray,
    detections: list[dict],
    x_offset: int,
    y_offset: int,
    scale: float,
    color: tuple[int, int, int],
    *,
    draw_center: bool = False,
    label_prefix: str = "",
) -> None:
    """把全分辨率检测结果按缩放比例绘制到 annotated 视频。"""
    for det in detections:
        label = det.get("label")
        draw_color = color
        if label == "stationary_object":
            draw_color = STATIONARY_BOX_COLOR

        x1 = int(det["x1"] * scale) + x_offset
        y1 = int(det["y1"] * scale) + y_offset
        x2 = int(det["x2"] * scale) + x_offset
        y2 = int(det["y2"] * scale) + y_offset
        cv2.rectangle(out, (x1, y1), (x2, y2), draw_color, 2)

        if label == "tennis_ball":
            conf_text = f"B {det['conf']:.2f}"
        elif label == "stationary_object":
            conf_text = f"S {det['conf']:.2f}"
        else:
            conf_text = f"{label_prefix}{det['conf']:.2f}"
        cv2.putText(
            out, conf_text, (x1, max(y_offset + 20, y1 - 5)),
            FONT, FONT_SCALE, draw_color, FONT_THICKNESS,
        )

        if draw_center:
            cx = int(det["x"] * scale) + x_offset
            cy = int(det["y"] * scale) + y_offset
            cv2.drawMarker(
                out, (cx, cy), draw_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )


def draw_racket_detections(
    out: np.ndarray,
    detections: list[dict],
    x_offset: int,
    y_offset: int,
    scale: float,
) -> None:
    """Draw ArmCalibration-style racket bbox, keypoints, and center."""
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue

        accepted = bool(det.get("accepted", False))
        draw_color = RACKET_BOX_COLOR if accepted else (0, 165, 255)
        x1 = int(bbox["x1"] * scale) + x_offset
        y1 = int(bbox["y1"] * scale) + y_offset
        x2 = int(bbox["x2"] * scale) + x_offset
        y2 = int(bbox["y2"] * scale) + y_offset
        cv2.rectangle(out, (x1, y1), (x2, y2), draw_color, 2)

        score_text = f"R {bbox.get('confidence', det.get('conf', 0.0)):.2f}"
        if not accepted and det.get("failure_reason"):
            score_text += f" {det['failure_reason']}"
        cv2.putText(
            out,
            score_text,
            (x1, max(y_offset + 20, y1 - 5)),
            FONT,
            0.9,
            draw_color,
            2,
            )        

        for keypoint in det.get("keypoints", []):
            kp_x = int(keypoint["x"] * scale) + x_offset
            kp_y = int(keypoint["y"] * scale) + y_offset
            if keypoint.get("used_for_center"):
                kp_color = (0, 255, 0) if keypoint.get("valid") else (0, 165, 255)
            else:
                kp_color = (255, 200, 0)
            cv2.circle(out, (kp_x, kp_y), 5, kp_color, -1)
            cv2.putText(
                out,
                f"{keypoint['id']}:{keypoint['score']:.1f}",
                (kp_x + 6, kp_y - 6),
                FONT,
                0.45,
                kp_color,
                1,
            )

        if accepted and "x" in det and "y" in det:
            cx = int(det["x"] * scale) + x_offset
            cy = int(det["y"] * scale) + y_offset
            cv2.drawMarker(
                out,
                (cx, cy),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=2,
            )


def annotate_frame(
    img: np.ndarray,
    frame_data: dict,
    serials: list[str],
    n_cams: int,
    panel_w: int,
    panel_h: int,
    layout_cols: int,
    *,
    show_racket: bool = False,
    relative_time_s: float | None = None,
) -> np.ndarray:
    """在拼接画面上绘制球/球拍/3D/状态等离线标注。"""
    h, w = img.shape[:2]
    out = img.copy()
    cols = layout_cols
    rows = max(1, math.ceil(n_cams / max(1, cols)))
    scale = 0.5

    detections = frame_data.get("detections", {})
    racket_detections = frame_data.get("racket_detections", {})
    tiles = frame_data.get("tiles", {})
    frame_car_loc = frame_data.get("car_loc")
    detection_counts = frame_data.get("detection_counts", {})

    for cam_idx, sn in enumerate(serials):
        color = BOX_COLORS[cam_idx % len(BOX_COLORS)]
        x_offset, y_offset = grid_slot(cam_idx, panel_w, panel_h, cols=cols)
        counts = detection_counts.get(sn)
        if counts is None:
            dets = detections.get(sn, [])
            counts = {
                "tennis_ball": sum(
                    1 for det in dets if det.get("label", "tennis_ball") == "tennis_ball"
                ),
                "stationary_object": sum(
                    1 for det in dets if det.get("label", "tennis_ball") == "stationary_object"
                ),
            }
        ball_count = counts.get("tennis_ball", 0)
        static_count = counts.get("stationary_object", 0)
        car_count = 0
        if frame_car_loc and "pixels" in frame_car_loc and sn in frame_car_loc["pixels"]:
            car_count = 1

        draw_scaled_detections(
            out,
            detections.get(sn, []),
            x_offset,
            y_offset,
            scale,
            color,
        )

        if show_racket:
            draw_racket_detections(
                out,
                racket_detections.get(sn, []),
                x_offset,
                y_offset,
                scale,
            )

        if frame_car_loc and "pixels" in frame_car_loc and sn in frame_car_loc["pixels"]:
            px, py = frame_car_loc["pixels"][sn]
            cx = int(px * scale) + x_offset
            cy = int(py * scale) + y_offset
            cv2.drawMarker(
                out, (cx, cy), (0, 200, 255),
                cv2.MARKER_DIAMOND, 20, 2,
            )

        draw_badge(
            out,
            f"{sn[-3:]}  BALL {ball_count}  STATIC {static_count}  CAR {car_count}",
            x_offset + 10,
            y_offset + 10,
            color,
        )

    for col in range(1, cols):
        x = panel_w * col
        cv2.line(out, (x, 0), (x, h), (100, 100, 100), 1)
    for row in range(1, rows):
        y = panel_h * row
        cv2.line(out, (0, y), (w, y), (100, 100, 100), 1)

    line_h = 40
    lines: list[tuple[str, tuple[int, int, int]]] = []

    lines.append((
        f"#{frame_data['idx']}  "
        f"{('t=' + format(relative_time_s, '.3f') + 's  ') if relative_time_s is not None else ''}"
        f"perf={frame_data.get('exposure_pc', 0):.6f}s  "
        f"lat={frame_data.get('latency_ms', 0):.0f}ms",
        TEXT_COLOR,
    ))

    if show_racket:
        racket_parts = []
        for sn in serials:
            dets = racket_detections.get(sn, [])
            accepted_count = sum(1 for det in dets if det.get("accepted"))
            detected_count = sum(1 for det in dets if det.get("detected", True))
            racket_parts.append(f"{sn[-3:]}={accepted_count}/{detected_count}")
        lines.append((f"racket: {'  '.join(racket_parts)}", RACKET_BOX_COLOR))

    ball3d = frame_data.get("ball3d")
    if ball3d:
        cams = "+".join(s[-3:] for s in ball3d["cameras"])
        lines.append((
            f"3D: {_format_xyz_m(ball3d['x'], ball3d['y'], ball3d['z'])}  "
            f"reproj={ball3d['reproj']:.1f}px  cams={cams}  conf={ball3d['conf']:.2f}",
            TEXT_3D_COLOR,
        ))

    racket3d = frame_data.get("racket3d")
    if show_racket and racket3d:
        cams = "+".join(s[-3:] for s in racket3d["cameras"])
        lines.append((
            f"R3D: {_format_xyz_m(racket3d['x'], racket3d['y'], racket3d['z'])}  "
            f"reproj={racket3d['reproj']:.1f}px  cams={cams}  face_min={racket3d['face_min']:.1f}",
            TEXT_RACKET_3D_COLOR,
        ))

    state = frame_data.get("state", "idle")
    state_color = STATE_COLORS.get(state, TEXT_COLOR)
    state_str = f"curve3: {state}"
    pred = frame_data.get("prediction")
    if pred:
        state_str += (
            f"  hit={_format_xyz_m(pred['x'], pred['y'], pred['z'])} "
            f"stage={pred['stage']} lead={pred['lead_ms']}ms"
        )
    lines.append((state_str, state_color))

    if frame_car_loc:
        cams = "+".join(s[-3:] for s in frame_car_loc["cameras_used"])
        lines.append((
            f"car: {_format_xyz_m(frame_car_loc['x'], frame_car_loc['y'], frame_car_loc['z'])}  "
            f"yaw={math.degrees(frame_car_loc['yaw']):.1f}deg  cams={cams}",
            (0, 200, 255),
        ))

    y = h - 15
    for text, color in reversed(lines):
        cv2.putText(out, text, (10, y), FONT, FONT_SCALE, color, FONT_THICKNESS)
        y -= line_h

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="离线标注视频，并补充球拍 2D/3D 结果")
    parser.add_argument("--input", required=True, help="输入 tracker JSON 路径")
    parser.add_argument("--video", default=None, help="原始拼接视频路径，默认自动查找同名 .mp4/.avi")
    parser.add_argument("--output", default=None, help="输出 annotated 视频路径，默认同目录 _annotated.avi")
    parser.add_argument("--no-output-video", action="store_true", help="只更新 JSON/HTML 所需结果，不写 annotated 视频")
    parser.add_argument("--json-output", default=None, help="补充后的 merged JSON 输出路径，默认覆写输入 JSON")
    parser.add_argument("--racket-json-output", default=None, help="单独输出球拍 2D/3D 与帧映射 JSON")
    parser.add_argument("--tracker-config", default=str(_DEFAULT_TRACKER_CONFIG), help="tracker.json 路径")
    parser.add_argument("--racket-model", default=str(_DEFAULT_RACKET_MODEL), help="球拍 bbox 模型路径")
    parser.add_argument("--racket-pose-model", default=str(_DEFAULT_RACKET_POSE_MODEL), help="球拍关键点模型路径")
    parser.add_argument("--racket-conf", type=float, default=0.25, help="球拍 bbox 置信度阈值")
    parser.add_argument("--racket-keypoint-threshold", type=float, default=40.0, help="ArmCalibration 同款 0-3 关键点分数阈值")
    parser.add_argument("--racket-min-face-valid-keypoints", type=int, default=4, help="ArmCalibration 同款中心关键点最少有效个数")
    parser.add_argument("--no-racket", action="store_true", help="只做旧标注，不补充球拍结果")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧，便于快速验证")
    args = parser.parse_args()

    json_path = Path(args.input)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    normalize_tracker_json_to_m(data)

    video_path = (
        Path(args.video)
        if args.video
        else guess_tracker_video_path(json_path, data)
    )
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return

    output_path = args.output or str(json_path.with_name(json_path.stem + "_annotated.avi"))
    json_output_path = Path(args.json_output) if args.json_output else json_path
    racket_json_output_path = (
        Path(args.racket_json_output)
        if args.racket_json_output
        else None
    )

    serials = data["config"]["serials"]
    n_cams = len(serials)
    frames_data = data["frames"]
    racket_enabled = not args.no_racket

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols, rows = infer_stitched_grid(n_cams, w, h)
    panel_w = w // cols
    panel_h = h // rows
    frame_mapping, has_exact_mapping = build_video_frame_mapping(data, total)
    time_reference = extract_time_reference(data)
    sync_video_frame_metadata(data, frame_mapping, has_exact_mapping)

    if racket_enabled:
        clear_racket_results(data)

    writer = None
    if not args.no_output_video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"输入视频: {video_path} ({w}x{h}, {fps:.0f}fps, {total} 帧)")
    print(f"JSON 帧数: {len(frames_data)}")
    if has_exact_mapping:
        print(f"视频映射: 使用 JSON video_frame_indices（{len(frame_mapping)} 帧精确对齐）")
    else:
        print("视频映射: JSON 不含 video_frame_indices，退化为按帧号 1:1 对齐（若录制时丢帧，标注可能漂移）")
    if args.no_output_video:
        print("输出视频: disabled (--no-output-video)")
    else:
        print(f"输出视频: {output_path}")
    if racket_enabled:
        print(f"球拍模型: {args.racket_model}")
        print(f"球拍关键点模型: {args.racket_pose_model}")
        print(f"JSON 输出: {json_output_path}")
        if racket_json_output_path is not None:
            print(f"球拍 JSON 输出: {racket_json_output_path}")

    frame_idx = 0
    n_annotated = 0
    racket_pipeline: Optional[RacketPipeline] = None
    racket_observations: list[dict] = []
    racket_frames_processed = 0

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ret, img = cap.read()
        if not ret:
            break

        if frame_idx < len(frame_mapping):
            fd = frames_data[frame_mapping[frame_idx]]

            if racket_enabled:
                if racket_pipeline is None:
                    racket_pipeline = init_racket_pipeline(
                        first_frame=img,
                        serials=serials,
                        racket_model_path=args.racket_model,
                        conf_threshold=args.racket_conf,
                        pose_model_path=args.racket_pose_model,
                        keypoint_score_threshold=args.racket_keypoint_threshold,
                        min_face_valid_keypoints=args.racket_min_face_valid_keypoints,
                    )
                    print(
                        "球拍关键点预热完成: "
                        f"pose={racket_pipeline.pose_model_path}, "
                        f"thr={racket_pipeline.keypoint_score_threshold:.1f}, "
                        f"min_face_valid={racket_pipeline.min_face_valid_keypoints}"
                    )

                frame_time = build_panel_timestamp(fd, frame_idx, fps)
                relative_time_s = build_relative_frame_time_s(
                    fd, frame_idx, fps, time_reference
                )
                racket_dets, racket3d = detect_racket_frame(
                    img, serials, racket_pipeline
                )
                apply_racket_results(
                    fd,
                    racket_dets,
                    racket3d,
                    keypoint_score_threshold=racket_pipeline.keypoint_score_threshold,
                )
                racket_frames_processed += 1

                if racket3d is not None:
                    racket_observations.append({
                        "frame_idx": fd.get("idx", frame_mapping[frame_idx]),
                        "video_frame_idx": frame_idx,
                        "x": racket3d.x,
                        "y": racket3d.y,
                        "z": racket3d.z,
                        "t": frame_time,
                        "elapsed_s": relative_time_s,
                        "reproj_err": racket3d.reprojection_error,
                        "confidence": racket3d.face_keypoint_score_min,
                        "face_keypoint_score_min": racket3d.face_keypoint_score_min,
                        "cameras_used": racket3d.cameras_used,
                    })

            annotated = annotate_frame(
                img,
                fd,
                serials,
                n_cams,
                panel_w,
                panel_h,
                cols,
                show_racket=racket_enabled,
                relative_time_s=build_relative_frame_time_s(
                    fd, frame_idx, fps, time_reference
                ),
            )
            n_annotated += 1
        else:
            annotated = img

        if writer is not None:
            writer.write(annotated)
        frame_idx += 1

        if frame_idx % 200 == 0:
            if racket_enabled:
                print(
                    f"  {frame_idx}/{total} 帧... "
                    f"racket_3d={len(racket_observations)}"
                )
            else:
                print(f"  {frame_idx}/{total} 帧...")

    cap.release()
    if writer is not None:
        writer.release()

    if racket_enabled:
        config = data.setdefault("config", {})
        summary = data.setdefault("summary", {})
        config["racket_model_path"] = str(
            Path(args.racket_model)
        )
        config["racket_pose_model_path"] = str(
            Path(args.racket_pose_model)
        )
        config["racket_conf_threshold"] = args.racket_conf
        config["racket_keypoint_score_threshold"] = args.racket_keypoint_threshold
        config["racket_min_face_valid_keypoints"] = args.racket_min_face_valid_keypoints
        data["racket_observations"] = racket_observations
        summary["racket_observations_3d"] = len(racket_observations)
        summary["racket_frames_processed"] = racket_frames_processed

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"JSON 已更新: {json_output_path}")
        print(f"球拍 3D 观测数: {len(racket_observations)}")
        if racket_json_output_path is not None:
            racket_payload = build_racket_json_payload(
                data,
                source_json_path=json_path,
                source_video_path=video_path,
            )
            with open(racket_json_output_path, "w", encoding="utf-8") as f:
                json.dump(racket_payload, f, ensure_ascii=False, indent=2)
            print(f"球拍 JSON 已输出: {racket_json_output_path}")

    if writer is not None:
        print(f"完成：{n_annotated} 帧已标注，输出到 {output_path}")
    else:
        print(f"完成：{n_annotated} 帧已处理，未写出 annotated 视频")


if __name__ == "__main__":
    main()
