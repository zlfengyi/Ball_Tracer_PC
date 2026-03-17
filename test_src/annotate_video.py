# -*- coding: utf-8 -*-
"""
离线标注脚本：读取拼接视频 + JSON 数据，生成带标注的视频。

在原有球框/球 3D/curve3 标注之外，还会在离线阶段调用 `yolo_model/racket.onnx`
识别球拍，复用与网球一致的 search/hold/track 分片逻辑，对球拍中点做多相机 3D
定位，并将结果补充回 JSON。

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

from src.ball_detector import BallDetection, BallDetector
from src.ball_localizer import Ball3D, BallLocalizer
from src.tile_manager import TileManager, TileRect

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.1
FONT_THICKNESS = 2
BOX_COLORS = [
    (0, 255, 0),       # 绿色 - 1号相机
    (0, 165, 255),     # 橙色 - 2号相机
    (255, 100, 100),   # 蓝色 - 3号相机
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
_DEFAULT_TRACKER_CONFIG = _PROJECT_ROOT / "src" / "config" / "tracker.json"


@dataclass
class RacketPipeline:
    detector: BallDetector
    localizer: BallLocalizer
    tile_mgr: TileManager
    engine_batch: int
    min_cameras_for_3d: int
    max_reproj_error_px: float


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


def split_stitched_panels(
    img: np.ndarray,
    serials: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    """将拼接视频帧按相机顺序拆成 panel。"""
    h, w = img.shape[:2]
    n_cams = len(serials)
    panel_w = w // n_cams
    panels: dict[str, np.ndarray] = {}
    for i, sn in enumerate(serials):
        x1 = i * panel_w
        x2 = w if i == n_cams - 1 else (i + 1) * panel_w
        panels[sn] = img[:, x1:x2]
    return panels, panel_w


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


def scale_panel_to_full(panel: np.ndarray) -> np.ndarray:
    """把半分辨率 panel 拉回原始坐标系大小，便于复用在线分片逻辑。"""
    h, w = panel.shape[:2]
    return cv2.resize(panel, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)


def yolo_detect_n(
    detector: BallDetector,
    img_list: list[np.ndarray],
    engine_batch: int,
) -> list[list[BallDetection]]:
    """按模型支持的 batch size 拆分推理。"""
    if not img_list:
        return []

    if len(img_list) <= engine_batch:
        padded = img_list[:]
        while len(padded) < engine_batch:
            padded.append(padded[-1])
        results = detector.detect_batch(padded)
        return results[:len(img_list)]

    detections_list: list[list[BallDetection]] = []
    for i in range(0, len(img_list), engine_batch):
        batch = img_list[i:i + engine_batch]
        actual_n = len(batch)
        while len(batch) < engine_batch:
            batch.append(batch[-1])
        results = detector.detect_batch(batch)
        detections_list.extend(results[:actual_n])
    return detections_list


def init_racket_pipeline(
    first_frame: np.ndarray,
    serials: list[str],
    tracker_cfg: dict,
    racket_model_path: str | Path,
    conf_threshold: float,
) -> RacketPipeline:
    """初始化球拍检测 + 分片追踪 + 3D 定位上下文。"""
    racket_model_path = Path(racket_model_path)
    if not racket_model_path.exists():
        raise FileNotFoundError(f"找不到球拍模型: {racket_model_path}")

    detector = BallDetector(
        model_path=racket_model_path,
        conf_threshold=conf_threshold,
        max_box_aspect_ratio=None,
    )
    localizer = BallLocalizer(detector=detector, conf_threshold=conf_threshold)

    panels_half, _ = split_stitched_panels(first_frame, serials)
    panels_full = {
        sn: scale_panel_to_full(panel)
        for sn, panel in panels_half.items()
    }
    camera_sizes = {
        sn: (panel.shape[1], panel.shape[0])
        for sn, panel in panels_full.items()
    }
    tile_mgr = TileManager(
        camera_sizes,
        tile_size=tracker_cfg.get("tile_size", 1280),
        track_timeout=tracker_cfg.get("track_timeout_s", 0.3),
        search_hold_frames=tracker_cfg.get("search_hold_frames", 4),
    )

    warmup_tiles = []
    for sn in serials:
        crop, _ = tile_mgr.get_tile(sn, panels_full[sn], current_time=0.0)
        warmup_tiles.append(crop)

    engine_batch = max(1, len(warmup_tiles))
    module_name = ""
    for try_batch in range(engine_batch, 0, -1):
        try:
            yolo_detect_n(detector, warmup_tiles, try_batch)
            engine_batch = try_batch
            break
        except ModuleNotFoundError as e:
            module_name = e.name or ""
            raise RuntimeError(
                "球拍 ONNX 推理缺少依赖，请安装 onnx / onnxruntime 后重试。"
            ) from e
        except Exception:
            continue

    if module_name:
        raise RuntimeError(
            "球拍 ONNX 推理缺少依赖，请安装 onnx / onnxruntime 后重试。"
        )

    for _ in range(2):
        yolo_detect_n(detector, warmup_tiles, engine_batch)

    return RacketPipeline(
        detector=detector,
        localizer=localizer,
        tile_mgr=tile_mgr,
        engine_batch=engine_batch,
        min_cameras_for_3d=tracker_cfg.get("min_cameras_for_3d", 2),
        max_reproj_error_px=tracker_cfg.get("max_reproj_error_px", 15.0),
    )


def detect_racket_frame(
    img: np.ndarray,
    serials: list[str],
    frame_time: float,
    pipeline: RacketPipeline,
) -> tuple[dict[str, list[BallDetection]], dict[str, TileRect], Optional[Ball3D]]:
    """在一帧拼接视频上做球拍 2D 检测 + 3D 定位。"""
    panels_half, _ = split_stitched_panels(img, serials)
    panels_full = {
        sn: scale_panel_to_full(panel)
        for sn, panel in panels_half.items()
    }

    tile_imgs: list[np.ndarray] = []
    frame_tiles: dict[str, TileRect] = {}
    for sn in serials:
        crop, tile_rect = pipeline.tile_mgr.get_tile(sn, panels_full[sn], frame_time)
        tile_imgs.append(crop)
        frame_tiles[sn] = tile_rect

    det_results = yolo_detect_n(pipeline.detector, tile_imgs, pipeline.engine_batch)
    all_detections: dict[str, list[BallDetection]] = {}
    for sn, dets in zip(serials, det_results):
        all_detections[sn] = [
            TileManager.map_detection_to_full(det, frame_tiles[sn])
            for det in dets
        ]

    primary_dets = {
        sn: dets[0]
        for sn, dets in all_detections.items()
        if dets
    }

    racket3d: Optional[Ball3D] = None
    if len(primary_dets) >= pipeline.min_cameras_for_3d:
        candidate = pipeline.localizer.triangulate(primary_dets)
        if candidate.reprojection_error <= pipeline.max_reproj_error_px:
            racket3d = candidate
            for sn, det in primary_dets.items():
                pipeline.tile_mgr.on_3d_located(sn, det.x, det.y, frame_time)
        else:
            for sn in primary_dets:
                pipeline.tile_mgr.on_2d_detected(sn, frame_tiles[sn])
    else:
        for sn in primary_dets:
            pipeline.tile_mgr.on_2d_detected(sn, frame_tiles[sn])

    return all_detections, frame_tiles, racket3d


def serialize_detections(
    detections: dict[str, list[BallDetection]],
) -> dict[str, list[dict]]:
    serialized: dict[str, list[dict]] = {}
    for sn, dets in detections.items():
        if not dets:
            continue
        serialized[sn] = [
            {
                "x": round(det.x),
                "y": round(det.y),
                "x1": round(det.x1),
                "y1": round(det.y1),
                "x2": round(det.x2),
                "y2": round(det.y2),
                "conf": round(det.confidence, 3),
            }
            for det in dets
        ]
    return serialized


def serialize_tiles(tiles: dict[str, TileRect]) -> dict[str, dict]:
    return {
        sn: {"x": tile.x, "y": tile.y, "w": tile.w, "h": tile.h}
        for sn, tile in tiles.items()
    }


def serialize_3d(obj3d: Ball3D) -> dict:
    return {
        "x": round(obj3d.x, 1),
        "y": round(obj3d.y, 1),
        "z": round(obj3d.z, 1),
        "reproj": round(obj3d.reprojection_error, 1),
        "conf": round(obj3d.confidence, 3),
        "cameras": obj3d.cameras_used,
    }


def apply_racket_results(
    frame_data: dict,
    detections: dict[str, list[BallDetection]],
    tiles: dict[str, TileRect],
    racket3d: Optional[Ball3D],
) -> None:
    """把当前帧的球拍结果写回 JSON frame entry。"""
    frame_data.pop("racket_detections", None)
    frame_data.pop("racket_tiles", None)
    frame_data.pop("racket3d", None)

    serialized_dets = serialize_detections(detections)
    if serialized_dets:
        frame_data["racket_detections"] = serialized_dets

    if tiles:
        frame_data["racket_tiles"] = serialize_tiles(tiles)

    if racket3d is not None:
        frame_data["racket3d"] = serialize_3d(racket3d)


def clear_racket_results(data: dict) -> None:
    """清理旧的球拍结果，避免重复运行时留下脏数据。"""
    for frame_data in data.get("frames", []):
        frame_data.pop("racket_detections", None)
        frame_data.pop("racket_tiles", None)
        frame_data.pop("racket3d", None)
    data.pop("racket_observations", None)

    summary = data.get("summary")
    if isinstance(summary, dict):
        summary.pop("racket_observations_3d", None)
        summary.pop("racket_frames_processed", None)


def draw_scaled_detections(
    out: np.ndarray,
    detections: list[dict],
    x_offset: int,
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
        y1 = int(det["y1"] * scale)
        x2 = int(det["x2"] * scale) + x_offset
        y2 = int(det["y2"] * scale)
        cv2.rectangle(out, (x1, y1), (x2, y2), draw_color, 2)

        if label == "tennis_ball":
            conf_text = f"B {det['conf']:.2f}"
        elif label == "stationary_object":
            conf_text = f"S {det['conf']:.2f}"
        else:
            conf_text = f"{label_prefix}{det['conf']:.2f}"
        cv2.putText(
            out, conf_text, (x1, max(20, y1 - 5)),
            FONT, FONT_SCALE, draw_color, FONT_THICKNESS,
        )

        if draw_center:
            cx = int(det["x"] * scale) + x_offset
            cy = int(det["y"] * scale)
            cv2.drawMarker(
                out, (cx, cy), draw_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )


def annotate_frame(
    img: np.ndarray,
    frame_data: dict,
    serials: list[str],
    n_cams: int,
    panel_w: int,
    *,
    show_racket: bool = False,
) -> np.ndarray:
    """在拼接画面上绘制球/球拍/3D/状态等离线标注。"""
    h, w = img.shape[:2]
    out = img.copy()
    scale = panel_w / (panel_w * 2)

    detections = frame_data.get("detections", {})
    racket_detections = frame_data.get("racket_detections", {})
    tiles = frame_data.get("tiles", {})

    for cam_idx, sn in enumerate(serials):
        color = BOX_COLORS[cam_idx % len(BOX_COLORS)]
        x_offset = cam_idx * panel_w

        draw_scaled_detections(
            out,
            detections.get(sn, []),
            x_offset,
            scale,
            color,
        )

        if show_racket:
            draw_scaled_detections(
                out,
                racket_detections.get(sn, []),
                x_offset,
                scale,
                RACKET_BOX_COLOR,
                draw_center=True,
                label_prefix="R ",
            )

        if sn in tiles:
            tile = tiles[sn]
            tx = int(tile["x"] * scale) + x_offset
            ty = int(tile["y"] * scale)
            tw = int(tile["w"] * scale)
            th = int(tile["h"] * scale)
            cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2)

        car_loc = frame_data.get("car_loc")
        if car_loc and "pixels" in car_loc and sn in car_loc["pixels"]:
            px, py = car_loc["pixels"][sn]
            cx = int(px * scale) + x_offset
            cy = int(py * scale)
            cv2.drawMarker(
                out, (cx, cy), (0, 200, 255),
                cv2.MARKER_DIAMOND, 20, 2,
            )

        cv2.putText(
            out, sn[-3:], (x_offset + 10, h - 15),
            FONT, 1.0, color, 2,
        )

    for i in range(1, n_cams):
        x = panel_w * i
        cv2.line(out, (x, 0), (x, h), (100, 100, 100), 1)

    line_h = 40
    lines: list[tuple[str, tuple[int, int, int]]] = []

    lines.append((
        f"#{frame_data['idx']}  {frame_data.get('exposure_time', '')}  "
        f"lat={frame_data.get('latency_ms', 0):.0f}ms",
        TEXT_COLOR,
    ))

    detection_counts = frame_data.get("detection_counts", {})
    det_parts = []
    for sn in serials:
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
        stationary_count = counts.get("stationary_object", 0)
        tennis_ball_count = counts.get("tennis_ball", 0)
        if stationary_count > 0:
            det_parts.append(f"{sn[-3:]}=b{tennis_ball_count}/s{stationary_count}")
        else:
            det_parts.append(f"{sn[-3:]}=b{tennis_ball_count}")
    lines.append((f"det: {'  '.join(det_parts)}", TEXT_COLOR))

    if show_racket:
        racket_parts = [f"{sn[-3:]}={len(racket_detections.get(sn, []))}" for sn in serials]
        lines.append((f"racket: {'  '.join(racket_parts)}", RACKET_BOX_COLOR))

    ball3d = frame_data.get("ball3d")
    if ball3d:
        cams = "+".join(s[-3:] for s in ball3d["cameras"])
        lines.append((
            f"3D: ({ball3d['x']:.0f}, {ball3d['y']:.0f}, {ball3d['z']:.0f}) mm  "
            f"reproj={ball3d['reproj']:.1f}px  cams={cams}  conf={ball3d['conf']:.2f}",
            TEXT_3D_COLOR,
        ))

    racket3d = frame_data.get("racket3d")
    if show_racket and racket3d:
        cams = "+".join(s[-3:] for s in racket3d["cameras"])
        lines.append((
            f"R3D: ({racket3d['x']:.0f}, {racket3d['y']:.0f}, {racket3d['z']:.0f}) mm  "
            f"reproj={racket3d['reproj']:.1f}px  cams={cams}  conf={racket3d['conf']:.2f}",
            TEXT_RACKET_3D_COLOR,
        ))

    state = frame_data.get("state", "idle")
    state_color = STATE_COLORS.get(state, TEXT_COLOR)
    state_str = f"curve3: {state}"
    pred = frame_data.get("prediction")
    if pred:
        state_str += (
            f"  hit=({pred['x']:.0f}, {pred['y']:.0f}, {pred['z']:.0f}) "
            f"stage={pred['stage']} lead={pred['lead_ms']}ms"
        )
    lines.append((state_str, state_color))

    car_loc = frame_data.get("car_loc")
    if car_loc:
        cams = "+".join(s[-3:] for s in car_loc["cameras_used"])
        lines.append((
            f"car: ({car_loc['x']:.0f}, {car_loc['y']:.0f}, {car_loc['z']:.0f}) mm  "
            f"yaw={math.degrees(car_loc['yaw']):.1f}deg  cams={cams}",
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
    parser.add_argument("--video", default=None, help="原始拼接视频路径，默认与 JSON 同名 .avi")
    parser.add_argument("--output", default=None, help="输出 annotated 视频路径，默认同目录 _annotated.avi")
    parser.add_argument("--json-output", default=None, help="补充后的 JSON 输出路径，默认覆写输入 JSON")
    parser.add_argument("--tracker-config", default=str(_DEFAULT_TRACKER_CONFIG), help="tracker.json 路径")
    parser.add_argument("--racket-model", default=str(_DEFAULT_RACKET_MODEL), help="球拍 YOLO 模型路径")
    parser.add_argument("--racket-conf", type=float, default=0.25, help="球拍检测置信度阈值")
    parser.add_argument("--no-racket", action="store_true", help="只做旧标注，不补充球拍结果")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧，便于快速验证")
    args = parser.parse_args()

    json_path = Path(args.input)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    video_path = Path(args.video) if args.video else json_path.with_suffix(".avi")
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return

    output_path = args.output or str(json_path.with_name(json_path.stem + "_annotated.avi"))
    json_output_path = Path(args.json_output) if args.json_output else json_path

    serials = data["config"]["serials"]
    n_cams = len(serials)
    frames_data = data["frames"]
    tracker_cfg = load_tracker_config(args.tracker_config)
    racket_enabled = not args.no_racket

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = w // n_cams
    frame_mapping, has_exact_mapping = build_video_frame_mapping(data, total)

    if racket_enabled:
        clear_racket_results(data)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"输入视频: {video_path} ({w}x{h}, {fps:.0f}fps, {total} 帧)")
    print(f"JSON 帧数: {len(frames_data)}")
    if has_exact_mapping:
        print(f"视频映射: 使用 JSON video_frame_indices（{len(frame_mapping)} 帧精确对齐）")
    else:
        print("视频映射: JSON 不含 video_frame_indices，退化为按帧号 1:1 对齐（若录制时丢帧，标注可能漂移）")
    print(f"输出视频: {output_path}")
    if racket_enabled:
        print(f"球拍模型: {args.racket_model}")
        print(f"JSON 输出: {json_output_path}")

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
                        tracker_cfg=tracker_cfg,
                        racket_model_path=args.racket_model,
                        conf_threshold=args.racket_conf,
                    )
                    print(
                        f"球拍 YOLO 预热完成: batch={racket_pipeline.engine_batch}, "
                        f"min_cams={racket_pipeline.min_cameras_for_3d}, "
                        f"max_reproj={racket_pipeline.max_reproj_error_px:.1f}px"
                    )

                frame_time = build_panel_timestamp(fd, frame_idx, fps)
                racket_dets, racket_tiles, racket3d = detect_racket_frame(
                    img, serials, frame_time, racket_pipeline
                )
                apply_racket_results(fd, racket_dets, racket_tiles, racket3d)
                racket_frames_processed += 1

                if racket3d is not None:
                    racket_observations.append({
                        "x": racket3d.x,
                        "y": racket3d.y,
                        "z": racket3d.z,
                        "t": frame_time,
                        "reproj_err": racket3d.reprojection_error,
                        "confidence": racket3d.confidence,
                        "cameras_used": racket3d.cameras_used,
                    })

            annotated = annotate_frame(
                img, fd, serials, n_cams, panel_w, show_racket=racket_enabled
            )
            n_annotated += 1
        else:
            annotated = img

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
    writer.release()

    if racket_enabled:
        config = data.setdefault("config", {})
        summary = data.setdefault("summary", {})
        config["racket_model_path"] = str(
            racket_pipeline.detector.model_path if racket_pipeline else args.racket_model
        )
        config["racket_conf_threshold"] = args.racket_conf
        data["racket_observations"] = racket_observations
        summary["racket_observations_3d"] = len(racket_observations)
        summary["racket_frames_processed"] = racket_frames_processed

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"JSON 已更新: {json_output_path}")
        print(f"球拍 3D 观测数: {len(racket_observations)}")

    print(f"完成：{n_annotated} 帧已标注，输出到 {output_path}")


if __name__ == "__main__":
    main()
