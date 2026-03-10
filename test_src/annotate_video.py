# -*- coding: utf-8 -*-
"""
离线标注脚本：读取原始视频 + JSON 数据，生成带标注的视频。

用法：
  python test_src/annotate_video.py --input tracker_output/tracker_1280.json
  python test_src/annotate_video.py --input tracker_output/tracker_1280.json --output annotated.avi
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.1
FONT_THICKNESS = 2
BOX_COLORS = [
    (0, 255, 0),       # 绿色 — 1号相机
    (0, 165, 255),     # 橙色 — 2号相机
    (255, 100, 100),   # 蓝色 — 3号相机
]
TEXT_COLOR = (255, 255, 255)
TEXT_3D_COLOR = (0, 255, 255)
STATE_COLORS = {
    "idle":         (128, 128, 128),
    "tracking_s0":  (255, 200, 0),
    "in_landing":   (0, 165, 255),
    "tracking_s1":  (0, 255, 0),
    "done":         (0, 0, 255),
}


def annotate_frame(
    img: np.ndarray,
    frame_data: dict,
    serials: list[str],
    n_cams: int,
    panel_w: int,
) -> np.ndarray:
    """在拼接画面上绘制标注。"""
    h, w = img.shape[:2]
    out = img.copy()

    # 每台相机面板的原始分辨率（从 JSON config 推断）
    sx = panel_w / (w / n_cams)  # 应该是 1.0（已经是半分辨率）
    # 实际上视频已是半分辨率拼接，检测坐标是全分辨率，需要缩放
    scale = panel_w / (panel_w * 2)  # 0.5

    # 绘制检测框
    detections = frame_data.get("detections", {})
    for cam_idx, sn in enumerate(serials):
        color = BOX_COLORS[cam_idx % len(BOX_COLORS)]
        x_offset = cam_idx * panel_w
        dets = detections.get(sn, [])
        for d in dets:
            x1 = int(d["x1"] * scale) + x_offset
            y1 = int(d["y1"] * scale)
            x2 = int(d["x2"] * scale) + x_offset
            y2 = int(d["y2"] * scale)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{d['conf']:.2f}", (x1, y1 - 5),
                        FONT, FONT_SCALE, color, FONT_THICKNESS)

        # 切片区域
        tiles = frame_data.get("tiles", {})
        if sn in tiles:
            t = tiles[sn]
            tx = int(t["x"] * scale) + x_offset
            ty = int(t["y"] * scale)
            tw = int(t["w"] * scale)
            th = int(t["h"] * scale)
            cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2)

        # AprilTag 标记
        car_loc = frame_data.get("car_loc")
        if car_loc and "pixels" in car_loc and sn in car_loc["pixels"]:
            px, py = car_loc["pixels"][sn]
            cx = int(px * scale) + x_offset
            cy = int(py * scale)
            cv2.drawMarker(out, (cx, cy), (0, 200, 255),
                           cv2.MARKER_DIAMOND, 20, 2)

        # 相机标签
        cv2.putText(out, sn[-3:], (x_offset + 10, h - 15),
                    FONT, 1.0, color, 2)

    # 分隔线
    for i in range(1, n_cams):
        x = panel_w * i
        cv2.line(out, (x, 0), (x, h), (100, 100, 100), 1)

    # 文字信息（从底部向上）
    line_h = 40
    lines = []

    lines.append((
        f"#{frame_data['idx']}  {frame_data.get('exposure_time', '')}  "
        f"lat={frame_data.get('latency_ms', 0):.0f}ms",
        TEXT_COLOR,
    ))

    # 检测统计
    det_parts = []
    for sn in serials:
        n = len(detections.get(sn, []))
        det_parts.append(f"{sn[-3:]}={n}")
    lines.append((f"det: {'  '.join(det_parts)}", TEXT_COLOR))

    # 3D 球
    ball3d = frame_data.get("ball3d")
    if ball3d:
        cams = "+".join(s[-3:] for s in ball3d["cameras"])
        lines.append((
            f"3D: ({ball3d['x']:.0f}, {ball3d['y']:.0f}, {ball3d['z']:.0f}) mm  "
            f"reproj={ball3d['reproj']:.1f}px  cams={cams}  conf={ball3d['conf']:.2f}",
            TEXT_3D_COLOR,
        ))

    # 状态
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

    # 小车
    car_loc = frame_data.get("car_loc")
    if car_loc:
        cams = "+".join(s[-3:] for s in car_loc["cameras_used"])
        lines.append((
            f"car: ({car_loc['x']:.0f}, {car_loc['y']:.0f}, {car_loc['z']:.0f}) mm  "
            f"yaw={math.degrees(car_loc['yaw']):.1f}deg  cams={cams}",
            (0, 200, 255),
        ))

    # 从底部向上绘制
    y = h - 15
    for text, color in reversed(lines):
        cv2.putText(out, text, (10, y), FONT, FONT_SCALE, color, FONT_THICKNESS)
        y -= line_h

    return out


def main():
    parser = argparse.ArgumentParser(description="离线标注视频")
    parser.add_argument("--input", required=True, help="JSON 文件路径")
    parser.add_argument("--output", default=None, help="输出视频路径（默认同目录 _annotated.avi）")
    args = parser.parse_args()

    json_path = Path(args.input)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 找到对应的原始视频
    video_path = json_path.with_suffix(".avi")
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return

    output_path = args.output or str(json_path.with_name(
        json_path.stem + "_annotated.avi"))

    serials = data["config"]["serials"]
    n_cams = len(serials)
    frames_data = data["frames"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = w // n_cams

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"输入视频: {video_path} ({w}x{h}, {fps:.0f}fps, {total} 帧)")
    print(f"JSON 帧数: {len(frames_data)}")
    print(f"输出视频: {output_path}")

    frame_idx = 0
    n_annotated = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        if frame_idx < len(frames_data):
            fd = frames_data[frame_idx]
            annotated = annotate_frame(img, fd, serials, n_cams, panel_w)
            n_annotated += 1
        else:
            annotated = img

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total} 帧...")

    cap.release()
    writer.release()
    print(f"完成：{n_annotated} 帧已标注，输出到 {output_path}")


if __name__ == "__main__":
    main()
