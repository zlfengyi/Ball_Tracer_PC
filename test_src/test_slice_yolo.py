# -*- coding: utf-8 -*-
"""
测试切片 YOLO 推理性能：将 2448x1348 图像切成 3x2=6 片，
每张图 batch=6 一次推理，左右各一次，共 2 次调用。

对比方案：
  A) 原始 TRT  batch=2  全图 imgsz=640
  B) .pt 模型  batch=6  切片 imgsz=640  x2次 (左右各一次)
  C) .pt 模型  batch=2  全图 imgsz=1280
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# ── 参数 ──
COLS, ROWS = 3, 2
OVERLAP = 0.1          # 10% 重叠
IMG_W, IMG_H = 2448, 1348
WARMUP = 10
REPEAT = 50


def make_slices(img: np.ndarray, cols: int, rows: int,
                overlap: float) -> list[dict]:
    """将图像切成 cols x rows 片，返回每片的 crop 和原图偏移。"""
    h, w = img.shape[:2]
    base_w = w / cols
    base_h = h / rows
    pad_w = int(base_w * overlap / 2)
    pad_h = int(base_h * overlap / 2)

    slices = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * base_w) - pad_w)
            y1 = max(0, int(r * base_h) - pad_h)
            x2 = min(w, int((c + 1) * base_w) + pad_w)
            y2 = min(h, int((r + 1) * base_h) + pad_h)
            slices.append({
                "crop": img[y1:y2, x1:x2],
                "offset_x": x1,
                "offset_y": y1,
            })
    return slices


def remap_detections(batch_results, slices_info):
    """将切片检测结果映射回原图坐标。"""
    dets = []
    for i, sl in enumerate(slices_info):
        for d in batch_results[i]:
            dets.append((
                d.x + sl["offset_x"],
                d.y + sl["offset_y"],
                d.confidence,
                d.x2 - d.x1,
                d.y2 - d.y1,
            ))
    return dets


def main():
    from src.ball_detector import BallDetector, BallDetection
    from ultralytics import YOLO

    print("=" * 60)
    print("切片 YOLO 推理性能测试")
    print("=" * 60)

    # ── 初始化 ──
    detector_trt = BallDetector()  # 默认加载 .engine
    print(f"TRT 模型: {detector_trt.model_path}")

    pt_path = "yolo_model/tennis_yolo26_v2_20260203.pt"
    detector_pt = BallDetector(pt_path, half=True)
    print(f"PT  模型: {detector_pt.model_path}")

    # 采集真实图像 or 随机图
    try:
        from src import SyncCapture, frame_to_numpy
        print("\n尝试从相机采集真实图像...")
        with SyncCapture.from_config() as cap:
            frames = cap.get_frames(timeout_s=3)
        img_left = frame_to_numpy(frames[0])
        img_right = frame_to_numpy(frames[1])
        print(f"  采集成功: {img_left.shape}")
    except Exception as e:
        print(f"  无相机，使用随机图像: {e}")
        img_left = np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        img_right = np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)

    # ═══════════════════════════════════════════════════════════
    # 方案 A: TRT engine, 全图 batch=2, imgsz=640
    # ═══════════════════════════════════════════════════════════
    print(f"\n--- 方案 A: TRT 全图 batch=2, imgsz=640 ---")
    for _ in range(WARMUP):
        detector_trt.detect_batch([img_left, img_right])

    times_a = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        res = detector_trt.detect_batch([img_left, img_right])
        t1 = time.perf_counter()
        times_a.append((t1 - t0) * 1000)
    avg_a = sum(times_a) / len(times_a)
    print(f"  推理时间: {avg_a:.1f}ms (avg of {REPEAT})")
    print(f"  检测数: 左={len(res[0])}, 右={len(res[1])}")
    for side, dets in zip(["左", "右"], res):
        for d in dets:
            print(f"    {side}: ({d.x:.0f}, {d.y:.0f}) conf={d.confidence:.3f}"
                  f"  box=({d.x2-d.x1:.0f}x{d.y2-d.y1:.0f})")

    # ═══════════════════════════════════════════════════════════
    # 方案 B: .pt 模型, 切片 batch=6, 左右各一次, 共 2 次推理
    # ═══════════════════════════════════════════════════════════
    print(f"\n--- 方案 B: PT 切片 {COLS}x{ROWS}=6, batch=6 x2次, imgsz=640 ---")
    slices_left = make_slices(img_left, COLS, ROWS, OVERLAP)
    slices_right = make_slices(img_right, COLS, ROWS, OVERLAP)
    print(f"  每片尺寸 (含 {OVERLAP*100:.0f}% 重叠):")
    for i, s in enumerate(slices_left):
        h, w = s["crop"].shape[:2]
        print(f"    片{i}: {w}x{h}  offset=({s['offset_x']}, {s['offset_y']})")

    crops_left = [s["crop"] for s in slices_left]
    crops_right = [s["crop"] for s in slices_right]

    # 预热
    for _ in range(WARMUP):
        detector_pt.detect_batch(crops_left)
        detector_pt.detect_batch(crops_right)

    times_b = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        res_left = detector_pt.detect_batch(crops_left)     # batch=6
        res_right = detector_pt.detect_batch(crops_right)   # batch=6
        t1 = time.perf_counter()
        times_b.append((t1 - t0) * 1000)

    dets_b_left = remap_detections(res_left, slices_left)
    dets_b_right = remap_detections(res_right, slices_right)
    avg_b = sum(times_b) / len(times_b)
    print(f"  推理时间: {avg_b:.1f}ms (2次 batch=6, avg of {REPEAT})")
    print(f"  检测数: 左={len(dets_b_left)}, 右={len(dets_b_right)}")
    for side, dets in [("左", dets_b_left), ("右", dets_b_right)]:
        for d in dets:
            print(f"    {side}: ({d[0]:.0f}, {d[1]:.0f}) conf={d[2]:.3f}"
                  f"  box=({d[3]:.0f}x{d[4]:.0f})")

    # ═══════════════════════════════════════════════════════════
    # 方案 C: .pt 模型, 全图 batch=2, imgsz=1280
    # ═══════════════════════════════════════════════════════════
    print(f"\n--- 方案 C: PT 全图 batch=2, imgsz=1280 ---")
    for _ in range(WARMUP):
        detector_pt._model.predict([img_left, img_right],
                                   conf=0.25, verbose=False, imgsz=1280,
                                   half=True)
    times_c = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        results = detector_pt._model.predict([img_left, img_right],
                                             conf=0.25, verbose=False,
                                             imgsz=1280, half=True)
        t1 = time.perf_counter()
        times_c.append((t1 - t0) * 1000)
    dets_c = [BallDetector._parse_boxes(r) for r in results]
    avg_c = sum(times_c) / len(times_c)
    print(f"  推理时间: {avg_c:.1f}ms (avg of {REPEAT})")
    print(f"  检测数: 左={len(dets_c[0])}, 右={len(dets_c[1])}")
    for side, dets in zip(["左", "右"], dets_c):
        for d in dets:
            print(f"    {side}: ({d.x:.0f}, {d.y:.0f}) conf={d.confidence:.3f}"
                  f"  box=({d.x2-d.x1:.0f}x{d.y2-d.y1:.0f})")

    # ═══════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("汇总对比:")
    print(f"  A) TRT 全图 640  batch=2:           {avg_a:6.1f}ms")
    print(f"  B) PT  切片 640  batch=6 x2:        {avg_b:6.1f}ms")
    print(f"  C) PT  全图 1280 batch=2:           {avg_c:6.1f}ms")
    print(f"\n  30fps 预算: 33.3ms (仅 YOLO 部分)")
    for label, t in [("A", avg_a), ("B", avg_b), ("C", avg_c)]:
        ok = "OK" if t < 33.3 else "超预算!"
        print(f"  {label}: {t:.1f}ms  {ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
