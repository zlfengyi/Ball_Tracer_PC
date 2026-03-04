# -*- coding: utf-8 -*-
"""
棋盘格角点检测：独立检测所有图片中的棋盘格角点，保存检测结果和可视化图片。

用法：
  python -m calibration.detect_corners --images images/005_checker_030321
  python -m calibration.detect_corners --images images/005_checker_030321 --range-end 478

输出（保存在图片目录内）：
  {image_dir}/corner_detections.json     — 所有相机所有帧的角点坐标
  {image_dir}/{serial}/det/              — 角点可视化图片
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


def detect_checkerboard(gray: np.ndarray, pattern: tuple[int, int]
                        ) -> np.ndarray | None:
    """检测棋盘格内角点，返回 (N,1,2) float32 或 None。"""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, pattern, None, flags)
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def process_one_image(path: Path, pattern: tuple[int, int],
                      det_dir: Path) -> dict | None:
    """处理单张图片：检测角点 + 保存可视化。返回角点数据或 None。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    corners = detect_checkerboard(img, pattern)
    h, w = img.shape[:2]

    # 保存可视化
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        cv2.drawChessboardCorners(vis, pattern, corners, True)
    else:
        cv2.putText(vis, "NOT DETECTED", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    det_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(det_dir / f"{path.stem}_det.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 80])

    if corners is None:
        return None

    return {
        "corners": corners.reshape(-1, 2).tolist(),
        "image_size": [w, h],
    }


def main():
    parser = argparse.ArgumentParser(description="棋盘格角点检测")
    parser.add_argument("--images", type=str, required=True,
                        help="图片目录 (相对于 calibration/)")
    parser.add_argument("--serials", type=str, nargs="+", default=None,
                        help="相机序列号 (默认: 从 camera.json 读取)")
    parser.add_argument("--range-start", type=int, default=1)
    parser.add_argument("--range-end", type=int, default=500)
    parser.add_argument("--inner-cols", type=int, default=8)
    parser.add_argument("--inner-rows", type=int, default=11)
    parser.add_argument("--workers", type=int, default=4,
                        help="并行线程数 (默认: 4)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent
    image_dir = calib_root / args.images

    # 确定相机列表
    if args.serials:
        serials = args.serials
    else:
        camera_json = project_root / "src" / "config" / "camera.json"
        with open(camera_json, encoding="utf-8") as f:
            cam_cfg = json.load(f)
        serials = cam_cfg.get("slave_serials", [])
        if not serials:
            print("错误: camera.json 中无 slave_serials")
            sys.exit(1)

    pattern = (args.inner_cols, args.inner_rows)
    start, end = args.range_start, args.range_end
    total = end - start + 1

    print(f"=== 棋盘格角点检测 ===")
    print(f"  图片目录: {image_dir}")
    print(f"  相机: {serials}")
    print(f"  范围: {start:03d} ~ {end:03d} ({total} 张)")
    print(f"  棋盘格: {args.inner_cols}x{args.inner_rows}")
    print(f"  并行线程: {args.workers}")
    print()

    # 收集所有任务
    tasks = []  # (serial, idx, path, det_dir)
    for sn in serials:
        det_dir = image_dir / sn / "det"
        for idx in range(start, end + 1):
            path = image_dir / sn / f"{idx:03d}.png"
            if path.exists():
                tasks.append((sn, idx, path, det_dir))

    print(f"  共 {len(tasks)} 张图片待检测")
    print()

    # 并行检测
    results = {}  # {serial: {idx_str: {corners, image_size}}}
    for sn in serials:
        results[sn] = {}

    detected_count = {sn: 0 for sn in serials}
    total_count = {sn: 0 for sn in serials}
    processed = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for sn, idx, path, det_dir in tasks:
            fut = pool.submit(process_one_image, path, pattern, det_dir)
            futures[fut] = (sn, idx)

        for fut in as_completed(futures):
            sn, idx = futures[fut]
            total_count[sn] += 1
            processed += 1

            try:
                det = fut.result()
            except Exception as e:
                print(f"  错误: {sn}/{idx:03d} — {e}")
                det = None

            if det is not None:
                results[sn][str(idx)] = det
                detected_count[sn] += 1

            # 进度
            if processed % 50 == 0 or processed == len(tasks):
                elapsed = time.monotonic() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - processed) / speed if speed > 0 else 0
                total_det = sum(detected_count.values())
                print(f"  [{processed}/{len(tasks)}] "
                      f"{elapsed:.0f}s, {speed:.1f} img/s, "
                      f"ETA {eta:.0f}s, "
                      f"已检测到 {total_det} 个角点帧")

    # 汇总
    elapsed = time.monotonic() - t0
    print(f"\n=== 检测完成 ({elapsed:.1f}s) ===")
    for sn in serials:
        n = detected_count[sn]
        t = total_count[sn]
        pct = 100 * n / t if t > 0 else 0
        print(f"  {sn}: {n}/{t} ({pct:.1f}%)")

    # 保存 JSON
    output = {
        "board": {
            "inner_cols": args.inner_cols,
            "inner_rows": args.inner_rows,
        },
        "cameras": results,
    }
    out_path = image_dir / "corner_detections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\n  检测结果已保存: {out_path}")

    # 文件大小
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
