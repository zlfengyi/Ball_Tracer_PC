# -*- coding: utf-8 -*-
"""
三目网球 3D 定位测试 (Step 8.5)。

拍摄一帧三目图片 → YOLO 检测网球 → 多视图三角测量 → 打印 3D 位置。

用法：
  python test_src/test_ball_localizer.py
  python test_src/test_ball_localizer.py --frames 10
"""

import argparse
import math
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from src import SyncCapture, frame_to_numpy, BallDetector
from src.ball_localizer import BallLocalizer, Ball3D


def _yolo_detect_n(detector, img_list, engine_batch):
    """按 engine 支持的 batch size 拆分调用 YOLO。"""
    if len(img_list) <= engine_batch:
        padded = img_list[:]
        while len(padded) < engine_batch:
            padded.append(padded[-1])
        results = detector.detect_batch(padded)
        return results[:len(img_list)]

    detections_list = []
    for i in range(0, len(img_list), engine_batch):
        batch = img_list[i:i + engine_batch]
        actual_n = len(batch)
        while len(batch) < engine_batch:
            batch.append(batch[-1])
        r = detector.detect_batch(batch)
        detections_list.extend(r[:actual_n])
    return detections_list


def main():
    parser = argparse.ArgumentParser(description="三目网球 3D 定位测试 (Step 8.5)")
    parser.add_argument("--frames", type=int, default=1,
                        help="拍摄帧数（默认 1）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Step 8.5 — 三目网球 3D 定位测试")
    print("=" * 60)

    print("\n[1/3] 初始化 BallDetector (YOLO)...")
    detector = BallDetector()
    print(f"  模型: {detector.model_path}")

    print("[2/3] 初始化 BallLocalizer (multi_calib.json)...")
    localizer = BallLocalizer(detector=detector)
    cam_serials = localizer.serials
    print(f"  相机: {cam_serials}")

    print("[3/3] 打开同步相机...")
    with SyncCapture.from_config() as cap:
        sync_sns = cap.sync_serials
        n_cams = len([sn for sn in cam_serials if sn in sync_sns])
        print(f"  同步相机: {sync_sns} ({n_cams} 台匹配)")

        print("  等待相机稳定 (1s)...")
        time.sleep(1.0)

        # YOLO 预热
        print("  YOLO 预热中...")
        first_frames = cap.get_frames(timeout_s=3.0)
        if first_frames is None:
            print("*** 错误: 无法获取初始帧 ***")
            return 1

        warmup_img = frame_to_numpy(first_frames[sync_sns[0]])
        engine_batch = n_cams
        for try_batch in [n_cams, n_cams - 1, 2, 1]:
            try:
                detector.detect_batch([warmup_img] * try_batch)
                engine_batch = try_batch
                break
            except Exception:
                continue
        for _ in range(4):
            _yolo_detect_n(detector, [warmup_img] * n_cams, engine_batch)
        print(f"  预热完成（engine batch={engine_batch}）")

        # 拍摄
        print(f"\n开始拍摄 {args.frames} 帧...\n")

        for frame_idx in range(args.frames):
            frames = cap.get_frames(timeout_s=2.0)
            if frames is None:
                print(f"  帧 {frame_idx}: 超时")
                continue

            # 解码
            images = {}
            for sn in cam_serials:
                if sn in frames:
                    images[sn] = frame_to_numpy(frames[sn])

            # YOLO 检测
            img_sns = [sn for sn in cam_serials if sn in images]
            img_list = [images[sn] for sn in img_sns]
            t0 = time.perf_counter()
            det_results = _yolo_detect_n(detector, img_list, engine_batch)
            dt_yolo = (time.perf_counter() - t0) * 1000

            # 打印每台相机检测结果
            print(f"帧 {frame_idx}:")
            all_dets = {}
            for sn, dets in zip(img_sns, det_results):
                all_dets[sn] = dets
                if dets:
                    for d in dets:
                        print(f"  {sn}: 网球 ({d.x:.0f}, {d.y:.0f}) "
                              f"conf={d.confidence:.3f}")
                else:
                    print(f"  {sn}: 未检测到")

            # 三角测量
            good_dets = {
                sn: dets[0]
                for sn, dets in all_dets.items()
                if len(dets) == 1
            }

            if len(good_dets) >= 2:
                t1 = time.perf_counter()
                ball3d = localizer.triangulate(good_dets)
                dt_tri = (time.perf_counter() - t1) * 1000

                cams = "+".join(s[-3:] for s in ball3d.cameras_used)
                print(f"  -> 3D: ({ball3d.x:.0f}, {ball3d.y:.0f}, {ball3d.z:.0f}) mm")
                print(f"     相机: {cams}")
                print(f"     重投影误差: {ball3d.reprojection_error:.1f}px")
                print(f"     置信度: {ball3d.confidence:.3f}")
                print(f"     YOLO={dt_yolo:.1f}ms  三角={dt_tri:.1f}ms")
            else:
                print(f"  -> 不足 2 台相机检测到网球（{len(good_dets)} 台）")

            print()

    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
