# -*- coding: utf-8 -*-
"""
3目相机采集 + YOLO 检测延迟测试 (Step 8.2)。

流程：
  1. SyncCapture.from_config() 打开 3 目相机（DA8199243/285/402）
  2. YOLO 预热（5 次），自动检测 engine batch size
  3. 采集 N 帧，每帧：GigE传输 → Bayer 解码 → YOLO → 测量各阶段延迟
  4. 保存第一帧 3 张原图到 test_trinocular_output/
  5. 打印各阶段延迟统计

用法：
  python test_src/test_trinocular.py [--frames 30] [--display]
"""

import argparse
import sys
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src import SyncCapture, frame_to_numpy, BallDetector


def _yolo_detect_n(detector, img_list, engine_batch):
    """按 engine 支持的 batch size 拆分调用 YOLO。"""
    if len(img_list) <= engine_batch:
        # 需要凑齐 engine_batch（TensorRT 固定 batch）
        padded = img_list[:]
        while len(padded) < engine_batch:
            padded.append(padded[-1])
        results = detector.detect_batch(padded)
        return results[:len(img_list)]

    # 拆分成多次调用
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
    parser = argparse.ArgumentParser(description="3目相机延迟测试 (Step 8.2)")
    parser.add_argument("--frames", type=int, default=30, help="采集帧数（默认 30）")
    parser.add_argument("--display", action="store_true", help="实时显示拼接画面")
    args = parser.parse_args()

    save_dir = root / "test_trinocular_output"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 初始化 ──
    print("=" * 60)
    print("Step 8.2 — 3目相机采集 + YOLO 延迟测试")
    print("=" * 60)

    print("\n[1/3] 初始化 BallDetector (YOLO)...")
    detector = BallDetector()
    print(f"  模型: {detector.model_path}")

    print("[2/3] 打开 3 目同步相机...")
    with SyncCapture.from_config() as cap:
        sync_sns = cap.sync_serials
        n_cams = len(sync_sns)
        print(f"  同步相机: {sync_sns} ({n_cams} 台)")

        if n_cams < 3:
            print(f"*** 警告: 期望 3 台同步相机，实际 {n_cams} 台 ***")

        print("  等待相机稳定 (1.5s)...")
        time.sleep(1.5)

        # 获取第一帧确认分辨率
        first_frames = cap.get_frames(timeout_s=3.0)
        if first_frames is None:
            print("*** 错误: 无法获取初始帧 ***")
            return 1

        for sn in sync_sns:
            if sn in first_frames:
                f = first_frames[sn]
                print(f"  {sn}: {f.width}x{f.height}")

        # ── YOLO 预热 + 自动检测 batch size ──
        print("\n[3/3] YOLO 预热中...")
        warmup_img = frame_to_numpy(first_frames[sync_sns[0]])

        # 尝试 batch=N，失败则尝试更小的 batch
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

        if engine_batch >= n_cams:
            print(f"  预热完成（batch={engine_batch}，单次推理）")
        else:
            import math
            n_calls = math.ceil(n_cams / engine_batch)
            print(f"  预热完成（engine batch={engine_batch}，"
                  f"需 {n_calls} 次推理处理 {n_cams} 张图）")

        # ── 采集循环 ──
        print(f"\n开始采集 {args.frames} 帧...\n")

        saved_first = False
        transfer_times = []  # GigE 传输耗时（ms）：arrival_perf - exposure_start_pc
        decode_times = []    # Bayer 解码耗时（ms）
        yolo_times = []      # YOLO 推理耗时（ms）
        latencies = []       # 端到端延迟：曝光 → YOLO 完成（ms）
        per_cam_transfer = {sn: [] for sn in sync_sns}
        det_counts = {sn: [] for sn in sync_sns}

        for frame_idx in range(args.frames):
            frames = cap.get_frames(timeout_s=2.0)
            if frames is None:
                print(f"  帧 {frame_idx}: 超时")
                continue

            # GigE 传输时间：每相机 arrival_perf - exposure_start_pc
            for sn in sync_sns:
                if sn in frames:
                    f = frames[sn]
                    xfer_ms = (f.arrival_perf - f.exposure_start_pc) * 1000.0
                    per_cam_transfer[sn].append(xfer_ms)
            # 整组传输时间：最晚到达 - 最早曝光
            arrivals = [frames[sn].arrival_perf for sn in sync_sns if sn in frames]
            exp_starts = [frames[sn].exposure_start_pc for sn in sync_sns if sn in frames]
            transfer_ms = (max(arrivals) - min(exp_starts)) * 1000.0
            transfer_times.append(transfer_ms)

            exposure_pc = sum(exp_starts) / len(exp_starts)

            # Bayer 解码
            t0 = time.perf_counter()
            images = {}
            for sn in sync_sns:
                if sn in frames:
                    images[sn] = frame_to_numpy(frames[sn])
            t1 = time.perf_counter()
            decode_ms = (t1 - t0) * 1000.0
            decode_times.append(decode_ms)

            # YOLO 检测
            img_sns = [sn for sn in sync_sns if sn in images]
            img_list = [images[sn] for sn in img_sns]
            detections_list = _yolo_detect_n(detector, img_list, engine_batch)
            t2 = time.perf_counter()
            yolo_ms = (t2 - t1) * 1000.0
            yolo_times.append(yolo_ms)

            # 端到端延迟 = YOLO 出结果时刻 - 曝光时刻
            latency_ms = (t2 - exposure_pc) * 1000.0
            latencies.append(latency_ms)

            # 记录检测数量
            for i, sn in enumerate(img_sns):
                det_counts[sn].append(len(detections_list[i]))

            # 保存第一帧图片
            if not saved_first:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for sn in sync_sns:
                    if sn in images:
                        path = save_dir / f"{sn}_{ts}.png"
                        cv2.imwrite(str(path), images[sn])
                        print(f"  已保存: {path.name} "
                              f"({images[sn].shape[1]}x{images[sn].shape[0]})")
                saved_first = True
                print()

            # 显示
            if args.display and images:
                imgs = [images[sn] for sn in sync_sns if sn in images]
                h_target = imgs[0].shape[0] // 2
                w_target = imgs[0].shape[1] // 3
                resized = [cv2.resize(img, (w_target, h_target)) for img in imgs]
                stitched = np.hstack(resized)
                cv2.imshow("Trinocular", stitched)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 每 10 帧打印进度
            if (frame_idx + 1) % 10 == 0:
                avg_xfer = np.mean(transfer_times[-10:])
                avg_dec = np.mean(decode_times[-10:])
                avg_yolo = np.mean(yolo_times[-10:])
                avg_lat = np.mean(latencies[-10:])
                print(f"  [{frame_idx+1}/{args.frames}] "
                      f"xfer={avg_xfer:.0f}ms  decode={avg_dec:.1f}ms  "
                      f"yolo={avg_yolo:.1f}ms  total={avg_lat:.0f}ms")

        if args.display:
            cv2.destroyAllWindows()

    # ── 统计 ──
    print(f"\n{'=' * 60}")
    print(f"  采集帧数:     {len(latencies)}/{args.frames}")
    print(f"  同步相机:     {sync_sns}")
    print(f"  YOLO engine:  batch={engine_batch}")

    if latencies:
        def _stat(name, data, unit="ms"):
            print(f"\n  {name}:")
            print(f"    avg={np.mean(data):.1f}{unit}  "
                  f"min={np.min(data):.1f}{unit}  "
                  f"max={np.max(data):.1f}{unit}")

        _stat(f"GigE 传输 (曝光→到达PC, 3台整组)", transfer_times)

        print(f"\n  各相机 GigE 传输:")
        for sn in sync_sns:
            if per_cam_transfer[sn]:
                d = per_cam_transfer[sn]
                print(f"    {sn}: avg={np.mean(d):.1f}ms  "
                      f"min={np.min(d):.1f}ms  max={np.max(d):.1f}ms")

        _stat(f"Bayer 解码 ({n_cams} 张)", decode_times)
        _stat(f"YOLO 推理 ({n_cams} 张)", yolo_times)
        _stat(f"端到端延迟 (曝光→YOLO完成)", latencies)

        print(f"\n  各相机 YOLO 检测数量（平均/帧）:")
        for sn in sync_sns:
            if det_counts[sn]:
                print(f"    {sn}: {np.mean(det_counts[sn]):.2f}")

    print(f"\n  样本图片: {save_dir}/")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
