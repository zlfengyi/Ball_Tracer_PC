# -*- coding: utf-8 -*-
"""
地面标定图片采集：拍摄标定板平放地面的同步图片对。

曝光时间自动设为 camera.json 中的 3 倍，以保证地面图像亮度充足。

用法：
  python ground_capture.py [--count 30] [--duration 60] [--output calibration_images]

输出：
  calibration_images/
    DA8199285/
      ground_001.png
      ground_002.png
      ...
    DA8199402/
      ground_001.png
      ground_002.png
      ...
"""

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
calib_root = Path(__file__).resolve().parent

from src import SyncCapture, frame_to_numpy

# 曝光倍数
EXPOSURE_MULTIPLIER = 4.5
GAIN_DIVISOR = 2


def main():
    parser = argparse.ArgumentParser(description="地面标定图片采集")
    parser.add_argument("--count", type=int, default=1,
                        help="拍摄组数（默认 1）")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="总时长秒（默认 10）")
    parser.add_argument("--output", type=str, default="images",
                        help="输出目录 (相对于 calibration/)")
    args = parser.parse_args()

    output_dir = calib_root / args.output
    interval = args.duration / args.count

    # 读取配置中的曝光时间
    config_path = project_root / "src" / "config" / "camera.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base_exposure = cfg.get("exposure_us", 1800.0)
    base_gain = cfg.get("gain_db", 20.0)
    ground_exposure = base_exposure * EXPOSURE_MULTIPLIER
    ground_gain = base_gain / GAIN_DIVISOR

    print(f"=== 地面标定图片采集 ===")
    print(f"  拍摄组数: {args.count}")
    print(f"  总时长:   {args.duration}s")
    print(f"  拍摄间隔: {interval:.1f}s")
    print(f"  曝光时间: {base_exposure:.0f} -> {ground_exposure:.0f} us (x{EXPOSURE_MULTIPLIER})")
    print(f"  增益:     {base_gain:.1f} -> {ground_gain:.1f} dB (/{GAIN_DIVISOR})")
    print(f"  输出目录: {output_dir}")
    print()

    print("加载相机配置 (config/camera.json)...")
    with SyncCapture.from_config(exposure_us=ground_exposure, gain_db=ground_gain) as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")

        # 创建每台相机的子目录
        for sn in sync_sns:
            (output_dir / sn).mkdir(parents=True, exist_ok=True)

        print("等待稳定 (2s)...")
        time.sleep(2.0)

        import cv2
        captured = 0
        print(f"\n开始拍摄（{args.duration}s 内拍 {args.count} 组）...")
        print("请将标定板平放在地面上，保持不动。\n")
        t_start = time.perf_counter()
        next_capture = t_start

        while captured < args.count:
            now = time.perf_counter()
            elapsed = now - t_start

            # 超时保护
            if elapsed > args.duration + 10:
                print(f"\n超时，已拍 {captured}/{args.count} 组")
                break

            # 还没到下一次拍摄时间，先排空帧保持同步
            if now < next_capture:
                cap.get_frames(timeout_s=0.05)
                time.sleep(0.01)
                continue

            # 拍摄
            frames = cap.get_frames(timeout_s=2.0)
            if frames is None:
                print(f"  [{captured+1}/{args.count}] 超时，重试...")
                continue

            captured += 1
            idx = f"{captured:03d}"
            for sn, f in frames.items():
                img = frame_to_numpy(f)
                path = output_dir / sn / f"ground_{idx}.png"
                cv2.imwrite(str(path), img)

            remaining = args.duration - elapsed
            print(f"  [{captured}/{args.count}] 已保存  "
                  f"剩余 {remaining:.0f}s  "
                  f"间隔 {interval:.1f}s")

            next_capture += interval

    print(f"\n完成！共拍摄 {captured} 组，保存在 {output_dir}")
    for sn in sync_sns:
        n = len(list((output_dir / sn).glob("ground_*.png")))
        print(f"  {sn}: {n} 张")


if __name__ == "__main__":
    main()
