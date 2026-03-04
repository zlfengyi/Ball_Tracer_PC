# -*- coding: utf-8 -*-
"""
标定图片采集：在指定时间内匀速拍摄指定数量的同步图片组（硬触发）。

用法：
  python calibration_capture.py [--count 100] [--duration 180] [--content checker]
  python calibration_capture.py --count 600 --duration 180 --exposure 8000 --gain 25 --content checker

输出目录结构（自动命名 {编号}_{内容}_{月日时}）：
  calibration/images/
    001_checker_030318/
      DA8199243/
        001.png  002.png  ...
      DA8199285/
        001.png  002.png  ...
      DA8199402/
        001.png  002.png  ...
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
calib_root = Path(__file__).resolve().parent

from src import SyncCapture, frame_to_numpy


def _auto_output_dir(images_root: Path, content: str) -> Path:
    """自动生成输出目录：{编号}_{内容}_{月日时}"""
    images_root.mkdir(parents=True, exist_ok=True)
    # 扫描现有目录，自动递增编号
    nums = []
    if images_root.exists():
        for d in sorted(images_root.iterdir()):
            if d.is_dir():
                try:
                    nums.append(int(d.name.split("_")[0]))
                except (ValueError, IndexError):
                    pass
    next_num = max(nums, default=0) + 1
    timestamp = datetime.now().strftime("%m%d%H")
    return images_root / f"{next_num:03d}_{content}_{timestamp}"


def main():
    parser = argparse.ArgumentParser(description="标定图片采集（硬触发）")
    parser.add_argument("--count", type=int, default=100, help="拍摄组数（默认 100）")
    parser.add_argument("--duration", type=float, default=180.0, help="总时长秒（默认 180）")
    parser.add_argument("--content", type=str, default="calib", help="拍摄内容描述，用于目录命名")
    parser.add_argument("--output", type=str, default="", help="输出目录（留空则自动生成）")
    parser.add_argument("--exposure", type=float, default=10000, help="曝光时间 us（默认 10000=10ms）")
    parser.add_argument("--gain", type=float, default=25, help="增益 dB（默认 25）")
    args = parser.parse_args()

    images_root = calib_root / "images"
    if args.output:
        output_dir = calib_root / args.output
    else:
        output_dir = _auto_output_dir(images_root, args.content)

    interval = args.duration / args.count

    print(f"=== 标定图片采集（硬触发） ===")
    print(f"  拍摄组数: {args.count}")
    print(f"  总时长:   {args.duration}s")
    print(f"  拍摄间隔: {interval:.1f}s")
    print(f"  输出目录: {output_dir}")
    if args.exposure > 0:
        print(f"  曝光时间: {args.exposure} us")
    if args.gain >= 0:
        print(f"  增益:     {args.gain} dB")
    print()

    print("加载相机配置 (config/camera.json)...")
    overrides = {}
    if args.exposure > 0:
        overrides["exposure_us"] = args.exposure
    if args.gain >= 0:
        overrides["gain_db"] = args.gain
    with SyncCapture.from_config(**overrides) as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")

        # 创建每台相机的子目录
        for sn in sync_sns:
            (output_dir / sn).mkdir(parents=True, exist_ok=True)

        print("等待稳定 (2s)...")
        time.sleep(2.0)

        import cv2
        captured = 0
        print(f"\n开始拍摄（{args.duration}s 内拍 {args.count} 组）...\n")
        t_start = time.monotonic()
        next_capture = t_start

        while captured < args.count:
            now = time.monotonic()
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
                path = output_dir / sn / f"{idx}.png"
                cv2.imwrite(str(path), img)

            remaining = args.duration - elapsed
            print(f"  [{captured}/{args.count}] 已保存  "
                  f"剩余 {remaining:.0f}s  "
                  f"间隔 {interval:.1f}s")

            next_capture += interval

    print(f"\n完成！共拍摄 {captured} 组，保存在 {output_dir}")
    for sn in sync_sns:
        n = len(list((output_dir / sn).glob("*.png")))
        print(f"  {sn}: {n} 张")


if __name__ == "__main__":
    main()
