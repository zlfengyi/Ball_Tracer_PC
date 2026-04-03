# -*- coding: utf-8 -*-
"""
车辆 AprilTag 3D 实时定位测试（四相机）。

从四相机持续采集同步图片，检测 AprilTag，三角测量得到 3D 位置，
实时打印结果。Ctrl+C 停止。

用法：
  python test_src/test_car_localizer.py
  python test_src/test_car_localizer.py --exposure 10000 --gain 25
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from src import SyncCapture, frame_to_numpy
from src.car_localizer import CarLocalizer, CarLoc


def main():
    parser = argparse.ArgumentParser(description="车辆 AprilTag 3D 实时定位（四相机）")
    parser.add_argument("--exposure", type=float, default=0,
                        help="曝光时间 μs (默认 0=使用 camera.json)")
    parser.add_argument("--gain", type=float, default=-1,
                        help="增益 dB (默认 -1=使用 camera.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("  四相机车辆 AprilTag 实时定位")
    print("=" * 60)

    print("\n[1/2] 初始化 CarLocalizer (four_camera_calib.json)...")
    localizer = CarLocalizer()
    print(f"  相机: {localizer.serials}")

    overrides = {}
    if args.exposure > 0:
        overrides["exposure_us"] = args.exposure
    if args.gain >= 0:
        overrides["gain_db"] = args.gain
    print(f"[2/2] 打开同步相机 (曝光={args.exposure if args.exposure > 0 else 'default'}μs, "
          f"增益={args.gain if args.gain >= 0 else 'default'}dB)...")
    with SyncCapture.from_config(**overrides) as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")
        print("  等待稳定 (1s)...")
        time.sleep(1.0)

        print(f"\n开始实时定位，Ctrl+C 停止...\n")
        print(f"{'帧':>4s}  {'相机':>12s}  {'X(m)':>8s}  {'Y(m)':>8s}  {'Z(m)':>8s}  "
              f"{'误差':>6s}  {'延迟':>6s}")
        print("-" * 65)

        frame_count = 0
        success_count = 0
        results: list[CarLoc] = []

        try:
            while True:
                frames = cap.get_frames(timeout_s=1.0)
                if frames is None:
                    continue

                frame_count += 1
                t_pc = time.perf_counter()

                # 解码图像
                images = {}
                for sn, f in frames.items():
                    if sn in localizer.serials:
                        images[sn] = frame_to_numpy(f)

                # 定位
                t0 = time.perf_counter()
                car_loc = localizer.locate(images, t=t_pc)
                dt_ms = (time.perf_counter() - t0) * 1000

                if car_loc is not None:
                    success_count += 1
                    results.append(car_loc)
                    cams = "+".join(s[-3:] for s in car_loc.cameras_used)
                    print(
                        f"{frame_count:4d}  {cams:>12s}  "
                        f"{car_loc.x:8.3f}  {car_loc.y:8.3f}  {car_loc.z:8.3f}  "
                        f"{car_loc.reprojection_error:5.1f}px  "
                        f"{dt_ms:5.1f}ms"
                    )
                else:
                    if frame_count % 30 == 0:
                        print(f"{frame_count:4d}  {'---':>12s}  未检测到")

        except KeyboardInterrupt:
            print(f"\n\n停止。")

    # ── 统计 ──
    print(f"\n{'=' * 60}")
    print(f"  总帧数:   {frame_count}")
    print(f"  成功定位: {success_count} ({100*success_count/max(frame_count,1):.0f}%)")

    if results:
        xs = np.array([r.x for r in results])
        ys = np.array([r.y for r in results])
        zs = np.array([r.z for r in results])
        errs = np.array([r.reprojection_error for r in results])

        print(f"\n  3D 坐标统计 (m):")
        print(f"    X: mean={xs.mean():.4f}  std={xs.std():.4f}")
        print(f"    Y: mean={ys.mean():.4f}  std={ys.std():.4f}")
        print(f"    Z: mean={zs.mean():.4f}  std={zs.std():.4f}")
        print(f"    重投影误差: mean={errs.mean():.2f}px  max={errs.max():.2f}px")

        # 统计每种相机组合出现频次
        combo_count = {}
        for r in results:
            key = "+".join(sorted(r.cameras_used))
            combo_count[key] = combo_count.get(key, 0) + 1
        print(f"\n  相机组合:")
        for combo, cnt in sorted(combo_count.items(), key=lambda x: -x[1]):
            print(f"    {combo}: {cnt} 帧")

    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
