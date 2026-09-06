# -*- coding: utf-8 -*-
"""
车辆 AprilTag 3D 实时定位测试（四相机，双 tag 联合拟合）。

从四相机持续采集同步图片，检测车载 AprilTag（id0/id1），联合刚体拟合
车位姿，实时打印结果。Ctrl+C 停止。

用法：
  python test_src/test_car_localizer.py
  python test_src/test_car_localizer.py --field 18        # 18 楼场地
  python test_src/test_car_localizer.py --exposure 10000 --gain 25

注意：18 楼相机正装，ReverseX/Y 已写死在 src/ball_grabber.py，无需再设环境变量
（run_tracker.ps1 里对应 -CameraReverse180 0）。
"""

import argparse
import math
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src import SyncCapture, frame_to_numpy
from src.car_localizer import CarLocalizer, CarLoc


def main():
    parser = argparse.ArgumentParser(
        description="车辆 AprilTag 实时定位（四相机，双 tag）")
    parser.add_argument("--field", default="",
                        help="场地后缀（如 18）；空 = 默认场地")
    parser.add_argument("--exposure", type=float, default=0,
                        help="曝光时间 μs (默认 0=使用 camera_18.json)")
    parser.add_argument("--gain", type=float, default=-1,
                        help="增益 dB (默认 -1=使用 camera_18.json)")
    args = parser.parse_args()

    suffix = f"_{args.field}" if args.field else ""
    config_dir = _PROJECT_ROOT / "src" / "config"
    camera_config = str(config_dir / f"camera{suffix}.json")
    calib_config = str(config_dir / f"four_camera_calib{suffix}.json")

    print("=" * 78)
    print("  四相机车辆 AprilTag 实时定位（双 tag 联合拟合）")
    print("=" * 78)

    print(f"\n[1/2] 初始化 CarLocalizer ({Path(calib_config).name})...")
    localizer = CarLocalizer(calib_config_path=calib_config)
    print(f"  相机: {localizer.serials}")
    print(f"  车载 tag: {localizer.tag_ids}")

    overrides = {}
    if args.exposure > 0:
        overrides["exposure_us"] = args.exposure
    if args.gain >= 0:
        overrides["gain_db"] = args.gain
    print(f"[2/2] 打开同步相机 ({Path(camera_config).name}, "
          f"曝光={args.exposure if args.exposure > 0 else 'default'}μs, "
          f"增益={args.gain if args.gain >= 0 else 'default'}dB)...")
    with SyncCapture.from_config(camera_config, **overrides) as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")
        print("  等待稳定 (1s)...")
        time.sleep(1.0)

        print(f"\n开始实时定位，Ctrl+C 停止...\n")
        print(f"{'帧':>4s}  {'相机':>12s}  {'tags':>6s}  {'X(m)':>8s}  {'Y(m)':>8s}  "
              f"{'yaw°':>7s}  {'yv':>2s}  {'误差':>6s}  {'延迟':>6s}")
        print("-" * 78)

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
                    tags = "+".join(str(t) for t in car_loc.tag_ids)
                    print(
                        f"{frame_count:4d}  {cams:>12s}  {tags:>6s}  "
                        f"{car_loc.x:8.3f}  {car_loc.y:8.3f}  "
                        f"{'    n/a' if car_loc.yaw is None else f'{math.degrees(car_loc.yaw):7.2f}'}  "
                        f"{'Y' if car_loc.yaw_valid else 'n':>2s}  "
                        f"{car_loc.reprojection_error:5.1f}px  "
                        f"{dt_ms:5.1f}ms"
                    )
                else:
                    if frame_count % 30 == 0:
                        print(f"{frame_count:4d}  {'---':>12s}  未检测到")

        except KeyboardInterrupt:
            print(f"\n\n停止。")

    # ── 统计 ──
    print(f"\n{'=' * 78}")
    print(f"  总帧数:   {frame_count}")
    print(f"  成功定位: {success_count} ({100*success_count/max(frame_count,1):.0f}%)")

    if results:
        xs = np.array([r.x for r in results])
        ys = np.array([r.y for r in results])
        yaws = np.degrees(np.array([r.yaw for r in results]))
        errs = np.array([r.reprojection_error for r in results])
        dual = sum(1 for r in results if len(r.tag_ids) >= 2)

        print(f"\n  位姿统计:")
        print(f"    X:   mean={xs.mean():.4f}m  std={xs.std()*1000:.1f}mm")
        print(f"    Y:   mean={ys.mean():.4f}m  std={ys.std()*1000:.1f}mm")
        print(f"    yaw: mean={yaws.mean():+.2f}°  std={yaws.std():.3f}°")
        print(f"    双tag帧: {dual}/{len(results)}  "
              f"yaw_valid 率: {np.mean([r.yaw_valid for r in results]):.2f}")
        print(f"    重投影误差: mean={errs.mean():.2f}px  max={errs.max():.2f}px")

        # 统计每种相机组合出现频次
        combo_count = {}
        for r in results:
            key = "+".join(sorted(s[-3:] for s in r.cameras_used))
            combo_count[key] = combo_count.get(key, 0) + 1
        print(f"\n  相机组合:")
        for combo, cnt in sorted(combo_count.items(), key=lambda x: -x[1]):
            print(f"    {combo}: {cnt} 帧")

    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
