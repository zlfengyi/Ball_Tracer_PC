# -*- coding: utf-8 -*-
"""
车辆 AprilTag 3D 定位测试。

从双目相机采集同步图片，检测 AprilTag，三角测量得到 3D 位置，
打印结果并保存标注图片。

用法：
  python test_src/test_car_localizer.py [--count 10] [--interval 0.5]
                                         [--output test_car_output]
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
from src.car_localizer import CarLocalizer, CarDetection, CarLoc


def draw_apriltag(
    image: np.ndarray,
    det: CarDetection,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """在图像上绘制 AprilTag 角点、中心和 ID。"""
    corners = det.corners.astype(int)
    # 画四边形
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, 2)
    # 画中心
    cx, cy = int(det.cx), int(det.cy)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    # 标 ID
    cv2.putText(
        image, f"id={det.tag_id}",
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )


def main():
    parser = argparse.ArgumentParser(description="车辆 AprilTag 3D 定位测试")
    parser.add_argument("--count", type=int, default=10, help="采集帧数 (默认 10)")
    parser.add_argument("--interval", type=float, default=0.5, help="采集间隔秒 (默认 0.5)")
    parser.add_argument("--output", type=str, default="test_car_output", help="输出目录")
    parser.add_argument("--exposure", type=float, default=24000.0,
                        help="曝光时间 μs (默认 24000，比球追踪的 1800 亮很多)")
    parser.add_argument("--gain", type=float, default=10.0,
                        help="增益 dB (默认 10)")
    args = parser.parse_args()

    output_dir = _PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 6.2 — 车辆 AprilTag 3D 定位测试")
    print("=" * 60)

    print("\n[1/3] 初始化 CarLocalizer...")
    localizer = CarLocalizer()
    serial_left = localizer.serial_left
    serial_right = localizer.serial_right
    print(f"  左相机: {serial_left}, 右相机: {serial_right}")

    print(f"[2/3] 打开同步相机 (曝光={args.exposure}μs, 增益={args.gain}dB)...")
    with SyncCapture.from_config(exposure_us=args.exposure, gain_db=args.gain) as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")

        if serial_left not in sync_sns or serial_right not in sync_sns:
            print(f"*** 错误: 标定相机 {serial_left}/{serial_right} "
                  f"不在同步列表 {sync_sns} 中 ***")
            return 1

        print("  等待稳定 (1s)...")
        time.sleep(1.0)

        print(f"\n[3/3] 开始采集 {args.count} 帧 (间隔 {args.interval}s)...\n")

        results: list[CarLoc] = []
        last_img_left = None
        last_img_right = None
        last_det_left: list[CarDetection] = []
        last_det_right: list[CarDetection] = []

        for i in range(args.count):
            frames = cap.get_frames(timeout_s=2.0)
            if frames is None:
                print(f"  [{i+1}/{args.count}] 超时")
                continue

            t_pc = time.perf_counter()
            img_left = frame_to_numpy(frames[serial_left])
            img_right = frame_to_numpy(frames[serial_right])

            dets_left = localizer.detect(img_left)
            dets_right = localizer.detect(img_right)

            last_img_left = img_left
            last_img_right = img_right
            last_det_left = dets_left
            last_det_right = dets_right

            # 按 tag_id 匹配
            ids_right = {d.tag_id: d for d in dets_right}
            matched = False
            for d1 in dets_left:
                if d1.tag_id in ids_right:
                    d2 = ids_right[d1.tag_id]
                    car_loc = localizer.triangulate(d1, d2, t=t_pc)
                    results.append(car_loc)
                    matched = True
                    print(
                        f"  [{i+1}/{args.count}] tag={car_loc.tag_id}  "
                        f"pos=({car_loc.x:.0f}, {car_loc.y:.0f}, {car_loc.z:.0f}) mm  "
                        f"reproj={car_loc.reprojection_error:.1f}px"
                    )

            if not matched:
                nl, nr = len(dets_left), len(dets_right)
                ids_l = [d.tag_id for d in dets_left]
                ids_r = [d.tag_id for d in dets_right]
                print(
                    f"  [{i+1}/{args.count}] 未匹配  "
                    f"L={nl} {ids_l}  R={nr} {ids_r}"
                )

            if i < args.count - 1:
                # 排空帧直到下次采集
                t_next = time.monotonic() + args.interval
                while time.monotonic() < t_next:
                    cap.get_frames(timeout_s=0.05)
                    time.sleep(0.01)

    # ── 保存标注图片 ──
    if last_img_left is not None:
        annotated_left = last_img_left.copy()
        annotated_right = last_img_right.copy()
        for d in last_det_left:
            draw_apriltag(annotated_left, d, (0, 255, 0))
        for d in last_det_right:
            draw_apriltag(annotated_right, d, (0, 165, 255))

        stitched = np.hstack([annotated_left, annotated_right])
        half = cv2.resize(stitched, (stitched.shape[1] // 2, stitched.shape[0] // 2))
        save_path = output_dir / "car_detection.png"
        cv2.imwrite(str(save_path), half)
        print(f"\n标注图片已保存: {save_path}")

    # ── 统计 ──
    print(f"\n{'=' * 60}")
    print(f"  总帧数:      {args.count}")
    print(f"  成功定位:    {len(results)}")

    if results:
        xs = np.array([r.x for r in results])
        ys = np.array([r.y for r in results])
        zs = np.array([r.z for r in results])
        errs = np.array([r.reprojection_error for r in results])

        print(f"\n  3D 坐标统计 (mm):")
        print(f"    X: mean={xs.mean():.1f}  std={xs.std():.1f}  "
              f"range=[{xs.min():.1f}, {xs.max():.1f}]")
        print(f"    Y: mean={ys.mean():.1f}  std={ys.std():.1f}  "
              f"range=[{ys.min():.1f}, {ys.max():.1f}]")
        print(f"    Z: mean={zs.mean():.1f}  std={zs.std():.1f}  "
              f"range=[{zs.min():.1f}, {zs.max():.1f}]")
        print(f"    重投影误差: mean={errs.mean():.2f}px  "
              f"max={errs.max():.2f}px")

    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
