# -*- coding: utf-8 -*-
"""
多目相机标定脚本 — 全局 BA 标定并保存结果。

用法:
    python -m calibration.run_calibration
    python -m calibration.run_calibration --serials DA8199285 DA8199402 DA8199243
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from calibration.multi_calibrator import (
    BoardConfig,
    MultiCalibrator,
    MultiCalibResult,
)


def _print_results(r: MultiCalibResult) -> None:
    print()
    print("=" * 60)
    print("              标定结果")
    print("=" * 60)

    print(f"\n  参考相机: {r.reference_serial}")
    print(f"  相机数量: {len(r.cameras)}")

    print(f"\n  重投影误差 (RMS, pixels):")
    print(f"    全局:  {r.total_rms:.4f}")
    for sn, rms in r.per_camera_rms.items():
        print(f"    {sn}: {rms:.4f}")

    print(f"\n  有效图像: {r.num_images}")
    print(f"  观测角点: {r.num_observations}")

    print(f"\n  焦距 (pixels):")
    for sn, cam in r.cameras.items():
        print(f"    {sn}: fx={cam.K[0,0]:.1f}  fy={cam.K[1,1]:.1f}")

    print(f"\n  主点 (pixels):")
    for sn, cam in r.cameras.items():
        print(f"    {sn}: cx={cam.K[0,2]:.1f}  cy={cam.K[1,2]:.1f}")

    print(f"\n  相机间外参 (到参考相机 {r.reference_serial}):")
    for sn, cam in r.cameras.items():
        t = cam.t_to_ref.ravel()
        dist = np.linalg.norm(t)
        print(f"    {sn}: t=[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]  "
              f"dist={dist:.1f}mm")

    print()


def main():
    parser = argparse.ArgumentParser(description="多目相机标定 (全局 BA)")
    parser.add_argument("--images", type=str, default="images",
                        help="标定图片根目录 (默认: images，相对于 calibration/)")
    parser.add_argument("--serials", type=str, nargs="+", default=None,
                        help="相机序列号列表 (默认: 从 camera.json 读取)")
    parser.add_argument("--reference", type=str, default=None,
                        help="参考相机序列号 (默认: slave_serials 中第二个)")
    parser.add_argument("--output", type=str, default="src/config/multi_calib.json",
                        help="输出文件路径 (默认: src/config/multi_calib.json)")
    parser.add_argument("--range-start", type=int, default=1,
                        help="标定图片起始编号 (默认: 1)")
    parser.add_argument("--range-end", type=int, default=500,
                        help="标定图片结束编号 (默认: 500)")
    parser.add_argument("--inner-cols", type=int, default=8,
                        help="棋盘格内角点列数 (默认: 8, 即 9 列方格)")
    parser.add_argument("--inner-rows", type=int, default=11,
                        help="棋盘格内角点行数 (默认: 11, 即 12 行方格)")
    parser.add_argument("--square-size", type=float, default=45.0,
                        help="方格边长 mm (默认: 45.0)")
    parser.add_argument("--annotate", action="store_true",
                        help="保存角点检测标注图")
    parser.add_argument("--fix-intrinsics", action="store_true", default=True,
                        help="固定内参，BA 仅优化外参+板位姿 (默认: True)")
    parser.add_argument("--no-fix-intrinsics", dest="fix_intrinsics",
                        action="store_false",
                        help="BA 同时优化内参+外参+板位姿")
    parser.add_argument("--max-images", type=int, default=50,
                        help="BA 最大使用图片数（0=不限制，默认: 50）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent
    image_dir = calib_root / args.images
    output_path = project_root / args.output

    # 确定相机列表
    if args.serials:
        serials = args.serials
    else:
        # 从 camera.json 读取
        camera_json = project_root / "src" / "config" / "camera.json"
        with open(camera_json, encoding="utf-8") as f:
            cam_cfg = json.load(f)
        serials = cam_cfg.get("slave_serials", [])
        if not serials:
            print("错误: camera.json 中无 slave_serials")
            sys.exit(1)

    # 参考相机：默认 DA8199285（中间相机）
    reference = args.reference
    if reference is None:
        reference = "DA8199285" if "DA8199285" in serials else serials[0]

    # 检查目录
    for sn in serials:
        d = image_dir / sn
        if not d.exists():
            print(f"错误: 图片目录不存在 — {d}")
            sys.exit(1)

    print("=" * 60)
    print("         多目相机标定 (全局 BA)")
    print("=" * 60)
    print(f"  相机: {serials}")
    print(f"  参考: {reference}")
    print(f"  图片目录: {image_dir}")
    print(f"  标定范围: {args.range_start:03d} ~ {args.range_end:03d}")
    print(f"  棋盘格: {args.inner_cols}x{args.inner_rows} 内角点, "
          f"方格 {args.square_size}mm")
    print(f"  固定内参: {args.fix_intrinsics}")
    print(f"  最大图片: {args.max_images if args.max_images > 0 else '不限制'}")
    print(f"  输出: {output_path}")

    board = BoardConfig(
        inner_cols=args.inner_cols,
        inner_rows=args.inner_rows,
        square_size=args.square_size,
    )

    calibrator = MultiCalibrator(
        serials=serials,
        image_dir=image_dir,
        reference_serial=reference,
        board=board,
        image_range=(args.range_start, args.range_end),
        save_annotations=args.annotate,
        fix_intrinsics=args.fix_intrinsics,
        max_images=args.max_images,
    )

    result = calibrator.run()
    _print_results(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"标定结果已保存: {output_path}")


if __name__ == "__main__":
    main()
