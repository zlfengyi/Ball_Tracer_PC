# -*- coding: utf-8 -*-
"""Run multi-camera extrinsics calibration with global bundle adjustment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from calibration.four_camera_calib_common import (
    CAMERA_CONFIG_PATH,
    DEFAULT_INTRINSICS_CONFIG_PATH,
    load_camera_config,
    load_sync_serials,
    rel_or_abs,
)
from calibration.multi_calibrator import (
    BoardConfig,
    MultiCalibrator,
    MultiCalibResult,
)


def _print_results(result: MultiCalibResult) -> None:
    print()
    print("=" * 60)
    print("              Calibration Result")
    print("=" * 60)

    print(f"\n  Reference camera: {result.reference_serial}")
    print(f"  Camera count: {len(result.cameras)}")

    print("\n  Reprojection RMS (pixels):")
    print(f"    total: {result.total_rms:.4f}")
    for sn, rms in result.per_camera_rms.items():
        print(f"    {sn}: {rms:.4f}")

    print(f"\n  Valid images: {result.num_images}")
    print(f"  Corner observations: {result.num_observations}")

    print("\n  Extrinsics to reference:")
    for sn, cam in result.cameras.items():
        t = cam.t_to_ref.ravel()
        dist = np.linalg.norm(t)
        print(f"    {sn}: t=[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm  dist={dist:.1f} mm")


def _load_intrinsics(path: Path) -> dict[str, dict[str, np.ndarray | tuple[int, int]]]:
    with open(path, encoding="utf-8") as inp:
        data = json.load(inp)

    cameras = data.get("cameras", {})
    if not cameras:
        raise RuntimeError(f"No cameras found in intrinsics file: {path}")

    intrinsics = {}
    for sn, cam in cameras.items():
        intrinsics[sn] = {
            "K": np.array(cam["K"], dtype=np.float64),
            "D": np.array(cam["D"], dtype=np.float64),
            "image_size": tuple(int(v) for v in cam["image_size"]),
        }
    return intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera calibration (global BA)")
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="Calibration image root directory.",
    )
    parser.add_argument(
        "--serials",
        type=str,
        nargs="+",
        default=None,
        help="Camera serials. Defaults to current four-camera config.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference camera serial. Defaults to current master serial.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/config/multi_calib.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default="",
        help="Optional fixed intrinsics JSON path. Defaults to src/config/four_camera_intrinsics.json when present.",
    )
    parser.add_argument("--range-start", type=int, default=1)
    parser.add_argument("--range-end", type=int, default=500)
    parser.add_argument("--inner-cols", type=int, default=8)
    parser.add_argument("--inner-rows", type=int, default=11)
    parser.add_argument("--square-size", type=float, default=45.0)
    parser.add_argument("--annotate", action="store_true", help="Save checkerboard annotations.")
    parser.add_argument(
        "--fix-intrinsics",
        action="store_true",
        default=True,
        help="Keep intrinsics fixed during BA.",
    )
    parser.add_argument(
        "--no-fix-intrinsics",
        dest="fix_intrinsics",
        action="store_false",
        help="Optimize intrinsics and extrinsics together.",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--min-cameras", type=int, default=2)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.8,
        help="Ignore cached corner detections whose score is below this threshold.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent

    raw_image_dir = Path(args.images)
    if raw_image_dir.is_absolute():
        image_dir = raw_image_dir
    else:
        project_candidate = project_root / raw_image_dir
        image_dir = project_candidate if project_candidate.exists() else calib_root / raw_image_dir

    raw_output = Path(args.output)
    output_path = raw_output if raw_output.is_absolute() else project_root / raw_output

    intrinsics_path = None
    provided_intrinsics = None
    if args.intrinsics:
        raw_intr_path = Path(args.intrinsics)
        intrinsics_path = raw_intr_path if raw_intr_path.is_absolute() else project_root / raw_intr_path
        provided_intrinsics = _load_intrinsics(intrinsics_path)
    elif DEFAULT_INTRINSICS_CONFIG_PATH.exists():
        intrinsics_path = DEFAULT_INTRINSICS_CONFIG_PATH
        provided_intrinsics = _load_intrinsics(intrinsics_path)

    if args.serials:
        serials = args.serials
        cam_cfg = {}
    else:
        cam_cfg = load_camera_config(CAMERA_CONFIG_PATH)
        serials = load_sync_serials(CAMERA_CONFIG_PATH)
        if not serials:
            print("Error: no cameras found in camera.json")
            sys.exit(1)

    reference = args.reference
    if reference is None:
        default_ref = cam_cfg.get("master_serial")
        reference = default_ref if default_ref in serials else serials[0]

    for sn in serials:
        d = image_dir / sn
        if not d.exists():
            print(f"Error: missing image directory {d}")
            sys.exit(1)

    print("=" * 60)
    print("         Multi-Camera Calibration (Global BA)")
    print("=" * 60)
    print(f"  Cameras: {serials}")
    print(f"  Reference: {reference}")
    print(f"  Image dir: {rel_or_abs(image_dir)}")
    print(f"  Range: {args.range_start:03d} ~ {args.range_end:03d}")
    print(f"  Board: {args.inner_cols}x{args.inner_rows} inner corners, {args.square_size} mm")
    print(f"  Fix intrinsics: {args.fix_intrinsics}")
    if intrinsics_path is not None:
        print(f"  Intrinsics input: {intrinsics_path}")
    print(f"  Max images: {args.max_images if args.max_images > 0 else 'unlimited'}")
    print(f"  Min cameras: {args.min_cameras}")
    print(f"  Score threshold: {args.score_threshold:.3f}")
    print(f"  Output: {output_path}")

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
        min_cameras_per_board=args.min_cameras,
        detection_score_threshold=args.score_threshold,
        provided_intrinsics=provided_intrinsics,
    )

    result = calibrator.run()
    _print_results(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved calibration result to {output_path}")


if __name__ == "__main__":
    main()
