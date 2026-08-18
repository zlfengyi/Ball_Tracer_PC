# -*- coding: utf-8 -*-
"""Capture four-camera checkerboard data for intrinsics calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from calibration.four_camera_calib_common import (
    CAMERA_CONFIG_PATH,
    DEFAULT_BOARD_COLS,
    DEFAULT_BOARD_ROWS,
    DEFAULT_SQUARE_SIZE_MM,
    FOUR_CAMERA_CALIB_ROOT,
    create_session_dir,
    load_camera_config,
    load_sync_serials,
    rel_or_abs,
)
from calibration.phone_preview import PhonePreviewServer
from src import SyncCapture, frame_to_numpy


def _build_manifest(session_dir: Path, args, interval_s: float, overrides: dict,
                    expected_serials: list[str], camera_config_path: Path) -> dict:
    board_cols = args.board_cols
    board_rows = args.board_rows
    return {
        "created_perf_counter_s": time.perf_counter(),
        "session_name": session_dir.name,
        "content": args.content,
        "data_root": rel_or_abs(FOUR_CAMERA_CALIB_ROOT),
        "session_dir": rel_or_abs(session_dir),
        "capture": {
            "target_frames": args.count,
            "duration_s": args.duration,
            "interval_s": interval_s,
            "dry_run": bool(args.dry_run),
        },
        "board": {
            "square_cols": board_cols,
            "square_rows": board_rows,
            "inner_cols": board_cols - 1,
            "inner_rows": board_rows - 1,
            "square_size_mm": args.square_size,
        },
        "camera_config": {
            "path": rel_or_abs(camera_config_path),
            "overrides": overrides,
            "expected_serials": expected_serials,
            "snapshot": load_camera_config(camera_config_path),
        },
        "results": {
            "captured_frames": 0,
            "per_camera_counts": {},
            "serials_captured": [],
        },
    }


def _write_manifest(session_dir: Path, manifest: dict) -> None:
    with open(session_dir / "session.json", "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture synchronized checkerboard images.")
    parser.add_argument("--count", type=int, default=500, help="Number of frames to save (default 500).")
    parser.add_argument("--duration", type=float, default=120.0, help="Total capture duration in seconds (default 120).")
    parser.add_argument("--content", type=str, default="chessboard", help="Content tag for the session name.")
    parser.add_argument("--session", type=str, default="", help="Explicit session directory name.")
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE_MM,
                        help="Checkerboard square size in mm (default 45.0).")
    parser.add_argument("--board-cols", type=int, default=DEFAULT_BOARD_COLS,
                        help="Checkerboard square columns (default 9).")
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD_ROWS,
                        help="Checkerboard square rows (default 12).")
    parser.add_argument("--exposure", type=float, default=-1.0, help="Exposure override in microseconds (negative skips override).")
    parser.add_argument("--gain", type=float, default=-1.0, help="Gain override in dB (negative skips override).")
    parser.add_argument("--pixel-format", type=str, default="", help="Pixel format override, e.g. Mono8.")
    parser.add_argument("--camera-config", type=str, default=str(CAMERA_CONFIG_PATH),
                        help="Camera JSON path relative to the project root.")
    parser.add_argument("--dry-run", action="store_true", help="Create the directories without opening cameras.")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be greater than zero")
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    if args.board_cols < 2 or args.board_rows < 2:
        parser.error("--board-cols and --board-rows must both be >= 2")

    raw_camera_config = Path(args.camera_config)
    camera_config_path = (
        raw_camera_config if raw_camera_config.is_absolute()
        else project_root / raw_camera_config
    )
    if not camera_config_path.is_file():
        parser.error(f"camera config does not exist: {camera_config_path}")

    data_root = FOUR_CAMERA_CALIB_ROOT
    session_dir = create_session_dir(data_root, args.content, args.session)

    interval = args.duration / args.count
    overrides = {}
    if args.exposure > 0:
        overrides["exposure_us"] = args.exposure
    if args.gain >= 0:
        overrides["gain_db"] = args.gain
    if args.pixel_format.strip():
        overrides["pixel_format"] = args.pixel_format.strip()

    pre_serials = load_sync_serials(camera_config_path, overrides)
    for sn in pre_serials:
        (session_dir / sn).mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(
        session_dir, args, interval, overrides, pre_serials, camera_config_path
    )
    _write_manifest(session_dir, manifest)

    print("=== Four-Camera Intrinsics Capture ===")
    print(f"  Session: {rel_or_abs(session_dir)}")
    print(f"  Frames: {args.count}")
    print(f"  Duration: {args.duration:.1f}s  Interval: {interval:.2f}s")
    print(f"  Checkerboard squares: {args.board_cols}x{args.board_rows} @ {args.square_size}mm")
    print(f"  Inner corners: {args.board_cols - 1}x{args.board_rows - 1}")
    print(f"  Camera config: {rel_or_abs(camera_config_path)}")
    if args.exposure > 0:
        print(f"  Exposure override: {args.exposure} us")
    if args.gain >= 0:
        print(f"  Gain override: {args.gain} dB")
    if args.pixel_format.strip():
        print(f"  Pixel format override: {args.pixel_format.strip()}")
    print(f"  Data root: {rel_or_abs(data_root)}")

    if args.dry_run:
        print("Dry run flag enabled; skipping camera capture.")
        return

    print("Loading capture configuration...")
    with SyncCapture.from_config(str(camera_config_path), **overrides) as cap:
        sync_serials = cap.sync_serials
        print(f"  Synchronized cameras: {sync_serials}")
        for sn in sync_serials:
            (session_dir / sn).mkdir(parents=True, exist_ok=True)
        manifest["results"]["serials_captured"] = list(sync_serials)
        _write_manifest(session_dir, manifest)

        import cv2

        preview = PhonePreviewServer(sync_serials, args.count)
        preview.start()
        try:
            print("  Phone preview:")
            for url in preview.urls:
                print(f"    {url}")

            print("Warming up (2s)...")
            time.sleep(2.0)

            captured = 0
            t_start = time.perf_counter()
            next_capture = t_start
            print()
            print(f"Beginning capture of {args.count} frames over {args.duration:.1f}s...")

            while captured < args.count:
                frames = cap.get_frames(timeout_s=0.2)
                now = time.perf_counter()
                elapsed = now - t_start

                if elapsed > args.duration + 10:
                    print(f"Timeout after {elapsed:.1f}s ({captured}/{args.count} frames)")
                    break

                save_frame = now >= next_capture
                preview_frame = preview.should_publish(now)
                if frames is None:
                    if save_frame:
                        print(f"  [{captured+1}/{args.count}] timeout waiting for synced frames")
                    continue
                if not save_frame and not preview_frame:
                    continue

                images = {sn: frame_to_numpy(frame) for sn, frame in frames.items()}
                if preview_frame:
                    preview.publish(images, now)
                if not save_frame:
                    continue

                captured += 1
                idx = f"{captured:04d}"
                for sn, image in images.items():
                    path = session_dir / sn / f"{idx}.png"
                    cv2.imwrite(str(path), image)

                preview.update_status(captured, elapsed)
                remaining = max(0.0, args.duration - elapsed)
                print(f"  [{captured}/{args.count}] remaining {remaining:.1f}s  interval {interval:.2f}s")
                next_capture += interval
        finally:
            preview.close()

        print(f"\nCapture complete: {captured} frames stored in {session_dir}")
        manifest["results"]["captured_frames"] = captured
        for sn in sync_serials:
            count = len(list((session_dir / sn).glob("*.png")))
            print(f"  {sn}: {count} images")
            manifest["results"]["per_camera_counts"][sn] = count
        _write_manifest(session_dir, manifest)


if __name__ == "__main__":
    main()
