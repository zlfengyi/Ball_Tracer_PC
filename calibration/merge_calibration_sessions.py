# -*- coding: utf-8 -*-
"""Merge one or more four-camera calibration sessions into a single dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from calibration.four_camera_calib_common import (
    FOUR_CAMERA_CALIB_ROOT,
    load_sync_serials,
    rel_or_abs,
    resolve_session_dir,
)


def _image_indices(cam_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in sorted(cam_dir.glob("*.png")):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return sorted(indices)


def _load_detection_cache(session_dir: Path) -> dict:
    cache_path = session_dir / "corner_detections.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing corner detection cache: {cache_path}")
    with open(cache_path, encoding="utf-8") as inp:
        return json.load(inp)


def _validate_detection_caches(
    sessions: list[Path],
    serials: list[str],
) -> list[dict]:
    if not sessions:
        raise ValueError("At least one calibration session is required")

    expected_serials = set(serials)
    expected_metadata = None
    caches = []

    for session_dir in sessions:
        cache = _load_detection_cache(session_dir)
        for key in ("board", "quality_metric", "min_parity_confidence", "cameras"):
            if key not in cache:
                raise ValueError(f"{session_dir}: cache is missing {key!r}")

        board = cache["board"]
        if not isinstance(board, dict):
            raise ValueError(f"{session_dir}: cache board must be an object")
        try:
            inner_cols = board["inner_cols"]
            inner_rows = board["inner_rows"]
        except KeyError as exc:
            raise ValueError(f"{session_dir}: cache board is missing {exc.args[0]!r}") from exc
        if (not isinstance(inner_cols, int) or isinstance(inner_cols, bool)
                or not isinstance(inner_rows, int) or isinstance(inner_rows, bool)
                or inner_cols <= 0 or inner_rows <= 0):
            raise ValueError(f"{session_dir}: board dimensions must be positive integers")

        cameras = cache["cameras"]
        if not isinstance(cameras, dict):
            raise ValueError(f"{session_dir}: cache cameras must be an object")
        cache_serials = set(cameras)
        if cache_serials != expected_serials:
            raise ValueError(
                f"{session_dir}: cache camera serials {sorted(cache_serials)} "
                f"do not match requested serials {sorted(expected_serials)}"
            )

        metadata = {
            "board": board,
            "quality_metric": cache["quality_metric"],
            "min_parity_confidence": cache["min_parity_confidence"],
        }
        if expected_metadata is None:
            expected_metadata = metadata
        elif metadata != expected_metadata:
            raise ValueError(f"{session_dir}: cache metadata does not match the first session")

        expected_corner_count = inner_cols * inner_rows
        for sn, detections in cameras.items():
            if not isinstance(detections, dict):
                raise ValueError(f"{session_dir}: detections for {sn} must be an object")
            for frame_idx, detection in detections.items():
                corners = detection.get("corners") if isinstance(detection, dict) else None
                if not isinstance(corners, list) or len(corners) != expected_corner_count:
                    actual = len(corners) if isinstance(corners, list) else "missing"
                    raise ValueError(
                        f"{session_dir}: {sn} frame {frame_idx} has {actual} corners; "
                        f"expected {expected_corner_count}"
                    )

        caches.append(cache)

    return caches


def _link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _session_frame_indices(session_dir: Path, serials: list[str]) -> list[int]:
    expected: set[int] | None = None
    expected_serial = None
    for sn in serials:
        cam_dir = session_dir / sn
        if not cam_dir.exists():
            raise FileNotFoundError(f"Missing camera directory: {cam_dir}")
        indices = set(_image_indices(cam_dir))
        if expected is None:
            expected = indices
            expected_serial = sn
        elif indices != expected:
            raise ValueError(
                f"{session_dir}: image frame sets differ between "
                f"{expected_serial} and {sn}"
            )
    if not expected:
        raise ValueError(f"{session_dir}: no calibration images found")
    return sorted(expected)


def merge_sessions(
    sessions: list[Path],
    output_dir: Path,
    serials: list[str],
    use_copy: bool,
) -> Path:
    started_perf_counter_s = time.perf_counter()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    session_caches = _validate_detection_caches(sessions, serials)

    output_dir.mkdir(parents=True, exist_ok=True)
    for sn in serials:
        (output_dir / sn).mkdir(parents=True, exist_ok=True)

    merged_cache = {
        "board": session_caches[0]["board"],
        "quality_metric": session_caches[0]["quality_metric"],
        "min_parity_confidence": session_caches[0]["min_parity_confidence"],
        "cameras": {sn: {} for sn in serials},
    }
    mapping: list[dict] = []
    next_idx = 1

    for session_dir, session_cache in zip(sessions, session_caches):
        frame_indices = _session_frame_indices(session_dir, serials)
        for src_idx in frame_indices:
            merged_idx = next_idx
            next_idx += 1

            mapping.append(
                {
                    "merged_index": merged_idx,
                    "source_session": rel_or_abs(session_dir),
                    "source_index": src_idx,
                }
            )

            for sn in serials:
                src_img = session_dir / sn / f"{src_idx:04d}.png"
                if not src_img.exists():
                    src_img = session_dir / sn / f"{src_idx:03d}.png"
                if not src_img.exists():
                    raise FileNotFoundError(f"Missing image for {sn} frame {src_idx}: {src_img}")

                dst_img = output_dir / sn / f"{merged_idx:04d}.png"
                _link_or_copy(src_img, dst_img, use_copy=use_copy)

                det = session_cache.get("cameras", {}).get(sn, {}).get(str(src_idx))
                if det is not None:
                    merged_cache["cameras"][sn][str(merged_idx)] = det

    mapping_path = output_dir / "session_sources.json"
    completed_perf_counter_s = time.perf_counter()
    with open(mapping_path, "w", encoding="utf-8") as out:
        json.dump(
            {
                "completed_perf_counter_s": completed_perf_counter_s,
                "elapsed_s": completed_perf_counter_s - started_perf_counter_s,
                "sessions": [rel_or_abs(path) for path in sessions],
                "total_frames": len(mapping),
                "serials": serials,
                "mapping": mapping,
            },
            out,
            indent=2,
            ensure_ascii=False,
        )

    cache_path = output_dir / "corner_detections.json"
    with open(cache_path, "w", encoding="utf-8") as out:
        json.dump(merged_cache, out, ensure_ascii=False)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge four-camera calibration sessions.")
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="Session directories to merge in order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged dataset directory.",
    )
    parser.add_argument(
        "--serials",
        nargs="+",
        default=None,
        help="Camera serials. Defaults to current four-camera config.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating hard links.",
    )
    args = parser.parse_args()

    sessions = [resolve_session_dir(raw) for raw in args.sessions]
    raw_output = Path(args.output)
    if raw_output.is_absolute():
        output_dir = raw_output
    elif raw_output.parts and raw_output.parts[0] == "data":
        output_dir = project_root / raw_output
    elif len(raw_output.parts) > 1:
        output_dir = project_root / raw_output
    else:
        output_dir = FOUR_CAMERA_CALIB_ROOT / raw_output

    if args.serials:
        serials = args.serials
    else:
        serials = load_sync_serials()
        if not serials:
            raise RuntimeError("No camera serials found in camera.json")

    merged_dir = merge_sessions(
        sessions=sessions,
        output_dir=output_dir,
        serials=serials,
        use_copy=args.copy,
    )

    print("Merged calibration sessions:")
    for session_dir in sessions:
        print(f"  - {rel_or_abs(session_dir)}")
    print(f"Output: {rel_or_abs(merged_dir)}")
    print(f"Serials: {serials}")
    print(f"Mode: {'copy' if args.copy else 'hardlink-or-copy'}")


if __name__ == "__main__":
    main()
