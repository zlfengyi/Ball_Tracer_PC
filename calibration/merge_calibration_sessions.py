# -*- coding: utf-8 -*-
"""Merge one or more four-camera calibration sessions into a single dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
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
        return {}
    with open(cache_path, encoding="utf-8") as inp:
        return json.load(inp)


def _link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists():
        dst.unlink()
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _session_frame_indices(session_dir: Path, serials: list[str]) -> list[int]:
    common: set[int] | None = None
    for sn in serials:
        cam_dir = session_dir / sn
        if not cam_dir.exists():
            raise FileNotFoundError(f"Missing camera directory: {cam_dir}")
        indices = set(_image_indices(cam_dir))
        common = indices if common is None else (common & indices)
    return sorted(common or [])


def merge_sessions(
    sessions: list[Path],
    output_dir: Path,
    serials: list[str],
    use_copy: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sn in serials:
        (output_dir / sn).mkdir(parents=True, exist_ok=True)

    merged_cache = {
        "board": {},
        "cameras": {sn: {} for sn in serials},
    }
    mapping: list[dict] = []
    next_idx = 1

    for session_dir in sessions:
        session_cache = _load_detection_cache(session_dir)
        if not merged_cache["board"] and session_cache.get("board"):
            merged_cache["board"] = session_cache["board"]

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
    with open(mapping_path, "w", encoding="utf-8") as out:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "sessions": [rel_or_abs(path) for path in sessions],
                "total_frames": len(mapping),
                "serials": serials,
                "mapping": mapping,
            },
            out,
            indent=2,
            ensure_ascii=False,
        )

    if merged_cache["board"]:
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
