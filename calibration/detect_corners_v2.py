# -*- coding: utf-8 -*-
"""Checkerboard corner detection with color overlays and quality scoring."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

_CANONICAL_PARITY_SIGN = 1.0


def _is_valid_corners(corners: np.ndarray | None, pattern: tuple[int, int]) -> bool:
    return corners is not None and len(corners) == pattern[0] * pattern[1]


def _sample_patch(gray: np.ndarray, pt: np.ndarray, radius: int = 1) -> float:
    x = int(round(float(pt[0])))
    y = int(round(float(pt[1])))
    x = max(radius, min(gray.shape[1] - radius - 1, x))
    y = max(radius, min(gray.shape[0] - radius - 1, y))
    patch = gray[y - radius:y + radius + 1, x - radius:x + radius + 1]
    return float(patch.mean())


def checker_parity_sign(gray: np.ndarray, corners: np.ndarray,
                        cols: int, rows: int) -> float:
    """
    Return a signed checkerboard parity score for the current corner ordering.

    We sample the four quadrants around the first corner in the returned grid.
    Because the board has odd/even square parity, the physically correct board
    origin yields a stable diagonal intensity sign across cameras, while the
    180-degree reversed ordering flips the sign.
    """
    grid = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 2)
    origin = grid[0, 0]
    vx = grid[0, 1] - grid[0, 0]
    vy = grid[1, 0] - grid[0, 0]
    scale = 0.25

    upper_left = _sample_patch(gray, origin - scale * vx - scale * vy)
    upper_right = _sample_patch(gray, origin + scale * vx - scale * vy)
    lower_left = _sample_patch(gray, origin - scale * vx + scale * vy)
    lower_right = _sample_patch(gray, origin + scale * vx + scale * vy)
    return (upper_left + lower_right) - (upper_right + lower_left)


def canonicalize_corner_order(gray: np.ndarray, corners: np.ndarray,
                              cols: int, rows: int) -> tuple[np.ndarray, float, bool]:
    """
    Force a consistent physical checkerboard origin across cameras/sessions.

    The raw detector output is already grid-ordered, but some cameras/sessions
    return the grid with a 180-degree ambiguity. We resolve that ambiguity by
    enforcing a fixed checkerboard parity sign at the first corner.
    """
    parity = checker_parity_sign(gray, corners, cols, rows)
    reversed_to_canonical = False
    canonical = corners.astype(np.float32, copy=True)
    if parity * _CANONICAL_PARITY_SIGN < 0:
        canonical = canonical[::-1].copy()
        parity = checker_parity_sign(gray, canonical, cols, rows)
        reversed_to_canonical = True
    return canonical, float(parity), reversed_to_canonical


def _draw_corner_labels(vis: np.ndarray, corners: np.ndarray,
                        cols: int, rows: int) -> None:
    grid = corners.reshape(rows, cols, 2)
    for row in range(rows):
        for col in range(cols):
            x, y = grid[row, col]
            cv2.putText(
                vis,
                f"{row},{col}",
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """
    Detect checkerboard corners.

    Prefer the SB detector so we avoid the classic adaptive-threshold pipeline
    that can visually shrink dark squares on difficult images. Fall back to the
    classic detector without adaptive-threshold/normalize flags if SB fails.
    """
    corners = None
    method = "none"

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flag_sets = [
            0,
            getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0),
        ]
        for flags in sb_flag_sets:
            try:
                ret, cand = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
            except cv2.error:
                continue
            if ret and _is_valid_corners(cand, pattern):
                corners = cand.astype(np.float32)
                method = "sb"
                break

    if corners is None:
        classic_flag_sets = [
            0,
            cv2.CALIB_CB_FAST_CHECK,
        ]
        for flags in classic_flag_sets:
            ret, cand = cv2.findChessboardCorners(gray, pattern, None, flags)
            if ret and _is_valid_corners(cand, pattern):
                corners = cand
                method = "classic"
                break

    if corners is None:
        return None, method

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.0001,
    )
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    return corners, method


def grid_quality_score(corners: np.ndarray, cols: int, rows: int) -> float:
    """Estimate how regular the detected corner grid is."""
    pts = corners.reshape(-1, 2)
    if len(pts) != rows * cols:
        return 0.0

    grid = pts.reshape(rows, cols, 2)

    row_dists = []
    for r in range(rows):
        for c in range(cols - 1):
            row_dists.append(np.linalg.norm(grid[r, c + 1] - grid[r, c]))

    col_dists = []
    for r in range(rows - 1):
        for c in range(cols):
            col_dists.append(np.linalg.norm(grid[r + 1, c] - grid[r, c]))

    row_dists = np.array(row_dists, dtype=np.float64)
    col_dists = np.array(col_dists, dtype=np.float64)
    if row_dists.size == 0 or col_dists.size == 0:
        return 0.0
    if row_dists.mean() < 1e-6 or col_dists.mean() < 1e-6:
        return 0.0

    row_cv = row_dists.std() / row_dists.mean()
    col_cv = col_dists.std() / col_dists.mean()
    avg_cv = (row_cv + col_cv) / 2.0
    return round(max(0.0, 1.0 - avg_cv * 5.0), 4)


def process_image(
    path: Path,
    pattern: tuple[int, int],
    det_dir: Path | None,
    cols: int,
    rows: int,
    save_failed: bool = True,
) -> dict | None:
    """Detect corners, score them, and optionally save a visualization."""
    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color is None:
        return None

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    corners, method = detect_corners(gray, pattern)
    parity = 0.0
    reversed_to_canonical = False
    if corners is not None:
        corners, parity, reversed_to_canonical = canonicalize_corner_order(
            gray, corners, cols, rows
        )

    if det_dir is not None:
        vis = color.copy()
        if corners is not None:
            score = grid_quality_score(corners, cols, rows)
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            _draw_corner_labels(vis, corners, cols, rows)
            color_code = (0, 255, 0) if score > 0.8 else (0, 165, 255) if score > 0.5 else (0, 0, 255)
            cv2.putText(
                vis,
                f"score={score:.3f}  method={method}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color_code,
                2,
            )
            cv2.putText(
                vis,
                f"parity={parity:.1f}  reversed={reversed_to_canonical}",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color_code,
                2,
            )
            det_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(det_dir / f"{path.stem}_det.png"), vis)
        elif save_failed:
            cv2.putText(
                vis,
                f"NOT DETECTED  method={method}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                3,
            )
            det_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(det_dir / f"{path.stem}_det.png"), vis)

    if corners is None:
        return None

    score = grid_quality_score(corners, cols, rows)
    return {
        "corners": corners.reshape(-1, 2).tolist(),
        "image_size": [w, h],
        "score": float(score),
        "detector": method,
        "parity_sign": float(parity),
        "reversed_to_canonical": bool(reversed_to_canonical),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect checkerboard corners with scoring.")
    parser.add_argument("--images", type=str, required=True, help="Image directory relative to calibration/.")
    parser.add_argument("--serials", type=str, nargs="+", default=None)
    parser.add_argument("--range-start", type=int, default=1)
    parser.add_argument("--range-end", type=int, default=500)
    parser.add_argument("--inner-cols", type=int, default=8)
    parser.add_argument("--inner-rows", type=int, default=11)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent
    image_dir = calib_root / args.images

    if args.serials:
        serials = args.serials
    else:
        camera_json = project_root / "src" / "config" / "camera.json"
        with open(camera_json, encoding="utf-8") as inp:
            cam_cfg = json.load(inp)
        serials = cam_cfg.get("slave_serials", [])

    pattern = (args.inner_cols, args.inner_rows)
    threshold = args.score_threshold

    print("=" * 60)
    print("Checkerboard Corner Detection v2")
    print("=" * 60)
    print(f"  Directory: {image_dir}")
    print(f"  Cameras: {serials}")
    print(f"  Range: {args.range_start:03d}~{args.range_end:03d}")
    print(f"  Board: {args.inner_cols}x{args.inner_rows}")
    print(f"  Threshold: {threshold}")

    tasks = []
    for sn in serials:
        det_dir = image_dir / sn / "det_v2"
        for idx in range(args.range_start, args.range_end + 1):
            path = image_dir / sn / f"{idx:03d}.png"
            if path.exists():
                tasks.append((sn, idx, path, det_dir))

    print(f"  Images: {len(tasks)}")

    results = {sn: {} for sn in serials}
    stats = {
        sn: {"total": 0, "detected": 0, "good": 0, "bad": 0, "scores": []}
        for sn in serials
    }
    processed = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_image, path, pattern, det_dir, args.inner_cols, args.inner_rows): (sn, idx)
            for sn, idx, path, det_dir in tasks
        }
        for fut in as_completed(futures):
            sn, idx = futures[fut]
            stats[sn]["total"] += 1
            processed += 1

            try:
                det = fut.result()
            except Exception as exc:
                print(f"  Error: {sn}/{idx:03d} - {exc}")
                det = None

            if det is not None:
                results[sn][str(idx)] = det
                stats[sn]["detected"] += 1
                score = det["score"]
                stats[sn]["scores"].append(score)
                if score >= threshold:
                    stats[sn]["good"] += 1
                else:
                    stats[sn]["bad"] += 1

            if processed % 100 == 0 or processed == len(tasks):
                elapsed = time.perf_counter() - t0
                speed = processed / elapsed if elapsed > 0 else 0.0
                print(f"  [{processed}/{len(tasks)}] {elapsed:.0f}s, {speed:.1f} img/s")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Detection complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    for sn in serials:
        s = stats[sn]
        scores = np.array(s["scores"]) if s["scores"] else np.array([0.0])
        print(f"\n  {sn}:")
        print(f"    detected: {s['detected']}/{s['total']}")
        print(
            f"    score: median={np.median(scores):.3f} "
            f"mean={np.mean(scores):.3f} min={np.min(scores):.3f}"
        )
        print(f"    good (>= {threshold}): {s['good']}")
        print(f"    bad  (<  {threshold}): {s['bad']}")

    bad_dir = image_dir / "bad_corners"
    if bad_dir.exists():
        shutil.rmtree(bad_dir)

    bad_count = 0
    for sn in serials:
        for idx_s, det in results[sn].items():
            if det["score"] >= threshold:
                continue
            idx = int(idx_s)
            sn_dir = bad_dir / sn
            sn_dir.mkdir(parents=True, exist_ok=True)
            src = image_dir / sn / f"{idx:03d}.png"
            if src.exists():
                tag = f"score{det['score']:.3f}"
                shutil.copy2(src, sn_dir / f"{idx:03d}_{tag}.png")
            src_det = image_dir / sn / "det_v2" / f"{idx:03d}_det.png"
            if src_det.exists():
                shutil.copy2(src_det, sn_dir / f"{idx:03d}_{tag}_det.png")
            bad_count += 1

    print(f"\n  Bad frames: {bad_count}, exported to {bad_dir}")

    output = {
        "board": {"inner_cols": args.inner_cols, "inner_rows": args.inner_rows},
        "cameras": results,
    }
    out_path = image_dir / "corner_detections.json"
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
