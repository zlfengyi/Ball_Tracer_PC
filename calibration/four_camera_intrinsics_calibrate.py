# -*- coding: utf-8 -*-
"""Run checkerboard detection and per-camera intrinsics calibration for the four-camera rig."""

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

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from calibration.detect_corners_v2 import (
    DETECTION_QUALITY_METRIC,
    MIN_PARITY_CONFIDENCE,
    process_image,
)
from calibration.four_camera_calib_common import (
    DEFAULT_INNER_COLS,
    DEFAULT_INNER_ROWS,
    DEFAULT_SQUARE_SIZE_MM,
    FOUR_CAMERA_CALIB_ROOT,
    latest_session_dir,
    load_sync_serials,
    rel_or_abs,
    resolve_session_dir,
)


def _make_obj_points(inner_cols: int, inner_rows: int, square_size: float) -> np.ndarray:
    pts = np.zeros((inner_cols * inner_rows, 3), np.float32)
    for row in range(inner_rows):
        for col in range(inner_cols):
            pts[row * inner_cols + col] = [col * square_size, row * square_size, 0.0]
    return pts


def _sample_evenly(items: list, max_count: int) -> list:
    if max_count <= 0 or len(items) <= max_count:
        return list(items)
    step = len(items) / max_count
    return [items[int(i * step)] for i in range(max_count)]


def _image_path(image_dir: Path, serial: str, index: int) -> Path | None:
    for digits in (4, 3):
        path = image_dir / serial / f"{index:0{digits}d}.png"
        if path.exists():
            return path
    return None


def _count_image_files(image_dir: Path, serials: list[str], range_start: int, range_end: int) -> dict[str, int]:
    totals = {}
    for sn in serials:
        total = 0
        for idx in range(range_start, range_end + 1):
            if _image_path(image_dir, sn, idx) is not None:
                total += 1
        totals[sn] = total
    return totals


def _make_detection_stats(totals: dict[str, int], results: dict[str, dict[str, dict]],
                          score_lists: dict[str, list[float]], score_threshold: float) -> dict[str, dict]:
    stats = {}
    for sn, total in totals.items():
        scores = np.array(score_lists.get(sn, []), dtype=np.float64)
        detected = len(results.get(sn, {}))
        good = int(np.sum(scores >= score_threshold)) if scores.size else 0
        bad = int(np.sum(scores < score_threshold)) if scores.size else 0
        stats[sn] = {
            "total_images": int(total),
            "detected": int(detected),
            "good_frames": int(good),
            "bad_frames": int(bad),
            "detection_rate": float(detected / total) if total > 0 else 0.0,
            "score_mean": float(scores.mean()) if scores.size else 0.0,
            "score_median": float(np.median(scores)) if scores.size else 0.0,
            "score_min": float(scores.min()) if scores.size else 0.0,
            "score_max": float(scores.max()) if scores.size else 0.0,
        }
    return stats


def _print_detection_stats(stats: dict[str, dict], score_threshold: float) -> None:
    print("\nDetection stats:")
    for sn, s in stats.items():
        print(
            f"  {sn}: detected={s['detected']}/{s['total_images']} "
            f"({100.0 * s['detection_rate']:.1f}%), "
            f"good={s['good_frames']}, bad={s['bad_frames']} "
            f"(threshold={score_threshold}), "
            f"score median={s['score_median']:.3f}, mean={s['score_mean']:.3f}"
        )


def _load_cached_detections(cache_path: Path, image_dir: Path, serials: list[str],
                            inner_cols: int, inner_rows: int,
                            range_start: int, range_end: int, score_threshold: float
                            ) -> tuple[dict[str, dict[str, dict]], dict[str, list[float]], dict[str, dict]]:
    with open(cache_path, encoding="utf-8") as inp:
        cache = json.load(inp)

    board = cache.get("board")
    expected_board = {"inner_cols": inner_cols, "inner_rows": inner_rows}
    if not isinstance(board, dict) or any(board.get(key) != value for key, value in expected_board.items()):
        raise RuntimeError(
            f"Detection cache board mismatch: expected {expected_board}, got {board}"
        )
    if cache.get("quality_metric") != DETECTION_QUALITY_METRIC:
        raise RuntimeError(
            "Detection cache quality_metric mismatch: "
            f"expected {DETECTION_QUALITY_METRIC!r}, got {cache.get('quality_metric')!r}"
        )
    cached_min_parity = cache.get("min_parity_confidence")
    if (isinstance(cached_min_parity, bool)
            or not isinstance(cached_min_parity, (int, float))
            or not np.isfinite(float(cached_min_parity))
            or float(cached_min_parity) != MIN_PARITY_CONFIDENCE):
        raise RuntimeError(
            "Detection cache min_parity_confidence mismatch: "
            f"expected {MIN_PARITY_CONFIDENCE}, got {cached_min_parity!r}"
        )

    results = {sn: {} for sn in serials}
    scores = {sn: [] for sn in serials}
    cameras = cache.get("cameras", {})
    if not isinstance(cameras, dict):
        raise RuntimeError("Detection cache cameras must be an object")

    required_fields = {
        "score",
        "parity_sign",
        "reversed_to_canonical",
        "corners",
        "image_size",
    }
    expected_corner_count = inner_cols * inner_rows
    for sn in serials:
        camera_detections = cameras.get(sn)
        if not isinstance(camera_detections, dict):
            raise RuntimeError(f"Detection cache missing camera object: {sn}")

        camera_size = None
        seen_indices = set()
        for idx_s, det in camera_detections.items():
            try:
                idx = int(idx_s)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid frame index for {sn}: {idx_s!r}") from exc
            if idx <= 0 or idx in seen_indices:
                raise RuntimeError(f"Invalid or duplicate frame index for {sn}: {idx_s!r}")
            seen_indices.add(idx)

            image_path = _image_path(image_dir, sn, idx)
            if image_path is None:
                raise RuntimeError(f"Cached detection image is missing: {sn}/{idx}")
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Cached detection image is unreadable: {image_path}")
            actual_size = (int(image.shape[1]), int(image.shape[0]))

            if not isinstance(det, dict):
                raise RuntimeError(f"Cached detection must be an object: {sn}/{idx}")
            missing = required_fields - det.keys()
            if missing:
                raise RuntimeError(
                    f"Cached detection {sn}/{idx} missing fields: {sorted(missing)}"
                )

            score = det["score"]
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not np.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0):
                raise RuntimeError(f"Invalid cached score for {sn}/{idx}: {score!r}")

            parity = det["parity_sign"]
            if (isinstance(parity, bool) or not isinstance(parity, (int, float))
                    or not np.isfinite(float(parity))
                    or float(parity) < MIN_PARITY_CONFIDENCE):
                raise RuntimeError(f"Invalid cached parity_sign for {sn}/{idx}: {parity!r}")
            if not isinstance(det["reversed_to_canonical"], bool):
                raise RuntimeError(
                    f"Invalid cached reversed_to_canonical for {sn}/{idx}: "
                    f"{det['reversed_to_canonical']!r}"
                )

            image_size = det["image_size"]
            if (not isinstance(image_size, list) or len(image_size) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                           for value in image_size)):
                raise RuntimeError(f"Invalid cached image_size for {sn}/{idx}: {image_size!r}")
            detection_size = tuple(image_size)
            if detection_size != actual_size:
                raise RuntimeError(
                    f"Cached image_size mismatch for {sn}/{idx}: "
                    f"cache={detection_size}, image={actual_size}"
                )
            if camera_size is None:
                camera_size = detection_size
            elif detection_size != camera_size:
                raise RuntimeError(
                    f"Inconsistent cached image_size for {sn}: "
                    f"expected {camera_size}, got {detection_size} at frame {idx}"
                )

            try:
                corners = np.asarray(det["corners"], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid cached corners for {sn}/{idx}") from exc
            if corners.shape != (expected_corner_count, 2) or not np.all(np.isfinite(corners)):
                raise RuntimeError(
                    f"Invalid cached corners for {sn}/{idx}: "
                    f"expected shape {(expected_corner_count, 2)}, got {corners.shape}"
                )
            width, height = detection_size
            if (np.any(corners[:, 0] < 0.0) or np.any(corners[:, 0] >= width)
                    or np.any(corners[:, 1] < 0.0) or np.any(corners[:, 1] >= height)):
                raise RuntimeError(f"Cached corners outside image bounds for {sn}/{idx}")

            if idx < range_start or idx > range_end:
                continue
            results[sn][idx_s] = det
            scores[sn].append(float(score))
    totals = _count_image_files(image_dir, serials, range_start, range_end)
    stats = _make_detection_stats(totals, results, scores, score_threshold)
    return results, scores, stats


def _detect_corners(image_dir: Path, serials: list[str], inner_cols: int, inner_rows: int,
                    range_start: int, range_end: int, workers: int,
                    score_threshold: float, recognition_root: Path
                    ) -> tuple[dict[str, dict[str, dict]], dict[str, list[float]], dict[str, dict]]:
    pattern = (inner_cols, inner_rows)
    recognition_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    totals = {}
    for sn in serials:
        det_dir = recognition_root / sn
        det_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for idx in range(range_start, range_end + 1):
            path = _image_path(image_dir, sn, idx)
            if path is not None:
                tasks.append((sn, idx, path, det_dir))
                total += 1
        totals[sn] = total

    if not tasks:
        raise FileNotFoundError(f"No calibration images found under {image_dir}")

    print(f"  Detecting corners from {len(tasks)} images...")
    results = {sn: {} for sn in serials}
    scores = {sn: [] for sn in serials}
    processed = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_image,
                path,
                pattern,
                det_dir,
                inner_cols,
                inner_rows,
                save_failed=False,
            ): (sn, idx)
            for sn, idx, path, det_dir in tasks
        }
        for fut in as_completed(futures):
            sn, idx = futures[fut]
            processed += 1
            try:
                det = fut.result()
            except Exception as exc:
                print(f"  Error on {sn}/{idx:04d}: {exc}")
                det = None
            if det is not None:
                results[sn][str(idx)] = det
                scores[sn].append(float(det["score"]))

            if processed % 100 == 0 or processed == len(tasks):
                elapsed = time.perf_counter() - t0
                speed = processed / elapsed if elapsed > 0 else 0.0
                print(f"  [{processed}/{len(tasks)}] {elapsed:.0f}s, {speed:.1f} img/s")

    stats = _make_detection_stats(totals, results, scores, score_threshold)
    _print_detection_stats(stats, score_threshold)

    bad_dir = recognition_root / "bad_corners"
    if bad_dir.exists():
        shutil.rmtree(bad_dir)

    bad_count = 0
    for sn in serials:
        for idx_s, det in results[sn].items():
            if float(det["score"]) >= score_threshold:
                continue
            idx = int(idx_s)
            target_dir = bad_dir / sn
            target_dir.mkdir(parents=True, exist_ok=True)
            src = _image_path(image_dir, sn, idx)
            if src is not None:
                tag = f"score{float(det['score']):.3f}"
                shutil.copy2(src, target_dir / f"{idx:04d}_{tag}.png")
            src_det = recognition_root / sn / f"{idx:04d}_det.png"
            if not src_det.exists():
                src_det = recognition_root / sn / f"{idx:03d}_det.png"
            if not src_det.exists():
                src_det = recognition_root / sn / f"{idx:04d}_det.jpg"
            if not src_det.exists():
                src_det = recognition_root / sn / f"{idx:03d}_det.jpg"
            if src_det.exists():
                shutil.copy2(src_det, target_dir / src_det.name.replace("_det", f"_score{float(det['score']):.3f}_det"))
                bad_count += 1

    output = {
        "board": {
            "inner_cols": inner_cols,
            "inner_rows": inner_rows,
        },
        "quality_metric": DETECTION_QUALITY_METRIC,
        "min_parity_confidence": MIN_PARITY_CONFIDENCE,
        "cameras": results,
    }
    out_path = image_dir / "corner_detections.json"
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False)
    summary_path = recognition_root / "detection_summary.json"
    completed_perf_counter_s = time.perf_counter()
    with open(summary_path, "w", encoding="utf-8") as out:
        json.dump(
            {
                "completed_perf_counter_s": completed_perf_counter_s,
                "elapsed_s": completed_perf_counter_s - t0,
                "dataset": rel_or_abs(image_dir),
                "recognition_root": rel_or_abs(recognition_root),
                "quality_metric": DETECTION_QUALITY_METRIC,
                "min_parity_confidence": MIN_PARITY_CONFIDENCE,
                "score_threshold": score_threshold,
                "stats": stats,
                "bad_frame_exports": bad_count,
            },
            out,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Corner detections saved to {rel_or_abs(out_path)}")
    print(f"  Recognition results saved to {rel_or_abs(recognition_root)}")
    print(f"  Detection summary saved to {rel_or_abs(summary_path)}")
    print(f"  Bad frames exported: {bad_count}")
    return results, scores, stats


def _coverage_metrics(all_points: np.ndarray, image_size: tuple[int, int]) -> dict:
    width, height = image_size
    x_min = float(np.min(all_points[:, 0]))
    x_max = float(np.max(all_points[:, 0]))
    y_min = float(np.min(all_points[:, 1]))
    y_max = float(np.max(all_points[:, 1]))
    x_pct = 100.0 * (x_max - x_min) / max(width, 1)
    y_pct = 100.0 * (y_max - y_min) / max(height, 1)
    area_pct = x_pct * y_pct / 100.0
    return {
        "x_bounds_px": [x_min, x_max],
        "y_bounds_px": [y_min, y_max],
        "x_coverage_pct": x_pct,
        "y_coverage_pct": y_pct,
        "area_coverage_pct": area_pct,
    }


def _calibrate_one_camera(sn: str, detections: dict[str, dict], obj_pts: np.ndarray,
                          score_threshold: float, max_frames: int,
                          initial_camera: dict | None = None) -> dict:
    usable = []
    for idx_s, det in sorted(detections.items(), key=lambda item: int(item[0])):
        score = float(det["score"])
        if score < score_threshold:
            continue
        usable.append((idx_s, det))

    if len(usable) < 8:
        raise RuntimeError(f"{sn}: not enough usable frames ({len(usable)} < 8)")

    usable = _sample_evenly(usable, max_frames)
    if len(usable) < 8:
        raise RuntimeError(
            f"{sn}: max_frames leaves too few calibration frames ({len(usable)} < 8)"
        )

    sampled_count = len(usable)
    image_size = tuple(int(v) for v in usable[0][1]["image_size"])

    def calibrate(frames, initial):
        obj_list = [obj_pts for _ in frames]
        img_list = [
            np.asarray(det["corners"], dtype=np.float32).reshape(-1, 1, 2)
            for _, det in frames
        ]
        if initial is None:
            result = cv2.calibrateCamera(obj_list, img_list, image_size, None, None)
        else:
            result = cv2.calibrateCamera(
                obj_list,
                img_list,
                image_size,
                np.asarray(initial["K"], dtype=np.float64).reshape(3, 3).copy(),
                np.asarray(initial["D"], dtype=np.float64).ravel().copy(),
                flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            )
        rms, K, D, rvecs, tvecs = result
        frame_rms = []
        for image_points, rvec, tvec in zip(img_list, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            delta = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
            frame_rms.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
        return rms, K, D, rvecs, tvecs, np.asarray(frame_rms, dtype=np.float64)

    rejected_frames = []
    calibration = None
    initial = initial_camera
    for round_number in range(1, 4):
        calibration = calibrate(usable, initial)
        initial = {"K": calibration[1], "D": calibration[2]}
        frame_rms = calibration[-1]
        median_rms = float(np.median(frame_rms))
        mad_rms = float(np.median(np.abs(frame_rms - median_rms)))
        robust_sigma = 1.4826 * mad_rms
        threshold = max(
            1.0,
            median_rms * 3.0,
            median_rms + 4.5 * robust_sigma,
        )
        candidates = np.flatnonzero(frame_rms > threshold)
        reject_budget = len(usable) - 8
        if candidates.size == 0 or reject_budget <= 0:
            break

        reject_positions = sorted(
            candidates.tolist(), key=lambda pos: frame_rms[pos], reverse=True
        )[:reject_budget]
        reject_set = set(reject_positions)
        for pos in reject_positions:
            rejected_frames.append(
                {
                    "index": int(usable[pos][0]),
                    "round": round_number,
                    "frame_rms_px": float(frame_rms[pos]),
                    "threshold_px": float(threshold),
                }
            )
        usable = [frame for pos, frame in enumerate(usable) if pos not in reject_set]
        calibration = None

    if calibration is None:
        calibration = calibrate(usable, initial)
    rms, K, D, rvecs, tvecs, frame_rms_errors = calibration

    numeric_outputs = [
        np.asarray([rms]),
        np.asarray(K),
        np.asarray(D),
        *[np.asarray(value) for value in rvecs],
        *[np.asarray(value) for value in tvecs],
        frame_rms_errors,
    ]
    if not all(np.all(np.isfinite(value)) for value in numeric_outputs):
        raise RuntimeError(f"{sn}: calibrateCamera returned non-finite values")
    if float(K[0, 0]) <= 0.0 or float(K[1, 1]) <= 0.0:
        raise RuntimeError(
            f"{sn}: calibrateCamera returned invalid focal lengths "
            f"fx={K[0, 0]}, fy={K[1, 1]}"
        )

    point_errors = []
    all_points = []
    for (idx_s, det), rvec, tvec in zip(usable, rvecs, tvecs):
        obs = np.array(det["corners"], dtype=np.float32).reshape(-1, 2)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - obs, axis=1)
        point_errors.extend(err.tolist())
        all_points.append(obs)

    all_points_np = np.concatenate(all_points, axis=0)
    coverage = _coverage_metrics(all_points_np, image_size)
    scores = np.array([float(det["score"]) for _, det in usable], dtype=np.float64)
    point_errors_np = np.array(point_errors, dtype=np.float64)

    return {
        "serial": sn,
        "image_size": list(image_size),
        "frames_detected": len(detections),
        "frames_used": len(usable),
        "used_indices": [int(idx_s) for idx_s, _ in usable],
        "score_threshold": score_threshold,
        "rms": float(rms),
        "K": K.tolist(),
        "D": D.ravel().tolist(),
        "diagnostics": {
            "frames_sampled": sampled_count,
            "rejected_frames": rejected_frames,
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "point_error_mean_px": float(point_errors_np.mean()),
            "point_error_median_px": float(np.median(point_errors_np)),
            "point_error_p95_px": float(np.percentile(point_errors_np, 95)),
            "point_error_max_px": float(point_errors_np.max()),
            "frame_rms_mean_px": float(frame_rms_errors.mean()),
            "frame_rms_median_px": float(np.median(frame_rms_errors)),
            "frame_rms_max_px": float(frame_rms_errors.max()),
            "coverage": coverage,
        },
    }


def main() -> None:
    run_started_perf_counter_s = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run four-camera intrinsics calibration.")
    parser.add_argument(
        "--images",
        type=str,
        default="",
        help="Session path or session name. Defaults to the latest data/four_camera_calibration session.",
    )
    parser.add_argument("--serials", type=str, nargs="+", default=None,
                        help="Camera serials to calibrate. Defaults to the current four-camera config.")
    parser.add_argument("--range-start", type=int, default=1)
    parser.add_argument("--range-end", type=int, default=500)
    parser.add_argument("--inner-cols", type=int, default=DEFAULT_INNER_COLS)
    parser.add_argument("--inner-rows", type=int, default=DEFAULT_INNER_ROWS)
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE_MM)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=120,
                        help="Maximum good frames per camera for calibrateCamera (0 means unlimited).")
    parser.add_argument("--redetect", action="store_true",
                        help="Ignore cached corner_detections.json and run detection again.")
    parser.add_argument("--detect-only", action="store_true",
                        help="Only run corner detection and summary generation; skip calibrateCamera.")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path. Defaults to <session>/intrinsics.json.")
    parser.add_argument("--recognition-dir", type=str, default="recognition_results",
                        help="Directory for corner-visualization outputs, relative to the session by default.")
    args = parser.parse_args()

    if 0 < args.max_frames < 8:
        parser.error("--max-frames must be 0 or at least 8")

    image_dir = latest_session_dir() if not args.images else resolve_session_dir(args.images)
    if not image_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {image_dir}")

    serials = args.serials or load_sync_serials()
    output_path = Path(args.output) if args.output else image_dir / "intrinsics.json"
    if not output_path.is_absolute():
        output_path = project_root / output_path
    recognition_dir = Path(args.recognition_dir)
    if not recognition_dir.is_absolute():
        recognition_dir = image_dir / recognition_dir

    print("=" * 60)
    print("      Four-Camera Intrinsics Calibration")
    print("=" * 60)
    print(f"  Dataset: {rel_or_abs(image_dir)}")
    print(f"  Cameras: {serials}")
    print(f"  Range: {args.range_start:04d} ~ {args.range_end:04d}")
    print(f"  Board: {args.inner_cols}x{args.inner_rows} inner corners, {args.square_size}mm")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Max frames per camera: {args.max_frames if args.max_frames > 0 else 'unlimited'}")
    print(f"  Recognition results: {rel_or_abs(recognition_dir)}")
    print(f"  Output: {rel_or_abs(output_path)}")

    cache_path = image_dir / "corner_detections.json"
    if cache_path.exists() and not args.redetect:
        print(f"  Using cached detections: {rel_or_abs(cache_path)}")
        results, detection_scores, detection_stats = _load_cached_detections(
            cache_path,
            image_dir,
            serials,
            args.inner_cols,
            args.inner_rows,
            args.range_start,
            args.range_end,
            args.score_threshold,
        )
        _print_detection_stats(detection_stats, args.score_threshold)
    else:
        results, detection_scores, detection_stats = _detect_corners(
            image_dir=image_dir,
            serials=serials,
            inner_cols=args.inner_cols,
            inner_rows=args.inner_rows,
            range_start=args.range_start,
            range_end=args.range_end,
            workers=args.workers,
            score_threshold=args.score_threshold,
            recognition_root=recognition_dir,
        )

    if args.detect_only:
        print("\nDetection-only mode enabled; skipping intrinsics calibration.")
        return

    obj_pts = _make_obj_points(args.inner_cols, args.inner_rows, args.square_size)
    cameras = {}
    rms_values = []
    used_counts = []
    for sn in serials:
        calib = _calibrate_one_camera(
            sn=sn,
            detections=results.get(sn, {}),
            obj_pts=obj_pts,
            score_threshold=args.score_threshold,
            max_frames=args.max_frames,
        )
        cameras[sn] = calib
        rms_values.append(calib["rms"])
        used_counts.append(calib["frames_used"])
        coverage = calib["diagnostics"]["coverage"]
        rejected_count = len(calib["diagnostics"]["rejected_frames"])
        print(f"  {sn}: rms={calib['rms']:.4f}px  used={calib['frames_used']}  "
              f"rejected={rejected_count}  coverage={coverage['area_coverage_pct']:.1f}%")

    completed_perf_counter_s = time.perf_counter()
    output = {
        "completed_perf_counter_s": completed_perf_counter_s,
        "elapsed_s": completed_perf_counter_s - run_started_perf_counter_s,
        "dataset": rel_or_abs(image_dir),
        "board": {
            "inner_cols": args.inner_cols,
            "inner_rows": args.inner_rows,
            "square_size_mm": args.square_size,
        },
        "selection": {
            "range_start": args.range_start,
            "range_end": args.range_end,
            "score_threshold": args.score_threshold,
            "max_frames": args.max_frames,
        },
        "recognition_results": rel_or_abs(recognition_dir),
        "quality_metric": DETECTION_QUALITY_METRIC,
        "min_parity_confidence": MIN_PARITY_CONFIDENCE,
        "detection_stats": detection_stats,
        "summary": {
            "num_cameras": len(cameras),
            "mean_rms_px": float(np.mean(np.array(rms_values, dtype=np.float64))),
            "min_frames_used": int(min(used_counts)),
            "max_frames_used": int(max(used_counts)),
            "score_samples": {
                sn: len(detection_scores.get(sn, [])) for sn in serials
            },
        },
        "cameras": cameras,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2, ensure_ascii=False)
    print(f"\nSaved intrinsics to {rel_or_abs(output_path)}")


if __name__ == "__main__":
    main()
