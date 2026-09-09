"""Offline-only black-dot triangulation for a raw V04 four-camera session."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_v04_sweet_spot_map as sweet
from run_v04_live_40 import FIXED_CAR, LOCKED_MARKER_OFFSET_TOOL_M


SESSION = sweet.PROJECT_ROOT / "arm_controller_data" / "v04_raw_wide_40_319876855513500"
OUTPUT = SESSION / "offline_sweet_spot_analysis_v2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _load_records() -> list[dict[str, Any]]:
    path = SESSION / "records.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 40 or [int(item["index"]) for item in records] != list(range(40)):
        raise RuntimeError("expected exactly 40 ordered raw records")
    return records


def _load_images(record: dict[str, Any], serials: list[str]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for serial in serials:
        relative = Path(str(record["files"][serial]).replace("\\", "/"))
        path = SESSION / relative
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {path}")
        images[serial] = image
    return images


def _nearest(candidates: list[sweet.MarkerCandidate], anchor: np.ndarray) -> sweet.MarkerCandidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(np.linalg.norm(np.asarray(item.uv) - anchor)))


def _anchor_local_candidates(
    image: np.ndarray, anchor_uv: tuple[float, float]
) -> list[sweet.MarkerCandidate]:
    """Retain dark compact components near FK even when the global top-8 drops one."""
    height, width = image.shape[:2]
    ax, ay = anchor_uv
    radius = 55
    x0 = max(0, int(math.floor(ax - radius)))
    y0 = max(0, int(math.floor(ay - radius)))
    x1 = min(width, int(math.ceil(ax + radius)))
    y1 = min(height, int(math.ceil(ay + radius)))
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.size == 0:
        return []
    background = float(np.median(gray))
    ys, xs = np.indices(gray.shape)
    thresholds = sorted(
        set(float(v) for v in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56]
            + list(np.percentile(gray, [5, 10, 15, 20, 25, 30])))
    )
    found: list[sweet.MarkerCandidate] = []
    for threshold in thresholds:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (gray < threshold).astype(np.uint8), 8
        )
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if not (8 <= area <= 1200 and 3 <= w <= 50 and 3 <= h <= 50):
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            fill = area / float(w * h)
            if aspect > 2.4 or fill < 0.30:
                continue
            component = labels == label
            weights = np.where(component, np.maximum(background - gray, 1.0), 0.0)
            mass = float(weights.sum())
            if mass <= 0.0:
                continue
            u = x0 + float((weights * xs).sum() / mass)
            v = y0 + float((weights * ys).sum() / mass)
            distance = math.hypot(u - ax, v - ay)
            if distance > 45.0:
                continue
            contrast = float(np.median(background - gray[component]))
            found.append(
                sweet.MarkerCandidate(
                    uv=(u, v),
                    score=max(contrast, 1.0) * fill * min(area, 250) / (1.0 + (distance / 25.0) ** 2),
                    area=area,
                    bbox_xywh=(x0 + x, y0 + y, w, h),
                )
            )
    found.sort(key=lambda item: (float(np.linalg.norm(np.asarray(item.uv) - np.asarray(anchor_uv))), -item.score))
    deduped: list[sweet.MarkerCandidate] = []
    for item in found:
        if all(np.linalg.norm(np.subtract(item.uv, old.uv)) > 3.0 for old in deduped):
            deduped.append(item)
        if len(deduped) == 12:
            break
    return deduped


def _combined_candidates(
    image: np.ndarray, anchor_uv: tuple[float, float]
) -> list[sweet.MarkerCandidate]:
    combined = [*_anchor_local_candidates(image, anchor_uv), *sweet.find_marker_candidates(image, anchor_uv)]
    combined.sort(key=lambda item: float(np.linalg.norm(np.asarray(item.uv) - np.asarray(anchor_uv))))
    deduped: list[sweet.MarkerCandidate] = []
    for item in combined:
        if all(np.linalg.norm(np.subtract(item.uv, old.uv)) > 3.0 for old in deduped):
            deduped.append(item)
        if len(deduped) == 12:
            break
    return deduped


def _review_reasons(
    mode: str,
    fit: sweet.MarkerFit | None,
    nearest_distances: dict[str, float],
) -> list[str]:
    if fit is None:
        return ["no_geometry_fit"]
    reasons: list[str] = []
    if mode != "nearest_anchor":
        reasons.append("multi_candidate_fallback")
    if fit.expected_distance_mm > 20.0:
        reasons.append("anchor_3d_distance_gt_20mm")
    if fit.point.max_px > 2.0:
        reasons.append("reproj_max_gt_2px")
    if max(fit.loo_delta_mm.values()) > 7.0:
        reasons.append("loo_max_gt_7mm")
    if max(fit.loo_heldout_px.values()) > 4.0:
        reasons.append("heldout_max_gt_4px")
    if max(nearest_distances.values(), default=math.inf) > 60.0:
        reasons.append("nearest_candidate_gt_60px_from_anchor")
    return reasons


def _render_review(
    index: int,
    images: dict[str, np.ndarray],
    serials: list[str],
    anchors: dict[str, tuple[float, float]],
    candidates: dict[str, list[sweet.MarkerCandidate]],
    fit: sweet.MarkerFit | None,
    status: str,
    reasons: list[str],
) -> None:
    tiles: list[np.ndarray] = []
    for serial in serials:
        image = images[serial].copy()
        anchor = tuple(int(round(v)) for v in anchors[serial])
        cv2.drawMarker(image, anchor, (0, 255, 255), cv2.MARKER_CROSS, 42, 3)
        for candidate in candidates[serial]:
            x, y, w, h = candidate.bbox_xywh
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 80, 0), 2)
        if fit is not None:
            uv = tuple(int(round(v)) for v in fit.pixels[serial])
            cv2.drawMarker(image, uv, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 42, 4)
        cv2.putText(image, serial, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        tiles.append(cv2.resize(image, (1024, 768), interpolation=cv2.INTER_AREA))
    canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    caption = f"point {index:02d} {status} " + (",".join(reasons) if reasons else "auto-pass")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(OUTPUT / "review_overlays" / f"point_{index:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    if OUTPUT.exists():
        existing_files = [path for path in OUTPUT.rglob("*") if path.is_file()]
        if existing_files:
            raise RuntimeError(f"refusing to overwrite existing analysis output: {OUTPUT}")
    (OUTPUT / "review_overlays").mkdir(parents=True, exist_ok=True)

    records = _load_records()
    serials, cameras = sweet.load_camera_models()
    kin, config = sweet._load_v04_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    results: list[dict[str, Any]] = []

    for record in records:
        index = int(record["index"])
        q = np.asarray(record["q_measured_rad"], dtype=np.float64)
        # The compact planar FK deliberately rejects encoder offsets on the three
        # fixed axes. Their commanded pose was zero throughout this session; use
        # measured J2-J4 and the exact fixed-axis contract for the image anchor.
        q_fk = q.copy()
        q_fk[[0, 4, 5]] = 0.0
        expected = sweet._expected_sweet_world_mm(
            q_fk, FIXED_CAR, kin, z_offset, LOCKED_MARKER_OFFSET_TOOL_M
        )
        anchors = {
            serial: tuple(float(v) for v in sweet.project_raw(cameras[serial], expected))
            for serial in serials
        }
        images = _load_images(record, serials)
        candidates = {
            serial: _combined_candidates(images[serial], anchors[serial])
            for serial in serials
        }
        nearest = {
            serial: _nearest(candidates[serial], np.asarray(anchors[serial]))
            for serial in serials
        }
        nearest_distances = {
            serial: (
                float(np.linalg.norm(np.asarray(nearest[serial].uv) - np.asarray(anchors[serial])))
                if nearest[serial] is not None
                else math.inf
            )
            for serial in serials
        }
        selected = {
            serial: ([nearest[serial]] if nearest[serial] is not None else [])
            for serial in serials
        }
        fit = sweet.solve_marker_4cam(
            selected, cameras, expected, sweet.MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM
        )
        mode = "nearest_anchor"
        if fit is None:
            fit = sweet.solve_marker_4cam(
                candidates, cameras, expected, sweet.MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM
            )
            mode = "multi_candidate_fallback"

        reasons = _review_reasons(mode, fit, nearest_distances)
        result: dict[str, Any] = {
            "index": index,
            "planned_x_m": float(record["planned"]["x_m"]),
            "planned_z_model_m": float(record["planned"]["z_model_m"]),
            "q_measured_rad": q.tolist(),
            "q_measured_deg": np.degrees(q).tolist(),
            "q_fk_anchor_rad": q_fk.tolist(),
            "expected_world_m": (expected / 1000.0).tolist(),
            "expected_car_m": sweet._sweet_in_car_m(expected, FIXED_CAR),
            "candidate_counts": {serial: len(candidates[serial]) for serial in serials},
            "nearest_anchor_distance_px": nearest_distances,
            "anchor_uv": anchors,
            "selection_mode": mode,
            "status": "failed" if fit is None else "success",
            "manual_review": bool(reasons),
            "review_reasons": reasons,
            "review_image": f"review_overlays/point_{index:02d}.jpg",
        }
        if fit is not None:
            result.update(
                {
                    "pixels": fit.pixels,
                    "sweet_world_m": (fit.point.xyz_mm / 1000.0).tolist(),
                    "sweet_car_m": sweet._sweet_in_car_m(fit.point.xyz_mm, FIXED_CAR),
                    "reproj_px": fit.point.radial_errors_px,
                    "reproj_rms_px": fit.point.rms_px,
                    "reproj_max_px": fit.point.max_px,
                    "loo_delta_mm": fit.loo_delta_mm,
                    "loo_max_mm": max(fit.loo_delta_mm.values()),
                    "loo_heldout_px": fit.loo_heldout_px,
                    "heldout_max_px": max(fit.loo_heldout_px.values()),
                    "anchor_distance_mm": fit.expected_distance_mm,
                }
            )
        results.append(_jsonable(result))
        _render_review(index, images, serials, anchors, candidates, fit, result["status"], reasons)
        print(
            f"{index:02d} {result['status']} mode={mode} "
            f"anchor_mm={result.get('anchor_distance_mm', math.nan):.2f} "
            f"reproj={result.get('reproj_max_px', math.nan):.2f}px "
            f"review={bool(reasons)}",
            flush=True,
        )

    jsonl_path = OUTPUT / "sweet_spot_results.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in results),
        encoding="utf-8",
    )

    successes = [item for item in results if item["status"] == "success"]
    reviews = [item for item in results if item["manual_review"]]
    with (OUTPUT / "joints_to_sweet_spot.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "index", "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
            "sweet_world_x_m", "sweet_world_y_m", "sweet_world_z_m",
            "sweet_car_x_m", "sweet_car_y_m", "sweet_car_z_m",
            "reproj_rms_px", "reproj_max_px", "loo_max_mm", "heldout_max_px",
            "anchor_distance_mm", "selection_mode", "manual_review", "review_reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            row: dict[str, Any] = {
                "index": item["index"],
                **{f"j{i + 1}_deg": item["q_measured_deg"][i] for i in range(6)},
                "selection_mode": item["selection_mode"],
                "manual_review": item["manual_review"],
                "review_reasons": ";".join(item["review_reasons"]),
            }
            if item["status"] == "success":
                row.update(
                    {
                        **{f"sweet_world_{axis}_m": item["sweet_world_m"][i] for i, axis in enumerate("xyz")},
                        **{f"sweet_car_{axis}_m": item["sweet_car_m"][i] for i, axis in enumerate("xyz")},
                        **{key: item[key] for key in (
                            "reproj_rms_px", "reproj_max_px", "loo_max_mm", "heldout_max_px", "anchor_distance_mm"
                        )},
                    }
                )
            writer.writerow(row)

    def metric(name: str) -> dict[str, float] | None:
        values = np.asarray([float(item[name]) for item in successes], dtype=np.float64)
        if not len(values):
            return None
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    summary = {
        "session": str(SESSION),
        "offline_only": True,
        "input_records": len(records),
        "input_images": len(records) * len(serials),
        "success_count": len(successes),
        "failure_indices": [item["index"] for item in results if item["status"] != "success"],
        "manual_review_count": len(reviews),
        "manual_review_indices": [item["index"] for item in reviews],
        "nearest_anchor_count": sum(item["selection_mode"] == "nearest_anchor" for item in successes),
        "multi_candidate_fallback_count": sum(item["selection_mode"] == "multi_candidate_fallback" for item in successes),
        "metrics": {name: metric(name) for name in (
            "reproj_rms_px", "reproj_max_px", "loo_max_mm", "heldout_max_px", "anchor_distance_mm"
        )},
        "gates": {
            "reproj_max_px": sweet.MARKER_MAX_REPROJ_PX,
            "loo_max_mm": sweet.MARKER_MAX_LOO_MM,
            "heldout_max_px": sweet.MARKER_MAX_HELDOUT_PX,
            "anchor_distance_mm": sweet.MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM,
        },
        "fixed_car": {"x_m": FIXED_CAR.x, "y_m": FIXED_CAR.y, "yaw_rad": FIXED_CAR.yaw},
        "locked_marker_offset_tool_m": LOCKED_MARKER_OFFSET_TOOL_M.tolist(),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(successes) == 40 else 2


if __name__ == "__main__":
    raise SystemExit(main())
