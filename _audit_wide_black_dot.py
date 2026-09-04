from __future__ import annotations

import itertools
import json
import csv
from pathlib import Path

import cv2
import numpy as np

from ArmCalibration.capture_v04_sweet_spot_map import (
    CALIBRATION_PATH,
    VEHICLE_CONFIG_PATH,
    _car_pose,
    _sweet_in_car_m,
    load_camera_models,
    project_raw,
    triangulate_refined,
)


ROOT = Path(r"D:\Ball_Tracer_PC\arm_controller_data\v04_raw_wide_40_319876855513500")
OUTPUT = ROOT / "offline_sweet_spot_analysis_v2" / "dark_rectangle_complete_40"
SERIALS, CAMERAS = load_camera_models()
INITIAL = {
    "DB0260414": np.array([1549.7, 407.8]),
    "DB0260373": np.array([914.5, 395.5]),
    "DB0260405": np.array([1440.7, 385.1]),
    "DB0260378": np.array([1068.3, 514.2]),
}


def candidates(image: np.ndarray, anchor: np.ndarray, radius: int = 180) -> list[tuple[np.ndarray, float, tuple]]:
    """Independent dark-rectangle detector.

    The marker often contains a bright vertical slit. Morphological closing groups
    the two black halves before the rectangle center is measured.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0 = max(0, int(anchor[0] - radius))
    y0 = max(0, int(anchor[1] - radius))
    x1 = min(w, int(anchor[0] + radius + 1))
    y1 = min(h, int(anchor[1] + radius + 1))
    roi = gray[y0:y1, x0:x1]
    found = []
    for threshold in (8, 12, 16, 20, 24, 28, 32):
        raw = (roi < threshold).astype(np.uint8)
        for kw, kh in ((5, 3), (7, 3), (9, 5)):
            closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
            for label in range(1, n):
                x, y, bw, bh, area_closed = (int(v) for v in stats[label])
                if not (18 <= area_closed <= 900 and 5 <= bw <= 45 and 5 <= bh <= 45):
                    continue
                aspect = max(bw, bh) / min(bw, bh)
                if aspect > 2.8:
                    continue
                component = labels[y:y + bh, x:x + bw] == label
                actual = raw[y:y + bh, x:x + bw].astype(bool) & component
                yy, xx = np.nonzero(actual)
                if len(xx) < 12:
                    continue
                # Close can bridge a specular slit, but use the geometric extent
                # of actual black pixels; midpoint is less slit-biased than centroid.
                cx = x0 + x + 0.5 * (float(xx.min()) + float(xx.max()))
                cy = y0 + y + 0.5 * (float(yy.min()) + float(yy.max()))
                fill = len(xx) / float(bw * bh)
                if fill < 0.22:
                    continue
                pad = 8
                ax0, ay0 = max(0, x - pad), max(0, y - pad)
                ax1, ay1 = min(roi.shape[1], x + bw + pad), min(roi.shape[0], y + bh + pad)
                patch = roi[ay0:ay1, ax0:ax1]
                contrast = float(np.median(patch)) - float(np.median(roi[y:y + bh, x:x + bw][actual]))
                if contrast < 3.0:
                    continue
                dist = float(np.linalg.norm(np.array([cx, cy]) - anchor))
                square = np.exp(-0.7 * abs(np.log(bw / bh)))
                score = contrast * fill * square * min(len(xx), 200) / (1.0 + (dist / 90.0) ** 2)
                found.append((np.array([cx, cy]), float(score), (threshold, kw, kh, bw, bh, len(xx), contrast)))
    found.sort(key=lambda x: x[1], reverse=True)
    deduped = []
    for item in found:
        if all(np.linalg.norm(item[0] - old[0]) > 2.5 for old in deduped):
            deduped.append(item)
        if len(deduped) >= 16:
            break
    return deduped


def loo_metrics(pixels):
    full = triangulate_refined({s: tuple(pixels[s]) for s in SERIALS}, CAMERAS)
    delta, held = {}, {}
    for dropped in SERIALS:
        fit = triangulate_refined({s: tuple(pixels[s]) for s in SERIALS if s != dropped}, CAMERAS)
        delta[dropped] = float(np.linalg.norm(fit.xyz_mm - full.xyz_mm))
        held[dropped] = float(np.linalg.norm(project_raw(CAMERAS[dropped], fit.xyz_mm) - pixels[dropped]))
    return full, delta, held


def choose(per_cam, predicted_uv, previous_xyz, max_step):
    proposed = {}
    for a, b in itertools.combinations(SERIALS, 2):
        for ia, ca in enumerate(per_cam[a][:12]):
            for ib, cb in enumerate(per_cam[b][:12]):
                try:
                    seed = triangulate_refined({a: tuple(ca[0]), b: tuple(cb[0])}, CAMERAS).xyz_mm
                except Exception:
                    continue
                if previous_xyz is not None and np.linalg.norm(seed - previous_xyz) > max_step * 1.8:
                    continue
                choice = []
                association = 0.0
                for s in SERIALS:
                    uv = project_raw(CAMERAS[s], seed)
                    ds = [float(np.linalg.norm(c[0] - uv)) for c in per_cam[s]]
                    j = int(np.argmin(ds))
                    if ds[j] > 14.0:
                        break
                    choice.append(j)
                    association += ds[j]
                if len(choice) == 4:
                    proposed[tuple(choice)] = min(proposed.get(tuple(choice), 1e9), association)

    best = None
    for choice, assoc in sorted(proposed.items(), key=lambda x: x[1])[:80]:
        pix = {s: per_cam[s][j][0] for s, j in zip(SERIALS, choice)}
        try:
            fit, delta, held = loo_metrics(pix)
        except Exception:
            continue
        step = 0.0 if previous_xyz is None else float(np.linalg.norm(fit.xyz_mm - previous_xyz))
        if fit.max_px > 5.0 or max(delta.values()) > 30 or max(held.values()) > 18 or step > max_step * 1.25:
            continue
        pred_px = sum(float(np.linalg.norm(pix[s] - predicted_uv[s])) for s in SERIALS)
        # Tracking identity is primary here. Many permanent black structures on
        # the arm can also give excellent four-view geometry.
        score = (pred_px / 4.0, fit.rms_px, max(delta.values()), step, assoc)
        if best is None or score < best[0]:
            best = (score, fit, pix, delta, held, choice)
    return best


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "review_overlays").mkdir()
    records = [json.loads(line) for line in (ROOT / "records.jsonl").read_text().splitlines()]
    from src.car_localizer import CarLocalizer
    car_localizer = CarLocalizer(
        calib_config_path=str(CALIBRATION_PATH),
        vehicle_config_path=str(VEHICLE_CONFIG_PATH),
    )
    previous_xyz = None
    uv_history = {s: [] for s in SERIALS}
    results = []
    for i, record in enumerate(records):
        predicted = {}
        per_cam = {}
        for s in SERIALS:
            if not uv_history[s]:
                predicted[s] = INITIAL[s]
            elif len(uv_history[s]) == 1:
                predicted[s] = uv_history[s][-1]
            else:
                delta = uv_history[s][-1] - uv_history[s][-2]
                # Extrapolation only helps inside a snake row. At row turns it is
                # safer to stay at the last observed position.
                same_row = records[i]["planned"]["grid_z"] == records[i - 1]["planned"]["grid_z"]
                predicted[s] = uv_history[s][-1] + (delta if same_row else 0.0)
            image = cv2.imread(str(ROOT / record["files"][s]))
            per_cam[s] = candidates(image, predicted[s])
            if not per_cam[s]:
                raise RuntimeError(f"point {i} {s}: no candidates")
        max_step = 90.0 if i and records[i]["planned"]["grid_z"] == records[i - 1]["planned"]["grid_z"] else 210.0
        selected = choose(per_cam, predicted, previous_xyz, max_step)
        if selected is None:
            counts = {s: len(per_cam[s]) for s in SERIALS}
            raise RuntimeError(f"point {i}: no 4cam fit, candidates={counts}")
        score, fit, pix, delta, held, choice = selected
        for s in SERIALS:
            uv_history[s].append(pix[s])
        previous_xyz = fit.xyz_mm
        item = {
            "index": i,
            "xyz_mm": [float(x) for x in fit.xyz_mm],
            "pixels": {s: [float(x) for x in pix[s]] for s in SERIALS},
            "rms_px": float(fit.rms_px),
            "max_px": float(fit.max_px),
            "max_loo_mm": max(delta.values()),
            "max_heldout_px": max(held.values()),
            "per_camera": {
                s: {
                    "pixel_uv": [float(x) for x in pix[s]],
                    "reproj_error_px": float(fit.radial_errors_px[s]),
                    "loo_delta_mm_when_dropped": float(delta[s]),
                    "heldout_error_px_when_dropped": float(held[s]),
                    "candidate_meta": list(per_cam[s][j][2]),
                }
                for s, j in zip(SERIALS, choice)
            },
            "candidate_meta": {s: per_cam[s][j][2] for s, j in zip(SERIALS, choice)},
        }
        images = {s: cv2.imread(str(ROOT / record["files"][s])) for s in SERIALS}
        car = _car_pose(images, car_localizer)
        item["q_measured_rad"] = [float(x) for x in record["q_measured_rad"]]
        item["q_measured_deg"] = [float(x) for x in record["q_measured_deg"]]
        item["world_xyz_m"] = [float(x / 1000.0) for x in fit.xyz_mm]
        item["car_xyz_m"] = [float(x) for x in _sweet_in_car_m(fit.xyz_mm, car)]
        item["car_pose"] = {
            "x_m": float(car.x), "y_m": float(car.y), "yaw_rad": float(car.yaw),
            "reprojection_error_px": float(car.reprojection_error),
        }
        item["planned"] = record["planned"]
        item["quality"] = "pass" if (
            fit.max_px <= 3.5 and max(delta.values()) <= 15.0 and max(held.values()) <= 6.0
        ) else "review"
        item["review_recommended"] = bool(
            fit.max_px > 2.5 or max(delta.values()) > 10.0 or max(held.values()) > 4.0
        )
        results.append(item)
        print(json.dumps(item, separators=(",", ":")))
    review_indices = [x["index"] for x in results if x["review_recommended"]]
    overlay_indices = sorted(set([0, 20, 39] + review_indices))
    for i in overlay_indices:
        record = records[i]
        tiles = []
        for s in SERIALS:
            image = cv2.imread(str(ROOT / record["files"][s]))
            uv = tuple(int(round(x)) for x in results[i]["pixels"][s])
            cv2.circle(image, uv, 24, (0, 255, 0), 4)
            cv2.drawMarker(image, uv, (0, 255, 255), cv2.MARKER_CROSS, 42, 3)
            cv2.putText(image, f"p{i:02d} {s}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            tiles.append(cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA))
        canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(str(OUTPUT / "review_overlays" / f"point_{i:02d}_4cam.png"), canvas)

    jsonl_path = OUTPUT / "sweet_spot_complete_40.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_fields = ["index", "quality", "review_recommended", "planned_x_m", "planned_z_model_m"]
    csv_fields += [f"j{i}_{unit}" for unit in ("rad", "deg") for i in range(1, 7)]
    csv_fields += [f"world_{a}_m" for a in "xyz"] + [f"world_{a}_mm" for a in "xyz"]
    csv_fields += [f"car_{a}_m" for a in "xyz"]
    for s in SERIALS:
        csv_fields += [f"{s}_u_px", f"{s}_v_px", f"{s}_reproj_px"]
    csv_fields += ["reproj_rms_px", "reproj_max_px", "loo_max_mm", "heldout_max_px"]
    with (OUTPUT / "joints_to_sweet_spot_complete_40.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for x in results:
            row = {
                "index": x["index"], "quality": x["quality"],
                "review_recommended": x["review_recommended"],
                "planned_x_m": x["planned"]["x_m"],
                "planned_z_model_m": x["planned"]["z_model_m"],
                "reproj_rms_px": x["rms_px"], "reproj_max_px": x["max_px"],
                "loo_max_mm": x["max_loo_mm"], "heldout_max_px": x["max_heldout_px"],
            }
            for unit, values in (("rad", x["q_measured_rad"]), ("deg", x["q_measured_deg"])):
                row.update({f"j{i}_{unit}": v for i, v in enumerate(values, 1)})
            row.update({f"world_{a}_m": v for a, v in zip("xyz", x["world_xyz_m"])})
            row.update({f"world_{a}_mm": v for a, v in zip("xyz", x["xyz_mm"])})
            row.update({f"car_{a}_m": v for a, v in zip("xyz", x["car_xyz_m"])})
            for s in SERIALS:
                row[f"{s}_u_px"], row[f"{s}_v_px"] = x["pixels"][s]
                row[f"{s}_reproj_px"] = x["per_camera"][s]["reproj_error_px"]
            writer.writerow(row)

    summary = {
        "n": len(results),
        "pass_count": sum(x["quality"] == "pass" for x in results),
        "review_count": sum(x["quality"] == "review" for x in results),
        "review_recommended_indices": review_indices,
        "overlay_indices": overlay_indices,
        "gates": {"reproj_max_px": 3.5, "loo_max_mm": 15.0, "heldout_max_px": 6.0},
        "method": "independent dark-rectangle morphological closing plus four-camera geometry and temporal tracking",
        "rms_mean": float(np.mean([x["rms_px"] for x in results])),
        "rms_p95": float(np.percentile([x["rms_px"] for x in results], 95)),
        "rms_max": float(np.max([x["rms_px"] for x in results])),
        "loo_max": float(np.max([x["max_loo_mm"] for x in results])),
        "heldout_max": float(np.max([x["max_heldout_px"] for x in results])),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Independent complete-40 black sweet-spot analysis\n\n"
        "The raw PNGs and first strict analysis were not modified. The detector closes the central "
        "specular slit before measuring the full black rectangle center. Quality gates: max reprojection "
        "3.5 px, leave-one-out 15 mm, held-out 6 px. Every input row is retained and labelled.\n",
        encoding="utf-8",
    )
    print("SUMMARY", json.dumps(summary, separators=(",", ":")))
    print("OUTPUT", OUTPUT)


if __name__ == "__main__":
    main()
