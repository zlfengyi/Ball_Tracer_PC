#!/usr/bin/env python3
"""Compare four-camera black-marker observations with full V04 FK for one static probe."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import _audit_wide_black_dot as dark
from ArmCalibration import capture_v04_sweet_spot_map as sweet
from src.car_localizer import CarLocalizer


SESSION = PROJECT_ROOT / "arm_controller_data" / "v04_visual_tcp_probe_87599693444200"
OUTPUT = SESSION / "offline_fk_visual_analysis_v1"


def _world_from_car(point_car_m: np.ndarray, car: Any) -> np.ndarray:
    c = math.cos(float(car.yaw))
    s = math.sin(float(car.yaw))
    x, y, z = (float(value) for value in point_car_m)
    return np.asarray(
        [float(car.x) + c * x - s * y, float(car.y) + s * x + c * y, z],
        dtype=np.float64,
    )


def _metric(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=1).tolist() if len(values) > 1 else np.zeros(values.shape[1]).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def _render_overlay(
    path: Path,
    images: dict[str, np.ndarray],
    predicted_uv: dict[str, np.ndarray],
    pixels: dict[str, np.ndarray],
    label: str,
) -> None:
    tiles: list[np.ndarray] = []
    for serial in dark.SERIALS:
        image = images[serial].copy()
        predicted = tuple(int(round(value)) for value in predicted_uv[serial])
        selected = tuple(int(round(value)) for value in pixels[serial])
        cv2.drawMarker(image, predicted, (0, 255, 255), cv2.MARKER_CROSS, 42, 3)
        cv2.circle(image, selected, 28, (0, 255, 0), 4)
        cv2.drawMarker(image, selected, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 44, 3)
        cv2.putText(
            image, serial, (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
        )
        tiles.append(cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA))
    canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write {path}")


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {OUTPUT}")
    overlays = OUTPUT / "review_overlays"
    overlays.mkdir(parents=True)

    records = [
        json.loads(line)
        for line in (SESSION / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 16:
        raise RuntimeError(f"expected 16 records, got {len(records)}")

    serials, cameras = sweet.load_camera_models()
    if serials != dark.SERIALS:
        raise RuntimeError("camera serial order changed")
    kin, config = sweet._load_v04_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    car_localizer = CarLocalizer(
        calib_config_path=str(sweet.CALIBRATION_PATH),
        vehicle_config_path=str(sweet.VEHICLE_CONFIG_PATH),
    )

    previous_visual_world_mm: np.ndarray | None = None
    previous_fk_world_mm: np.ndarray | None = None
    results: list[dict[str, Any]] = []

    for record in records:
        images = {
            serial: cv2.imread(str(SESSION / record["files"][serial]))
            for serial in serials
        }
        if any(image is None for image in images.values()):
            raise RuntimeError(f"{record['trial_id']}: failed to load four images")

        car = sweet._car_pose(images, car_localizer)
        q = np.asarray(record["joint_at_exposure"]["q_rad"], dtype=np.float64)
        fk_car_model_m = np.asarray(kin.fk(q)["tcp"], dtype=np.float64)
        fk_car_ground_m = fk_car_model_m.copy()
        fk_car_ground_m[2] -= z_offset
        fk_world_m = _world_from_car(fk_car_ground_m, car)
        fk_world_mm = 1000.0 * fk_world_m

        predicted_xyz_mm = (
            fk_world_mm
            if previous_visual_world_mm is None
            else previous_visual_world_mm + fk_world_mm - previous_fk_world_mm
        )
        predicted_uv = {
            serial: dark.project_raw(cameras[serial], predicted_xyz_mm)
            for serial in serials
        }
        per_camera = {
            serial: dark.candidates(images[serial], predicted_uv[serial], radius=180)
            for serial in serials
        }
        if any(not candidates for candidates in per_camera.values()):
            counts = {serial: len(per_camera[serial]) for serial in serials}
            raise RuntimeError(f"{record['trial_id']}: missing black-marker candidates {counts}")

        selected = dark.choose(
            per_camera,
            predicted_uv,
            previous_visual_world_mm,
            max_step=110.0,
        )
        if selected is None:
            counts = {serial: len(per_camera[serial]) for serial in serials}
            raise RuntimeError(f"{record['trial_id']}: no four-camera black-marker fit {counts}")
        _score, fit, pixels, loo_delta, heldout, choice = selected
        previous_visual_world_mm = fit.xyz_mm.copy()
        previous_fk_world_mm = fk_world_mm.copy()

        visual_car_m = np.asarray(sweet._sweet_in_car_m(fit.xyz_mm, car), dtype=np.float64)
        delta_mm = 1000.0 * (visual_car_m - fk_car_ground_m)
        target_car_m = np.asarray(
            [record["target"]["x_m"], 0.0, record["target"]["z_ground_m"]],
            dtype=np.float64,
        )
        fk_minus_target_mm = 1000.0 * (fk_car_ground_m - target_car_m)
        visual_minus_target_mm = 1000.0 * (visual_car_m - target_car_m)
        loo_max_mm = float(max(loo_delta.values()))
        heldout_max_px = float(max(heldout.values()))
        quality = (
            "pass"
            if fit.max_px <= sweet.MARKER_MAX_REPROJ_PX
            and loo_max_mm < sweet.MARKER_MAX_LOO_MM
            and heldout_max_px <= sweet.MARKER_MAX_HELDOUT_PX
            and float(car.reprojection_error) <= 4.0
            else "review"
        )
        item = {
            "schema": "v04_visual_tcp_fk/v1",
            "trial_id": record["trial_id"],
            "index": int(record["index"]),
            "phase": record["phase"],
            "point_id": record["point_id"],
            "quality": quality,
            "q_rad": q.tolist(),
            "target_car_ground_m": target_car_m.tolist(),
            "fk_car_model_m": fk_car_model_m.tolist(),
            "fk_car_ground_m": fk_car_ground_m.tolist(),
            "fk_world_m": fk_world_m.tolist(),
            "visual_world_m": (fit.xyz_mm / 1000.0).tolist(),
            "visual_car_m": visual_car_m.tolist(),
            "visual_minus_fk_mm": delta_mm.tolist(),
            "fk_minus_target_mm": fk_minus_target_mm.tolist(),
            "visual_minus_target_mm": visual_minus_target_mm.tolist(),
            "visual_minus_fk_norm_mm": float(np.linalg.norm(delta_mm)),
            "pixels": {serial: np.asarray(pixels[serial]).tolist() for serial in serials},
            "candidate_counts": {serial: len(per_camera[serial]) for serial in serials},
            "reprojection_rms_px": float(fit.rms_px),
            "reprojection_max_px": float(fit.max_px),
            "loo_max_mm": loo_max_mm,
            "heldout_max_px": heldout_max_px,
            "fk_prior_distance_mm": float(np.linalg.norm(fit.xyz_mm - fk_world_mm)),
            "car_pose": {
                "x_m": float(car.x),
                "y_m": float(car.y),
                "yaw_rad": float(car.yaw),
                "reprojection_error_px": float(car.reprojection_error),
            },
            "exposure_joint_delta_ms": float(record["joint_at_exposure"]["exposure_delta_ms"]),
        }
        results.append(item)
        _render_overlay(
            overlays / f"{record['index'] + 1:02d}_{record['trial_id']}_{record['point_id']}.png",
            images,
            predicted_uv,
            pixels,
            (
                f"{record['trial_id']} {record['point_id']} visual-FK mm "
                f"x={delta_mm[0]:+.1f} y={delta_mm[1]:+.1f} z={delta_mm[2]:+.1f} {quality}"
            ),
        )
        print(
            f"{record['trial_id']} {record['point_id']:>5} "
            f"d_mm={delta_mm[0]:+6.1f}/{delta_mm[1]:+6.1f}/{delta_mm[2]:+6.1f} "
            f"rms={fit.rms_px:.3f}px loo={loo_max_mm:.2f}mm {quality}",
            flush=True,
        )

    (OUTPUT / "results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    with (OUTPUT / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "trial_id", "point_id", "phase", "quality",
            "fk_x_m", "fk_y_m", "fk_z_ground_m",
            "visual_x_m", "visual_y_m", "visual_z_m",
            "dx_mm", "dy_mm", "dz_mm", "d_norm_mm",
            "reprojection_rms_px", "reprojection_max_px", "loo_max_mm",
            "heldout_max_px", "car_reprojection_px",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            fk = item["fk_car_ground_m"]
            visual = item["visual_car_m"]
            delta = item["visual_minus_fk_mm"]
            writer.writerow(
                {
                    "trial_id": item["trial_id"], "point_id": item["point_id"],
                    "phase": item["phase"], "quality": item["quality"],
                    "fk_x_m": fk[0], "fk_y_m": fk[1], "fk_z_ground_m": fk[2],
                    "visual_x_m": visual[0], "visual_y_m": visual[1], "visual_z_m": visual[2],
                    "dx_mm": delta[0], "dy_mm": delta[1], "dz_mm": delta[2],
                    "d_norm_mm": item["visual_minus_fk_norm_mm"],
                    "reprojection_rms_px": item["reprojection_rms_px"],
                    "reprojection_max_px": item["reprojection_max_px"],
                    "loo_max_mm": item["loo_max_mm"],
                    "heldout_max_px": item["heldout_max_px"],
                    "car_reprojection_px": item["car_pose"]["reprojection_error_px"],
                }
            )

    passed = [item for item in results if item["quality"] == "pass"]
    if not passed:
        raise RuntimeError("no visual rows passed the strict quality gates")
    errors = np.asarray([item["visual_minus_fk_mm"] for item in passed], dtype=np.float64)
    point_summaries: dict[str, Any] = {}
    for point_id in sorted({item["point_id"] for item in passed}):
        group = [item for item in passed if item["point_id"] == point_id]
        group_errors = np.asarray([item["visual_minus_fk_mm"] for item in group])
        group_visual = 1000.0 * np.asarray([item["visual_car_m"] for item in group])
        pairwise = np.linalg.norm(
            group_visual[:, np.newaxis, :] - group_visual[np.newaxis, :, :], axis=2
        )
        point_summaries[point_id] = {
            "n": len(group),
            "trials": [item["trial_id"] for item in group],
            "error_mm": _metric(group_errors),
            "visual_repeatability_max_pairwise_mm": float(np.max(pairwise)),
            "delta_peak_to_peak_mm": np.ptp(group_errors, axis=0).tolist(),
        }

    equal_point_bias_mm = np.mean(
        [point_summaries[point_id]["error_mm"]["mean"] for point_id in point_summaries],
        axis=0,
    )
    residual = errors - equal_point_bias_mm
    summary = {
        "schema": "v04_visual_tcp_fk_summary/v1",
        "session": str(SESSION),
        "input_count": len(results),
        "pass_count": len(passed),
        "review_trials": [item["trial_id"] for item in results if item["quality"] != "pass"],
        "definition": "visual_minus_fk in V04 car frame; x right, y forward, z ground-up",
        "z_conversion": {"hit_pos_z_offset_m": z_offset, "formula": "z_ground=z_model-hit_pos_z_offset"},
        "quality_gates": {
            "reprojection_max_px": sweet.MARKER_MAX_REPROJ_PX,
            "loo_max_mm_strict_lt": sweet.MARKER_MAX_LOO_MM,
            "heldout_max_px": sweet.MARKER_MAX_HELDOUT_PX,
            "car_reprojection_max_px": 4.0,
        },
        "error_mm_all_passed": _metric(errors),
        "equal_point_bias_mm": equal_point_bias_mm.tolist(),
        "bias_removed_residual_rms_mm": np.sqrt(np.mean(residual * residual, axis=0)).tolist(),
        "bias_removed_residual_norm_rms_mm": float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
        ),
        "quality": {
            "reprojection_rms_px": _metric(
                np.asarray([[item["reprojection_rms_px"]] for item in passed])
            ),
            "reprojection_max_px_max": max(item["reprojection_max_px"] for item in passed),
            "loo_max_mm_max": max(item["loo_max_mm"] for item in passed),
            "heldout_max_px_max": max(item["heldout_max_px"] for item in passed),
            "car_reprojection_px_max": max(
                item["car_pose"]["reprojection_error_px"] for item in passed
            ),
        },
        "per_point": point_summaries,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)
    print(f"OUTPUT {OUTPUT}", flush=True)
    return 0 if len(passed) == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
