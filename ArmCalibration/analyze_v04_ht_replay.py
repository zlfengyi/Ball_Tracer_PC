#!/usr/bin/env python3
"""Compare arm FK against four-camera racket localisation for one HT-replay capture.

Every pose in the capture holds still, so both sides can be reduced to one point in
the car body frame: FK from the joint state at exposure, vision from the racket-face
keypoint centre (and optionally the black marker, seeded by the FK projection).
"""
from __future__ import annotations

import argparse
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

from ArmCalibration import capture_v04_ht_replay as replay
from ArmCalibration import capture_v04_sweet_spot_map as sweet
from src.car_localizer import CarLocalizer
from src.racket_localizer import RacketLocalizer

RACKET_MIN_CAMERAS = 3
RACKET_MAX_REPROJ_PX = 8.0
CAR_MAX_REPROJ_PX = 4.0


def _world_from_car(point_car_m: np.ndarray, car: Any) -> np.ndarray:
    c = math.cos(float(car.yaw))
    s = math.sin(float(car.yaw))
    x, y, z = (float(value) for value in point_car_m)
    return np.asarray([float(car.x) + c * x - s * y, float(car.y) + s * x + c * y, z])


def _render_overlay(path: Path, images, serials, fk_uv, visual_uv, label: str) -> None:
    tiles = []
    for serial in serials:
        image = images[serial].copy()
        fk_pixel = tuple(int(round(value)) for value in fk_uv[serial])
        cv2.drawMarker(image, fk_pixel, (0, 255, 255), cv2.MARKER_CROSS, 46, 3)
        if visual_uv.get(serial) is not None:
            seen = tuple(int(round(value)) for value in visual_uv[serial])
            cv2.circle(image, seen, 26, (255, 0, 255), 4)
            cv2.line(image, fk_pixel, seen, (255, 0, 255), 2)
        cv2.putText(image, serial, (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA))
    canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write {path}")


def _black_marker(images, serials, fk_world_mm, previous_world_mm):
    import _audit_wide_black_dot as dark

    predicted_uv = {s: dark.project_raw(dark.CAMERAS[s], fk_world_mm) for s in serials}
    per_camera = {s: dark.candidates(images[s], predicted_uv[s], radius=180) for s in serials}
    if any(not candidates for candidates in per_camera.values()):
        return None
    selected = dark.choose(per_camera, predicted_uv, previous_world_mm, max_step=400.0)
    if selected is None:
        return None
    _score, fit, pixels, loo_delta, heldout, _choice = selected
    return {
        "world_mm": fit.xyz_mm,
        "pixels": pixels,
        "rms_px": float(fit.rms_px),
        "max_px": float(fit.max_px),
        "loo_max_mm": float(max(loo_delta.values())),
        "heldout_max_px": float(max(heldout.values())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="a v04_ht_replay_* directory under arm_controller_data")
    parser.add_argument("--black-marker", action="store_true", help="also fit the racket black marker")
    parser.add_argument("--racket-keypoint-threshold", type=float, default=40.0)
    args = parser.parse_args(argv)

    session = Path(args.session_dir)
    if not session.is_absolute():
        session = PROJECT_ROOT / "arm_controller_data" / session
    output = session / "fk_visual_analysis"
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing analysis: {output}")
    overlays = output / "review_overlays"
    overlays.mkdir(parents=True)

    records = [
        json.loads(line)
        for line in (session / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"{session}: results.jsonl is empty")

    kin, config = replay.load_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    serials, _cameras = sweet.load_camera_models()
    car_localizer = CarLocalizer(
        calib_config_path=str(sweet.CALIBRATION_PATH),
        vehicle_config_path=str(sweet.VEHICLE_CONFIG_PATH),
    )
    racket_localizer = RacketLocalizer(
        calib_config_path=str(sweet.CALIBRATION_PATH),
        keypoint_score_threshold=args.racket_keypoint_threshold,
    )

    # The car never moves during a capture, so one pooled pose beats a per-frame fit:
    # a per-frame yaw carries the two-tag baseline noise straight into every residual,
    # and the arm blocks a tag outright in some of the high poses.
    per_frame_car: list[dict[str, float]] = []
    for record in records:
        images = {s: cv2.imread(str(session / record["files"][s])) for s in serials}
        if any(image is None for image in images.values()):
            raise SystemExit(f"{record['trial_id']}: failed to load four images")
        try:
            car = sweet._car_pose(images, car_localizer)
        except sweet.SafetyError as exc:
            print(f"{record['trial_id']}: car pose unusable ({exc})", flush=True)
            continue
        per_frame_car.append(
            {
                "trial_id": record["trial_id"],
                "x_m": float(car.x),
                "y_m": float(car.y),
                "yaw_rad": float(car.yaw),
                "reprojection_error_px": float(car.reprojection_error),
            }
        )
    if not per_frame_car:
        raise SystemExit(f"{session}: no frame yielded a usable car pose")

    class PooledCar:
        pass

    pooled = PooledCar()
    pooled.x = float(np.median([row["x_m"] for row in per_frame_car]))
    pooled.y = float(np.median([row["y_m"] for row in per_frame_car]))
    pooled.yaw = float(np.median([row["yaw_rad"] for row in per_frame_car]))
    pooled.reprojection_error = float(np.median([row["reprojection_error_px"] for row in per_frame_car]))
    car_spread_mm = [
        1000.0 * float(np.ptp([row["x_m"] for row in per_frame_car])),
        1000.0 * float(np.ptp([row["y_m"] for row in per_frame_car])),
    ]
    car_yaw_spread_deg = math.degrees(float(np.ptp([row["yaw_rad"] for row in per_frame_car])))
    print(
        f"\npooled car pose from {len(per_frame_car)}/{len(records)} frames: "
        f"({1000.0 * pooled.x:.1f}, {1000.0 * pooled.y:.1f})mm yaw={math.degrees(pooled.yaw):+.3f}deg "
        f"reproj={pooled.reprojection_error:.2f}px | frame spread x/y={car_spread_mm[0]:.1f}/"
        f"{car_spread_mm[1]:.1f}mm yaw={car_yaw_spread_deg:.3f}deg\n"
    )

    previous_marker_world_mm: np.ndarray | None = None
    results: list[dict[str, Any]] = []
    for record in records:
        images = {s: cv2.imread(str(session / record["files"][s])) for s in serials}
        if any(image is None for image in images.values()):
            raise SystemExit(f"{record['trial_id']}: failed to load four images")

        car = pooled
        q = np.asarray(record["joint_at_exposure"]["q_rad"], dtype=np.float64)
        fk_car_model_m = np.asarray(kin.fk(q)["tcp"], dtype=np.float64)
        fk_car_ground_m = fk_car_model_m.copy()
        fk_car_ground_m[2] -= z_offset
        fk_world_m = _world_from_car(fk_car_ground_m, car)
        fk_world_mm = 1000.0 * fk_world_m
        fk_uv = {s: racket_localizer.project_world_m(s, fk_world_m) for s in serials}

        detections, racket = racket_localizer.locate(images)
        accepted = {s: d for s, d in detections.items() if d.accepted and d.center_xy is not None}
        if len(accepted) >= RACKET_MIN_CAMERAS:
            racket = racket_localizer.select_and_triangulate(
                accepted,
                min_cameras=RACKET_MIN_CAMERAS,
                max_reprojection_error_px=RACKET_MAX_REPROJ_PX,
            ) or racket

        target_car_m = np.asarray(
            [record["target"]["x_m"], 0.0, record["target"]["z_ground_m"]], dtype=np.float64
        )
        item: dict[str, Any] = {
            "schema": "v04_ht_replay_fk_visual/v1",
            "trial_id": record["trial_id"],
            "index": int(record["index"]),
            "phase": record["phase"],
            "point": str(record["point"]),
            "q_rad": q.tolist(),
            "target_car_ground_m": target_car_m.tolist(),
            "fk_car_model_m": fk_car_model_m.tolist(),
            "fk_car_ground_m": fk_car_ground_m.tolist(),
            "fk_world_m": fk_world_m.tolist(),
            "fk_minus_target_mm": (1000.0 * (fk_car_ground_m - target_car_m)).tolist(),
            "car_pose": {
                "x_m": float(car.x),
                "y_m": float(car.y),
                "yaw_rad": float(car.yaw),
                "reprojection_error_px": float(car.reprojection_error),
            },
            "racket_accepted_cameras": sorted(accepted),
            "exposure_joint_delta_ms": float(record["joint_at_exposure"]["exposure_delta_ms"]),
        }

        visual_uv: dict[str, Any] = {s: None for s in serials}
        if racket is None:
            item["racket"] = None
            item["quality"] = "no_racket_fix"
        else:
            visual_world_m = np.asarray([racket.x, racket.y, racket.z], dtype=np.float64) / 1000.0
            visual_car_m = np.asarray(
                sweet._sweet_in_car_m(1000.0 * visual_world_m, car), dtype=np.float64
            )
            delta_mm = 1000.0 * (visual_car_m - fk_car_ground_m)
            item["racket"] = {
                "world_m": visual_world_m.tolist(),
                "car_m": visual_car_m.tolist(),
                "cameras_used": list(racket.cameras_used),
                "reprojection_error_px": float(racket.reprojection_error),
                "face_keypoint_score_min": float(racket.face_keypoint_score_min),
            }
            item["racket_minus_fk_mm"] = delta_mm.tolist()
            item["racket_minus_fk_norm_mm"] = float(np.linalg.norm(delta_mm))
            item["racket_minus_target_mm"] = (1000.0 * (visual_car_m - target_car_m)).tolist()
            item["quality"] = (
                "pass"
                if len(racket.cameras_used) >= RACKET_MIN_CAMERAS
                and racket.reprojection_error <= RACKET_MAX_REPROJ_PX
                and float(car.reprojection_error) <= CAR_MAX_REPROJ_PX
                else "review"
            )
            for serial in racket.cameras_used:
                visual_uv[serial] = racket.pixels[serial]

        if args.black_marker:
            marker = _black_marker(images, serials, fk_world_mm, previous_marker_world_mm)
            if marker is None:
                item["black_marker"] = None
            else:
                previous_marker_world_mm = marker["world_mm"].copy()
                marker_car_m = np.asarray(
                    sweet._sweet_in_car_m(marker["world_mm"], car), dtype=np.float64
                )
                item["black_marker"] = {
                    "world_m": (marker["world_mm"] / 1000.0).tolist(),
                    "car_m": marker_car_m.tolist(),
                    "reprojection_rms_px": marker["rms_px"],
                    "reprojection_max_px": marker["max_px"],
                    "loo_max_mm": marker["loo_max_mm"],
                    "heldout_max_px": marker["heldout_max_px"],
                }
                item["marker_minus_fk_mm"] = (
                    1000.0 * (marker_car_m - fk_car_ground_m)
                ).tolist()

        results.append(item)
        delta = item.get("racket_minus_fk_mm")
        _render_overlay(
            overlays / f"{record['index'] + 1:02d}_{record['trial_id']}_{record['point']}.png",
            images,
            serials,
            fk_uv,
            visual_uv,
            (
                f"{record['trial_id']} {record['point']} "
                + (
                    "racket-FK mm x={:+.1f} y={:+.1f} z={:+.1f}".format(*delta)
                    if delta
                    else "no racket fix"
                )
                + f" {item['quality']}"
            ),
        )
        print(
            f"{record['trial_id']} {record['point']:>7} {item['phase']:>10} "
            + (
                "d_mm={:+6.1f}/{:+6.1f}/{:+6.1f}".format(*delta)
                if delta
                else "d_mm=      -/     -/     -"
            )
            + f" ncam={len(item['racket_accepted_cameras'])} {item['quality']}",
            flush=True,
        )

    (output / "results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8"
    )
    fields = [
        "trial_id", "point", "phase", "quality",
        "fk_x_m", "fk_y_m", "fk_z_ground_m",
        "racket_x_m", "racket_y_m", "racket_z_m",
        "dx_mm", "dy_mm", "dz_mm", "d_norm_mm",
        "racket_reprojection_px", "car_reprojection_px",
    ]
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            fk = item["fk_car_ground_m"]
            racket = item.get("racket")
            delta = item.get("racket_minus_fk_mm") or [float("nan")] * 3
            writer.writerow(
                {
                    "trial_id": item["trial_id"],
                    "point": item["point"],
                    "phase": item["phase"],
                    "quality": item["quality"],
                    "fk_x_m": fk[0], "fk_y_m": fk[1], "fk_z_ground_m": fk[2],
                    "racket_x_m": racket["car_m"][0] if racket else "",
                    "racket_y_m": racket["car_m"][1] if racket else "",
                    "racket_z_m": racket["car_m"][2] if racket else "",
                    "dx_mm": delta[0], "dy_mm": delta[1], "dz_mm": delta[2],
                    "d_norm_mm": item.get("racket_minus_fk_norm_mm", ""),
                    "racket_reprojection_px": racket["reprojection_error_px"] if racket else "",
                    "car_reprojection_px": item["car_pose"]["reprojection_error_px"],
                }
            )

    usable = [item for item in results if item.get("racket_minus_fk_mm")]
    summary: dict[str, Any] = {
        "schema": "v04_ht_replay_fk_visual/v1",
        "session": str(session),
        "poses": len(results),
        "racket_fixes": len(usable),
        "pass_count": sum(1 for item in results if item["quality"] == "pass"),
        "car_pose": {
            "pooled_x_m": pooled.x,
            "pooled_y_m": pooled.y,
            "pooled_yaw_rad": pooled.yaw,
            "pooled_reprojection_px": pooled.reprojection_error,
            "frames_used": len(per_frame_car),
            "frames_total": len(records),
            "frame_spread_x_mm": car_spread_mm[0],
            "frame_spread_y_mm": car_spread_mm[1],
            "frame_spread_yaw_deg": car_yaw_spread_deg,
            "per_frame": per_frame_car,
        },
    }
    if usable:
        deltas = np.asarray([item["racket_minus_fk_mm"] for item in usable], dtype=np.float64)
        summary["racket_minus_fk_mm"] = {
            "mean": deltas.mean(axis=0).tolist(),
            "median": np.median(deltas, axis=0).tolist(),
            "std": (deltas.std(axis=0, ddof=1) if len(deltas) > 1 else np.zeros(3)).tolist(),
            "min": deltas.min(axis=0).tolist(),
            "max": deltas.max(axis=0).tolist(),
        }
        by_point: dict[str, Any] = {}
        for point in sorted({item["point"] for item in usable}):
            rows = np.asarray(
                [item["racket_minus_fk_mm"] for item in usable if item["point"] == point],
                dtype=np.float64,
            )
            by_point[point] = {
                "n": int(len(rows)),
                "mean": rows.mean(axis=0).tolist(),
                "spread": (rows.max(axis=0) - rows.min(axis=0)).tolist(),
            }
        summary["by_point"] = by_point
        print("\n=== racket - FK, car body frame (mm) ===")
        for name, axis in zip(("x_right", "y_fwd", "z_up"), range(3)):
            column = deltas[:, axis]
            print(
                f"{name:8s} mean={column.mean():+8.1f}  median={np.median(column):+8.1f}  "
                f"sd={column.std(ddof=1) if len(column) > 1 else 0.0:6.2f}  "
                f"min={column.min():+8.1f}  max={column.max():+8.1f}"
            )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
