"""Emit ungated four-view/LOO quality for strict failures from the V04 batch."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_v04_raw_wide_40 as batch
import capture_v04_sweet_spot_map as sweet
from run_v04_live_40 import FIXED_CAR, LOCKED_MARKER_OFFSET_TOOL_M


OUTPUT = batch.SESSION / "offline_sweet_spot_analysis_v2"


def _loo(
    pixels: dict[str, tuple[float, float]],
    cameras: dict[str, sweet.CameraModel],
    fit4: sweet.PointFit,
) -> tuple[dict[str, float], dict[str, float]]:
    loo_delta: dict[str, float] = {}
    heldout: dict[str, float] = {}
    for dropped in cameras:
        fit3 = sweet.triangulate_refined(
            {serial: uv for serial, uv in pixels.items() if serial != dropped}, cameras
        )
        loo_delta[dropped] = float(np.linalg.norm(fit3.xyz_mm - fit4.xyz_mm))
        heldout[dropped] = float(
            np.linalg.norm(sweet.project_raw(cameras[dropped], fit3.xyz_mm) - pixels[dropped])
        )
    return loo_delta, heldout


def main() -> int:
    strict = [
        json.loads(line)
        for line in (OUTPUT / "sweet_spot_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    records = batch._load_records()
    serials, cameras = sweet.load_camera_models()
    kin, config = sweet._load_v04_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    raw_failures: list[dict] = []

    for strict_item in strict:
        if strict_item["status"] == "success":
            continue
        index = int(strict_item["index"])
        record = records[index]
        q = np.asarray(record["q_measured_rad"], dtype=np.float64)
        q_fk = q.copy()
        q_fk[[0, 4, 5]] = 0.0
        expected = sweet._expected_sweet_world_mm(
            q_fk, FIXED_CAR, kin, z_offset, LOCKED_MARKER_OFFSET_TOOL_M
        )
        anchors = {
            serial: tuple(float(v) for v in sweet.project_raw(cameras[serial], expected))
            for serial in serials
        }
        images = batch._load_images(record, serials)
        candidates = {
            serial: batch._combined_candidates(images[serial], anchors[serial])
            for serial in serials
        }
        chosen = {
            serial: min(
                candidates[serial],
                key=lambda item: float(np.linalg.norm(np.asarray(item.uv) - np.asarray(anchors[serial]))),
            )
            for serial in serials
        }
        pixels = {serial: chosen[serial].uv for serial in serials}
        fit4 = sweet.triangulate_refined(pixels, cameras)
        loo_delta, heldout = _loo(pixels, cameras, fit4)
        anchor_distance = float(np.linalg.norm(fit4.xyz_mm - expected))
        violations: list[str] = []
        if fit4.max_px > sweet.MARKER_MAX_REPROJ_PX:
            violations.append("reproj_max")
        if max(loo_delta.values()) >= sweet.MARKER_MAX_LOO_MM:
            violations.append("loo_max")
        if max(heldout.values()) > sweet.MARKER_MAX_HELDOUT_PX:
            violations.append("heldout_max")
        if anchor_distance > sweet.MARKER_TRACKED_MAX_EXPECTED_DISTANCE_MM:
            violations.append("anchor_distance")
        raw = {
            "index": index,
            "strict_accepted": False,
            "manual_review": True,
            "q_measured_rad": q.tolist(),
            "q_measured_deg": np.degrees(q).tolist(),
            "pixels": pixels,
            "sweet_world_m": (fit4.xyz_mm / 1000.0).tolist(),
            "sweet_car_m": sweet._sweet_in_car_m(fit4.xyz_mm, FIXED_CAR),
            "reproj_px": fit4.radial_errors_px,
            "reproj_rms_px": fit4.rms_px,
            "reproj_max_px": fit4.max_px,
            "loo_delta_mm": loo_delta,
            "loo_max_mm": max(loo_delta.values()),
            "loo_heldout_px": heldout,
            "heldout_max_px": max(heldout.values()),
            "anchor_distance_mm": anchor_distance,
            "strict_gate_violations": violations,
        }
        raw_failures.append(raw)
        print(
            f"{index:02d} raw reproj={fit4.max_px:.3f}px loo={raw['loo_max_mm']:.3f}mm "
            f"heldout={raw['heldout_max_px']:.3f}px anchor={anchor_distance:.3f}mm "
            f"violations={violations}",
            flush=True,
        )

        tiles = []
        for serial in serials:
            image = images[serial].copy()
            anchor = tuple(int(round(v)) for v in anchors[serial])
            uv = tuple(int(round(v)) for v in pixels[serial])
            cv2.drawMarker(image, anchor, (0, 255, 255), cv2.MARKER_CROSS, 42, 3)
            cv2.drawMarker(image, uv, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 42, 4)
            cv2.putText(image, serial, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            tiles.append(cv2.resize(image, (1024, 768), interpolation=cv2.INTER_AREA))
        canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        overlay_dir = OUTPUT / "raw_failure_overlays"
        overlay_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(overlay_dir / f"point_{index:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

    (OUTPUT / "failed_raw_four_camera_quality.json").write_text(
        json.dumps(raw_failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw_by_index = {int(item["index"]): item for item in raw_failures}
    combined: list[dict] = []
    for item in strict:
        if item["status"] == "success":
            combined.append({**item, "strict_accepted": True})
        else:
            combined.append({**item, **raw_by_index[int(item["index"])], "status": "raw_only"})
    (OUTPUT / "all_40_with_raw_quality.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in combined), encoding="utf-8"
    )

    fields = [
        "index", "strict_accepted", "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
        "sweet_world_x_m", "sweet_world_y_m", "sweet_world_z_m",
        "sweet_car_x_m", "sweet_car_y_m", "sweet_car_z_m",
        "reproj_rms_px", "reproj_max_px", "loo_max_mm", "heldout_max_px", "anchor_distance_mm",
        "strict_gate_violations",
    ]
    with (OUTPUT / "joints_to_sweet_spot_all_40.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in combined:
            row = {
                "index": item["index"],
                "strict_accepted": item["strict_accepted"],
                **{f"j{i + 1}_deg": item["q_measured_deg"][i] for i in range(6)},
                **{f"sweet_world_{axis}_m": item["sweet_world_m"][i] for i, axis in enumerate("xyz")},
                **{f"sweet_car_{axis}_m": item["sweet_car_m"][i] for i, axis in enumerate("xyz")},
                **{name: item[name] for name in (
                    "reproj_rms_px", "reproj_max_px", "loo_max_mm", "heldout_max_px", "anchor_distance_mm"
                )},
                "strict_gate_violations": ";".join(item.get("strict_gate_violations", [])),
            }
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
