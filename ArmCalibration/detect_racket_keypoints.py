# -*- coding: utf-8 -*-
"""15.3: detect racket keypoints on captured images and save overlays."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ArmCalibration.common import ARM_DATA_ROOT, load_json, rel_or_abs, save_json
from src.ball_detector import BallDetector
from yolo_model.racket_pose import RacketPose

DEFAULT_RACKET_MODEL_PATH = project_root / "yolo_model" / "racket.onnx"
DEFAULT_RACKET_POSE_MODEL_PATH = project_root / "yolo_model" / "racket_pose.onnx"
CENTER_KEYPOINT_IDS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="15.3 detect racket keypoints and save annotated images.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="",
        help="Session directory path or ArmCalibration/data/<session_name>. Defaults to the latest session.",
    )
    parser.add_argument("--racket-model", type=str, default=str(DEFAULT_RACKET_MODEL_PATH))
    parser.add_argument("--pose-model", type=str, default=str(DEFAULT_RACKET_POSE_MODEL_PATH))
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument(
        "--keypoint-score-threshold",
        type=float,
        default=40.0,
        help="Per-face-keypoint (0-3) score threshold used for acceptance statistics.",
    )
    parser.add_argument(
        "--min-face-valid-keypoints",
        type=int,
        default=4,
        help="Minimum count of face keypoints (0-3) with score >= threshold for accepting one camera image.",
    )
    parser.add_argument(
        "--min-accepted-cameras",
        type=int,
        default=2,
        help="Minimum accepted camera count for accepting one captured sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Only process the first N samples. 0 means all samples.",
    )
    parser.add_argument(
        "--accepted-flat-dir-name",
        type=str,
        default="racket_pose_accepted_flat",
        help="Flat export directory name for accepted annotated images under the session root.",
    )
    parser.add_argument(
        "--overlay-format",
        type=str,
        default="jpg",
        choices=("jpg", "png"),
        help="Image format for annotated outputs.",
    )
    parser.add_argument(
        "--overlay-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for annotated outputs when --overlay-format=jpg.",
    )
    return parser.parse_args()


def resolve_session_dir(raw: str) -> Path:
    if raw:
        candidate = Path(raw)
        if candidate.exists():
            return candidate.resolve()
        candidate = ARM_DATA_ROOT / raw
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Session directory not found: {raw}")

    sessions = sorted(
        [entry for entry in ARM_DATA_ROOT.iterdir() if entry.is_dir()],
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        raise FileNotFoundError(f"No session directories found under {ARM_DATA_ROOT}")
    return sessions[0]


def load_sample_dirs(session_dir: Path, max_samples: int) -> list[Path]:
    sample_dirs = sorted(
        [
            entry for entry in session_dir.iterdir()
            if entry.is_dir() and entry.name.startswith("sample_") and (entry / "sample.json").exists()
        ]
    )
    if max_samples > 0:
        return sample_dirs[:max_samples]
    return sample_dirs


def round_float(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def overlay_filename(image_name: str, overlay_format: str) -> str:
    return f"{Path(image_name).stem}.{overlay_format}"


def write_overlay_image(path: Path, image: np.ndarray, *, overlay_format: str, jpeg_quality: int) -> None:
    params: list[int] = []
    if overlay_format == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise RuntimeError(f"Failed to write overlay image: {path}")


def detect_single_image(
    image: np.ndarray,
    detector: BallDetector,
    pose_model: RacketPose,
    *,
    bbox_conf: float,
    keypoint_score_threshold: float,
    min_face_valid_keypoints: int,
) -> dict:
    detections = detector.detect(image, conf=bbox_conf)
    if not detections:
        return {
            "detected": False,
            "accepted": False,
            "failure_reason": "no_racket_bbox",
        }

    det = detections[0]
    bbox = [float(det.x1), float(det.y1), float(det.x2), float(det.y2)]
    keypoints, scores = pose_model(image, bbox)
    face_keypoints = keypoints[list(CENTER_KEYPOINT_IDS)]
    face_scores = scores[list(CENTER_KEYPOINT_IDS)]
    center_xy = face_keypoints.mean(axis=0)
    all_keypoints_center_xy = keypoints.mean(axis=0)
    valid_mask = scores >= keypoint_score_threshold
    face_valid_mask = face_scores >= keypoint_score_threshold
    valid_count = int(valid_mask.sum())
    face_valid_count = int(face_valid_mask.sum())
    failed_face_keypoint_ids = [
        int(CENTER_KEYPOINT_IDS[idx])
        for idx, is_valid in enumerate(face_valid_mask)
        if not bool(is_valid)
    ]
    accepted = True
    failure_reason = ""
    if face_valid_count < min_face_valid_keypoints:
        accepted = False
        failure_reason = "low_face_keypoint_confidence"

    keypoint_entries = []
    for idx, (point, score, is_valid) in enumerate(zip(keypoints, scores, valid_mask)):
        keypoint_entries.append(
            {
                "id": int(idx),
                "x": round_float(point[0], 2),
                "y": round_float(point[1], 2),
                "score": round_float(score, 3),
                "valid": bool(is_valid),
                "used_for_center": bool(idx in CENTER_KEYPOINT_IDS),
            }
        )

    return {
        "detected": True,
        "accepted": bool(accepted),
        "failure_reason": failure_reason,
        "bbox": {
            "x1": round_float(det.x1, 2),
            "y1": round_float(det.y1, 2),
            "x2": round_float(det.x2, 2),
            "y2": round_float(det.y2, 2),
            "confidence": round_float(det.confidence, 4),
        },
        "keypoints": keypoint_entries,
        "center_xy": [
            round_float(center_xy[0], 2),
            round_float(center_xy[1], 2),
        ],
        "all_keypoints_center_xy": [
            round_float(all_keypoints_center_xy[0], 2),
            round_float(all_keypoints_center_xy[1], 2),
        ],
        "center_keypoint_ids": list(CENTER_KEYPOINT_IDS),
        "keypoint_score_mean": round_float(float(scores.mean()), 3),
        "keypoint_score_min": round_float(float(scores.min()), 3),
        "keypoint_score_max": round_float(float(scores.max()), 3),
        "valid_keypoint_count": int(valid_count),
        "face_keypoint_score_mean": round_float(float(face_scores.mean()), 3),
        "face_keypoint_score_min": round_float(float(face_scores.min()), 3),
        "face_valid_keypoint_count": int(face_valid_count),
        "failed_face_keypoint_ids": failed_face_keypoint_ids,
    }


def render_overlay(image: np.ndarray, result: dict, threshold: float) -> np.ndarray:
    vis = image.copy()
    status_color = (0, 180, 0) if result["accepted"] else (0, 0, 255)
    title = "accepted" if result["accepted"] else f"rejected: {result['failure_reason']}"
    cv2.putText(
        vis,
        title,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3,
    )

    if not result["detected"]:
        return vis

    bbox = result["bbox"]
    cv2.rectangle(
        vis,
        (int(round(bbox["x1"])), int(round(bbox["y1"]))),
        (int(round(bbox["x2"])), int(round(bbox["y2"]))),
        (255, 0, 255),
        3,
    )
    cv2.putText(
        vis,
        f"bbox={bbox['confidence']:.3f}",
        (int(round(bbox["x1"])), max(35, int(round(bbox["y1"])) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        2,
    )

    for keypoint in result["keypoints"]:
        if not keypoint["used_for_center"]:
            color = (255, 200, 0)
        else:
            color = (0, 255, 0) if keypoint["valid"] else (0, 165, 255)
        x = int(round(keypoint["x"]))
        y = int(round(keypoint["y"]))
        cv2.circle(vis, (x, y), 10, color, -1)
        cv2.putText(
            vis,
            f"{keypoint['id']}:{keypoint['score']:.1f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

    center_x = int(round(result["center_xy"][0]))
    center_y = int(round(result["center_xy"][1]))
    cv2.drawMarker(
        vis,
        (center_x, center_y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=3,
    )
    cv2.putText(
        vis,
        (
            f"face_center(0-3)=({result['center_xy'][0]:.1f}, {result['center_xy'][1]:.1f})  "
            f"face_valid={result['face_valid_keypoint_count']}/4  thr={threshold:.1f}"
        ),
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"face_min_score={result['face_keypoint_score_min']:.1f}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
    )
    return vis


def serial_summary(camera_results: list[dict]) -> dict:
    accepted = [entry for entry in camera_results if entry.get("detected")]
    if not accepted:
        return {
            "images_total": len(camera_results),
            "images_detected": 0,
            "images_accepted": 0,
        }

    means = [entry["keypoint_score_mean"] for entry in accepted]
    mins = [entry["keypoint_score_min"] for entry in accepted]
    bbox_confs = [entry["bbox"]["confidence"] for entry in accepted]
    face_mins = [entry["face_keypoint_score_min"] for entry in accepted]
    return {
        "images_total": len(camera_results),
        "images_detected": len(accepted),
        "images_accepted": sum(1 for entry in accepted if entry["accepted"]),
        "bbox_confidence_mean": round_float(mean(bbox_confs), 4),
        "keypoint_score_mean_mean": round_float(mean(means), 3),
        "keypoint_score_min_mean": round_float(mean(mins), 3),
        "keypoint_score_min_min": round_float(min(mins), 3),
        "face_keypoint_score_min_mean": round_float(mean(face_mins), 3),
        "face_keypoint_score_min_min": round_float(min(face_mins), 3),
    }


def main() -> int:
    args = parse_args()
    session_dir = resolve_session_dir(args.session)
    sample_dirs = load_sample_dirs(session_dir, args.max_samples)
    if not sample_dirs:
        raise SystemExit(f"No sample directories found under {session_dir}")

    detector = BallDetector(
        model_path=Path(args.racket_model),
        conf_threshold=args.bbox_conf,
        max_box_aspect_ratio=None,
    )
    pose_model = RacketPose(
        str(Path(args.pose_model)),
        providers=["CPUExecutionProvider"],
    )

    session_manifest_path = session_dir / "session.json"
    session_manifest = load_json(session_manifest_path)
    processed_at = datetime.now().isoformat(timespec="seconds")
    accepted_flat_dir = session_dir / args.accepted_flat_dir_name
    shutil.rmtree(accepted_flat_dir, ignore_errors=True)
    accepted_flat_dir.mkdir(parents=True, exist_ok=True)

    camera_buckets: dict[str, list[dict]] = {}
    accepted_sample_count = 0
    accepted_camera_count = 0
    accepted_flat_export_count = 0

    print("=== ArmCalibration Racket Pose ===")
    print(f"  Session: {rel_or_abs(session_dir)}")
    print(f"  Samples: {len(sample_dirs)}")
    print(f"  BBox model: {rel_or_abs(Path(args.racket_model))}")
    print(f"  Pose model: {rel_or_abs(Path(args.pose_model))}")
    print(f"  Accepted flat export: {rel_or_abs(accepted_flat_dir)}")

    for sample_dir in sample_dirs:
        sample_path = sample_dir / "sample.json"
        sample_payload = load_json(sample_path)
        overlay_dir = sample_dir / "racket_pose"
        shutil.rmtree(overlay_dir, ignore_errors=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        camera_results: dict[str, dict] = {}
        accepted_this_sample = 0

        for serial, image_name in sample_payload.get("images", {}).items():
            image_path = sample_dir / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                result = {
                    "detected": False,
                    "accepted": False,
                    "failure_reason": "image_read_failed",
                }
                overlay = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                result = detect_single_image(
                    image,
                    detector,
                    pose_model,
                    bbox_conf=args.bbox_conf,
                    keypoint_score_threshold=args.keypoint_score_threshold,
                    min_face_valid_keypoints=args.min_face_valid_keypoints,
                )
                overlay = render_overlay(
                    image,
                    result,
                    threshold=args.keypoint_score_threshold,
                )

            overlay_name = overlay_filename(image_name, args.overlay_format)
            overlay_path = overlay_dir / overlay_name
            write_overlay_image(
                overlay_path,
                overlay,
                overlay_format=args.overlay_format,
                jpeg_quality=args.overlay_jpeg_quality,
            )
            result["image_path"] = image_name
            result["overlay_path"] = str(Path("racket_pose") / overlay_name)
            camera_results[serial] = result
            camera_buckets.setdefault(serial, []).append(result)
            if result["accepted"]:
                accepted_this_sample += 1
                accepted_camera_count += 1
                flat_name = f"{sample_dir.name}__{serial}.{args.overlay_format}"
                write_overlay_image(
                    accepted_flat_dir / flat_name,
                    overlay,
                    overlay_format=args.overlay_format,
                    jpeg_quality=args.overlay_jpeg_quality,
                )
                accepted_flat_export_count += 1

        sample_pose_summary = {
            "processed_at": processed_at,
            "bbox_model": rel_or_abs(Path(args.racket_model)),
            "pose_model": rel_or_abs(Path(args.pose_model)),
            "center_keypoint_ids": list(CENTER_KEYPOINT_IDS),
            "keypoint_score_threshold": round_float(args.keypoint_score_threshold, 3),
            "min_face_valid_keypoints": int(args.min_face_valid_keypoints),
            "min_accepted_cameras": int(args.min_accepted_cameras),
            "accepted_camera_count": int(accepted_this_sample),
            "accepted_for_poe": bool(accepted_this_sample >= args.min_accepted_cameras),
            "cameras": camera_results,
        }
        sample_payload["racket_pose"] = sample_pose_summary
        save_json(sample_path, sample_payload)

        if sample_pose_summary["accepted_for_poe"]:
            accepted_sample_count += 1
        print(
            f"  {sample_dir.name}: accepted_cameras="
            f"{accepted_this_sample}/{len(camera_results)}"
        )

    summary_payload = {
        "processed_at": processed_at,
        "session_dir": rel_or_abs(session_dir),
        "bbox_model": rel_or_abs(Path(args.racket_model)),
        "pose_model": rel_or_abs(Path(args.pose_model)),
        "thresholds": {
            "bbox_conf": round_float(args.bbox_conf, 4),
            "keypoint_score_threshold": round_float(args.keypoint_score_threshold, 3),
            "min_face_valid_keypoints": int(args.min_face_valid_keypoints),
            "min_accepted_cameras": int(args.min_accepted_cameras),
        },
        "sample_count": len(sample_dirs),
        "accepted_sample_count": int(accepted_sample_count),
        "accepted_camera_count": int(accepted_camera_count),
        "accepted_flat_export_dir": rel_or_abs(accepted_flat_dir),
        "accepted_flat_export_count": int(accepted_flat_export_count),
        "per_camera": {
            serial: serial_summary(results)
            for serial, results in sorted(camera_buckets.items())
        },
    }
    summary_path = session_dir / "racket_pose_summary.json"
    save_json(summary_path, summary_payload)

    session_manifest["racket_pose"] = {
        "processed_at": processed_at,
        "summary_path": rel_or_abs(summary_path),
        "accepted_sample_count": int(accepted_sample_count),
        "accepted_camera_count": int(accepted_camera_count),
        "accepted_flat_export_dir": rel_or_abs(accepted_flat_dir),
        "accepted_flat_export_count": int(accepted_flat_export_count),
    }
    save_json(session_manifest_path, session_manifest)

    print("\nRacket pose detection finished.")
    print(f"  Summary: {rel_or_abs(summary_path)}")
    print(f"  Accepted flat images: {accepted_flat_export_count}")
    print(
        f"  Accepted samples: {accepted_sample_count}/{len(sample_dirs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
