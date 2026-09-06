# -*- coding: utf-8 -*-
"""Racket-center multi-camera localization based on racket bbox + keypoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .ball_detector import BallDetector
from .cv_linalg import projection_matrix, smallest_right_singular_vector
from yolo_model.racket_pose import RacketPose


_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIB_CONFIG = _SRC_DIR / "config" / "four_camera_calib_18.json"
_DEFAULT_RACKET_MODEL = _SRC_DIR.parent / "yolo_model" / "racket.onnx"
_DEFAULT_RACKET_POSE_MODEL = _SRC_DIR.parent / "yolo_model" / "racket_pose.onnx"
_CENTER_KEYPOINT_IDS = (0, 1, 2, 3)


@dataclass
class RacketDetection:
    serial: str
    detected: bool
    accepted: bool
    failure_reason: str
    bbox_confidence: float = 0.0
    bbox_xyxy: tuple[float, float, float, float] | None = None
    center_xy: tuple[float, float] | None = None
    face_keypoint_score_min: float = 0.0
    face_valid_keypoint_count: int = 0
    keypoints_xy: np.ndarray | None = None
    keypoint_scores: np.ndarray | None = None


@dataclass
class RacketLoc:
    """Racket-face centre in calibration world coordinates (millimetres)."""

    x: float
    y: float
    z: float
    cameras_used: list[str]
    pixels: dict[str, tuple[float, float]]
    reprojection_error: float
    face_keypoint_score_min: float
    reprojection_errors: dict[str, float] = field(default_factory=dict)


class RacketLocalizer:
    """Detect racket centers in each camera and triangulate their 3D world position."""

    def __init__(
        self,
        calib_config_path: Optional[str] = None,
        racket_model_path: Optional[str | Path] = None,
        pose_model_path: Optional[str | Path] = None,
        *,
        bbox_conf: float = 0.25,
        keypoint_score_threshold: float = 40.0,
        min_valid_keypoints: int = 4,
        bbox_onnx_providers: Optional[list[str]] = None,
        pose_providers: Optional[list[str]] = None,
    ) -> None:
        self._bbox_conf = float(bbox_conf)
        self._keypoint_score_threshold = float(keypoint_score_threshold)
        self._min_valid_keypoints = int(min_valid_keypoints)
        self._load_calib(calib_config_path or str(_DEFAULT_CALIB_CONFIG))
        self._detector = BallDetector(
            model_path=racket_model_path or _DEFAULT_RACKET_MODEL,
            conf_threshold=self._bbox_conf,
            max_box_aspect_ratio=None,
            onnx_providers=bbox_onnx_providers,
        )
        self._pose_model = RacketPose(
            str(pose_model_path or _DEFAULT_RACKET_POSE_MODEL),
            providers=pose_providers,
        )

    def _load_calib(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)

        self._serials = list(cfg["cameras"].keys())
        self._K = {}
        self._D = {}
        self._P = {}
        self._R = {}
        self._t = {}
        self._rvec = {}

        for sn, cd in cfg["cameras"].items():
            K = np.array(cd["K"], dtype=np.float64).reshape(3, 3)
            D = np.array(cd["D"], dtype=np.float64).ravel()
            R = np.array(cd["R_world"], dtype=np.float64).reshape(3, 3)
            t = np.array(cd["t_world"], dtype=np.float64).reshape(3, 1)
            self._K[sn] = K
            self._D[sn] = D
            self._P[sn] = projection_matrix(K, R, t)
            self._R[sn] = R
            self._t[sn] = t
            self._rvec[sn] = cv2.Rodrigues(R)[0]

    @property
    def serials(self) -> list[str]:
        return list(self._serials)

    @property
    def provider_info(self) -> dict[str, list[str]]:
        bbox_session = getattr(self._detector, "_onnx_session", None)
        return {
            "bbox": (
                list(bbox_session.get_providers())
                if bbox_session is not None else []
            ),
            "pose": list(self._pose_model.session.get_providers()),
        }

    def detect_candidates(
        self,
        image: np.ndarray,
        serial: str = "",
        image_origin_xy: tuple[int, int] = (0, 0),
    ) -> list[RacketDetection]:
        """Detect real racket bboxes and estimate face keypoints 0..3.

        ``image`` may be a player crop.  ``image_origin_xy`` maps every bbox and
        keypoint back to native full-frame pixels before triangulation.
        """
        detections = self._detector.detect(image, conf=self._bbox_conf)
        origin = np.asarray(image_origin_xy, dtype=np.float64)
        results: list[RacketDetection] = []
        for det in detections:
            crop_bbox = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
            keypoints, scores = self._pose_model(image, crop_bbox)
            keypoints = np.asarray(keypoints, dtype=np.float64)
            scores = np.asarray(scores, dtype=np.float64)
            keypoints += origin
            bbox = (
                crop_bbox[0] + origin[0],
                crop_bbox[1] + origin[1],
                crop_bbox[2] + origin[0],
                crop_bbox[3] + origin[1],
            )
            face_points = keypoints[list(_CENTER_KEYPOINT_IDS)]
            face_scores = scores[list(_CENTER_KEYPOINT_IDS)]
            face_valid_mask = face_scores >= self._keypoint_score_threshold
            face_valid_count = int(face_valid_mask.sum())
            center_xy = face_points.mean(axis=0)
            accepted = bool(face_valid_count >= self._min_valid_keypoints)
            failure_reason = "" if accepted else "low_face_keypoint_confidence"
            results.append(RacketDetection(
                serial=serial,
                detected=True,
                accepted=accepted,
                failure_reason=failure_reason,
                bbox_confidence=float(det.confidence),
                bbox_xyxy=bbox,
                center_xy=(float(center_xy[0]), float(center_xy[1])),
                face_keypoint_score_min=float(face_scores.min()),
                face_valid_keypoint_count=face_valid_count,
                keypoints_xy=keypoints,
                keypoint_scores=scores,
            ))
        results.sort(key=lambda item: item.bbox_confidence, reverse=True)
        return results

    def detect(self, image: np.ndarray, serial: str = "") -> RacketDetection:
        candidates = self.detect_candidates(image, serial=serial)
        if candidates:
            return candidates[0]
        return RacketDetection(
            serial=serial,
            detected=False,
            accepted=False,
            failure_reason="no_racket_bbox",
        )

    def detect_all(self, images: dict[str, np.ndarray]) -> dict[str, RacketDetection]:
        return {sn: self.detect(image, serial=sn) for sn, image in images.items()}

    def triangulate(self, detections: dict[str, RacketDetection]) -> RacketLoc:
        serials = list(detections.keys())
        if len(serials) < 2:
            raise ValueError("racket triangulation requires at least two cameras")
        A = []
        for sn in serials:
            if sn not in self._P:
                raise ValueError(f"camera is absent from calibration: {sn}")
            center_xy = detections[sn].center_xy
            if center_xy is None:
                raise ValueError(f"missing racket center for triangulation camera {sn}")
            if not all(math.isfinite(value) for value in center_xy):
                raise ValueError(f"nonfinite racket center for triangulation camera {sn}")
            u, v = self._undistort_point(center_xy[0], center_xy[1], self._K[sn], self._D[sn])
            P = self._P[sn]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.array(A, dtype=np.float64)
        X = smallest_right_singular_vector(A)
        if not np.all(np.isfinite(X)) or abs(float(X[3])) < 1e-12:
            raise ValueError("degenerate racket triangulation")
        pts_3d = X[:3] / X[3]
        if not np.all(np.isfinite(pts_3d)):
            raise ValueError("nonfinite racket triangulation")

        pixels: dict[str, tuple[float, float]] = {}
        reprojection_errors: dict[str, float] = {}
        face_mins = []
        for sn in serials:
            det = detections[sn]
            assert det.center_xy is not None
            camera_point = self._R[sn] @ pts_3d.reshape(3, 1) + self._t[sn]
            if float(camera_point[2, 0]) <= 0.0:
                raise ValueError(f"racket point is behind camera {sn}")
            pixels[sn] = det.center_xy
            face_mins.append(det.face_keypoint_score_min)
            proj, _ = cv2.projectPoints(
                pts_3d.reshape(1, 3),
                self._rvec[sn],
                self._t[sn],
                self._K[sn],
                self._D[sn],
            )
            proj = proj.reshape(2)
            err = np.sqrt((proj[0] - det.center_xy[0]) ** 2 + (proj[1] - det.center_xy[1]) ** 2)
            reprojection_errors[sn] = float(err)

        return RacketLoc(
            x=float(pts_3d[0]),
            y=float(pts_3d[1]),
            z=float(pts_3d[2]),
            cameras_used=serials,
            pixels=pixels,
            reprojection_error=float(np.mean(list(reprojection_errors.values()))),
            face_keypoint_score_min=float(min(face_mins)) if face_mins else 0.0,
            reprojection_errors=reprojection_errors,
        )

    def select_and_triangulate(
        self,
        detections: dict[str, RacketDetection],
        *,
        min_cameras: int = 3,
        max_reprojection_error_px: float = 8.0,
    ) -> Optional[RacketLoc]:
        """Return the largest camera subset whose every reprojection is valid.

        Searching subsets from largest to smallest prevents a single bad view
        from poisoning an otherwise consistent 3-view solution.  Two-view
        solutions are not used by the impact estimator because their near-zero
        DLT residual cannot independently prove correspondence.
        """
        selected = self.select_candidate_combination(
            {serial: [detection] for serial, detection in detections.items()},
            min_cameras=min_cameras,
            max_reprojection_error_px=max_reprojection_error_px,
            max_candidates_per_camera=1,
        )
        return selected[0] if selected is not None else None

    def select_candidate_combination(
        self,
        candidates: dict[str, list[RacketDetection]],
        *,
        min_cameras: int = 3,
        max_reprojection_error_px: float = 8.0,
        max_candidates_per_camera: int = 3,
    ) -> Optional[tuple[RacketLoc, dict[str, RacketDetection]]]:
        """Jointly choose per-camera racket candidates and triangulation inliers.

        Candidate correspondence is a multi-camera decision.  In particular,
        the highest-confidence bbox in one camera is not selected before
        geometry has a chance to reject it.
        """
        usable = {
            serial: sorted(
                (
                    detection for detection in detections
                    if detection.accepted and detection.center_xy is not None
                ),
                key=lambda detection: detection.bbox_confidence,
                reverse=True,
            )[:max_candidates_per_camera]
            for serial, detections in candidates.items()
            if serial in self._P
        }
        usable = {serial: detections for serial, detections in usable.items() if detections}
        serials = sorted(usable)
        if len(serials) < min_cameras:
            return None
        for count in range(len(serials), min_cameras - 1, -1):
            best: tuple[RacketLoc, dict[str, RacketDetection]] | None = None
            for subset in combinations(serials, count):
                for choice in product(*(usable[serial] for serial in subset)):
                    picked = dict(zip(subset, choice))
                    try:
                        loc = self.triangulate(picked)
                    except (FloatingPointError, ValueError):
                        continue
                    if not loc.reprojection_errors:
                        continue
                    if max(loc.reprojection_errors.values()) > max_reprojection_error_px:
                        continue
                    if best is None or loc.reprojection_error < best[0].reprojection_error:
                        best = (loc, picked)
            if best is not None:
                return best
        return None

    def project_world_m(
        self,
        serial: str,
        xyz_world_m: tuple[float, float, float] | np.ndarray,
    ) -> tuple[float, float]:
        """Project a calibration-world point in metres to original pixels."""
        if serial not in self._P:
            raise KeyError(f"camera is absent from calibration: {serial}")
        xyz_world_mm = np.asarray(xyz_world_m, dtype=np.float64).reshape(1, 3) * 1000.0
        pixel, _ = cv2.projectPoints(
            xyz_world_mm,
            self._rvec[serial],
            self._t[serial],
            self._K[serial],
            self._D[serial],
        )
        u, v = pixel.reshape(2)
        return float(u), float(v)

    def locate(
        self,
        images: dict[str, np.ndarray],
    ) -> tuple[dict[str, RacketDetection], Optional[RacketLoc]]:
        all_dets = self.detect_all(images)
        accepted = {sn: det for sn, det in all_dets.items() if det.accepted and det.center_xy is not None}
        if len(accepted) < 2:
            return all_dets, None
        return all_dets, self.triangulate(accepted)

    @staticmethod
    def _undistort_point(
        u: float, v: float, K: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, K, D, P=K)
        return undist[0, 0]
