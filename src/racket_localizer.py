# -*- coding: utf-8 -*-
"""Racket-center multi-camera localization based on racket bbox + keypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .ball_detector import BallDetector
from .cv_linalg import matvec, projection_matrix, smallest_right_singular_vector
from yolo_model.racket_pose import RacketPose


_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIB_CONFIG = _SRC_DIR / "config" / "four_camera_calib.json"
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
    x: float
    y: float
    z: float
    cameras_used: list[str]
    pixels: dict[str, tuple[float, float]]
    reprojection_error: float
    face_keypoint_score_min: float


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

        for sn, cd in cfg["cameras"].items():
            K = np.array(cd["K"], dtype=np.float64).reshape(3, 3)
            D = np.array(cd["D"], dtype=np.float64).ravel()
            R = np.array(cd["R_world"], dtype=np.float64).reshape(3, 3)
            t = np.array(cd["t_world"], dtype=np.float64).reshape(3, 1)
            self._K[sn] = K
            self._D[sn] = D
            self._P[sn] = projection_matrix(K, R, t)

    @property
    def serials(self) -> list[str]:
        return list(self._serials)

    def detect(self, image: np.ndarray, serial: str = "") -> RacketDetection:
        detections = self._detector.detect(image, conf=self._bbox_conf)
        if not detections:
            return RacketDetection(
                serial=serial,
                detected=False,
                accepted=False,
                failure_reason="no_racket_bbox",
            )

        det = detections[0]
        bbox = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
        keypoints, scores = self._pose_model(image, bbox)
        face_points = keypoints[list(_CENTER_KEYPOINT_IDS)]
        face_scores = scores[list(_CENTER_KEYPOINT_IDS)]
        face_valid_mask = face_scores >= self._keypoint_score_threshold
        face_valid_count = int(face_valid_mask.sum())
        center_xy = face_points.mean(axis=0)
        accepted = bool(face_valid_count >= self._min_valid_keypoints)
        failure_reason = "" if accepted else "low_face_keypoint_confidence"

        return RacketDetection(
            serial=serial,
            detected=True,
            accepted=accepted,
            failure_reason=failure_reason,
            bbox_confidence=float(det.confidence),
            bbox_xyxy=bbox,
            center_xy=(float(center_xy[0]), float(center_xy[1])),
            face_keypoint_score_min=float(face_scores.min()),
            face_valid_keypoint_count=face_valid_count,
            keypoints_xy=np.array(keypoints, dtype=np.float64),
            keypoint_scores=np.array(scores, dtype=np.float64),
        )

    def detect_all(self, images: dict[str, np.ndarray]) -> dict[str, RacketDetection]:
        return {sn: self.detect(image, serial=sn) for sn, image in images.items()}

    def triangulate(self, detections: dict[str, RacketDetection]) -> RacketLoc:
        serials = list(detections.keys())
        A = []
        for sn in serials:
            center_xy = detections[sn].center_xy
            if center_xy is None:
                raise ValueError(f"missing racket center for triangulation camera {sn}")
            u, v = self._undistort_point(center_xy[0], center_xy[1], self._K[sn], self._D[sn])
            P = self._P[sn]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.array(A, dtype=np.float64)
        X = smallest_right_singular_vector(A)
        pts_3d = X[:3] / X[3]

        pixels: dict[str, tuple[float, float]] = {}
        errs = []
        face_mins = []
        for sn in serials:
            det = detections[sn]
            assert det.center_xy is not None
            pixels[sn] = det.center_xy
            face_mins.append(det.face_keypoint_score_min)
            pt_h = np.append(pts_3d, 1.0)
            proj = matvec(self._P[sn], pt_h)
            proj = proj[:2] / proj[2]
            err = np.sqrt((proj[0] - det.center_xy[0]) ** 2 + (proj[1] - det.center_xy[1]) ** 2)
            errs.append(err)

        return RacketLoc(
            x=float(pts_3d[0]),
            y=float(pts_3d[1]),
            z=float(pts_3d[2]),
            cameras_used=serials,
            pixels=pixels,
            reprojection_error=float(np.mean(errs)),
            face_keypoint_score_min=float(min(face_mins)) if face_mins else 0.0,
        )

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
