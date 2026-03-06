# -*- coding: utf-8 -*-
"""
网球 3D 定位模块 — 多视图三角测量。

流程：
  1. 接收多台相机的同步 BGR 图像
  2. BallDetector (YOLO) 分别检测网球像素坐标
  3. cv2.undistortPoints 去镜头畸变
  4. 多视图 DLT 三角测量求 3D 世界坐标
  5. 计算重投影误差评估定位精度

标定参数从 src/config/multi_calib.json 加载。

管线中两种使用方式：
  A. locate(images)  — 检测 + 三角测量一步到位
  B. 先外部 detect_batch 得到 BallDetection，再调用
     triangulate(detections) — 检测与三角测量分离，
     便于在图像上绘制检测框
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .ball_detector import BallDetector, BallDetection

_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIB_CONFIG = _SRC_DIR / "config" / "multi_calib.json"


@dataclass
class Ball3D:
    """网球 3D 定位结果。"""
    x: float                       # 世界坐标 X（mm）
    y: float                       # 世界坐标 Y（mm）
    z: float                       # 世界坐标 Z（mm）
    cameras_used: list[str]        # 参与三角测量的相机序列号
    pixels: dict[str, tuple[float, float]]  # {序列号: (u, v)}
    confidence: float              # 参与相机检测置信度的最小值
    reprojection_error: float      # 平均重投影误差（像素）


class BallLocalizer:
    """
    多目网球 3D 定位器。

    在多张同步图像中检测网球，用检测到网球的 2+ 台相机
    进行多视图 DLT 三角测量得到 3D 世界坐标。

    用法::

        localizer = BallLocalizer()  # 自动加载 config/multi_calib.json
        result = localizer.locate({"DA8199285": img1, "DA8199402": img2, "DA8199243": img3})
        if result is not None:
            print(f"网球 3D: ({result.x:.1f}, {result.y:.1f}, {result.z:.1f}) mm")
    """

    def __init__(
        self,
        calib_config_path: Optional[str] = None,
        detector: Optional[BallDetector] = None,
        conf_threshold: float = 0.25,
    ):
        config_path = calib_config_path or str(_DEFAULT_CALIB_CONFIG)
        self._load_calib(config_path)
        self._detector = detector or BallDetector(conf_threshold=conf_threshold)
        self._conf = conf_threshold

    def _load_calib(self, path: str) -> None:
        """加载多目标定参数。"""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)

        self._serials = list(cfg["cameras"].keys())
        self._K = {}
        self._D = {}
        self._P = {}  # 投影矩阵 3x4

        for sn, cd in cfg["cameras"].items():
            K = np.array(cd["K"], dtype=np.float64).reshape(3, 3)
            D = np.array(cd["D"], dtype=np.float64).ravel()
            R = np.array(cd["R_world"], dtype=np.float64).reshape(3, 3)
            t = np.array(cd["t_world"], dtype=np.float64).reshape(3, 1)
            self._K[sn] = K
            self._D[sn] = D
            self._P[sn] = K @ np.hstack([R, t])

    @property
    def serials(self) -> list[str]:
        """所有标定相机序列号。"""
        return list(self._serials)

    def locate(
        self,
        images: dict[str, np.ndarray],
        conf: Optional[float] = None,
    ) -> Optional[Ball3D]:
        """
        对多张同步图片进行网球 3D 定位（检测 + 三角测量一步完成）。

        在所有图像中 YOLO 检测网球，找恰好检测到 1 个网球的相机，
        若 ≥2 台则进行多视图三角测量。

        Args:
            images: {序列号: BGR 图像}
            conf: 检测置信度阈值（覆盖默认值）。

        Returns:
            Ball3D 结果，或 None（不足 2 台相机检测到恰好 1 个网球）。
        """
        conf = conf or self._conf

        # 检测所有图像
        sns = list(images.keys())
        img_list = [images[sn] for sn in sns]
        det_results = self._detector.detect_batch(img_list)

        # 收集恰好检测到 1 个网球的相机
        detections = {}
        for sn, dets in zip(sns, det_results):
            if len(dets) == 1:
                detections[sn] = dets[0]

        if len(detections) < 2:
            return None

        return self.triangulate(detections)

    def triangulate(
        self,
        detections: dict[str, BallDetection],
    ) -> Ball3D:
        """
        对多台相机中检测到的网球像素坐标进行 DLT 三角测量。

        Args:
            detections: {序列号: BallDetection}，至少 2 台相机。

        Returns:
            Ball3D 3D 定位结果。
        """
        serials = list(detections.keys())

        # 去畸变
        undist_pts = {}
        for sn in serials:
            det = detections[sn]
            undist_pts[sn] = self._undistort_point(
                det.x, det.y, self._K[sn], self._D[sn]
            )

        # DLT: 构建 A 矩阵 (2N x 4)
        A = []
        for sn in serials:
            u, v = undist_pts[sn]
            P = self._P[sn]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.array(A)

        # SVD 求解
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pts_3d = (X[:3] / X[3])

        # 重投影误差
        pixels = {}
        errs = []
        for sn in serials:
            det = detections[sn]
            pixels[sn] = (det.x, det.y)
            pt_h = np.append(pts_3d, 1.0)
            proj = self._P[sn] @ pt_h
            proj = proj[:2] / proj[2]
            err = np.sqrt((proj[0] - det.x) ** 2 + (proj[1] - det.y) ** 2)
            errs.append(err)

        return Ball3D(
            x=float(pts_3d[0]),
            y=float(pts_3d[1]),
            z=float(pts_3d[2]),
            cameras_used=serials,
            pixels=pixels,
            confidence=min(detections[sn].confidence for sn in serials),
            reprojection_error=float(np.mean(errs)),
        )

    @staticmethod
    def _undistort_point(
        u: float, v: float, K: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """对单个像素坐标去畸变，返回 shape=(2,) 的去畸变像素坐标。"""
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, K, D, P=K)
        return undist[0, 0]
