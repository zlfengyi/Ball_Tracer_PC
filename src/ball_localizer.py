# -*- coding: utf-8 -*-
"""
网球 3D 定位模块 — 双目视觉三角测量。

流程：
  1. 接收两台从相机（DA8199402 / DA8199285）的同步 BGR 图像
  2. BallDetector (YOLO) 分别检测网球像素坐标
  3. cv2.undistortPoints 去镜头畸变
  4. cv2.triangulatePoints 三角测量求 3D 世界坐标
  5. 计算重投影误差评估定位精度

标定参数从 config/stereo_calib.json 加载（K1/D1/R1_world/t1_world 等）。

管线中两种使用方式：
  A. locate(img1, img2)  — 检测 + 三角测量一步到位
  B. 先外部 detect_batch 得到 BallDetection，再调用
     triangulate(det1, det2) — 检测与三角测量分离，
     便于在图像上绘制检测框（步骤 4.5 需要）
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
_PROJECT_ROOT = _SRC_DIR.parent
_DEFAULT_STEREO_CONFIG = _SRC_DIR / "config" / "stereo_calib.json"


@dataclass
class Ball3D:
    """网球 3D 定位结果。"""
    x: float       # 世界坐标 X（mm）
    y: float       # 世界坐标 Y（mm）
    z: float       # 世界坐标 Z（mm）
    pixel_1: tuple[float, float]  # 相机1 (left) 像素坐标 (u, v)
    pixel_2: tuple[float, float]  # 相机2 (right) 像素坐标 (u, v)
    confidence: float             # 两张图检测置信度的最小值
    reprojection_error: float     # 重投影误差（像素）


class BallLocalizer:
    """
    双目网球 3D 定位器。

    用法::

        localizer = BallLocalizer()  # 自动加载 config/stereo_calib.json
        result = localizer.locate(img_left, img_right)
        if result is not None:
            print(f"网球 3D 坐标: ({result.x:.1f}, {result.y:.1f}, {result.z:.1f}) mm")
    """

    def __init__(
        self,
        stereo_config_path: Optional[str] = None,
        detector: Optional[BallDetector] = None,
        conf_threshold: float = 0.25,
    ):
        """
        Args:
            stereo_config_path: 双目标定参数文件路径。None 时使用 config/stereo_calib.json。
            detector: 复用已有的 BallDetector 实例。None 时自动创建。
            conf_threshold: 网球检测置信度阈值。
        """
        # 加载标定参数
        config_path = stereo_config_path or str(_DEFAULT_STEREO_CONFIG)
        self._load_stereo_params(config_path)

        # 检测器
        self._detector = detector or BallDetector(conf_threshold=conf_threshold)
        self._conf = conf_threshold

    def _load_stereo_params(self, path: str) -> None:
        """
        从 JSON 文件加载双目标定参数。

        stereo_calib.json 中的关键字段：
          - K1, K2:       3×3 相机内参矩阵
          - D1, D2:       畸变系数
          - R1_world, t1_world: 相机1 外参（世界→相机变换，来自 solvePnP）
          - R2_world, t2_world: 相机2 外参
          - serial_left, serial_right: 对应相机序列号

        投影矩阵: P = K @ [R_world | t_world]  (3×4)
        """
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)

        # 相机内参矩阵 3x3
        self._K1 = np.array(cfg["K1"], dtype=np.float64).reshape(3, 3)
        self._K2 = np.array(cfg["K2"], dtype=np.float64).reshape(3, 3)

        # 畸变系数
        self._D1 = np.array(cfg["D1"], dtype=np.float64).ravel()
        self._D2 = np.array(cfg["D2"], dtype=np.float64).ravel()

        # 相机外参（世界→相机 变换），来自 stereo_calibrator 的 ground registration
        # R_world 来自 cv2.Rodrigues(rvec)，t_world 来自 solvePnP tvec
        self._R1 = np.array(cfg["R1_world"], dtype=np.float64).reshape(3, 3)
        self._T1 = np.array(cfg["t1_world"], dtype=np.float64).reshape(3, 1)
        self._R2 = np.array(cfg["R2_world"], dtype=np.float64).reshape(3, 3)
        self._T2 = np.array(cfg["t2_world"], dtype=np.float64).reshape(3, 1)

        # 投影矩阵 P = K @ [R | T]  (3x4)
        self._P1 = self._K1 @ np.hstack([self._R1, self._T1])
        self._P2 = self._K2 @ np.hstack([self._R2, self._T2])

        # 相机序列号（供管线匹配 SyncCapture 帧使用）
        self._serial_left = cfg.get("serial_left", "")
        self._serial_right = cfg.get("serial_right", "")

    @property
    def serial_left(self) -> str:
        """左相机（cam1）序列号，对应 K1/D1/R1/T1。"""
        return self._serial_left

    @property
    def serial_right(self) -> str:
        """右相机（cam2）序列号，对应 K2/D2/R2/T2。"""
        return self._serial_right

    def locate(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        conf: Optional[float] = None,
    ) -> Optional[Ball3D]:
        """
        对两张同步图片进行网球 3D 定位（检测 + 三角测量一步完成）。

        Args:
            img1: 相机1 (left) 的 BGR 图像。
            img2: 相机2 (right) 的 BGR 图像。
            conf: 检测置信度阈值（覆盖默认值）。

        Returns:
            Ball3D 结果，或 None（未检测到 / 不是恰好各一个网球）。
        """
        conf = conf or self._conf

        det1 = self._detector.detect(img1, conf=conf)
        det2 = self._detector.detect(img2, conf=conf)

        # 必须各检测到恰好 1 个网球
        if len(det1) != 1 or len(det2) != 1:
            return None

        return self.triangulate(det1[0], det2[0])

    def triangulate(
        self,
        det1: BallDetection,
        det2: BallDetection,
    ) -> Ball3D:
        """
        对已检测到的两个网球像素坐标进行三角测量。

        管线使用此方法时，先用 BallDetector.detect_batch() 批量检测
        得到 BallDetection（同时可用于在图像上画检测框），再传入此方法
        进行 3D 定位。

        Args:
            det1: 相机1 (left) 的检测结果。
            det2: 相机2 (right) 的检测结果。

        Returns:
            Ball3D 3D 定位结果。
        """
        # 去畸变像素坐标
        pt1_undist = self._undistort_point(det1.x, det1.y, self._K1, self._D1)
        pt2_undist = self._undistort_point(det2.x, det2.y, self._K2, self._D2)

        # 三角测量
        pts_4d = cv2.triangulatePoints(
            self._P1, self._P2,
            pt1_undist.reshape(2, 1).astype(np.float64),
            pt2_undist.reshape(2, 1).astype(np.float64),
        )

        # 齐次坐标 → 3D
        pts_3d = (pts_4d[:3] / pts_4d[3]).ravel()

        # 计算重投影误差
        reproj_err = self._reprojection_error(
            pts_3d, det1.x, det1.y, det2.x, det2.y
        )

        return Ball3D(
            x=float(pts_3d[0]),
            y=float(pts_3d[1]),
            z=float(pts_3d[2]),
            pixel_1=(det1.x, det1.y),
            pixel_2=(det2.x, det2.y),
            confidence=min(det1.confidence, det2.confidence),
            reprojection_error=reproj_err,
        )

    @staticmethod
    def _undistort_point(
        u: float, v: float, K: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """对单个像素坐标去畸变，返回 shape=(2,) 的去畸变像素坐标。"""
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, K, D, P=K)
        return undist[0, 0]  # shape (2,)

    def _reprojection_error(
        self,
        pt_3d: np.ndarray,
        u1: float, v1: float,
        u2: float, v2: float,
    ) -> float:
        """计算 3D 点重投影到两个相机的平均像素误差。"""
        pt_h = np.append(pt_3d, 1.0)  # 齐次 (4,)

        # 重投影到相机1
        proj1 = self._P1 @ pt_h
        proj1 = proj1[:2] / proj1[2]

        # 重投影到相机2
        proj2 = self._P2 @ pt_h
        proj2 = proj2[:2] / proj2[2]

        err1 = np.sqrt((proj1[0] - u1) ** 2 + (proj1[1] - v1) ** 2)
        err2 = np.sqrt((proj2[0] - u2) ** 2 + (proj2[1] - v2) ** 2)

        return float((err1 + err2) / 2.0)
