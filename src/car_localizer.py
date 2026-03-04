# -*- coding: utf-8 -*-
"""
车辆 3D 定位模块 — AprilTag 双目视觉三角测量。

流程：
  1. 接收两台相机的同步 BGR 图像
  2. cv2.aruco 检测 AprilTag (tag36h11) 的 4 个角点
  3. 取角点中心作为 tag 像素坐标
  4. cv2.undistortPoints 去镜头畸变
  5. cv2.triangulatePoints 三角测量求 3D 世界坐标
  6. 计算重投影误差评估定位精度

标定参数从 src/config/stereo_calib.json 加载。

用法：
  localizer = CarLocalizer()
  result = localizer.locate(img_left, img_right, t=time.perf_counter())
  if result is not None:
      print(f"车辆 3D: ({result.x:.0f}, {result.y:.0f}, {result.z:.0f}) mm")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_STEREO_CONFIG = _SRC_DIR / "config" / "stereo_calib.json"


@dataclass
class CarDetection:
    """单张图像中的 AprilTag 检测结果。"""
    tag_id: int                    # AprilTag ID
    cx: float                      # tag 中心 x (pixels)
    cy: float                      # tag 中心 y (pixels)
    corners: np.ndarray            # 4 个角点 shape=(4, 2)


@dataclass
class CarLoc:
    """车辆 3D 定位结果。"""
    x: float                       # 世界坐标 X (mm)
    y: float                       # 世界坐标 Y (mm)
    z: float                       # 世界坐标 Z (mm)
    t: float                       # 时间戳 (perf_counter)
    tag_id: int                    # 检测到的 tag ID
    pixel_1: tuple[float, float]   # 左相机像素坐标 (u, v)
    pixel_2: tuple[float, float]   # 右相机像素坐标 (u, v)
    reprojection_error: float      # 重投影误差 (px)


class CarLocalizer:
    """
    双目车辆 3D 定位器（基于 AprilTag）。

    在两张同步图像中检测 AprilTag，通过 tag_id 匹配左右图的同一 tag，
    对 tag 中心进行三角测量得到 3D 世界坐标。
    """

    def __init__(
        self,
        stereo_config_path: Optional[str] = None,
    ):
        config_path = stereo_config_path or str(_DEFAULT_STEREO_CONFIG)
        self._load_stereo_params(config_path)
        self._init_aruco_detector()

    # ── 初始化 ──────────────────────────────────────────────────────────

    def _load_stereo_params(self, path: str) -> None:
        """加载双目标定参数（与 BallLocalizer 相同逻辑）。"""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)

        self._K1 = np.array(cfg["K1"], dtype=np.float64).reshape(3, 3)
        self._K2 = np.array(cfg["K2"], dtype=np.float64).reshape(3, 3)
        self._D1 = np.array(cfg["D1"], dtype=np.float64).ravel()
        self._D2 = np.array(cfg["D2"], dtype=np.float64).ravel()

        self._R1 = np.array(cfg["R1_world"], dtype=np.float64).reshape(3, 3)
        self._T1 = np.array(cfg["t1_world"], dtype=np.float64).reshape(3, 1)
        self._R2 = np.array(cfg["R2_world"], dtype=np.float64).reshape(3, 3)
        self._T2 = np.array(cfg["t2_world"], dtype=np.float64).reshape(3, 1)

        self._P1 = self._K1 @ np.hstack([self._R1, self._T1])
        self._P2 = self._K2 @ np.hstack([self._R2, self._T2])

        self._serial_left = cfg.get("serial_left", "")
        self._serial_right = cfg.get("serial_right", "")

    def _init_aruco_detector(self) -> None:
        """创建 AprilTag 36h11 检测器。"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def serial_left(self) -> str:
        return self._serial_left

    @property
    def serial_right(self) -> str:
        return self._serial_right

    # ── 检测 ──────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list[CarDetection]:
        """
        检测图像中的所有 AprilTag (tag36h11)。

        Args:
            image: BGR 图像。

        Returns:
            检测结果列表（可能为空）。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners_list, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return []

        results = []
        for i, tag_id in enumerate(ids.ravel()):
            corners = corners_list[i].reshape(4, 2)  # (4, 2)
            cx = float(corners[:, 0].mean())
            cy = float(corners[:, 1].mean())
            results.append(CarDetection(
                tag_id=int(tag_id),
                cx=cx,
                cy=cy,
                corners=corners,
            ))
        return results

    # ── 三角测量 ──────────────────────────────────────────────────────

    def triangulate(
        self,
        det1: CarDetection,
        det2: CarDetection,
        t: float = 0.0,
    ) -> CarLoc:
        """
        对左右图中同一 AprilTag 的中心进行三角测量。

        Args:
            det1: 左相机检测结果。
            det2: 右相机检测结果。
            t: 时间戳 (perf_counter)。

        Returns:
            CarLoc 3D 定位结果。
        """
        pt1_undist = self._undistort_point(det1.cx, det1.cy, self._K1, self._D1)
        pt2_undist = self._undistort_point(det2.cx, det2.cy, self._K2, self._D2)

        pts_4d = cv2.triangulatePoints(
            self._P1, self._P2,
            pt1_undist.reshape(2, 1).astype(np.float64),
            pt2_undist.reshape(2, 1).astype(np.float64),
        )
        pts_3d = (pts_4d[:3] / pts_4d[3]).ravel()

        reproj_err = self._reprojection_error(
            pts_3d, det1.cx, det1.cy, det2.cx, det2.cy
        )

        return CarLoc(
            x=float(pts_3d[0]),
            y=float(pts_3d[1]),
            z=float(pts_3d[2]),
            t=t,
            tag_id=det1.tag_id,
            pixel_1=(det1.cx, det1.cy),
            pixel_2=(det2.cx, det2.cy),
            reprojection_error=reproj_err,
        )

    # ── 一步到位 ──────────────────────────────────────────────────────

    def locate(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        t: float = 0.0,
    ) -> Optional[CarLoc]:
        """
        检测 + 三角测量一步完成。

        在两张图中各检测 AprilTag，按 tag_id 匹配后三角测量。
        若匹配到多个相同 ID 的 tag，取第一个。

        Args:
            img1: 左相机 BGR 图像。
            img2: 右相机 BGR 图像。
            t: 时间戳。

        Returns:
            CarLoc 或 None（未检测到匹配的 tag）。
        """
        dets1 = self.detect(img1)
        dets2 = self.detect(img2)

        if not dets1 or not dets2:
            return None

        # 按 tag_id 匹配
        ids2 = {d.tag_id: d for d in dets2}
        for d1 in dets1:
            if d1.tag_id in ids2:
                return self.triangulate(d1, ids2[d1.tag_id], t)

        return None

    # ── 内部方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _undistort_point(
        u: float, v: float, K: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """对单个像素坐标去畸变，返回 shape=(2,) 的去畸变像素坐标。"""
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, K, D, P=K)
        return undist[0, 0]

    def _reprojection_error(
        self,
        pt_3d: np.ndarray,
        u1: float, v1: float,
        u2: float, v2: float,
    ) -> float:
        """计算 3D 点重投影到两个相机的平均像素误差。"""
        pt_h = np.append(pt_3d, 1.0)

        proj1 = self._P1 @ pt_h
        proj1 = proj1[:2] / proj1[2]

        proj2 = self._P2 @ pt_h
        proj2 = proj2[:2] / proj2[2]

        err1 = np.sqrt((proj1[0] - u1) ** 2 + (proj1[1] - v1) ** 2)
        err2 = np.sqrt((proj2[0] - u2) ** 2 + (proj2[1] - v2) ** 2)

        return float((err1 + err2) / 2.0)
