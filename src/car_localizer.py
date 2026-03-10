# -*- coding: utf-8 -*-
"""
车辆 3D 定位模块 — AprilTag 多目视觉三角测量。

流程：
  1. 接收多台相机的同步 BGR 图像
  2. cv2.aruco 检测 AprilTag (tag36h11) 的 4 个角点
  3. 取角点中心作为 tag 像素坐标
  4. cv2.undistortPoints 去镜头畸变
  5. 多视图 DLT 三角测量求 3D 世界坐标
  6. 计算重投影误差评估定位精度

标定参数从 src/config/multi_calib.json 加载。

用法：
  localizer = CarLocalizer()
  result = localizer.locate({"DA8199285": img1, "DA8199402": img2, "DA8199243": img3})
  if result is not None:
      print(f"车辆 3D: ({result.x:.0f}, {result.y:.0f}, {result.z:.0f}) mm")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIB_CONFIG = _SRC_DIR / "config" / "multi_calib.json"


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
    cameras_used: list[str]        # 参与三角测量的相机序列号
    pixels: dict[str, tuple[float, float]]  # {序列号: (u, v)}
    reprojection_error: float      # 平均重投影误差 (px)
    yaw: float = 0.0              # 绕 z 轴旋转角 (rad)


class CarLocalizer:
    """
    多目车辆 3D 定位器（基于 AprilTag）。

    在多张同步图像中检测 AprilTag，通过 tag_id 匹配，
    对 tag 中心进行多视图三角测量得到 3D 世界坐标。
    """

    def __init__(
        self,
        calib_config_path: Optional[str] = None,
    ):
        config_path = calib_config_path or str(_DEFAULT_CALIB_CONFIG)
        self._load_calib(config_path)
        self._init_aruco_detector()
        self._pool = ThreadPoolExecutor(max_workers=3)

    # ── 初始化 ──────────────────────────────────────────────────────────

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

    def _init_aruco_detector(self) -> None:
        """创建优化后的 AprilTag 36h11 检测器。"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        params = cv2.aruco.DetectorParameters()
        # 提高 cell 采样分辨率，改善斜视角下的解码成功率
        params.perspectiveRemovePixelPerCell = 8
        # 忽略 cell 边缘 30%，减少边缘串扰
        params.perspectiveRemoveIgnoredMarginPerCell = 0.3
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def serials(self) -> list[str]:
        return list(self._serials)

    # ── 检测 ──────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list[CarDetection]:
        """检测图像下 2/3 区域中的所有 AprilTag (tag36h11)。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h = gray.shape[0]
        y_offset = h // 3
        roi = gray[y_offset:]
        corners_list, ids, _ = self._detector.detectMarkers(roi)

        if ids is None:
            return []

        results = []
        for i, tag_id in enumerate(ids.ravel()):
            corners = corners_list[i].reshape(4, 2)
            corners[:, 1] += y_offset
            cx = float(corners[:, 0].mean())
            cy = float(corners[:, 1].mean())
            results.append(CarDetection(
                tag_id=int(tag_id),
                cx=cx,
                cy=cy,
                corners=corners,
            ))
        return results

    # ── 多视图三角测量 ─────────────────────────────────────────────────

    def triangulate(
        self,
        detections: dict[str, CarDetection],
        t: float = 0.0,
    ) -> CarLoc:
        """
        对多台相机中同一 AprilTag 的中心进行 DLT 三角测量。

        Args:
            detections: {序列号: CarDetection}，至少 2 台相机。
            t: 时间戳 (perf_counter)。

        Returns:
            CarLoc 3D 定位结果。
        """
        serials = list(detections.keys())
        tag_id = detections[serials[0]].tag_id

        # 三角测量 tag 中心
        center_px = {sn: (det.cx, det.cy) for sn, det in detections.items()}
        pts_3d = self._triangulate_point(serials, center_px)

        # 三角测量角点 0 和 1（tag 上边缘）计算 yaw
        c0_px = {sn: (float(det.corners[0, 0]), float(det.corners[0, 1]))
                 for sn, det in detections.items()}
        c1_px = {sn: (float(det.corners[1, 0]), float(det.corners[1, 1]))
                 for sn, det in detections.items()}
        p0 = self._triangulate_point(serials, c0_px)
        p1 = self._triangulate_point(serials, c1_px)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        yaw = math.atan2(dy, dx)

        # 重投影误差
        pixels = {}
        errs = []
        for sn in serials:
            det = detections[sn]
            pixels[sn] = (det.cx, det.cy)
            pt_h = np.append(pts_3d, 1.0)
            proj = self._P[sn] @ pt_h
            proj = proj[:2] / proj[2]
            err = np.sqrt((proj[0] - det.cx) ** 2 + (proj[1] - det.cy) ** 2)
            errs.append(err)

        return CarLoc(
            x=float(pts_3d[0]),
            y=float(pts_3d[1]),
            z=float(pts_3d[2]),
            t=t,
            tag_id=tag_id,
            cameras_used=serials,
            pixels=pixels,
            reprojection_error=float(np.mean(errs)),
            yaw=yaw,
        )

    # ── 一步到位 ──────────────────────────────────────────────────────

    def locate(
        self,
        images: dict[str, np.ndarray],
        t: float = 0.0,
    ) -> Optional[CarLoc]:
        """
        检测 + 三角测量一步完成。

        在所有图像中检测 AprilTag，按 tag_id 匹配，
        用检测到同一 tag 的 2+ 台相机进行三角测量。

        Args:
            images: {序列号: BGR 图像}
            t: 时间戳。

        Returns:
            CarLoc 或 None（不足 2 台相机检测到匹配 tag）。
        """
        # 并行检测所有相机（复用线程池）
        all_dets = {}
        futures = {self._pool.submit(self.detect, img): sn for sn, img in images.items()}
        for fut in futures:
            all_dets[futures[fut]] = fut.result()

        # 统计每个 tag_id 被哪些相机检测到
        tag_cameras = {}  # {tag_id: {sn: CarDetection}}
        for sn, dets in all_dets.items():
            for d in dets:
                tag_cameras.setdefault(d.tag_id, {})[sn] = d

        # 找到被 >=2 台相机检测到的 tag，取检测相机数最多的
        best_tag = None
        best_count = 0
        for tag_id, cam_dets in tag_cameras.items():
            if len(cam_dets) >= 2 and len(cam_dets) > best_count:
                best_tag = tag_id
                best_count = len(cam_dets)

        if best_tag is None:
            return None

        return self.triangulate(tag_cameras[best_tag], t)

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _triangulate_point(
        self,
        serials: list[str],
        pixel_coords: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        """对单个像素点进行多视图 DLT 三角测量，返回 3D 坐标 shape=(3,)。"""
        A = []
        for sn in serials:
            u, v = self._undistort_point(
                pixel_coords[sn][0], pixel_coords[sn][1],
                self._K[sn], self._D[sn],
            )
            P = self._P[sn]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    @staticmethod
    def _undistort_point(
        u: float, v: float, K: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """对单个像素坐标去畸变，返回 shape=(2,) 的去畸变像素坐标。"""
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, K, D, P=K)
        return undist[0, 0]
