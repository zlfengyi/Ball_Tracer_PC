# -*- coding: utf-8 -*-
"""
车辆定位模块 — 竖直 AprilTag 的多目刚体位姿估计。

流程：
  1. 接收多台相机的同步 BGR 图像
  2. cv2.aruco 检测 AprilTag (tag36h11) 的 4 个角点
  3. 以已知边长、固定中心高度和竖直安装约束 4 个角点
  4. 在原始畸变像素域联合优化 tag 的世界系 x/y/yaw
  5. 按四角重投影误差剔除异常相机
  6. 将 tag 到底盘中心的车体系外参变换到世界系

默认标定参数从 src/config/four_camera_calib.json 加载。

用法：
  localizer = CarLocalizer()
  result = localizer.locate({
      "DA7403103": img1,
      "DA8571029": img2,
      "DA7403087": img3,
      "DA8474746": img4,
  })
  if result is not None:
      print(f"车辆位置: ({result.x:.3f}, {result.y:.3f}, {result.z:.3f}) m")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .cv_linalg import (
    projection_matrix,
    smallest_right_singular_vector,
    solve_least_squares,
)

_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIB_CONFIG = _SRC_DIR / "config" / "four_camera_calib.json"
_DEFAULT_ARM_POE_CONFIG = _SRC_DIR / "config" / "arm_poe_racket_center.json"
_WORLD_SCALE_M_PER_MM = 1.0 / 1000.0
_HUBER_DELTA_PX = 3.0
_CAMERA_OUTLIER_MIN_PX = 4.0
_CAMERA_OUTLIER_RATIO = 2.5
_YAW_MAX_REPROJECTION_ERROR_PX = 8.0


@dataclass
class CarDetection:
    """单张图像中的 AprilTag 检测结果。"""
    tag_id: int                    # AprilTag ID
    cx: float                      # tag 中心 x (pixels)
    cy: float                      # tag 中心 y (pixels)
    corners: np.ndarray            # 4 个角点 shape=(4, 2)


@dataclass
class CarLoc:
    """车辆底盘中心在球场世界系中的定位结果。"""
    x: float                       # 世界坐标 X (m)
    y: float                       # 世界坐标 Y (m)
    z: float                       # 世界坐标 Z (m)
    t: float                       # 时间戳 (perf_counter)
    tag_id: int                    # 检测到的 tag ID
    cameras_used: list[str]        # 参与最终刚体拟合的相机序列号
    pixels: dict[str, tuple[float, float]]  # {序列号: (u, v)}
    reprojection_error: float      # 平均重投影误差 (px)
    yaw: float                     # 底盘绕 z 轴旋转角 (rad)
    yaw_valid: bool                # 3+ 台相机且四角误差合格时才允许修正底盘 yaw


class CarLocalizer:
    """
    多目车辆定位器（基于上边水平、平面竖直的刚性 AprilTag）。

    在多张同步图像中检测 AprilTag，通过 tag_id 匹配，
    用 tag 的真实边长、固定中心高度和竖直安装联合估计世界系 x/y/yaw。
    """

    def __init__(
        self,
        calib_config_path: Optional[str] = None,
    ):
        config_path = calib_config_path or str(_DEFAULT_CALIB_CONFIG)
        self._load_calib(config_path)
        self._load_vehicle_reference()
        self._init_aruco_detector()
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self._serials)))

    # ── 初始化 ──────────────────────────────────────────────────────────

    def _load_calib(self, path: str) -> None:
        """加载多目标定参数。"""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)

        self._serials = list(cfg["cameras"].keys())
        self._K = {}
        self._D = {}
        self._t = {}
        self._rvec = {}
        self._P = {}  # 投影矩阵 3x4

        for sn, cd in cfg["cameras"].items():
            K = np.array(cd["K"], dtype=np.float64).reshape(3, 3)
            D = np.array(cd["D"], dtype=np.float64).ravel()
            R = np.array(cd["R_world"], dtype=np.float64).reshape(3, 3)
            t = np.array(cd["t_world"], dtype=np.float64).reshape(3, 1)
            self._K[sn] = K
            self._D[sn] = D
            self._t[sn] = t
            self._rvec[sn] = cv2.Rodrigues(R)[0]
            self._P[sn] = projection_matrix(K, R, t)

    def _load_vehicle_reference(self) -> None:
        with open(_DEFAULT_ARM_POE_CONFIG, encoding="utf-8") as f:
            vehicle_cfg = json.load(f)["vehicle_reference"]
        self._apriltag_to_car_base_offset_m = np.array(
            vehicle_cfg["apriltag_center_to_car_base_offset_m"],
            dtype=np.float64,
        ).reshape(3)
        self._apriltag_black_edge_mm = (
            float(vehicle_cfg["apriltag_black_edge_m"]) / _WORLD_SCALE_M_PER_MM
        )
        self._apriltag_center_height_mm = (
            float(vehicle_cfg["apriltag_center_height_m"]) / _WORLD_SCALE_M_PER_MM
        )
        self._apriltag_to_car_yaw_offset_rad = float(
            vehicle_cfg["apriltag_to_car_yaw_offset_rad"]
        )
        half_edge = 0.5 * self._apriltag_black_edge_mm
        # ArUco 角点顺序为左上、右上、右下、左下；tag 平面竖直，0→1 定义 +x/yaw。
        self._tag_corners_local_mm = np.array(
            [
                [-half_edge, 0.0, half_edge],
                [half_edge, 0.0, half_edge],
                [half_edge, 0.0, -half_edge],
                [-half_edge, 0.0, -half_edge],
            ],
            dtype=np.float64,
        )

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
        # yaw 对角点误差很敏感；检测后使用灰度梯度做亚像素角点细化。
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def serials(self) -> list[str]:
        return list(self._serials)

    @property
    def apriltag_to_car_base_offset_m(self) -> np.ndarray:
        return self._apriltag_to_car_base_offset_m.copy()

    # ── 检测 ──────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list[CarDetection]:
        """检测整张图像中的所有 AprilTag (tag36h11)。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners_list, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return []

        results = []
        for i, tag_id in enumerate(ids.ravel()):
            corners = corners_list[i].reshape(4, 2)
            cx = float(corners[:, 0].mean())
            cy = float(corners[:, 1].mean())
            results.append(CarDetection(
                tag_id=int(tag_id),
                cx=cx,
                cy=cy,
                corners=corners,
            ))
        return results

    # ── 多视图刚体位姿估计 ─────────────────────────────────────────────

    def estimate_pose(
        self,
        detections: dict[str, CarDetection],
        t: float = 0.0,
    ) -> CarLoc:
        """
        对多台相机中同一竖直 AprilTag 的四角做固定中心高度刚体拟合。

        Args:
            detections: {序列号: CarDetection}，至少 2 台相机。
            t: 时间戳 (perf_counter)。

        Returns:
            CarLoc 3D 定位结果。
        """
        serials = list(detections.keys())
        if len(serials) < 2:
            raise ValueError("car localization requires at least two cameras")
        tag_id = detections[serials[0]].tag_id
        if any(det.tag_id != tag_id for det in detections.values()):
            raise ValueError("car localization detections must have the same tag_id")

        # DLT 只提供联合优化初值；最终位置和 yaw 均由固定高度、固定边长的四角模型给出。
        center_px = {sn: (det.cx, det.cy) for sn, det in detections.items()}
        center_3d = self._triangulate_point(serials, center_px)
        c0_px = {sn: (float(det.corners[0, 0]), float(det.corners[0, 1]))
                 for sn, det in detections.items()}
        c1_px = {sn: (float(det.corners[1, 0]), float(det.corners[1, 1]))
                 for sn, det in detections.items()}
        p0 = self._triangulate_point(serials, c0_px)
        p1 = self._triangulate_point(serials, c1_px)
        pose = np.array(
            [
                center_3d[0],
                center_3d[1],
                math.atan2(p1[1] - p0[1], p1[0] - p0[0]),
            ],
            dtype=np.float64,
        )

        # Huber 拟合后按相机四角 RMS 逐台剔除明显离群者；允许降为两相机位置结果，
        # 但两相机的短基线 yaw 不得用于覆盖 RK 上的 IMU/nav yaw。
        while True:
            pose, residual = self._fit_tag_pose(pose, serials, detections)
            if len(serials) <= 2:
                break
            camera_rms = {
                sn: float(np.sqrt(np.mean(residual[i * 8:(i + 1) * 8] ** 2)))
                for i, sn in enumerate(serials)
            }
            worst = max(serials, key=camera_rms.__getitem__)
            other_median = float(np.median([
                camera_rms[sn] for sn in serials if sn != worst
            ]))
            outlier_limit = max(
                _CAMERA_OUTLIER_MIN_PX,
                _CAMERA_OUTLIER_RATIO * other_median,
            )
            if camera_rms[worst] <= outlier_limit:
                break
            serials.remove(worst)

        corner_errors = np.linalg.norm(residual.reshape(-1, 2), axis=1)
        reprojection_error = float(np.mean(corner_errors))
        yaw = self._normalize_angle(
            float(pose[2]) + self._apriltag_to_car_yaw_offset_rad
        )
        c = math.cos(yaw)
        s = math.sin(yaw)
        offset = self._apriltag_to_car_base_offset_m
        car_pos_m = np.array(
            [
                pose[0] * _WORLD_SCALE_M_PER_MM + c * offset[0] - s * offset[1],
                pose[1] * _WORLD_SCALE_M_PER_MM + s * offset[0] + c * offset[1],
                self._apriltag_center_height_mm * _WORLD_SCALE_M_PER_MM + offset[2],
            ],
            dtype=np.float64,
        )

        return CarLoc(
            x=float(car_pos_m[0]),
            y=float(car_pos_m[1]),
            z=float(car_pos_m[2]),
            t=t,
            tag_id=tag_id,
            cameras_used=serials,
            pixels={sn: (detections[sn].cx, detections[sn].cy) for sn in serials},
            reprojection_error=reprojection_error,
            yaw=yaw,
            yaw_valid=(
                len(serials) >= 3
                and reprojection_error <= _YAW_MAX_REPROJECTION_ERROR_PX
            ),
        )

    # ── 一步到位 ──────────────────────────────────────────────────────

    def locate(
        self,
        images: dict[str, np.ndarray],
        t: float = 0.0,
    ) -> Optional[CarLoc]:
        """
        检测 + 刚体位姿估计一步完成。

        在所有图像中检测 AprilTag，按 tag_id 匹配，
        用检测到同一 tag 的 2+ 台相机进行固定高度刚体拟合。

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

        return self.estimate_pose(tag_cameras[best_tag], t)

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _fit_tag_pose(
        self,
        initial_pose: np.ndarray,
        serials: list[str],
        detections: dict[str, CarDetection],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Huber Gauss-Newton 优化 [tag_x_mm, tag_y_mm, tag_yaw_rad]。"""
        pose = initial_pose.copy()
        eps = np.array([0.1, 0.1, 1e-4], dtype=np.float64)
        max_step = np.array([200.0, 200.0, 0.35], dtype=np.float64)

        for _ in range(10):
            residual = self._corner_residuals(pose, serials, detections)
            corner_norms = np.linalg.norm(residual.reshape(-1, 2), axis=1)
            corner_weights = np.ones_like(corner_norms)
            large = corner_norms > _HUBER_DELTA_PX
            corner_weights[large] = _HUBER_DELTA_PX / corner_norms[large]
            row_weights = np.repeat(np.sqrt(corner_weights), 2)

            jacobian = np.empty((residual.size, 3), dtype=np.float64)
            for column in range(3):
                plus = pose.copy()
                minus = pose.copy()
                plus[column] += eps[column]
                minus[column] -= eps[column]
                jacobian[:, column] = (
                    self._corner_residuals(plus, serials, detections)
                    - self._corner_residuals(minus, serials, detections)
                ) / (2.0 * eps[column])

            step = solve_least_squares(
                jacobian * row_weights[:, None],
                -residual * row_weights,
            )
            step = np.clip(step, -max_step, max_step)
            current_cost = self._huber_cost(corner_norms)
            accepted = False
            for _ in range(5):
                candidate = pose + step
                candidate[2] = self._normalize_angle(float(candidate[2]))
                candidate_residual = self._corner_residuals(
                    candidate, serials, detections
                )
                if self._huber_cost(np.linalg.norm(
                    candidate_residual.reshape(-1, 2), axis=1
                )) < current_cost:
                    pose = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted or (
                np.hypot(step[0], step[1]) < 0.01 and abs(step[2]) < 1e-6
            ):
                break

        return pose, self._corner_residuals(pose, serials, detections)

    def _corner_residuals(
        self,
        pose: np.ndarray,
        serials: list[str],
        detections: dict[str, CarDetection],
    ) -> np.ndarray:
        """在原始畸变像素域返回所有相机的四角重投影残差。"""
        c = math.cos(float(pose[2]))
        s = math.sin(float(pose[2]))
        ox = self._tag_corners_local_mm[:, 0]
        oy = self._tag_corners_local_mm[:, 1]
        world_corners = np.empty((4, 3), dtype=np.float64)
        world_corners[:, 0] = pose[0] + c * ox - s * oy
        world_corners[:, 1] = pose[1] + s * ox + c * oy
        world_corners[:, 2] = (
            self._apriltag_center_height_mm + self._tag_corners_local_mm[:, 2]
        )

        residuals = []
        for sn in serials:
            projected, _ = cv2.projectPoints(
                world_corners,
                self._rvec[sn],
                self._t[sn],
                self._K[sn],
                self._D[sn],
            )
            residuals.append(
                (projected.reshape(4, 2) - detections[sn].corners).reshape(-1)
            )
        return np.concatenate(residuals)

    @staticmethod
    def _huber_cost(norms: np.ndarray) -> float:
        small = norms <= _HUBER_DELTA_PX
        return float(
            0.5 * np.sum(norms[small] ** 2)
            + np.sum(_HUBER_DELTA_PX * (
                norms[~small] - 0.5 * _HUBER_DELTA_PX
            ))
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

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
        X = smallest_right_singular_vector(A)
        return X[:3] / X[3]

    @staticmethod
    def _undistort_point(
        u: float, v: float, K: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """对单个像素坐标去畸变，返回 shape=(2,) 的去畸变像素坐标。"""
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, K, D, P=K)
        return undist[0, 0]
