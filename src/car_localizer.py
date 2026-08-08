# -*- coding: utf-8 -*-
"""
车辆定位模块 — 车载多 AprilTag 的多目刚体位姿估计。

车上刚性安装若干块布局已知的竖直 AprilTag（当前两块：id0 右后、id1 左前），
布局（车体系中心 + 完整安装旋转）由 test_src/measure_car_tag_layout.py 实测，
写入 src/config/arm_poe_racket_center.json 的 vehicle_reference.apriltags。

流程：
  1. 接收多台相机的同步 BGR 图像
  2. cv2.aruco 检测 AprilTag (tag36h11) 的 4 个角点
  3. 所有相机 × 所有已配置 tag 的角点纳入同一刚体模型：
     世界角点 = 车位姿 (x, y, yaw) ∘ 车体系 tag 布局 ∘ tag 印面角点
  4. 在原始畸变像素域联合优化车的 x/y/yaw
  5. 按 (相机, tag) 视图的四角重投影误差剔除异常视图
  6. 直接输出车底盘中心位姿（车心即优化变量，无需 tag→车心换算）

双 tag 同时可见时 ~0.9 m 的中心基线让 yaw 远优于单 tag 16 cm 边长的短基线。
产品约定：locate() 只在两块 tag 都参与拟合时给出结果（单 tag 不发布）；
单 tag 拟合仅供诊断（estimate_car_pose / estimate_pose 直调）。

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
from dataclasses import dataclass, field
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
_VIEW_OUTLIER_MIN_PX = 4.0
_VIEW_OUTLIER_RATIO = 2.5
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
    tag_id: int                    # 主 tag（拟合中相机数最多；并列取小 id）
    cameras_used: list[str]        # 参与最终刚体拟合的相机序列号
    pixels: dict[str, tuple[float, float]]  # {序列号: 主 tag 中心 (u, v)}
    reprojection_error: float      # 平均重投影误差 (px)
    yaw: float                     # 底盘绕 z 轴旋转角 (rad)
    yaw_valid: bool                # 双 tag 或 3+ 相机、且四角误差合格才允许修正底盘 yaw
    tag_ids: list[int] = field(default_factory=list)  # 参与最终拟合的全部 tag
    # 参与最终拟合的各 (相机, tag) 检测角点 {序列号: {tag_id: (4,2) 像素}}，供离线叠加画识别框
    corners_px: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)


@dataclass
class _TagSpec:
    """单块车载 tag 的布局参数（车体系）。"""
    tag_id: int
    center_car_mm: np.ndarray      # (3,) 车体系中心，z 为离地高度
    R_tag_car: np.ndarray          # 3x3，列 = [印面右 e1, 印面上 e2, 朝外法线 n]


def tag_rotation_from_angles(
    face_azimuth_rad: float,
    inplane_rotation_rad: float = 0.0,
) -> np.ndarray:
    """由『版面朝外法线方位角 + 面内旋转』构造竖直 tag 的安装旋转。

    face_azimuth: 法线在车体系水平面内的方位角（atan2(ny, nx)）。
    inplane_rotation: 印面绕法线的旋转（180° = 倒贴）。
    返回 3x3，列 = [印面右, 印面上, 朝外法线]。
    """
    n = np.array([
        math.cos(face_azimuth_rad), math.sin(face_azimuth_rad), 0.0,
    ])
    e1 = np.cross([0.0, 0.0, 1.0], n)          # 印面右（正贴时水平）
    e2 = np.array([0.0, 0.0, 1.0])             # 印面上（正贴时竖直向上）
    cr = math.cos(inplane_rotation_rad)
    sr = math.sin(inplane_rotation_rad)
    e1r = cr * e1 + sr * e2
    e2r = -sr * e1 + cr * e2
    return np.column_stack([e1r, e2r, n])


class CarLocalizer:
    """
    多目车辆定位器（基于车载已知布局的刚性 AprilTag 组）。

    在多张同步图像中检测 AprilTag，将所有已配置 tag 的角点观测合并，
    联合估计车底盘中心的世界系 x/y/yaw。
    """

    def __init__(
        self,
        calib_config_path: Optional[str] = None,
        vehicle_config_path: Optional[str] = None,
    ):
        config_path = calib_config_path or str(_DEFAULT_CALIB_CONFIG)
        self._load_calib(config_path)
        self._load_vehicle_reference(
            vehicle_config_path or str(_DEFAULT_ARM_POE_CONFIG)
        )
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

    def _load_vehicle_reference(self, path: str) -> None:
        """加载车载 tag 布局（车体系：+x=右、+y=前、z 离地）。"""
        with open(path, encoding="utf-8") as f:
            vehicle_cfg = json.load(f)["vehicle_reference"]
        self._apriltag_black_edge_mm = (
            float(vehicle_cfg["apriltag_black_edge_m"]) / _WORLD_SCALE_M_PER_MM
        )
        half_edge = 0.5 * self._apriltag_black_edge_mm
        # 印面坐标 (u=印面右, v=印面上)，ArUco 角点顺序左上、右上、右下、左下
        self._corners_printed_mm = np.array(
            [
                [-half_edge, half_edge],
                [half_edge, half_edge],
                [half_edge, -half_edge],
                [-half_edge, -half_edge],
            ],
            dtype=np.float64,
        )
        self._tags: dict[int, _TagSpec] = {}
        for key, tag_cfg in vehicle_cfg["apriltags"].items():
            tag_id = int(key)
            center_mm = (
                np.array(tag_cfg["center_car_m"], dtype=np.float64).reshape(3)
                / _WORLD_SCALE_M_PER_MM
            )
            if "R_tag_car" in tag_cfg:
                R = np.array(
                    tag_cfg["R_tag_car"], dtype=np.float64
                ).reshape(3, 3)
            else:
                R = tag_rotation_from_angles(
                    math.radians(float(tag_cfg["face_azimuth_deg"])),
                    math.radians(float(tag_cfg.get("inplane_rotation_deg", 0.0))),
                )
            self._tags[tag_id] = _TagSpec(
                tag_id=tag_id,
                center_car_mm=center_mm,
                R_tag_car=R,
            )
        if not self._tags:
            raise ValueError(f"vehicle_reference.apriltags is empty in {path}")

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
    def tag_ids(self) -> list[int]:
        """已配置的车载 tag id。"""
        return sorted(self._tags)

    @property
    def tag_layout_m(self) -> dict[int, np.ndarray]:
        """{tag_id: 车体系中心 (3,) m}。"""
        return {
            tag_id: spec.center_car_mm * _WORLD_SCALE_M_PER_MM
            for tag_id, spec in self._tags.items()
        }

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

    def estimate_car_pose(
        self,
        tag_detections: dict[int, dict[str, CarDetection]],
        t: float = 0.0,
    ) -> Optional[CarLoc]:
        """
        对多台相机中若干车载 tag 的四角做车位姿联合刚体拟合。

        Args:
            tag_detections: {tag_id: {序列号: CarDetection}}，只使用已配置的
                tag；至少一块 tag 需被 >=2 台相机看到（用于初值三角化），
                其余 1 台相机的 tag 观测也会加入联合优化。
            t: 时间戳 (perf_counter)。

        Returns:
            CarLoc，或 None（初值不可解 / 剔除后有效相机不足 2 台）。
        """
        usable = {
            tag_id: dict(cam_dets)
            for tag_id, cam_dets in tag_detections.items()
            if tag_id in self._tags and cam_dets
        }
        pose = self._initial_pose(usable)
        if pose is None:
            return None

        # (序列号, tag_id) 为一个视图单元，各贡献 4 角 8 行残差。
        units: list[tuple[str, int, CarDetection]] = [
            (sn, tag_id, det)
            for tag_id, cam_dets in usable.items()
            for sn, det in cam_dets.items()
        ]

        # Huber 拟合后按视图四角 RMS 逐个剔除明显离群者；剩 2 个视图时停止。
        while True:
            pose, residual = self._fit_car_pose(pose, units)
            if len(units) <= 2:
                break
            view_rms = [
                float(np.sqrt(np.mean(residual[i * 8:(i + 1) * 8] ** 2)))
                for i in range(len(units))
            ]
            worst = int(np.argmax(view_rms))
            other_median = float(np.median(
                [rms for i, rms in enumerate(view_rms) if i != worst]
            ))
            outlier_limit = max(
                _VIEW_OUTLIER_MIN_PX,
                _VIEW_OUTLIER_RATIO * other_median,
            )
            if view_rms[worst] <= outlier_limit:
                break
            units.pop(worst)

        cameras_used = list(dict.fromkeys(sn for sn, _, _ in units))
        if len(cameras_used) < 2:
            return None

        corner_errors = np.linalg.norm(residual.reshape(-1, 2), axis=1)
        reprojection_error = float(np.mean(corner_errors))
        yaw = self._normalize_angle(float(pose[2]))

        tag_camera_count: dict[int, int] = {}
        for _, tag_id, _ in units:
            tag_camera_count[tag_id] = tag_camera_count.get(tag_id, 0) + 1
        tag_ids = sorted(tag_camera_count)
        primary_tag = min(
            tag_camera_count,
            key=lambda tag_id: (-tag_camera_count[tag_id], tag_id),
        )

        pixels: dict[str, tuple[float, float]] = {}
        corners_px: dict[str, dict[int, np.ndarray]] = {}
        for sn, tag_id, det in units:
            if sn not in pixels or tag_id == primary_tag:
                pixels[sn] = (det.cx, det.cy)
            corners_px.setdefault(sn, {})[tag_id] = det.corners

        return CarLoc(
            x=float(pose[0]) * _WORLD_SCALE_M_PER_MM,
            y=float(pose[1]) * _WORLD_SCALE_M_PER_MM,
            z=0.0,
            t=t,
            tag_id=primary_tag,
            cameras_used=cameras_used,
            pixels=pixels,
            reprojection_error=reprojection_error,
            yaw=yaw,
            yaw_valid=(
                reprojection_error <= _YAW_MAX_REPROJECTION_ERROR_PX
                and (len(tag_ids) >= 2 or len(cameras_used) >= 3)
            ),
            tag_ids=tag_ids,
            corners_px=corners_px,
        )

    def estimate_pose(
        self,
        detections: dict[str, CarDetection],
        t: float = 0.0,
    ) -> Optional[CarLoc]:
        """单 tag 兼容入口：多相机同一 tag 的检测 → 车位姿。"""
        serials = list(detections.keys())
        if len(serials) < 2:
            raise ValueError("car localization requires at least two cameras")
        tag_id = detections[serials[0]].tag_id
        if any(det.tag_id != tag_id for det in detections.values()):
            raise ValueError("car localization detections must have the same tag_id")
        if tag_id not in self._tags:
            raise ValueError(f"tag {tag_id} is not in vehicle_reference.apriltags")
        return self.estimate_car_pose({tag_id: dict(detections)}, t)

    # ── 一步到位 ──────────────────────────────────────────────────────

    def locate(
        self,
        images: dict[str, np.ndarray],
        t: float = 0.0,
        min_tags: int = 2,
    ) -> Optional[CarLoc]:
        """
        检测 + 车位姿联合估计一步完成。

        在所有图像中检测 AprilTag，收集所有已配置车载 tag 的观测，
        联合做车位姿刚体拟合。

        默认要求至少 2 块车载 tag 参与最终拟合才返回结果——单 tag 位姿
        的 yaw 基线短、且经安装杠杆放大位置误差，不发布（产品约定）。
        生产链路（tracker 发布 /pc_car_loc、离线补标）都走本入口；
        诊断需要单 tag 结果时传 min_tags=1 或直接用 estimate_car_pose。

        Args:
            images: {序列号: BGR 图像}
            t: 时间戳。
            min_tags: 最终拟合中最少 tag 块数。

        Returns:
            CarLoc 或 None（无 tag 被 >=2 台相机检测到 / 参与拟合的
            tag 数不足 min_tags）。
        """
        # 并行检测所有相机（复用线程池）
        all_dets = {}
        futures = {self._pool.submit(self.detect, img): sn for sn, img in images.items()}
        for fut in futures:
            all_dets[futures[fut]] = fut.result()

        # 收集已配置 tag 的检测：{tag_id: {sn: CarDetection}}
        tag_cameras: dict[int, dict[str, CarDetection]] = {}
        for sn, dets in all_dets.items():
            for d in dets:
                if d.tag_id in self._tags:
                    tag_cameras.setdefault(d.tag_id, {})[sn] = d

        if len(tag_cameras) < min_tags:
            return None
        if not any(len(cam_dets) >= 2 for cam_dets in tag_cameras.values()):
            return None

        result = self.estimate_car_pose(tag_cameras, t)
        if result is not None and len(result.tag_ids) < min_tags:
            # 离群剔除后有 tag 整块出局（如仅剩单视图且被剔），不发布
            return None
        return result

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _tag_world_corners(
        self, pose: np.ndarray, tag_id: int
    ) -> np.ndarray:
        """车位姿 [x_mm, y_mm, yaw] 下某 tag 的 4 个世界角点 (4,3) mm。"""
        c = math.cos(float(pose[2]))
        s = math.sin(float(pose[2]))
        Rz = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        spec = self._tags[tag_id]
        center_w = np.array(
            [
                pose[0] + c * spec.center_car_mm[0] - s * spec.center_car_mm[1],
                pose[1] + s * spec.center_car_mm[0] + c * spec.center_car_mm[1],
                spec.center_car_mm[2],
            ],
            dtype=np.float64,
        )
        R_w = Rz @ spec.R_tag_car
        # (4,2) 印面坐标 × [e1_w, e2_w]^T → (4,3)
        return center_w[None, :] + self._corners_printed_mm @ R_w[:, :2].T

    def _car_corner_residuals(
        self,
        pose: np.ndarray,
        units: list[tuple[str, int, CarDetection]],
    ) -> np.ndarray:
        """在原始畸变像素域返回所有 (相机, tag) 视图的四角重投影残差。"""
        world_corners = {
            tag_id: self._tag_world_corners(pose, tag_id)
            for tag_id in {tag_id for _, tag_id, _ in units}
        }
        residuals = []
        for sn, tag_id, det in units:
            projected, _ = cv2.projectPoints(
                world_corners[tag_id],
                self._rvec[sn],
                self._t[sn],
                self._K[sn],
                self._D[sn],
            )
            residuals.append(
                (projected.reshape(4, 2) - det.corners).reshape(-1)
            )
        return np.concatenate(residuals)

    def _fit_car_pose(
        self,
        initial_pose: np.ndarray,
        units: list[tuple[str, int, CarDetection]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Huber Gauss-Newton 优化车位姿 [x_mm, y_mm, yaw_rad]。"""
        pose = initial_pose.copy()
        eps = np.array([0.1, 0.1, 1e-4], dtype=np.float64)
        max_step = np.array([200.0, 200.0, 0.35], dtype=np.float64)

        for _ in range(10):
            residual = self._car_corner_residuals(pose, units)
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
                    self._car_corner_residuals(plus, units)
                    - self._car_corner_residuals(minus, units)
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
                candidate_residual = self._car_corner_residuals(candidate, units)
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

        return pose, self._car_corner_residuals(pose, units)

    def _initial_pose(
        self,
        tag_detections: dict[int, dict[str, CarDetection]],
    ) -> Optional[np.ndarray]:
        """DLT 三角化生成车位姿初值 [x_mm, y_mm, yaw_rad]。

        双 tag（各 >=2 相机）：由两中心基线解 yaw；单 tag：由印面角点边
        方向 + 安装旋转解 yaw。
        """
        centers: dict[int, np.ndarray] = {}
        for tag_id, cam_dets in tag_detections.items():
            if len(cam_dets) < 2:
                continue
            sns = list(cam_dets)
            centers[tag_id] = self._triangulate_point(
                sns,
                {sn: (cam_dets[sn].cx, cam_dets[sn].cy) for sn in sns},
            )
        if not centers:
            return None

        if len(centers) >= 2:
            tag_a, tag_b = sorted(centers)[:2]
            w_ab = centers[tag_b][:2] - centers[tag_a][:2]
            a_ab = (
                self._tags[tag_b].center_car_mm[:2]
                - self._tags[tag_a].center_car_mm[:2]
            )
            yaw = (
                math.atan2(w_ab[1], w_ab[0]) - math.atan2(a_ab[1], a_ab[0])
            )
            c = math.cos(yaw)
            s = math.sin(yaw)
            xy = np.zeros(2)
            for tag_id in (tag_a, tag_b):
                a = self._tags[tag_id].center_car_mm[:2]
                xy += centers[tag_id][:2] - np.array(
                    [c * a[0] - s * a[1], s * a[0] + c * a[1]]
                )
            xy *= 0.5
        else:
            (tag_id, center_w), = centers.items()
            cam_dets = tag_detections[tag_id]
            sns = list(cam_dets)
            spec = self._tags[tag_id]
            # 选车体系水平分量更大的印面轴，避免近竖直边的方位角退化
            e1_h = float(np.hypot(*spec.R_tag_car[:2, 0]))
            e2_h = float(np.hypot(*spec.R_tag_car[:2, 1]))
            if e1_h >= e2_h:
                idx_a, idx_b = 0, 1                       # c0 -> c1 沿印面右
                axis_car = spec.R_tag_car[:2, 0]
            else:
                idx_a, idx_b = 3, 0                       # c3 -> c0 沿印面上
                axis_car = spec.R_tag_car[:2, 1]
            p_a = self._triangulate_point(sns, {
                sn: (
                    float(cam_dets[sn].corners[idx_a, 0]),
                    float(cam_dets[sn].corners[idx_a, 1]),
                )
                for sn in sns
            })
            p_b = self._triangulate_point(sns, {
                sn: (
                    float(cam_dets[sn].corners[idx_b, 0]),
                    float(cam_dets[sn].corners[idx_b, 1]),
                )
                for sn in sns
            })
            yaw = (
                math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
                - math.atan2(axis_car[1], axis_car[0])
            )
            c = math.cos(yaw)
            s = math.sin(yaw)
            a = spec.center_car_mm[:2]
            xy = center_w[:2] - np.array(
                [c * a[0] - s * a[1], s * a[0] + c * a[1]]
            )

        return np.array(
            [xy[0], xy[1], self._normalize_angle(yaw)], dtype=np.float64
        )

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
