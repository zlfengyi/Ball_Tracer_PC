# -*- coding: utf-8 -*-
"""
网球 3D 定位模块 — 多视图三角测量。

流程：
  1. 接收多台相机的同步 BGR 图像
  2. BallDetector (YOLO) 分别检测网球像素坐标
  3. cv2.undistortPoints 去镜头畸变
  4. 多视图 DLT 三角测量求 3D 世界坐标
  5. 计算重投影误差评估定位精度

默认标定参数从 src/config/four_camera_calib_18.json 加载。

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
from .cv_linalg import matvec, projection_matrix, smallest_right_singular_vector

_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIB_CONFIG = _SRC_DIR / "config" / "four_camera_calib_18.json"
_WORLD_SCALE_M_PER_MM = 1.0 / 1000.0
# 多假设三角化默认参数（与 config/tracker.json 的同名项对齐；这里只是库的兜底值）
_DEFAULT_MAX_REPROJ_PX = 15.0
# 每台相机取几个候选框。**默认 1**：0811 两场离线回放实测，放到 2（枚举各相机
# 的第 2 个框）会多救回约 1/3 的帧，但代价是"三台相机同时锁到同一颗幽灵球"这种
# 几何上完全自洽、语义上完全错误的解——新增点偏离局部物理拟合的 p90 从 31cm
# 恶化到 251cm（另一场 24cm→731cm），出现米级错点。纯几何没有"我在跟哪颗球"
# 的概念，判不出来。等 tracker 预测位置门控（按预测投影选框）落地后再放开。
_DEFAULT_MAX_PER_CAMERA = 1


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

        localizer = BallLocalizer()  # 自动加载 config/four_camera_calib_18.json
        result = localizer.locate({
            "DB0260414": img1,
            "DB0260373": img2,
            "DA7403087": img3,
            "DA8474746": img4,
        })
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
            self._P[sn] = projection_matrix(K, R, t)

    @property
    def serials(self) -> list[str]:
        """所有标定相机序列号。"""
        return list(self._serials)

    def select_and_triangulate(
        self,
        candidates: dict[str, list[BallDetection]],
        *,
        min_cameras: int = 2,
        max_reproj_error_px: float = _DEFAULT_MAX_REPROJ_PX,
        max_per_camera: int = _DEFAULT_MAX_PER_CAMERA,
        require_exactly_one: bool = False,
    ) -> Optional[Ball3D]:
        """多假设三角化：每台相机可能框到不止一颗球，按重投影一致性择优。

        为什么需要它（0811 两场实测）：场上常有第二颗球（地上的、手里的），
        旧规则「一台相机恰好检出 1 个球才算数」会把冒出第 2 个框的整台相机
        丢掉——击球前 [20,750]ms 窗口里 17.9% 的帧因此凑不齐 2 台相机而无
        3D；另有 13.6% 的帧是各相机各锁一颗**不同**的球，配错后重投影
        >15px 被拒。两种失效都源于「每台相机独立决定用哪个框」。

        做法：把选择权交给几何一致性——枚举各相机的候选框组合，
        **按参与相机数从多到少分层**，层内取重投影误差最小者，第一个过门
        的层即答案。

        ⚠ 分层是关键，不能直接全局取误差最小：2 视图解只有 1 个自由度冗余，
        误差几乎恒等于 0，全局最小会稳定选中「两台相机各锁一颗不同球」的错
        解。要求尽可能多的相机同时自洽，才是真正的判别力来源。

        Args:
            candidates: {序列号: [BallDetection]}，只应含 is_tennis_ball 的框。
            min_cameras: 最少参与相机数。
            max_reproj_error_px: 平均重投影误差门限。
            max_per_camera: 每台相机最多取几个候选框（按置信度降序）。
            require_exactly_one: True 时退回 2026-08-11 之前的行为——只用
                恰好检出 1 个球的相机，多框的整台丢弃。留作 A/B 回退开关。

        Returns:
            Ball3D，或 None（没有任何组合能在门限内自洽）。
        """
        if require_exactly_one:
            picked = {sn: dets[0] for sn, dets in candidates.items() if len(dets) == 1}
            if len(picked) < min_cameras:
                return None
            result = self.triangulate(picked)
            return result if result.reprojection_error <= max_reproj_error_px else None

        # 每台相机按置信度降序取前 max_per_camera 个候选
        pool = {
            sn: sorted(dets, key=lambda d: -d.confidence)[:max(1, max_per_camera)]
            for sn, dets in candidates.items()
            if dets
        }
        serials = sorted(pool)
        if len(serials) < min_cameras:
            return None

        from itertools import combinations, product

        # 全都只有一个候选、且用满所有相机就能自洽：组合唯一，零额外开销
        single_only = all(len(pool[sn]) == 1 for sn in serials)
        if single_only:
            result = self.triangulate({sn: pool[sn][0] for sn in serials})
            if result.reprojection_error <= max_reproj_error_px:
                return result

        # ⚠ 只允许下探到 3 台。2 视图 DLT 只有 1 个自由度冗余，重投影误差几乎
        # 恒等于 0，**没有判别力**——两台锁真球、两台锁幽灵球时，任取一边的
        # 2 视图组合都"完美"，选谁纯属抛硬币。3 台起才是真正的一致性证据。
        # 唯一的例外是本来就只有 2 台相机看到球（没有在做选择），走上面那条。
        floor = max(min_cameras, 3)
        for k in range(len(serials), floor - 1, -1):
            best: Optional[Ball3D] = None
            for subset in combinations(serials, k):
                for choice in product(*(pool[sn] for sn in subset)):
                    result = self.triangulate(dict(zip(subset, choice)))
                    if best is None or result.reprojection_error < best.reprojection_error:
                        best = result
            if best is not None and best.reprojection_error <= max_reproj_error_px:
                return best

        # 只有 2 台相机可用时不存在"挑哪两台"的问题，按老规矩收（每台恰好 1 个框）
        if len(serials) == min_cameras and not single_only:
            best = None
            for choice in product(*(pool[sn] for sn in serials)):
                result = self.triangulate(dict(zip(serials, choice)))
                if best is None or result.reprojection_error < best.reprojection_error:
                    best = result
            if best is not None and best.reprojection_error <= max_reproj_error_px:
                return best

        # 兜底：回到旧规则（只用恰好检出 1 个球的相机）。保证本方法是旧行为的
        # **严格超集**——否则会出现"旧规则能出、新规则反而空"的倒退：一台相机
        # 的最高置信度框恰好是幽灵球时，它会污染候选池，把 2v2 这种无解局面
        # 造出来，而旧规则本来会直接忽略这台多框相机。
        picked = {sn: dets[0] for sn, dets in candidates.items() if len(dets) == 1}
        if len(picked) >= min_cameras:
            result = self.triangulate(picked)
            if result.reprojection_error <= max_reproj_error_px:
                return result
        return None

    def locate(
        self,
        images: dict[str, np.ndarray],
        conf: Optional[float] = None,
        *,
        max_reproj_error_px: float = _DEFAULT_MAX_REPROJ_PX,
        max_per_camera: int = _DEFAULT_MAX_PER_CAMERA,
        require_exactly_one: bool = False,
    ) -> Optional[Ball3D]:
        """
        对多张同步图片进行网球 3D 定位（检测 + 三角测量一步完成）。

        在所有图像中 YOLO 检测网球，再交给 select_and_triangulate 按几何
        一致性挑组合（2026-08-11 起；此前是「恰好 1 个框的相机才算数」）。

        Args:
            images: {序列号: BGR 图像}
            conf: 检测置信度阈值（覆盖默认值）。
            max_reproj_error_px / max_per_camera / require_exactly_one:
                见 select_and_triangulate。

        Returns:
            Ball3D 结果，或 None。
        """
        conf = conf or self._conf

        # 检测所有图像
        sns = list(images.keys())
        img_list = [images[sn] for sn in sns]
        det_results = self._detector.detect_batch(img_list)

        candidates = {
            sn: [det for det in dets if det.is_tennis_ball]
            for sn, dets in zip(sns, det_results)
        }
        return self.select_and_triangulate(
            candidates,
            max_reproj_error_px=max_reproj_error_px,
            max_per_camera=max_per_camera,
            require_exactly_one=require_exactly_one,
        )

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
        X = smallest_right_singular_vector(A)
        pts_3d = (X[:3] / X[3])

        # 重投影误差
        pixels = {}
        errs = []
        for sn in serials:
            det = detections[sn]
            pixels[sn] = (det.x, det.y)
            pt_h = np.append(pts_3d, 1.0)
            proj = matvec(self._P[sn], pt_h)
            proj = proj[:2] / proj[2]
            err = np.sqrt((proj[0] - det.x) ** 2 + (proj[1] - det.y) ** 2)
            errs.append(err)

        pts_3d_m = pts_3d * _WORLD_SCALE_M_PER_MM

        return Ball3D(
            x=float(pts_3d_m[0]),
            y=float(pts_3d_m[1]),
            z=float(pts_3d_m[2]),
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
