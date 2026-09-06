# -*- coding: utf-8 -*-
"""
车辆定位模块 — 车载多 AprilTag 的多目刚体位姿估计。

车上刚性安装若干块布局已知的竖直 AprilTag（当前两块：id0 右后、id1 左前），
布局（车体系中心 + 完整安装旋转）由 test_src/measure_car_tag_layout.py 实测，
写入车体配置的 vehicle_reference.apriltags。默认读 v0.4 车的
src/config/vehicle_v04.json；v0.3 车的布局仍在 src/config/arm_poe_racket_center.json
里，用 run_tracker.py --car-config 指过去即可回退。

流程：
  1. 接收多台相机的同步 BGR 图像
  2. cv2.aruco 检测 AprilTag (tag36h11) 的 4 个角点
  3. 所有相机 × 所有已配置 tag 的角点纳入同一刚体模型：
     世界角点 = 车位姿 (x, y, yaw) ∘ 车体系 tag 布局 ∘ tag 印面角点
  4. 在原始畸变像素域联合优化车的 x/y/yaw
  5. 按 (相机, tag) 视图的四角重投影误差剔除异常视图
  6. 直接输出车底盘中心位姿（车心即优化变量，无需 tag→车心换算）

双 tag 同时可见时 ~0.9 m 的中心基线让 yaw 远优于单 tag 16 cm 边长的短基线。
产品约定：locate() 首选两块 tag 联合拟合（x/y/yaw 全解）；只剩一块 tag 时退化为
「冻结最近一次可信 yaw、只解 x/y」并返回 yaw=None，由消费端保持自身 yaw 不更新
（2026-08-11 起；此前是单 tag 一律不发布）。自由 yaw 的单 tag 拟合仍只供诊断
（estimate_car_pose / estimate_pose 直调）。

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
_DEFAULT_VEHICLE_CONFIG = _SRC_DIR / "config" / "vehicle_v04.json"
_WORLD_SCALE_M_PER_MM = 1.0 / 1000.0
_HUBER_DELTA_PX = 3.0
_VIEW_OUTLIER_MIN_PX = 4.0
_VIEW_OUTLIER_RATIO = 2.5
_MAX_REPROJECTION_ERROR_PX = 8.0
_CAR_X_BOUNDS_M = (-3.0, 3.0)
_CAR_Y_BOUNDS_M = (0.0, 9.0)
_CAR_MAX_SPEED_MPS = 4.0
_CAR_JUMP_SLACK_M = 0.10
# 单 tag 退化拟合时，冻结用的「最近一次双 tag yaw」允许多陈旧 (s)。
# 车 yaw 变化慢：0811 场实测各定位空洞两端 yaw 差 1~5°/1.2~2.2s（≈3°/s 上界），
# 0.5s 陈旧 ⇒ ~1.5°，经 tag0 的 0.42m 安装杠杆只值 ~11mm，比整帧丢定位好两个量级。
# 再旧就不发——单 tag 自己的印面短基线 yaw（0.161m）不拿来解位置，见 locate()。
_SINGLE_TAG_YAW_MAX_AGE_S = 0.5


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
    # 底盘绕 z 轴旋转角 (rad)。None = 本帧给不出可信 yaw（单 tag 退化，位置是拿
    # 冻结的历史 yaw 解出来的），消费端必须保持自身 yaw 不更新，只吃 x/y。
    yaw: Optional[float]
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
            vehicle_config_path or str(_DEFAULT_VEHICLE_CONFIG)
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
        # 候选周长封顶：默认 4.0 让网柱/场地结构等巨型轮廓也进解码流程白耗时。
        # 0901 在 18F 真图（004_current 标定照 + 115811 live 帧）实测：单加 0.6 省 30%
        # （23.5→16.4ms/帧），检出与角点逐位不变；车 tag 现距边长 41~51px（周长率
        # ~0.09），到 0.6 还有 6 倍余量。⚠勿加 minMarkerPerimeterRate——远处小 tag
        # 有被杀风险，实测它也只再省 <1ms。
        params.maxMarkerPerimeterRate = 0.6
        # yaw 对角点误差很敏感；检测后使用灰度梯度做亚像素角点细化。
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        # 暗光重试计数（诊断用）：raw 全空 → 拉对比度重试的次数 / 其中救回来的次数
        self.low_light_retries = 0
        self.low_light_recovered = 0
        # 暗场自适应：连续多次「raw 空、增强能救」之后就直接从增强图起手。
        # 否则暗场每帧要跑两遍 detectMarkers（~100ms 一遍），后台定位跟不上：
        # 2026-08-09 050621 场实测 car_loc_dropped 从历来的 0 涨到 48%。
        # 直接起手后暗场恢复成一遍，代价回到修复前水平。
        self._prefer_enhanced = False
        self._raw_fail_streak = 0
        self._enhanced_since_probe = 0
        # 单 tag 退化路径用的冻结 yaw：只由「双 tag 且 yaw_valid」的结果刷新，
        # 单 tag 帧自己解出的 yaw 绝不回灌（否则误差会沿着退化链自我累积）。
        self._hold_yaw: Optional[float] = None
        self._hold_yaw_t: Optional[float] = None
        self._last_accepted_loc: Optional[CarLoc] = None
        self.single_tag_frames = 0     # 诊断计数：走退化路径发出的帧数

        # 逐相机跟踪窗 {序列号: ((x0, y0, x1, y1), 上次检出块数)}。每个线程只碰
        # 自己那台的键，dict 赋值在 GIL 下是原子的，locate() 的线程池可直接并发写。
        self._roi_state: dict[str, tuple[tuple[int, int, int, int], int]] = {}
        self._roi_since_full: dict[str, int] = {}
        self._roi_miss: dict[str, int] = {}

    # 连续 N 次 raw 空而增强有效 → 切到「增强起手」；之后每 M 次回探一次 raw，
    # 免得光线恢复了还一直吃增强图（增强图角点有 ~0.27° 偏移，能不用就不用）。
    _PREFER_ENHANCED_AFTER = 3
    _RAW_PROBE_EVERY = 60

    @staticmethod
    def _enhance_for_low_light(gray: np.ndarray) -> np.ndarray:
        """暗光下把 tag 的局部对比度拉回可解码范围。

        参数在「救回率」和「角点保真」之间实测选出（034554 场 288 个角点，与 raw
        检出角点比，tag 边长中位 45.7px）：linear stretch 只偏 0.04px 但仅救回 3/6；
        CLAHE 2.0/16 全救回且偏 0.213px≈0.27°；gamma0.60 0.446px；CLAHE 3.0/8 0.630px。
        增强图是非线性的，角点必然有偏移，而车 yaw 对角点敏感，所以要挑保真最好的一档。
        （试过「增强图定位 + 回原图重跑 cornerSubPix」把几何拉回来：暗图梯度太弱，
         细化会跑飞，p90 从 1.0px 恶化到 5.4px，已放弃。）
        CLAHE 对象每次新建：detect() 跑在 ThreadPoolExecutor 里，共享内部缓冲不安全。
        """
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(gray)

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

    # 车是慢速刚体：上一帧 tag 在哪，这一帧就在附近。全幅 detectMarkers 实测
    # 24~36ms/相机（4 台并发的 locate() 要 89ms，把车定位硬封在 ~11Hz），裁到上次
    # tag bbox 外扩 _ROI_PAD 之后是 6ms（0906 在 18F 真帧实测，检出与全幅逐位一致）。
    # 窗口里检出数少于上次就整幅重检一次 —— tag 被遮挡或新进画面最多多花一次全幅
    # 代价，不会永久漏；窗口全空同理落回全幅，所以最坏情况就是今天的成本。
    _ROI_PAD = 300      # px；按 tag 边长 ~45px 折算约 1m 车体位移余量
    # 窗口按「上次检出的所有 tag 的并集」画，而 tag 全都刚性长在车上：窗口里只要
    # 还有 tag，暂时被挡住的那块回来时也仍在窗口内。所以少检出一块**不**触发重扫，
    # 只有窗口连续空这么多次才认为真跟丢，去整幅重扫。空窗那几次该相机本帧不贡献
    # 观测（另外几台还在），比每少一块就全幅重来便宜得多。窗口 300px 折合约 1m
    # 车体位移，而两次采样才隔 25~50ms，物理上跑不出去，空窗基本只来自遮挡/反光。
    _ROI_MISS_LIMIT = 3
    # 安全网：窗口缩到单块 tag 上、另一块回来时恰好落在窗外，靠周期性整幅重扫兜。
    # 看全所有配置 tag 时不重扫，正常场次不花钱。
    _ROI_RESCAN_EVERY = 60

    def detect(self, image: np.ndarray, serial: str = "") -> list[CarDetection]:
        """检测整张图像中的所有 AprilTag (tag36h11)。

        serial 非空时走跟踪窗快路径。角点一律还原成**全图坐标**再返回，标定内参
        和三角化都按全图算，下游无需感知窗口的存在。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        roi_state = self._roi_state.get(serial) if serial else None
        if roi_state is not None and roi_state[1] < len(self._tags) and (
            self._roi_since_full.get(serial, 0) >= self._ROI_RESCAN_EVERY
        ):
            roi_state = None
        if roi_state is not None:
            (x0, y0, x1, y1), _ = roi_state
            sub = gray[y0:y1, x0:x1]
            # 快路径不碰暗光状态机（它是全图口径的），只跟随它已经判定的暗场结论，
            # 否则暗场里窗口每帧必空、白白多跑一遍。
            if self._prefer_enhanced:
                sub = self._enhance_for_low_light(sub)
            corners_list, ids, _ = self._detector.detectMarkers(
                np.ascontiguousarray(sub))
            if ids is not None:
                self._roi_miss[serial] = 0
                return self._collect(
                    corners_list, ids, serial, x0, y0, gray.shape, full=False)
            self._roi_miss[serial] = self._roi_miss.get(serial, 0) + 1
            if self._roi_miss[serial] < self._ROI_MISS_LIMIT:
                return []

        corners_list, ids = self._detect_full(gray)
        if serial:
            self._roi_miss[serial] = 0
        if ids is None:
            self._roi_state.pop(serial, None)
            self._roi_since_full.pop(serial, None)
            return []
        return self._collect(
            corners_list, ids, serial, 0, 0, gray.shape, full=True)

    def _collect(
        self,
        corners_list,
        ids,
        serial: str,
        x0: int,
        y0: int,
        shape: tuple[int, ...],
        *,
        full: bool,
    ) -> list[CarDetection]:
        """窗口坐标 → 全图坐标，顺手刷新该相机的跟踪窗。"""
        results = []
        for i, tag_id in enumerate(ids.ravel()):
            corners = corners_list[i].reshape(4, 2) + np.array(
                [x0, y0], dtype=np.float32)
            results.append(CarDetection(
                tag_id=int(tag_id),
                cx=float(corners[:, 0].mean()),
                cy=float(corners[:, 1].mean()),
                corners=corners,
            ))
        if serial and results:
            pts = np.concatenate([d.corners for d in results], axis=0)
            h, w = shape[0], shape[1]
            pad = self._ROI_PAD
            self._roi_state[serial] = ((
                max(0, int(pts[:, 0].min()) - pad),
                max(0, int(pts[:, 1].min()) - pad),
                min(w, int(pts[:, 0].max()) + pad),
                min(h, int(pts[:, 1].max()) + pad),
            ), len(results))
            self._roi_since_full[serial] = (
                0 if full else self._roi_since_full.get(serial, 0) + 1)
        return results

    def _detect_full(self, gray: np.ndarray):
        """全幅检测 + 暗光救援状态机，返回 (corners_list, ids)。"""

        # 暗光救援：光线掉下去后 tag 白块只剩 20~30 灰度（黑块 ~2），ArUco 的自适应
        # 阈值取不出边缘，整张图一个 tag 都出不来 —— 2026-08-09 夜实测就是这样：场景
        # 均值和能用的场次一样（24.5 vs 24.1），但 tag 局部动态范围从 ~48 掉到 ~28，
        # car_loc 直接 100% miss。拉一次局部对比度再检即可全部救回，且不必动曝光
        # （动曝光会加重回球段拖影，见 detection_shape_gate 那条链路）。
        probe_raw = (self._enhanced_since_probe >= self._RAW_PROBE_EVERY)
        if self._prefer_enhanced and not probe_raw:
            # 暗场稳态：跳过注定失败的 raw 一遍，只跑增强图，成本回到一次 detectMarkers
            self._enhanced_since_probe += 1
            self.low_light_retries += 1
            corners_list, ids, _ = self._detector.detectMarkers(
                self._enhance_for_low_light(gray))
            if ids is None:
                self._prefer_enhanced = False       # 连增强都不行了，回到常规路径重新判断
                return None, None
            self.low_light_recovered += 1
        else:
            corners_list, ids, _ = self._detector.detectMarkers(gray)
            if ids is not None:
                self._prefer_enhanced = False       # raw 能用（可能光线恢复了）
                self._raw_fail_streak = 0
                self._enhanced_since_probe = 0
            else:
                self._enhanced_since_probe = 0
                self.low_light_retries += 1
                corners_list, ids, _ = self._detector.detectMarkers(
                    self._enhance_for_low_light(gray))
                if ids is None:
                    self._raw_fail_streak = 0
                    return None, None
                self.low_light_recovered += 1
                self._raw_fail_streak += 1
                if self._raw_fail_streak >= self._PREFER_ENHANCED_AFTER:
                    self._prefer_enhanced = True

        return corners_list, ids

    # ── 多视图刚体位姿估计 ─────────────────────────────────────────────

    def estimate_car_pose(
        self,
        tag_detections: dict[int, dict[str, CarDetection]],
        t: float = 0.0,
        fixed_yaw: Optional[float] = None,
    ) -> Optional[CarLoc]:
        """
        对多台相机中若干车载 tag 的四角做车位姿联合刚体拟合。

        Args:
            tag_detections: {tag_id: {序列号: CarDetection}}，只使用已配置的
                tag；至少一块 tag 需被 >=2 台相机看到（用于初值三角化），
                其余 1 台相机的 tag 观测也会加入联合优化。
            t: 时间戳 (perf_counter)。
            fixed_yaw: 给定时 yaw 冻结在该值上只解 x/y，返回的 CarLoc.yaw 为
                None、yaw_valid 为 False（单 tag 退化路径，见 locate()）。

        Returns:
            CarLoc，或 None（初值不可解 / 剔除后有效相机不足 2 台）。
        """
        usable = {
            tag_id: dict(cam_dets)
            for tag_id, cam_dets in tag_detections.items()
            if tag_id in self._tags and cam_dets
        }
        pose = self._initial_pose(usable, fixed_yaw)
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
            pose, residual = self._fit_car_pose(
                pose, units, free_yaw=(fixed_yaw is None)
            )
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
            # yaw 冻结解出来的位姿不带 yaw 信息：本帧的 yaw 是历史值，原样发回去
            # 会让消费端把陈旧值当新观测反复吸收，必须显式报 None。
            yaw=None if fixed_yaw is not None else yaw,
            yaw_valid=(
                fixed_yaw is None
                and reprojection_error <= _MAX_REPROJECTION_ERROR_PX
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
        single_tag_fallback: bool = True,
    ) -> Optional[CarLoc]:
        """
        检测 + 车位姿联合估计一步完成。

        在所有图像中检测 AprilTag，收集所有已配置车载 tag 的观测，
        联合做车位姿刚体拟合。

        **首选路径**是 min_tags 块 tag 联合拟合（默认 2），x/y/yaw 全解。

        **退化路径**（single_tag_fallback，2026-08-11 加）：首选路径拿不到结果、
        但仍有一块 tag 被 >=2 台相机看到时，用最近一次可信 yaw 冻结着只解 x/y，
        返回 `yaw=None` 的 CarLoc。动机是 0811 053055 场实测：tag1 被臂座自遮挡，
        373/378 两台常年看不到它，而与门要求两块都在，于是击球瞬间成片丢定位
        （18 抛里 4 抛的真值因此为空，空洞 0.65~2.2s）；同批 miss 帧里 tag0 在
        >=2 台相机的检出率是 15/15。位置来自 tag 中心多视图三角化（无短基线
        问题），只有 0.42m 安装杠杆要乘 yaw 误差：冻结 yaw 陈旧 <=0.5s ⇒ ~1.5°
        ⇒ ~11mm，远好于整帧丢定位。yaw 一律不发（None），由消费端保持自身值。

        Args:
            images: {序列号: BGR 图像}
            t: 时间戳。
            min_tags: 首选路径中最少 tag 块数。
            single_tag_fallback: 关掉即回到 2026-08-11 之前的纯与门行为。

        Returns:
            CarLoc 或 None（无 tag 被 >=2 台相机检测到 / 首选路径失败且退化
            路径不可用）。
        """
        # 并行检测所有相机（复用线程池）
        all_dets = {}
        futures = {self._pool.submit(self.detect, img, sn): sn
                   for sn, img in images.items()}
        for fut in futures:
            all_dets[futures[fut]] = fut.result()

        # 收集已配置 tag 的检测：{tag_id: {sn: CarDetection}}。
        # 同一相机同一车载 tag_id 出现多次时，该相机本帧的整条车体观测有歧义，
        # 所有 tag 一起丢弃；绝不能让后一个检测静默覆盖前一个。
        tag_cameras: dict[int, dict[str, CarDetection]] = {}
        for sn, dets in all_dets.items():
            configured_ids = [d.tag_id for d in dets if d.tag_id in self._tags]
            if len(configured_ids) != len(set(configured_ids)):
                continue
            for d in dets:
                if d.tag_id in self._tags:
                    tag_cameras.setdefault(d.tag_id, {})[sn] = d

        # 无论走哪条路径，都至少要有一块 tag 能被三角化出中心
        if not any(len(cam_dets) >= 2 for cam_dets in tag_cameras.values()):
            return None

        if len(tag_cameras) >= min_tags:
            result = self.estimate_car_pose(tag_cameras, t)
            # 离群剔除后有 tag 整块出局（如仅剩单视图且被剔）也算首选路径失败
            if result is not None and len(result.tag_ids) >= min_tags:
                # 已经形成首选解却没过最终质量门时，整帧拒绝；不能拿同一批污染
                # 观测再绕进单 tag fallback。
                return self._finalize_location(result, single_tag_fallback=False)

        if not single_tag_fallback:
            return None
        yaw = self._fresh_hold_yaw(t)
        if yaw is None:
            # 还没锁定过 yaw（或已经太旧）：单 tag 的印面短基线 yaw 不足以撑住
            # 0.42m 杠杆，宁可不发，别用一个说不清误差的位置污染下游。
            return None
        result = self.estimate_car_pose(tag_cameras, t, fixed_yaw=yaw)
        if result is None:
            return None
        return self._finalize_location(result, single_tag_fallback=True)

    def _finalize_location(
        self,
        result: CarLoc,
        *,
        single_tag_fallback: bool,
    ) -> Optional[CarLoc]:
        """最终 fail-closed 门；只有通过的结果才能刷新定位状态并向下游返回。"""
        if not all(math.isfinite(value) for value in (
            result.x, result.y, result.t, result.reprojection_error,
        )):
            return None
        if result.yaw_valid and (
            result.yaw is None or not math.isfinite(result.yaw)
        ):
            return None
        if not (0.0 <= result.reprojection_error <= _MAX_REPROJECTION_ERROR_PX):
            return None
        if not (
            _CAR_X_BOUNDS_M[0] <= result.x <= _CAR_X_BOUNDS_M[1]
            and _CAR_Y_BOUNDS_M[0] <= result.y <= _CAR_Y_BOUNDS_M[1]
        ):
            return None

        if self._last_accepted_loc is not None:
            dt = result.t - self._last_accepted_loc.t
            if dt <= 0.0:
                return None
            distance = math.hypot(
                result.x - self._last_accepted_loc.x,
                result.y - self._last_accepted_loc.y,
            )
            if distance > _CAR_JUMP_SLACK_M + _CAR_MAX_SPEED_MPS * dt:
                return None

        self._last_accepted_loc = result
        if result.yaw is not None and result.yaw_valid:
            self._hold_yaw = result.yaw
            self._hold_yaw_t = result.t
        if single_tag_fallback:
            self.single_tag_frames += 1
        return result

    def _fresh_hold_yaw(self, t: float) -> Optional[float]:
        """最近一次可信 yaw，超过 _SINGLE_TAG_YAW_MAX_AGE_S 就作废。"""
        if self._hold_yaw is None or self._hold_yaw_t is None:
            return None
        age = t - self._hold_yaw_t
        if not (0.0 <= age <= _SINGLE_TAG_YAW_MAX_AGE_S):
            return None
        return self._hold_yaw

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
        free_yaw: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Huber Gauss-Newton 优化车位姿 [x_mm, y_mm, yaw_rad]。

        free_yaw=False 时 yaw 冻结在初值上，只解 x/y（单 tag 退化用）：印面
        0.161m 短基线解出的 yaw 会经 0.42m 安装杠杆放大成位置误差，宁可用
        最近一次双 tag 的 yaw 冻住，把这一帧的自由度降到 2。
        """
        pose = initial_pose.copy()
        eps = np.array([0.1, 0.1, 1e-4], dtype=np.float64)
        max_step = np.array([200.0, 200.0, 0.35], dtype=np.float64)
        columns = (0, 1, 2) if free_yaw else (0, 1)

        for _ in range(10):
            residual = self._car_corner_residuals(pose, units)
            corner_norms = np.linalg.norm(residual.reshape(-1, 2), axis=1)
            corner_weights = np.ones_like(corner_norms)
            large = corner_norms > _HUBER_DELTA_PX
            corner_weights[large] = _HUBER_DELTA_PX / corner_norms[large]
            row_weights = np.repeat(np.sqrt(corner_weights), 2)

            jacobian = np.empty((residual.size, len(columns)), dtype=np.float64)
            for slot, column in enumerate(columns):
                plus = pose.copy()
                minus = pose.copy()
                plus[column] += eps[column]
                minus[column] -= eps[column]
                jacobian[:, slot] = (
                    self._car_corner_residuals(plus, units)
                    - self._car_corner_residuals(minus, units)
                ) / (2.0 * eps[column])

            free_step = solve_least_squares(
                jacobian * row_weights[:, None],
                -residual * row_weights,
            )
            step = np.zeros(3, dtype=np.float64)
            step[list(columns)] = free_step
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

    def _xy_from_centers(
        self,
        centers: dict[int, np.ndarray],
        yaw: float,
    ) -> np.ndarray:
        """已知 yaw 时，由若干 tag 的三角化中心反解车心 xy（多块取平均）。"""
        c = math.cos(yaw)
        s = math.sin(yaw)
        xy = np.zeros(2, dtype=np.float64)
        for tag_id, center_w in centers.items():
            a = self._tags[tag_id].center_car_mm[:2]
            xy += center_w[:2] - np.array(
                [c * a[0] - s * a[1], s * a[0] + c * a[1]]
            )
        return xy / len(centers)

    def _initial_pose(
        self,
        tag_detections: dict[int, dict[str, CarDetection]],
        fixed_yaw: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """DLT 三角化生成车位姿初值 [x_mm, y_mm, yaw_rad]。

        双 tag（各 >=2 相机）：由两中心基线解 yaw；单 tag：由印面角点边
        方向 + 安装旋转解 yaw。fixed_yaw 给定时直接用它，不做任何 yaw 估计
        （单 tag 退化路径：位置纯靠 tag 中心三角化 + 冻结 yaw 的安装杠杆）。
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

        if fixed_yaw is not None:
            yaw = float(fixed_yaw)
            xy = self._xy_from_centers(centers, yaw)
        elif len(centers) >= 2:
            tag_a, tag_b = sorted(centers)[:2]
            w_ab = centers[tag_b][:2] - centers[tag_a][:2]
            a_ab = (
                self._tags[tag_b].center_car_mm[:2]
                - self._tags[tag_a].center_car_mm[:2]
            )
            yaw = (
                math.atan2(w_ab[1], w_ab[0]) - math.atan2(a_ab[1], a_ab[0])
            )
            xy = self._xy_from_centers(
                {tag_id: centers[tag_id] for tag_id in (tag_a, tag_b)}, yaw
            )
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
            xy = self._xy_from_centers({tag_id: center_w}, yaw)

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
