# -*- coding: utf-8 -*-
"""
双目相机标定模块 — 黑白网格板检测、双目标定、地面坐标注册。

将 AprilTag 板退化为 6×6 黑方格网格板，
使用 SimpleBlobDetector + findCirclesGrid 检测全部 36 个 tag 中心。
支持自动左右相机判定、带立体约束的联合 PnP 优化。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- 默认标定板参数 ----------
DEFAULT_GRID_COLS = 6          # tag 列数
DEFAULT_GRID_ROWS = 6          # tag 行数
DEFAULT_TAG_SPACING = 71.5     # tag 中心间距 mm
DEFAULT_TAG_SIZE = 55.0        # 单个 tag 边长 mm
DEFAULT_TAG_FAMILY = "tag36h11"

# 检测阈值
_MIN_VALID_PAIRS = 5           # 标定所需最少有效图像对
_ROTATION_ERR_THRESHOLD = 5.0  # 旋转校正后最大允许误差 (px)


# ================================================================
#  数据类
# ================================================================

@dataclass
class BoardConfig:
    """AprilTag 网格板参数。"""
    grid_cols: int = DEFAULT_GRID_COLS       # 列数
    grid_rows: int = DEFAULT_GRID_ROWS       # 行数
    tag_spacing: float = DEFAULT_TAG_SPACING  # 中心间距 mm
    tag_size: float = DEFAULT_TAG_SIZE        # tag 边长 mm
    tag_family: str = DEFAULT_TAG_FAMILY      # tag 家族


@dataclass
class StereoCalibResult:
    """双目标定完整结果。"""
    # 内参
    K1: np.ndarray = field(default_factory=lambda: np.eye(3))
    D1: np.ndarray = field(default_factory=lambda: np.zeros(5))
    K2: np.ndarray = field(default_factory=lambda: np.eye(3))
    D2: np.ndarray = field(default_factory=lambda: np.zeros(5))
    # 双目外参 (cam2 = R_stereo @ cam1 + T_stereo)
    R_stereo: np.ndarray = field(default_factory=lambda: np.eye(3))
    T_stereo: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    E: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    F: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    # 世界位姿（地面注册后填充）
    R1_world: Optional[np.ndarray] = None
    t1_world: Optional[np.ndarray] = None
    pos1_world: Optional[np.ndarray] = None
    R2_world: Optional[np.ndarray] = None
    t2_world: Optional[np.ndarray] = None
    pos2_world: Optional[np.ndarray] = None
    # 相机序列号（左在前）
    serial_left: str = ""
    serial_right: str = ""
    # 诊断
    stereo_rms: float = 0.0
    mono_rms1: float = 0.0
    mono_rms2: float = 0.0
    ground_reproj_error: float = 0.0
    num_valid_pairs: int = 0
    num_total_pairs: int = 0
    image_size: tuple[int, int] = (0, 0)
    board_config: Optional[BoardConfig] = None
    dict_name: str = ""

    # ---------- 序列化 ----------

    def save(self, path: Path) -> None:
        """将标定结果保存为 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        data = {
            "serial_left": self.serial_left,
            "serial_right": self.serial_right,
            "image_size": list(self.image_size),
            "K1": _convert(self.K1), "D1": _convert(self.D1),
            "K2": _convert(self.K2), "D2": _convert(self.D2),
            "R_stereo": _convert(self.R_stereo),
            "T_stereo": _convert(self.T_stereo),
            "E": _convert(self.E), "F": _convert(self.F),
            "R1_world": _convert(self.R1_world) if self.R1_world is not None else None,
            "t1_world": _convert(self.t1_world) if self.t1_world is not None else None,
            "pos1_world": _convert(self.pos1_world) if self.pos1_world is not None else None,
            "R2_world": _convert(self.R2_world) if self.R2_world is not None else None,
            "t2_world": _convert(self.t2_world) if self.t2_world is not None else None,
            "pos2_world": _convert(self.pos2_world) if self.pos2_world is not None else None,
            "diagnostics": {
                "stereo_rms": self.stereo_rms,
                "mono_rms1": self.mono_rms1,
                "mono_rms2": self.mono_rms2,
                "ground_reproj_error": self.ground_reproj_error,
                "num_valid_pairs": self.num_valid_pairs,
                "num_total_pairs": self.num_total_pairs,
                "dict_name": self.dict_name,
            },
            "board": {
                "grid_cols": self.board_config.grid_cols,
                "grid_rows": self.board_config.grid_rows,
                "tag_spacing": self.board_config.tag_spacing,
                "tag_size": self.board_config.tag_size,
                "tag_family": self.board_config.tag_family,
            } if self.board_config else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> StereoCalibResult:
        """从 JSON 加载标定结果。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _to_arr(v):
            return np.array(v, dtype=np.float64) if v is not None else None

        diag = data.get("diagnostics", {})
        board_data = data.get("board")
        board = BoardConfig(**board_data) if board_data else None

        return cls(
            K1=_to_arr(data["K1"]), D1=_to_arr(data["D1"]),
            K2=_to_arr(data["K2"]), D2=_to_arr(data["D2"]),
            R_stereo=_to_arr(data["R_stereo"]),
            T_stereo=_to_arr(data["T_stereo"]),
            E=_to_arr(data["E"]), F=_to_arr(data["F"]),
            R1_world=_to_arr(data.get("R1_world")),
            t1_world=_to_arr(data.get("t1_world")),
            pos1_world=_to_arr(data.get("pos1_world")),
            R2_world=_to_arr(data.get("R2_world")),
            t2_world=_to_arr(data.get("t2_world")),
            pos2_world=_to_arr(data.get("pos2_world")),
            serial_left=data.get("serial_left", ""),
            serial_right=data.get("serial_right", ""),
            stereo_rms=diag.get("stereo_rms", 0.0),
            mono_rms1=diag.get("mono_rms1", 0.0),
            mono_rms2=diag.get("mono_rms2", 0.0),
            ground_reproj_error=diag.get("ground_reproj_error", 0.0),
            num_valid_pairs=diag.get("num_valid_pairs", 0),
            num_total_pairs=diag.get("num_total_pairs", 0),
            image_size=tuple(data.get("image_size", (0, 0))),
            board_config=board,
            dict_name=diag.get("dict_name", ""),
        )


# ================================================================
#  Blob 网格检测（核心检测模块）
# ================================================================

def _make_blob_detector(min_area: int = 200, max_area: int = 10000,
                        min_inertia: float = 0.4,
                        min_convexity: float = 0.5) -> cv2.SimpleBlobDetector:
    """创建用于检测黑色 tag 方块的 blob 检测器。"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0           # 检测暗色 blob
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = min_convexity
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia
    return cv2.SimpleBlobDetector_create(params)


def _detect_grid(gray: np.ndarray, board: BoardConfig,
                 blob_det: cv2.SimpleBlobDetector) -> Optional[np.ndarray]:
    """
    在图像中检测 6×6 黑方格网格的全部中心点。

    将每个 AprilTag 视为纯黑方块，用 findCirclesGrid 找到完整网格。
    返回反色图像上的检测结果（因为 blob 检测器找白色 blob，
    而我们的 tag 是黑色的，所以先反色）。

    Returns:
        (36, 1, 2) float32 中心坐标数组，或 None（未找到完整网格）
    """
    inv = cv2.bitwise_not(gray)
    pattern = (board.grid_cols, board.grid_rows)
    flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
    found, centers = cv2.findCirclesGrid(inv, pattern, None, flags, blob_det)
    if found:
        return centers
    return None


def _draw_detection(gray: np.ndarray, centers: Optional[np.ndarray],
                     board: BoardConfig, label: str = "") -> np.ndarray:
    """在灰度图上绘制检测到的网格中心点，返回彩色标注图。"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if centers is not None:
        pts = centers.reshape(-1, 2)
        n = board.grid_cols
        for i, (px, py) in enumerate(pts):
            r, c = i // n, i % n
            cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 0), 2)
            cv2.putText(vis, f"{r},{c}", (int(px) + 6, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        status = f"OK ({len(pts)} pts)"
        color = (0, 255, 0)
    else:
        status = "FAIL"
        color = (0, 0, 255)
    if label:
        cv2.putText(vis, f"{label}: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis


def _detect_grid_masked(gray: np.ndarray, board: BoardConfig) -> Optional[np.ndarray]:
    """
    板子区域裁剪检测：先定位白色板子，裁剪后用标准方法检测。

    适用于背景杂乱的地面图像。步骤:
    1. 阈值定位白色板子区域
    2. 裁剪到板子附近区域
    3. 在裁剪区域内运行标准检测（多组参数）
    4. 将坐标映射回原图
    """
    h, w = gray.shape
    pattern = (board.grid_cols, board.grid_rows)
    flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

    param_sets = [
        dict(min_area=200, max_area=10000, min_inertia=0.4, min_convexity=0.5),
        dict(min_area=80,  max_area=8000, min_inertia=0.3, min_convexity=0.4),
        dict(min_area=80,  max_area=10000, min_inertia=0.2, min_convexity=0.3),
        dict(min_area=30,  max_area=10000, min_inertia=0.2, min_convexity=0.3),
    ]

    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    for thresh_val in [170, 160, 150, 140, 130, 120, 110]:
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
        kern = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < 2000 or area > h * w * 0.3:
                continue
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if rw == 0 or rh == 0:
                continue
            if max(rw, rh) / min(rw, rh) > 4.0:
                continue

            # 裁剪到板子区域 + 50% 边距
            x, y, bw, bh = cv2.boundingRect(c)
            pad_x = int(bw * 0.5)
            pad_y = int(bh * 0.5)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + bw + pad_x)
            y1 = min(h, y + bh + pad_y)
            crop = gray[y0:y1, x0:x1]

            inv_crop = cv2.bitwise_not(crop)

            for ps in param_sets:
                det = _make_blob_detector(**ps)
                found, centers = cv2.findCirclesGrid(
                    inv_crop, pattern, None, flags, det)
                if found:
                    # 坐标映射回原图
                    centers[:, :, 0] += x0
                    centers[:, :, 1] += y0
                    log.info("裁剪检测成功 (thresh=%d, area=%.0f, crop=%dx%d+%d+%d)",
                             thresh_val, area, x1 - x0, y1 - y0, x0, y0)
                    return centers
    return None


def _grid_obj_points(board: BoardConfig) -> np.ndarray:
    """
    生成 6×6 网格中心的 3D 坐标。

    findCirclesGrid 返回顺序：从图像中的某个角开始，逐行从左到右。
    3D 坐标使用标定坐标系（任意一致即可，标定不依赖绝对朝向）：
      第 i 个点: row = i // cols, col = i % cols
      3D = (col * spacing, row * spacing, 0)

    Returns:
        (36, 3) float32
    """
    pts = []
    for row in range(board.grid_rows):
        for col in range(board.grid_cols):
            pts.append([col * board.tag_spacing, row * board.tag_spacing, 0.0])
    return np.array(pts, dtype=np.float32)


def _grid_obj_points_world(board: BoardConfig, rotation: int = 0) -> np.ndarray:
    """
    生成 6×6 网格中心的世界坐标。

    世界坐标系：原点在板子左下角（图像中板子下方），X 向右，Y 向上，Z=0。

    rotation: 0-3，表示 findCirclesGrid 返回顺序与世界坐标的旋转关系。
      0: 第0点=世界左上 (row=0, col=0 → world Y 最大)
      1: 顺时针90度
      2: 180度
      3: 逆时针90度

    Returns:
        (36, 3) float64
    """
    rows, cols = board.grid_rows, board.grid_cols
    sp = board.tag_spacing
    pts = []

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        if rotation == 0:
            # 第 0 点 = 图像左上 → 世界左上 (x 小, y 大)
            x = c * sp
            y = (rows - 1 - r) * sp
        elif rotation == 1:
            # 顺时针 90 度
            x = r * sp
            y = c * sp
        elif rotation == 2:
            # 180 度
            x = (cols - 1 - c) * sp
            y = r * sp
        elif rotation == 3:
            # 逆时针 90 度
            x = (rows - 1 - r) * sp
            y = (cols - 1 - c) * sp
        else:
            raise ValueError(f"rotation must be 0-3, got {rotation}")
        pts.append([x, y, 0.0])

    return np.array(pts, dtype=np.float64)


def _grid_obj_points_vertical(board: BoardConfig, rotation: int = 0,
                               z_offset: float = 0.0) -> np.ndarray:
    """
    生成垂直放置标定板的 6×6 网格中心世界坐标。

    世界坐标系：
      X 沿板子水平方向（向右）
      Y 垂直板面方向（从板面指向相机）= 0
      Z 竖直向上，Z=0 为板子底边（地面）

    z_offset: 板子底边到最底排 tag 中心的距离 (mm)，
              即 board_bottom_margin + tag_size/2。

    rotation: 0-3，同 _grid_obj_points_world。

    Returns:
        (36, 3) float64
    """
    rows, cols = board.grid_rows, board.grid_cols
    sp = board.tag_spacing
    pts = []

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        if rotation == 0:
            x = c * sp
            z = z_offset + (rows - 1 - r) * sp
        elif rotation == 1:
            x = r * sp
            z = z_offset + c * sp
        elif rotation == 2:
            x = (cols - 1 - c) * sp
            z = z_offset + r * sp
        elif rotation == 3:
            x = (rows - 1 - r) * sp
            z = z_offset + (cols - 1 - c) * sp
        else:
            raise ValueError(f"rotation must be 0-3, got {rotation}")
        pts.append([x, 0.0, z])

    return np.array(pts, dtype=np.float64)


def _grid_rotation_perms(n: int) -> list[np.ndarray]:
    """
    生成 n×n 网格在 row-major 排列下 4 种旋转的索引排列。

    findCirclesGrid 对正方形对称网格有 4 重旋转歧义 (0°, 90°CW, 180°, 90°CCW)。
    返回 4 个排列数组, perm[rot] 使得 centers[perm[rot]] 等价于
    将 centers 旋转 rot*90° 后的排列。

    Returns:
        [perm_0, perm_90cw, perm_180, perm_90ccw], each (n*n,) int
    """
    perms = []
    for rot in range(4):
        perm = np.empty(n * n, dtype=int)
        for j in range(n * n):
            r, c = j // n, j % n
            if rot == 0:
                perm[j] = r * n + c
            elif rot == 1:   # 90° CW: (r,c) → (c, n-1-r)
                perm[j] = c * n + (n - 1 - r)
            elif rot == 2:   # 180°: (r,c) → (n-1-r, n-1-c)
                perm[j] = (n - 1 - r) * n + (n - 1 - c)
            elif rot == 3:   # 90° CCW: (r,c) → (n-1-c, r)
                perm[j] = (n - 1 - c) * n + r
        perms.append(perm)
    return perms


# ================================================================
#  地面注册工具函数
# ================================================================

def _sampson_distance(pts1: np.ndarray, pts2: np.ndarray,
                      F: np.ndarray) -> np.ndarray:
    """
    计算点对的 Sampson 距离 (对极误差的一阶近似)。

    Args:
        pts1: (N, 2) 左相机 2D 点
        pts2: (N, 2) 右相机 2D 点
        F: (3, 3) 基础矩阵

    Returns:
        (N,) Sampson 距离 (像素)
    """
    ones = np.ones((len(pts1), 1))
    p1h = np.hstack([pts1, ones])
    p2h = np.hstack([pts2, ones])
    Fp1 = (F @ p1h.T).T       # (N, 3)
    FTp2 = (F.T @ p2h.T).T    # (N, 3)
    num = np.sum(p2h * Fp1, axis=1) ** 2
    den = Fp1[:, 0]**2 + Fp1[:, 1]**2 + FTp2[:, 0]**2 + FTp2[:, 1]**2
    return np.sqrt(num / den)


def _ground_residuals(params, obj_pts, img_pts_1, img_pts_2,
                      K1, D1, K2, D2, R_stereo, T_stereo):
    """6-DOF 联合残差：左相机 rvec/tvec，右相机由立体约束推导。"""
    rvec1 = params[:3].reshape(3, 1)
    tvec1 = params[3:6].reshape(3, 1)

    proj1, _ = cv2.projectPoints(obj_pts, rvec1, tvec1, K1, D1)
    res1 = (proj1.reshape(-1, 2) - img_pts_1.reshape(-1, 2)).ravel()

    R1, _ = cv2.Rodrigues(rvec1)
    R2 = R_stereo @ R1
    t2 = R_stereo @ tvec1 + T_stereo.reshape(3, 1)
    rvec2, _ = cv2.Rodrigues(R2)

    proj2, _ = cv2.projectPoints(obj_pts, rvec2, t2, K2, D2)
    res2 = (proj2.reshape(-1, 2) - img_pts_2.reshape(-1, 2)).ravel()

    return np.concatenate([res1, res2])


# ================================================================
#  StereoCalibrator
# ================================================================

class StereoCalibrator:
    """
    双目相机标定器（黑白网格板，将 AprilTag 板退化为纯黑方格网格）。

    用法::

        calibrator = StereoCalibrator(
            image_dir_1=Path("calibration/images/DA8199285"),
            image_dir_2=Path("calibration/images/DA8199402"),
            serial_1="DA8199285", serial_2="DA8199402",
        )
        result = calibrator.run()
        result.save(Path("config/stereo_calib.json"))
    """

    def __init__(
        self,
        image_dir_1: Path,
        image_dir_2: Path,
        serial_1: str = "",
        serial_2: str = "",
        board: Optional[BoardConfig] = None,
        ground_index: int = 52,
        calibration_range: tuple[int, int] = (1, 51),
        save_annotations: bool = False,
        ground_vertical: bool = False,
        ground_bottom_margin: float = 0.0,
    ):
        self._dir1 = Path(image_dir_1)
        self._dir2 = Path(image_dir_2)
        self._serial1 = serial_1
        self._serial2 = serial_2
        self._board = board or BoardConfig()
        self._ground_idx = ground_index
        self._cal_range = calibration_range
        self._save_annotations = save_annotations
        self._ground_vertical = ground_vertical
        # z_offset = 板底边到最底排 tag 中心距离
        self._ground_z_offset = ground_bottom_margin + board.tag_size / 2 if ground_vertical else 0.0

    # ---------- 公开接口 ----------

    def run(self) -> StereoCalibResult:
        """执行完整标定流程。"""
        b = self._board
        blob_det = _make_blob_detector()

        # 1. 检测所有图像对
        (obj_pts_list, img1_list, img2_list,
         valid_indices, image_size) = self._detect_all(blob_det, b)

        num_total = self._cal_range[1] - self._cal_range[0] + 1
        log.info("有效图像对: %d / %d", len(valid_indices), num_total)
        if len(valid_indices) < _MIN_VALID_PAIRS:
            raise RuntimeError(
                f"有效图像对不足: {len(valid_indices)} < {_MIN_VALID_PAIRS}")

        # 2. 双目标定
        (K1, D1, K2, D2, R_stereo, T_stereo, E, F,
         stereo_rms, mono_rms1, mono_rms2) = self._stereo_calibrate(
            obj_pts_list, img1_list, img2_list, image_size)

        log.info("单目 RMS: cam1=%.4f  cam2=%.4f", mono_rms1, mono_rms2)
        log.info("双目 RMS: %.4f", stereo_rms)
        log.info("基线: %.1f mm", np.linalg.norm(T_stereo))

        # 3. 左右判定
        need_swap, serial_left, serial_right = self._determine_left_right(
            R_stereo, T_stereo, self._serial1, self._serial2)
        if need_swap:
            log.info("交换左右: %s=左, %s=右", serial_left, serial_right)
            K1, K2 = K2, K1
            D1, D2 = D2, D1
            R_stereo_new = R_stereo.T
            T_stereo_new = -R_stereo.T @ T_stereo
            R_stereo, T_stereo = R_stereo_new, T_stereo_new
            E, F = E.T, F.T
            mono_rms1, mono_rms2 = mono_rms2, mono_rms1
            self._dir1, self._dir2 = self._dir2, self._dir1

        # 4. 地面注册（可选）
        R1w = t1w = pos1w = R2w = t2w = pos2w = None
        ground_err = 0.0
        try:
            R1w, t1w, pos1w, R2w, t2w, pos2w, ground_err = self._register_ground(
                b, K1, D1, K2, D2, R_stereo, T_stereo, F)
            log.info("地面注册 RMS: %.4f px", ground_err)
            log.info("左相机世界坐标: %s", pos1w.ravel())
            log.info("右相机世界坐标: %s", pos2w.ravel())
        except RuntimeError as e:
            log.warning("地面注册跳过: %s", e)

        return StereoCalibResult(
            K1=K1, D1=D1, K2=K2, D2=D2,
            R_stereo=R_stereo, T_stereo=T_stereo, E=E, F=F,
            R1_world=R1w, t1_world=t1w, pos1_world=pos1w,
            R2_world=R2w, t2_world=t2w, pos2_world=pos2w,
            serial_left=serial_left, serial_right=serial_right,
            stereo_rms=stereo_rms, mono_rms1=mono_rms1, mono_rms2=mono_rms2,
            ground_reproj_error=ground_err,
            num_valid_pairs=len(valid_indices), num_total_pairs=num_total,
            image_size=image_size,
            board_config=b,
            dict_name=b.tag_family,
        )

    # ---------- 网格检测 ----------

    def _detect_all(self, blob_det, board):
        """检测所有标定图像对，解决 4 重旋转歧义后返回一致的网格中心点。

        findCirclesGrid 对 N×N 正方形对称网格有 4 重旋转歧义
        (0°, 90°CW, 180°, 90°CCW)。两台相机可能各选择不同旋转。

        解决方法：
        1. 检测所有图像对的网格中心
        2. 用少量初始对估计粗略立体参数
        3. 对每对图像测试 cam2 的 4 种旋转，选择与立体参数一致的旋转
        4. 剔除任何旋转都无法拟合的异常对
        """
        obj_pts = _grid_obj_points(board)
        n = board.grid_cols   # assume square grid
        perms = _grid_rotation_perms(n)

        # --- 第一步: 检测所有图像对 ---
        raw_pairs = []   # (idx, centers1, centers2)
        image_size = None

        start, end = self._cal_range
        for idx in range(start, end + 1):
            path1 = self._dir1 / f"{idx:03d}.png"
            path2 = self._dir2 / f"{idx:03d}.png"
            if not path1.exists() or not path2.exists():
                continue

            gray1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
            gray2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)
            if gray1 is None or gray2 is None:
                continue

            if image_size is None:
                image_size = (gray1.shape[1], gray1.shape[0])

            centers1 = _detect_grid(gray1, board, blob_det)
            centers2 = _detect_grid(gray2, board, blob_det)

            # 保存标注图
            if self._save_annotations:
                vis1 = _draw_detection(gray1, centers1, board, f"{idx:03d}")
                vis2 = _draw_detection(gray2, centers2, board, f"{idx:03d}")
                cv2.imwrite(str(self._dir1 / f"{idx:03d}_det.png"), vis1)
                cv2.imwrite(str(self._dir2 / f"{idx:03d}_det.png"), vis2)

            if centers1 is None or centers2 is None:
                continue

            raw_pairs.append((idx,
                              centers1.astype(np.float32),
                              centers2.astype(np.float32)))

        if image_size is None:
            raise RuntimeError("未能读取任何图片")

        log.info("网格检测: %d 对图像中检测到完整网格", len(raw_pairs))

        if len(raw_pairs) < _MIN_VALID_PAIRS:
            raise RuntimeError(
                f"检测到的图像对不足: {len(raw_pairs)} < {_MIN_VALID_PAIRS}")

        # --- 第二步: 用全部对做单目标定获取可靠内参 ---
        # 旋转歧义不影响单目标定的内参: 不同旋转只改变 per-view 外参,
        # K 和 D 作为相机固有属性仍然被正确估计。
        all_img1_raw = [p[1] for p in raw_pairs]
        all_img2_raw = [p[2] for p in raw_pairs]
        _, K1, D1, _, _ = cv2.calibrateCamera(
            [obj_pts] * len(raw_pairs), all_img1_raw, image_size, None, None)
        _, K2, D2, _, _ = cv2.calibrateCamera(
            [obj_pts] * len(raw_pairs), all_img2_raw, image_size, None, None)
        log.info("单目内参: K1 fx=%.1f fy=%.1f, K2 fx=%.1f fy=%.1f",
                 K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1])

        # --- 第三步: 用前几对估计初始立体参数 ---
        # 对初始对尝试 cam2 的 4 种旋转, 选 RMS 最低的
        # 选取间隔均匀的帧，避免连续帧位姿过于相似导致几何退化
        if len(raw_pairs) <= 5:
            init_indices = list(range(len(raw_pairs)))
        else:
            step = len(raw_pairs) / 5
            init_indices = [int(i * step) for i in range(5)]
        init_n = len(init_indices)
        init_img1 = [all_img1_raw[i] for i in init_indices]
        log.info("初始对选取: %d 帧 (indices=%s, frame_ids=%s)",
                 init_n, init_indices,
                 [raw_pairs[i][0] for i in init_indices])

        best_init_rms = float('inf')
        best_init_rot = 0
        for rot in range(4):
            init_img2 = [raw_pairs[i][2][perms[rot]].copy()
                         for i in init_indices]
            try:
                srms, *_ = cv2.stereoCalibrate(
                    [obj_pts] * init_n, init_img1, init_img2,
                    K1.copy(), D1.copy(), K2.copy(), D2.copy(), image_size,
                    flags=cv2.CALIB_FIX_INTRINSIC)
                if srms < best_init_rms:
                    best_init_rms = srms
                    best_init_rot = rot
            except cv2.error:
                continue

        log.info("初始对最佳旋转: %d (RMS=%.4f)", best_init_rot, best_init_rms)

        # 用最佳初始旋转获得参考立体参数
        init_img2_best = [raw_pairs[i][2][perms[best_init_rot]].copy()
                          for i in init_indices]
        _, K1, D1, K2, D2, R_ref, T_ref, _, _ = cv2.stereoCalibrate(
            [obj_pts] * init_n, init_img1, init_img2_best,
            K1.copy(), D1.copy(), K2.copy(), D2.copy(), image_size,
            flags=cv2.CALIB_FIX_INTRINSIC)

        # --- 第四步: 用参考立体参数为每对选择最佳旋转 ---
        obj_pts_list = []
        img1_list = []
        img2_list = []
        valid_indices = []
        rot_counts = [0, 0, 0, 0]

        for idx, c1, c2 in raw_pairs:
            ok, rvec1, tvec1 = cv2.solvePnP(
                obj_pts, c1, K1, D1, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            R1, _ = cv2.Rodrigues(rvec1)
            R2 = R_ref @ R1
            t2 = R_ref @ tvec1 + T_ref
            rvec2, _ = cv2.Rodrigues(R2)
            proj2, _ = cv2.projectPoints(obj_pts, rvec2, t2, K2, D2)
            proj2_flat = proj2.reshape(-1, 2)

            best_err = float('inf')
            best_rot = 0
            for rot in range(4):
                c2_rot = c2[perms[rot]]
                err = np.sqrt(np.mean(
                    (proj2_flat - c2_rot.reshape(-1, 2)) ** 2))
                if err < best_err:
                    best_err = err
                    best_rot = rot

            if best_err > _ROTATION_ERR_THRESHOLD:
                log.debug("剔除 %03d: best_err=%.1f (rot=%d)", idx, best_err, best_rot)
                continue

            obj_pts_list.append(obj_pts)
            img1_list.append(c1)
            img2_list.append(c2[perms[best_rot]].copy())
            valid_indices.append(idx)
            rot_counts[best_rot] += 1

        log.info("旋转校正: 0°=%d  90°CW=%d  180°=%d  90°CCW=%d  剔除=%d",
                 rot_counts[0], rot_counts[1], rot_counts[2], rot_counts[3],
                 len(raw_pairs) - len(valid_indices))

        return obj_pts_list, img1_list, img2_list, valid_indices, image_size

    # ---------- 标定 ----------

    def _stereo_calibrate(self, obj_pts_list, img_pts_1, img_pts_2, image_size):
        """单目标定 + 双目标定。"""
        mono_rms1, K1, D1, _, _ = cv2.calibrateCamera(
            obj_pts_list, img_pts_1, image_size, None, None)
        mono_rms2, K2, D2, _, _ = cv2.calibrateCamera(
            obj_pts_list, img_pts_2, image_size, None, None)

        stereo_rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_pts_list, img_pts_1, img_pts_2,
            K1, D1, K2, D2, image_size,
            flags=cv2.CALIB_FIX_INTRINSIC)

        return K1, D1, K2, D2, R, T, E, F, stereo_rms, mono_rms1, mono_rms2

    @staticmethod
    def _determine_left_right(R_stereo, T_stereo, serial_1, serial_2):
        """根据 T_stereo[0] 符号判断左右。"""
        tx = float(T_stereo.ravel()[0])
        if tx > 0:
            return True, serial_2, serial_1
        else:
            return False, serial_1, serial_2

    # ---------- 地面注册 ----------

    def _register_ground(self, board, K1, D1, K2, D2,
                         R_stereo, T_stereo, F_stereo):
        """
        地面坐标注册：标准 blob 检测 + 联合 PnP 优化。

        流程:
        1. 检测地面图像中的网格
        2. 用对极约束 (Sampson距离) 解决旋转歧义
        3. 过滤重复点和高误差点
        4. 去畸变后三角测量 → Kabsch 初始猜测
        5. 联合 PnP 优化 6-DOF 位姿
        """
        # 查找地面图像
        # 注意: self._dir1/self._dir2 在 run() 中已根据左右判定交换过，
        # 此处直接使用即可 (dir1=左, dir2=右)
        path_left, path_right = self._find_ground_images()

        gray_l = cv2.imread(str(path_left), cv2.IMREAD_GRAYSCALE)
        gray_r = cv2.imread(str(path_right), cv2.IMREAD_GRAYSCALE)
        if gray_l is None or gray_r is None:
            raise RuntimeError(f"无法读取地面参考图像: {path_left}, {path_right}")

        blob_det = _make_blob_detector()
        centers_l = _detect_grid(gray_l, board, blob_det)
        centers_r = _detect_grid(gray_r, board, blob_det)

        # 回退: board-mask 模式
        if centers_l is None:
            log.info("左地面图标准检测失败，使用 board-mask 模式...")
            centers_l = _detect_grid_masked(gray_l, board)
        if centers_r is None:
            log.info("右地面图标准检测失败，使用 board-mask 模式...")
            centers_r = _detect_grid_masked(gray_r, board)

        # 保存地面图标注
        if self._save_annotations:
            vis_l = _draw_detection(gray_l, centers_l, board, "ground-L")
            vis_r = _draw_detection(gray_r, centers_r, board, "ground-R")
            cv2.imwrite(str(path_left.with_name(
                path_left.stem + "_det.png")), vis_l)
            cv2.imwrite(str(path_right.with_name(
                path_right.stem + "_det.png")), vis_r)

        if centers_l is None or centers_r is None:
            raise RuntimeError(
                f"地面图像网格检测失败: "
                f"左={'OK' if centers_l is not None else 'FAIL'}, "
                f"右={'OK' if centers_r is not None else 'FAIL'}")

        n = board.grid_cols
        n_pts = n * n
        perms = _grid_rotation_perms(n)
        pts_l_raw = centers_l.reshape(-1, 2).astype(np.float64)
        pts_r_raw = centers_r.reshape(-1, 2).astype(np.float64)

        # ----- 1. 用对极约束 (Sampson 距离) 解决旋转歧义 -----
        # 对称网格有 16 种左右旋转组合，正确组合有最小对极误差。
        # 比三角测量间距更可靠 (三角测量对错误对应也可能产生看似合理的间距)。
        best_sampson = float('inf')
        best_lrot, best_rrot = 0, 0

        for lrot in range(4):
            pl = pts_l_raw[perms[lrot]]
            for rrot in range(4):
                pr = pts_r_raw[perms[rrot]]
                sampson = _sampson_distance(pl, pr, F_stereo)
                mean_s = np.mean(sampson)
                log.debug("地面旋转 L=%d R=%d: Sampson=%.2fpx", lrot, rrot, mean_s)
                if mean_s < best_sampson:
                    best_sampson = mean_s
                    best_lrot, best_rrot = lrot, rrot

        log.info("地面旋转: 最佳 L=%d R=%d (Sampson=%.2fpx)", best_lrot, best_rrot,
                 best_sampson)

        # 应用旋转排列
        ip_l = pts_l_raw[perms[best_lrot]].copy()
        ip_r = pts_r_raw[perms[best_rrot]].copy()

        # ----- 2. 过滤重复点和高对极误差点 -----
        # findCirclesGrid 对远距离小目标可能返回重复坐标 (blob 不足时复用)
        sampson = _sampson_distance(ip_l, ip_r, F_stereo)
        good_mask = np.ones(n_pts, dtype=bool)

        # 检测重复坐标
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                if np.linalg.norm(ip_l[i] - ip_l[j]) < 1.0:
                    good_mask[i] = False
                    good_mask[j] = False
                if np.linalg.norm(ip_r[i] - ip_r[j]) < 1.0:
                    good_mask[i] = False
                    good_mask[j] = False

        # 按 Sampson 距离过滤
        sampson_thresh = max(15.0, np.median(sampson) * 2)
        good_mask &= (sampson < sampson_thresh)

        good_idx = np.where(good_mask)[0]
        n_good = len(good_idx)
        log.info("地面点过滤: %d/%d 通过 (重复+Sampson>%.0f 被移除)",
                 n_good, n_pts, sampson_thresh)

        if n_good < 12:
            log.warning("过滤后点太少 (%d)，使用全部 %d 点", n_good, n_pts)
            good_idx = np.arange(n_pts)
            n_good = n_pts

        ip_l_good = ip_l[good_idx]
        ip_r_good = ip_r[good_idx]

        # ----- 3. 去畸变后三角测量 -----
        # triangulatePoints 使用线性投影矩阵 (无畸变)，
        # 输入必须是去畸变坐标，否则远离光心的点误差显著
        ip_l_undist = cv2.undistortPoints(
            ip_l_good.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)
        ip_r_undist = cv2.undistortPoints(
            ip_r_good.reshape(-1, 1, 2), K2, D2, P=K2).reshape(-1, 2)

        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K2 @ np.hstack([R_stereo, T_stereo])
        pts4d = cv2.triangulatePoints(P1, P2, ip_l_undist.T, ip_r_undist.T)
        pts3d_cam = (pts4d[:3] / pts4d[3:]).T  # (n_good, 3) 左相机坐标系

        # 检查三角测量质量
        proj_l, _ = cv2.projectPoints(
            pts3d_cam, np.zeros((3, 1)), np.zeros((3, 1)), K1, D1)
        tri_err = np.sqrt(np.mean(
            (proj_l.reshape(-1, 2) - ip_l_good) ** 2))
        log.info("三角测量重投影误差: %.2fpx (%d 点)", tri_err, n_good)

        # ----- 4. 尝试 4 种世界坐标旋转 -----
        # Kabsch 初始猜测 + 联合优化 (使用原始畸变坐标，projectPoints 内部处理畸变)
        best_result = None
        best_err = float('inf')
        best_rot = 0

        for rot in range(4):
            if self._ground_vertical:
                obj_full = _grid_obj_points_vertical(
                    board, rotation=rot, z_offset=self._ground_z_offset)
            else:
                obj_full = _grid_obj_points_world(board, rotation=rot)

            obj_good = obj_full[good_idx]

            # Kabsch 算法: 3D-3D 刚体配准
            mu_w = obj_good.mean(axis=0)
            mu_c = pts3d_cam.mean(axis=0)
            W = (obj_good - mu_w).astype(np.float64)
            C = pts3d_cam - mu_c
            H = W.T @ C
            U, S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R_init = Vt.T @ np.diag([1, 1, d]) @ U.T
            t_init = mu_c - R_init @ mu_w
            rvec_init, _ = cv2.Rodrigues(R_init)

            # 联合 PnP 优化 (使用过滤后的畸变坐标)
            x0 = np.concatenate([rvec_init.ravel(), t_init.ravel()])
            result = least_squares(
                _ground_residuals, x0,
                args=(obj_good, ip_l_good, ip_r_good,
                      K1, D1, K2, D2, R_stereo, T_stereo),
                method='lm')

            opt_rms = np.sqrt(np.mean(result.fun ** 2))

            rvec1 = result.x[:3].reshape(3, 1)
            tvec1 = result.x[3:6].reshape(3, 1)

            R1, _ = cv2.Rodrigues(rvec1)
            pos1 = -R1.T @ tvec1

            R2 = R_stereo @ R1
            t2 = R_stereo @ tvec1 + T_stereo.reshape(3, 1)
            pos2 = -R2.T @ t2

            z1 = float(pos1.ravel()[2])
            z2 = float(pos2.ravel()[2])
            physically_valid = z1 > 0 and z2 > 0

            log.info("  世界旋转 %d: RMS=%.2f, 左=[%.0f,%.0f,%.0f] 右=[%.0f,%.0f,%.0f] %s",
                      rot, opt_rms,
                      pos1.ravel()[0], pos1.ravel()[1], pos1.ravel()[2],
                      pos2.ravel()[0], pos2.ravel()[1], pos2.ravel()[2],
                      "[OK]" if physically_valid else "[x]")

            if physically_valid and opt_rms < best_err:
                best_err = opt_rms
                best_rot = rot
                best_result = (R1, tvec1, pos1, R2, t2, pos2, opt_rms)

        if best_result is None:
            raise RuntimeError("地面注册失败：所有旋转方向均不合理")

        log.info("地面注册: 旋转方向 %d, RMS=%.2f px", best_rot, best_err)

        R1, tvec1, pos1, R2, t2, pos2, reproj_err = best_result
        return R1, tvec1, pos1, R2, t2, pos2, reproj_err

    def _find_ground_images(self):
        """查找地面参考图像路径，支持多种命名格式。

        注意: self._dir1/self._dir2 在 run() 中已根据左右判定交换过，
        self._dir1=左相机目录, self._dir2=右相机目录。
        """
        dir1 = self._dir1  # 左相机
        dir2 = self._dir2  # 右相机

        # 格式 1: ground_XXX.png（排除 _det.png 标注图）
        matches1 = sorted(p for p in dir1.glob("ground_*.png")
                          if not p.stem.endswith("_det"))
        matches2 = sorted(p for p in dir2.glob("ground_*.png")
                          if not p.stem.endswith("_det"))
        if matches1 and matches2:
            log.info("找到 %d 张地面图像，使用第一张: %s",
                     len(matches1), matches1[0].name)
            return matches1[0], matches2[0]

        # 格式 2: {index:03d}.png（旧格式）
        p1 = dir1 / f"{self._ground_idx:03d}.png"
        p2 = dir2 / f"{self._ground_idx:03d}.png"
        if p1.exists() and p2.exists():
            return p1, p2

        raise RuntimeError(
            f"未找到地面参考图像。尝试过:\n"
            f"  {dir1 / 'ground_*.png'}\n"
            f"  {dir1 / f'{self._ground_idx:03d}.png'}")
