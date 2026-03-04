# -*- coding: utf-8 -*-
"""
多目相机标定模块 — 棋盘格检测 + 全局 Bundle Adjustment。

一次性优化所有相机的内参、相机间外参、每帧标定板位姿，
最小化全部角点重投影误差（Huber 鲁棒核）。

用法::

    calibrator = MultiCalibrator(
        serials=["DA8199285", "DA8199402", "DA8199243"],
        image_dir=Path("calibration/images"),
        reference_serial="DA8199285",
    )
    result = calibrator.run()
    result.save(Path("src/config/multi_calib.json"))
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
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MIN_VALID_IMAGES = 5


# ================================================================
#  数据类
# ================================================================

@dataclass
class BoardConfig:
    """棋盘格标定板参数。"""
    inner_cols: int = 8          # 内角点列数（方格数-1）
    inner_rows: int = 11         # 内角点行数（方格数-1）
    square_size: float = 50.0    # 方格边长 mm


@dataclass
class CameraCalib:
    """单台相机标定结果。"""
    serial: str
    K: np.ndarray
    D: np.ndarray
    image_size: tuple[int, int]
    R_to_ref: np.ndarray
    t_to_ref: np.ndarray
    R_world: Optional[np.ndarray] = None
    t_world: Optional[np.ndarray] = None
    pos_world: Optional[np.ndarray] = None


@dataclass
class MultiCalibResult:
    """多目标定完整结果。"""
    reference_serial: str
    cameras: dict[str, CameraCalib] = field(default_factory=dict)
    board_config: Optional[BoardConfig] = None
    total_rms: float = 0.0
    per_camera_rms: dict[str, float] = field(default_factory=dict)
    num_images: int = 0
    num_observations: int = 0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _c(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        cameras = {}
        for sn, cam in self.cameras.items():
            d = {
                "K": _c(cam.K),
                "D": _c(cam.D),
                "image_size": list(cam.image_size),
                "R_to_ref": _c(cam.R_to_ref),
                "t_to_ref": _c(cam.t_to_ref),
            }
            if cam.R_world is not None:
                d["R_world"] = _c(cam.R_world)
                d["t_world"] = _c(cam.t_world)
                d["pos_world"] = _c(cam.pos_world)
            cameras[sn] = d

        data = {
            "reference_serial": self.reference_serial,
            "cameras": cameras,
            "diagnostics": {
                "total_rms": self.total_rms,
                "per_camera_rms": self.per_camera_rms,
                "num_images": self.num_images,
                "num_observations": self.num_observations,
            },
            "board": {
                "inner_cols": self.board_config.inner_cols,
                "inner_rows": self.board_config.inner_rows,
                "square_size": self.board_config.square_size,
            } if self.board_config else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log.info("标定结果已保存: %s", path)

    @classmethod
    def load(cls, path: Path) -> MultiCalibResult:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _a(v):
            return np.array(v, dtype=np.float64) if v is not None else None

        board_data = data.get("board")
        board = BoardConfig(**board_data) if board_data else None
        diag = data.get("diagnostics", {})

        cameras = {}
        for sn, cd in data.get("cameras", {}).items():
            cameras[sn] = CameraCalib(
                serial=sn,
                K=_a(cd["K"]).reshape(3, 3),
                D=_a(cd["D"]).ravel(),
                image_size=tuple(cd["image_size"]),
                R_to_ref=_a(cd["R_to_ref"]).reshape(3, 3),
                t_to_ref=_a(cd["t_to_ref"]).ravel(),
                R_world=_a(cd.get("R_world")),
                t_world=_a(cd.get("t_world")),
                pos_world=_a(cd.get("pos_world")),
            )

        return cls(
            reference_serial=data["reference_serial"],
            cameras=cameras,
            board_config=board,
            total_rms=diag.get("total_rms", 0.0),
            per_camera_rms=diag.get("per_camera_rms", {}),
            num_images=diag.get("num_images", 0),
            num_observations=diag.get("num_observations", 0),
        )


# ================================================================
#  棋盘格检测与 3D 坐标
# ================================================================

def _make_obj_points(board: BoardConfig) -> np.ndarray:
    """生成棋盘格内角点的 3D 坐标 (cols*rows, 3) float32。"""
    pts = np.zeros((board.inner_cols * board.inner_rows, 3), dtype=np.float32)
    for i in range(board.inner_rows):
        for j in range(board.inner_cols):
            pts[i * board.inner_cols + j] = [
                j * board.square_size,
                i * board.square_size,
                0.0,
            ]
    return pts


def _detect_checkerboard(gray: np.ndarray, board: BoardConfig
                         ) -> Optional[np.ndarray]:
    """
    检测棋盘格内角点。

    Returns:
        (N, 1, 2) float32 亚像素精度角点，或 None
    """
    pattern = (board.inner_cols, board.inner_rows)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, pattern, None, flags)
    if not ret:
        return None
    # 亚像素精化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


# ================================================================
#  SE(3) 工具
# ================================================================

def _rvec_tvec_to_mat(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).ravel())
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


def _mat_to_rvec_tvec(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.ravel(), T[:3, 3].copy()


def _project(pts_3d, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
    return proj.reshape(-1, 2)


# ================================================================
#  MultiCalibrator
# ================================================================

class MultiCalibrator:
    """
    多目相机全局 BA 标定器（棋盘格）。

    流程：
      1. findChessboardCorners 检测每帧
      2. calibrateCamera 初始化内参
      3. solvePnP 初始化板位姿 + 相机间外参
      4. 全局 BA 联合优化
    """

    def __init__(
        self,
        serials: list[str],
        image_dir: Path,
        reference_serial: str = "",
        board: Optional[BoardConfig] = None,
        image_range: tuple[int, int] = (1, 100),
        save_annotations: bool = False,
        fix_intrinsics: bool = False,
        max_images: int = 0,
    ):
        self._serials = serials
        self._image_dir = Path(image_dir)
        self._ref_serial = reference_serial or serials[0]
        self._board = board or BoardConfig()
        self._image_range = image_range
        self._save_annotations = save_annotations
        self._fix_intrinsics = fix_intrinsics
        self._max_images = max_images

        if self._ref_serial not in self._serials:
            raise ValueError(f"参考相机 {self._ref_serial} 不在列表 {self._serials} 中")

    def run(self) -> MultiCalibResult:
        board = self._board
        obj_pts = _make_obj_points(board)
        n_cams = len(self._serials)
        n_corners = board.inner_cols * board.inner_rows

        # ── 1. 检测 ──
        print(f"\n[1/4] 检测棋盘格 ({board.inner_cols}x{board.inner_rows})...")
        detections, image_sizes = self._detect_all(board)

        total_dets = sum(
            sum(1 for sn in cam_dets) for cam_dets in detections.values())
        print(f"  共 {len(detections)} 帧有检测, {total_dets} 个相机-帧对")

        # ── 2. 初始化内参 ──
        print(f"\n[2/4] 初始化内参 (calibrateCamera)...")
        init_K, init_D = self._init_intrinsics(detections, image_sizes,
                                                board, obj_pts)

        # ── 3. 初始化外参 + 板位姿 ──
        print(f"\n[3/4] 初始化外参 (solvePnP)...")
        (init_cam_poses, init_board_poses,
         valid_images) = self._init_extrinsics(detections, init_K, init_D,
                                               obj_pts)

        if len(valid_images) < _MIN_VALID_IMAGES:
            raise RuntimeError(
                f"有效图像不足: {len(valid_images)} < {_MIN_VALID_IMAGES}")

        print(f"  有效图像: {len(valid_images)} (3台同时检测成功)")
        for sn in self._serials:
            if sn != self._ref_serial and sn in init_cam_poses:
                rv, tv = _mat_to_rvec_tvec(init_cam_poses[sn])
                print(f"  {sn} → ref: t=[{tv[0]:.1f}, {tv[1]:.1f}, {tv[2]:.1f}]mm")

        # ── 4. 全局 BA ──
        if self._fix_intrinsics:
            print(f"\n[4/4] Bundle Adjustment (内参固定，仅优化外参+板位姿)...")
        else:
            print(f"\n[4/4] Bundle Adjustment (内参+外参+板位姿)...")
        result = self._bundle_adjust(
            detections, valid_images, image_sizes,
            init_K, init_D, init_cam_poses, init_board_poses,
            board, obj_pts)

        return result

    # ────────────────────────────────────────────────────────────
    #  检测
    # ────────────────────────────────────────────────────────────

    def _detect_all(self, board):
        """检测所有图片中的棋盘格。优先从 corner_detections.json 缓存读取。"""
        import time as _time
        detections = {}   # {img_idx: {serial: corners(N,1,2)}}
        image_sizes = {}

        start, end = self._image_range

        # 尝试从缓存读取
        cache_path = self._image_dir / "corner_detections.json"
        if cache_path.exists():
            print(f"  从缓存读取: {cache_path}")
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)

            cam_cache = cache.get("cameras", {})
            for sn in self._serials:
                sn_data = cam_cache.get(sn, {})
                for idx_str, det_data in sn_data.items():
                    idx = int(idx_str)
                    if idx < start or idx > end:
                        continue
                    corners = np.array(det_data["corners"],
                                       dtype=np.float32).reshape(-1, 1, 2)
                    if sn not in image_sizes:
                        image_sizes[sn] = tuple(det_data["image_size"])
                    if idx not in detections:
                        detections[idx] = {}
                    detections[idx][sn] = corners

            total = end - start + 1
            print(f"  缓存加载完成")
            for sn in self._serials:
                n = sum(1 for d in detections.values() if sn in d)
                print(f"  {sn}: {n}/{total} 帧检测成功")

            return detections, image_sizes

        # 无缓存，实时检测
        print(f"  未找到缓存，实时检测（建议先运行 detect_corners.py）")
        pattern = (board.inner_cols, board.inner_rows)
        total = end - start + 1
        t0 = _time.monotonic()
        processed = 0

        for idx in range(start, end + 1):
            cam_dets = {}
            for sn in self._serials:
                path = self._image_dir / sn / f"{idx:03d}.png"
                if not path.exists():
                    continue
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                if sn not in image_sizes:
                    image_sizes[sn] = (img.shape[1], img.shape[0])

                corners = _detect_checkerboard(img, board)
                if corners is not None:
                    cam_dets[sn] = corners

            if cam_dets:
                detections[idx] = cam_dets

            processed += 1
            if processed % 20 == 0 or processed == total:
                elapsed = _time.monotonic() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / speed if speed > 0 else 0
                n_det = len(cam_dets)
                print(f"  [{processed}/{total}] "
                      f"{elapsed:.0f}s elapsed, "
                      f"{speed:.1f} img/s, "
                      f"ETA {eta:.0f}s, "
                      f"本帧检测到 {n_det}/{len(self._serials)} 相机")

        total_elapsed = _time.monotonic() - t0
        print(f"\n  检测完成，耗时 {total_elapsed:.1f}s")
        for sn in self._serials:
            n = sum(1 for d in detections.values() if sn in d)
            print(f"  {sn}: {n}/{total} 帧检测成功")

        return detections, image_sizes

    # ────────────────────────────────────────────────────────────
    #  初始化内参
    # ────────────────────────────────────────────────────────────

    def _init_intrinsics(self, detections, image_sizes, board, obj_pts):
        init_K = {}
        init_D = {}

        # 每台相机分别标定并过滤坏帧
        bad_frames = set()  # 在任何相机中表现差的帧

        # calibrateCamera 限制帧数（>80 帧即可得到稳定内参）
        max_calib = max(self._max_images, 80) if self._max_images > 0 else 0

        for sn in self._serials:
            frame_indices = []
            obj_list = []
            img_list = []
            for idx, cam_dets in sorted(detections.items()):
                if sn not in cam_dets:
                    continue
                frame_indices.append(idx)
                obj_list.append(obj_pts)
                img_list.append(cam_dets[sn])

            # 均匀抽样加速 calibrateCamera
            if max_calib > 0 and len(obj_list) > max_calib:
                total = len(obj_list)
                step = total / max_calib
                sel = [int(i * step) for i in range(max_calib)]
                frame_indices = [frame_indices[i] for i in sel]
                obj_list = [obj_list[i] for i in sel]
                img_list = [img_list[i] for i in sel]
                log.info("  %s: calibrateCamera 抽样 %d/%d 帧", sn, max_calib, total)

            if len(obj_list) < _MIN_VALID_IMAGES:
                raise RuntimeError(
                    f"相机 {sn} 有效图像不足: {len(obj_list)} < {_MIN_VALID_IMAGES}")

            size = image_sizes[sn]
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                obj_list, img_list, size, None, None)

            # 计算每帧重投影误差
            per_frame_rms = []
            for i in range(len(obj_list)):
                proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i],
                                            K, D)
                err = np.sqrt(np.mean((proj.reshape(-1, 2) -
                                       img_list[i].reshape(-1, 2)) ** 2))
                per_frame_rms.append(err)

            per_frame_rms = np.array(per_frame_rms)
            median_err = np.median(per_frame_rms)
            threshold = max(median_err * 3.0, 1.0)  # 3倍中位数，但不低于1.0px

            n_bad = 0
            for i, err in enumerate(per_frame_rms):
                if err > threshold:
                    bad_frames.add(frame_indices[i])
                    n_bad += 1

            print(f"  {sn}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                  f"rms={rms:.3f} ({len(obj_list)} 图, size={size}) "
                  f"median_err={median_err:.3f} bad={n_bad}")

            init_K[sn] = K
            init_D[sn] = D.ravel()[:5]

        # 过滤坏帧
        if bad_frames:
            print(f"\n  过滤 {len(bad_frames)} 坏帧, 重新标定...")
            for idx in bad_frames:
                if idx in detections:
                    del detections[idx]

            # 重新标定
            for sn in self._serials:
                obj_list = []
                img_list = []
                for idx, cam_dets in sorted(detections.items()):
                    if sn not in cam_dets:
                        continue
                    obj_list.append(obj_pts)
                    img_list.append(cam_dets[sn])

                if len(obj_list) < _MIN_VALID_IMAGES:
                    raise RuntimeError(
                        f"过滤后相机 {sn} 有效图像不足: {len(obj_list)}")

                # 均匀抽样加速
                if max_calib > 0 and len(obj_list) > max_calib:
                    total = len(obj_list)
                    step = total / max_calib
                    sel = [int(i * step) for i in range(max_calib)]
                    obj_list = [obj_list[i] for i in sel]
                    img_list = [img_list[i] for i in sel]

                size = image_sizes[sn]
                rms, K, D, _, _ = cv2.calibrateCamera(
                    obj_list, img_list, size, None, None)
                init_K[sn] = K
                init_D[sn] = D.ravel()[:5]
                print(f"  {sn}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                      f"rms={rms:.3f} ({len(obj_list)} 图)")

        return init_K, init_D

    # ────────────────────────────────────────────────────────────
    #  初始化外参
    # ────────────────────────────────────────────────────────────

    def _init_extrinsics(self, detections, init_K, init_D, obj_pts):
        ref = self._ref_serial

        # 每台相机每帧 solvePnP
        cam_frame_poses = {sn: {} for sn in self._serials}

        for idx, cam_dets in detections.items():
            for sn in self._serials:
                if sn not in cam_dets:
                    continue
                corners = cam_dets[sn]
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners,
                    init_K[sn], init_D[sn],
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    cam_frame_poses[sn][idx] = _rvec_tvec_to_mat(rvec, tvec)

        # 板位姿 = 参考相机坐标系下
        board_poses = {}
        for idx, T in cam_frame_poses[ref].items():
            board_poses[idx] = T

        # 有效图像：所有相机都有检测的帧
        valid_images = sorted(
            idx for idx in board_poses
            if all(idx in cam_frame_poses[sn] for sn in self._serials))

        # 均匀抽样限制图片数量（加速 BA）
        if self._max_images > 0 and len(valid_images) > self._max_images:
            n = self._max_images
            step = len(valid_images) / n
            valid_images = [valid_images[int(i * step)] for i in range(n)]
            log.info("抽样 %d/%d 帧用于 BA", n, len(valid_images))

        # 计算相机间外参
        cam_poses = {ref: np.eye(4)}

        for sn in self._serials:
            if sn == ref:
                continue

            R_estimates = []
            t_estimates = []
            for idx in valid_images:
                if idx not in cam_frame_poses[sn]:
                    continue
                T_cam_board = cam_frame_poses[sn][idx]
                T_ref_board = board_poses[idx]
                T_cam_ref = T_cam_board @ np.linalg.inv(T_ref_board)
                R_estimates.append(T_cam_ref[:3, :3])
                t_estimates.append(T_cam_ref[:3, 3])

            if not R_estimates:
                raise RuntimeError(f"相机 {sn} 无法计算到参考相机的外参")

            # 旋转取四元数平均
            quats = np.array([Rotation.from_matrix(R).as_quat()
                              for R in R_estimates])
            for i in range(1, len(quats)):
                if np.dot(quats[i], quats[0]) < 0:
                    quats[i] = -quats[i]
            avg_quat = np.mean(quats, axis=0)
            avg_quat /= np.linalg.norm(avg_quat)
            R_avg = Rotation.from_quat(avg_quat).as_matrix()
            t_avg = np.median(np.array(t_estimates), axis=0)

            T = np.eye(4)
            T[:3, :3] = R_avg
            T[:3, 3] = t_avg
            cam_poses[sn] = T

            log.info("%s → ref: %d 帧, t=[%.1f, %.1f, %.1f]",
                     sn, len(R_estimates), t_avg[0], t_avg[1], t_avg[2])

        return cam_poses, board_poses, valid_images

    # ────────────────────────────────────────────────────────────
    #  Bundle Adjustment
    # ────────────────────────────────────────────────────────────

    def _bundle_adjust(self, detections, valid_images, image_sizes,
                       init_K, init_D, init_cam_poses, init_board_poses,
                       board, obj_pts):
        """
        全局 BA。

        fix_intrinsics=False 时参数向量:
          [cam_intrinsics(n_cams * 9) |
           cam_extrinsics((n_cams-1) * 6) |
           board_poses(n_boards * 6)]

        fix_intrinsics=True 时参数向量:
          [cam_extrinsics((n_cams-1) * 6) |
           board_poses(n_boards * 6)]
        """
        fix_intr = self._fix_intrinsics
        n_cams = len(self._serials)
        non_ref_serials = [s for s in self._serials if s != self._ref_serial]
        board_indices = sorted(valid_images)
        n_boards = len(board_indices)
        board_idx_map = {idx: i for i, idx in enumerate(board_indices)}
        n_corners = board.inner_cols * board.inner_rows

        # ── 固定内参时，预计算 K/D 数组 ──
        if fix_intr:
            fixed_K = {}
            fixed_D = {}
            for sn in self._serials:
                fixed_K[sn] = init_K[sn].copy()
                fixed_D[sn] = init_D[sn].copy()

        # ── 构建参数向量 ──
        params = []
        if not fix_intr:
            # 内参
            for sn in self._serials:
                K = init_K[sn]
                D = init_D[sn]
                params.extend([K[0,0], K[1,1], K[0,2], K[1,2],
                               D[0], D[1], D[2], D[3], D[4]])
        # 外参（非参考相机）
        for sn in non_ref_serials:
            rv, tv = _mat_to_rvec_tvec(init_cam_poses[sn])
            params.extend(rv.tolist())
            params.extend(tv.tolist())
        # 板位姿
        for idx in board_indices:
            rv, tv = _mat_to_rvec_tvec(init_board_poses[idx])
            params.extend(rv.tolist())
            params.extend(tv.tolist())

        x0 = np.array(params, dtype=np.float64)

        # ── 收集观测 ──
        # (cam_idx, board_list_idx, corners(N,2))
        observations = []
        for idx in board_indices:
            cam_dets = detections[idx]
            bi = board_idx_map[idx]
            for ci, sn in enumerate(self._serials):
                if sn not in cam_dets:
                    continue
                corners = cam_dets[sn].reshape(-1, 2)
                observations.append((ci, bi, corners))

        total_obs = len(observations) * n_corners
        print(f"  观测: {len(observations)} 组, {total_obs} 角点")

        # ── 偏移量 ──
        if fix_intr:
            ext_off = 0
        else:
            ext_off = n_cams * 9
        board_off = ext_off + len(non_ref_serials) * 6
        non_ref_map = {sn: i for i, sn in enumerate(non_ref_serials)}

        def _unpack_K_D(x, ci):
            if fix_intr:
                sn = self._serials[ci]
                return fixed_K[sn], fixed_D[sn]
            o = ci * 9
            fx, fy, cx, cy = x[o:o+4]
            k1, k2, p1, p2, k5 = x[o+4:o+9]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            D = np.array([k1, k2, p1, p2, k5])
            return K, D

        def _unpack_ext(x, ci):
            sn = self._serials[ci]
            if sn == self._ref_serial:
                return np.zeros(3), np.zeros(3)
            ei = non_ref_map[sn]
            o = ext_off + ei * 6
            return x[o:o+3], x[o+3:o+6]

        def _unpack_board(x, bi):
            o = board_off + bi * 6
            return x[o:o+3], x[o+3:o+6]

        def residuals(x):
            res = []
            for ci, bi, obs_2d in observations:
                K, D = _unpack_K_D(x, ci)
                cam_rv, cam_tv = _unpack_ext(x, ci)
                b_rv, b_tv = _unpack_board(x, bi)

                if self._serials[ci] == self._ref_serial:
                    rvec, tvec = b_rv, b_tv
                else:
                    R_cr, _ = cv2.Rodrigues(cam_rv.astype(np.float64))
                    R_b, _ = cv2.Rodrigues(b_rv.astype(np.float64))
                    R_tot = R_cr @ R_b
                    t_tot = R_cr @ b_tv + cam_tv
                    rvec_arr, _ = cv2.Rodrigues(R_tot)
                    rvec = rvec_arr.ravel()
                    tvec = t_tot

                proj = _project(obj_pts, rvec, tvec, K, D)
                res.append((proj - obs_2d).ravel())
            return np.concatenate(res)

        # ── 稀疏 Jacobian ──
        n_res = total_obs * 2
        n_params = len(x0)

        from scipy.sparse import lil_matrix
        J_sp = lil_matrix((n_res, n_params), dtype=int)

        row = 0
        for ci, bi, obs_2d in observations:
            n_r = n_corners * 2
            # 内参（仅非固定时）
            if not fix_intr:
                co = ci * 9
                J_sp[row:row+n_r, co:co+9] = 1
            # 外参
            sn = self._serials[ci]
            if sn != self._ref_serial:
                eo = ext_off + non_ref_map[sn] * 6
                J_sp[row:row+n_r, eo:eo+6] = 1
            # 板位姿
            bo = board_off + bi * 6
            J_sp[row:row+n_r, bo:bo+6] = 1
            row += n_r

        if fix_intr:
            print(f"  参数: {n_params} (内参固定, "
                  f"{len(non_ref_serials)}x6 外参 + {n_boards}x6 板位姿)")
        else:
            print(f"  参数: {n_params} ({n_cams}x9 内参 + "
                  f"{len(non_ref_serials)}x6 外参 + {n_boards}x6 板位姿)")
        print(f"  残差: {n_res}")
        print(f"  优化中...")

        result = least_squares(
            residuals, x0,
            jac_sparsity=J_sp,
            verbose=2,
            x_scale='jac',
            loss='huber',
            f_scale=1.0,
            method='trf',
            max_nfev=5000,
        )

        x_opt = result.x
        total_rms = np.sqrt(np.mean(result.fun ** 2))
        print(f"\n  BA 收敛: cost={result.cost:.2f}, total_rms={total_rms:.3f}px")

        # ── 提取结果 ──
        cameras = {}
        per_camera_rms = {}

        for ci, sn in enumerate(self._serials):
            K, D = _unpack_K_D(x_opt, ci)
            cam_rv, cam_tv = _unpack_ext(x_opt, ci)

            if sn == self._ref_serial:
                R_to_ref = np.eye(3)
                t_to_ref = np.zeros(3)
            else:
                R_to_ref, _ = cv2.Rodrigues(cam_rv.astype(np.float64))
                t_to_ref = cam_tv.copy()

            cameras[sn] = CameraCalib(
                serial=sn, K=K.copy(), D=D.copy(),
                image_size=image_sizes[sn],
                R_to_ref=R_to_ref,
                t_to_ref=t_to_ref.reshape(3, 1),
            )

            # 每台相机 RMS
            cam_res = []
            for obs_ci, bi, obs_2d in observations:
                if obs_ci != ci:
                    continue
                b_rv, b_tv = _unpack_board(x_opt, bi)
                if sn == self._ref_serial:
                    rvec, tvec = b_rv, b_tv
                else:
                    R_cr, _ = cv2.Rodrigues(cam_rv.astype(np.float64))
                    R_b, _ = cv2.Rodrigues(b_rv.astype(np.float64))
                    R_tot = R_cr @ R_b
                    t_tot = R_cr @ b_tv + cam_tv
                    rvec_arr, _ = cv2.Rodrigues(R_tot)
                    rvec = rvec_arr.ravel()
                    tvec = t_tot
                proj = _project(obj_pts, rvec, tvec, K, D)
                cam_res.append(((proj - obs_2d) ** 2).sum(axis=1))

            if cam_res:
                rms = np.sqrt(np.mean(np.concatenate(cam_res)))
                per_camera_rms[sn] = float(rms)
                print(f"  {sn}: rms={rms:.3f}px")

        return MultiCalibResult(
            reference_serial=self._ref_serial,
            cameras=cameras,
            board_config=board,
            total_rms=float(total_rms),
            per_camera_rms=per_camera_rms,
            num_images=len(valid_images),
            num_observations=total_obs,
        )
