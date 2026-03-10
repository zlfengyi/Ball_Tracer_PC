# -*- coding: utf-8 -*-
"""
1280x1280 分片管理器 — 每台相机独立的 YOLO 检测区域管理。

裁剪 1280x1280 切片后压缩到 640x640 送入 YOLO，检测坐标乘 2 还原到切片坐标
再加上切片偏移得到全图坐标。

每台相机独立的三种状态：
  - 搜索模式：逐帧轮询预定义切片区域
  - 持续模式：某切片 YOLO 检测到球但 3D 定位失败，持续该切片 N 帧再继续搜索
  - 跟踪模式：3D 定位成功，以球位置为中心跟踪（最多持续 track_timeout_s）

状态转换：
  搜索 → YOLO检测到球 → 持续（最多 search_hold_frames 帧）
  持续 → 3D定位成功   → 跟踪
  持续 → 持续帧数用完  → 搜索（继续下一个切片）
  跟踪 → 超时          → 搜索
  跟踪 → 3D定位成功    → 跟踪（刷新位置和时间）

用法::

    tile_mgr = TileManager({"DA8199285": (2448, 2048)})

    # 每帧
    crop_640, tile = tile_mgr.get_tile("DA8199285", image, current_time)
    dets = detector.detect(crop_640)
    dets_full = [TileManager.map_detection_to_full(d, tile) for d in dets]

    # YOLO 检测到球但 3D 定位失败
    tile_mgr.on_2d_detected("DA8199285", tile)

    # 3D 定位成功
    tile_mgr.on_3d_located("DA8199285", det.x, det.y, current_time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .ball_detector import BallDetection


class _TileMode(Enum):
    SEARCH = "search"
    HOLD = "hold"
    TRACK = "track"


@dataclass
class TileRect:
    """切片区域（全图坐标）。"""
    x: int
    y: int
    w: int = 1280
    h: int = 1280


@dataclass
class _CameraState:
    """单台相机的分片状态。"""
    search_tiles: list[TileRect] = field(default_factory=list)
    search_idx: int = 0
    mode: _TileMode = _TileMode.SEARCH
    # 持续模式
    hold_remaining: int = 0
    hold_tile: TileRect | None = None
    # 跟踪模式
    track_x: float = 0.0
    track_y: float = 0.0
    track_time: float = -1.0


class TileManager:
    """每台相机独立的 1280x1280 分片管理器（压缩 640x640 送 YOLO）。"""

    RESIZE_TO = 640

    def __init__(
        self,
        camera_sizes: dict[str, tuple[int, int]],
        tile_size: int = 1280,
        track_timeout: float = 0.3,
        search_hold_frames: int = 4,
    ):
        self._tile_size = tile_size
        self._track_timeout = track_timeout
        self._search_hold_frames = search_hold_frames
        self._states: dict[str, _CameraState] = {}

        for sn, (w, h) in camera_sizes.items():
            tiles = self._compute_search_tiles(w, h, tile_size)
            self._states[sn] = _CameraState(search_tiles=tiles)

    @staticmethod
    def _compute_search_tiles(
        img_w: int, img_h: int, tile_size: int
    ) -> list[TileRect]:
        """为给定分辨率计算均匀分布的搜索切片。"""
        def _positions(length: int, size: int) -> list[int]:
            if length <= size:
                return [0]
            n = max(2, round(length / size))
            step = (length - size) / (n - 1) if n > 1 else 0
            return [round(step * i) for i in range(n)]

        xs = _positions(img_w, tile_size)
        ys = _positions(img_h, tile_size)

        tiles = []
        for y in ys:
            for x in xs:
                tiles.append(TileRect(
                    x=x, y=y,
                    w=min(tile_size, img_w - x),
                    h=min(tile_size, img_h - y),
                ))
        return tiles

    def get_tile(
        self, sn: str, image: np.ndarray, current_time: float
    ) -> tuple[np.ndarray, TileRect]:
        """
        返回当前帧该相机应使用的切片（已压缩到 640x640）。

        Returns:
            (压缩后的 640x640 图像, 切片区域（全图坐标）)
        """
        state = self._states[sn]
        h, w = image.shape[:2]

        if state.mode == _TileMode.TRACK:
            if current_time - state.track_time >= self._track_timeout:
                state.mode = _TileMode.SEARCH
            else:
                tile = self._center_tile(
                    int(state.track_x), int(state.track_y), w, h
                )
                return self._crop_and_resize(image, tile), tile

        if state.mode == _TileMode.HOLD:
            if state.hold_remaining > 0:
                state.hold_remaining -= 1
                tile = state.hold_tile
                return self._crop_and_resize(image, tile), tile
            else:
                state.mode = _TileMode.SEARCH

        # 搜索模式
        idx = state.search_idx % len(state.search_tiles)
        tile = state.search_tiles[idx]
        state.search_idx += 1
        return self._crop_and_resize(image, tile), tile

    def on_2d_detected(self, sn: str, tile: TileRect) -> None:
        """YOLO 检测到球但 3D 定位失败时调用 -- 进入持续模式。"""
        state = self._states[sn]
        if state.mode == _TileMode.SEARCH:
            state.mode = _TileMode.HOLD
            state.hold_tile = tile
            state.hold_remaining = self._search_hold_frames - 1

    def on_3d_located(
        self, sn: str, det_x: float, det_y: float, time: float
    ) -> None:
        """3D 定位成功时调用 -- 进入/刷新跟踪模式（全图坐标）。"""
        state = self._states[sn]
        state.mode = _TileMode.TRACK
        state.track_x = det_x
        state.track_y = det_y
        state.track_time = time
        state.hold_remaining = 0

    def _center_tile(
        self, cx: int, cy: int, img_w: int, img_h: int
    ) -> TileRect:
        """以 (cx, cy) 为中心生成切片，clamp 到图片边界。"""
        ts = self._tile_size
        x = max(0, min(cx - ts // 2, img_w - ts))
        y = max(0, min(cy - ts // 2, img_h - ts))
        return TileRect(
            x=x, y=y,
            w=min(ts, img_w - x),
            h=min(ts, img_h - y),
        )

    def _crop_and_resize(
        self, image: np.ndarray, tile: TileRect
    ) -> np.ndarray:
        crop = image[tile.y:tile.y + tile.h, tile.x:tile.x + tile.w]
        return cv2.resize(crop, (self.RESIZE_TO, self.RESIZE_TO))

    @staticmethod
    def map_detection_to_full(
        det: BallDetection, tile: TileRect
    ) -> BallDetection:
        """将 640x640 上的检测坐标映射回全图坐标。"""
        sx = tile.w / TileManager.RESIZE_TO
        sy = tile.h / TileManager.RESIZE_TO
        return BallDetection(
            x=det.x * sx + tile.x,
            y=det.y * sy + tile.y,
            confidence=det.confidence,
            x1=det.x1 * sx + tile.x,
            y1=det.y1 * sy + tile.y,
            x2=det.x2 * sx + tile.x,
            y2=det.y2 * sy + tile.y,
        )

    def get_search_tile_count(self, sn: str) -> int:
        """返回该相机的搜索切片总数。"""
        return len(self._states[sn].search_tiles)

    @property
    def serials(self) -> list[str]:
        return list(self._states.keys())
