# -*- coding: utf-8 -*-
"""
1280x1280 分片管理器 — 每台相机独立的 YOLO 检测区域管理。

裁剪 1280x1280 切片后压缩到 640x640 送入 YOLO，检测坐标乘 2 还原到切片坐标
再加上切片偏移得到全图坐标。

两种工作模式：
  - 跟踪模式（上次检测 <0.3s）：以上次检测到的网球位置为中心裁剪 1280x1280
  - 搜索模式（上次检测 >0.3s）：按预定义切片区域逐帧轮询

用法::

    tile_mgr = TileManager({"DA8199285": (2248, 1348)})

    # 每帧
    crop_640, tile = tile_mgr.get_tile("DA8199285", image, current_time)
    dets = detector.detect(crop_640)  # 在 640x640 上检测
    dets_full = [TileManager.map_detection_to_full(d, tile) for d in dets]

    # 成功 3D 定位后更新（全图坐标）
    tile_mgr.update_tracking("DA8199285", det.x, det.y, current_time)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ball_detector import BallDetection


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
    last_x: float = 0.0
    last_y: float = 0.0
    last_time: float = -1.0  # 负数表示从未检测到


class TileManager:
    """每台相机独立的 1280x1280 分片管理器（压缩 640x640 送 YOLO）。"""

    RESIZE_TO = 640  # 切片压缩到此尺寸后送 YOLO

    def __init__(
        self,
        camera_sizes: dict[str, tuple[int, int]],
        tile_size: int = 1280,
        track_timeout: float = 0.3,
    ):
        self._tile_size = tile_size
        self._track_timeout = track_timeout
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

        if (state.last_time >= 0 and
                current_time - state.last_time < self._track_timeout):
            tile = self._center_tile(
                int(state.last_x), int(state.last_y), w, h
            )
        else:
            idx = state.search_idx % len(state.search_tiles)
            tile = state.search_tiles[idx]
            state.search_idx += 1

        crop = image[tile.y:tile.y + tile.h, tile.x:tile.x + tile.w]
        resized = cv2.resize(crop, (self.RESIZE_TO, self.RESIZE_TO))
        return resized, tile

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

    def update_tracking(
        self, sn: str, det_x: float, det_y: float, time: float
    ) -> None:
        """成功检测到球且有 3D 位置时调用，更新跟踪状态（全图坐标）。"""
        state = self._states[sn]
        state.last_x = det_x
        state.last_y = det_y
        state.last_time = time

    @staticmethod
    def map_detection_to_full(
        det: BallDetection, tile: TileRect
    ) -> BallDetection:
        """将 640x640 上的检测坐标映射回全图坐标。

        映射：先按 tile 实际尺寸 / 640 缩放，再加 tile 偏移。
        """
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
