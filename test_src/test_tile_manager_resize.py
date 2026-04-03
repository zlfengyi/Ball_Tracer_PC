from __future__ import annotations

import numpy as np

from src.ball_detector import BallDetection
from src.tile_manager import TileManager, TileRect


def test_tile_manager_uses_configured_resize():
    image = np.zeros((1536, 2048, 3), dtype=np.uint8)
    tile_mgr = TileManager(
        {"cam0": (2048, 1536)},
        tile_size=1280,
        resize_to=512,
    )

    crop, tile = tile_mgr.get_tile("cam0", image, current_time=0.0)

    assert crop.shape == (512, 512, 3)
    assert tile == TileRect(x=0, y=0, w=1280, h=1280)


def test_map_detection_to_full_uses_runtime_resize():
    det = BallDetection(
        x=256.0,
        y=256.0,
        confidence=0.9,
        x1=200.0,
        y1=220.0,
        x2=300.0,
        y2=320.0,
    )
    tile = TileRect(x=100, y=200, w=1280, h=1280)

    mapped = TileManager.map_detection_to_full(det, tile, resize_to=512)

    assert mapped.x == 740.0
    assert mapped.y == 840.0
    assert mapped.x1 == 600.0
    assert mapped.y1 == 750.0
    assert mapped.x2 == 850.0
    assert mapped.y2 == 1000.0
