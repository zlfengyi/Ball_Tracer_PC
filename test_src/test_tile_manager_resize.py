from __future__ import annotations

import pytest

from src.ball_detector import BallDetection
from src.tile_manager import TileManager, TileRect


def test_tile_manager_selects_search_tile():
    tile_mgr = TileManager(
        {"cam0": (2048, 1536)},
        tile_size=1280,
        resize_to=512,
    )

    tile = tile_mgr.select_tile("cam0", 2048, 1536, current_time=0.0)

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


@pytest.mark.parametrize(
    ("configured", "image", "expected_side"),
    [(1280, (2048, 1304), 1280), (1280, (2048, 928), 928), (896, (2048, 816), 816)],
)
def test_search_tiles_stay_square_when_the_roi_is_shorter_than_tile_size(
    configured, image, expected_side
):
    """Non-square tiles would squash the ball on the way to the 416x416 input.

    A tile is resized to resize_to x resize_to, so a 1280x928 tile scales the
    ball 1.38x anisotropically and it then trips detection_postprocess's
    max_box_aspect_ratio gate — silently, with no error anywhere. Shrinking the
    sensor ROI to buy frame rate is exactly when that happens, so the tile is
    clamped to the shortest side instead of trusting the config.
    """
    manager = TileManager({"cam": image}, tile_size=configured, resize_to=416)
    tiles = manager._states["cam"].search_tiles
    assert manager._tile_size == expected_side
    assert tiles, "expected at least one search tile"
    for tile in tiles:
        assert tile.w == tile.h == expected_side
