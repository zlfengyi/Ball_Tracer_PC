from __future__ import annotations

from test_src.annotate_video import (
    apply_racket_results,
    build_video_frame_mapping,
    split_stitched_panels,
)
from src.ball_detector import BallDetection
from src.tile_manager import TileRect


def test_build_video_frame_mapping_fallback():
    data = {"frames": [{}, {}, {}]}
    mapping, exact = build_video_frame_mapping(data, total_video_frames=5)
    assert mapping == [0, 1, 2]
    assert exact is False


def test_split_stitched_panels_keeps_serial_order():
    import numpy as np

    img = np.zeros((10, 12, 3), dtype=np.uint8)
    img[:, 0:4] = 10
    img[:, 4:8] = 20
    img[:, 8:12] = 30

    panels, panel_w = split_stitched_panels(img, ["A", "B", "C"])
    assert panel_w == 4
    assert int(panels["A"][0, 0, 0]) == 10
    assert int(panels["B"][0, 0, 0]) == 20
    assert int(panels["C"][0, 0, 0]) == 30


def test_apply_racket_results_serializes_frame_fields():
    frame_data = {"idx": 0}
    detections = {
        "cam1": [
            BallDetection(
                x=100.4, y=200.6, confidence=0.91,
                x1=80.2, y1=180.3, x2=120.7, y2=220.9,
            )
        ]
    }
    tiles = {"cam1": TileRect(x=64, y=32, w=1280, h=1280)}

    apply_racket_results(frame_data, detections, tiles, racket3d=None)

    assert "racket_detections" in frame_data
    assert frame_data["racket_detections"]["cam1"][0]["x"] == 100
    assert frame_data["racket_tiles"]["cam1"] == {"x": 64, "y": 32, "w": 1280, "h": 1280}
