from __future__ import annotations

from test_src.annotate_video import (
    apply_racket_results,
    build_video_frame_mapping,
    guess_tracker_video_path,
    split_stitched_panels,
)
from src.racket_localizer import RacketDetection


def test_build_video_frame_mapping_fallback():
    data = {"frames": [{}, {}, {}]}
    mapping, exact = build_video_frame_mapping(data, total_video_frames=5)
    assert mapping == [0, 1, 2]
    assert exact is False


def test_split_stitched_panels_keeps_serial_order():
    import numpy as np

    img = np.zeros((10, 12, 3), dtype=np.uint8)
    img[0:5, 0:6] = 10
    img[0:5, 6:12] = 20
    img[5:10, 0:6] = 30

    panels, panel_w, panel_h = split_stitched_panels(img, ["A", "B", "C"])
    assert panel_w == 6
    assert panel_h == 5
    assert int(panels["A"][0, 0, 0]) == 10
    assert int(panels["B"][0, 0, 0]) == 20
    assert int(panels["C"][0, 0, 0]) == 30


def test_apply_racket_results_serializes_frame_fields():
    frame_data = {"idx": 0}
    detections = {
        "cam1": RacketDetection(
            serial="cam1",
            detected=True,
            accepted=True,
            failure_reason="",
            bbox_confidence=0.91,
            bbox_xyxy=(80.2, 180.3, 120.7, 220.9),
            center_xy=(100.4, 200.6),
            face_keypoint_score_min=55.0,
            face_valid_keypoint_count=4,
        )
    }

    apply_racket_results(
        frame_data,
        detections,
        racket3d=None,
        keypoint_score_threshold=40.0,
    )

    assert "racket_detections" in frame_data
    assert frame_data["racket_detections"]["cam1"][0]["center_xy"] == [100.4, 200.6]
    assert frame_data["racket_detections"]["cam1"][0]["bbox"] == {
        "x1": 80.2,
        "y1": 180.3,
        "x2": 120.7,
        "y2": 220.9,
        "confidence": 0.91,
    }


def test_guess_tracker_video_path_prefers_json_artifact_path(tmp_path):
    json_path = tmp_path / "tracker_20260401_123000.json"
    json_path.write_text("{}", encoding="utf-8")
    video_path = tmp_path / "tracker_20260401_123000.mp4"
    video_path.write_bytes(b"")

    data = {
        "config": {
            "video_output": {
                "artifact_path": str(video_path),
            }
        }
    }

    assert guess_tracker_video_path(json_path, data) == video_path
