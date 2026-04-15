from __future__ import annotations

from test_src.annotate_video import (
    apply_car_result,
    apply_racket_results,
    build_video_frame_mapping,
    clear_car_results,
    describe_car_loc_status,
    extract_fullres_panels,
    guess_tracker_video_path,
    split_stitched_panels,
)
from src.car_localizer import CarLoc
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


def test_extract_fullres_panels_restores_panel_resolution():
    import numpy as np

    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[0:2, 0:3] = 10
    img[0:2, 3:6] = 20
    img[2:4, 0:3] = 30
    img[2:4, 3:6] = 40

    panels = extract_fullres_panels(img, ["A", "B", "C", "D"])

    assert panels["A"].shape == (4, 6, 3)
    assert panels["B"].shape == (4, 6, 3)
    assert panels["C"].shape == (4, 6, 3)
    assert panels["D"].shape == (4, 6, 3)
    assert int(panels["A"][0, 0, 0]) == 10
    assert int(panels["B"][0, 0, 0]) == 20
    assert int(panels["C"][0, 0, 0]) == 30
    assert int(panels["D"][0, 0, 0]) == 40


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


def test_describe_car_loc_status_marks_skipped_frames():
    text, color = describe_car_loc_status(
        {"car_loc_status": "skipped"},
        sample_every_frames=6,
    )
    assert text == "AprilTag: skipped  sample=1/6"
    assert color == (160, 160, 160)


def test_describe_car_loc_status_marks_sampled_miss():
    text, color = describe_car_loc_status(
        {"car_loc_status": "miss"},
        sample_every_frames=6,
    )
    assert text == "AprilTag: sampled, no tag  sample=1/6"
    assert color == (0, 165, 255)


def test_describe_car_loc_status_marks_dropped_backlog():
    text, color = describe_car_loc_status(
        {"car_loc_status": "dropped"},
        sample_every_frames=2,
    )
    assert text == "AprilTag: dropped backlog  sample=1/2"
    assert color == (0, 96, 255)


def test_apply_car_result_serializes_frame_fields():
    frame_data = {"idx": 12}
    car_loc = CarLoc(
        x=1.23456,
        y=2.34567,
        z=0.45678,
        t=123.456,
        tag_id=7,
        cameras_used=["cam1", "cam2"],
        pixels={"cam1": (10.2, 20.6)},
        reprojection_error=1.234,
        yaw=0.5,
    )

    apply_car_result(frame_data, car_loc, elapsed_s=3.21)

    assert frame_data["car_loc_status"] == "hit"
    assert frame_data["car_loc_sampled"] is True
    assert frame_data["car_loc"]["t"] == 123.456
    assert frame_data["car_loc"]["elapsed_s"] == 3.21
    assert frame_data["car_loc"]["pixels"]["cam1"] == [10, 21]


def test_clear_car_results_removes_old_entries():
    data = {
        "frames": [{
            "car_loc": {"x": 1.0},
            "car_loc_sampled": True,
            "car_loc_status": "hit",
        }],
        "car_locs": [{"x": 1.0}],
        "summary": {
            "car_locs": 1,
            "car_loc_sampled_frames": 1,
            "car_loc_misses": 0,
            "car_loc_dropped_frames": 1,
        },
    }

    clear_car_results(data)

    assert data["frames"][0] == {}
    assert data["car_locs"] == []
    assert data["summary"] == {}
