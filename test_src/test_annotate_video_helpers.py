from __future__ import annotations

import math
import random

from test_src.annotate_video import (
    CAR_MARK_COLOR,
    RETURN_COLOR,
    TARGET_COLOR,
    apply_car_result,
    apply_racket_results,
    bot_sample_near,
    build_video_frame_mapping,
    clear_car_results,
    describe_car_loc_status,
    detect_return_events,
    draw_car_target_overlay,
    draw_racket_detections,
    draw_return_vector,
    estimate_rk_bias_from_hits,
    estimate_rk_time_bias,
    extract_fullres_panels,
    guess_tracker_video_path,
    split_stitched_panels,
    target_episodes,
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


_T_HIT = 100.0
_HIT_POS = (0.6, 0.4, 0.5)
_V_IN = (-0.05, -4.5, -1.0)
_V_OUT = (0.8, 5.5, 1.2)
_G = 9.81


def _synth_observations(
    *,
    t_hit: float = _T_HIT,
    with_return: bool = True,
    ground_clutter: bool = False,
    occlusion_gap: tuple[float, float] | None = None,
    noise: float = 0.002,
    seed: int = 7,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    def add(t: float, x: float, y: float, z: float) -> None:
        rows.append({
            "t": round(t, 5),
            "x": round(x + rng.gauss(0.0, noise), 5),
            "y": round(y + rng.gauss(0.0, noise), 5),
            "z": round(z + rng.gauss(0.0, noise), 5),
        })

    t = t_hit - 0.5
    while t < t_hit - 0.03:  # 触球邻域 [−30,+40]ms 遮挡断档
        u = t - t_hit
        add(
            t,
            _HIT_POS[0] + _V_IN[0] * u,
            _HIT_POS[1] + _V_IN[1] * u,
            _HIT_POS[2] + _V_IN[2] * u - 0.5 * _G * u * u,
        )
        t += 0.01
    t = t_hit + 0.04
    while t < t_hit + 0.45:
        u = t - t_hit
        if occlusion_gap is not None and occlusion_gap[0] <= u <= occlusion_gap[1]:
            t += 0.01
            continue
        if with_return:
            add(
                t,
                _HIT_POS[0] + _V_OUT[0] * u,
                _HIT_POS[1] + _V_OUT[1] * u,
                _HIT_POS[2] + _V_OUT[2] * u - 0.5 * _G * u * u,
            )
        else:
            add(
                t,
                _HIT_POS[0] + _V_IN[0] * u,
                _HIT_POS[1] + _V_IN[1] * u,
                max(0.03, _HIT_POS[2] + _V_IN[2] * u - 0.5 * _G * u * u),
            )
        t += 0.01
    if ground_clutter:
        t = t_hit - 0.5
        while t < t_hit + 0.45:
            add(t, 1.5, 1.0, 0.05)
            t += 0.01
    rows.sort(key=lambda r: r["t"])
    return rows


def test_detect_return_events_clean_hit():
    events = detect_return_events(_synth_observations())
    assert len(events) == 1
    ev = events[0]
    yaw_truth = math.degrees(math.atan2(_V_OUT[0], _V_OUT[1]))
    pitch_truth = math.degrees(math.atan2(_V_OUT[2], math.hypot(_V_OUT[0], _V_OUT[1])))
    assert abs(ev["t_hit"] - _T_HIT) < 0.02
    assert abs(ev["yaw_deg"] - yaw_truth) < 2.0
    assert abs(ev["pitch_deg"] - pitch_truth) < 2.5
    assert abs(ev["speed"] - math.hypot(*_V_OUT)) < 0.4
    assert ev["t_end"] == ev["t_hit"] + 0.40
    assert ev["seg_t_end"] > ev["t_hit"] + 0.25


def test_detect_return_events_miss_gives_none():
    assert detect_return_events(_synth_observations(with_return=False)) == []


def test_detect_return_events_ground_clutter_filtered():
    events = detect_return_events(_synth_observations(ground_clutter=True))
    assert len(events) == 1
    yaw_truth = math.degrees(math.atan2(_V_OUT[0], _V_OUT[1]))
    assert abs(events[0]["yaw_deg"] - yaw_truth) < 2.0


def test_detect_return_events_occlusion_gap_after_hit():
    """触球后被拍/臂遮挡断档 200ms：仍应用遮挡后的真弧给出方向。"""
    events = detect_return_events(_synth_observations(occlusion_gap=(0.05, 0.25)))
    assert len(events) == 1
    yaw_truth = math.degrees(math.atan2(_V_OUT[0], _V_OUT[1]))
    assert abs(events[0]["yaw_deg"] - yaw_truth) < 2.5
    assert abs(events[0]["speed"] - math.hypot(*_V_OUT)) < 0.6


def test_detect_return_events_two_hits():
    obs = _synth_observations(seed=1) + _synth_observations(t_hit=_T_HIT + 3.0, seed=2)
    obs.sort(key=lambda r: r["t"])
    events = detect_return_events(obs)
    assert len(events) == 2
    assert abs(events[0]["t_hit"] - _T_HIT) < 0.02
    assert abs(events[1]["t_hit"] - _T_HIT - 3.0) < 0.02


def _synth_shared_poses(n: int, bias: float):
    """RK world 位姿回显行 + PC car_locs 行（PC 时轴 = RK 时轴 + bias）。"""
    rk_rows = []
    pc_rows = []
    for i in range(n):
        x, y, yaw = 1.0 + 0.01 * i, 2.0 - 0.02 * i, 0.1 + 0.001 * i
        rk_t = 10.0 + 0.5 * i
        rk_rows.append((rk_t, x, y, yaw))
        pc_rows.append((rk_t + bias, x, y, yaw))
    return rk_rows, pc_rows


def test_estimate_rk_time_bias_recovers_offset():
    rk_rows, pc_rows = _synth_shared_poses(15, bias=4.2)
    est = estimate_rk_time_bias(rk_rows, pc_rows)
    assert est is not None
    bias, anchors, mad = est
    assert abs(bias - 4.2) < 1e-9
    assert anchors == 15
    assert mad < 1e-9


def test_estimate_rk_time_bias_too_few_anchors():
    rk_rows, pc_rows = _synth_shared_poses(5, bias=4.2)
    assert estimate_rk_time_bias(rk_rows, pc_rows) is None


def test_estimate_rk_time_bias_drops_ambiguous_pc_keys():
    rk_rows, pc_rows = _synth_shared_poses(12, bias=4.2)
    # 同一位姿在 PC 侧出现两次 → 锚歧义，应被丢弃而不是给错 bias
    dup_t, dup_x, dup_y, dup_yaw = pc_rows[0]
    pc_rows.append((dup_t + 30.0, dup_x, dup_y, dup_yaw))
    est = estimate_rk_time_bias(rk_rows, pc_rows)
    assert est is not None
    assert est[1] == 11
    assert abs(est[0] - 4.2) < 1e-9


def test_estimate_rk_bias_from_hits_mode_cluster():
    throw_hts = [23.2, 30.3, 48.9, 105.5, 172.0]
    # 4 次真实回球（bias=-4.2、各带 ±20ms 预测误差）+ 1 个无配对孤立事件
    hits = [26.08, 44.72, 101.31, 167.82, 250.0]
    est = estimate_rk_bias_from_hits(throw_hts, hits)
    assert est is not None
    bias, n, mad = est
    assert abs(bias - (-4.19)) < 0.05
    assert n == 4
    assert mad < 0.05


def test_estimate_rk_bias_from_hits_needs_cluster():
    # 只有两对且互相不成簇 → 不给 bias
    assert estimate_rk_bias_from_hits([10.0, 90.0], [20.0, 50.0]) is None


def _bot_fixture():
    samples = [
        {"t": 10.0, "x": 1.0, "y": 2.0, "tx": None, "ty": None, "active": False, "phase": "IDLE"},
        {"t": 10.1, "x": 1.0, "y": 2.0, "tx": 1.5, "ty": 2.5, "active": True, "phase": "RUN"},
        {"t": 10.2, "x": 1.1, "y": 2.1, "tx": 1.5, "ty": 2.5, "active": True, "phase": "RUN"},
        {"t": 15.0, "x": 1.2, "y": 2.2, "tx": 1.8, "ty": 2.0, "active": True, "phase": "RUN"},
    ]
    return {"samples": samples, "sample_ts": [s["t"] for s in samples], "pose_rows": []}


def test_bot_sample_near_picks_nearest_within_tolerance():
    rk_bot = _bot_fixture()
    assert bot_sample_near(rk_bot, 10.14)["t"] == 10.1
    assert bot_sample_near(rk_bot, 10.19)["t"] == 10.2
    assert bot_sample_near(rk_bot, 12.0) is None


def test_target_episodes_groups_by_gap():
    eps = target_episodes(_bot_fixture(), bias=4.0)
    assert eps == [(14.1, 14.2), (19.0, 19.0)]


def test_draw_car_target_overlay_marks_panel():
    import numpy as np

    cam = {
        "K": np.array([[200.0, 0.0, 512.0], [0.0, 200.0, 512.0], [0.0, 0.0, 1.0]]),
        "D": np.zeros(5),
        "R": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        "t": np.zeros(3),
    }
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    sample = {"t": 10.0, "x": 0.3, "y": 0.8, "tx": 0.6, "ty": 1.0, "active": True, "phase": "RUN"}
    draw_car_target_overlay(canvas, sample, (0.3, 0.8), ["CAM_A"], 512, 512, 1, {"CAM_A": cam})
    red = (canvas[:, :, 2] == TARGET_COLOR[2]) & (canvas[:, :, 0] < 60)
    green = (canvas[:, :, 1] == CAR_MARK_COLOR[1]) & (canvas[:, :, 2] < 60)
    assert red.any()
    assert green.any()


def test_draw_return_vector_marks_panel():
    import numpy as np

    events = detect_return_events(_synth_observations())
    assert len(events) == 1
    # 相机在原点看向世界 +y：cam x=世界x、cam y=−世界z、cam z=世界y（mm）
    cam = {
        "K": np.array([[200.0, 0.0, 512.0], [0.0, 200.0, 512.0], [0.0, 0.0, 1.0]]),
        "D": np.zeros(5),
        "R": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        "t": np.zeros(3),
    }
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    draw_return_vector(
        canvas,
        events[0],
        events[0]["t_hit"] + 0.1,
        None,  # 无 ball3d，走出弧拟合位置
        ["CAM_A"],
        512,
        512,
        1,
        {"CAM_A": cam},
    )
    assert np.any(canvas[:, :, 0] == RETURN_COLOR[0])
    assert np.any((canvas != 0).any(axis=2))
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


def test_draw_racket_detections_only_draws_keypoints_and_center(monkeypatch):
    import cv2
    import numpy as np

    calls = {"rectangle": [], "text": [], "circle": [], "marker": []}
    monkeypatch.setattr(cv2, "rectangle", lambda *args, **kwargs: calls["rectangle"].append((args, kwargs)))
    monkeypatch.setattr(cv2, "putText", lambda *args, **kwargs: calls["text"].append((args, kwargs)))
    monkeypatch.setattr(cv2, "circle", lambda *args, **kwargs: calls["circle"].append((args, kwargs)))
    monkeypatch.setattr(cv2, "drawMarker", lambda *args, **kwargs: calls["marker"].append((args, kwargs)))

    draw_racket_detections(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [{
            "accepted": False,
            "bbox": {"x1": 80, "y1": 100, "x2": 160, "y2": 180, "confidence": 0.9},
            "keypoints": [
                {"id": 0, "x": 100, "y": 120, "score": 50, "valid": True, "used_for_center": True},
                {"id": 4, "x": 140, "y": 160, "score": 30, "valid": False, "used_for_center": False},
            ],
            "x": 120,
            "y": 140,
        }],
        x_offset=10,
        y_offset=20,
        scale=0.5,
    )

    assert calls["rectangle"] == []
    assert calls["text"] == []
    assert len(calls["circle"]) == 2
    assert len(calls["marker"]) == 1


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
        yaw_valid=False,
    )

    apply_car_result(frame_data, car_loc, elapsed_s=3.21)

    assert frame_data["car_loc_status"] == "hit"
    assert frame_data["car_loc_sampled"] is True
    assert frame_data["car_loc"]["t"] == 123.456
    assert frame_data["car_loc"]["elapsed_s"] == 3.21
    assert frame_data["car_loc"]["yaw_valid"] is False
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
