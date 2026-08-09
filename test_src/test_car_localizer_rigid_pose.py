from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from src.car_localizer import (
    CarDetection,
    CarLocalizer,
    tag_rotation_from_angles,
)


TAG_EDGE_M = 0.161
# 近似实车布局：id0 右后低位倒贴、id1 左前高位正贴，均朝车尾（-y_car）
LAYOUT = {
    0: {
        "center": np.array([0.2835, -0.42, 0.1915]),
        "az_deg": -90.0,
        "rho_deg": 180.0,
    },
    1: {
        "center": np.array([-0.2445, 0.274, 0.4026]),
        "az_deg": -90.0,
        "rho_deg": 0.0,
    },
}
K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])


def _look_at(camera_m: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    forward = target_m - camera_m
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def _write_calibration(path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    camera_positions_m = {
        "cam0": np.array([-3.0, -2.0, 3.0]),
        "cam1": np.array([3.0, -2.0, 3.0]),
        "cam2": np.array([-3.0, 6.0, 3.0]),
        "cam3": np.array([3.0, 6.0, 3.0]),
    }
    target_m = np.array([0.0, 2.0, 0.3])
    rotations = {
        serial: _look_at(position, target_m)
        for serial, position in camera_positions_m.items()
    }
    cameras = {}
    for serial, position_m in camera_positions_m.items():
        R = rotations[serial]
        t = -R.dot(position_m * 1000.0)
        cameras[serial] = {
            "K": K.tolist(),
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R_world": R.tolist(),
            "t_world": t.reshape(3, 1).tolist(),
        }
    path.write_text(
        json.dumps({"reference_serial": "cam0", "cameras": cameras}),
        encoding="utf-8",
    )
    return rotations, camera_positions_m


def _write_vehicle(path) -> None:
    # id0 走 R_tag_car 矩阵分支，id1 走 face_azimuth/inplane 角度分支
    r0 = tag_rotation_from_angles(
        math.radians(LAYOUT[0]["az_deg"]), math.radians(LAYOUT[0]["rho_deg"])
    )
    cfg = {
        "vehicle_reference": {
            "apriltag_black_edge_m": TAG_EDGE_M,
            "apriltags": {
                "0": {
                    "center_car_m": LAYOUT[0]["center"].tolist(),
                    "R_tag_car": r0.tolist(),
                },
                "1": {
                    "center_car_m": LAYOUT[1]["center"].tolist(),
                    "face_azimuth_deg": LAYOUT[1]["az_deg"],
                    "inplane_rotation_deg": LAYOUT[1]["rho_deg"],
                },
            },
        }
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def _tag_world_corners_mm(
    tag_id: int, car_x: float, car_y: float, car_yaw: float
) -> np.ndarray:
    """独立于被测代码的角点生成（印面顺序 左上/右上/右下/左下）。"""
    spec = LAYOUT[tag_id]
    r_tag = tag_rotation_from_angles(
        math.radians(spec["az_deg"]), math.radians(spec["rho_deg"])
    )
    c, s = math.cos(car_yaw), math.sin(car_yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    center = spec["center"]
    center_w = np.array([
        car_x + c * center[0] - s * center[1],
        car_y + s * center[0] + c * center[1],
        center[2],
    ])
    r_w = rz @ r_tag
    half = TAG_EDGE_M / 2.0
    printed = np.array([
        [-half, half], [half, half], [half, -half], [-half, -half],
    ])
    return (center_w[None, :] + printed @ r_w[:, :2].T) * 1000.0


def _project(corners_mm: np.ndarray, R: np.ndarray, position_m: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R.dot(position_m * 1000.0)
    projected, _ = cv2.projectPoints(corners_mm, rvec, tvec, K, np.zeros(5))
    return projected.reshape(4, 2)


def _detection(tag_id: int, corners: np.ndarray) -> CarDetection:
    return CarDetection(
        tag_id=tag_id,
        cx=float(corners[:, 0].mean()),
        cy=float(corners[:, 1].mean()),
        corners=corners,
    )


def _make_case(
    tmp_path,
    *,
    x_m: float,
    y_m: float,
    yaw: float,
    tag_cameras: dict[int, list[str]],
):
    calib_path = tmp_path / "calib.json"
    vehicle_path = tmp_path / "vehicle.json"
    rotations, positions_m = _write_calibration(calib_path)
    _write_vehicle(vehicle_path)
    localizer = CarLocalizer(str(calib_path), str(vehicle_path))

    tag_detections: dict[int, dict[str, CarDetection]] = {}
    for tag_id, serials in tag_cameras.items():
        corners_mm = _tag_world_corners_mm(tag_id, x_m, y_m, yaw)
        for sn in serials:
            corners_px = _project(corners_mm, rotations[sn], positions_m[sn])
            tag_detections.setdefault(tag_id, {})[sn] = _detection(
                tag_id, corners_px
            )
    return localizer, tag_detections


def _yaw_error(actual: float, expected: float) -> float:
    return math.atan2(math.sin(actual - expected), math.cos(actual - expected))


ALL_CAMS = ["cam0", "cam1", "cam2", "cam3"]


def test_dual_tag_recovers_car_pose_and_reports_both_tags(tmp_path):
    expected = (0.35, 2.4, 0.43)
    localizer, dets = _make_case(
        tmp_path,
        x_m=expected[0], y_m=expected[1], yaw=expected[2],
        tag_cameras={0: ALL_CAMS, 1: ALL_CAMS},
    )

    result = localizer.estimate_car_pose(dets)

    assert result is not None
    assert result.tag_ids == [0, 1]
    assert result.yaw_valid is True
    assert set(result.cameras_used) == set(ALL_CAMS)
    assert result.x == pytest.approx(expected[0], abs=0.002)
    assert result.y == pytest.approx(expected[1], abs=0.002)
    assert result.z == pytest.approx(0.0, abs=1e-9)
    assert abs(_yaw_error(result.yaw, expected[2])) < math.radians(0.2)


@pytest.mark.parametrize("tag_id", [0, 1])
def test_single_tag_fallback_recovers_car_pose(tmp_path, tag_id):
    expected = (-0.25, 1.8, -0.37)
    localizer, dets = _make_case(
        tmp_path,
        x_m=expected[0], y_m=expected[1], yaw=expected[2],
        tag_cameras={tag_id: ALL_CAMS},
    )

    result = localizer.estimate_pose(dets[tag_id])

    assert result is not None
    assert result.tag_ids == [tag_id]
    assert result.tag_id == tag_id
    assert result.yaw_valid is True
    assert result.x == pytest.approx(expected[0], abs=0.002)
    assert result.y == pytest.approx(expected[1], abs=0.002)
    assert abs(_yaw_error(result.yaw, expected[2])) < math.radians(0.2)


def test_flipped_tag_layout_matters(tmp_path):
    """id0 倒贴（面内 180°）：若代码忽略面内旋转，角点模型上下颠倒必然拟合失败。"""
    expected = (0.1, 2.2, 0.2)
    localizer, dets = _make_case(
        tmp_path,
        x_m=expected[0], y_m=expected[1], yaw=expected[2],
        tag_cameras={0: ALL_CAMS},
    )

    result = localizer.estimate_pose(dets[0])

    assert result is not None
    assert result.reprojection_error < 0.5
    assert result.x == pytest.approx(expected[0], abs=0.002)
    assert result.y == pytest.approx(expected[1], abs=0.002)


def test_bad_corner_view_is_rejected_without_losing_camera(tmp_path):
    expected = (0.3, 2.0, 0.5)
    localizer, dets = _make_case(
        tmp_path,
        x_m=expected[0], y_m=expected[1], yaw=expected[2],
        tag_cameras={0: ALL_CAMS, 1: ALL_CAMS},
    )
    bad = dets[1]["cam3"]
    bad.corners = bad.corners + np.array(
        [[90.0, -70.0], [130.0, 40.0], [-80.0, 100.0], [-120.0, -50.0]]
    )

    result = localizer.estimate_car_pose(dets)

    assert result is not None
    # cam3 的 id0 视图仍在，相机不丢
    assert set(result.cameras_used) == set(ALL_CAMS)
    assert result.tag_ids == [0, 1]
    assert result.reprojection_error < 0.5
    assert result.x == pytest.approx(expected[0], abs=0.002)
    assert result.y == pytest.approx(expected[1], abs=0.002)
    assert abs(_yaw_error(result.yaw, expected[2])) < math.radians(0.2)


def test_two_cameras_single_tag_does_not_publish_visual_yaw(tmp_path):
    localizer, dets = _make_case(
        tmp_path,
        x_m=0.2, y_m=2.1, yaw=0.25,
        tag_cameras={0: ["cam0", "cam1"]},
    )

    result = localizer.estimate_pose(dets[0])

    assert result is not None
    assert result.yaw_valid is False
    assert set(result.cameras_used) == {"cam0", "cam1"}


def test_two_cameras_dual_tag_publishes_yaw(tmp_path):
    """双 tag 中心基线 ~0.9m，即便只有 2 台相机 yaw 也可用。"""
    expected = (0.2, 2.1, 0.25)
    localizer, dets = _make_case(
        tmp_path,
        x_m=expected[0], y_m=expected[1], yaw=expected[2],
        tag_cameras={0: ["cam0", "cam1"], 1: ["cam0", "cam1"]},
    )

    result = localizer.estimate_car_pose(dets)

    assert result is not None
    assert result.yaw_valid is True
    assert abs(_yaw_error(result.yaw, expected[2])) < math.radians(0.2)


def test_one_camera_tag_still_contributes(tmp_path):
    expected = (0.15, 2.3, -0.2)
    localizer, dets = _make_case(
        tmp_path,
        x_m=expected[0], y_m=expected[1], yaw=expected[2],
        tag_cameras={0: ["cam0", "cam1", "cam2"], 1: ["cam3"]},
    )

    result = localizer.estimate_car_pose(dets)

    assert result is not None
    assert result.tag_ids == [0, 1]
    assert set(result.cameras_used) == set(ALL_CAMS)
    assert result.x == pytest.approx(expected[0], abs=0.002)
    assert result.y == pytest.approx(expected[1], abs=0.002)


def test_unknown_tag_id_raises(tmp_path):
    localizer, dets = _make_case(
        tmp_path,
        x_m=0.0, y_m=2.0, yaw=0.0,
        tag_cameras={0: ["cam0", "cam1"]},
    )
    for det in dets[0].values():
        det.tag_id = 9

    with pytest.raises(ValueError):
        localizer.estimate_pose(dets[0])


def test_estimate_car_pose_without_triangulatable_tag_returns_none(tmp_path):
    localizer, dets = _make_case(
        tmp_path,
        x_m=0.0, y_m=2.0, yaw=0.0,
        tag_cameras={0: ["cam0"], 1: ["cam1"]},
    )

    assert localizer.estimate_car_pose(dets) is None


def _fake_locate_images(localizer, dets):
    """伪造 detect：用 1x1 标记图把 locate 走成端到端路径。"""
    per_cam: dict[str, list[CarDetection]] = {sn: [] for sn in ALL_CAMS}
    for cam_dets in dets.values():
        for sn, det in cam_dets.items():
            per_cam[sn].append(det)
    images = {
        sn: np.full((1, 1), i, dtype=np.uint8)
        for i, sn in enumerate(ALL_CAMS)
    }
    marker_to_sn = dict(enumerate(ALL_CAMS))
    localizer.detect = (  # type: ignore[method-assign]
        lambda img: per_cam[marker_to_sn[int(img[0, 0])]]
    )
    return images


def test_locate_publishes_only_with_both_tags(tmp_path):
    localizer, dets = _make_case(
        tmp_path,
        x_m=0.3, y_m=2.2, yaw=0.1,
        tag_cameras={0: ALL_CAMS, 1: ALL_CAMS},
    )
    images = _fake_locate_images(localizer, dets)

    result = localizer.locate(images)

    assert result is not None
    assert result.tag_ids == [0, 1]


def test_locate_rejects_single_tag_by_default(tmp_path):
    localizer, dets = _make_case(
        tmp_path,
        x_m=0.3, y_m=2.2, yaw=0.1,
        tag_cameras={0: ALL_CAMS},
    )
    images = _fake_locate_images(localizer, dets)

    assert localizer.locate(images) is None
    # 诊断口径仍可解
    assert localizer.locate(images, min_tags=1) is not None


def test_locate_rejects_when_outlier_removal_drops_a_tag(tmp_path):
    localizer, dets = _make_case(
        tmp_path,
        x_m=0.3, y_m=2.2, yaw=0.1,
        tag_cameras={0: ALL_CAMS, 1: ["cam3"]},
    )
    bad = dets[1]["cam3"]
    bad.corners = bad.corners + np.array(
        [[90.0, -70.0], [130.0, 40.0], [-80.0, 100.0], [-120.0, -50.0]]
    )
    images = _fake_locate_images(localizer, dets)

    assert localizer.locate(images) is None


# ── 暗光 AprilTag 救援 ────────────────────────────────────────────────────
# 2026-08-09 夜实测：光线掉下去后 tag 白块只剩 ~25 灰度（黑块 ~2），ArUco 自适应
# 阈值取不出边缘，car_loc 从 miss 1.4% 直接变成 100%；而场景均值与可用场次相同
# （24.5 vs 24.1），所以拉曝光救不回来——必须拉局部对比度。detect() 因此在 raw
# 全空时用 CLAHE 重试一次。这里用打桩 detectMarkers 确定性地验证这条分支。


def _localizer(tmp_path):
    calib_path = tmp_path / "calib.json"
    vehicle_path = tmp_path / "vehicle.json"
    _write_calibration(calib_path)
    _write_vehicle(vehicle_path)
    return CarLocalizer(str(calib_path), str(vehicle_path))


class _StubDetector:
    """按调用次序返回 results 里的 (corners, ids)，并记录每次收到的图。

    cv2.aruco.ArucoDetector 是 C++ 绑定、方法只读，只能整体替换。
    """

    def __init__(self, results):
        self._results = results
        self.seen: list[np.ndarray] = []

    def detectMarkers(self, gray):
        self.seen.append(gray)
        corners, ids = self._results[min(len(self.seen) - 1, len(self._results) - 1)]
        return corners, ids, None


def _stub_detect_markers(localizer, results):
    stub = _StubDetector(results)
    localizer._detector = stub
    return stub.seen


def test_detect_retries_with_contrast_boost_when_raw_finds_nothing(tmp_path):
    localizer = _localizer(tmp_path)
    corners = [np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]])]
    seen = _stub_detect_markers(
        localizer,
        [(None, None), (corners, np.array([[0]]))],   # raw 空 → 重试命中
    )
    gray = np.full((64, 64), 8, np.uint8)
    gray[20:40, 20:40] = 26                            # 暗光低对比

    dets = localizer.detect(gray)

    assert [d.tag_id for d in dets] == [0]
    assert localizer.low_light_retries == 1
    assert localizer.low_light_recovered == 1
    assert len(seen) == 2, "raw 一次 + 重试一次"
    # 第二次收到的必须是增强过的图（对比度被拉开），不是原图
    assert seen[1].ptp() > seen[0].ptp()


def test_detect_does_not_retry_when_raw_succeeds(tmp_path):
    localizer = _localizer(tmp_path)
    corners = [np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]])]
    seen = _stub_detect_markers(localizer, [(corners, np.array([[1]]))])

    dets = localizer.detect(np.full((64, 64), 120, np.uint8))

    assert [d.tag_id for d in dets] == [1]
    assert localizer.low_light_retries == 0, "亮场不应付重试的 ~100ms"
    assert len(seen) == 1


def test_detect_counts_retry_but_not_recovery_when_boost_also_fails(tmp_path):
    localizer = _localizer(tmp_path)
    _stub_detect_markers(localizer, [(None, None), (None, None)])

    assert localizer.detect(np.full((64, 64), 3, np.uint8)) == []
    assert localizer.low_light_retries == 1
    assert localizer.low_light_recovered == 0


_STUB_CORNERS = [np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]])]


def _ptp(img) -> int:
    return int(img.max()) - int(img.min())


def _low_contrast_image():
    """暗且低对比：raw 检不出，但增强后能拉开动态范围。"""
    img = np.full((64, 64), 8, np.uint8)
    img[20:40, 20:40] = 26
    return img


class _ContrastStubDetector:
    """按图像动态范围决定成败，而不是按调用次序。

    真实行为是「同一帧 raw 失败、增强后成功」，按次序打桩会在状态机切换后错位
    （切换后不再跑 raw，次序桩就会把本该失败的那一档返回成功）。
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.seen: list[np.ndarray] = []

    def detectMarkers(self, gray):
        self.seen.append(gray)
        if _ptp(gray) < self.threshold:
            return None, None, None
        return _STUB_CORNERS, np.array([[0]]), None


def _stub_by_contrast(localizer, image):
    """阈值由真实增强函数的效果推出，CLAHE 参数一改这里会立刻失效并报错。"""
    raw_ptp = _ptp(image)
    enhanced_ptp = _ptp(CarLocalizer._enhance_for_low_light(image))
    assert enhanced_ptp > raw_ptp, "增强必须真的拉开动态范围，否则本用例无意义"
    stub = _ContrastStubDetector((raw_ptp + enhanced_ptp) / 2.0)
    localizer._detector = stub
    return stub.seen


def test_detect_switches_to_enhanced_first_after_repeated_dark_frames(tmp_path):
    """暗场稳态下不能每帧都白跑一遍注定失败的 raw。

    2026-08-09 050621 场实测：每帧两遍 detectMarkers（~100ms 一遍）让后台定位
    跟不上，car_loc_dropped 从历来的 0 涨到 48%。连续几帧确认 raw 无效后应改为
    直接从增强图起手，暗场成本回到一次。
    """
    localizer = _localizer(tmp_path)
    dark = _low_contrast_image()
    seen = _stub_by_contrast(localizer, dark)

    for _ in range(localizer._PREFER_ENHANCED_AFTER):
        assert [d.tag_id for d in localizer.detect(dark)] == [0]
    assert len(seen) == 2 * localizer._PREFER_ENHANCED_AFTER, "切换前每帧两遍"
    assert localizer._prefer_enhanced is True

    before = len(seen)
    assert [d.tag_id for d in localizer.detect(dark)] == [0]
    assert len(seen) - before == 1, "切换后每帧只跑一遍（增强图）"


def test_detect_probes_raw_again_so_recovered_light_stops_using_enhanced(tmp_path):
    """光线恢复后要回到 raw —— 增强图角点有 ~0.27° 偏移，能不用就不用。"""
    localizer = _localizer(tmp_path)
    seen = _stub_by_contrast(localizer, _low_contrast_image())
    bright = np.full((64, 64), 40, np.uint8)
    bright[20:40, 20:40] = 200                                     # raw 就够检出

    localizer._prefer_enhanced = True
    localizer._enhanced_since_probe = localizer._RAW_PROBE_EVERY   # 该回探了

    assert [d.tag_id for d in localizer.detect(bright)] == [0]
    assert localizer._prefer_enhanced is False, "光线恢复后应停用增强图"
    assert len(seen) == 1, "回探命中就不该再跑增强图"
