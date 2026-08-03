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
