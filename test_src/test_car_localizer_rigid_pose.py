from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from src.car_localizer import CarDetection, CarLocalizer


TAG_EDGE_M = 0.161
TAG_Z_M = 0.61


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
    target_m = np.array([0.0, 2.0, TAG_Z_M])
    K = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]]
    )
    rotations = {
        serial: _look_at(position, target_m)
        for serial, position in camera_positions_m.items()
    }
    cameras = {}
    for serial, position_m in camera_positions_m.items():
        R = rotations[serial]
        position_mm = position_m * 1000.0
        t = -R.dot(position_mm)
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


def _tag_corners_mm(x_m: float, y_m: float, yaw: float) -> np.ndarray:
    half = TAG_EDGE_M / 2.0
    local = np.array(
        [
            [-half, 0.0, half],
            [half, 0.0, half],
            [half, 0.0, -half],
            [-half, 0.0, -half],
        ]
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    center_m = np.array([x_m, y_m, TAG_Z_M])
    return (local.dot(rotation.T) + center_m) * 1000.0


def _detection(corners: np.ndarray) -> CarDetection:
    return CarDetection(
        tag_id=0,
        cx=float(corners[:, 0].mean()),
        cy=float(corners[:, 1].mean()),
        corners=corners,
    )


def _make_case(tmp_path, *, x_m: float, y_m: float, yaw: float):
    calib_path = tmp_path / "calib.json"
    rotations, positions_m = _write_calibration(calib_path)
    localizer = CarLocalizer(str(calib_path))
    localizer._apriltag_to_car_base_offset_m = np.zeros(3)
    localizer._apriltag_to_car_yaw_offset_rad = 0.0

    world_corners_mm = _tag_corners_mm(x_m, y_m, yaw)
    K = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]]
    )
    detections = {}
    for serial, R in rotations.items():
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R.dot(positions_m[serial] * 1000.0)
        projected, _ = cv2.projectPoints(
            world_corners_mm,
            rvec,
            tvec,
            K,
            np.zeros(5),
        )
        detections[serial] = _detection(projected.reshape(4, 2))
    return localizer, detections


def _yaw_error(actual: float, expected: float) -> float:
    return math.atan2(math.sin(actual - expected), math.cos(actual - expected))


@pytest.mark.parametrize("camera_count", [3, 4])
def test_rigid_tag_pose_recovers_xy_and_yaw_from_three_or_four_cameras(
    tmp_path, camera_count
):
    expected_x, expected_y, expected_yaw = 0.35, 2.4, 0.43
    localizer, all_detections = _make_case(
        tmp_path,
        x_m=expected_x,
        y_m=expected_y,
        yaw=expected_yaw,
    )
    detections = dict(list(all_detections.items())[:camera_count])

    result = localizer.estimate_pose(detections)

    assert result.yaw_valid is True
    assert set(result.cameras_used) == set(detections)
    assert result.x == pytest.approx(expected_x, abs=0.002)
    assert result.y == pytest.approx(expected_y, abs=0.002)
    assert result.z == pytest.approx(TAG_Z_M, abs=1e-9)
    assert abs(_yaw_error(result.yaw, expected_yaw)) < math.radians(0.2)


def test_rigid_tag_pose_rejects_one_camera_with_bad_corners(tmp_path):
    expected_x, expected_y, expected_yaw = -0.25, 1.8, -0.37
    localizer, detections = _make_case(
        tmp_path,
        x_m=expected_x,
        y_m=expected_y,
        yaw=expected_yaw,
    )
    bad_serial = "cam3"
    bad_corners = detections[bad_serial].corners.copy()
    bad_corners += np.array(
        [[90.0, -70.0], [130.0, 40.0], [-80.0, 100.0], [-120.0, -50.0]]
    )
    detections[bad_serial] = _detection(bad_corners)

    result = localizer.estimate_pose(detections)

    assert result.yaw_valid is True
    assert set(result.cameras_used) == {"cam0", "cam1", "cam2"}
    assert result.x == pytest.approx(expected_x, abs=0.002)
    assert result.y == pytest.approx(expected_y, abs=0.002)
    assert result.z == pytest.approx(TAG_Z_M, abs=1e-9)
    assert abs(_yaw_error(result.yaw, expected_yaw)) < math.radians(0.2)


def test_two_cameras_do_not_publish_visual_yaw(tmp_path):
    localizer, all_detections = _make_case(
        tmp_path,
        x_m=0.2,
        y_m=2.1,
        yaw=0.25,
    )
    detections = dict(list(all_detections.items())[:2])

    result = localizer.estimate_pose(detections)

    assert result.yaw_valid is False
    assert set(result.cameras_used) == set(detections)


def test_three_cameras_with_large_corner_error_do_not_publish_yaw(tmp_path):
    localizer, all_detections = _make_case(
        tmp_path,
        x_m=0.2,
        y_m=2.1,
        yaw=0.25,
    )
    detections = dict(list(all_detections.items())[:3])
    distortions = (
        np.array([[20.0, 0.0], [-20.0, 0.0], [20.0, 0.0], [-20.0, 0.0]]),
        np.array([[0.0, 20.0], [0.0, -20.0], [0.0, 20.0], [0.0, -20.0]]),
        np.array([[15.0, 15.0], [-15.0, -15.0], [15.0, 15.0], [-15.0, -15.0]]),
    )
    for detection, distortion in zip(detections.values(), distortions):
        detection.corners += distortion

    result = localizer.estimate_pose(detections)

    assert len(result.cameras_used) == 3
    assert result.reprojection_error > 8.0
    assert result.yaw_valid is False


def test_car_base_offset_rotates_with_car_yaw(tmp_path):
    tag_x, tag_y, yaw = 0.3, 2.0, math.pi / 2.0
    localizer, detections = _make_case(
        tmp_path,
        x_m=tag_x,
        y_m=tag_y,
        yaw=yaw,
    )
    localizer._apriltag_to_car_base_offset_m = np.array([0.04, 0.16, -0.61])

    result = localizer.estimate_pose(detections)

    assert result.x == pytest.approx(tag_x - 0.16, abs=0.002)
    assert result.y == pytest.approx(tag_y + 0.04, abs=0.002)
    assert result.z == pytest.approx(0.0, abs=1e-9)
