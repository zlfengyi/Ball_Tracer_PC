from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import racket_bbox_bundle as module
from src.racket_bbox_bundle import (
    BBoxObservation,
    BundleGates,
    CameraCalibration,
    fit_bbox_bundle,
    load_cameras,
    project_world_m,
)


_CONTACT_TIME_S = 1000.0
_FRAME_TIMES = {
    100: -0.12,
    101: -0.08,
    102: -0.02,
    103: 0.02,
}


def _camera(translation_mm: tuple[float, float, float]) -> CameraCalibration:
    rotation = np.eye(3, dtype=np.float64)
    return CameraCalibration(
        K=np.asarray(
            [[800.0, 0.0, 640.0], [0.0, 810.0, 480.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        D=np.asarray([0.008, -0.001, 0.0002, -0.0001, 0.0]),
        R=rotation,
        t=np.asarray(translation_mm, dtype=np.float64).reshape(3, 1),
        rvec=cv2.Rodrigues(rotation)[0],
    )


def _cameras() -> dict[str, CameraCalibration]:
    return {
        "cam_a": _camera((0.0, 0.0, 0.0)),
        "cam_b": _camera((-800.0, 0.0, 0.0)),
        "cam_c": _camera((0.0, -700.0, 0.0)),
    }


def _observations(
    position_m=(0.1, -0.2, 5.0),
    velocity_mps=(0.4, -0.6, 1.2),
    *,
    prepend_filtered: bool = False,
    add_outliers: bool = False,
):
    cameras = _cameras()
    observations = []
    good_indices = []
    outlier_indices = []
    if prepend_filtered:
        observations.append(
            BBoxObservation(
                frame_id=99,
                serial="unknown",
                exposure_center_s=_CONTACT_TIME_S - 0.1,
                center_xy=(1.0, 1.0),
                bbox_confidence=1.0,
            )
        )
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    for frame_id, tau in _FRAME_TIMES.items():
        point = position + velocity * tau
        for serial in cameras:
            pixel = project_world_m(cameras, serial, tuple(point))
            good_indices.append(len(observations))
            observations.append(
                BBoxObservation(
                    frame_id=frame_id,
                    serial=serial,
                    exposure_center_s=_CONTACT_TIME_S + tau,
                    center_xy=pixel,
                    bbox_confidence=0.9,
                )
            )
        if add_outliers:
            outlier_indices.append(len(observations))
            observations.append(
                BBoxObservation(
                    frame_id=frame_id,
                    serial="cam_a",
                    exposure_center_s=_CONTACT_TIME_S + tau,
                    center_xy=(pixel[0] + 180.0, pixel[1] - 130.0),
                    bbox_confidence=0.95,
                )
            )
    return observations, cameras, good_indices, outlier_indices


def _gates(**changes) -> BundleGates:
    values = {
        "world_volume_m": ((-2.0, 2.0), (-2.0, 2.0), (3.0, 7.0)),
        "bbox_confidence_min": 0.05,
    }
    values.update(changes)
    return BundleGates(**values)


def test_module_has_no_legacy_localizer_dependency():
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("racket_localizer", "RacketPose", "racket_pose", "keypoint"):
        assert forbidden not in source


def test_native_pixel_bundle_recovers_bbox_center_proxy_and_input_inliers():
    observations, cameras, good_indices, outlier_indices = _observations(
        prepend_filtered=True,
        add_outliers=True,
    )

    result = fit_bbox_bundle(
        observations,
        _CONTACT_TIME_S,
        cameras,
        _gates(),
    )
    repeated = fit_bbox_bundle(
        observations,
        _CONTACT_TIME_S,
        cameras,
        _gates(),
    )

    assert result["accepted"]
    assert result["reason"] == "accepted"
    assert result["observation_semantics"] == "racket_head_bbox_geometric_center"
    assert result["bbox_center_position_world_m"] == pytest.approx(
        (0.1, -0.2, 5.0), abs=1e-8
    )
    assert result["bbox_center_velocity_world_mps"] == pytest.approx(
        (0.4, -0.6, 1.2), abs=1e-8
    )
    assert result["bbox_center_vz_world_mps"] == pytest.approx(1.2, abs=1e-8)
    assert result["supported_frames"] == [100, 101, 102, 103]
    assert result["fit_span_s"] == pytest.approx(0.14)
    assert result["inlier_observation_indices"] == good_indices
    assert not set(result["inlier_observation_indices"]) & set(outlier_indices)
    assert result["inlier_observations"] == len(good_indices)
    assert len(result["leave_one_frame_bbox_center_vz_mps"]) == 4
    assert "position_world_m" not in result
    assert "vz_world_mps" not in result
    assert repeated["inlier_observation_indices"] == result["inlier_observation_indices"]
    assert repeated["bbox_center_velocity_world_mps"] == pytest.approx(
        result["bbox_center_velocity_world_mps"]
    )


def test_negative_bbox_center_velocity_is_preserved():
    observations, cameras, _, _ = _observations(velocity_mps=(0.2, -0.4, -1.5))
    result = fit_bbox_bundle(observations, _CONTACT_TIME_S, cameras, _gates())

    assert result["accepted"]
    assert result["bbox_center_vz_world_mps"] == pytest.approx(-1.5, abs=1e-8)
    assert all(value < 0.0 for value in result["leave_one_frame_bbox_center_vz_mps"])


def test_fewer_than_three_supported_frames_is_rejected_before_fit():
    observations, cameras, _, _ = _observations()
    two_frames = [item for item in observations if item.frame_id in (100, 101)]

    result = fit_bbox_bundle(two_frames, _CONTACT_TIME_S, cameras, _gates())

    assert not result["accepted"]
    assert result["reason"] == "insufficient_bbox_observations"
    assert "bbox_center_vz_world_mps" not in result


def test_many_single_camera_frames_plus_one_two_camera_frame_cannot_bundle():
    observations, cameras, _, _ = _observations()
    sparse = [item for item in observations if item.serial == "cam_a"]
    sparse.append(
        next(
            item
            for item in observations
            if item.frame_id == 102 and item.serial == "cam_b"
        )
    )
    sparse.extend(
        BBoxObservation(
            frame_id=item.frame_id,
            serial=item.serial,
            exposure_center_s=item.exposure_center_s,
            center_xy=item.center_xy,
            bbox_confidence=0.08,
        )
        for item in sparse[:2]
    )

    all_inliers = np.ones(len(sparse), dtype=bool)
    assert module._supported_frames(sparse, all_inliers) == [102]

    result = fit_bbox_bundle(sparse, _CONTACT_TIME_S, cameras, _gates())

    assert not result["accepted"]
    assert result["reason"] == "no_bundle_inlier_model"
    assert "bbox_center_vz_world_mps" not in result


@pytest.mark.parametrize(
    ("position_m", "velocity_mps", "gates", "reason"),
    [
        (
            (3.6, 0.0, 5.0),
            (0.0, 0.0, 1.0),
            _gates(world_volume_m=((-3.45, 3.45), (-3.0, 3.0), (3.0, 7.0))),
            "bundle_world_or_speed_gate",
        ),
        (
            (0.0, 0.0, 5.0),
            (40.0, 0.0, 1.0),
            _gates(),
            "bundle_world_or_speed_gate",
        ),
        (
            (0.0, 0.0, 5.0),
            (0.2, 0.0, 0.20),
            _gates(),
            "weak_or_implausible_vz",
        ),
    ],
)
def test_terminal_physical_gates_keep_proxy_diagnostics(
    position_m,
    velocity_mps,
    gates,
    reason,
):
    observations, cameras, _, _ = _observations(position_m, velocity_mps)

    result = fit_bbox_bundle(observations, _CONTACT_TIME_S, cameras, gates)

    assert not result["accepted"]
    assert result["reason"] == reason
    assert "bbox_center_position_world_m" in result
    assert "inlier_observation_indices" in result


def test_bundle_geometry_margin_accepts_position_inside_player_box_plus_060():
    observations, cameras, _, _ = _observations(position_m=(3.55, 0.0, 3.7))
    geometry_volume = _gates(
        world_volume_m=((-3.60, 3.60), (-3.60, 3.60), (-0.60, 3.80)),
    )

    result = fit_bbox_bundle(observations, _CONTACT_TIME_S, cameras, geometry_volume)

    assert result["accepted"]
    assert result["bbox_center_position_world_m"][0] == pytest.approx(3.55)


def test_duplicate_nearby_candidates_cannot_repeat_weight_or_win_by_count():
    observations, cameras, good_indices, _ = _observations()
    duplicated = list(observations)
    duplicate_indices = []
    for item in observations:
        duplicate_indices.append(len(duplicated))
        duplicated.append(
            BBoxObservation(
                frame_id=item.frame_id,
                serial=item.serial,
                exposure_center_s=item.exposure_center_s,
                center_xy=(item.center_xy[0] + 0.3, item.center_xy[1] - 0.2),
                bbox_confidence=0.99,
            )
        )

    result = fit_bbox_bundle(duplicated, _CONTACT_TIME_S, cameras, _gates())

    assert result["accepted"]
    inliers = result["inlier_observation_indices"]
    cells = {(duplicated[index].frame_id, duplicated[index].serial) for index in inliers}
    assert len(inliers) == len(cells) == len(good_indices)
    assert set(inliers) <= set(good_indices) | set(duplicate_indices)


def test_nonfinite_residual_is_never_selected_as_camera_frame_candidate():
    observations = [
        BBoxObservation(100, "cam_a", 1.0, (10.0, 20.0), 0.9),
        BBoxObservation(100, "cam_a", 1.0, (11.0, 21.0), 0.8),
    ]

    selected = module._one_candidate_per_camera_frame(
        observations,
        np.asarray([np.nan, 2.0]),
        threshold_px=8.0,
    )

    assert selected.tolist() == [False, True]


def test_leave_one_frame_rejects_only_a_confident_opposite_sign():
    threshold = 0.45
    assert not module._confidently_opposite(1.0, [0.8, -0.10, 0.9], threshold)
    assert module._confidently_opposite(1.0, [0.8, -0.46, 0.9], threshold)
    assert not module._confidently_opposite(-1.0, [-0.8, 0.10, -0.9], threshold)
    assert module._confidently_opposite(-1.0, [-0.8, 0.46, -0.9], threshold)


def test_load_cameras_and_project_world_use_calibration_native_pixels(tmp_path):
    calibration = {
        "cameras": {
            "camera": {
                "K": [[800.0, 0.0, 640.0], [0.0, 810.0, 480.0], [0.0, 0.0, 1.0]],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
                "R_world": np.eye(3).tolist(),
                "t_world": [0.0, 0.0, 0.0],
            }
        }
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    cameras = load_cameras(path)

    assert set(cameras) == {"camera"}
    assert project_world_m(cameras, "camera", (1.0, 0.5, 5.0)) == pytest.approx(
        (800.0, 561.0)
    )
    with pytest.raises(KeyError):
        project_world_m(cameras, "missing", (0.0, 0.0, 5.0))
