from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from src import racket_impact as module
from src.opponent_racket_bbox_localizer import OpponentRacketBBoxDetection
from src.racket_bbox_bundle import BundleGates, CameraCalibration
from src.racket_impact import RacketImpactEstimator


def _camera(translation_mm: tuple[float, float, float]) -> CameraCalibration:
    rotation = np.eye(3, dtype=np.float64)
    return CameraCalibration(
        K=np.asarray(
            [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        D=np.zeros(5, dtype=np.float64),
        R=rotation,
        t=np.asarray(translation_mm, dtype=np.float64).reshape(3, 1),
        rvec=cv2.Rodrigues(rotation)[0],
    )


def _cameras() -> dict[str, CameraCalibration]:
    return {
        "cam_a": _camera((0.0, 0.0, 0.0)),
        "cam_b": _camera((-800.0, 0.0, 0.0)),
    }


def _gates() -> BundleGates:
    return BundleGates(
        world_volume_m=((-2.0, 2.0), (-2.0, 2.0), (3.0, 7.0)),
        bbox_confidence_min=0.05,
        window_before_contact_s=0.125,
        window_after_contact_s=0.035,
    )


@dataclass
class _LocalizerCall:
    serial: str
    origin_xy: tuple[int, int]
    max_candidates: int
    image_value: int


class _FakeLocalizer:
    def __init__(self, *, clipped: bool = False) -> None:
        self.calls: list[_LocalizerCall] = []
        self._clipped = clipped

    @property
    def provider_info(self) -> dict[str, list[str]]:
        return {"opponent_racket_bbox": ["FakeExecutionProvider"]}

    def detect_candidates(
        self,
        image: np.ndarray,
        *,
        serial: str,
        image_origin_xy: tuple[int, int],
        max_candidates: int,
    ) -> list[OpponentRacketBBoxDetection]:
        self.calls.append(
            _LocalizerCall(
                serial=serial,
                origin_xy=image_origin_xy,
                max_candidates=max_candidates,
                image_value=int(image[0, 0, 0]),
            )
        )
        origin_x, origin_y = image_origin_xy
        if self._clipped:
            boxes = [(origin_x + 1.0, origin_y + 20.0, origin_x + 31.0, origin_y + 70.0)]
            scores = [0.90]
        else:
            boxes = [
                (origin_x + 30.0, origin_y + 30.0, origin_x + 70.0, origin_y + 90.0),
                (origin_x + 45.0, origin_y + 40.0, origin_x + 85.0, origin_y + 100.0),
                (origin_x + 60.0, origin_y + 50.0, origin_x + 100.0, origin_y + 110.0),
            ]
            scores = [0.70, 0.95, 0.80]
        return [
            OpponentRacketBBoxDetection(
                serial=serial,
                bbox_confidence=score,
                bbox_xyxy=box,
                center_xy=(-1.0, -1.0),
            )
            for box, score in zip(boxes, scores)
        ][:max_candidates]


def _estimator(localizer: _FakeLocalizer) -> RacketImpactEstimator:
    return RacketImpactEstimator(
        {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (4.0, 6.0),
        },
        localizer=localizer,
        cameras=_cameras(),
        bundle_gates=_gates(),
        max_candidates_per_camera_frame=3,
        bbox_edge_margin_px=8.0,
        player_roi_padding_px=40,
    )


def _frame(
    estimator: RacketImpactEstimator,
    video_frame_idx: int,
    exposure_center_pc: float,
):
    image = np.full((960, 1280, 3), video_frame_idx % 251, dtype=np.uint8)
    return estimator.prepare_frame(
        video_frame_idx,
        exposure_center_pc,
        {"cam_a": image, "cam_b": image},
    )


def test_prepare_frame_preserves_exact_identity_and_native_crop_mapping():
    estimator = _estimator(_FakeLocalizer())
    image = np.full((960, 1280, 3), 17, dtype=np.uint8)

    frame = estimator.prepare_frame(123456, 9876.54321, {"cam_a": image})
    image[:] = 99

    assert frame.video_frame_idx == 123456
    assert frame.exposure_center_pc == 9876.54321
    assert set(frame.camera_crops) == {"cam_a"}
    crop = frame.camera_crops["cam_a"]
    assert crop.origin_xy[0] % 2 == 0
    assert crop.origin_xy[1] % 2 == 0
    assert crop.native_size_wh == (1280, 960)
    assert crop.native_roi_xyxy == (
        float(crop.origin_xy[0]),
        float(crop.origin_xy[1]),
        float(crop.origin_xy[0] + crop.image.shape[1]),
        float(crop.origin_xy[1] + crop.image.shape[0]),
    )
    assert np.all(crop.image == 17)


def test_measure_uses_exact_pc_times_native_centres_and_one_bundle_call(monkeypatch):
    localizer = _FakeLocalizer()
    estimator = _estimator(localizer)
    anchor = 1000.0
    frames = [
        _frame(estimator, 90, anchor - 0.126),
        _frame(estimator, 101, anchor - 0.125),
        _frame(estimator, 205, anchor - 0.090),
        _frame(estimator, 309, anchor + 0.035),
        _frame(estimator, 400, anchor + 0.036),
    ]
    bundle_calls = []

    def fake_bundle(observations, contact_time_s, cameras, gates):
        bundle_calls.append((list(observations), contact_time_s, cameras, gates))
        return {
            "accepted": True,
            "reason": "accepted",
            "observation_semantics": "racket_head_bbox_geometric_center",
            "input_observations": len(observations),
            "eligible_observations": len(observations),
            "bbox_center_position_world_m": [0.1, -0.2, 5.0],
            "bbox_center_velocity_world_mps": [0.4, -0.6, 1.25],
            "bbox_center_vz_world_mps": 1.25,
            "supported_frames": [101, 205, 309],
            "fit_span_s": 0.13,
            "inlier_observations": len(observations),
            "inlier_observation_indices": list(range(len(observations))),
            "mean_reprojection_error_px": 0.5,
            "max_reprojection_error_px": 1.0,
            "leave_one_frame_bbox_center_vz_mps": [1.1, 1.2, 1.3],
        }

    monkeypatch.setattr(module, "fit_bbox_bundle", fake_bundle)
    measurement = estimator.measure(frames, anchor)

    assert len(bundle_calls) == 1
    observations, passed_anchor, passed_cameras, passed_gates = bundle_calls[0]
    assert passed_anchor == anchor
    assert passed_cameras.keys() == _cameras().keys()
    assert passed_gates is estimator._bundle_gates
    assert {item.frame_id for item in observations} == {101, 205, 309}
    assert {item.exposure_center_s for item in observations} == {
        anchor - 0.125,
        anchor - 0.090,
        anchor + 0.035,
    }
    assert len(observations) == 3 * 2 * 3
    assert all(call.max_candidates == 3 for call in localizer.calls)
    assert len(localizer.calls) == 3 * 2

    assert measurement.accepted
    assert measurement.reason == "accepted"
    assert measurement.bbox_center_vz_world_mps == 1.25
    assert measurement.observation_semantics == (
        "racket_head_bbox_geometric_center_native_pixel"
    )
    assert measurement.velocity_semantics == (
        "racket_head_bbox_center_world_velocity_proxy"
    )
    assert measurement.n_input_frames == 5
    assert measurement.n_contact_window_frames == 3
    assert measurement.camera_support_frames == {"cam_a": 3, "cam_b": 3}
    assert measurement.bundle_diagnostics["inlier_observation_indices"] == list(
        range(18)
    )
    assert len(measurement.raw_bbox_observations) == 18

    first = measurement.raw_bbox_observations[0]
    assert first.video_frame_idx == 101
    assert first.exposure_center_pc == anchor - 0.125
    assert first.time_to_anchor_s == pytest.approx(-0.125)
    assert first.serial == "cam_a"
    assert first.candidate_rank == 1
    assert first.bbox_confidence == 0.95
    assert first.bbox_center_native_xy == pytest.approx(
        (
            (first.bbox_native_xyxy[0] + first.bbox_native_xyxy[2]) / 2.0,
            (first.bbox_native_xyxy[1] + first.bbox_native_xyxy[3]) / 2.0,
        )
    )
    assert observations[0].center_xy == first.bbox_center_native_xy


def test_rejected_bundle_keeps_candidate_only_in_diagnostics(monkeypatch):
    localizer = _FakeLocalizer(clipped=True)
    estimator = _estimator(localizer)
    bundle_calls = []

    def fake_bundle(observations, contact_time_s, cameras, gates):
        bundle_calls.append(list(observations))
        return {
            "accepted": False,
            "reason": "insufficient_bbox_observations",
            "observation_semantics": "racket_head_bbox_geometric_center",
            "input_observations": 0,
            "eligible_observations": 0,
            "bbox_center_vz_world_mps": -1.8,
        }

    monkeypatch.setattr(module, "fit_bbox_bundle", fake_bundle)
    measurement = estimator.measure([_frame(estimator, 77, 99.90)], 100.0)

    assert len(bundle_calls) == 1
    assert bundle_calls[0] == []
    assert not measurement.accepted
    assert measurement.reason == "insufficient_bbox_observations"
    assert measurement.bbox_center_vz_world_mps is None
    assert measurement.bundle_diagnostics["bbox_center_vz_world_mps"] == -1.8
    assert measurement.raw_bbox_observations == ()
    assert measurement.rejection_counts == {"racket_clipped_by_player_crop": 2}
    assert estimator.provider_info == {
        "opponent_racket_bbox": ["FakeExecutionProvider"]
    }


@pytest.mark.parametrize("video_frame_idx", [-1, 1.5, True])
def test_prepare_frame_rejects_non_exact_video_frame_identity(video_frame_idx):
    estimator = _estimator(_FakeLocalizer())
    with pytest.raises(ValueError, match="video_frame_idx"):
        estimator.prepare_frame(video_frame_idx, 1.0, {})
