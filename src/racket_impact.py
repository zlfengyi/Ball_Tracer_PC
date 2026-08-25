"""Event-level racket-head bbox-centre velocity measurement.

The estimator buffers exact synchronized player crops, keeps detector bbox
geometry in native camera pixels, and calls the multi-camera bundle fitter once
per impact event.  The resulting velocity is a bbox-centre proxy; it is not a
calibrated racket-face speed or a direct observation of ball spin.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Sequence

import numpy as np

from .opponent_racket_bbox_localizer import (
    OpponentRacketBBoxDetection,
    OpponentRacketBBoxLocalizer,
)
from .racket_bbox_bundle import (
    BBoxObservation,
    BundleGates,
    CameraCalibration,
    fit_bbox_bundle,
    project_world_m,
)


_OBSERVATION_SEMANTICS = "racket_head_bbox_geometric_center_native_pixel"
_VELOCITY_SEMANTICS = "racket_head_bbox_center_world_velocity_proxy"


@dataclass(frozen=True)
class RacketCameraCrop:
    """Buffered player crop and its exact mapping into the native image."""

    image: np.ndarray
    origin_xy: tuple[int, int]
    native_size_wh: tuple[int, int]

    @property
    def native_roi_xyxy(self) -> tuple[float, float, float, float]:
        x0, y0 = self.origin_xy
        return (
            float(x0),
            float(y0),
            float(x0 + self.image.shape[1]),
            float(y0 + self.image.shape[0]),
        )


@dataclass(frozen=True)
class SynchronizedRacketFrame:
    """One exact video frame on the PC ``perf_counter`` exposure time axis."""

    video_frame_idx: int
    exposure_center_pc: float
    camera_crops: Mapping[str, RacketCameraCrop]


@dataclass(frozen=True)
class RacketBBoxObservationRecord:
    """Raw detector bbox retained for bundle-result auditability."""

    video_frame_idx: int
    exposure_center_pc: float
    time_to_anchor_s: float
    serial: str
    candidate_rank: int
    bbox_confidence: float
    bbox_native_xyxy: tuple[float, float, float, float]
    bbox_center_native_xy: tuple[float, float]


@dataclass(frozen=True)
class RacketImpactMeasurement:
    """One bundle decision and the unmodified bbox evidence behind it."""

    accepted: bool
    reason: str
    contact_anchor_pc: float
    window_start_pc: float
    window_end_pc: float
    observation_semantics: str
    velocity_semantics: str
    raw_bbox_observations: tuple[RacketBBoxObservationRecord, ...]
    bundle_diagnostics: dict
    bbox_center_vz_world_mps: float | None = None
    n_input_frames: int = 0
    n_contact_window_frames: int = 0
    camera_support_frames: dict[str, int] = field(default_factory=dict)
    rejection_counts: dict[str, int] = field(default_factory=dict)


class RacketImpactEstimator:
    """Measure one event with native bbox centres and one bundle invocation."""

    def __init__(
        self,
        player_box_world_m: Mapping[str, Sequence[float]],
        *,
        localizer: OpponentRacketBBoxLocalizer,
        cameras: Mapping[str, CameraCalibration],
        bundle_gates: BundleGates,
        max_candidates_per_camera_frame: int = 3,
        bbox_edge_margin_px: float = 8.0,
        player_roi_padding_px: int = 40,
    ) -> None:
        self._player_box = self._validate_player_box(player_box_world_m)
        if not cameras:
            raise ValueError("at least one calibrated camera is required")
        self._cameras = dict(cameras)
        self._localizer = localizer
        self._bundle_gates = bundle_gates
        self._max_candidates = int(max_candidates_per_camera_frame)
        self._bbox_edge_margin_px = float(bbox_edge_margin_px)
        self._player_roi_padding_px = int(player_roi_padding_px)
        if self._max_candidates < 1:
            raise ValueError("max_candidates_per_camera_frame must be positive")
        if not math.isfinite(self._bbox_edge_margin_px) or self._bbox_edge_margin_px < 0.0:
            raise ValueError("bbox_edge_margin_px must be finite and nonnegative")
        if self._player_roi_padding_px < 0:
            raise ValueError("player_roi_padding_px must be nonnegative")
        if (
            not math.isfinite(bundle_gates.window_before_contact_s)
            or not math.isfinite(bundle_gates.window_after_contact_s)
            or bundle_gates.window_before_contact_s <= 0.0
            or bundle_gates.window_after_contact_s < 0.0
        ):
            raise ValueError("bundle contact window must be finite and nonnegative")
        self._roi_cache: dict[
            tuple[str, int, int], tuple[float, float, float, float]
        ] = {}

    @property
    def provider_info(self) -> dict[str, list[str]]:
        return self._localizer.provider_info

    @staticmethod
    def _validate_player_box(
        box: Mapping[str, Sequence[float]],
    ) -> dict[str, tuple[float, float]]:
        result = {}
        for axis in ("x", "y", "z"):
            values = tuple(float(value) for value in box[axis])
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"player_box_world_m.{axis} must contain two finite values")
            if values[0] >= values[1]:
                raise ValueError(f"player_box_world_m.{axis} must be increasing")
            result[axis] = values
        return result

    def prepare_frame(
        self,
        video_frame_idx: int,
        exposure_center_pc: float,
        images: Mapping[str, np.ndarray],
    ) -> SynchronizedRacketFrame:
        """Copy player crops without inference, preserving exact frame identity."""

        if isinstance(video_frame_idx, bool) or not isinstance(video_frame_idx, int):
            raise ValueError("video_frame_idx must be an integer")
        if video_frame_idx < 0:
            raise ValueError("video_frame_idx must be nonnegative")
        timestamp = float(exposure_center_pc)
        if not math.isfinite(timestamp):
            raise ValueError("exposure_center_pc must be finite")

        crops: dict[str, RacketCameraCrop] = {}
        for serial in self._cameras:
            image = images.get(serial)
            if (
                image is None
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
            ):
                continue
            height, width = image.shape[:2]
            roi = self._player_roi(serial, image.shape)
            if roi is None:
                continue
            x0 = max(0, int(math.floor(roi[0]))) & ~1
            y0 = max(0, int(math.floor(roi[1]))) & ~1
            x1 = min(width, int(math.ceil(roi[2])))
            y1 = min(height, int(math.ceil(roi[3])))
            if x1 & 1 and x1 < width:
                x1 += 1
            if y1 & 1 and y1 < height:
                y1 += 1
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crops[serial] = RacketCameraCrop(
                image=np.ascontiguousarray(image[y0:y1, x0:x1]).copy(),
                origin_xy=(x0, y0),
                native_size_wh=(width, height),
            )
        return SynchronizedRacketFrame(
            video_frame_idx=video_frame_idx,
            exposure_center_pc=timestamp,
            camera_crops=crops,
        )

    def measure(
        self,
        frames: Sequence[SynchronizedRacketFrame],
        contact_anchor_pc: float,
    ) -> RacketImpactMeasurement:
        """Collect all gated bbox candidates, then fit the event exactly once."""

        anchor = float(contact_anchor_pc)
        if not math.isfinite(anchor):
            raise ValueError("contact_anchor_pc must be finite")
        window_start = anchor - self._bundle_gates.window_before_contact_s
        window_end = anchor + self._bundle_gates.window_after_contact_s
        selected = sorted(
            (
                frame
                for frame in frames
                if window_start <= frame.exposure_center_pc <= window_end
            ),
            key=lambda frame: (frame.exposure_center_pc, frame.video_frame_idx),
        )

        raw_records: list[RacketBBoxObservationRecord] = []
        bundle_observations: list[BBoxObservation] = []
        camera_support: Counter[str] = Counter()
        rejected: Counter[str] = Counter()

        for frame in selected:
            for serial in self._cameras:
                crop = frame.camera_crops.get(serial)
                if crop is None:
                    rejected["missing_player_crop"] += 1
                    continue
                detections = self._localizer.detect_candidates(
                    crop.image,
                    serial=serial,
                    image_origin_xy=crop.origin_xy,
                    max_candidates=self._max_candidates,
                )
                if not detections:
                    rejected["no_racket_bbox_in_player_crop"] += 1
                    continue
                ordered = sorted(
                    detections,
                    key=lambda item: item.bbox_confidence,
                    reverse=True,
                )[: self._max_candidates]
                camera_has_observation = False
                for candidate_rank, detection in enumerate(ordered, start=1):
                    reason, bbox, center = self._gate_detection(
                        detection,
                        serial,
                        crop,
                    )
                    if reason:
                        rejected[reason] += 1
                        continue
                    record = RacketBBoxObservationRecord(
                        video_frame_idx=frame.video_frame_idx,
                        exposure_center_pc=frame.exposure_center_pc,
                        time_to_anchor_s=frame.exposure_center_pc - anchor,
                        serial=serial,
                        candidate_rank=candidate_rank,
                        bbox_confidence=float(detection.bbox_confidence),
                        bbox_native_xyxy=bbox,
                        bbox_center_native_xy=center,
                    )
                    raw_records.append(record)
                    bundle_observations.append(
                        BBoxObservation(
                            frame_id=frame.video_frame_idx,
                            serial=serial,
                            exposure_center_s=frame.exposure_center_pc,
                            center_xy=center,
                            bbox_confidence=float(detection.bbox_confidence),
                        )
                    )
                    camera_has_observation = True
                if camera_has_observation:
                    camera_support[serial] += 1

        diagnostics = fit_bbox_bundle(
            bundle_observations,
            anchor,
            self._cameras,
            self._bundle_gates,
        )
        accepted = bool(diagnostics["accepted"])
        vz = (
            float(diagnostics["bbox_center_vz_world_mps"])
            if accepted
            else None
        )
        return RacketImpactMeasurement(
            accepted=accepted,
            reason=str(diagnostics["reason"]),
            contact_anchor_pc=anchor,
            window_start_pc=window_start,
            window_end_pc=window_end,
            observation_semantics=_OBSERVATION_SEMANTICS,
            velocity_semantics=_VELOCITY_SEMANTICS,
            raw_bbox_observations=tuple(raw_records),
            bundle_diagnostics=dict(diagnostics),
            bbox_center_vz_world_mps=vz,
            n_input_frames=len(frames),
            n_contact_window_frames=len(selected),
            camera_support_frames=dict(camera_support),
            rejection_counts=dict(rejected),
        )

    def _player_roi(
        self,
        serial: str,
        image_shape: tuple[int, ...],
    ) -> tuple[float, float, float, float] | None:
        height, width = int(image_shape[0]), int(image_shape[1])
        key = (serial, width, height)
        cached = self._roi_cache.get(key)
        if cached is not None:
            return cached
        projected = []
        for x, y, z in product(
            self._player_box["x"],
            self._player_box["y"],
            self._player_box["z"],
        ):
            try:
                pixel = project_world_m(self._cameras, serial, (x, y, z))
            except (KeyError, ValueError):
                return None
            if not all(math.isfinite(value) for value in pixel):
                return None
            projected.append(pixel)
        pixels = np.asarray(projected, dtype=np.float64)
        padding = self._player_roi_padding_px
        x0 = max(0.0, float(pixels[:, 0].min()) - padding)
        y0 = max(0.0, float(pixels[:, 1].min()) - padding)
        x1 = min(float(width), float(pixels[:, 0].max()) + padding)
        y1 = min(float(height), float(pixels[:, 1].max()) + padding)
        if x1 - x0 < 16.0 or y1 - y0 < 16.0:
            return None
        roi = (x0, y0, x1, y1)
        self._roi_cache[key] = roi
        return roi

    def _gate_detection(
        self,
        detection: OpponentRacketBBoxDetection,
        serial: str,
        crop: RacketCameraCrop,
    ) -> tuple[
        str,
        tuple[float, float, float, float],
        tuple[float, float],
    ]:
        empty_bbox = (math.nan, math.nan, math.nan, math.nan)
        empty_center = (math.nan, math.nan)
        if detection.serial != serial:
            return "racket_camera_mismatch", empty_bbox, empty_center
        confidence = float(detection.bbox_confidence)
        if not math.isfinite(confidence):
            return "invalid_racket_geometry", empty_bbox, empty_center
        if confidence < self._bundle_gates.bbox_confidence_min:
            return "low_racket_bbox_confidence", empty_bbox, empty_center
        bbox = tuple(float(value) for value in detection.bbox_xyxy)
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            return "invalid_racket_geometry", empty_bbox, empty_center
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return "invalid_racket_geometry", empty_bbox, empty_center
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        width, height = crop.native_size_wh
        margin = self._bbox_edge_margin_px
        if (
            x0 <= margin
            or y0 <= margin
            or x1 >= width - 1 - margin
            or y1 >= height - 1 - margin
        ):
            return "racket_clipped_by_image_edge", bbox, center
        crop_x0, crop_y0, crop_x1, crop_y1 = crop.native_roi_xyxy
        if (
            x0 <= crop_x0 + margin
            or y0 <= crop_y0 + margin
            or x1 >= crop_x1 - margin
            or y1 >= crop_y1 - margin
        ):
            return "racket_clipped_by_player_crop", bbox, center
        return "", bbox, center


__all__ = [
    "RacketBBoxObservationRecord",
    "RacketCameraCrop",
    "RacketImpactEstimator",
    "RacketImpactMeasurement",
    "SynchronizedRacketFrame",
]
