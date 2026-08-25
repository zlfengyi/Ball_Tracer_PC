from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src import opponent_racket_bbox_localizer as module
from src.opponent_racket_bbox_localizer import OpponentRacketBBoxLocalizer


@dataclass
class _TensorSpec:
    name: str
    shape: list


class _ModelMeta:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.custom_metadata_map = metadata


class _Session:
    def __init__(
        self,
        *,
        metadata: dict[str, str] | None = None,
        input_shape: list | None = None,
        output_shape: list | None = None,
        runtime_output: np.ndarray | None = None,
    ) -> None:
        self._metadata = metadata or {
            "task": "detect",
            "names": "{0: 'racket_head'}",
        }
        self._inputs = [_TensorSpec("images", input_shape or [1, 3, 320, 320])]
        self._outputs = [_TensorSpec("output0", output_shape or [1, 5, 4])]
        output_count = self._outputs[0].shape[2]
        self._runtime_output = (
            runtime_output
            if runtime_output is not None
            else np.zeros(
                (1, 5, output_count if isinstance(output_count, int) else 4),
                dtype=np.float32,
            )
        )
        self.last_feed = None

    def get_modelmeta(self):
        return _ModelMeta(self._metadata)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feed):
        assert output_names == ["output0"]
        self.last_feed = feed
        return [self._runtime_output]


def _localizer(monkeypatch, session: _Session, **kwargs):
    captured = {}

    def create(path, *, providers):
        captured["path"] = path
        captured["providers"] = providers
        return session

    monkeypatch.setattr(module.ort, "InferenceSession", create)
    localizer = OpponentRacketBBoxLocalizer(
        "opponent.onnx",
        bbox_confidence_min=kwargs.get("bbox_confidence_min", 0.20),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        nms_iou=kwargs.get("nms_iou", 0.45),
    )
    return localizer, captured


def test_module_has_no_legacy_localizer_dependency():
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("racket_localizer", "RacketPose", "racket_pose", "keypoint"):
        assert forbidden not in source


def test_detect_candidates_maps_letterboxed_crop_to_native_pixels(monkeypatch):
    # Crop is 200x100: scale=1.6, top padding=80.  The first two boxes overlap
    # and NMS keeps only the 0.90 candidate.
    predictions = np.asarray(
        [
            [80.0, 128.0, 96.0, 64.0, 0.90],
            [80.0, 128.0, 89.6, 57.6, 0.80],
            [240.0, 144.0, 96.0, 64.0, 0.70],
            [20.0, 100.0, 10.0, 10.0, 0.10],
        ],
        dtype=np.float32,
    )
    session = _Session(runtime_output=predictions.T[None])
    localizer, captured = _localizer(monkeypatch, session)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    detections = localizer.detect_candidates(
        image,
        serial="DB0260414",
        image_origin_xy=(1000, 2000),
    )

    assert captured == {
        "path": "opponent.onnx",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    assert localizer.provider_info == {
        "opponent_racket_bbox": [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    }
    assert len(detections) == 2
    assert detections[0].serial == "DB0260414"
    assert detections[0].bbox_confidence == pytest.approx(0.90)
    assert detections[0].bbox_xyxy == pytest.approx((1020, 2010, 1080, 2050))
    assert detections[0].center_xy == pytest.approx((1050, 2030))
    assert detections[1].bbox_xyxy == pytest.approx((1120, 2020, 1180, 2060))
    tensor = session.last_feed["images"]
    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 100, 100] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metadata",
    [
        {"task": "pose", "names": "{0: 'racket_head'}"},
        {"task": "detect", "names": "{0: 'racket'}"},
        {"task": "detect", "names": "not a mapping"},
    ],
)
def test_model_metadata_is_fail_closed(monkeypatch, metadata):
    session = _Session(metadata=metadata)
    monkeypatch.setattr(
        module.ort,
        "InferenceSession",
        lambda path, *, providers: session,
    )
    with pytest.raises(ValueError):
        OpponentRacketBBoxLocalizer(
            "bad.onnx",
            bbox_confidence_min=0.05,
            providers=["CPUExecutionProvider"],
        )


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [
        ([1, 3, 640, 640], [1, 5, 4]),
        ([1, 3, 320, 320], [1, 6, 4]),
        ([1, 3, 320, 320], [1, 5, "anchors"]),
    ],
)
def test_model_tensor_contract_is_exact(monkeypatch, input_shape, output_shape):
    session = _Session(input_shape=input_shape, output_shape=output_shape)
    monkeypatch.setattr(
        module.ort,
        "InferenceSession",
        lambda path, *, providers: session,
    )
    with pytest.raises(ValueError):
        OpponentRacketBBoxLocalizer(
            "bad.onnx",
            bbox_confidence_min=0.05,
            providers=["CPUExecutionProvider"],
        )


def test_runtime_output_shape_is_fail_closed(monkeypatch):
    session = _Session(runtime_output=np.zeros((1, 5, 3), dtype=np.float32))
    localizer, _ = _localizer(monkeypatch, session)
    with pytest.raises(ValueError, match="runtime output"):
        localizer.detect_candidates(
            np.zeros((32, 32, 3), dtype=np.uint8),
            serial="camera",
        )


def test_invalid_detection_inputs_are_rejected(monkeypatch):
    localizer, _ = _localizer(monkeypatch, _Session())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8 BGR"):
        localizer.detect_candidates(image.astype(np.float32), serial="camera")
    with pytest.raises(ValueError, match="serial"):
        localizer.detect_candidates(image, serial="")
    with pytest.raises(ValueError, match="max_candidates"):
        localizer.detect_candidates(image, serial="camera", max_candidates=0)
