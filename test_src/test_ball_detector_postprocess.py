from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _PROJECT_ROOT / "src" / "ball_detector.py"
_SPEC = importlib.util.spec_from_file_location(
    "ball_detector_under_test", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load ball_detector module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BallDetection = _MODULE.BallDetection
BallDetector = _MODULE.BallDetector


def _det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    label: str = "tennis_ball",
) -> BallDetection:
    return BallDetection(
        x=(x1 + x2) / 2.0,
        y=(y1 + y2) / 2.0,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        label=label,
    )


class _BulkOnlyTensor:
    def __init__(self, values: list) -> None:
        self._values = np.asarray(values, dtype=np.float32)
        self.cpu_calls = 0

    def __getitem__(self, index):
        raise AssertionError("box tensors must be transferred in bulk")

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        return self._values


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _BulkOnlyTensor([
            [100.0, 200.0, 120.0, 224.0],
            [300.0, 400.0, 330.0, 430.0],
        ])
        self.conf = _BulkOnlyTensor([0.75, 0.9])

    def __len__(self) -> int:
        return 2


def test_parse_boxes_transfers_tensor_columns_in_bulk() -> None:
    boxes = _FakeBoxes()
    result = type("FakeResult", (), {"boxes": boxes})()

    detections = BallDetector._parse_boxes(result)

    assert boxes.xyxy.cpu_calls == 1
    assert boxes.conf.cpu_calls == 1
    assert np.allclose([det.confidence for det in detections], [0.9, 0.75])
    assert detections[0].x == 315.0
    assert detections[0].y == 415.0


def test_postprocess_removes_near_identical_duplicates() -> None:
    detections = [
        _det(100, 100, 130, 130, 0.91),
        _det(100, 100, 130, 130, 0.42),
        _det(300, 300, 332, 332, 0.73),
    ]

    processed = BallDetector.postprocess_detections(
        detections,
        duplicate_iou_threshold=0.95,
        max_box_aspect_ratio=1.2,
    )

    assert len(processed) == 2
    assert processed[0].confidence == 0.91
    assert processed[1].confidence == 0.73


def test_postprocess_filters_rectangles_over_twenty_percent() -> None:
    detections = [
        _det(0, 0, 30, 30, 0.95),
        _det(50, 50, 80, 87, 0.88),
    ]

    processed = BallDetector.postprocess_detections(
        detections,
        duplicate_iou_threshold=0.95,
        max_box_aspect_ratio=1.2,
    )

    assert len(processed) == 1
    assert processed[0].width == 30
    assert processed[0].height == 30


def test_postprocess_keeps_box_at_exact_twenty_percent_ratio() -> None:
    detections = [_det(0, 0, 30, 36, 0.88)]

    processed = BallDetector.postprocess_detections(
        detections,
        duplicate_iou_threshold=0.95,
        max_box_aspect_ratio=1.2,
    )

    assert len(processed) == 1
    assert math.isclose(processed[0].aspect_ratio, 1.2, rel_tol=1e-9)


def test_shape_gate_stats_count_rejected_tennis_balls() -> None:
    """形状门要能事后回答「这场漏检是不是拖影把球框拉长了」。"""
    BallDetector.reset_shape_gate_stats()

    detections = [
        _det(0, 0, 30, 30, 0.95),            # 圆，留
        _det(50, 50, 80, 110, 0.88),         # 长宽比 2.0，拦
        _det(200, 200, 230, 245, 0.70),      # 长宽比 1.5，拦
        _det(400, 400, 430, 490, 0.60, label="stationary_object"),  # 非网球，不计
    ]

    processed = BallDetector.postprocess_detections(
        detections,
        duplicate_iou_threshold=0.95,
        max_box_aspect_ratio=1.2,
    )

    assert len(processed) == 1
    stats = BallDetector.shape_gate_stats
    assert stats["ball_kept"] == 1
    assert stats["ball_rejected"] == 2
    assert math.isclose(stats["rejected_aspect_max"], 2.0, rel_tol=1e-9)
    assert math.isclose(stats["rejected_aspect_sum"], 3.5, rel_tol=1e-9)


def test_shape_gate_stats_reset_clears_counters() -> None:
    BallDetector.reset_shape_gate_stats()
    BallDetector.postprocess_detections(
        [_det(0, 0, 30, 90, 0.9)],
        duplicate_iou_threshold=0.95,
        max_box_aspect_ratio=1.2,
    )
    assert BallDetector.shape_gate_stats["ball_rejected"] == 1

    BallDetector.reset_shape_gate_stats()
    assert BallDetector.shape_gate_stats["ball_rejected"] == 0
    assert BallDetector.shape_gate_stats["rejected_aspect_max"] == 0.0
