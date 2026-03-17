from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


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
