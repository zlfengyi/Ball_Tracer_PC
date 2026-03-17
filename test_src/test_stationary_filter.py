from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _PROJECT_ROOT / "src" / "stationary_filter.py"
_SPEC = importlib.util.spec_from_file_location(
    "stationary_filter_under_test", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load stationary_filter module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

StationaryObjectFilter = _MODULE.StationaryObjectFilter
STATIONARY_OBJECT_LABEL = _MODULE.STATIONARY_OBJECT_LABEL
TENNIS_BALL_LABEL = _MODULE.TENNIS_BALL_LABEL


@dataclass
class FakeDetection:
    x: float
    y: float
    label: str = TENNIS_BALL_LABEL


def test_marks_detection_stationary_on_sixth_hit_within_radius() -> None:
    detector_filter = StationaryObjectFilter(
        window_s=15.0,
        radius_px=2.0,
        min_occurrences=6,
    )

    labels = []
    for t, (x, y) in enumerate(
        [(100.0, 100.0), (101.0, 100.0), (100.0, 101.0), (99.0, 100.0), (100.0, 99.0), (100.0, 100.0)]
    ):
        det = FakeDetection(x=x, y=y)
        labels.append(detector_filter.classify("cam_a", [det], float(t))[0].label)

    assert labels[:5] == [TENNIS_BALL_LABEL] * 5
    assert labels[5] == STATIONARY_OBJECT_LABEL


def test_expired_history_no_longer_marks_stationary() -> None:
    detector_filter = StationaryObjectFilter(
        window_s=15.0,
        radius_px=2.0,
        min_occurrences=6,
    )

    for t in range(5):
        det = FakeDetection(x=200.0, y=300.0)
        assert detector_filter.classify("cam_a", [det], float(t))[0].label == TENNIS_BALL_LABEL

    late_det = FakeDetection(x=200.0, y=300.0)
    classified = detector_filter.classify("cam_a", [late_det], 20.0)[0]
    assert classified.label == TENNIS_BALL_LABEL


def test_history_is_kept_per_camera() -> None:
    detector_filter = StationaryObjectFilter(
        window_s=15.0,
        radius_px=2.0,
        min_occurrences=6,
    )

    for t in range(5):
        det = FakeDetection(x=50.0, y=60.0)
        detector_filter.classify("cam_a", [det], float(t))

    cam_a_det = FakeDetection(x=50.0, y=60.0)
    cam_b_det = FakeDetection(x=50.0, y=60.0)

    assert detector_filter.classify("cam_a", [cam_a_det], 5.0)[0].label == STATIONARY_OBJECT_LABEL
    assert detector_filter.classify("cam_b", [cam_b_det], 5.0)[0].label == TENNIS_BALL_LABEL
