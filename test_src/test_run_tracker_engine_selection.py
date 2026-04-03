from __future__ import annotations

from pathlib import Path

from src.run_tracker import (
    _infer_engine_batch_from_model_path,
    _infer_model_input_size_from_model_path,
    _resolve_engine_batch,
    _select_detector_model_for_active_cams,
)


def test_infer_engine_batch_from_model_path():
    path = Path("yolo_model/tennis_yolo26_v2_20260203_b3_640.engine")
    assert _infer_engine_batch_from_model_path(path) == 3


def test_infer_model_input_size_from_model_path():
    path = Path("yolo_model/tennis_yolo26_v2_20260203_b3_640.engine")
    assert _infer_model_input_size_from_model_path(path) == 640


def test_select_detector_model_for_active_cams_prefers_matching_batch(tmp_path):
    current = tmp_path / "tennis_yolo26_v2_20260203_b4_640.engine"
    current.write_bytes(b"")
    matched = tmp_path / "tennis_yolo26_v2_20260203_b3_640.engine"
    matched.write_bytes(b"")

    selected = _select_detector_model_for_active_cams(current, 3)

    assert selected == matched


def test_select_detector_model_for_active_cams_keeps_current_when_no_match(tmp_path):
    current = tmp_path / "tennis_yolo26_v2_20260203_b4_640.engine"
    current.write_bytes(b"")

    selected = _select_detector_model_for_active_cams(current, 3)

    assert selected == current


def test_select_detector_model_for_active_cams_prefers_matching_batch_and_input_size(tmp_path):
    current = tmp_path / "tennis_yolo26_v2_20260203_b4_640.engine"
    current.write_bytes(b"")
    matched = tmp_path / "tennis_yolo26_v2_20260203_b3_512.engine"
    matched.write_bytes(b"")
    fallback = tmp_path / "tennis_yolo26_v2_20260203_b3_640.engine"
    fallback.write_bytes(b"")

    selected = _select_detector_model_for_active_cams(
        current,
        3,
        target_input_size=512,
    )

    assert selected == matched


def test_select_detector_model_for_active_cams_falls_back_to_nearest_input_size(tmp_path):
    current = tmp_path / "tennis_yolo26_v2_20260203_b4_640.engine"
    current.write_bytes(b"")
    fallback = tmp_path / "tennis_yolo26_v2_20260203_b3_640.engine"
    fallback.write_bytes(b"")

    selected = _select_detector_model_for_active_cams(
        current,
        3,
        target_input_size=512,
    )

    assert selected == fallback


class _FakeDetector:
    def __init__(self, model_path: Path, accepted_batches: set[int]):
        self.model_path = model_path
        self.accepted_batches = set(accepted_batches)
        self.calls: list[int] = []

    def detect_batch(self, images):
        batch = len(images)
        self.calls.append(batch)
        if batch not in self.accepted_batches:
            raise RuntimeError(f"unsupported batch={batch}")
        return [None] * batch


def test_resolve_engine_batch_prefers_fixed_engine_batch_over_runtime_chunking():
    detector = _FakeDetector(
        Path("yolo_model/tennis_yolo26_v2_20260203_b3_640.engine"),
        accepted_batches={3, 4},
    )

    engine_batch = _resolve_engine_batch(
        detector,
        warmup_img=object(),
        n_ball_detect_cams=3,
        n_cams=4,
    )

    assert engine_batch == 3
    assert detector.calls == [3]
