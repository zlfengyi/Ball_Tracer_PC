from __future__ import annotations

import os
import queue
from pathlib import Path

from src.ros2_support import ROS2_ROOT
from src.run_tracker import (
    CarLocJob,
    _submit_latest,
    _infer_engine_batch_from_model_path,
    _infer_model_input_size_from_model_path,
    _report_tool_env,
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


def test_report_tool_env_drops_ros2_pythonpath(monkeypatch):
    """post-run 报告工具必须拿到不含 ROS2 site-packages 的 PYTHONPATH：
    否则 venv 解释器 import numpy 会命中 conda 版、C 扩展加载失败，
    generate_curve3_html 的拍面yaw两列静默变「—」、annotate_video 直接挂。"""
    ros2_sp = str(ROS2_ROOT / "Lib" / "site-packages")
    keep = str(Path("D:/Ball_Tracer_PC"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([ros2_sp, keep]))

    env = _report_tool_env()

    assert env["PYTHONPATH"] == keep
    assert os.environ["PYTHONPATH"].startswith(ros2_sp)  # 父进程自己的不动


def test_report_tool_env_drops_pythonpath_key_when_only_ros2(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROS2_ROOT / "Lib" / "site-packages"))

    assert "PYTHONPATH" not in _report_tool_env()


def test_report_tool_env_keeps_env_without_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    assert "PYTHONPATH" not in _report_tool_env()


def test_car_submit_latest_returns_stale_on_full_queue():
    job_queue: queue.Queue[CarLocJob | None] = queue.Queue(maxsize=1)
    stale = CarLocJob(frame_idx=10, exposure_pc=1.0, elapsed_s=0.0, images={})
    latest = CarLocJob(frame_idx=12, exposure_pc=2.0, elapsed_s=0.1, images={})

    assert _submit_latest(job_queue, stale) is None
    evicted = _submit_latest(job_queue, latest)

    queued = job_queue.get_nowait()
    assert evicted is stale
    assert queued is latest
