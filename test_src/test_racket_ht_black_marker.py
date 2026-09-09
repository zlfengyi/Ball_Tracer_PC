from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from racket_ht_black_marker import (
    CameraModel,
    NothingToMeasure,
    _grid_panels,
    _measurement_context,
)


SERIALS = ["cam0", "cam1", "cam2", "cam3"]


def _write_inputs(tmp_path: Path, *, align_err: float = 0.01, tab_errors=None):
    tracker = tmp_path / "tracker.json"
    arm = tmp_path / "arm.json"
    rk = tmp_path / "rk.json"
    tables = tmp_path / "tables.json"
    tracker.write_text(
        json.dumps(
            {
                "config": {
                    "car_config_path": "vehicle_v04.json",
                    "video_frame_mapping_exact": True,
                    "video_output": {
                        "layout": "grid",
                        "grid_cols": 2,
                        "grid_rows": 2,
                        "serial_order": SERIALS,
                    },
                    "camera_settings": {
                        serial: {"exposure_us": 9000.0} for serial in SERIALS
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    arm.write_text(json.dumps({"car": "v04"}), encoding="utf-8")
    rk.write_text(json.dumps({"t0": 100.0}), encoding="utf-8")
    tables.write_text(
        json.dumps(
            {
                "align": {
                    "auto": {
                        "bias": 2.5,
                        "err": align_err,
                        "n": 60,
                        "flights": 4,
                        "requiredFlights": 3,
                        "windowSource": "bridge",
                    },
                    "timeMap": {"scale": 1, "bias": 2.5},
                },
                "arm_contract": {
                    "schema": "arm_final_ht/v4",
                    "rkT0": 100.0,
                    "zPhasePolicy": {
                        "maxAbsOffsetMs": 100,
                        "appliesTo": "all_pc_sampling",
                        "rkUse": "global_baseline",
                    },
                    "calibration": {"zOff": -0.1471},
                    "rows": [
                        {
                            "reportRow": 1,
                            "accepted": True,
                            "finalMismatch": False,
                            "finalHtRkAbs": 110.0,
                            "finalHtPcBaselineElapsed": 12.5,
                            "finalHtPcSampleElapsed": 12.525,
                            "zPhase": {"usable": True, "deltaS": 0.025},
                        },
                        {
                            "reportRow": 2,
                            "accepted": False,
                            "finalMismatch": False,
                            "finalHtRkAbs": None,
                            "finalHtPcBaselineElapsed": None,
                        },
                    ],
                },
                "script_error": None,
                "tab_errors": tab_errors or [],
            }
        ),
        encoding="utf-8",
    )
    return tracker, arm, rk, tables


def test_measurement_context_uses_report_final_ht_and_exposure_center(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    _, _, _, serials, exposure_offset, z_offset, targets = (
        _measurement_context(*paths)
    )
    assert serials == SERIALS
    assert exposure_offset == pytest.approx(0.0045)
    assert z_offset == pytest.approx(-0.1471)
    assert targets == [{
        "report_row": 1,
        "final_ht_pc_elapsed_s": 12.525,
        "rk_to_pc_bias_s": 2.525,
    }]


def test_measurement_context_requires_usable_throw_phase(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    row = payload["arm_contract"]["rows"][0]
    row["zPhase"] = {"usable": False, "deltaS": None}
    row["finalHtPcSampleElapsed"] = None
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="usable zPhase"):
        _measurement_context(*paths)


def test_measurement_context_rejects_inconsistent_pc_sample(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["rows"][0]["finalHtPcSampleElapsed"] = 12.526
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline plus zPhase"):
        _measurement_context(*paths)


def test_measurement_context_rejects_old_visual_contract(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["schema"] = "arm_final_ht/v3"
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="arm_final_ht/v4"):
        _measurement_context(*paths)


def test_measurement_context_skips_a_session_without_throws(tmp_path: Path):
    """0 抛的场次（RK 没发过 /predict_hit_pos）只是没东西可测，不是报告不可信。

    报告页对空场次也发 contract（rows=[]），所以这里能和"合同缺失=报告代码太旧"分开：
    前者跳过让后处理继续，后者必须报错。对齐质量此时不参与判定。
    """
    paths = _write_inputs(tmp_path, align_err=0.5)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["rows"] = []
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NothingToMeasure):
        _measurement_context(*paths)


def test_measurement_context_still_rejects_a_missing_contract(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"] = None
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="arm_final_ht/v4"):
        _measurement_context(*paths)


@pytest.mark.parametrize(
    ("align_err", "tab_errors"),
    [(0.081, None), (0.01, ["sw(5): failed"])],
)
def test_measurement_context_rejects_untrusted_report(
    tmp_path: Path, align_err: float, tab_errors
):
    paths = _write_inputs(tmp_path, align_err=align_err, tab_errors=tab_errors)
    with pytest.raises(ValueError):
        _measurement_context(*paths)


def _camera(image_size: tuple[int, int]) -> CameraModel:
    return CameraModel(
        serial="cam",
        K=np.eye(3),
        D=np.zeros(5),
        R=np.eye(3),
        t=np.zeros(3),
        rvec=np.zeros(3),
        P=np.zeros((3, 4)),
        image_size=image_size,
    )


@pytest.mark.parametrize("frame_size", [(2048, 1536), (2048, 1304)])
def test_grid_panels_restore_the_captured_frame_not_the_calibrated_height(frame_size):
    """A bottom-cropped sensor ROI must not be stretched back to the calib height.

    camera_18.json roi_height shortened the capture to 1304 rows while the
    calibration still records 1536; restoring panels to the calibrated size
    scaled every row by 1.178 and shifted projected anchors ~0.18*v px.
    """
    width, height = frame_size
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    # one bright pixel per quadrant, at a known place inside each panel
    for index in range(4):
        x = (index % 2) * (width // 2) + 100
        y = (index // 2) * (height // 2) + 200
        grid[y, x] = (0, 0, 255)
    cameras = {serial: _camera((2048, 1536)) for serial in SERIALS}
    panels = _grid_panels(grid, SERIALS, cameras)
    for serial in SERIALS:
        panel = panels[serial]
        assert panel.shape[:2] == (height, width)
        ys, xs = np.nonzero(panel[:, :, 2])
        # half-scale panel restored by exactly 2x: (100, 200) -> (200, 400)
        assert int(round(xs.mean())) == pytest.approx(200, abs=2)
        assert int(round(ys.mean())) == pytest.approx(400, abs=2)


def test_grid_panels_reject_a_video_that_is_not_a_roi_of_the_calibrated_frame():
    grid = np.zeros((1304, 1600, 3), dtype=np.uint8)
    cameras = {serial: _camera((2048, 1536)) for serial in SERIALS}
    with pytest.raises(ValueError, match="bottom-cropped ROI"):
        _grid_panels(grid, SERIALS, cameras)
