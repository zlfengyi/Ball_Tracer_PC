from __future__ import annotations

import json
from pathlib import Path

import pytest

from racket_ht_black_marker import _measurement_context


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
