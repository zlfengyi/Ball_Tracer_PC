from __future__ import annotations

import json

import pytest

from src.arm_poe import ArmPoePositionModel


def _write_config(path, *, z_offset_base_mm: float | None) -> None:
    payload = {
        "poe_model_position_only": {
            "joint_count": 1,
            "joint_angle_offsets_rad": [0.0],
            "home_point_base_mm": [100.0, 0.0, 0.0],
            "space_axes_base": [
                {
                    "joint_index": 1,
                    "omega": [0.0, 0.0, 1.0],
                    "q_mm": [0.0, 0.0, 0.0],
                    "v_mm": [0.0, 0.0, 0.0],
                }
            ],
        },
        "T_base_in_world": {
            "R": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "t_mm": [0.0, 0.0, 50.0],
        },
    }
    if z_offset_base_mm is not None:
        payload["z_axis_correction"] = {
            "enabled": True,
            "offset_base_mm": float(z_offset_base_mm),
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_forward_without_z_correction_keeps_original_geometry(tmp_path):
    cfg = tmp_path / "poe.json"
    _write_config(cfg, z_offset_base_mm=None)

    model = ArmPoePositionModel(config_path=cfg)
    fk = model.forward([0.0])

    assert model.z_offset_base_mm == pytest.approx(0.0)
    assert fk.point_base_mm.tolist() == pytest.approx([100.0, 0.0, 0.0])
    assert fk.point_world_mm.tolist() == pytest.approx([100.0, 0.0, 50.0])


def test_forward_applies_configured_base_z_correction(tmp_path):
    cfg = tmp_path / "poe_zcorr.json"
    _write_config(cfg, z_offset_base_mm=-40.0)

    model = ArmPoePositionModel(config_path=cfg)
    fk = model.forward([0.0])

    assert model.z_offset_base_mm == pytest.approx(-40.0)
    assert fk.point_base_mm.tolist() == pytest.approx([100.0, 0.0, -40.0])
    assert fk.point_world_mm.tolist() == pytest.approx([100.0, 0.0, 10.0])
