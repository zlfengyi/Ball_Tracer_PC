from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.car_localizer import CarLocalizer

VEHICLE_V04_CONFIG = (
    Path(__file__).resolve().parent.parent / "src" / "config" / "vehicle_v04.json"
)
# 用户 2026-08-15 复测的 v0.4 车实测中心（cm→m）。id0 左上/左前、id1 右下/右后。
# ⚠ 早间建档时这两组值被对调过（id0 当成右后），同批实拍角点重投影 44px、yaw 翻 180°、
# 车心偏 (-13.5, -32.1) cm；改回来后 2.07px。这个测试就是防它再被换回去。
EXPECTED_CENTER_M = {
    0: np.array([-0.4355, 0.121, 0.3165]),
    1: np.array([0.3185, -0.429, 0.2875]),
}


def test_shipped_v04_layout_matches_measured_centers():
    localizer = CarLocalizer(vehicle_config_path=str(VEHICLE_V04_CONFIG))

    assert localizer.tag_ids == [0, 1]
    layout = localizer.tag_layout_m
    for tag_id, expected in EXPECTED_CENTER_M.items():
        np.testing.assert_allclose(layout[tag_id], expected, atol=1e-9)


def test_v04_tag0_is_higher_than_tag1():
    """z 是唯一能判 id 归属的量。

    两块 tag 对调 == 绕两者中点转 180°，x/y 的间距完全不变（0.9333m 两种归属都成立），
    只有 z 不参与这个平面旋转。实测 tag0 世界 z=0.3168 > tag1 z=0.2838，差 +33mm。
    """
    layout = CarLocalizer(vehicle_config_path=str(VEHICLE_V04_CONFIG)).tag_layout_m
    assert layout[0][2] > layout[1][2]
    assert 0.020 < layout[0][2] - layout[1][2] < 0.045


def test_v04_tags_are_both_upright_facing_rear():
    """贴法由固定相机三角化四角点实测：两块印面『上』都朝世界 +z ⇒ 都是正贴。"""
    cfg = json.loads(VEHICLE_V04_CONFIG.read_text(encoding="utf-8"))
    tags = cfg["vehicle_reference"]["apriltags"]
    for key in ("0", "1"):
        assert tags[key]["inplane_rotation_deg"] == 0.0
        assert tags[key]["face_azimuth_deg"] == -90.0
    assert cfg["vehicle_reference"]["layout_status"] == "measured"
