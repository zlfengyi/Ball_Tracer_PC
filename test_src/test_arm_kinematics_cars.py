# -*- coding: utf-8 -*-
"""臂 FK 的车型合同：v0.3 与 v0.4 是两台不同的臂，报告必须按本场车型算 TCP。

黄金向量取自臂端自己导出的资产（arm_controller `tools/export_cpp_assets.py` →
`cpp/arm_controller_cpp/assets/<car>/test_vectors.json` 的 ik_hit / face_lookup 组）：
每条是「臂端 IK 解出的位形 q」+「它本该把拍心放到的 (x, z)」。我们这边的 FK 独立算一遍，
必须落回同一个点——这才是跨实现对账，而不是拿本文件的常数自证。

⚠ 这组测试守的是 0816 那次事故：报告端 FK 写死 v0.3 链，v0.4 场次整场 TCP 偏
(−5.4, −8.6)cm、FK 拍速低报 5%，而拍面 yaw/pitch 两车逐拍恒等——页面上没有任何征兆。
所以既要验「各车自己对」，也要验「拿错车一定偏得看得见」，还要验「车型推断/缺省不猜」。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import extract_arm_bag as eab  # noqa: E402

# 臂端导出的黄金向量（逐条抄进来，免得测试依赖隔壁 tennis-man checkout 的存在与分支）。
# 来源：v03 = origin/c++version:cpp/arm_controller_cpp/assets/test_vectors.json
#      v04 = arm_controller-unify@0e1104f:cpp/arm_controller_cpp/assets/v04/test_vectors.json
# 每条 = (x, z, q6)：臂端 face_lookup/ik_hit 声称该位形把拍心放在 (x, y≈0, z)。
GOLDEN = {
    "v03": [
        (1.0, 1.15,
         [-1.085469451873422e-07, 0.25659757569089414, 1.4400962568581046,
          1.6966939525489886, 1.8854693266225223e-07, 0.31415906535899574]),
        (1.0, 1.136,
         [-0.00939710960437435, 0.251369198185597, 1.4556222108187986,
          1.687440562872557, 0.015974784544359796, 0.3243621562537498]),
        (1.0, 1.22,
         [0.0, 0.2548373386138545, 1.2085464803674686, 1.463383818981323, 0.0, 0.0]),
        (0.8, 1.0,
         [0.0, -0.021913096108918673, 2.3326005193252524, 2.3106874232163337, 0.0, 0.0]),
    ],
    "v04": [
        (1.0, 1.15,
         [-0.051970737187912314, 0.2909985676928369, 1.553974563237923,
          1.7222877928352092, 0.09118969644015855, 0.3053721642437658]),
        (1.0, 1.136,
         [-0.062136554915039145, 0.2899976768804198, 1.5641746368566103,
          1.7115920079545708, 0.1090724415160484, 0.31239731944623406]),
        (1.0, 1.22,
         [0.0, 0.32228456810630046, 1.510894547632402,
          1.8331791157387016, 0.0, 0.0]),
        (0.8, 1.0,
         [0.0, 0.3661559666399792, 2.534715766627718,
          2.9008717332676968, 0.0, 0.0]),
    ],
}


def test_v04_tcp_distance_matches_current_calibration():
    assert eab.CAR_MODELS["v04"].tcp_distance == pytest.approx(0.548946367, abs=1e-12)


@pytest.mark.parametrize("car", sorted(GOLDEN))
def test_fk_matches_arm_side_golden_vectors(car):
    """各车 FK 落回臂端 IK 声称的击球点（含 y≈0：拍心该落在击球平面上）。"""
    pytest.importorskip("numpy")
    eab.use_car(car)
    for x, z, q in GOLDEN[car]:
        tcp = eab.fk(q)["tcp"]
        assert tcp[0] == pytest.approx(x, abs=1e-3), f"{car} x @({x},{z})"
        assert tcp[2] == pytest.approx(z, abs=1e-3), f"{car} z @({x},{z})"
        assert abs(tcp[1]) < 0.01, f"{car} 拍心不在击球平面上 @({x},{z})"


def test_wrong_car_is_off_by_centimetres_not_millimetres():
    """拿错车不是小数点级误差：当前 v0.4 黄金位形用 v0.3 链算会低约 12cm。"""
    pytest.importorskip("numpy")
    for x, z, q in GOLDEN["v04"]:
        good = eab.fk(q, car="v04")["tcp"]
        wrong = eab.fk(q, car="v03")["tcp"]
        assert wrong[2] - good[2] < -0.05, "v0.3 链算 v0.4 位形应显著偏低"
        assert math.hypot(wrong[0] - good[0], wrong[2] - good[2]) > 0.08


def test_face_angles_are_identical_across_cars():
    """拍面法向在两车下逐条恒等——所以角度列看不出选错车，位置列才是判据。
    （0816 实测 11 拍逐拍相同；这条固化那个观察，免得有人拿角度列去"验证"车型。）"""
    pytest.importorskip("numpy")
    for _, _, q in GOLDEN["v04"] + GOLDEN["v03"]:
        n3 = eab.fk(q, car="v03")["face_normal"]
        n4 = eab.fk(q, car="v04")["face_normal"]
        assert all(abs(a - b) < 1e-6 for a, b in zip(n3, n4))


def test_no_default_car():
    """一次都没选过车型就调 fk() 必须抛——静默用另一台车的链是这套代码最贵的错误。"""
    pytest.importorskip("numpy")
    eab._ACTIVE = None
    with pytest.raises(RuntimeError, match="还没选车型"):
        eab.fk([0.0] * 6)
    eab.use_car("v04")  # 复位，免得影响同进程其它用例


def test_car_for_tracker_json_reads_layout_config(tmp_path):
    """车型来源 = tracker JSON 里 run_tracker 按 --car 落下的 car_config_path。
    0815 之前没这个字段（当时只有 v0.3 一台车）→ v03；认不出的布局文件必须抛，不能猜。"""
    def write(name, config):
        p = tmp_path / name
        p.write_text(json.dumps({"config": config}), encoding="utf-8")
        return p

    car, why = eab.car_for_tracker_json(
        write("v04.json", {"car_config_path": r"D:\Ball_Tracer_PC\src\config\vehicle_v04.json"}))
    assert car == "v04" and "vehicle_v04.json" in why
    car, why = eab.car_for_tracker_json(
        write("v03.json", {"car_config_path": "/x/arm_poe_racket_center.json"}))
    assert car == "v03"
    car, why = eab.car_for_tracker_json(write("old.json", {"fps": 29.0}))
    assert car == "v03" and "0815" in why
    with pytest.raises(RuntimeError, match="认不出车型"):
        eab.car_for_tracker_json(write("new.json", {"car_config_path": "vehicle_v05.json"}))


def test_car_for_session_precedence(tmp_path):
    """显式 > arm JSON 自述 > tracker JSON；三条都没有就抛。"""
    tracker = tmp_path / "t.json"
    tracker.write_text(json.dumps({"config": {"car_config_path": "vehicle_v04.json"}}),
                       encoding="utf-8")
    assert eab.car_for_session({"car": "v03"}, tracker, explicit="v04")[0] == "v04"
    assert eab.car_for_session({"car": "v03"}, tracker)[0] == "v03"
    assert eab.car_for_session({}, tracker)[0] == "v04"
    with pytest.raises(RuntimeError, match="推不出车型"):
        eab.car_for_session({}, None)


def test_recompute_tcp_rewrites_derived_field():
    """arm JSON 的 tcp 只是派生量：换车重算即可，不必重跑 rosbag 提取。
    关节残缺的行置 None（报告列显示 —），不能留着别的车算出来的旧值。"""
    pytest.importorskip("numpy")
    q = [0.1, 0.2, 1.4, 1.6, 0.0, 0.3]
    rows = [{"position": q, "tcp": [9.0, 9.0, 9.0]},
            {"position": [0.1, None, 1.4, 1.6, 0.0, 0.3], "tcp": [9.0, 9.0, 9.0]}]
    eab.use_car("v04")
    assert eab.recompute_tcp(rows) == 2
    assert rows[0]["tcp"] == [round(float(v), 4) for v in eab.fk(q)["tcp"]]
    assert rows[1]["tcp"] is None


def test_extractor_records_car_in_output():
    """出 arm JSON 必须把车型写进去（car/car_source/fk_source），报告端才不用再猜。"""
    source = (SRC_DIR / "extract_arm_bag.py").read_text(encoding="utf-8")
    assert '"car": car,' in source
    assert '"car_source": car_source,' in source
    assert '"fk_source": f"extract_arm_bag.fk({car})",' in source
    assert '"--car"' in source and "choices=sorted(CAR_MODELS)" in source


def test_run_tracker_passes_car_to_extractor():
    """run_tracker 把启动时选的车型透给 extract_arm_bag（--car-config 直给时留空、由 JSON 推）。"""
    source = (SRC_DIR.parent / "src" / "run_tracker.py").read_text(encoding="utf-8")
    assert 'arm_command.extend(["--car", car])' in source
    assert "car=args.car," in source
