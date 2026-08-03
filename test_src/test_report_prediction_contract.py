# -*- coding: utf-8 -*-
"""回归测试：RK≈300ms消息与机械臂最后accepted必须是两套独立合同。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
RK_EXTRACTOR = Path(__file__).resolve().parent / "extract_rk_tracking_bag.py"
NODE = shutil.which("node")


def _core(begin: str, end: str) -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        rf"// \[\[{re.escape(begin)}\]\].*?\n(.*)// \[\[{re.escape(end)}\]\]",
        text,
        re.S,
    )
    assert match, f"缺少 {begin}/{end} 标记"
    return match.group(1)


def _run_node(tmp_path: Path, body: str):
    script = tmp_path / "prediction_contract_harness.js"
    script.write_text(body, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_rk300_fields_come_from_one_s1_message(tmp_path):
    core = _core("rk300-contract-core-begin", "rk300-contract-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        "const RK={pred:{t:[0.9,1.0,1.1,1.2],y:{"
        "ht_rel:[1.2,1.6,1.405,1.49],y:[4.2,3.6,99,98],x:[50,51,52,53],"
        "rel_x:[99,10,20,30],rel_y:[99,11,21,31],rel_z:[199,110,120,130],"
        "car_pred_x:[900,100,200,300],car_pred_y:[901,101,201,301]}}};\n"
        "const rkPredStage=[0,0,1,1]; const rkPredNFit=[2,4,5,6];\n"
        f"{core}\n"
        "console.log(JSON.stringify(rkThrows[0]));\n"
    )
    throw = _run_node(tmp_path, harness)

    assert throw["ref300T"] == pytest.approx(1.1)
    assert throw["ref300Ht"] == pytest.approx(1.405)
    assert throw["ref300Lead"] == pytest.approx(0.305)
    assert throw["ref300X"] == 20
    assert throw["ref300Y"] == 21
    assert throw["ref300Z"] == 120
    assert throw["ref300Xw"] == 52
    assert throw["ref300CarX"] == 200
    assert throw["ref300CarY"] == 201
    assert throw["ref300NFit"] == 5
    assert throw["ref300Idx"] == 2
    assert throw["lastS0T"] == pytest.approx(1.0)
    assert throw["lastS0Y"] == pytest.approx(3.6)
    assert throw["lastS0Idx"] == 1


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_stage1_is_recomputed_at_same_throw_rk_stage0_y(tmp_path):
    core = _core("pc-s1-rk-s0-core-begin", "pc-s1-rk-s0-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const cfg={k_drag:0}; const relTime=t=>t; const rkToPc=t=>t+10;\n"
        "const rkThrows=["
        "{firstT:0,lastT:2,lastS0T:1,lastS0Y:3.6,hasS1:true},"
        "{firstT:20,lastT:22,lastS0T:21,lastS0Y:6.5,hasS1:true}];\n"
        f"{core}\n"
        "const rows=["
        "{x:1,y:4,z:1.2,vx:.5,vy:-2,vz:1,stage:1,ct:10.5,ht:11},"
        "{x:10,y:7,z:2,vx:-1,vy:-1,vz:2,stage:1,ct:30.5,ht:31},"
        "{x:0,y:1,z:1,vx:0,vy:-1,vz:0,stage:1,ct:50,ht:51}];\n"
        "console.log(JSON.stringify({"
        "hits:rows.map(pcS1AtRkS0Y).filter(Boolean),"
        "belowGround:pcS1AtWorldY({x:1,y:4,z:1.2,vx:.5,vy:-2,vz:1,stage:1,ct:10.5,ht:11},6)}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert len(result["hits"]) == 2
    first, second = result["hits"]
    assert (first["x"], first["y"], first["z"], first["ht"]) == pytest.approx(
        (1.1, 3.6, 1.204, 11.2), abs=1e-6
    )
    assert (second["x"], second["y"], second["z"], second["ht"]) == pytest.approx(
        (9.5, 6.5, 1.775, 31.5), abs=1e-6
    )
    assert first["ct"] == 10.5 and second["ct"] == 30.5
    assert first["stage"] == second["stage"] == 1
    assert result["belowGround"] is None


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_accepted_is_matched_by_source_ct_only(tmp_path):
    core = _core("accepted-match-core-begin", "accepted-match-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const reportThrows=["
        "{firstT:0,lastT:1,ref300T:0.5},"
        "{firstT:2,lastT:3,ref300T:2.5}];\n"
        f"{core}\n"
        "console.log(JSON.stringify({matched:reportThrows.indexOf(matchThrowByAcceptedCt(2.2)),"
        "unmatched:matchThrowByAcceptedCt(1.5)}));\n"
    )
    result = _run_node(tmp_path, harness)
    assert result == {"matched": 1, "unmatched": None}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_accepted_raw_message_match_rejects_adjacent_prediction(tmp_path):
    core = _core(
        "arm-prediction-match-core-begin",
        "arm-prediction-match-core-end",
    )
    harness = (
        f"{core}\n"
        "const acceptT=10.0, acceptX=1.0375, acceptZ=1.3362, duration=0.3216;\n"
        "const actual={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216};\n"
        "const adjacent={rel_x:1.0376,rel_z:1.5058,relSrc:'target',ht:10.3221};\n"
        "// 0802 起臂控在 duration 内补偿舵机滞后 τ1≈9.2ms：ht 差 ~9ms 仍必须回配成功\n"
        "const servoLag={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216+0.0092};\n"
        "const crossThrow={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216+0.5};\n"
        "console.log(JSON.stringify({"
        "actual:armPredictionMatchesAccepted(actual,acceptT,acceptX,acceptZ,duration),"
        "adjacent:armPredictionMatchesAccepted(adjacent,acceptT,acceptX,acceptZ,duration),"
        "servoLag:armPredictionMatchesAccepted(servoLag,acceptT,acceptX,acceptZ,duration),"
        "crossThrow:armPredictionMatchesAccepted(crossThrow,acceptT,acceptX,acceptZ,duration)}));\n"
    )
    result = _run_node(tmp_path, harness)
    assert result == {
        "actual": True,
        "adjacent": False,
        "servoLag": True,
        "crossThrow": False,
    }


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_bot_run_end_is_normal_one_to_one_and_uses_target_minus_actual(tmp_path):
    core = _core("bot-run-end-core-begin", "bot-run-end-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        "const duplicate={ht:2.06}, first={ht:2.005}, second={ht:5.005}, stale={ht:8.0};\n"
        "const reportThrows=[duplicate,first,second,stale];\n"
        "const RK={bot:{t:[1.9,2.0,2.01,4.9,5.0,5.01,7.9,8.0,8.01],y:{"
        "phase:['RUN','RUN','BRAKE_IN_SWING','RUN','RUN','BRAKE_IN_SWING','RUN','RUN','BRAKE_AFTER_SWING'],"
        "x:[0,1,1,0,3,3,0,5,5],y:[0,2,2,0,4,4,0,6,6],"
        "target_x:[0,1.1,null,0,3.05,null,0,5.1,null],"
        "target_y:[0,1.8,null,0,4.07,null,0,6.1,null]}}};\n"
        f"{core}\n"
        "const a=botRunEndForThrow(first), b=botRunEndForThrow(second);\n"
        "console.log(JSON.stringify({duplicate:botRunEndForThrow(duplicate),"
        "first:{t:a.t,dx:(a.tx-a.x)*100,dy:(a.ty-a.y)*100},"
        "second:{t:b.t,dx:(b.tx-b.x)*100,dy:(b.ty-b.y)*100},"
        "stale:botRunEndForThrow(stale)}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert result["duplicate"] is None
    assert result["first"] == pytest.approx({"t": 2.0, "dx": 10.0, "dy": -20.0})
    assert result["second"] == pytest.approx({"t": 5.0, "dx": 5.0, "dy": 7.0})
    assert result["stale"] is None


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_ht_true_intersects_ball_fit_with_actual_car_and_rejects_pollution(tmp_path):
    """HT真实：S1 期球世界y拟合（3σ 剔跳变、z 门槛）× 车实际y挥拍前外推线 求交。"""
    core = _core("ht-alls1-core-begin", "ht-alls1-core-end")
    # 球：y(t)=1.0−3.5·(t−2.0)，t∈[1.56,1.95] 共 14 点；混入 1 个 +0.5m 跳变污染点
    # （t=1.80）与 1 个 z=0.05 贴地点（都必须被剔除/过滤）。
    # 车：y(t)=0.9+0.5·(t−2.0)（向球移动），挥拍塌陷伪迹点在 t≥1.973（窗外，必须无影响）。
    # 交点：1.0−3.5u = 0.9+0.5u → u=+0.025 → tRel=2.025。
    ts_list, ys_list, zs_list = [], [], []
    for i in range(14):
        t = 1.56 + 0.03 * i
        y = 1.0 - 3.5 * (t - 2.0)
        if abs(t - 1.80) < 1e-9:
            y += 0.5
        ts_list.append(round(t, 5))
        ys_list.append(round(y, 6))
        zs_list.append(0.4)
    ts_list.append(1.70)
    ys_list.append(99.0)
    zs_list.append(0.05)
    bt_list, by_list = [], []
    for i in range(20):
        t = 1.80 + 0.01 * i
        y = 0.9 + 0.5 * (t - 2.0)
        if t >= 1.973:
            y -= 0.05  # 挥拍塌陷伪迹
        bt_list.append(round(t, 5))
        by_list.append(round(y, 6))
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        f"const RK={{world:{{t:{json.dumps(ts_list)},y:{{y:{json.dumps(ys_list)},z:{json.dumps(zs_list)}}}}},"
        f"bot:{{t:{json.dumps(bt_list)},y:{{y:{json.dumps(by_list)}}}}}}};\n"
        f"{core}\n"
        "const fin=htAllS1ForThrow({ht:2.0,lastS0Y:1.0});\n"
        "const miss=htAllS1ForThrow({ht:99.0,lastS0Y:1.0});\n"
        "console.log(JSON.stringify({fin,miss}));\n"
    )
    result = _run_node(tmp_path, harness)

    fin = result["fin"]
    assert fin is not None
    assert fin["nWin"] == 14            # 贴地点被 z 门槛挡在窗外
    assert fin["n"] == 13               # 污染点被 3σ 剔除
    assert fin["tRel"] == pytest.approx(2.025, abs=2e-3)
    assert fin["vy"] == pytest.approx(-3.5, abs=0.05)
    assert fin["carVy"] == pytest.approx(0.5, abs=0.03)   # 塌陷伪迹在窗外，不进车拟合
    assert fin["eA"] == pytest.approx(-100.0, abs=3)      # 车@ht − 冻结面 = −100mm
    assert result["miss"] is None       # 窗内无球观测不产出


def test_main_pc_truth_uses_rk_ht_and_accepted_uses_own_ht():
    source = SRC.read_text(encoding="utf-8")
    assert "const truth=pcTruthAt(htPc);" in source
    # HT err@300 锚在全S1重估 HT 上；@300 消息 ht 只留在 hE300/备查里
    assert "const hE300=(fin&&isNum(th.ref300Ht))?(th.ref300Ht-fin.tRel)*1000:null;" in source
    assert "<th>HT真实(球×车)<br>(s,PC轴)</th><th>HT err@300<br>(ms)</th>" in source
    assert "'<td>'+htPc.toFixed(3)+'</td>'" not in source
    assert "const truth=pcTruthAt(accHtPc);" in source
    assert "<td>'+pcTruthCell(truth,true)+'</td>" in source
    assert "<td>'+pcTruthCell(truth)+'</td>" in source
    # 0803 起已删列：开始触球t/PC球×车相交t/HT−开始触球/HT−PC相交/RK−PC dx/dz/
    # PC真值@开始触球，相关实现与辅助函数不应残留
    assert "touchT" not in source
    assert "pcMeetTrueAt" not in source
    assert "TOUCH_DWELL_LEAD_S" not in source
    assert "开始触球" not in source
    assert "球×车相交" not in source
    assert "y:fy[0]-c.y" in source
    assert "x:fx[0]-c.x" in source
    coord_note = "PC真值 x/y = 拟合球世界 x/y − 同时刻插值车世界 x/y，采用世界坐标轴，不随车体 yaw 旋转"
    assert source.index('id="p5"') < source.index(coord_note) < source.index('id="rk300Tbl"')
    assert "preHt" not in source
    assert "hitTableHtml" not in source


def test_rk300_table_includes_last_accepted_target_and_tcp_at_accepted_ht():
    source = SRC.read_text(encoding="utf-8")
    assert "const accepted=lastAcceptedForThrow(th);" in source
    assert "const accHt=accepted&&isNum(accepted.wht)?accepted.wht-RK.t0:null;" in source
    assert "const tcp=accHt!=null?tcpAt(accHt):null;" in source
    assert "const tcpWorld=tcp?[tcp[0],tcp[1],tcp[2]-(isNum(armZOff)?armZOff:0)]:null;" in source
    assert "const tcpCell=tcpWorld?tableXyz(tcpWorld[0],tcpWorld[1],tcpWorld[2]):tableFmt(null,4);" in source
    assert "const tcpAcceptedDx=accepted&&tcpWorld&&isNum(accepted.wx)?(tcpWorld[0]-accepted.wx)*100:null;" in source
    assert "const tcpAcceptedDz=accepted&&tcpWorld&&isNum(accepted.wz)?(tcpWorld[2]-accepted.wz)*100:null;" in source
    assert "armPredictionMatchesAccepted(p,e.t+RK.t0,rec.tx,rec.tz,dur)" in source
    headers = [
        "<th>车RUN末帧 目标−实际 dx/dy(cm)<br>(RK世界系)</th>",
        "<th>RK@≈300ms预测车@HT−RUN末实际 dx/dy(cm)<br>(RK世界系)</th>",
        "<th>机械臂最后accepted目标 x/z(m)</th>",
        "<th>PC真值@HT300 x/y/z(m)</th>",
        "<th>TCP@accepted HT x/y/z(m)</th>",
        "<th>TCP−accepted dx/dz(cm)</th>",
        "<th>拍面yaw@accepted HT(°,世界系)</th>",
    ]
    assert [source.index(header) for header in headers] == sorted(
        source.index(header) for header in headers
    )
    assert "<th>/tennis/status 字符串</th>" not in source
    assert "<th>/predict_hit_pos 字符串</th>" not in source
    assert "const tcpWorld=tcp?[tcp[0],tcp[1],tcp[2]-(isNum(armZOff)?armZOff:0)]:null;" in source
    assert "tableXyz(tcpWorld[0],tcpWorld[1],tcpWorld[2])" in source
    assert "(runEnd.tx-runEnd.x)*100" in source
    assert "(runEnd.ty-runEnd.y)*100" in source
    assert source.count("<td>'+runTargetError+'</td>") == 2
    assert "(th.ref300CarX-runEnd.x)*100" in source
    assert "(th.ref300CarY-runEnd.y)*100" in source
    assert source.count("<td>'+rkPredCarError+'</td>") == 2


def test_add_face_yaw_wiring_and_fk_properties():
    """拍面yaw列的 Python 侧：_add_face_yaw 正确附加 fy、跳过残缺关节；
    FK 性质：零位 fy=0（拍面朝正前）、J1 为垂直轴（Δfy=−Δq1 精确）。"""
    pytest.importorskip("numpy")
    import math
    import sys

    sys.path.insert(0, str(SRC.parent))
    try:
        from generate_curve3_html import _add_face_yaw
    finally:
        sys.path.pop(0)

    q = [0.1, -0.2, 0.3, 0.15, -0.4, 0.25]
    q1p = [0.3] + q[1:]
    arm = {"states": [
        {"t": 1.0, "position": q},
        {"t": 1.1, "position": [0.1, None, 0.3, 0.15, -0.4, 0.25]},  # 关节缺失
        {"t": 1.2, "position": q[:5]},                               # 非 6 关节
        {"t": 1.3},                                                  # 无 position
        {"t": 1.4, "position": [0.0] * 6},
        {"t": 1.5, "position": q1p},
    ]}
    _add_face_yaw(arm)
    s = arm["states"]
    assert isinstance(s[0].get("fy"), float)
    assert all("fy" not in r for r in s[1:4])
    assert s[4]["fy"] == pytest.approx(0.0, abs=1e-6)          # 零位拍面朝正前
    assert s[5]["fy"] - s[0]["fy"] == pytest.approx(           # J1 垂直轴：Δfy = −Δq1
        -math.degrees(0.2), abs=0.02
    )


def test_rk300_table_shows_reject_reasons_only_without_accepted():
    source = SRC.read_text(encoding="utf-8")
    assert "const rejectNote=accepted?'—':rejectNoteForThrow(th);" in source
    assert "<th>备注</th>" in source
    assert "^reject hit: (.+)$" in source
    assert source.count('<td class="armTblNote"><div>\'+rejectNote+\'</div></td>') == 2
    assert "-webkit-line-clamp:2" in source


def test_rk_plot_uses_predict_hit_car_position_and_removes_old_traces():
    source = SRC.read_text(encoding="utf-8")
    extractor = RK_EXTRACTOR.read_text(encoding="utf-8")

    assert 'car_pred_x=payload.get("car_pred_x")' in extractor
    assert 'car_pred_y=payload.get("car_pred_y")' in extractor
    assert "rkPredTr('car_pred_x','/predict_hit_pos car_pred_x'" in source
    assert "rkPredTr('car_pred_y','/predict_hit_pos car_pred_y'" in source

    removed = [
        "RK Ball-Car dX",
        "RK Ball-Car dY",
        "RK Ball-Car XY Dist",
        "PC Car Z",
        "PC Car Yaw x10",
        "PC Hit remaining(ms)",
        "PC Ball-Car dX",
        "PC Ball-Car dY",
        "PC Ball-Car XY Dist",
        "PC Truth X",
        "PC Truth Y",
        "PC Truth Z",
    ]
    assert not any(name in source for name in removed)
    assert "pcCarTr('x','PC Car X'" in source
    assert "pcCarTr('y','PC Car Y'" in source
    assert "PC Hit X" in source
    assert "PC Hit Y" in source
    assert "PC Hit Z" in source


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_visual_racket_fit_estimates_position_velocity_and_extrapolation_error(tmp_path):
    core = _core("racket-fit-core-begin", "racket-fit-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        f"{core}\n"
        "const interp=[-0.15,-0.10,-0.05,0,0.05,0.10,0.15].map(t=>"
        "({t,x:1+2*t+3*t*t,y:0,z:1.5-0.4*t+2*t*t}));\n"
        "const extra=[-0.40,-0.35,-0.30,-0.25,-0.20,-0.15].map(t=>"
        "({t,x:1+2*t,y:0,z:1.5-0.4*t}));\n"
        "console.log(JSON.stringify({interp:fitVisualRacketRows(interp,0),"
        "extra:fitVisualRacketRows(extra,0)}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert result["interp"]["x"] == pytest.approx(1.0)
    assert result["interp"]["z"] == pytest.approx(1.5)
    assert result["interp"]["vx"] == pytest.approx(2.0)
    assert result["interp"]["mode"] == "interpolation"
    assert result["extra"]["x"] == pytest.approx(1.0)
    assert result["extra"]["vx"] == pytest.approx(2.0)
    assert result["extra"]["mode"] == "extrapolation"
    assert result["extra"]["err"] == pytest.approx(0.225)
