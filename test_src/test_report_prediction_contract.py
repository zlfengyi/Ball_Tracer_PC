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
        "// z 偏移是逐场自标定量（armConstCal）：本例复刻 0803 及以前的 −0.153\n"
        "ARM_HIT_Z_OFFSET=-0.153;\n"
        "const acceptT=10.0, acceptX=1.0375, acceptZ=1.3362, duration=0.3216;\n"
        "const actual={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216};\n"
        "const adjacent={rel_x:1.0376,rel_z:1.5058,relSrc:'target',ht:10.3221};\n"
        "// ht−(acc_t+dur) 的系统差 = 臂内提前量 − 发布开销（0802/0803晨 ~+9ms）：必须仍回配成功\n"
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


def _const_cal_harness(events: list) -> str:
    """臂端常量自标定夹具：喂事件流，吐出标定结果与被改写的两个常量。"""
    core = _core("arm-const-cal-core-begin", "arm-const-cal-core-end")
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:0};\n"
        "let ARM_HIT_Z_OFFSET=-0.164;\n"
        "let HIT_TIME_ADVANCE_SEC=0.0;\n"
        f"const ARM={{events:{json.dumps(events)}}};\n"
        "const armPreds=ARM.events.filter(e=>e.topic==='/predict_hit_pos').map(e=>{"
        "const p=JSON.parse(e.text);"
        "return {t:e.t,rel_x:p.rel_x,rel_z:p.rel_z,ht:p.ht,relSrc:p.rel_src};});\n"
        f"{core}\n"
        "console.log(JSON.stringify({cal:armConstCal,z:ARM_HIT_Z_OFFSET,adv:HIT_TIME_ADVANCE_SEC}));\n"
    )


def _cal_session(z_offset: float, advance: float, overhead: float = 0.001) -> list:
    """造 4 拍：每拍 3 条预测（同 rel_x，rel_z 各异）+ 命中最后一条的 accepted。

    臂内：duration = ht − advance − now，accepted 打印 z = rel_z + z_offset，
    状态发布比消息到达晚 overhead。
    """
    events = []
    for k in range(4):
        base = 10.0 * (k + 1)
        for j in range(3):
            t = base + 0.03 * j
            ht = base + 0.40 + 0.001 * j
            rel_z = 1.30 + 0.01 * j
            events.append(_pred_event(t, ht, rel_x=1.0375, rel_z=rel_z))
            dur = round(ht - advance - t, 3)
            text = (f"accepted hit x=1.0375 z={rel_z + z_offset:.4f} "
                    f"duration={dur:.4f} hit_time=0.2500")
            events.append(_status_event(t + overhead, text))
    return events


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("z_offset,advance", [(-0.164, 0.0), (-0.153, 0.010), (-0.153, 0.015)])
def test_arm_constants_are_calibrated_per_session(tmp_path, z_offset, advance):
    """z 偏移与臂内提前量随 arm_controller config.py 改版跳变，必须逐场标定回来。

    写死时 0804 场（−0.164/0ms）用老常数（−0.153/10ms）回配命中 0/80，
    整张北极星表的 accepted 系列列全变 —。同一抛内相邻消息会投错票，靠众数压掉。
    """
    result = _run_node(tmp_path, _const_cal_harness(_cal_session(z_offset, advance)))

    assert result["cal"]["zOff"] == pytest.approx(z_offset, abs=1e-9)
    assert result["cal"]["n"] == 12          # 正确 z 的票数
    assert result["cal"]["total"] == 24      # 含同抛相邻消息的错票
    assert result["z"] == pytest.approx(z_offset, abs=1e-9)
    assert result["adv"] == pytest.approx(advance, abs=1e-9)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_arm_constants_keep_defaults_when_votes_too_few(tmp_path):
    """票数 <3 时不拿噪声改合同：两个常量保持缺省（页面注记提示去查这里）。"""
    events = _cal_session(-0.153, 0.010)[:2]   # 只留 1 条预测 + 1 条 accepted
    result = _run_node(tmp_path, _const_cal_harness(events))

    assert result["cal"]["zOff"] is None
    assert result["cal"]["n"] == 1
    assert result["z"] == pytest.approx(-0.164)
    assert result["adv"] == pytest.approx(0.0)


def _swing_ht_harness(events: str) -> str:
    """挥拍中 ht 重定相重建段的 node 夹具：喂事件流，吐出 hit marks。"""
    core = _core("arm-swing-ht-core-begin", "arm-swing-ht-core-end")
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:0};\n"
        "const HIT_TIME_ADVANCE_SEC=0.010;\n"
        "const SWING_HT_UPDATE_LEAD_SEC=0.100;\n"
        "const SWING_HT_UPDATE_MIN_REMAINING_SEC=0.060;\n"
        "const ARM_HIT_Z_OFFSET=-0.153;\n"
        "const armPredictionMatchesAccepted=(p,acceptT,acceptX,acceptZ,acceptDuration)=>"
        "Math.abs(p.rel_x-acceptX)<5e-4 && Math.abs(p.rel_z+ARM_HIT_Z_OFFSET-acceptZ)<5e-4"
        " && Math.abs(p.ht-(acceptT+acceptDuration))<3e-2;\n"
        f"const ARM={{events:{events}}};\n"
        f"{core}\n"
        "console.log(JSON.stringify(_armHit.marks.filter(h=>h.label==='hit')));\n"
    )


def _pred_event(t, ht, rel_x=1.0, rel_z=1.2):
    """构造一条 /predict_hit_pos 事件（accepted 回配靠 rel_x/rel_z 双 5e-4）。"""
    payload = {
        "x": rel_x, "y": 0.0, "z": rel_z, "stage": 1, "ct": t, "ht": ht,
        "duration": ht - t, "n_bounce_fit": 4, "rel_x": rel_x, "rel_y": 0.0,
        "rel_z": rel_z, "car_pred_x": 0.0, "car_pred_y": 0.0, "car_yaw": 0.0,
        "rel_src": "target",
    }
    return {"t": t, "topic": "/predict_hit_pos", "text": json.dumps(payload)}


def _status_event(t, text):
    return {"t": t, "topic": "/tennis/status", "text": text}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_swing_ht_replan_consumes_last_late_saved(tmp_path):
    """挥拍中 ht 重定相：被消费的是最后一条 late ht saved，finalHt 取它的原始 ht。

    accepted @10.000 ht=10.510 → 臂内触球 10.500、挥拍起 10.250、重定相触发 10.400。
    三条 late（10.30/10.35/10.39）依次覆盖，最后一条 ht=10.516 → 新触球 10.506（+6ms），
    剩余 106ms ≥ 60ms 生效。触发点之后的消息在线上是 reject，这里不进事件流。
    """
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0470 duration=0.5000 hit_time=0.2500"),
        _pred_event(10.26, 10.512), _status_event(10.300, "late ht saved: contact in 0.202s"),
        _pred_event(10.31, 10.514), _status_event(10.350, "late ht saved: contact in 0.154s"),
        _pred_event(10.35, 10.516), _status_event(10.390, "late ht saved: contact in 0.116s"),
    ]
    marks = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))

    assert len(marks) == 1
    h = marks[0]
    assert h["done"] == pytest.approx(10.500)          # 老触球不被改写
    assert h["start"] == pytest.approx(10.250)
    assert h["lastAcceptT"] == pytest.approx(10.000)   # 最后一条 accepted
    assert h["lastUpdateT"] == pytest.approx(10.390)   # 最后一条被受理的更新
    assert h["reswing"]["ok"] is True
    assert h["reswing"]["n"] == 3
    assert h["reswing"]["trig"] == pytest.approx(10.400)
    assert h["reswing"]["newDone"] == pytest.approx(10.506)
    assert h["reswing"]["delta"] == pytest.approx(6.0)
    assert h["finalDone"] == pytest.approx(10.506)
    assert h["finalHt"] == pytest.approx(10.516)       # 原始 ht，不减 10ms
    assert h["finalCt"] == pytest.approx(10.35)        # 与 finalHt 同源那条的 ct
    assert h["wht"] == pytest.approx(10.510)           # accepted 那条保持不变


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_swing_ht_replan_skipped_when_remaining_too_short(tmp_path):
    """新触球距触发点 <60ms 时控制器放弃重定相：finalHt/finalDone 退回最后一条 accepted。"""
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0470 duration=0.5000 hit_time=0.2500"),
        # 新触球 10.450（比老触球早 50ms）→ 距触发点 10.400 只剩 50ms
        _pred_event(10.35, 10.460), _status_event(10.390, "late ht saved: contact in 0.060s"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"]["ok"] is False
    assert h["reswing"]["remain"] == pytest.approx(0.050)
    assert h["finalDone"] == pytest.approx(10.500)
    assert h["finalHt"] == pytest.approx(10.510)
    assert h["lastUpdateT"] == pytest.approx(10.390)   # 受理了，只是没用上


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_swing_without_late_update_reports_no_replan(tmp_path):
    """挥拍窗内没有新预测：reswing 为 null，最后更新时刻即最后一条 accepted。"""
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0470 duration=0.5000 hit_time=0.2500"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"] is None
    assert h["lastUpdateT"] == pytest.approx(10.000)
    assert h["finalHt"] == pytest.approx(10.510)
    assert h["finalDone"] == pytest.approx(10.500)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_late_ht_pairing_self_check_rejects_wrong_source(tmp_path):
    """顺序配对必须自校验：t+duration+10ms 与原消息 ht 差 >8ms 就不认，ht 置 null。

    此时 hitTime 退回 status 时刻重建值（仍可判重定相），但 finalHt 不被污染。
    """
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0470 duration=0.5000 hit_time=0.2500"),
        # 这条预测的 ht 与状态自述的触球时刻差 100ms（模拟队列错位）
        _pred_event(10.35, 10.616), _status_event(10.390, "late ht saved: contact in 0.116s"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"]["ht"] is None
    assert h["reswing"]["newDone"] == pytest.approx(10.506)   # 退回 status t+duration
    assert h["finalHt"] == pytest.approx(10.510)              # 不采信失配的 ht
    assert h["finalCt"] == pytest.approx(9.95)                # ct 与 ht 同源一起退回


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
def test_ball_car_gap_measured_at_given_time_and_rejects_pollution(tmp_path):
    """球面−车 dx/dy/dz(t)：S1 期球世界三轴拟合（3σ 剔跳变、z 门槛）× 车实际 x/y 挥拍前外推线。"""
    core = _core("ball-car-gap-core-begin", "ball-car-gap-core-end")
    # 球：y(t)=1.0−3.5·(t−2.0)、x(t)=0.2+1.0·(t−2.0)、z(t)=1.2−1.0·(t−2.0)−4.905·(t−2.0)²，
    # t∈[1.56,1.95] 共 14 点；t=1.80 一点同时污染 y(+0.5m) 与 x(+0.3m)——按 y 轴 3σ 剔掉后
    # x 拟合必须跟着干净（三轴同进同出，rmsX≈0）；另加 1 个 z=0.05 贴地点（z 门槛过滤）。
    # 车：y(t)=0.9+0.5·(t−2.0)（向球移动）、x(t)=−0.7+0.3·(t−2.0)，挥拍塌陷伪迹点在
    # t≥1.973（窗外，必须无影响）。
    # dy(u)=球心y−R球−车y=0.067−4.0u：u=0 时 +67mm（该时刻球还没够到，评估点偏早
    # 16.75ms）、真实触球在 u=+0.01675（dy=0）、u=+0.03 时 −53mm（偏晚 13.25ms）。
    # u=0 时 dx=0.2−(−0.7)=+900mm、dz=球心z=+1200mm（车中心 z≡地面 0，不扣球半径）。
    ts_list, xs_list, ys_list, zs_list = [], [], [], []
    for i in range(14):
        t = 1.56 + 0.03 * i
        u = t - 2.0
        x, y = 0.2 + 1.0 * u, 1.0 - 3.5 * u
        if abs(t - 1.80) < 1e-9:
            x += 0.3
            y += 0.5
        ts_list.append(round(t, 5))
        xs_list.append(round(x, 6))
        ys_list.append(round(y, 6))
        zs_list.append(round(1.2 - 1.0 * u - 4.905 * u * u, 6))
    ts_list.append(1.70)
    xs_list.append(99.0)
    ys_list.append(99.0)
    zs_list.append(0.05)
    bt_list, bx_list, by_list = [], [], []
    for i in range(20):
        t = 1.80 + 0.01 * i
        u = t - 2.0
        y = 0.9 + 0.5 * u
        if t >= 1.973:
            y -= 0.05  # 挥拍塌陷伪迹
        bt_list.append(round(t, 5))
        bx_list.append(round(-0.7 + 0.3 * u, 6))
        by_list.append(round(y, 6))
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        f"const RK={{world:{{t:{json.dumps(ts_list)},y:{{x:{json.dumps(xs_list)},"
        f"y:{json.dumps(ys_list)},z:{json.dumps(zs_list)}}}}},"
        f"bot:{{t:{json.dumps(bt_list)},y:{{x:{json.dumps(bx_list)},y:{json.dumps(by_list)}}}}}}};\n"
        f"{core}\n"
        "const th={ht:2.0,lastS0Y:1.0};\n"
        "console.log(JSON.stringify({at:ballCarGapForThrow(th,2.0),"
        "contact:ballCarGapForThrow(th,2.01675),late:ballCarGapForThrow(th,2.03),"
        "far:ballCarGapForThrow(th,2.4),miss:ballCarGapForThrow({ht:99.0,lastS0Y:1.0},99.0)}));\n"
    )
    result = _run_node(tmp_path, harness)

    at = result["at"]
    assert at is not None
    assert at["nWin"] == 14            # 贴地点被 z 门槛挡在窗外
    assert at["n"] == 13               # 污染点被 3σ 剔除
    assert at["dy"] * 1000 == pytest.approx(67.0, abs=2)    # 球面还差 67mm 够到车 y 面
    assert at["dx"] * 1000 == pytest.approx(900.0, abs=3)   # 球心x−车x，不扣球半径
    assert at["dz"] * 1000 == pytest.approx(1200.0, abs=3)  # 球心z（车中心z≡地面0）
    assert at["ballX"] == pytest.approx(0.2, abs=3e-3)
    assert at["ballY"] == pytest.approx(1.0, abs=2e-3)
    assert at["ballZ"] == pytest.approx(1.2, abs=3e-3)
    assert at["carX"] == pytest.approx(-0.7, abs=2e-3)
    assert at["carY"] == pytest.approx(0.9, abs=2e-3)
    assert at["rmsX"] * 1000 == pytest.approx(0.0, abs=1)   # y轴剔掉的点也退出x拟合
    assert at["rmsZ"] * 1000 == pytest.approx(0.0, abs=1)
    assert at["dtMs"] == pytest.approx(-16.75, abs=0.5)     # 负=该时刻偏早
    assert at["vy"] == pytest.approx(-3.5, abs=0.05)
    assert at["vRel"] == pytest.approx(-4.0, abs=0.06)
    assert at["carVy"] == pytest.approx(0.5, abs=0.03)   # 塌陷伪迹在窗外，不进车拟合
    assert at["carVx"] == pytest.approx(0.3, abs=0.03)
    assert at["eA"] == pytest.approx(-100.0, abs=3)      # 车@ht − 冻结面 = −100mm
    # 真实触球处 dy 归零；再晚 13.25ms 球已穿过拍面，dy 转负（口径与旧 HT err 同向）
    assert result["contact"]["dy"] * 1000 == pytest.approx(0.0, abs=2)
    assert result["contact"]["dtMs"] == pytest.approx(0.0, abs=0.5)
    assert result["late"]["dy"] * 1000 == pytest.approx(-53.0, abs=2)
    assert result["late"]["dtMs"] == pytest.approx(13.25, abs=0.5)
    assert result["far"] is None        # 评估点离拟合窗 >300ms 不外推
    assert result["miss"] is None       # 窗内无球观测不产出


def test_main_pc_truth_uses_rk_ht_and_accepted_uses_own_ht():
    source = SRC.read_text(encoding="utf-8")
    assert "const truth=pcTruthAt(htPc);" in source
    # 0805 起：全量RK重估的第三个 ht 不再单列，击球误差改成 accepted HT 上的两列空间量——
    # dy 是时序误差的空间形态，dx/dz 是 @300 预测击球点与真值的落点差（两侧同轴同基准）
    # 两列都锚在「臂最后更新HT」= 臂真正执行的击球时刻（重定相生效时是那条 late ht saved），
    # 不是 PC真值/TCP/车yaw 三列用的 accepted HT——两个锚之差就是 Δht 重定相列
    assert "const gapFin=finalHt!=null?ballCarGapForThrow(th,finalHt):null;" in source
    assert "球面y−车y @臂最后更新HT<br>(mm, RK全量真值)</th>" in source
    assert "击球点@300预测 − RK全量真值@臂最后更新HT<br>dx/dz(mm, 世界轴)</th>" in source
    assert "((th.ref300Xw-th.ref300CarX)-gapFin.dx)*1000" in source
    assert "const aimDz=(gapFin&&isNum(th.ref300Z))?(th.ref300Z-gapFin.dz)*1000:null;" in source
    assert "htAllS1ForThrow" not in source
    assert "hE300" not in source
    assert "<th>HT真实(触球)<br>(s,PC轴)</th>" not in source
    assert "HT err@300<br>" not in source
    assert "'<td>'+htPc.toFixed(3)+'</td>'" not in source
    assert "const truth=pcTruthAt(accHtPc);" in source
    assert "<td>'+pcTruthCell(truth,true)+'</td>" in source
    assert "<td>'+pcTruthCell(truth)+'</td>" in source
    # 主表第二列 PC 真值：同一套拟合，评估时刻换成臂最后 accepted 的原消息 ht（与 TCP 同锚）
    assert "const truthAcc=accHt!=null?pcTruthAt(rkToPc(accHt)):null;" in source
    # 有/无 S1@300 两条渲染路径都要出这一列（该列只依赖 accepted，不依赖 @300 参考消息）
    assert source.count("pcTruthCell(truthAcc,true)") == 2
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
        "PC真值@accepted HT x/y/z(m)</th>",
        "<th>TCP@accepted HT x/y/z(m)</th>",
        "<th>TCP−accepted dx/dz(cm)</th>",
        "最后更新−挥拍起<br>(ms)</th>",
        "盲区 ht−ct@臂最后更新<br>(ms)</th>",
        "Δht 重定相<br>(ms)</th>",
        "车yaw@accepted HT(°)</th>",
        "<th>拍面yaw@臂最后更新HT(°,世界系)</th>",
        "<th>拍面yaw@臂最后更新HT−10ms(°,世界系)</th>",
    ]
    assert [source.index(header) for header in headers] == sorted(
        source.index(header) for header in headers
    )
    # 两列拍面 yaw 锚在「臂最后更新 HT」（含挥拍中 ht 重定相消费的那条），不是最后一条 accepted
    assert "const finalHt=accepted&&isNum(accepted.finalHt)?accepted.finalHt-RK.t0:accHt;" in source
    assert "faceYawWorldAt(finalHt)" in source
    assert "faceYawWorldPreAt(finalHt)" in source
    assert "faceYawWorldAt(accHt)" not in source
    assert "faceYawWorldPreAt(accHt)" not in source
    # 盲区列的 ct/ht 必须同源，且与主表 lead(ms)（@300ms 参考消息）不是一回事
    assert "const finalCt=accepted&&isNum(accepted.finalCt)?accepted.finalCt-RK.t0:null;" in source
    assert "const blind=(finalHt!=null&&finalCt!=null)?(finalHt-finalCt)*1000:null;" in source
    # 0803 已删列：视觉球拍@accepted HT（相对小车），其专用取值/单元格实现不应残留
    assert "视觉球拍" not in source
    assert "visualRacketAt" not in source
    assert "visualRacketCell" not in source
    # 车yaw@accepted HT 列：/bot_state 瞬时值，锚在最后 accepted 的原消息 ht（同 PC真值/TCP 列）
    assert "const carYawAcc=accHt!=null?botYawDegAt(accHt):null;" in source
    assert "const carYawRate=accHt!=null?imuYawRateDegAt(accHt):null;" in source
    assert source.count("<td>'+carYawCell+'</td>") == 2
    # 角速度只能取 IMU 零滞后原值，禁止对有 0.3~0.5s 滞后的 bot_state yaw 数值求导
    assert "ys(RK.imu,'yaw_speed')" in source
    # HT−10ms 列：fy 同窗拟合在 HT−10ms 取值；车 yaw 用 /bot_state 瞬时值（IMU 连续更新）
    assert "const FACE_YAW_PRE_S=0.010;" in source
    assert "fitFaceYawTo(accHtRk,tEval)" in source
    assert "botYawDegAt(tEval)" in source
    assert "ys(RK.bot,'yaw')" in source
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
