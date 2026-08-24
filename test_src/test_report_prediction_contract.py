# -*- coding: utf-8 -*-
"""回归测试：RK≈300ms消息与机械臂最后accepted必须是两套独立合同。"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
RK_EXTRACTOR = Path(__file__).resolve().parent / "extract_rk_tracking_bag.py"
TABLE_EXPORTER = Path(__file__).resolve().parent / "export_report_tables.py"
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
        # 显式 utf-8：夹具会吐中文（表头红条文案），按 Windows 默认 cp1252 解码会炸
        [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30
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
def test_last_target_change_matches_prediction_by_deadline_not_ct(tmp_path):
    """末次坐标变化用 bot.t+remaining ↔ pred.ht 回配；更晚 ct 的 deadline-only 消息不能抢走。"""
    run_core = _core("bot-run-end-core-begin", "bot-run-end-core-end")
    pred_core = _core("last-target-pred-core-begin", "last-target-pred-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        "const th={ht:1.25,firstIdx:0,lastIdx:2}; const reportThrows=[th];\n"
        "const rkPredStage=[1,1,1], rkPredNFit=[4,5,6];\n"
        "const RK={"
        "bot:{t:[1.00,1.05,1.10,1.15,1.20,1.25,1.26],y:{"
        "phase:['RUN','RUN','RUN','RUN','RUN','RUN','BRAKE_IN_SWING'],"
        "x:[0,0,0,0,0,2.25,2.25],y:[0,0,0,0,0,3.35,3.35],"
        "target_x:[1,1,1,4,4,4,null],target_y:[2,3,3,3,3,3,null],"
        "remaining:[1.00,.95,.90,.65,.69,.64,null]}},"
        "pred:{t:[1.00,1.10,1.14],y:{"
        "ht_rel:[2.00,1.80,1.89],x:[.50,.60,999],rel_x:[.70,.82,999],rel_z:[1.10,1.24,999],"
        "car_pred_x:[2.00,2.20,999],car_pred_y:[3.00,3.30,999]}}};\n"
        f"{run_core}\n{pred_core}\n"
        "const end=botRunEndForThrow(th), p=lastTargetPredictionForThrow(th,end);\n"
        "console.log(JSON.stringify({end,ref:p}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert result["end"]["targetChange"] == pytest.approx(
        {"t": 1.15, "x": 4, "y": 3, "idx": 3, "deadline": 1.80}
    )
    assert result["ref"] == pytest.approx(
        {
            "idx": 1, "ct": 1.10, "ht": 1.80, "lead": .70, "htError": 0,
            "stage": 1, "nFit": 5, "worldX": .60, "relX": .82, "relZ": 1.24,
            "carX": 2.20, "carY": 3.30,
        }
    )


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
    """值键（兜底路径）：x/z 双 5e-4 + ht 30ms，两个自标定量都进算式。"""
    core = _core(
        "arm-prediction-match-core-begin",
        "arm-prediction-match-core-end",
    )
    harness = (
        f"{core}\n"
        "// 两个都是逐场自标定量（armConstCal）：本例复刻 0803 及以前的 z −0.153、x 无变换\n"
        "ARM_HIT_Z_OFFSET=-0.153;\n"
        "const acceptT=10.0, acceptX=1.0375, acceptZ=1.3362, duration=0.3216;\n"
        "const actual={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216};\n"
        "const adjacent={rel_x:1.0376,rel_z:1.5058,relSrc:'target',ht:10.3221};\n"
        "// ht−(acc_t+dur) 的系统差 = 臂内提前量 − 发布开销（0802/0803晨 ~+9ms）：必须仍回配成功\n"
        "const servoLag={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216+0.0092};\n"
        "const crossThrow={rel_x:1.0375,rel_z:1.4892,relSrc:'target',ht:10.3216+0.5};\n"
        "const call=p=>armPredictionMatchesAccepted(p,acceptT,acceptX,acceptZ,duration);\n"
        "const before={actual:call(actual),adjacent:call(adjacent),"
        "servoLag:call(servoLag),crossThrow:call(crossThrow)};\n"
        "// 臂端 0811 起 x/=cos5°：x 比例也自标定进来后，同一条消息必须还认得出\n"
        "ARM_HIT_X_SCALE=1/Math.cos(5*Math.PI/180);\n"
        "const scaledAcceptX=Number((1.0375*ARM_HIT_X_SCALE).toFixed(4));\n"
        "const scaled=armPredictionMatchesAccepted(actual,acceptT,scaledAcceptX,acceptZ,duration);\n"
        "const stale=armPredictionMatchesAccepted(actual,acceptT,acceptX,acceptZ,duration);\n"
        "console.log(JSON.stringify(Object.assign(before,{scaled,stale})));\n"
    )
    result = _run_node(tmp_path, harness)
    assert result == {
        "actual": True,
        "adjacent": False,
        "servoLag": True,
        "crossThrow": False,
        "scaled": True,     # 按本场标定的 x 比例回配
        "stale": False,     # 不标定就整场失配（0811 113734 场 115/115 全灭的机理）
    }


def _const_cal_harness(events: list) -> str:
    """臂端常量自标定夹具：喂事件流，吐出序号对齐、标定结果与被改写的三个常量。"""
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:0};\n"
        "let HIT_TIME_ADVANCE_SEC=0.0;\n"
        f"const ARM={{events:{json.dumps(events)}}};\n"
        + _core("arm-prediction-match-core-begin", "arm-prediction-match-core-end") + "\n"
        "const armPreds=ARM.events.filter(e=>e.topic==='/predict_hit_pos').map(e=>{"
        "const p=JSON.parse(e.text);"
        "return {t:e.t,rel_x:p.rel_x,rel_z:p.rel_z,ht:p.ht,relSrc:p.rel_src};});\n"
        + _core("arm-pred-align-core-begin", "arm-pred-align-core-end") + "\n"
        + _core("arm-const-cal-core-begin", "arm-const-cal-core-end") + "\n"
        # 标定完再拿值键复核每条 accepted↔源消息：标定对了值键就该整场认得出
        "const recheck=armHitStatuses.map(s=>{const m=armAcceptedHitRe.exec(s.text);"
        "if(!m) return null; const dur=Number(m[3]); const p=armPredForStatus(s,dur);"
        "if(!p) return null;"
        "return armPredictionMatchesAccepted(p,s.t+RK.t0,Number(m[1]),Number(m[2]),dur);})"
        ".filter(v=>v!==null);\n"
        "console.log(JSON.stringify({cal:armConstCal,align:armPredAlign,z:ARM_HIT_Z_OFFSET,"
        "x:ARM_HIT_X_SCALE,adv:HIT_TIME_ADVANCE_SEC,"
        "recheck:{n:recheck.length,ok:recheck.filter(Boolean).length}}));\n"
    )


def _cal_session(z_offset: float, advance: float, overhead: float = 0.001,
                 x_scale: float = 1.0, throws: int = 4, per_throw: int = 3) -> list:
    """造 N 拍：每拍若干条预测（同 rel_x，rel_z 各异）+ 每条各回一条 accepted。

    臂内：duration = ht − advance − now，accepted 打印 z = rel_z + z_offset、
    x = rel_x × x_scale（0811 起 = 1/cos(kHitYawExtraRad)），状态发布比消息到达晚 overhead。
    "收一条必回一条" 是 on_hit_pos 的结构合同，序号对齐就架在它上面。
    per_throw=1 时同抛内没有相邻消息，每条 accepted 只投一票——用来单测 δ 的认票门槛。
    """
    events = []
    for k in range(throws):
        base = 10.0 * (k + 1)
        for j in range(per_throw):
            t = base + 0.03 * j
            ht = base + 0.40 + 0.001 * j
            rel_z = 1.30 + 0.01 * j
            events.append(_pred_event(t, ht, rel_x=1.0375, rel_z=rel_z))
            dur = round(ht - advance - t, 3)
            text = (f"accepted hit x={1.0375 * x_scale:.4f} z={rel_z + z_offset:.4f} "
                    f"duration={dur:.4f} hit_time=0.2500")
            events.append(_status_event(t + overhead, text))
    return events


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("z_offset,advance", [(-0.164, 0.0), (-0.153, 0.010), (-0.153, 0.015)])
def test_arm_constants_are_calibrated_per_session(tmp_path, z_offset, advance):
    """z 偏移与臂内提前量随 arm_controller 改版跳变，必须逐场标定回来。

    写死时 0804 场（−0.164/0ms）用老常数（−0.153/10ms）回配命中 0/80，
    整张北极星表的 accepted 系列列全变 —。样本由序号对齐给出，故不含同抛相邻消息的错票。
    """
    result = _run_node(tmp_path, _const_cal_harness(_cal_session(z_offset, advance)))

    assert result["align"]["delta"] == 0     # hit_pos 派生状态与预测消息一对一保序
    assert result["align"]["n"] == 12
    assert result["cal"]["zOff"] == pytest.approx(z_offset, abs=1e-9)
    assert result["cal"]["n"] == 12          # 正确 z 的票数
    assert result["cal"]["total"] == 12      # 序号对齐后不再有同抛相邻消息的错票
    assert result["z"] == pytest.approx(z_offset, abs=1e-9)
    assert result["adv"] == pytest.approx(advance, abs=1e-9)
    assert result["recheck"] == {"n": 12, "ok": 12}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_arm_target_x_transform_is_calibrated_not_fatal(tmp_path):
    """臂端对目标 x 做单点变换（0811 kHitYawExtraRad=5° → x/=cos5°）不许打死回配。

    这正是 0811 113734 场：115 条 accepted 全部回配失败、armConstCal 一票、
    accepted 目标/击球真值/TCP 各列整列 —。回配主键改成序号键后，x 比例只是个被标定出来的量。
    """
    scale = 1 / math.cos(math.radians(5.0))
    result = _run_node(tmp_path, _const_cal_harness(
        _cal_session(-0.164, 0.0, x_scale=scale)))

    assert result["align"]["delta"] == 0
    assert result["cal"]["zOff"] == pytest.approx(-0.164, abs=1e-9)
    assert result["cal"]["xScale"] == pytest.approx(scale, abs=2e-4)
    assert result["x"] == pytest.approx(scale, abs=2e-4)
    assert result["adv"] == pytest.approx(0.0, abs=1e-9)
    assert result["recheck"] == {"n": 12, "ok": 12}     # 标定回来后值键也认得出


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_arm_alignment_rejected_when_one_to_one_broken(tmp_path):
    """bag 少收状态 → 序号差中途跳变：δ 众数不过 6 成就整场退回值键，不许硬套。

    0719 100823 那场（少 11 条状态）δ=−7 只拿 4/9 票，却能骗过逐条弱键自校验，
    把 9 条回配打成 4 条、z 标定带歪 13mm。
    """
    events = _cal_session(-0.164, 0.0, throws=8, per_throw=1)
    # 第 4 拍丢一条状态：其后所有状态的序号整体前移一格，票数分成 3(δ=0) : 4(δ=−1)
    drop = next(i for i, e in enumerate(events)
                if e["topic"] == "/tennis/status" and e["t"] > 39.0)
    events.pop(drop)
    result = _run_node(tmp_path, _const_cal_harness(events))

    assert result["align"]["delta"] is None      # 4/7 不足 6 成，不认
    assert result["align"]["n"] == 4             # 众数票数够 3，但只占 4/7
    assert result["cal"]["zOff"] == pytest.approx(-0.164, abs=1e-9)   # 值键自举兜底
    assert result["z"] == pytest.approx(-0.164, abs=1e-9)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_arm_constants_keep_defaults_when_votes_too_few(tmp_path):
    """票数 <3 时不拿噪声改合同：三个常量保持缺省（页面红条提示去查这里）。"""
    events = _cal_session(-0.153, 0.010)[:2]   # 只留 1 条预测 + 1 条 accepted
    result = _run_node(tmp_path, _const_cal_harness(events))

    assert result["align"]["delta"] is None
    assert result["cal"]["zOff"] is None
    assert result["cal"]["n"] == 1
    assert result["z"] == pytest.approx(-0.164)
    assert result["x"] == pytest.approx(1.0)
    assert result["adv"] == pytest.approx(0.0)


def _arm_hit_harness(events: str, tail: str) -> str:
    """整条回配链路（序号对齐 + 逐场自标定 + 值键复核 + 重定相重建）的 node 夹具。"""
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:0};\n"
        "let HIT_TIME_ADVANCE_SEC=0.010;\n"
        "const SWING_HT_UPDATE_LEAD_SEC=0.100;\n"
        "const SWING_HT_UPDATE_MIN_REMAINING_SEC=0.060;\n"
        f"const ARM={{events:{events}}};\n"
        + _core("arm-prediction-match-core-begin", "arm-prediction-match-core-end") + "\n"
        + _core("arm-swing-ht-core-begin", "arm-swing-ht-core-end") + "\n"
        + tail
    )


def _swing_ht_harness(events: str) -> str:
    """挥拍中 ht 重定相重建段的 node 夹具：喂事件流，吐出 hit marks。"""
    return _arm_hit_harness(
        events,
        "console.log(JSON.stringify(_armHit.marks.filter(h=>h.label==='hit')));\n",
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
        _status_event(10.000, "accepted hit x=1.0000 z=1.0360 duration=0.5000 hit_time=0.2500"),
        _pred_event(10.26, 10.512), _status_event(10.300, "late ht saved: contact in 0.202s"),
        _pred_event(10.31, 10.514), _status_event(10.350, "late ht saved: contact in 0.154s"),
        _pred_event(10.35, 10.516), _status_event(10.390, "late ht saved: contact in 0.116s"),
        # 无 sweep_w 的旧状态仍受老 done+50ms 归属窗约束，不得抢走最后消费时刻。
        _pred_event(10.56, 10.700), _status_event(10.600, "late ht saved: contact in 0.090s"),
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
def test_continuous_sweep_keeps_raw_last_saved_after_old_done(tmp_path):
    """连续 sweep 的 raw last-saved 可晚于老 done+50ms，但 mode=2 不代表 profile 重解。"""
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0360 duration=0.5000 hit_time=0.2500"),
        _pred_event(10.35, 10.516),
        _status_event(10.390, "late ht saved: contact in 0.116s sweep_w=8.50 mode=0"),
        # 旧报告以 done+50ms=10.550 截断，会漏掉下面两条已被 update_ht() 消费的更新。
        _pred_event(10.52, 10.710),
        _status_event(10.580, "late ht saved: contact in 0.120s sweep_w=8.50 mode=2"),
        # advance 整数 ms 自标定 + duration 毫秒打印会留下 −2.8ms gap；sweep 状态仍是强消费证据。
        _pred_event(10.56, 10.7228),
        _status_event(10.600, "late ht saved: contact in 0.110s sweep_w=8.50 mode=2"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"]["continuousSweep"] is True
    assert h["reswing"]["n"] == 3
    assert h["reswing"]["nCoast"] == 2
    assert h["sweepCoast"] is True
    assert [late["mode"] for late in h["lates"]] == [0, 2, 2]
    assert [late["ht"] for late in h["lates"]] == pytest.approx([10.516, 10.710, 10.7228])
    assert h["lastUpdateT"] == pytest.approx(10.600)
    assert h["finalDone"] == pytest.approx(10.7128)
    assert h["finalHt"] == pytest.approx(10.7228)
    assert h["finalCt"] == pytest.approx(10.56)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_continuous_sweep_saved_bypasses_legacy_remaining_gate(tmp_path):
    """带 sweep_w 的 raw HT 已写入 update_ht，不能再被旧版 60ms 一次性门否掉。"""
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0360 duration=0.5000 hit_time=0.2500"),
        # 内部触球 10.450，距旧触发点 10.400 仅 50ms；连续 sweep 仍已消费该 HT。
        _pred_event(10.35, 10.460),
        _status_event(10.390, "late ht saved: contact in 0.060s sweep_w=8.50 mode=0"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"]["continuousSweep"] is True
    assert h["reswing"]["nCoast"] == 0
    assert h["sweepCoast"] is False
    assert h["reswing"]["remain"] == pytest.approx(0.050)
    assert h["reswing"]["ok"] is True
    assert h["finalHt"] == pytest.approx(10.460)
    assert h["finalCt"] == pytest.approx(10.35)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_mode2_tcp_sample_uses_actual_plane_instead_of_raw_final_ht(tmp_path):
    """mode=2 时五个 TCP 数整体取实际过面，不能混入随挥阶段的 raw finalHt。"""
    script = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const lerp=(a,b,f)=>a+(b-a)*f;\n"
        "const RK={t0:0};\n"
        "const armTcpRows=["
        "{t:10.000,tcp:[0.9430,-0.0100,1.0300]},"
        "{t:10.010,tcp:[0.9432, 0.0100,1.0440]},"
        "{t:10.100,tcp:[0.7000, 0.0200,1.1000]},"
        "{t:10.110,tcp:[0.7000,-0.0200,1.1000]}];\n"
        + _core("arm-execution-contact-core-begin", "arm-execution-contact-core-end")
        + "\nconst h={tx:0.9431,tz:1.0296,tgtYawExtraDeg:0,start:9.9,done:10.0,"
          "finalHt:10.105,wht:10.0,hitT:0.25,"
          "reswing:{continuousSweep:true,lastMode:2,nCoast:2}};\n"
          "const exec=armExecutionContactAt(h);\n"
          "const raw=[0,1,2].map(k=>lerp(armTcpRows[2].tcp[k],armTcpRows[3].tcp[k],0.5));\n"
          "console.log(JSON.stringify({exec,raw,rawHt:h.finalHt}));\n"
    )
    result = _run_node(tmp_path, script)

    assert result["exec"]["t"] == pytest.approx(10.005)
    assert result["exec"]["t"] != pytest.approx(result["rawHt"])
    assert result["exec"]["tcp"] == pytest.approx([0.9431, 0.0, 1.0370])
    assert result["exec"]["target"] == pytest.approx([0.9431, 0.0, 1.0296])
    assert result["raw"] == pytest.approx([0.7000, 0.0, 1.1000])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_tcp_five_sample_switches_whole_cell_for_mode2(tmp_path):
    """健康抛取 raw；Coast 五数整体取 exec；没有可信过面时不回退 raw。"""
    script = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const raw=[0.70,0.20,1.10], exec={t:9.9,tcp:[0.943,0,1.045]};\n"
        "const tcpAt=t=>raw; let crossing=exec;\n"
        "const armExecutionContactAt=h=>crossing;\n"
        + _core("arm-tcp-five-sample-core-begin", "arm-tcp-five-sample-core-end")
        + "\nconst healthy=armTcpFiveSampleAt({reswing:{lastMode:0}},10.1);\n"
          "const coast=armTcpFiveSampleAt({reswing:{lastMode:2}},10.1);\n"
          "crossing=null;\n"
          "const missing=armTcpFiveSampleAt({reswing:{lastMode:2}},10.1);\n"
          "console.log(JSON.stringify({healthy,coast,missing}));\n"
    )
    result = _run_node(tmp_path, script)

    assert result["healthy"] == {"t": 10.1, "tcp": [0.70, 0.20, 1.10], "usesExec": False}
    assert result["coast"] == {"t": 9.9, "tcp": [0.943, 0, 1.045], "usesExec": True}
    assert result["missing"] is None


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_swing_ht_replan_skipped_when_remaining_too_short(tmp_path):
    """新触球距触发点 <60ms 时控制器放弃重定相：finalHt/finalDone 退回最后一条 accepted。"""
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0360 duration=0.5000 hit_time=0.2500"),
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
        _status_event(10.000, "accepted hit x=1.0000 z=1.0360 duration=0.5000 hit_time=0.2500"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"] is None
    assert h["lastUpdateT"] == pytest.approx(10.000)
    assert h["finalHt"] == pytest.approx(10.510)
    assert h["finalDone"] == pytest.approx(10.500)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_late_ht_pairing_self_check_rejects_wrong_source(tmp_path):
    """顺序配对必须自校验：t+duration+10ms 与原消息 ht 差 >8ms 就不认，ht 置 null。

    此时 hitTime 退回 status 时刻重建值（仍可判重定相），但 finalHt/finalCt 必须清空，
    让所有空间列显式缺失，绝不能退回 accepted HT 冒充最后消费时刻。
    """
    events = [
        _pred_event(9.95, 10.510),
        _status_event(10.000, "accepted hit x=1.0000 z=1.0360 duration=0.5000 hit_time=0.2500"),
        # 这条预测的 ht 与状态自述的触球时刻差 100ms（模拟队列错位）
        _pred_event(10.35, 10.616), _status_event(10.390, "late ht saved: contact in 0.116s"),
    ]
    h = _run_node(tmp_path, _swing_ht_harness(json.dumps(events)))[0]

    assert h["reswing"]["ht"] is None
    assert h["reswing"]["newDone"] == pytest.approx(10.506)   # 退回 status t+duration
    assert h["finalMismatch"] is True
    assert h["finalHt"] is None
    assert h["finalCt"] is None


def _dup_session(x_scale: float = 1.0, drop_status_after: float | None = None,
                 dx: float = 0.0002, dz: float = 0.0002) -> list:
    """造 4 拍，每拍两条预测：先到的被 accept，后到的落在挥拍窗里被 reject。

    dx/dz = 后到那条与被消费那条的间距。缺省 0.2mm 是"值键分不开"的实况（0804/0808/0811
    三场各实测到 1~2 条：就近搜会选中晚到 33ms 的那条）；给 3mm 则是同抛相邻消息的常态间距。
    accepted 的 x 再乘 x_scale，复刻臂端 0811 起的 x/=cos5° 变换。
    """
    events = []
    for k in range(4):
        base = 10.0 * (k + 1)
        events.append(_pred_event(base, base + 0.500, rel_x=1.0358, rel_z=1.2281))
        events.append(_status_event(base + 0.048, (
            f"accepted hit x={1.0358 * x_scale:.4f} z=1.0641 "
            f"duration={0.500 - 0.048:.4f} hit_time=0.2500")))
        events.append(_pred_event(base + 0.033, base + 0.504,
                                  rel_x=1.0358 - dx, rel_z=1.2281 + dz))
        events.append(_status_event(base + 0.081, "reject hit: hit phase in progress"))
    if drop_status_after is not None:
        drop = next(i for i, e in enumerate(events)
                    if e["topic"] == "/tennis/status" and e["t"] > drop_status_after)
        events.pop(drop)
    return events


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("x_scale", [1.0, 1 / math.cos(math.radians(5.0))])
def test_accepted_matches_the_message_it_consumed_not_the_nearest(tmp_path, x_scale):
    """回配必须落在臂真正消费的那条上：晚到 33ms 的那条 x/z 只差 0.2mm，值键分不开。

    x_scale≠1 复刻 0811 kHitYawExtraRad：值键此时整场对不上，全靠序号键顶住。
    """
    result = _run_node(tmp_path, _arm_hit_harness(
        json.dumps(_dup_session(x_scale=x_scale)),
        "console.log(JSON.stringify({nAcc:_armHit.nAcc,nMatch:_armHit.nMatch,"
        "delta:armPredAlign.delta,xScale:armConstCal.xScale,"
        "wht:_armHit.marks.filter(h=>h.label==='hit').map(h=>h.wht)}));\n"))

    assert result["delta"] == 0
    assert (result["nAcc"], result["nMatch"]) == (4, 4)
    assert result["xScale"] == pytest.approx(x_scale, abs=2e-4)
    # 消费的是先到那条（ht=base+0.500），不是晚到的 base+0.504
    assert result["wht"] == pytest.approx([10.5, 20.5, 30.5, 40.5])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_value_keys_repair_alignment_when_a_status_is_missing(tmp_path):
    """序号对齐被"少收一条状态"整体带偏一格时，本场自标定的值键要把它掰回来。

    偏的那条同样过得了到达窗与时序键（相邻消息只差 33ms/4ms），弱键自校验看不出来；
    但它的 x/z 与 accepted 差着常态间距，命中不了值键，而正确那条严丝合缝。
    （间距退化到 0.2mm 时两条谁都对得上，此时错配代价 = 用了 33ms 前的同抛预测。）
    """
    result = _run_node(tmp_path, _arm_hit_harness(
        json.dumps(_dup_session(drop_status_after=20.05, dx=0.003, dz=0.006)),
        "console.log(JSON.stringify({nMatch:_armHit.nMatch,delta:armPredAlign.delta,"
        "wht:_armHit.marks.filter(h=>h.label==='hit').map(h=>h.wht)}));\n"))

    assert result["delta"] == -1        # 序号对齐被丢掉的那条状态带偏了一格
    assert result["nMatch"] == 4
    assert result["wht"] == pytest.approx([10.5, 20.5, 30.5, 40.5])


def _warn_harness(n_acc: int, n_match: int, parse_bad: int = 0) -> str:
    """表头红条夹具：喂回配统计，吐出 armDataWarnHtml。"""
    return (
        "const ARM={events:[]};\n"
        f"let armPredParseBad={parse_bad}, armPredTotal=722;\n"
        f"const _armHit={{nAcc:{n_acc},nMatch:{n_match}}};\n"
        f"const armPredAlign={{delta:null,n:1,nAcc:{n_acc}}};\n"
        "const armConstCal={xScale:null,zOff:null,adv:null};\n"
        + _core("arm-data-warn-core-begin", "arm-data-warn-core-end") + "\n"
        "console.log(JSON.stringify({html:armDataWarnHtml}));\n"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_arm_data_failure_is_shown_as_a_banner_not_a_page_of_dashes(tmp_path):
    """回配失败必须在表头喊出来：整表满屏 "—" 看上去像"这场没数据"而不是"报告读不了数据"。

    0809 103849（载荷截断）与 0811 113734（臂端改了 x 变换）两次都是这么被埋掉的，
    所以红条里要带上三把键各自的状态，下次一眼能定位是结构变了还是口径变了。
    """
    dead = _run_node(tmp_path, _warn_harness(115, 0))["html"]
    assert "#e94560" in dead                      # 全灭 = 红条
    assert "115 条 accepted hit 只回配上 0 条" in dead
    assert "序号对齐 δ=" in dead and "本场自标定 x×" in dead

    partial = _run_node(tmp_path, _warn_harness(115, 113))["html"]
    assert "#e0a24a" in partial and "#e94560" not in partial   # 部分失配 = 橙条
    assert "只回配上 113 条" in partial

    assert _run_node(tmp_path, _warn_harness(115, 115))["html"] == ""   # 全中就不出条
    truncated = _run_node(tmp_path, _warn_harness(115, 115, parse_bad=9))["html"]
    assert "载荷解析失败 9/722 条" in truncated and "#e94560" in truncated


def _plane_shift_harness(x_scale, tx: float, dy: float) -> str:
    """触球平面前移量的 node 夹具：喂本场自标定 x 比例 + 该拍目标 x + 实测 dy。"""
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const cmFmt=(v,d=1)=>isNum(v)?(Number(v)*100).toFixed(d):'—';\n"
        "const cmSigned=(v,d=1)=>isNum(v)?(v>=0?'+':'')+(Number(v)*100).toFixed(d):'—';\n"
        "const tableSigned=v=>isNum(v)?(v>=0?'+':'')+Number(v).toFixed(1):'—';\n"
        f"const armConstCal={{xScale:{'null' if x_scale is None else repr(x_scale)}}};\n"
        f"const accepted={{tx:{tx}}};\n"
        f"const gapFin={{dy:{dy}, vRel:-12.0}};\n"
        "const finalHt=1.0;\n"
        + _core("hit-plane-shift-core-begin", "hit-plane-shift-core-end") + "\n"
        "console.log(JSON.stringify({deg:hitYawExtraRad*180/Math.PI,"
        "shift:hitPlaneShift,note:hitPlaneNote}));\n"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_hit_plane_shift_is_derived_from_the_self_calibrated_x_scale(tmp_path):
    """臂端整体多转 δ 把触球平面搬离车 y 面 → 「球面y−车y」列的零点跟着搬。

    列的口径不动（用户 0812 拍板），但零点必须在悬停里写出来：0811 113734 场 dy=+4cm
    按老零点读是"球没够到、ht 偏早 3ms"，扣掉平面前移 9.1cm 后是"已穿过 5cm、ht 偏晚 4ms"，
    符号相反。δ 由 x 比例反解（同一处臂端变换 x/=cosδ），臂端下场改 δ 报告自动跟上。
    """
    scale = 1 / math.cos(math.radians(5.0))
    r = _run_node(tmp_path, _plane_shift_harness(scale, 1.0439, 0.040))
    assert r["deg"] == pytest.approx(5.0, abs=0.02)
    assert r["shift"] == pytest.approx(1.0439 * math.sin(math.radians(5.0)), abs=1e-6)
    assert "零点是 +9.1cm，不是 0" in r["note"]
    assert "扣掉后球面到拍面 -5.1cm" in r["note"]       # +4.0 − 9.1（cm，1位小数）
    assert "等效时序 +4.2ms" in r["note"]                # 正=该 ht 比真实触球晚

    # δ=0（臂端没这个变换 / 没标定出来）时整段不出现，与改前逐字相同
    for x_scale in (None, 1.0):
        off = _run_node(tmp_path, _plane_shift_harness(x_scale, 1.0439, 0.040))
        assert (off["deg"], off["shift"], off["note"]) == (0, 0, "")


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
    assert at["eA"] == pytest.approx(-0.100, abs=3e-3)   # 车@ht − 冻结面 = −0.1m（核心一律用米）
    # 真实触球处 dy 归零；再晚 13.25ms 球已穿过拍面，dy 转负（口径与旧 HT err 同向）
    assert result["contact"]["dy"] * 1000 == pytest.approx(0.0, abs=2)
    assert result["contact"]["dtMs"] == pytest.approx(0.0, abs=0.5)
    assert result["late"]["dy"] * 1000 == pytest.approx(-53.0, abs=2)
    assert result["late"]["dtMs"] == pytest.approx(13.25, abs=0.5)
    assert result["far"] is None        # 评估点离拟合窗 >300ms 不外推
    assert result["miss"] is None       # 窗内无球观测不产出


def test_main_pc_truth_and_aim_use_last_target_prediction_ht():
    source = SRC.read_text(encoding="utf-8")
    assert "const targetPredHtBaseline=targetPred?rkToPc(targetPred.ht):null;" in source
    assert "const targetPredSamplePc=targetPred?pcSampleTimeForThrow(th,targetPred.ht):null;" in source
    assert "const targetTruth=targetPredSamplePc!=null?pcTruthAt(targetPredSamplePc):null;" in source
    assert "const truth=pcTruthAt(htPc);" not in source
    # 0805 起：全量RK重估的第三个 ht 不再单列，击球误差改成两列空间量——
    # dy 是时序误差的空间形态，dx/dz 是末次 target 对应预测击球点与真值的落点差（两侧同轴同基准）
    # raw预测/盲区列统一锚「最后进入update_ht的raw HT」；mode=2时它不是执行接触时刻。
    # 机械空间执行误差单独锚 /joint_states FK 实际穿 accepted 平面。
    assert "const gapFin=finalHt!=null?ballCarGapForThrow(th,finalHt):null;" in source
    assert "球面y−车y @臂最后更新HT<br>(cm, RK全量真值)</th>" in source
    assert "击球点@末次target预测 − RK全量真值@臂最后更新HT<br>dx/dz(cm, 世界轴)</th>" in source
    # 0809 起两列改 cm：算式一律留米，换算只在 cmSigned/cmFmt 显示层做
    assert "? (targetPred.worldX-targetPred.carX)-gapFin.dx : null;" in source
    assert "const aimDz=(gapFin&&targetPred&&isNum(targetPred.relZ))?targetPred.relZ-gapFin.dz:null;" in source
    assert "th.ref300Xw-th.ref300CarX" not in source
    assert "isNum(th.ref300Z))?th.ref300Z-gapFin.dz" not in source
    assert "+cmSigned(aimDx)+'/'+cmSigned(aimDz)+'</span>'" in source
    assert "+cmSigned(gapFin.dy)+'</span>'" in source
    assert "htAllS1ForThrow" not in source
    assert "hE300" not in source
    assert "<th>HT真实(触球)<br>(s,PC轴)</th>" not in source
    assert "HT err@300<br>" not in source
    assert "'<td>'+htPc.toFixed(3)+'</td>'" not in source
    assert "const truth=accHtPcSample!=null?pcTruthAt(accHtPcSample):null;" in source
    assert source.count("<td>'+pcTruthCell(targetTruth,true,targetPredSamplePc)+'</td>") == 2
    assert "<td>'+pcTruthCell(truth,false,accHtPcSample)+'</td>" in source
    # 主表第二列 PC 真值：同一套拟合，评估时刻换成raw臂最后更新HT。
    assert "const truthAcc=finalHtPcSample!=null?pcTruthAt(finalHtPcSample):null;" in source
    # 有/无末次 target 对应预测两条渲染路径都要出这一列（该列只依赖 accepted）
    assert source.count("pcTruthCell(truthAcc,true,finalHtPcSample)") == 2
    # 空值必须带原因：取值时刻要传进单元格，否则 "—" 分不清缺小车位姿、缺球观测还是拟合没过门
    assert "const pcTruthCell = (f,withY=false,tPc=null) => {" in source
    assert "if(!f) return pcTruthMissCell(tPc);" in source
    assert "PC 小车定位在该时刻缺失" in source
    assert "PC 球观测不足" in source
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


def test_rk300_table_merges_tcp_xyz_and_last_accepted_error():
    source = SRC.read_text(encoding="utf-8")
    exporter = TABLE_EXPORTER.read_text(encoding="utf-8")
    assert "const accepted=lastAcceptedForThrow(th);" in source
    assert "const accHt=accepted&&isNum(accepted.wht)?accepted.wht-RK.t0:null;" in source
    # 正常抛 TCP 锚 raw 最后更新HT；最后状态 mode=2 时五个数整体切到实际 accepted 平面过面。
    assert "const rawTcp=finalHt!=null?tcpAt(finalHt):null;" in source
    assert "tcpAt(accHt)" not in source
    assert "const tcpSample=accepted?armTcpFiveSampleAt(accepted,finalHt):null;" in source
    assert "const tcpUsesExecContact=!!(tcpSample&&tcpSample.usesExec);" in source
    assert "const tcpSampleT=tcpSample?tcpSample.t:null;" in source
    assert "const tcp=tcpSample?tcpSample.tcp:null;" in source
    assert "const tcpYawDeg=tcpSampleT!=null?botYawDegAt(tcpSampleT):null;" in source
    assert "const tcpWorld=tcp?[tcp[0]*Math.cos(tcpYawRad)-tcp[1]*Math.sin(tcpYawRad)," in source
    assert "tcp[0]*Math.sin(tcpYawRad)+tcp[1]*Math.cos(tcpYawRad)," in source
    assert "tcp[2]-(isNum(armZOff)?armZOff:0)]:null;" in source
    # dx/dz 与 xyz 共用 tcpWorld，并减 status tx/tz；raw world_x-car_pred_x 含 rel_y，禁止回潮。
    assert "const tcpAcceptedTarget=accepted&&isNum(accepted.tx)&&isNum(accepted.tz)" in source
    assert "tcpAcceptedTarget[0]*Math.cos(tcpYawRad)-tcpAcceptedTarget[1]*Math.sin(tcpYawRad)" in source
    assert "tcpAcceptedTarget[0]*Math.sin(tcpYawRad)+tcpAcceptedTarget[1]*Math.cos(tcpYawRad)" in source
    assert "const tcpAcceptedDx=tcpWorld&&tcpAcceptedTargetWorld" in source
    assert "const tcpAcceptedDz=tcpWorld&&tcpAcceptedTargetWorld" in source
    assert "tableXyzCm(tcpWorld[0],tcpWorld[1],tcpWorld[2])+', '" in source
    assert "tcpAcceptedError" not in source
    assert "(finalHt!=null?rkToPc(finalHt).toFixed(3):'—')+'s 已与有效profile脱钩" in source
    assert "各减同一个本场自标定提前量" in source
    assert "各减10ms" not in source
    assert source.count("<td>'+tcpCell+'</td>") == 2
    assert "accepted.wxw-accepted.wcarx" not in source
    assert "mode=2 Coast" in source
    assert "armPredictionMatchesAccepted(p,e.t+RK.t0,rec.tx,rec.tz,dur)" in source
    headers = [
        "PC取样 zPhase<br>offset(ms)</th>",
        "最后改target t / 对应ct<br>(s,global PC轴)</th>",
        "<th>对应预测 HT<br>(s,global PC轴)</th>",
        "<th>对应预测击球 rel_x/z(cm)</th>",
        "<th>车RUN末帧 目标−实际 dx/dy(cm)<br>(RK世界系)</th>",
        "末次target对应预测车@HT−RUN末实际 dx/dy(cm)<br>(RK世界系)</th>",
        # 表头带悬停：本场回配成功率 + 自标定出来的臂端三个量（x 比例/z 偏移/提前量）
        "机械臂最后accepted目标 x/z(cm)</th>",
        "<th>PC真值@对应预测HT+zPhase x/y/z(cm)</th>",
        "PC真值@臂最后更新HT+zPhase x/y/z(cm)</th>",
        "TCP−车心@臂最后更新HT x/y/z(cm,世界轴)<br>tcp−last accepted（dx，dz）</th>",
        "最后更新−挥拍起<br>(ms)</th>",
        "盲区 ht−ct@臂最后更新<br>(ms)</th>",
        "Δht 重定相<br>(ms)</th>",
        "车yaw@臂最后更新HT(°)</th>",
        "目标挥拍速度/pitch<br>(m/s, °)</th>",
        "拍面yaw,pitch,speed@臂最后更新HT(°,°,m/s;世界系)</th>",
        "拍面yaw,pitch,speed@臂最后更新HT−12ms(°,世界系)</th>",
    ]
    assert [source.index(header) for header in headers] == sorted(
        source.index(header) for header in headers
    )
    assert "TCP@执行过面−accepted" not in source
    # 两列拍面角锚在「臂最后更新 HT」（含挥拍中 ht 重定相消费的那条），不是最后一条 accepted
    assert "? (accepted.finalMismatch ? null : (isNum(accepted.finalHt)?accepted.finalHt-RK.t0:accHt))" in source
    assert "continuousSweep || e.t<=cur.done+0.05" in source
    assert "const continuousSweep=/\\bsweep_w=/.test(e.text);" in source
    assert "mode:continuousSweep?statusNum(e.text,'mode'):null" in source
    assert "faceAnglesWorldAt(finalHt)" in source
    assert "faceAnglesWorldPreAt(finalHt)" in source
    assert "faceAnglesWorldAt(accHt)" not in source
    assert "faceAnglesWorldPreAt(accHt)" not in source
    # 两列各自同时给 yaw 与 pitch（同一份窗内拟合），单元格文本为 yaw/pitch
    assert "tableSigned(faceYaw.deg)+'/'+tableSigned(faceYaw.pitch)" in source
    assert "tableSigned(faceYawPre.deg)+'/'+tableSigned(faceYawPre.pitch)" in source
    # 两列都另带实测拍速（灰字 m/s）：各自取值时刻处直接插值，**不**沿用 yaw/pitch 的窗内
    # 线性外推（S 曲线下拍速强非线性，外推会高估 40%+）；触球锚/σ/指令侧/J1 分量走悬停
    assert "const swingSpeed=racketSpeedAt(finalHt);" in source
    assert "const s=interpRow(armSpeedRows,t,RACKET_SPEED_MAX_GAP_S);" in source
    assert "fitFaceAnglesTo(accHtRk,accHtRk)" in source          # yaw/pitch 仍是窗内拟合外推
    assert "swingSpeed.v.toFixed(2)+'m/s'" in source
    assert "swingSpeed.cmd" in source
    # 触球锚只许由指令速度平台首帧定位；实测速度叠着 σ~0.5m/s 伺服振荡，
    # 窗内 argmax 不是触球探测器（0808 首版的"峰值即触球"伪迹，不许回潮）
    assert "swingSpeed.contactDt" in source and "swingSpeed.measContact" in source
    assert "swingSpeed.osc" in source
    assert "peakDt" not in source
    # 目标挥拍速度/pitch 列：只吃 accepted 状态自带的计划量，两条渲染路径都要出
    assert "const tgtSpeed=accepted&&isNum(accepted.tgtSpeed)?accepted.tgtSpeed:null;" in source
    assert "const tgtPitch=accepted&&isNum(accepted.tgtPitch)?accepted.tgtPitch:null;" in source
    assert "tgtSpeed:statusNum(e.text,'speed')" in source
    assert "tgtSpeedReq:statusNum(e.text,'speed_req')" in source
    assert source.count("<td>'+tgtSpeedCell+'</td>") == 2
    # 盲区列的 ct/ht 必须同源，且与主表末次 target 对应预测不是一回事
    assert "const finalCt=accepted&&isNum(accepted.finalCt)?accepted.finalCt-RK.t0:null;" in source
    # 回配失配（重定相生效但拿不到同源 ht/ct）必须出 ⚠— 而不是退回 accepted 冒充：
    # 0809 场 #6/#13 曾把 ~174ms 的真盲区显示成 327/306ms，且无任何提示
    assert "const blindBad=!!(accepted&&accepted.finalMismatch);" in source
    assert (
        "const blind=(finalHt!=null&&finalCt!=null&&!blindBad)?(finalHt-finalCt)*1000:null;"
        in source
    )
    assert "h.finalHt=null; h.finalCt=null; h.finalMismatch=true;" in source
    assert "⚠—" in source
    # 0803 已删列：视觉球拍@accepted HT（相对小车），其专用取值/单元格实现不应残留
    assert "视觉球拍" not in source
    assert "visualRacketAt" not in source
    assert "visualRacketCell" not in source
    # 车yaw@臂最后更新HT 列：/bot_state 瞬时值，锚同全表其余空间量列（0807 起统一臂最后更新HT）
    assert "const carYawAcc=finalHt!=null?botYawDegAt(finalHt):null;" in source
    assert "const carYawRate=finalHt!=null?imuYawRateDegAt(finalHt):null;" in source
    assert source.count("<td>'+carYawCell+'</td>") == 2
    # 角速度只能取 IMU 零滞后原值，禁止对有 0.3~0.5s 滞后的 bot_state yaw 数值求导
    assert "ys(RK.imu,'yaw_speed')" in source
    # HT−12ms 列（0808 由 −10ms 挪到 −12ms：指令平台首帧定位的臂内触球锚中位 −11ms）：
    # fy/fp 同窗拟合在该刻取值、拍速同刻插值；车 yaw 用 /bot_state 瞬时值（IMU 连续更新）
    assert "const FACE_YAW_PRE_S=0.012;" in source
    assert "racketSpeedRawAt(finalHt-FACE_YAW_PRE_S)" in source
    assert "swingSpeedPre.v.toFixed(2)+'m/s'" in source
    assert "fitFaceAnglesTo(accHtRk,tEval)" in source
    assert "botYawDegAt(tEval)" in source
    assert "ys(RK.bot,'yaw')" in source
    assert "<th>/tennis/status 字符串</th>" not in source
    assert "<th>/predict_hit_pos 字符串</th>" not in source
    # 旧车体系 TCP 列与 accepted HT 锚点命名不应残留（lead 分表的 PC真值@accepted HT 除外，
    # 那边评估的是 accepted 消息自身预测质量，锚它自己的 ht 语义正确）
    assert "const tcpWorld=tcp?[tcp[0],tcp[1],tcp[2]-(isNum(armZOff)?armZOff:0)]:null;" not in source
    assert "TCP@accepted HT" not in source
    assert "PC真值@accepted HT x/y/z" not in source
    assert "tableXyzCm(tcpWorld[0],tcpWorld[1],tcpWorld[2])" in source
    assert "(runEnd.tx-runEnd.x)*100" in source
    assert "(runEnd.ty-runEnd.y)*100" in source
    assert source.count("<td>'+runTargetError+'</td>") == 2
    assert "(targetPred.carX-runEnd.x)*100" in source
    assert "(targetPred.carY-runEnd.y)*100" in source
    assert "(th.ref300CarX-runEnd.x)*100" not in source
    assert "(th.ref300CarY-runEnd.y)*100" not in source
    assert source.count("<td>'+targetPredCarError+'</td>") == 2
    assert "const targetPredHt=targetPredHtBaseline!=null?targetPredHtBaseline.toFixed(3):'—';" in source
    assert "const targetPredHit=targetPred?tableXzCm(targetPred.relX,targetPred.relZ):'—';" in source
    assert source.count("<td>'+targetPredHt+'</td>") == 2
    assert source.count("<td>'+targetPredHit+'</td>") == 2
    assert "'<br><span style=\"color:#a0a0c0\">ct '" not in source
    assert "' <span style=\"color:#a0a0c0\">(ct '" in source
    assert '"rk300Tbl": "北极星表（末次 Target 对应预测 / PC 真值 / 臂执行）"' in exporter


@pytest.mark.parametrize("car,fp0", [("v03", 15.42), ("v04", 15.42)])
def test_add_face_angles_wiring_and_fk_properties(car, fp0):
    """拍面yaw,pitch,speed 三量的 Python 侧：_add_face_angles 正确附加 fy/fp/vt、跳过残缺关节；
    FK 性质：零位 fy=fp=0（拍面朝正前且不上仰）、J1 为垂直轴（Δfy=−Δq1 精确、fp 分毫不动
    ——这正是 pitch 列不需要减车 yaw 的依据）；vt 的解析 Jacobian 与数值差分一致，
    且只有 J1 转时退化成 |q̇1|·r（r=hypot(tcp_x,tcp_y)），即 status speed= 的口径。
    两台车逐条都要成立：拍面角在两车下恒等（旋转链一致），位置/拍速则不同。"""
    pytest.importorskip("numpy")
    import math
    import sys

    sys.path.insert(0, str(SRC.parent))
    try:
        from generate_curve3_html import _add_face_angles
        import extract_arm_bag as eab
    finally:
        sys.path.pop(0)
    eab.use_car(car)
    fk = eab.fk

    q = [0.1, -0.2, 0.3, 0.15, -0.4, 0.25]
    q1p = [0.3] + q[1:]
    qd = [1.3, -0.4, 0.7, 0.2, -0.9, 0.5]
    arm = {"states": [
        {"t": 1.0, "position": q},
        {"t": 1.1, "position": [0.1, None, 0.3, 0.15, -0.4, 0.25]},  # 关节缺失
        {"t": 1.2, "position": q[:5]},                               # 非 6 关节
        {"t": 1.3},                                                  # 无 position
        {"t": 1.4, "position": [0.0] * 6},
        {"t": 1.5, "position": q1p},
        {"t": 1.6, "position": q, "velocity": qd},
        {"t": 1.7, "position": q, "velocity": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"t": 1.8, "position": q, "velocity": [0.0] * 6},
        {"t": 1.9, "position": q, "velocity": [0.1, None, 0, 0, 0, 0]},  # 速度残缺
    ]}
    _add_face_angles(arm, car=car)
    s = arm["states"]
    assert arm["fk_car"] == car and arm["fk_source"].startswith(f"extract_arm_bag.fk({car})")
    assert isinstance(s[0].get("fy"), float)
    assert isinstance(s[0].get("fp"), float)
    assert all("fy" not in r and "fp" not in r for r in s[1:4])
    assert s[4]["fy"] == pytest.approx(0.0, abs=1e-6)          # 零位拍面朝正前
    assert s[4]["fp"] == pytest.approx(0.0, abs=1e-6)          # 零位拍面不上仰
    assert s[5]["fy"] - s[0]["fy"] == pytest.approx(           # J1 垂直轴：Δfy = −Δq1
        -math.degrees(0.2), abs=0.02
    )
    assert s[5]["fp"] == s[0]["fp"]                            # 纯 z 转不动 n_z ⇒ pitch 不变
    assert s[0]["fp"] == pytest.approx(fp0, abs=0.01)          # 该位形的拍面上仰角（回归锚）
    # vt：只在有 6 个数值 velocity 的帧上出现
    assert "vt" not in s[0] and "vt" not in s[5] and "vt" not in s[9]
    # 解析 Jacobian ≡ 数值差分（沿 q̇ 方向前推一小步的位移速率）
    step = 1e-6
    tcp0 = fk(q)["tcp"]
    tcp1 = fk([a + step * b for a, b in zip(q, qd)])["tcp"]
    numeric = math.dist(tcp1, tcp0) / step
    assert s[6]["vt"] == pytest.approx(numeric, abs=1e-4)      # vt 落盘保留 4 位（0.1mm/s）
    # 只有 J1 转时退化成 |q̇1|·r（触球瞬间的构造，也是 status speed= 的口径）
    lever = math.hypot(tcp0[0], tcp0[1])
    assert s[7]["vt"] == pytest.approx(2.0 * lever, abs=1e-4)
    assert s[8]["vt"] == 0.0                                   # 静止帧拍速为 0


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
def test_visual_racket_bracket_keeps_nearest_raw_before_and_after(tmp_path):
    core = _core("racket-bracket-core-begin", "racket-bracket-core-end")
    harness = (
        f"{core}\n"
        "const rows=[-0.040,-0.020,-0.005,0.010,0.030,0.040].map(t=>({t}));\n"
        "console.log(JSON.stringify({both:bracketVisualRacketRows(rows,0),"
        "beforeOnly:bracketVisualRacketRows([{t:-0.006}],0)}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert result["both"]["before"]["t"] == pytest.approx(-0.005)
    assert result["both"]["after"]["t"] == pytest.approx(0.010)
    assert result["beforeOnly"]["before"]["t"] == pytest.approx(-0.006)
    assert result["beforeOnly"]["after"] is None
