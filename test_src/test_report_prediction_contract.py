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
        "ht_rel:[1.2,1.6,1.405,1.49],"
        "rel_x:[99,10,20,30],rel_y:[99,11,21,31],rel_z:[199,110,120,130]}}};\n"
        "const rkPredStage=[0,1,1,1]; const rkPredNFit=[2,4,5,6];\n"
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
    assert throw["ref300NFit"] == 5
    assert throw["ref300Idx"] == 2


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
        "console.log(JSON.stringify({"
        "actual:armPredictionMatchesAccepted(actual,acceptT,acceptX,acceptZ,duration),"
        "adjacent:armPredictionMatchesAccepted(adjacent,acceptT,acceptX,acceptZ,duration)}));\n"
    )
    result = _run_node(tmp_path, harness)
    assert result == {"actual": True, "adjacent": False}


def test_pc_truth_uses_each_contracts_own_ht():
    source = SRC.read_text(encoding="utf-8")
    assert "const truth=pcTruthAt(htPc);" in source
    assert "const truth=pcTruthAt(accHtPc);" in source
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
        "<th>机械臂最后accepted目标 x/z(m)</th>",
        "<th>PC真值@RK HT x/z(m)</th>",
        "<th>TCP@accepted HT x/y/z(m)</th>",
        "<th>TCP−accepted dx/dz(cm)</th>",
    ]
    assert [source.index(header) for header in headers] == sorted(
        source.index(header) for header in headers
    )
    assert "<th>/tennis/status 字符串</th>" not in source
    assert "<th>/predict_hit_pos 字符串</th>" not in source


def test_rk300_table_shows_reject_reasons_only_without_accepted():
    source = SRC.read_text(encoding="utf-8")
    assert "const rejectNote=accepted?'—':rejectNoteForThrow(th);" in source
    assert "<th>备注</th>" in source
    assert "^reject hit: (.+)$" in source


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
