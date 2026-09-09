# -*- coding: utf-8 -*-
"""回归测试：RK≈300ms消息与机械臂最后accepted必须是两套独立合同。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
RUN_TRACKER = Path(__file__).resolve().parents[1] / "src" / "run_tracker.py"
RK_EXTRACTOR = Path(__file__).resolve().parent / "extract_rk_tracking_bag.py"
TABLE_EXPORTER = Path(__file__).resolve().parent / "export_report_tables.py"
NODE = shutil.which("node")
BOT_POSE_JS_DEPS = (
    "const lerp=(a,b,f)=>a+(b-a)*f;\n"
    "const nearest=(rows,t)=>rows.length?rows.reduce((a,b)=>Math.abs(b.t-t)<Math.abs(a.t-t)?b:a):null;\n"
    "const interpRow=(rows,t,maxGap)=>{let lo=0;while(lo<rows.length&&rows[lo].t<t)lo++;"
    "if(lo<=0||lo>=rows.length)return null;const a=rows[lo-1],b=rows[lo];"
    "return b.t-a.t<=maxGap?{a,b,f:(t-a.t)/(b.t-a.t)}:null;};\n"
)


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


def _embedded_report_data(html_path: Path) -> dict:
    source = html_path.read_text(encoding="utf-8")
    match = re.search(r"^const D = (.*);$", source, re.M)
    assert match
    return json.loads(match.group(1))


def test_racket_impact_sidecar_requires_explicit_v3_provenance(tmp_path):
    tracker = tmp_path / "tracker_demo.json"
    tracker.write_text(
        json.dumps({"config": {}, "summary": {}, "frames": []}),
        encoding="utf-8",
    )
    automatic = tmp_path / "tracker_demo_racket_impact.json"
    automatic.write_text(
        json.dumps({
            "schema": "racket_impact/v3",
            "control_usage": "record_only",
            "frame_time_semantics": "mosaic_group_mean_exposure_center_pc_perf_counter",
            "vz_semantics": "racket_head_bbox_center_world_velocity_proxy",
            "source": {"tracker_json": str(tracker)},
            "racket_impact": [
                {
                    "status": "accepted",
                    "contact_anchor_status": "accepted",
                    "vision_evaluated": True,
                    "contact_anchor_t_rk": 10.0,
                    "vz_world_mps": 1.25,
                }
            ]
        }),
        encoding="utf-8",
    )
    auto_html = tmp_path / "auto.html"
    auto_run = subprocess.run(
        [
            sys.executable, str(SRC), "--input", str(tracker),
            "--output", str(auto_html), "--no-tables",
        ],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert auto_run.returncode == 0, auto_run.stderr
    auto_data = _embedded_report_data(auto_html)
    assert "racket_impact" not in auto_data
    assert "racket_impact_json_path" not in auto_data["config"]

    explicit = tmp_path / "chosen.json"
    explicit.write_text(
        json.dumps({
            "schema": "racket_impact/v3",
            "control_usage": "record_only",
            "frame_time_semantics": "mosaic_group_mean_exposure_center_pc_perf_counter",
            "vz_semantics": "racket_head_bbox_center_world_velocity_proxy",
            "source": {"tracker_json": str(tracker)},
            "racket_impact": [
                {
                    "status": "accepted",
                    "contact_anchor_status": "accepted",
                    "vision_evaluated": True,
                    "contact_anchor_t_rk": 11.0,
                    "vz_world_mps": -2.5,
                }
            ]
        }),
        encoding="utf-8",
    )
    explicit_html = tmp_path / "explicit.html"
    explicit_run = subprocess.run(
        [
            sys.executable, str(SRC), "--input", str(tracker),
            "--racket-impact-json", str(explicit),
            "--output", str(explicit_html), "--no-tables",
        ],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert explicit_run.returncode == 0, explicit_run.stderr
    explicit_data = _embedded_report_data(explicit_html)
    assert explicit_data["racket_impact"][0]["vz_world_mps"] == pytest.approx(-2.5)
    assert explicit_data["config"]["racket_impact_json_path"] == str(explicit)


def test_racket_impact_sidecar_source_contract_is_explicit_and_provenance_locked():
    source = SRC.read_text(encoding="utf-8")
    assert '"control_usage": "record_only"' in source
    assert '"frame_time_semantics": "mosaic_group_mean_exposure_center_pc_perf_counter"' in source
    assert '"vz_semantics": "racket_head_bbox_center_world_velocity_proxy"' in source
    assert 'source.get("tracker_json")' in source
    assert "Path(source_tracker).resolve() != Path(tracker_json_path).resolve()" in source
    assert "impact_path = racket_impact_json_path" in source
    assert 'base + "_racket_impact.json"' not in source

    help_run = subprocess.run(
        [sys.executable, str(SRC), "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert help_run.returncode == 0, help_run.stderr
    impact_help = re.search(
        r"--racket-impact-json RACKET_IMPACT_JSON\s+(.+?)(?=\n\s+--|\Z)",
        help_run.stdout,
        re.S,
    )
    assert impact_help
    assert "自动探测" not in impact_help.group(1)


def test_post_run_does_not_reuse_a_stale_racket_impact_sidecar():
    source = RUN_TRACKER.read_text(encoding="utf-8")
    assert "elif racket_impact_candidate.exists()" not in source
    assert source.count("racket_impact_json_path = racket_impact_candidate") == 1
    assert (
        "and racket_impact_candidate.exists():\n"
        "                racket_impact_json_path = racket_impact_candidate"
    ) in source


def test_racket_impact_v2_sidecar_is_rejected_without_fallback(tmp_path):
    tracker = tmp_path / "tracker.json"
    tracker.write_text(
        json.dumps({"config": {}, "summary": {}, "frames": []}),
        encoding="utf-8",
    )
    sidecar = tmp_path / "v2.json"
    sidecar.write_text(
        json.dumps({
            "schema": "racket_impact/v2",
            "racket_impact": [],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "report.html"

    result = subprocess.run(
        [
            sys.executable, str(SRC), "--input", str(tracker),
            "--racket-impact-json", str(sidecar),
            "--output", str(output), "--no-tables",
        ],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )

    assert result.returncode != 0
    assert "schema mismatch" in result.stderr


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
        "const rkPredRacketVz=[null,1.1,9.0,9.1]; "
        "const rkPredCorXyEff=[.50,.55,.90,.91]; "
        "const rkPredCorEff=[.80,.81,.95,.96]; "
        "const rkPredCorMeasReplay=[null,null,.846,null]; "
        "const rkPredCxyMeasReplay=[null,null,.621,null]; "
        "const rkPredCorMeasClosureMs=[null,null,-7.38,null];\n"
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
    assert throw["lastS0Rvz"] == pytest.approx(1.1)
    assert throw["lastS0CorXyEff"] == pytest.approx(.55)
    assert throw["lastS0CorEff"] == pytest.approx(.81)
    assert throw["corMeasReplay"] == pytest.approx(.846)
    assert throw["cxyMeasReplay"] == pytest.approx(.621)
    assert throw["corMeasClosureMs"] == pytest.approx(-7.38)
    assert throw["corMeasIdx"] == 2


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_racket_impact_and_s0_cor_keep_new_measurement_separate_from_old_rk_use(tmp_path):
    """accepted 才可显示 vz；rejected 只显示原因且不能泄漏其 vz。"""
    core = _core("racket-cor-core-begin", "racket-cor-core-end")
    cell_match = re.search(
        r"const racketCorCellHtml = .*?\n};",
        SRC.read_text(encoding="utf-8"),
        re.S,
    )
    assert cell_match
    cell_core = cell_match.group(0)
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const tableEsc=s=>String(s);\n"
        "const RK={t0:100};\n"
        "const reject={firstT:.5,lastS0Rvz:null,lastS0CorXyEff:null,lastS0CorEff:null};\n"
        "const visionReject={firstT:1.5,lastS0Rvz:null,lastS0CorXyEff:null,lastS0CorEff:null};\n"
        "const a={firstT:2.5,lastS0Rvz:1.137,lastS0CorXyEff:.70,lastS0CorEff:.82,"
        "corMeasReplay:.846,cxyMeasReplay:.621,corMeasClosureMs:-7.38,corMeasIdx:12};\n"
        "const hole={firstT:3.5,lastS0Rvz:null,lastS0CorXyEff:.58,lastS0CorEff:.83};\n"
        "const b={firstT:5.0,lastS0Rvz:null,lastS0CorXyEff:.60,lastS0CorEff:.84};\n"
        "const c={firstT:8.0,lastS0Rvz:1.428,lastS0CorXyEff:.7034,lastS0CorEff:.8387};\n"
        "const d={firstT:11.0,lastS0Rvz:null,lastS0CorXyEff:null,lastS0CorEff:null};\n"
        "const invalidMissingVision={firstT:12.5,lastS0Rvz:null,lastS0CorXyEff:null,lastS0CorEff:null};\n"
        "const invalidContradictory={firstT:14.0,lastS0Rvz:null,lastS0CorXyEff:null,lastS0CorEff:null};\n"
        "const reportThrows=[reject,visionReject,a,hole,b,c,d,invalidMissingVision,invalidContradictory];\n"
        "const evidence=v=>({accepted:true,reason:'accepted',"
        "observation_semantics:'racket_head_bbox_geometric_center_native_pixel',"
        "velocity_semantics:'racket_head_bbox_center_world_velocity_proxy',"
        "bbox_center_vz_world_mps:v,raw_bbox_observations:[{},{},{},{},{},{}],"
        "bundle_diagnostics:{accepted:true,reason:'accepted',"
        "observation_semantics:'racket_head_bbox_geometric_center',"
        "bbox_center_vz_world_mps:v,bbox_center_velocity_world_mps:[0,0,v],"
        "supported_frames:[10,11,12],fit_span_s:.08,max_reprojection_error_px:2}});\n"
        "const D={config:{cor_xy:9.99},racket_impact:["
        "{status:'rejected',contact_anchor_status:'rejected',vision_evaluated:false,"
        "contact_anchor_t_rk:100.05,vz_world_mps:99,failure_reason:'锚点拒绝'},"
        "{status:'rejected',contact_anchor_status:'accepted',vision_evaluated:true,"
        "contact_anchor_t_rk:101.05,vz_world_mps:99,failure_reason:'no_bundle_inlier_model',"
        "acceptance_mode:'strict',prefix_spread_s:.002,contact_point_spread_m:.03,"
        "measurement:{rejection_counts:{no_bbox:2},bundle_diagnostics:"
        "{supported_frames:[10,11,12],fit_span_s:.08,max_reprojection_error_px:7.2,"
        "leave_one_frame_bbox_center_vz_mps:[1.0,.9,1.1]}}},"
        "{status:'accepted',contact_anchor_status:'accepted',vision_evaluated:true,"
        "contact_anchor_t_rk:102.10,vz_world_mps:1.137,"
        "vz_semantics:'racket_head_bbox_center_world_velocity_proxy',"
        "acceptance_mode:'strict',measurement:evidence(1.137)},"
        "{status:'accepted',contact_anchor_status:'accepted',vision_evaluated:true,"
        "contact_anchor_t_rk:104.60,vz_world_mps:.10},"
        "{status:'accepted',contact_anchor_status:'accepted',vision_evaluated:true,"
        "contact_anchor_t_rk:107.60,vz_world_mps:-.896,"
        "vz_semantics:'racket_head_bbox_center_world_velocity_proxy',"
        "measurement:evidence(-.896)},"
        "{status:'rejected',contact_anchor_status:'accepted',"
        "contact_anchor_t_rk:112.50,vz_world_mps:99,failure_reason:'missing_vision_flag'},"
        "{status:'accepted',contact_anchor_status:'rejected',vision_evaluated:true,"
        "contact_anchor_t_rk:114.00,vz_world_mps:99}]};\n"
        f"{core}\n"
        f"{cell_core}\n"
        "const assigned=racketImpactAssignments();\n"
        "const result=reportThrows.map(th=>racketCorForThrow(th,assigned));\n"
        "const rejectedCell=racketCorCellHtml(reject,assigned);\n"
        "const visionRejectedCell=racketCorCellHtml(visionReject,assigned);\n"
        "const invalidMissingVisionCell=racketCorCellHtml(invalidMissingVision,assigned);\n"
        "const invalidContradictoryCell=racketCorCellHtml(invalidContradictory,assigned);\n"
        "const acceptedCell=racketCorCellHtml(a,assigned);\n"
        "const labels=[.301,.30,-.30,-.301,0].map(racketMotionForVz);\n"
        "D.racket_impact=undefined; const legacy=racketCorForThrow(a,racketImpactAssignments());\n"
        "console.log(JSON.stringify({result,rejectedCell,visionRejectedCell,invalidMissingVisionCell,invalidContradictoryCell,acceptedCell,labels,legacy}));\n"
    )
    out = _run_node(tmp_path, harness)

    (
        contact_rejected, vision_rejected, first, hole, level, downward, missing,
        invalid_missing_vision, invalid_contradictory,
    ) = out["result"]
    assert contact_rejected["status"] == "contact_rejected"
    assert contact_rejected["measuredVz"] is None
    assert contact_rejected["motion"] is None
    assert contact_rejected["failureReason"] == "锚点拒绝"
    assert out["rejectedCell"].endswith(
        ">— / S0 cxy —, cor_z — / 实测反弹 cxy —, cor_z — / RK旧拍头vz —</span>"
    )
    assert "+99.000m/s" not in out["rejectedCell"]
    assert "vision_evaluated=false" in out["rejectedCell"]

    assert vision_rejected["status"] == "vision_rejected"
    assert vision_rejected["measuredVz"] is None
    assert vision_rejected["acceptanceMode"] == "strict"
    assert vision_rejected["prefixSpread"] == pytest.approx(.002)
    assert vision_rejected["contactPointSpread"] == pytest.approx(.03)
    assert vision_rejected["reprojectionMax"] == pytest.approx(7.2)
    assert "+99.000m/s" not in out["visionRejectedCell"]
    assert "contact_acceptance_mode=strict" in out["visionRejectedCell"]
    assert "supported_frames=" in out["visionRejectedCell"]
    assert "bundle_fit_span_ms=80.00" in out["visionRejectedCell"]
    assert "bundle_max_reprojection_error_px=7.20" in out["visionRejectedCell"]
    assert "leave_one_frame_bbox_center_vz_mps=" in out["visionRejectedCell"]
    assert "rejection_counts=" in out["visionRejectedCell"]

    assert invalid_missing_vision["status"] == "invalid_v3_row"
    assert invalid_missing_vision["measuredVz"] is None
    assert "+99.000m/s" not in out["invalidMissingVisionCell"]
    assert "status=rejected" in out["invalidMissingVisionCell"]
    assert "vision_evaluated=undefined" in out["invalidMissingVisionCell"]
    assert "状态组合无效" in out["invalidMissingVisionCell"]

    assert invalid_contradictory["status"] == "invalid_v3_row"
    assert invalid_contradictory["measuredVz"] is None
    assert "+99.000m/s" not in out["invalidContradictoryCell"]
    assert "status=accepted" in out["invalidContradictoryCell"]
    assert "contact_anchor_status=rejected" in out["invalidContradictoryCell"]
    assert "状态组合无效" in out["invalidContradictoryCell"]
    assert (first["status"], first["motion"], first["measuredVz"]) == (
        "measured", "拍头上行", pytest.approx(1.137)
    )
    assert first["spinType"] == "上旋倾向"
    assert out["acceptedCell"].endswith(
        ">上旋倾向 +1.137 / S0 cxy 0.7000, cor_z 0.8200 / "
        "实测反弹 cxy 0.621, cor_z 0.846 / RK旧拍头vz +1.137</span>"
    )
    assert first["corXyEff"] == pytest.approx(.70)
    assert first["corEff"] == pytest.approx(.82)
    assert first["corMeasReplay"] == pytest.approx(.846)
    assert first["cxyMeasReplay"] == pytest.approx(.621)
    assert first["corMeasClosureMs"] == pytest.approx(-7.38)
    assert first["timeSource"] == "contact_anchor"
    assert first["matchedTRk"] == pytest.approx(102.10)
    assert first["anchorTRk"] == pytest.approx(102.10)
    assert hole["status"] == "missing"  # 后一 contact 不得按数组序号错配到这个空抛
    assert hole["corXyEff"] == pytest.approx(.58)
    assert level["status"] == "invalid_accepted_evidence"
    assert level["motion"] is None
    assert level["measuredVz"] is None
    assert level["timeSource"] == "contact_anchor"
    assert level["matchedTRk"] == pytest.approx(104.60)
    assert downward["motion"] == "拍头下行"
    assert downward["spinType"] == "下旋倾向"
    assert downward["measuredVz"] == pytest.approx(-.896)
    assert downward["usedRvz"] == pytest.approx(1.428)
    assert downward["corXyEff"] == pytest.approx(.7034)
    assert downward["corEff"] == pytest.approx(.8387)
    assert missing["status"] == "missing"
    assert missing["corXyEff"] is None  # 不从 D.config.cor_xy 回填
    assert missing["corEff"] is None
    assert out["labels"] == ["拍头上行", "拍头上行", "拍头下行", "拍头下行", "近水平"]
    assert out["legacy"]["status"] == "legacy_untrusted"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_last_target_change_matches_prediction_by_deadline_not_ct(tmp_path):
    """末次坐标变化用 bot.t+remaining ↔ pred.ht 回配；更晚 ct 的 deadline-only 消息不能抢走。"""
    run_core = _core("bot-run-end-core-begin", "bot-run-end-core-end")
    pred_core = _core("last-target-pred-core-begin", "last-target-pred-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        f"{BOT_POSE_JS_DEPS}"
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
        "const armAligned=true;\n"
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
    assert h["reswing"]["remain"] == pytest.approx(0.050)
    assert h["reswing"]["ok"] is True
    assert h["finalHt"] == pytest.approx(10.460)
    assert h["finalCt"] == pytest.approx(10.35)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_arm_point_world_contract_uses_yaw_and_ground_height(tmp_path):
    """x/y 按 yaw 旋转；z 从 FK 安装面换到地面，缺任一变换量就不出数。"""
    script = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        + _core("arm-point-world-core-begin", "arm-point-world-core-end")
        + "\nconst tcp=armPointWorld([2,1,3],90,-0.5);\n"
          "const accepted=[2,4];\n"
          "const error=[(tcp[0]-accepted[0])*100,(tcp[2]-accepted[1])*100];\n"
          "const missingYaw=armPointWorld([2,1,3],null,-0.5);\n"
          "const missingZOffset=armPointWorld([2,1,3],90,null);\n"
          "console.log(JSON.stringify({tcp,accepted,error,missingYaw,missingZOffset}));\n"
    )
    result = _run_node(tmp_path, script)

    assert result["tcp"] == pytest.approx([-1, 2, 3.5])
    assert result["accepted"] == [2, 4]
    assert result["error"] == pytest.approx([-300, -50])
    assert result["missingYaw"] is None
    assert result["missingZOffset"] is None


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






@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_bot_run_end_is_normal_one_to_one_and_uses_target_minus_actual(tmp_path):
    core = _core("bot-run-end-core-begin", "bot-run-end-core-end")
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        f"{BOT_POSE_JS_DEPS}"
        "const duplicate={ht:2.06}, first={ht:2.005}, second={ht:5.005}, stale={ht:8.0};\n"
        "const reportThrows=[duplicate,first,second,stale];\n"
        "const RK={bot:{t:[1.9,2.0,2.01,4.9,5.0,5.01,7.9,8.0,8.01],y:{"
        "imu_t:[1.88,1.98,2.03,4.88,4.98,5.03,7.88,7.98,null],"
        "phase:['RUN','RUN','BRAKE_IN_SWING','RUN','RUN','BRAKE_IN_SWING','RUN','RUN','BRAKE_AFTER_SWING'],"
        "x:[0,1,2,0,3,4,0,5,6],y:[0,2,3,0,4,5,0,6,7],"
        "target_x:[0,1.1,null,0,3.05,null,0,5.1,null],"
        "target_y:[0,1.8,null,0,4.07,null,0,6.1,null],"
        "remaining:[.2,.005,null,.2,.005,null,.2,.005,null]}}};\n"
        f"{core}\n"
        "const a=botRunEndForThrow(first), b=botRunEndForThrow(second);\n"
        "const ap=botPoseAtImuTime(a.targetChange.deadline), bp=botPoseAtImuTime(b.targetChange.deadline);\n"
        "console.log(JSON.stringify({duplicate:botRunEndForThrow(duplicate),"
        "first:{t:a.t,ht:a.targetChange.deadline,x:ap.x,y:ap.y,dx:(a.tx-ap.x)*100,dy:(a.ty-ap.y)*100},"
        "second:{t:b.t,ht:b.targetChange.deadline,x:bp.x,y:bp.y,dx:(b.tx-bp.x)*100,dy:(b.ty-bp.y)*100},"
        "wide:botPoseAtImuTime(2.10),missing:botPoseAtImuTime(0),stale:botRunEndForThrow(stale)}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert result["duplicate"] is None
    assert result["first"] == pytest.approx(
        {"t": 2.0, "ht": 2.005, "x": 1.5, "y": 2.5, "dx": -40.0, "dy": -70.0}
    )
    assert result["second"] == pytest.approx(
        {"t": 5.0, "ht": 5.005, "x": 3.5, "y": 4.5, "dx": -45.0, "dy": -43.0}
    )
    assert result["wide"] is None
    assert result["missing"] is None
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
        f"bot:{{t:{json.dumps([round(t + 0.02, 5) for t in bt_list])},y:{{"
        f"imu_t:{json.dumps(bt_list)},x:{json.dumps(bx_list)},y:{json.dumps(by_list)}}}}}}};\n"
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


def test_main_pc_truth_columns_anchor_pre300ht_and_finalht():
    """0904 起北极星表只剩两个预测锚：Pre300HT 与 FinalHT，PC 真值各在其 ht(+本抛 zPhase) 上取样。"""
    source = SRC.read_text(encoding="utf-8")
    assert "const finalHt=fin?fin.ht-RK.t0:null;" in source
    assert "const finalHtPcSample=finalHt!=null?pcSampleTimeForThrow(th,finalHt):null;" in source
    assert "const pre300Ht=pre?pre.ht-RK.t0:null;" in source
    assert "const pre300HtPcSample=pre300Ht!=null?pcSampleTimeForThrow(th,pre300Ht):null;" in source
    assert "const truthFin=finalHtPcSample!=null?pcTruthAt(finalHtPcSample):null;" in source
    assert "const truthPre=pre300HtPcSample!=null?pcTruthAt(pre300HtPcSample):null;" in source
    assert "pcTruthCell(truthPre,true,pre300HtPcSample)" in source
    assert "pcTruthCell(truthFin,true,finalHtPcSample)" in source
    assert "PC真值@Pre300HT+zPhase<br>x/y/z(cm)<br>(y为球接触面)</th>" in source
    assert "PC真值@FinalHT+zPhase<br>x/y/z(cm)<br>(y为球接触面)</th>" in source
    # 旧锚（末次 target 对应预测 HT / 臂最后更新HT / accepted HT）与随之撤下的列不得残留在主表
    for gone in ("targetPredHtBaseline", "targetPredSamplePc", "targetTruth", "truthAcc",
                 "PC真值@对应预测HT", "PC真值@臂最后更新HT", "球面y−车y<br>@臂最后更新HT",
                 "击球点@末次target预测", "const gapFin=", "hitPlaneShift", "htAllS1ForThrow", "hE300",
                 "<th>HT真实(触球)<br>(s,PC轴)</th>", "HT err@300<br>", "'<td>'+htPc.toFixed(3)+'</td>'"):
        assert gone not in source, gone
    # 空值必须带原因：取值时刻要传进单元格，否则 "—" 分不清缺小车位姿、缺球观测还是拟合没过门
    assert "const pcTruthCell = (f,withY=false,tPc=null) => {" in source
    assert "const R_BALL=0.033;" in source
    assert "const yValue=f.y-R_BALL;" in source
    assert "y=(球心world_y−R球3.3cm)−车体中心world_y" in source
    assert "y 均显示球接触面" in source
    assert "if(!f) return pcTruthMissCell(tPc);" in source
    assert "PC 小车定位在该时刻缺失" in source
    assert "PC 球观测不足" in source
    # Arm Accepted 分表仍评估 accepted 消息自身的预测质量，锚它自己的 ht
    assert "const truth=accHtPcSample!=null?pcTruthAt(accHtPcSample):null;" in source
    assert "<td>'+pcTruthCell(truth,false,accHtPcSample)+'</td>" in source
    # 0803 起已删列（开始触球/球×车相交/旧 preHt）不应残留；Pre300HT 是 0904 新口径，变量名不含旧 preHt
    for gone in ("touchT", "pcMeetTrueAt", "TOUCH_DWELL_LEAD_S", "开始触球", "球×车相交", "preHt", "hitTableHtml"):
        assert gone not in source, gone
    assert "y:fy[0]-c.y" in source
    assert "x:fx[0]-c.x" in source
    coord_note = "PC真值采用世界坐标轴，不随车体 yaw 旋转。x = 拟合球心 world_x − 同时刻插值车体中心 world_x"
    assert source.index('id="p5"') < source.index(coord_note) < source.index('id="rk300Tbl"')


def test_rk300_table_anchors_everything_on_finalht():
    """0904 用户定：北极星表全表唯一时间锚 = FinalHT（[[final-ht-core]]），列集固定 16 列。"""
    source = SRC.read_text(encoding="utf-8")
    exporter = TABLE_EXPORTER.read_text(encoding="utf-8")
    table = source.split("const rk300TableHtml = () => {")[1].split("const armAcceptedTableHtml = () => {")[0]
    assert "const fin=finalTargetForThrow(th);" in table
    assert "const pre=pre300ForThrow(th,fin);" in table
    assert "const accepted=fin?fin.accepted:lastAcceptedForThrow(th);" in table
    assert "const RL_TARGET_FREEZE_LEAD_S=0.030;" in source
    assert "const tRef=fin.ht-REF_LEAD_TARGET;" in source
    # TCP 只锚 FinalHT；不另找过面
    assert "const tcp=finalHt!=null?tcpAt(finalHt):null;" in table
    assert "tcpAt(accHt)" not in source
    assert "armExecutionContactAt" not in source
    assert "armTcpFiveSampleAt" not in source
    assert "const tcpYawDeg=carYawAcc;" in table
    assert "const tcpWorld=armPointWorld(tcp,tcpYawDeg,armConstCal.zOff);" in table
    assert "p[2]-zArmMinusWorld" in source
    assert "armZOff" not in source
    # 臂目标：RL=FinalHT 消息自身 rel_x/rel_z，规则=最后 accepted 的（late 只改 ht）；目标侧不做任何变换
    assert "const aimIsFinal=!!(fin&&(fin.source==='rl_status'||fin.source==='rl_recon'));" in table
    assert "const aim=aimIsFinal?[fin.relX,fin.relZ]" in table
    assert ":(accepted&&isNum(accepted.wx)&&isNum(accepted.wz)?[accepted.wx,accepted.wz]:null);" in table
    assert "const tcpAimDx=tcpWorld&&aim?(tcpWorld[0]-aim[0])*100:null;" in table
    assert "const tcpAimDz=tcpWorld&&aim?(tcpWorld[2]-aim[1])*100:null;" in table
    assert "直接取上游 /predict_hit_pos rel_x/z=(" in table
    assert "z_world=z_FK−zOffset=" in table
    assert "zOffset=臂模型z−世界z" in table
    assert "tcp−目标 dx/dz=" in table
    assert "tableXyzCm(tcpWorld[0],tcpWorld[1],tcpWorld[2])+', '" in table
    assert table.count("<td>'+tcpCell+'</td>") == 1
    assert "accepted.wxw-accepted.wcarx" not in source
    assert "armPredictionMatchesAccepted(p,e.t+rkT0,rec.tx,rec.tz,dur)" in source
    headers = [
        "末次target目标−实际车@FinalHT<br>dx/dy(cm)<br>(RK世界系)</th>",
        "末次target对应预测车<br>−实际车@FinalHT dx/dy(cm)<br>(RK世界系)</th>",
        "Pre300HT 预测击球点<br>rel_x/z(cm) @ht</th>",
        "FinalHT 预测击球点<br>rel_x/z(cm) @ht</th>",
        "Pre300HT−FinalHT<br>dx/dz(cm) Δht(ms)</th>",
        "PC真值@Pre300HT+zPhase<br>x/y/z(cm)<br>(y为球接触面)</th>",
        "PC真值@FinalHT+zPhase<br>x/y/z(cm)<br>(y为球接触面)</th>",
        "TCP@FinalHT x/y/z(cm,世界轴)<br>相对机械臂中心地面点z=0<br>tcp−臂目标（dx，dz）</th>",
        "视觉拍心−车心@FinalHT+zPhase附近<br>x/y/z(cm,世界轴)<br>人工轨迹或最近前/后＋ht前逐帧；视觉−同曝光TCP（dx，dy，dz）</th>",
        "车yaw@FinalHT<br>(°)</th>",
        "目标挥拍速度/yaw/pitch<br>(m/s, °, °)</th>",
        "拍面yaw,pitch / 世界拍心speed<br>@FinalHT<br>(°,°,m/s;世界系)</th>",
        "拍面yaw,pitch / 世界拍心speed<br>@FinalHT−12ms<br>(°,°,m/s;世界系)</th>",
        "PC回球<br>yaw/俯仰(°) / speed / Δt(ms)</th>",
        "实测 e_n,eff<br>(拍面法向)</th>",
    ]
    assert [table.index(h) for h in headers] == sorted(table.index(h) for h in headers)
    assert len(re.findall(r"<th[ >]", table)) == len(headers) + 1   # 加 # 列，恰 16 列（<thead> 不算）
    for gone in ("机械臂最后accepted目标<br>x/z(cm)</th>", "PC真值@对应预测HT+zPhase", "最后更新−挥拍起<br>(ms)</th>",
                 "盲区 ht−ct<br>@臂最后更新HT", "Δht 重定相<br>(ms)</th>", "目标规划出球<br>", "<th>消息</th>",
                 "<th>备注</th>", "PC视频判型 / S0 cxy, cor_z / 实测反弹 cxy, cor_z / RK旧拍头vz</th>",
                 "TCP@执行过面−accepted", "臂最后更新HT", "accHt"):
        assert gone not in table, gone
    # 规则模式的 FinalHT 仍来自 _armHit 的重定相重建（最后一条 late ht saved 的同源 ht/ct）
    assert "const ct=isNum(accepted.finalCt)?accepted.finalCt:accepted.wct;" in source
    assert "source=viaLate?'late_saved':'accepted';" in source
    assert "continuousSweep || e.t<=cur.done+0.05" in source
    assert "const continuousSweep=/\\bsweep_w=/.test(e.text);" in source
    assert "mode:continuousSweep?statusNum(e.text,'mode'):null" in source
    assert "h.finalHt=null; h.finalCt=null; h.finalMismatch=true;" in source
    # 拍面/拍速/车 yaw 全部锚 FinalHT
    assert "faceAnglesWorldAt(finalHt)" in table
    assert "faceAnglesWorldPreAt(finalHt)" in table
    assert "faceAnglesWorldAt(accHt)" not in source
    assert "faceAnglesWorldPreAt(accHt)" not in source
    assert "const botYaw=botYawDegAt(tEval);" in source
    assert "pcCarYawAt" not in source
    assert "faceYaw.carYaw" not in source
    assert "tableSigned(faceYaw.deg)+'/'+tableSigned(faceYaw.pitch)" in table
    assert "tableSigned(faceYawPre.deg)+'/'+tableSigned(faceYawPre.pitch)" in table
    assert "const swingSpeed=racketSpeedAt(finalHt);" in table
    assert "const s=interpRow(armSpeedRows,t,RACKET_SPEED_MAX_GAP_S);" in source
    assert "const fit=fitFaceAnglesTo(accHtRk,tEval);" in source
    assert "swingSpeed.speedWorld.toFixed(2)+'m/s'" in table
    for k in ("swingSpeed.speedArm", "swingSpeed.speedCar", "swingSpeed.speedTurn",
              "swingSpeed.vJ1-swingSpeed.cmdJ1", "swingSpeed.vJ1-tgtSpeed",
              "swingSpeed.contactDt", "swingSpeed.measContactJ1", "swingSpeed.oscJ1"):
        assert k in table, k
    assert "peakDt" not in source
    # 目标三量：RL 用 FinalHT 目标（rl_swing done 状态），规则用最后 accepted 计划量；yaw=−δ（世界系）
    assert "const doneNum=k=>done?statusNum(done.text,k):null;" in table
    assert "const tgtSpeed=isNum(doneNum('speed_req'))?doneNum('speed_req')" in table
    assert ":(accepted&&isNum(accepted.tgtSpeed)?accepted.tgtSpeed:null);" in table
    assert "const tgtYawWorldDeg=tgtYawExtraDeg!=null?-tgtYawExtraDeg" in table
    assert "yawExtra:Number(p.hit_yaw_extra), carYaw:Number(p.car_yaw)" in source
    assert "tgtSpeed:statusNum(e.text,'speed')" in source
    assert "tgtSpeedReq:statusNum(e.text,'speed_req')" in source
    assert table.count("<td>'+tgtCell+'</td>") == 1
    # 车yaw@FinalHT 列：/bot_state yaw 按它自己的 imu_t 物理轴严格夹取
    assert "const carYawAcc=finalHt!=null?botYawDegAt(finalHt):null;" in table
    assert "const carYawRate=finalHt!=null?imuYawRateDegAt(finalHt):null;" in table
    assert table.count("<td>'+carYawCell+'</td>") == 1
    # 世界拍速三条车体合同：vx/vy/yaw 走 bot_state imu_t；角速度走已回拨反馈延时的 IMU t
    assert "ys(RK.imu,'yaw_speed')" in source
    assert "const T=ys(RK.bot,'imu_t')||[], VX=ys(RK.bot,'vx'), VY=ys(RK.bot,'vy')" in source
    assert "const vCar=botVelocityAt(t), yawDeg=botYawDegAt(t), yawRate=imuYawRateAt(t);" in source
    assert "CA.bot_center.params.arm_forward_offset_m" in source
    # yaw 刚体项的杠杆必须是拍心点 p_head 本身（速度算的是哪个点，杠杆就得是哪个点）
    assert "racketWorldVelocity(vArm,head,vCar,yawDeg,yawRate,armForwardOffsetM)" in source
    assert "const head=[0,1,2].map(k=>lerp(s.a.p_head[k],s.b.p_head[k],s.f));" in source
    assert "不回退 arm-only" in source
    assert "const FACE_YAW_PRE_S=0.012;" in source
    assert "racketSpeedRawAt(finalHt-FACE_YAW_PRE_S)" in table
    assert "swingSpeedPre.speedWorld.toFixed(2)+'m/s'" in table
    assert "fitFaceAnglesTo(accHtRk,tEval)" in source
    assert "botYawDegAt(tEval)" in source
    assert "ys(RK.bot,'yaw')" in source
    assert "const T=ys(RK.bot,'imu_t')||[], Y=ys(RK.bot,'yaw')" in source
    for gone in ("<th>/tennis/status 字符串</th>", "<th>/predict_hit_pos 字符串</th>", "TCP−车心@臂最后更新HT",
                 "z−臂系z偏移", "臂基≡车心", "TCP@accepted HT", "PC真值@accepted HT x/y/z", "视觉球拍",
                 "visualRacketAt", "visualRacketCell"):
        assert gone not in source, gone
    # 车侧两列：末次 raw target / 末次target对应预测车，都减 FinalHT 时刻的实际车（imu_t 有界插值）
    assert "const t=ts(RK.bot), phase=ys(RK.bot,'phase');" in source
    assert "deadline:isNum(remaining[i])?t[i]+remaining[i]:null" in source
    assert "const actualAtFinal=finalHt!=null?botPoseAtImuTime(finalHt):null;" in table
    assert "(runEnd.tx-actualAtFinal.x)*100" in table
    assert "(runEnd.ty-actualAtFinal.y)*100" in table
    assert "(targetPred.carX-actualAtFinal.x)*100" in table
    assert "(targetPred.carY-actualAtFinal.y)*100" in table
    assert table.count("<td>'+targetActualCell+'</td>") == 1
    assert table.count("<td>'+predActualCell+'</td>") == 1
    assert "actualAtTargetHt" not in source
    assert "(th.ref300CarX-runEnd.x)*100" not in source
    # 两个预测点格 + 差值格
    assert "const preCell=predPointCell(pre,'Pre300HT','');" in table
    assert "const finCell=predPointCell(fin,'FinalHT',fin?' ['+srcLabel[fin.source]+']':'');" in table
    assert "tableSigned((pre.relX-fin.relX)*100)+'/'+tableSigned((pre.relZ-fin.relZ)*100)" in table
    assert "Δht'+tableSigned((pre.ht-fin.ht)*1000)+'ms" in table
    # 视频判型 / 反弹 COR 列已从主表撤下（实现保留给 racket-cor 单测）；e_n 列保留并锚 FinalHT
    assert "<td>'+racketCorCell+'</td>" not in source
    assert table.count("<td>'+enCell+'</td>") == 1
    assert "const enHitRk=ret&&finalHt!=null&&finalHtPcSample!=null" in table
    assert "finalHt+(ret.tHit-finalHtPcSample)" in table
    assert "faceAnglesWorldAt(finalHt,enHitRk)" in table
    assert "racketSpeedRawAt(enHitRk)" in table
    assert "racketNormalRestitution(" in table
    assert "[ret.incoming.vx,ret.incoming.vy,ret.incoming.vz]" in table
    assert "[ret.vx,ret.vy,ret.vz],enRacket.world,enNormal" in table
    assert "不是材料常数，也不是落地 cor_z/cxy 或世界y等效e" in table
    # 合同导出带挥拍模式与 FinalHT 来源，schema 不变（racket_ht_black_marker 只认 v4）
    contract = source.split("const publishArmHitContract = rows => {")[1]         .split("const rk300TableHtml = () => {")[0]
    assert "schema:'arm_final_ht/v4'," in contract
    assert "swingMode:ARM_SWING_MODE.mode," in contract
    assert "finalHtSource:fin?fin.source:null," in table
    assert "finalHtRkAbs:fin?fin.ht:null," in table
    # 0 抛的场次也发合同（rows=[]）：下游靠"有没有合同"区分报告太旧/页面挂了 与 本场没有抛
    assert "if(!reportThrows.length){ publishArmHitContract([]); return ''; }" in table
    assert "publishArmHitContract(armContractRows);" in table
    # racket-cor 实现合同不变（列撤下但函数留给单测与将来复用）
    cell = re.search(r"const racketCorCellHtml = .*?\n};", source, re.S)
    assert cell
    assert "<br>" not in cell.group(0)
    assert "impactText+' / S0 cxy '+corXyText+', cor_z '+corText" in cell.group(0)
    assert "' / 实测反弹 cxy '+measuredCxyText+', cor_z '+measuredCorText" in cell.group(0)
    assert "' / RK旧拍头vz '+usedValue" in cell.group(0)
    assert "cor_xy=vxy_out/vxy_in=" in source
    assert "cor_z=−vz_out/vz_in=" in source
    assert '"rk300Tbl": "北极星表（Pre300HT / FinalHT 预测 / PC 真值 / 臂执行）"' in exporter


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_racket_world_speed_is_vector_sum_with_car_center_offset(tmp_path):
    """世界拍速必须合成三维向量；覆盖反向抵消与 V04 臂座前移的纯 yaw 速度。"""
    harness = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        + _core("arm-point-world-core-begin", "arm-point-world-core-end") + "\n"
        + _core("racket-world-speed-core-begin", "racket-world-speed-core-end") + "\n"
        + "const mixed=racketWorldVelocity([1,2,3],[0.4,-0.2,1],[0.5,-0.3],90,2,0.045);\n"
        + "const staticCar=racketWorldVelocity([3,4,12],[0.4,-0.2,1],[0,0],37,0,0.045);\n"
        + "const cancel=racketWorldVelocity([2,0,0],[0.4,0,1],[-1,0],0,0,0.045);\n"
        + "const offsetYaw=racketWorldVelocity([0,0,0],[0,0,1],[0,0],0,2,0.045);\n"
        + "const missing=racketWorldVelocity([1,0,0],[0,0,1],[0,0],0,0,null);\n"
        + "const n0=faceNormalWorld(0,0), n90=faceNormalWorld(90,0);\n"
        + "const en=racketNormalRestitution([0,-4,0],[0,5,0],[0,2,0],n0);\n"
        + "const enShift=racketNormalRestitution([1,-1,-2],[1,8,-2],[1,5,-2],n0);\n"
        + "const enOblique=racketNormalRestitution([-4,2,0],[5,2,0],[2,2,0],n90);\n"
        + "const enNegative=racketNormalRestitution([0,-4,0],[0,1,0],[0,2,0],n0);\n"
        + "const enNotClosing=racketNormalRestitution([0,3,0],[0,5,0],[0,2,0],n0);\n"
        + "const enMissing=racketNormalRestitution([0,-4,0],[0,5,0],null,n0);\n"
        + "console.log(JSON.stringify({mixed,staticCar,cancel,offsetYaw,missing,n0,n90,en,enShift,enOblique,enNegative,enNotClosing,enMissing}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert result["mixed"]["armWorld"] == pytest.approx([-2.0, 1.0, 3.0])
    assert result["mixed"]["turnWorld"] == pytest.approx([-0.8, 0.31, 0.0])
    assert result["mixed"]["world"] == pytest.approx([-2.3, 1.01, 3.0])
    assert result["mixed"]["speedWorld"] == pytest.approx(math.sqrt(15.3101))
    assert result["staticCar"]["speedWorld"] == pytest.approx(13.0)
    assert result["cancel"]["speedWorld"] == pytest.approx(1.0)  # 不是标量 2+1
    assert result["offsetYaw"]["turnWorld"] == pytest.approx([-0.09, 0.0, 0.0])
    assert result["offsetYaw"]["speedWorld"] == pytest.approx(0.09)
    assert result["missing"] is None
    assert result["n0"] == pytest.approx([0.0, 1.0, 0.0])
    assert result["n90"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert result["en"]["uInN"] == pytest.approx(-6.0)
    assert result["en"]["uOutN"] == pytest.approx(3.0)
    assert result["en"]["en"] == pytest.approx(0.5)
    assert result["enShift"]["en"] == pytest.approx(0.5)  # 共同平移不改变相对速度
    assert result["enOblique"]["en"] == pytest.approx(0.5)
    assert result["enNegative"]["en"] == pytest.approx(-1.0 / 6.0)  # 异常值保留，不钳位
    assert result["enNotClosing"] is None
    assert result["enMissing"] is None


@pytest.mark.parametrize("car,fp0", [("v03", 15.42), ("v04", 15.42)])
def test_add_face_angles_wiring_and_fk_properties(car, fp0):
    """拍面yaw,pitch,拍心/拍速的 Python 侧：正确附加 fy/fp、跳过残缺关节；
    FK 性质：零位 fy=fp=0（拍面朝正前且不上仰）、J1 为垂直轴（Δfy=−Δq1 精确、fp 分毫不动
    ——这正是 pitch 列不需要减车 yaw 的依据）。
    拍速口径按车分（arm.head_model 自报）：v0.3 无柔度标定 → 解析 Jacobian，与数值差分一致、
    只有 J1 转时退化成 |q̇1|·r（status speed= 的口径）；v0.4 走冻结柔度模型 F，**必须同时有
    effort** 才出 p_head/v_tcp_arm，故这里这批没 effort 的帧在 v04 下一律不出速度（F 的细节
    见 test_add_face_angles_v04_head_is_frozen_compliance_central_difference）。
    拍面角在两车下恒等（旋转链一致），位置/拍速则不同。"""
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
    # 三维速度只在有 6 个数值 velocity 的帧上出现；不再只存无法与车速向量合成的模长。
    assert all("v_tcp_arm" not in s[i] for i in (0, 5, 9))
    assert all("vt" not in row for row in s)
    if car == "v04":
        # 冻结柔度模型三个输入缺一不可：这批帧没有 effort ⇒ 整批无 p_head、无拍速
        assert arm["head_model"]["kind"] == "frozen_compliance"
        assert arm["head_model"]["velocity"] == "central_difference"
        assert all("v_tcp_arm" not in row and "p_head" not in row for row in s)
        assert "冻结柔度模型" in arm["fk_source"]
        return
    assert arm["head_model"]["kind"] == "rigid_fk_jacobian"     # v0.3 没有柔度标定
    assert "拍速=解析 Jacobian" in arm["fk_source"]
    # 解析 Jacobian 三分量 ≡ 数值差分（沿 q̇ 方向前推一小步的位移速率）
    step = 1e-6
    tcp0 = fk(q)["tcp"]
    tcp1 = fk([a + step * b for a, b in zip(q, qd)])["tcp"]
    numeric_vec = [(float(tcp1[i]) - float(tcp0[i])) / step for i in range(3)]
    assert s[6]["v_tcp_arm"] == pytest.approx(numeric_vec, abs=1e-4)
    assert s[6]["p_head"] == pytest.approx([float(v) for v in tcp0], abs=1e-4)  # 刚性 TCP 原样
    # 只有 J1 转时退化成 |q̇1|·r（触球瞬间的构造，也是 status speed= 的口径）
    lever = math.hypot(tcp0[0], tcp0[1])
    assert math.dist(s[7]["v_tcp_arm"], [0.0, 0.0, 0.0]) == pytest.approx(
        2.0 * lever, abs=1e-4
    )
    assert s[8]["v_tcp_arm"] == [0.0, 0.0, 0.0]                # 静止帧拍速为 0


def test_add_face_angles_v04_head_is_frozen_compliance_central_difference():
    """v0.4 的拍心/拍速合同：p_head = 冻结柔度模型 F(q, q̇, τ)，拍速 = 它的 ±10ms 中心差分。

    钉四条：① 零载荷零速时 F 退化成「刚性 TCP + R·tool_offset + [dx,dy,0]」的纯几何偏置；
    ② 有力矩时 q_eff 真的按 c·τ 偏（J1 正负载荷不同系数），p_head 随之偏离刚性 TCP；
    ③ v_tcp_arm 就是相邻 ±10ms 两帧 p_head 的割线，不是 J(q)·q̇——同一批帧上两者数值不同；
    ④ 差分端点被 >30ms 的断档隔开时不出速度（空闲段 20Hz 会走到这条）。"""
    pytest.importorskip("numpy")
    import json
    import sys

    import numpy as np

    sys.path.insert(0, str(SRC.parent))
    try:
        from generate_curve3_html import _add_face_angles, _frozen_head_model
        import extract_arm_bag as eab
    finally:
        sys.path.pop(0)
    eab.use_car("v04")
    model = _frozen_head_model("v04")
    spec = json.loads(Path(model["path"]).read_text(encoding="utf-8"))
    link6 = eab.JOINTS[-1]["child"]

    def ref_point(q, qd, tau):
        c = spec["compliance_rad_per_Nm"]
        qe = list(map(float, q))
        qe[0] -= (c["j1_positive_effort"] * max(tau[0], 0.0)
                  + c["j1_negative_effort"] * min(tau[0], 0.0)
                  + spec["j1_velocity_seconds"] * qd[0])
        for j in (2, 3, 4):
            qe[j] -= c[f"j{j + 1}"] * tau[j]
        r = eab.fk(qe)
        rot = np.asarray(r["link_transforms"][link6][:3, :3], dtype=float)
        return (np.asarray(r["tcp"], dtype=float)
                + rot @ np.asarray(spec["tool_offset_m"], dtype=float)
                + np.asarray([*spec["base_xy_offset_m"], 0.0], dtype=float))

    q0 = [0.1, -0.2, 0.3, 0.15, -0.4, 0.25]
    qd = [4.0, -0.4, 0.7, 0.2, -0.9, 0.5]
    tau = [37.0, -6.0, 3.0, -8.0, 2.5, 0.4]
    dt = 0.010
    # 100Hz 一段（12 帧）+ 一段 60ms 断档后的孤立帧
    ts = [1.000 + k * dt for k in range(12)] + [1.170, 1.180]
    states = []
    for i, t in enumerate(ts):
        q = [a + b * (t - ts[0]) for a, b in zip(q0, qd)]
        states.append({"t": round(t, 6), "position": q,
                       "velocity": list(qd), "effort": list(tau)})
    # 静止零载荷帧（放在最后一段之后，自带 ±10ms 邻居）
    rest_t = [1.400, 1.410, 1.420]
    for t in rest_t:
        states.append({"t": t, "position": list(q0), "velocity": [0.0] * 6,
                       "effort": [0.0] * 6})
    arm = {"car": "v04", "states": states}
    _add_face_angles(arm, car="v04")
    s = arm["states"]

    # ① 零载荷零速 ⇒ 纯几何偏置
    rest = s[len(ts) + 1]
    rigid = np.asarray(eab.fk(q0)["tcp"], dtype=float)
    assert rest["p_head"] == pytest.approx(ref_point(q0, [0.0] * 6, [0.0] * 6), abs=1e-4)
    assert float(np.linalg.norm(np.asarray(rest["p_head"]) - rigid)) > 0.005   # 偏置非零
    assert rest["v_tcp_arm"] == [0.0, 0.0, 0.0]                # 静止帧拍速为 0

    # ② 有力矩时 q_eff 真的偏，p_head 离开刚性 TCP
    mid = s[5]
    assert mid["p_head"] == pytest.approx(ref_point(mid["position"], qd, tau), abs=1e-4)
    assert float(np.linalg.norm(
        np.asarray(mid["p_head"])
        - np.asarray(eab.fk(mid["position"])["tcp"], dtype=float))) > 0.02

    # ③ 拍速 ≡ 相邻 ±10ms 两帧 p_head 的割线，且明显不等于 J(q)·q̇
    # p_head 落盘只留 0.1mm（4 位），拿它反算割线自带 ±0.0001/0.02 = ±5mm/s 的量化噪声
    secant = [(b - a) / (2 * dt) for a, b in zip(s[4]["p_head"], s[6]["p_head"])]
    assert mid["v_tcp_arm"] == pytest.approx(secant, abs=6e-3)
    res = eab.fk(mid["position"])
    jac = np.zeros(3)
    for rate, joint in zip(qd, eab.JOINTS):
        frame = res["joint_frames"][joint["name"]]
        jac += rate * np.cross(frame[:3, :3] @ joint["axis"],
                               res["tcp"] - frame[:3, 3])
    assert float(np.linalg.norm(np.asarray(mid["v_tcp_arm"]) - jac)) > 0.05

    # ④ 端点被 >30ms 断档隔开 ⇒ 有 p_head 但没有拍速（首尾帧同理）
    assert "p_head" in s[len(ts) - 1] and "v_tcp_arm" not in s[len(ts) - 1]
    assert "p_head" in s[0] and "v_tcp_arm" not in s[0]
    assert arm["head_model"]["sha256"] == model["sha256"]
    assert arm["head_model"]["half_window_ms"] == pytest.approx(10.0)


def test_v04_head_point_matches_rl_arm_frozen_endpoint():
    """报告端的 p_head 必须与 rl_arm/env/frozen_endpoint.py 逐值相同。

    两边现在共用同一个资产文件，但公式是各自写的一份（报告端不 import rl_arm，避免把
    torch 拖进报告链）。这条测试就是那份「同一口径」的锁：训练环境评分用哪个点、
    报告页面上的拍速/e_n 就得是同一个点，否则线上线下的甜点定义会悄悄分叉。"""
    pytest.importorskip("numpy")
    pytest.importorskip("torch")          # frozen_endpoint 模块级 import torch
    import sys

    import numpy as np

    rl_arm = Path(os.environ.get("TENNIS_MAN_ROOT", "D:/tennis-man")) / "rl_arm"
    if not (rl_arm / "env" / "frozen_endpoint.py").is_file():
        pytest.skip(f"rl_arm checkout 不在 {rl_arm}")
    sys.path.insert(0, str(SRC.parent))
    sys.path.insert(0, str(rl_arm))
    try:
        from generate_curve3_html import _frozen_head_model, _head_effective_q
        import extract_arm_bag as eab
        from env.frozen_endpoint import FrozenEndpoint
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    eab.use_car("v04")
    head = _frozen_head_model("v04")
    link6 = eab.JOINTS[-1]["child"]

    class _FkAdapter:                      # FrozenEndpoint 的 numpy 分支只用这两样
        fk = staticmethod(eab.fk)
        JOINTS = eab.JOINTS

    rng = np.random.default_rng(20260907)
    q = rng.uniform(-1.0, 1.0, size=(16, 6))
    v = rng.uniform(-6.0, 6.0, size=(16, 6))
    tau = rng.uniform(-60.0, 60.0, size=(16, 6))
    ref = FrozenEndpoint(_FkAdapter()).point(q, v, tau)
    mine = []
    for qi, vi, ti in zip(q, v, tau):
        r = eab.fk(_head_effective_q(head, qi, vi, ti))
        mine.append(np.asarray(r["tcp"], dtype=float)
                    + np.asarray(r["link_transforms"][link6][:3, :3], dtype=float)
                    @ np.asarray(head["tool_offset"], dtype=float)
                    + np.asarray(head["base_offset"], dtype=float))
    assert np.abs(np.asarray(mine) - ref).max() == 0.0


def test_rk300_table_puts_reject_reasons_in_finalht_cell_when_no_accepted():
    """备注列已撤：本抛没有 FinalHT 时，臂端拒收原因进 FinalHT 格的悬停。"""
    source = SRC.read_text(encoding="utf-8")
    table = source.split("const rk300TableHtml = () => {")[1].split("const armAcceptedTableHtml = () => {")[0]
    assert "^reject hit: (.+)$" in source
    assert "rejectNoteForThrow(th).replace(/<br>/g,'；')" in table
    assert "本抛无 FinalHT（无 accepted / 无 RL 目标，底盘也无可回配的末次 target）；臂端拒收原因：" in table
    assert "<th>备注</th>" not in table
    assert "armTblNote" not in table


def test_rk_plot_uses_predict_hit_car_position_and_removes_old_traces():
    source = SRC.read_text(encoding="utf-8")
    extractor = RK_EXTRACTOR.read_text(encoding="utf-8")

    assert 'car_pred_x=payload.get("car_pred_x")' in extractor
    assert 'car_pred_y=payload.get("car_pred_y")' in extractor
    assert 'rvz=payload.get("rvz")' in extractor
    assert 'cor_xy_eff=payload.get("cor_xy_eff")' in extractor
    assert 'cor_eff=payload.get("cor_eff")' in extractor
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


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_reviewed_visual_racket_rows_opt_in_keeps_all_phases_sorted(tmp_path):
    core = _core("racket-bracket-core-begin", "racket-bracket-core-end")
    harness = (
        f"{core}\n"
        "const rows=["
        "{t:0.020,reportRow:16,manualReview:true,contactPhase:'post_contact'},"
        "{t:-0.030,reportRow:16,manualReview:true,contactPhase:'pre_contact'},"
        "{t:0.001,reportRow:16,manualReview:true,contactPhase:'contact_integrated'},"
        "{t:0.000,reportRow:15,manualReview:true,contactPhase:'contact_integrated'},"
        "{t:-0.010,reportRow:16,manualReview:false,contactPhase:'pre_contact'}];\n"
        "console.log(JSON.stringify({reviewed:reviewedVisualRacketRows(rows,16),"
        "single:reviewedVisualRacketRows(rows,15)}));\n"
    )
    result = _run_node(tmp_path, harness)

    assert [row["t"] for row in result["reviewed"]] == pytest.approx([-0.030, 0.001, 0.020])
    assert [row["contactPhase"] for row in result["reviewed"]] == [
        "pre_contact",
        "contact_integrated",
        "post_contact",
    ]
    assert result["single"] == []

    source = SRC.read_text(encoding="utf-8")
    assert "contact_integrated:'触球窗'" in source
    assert "expectedDistanceMm:r.expected_distance_mm" in source


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_visual_racket_rows_before_ht_keep_own_throw_and_full_window(tmp_path):
    core = _core("racket-bracket-core-begin", "racket-bracket-core-end")
    harness = (
        f"{core}\n"
        "const rows=["
        "{t:-0.470,reportRow:7},"
        "{t:-0.430,reportRow:7},"
        "{t:-0.100,reportRow:7},"
        "{t:-0.030,reportRow:8},"
        "{t:-0.020,reportRow:null},"
        "{t:0.010,reportRow:7}];\n"
        "console.log(JSON.stringify({own:visualRacketRowsBeforeHt(rows,7,0),"
        "legacy:visualRacketRowsBeforeHt(rows,null,0)}));\n"
    )
    result = _run_node(tmp_path, harness)

    # 本抛：只留自己 report_row 的帧（旧 sidecar 无 report_row 的行放行），窗内升序。
    assert [row["t"] for row in result["own"]] == pytest.approx([-0.430, -0.100, -0.020])
    # 旧 sidecar 整体无 report_row 时退回纯时间窗。
    assert [row["t"] for row in result["legacy"]] == pytest.approx(
        [-0.430, -0.100, -0.030, -0.020]
    )

    source = SRC.read_text(encoding="utf-8")
    # 35ms 内缺帧时，前/后条目回退到测量窗内最近帧（dt 如实显示），不做宽窗拟合。
    assert "const RACKET_RAW_FAR_MAX_SEC=0.46;" in source
    assert "visRawEntry(visPair.before||visPairFar.before,'前')" in source
    assert "visRawEntry(visPair.after||visPairFar.after,'后')" in source
    assert "visualRacketRowsBeforeHt(pcRacketRows,idx+1,visPcT)" in source


def _s1_pred(t, ht, rel_x=1.0, rel_z=1.2, n_fit=8, stage=1, rel_y=0.2):
    """构造一条 /predict_hit_pos（payload ct=t）。"""
    payload = {
        "x": rel_x, "y": 0.0, "z": rel_z, "stage": stage, "ct": t, "ht": ht, "duration": ht - t,
        "n_bounce_fit": n_fit, "rel_x": rel_x, "rel_y": rel_y, "rel_z": rel_z, "car_pred_x": 0.0,
        "car_pred_y": 0.0, "car_yaw": 0.0, "rel_src": "predictor", "hit_yaw_extra": 0.1,
    }
    return {"t": t, "topic": "/predict_hit_pos", "text": json.dumps(payload)}


def _final_ht_harness(events: list, override: str, accepted_js: str = "null") -> str:
    """FinalHT / Pre300HT 选取段（[[final-ht-core]]）的 node 夹具：事件流 + 模式覆盖 + lastAccepted 桩。"""
    status_num = ("const statusNum=(text,key)=>{const m=new RegExp('(?:^|\\\\s)'+key+"
                  "'=(-?[0-9]+(?:\\\\.[0-9]+)?)').exec(text||'');return m?Number(m[1]):null;};\n")
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const cmFmt=(v,d=1)=>isNum(v)?(Number(v)*100).toFixed(d):'—';\n"
        "const RK={t0:0};\n"
        # 无臂回退依赖（[[bot-run-end-core]] / [[last-target-pred-core]]）：本夹具只测臂侧取法，桩掉即可
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k]; const rkPredStage=[], rkPredNFit=[];\n"
        "const botRunEndForThrow=()=>null; const lastTargetPredictionForThrow=()=>null;\n"
        "const REF_LEAD_TARGET=0.3;\n"
        "const ARM_PRED_ARRIVE_MAX_SEC=0.25;\n"
        "const armAligned=true;\n"
        f"const ARM_SWING_MODE_OVERRIDE='{override}';\n"
        f"const ARM={{events:{json.dumps(events)}}};\n"
        + status_num +
        "const armPreds=ARM.events.filter(e=>e.topic==='/predict_hit_pos').map(e=>{const p=JSON.parse(e.text);"
        "return {t:e.t,rel_x:p.rel_x,rel_y:p.rel_y,rel_z:p.rel_z,ht:p.ht,ct:p.ct,stage:p.stage,"
        "nFit:p.n_bounce_fit,relSrc:p.rel_src};});\n"
        "const armHitStatuses=ARM.events.filter(e=>e.topic==='/tennis/status'&&"
        "/^(accepted hit |late ht saved|reject hit:|error hit_pos)/.test(e.text)).map((e,si)=>({si,t:e.t,text:e.text}));\n"
        "const armPredAlign={delta:0};\n"
        "const reportThrows=[{firstT:0,lastT:1e9,ht:0}];\n"
        "const matchThrowByAcceptedCt=ct=>reportThrows[0];\n"
        f"const lastAcceptedForThrow=th=>({accepted_js});\n"
        + _core("final-ht-core-begin", "final-ht-core-end") + "\n"
        "const fin=finalTargetForThrow(reportThrows[0]);\n"
        "const pre=pre300ForThrow(reportThrows[0],fin);\n"
        "console.log(JSON.stringify({mode:ARM_SWING_MODE.mode,"
        "fin:fin&&{ct:fin.ct,ht:fin.ht,relX:fin.relX,relZ:fin.relZ,source:fin.source,issues:fin.issues,"
        "nFrozen:fin.nFrozen,arrival:fin.arrival},pre:pre&&{ct:pre.ct,ht:pre.ht,devMs:pre.devMs}}));\n"
    )


def _rl_throw_events(ht=10.60, n=20, reply_delay=0.055, rl_done=None):
    """RL 场次一抛：每 30ms 一条 S1 预测（ct=10.00+0.03k，ht 恒定），臂回复晚 reply_delay。
    前 5 条回 accepted（首条=接管），其后 late ht saved；rl_done 给了就追加 rl_swing done 状态。"""
    events = []
    for k in range(n):
        ct = round(10.00 + 0.03 * k, 3)
        events.append(_s1_pred(ct, ht, rel_x=1.0 + 0.001 * k, rel_z=1.2 + 0.002 * k, n_fit=4 + k))
        reply = round(ct + reply_delay, 3)
        if k < 5:
            text = (f"accepted hit x={1.0 + 0.001 * k:.4f} z={1.2 + 0.002 * k - 0.164:.4f} "
                    f"duration={ht - reply:.4f} hit_time=0.2500")
        else:
            text = f"late ht saved: contact in {ht - reply:.3f}s sweep_w=3.5 mode=0"
        events.append(_status_event(reply, text))
    if rl_done is not None:
        events.append(_status_event(ht + 0.05, rl_done))
    events.sort(key=lambda e: e["t"])
    return events


_RL_ACCEPTED_STUB = ("{cmd:10.055,lastAcceptT:10.175,wct:10.12,wht:10.60,finalCt:10.12,finalHt:10.60,"
                     "finalMismatch:false,reswing:null,lates:[],lastUpdateT:10.175,wx:1.004,wz:1.208}")


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_ht_rl_reconstruction_respects_30ms_freeze(tmp_path):
    """旧 RL bag（无 rl_swing done）：到达代理=生产回复时刻，接管后 到达≥当前目标ht−30ms 的不接受。

    ht=10.60：k=19（ct 10.57，回复 10.625）与 k=18（ct 10.54，回复 10.595）落在冻结窗，
    k=17（ct 10.51，回复 10.565）是冻结前最后接受 → FinalHT；Pre300HT 取 ct 最接近 10.30 的 k=10。
    """
    out = _run_node(tmp_path, _final_ht_harness(_rl_throw_events(), "rl", _RL_ACCEPTED_STUB))
    assert out["mode"] == "rl"
    assert out["fin"]["source"] == "rl_recon"
    assert out["fin"]["ct"] == pytest.approx(10.51)
    assert out["fin"]["relX"] == pytest.approx(1.017)
    assert out["fin"]["nFrozen"] == 2
    assert out["fin"]["arrival"] == pytest.approx(10.565)
    assert out["fin"]["issues"] == []
    assert out["pre"]["ct"] == pytest.approx(10.30)
    assert out["pre"]["devMs"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_ht_rl_status_is_matched_by_ct_and_auto_detected(tmp_path):
    """新 bag：臂发 rl_swing done mode=active ct=…，auto 判定为 RL 且按 ct 精确回配（不靠到达代理）。"""
    done = ("rl_swing done mode=active ct=10.480000 ht=10.600000 t_arr=10.535000 x_geo=1.0160 z=1.0680 "
            "face_yaw=-0.1000 pitch=28.00 speed_req=4.50 stage=1 n_points=20 n_targets=17 n_frozen=3 "
            "t_end=10.630000")
    out = _run_node(tmp_path, _final_ht_harness(_rl_throw_events(rl_done=done), "auto", _RL_ACCEPTED_STUB))
    assert out["mode"] == "rl"
    assert out["fin"]["source"] == "rl_status"
    assert out["fin"]["ct"] == pytest.approx(10.48)
    assert out["fin"]["nFrozen"] == 3
    assert out["fin"]["arrival"] == pytest.approx(10.535)
    # 同一 bag 若强制按规则读：FinalHT 退回 lastAccepted 桩的 finalCt（10.12）
    out2 = _run_node(tmp_path, _final_ht_harness(_rl_throw_events(rl_done=done), "rules", _RL_ACCEPTED_STUB))
    assert out2["mode"] == "rules"
    assert out2["fin"]["source"] == "accepted"
    assert out2["fin"]["ct"] == pytest.approx(10.12)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_ht_rules_uses_last_late_saved_message_and_flags_contamination(tmp_path):
    """规则规划器：FinalHT = 最后一条 late ht saved 回配的消息（finalCt）；校验只标不换。"""
    stub = ("{cmd:10.055,lastAcceptT:10.175,wct:10.12,wht:10.60,finalCt:10.45,finalHt:10.60,"
            "finalMismatch:false,reswing:{ok:true},lates:[1,2,3],lastUpdateT:10.505,wx:1.004,wz:1.208}")
    out = _run_node(tmp_path, _final_ht_harness(_rl_throw_events(), "auto", stub))
    assert out["mode"] == "rules"
    assert out["fin"]["source"] == "late_saved"
    assert out["fin"]["ct"] == pytest.approx(10.45)
    assert out["fin"]["issues"] == []
    assert out["pre"]["ct"] == pytest.approx(10.30)
    # 末端脏数据：finalCt 指向一条 rel_y<0 且 stage 回退到 0 的消息 → 三条告警，值照旧不换
    events = _rl_throw_events()
    events.append(_s1_pred(10.58, 10.60, rel_x=1.3, rel_z=1.2, n_fit=0, stage=0, rel_y=-0.1))
    events.append(_status_event(10.635, "reject hit: stage 0"))
    events.sort(key=lambda e: e["t"])
    stub_dirty = stub.replace("finalCt:10.45", "finalCt:10.58")
    dirty = _run_node(tmp_path, _final_ht_harness(events, "rules", stub_dirty))
    assert dirty["fin"]["ct"] == pytest.approx(10.58)
    joined = "；".join(dirty["fin"]["issues"])
    assert "stage 回退" in joined and "rel_y<0" in joined and "跳变" in joined


def _chassis_fallback_harness(arm_aligned: bool, override: str, events=None, accepted_js: str = "null",
                              with_target_change: bool = True) -> str:
    """无臂 FinalHT 回退（用户 2026-09-06 定）：真实 bot-run-end / last-target-pred / final-ht 三段 + RK bot/pred 桩。

    RK.t0=100 检查回退锚换到了 RK 绝对轴；bot 桩与 test_last_target_change_matches_prediction_by_deadline_not_ct
    同一组：RUN 内最后一次 target 变化 t=1.15、deadline=1.80，对应 pred idx=1（ct 1.10 / ht 1.80）。
    """
    events = events or []
    status_num = ("const statusNum=(text,key)=>{const m=new RegExp('(?:^|\\s)'+key+"
                  "'=(-?[0-9]+(?:\\.[0-9]+)?)').exec(text||'');return m?Number(m[1]):null;};\n")
    target_x = "[1,1,1,4,4,4,null]" if with_target_change else "[null,null,null,null,null,null,null]"
    arm_js = f"{{events:{json.dumps(events)}}}" if arm_aligned else "null"
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const cmFmt=(v,d=1)=>isNum(v)?(Number(v)*100).toFixed(d):'—';\n"
        f"{BOT_POSE_JS_DEPS}"
        "const ts=s=>s.t; const ys=(s,k)=>s.y[k];\n"
        "const REF_LEAD_TARGET=0.3;\n"
        "const ARM_PRED_ARRIVE_MAX_SEC=0.25;\n"
        f"const armAligned={'true' if arm_aligned else 'false'};\n"
        f"const ARM_SWING_MODE_OVERRIDE='{override}';\n"
        f"const ARM={arm_js};\n"
        + status_num +
        "const armPreds=ARM?ARM.events.filter(e=>e.topic==='/predict_hit_pos').map(e=>{const p=JSON.parse(e.text);"
        "return {t:e.t,rel_x:p.rel_x,rel_y:p.rel_y,rel_z:p.rel_z,ht:p.ht,ct:p.ct,stage:p.stage,"
        "nFit:p.n_bounce_fit,relSrc:p.rel_src};}):[];\n"
        "const armHitStatuses=ARM?ARM.events.filter(e=>e.topic==='/tennis/status'&&"
        "/^(accepted hit |late ht saved|reject hit:|error hit_pos)/.test(e.text)).map((e,si)=>({si,t:e.t,text:e.text})):[];\n"
        "const armPredAlign={delta:0};\n"
        "const th={ht:1.25,firstT:1.00,lastT:1.14,firstIdx:0,lastIdx:2}; const reportThrows=[th];\n"
        "const matchThrowByAcceptedCt=ct=>reportThrows[0];\n"
        f"const lastAcceptedForThrow=th=>({accepted_js});\n"
        "const rkPredStage=[1,1,1], rkPredNFit=[4,5,6];\n"
        "const RK={t0:100,"
        "bot:{t:[1.00,1.05,1.10,1.15,1.20,1.25,1.26],y:{"
        "phase:['RUN','RUN','RUN','RUN','RUN','RUN','BRAKE_IN_SWING'],"
        "x:[0,0,0,0,0,2.25,2.25],y:[0,0,0,0,0,3.35,3.35],"
        f"target_x:{target_x},target_y:[2,3,3,3,3,3,null],"
        "remaining:[1.00,.95,.90,.65,.69,.64,null]}},"
        "pred:{t:[1.00,1.10,1.14],y:{"
        "ht_rel:[2.00,1.80,1.89],x:[.50,.60,999],rel_x:[.70,.78,999],rel_y:[.9,.8,.7],rel_z:[1.10,1.20,999],"
        "car_pred_x:[2.00,2.20,999],car_pred_y:[3.00,3.30,999]}}};\n"
        + _core("bot-run-end-core-begin", "bot-run-end-core-end") + "\n"
        + _core("last-target-pred-core-begin", "last-target-pred-core-end") + "\n"
        + _core("final-ht-core-begin", "final-ht-core-end") + "\n"
        "const fin=finalTargetForThrow(th);\n"
        "const pre=pre300ForThrow(th,fin);\n"
        "console.log(JSON.stringify({mode:ARM_SWING_MODE.mode,modeSource:ARM_SWING_MODE.source,"
        "fin:fin&&{ct:fin.ct,ht:fin.ht,relX:fin.relX,relZ:fin.relZ,stage:fin.stage,nFit:fin.nFit,source:fin.source,"
        "issues:fin.issues,fallback:fin.fallback,accepted:fin.accepted,note:fin.note},"
        "pre:pre&&{ct:pre.ct,ht:pre.ht,relX:pre.relX,devMs:pre.devMs}}));\n"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_ht_falls_back_to_chassis_last_target_when_arm_absent(tmp_path):
    """臂栈没跑（bag 无 /joint_states，0905_173206）：FinalHT=底盘末次 target 对应预测 [车]，Pre300HT 同抛 RK 流。"""
    out = _run_node(tmp_path, _chassis_fallback_harness(arm_aligned=False, override="auto"))
    assert out["mode"] == "rules"
    assert "无 /joint_states" in out["modeSource"]
    fin = out["fin"]
    assert fin["source"] == "chassis_target" and fin["fallback"] is True and fin["accepted"] is None
    assert fin["ct"] == pytest.approx(101.10) and fin["ht"] == pytest.approx(101.80)   # RK 绝对轴
    assert fin["relX"] == pytest.approx(0.78) and fin["relZ"] == pytest.approx(1.20)
    assert fin["stage"] == 1 and fin["nFit"] == 5
    assert fin["issues"] == []
    assert "末次 target" in fin["note"]
    # Pre300HT：早于 FinalHT 的 ct 里最接近 101.80−0.30=101.50 的只有 idx0（ct 101.00）
    assert out["pre"]["ct"] == pytest.approx(101.00) and out["pre"]["relX"] == pytest.approx(0.70)
    assert out["pre"]["devMs"] == pytest.approx(-500.0)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_ht_chassis_fallback_only_when_arm_gives_none(tmp_path):
    """臂在但本抛没受理（RL 无 done、无 accepted）→ 落到 [车]；臂受理了 → 仍是臂的 FinalHT，不被回退抢走。"""
    preds = [_s1_pred(round(10.00 + 0.03 * k, 3), 10.60, rel_x=1.0, rel_z=1.2, n_fit=4 + k) for k in range(5)]
    none = _run_node(tmp_path, _chassis_fallback_harness(arm_aligned=True, override="rl", events=preds))
    assert none["mode"] == "rl"
    assert none["fin"]["source"] == "chassis_target" and none["fin"]["fallback"] is True
    assert none["fin"]["ct"] == pytest.approx(101.10)

    real = _run_node(tmp_path, _chassis_fallback_harness(
        arm_aligned=True, override="rules", events=_rl_throw_events(), accepted_js=_RL_ACCEPTED_STUB))
    assert real["fin"]["source"] == "accepted" and real["fin"]["fallback"] is False
    assert real["fin"]["ct"] == pytest.approx(10.12)

    # 臂状态有但回配不上（rl_swing done 的 ct 不在预测流里）→ 回退到 [车]，并把臂侧告警带过去
    done = ("rl_swing done mode=active ct=10.999000 ht=10.600000 t_arr=10.535000 x_geo=1.0 z=1.0 "
            "face_yaw=0 pitch=28.00 speed_req=4.50 stage=1 n_points=20 n_targets=17 n_frozen=3 t_end=10.63")
    dangling = _run_node(tmp_path, _chassis_fallback_harness(
        arm_aligned=True, override="rl", events=_rl_throw_events(rl_done=done), accepted_js=_RL_ACCEPTED_STUB))
    assert dangling["fin"]["source"] == "chassis_target"
    assert any("rl_swing done" in s for s in dangling["fin"]["issues"])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_ht_chassis_fallback_needs_a_target_change(tmp_path):
    """车没下发过目标（RUN 内无 target 变化）→ 回退也没有锚，整行照旧为空，不拿别的时刻冒充。"""
    out = _run_node(tmp_path, _chassis_fallback_harness(arm_aligned=False, override="auto", with_target_change=False))
    assert out["fin"] is None and out["pre"] is None


def test_chassis_fallback_is_labelled_in_table_and_summary():
    source = SRC.read_text(encoding="utf-8")
    table = source.split("const rk300TableHtml = () => {")[1].split("const armAcceptedTableHtml = () => {")[0]
    assert "chassis_target:'车'" in table
    assert "finalHtFallback:!!(fin&&fin.fallback)," in table
    assert "回退为底盘末次 target 对应预测 [车]" in table
    assert "臂无 FinalHT 时回退=底盘末次 target 对应预测 [车]" in table
    # 回退不冒充臂目标：tcp−臂目标 / 目标拍速仍只认 RL/规则的源
    assert "const aimIsFinal=!!(fin&&(fin.source==='rl_status'||fin.source==='rl_recon'));" in table
    assert "if(!ARM) return {mode:'rules', source:'bag 无 /joint_states，臂栈未运行'};" in source
