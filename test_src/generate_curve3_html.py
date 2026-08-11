# -*- coding: utf-8 -*-
"""
Generate interactive HTML directly from JSON data.

Supported inputs:
1. tracker_*.json        Raw tracker output
2. *_replay.json         Replay output from test_curve3_replay.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _merge_racket_json(base_data: dict, racket_data: dict, racket_json_path: str | None) -> dict:
    merged = copy.deepcopy(base_data)
    merged_cfg = merged.setdefault("config", {})
    merged_summary = merged.setdefault("summary", {})
    merged_frames = merged.setdefault("frames", [])
    racket_cfg = racket_data.get("config", {})
    racket_summary = racket_data.get("summary", {})

    if racket_json_path:
        merged_cfg["racket_json_path"] = str(racket_json_path)

    for key in (
        "distance_unit",
        "first_frame_exposure_pc",
        "video_frame_mapping_exact",
        "racket_model_path",
        "racket_conf_threshold",
    ):
        if key in racket_cfg:
            merged_cfg[key] = racket_cfg[key]

    for key in (
        "video_frame_mapping_exact",
        "video_frames_mapped",
        "racket_observations_3d",
        "racket_frames_processed",
    ):
        if key in racket_summary:
            merged_summary[key] = racket_summary[key]

    if "racket_observations" in racket_data:
        merged["racket_observations"] = racket_data["racket_observations"]

    frame_by_idx = {
        frame.get("idx"): frame
        for frame in merged_frames
        if isinstance(frame, dict) and isinstance(frame.get("idx"), int)
    }

    for racket_frame in racket_data.get("frames", []):
        if not isinstance(racket_frame, dict):
            continue
        frame_idx = racket_frame.get("idx")
        target = frame_by_idx.get(frame_idx) if isinstance(frame_idx, int) else None
        if target is None:
            target = {}
            if isinstance(frame_idx, int):
                target["idx"] = frame_idx
                frame_by_idx[frame_idx] = target
            merged_frames.append(target)
        for key, value in racket_frame.items():
            if key == "idx":
                continue
            target[key] = value

    merged["frames"] = sorted(
        merged_frames,
        key=lambda frame: (
            0,
            frame.get("idx"),
        )
        if isinstance(frame, dict) and isinstance(frame.get("idx"), int)
        else (1, 0),
    )
    return merged


def _add_face_angles(arm) -> None:
    """给 arm.states 逐帧附加 fy/fp/vt：FK 拍面法向（link6 +X，前向规范化）在臂系的
    yaw（°，atan2(x,y) 口径，与 PC回球 yaw 同式）与 pitch（°，asin(n_z)，正=开面上仰），
    以及拍心线速度大小 vt（m/s）。
    单源复用 extract_arm_bag.fk（0801 dz/yawrate 分析脚本同一公式），不在 JS 里抄第二份 FK 链。
    **pitch 不需要减车 yaw**：J1/BASE_ROT 都是纯 z（垂直）转，只搬 n 的水平分量、不动 n_z，
    故臂系 pitch ≡ 世界 pitch（车无 roll/pitch 前提），比 yaw 少一个误差源。
    **vt 走解析 Jacobian**：同一次 fk 已经给出每个关节的 joint_frames，转轴 a_j=R_j·axis、
    轴上一点 o_j=p_j 都是现成的，故 v_tcp = Σ_j q̇_j·(a_j×(p_tcp−o_j))——与 6 次数值差分
    逐分量一致到 1e-6，但只需一次 FK（差分要 7 次，整场多花 ~30s）。速度大小与车 yaw 无关
    （纯 z 转不改模长），故 vt 同时是臂系值与世界系值。缺 velocity 的帧只出 fy/fp。
    关节残缺（None 或非 6 关节）时跳过该帧，报告列显示 —。
    导入失败不让整份报告挂掉，但必须打到 stderr：2026-08-05 前 run_tracker 的
    post-run 把 ROS2 pixi 的 site-packages 留在 PYTHONPATH 里，子进程 import numpy
    命中 conda 版 C 扩展加载失败，这里静默 return 让两列拍面角整场空白，只有手工
    重跑才有（已由 run_tracker._report_tool_env 根治）。"""
    try:
        _here = str(Path(__file__).resolve().parent)
        if _here not in sys.path:
            sys.path.insert(0, _here)   # 作为模块被别处 import 时 sys.path[0] 不是本目录
        import numpy as _np
        from extract_arm_bag import fk, JOINTS
    except Exception as exc:
        print(
            f"[report] 拍面yaw,pitch,拍速 三量不可用：import extract_arm_bag 失败（{exc!r}）",
            file=sys.stderr,
        )
        return
    import math as _m
    states = arm.get("states") if isinstance(arm, dict) else None
    if not isinstance(states, list):
        return
    link6 = JOINTS[-1]["child"]
    for s in states:
        q = s.get("position") if isinstance(s, dict) else None
        if not (isinstance(q, list) and len(q) == 6
                and all(isinstance(v, (int, float)) for v in q)):
            continue
        res = fk(q)
        rot = res["link_transforms"][link6]
        # R @ [1,0,0] = 第一列；前向规范化要同时翻 n_z，否则 pitch 会跟着 n_y 变号
        n0, n1, n2 = float(rot[0, 0]), float(rot[1, 0]), float(rot[2, 0])
        if n1 < 0:
            n0, n1, n2 = -n0, -n1, -n2
        s["fy"] = round(_m.degrees(_m.atan2(n0, n1)), 2)
        s["fp"] = round(_m.degrees(_m.asin(max(-1.0, min(1.0, n2)))), 2)
        qd = s.get("velocity")
        if not (isinstance(qd, list) and len(qd) == 6
                and all(isinstance(v, (int, float)) for v in qd)):
            continue
        tcp = res["tcp"]
        vel = _np.zeros(3)
        for rate, joint in zip(qd, JOINTS):
            frame = res["joint_frames"][joint["name"]]
            axis = frame[:3, :3] @ joint["axis"]
            vel += rate * _np.cross(axis, tcp - frame[:3, 3])
        s["vt"] = round(float(_np.linalg.norm(vel)), 4)


def generate_html(
    input_path: str,
    output_path: str,
    racket_json_path: str | None = None,
    arm_json_path: str | None = None,
    rk_tracking_json_path: str | None = None,
    rk_time_bias: float | None = None,
) -> None:
    data = _load_json(input_path)
    if racket_json_path:
        data = _merge_racket_json(data, _load_json(racket_json_path), racket_json_path)
    if arm_json_path:
        data["arm"] = _load_json(arm_json_path)
        _add_face_angles(data["arm"])
    if rk_tracking_json_path:
        data["rk_tracking"] = _load_json(rk_tracking_json_path)
    if rk_time_bias is not None:
        data["rk_time_bias_preset"] = rk_time_bias
    # annotate_video 写回的逐帧球拍 2D 检测（bbox+关键点）体积巨大且模板不消费，
    # 只保留 racket3d / racket_observations，避免 HTML 从 ~12MB 涨到 ~60MB
    for frame in data.get("frames", []):
        if isinstance(frame, dict):
            frame.pop("racket_detections", None)
    data_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("%%DATA_JSON%%", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Interactive HTML saved: {output_path}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracker / Curve3 Interactive</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0}
.hdr{padding:16px 24px;background:#16213e;border-bottom:1px solid #0f3460}
.hdr h1{font-size:20px;color:#e94560;margin-bottom:6px}
.hdr .st{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#a0a0c0}
.hdr .st span{background:#0f3460;padding:3px 10px;border-radius:4px}
.hdr .st .v{color:#e94560;font-weight:600}
.tabs{display:flex;gap:3px;padding:10px 24px 0}
.tab{padding:7px 18px;cursor:pointer;background:#16213e;border:1px solid #0f3460;border-bottom:none;
     border-radius:6px 6px 0 0;font-size:12px;color:#a0a0c0;user-select:none}
.tab:hover{background:#1a1a3e;color:#fff}
.tab.on{background:#1a1a2e;color:#e94560;border-color:#e94560;border-bottom:1px solid #1a1a2e}
.pnl{display:none}.pnl.on{display:block}
.cc{width:100%;padding:8px 20px}
.zt{display:flex;justify-content:flex-end;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 10px}
.ztl{font-size:12px;color:#a0a0c0}
.zb{appearance:none;border:1px solid #0f3460;background:#16213e;color:#d7d7eb;border-radius:999px;padding:4px 10px;
    font:inherit;font-size:12px;cursor:pointer;transition:background .18s ease,border-color .18s ease,transform .18s ease}
.zb:hover{background:#1a1a3e;border-color:#e94560;transform:translateY(-1px)}
.zb.on{background:#0f3460;border-color:#5cd0ff;color:#fff}
.zr{font-size:12px;color:#a0a0c0;min-width:44px;text-align:right}
.lc{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 10px}
.lb{appearance:none;display:inline-flex;align-items:center;gap:8px;border:1px solid #0f3460;background:#16213e;color:#d7d7eb;user-select:none;
    border-radius:999px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer;transition:background .18s ease,border-color .18s ease,opacity .18s ease,transform .18s ease}
.lb:hover{background:#1a1a3e;border-color:#e94560;transform:translateY(-1px)}
.lb.off{opacity:.45}
.ls{width:10px;height:10px;border-radius:999px;flex:0 0 10px;box-shadow:0 0 0 1px rgba(255,255,255,.15)}
.zx{overflow:hidden;padding-bottom:6px;border-radius:16px;transition:box-shadow .18s ease}
.cc.zoom-active .zx{box-shadow:0 0 0 1px rgba(92,208,255,.55),0 0 0 4px rgba(92,208,255,.10)}
.cb{width:100%;min-width:100%;height:780px;min-height:780px}
.cbt{width:100%;min-width:100%;height:2000px;min-height:2000px}
.armEv{padding:0 24px 4px;font-size:12px;color:#a0a0c0;line-height:1.7}
.armEv b{color:#e94560;font-weight:600}
.armTblWrap{overflow-x:auto}
.armTbl{border-collapse:collapse;margin:8px 0 4px;font-size:11.5px}
.armTbl th,.armTbl td{border:1px solid #0f3460;padding:3px 9px;text-align:right;white-space:nowrap}
.armTbl th{background:#16213e;color:#a0a0c0;font-weight:600}
.armTbl td{color:#d7d7eb}
.armTbl td.armTblNote{width:240px;min-width:240px;max-width:240px;text-align:left;white-space:normal}
.armTblNote>div{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;max-height:2.7em;line-height:1.35;overflow:hidden}
.rkCtl{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 10px;font-size:12px;color:#a0a0c0}
.rkCtl input{width:92px;border:1px solid #0f3460;background:#16213e;color:#fff;border-radius:4px;padding:4px 6px;font:inherit}
.rkCoordNote{margin:0 0 8px;padding:6px 10px;border:1px solid #0f3460;border-radius:6px;background:#16213e;font-size:12px;color:#d7d7eb}
.rkCoordNote b{color:#5cd0ff}
.mvSel{border:1px solid #0f3460;background:#16213e;color:#fff;border-radius:4px;padding:4px 8px;font:inherit;font-size:12px;max-width:380px}
.rkCtl input[type=range]{flex:1 1 160px;min-width:120px;width:auto;accent-color:#e94560;padding:0}
#mvClock{font-size:12px;color:#a0a0c0;min-width:210px;font-variant-numeric:tabular-nums}
.mvWrap{display:flex;gap:12px;align-items:stretch}
.mvPlot{flex:1 1 auto;min-width:0;border-radius:16px;overflow:hidden}
.mvPlot>div{width:100%;height:680px}
.mvSide{flex:0 0 300px;background:#16213e;border:1px solid #0f3460;border-radius:12px;padding:10px 14px;font-size:12px;color:#d7d7eb;align-self:flex-start}
.mvSide h3{font-size:11.5px;color:#5cd0ff;margin:12px 0 2px;font-weight:600}
.mvKV{display:flex;justify-content:space-between;gap:10px;padding:3px 0;border-bottom:1px dashed #0f3460}
.mvKV .k{color:#a0a0c0;white-space:nowrap}
.mvKV .v{color:#fff;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
#mvNote{font-size:11px;color:#a0a0c0;margin-top:10px;line-height:1.7}
</style>
</head>
<body>
<div class="hdr">
  <h1>Tracker / Curve3 Interactive</h1>
  <div class="st" id="st"></div>
</div>
<div class="tabs">
  <div class="tab on" id="tabRk" data-idx="5" onclick="sw(5)">RK≈300ms / PC</div>
  <div class="tab" data-idx="0" onclick="sw(0)">PC Data</div>
  <div class="tab" data-idx="2" onclick="sw(2)">3D Trajectory</div>
  <div class="tab" data-idx="3" onclick="sw(3)">Car Location</div>
  <div class="tab" id="tabArm" data-idx="4" onclick="sw(4)">Arm Accepted</div>
  <div class="tab" id="tabRkSignals" data-idx="6" onclick="sw(6)">RK Signals</div>
  <div class="tab" id="tabRkMove" data-idx="1" onclick="sw(1)">RK Car Move</div>
</div>
<div id="p0" class="pnl"><div class="cc"><div class="lc" id="l0"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c0" data-action="out">X-</button><button type="button" class="zb on" data-plot="c0" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c0" data-action="in">X+</button><span id="c0r" class="zr">1.00x</span></div><div class="zx"><div id="c0" class="cb"></div></div></div></div>
<div id="p1" class="pnl">
  <div class="cc">
    <div class="rkCtl">
      <span>移动</span><select id="mvSel" class="mvSel"></select>
      <button type="button" class="zb" id="mvFirst" title="回到首帧">⏮ 首帧</button>
      <button type="button" class="zb" id="mvPrev">◀ 上一帧</button>
      <button type="button" class="zb" id="mvPlay">▶ 播放</button>
      <button type="button" class="zb" id="mvNext">下一帧 ▶</button>
      <span>倍速</span><select id="mvSpeed" class="mvSel">
        <option value="0.05">0.05x</option><option value="0.1">0.1x</option>
        <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
        <option value="1" selected>1x</option><option value="2">2x</option>
      </select>
      <input id="mvSlider" type="range" min="0" max="0" value="0">
      <span id="mvClock"></span>
    </div>
    <div class="mvWrap">
      <div class="mvPlot"><div id="c1"></div></div>
      <div class="mvSide" id="mvSide">
        <div class="mvKV"><span class="k">帧</span><span class="v" id="mvFrameV">—</span></div>
        <div class="mvKV"><span class="k">t (RK 轴)</span><span class="v" id="mvTRk">—</span></div>
        <div class="mvKV"><span class="k">t (PC 报告轴)</span><span class="v" id="mvTPc">—</span></div>
        <div class="mvKV"><span class="k">阶段 phase</span><span class="v" id="mvPhase">—</span></div>
        <div class="mvKV"><span class="k">车位置</span><span class="v" id="mvPos">—</span></div>
        <div class="mvKV"><span class="k">目标位置</span><span class="v" id="mvTgt">—</span></div>
        <div class="mvKV"><span class="k">距目标距离</span><span class="v" id="mvDist">—</span></div>
        <div class="mvKV"><span class="k">剩余到位时间</span><span class="v" id="mvRem">—</span></div>
        <h3>IMU 车速（bot_state vx/vy，世界系）</h3>
        <div class="mvKV"><span class="k">|v|</span><span class="v" id="mvSpd">—</span></div>
        <div class="mvKV"><span class="k">vx / vy</span><span class="v" id="mvVxy">—</span></div>
        <h3>姿态</h3>
        <div class="mvKV"><span class="k">yaw</span><span class="v" id="mvYaw">—</span></div>
        <div class="mvKV"><span class="k">IMU yaw_speed</span><span class="v" id="mvImuW">—</span></div>
        <h3>舵轮</h3>
        <div class="mvKV"><span class="k">舵轮角 steer</span><span class="v" id="mvSteer">—</span></div>
        <div class="mvKV"><span class="k">目标 steer (cmd)</span><span class="v" id="mvSteerTgt">—</span></div>
        <div class="mvKV"><span class="k">旋转方向</span><span class="v" id="mvSteerDir">—</span></div>
        <div id="mvNote"></div>
      </div>
    </div>
  </div>
</div>
<div id="p2" class="pnl"><div class="cc"><div class="lc" id="l2"></div><div class="zt"><span class="ztl">X zoom</span><button type="button" class="zb" data-plot="c2" data-action="out">X-</button><button type="button" class="zb on" data-plot="c2" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c2" data-action="in">X+</button><span id="c2r" class="zr">n/a</span></div><div class="zx"><div id="c2" class="cb"></div></div></div></div>
<div id="p3" class="pnl"><div class="cc"><div class="lc" id="l3"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c3" data-action="out">X-</button><button type="button" class="zb on" data-plot="c3" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c3" data-action="in">X+</button><span id="c3r" class="zr">1.00x</span></div><div class="zx"><div id="c3" class="cb"></div></div></div></div>
<div id="p4" class="pnl">
  <div class="armEv" id="armEv"></div>
  <div class="cc"><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c4" data-action="out">X-</button><button type="button" class="zb on" data-plot="c4" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c4" data-action="in">X+</button><span id="c4r" class="zr">1.00x</span></div><div class="zx"><div id="c4" class="cbt"></div></div><div class="lc" id="l4" style="margin:10px 0 0"></div></div>
</div>
<div id="p5" class="pnl on">
  <div class="cc">
    <div class="rkCtl"><span>RK time bias(s)</span><input id="rkOff" type="number" step="0.0001" value="0"><button type="button" class="zb" id="rkApply">Apply</button><button type="button" class="zb" id="rkAuto">Auto align</button><span id="rkInfo"></span></div>
    <div class="rkCoordNote"><b>坐标说明：</b>PC真值 x/y = 拟合球世界 x/y − 同时刻插值车世界 x/y，采用世界坐标轴，不随车体 yaw 旋转（0731 起全表统一世界系；rel_x/rel_z 列除外，那是臂端车体系合同值）。PC S1 以同抛最后一条 RK S0 世界 y 为相会面，沿原 PC S1 drag 状态重求 x/z/HT/v；交点落到地面下时不显示。</div>
    <div class="armEv" id="rk300Tbl" style="padding:0 0 4px"></div>
    <div class="lc" id="l5"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c5" data-action="out">X-</button><button type="button" class="zb on" data-plot="c5" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c5" data-action="in">X+</button><span id="c5r" class="zr">1.00x</span></div><div class="zx"><div id="c5" class="cb"></div></div>
  </div>
</div>
<div id="p6" class="pnl">
  <div class="cc">
    <div class="rkCtl"><span>RK time bias(s)</span><input id="rkSigOff" type="number" step="0.0001" value="0"><button type="button" class="zb" id="rkSigApply">Apply</button><button type="button" class="zb" id="rkSigAuto">Auto align</button><span id="rkSigInfo"></span></div>
    <div class="lc" id="l6"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c6" data-action="out">X-</button><button type="button" class="zb on" data-plot="c6" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c6" data-action="in">X+</button><span id="c6r" class="zr">1.00x</span></div><div class="zx"><div id="c6" class="cb"></div></div>
  </div>
</div>

<script>
const D = %%DATA_JSON%%;
(function(){
const cfg = D.config || {};
const summary = D.summary || {};
const PLOT_CONFIG = {
  responsive:true,
  displayModeBar:true,
  scrollZoom:false,
  plotGlPixelRatio:1,
};
const preds = D.predictions || [];
const frames = Array.isArray(D.frames) ? D.frames : [];
const writtenVideoFrameIds = Array.isArray(D.video_frame_indices) ? D.video_frame_indices : [];
const distanceScale = cfg.distance_unit === 'm' ? 1.0 : 0.001;
const scaleVec3 = p => ({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale});
const obsRaw = (D.observations || []).map(o=>scaleVec3(o));
const racketObsRaw = (D.racket_observations || []).map(o=>scaleVec3(o));
const carFull = (D.car_locs || []).map(c=>({...c, x:c.x*distanceScale, y:c.y*distanceScale, z:c.z*distanceScale}));
const s0Full = preds.filter(p=>p.stage===0).map(p=>({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale}));
const s1Full = preds.filter(p=>p.stage===1).map(p=>({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale}));
const resets = D.reset_times || summary.reset_times || [];
const throws = D.throws || [];
const ARM = (D.arm && Array.isArray(D.arm.states) && D.arm.states.length) ? D.arm : null;
const RK = (D.rk_tracking && D.rk_tracking.world && Array.isArray(D.rk_tracking.world.t)) ? D.rk_tracking : null;
if(!ARM){
  const tabArm=document.getElementById('tabArm');
  if(tabArm) tabArm.style.display='none';
}
if(!RK){
  const tabRk=document.getElementById('tabRk');
  if(tabRk) tabRk.style.display='none';
  const tabRkSignals=document.getElementById('tabRkSignals');
  if(tabRkSignals) tabRkSignals.style.display='none';
  const tabRkMove=document.getElementById('tabRkMove');
  if(tabRkMove) tabRkMove.style.display='none';
}
window.__hasRK = !!RK;  // 无 RK 数据时首屏退回 PC Data
const sourceType = cfg.replay_source ? 'Replay JSON' : 'Tracker JSON';
const fps = cfg.fps || summary.actual_fps;
const durationS = cfg.duration_s || summary.duration_s;
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const firstNumeric = (items, key) => {
  for (const item of items) {
    if (item && isNum(item[key])) return item[key];
  }
  return null;
};
const firstFrameT0 =
  isNum(cfg.first_frame_exposure_pc) ? cfg.first_frame_exposure_pc :
  (frames.length > 0 && isNum(frames[0].exposure_pc) ? frames[0].exposure_pc : null);
const fallbackT0 = [firstNumeric(obsRaw, 't'), firstNumeric(racketObsRaw, 't'), firstNumeric(carFull, 't'), firstNumeric(preds, 'ct')]
  .find(v => v !== null);
const t0 = firstFrameT0 !== null ? firstFrameT0 : (fallbackT0 !== null ? fallbackT0 : 0);
const relTime = v => isNum(v) ? (v - t0) : 0;
const frameSeries = frames
  .filter(f => f && isNum(f.exposure_pc))
  .map(f => ({
    ...f,
    rel_s: relTime(f.exposure_pc),
    video_frame_idx: Number.isInteger(f.video_frame_idx) ? f.video_frame_idx : null,
  }));
const frameByIdx = new Map(frameSeries.filter(f => Number.isInteger(f.idx)).map(f => [f.idx, f]));
const mappedFramesFromVideo = writtenVideoFrameIds
  .map((frameId, videoFrameIdx) => {
    const f = frameByIdx.get(frameId);
    return f ? ({...f, video_frame_idx: videoFrameIdx}) : null;
  })
  .filter(Boolean);
const explicitVideoLinkedFrames = frameSeries.filter(f => f.video_frame_idx !== null);
const videoLinkedFrames = mappedFramesFromVideo.length > 0
  ? mappedFramesFromVideo
  : explicitVideoLinkedFrames;
const preferredFrames = videoLinkedFrames.length > 0 ? videoLinkedFrames : frameSeries;
const frameBallObs = preferredFrames
  .filter(f => f.ball3d)
  .map(f => ({
    ...scaleVec3(f.ball3d),
    t: f.exposure_pc,
    rel_s: f.rel_s,
    idx: f.idx,
    video_frame_idx: f.video_frame_idx,
  }));
const frameRacketObs = preferredFrames
  .filter(f => f.racket3d)
  .map(f => ({
    ...scaleVec3(f.racket3d),
    t: f.exposure_pc,
    rel_s: f.rel_s,
    idx: f.idx,
    video_frame_idx: f.video_frame_idx,
  }));
const obsFull = frameBallObs.length > 0
  ? frameBallObs
  : obsRaw.map(o => ({...o, rel_s: relTime(o.t), idx: null, video_frame_idx: null}));
const racketFull = frameRacketObs.length > 0
  ? frameRacketObs
  : racketObsRaw.map(o => ({
      ...o,
      rel_s: isNum(o.elapsed_s) ? o.elapsed_s : relTime(o.t),
      idx: Number.isInteger(o.frame_idx) ? o.frame_idx : null,
      video_frame_idx: Number.isInteger(o.video_frame_idx) ? o.video_frame_idx : null,
    }));
const obs = obsFull;
const racket = racketFull;
const car = carFull;
const s0 = s0Full;
let s1 = s1Full;
const pairedFrames = preferredFrames.filter(f => f.ball3d && f.racket3d);
const frameStartLabel =
  frames.length > 0 && isNum(frames[0].exposure_pc)
    ? `${Number(frames[0].exposure_pc).toFixed(6)}s`
    : null;
const ballSourceLabel = frameBallObs.length > 0 ? 'video-linked frames' : 'tracker observations';
const racketSourceLabel = frameRacketObs.length > 0 ? 'video-linked frames' : 'racket observations';
const g2 = trace => ({type:'scattergl', ...trace});
const buildPlots = [];
const builtPlots = new Set();
function ensurePlot(idx){
  if(builtPlots.has(idx)) return;
  const builder = buildPlots[idx];
  if(typeof builder !== 'function') return;
  builder();
  builtPlots.add(idx);
}
window.ensurePlot = ensurePlot;

const stat=(k,v)=>`<span>${k}: <span class="v">${v!=null?v:'-'}</span></span>`;
document.getElementById('st').innerHTML=[
  stat('Source', sourceType),
  cfg.replay_source ? stat('Replay source', cfg.replay_source) : '',
  isNum(cfg.first_frame_exposure_pc) ? stat('t0 perf', Number(cfg.first_frame_exposure_pc).toFixed(6)+'s') : '',
  frameStartLabel ? stat('Frame0 perf', frameStartLabel) : '',
  cfg.video_frame_mapping_exact != null ? stat('Frame map', cfg.video_frame_mapping_exact ? 'exact' : 'fallback') : '',
  summary.video_frames_mapped ? stat('Mapped video frames', summary.video_frames_mapped) : '',
  stat('Ball 3D', obsFull.length),
  stat('Ball src', ballSourceLabel),
  racketFull.length ? stat('Racket 3D', racketFull.length) : '',
  racket.length ? stat('Racket src', racketSourceLabel) : '',
  videoLinkedFrames.length ? stat('Video-linked frames', videoLinkedFrames.length) : '',
  pairedFrames.length ? stat('Ball+Racket same-frame', pairedFrames.length) : '',
  stat('S0 preds', s0Full.length),
  stat('S1 raw preds', s1Full.length),
  carFull.length ? stat('Car locs', carFull.length) : '',
  (cfg.car_localizer && Number.isInteger(cfg.car_localizer.sample_every_frames))
    ? stat('Car sample', `1/${cfg.car_localizer.sample_every_frames}`)
    : '',
  stat('2D render', 'full scattergl'),
  stat('Resets', resets.length),
  ARM ? stat('Arm states', ARM.states.length) : '',
  ARM ? stat('Arm cmds', ARM.commands.length) : '',
  ARM && ARM.duration_sec ? stat('Arm bag', ARM.duration_sec.toFixed(1)+'s') : '',
  RK ? stat('RK topics', Object.keys(RK.counts || {}).length) : '',
  RK && RK.world ? stat('RK world ball', RK.world.t.length) : '',
  throws.length ? stat('Throws', throws.length) : '',
  fps ? stat('FPS', fps.toFixed ? fps.toFixed(1) : fps) : '',
  cfg.noise_mm!=null ? stat('Noise', cfg.noise_mm+'mm') : '',
  cfg.cor!=null ? stat('COR', cfg.cor) : '',
  cfg.ideal_hit_z!=null ? stat('ideal_hit_z', (cfg.ideal_hit_z*distanceScale).toFixed(2)+'m') : '',
  cfg.min_stage1_points ? stat('min_s1', cfg.min_stage1_points) : '',
  durationS ? stat('Duration', durationS.toFixed ? durationS.toFixed(1)+'s' : durationS+'s') : '',
].filter(Boolean).join('');

const DL={paper_bgcolor:'#1a1a2e',plot_bgcolor:'#16213e',font:{color:'#e0e0e0',size:11},
  legend:{bgcolor:'rgba(22,33,62,0.9)',bordercolor:'#0f3460',borderwidth:1,font:{size:10},itemsizing:'constant'},
  hovermode:'closest',margin:{l:60,r:30,t:40,b:50}};
const GS={gridcolor:'#0f3460',zerolinecolor:'#0f3460'};
const predRemainingMs = p => (p && isNum(p.ht) && isNum(p.ct)) ? (p.ht - p.ct) * 1000 : null;

// ================= RK / Arm 共享数据层（首页对比表、Arm tab、RK Move 共用） =================
// [[align-core-begin]] —— test_report_time_align.py 抽取此段在 node 里回归测试,别在段内引新全局依赖
const ts = series => (series && Array.isArray(series.t)) ? series.t : [];
const ys = (series, key) => (series && series.y && Array.isArray(series.y[key])) ? series.y[key] : [];
const pairs = (series, key) => ts(series).map((t,i)=>({t:Number(t), v:Number(ys(series,key)[i])}))
  .filter(p=>isNum(p.t)&&isNum(p.v))
  .sort((a,b)=>a.t-b.t);
const pcRows = obs.map(o=>({t:isNum(o.rel_s)?o.rel_s:relTime(o.t), x:o.x, y:o.y, z:o.z}))
  .filter(p=>isNum(p.t))
  .sort((a,b)=>a.t-b.t);
// single=单 tag 退化解：只剩一块 tag 可见时，位置是拿冻结的历史 yaw 经 0.42m 安装
// 杠杆反解出来的，yaw 本身发 null（消费端保持自身值，见 car_localizer.locate）。
const pcCarRows = car.map(c=>({t:isNum(c.elapsed_s)?c.elapsed_s:relTime(c.t),
    x:c.x, y:c.y, z:c.z, yaw:c.yaw, single:Array.isArray(c.tag_ids)&&c.tag_ids.length===1}))
  .filter(p=>isNum(p.t))
  .sort((a,b)=>a.t-b.t);
// 位置插值用全部行，yaw 插值只能用带 yaw 的行——两者分开取，别让一个 null yaw 把
// 好好的位置一起废掉。
const pcCarYawRows = pcCarRows.filter(p=>isNum(p.yaw));
const nearest = (rows, t) => {
  let lo=0, hi=rows.length;
  while(lo<hi){
    const mid=(lo+hi)>>1;
    if(rows[mid].t<t) lo=mid+1; else hi=mid;
  }
  const cand=[];
  if(lo<rows.length) cand.push(rows[lo]);
  if(lo>0) cand.push(rows[lo-1]);
  if(!cand.length) return null;
  return cand.reduce((best,row)=>Math.abs(row.t-t)<Math.abs(best.t-t)?row:best,cand[0]);
};
const lerp = (a,b,f) => a + (b-a)*f;
// 有界线性插值：t 落在相邻两行之间且间隔 ≤ maxGap 才给结果，绝不外推
const interpRow = (rows, t, maxGap) => {
  if(!rows.length) return null;
  let lo=0, hi=rows.length;
  while(lo<hi){
    const mid=(lo+hi)>>1;
    if(rows[mid].t<t) lo=mid+1; else hi=mid;
  }
  if(lo<=0||lo>=rows.length) return null;
  const a=rows[lo-1], b=rows[lo];
  if(!(t>=a.t&&t<=b.t)||b.t-a.t>maxGap) return null;
  return {a, b, f:(t-a.t)/Math.max(1e-9,b.t-a.t)};
};
const interpPcVal = (rows,t,key,maxGap) => {
  const s=interpRow(rows,t,maxGap);
  return (s && isNum(s.a[key]) && isNum(s.b[key])) ? lerp(s.a[key],s.b[key],s.f) : null;
};
const median = values => {
  const sorted=[...values].sort((a,b)=>a-b);
  return sorted[sorted.length>>1];
};
const splitFlights = (rows, accept) => {
  const groups=[];
  let group=[];
  const flush=()=>{
    if(group.length>=5 && group[group.length-1].t-group[0].t>=0.2 && accept(group)){
      groups.push(group);
    }
    group=[];
  };
  for(const row of rows){
    if(group.length && row.t-group[group.length-1].t>0.5) flush();
    group.push(row);
  }
  flush();
  return groups;
};
const pcFlights = splitFlights(pcRows, rows=>{
  const zs=rows.map(row=>row.z).filter(isNum);
  return zs.length>=5 && Math.max(...zs)-Math.min(...zs)>=0.25;
});
const rkMovZ = (()=>{
  if(!RK) return [];
  const rows=pairs(RK.world,'z');
  const out=[];
  for(let i=1;i<rows.length;i++){
    const dt=rows[i].t-rows[i-1].t;
    if(dt>0&&dt<0.2&&Math.abs(rows[i].v-rows[i-1].v)/dt>0.4) out.push(rows[i]);
  }
  return out;
})();
const rkFlights = splitFlights(rkMovZ, ()=>true);
// 粗扫不能按全局固定步长抽 RK 点：短抛球会因抽样相位恰好少于 5 点而从评分中消失，
// 重复抛球形状随后把搜索带到“错一抛”的假谷。每抛等量取点，覆盖不再依赖相位。
const rkMovZCoarse = rkFlights.flatMap(flight=>{
  const count=Math.min(16,flight.length);
  return Array.from({length:count},(_,i)=>
    flight[Math.round(i*(flight.length-1)/Math.max(1,count-1))]);
}).sort((a,b)=>a.t-b.t);

// PC 发布的小车位姿会原样进入 RK world 的 bot_x/bot_y/bot_yaw。
// 用这些共同值拟合两台机器时钟的比例，避免把数分钟场次强行压成一个固定 offset。
const clockAnchor = (()=>{
  if(!RK || pcCarRows.length<20) return {scale:1,bias:null,anchors:0,mad:null};
  const wx=ys(RK.world,'bot_x'), wy=ys(RK.world,'bot_y'), wyaw=ys(RK.world,'bot_yaw');
  const rkPose=ts(RK.world).map((t,i)=>({
    t:Number(t),x:Number(wx[i]),y:Number(wy[i]),yaw:Number(wyaw[i]),
  })).filter(row=>isNum(row.t)&&isNum(row.x)&&isNum(row.y)&&isNum(row.yaw));
  const key=row=>row.x.toFixed(4)+','+row.y.toFixed(4)+','+row.yaw.toFixed(4);
  const pcUnique=new Map();
  for(const row of pcCarRows){
    if(!isNum(row.x)||!isNum(row.y)||!isNum(row.yaw)) continue;
    const k=key(row);
    pcUnique.set(k,pcUnique.has(k)?null:row.t);
  }
  const rkFirst=new Map();
  for(const row of rkPose){
    const k=key(row);
    if(!rkFirst.has(k)) rkFirst.set(k,row.t);
  }
  let anchors=[];
  for(const [k,pcT] of pcUnique){
    const rkT=rkFirst.get(k);
    if(isNum(pcT)&&isNum(rkT)) anchors.push({rk:rkT,pc:pcT,off:pcT-rkT});
  }
  if(anchors.length<20) return {scale:1,bias:null,anchors:anchors.length,mad:null};
  const off0=median(anchors.map(row=>row.off));
  const offMad=median(anchors.map(row=>Math.abs(row.off-off0)));
  // 真锚的离散只有 /pc_car_loc→bot_state 的管线延迟量级（几十 ms~0.3s）；
  // 位姿键重复时 rkFirst 会配到错的一条，这些错锚能偏出 ±14s。它们把 offMad 顶到
  // 秒级后 8×offMad 反而全放行，Theil–Sen 斜率被拖歪触发 scale 门、整组锚被丢弃
  // （0809 场：offMad 1.81s → 放行 ±14.5s → scale 0.9864 → bias=null）。给硬上限。
  const maxDev=Math.min(0.5,Math.max(0.2,8*offMad));
  anchors=anchors.filter(row=>Math.abs(row.off-off0)<=maxDev).sort((a,b)=>a.rk-b.rk);
  if(anchors.length<20 || anchors[anchors.length-1].rk-anchors[0].rk<30){
    return {scale:1,bias:null,anchors:anchors.length,mad:null};
  }
  const fitRows=anchors.length<=400 ? anchors : Array.from({length:400},(_,i)=>
    anchors[Math.round(i*(anchors.length-1)/399)]);
  const span=fitRows[fitRows.length-1].rk-fitRows[0].rk;
  const minSpan=Math.max(20,0.2*span);
  const slopes=[];
  for(let i=0;i<fitRows.length;i++){
    for(let j=i+1;j<fitRows.length;j++){
      const dt=fitRows[j].rk-fitRows[i].rk;
      if(dt>=minSpan) slopes.push((fitRows[j].pc-fitRows[i].pc)/dt);
    }
  }
  if(slopes.length<20) return {scale:1,bias:null,anchors:anchors.length,mad:null};
  const scale=median(slopes);
  if(Math.abs(scale-1)>0.005) return {scale:1,bias:null,anchors:anchors.length,mad:null};
  const bias=median(fitRows.map(row=>row.pc-scale*row.rk));
  const residualMad=median(fitRows.map(row=>Math.abs(row.pc-(scale*row.rk+bias))));
  if(residualMad>0.08) return {scale:1,bias:null,anchors:anchors.length,mad:residualMad};
  return {scale,bias,anchors:anchors.length,mad:residualMad};
})();

// 给定仿射时间映射 PC t = scale * RK t + bias，只比较每抛 z 形状；
// 每抛独立去除固定高度偏差，再对各抛误差取中位数，长轨迹不会压过短轨迹。
const scoreTimeMap = (scale,bias,rows) => {
  const matches=pcFlights.map(()=>({dz:[], first:null, last:null}));
  let flightIdx=0;
  for(const row of rows){
    const t=scale*row.t+bias;
    while(flightIdx<pcFlights.length && t>pcFlights[flightIdx][pcFlights[flightIdx].length-1].t) flightIdx+=1;
    if(flightIdx>=pcFlights.length) break;
    const flight=pcFlights[flightIdx];
    if(t<flight[0].t) continue;
    const v=interpPcVal(flight, t, 'z', 0.08);
    if(v==null) continue;
    const match=matches[flightIdx];
    match.dz.push(v-row.v);
    if(match.first==null) match.first=t;
    match.last=t;
  }
  const flightErrs=[];
  let n=0;
  for(const match of matches){
    if(match.dz.length<5 || match.last-match.first<0.2) continue;
    const bias=median(match.dz);
    flightErrs.push(median(match.dz.map(value=>Math.abs(value-bias))));
    n+=match.dz.length;
  }
  if(!flightErrs.length) return null;
  return {err:median(flightErrs), n, flights:flightErrs.length};
};
const estimateTimeMap = () => {
  const requiredFlights=clockAnchor.bias==null?3:2;
  const empty={scale:clockAnchor.scale,bias:null,err:null,n:0,flights:0,
    anchors:clockAnchor.anchors,anchorMad:clockAnchor.mad,requiredFlights};
  if(rkMovZ.length<30 || rkMovZCoarse.length<15 || pcFlights.length<requiredFlights) return empty;
  const scale=clockAnchor.scale;
  const lo=clockAnchor.bias==null
    ? Math.floor(pcRows[0].t-scale*rkMovZ[rkMovZ.length-1].t)-1
    : clockAnchor.bias-0.75;
  const hi=clockAnchor.bias==null
    ? Math.ceil(pcRows[pcRows.length-1].t-scale*rkMovZ[0].t)+1
    : clockAnchor.bias+0.75;
  const coarse=Math.max(0.005,Math.min(0.05,(hi-lo)/18000));
  const cands=[];
  for(let bias=lo; bias<=hi+1e-4; bias+=coarse){
    const s=scoreTimeMap(scale,bias,rkMovZCoarse);
    if(s) cands.push({scale,bias,...s});
  }
  if(!cands.length) return empty;
  const flightMax=cands.reduce((m,c)=>Math.max(m,c.flights),0);
  const minFlights=Math.max(requiredFlights,Math.ceil(0.6*flightMax));
  const ok=cands.filter(c=>c.flights>=minFlights);
  if(!ok.length) return empty;
  const coarseBest=ok.reduce((a,b)=>b.err<a.err?b:a,ok[0]);
  const win=Math.max(0.04,coarse*3);
  let best=null;
  for(let bias=coarseBest.bias-win; bias<=coarseBest.bias+win+1e-4; bias+=0.0002){
    const s=scoreTimeMap(scale,bias,rkMovZ);
    if(s && s.flights>=minFlights && (!best || s.err<best.err)) best={scale,bias,...s};
  }
  if(!best) return empty;
  return {...best,anchors:clockAnchor.anchors,anchorMad:clockAnchor.mad,requiredFlights};
};
// [[align-core-end]]
const auto = RK ? estimateTimeMap() : {
  scale:1,bias:null,err:null,n:0,flights:0,anchors:0,anchorMad:null,requiredFlights:3,
};
const presetBias = isNum(D.rk_time_bias_preset) ? Number(D.rk_time_bias_preset) : null;
let rkScale=isNum(auto.scale)?auto.scale:1;
let rkBias=Math.round((presetBias!=null?presetBias:(isNum(auto.bias)?auto.bias:0))*10000)/10000;
const rkToPc = t => isNum(Number(t)) ? rkScale*Number(t)+rkBias : null;
const publishRkTimeMap = () => {
  window.__rkTimeMap={scale:rkScale,bias:rkBias};
};
publishRkTimeMap();
window.__dbgAlign={
  pcRows:()=>pcRows,
  rkMovZ:()=>rkMovZ,
  clockAnchor:()=>clockAnchor,
  scoreTimeMap:(scale,bias)=>scoreTimeMap(scale,bias,rkMovZ),
  auto:()=>auto,
};
// 有外部 bias 预置时由操作者承担锚点来源；否则必须同时通过轨迹形状、点数和抛数质量门。
const alignBad = !!RK && presetBias==null &&
  (auto.bias==null || auto.err==null || auto.err>0.08 || auto.n<30 ||
   auto.flights<auto.requiredFlights);
const driftPpm=(rkScale-1)*1e6;
const alignWarnHtml = alignBad
  ? `<div style="border:1px solid #e94560;background:rgba(233,69,96,0.12);color:#e94560;font-weight:600;border-radius:8px;padding:8px 12px;margin:0 0 10px">`+
    `⚠ PC↔RK 自动对齐不可信（z 形状误差 ${auto.err==null?'n/a':auto.err.toFixed(3)+'m'} / `+
    `${auto.n} 点 / ${auto.flights||0} 抛；共享位姿锚 ${auto.anchors||0}）`+
    `——本页所有跨轴内容（北极星表 / RK 轨迹叠加 / Arm 对齐）不可靠。`+
    `常见原因：两侧共同球观测或共享小车位姿太少。`+
    `处置：RK 页手动 Apply time bias，或用 --rk-time-bias 预置外部锚。</div>`
  : (RK
    ? `<div style="font-size:11px;color:#7fbf9f;margin:0 0 6px">PC↔RK 对齐 ✓ `+
      `PC t = ${rkScale.toFixed(8)} × RK t + ${rkBias.toFixed(4)}s（drift ${driftPpm>=0?'+':''}${driftPpm.toFixed(1)}ppm）`+
      (presetBias!=null
        ? `（--rk-time-bias 外部锚预置）`
        : `（共享位姿 ${auto.anchors} 锚；z 形状 ${auto.err.toFixed(3)}m / ${auto.n} 点 / ${auto.flights} 抛）`)+
      `——对齐可信时，表内 Δt/dx/dz 的偏差应解读为预测/执行误差而非错轴。</div>`
    : '');
const rkPredStage = RK ? ys(RK.pred,'stage') : [];
const rkPredDurMs = (RK ? ys(RK.pred,'duration') : []).map(v=>isNum(v)?v*1000:null);
const rkPredNFit = RK ? ys(RK.pred,'n_bounce_fit') : [];
// 分抛：按 ht_rel 聚类 RK 预测消息；ref300* 字段严格来自同一条、提前量最接近 300ms 的 S1 消息。
// [[rk300-contract-core-begin]]
const REF_LEAD_TARGET=0.3;
const rkThrows = (()=>{
  if(!RK) return [];
  const t=ts(RK.pred), ht=ys(RK.pred,'ht_rel');
  const worldY=ys(RK.pred,'y');
  const worldX=ys(RK.pred,'x')||[];
  const relx=ys(RK.pred,'rel_x'), rely=ys(RK.pred,'rel_y'), relz=ys(RK.pred,'rel_z');
  const carPredX=ys(RK.pred,'car_pred_x'), carPredY=ys(RK.pred,'car_pred_y');
  const out=[];
  for(let i=0;i<t.length;i++){
    const ti=Number(t[i]);
    if(!isNum(ti) || !isNum(ht[i])) continue;
    const cur=out[out.length-1];
    const upd = th => {
      th.ht=ht[i]; th.lastT=ti; th.msgs=(th.msgs||0)+1;
      if(Number(rkPredStage[i])===0 && isNum(worldY[i])){
        th.lastS0Y=worldY[i]; th.lastS0T=ti; th.lastS0Idx=i;
      } else if(Number(rkPredStage[i])===1){
        th.hasS1=true;
      }
      if(isNum(relx[i])&&isNum(rely[i])&&isNum(relz[i])){
        th.stage=rkPredStage[i]; th.rel_x=relx[i]; th.rel_y=rely[i]; th.rel_z=relz[i];
        th.lastRelIdx=i; th.refT=ti;  // 最终 ref 定格的计算时刻（该消息的 ct，RK 相对轴）
        const lead=ht[i]-ti;
        if(lead>0 && Number(rkPredStage[i])===1){
          const dev=Math.abs(lead-REF_LEAD_TARGET);
          if(th.ref300T==null || dev<th.ref300LeadDev){
            th.ref300Stage=rkPredStage[i]; th.ref300X=relx[i]; th.ref300Y=rely[i]; th.ref300Z=relz[i];
            th.ref300Xw=isNum(worldX[i])?worldX[i]:null;
            th.ref300CarX=isNum(carPredX[i])?carPredX[i]:null;
            th.ref300CarY=isNum(carPredY[i])?carPredY[i]:null;
            th.ref300T=ti; th.ref300Ht=ht[i]; th.ref300Lead=lead; th.ref300LeadDev=dev;
            th.ref300NFit=isNum(rkPredNFit[i])?rkPredNFit[i]:null; th.ref300Idx=i;
          }
        }
      }
    };
    if(cur && Math.abs(ht[i]-cur.ht)<0.8 && ti-cur.lastT<2.0){
      upd(cur);
    } else {
      const th={ht:ht[i], firstT:ti, lastT:ti, msgs:0, stage:null, rel_x:null, rel_y:null, rel_z:null, lastRelIdx:null, refT:null,
                ref300Stage:null, ref300X:null, ref300Y:null, ref300Z:null, ref300T:null, ref300Ht:null,
                ref300Xw:null, ref300CarX:null, ref300CarY:null,
                ref300Lead:null, ref300LeadDev:null, ref300NFit:null, ref300Idx:null,
                lastS0Y:null, lastS0T:null, lastS0Idx:null, hasS1:false};
      upd(th);
      out.push(th);
    }
  }
  return out;
})();
// [[rk300-contract-core-end]]
// PC S1 原日志是“PC 自己最后 S0 y”处的 Curve4 drag 状态。报告用该状态沿同一
// 自治 ODE 前/后积分，在同抛 RK 最后一条 S0 世界 y 处重新求交；x/z/HT/v 必须一起更新。
// 若 RK/PC 时轴无法可靠匹配，或数学交点落到地面下，则该点不显示，不回退原 S1。
// [[pc-s1-rk-s0-core-begin]]
const pcS1ThrowAt = p => {
  const ctPc=relTime(p.ct);
  if(!isNum(ctPc)) return null;
  const candidates=rkThrows.filter(th=>th.hasS1 && isNum(th.lastS0Y) &&
    ctPc>=rkToPc(th.firstT)-0.5 && ctPc<=rkToPc(th.lastT)+0.5);
  if(!candidates.length) return null;
  return candidates.reduce((best,th)=>
    Math.abs(ctPc-rkToPc(th.lastS0T))<Math.abs(ctPc-rkToPc(best.lastS0T))?th:best,
    candidates[0]);
};
const pcS1AtWorldY = (p,targetY) => {
  const keys=['x','y','z','vx','vy','vz','ct','ht'];
  if(!isNum(targetY) || !keys.every(key=>isNum(p[key])) || !isNum(cfg.k_drag)) return null;
  if(Math.abs(targetY-p.y)<=1e-9) return {...p,y:targetY};
  if(Math.abs(p.vy)<1e-9) return null;
  const direction=Math.sign((targetY-p.y)/p.vy);
  if(!direction) return null;
  const rhs=s=>{
    const speed=Math.hypot(s[3],s[4],s[5]), k=Number(cfg.k_drag);
    return [s[3],s[4],s[5],-k*speed*s[3],-k*speed*s[4],-9.8-k*speed*s[5]];
  };
  const step=(s,h)=>{
    const k1=rhs(s);
    const k2=rhs(s.map((v,i)=>v+0.5*h*k1[i]));
    const k3=rhs(s.map((v,i)=>v+0.5*h*k2[i]));
    const k4=rhs(s.map((v,i)=>v+h*k3[i]));
    return s.map((v,i)=>v+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6);
  };
  const h=direction*0.002;
  let state=[p.x,p.y,p.z,p.vx,p.vy,p.vz], elapsed=0;
  for(let i=0;i<1000;i++){
    const next=step(state,h);
    const d0=state[1]-targetY, d1=next[1]-targetY;
    if(d0*d1<=0){
      const f=Math.abs(next[1]-state[1])<1e-12 ? 0 : (targetY-state[1])/(next[1]-state[1]);
      const hit=state.map((v,j)=>v+f*(next[j]-v));
      const ht=p.ht+elapsed+f*h;
      if(hit[2]<0) return null;
      return {...p,x:hit[0],y:targetY,z:hit[2],vx:hit[3],vy:hit[4],vz:hit[5],ht};
    }
    state=next;
    elapsed+=h;
  }
  return null;
};
const pcS1AtRkS0Y = p => {
  const th=pcS1ThrowAt(p);
  if(!th) return null;
  const hit=pcS1AtWorldY(p,th.lastS0Y);
  return hit ? {...hit,rkStage0Y:th.lastS0Y,rkStage0T:th.lastS0T} : null;
};
// [[pc-s1-rk-s0-core-end]]
s1 = RK && !alignBad ? s1Full.map(pcS1AtRkS0Y).filter(Boolean) : [];
// 小车位姿只做「被前后两条 /pc_car_loc 夹住」的有界插值，**不外推、不冻结**。
// 2026-08-09：原来插不上时会退到「最近邻 ≤0.3s」——车在跑的时候这等于把车钉死在
// 一百多毫秒前的位置，而真值 x/y = 球世界 − 车世界，冻结多少就假造多少。
// 0809_122035 #13（HT300=188.112s）实测：AprilTag 最后一帧 187.9999 之后断了 1.83s，
// 最近邻拿了 112ms 前那帧，车 vx≈−1.1/vy≈−1.2 m/s ⇒ 真值 x 少算 10.6cm、y 少算 11.4cm
// （报告给 94.7/−0.4，按 RK /bot_state 的车位姿应是 105.3/+11.0，y 甚至变号），
// 而格子里还挂着球侧的 ±0.8cm 误差棒，完全看不出车已经黑了。宁可整格 —，不冻结。
// 外推也不做：同场两抛（#13/#18）拿 RK /bot_state 当参照扫过 2~7 点 × 1/2 阶，
// 没有任何一档在两抛上同时好（6点1阶 −1.9cm/−9.5cm、5点2阶 −7.5cm/−2.3cm，
// 线性 2~4 点稳定差 4~7cm）——110ms 外推本身就值 5cm 上下，调不出来。
// 返回 ga/gb = 到前/后一条 /pc_car_loc 的时距，供上层算插值误差并显示。
// yaw 与位置分两套行插：单 tag 退化帧只有 x/y、yaw 为 null（车上一块 tag 被臂座
// 挡住时的常态，见 car_localizer.locate）。位置照旧用全部 /pc_car_loc 行夹住插值，
// yaw 单独在带 yaw 的行里夹（同样 0.5s 上限、同样不外推）；夹不住就 yaw=null，
// 只让依赖 yaw 的列（车体系 rel、车 yaw 列）缺失，世界轴的 PC 真值列照常出。
// [[car-at-core-begin]] —— test_report_car_pose_at.py 抽取此段在 node 里回归测试
const carAt = t => {
  const s=interpRow(pcCarRows,t,0.5);
  if(!s) return null;
  const sy=interpRow(pcCarYawRows,t,0.5);
  let yaw=null, yawGa=null, yawGb=null;
  if(sy){
    const dyaw=Math.atan2(Math.sin(sy.b.yaw-sy.a.yaw),Math.cos(sy.b.yaw-sy.a.yaw));
    yaw=sy.a.yaw+dyaw*sy.f;
    yawGa=t-sy.a.t; yawGb=sy.b.t-t;
  }
  return {x:lerp(s.a.x,s.b.x,s.f), y:lerp(s.a.y,s.b.y,s.f), yaw,
          ga:t-s.a.t, gb:s.b.t-t, yawGa, yawGb, single:!!(s.a.single||s.b.single)};
};
const relToCar = (b,c) => {
  if(!c || !isNum(c.yaw)) return null;      // yaw 缺失时转不到车体系，宁可整格空
  const dx=b.x-c.x, dy=b.y-c.y, cy=Math.cos(c.yaw), sy=Math.sin(c.yaw);
  return {x:cy*dx+sy*dy, y:-sy*dx+cy*dy, z:b.z};
};
// [[car-at-core-end]]
// PC 坐标真值统一入口：只使用目标时刻 20ms 前的入弧观测拟合，不再跨目标时刻插值。
// 这样不依赖“y 是否反向”识别触球，擦拍、拍框接触或触球附近丢帧都不会把出弧混进入弧。
// x/y 线性；z = 重力 + 空气阻力（λ=k_drag·水平速，与 RK fit_curve_gravity_drag 同款闭式，
// 无 k_drag 的旧 JSON/回放退回纯重力）+ 带界自由曲率 δ（|δ|≤2m/s²、≥8 点才放开，吸收
// 反弹旋转 Magnus 等平滑项——0731 实测纯重力在 0.7s 窗口有 2~5cm 系统性残差、上旋抛
// 另有 ~1.3m/s² 下压，都会顶破 35mm 门槛让真值列大量缺失）；至少 5 点、外推 ≤0.35s，
// 并以 x/z 最大残差门控。拟合失败返回 null，报告显示缺失，不退回可能跨越速度冲量的
// 线性插值。返回 {x,y,z,gap,err,resMax,dNear,delta}：x/y 为拟合球世界坐标−车世界坐标
// （0731 起世界轴，不转 yaw），z 为世界高度；gap/dNear 均为距最后入弧观测的时距，
// delta 为 z 附加曲率。
// [[pc-truth-core-begin]]
const PC_TRUTH_K_DRAG=(typeof cfg!=='undefined'&&Number.isFinite(Number(cfg.k_drag)))?Number(cfg.k_drag):0;
const PC_TRUTH_SPIN_MAX=2.0;  // z 附加自由曲率界 (m/s²)：反弹旋转 Magnus 实测 ~1.5
// 车位姿插值误差用的加速度界 (m/s²)：底盘 a_dec_max=3.0（travel_controller.hpp）。
// 恒加速下线性插值在 τ=0 的误差正好是 ½·a·ga·gb，故按这个闭式并进 err，
// 由 carAt 给的 ga/gb 直接算——AprilTag 名义 ~10Hz，正常 ga·gb 只值几 mm，
// 但 0.5s 上限那种「勉强夹住」的宽档能到 5cm，必须让它出现在误差棒里。
const PC_TRUTH_CAR_ACC=3.0;
// 单 tag 退化解的额外位置误差 (m)：位置由 tag 中心三角化 + **冻结的**历史 yaw 经
// 安装杠杆反解。冻结上限 0.5s（car_localizer._SINGLE_TAG_YAW_MAX_AGE_S）× 实测车
// yaw 变化率上界 ~3°/s ⇒ ~1.5°，乘 tag0 的 0.42m 杠杆 ≈ 11mm。双 tag 行不计这项。
const PC_TRUTH_SINGLE_TAG_ERR=0.011;
// 对称 3x3 Cramer：S=[s00,s01,s02,s11,s12,s22] 的上三角，解 M·x=r
const solve3=(S,r)=>{
  const [a,b,c,d,e,f]=S;
  const det=a*(d*f-e*e)-b*(b*f-e*c)+c*(b*e-d*c);
  if(Math.abs(det)<1e-12) return null;
  return [
    (r[0]*(d*f-e*e)-b*(r[1]*f-e*r[2])+c*(r[1]*e-d*r[2]))/det,
    (a*(r[1]*f-r[2]*e)-r[0]*(b*f-e*c)+c*(b*r[2]-r[1]*c))/det,
    (a*(d*r[2]-r[1]*e)-b*(b*r[2]-r[1]*c)+r[0]*(b*e-d*c))/det,
  ];
};
const pcTruthAt = tPc => {
  const c=carAt(tPc);
  if(!c) return null;
  let win=pcRows.filter(p=>tPc-p.t>=0.02 && tPc-p.t<=0.75);
  // 落地弹跳必须切掉，否则拟合窗横跨弹跳、残差门必爆。
  // 判据按**轨迹形状**（vz 由降转升的突增），不是绝对高度：
  // 29fps + 落地 |vz|≈5m/s ⇒ 半帧就走 8.7cm，弹跳最低点经常整个被跨过，
  // 旧的 z<0.12 判据因此频繁不触发 —— 2026-08-09 两场失败抛采到的 z_min 是
  // 0.126/0.154/0.165/0.180/0.464，全都在阈值之上。改判形状后 050621 由 9/14
  // 升到 13/14、024150 由 6/9 升到 7/9，两场均无回退。
  // 阈值与 [[pc-return-core]] 的 bounceCutRun 同款（降 <−0.5、反弹突增 >3m/s）。
  let loT=-Infinity;
  for(let i=1;i+1<win.length;i++){
    const dt0=win[i].t-win[i-1].t, dt1=win[i+1].t-win[i].t;
    if(!(dt0>0) || !(dt1>0)) continue;
    const vzA=(win[i].z-win[i-1].z)/dt0, vzB=(win[i+1].z-win[i].z)/dt1;
    // 切到最低点之后：最低点那一帧本身常被接触过程污染（曝光跨越触地），
    // 050621 #13 实测 vz 在它之后是 1.3→4.4m/s 的"向上加速"，物理上不可能；
    // 丢掉它，从干净的上升弧起拟合。
    if(vzA<-0.5 && vzB-vzA>3.0) loT=win[i].t;
  }
  // 贴地观测（z<0.12）无论如何都不该进入入弧拟合：可能是场上静止球或弹跳采样点
  win.forEach(p=>{ if(p.z<0.12 && p.t>loT) loT=p.t; });
  win=win.filter(p=>p.t>loT);
  const reg=(ts,vs)=>{
    const n=ts.length;
    let st=0,sv=0,stt=0,stv=0;
    for(let i=0;i<n;i++){ st+=ts[i]; sv+=vs[i]; stt+=ts[i]*ts[i]; stv+=ts[i]*vs[i]; }
    const den=n*stt-st*st;
    return Math.abs(den)<1e-9 ? null : [(sv*stt-st*stv)/den, (n*stv-st*sv)/den];
  };
  for(let attempt=0; attempt<3; attempt++){
    if(win.length<5 || tPc-win[win.length-1].t>0.35) return null;
    const ts=win.map(p=>p.t-tPc);
    const fx=reg(ts,win.map(p=>p.x));
    const fy=reg(ts,win.map(p=>p.y));
    if(!fx||!fy) return null;
    // z 模型 = 重力 + 空气阻力 + 带界自由曲率 δ（旋转 Magnus 等平滑未建模项）：
    //   z(u)=z0+c1·b1(u)+base(u)+δ/2·u²；阻力路径 b1=φ=(1−e^{−λu})/λ、base=−(g/λ)u、
    //   c1=vz0+g/λ，λ 用本窗口 x/y 拟合的水平速度；无 k_drag 退回 b1=u、base=−g/2·u²。
    // δ 仅在 ≥8 点时放开（3 参对少点窗口太软），|δ|≤PC_TRUTH_SPIN_MAX，夹界后按固定 δ
    // 退回 2 参。平滑旋转下压被 δ 吸收；跳变/混弧污染不是 u² 形态，仍会撑爆残差被门拦。
    const lam=PC_TRUTH_K_DRAG*Math.hypot(fx[1],fy[1]);
    const phi=lam>1e-6 ? (u=>-Math.expm1(-lam*u)/lam) : (u=>u);
    const base=lam>1e-6 ? (u=>-(9.81/lam)*u) : (u=>-4.905*u*u);
    const b1=ts.map(phi);
    const w=win.map((p,i)=>p.z-base(ts[i]));
    let fz=null, delta=0, zDof=2, zLever=null;
    if(win.length>=8){
      const n3=ts.length;
      let s01=0,s02=0,s11=0,s12=0,s22=0,r0=0,r1=0,r2=0;
      for(let i=0;i<n3;i++){
        const q=0.5*ts[i]*ts[i];
        s01+=b1[i]; s02+=q; s11+=b1[i]*b1[i]; s12+=b1[i]*q; s22+=q*q;
        r0+=w[i]; r1+=b1[i]*w[i]; r2+=q*w[i];
      }
      const S=[n3,s01,s02,s11,s12,s22];
      const sol=solve3(S,[r0,r1,r2]);
      if(sol && Math.abs(sol[2])<=PC_TRUTH_SPIN_MAX){
        const inv0=solve3(S,[1,0,0]);
        if(inv0 && inv0[0]>0){ fz=[sol[0],sol[1]]; delta=sol[2]; zDof=3; zLever=inv0[0]; }
      } else if(sol){
        delta=(sol[2]>0?1:-1)*PC_TRUTH_SPIN_MAX;
      }
    }
    if(!fz){
      const f2=reg(b1,w.map((v,i)=>v-0.5*delta*ts[i]*ts[i]));
      if(!f2) return null;
      fz=f2;
      const n2=ts.length;
      let m1=0; b1.forEach(v=>m1+=v); m1/=n2;
      let sb=0; b1.forEach(v=>sb+=(v-m1)*(v-m1));
      zLever=1/n2+m1*m1/Math.max(1e-12,sb);
    }
    const zAt=t=>fz[0]+fz[1]*phi(t)+base(t)+0.5*delta*t*t;
    let zMax=0, xMax=0;
    win.forEach((p,i)=>{
      zMax=Math.max(zMax,Math.abs(zAt(ts[i])-p.z));
      xMax=Math.max(xMax,Math.abs(fx[0]+fx[1]*ts[i]-p.x));
    });
    if(zMax<=0.035 && xMax<=0.05){
      const gap=tPc-win[win.length-1].t;
      const n=ts.length;
      let mean=0; ts.forEach(t=>mean+=t); mean/=n;
      let sxx=0; ts.forEach(t=>sxx+=(t-mean)*(t-mean));
      const lever=1/n+mean*mean/sxx;   // {1,u} 基在 τ=0 的预测方差杠杆（x 用；z 用自己基的杠杆）
      const se=(res,lev,dof)=>{
        let ss=0; res.forEach(r=>ss+=r*r);
        return Math.sqrt(ss/Math.max(1,n-dof)*lev);
      };
      const seX=se(win.map((p,i)=>fx[0]+fx[1]*ts[i]-p.x),lever,2);
      const seZ=se(win.map((p,i)=>zAt(ts[i])-p.z),zLever!=null?zLever:lever,zDof);
      const dNear=gap;
      const eModel=0.75*dNear*dNear;   // ½·(~1.5 m/s² 未建模气动)·d²
      // 车侧插值误差：x/y 是「球世界 − 车世界」，车插得糙一样进真值，必须一起算进 err
      // 并显示出来（0731 起 x/y 是差值口径，此前只报球侧误差是漏项）。
      const carGa=Number(c.ga)||0, carGb=Number(c.gb)||0;
      const eSingle=c.single?PC_TRUTH_SINGLE_TAG_ERR:0;
      const eCar=Math.hypot(0.5*PC_TRUTH_CAR_ACC*carGa*carGb, eSingle);
      const carGap=Math.max(carGa,carGb);
      const seMax=Math.max(seX,seZ);
      const resMax=Math.max(zMax,xMax);
      // 显示误差下限取 max|残差|：统计上均值估计可优于单点散布，但对外宣称
      // 不应低于模型对数据的实际解释能力（防"跨 0.3s 断档标 ±1mm"式过度自信）
      const err=Math.max(Math.sqrt(seMax*seMax+eModel*eModel+eCar*eCar), resMax);
      return {x:fx[0]-c.x, y:fy[0]-c.y, z:zAt(0), gap, err, resMax, dNear, delta,
              carGap, carGa, carGb, eCar, carSingleTag:!!c.single};
    }
    let gi=-1, gmax=0;
    for(let i=1;i<win.length;i++){
      const g=win[i].t-win[i-1].t;
      if(g>gmax){ gmax=g; gi=i; }
    }
    if(gi<0 || gmax<0.1) return null;
    win=win.slice(gi);                                 // 只保留最靠近目标时刻的连续入弧
  }
  return null;
};
// [[pc-truth-core-end]]
// PC 真值为空时说清是哪一道门拦的：光一个 "—" 分不清"这场没数据"和"这一刻缺小车位姿"。
// 0809 103849 场 9/16 抛为空，其中 8 抛是 PC AprilTag 小车定位在击球时刻整段黑掉
// （2~3.5s 无一帧成功），球观测本身有 14~20 点、外推只有 23~50ms——完全不是球的问题。
const pcTruthWhy = tPc => {
  if(!isNum(tPc)) return null;
  const n=nearest(pcCarRows,tPc);
  if(!carAt(tPc)){
    if(!n) return 'PC 小车定位在该时刻缺失：本场没有可用的 /pc_car_loc';
    const d=tPc-n.t;
    // 说清"要是照旧冻结这一帧会假造多少 cm"——不给量级，看的人分不清是无伤大雅还是致命。
    const i=pcCarRows.indexOf(n), p=(i>0?pcCarRows[i-1]:null);
    const v=(p && n.t-p.t>1e-3 && n.t-p.t<0.5)
      ? Math.hypot((n.x-p.x)/(n.t-p.t),(n.y-p.y)/(n.t-p.t)) : null;
    return 'PC 小车定位在该时刻没被前后两条 /pc_car_loc 夹住：最近一条在 '+
      (d>=0?'前 ':'后 ')+Math.round(Math.abs(d)*1000)+'ms'+
      '（车位姿只做 ≤0.5s 有界插值，不外推、不冻结）'+
      (v!=null ? '；最后一帧处车速 '+v.toFixed(2)+'m/s ⇒ 若照旧拿它当同时刻位姿，'+
                 '真值会被整体平移 ~'+(Math.abs(v*d)*100).toFixed(0)+'cm 量级'+
                 '（按等速折算；车在变速时实际略小）' : '')+
      '；球侧不影响';
  }
  const raw=pcRows.filter(p=>tPc-p.t>=0.02 && tPc-p.t<=0.75);
  if(raw.length<5) return 'PC 球观测不足：目标时刻前 [20,750]ms 只有 '+raw.length+' 点（需 ≥5）';
  const d=tPc-raw[raw.length-1].t;
  if(d>0.35) return 'PC 球观测断在目标时刻前 '+Math.round(d*1000)+'ms（外推上限 350ms）';
  return '入弧拟合未过残差门（切掉弹跳/断档后点数不足，或 max|残差| 超 z 3.5cm / x 5cm）';
};
const pcTruthMissCell = tPc => {
  const why=pcTruthWhy(tPc);
  return why ? '<span style="color:#c98a7a;cursor:help" title="'+tableEsc(why)+'">—</span>' : '—';
};
// devtools 排障出口（不进 core 段——core 段要在 node 里跑，没有 window）
window.__dbgPcTruth = {
  at: t => pcTruthAt(t),
  why: t => {
    const raw=pcRows.filter(p=>t-p.t>=0.02 && t-p.t<=0.75);
    const last=raw.length?raw[raw.length-1].t:null;
    return {car:!!carAt(t), nRaw:raw.length,
            dNear:last==null?null:+(t-last).toFixed(3), fit:pcTruthAt(t)!=null};
  },
  rows: () => pcRows,
};
// ---- 臂侧共享数据 ----
// v3 起臂数据与 RK 数据同钟（RK 单调钟绝对秒，extract_arm_bag.py 里由
// header.stamp 经常数换算）：减 RK.t0 落到 RK 相对轴后，与 RK 轨迹/预测
// 完全同轴 —— 全项目只有 PC/RK 两个时间轴，显示统一走 rkToPc 仿射映射。
// v2 旧文件（bag 接收轴）不再桥接：显示未对齐提示，重跑 extract_arm_bag.py 即可。
const armAligned = !!(ARM && ARM.time_axis==='rk_mono_abs_s' && RK && isNum(RK.t0));
if(armAligned){
  ['states','commands','events'].forEach(k=>{
    (ARM[k]||[]).forEach(r=>{ if(isNum(r.t)) r.t -= RK.t0; });
  });
}
const armToPc = t => armAligned ? rkToPc(t) : t;
// 击打时刻：accepted hit 状态自带 duration/hit_time，按 receive_hit 的公式
// start_hit = accept_t + duration − hit_time，finish_hit = accept_t + duration。
// 同一抛会连续 accept 多条更新：聚类为一组，cmd 取第一条（臂开始动作），
// start/done/目标取最后一条（最终执行的时序与目标）。
// 每条 accepted 用 x、机械臂当前 z 补偿和接触 HT 回配实际消费的 /predict_hit_pos，
// 并统计臂端 z 偏移 zOff = accepted_z − rel_z。
// [[arm-prediction-match-core-begin]]
// x/z 双 5e-4 精确匹配是唯一强键；ht 只作 sanity 防跨抛误配。ht−(accept_t+duration)
// 的系统差 = 臂内提前量 HIT_TIME_ADVANCE_SEC − 状态发布开销（0802/0803晨 ~+9ms、
// 0803夜 ~+14ms、0804 ~−1ms），5ms 容差会把全场回配打成 0（073646/064744 实测），
// 放宽到 30ms。z 偏移与提前量都随 arm_controller config.py 改版跳变，写死会让全场
// accepted 回配归零（0804 场 −0.153 命中 0/80），故这里只留缺省、由 armConstCal 逐场自标定。
let ARM_HIT_Z_OFFSET=-0.164;   // config.HIT_POS_Z_OFFSET：2026-08-04 起 −0.164（此前 −0.153）
const armPredictionMatchesAccepted = (p,acceptT,acceptX,acceptZ,acceptDuration) => {
  const zOffset=p.relSrc==='panel'?0:ARM_HIT_Z_OFFSET;
  return Math.abs(p.rel_x-acceptX)<5e-4 &&
    Math.abs(p.rel_z+zOffset-acceptZ)<5e-4 &&
    Math.abs(p.ht-(acceptT+acceptDuration))<3e-2;
};
// [[arm-prediction-match-core-end]]
// arm_controller 常量（config.py）：臂内触球 = 原始 ht − HIT_TIME_ADVANCE_SEC。
// 2026-08-04 起归零——方向修正统一收敛到 SWING_J1_LEAD_SEC 的 J1 角度提前（不动时间轴），
// 该量不再兼职修拍面 yaw；历史值 10ms（0802~0803晨）/15ms（0803夜）。同样只作缺省，
// 实际值由 armConstCal 从本场 accepted 自标定。
let HIT_TIME_ADVANCE_SEC=0.0;
// 挥拍中 ht 重定相（arm_controller 2026-08-03）：挥拍窗（触球前 HIT_T）内到达的新预测不再
// 拒收，只把触球时刻存进 swing.late_ht（status: `late ht saved: contact in Xs`）；到
// 「老触球 − SWING_HT_UPDATE_LEAD_SEC」那一个 tick 用最后存下的那条一次性重建挥拍段，
// 并置 ht_replanned=True——之后的消息一律 `reject hit: hit phase in progress`。故
// **该次挥拍最后一条 late ht saved 就是被消费的那条**（触发 tick 之后不可能再有 saved）。
// 新触球距触发点不足 SWING_HT_UPDATE_MIN_REMAINING_SEC 则放弃重定相、沿用老时间轴。
// reswing 结果只写 controller.last_status、没有发上 /tennis/status，所以这里按同款条件离线重建。
const SWING_HT_UPDATE_LEAD_SEC=0.100;
const SWING_HT_UPDATE_MIN_REMAINING_SEC=0.060;
// [[arm-swing-ht-core-begin]] —— test_report_prediction_contract.py 抽取此段在 node 里回归
// 解析失败必须计数：extract_arm_bag 曾把事件文本截到 500 字，RK 加字段后 payload 越过阈值
// → 这里整场 catch 掉 → armPreds 空 → 回配 0 票 → 臂表整列 —，页面上却没有任何报错
// （0809 103849 场）。截断已修，但解析失败仍要显式暴露，见 armDataWarnHtml。
let armPredParseBad=0, armPredTotal=0;
const armPreds = (()=>{
  if(!ARM) return [];
  const out=[];
  (ARM.events||[]).forEach(e=>{
    if(e.topic!=='/predict_hit_pos') return;
    armPredTotal+=1;
    try{
      const p=JSON.parse(e.text);
      out.push({t:e.t, rel_x:Number(p.rel_x), rel_y:Number(p.rel_y), rel_z:Number(p.rel_z),
                xWorld:Number(p.x), carPredX:Number(p.car_pred_x),
                duration:Number(p.duration), ht:Number(p.ht), ct:Number(p.ct), stage:Number(p.stage),
                nFit:Number(p.n_bounce_fit), relSrc:String(p.rel_src||'?')});
    }catch(err){ armPredParseBad+=1; }
  });
  return out;
})();
// [[arm-const-cal-core-begin]]
// 臂端两个配置量逐场自标定（写死会随控制端改版整场失效）：
//   候选对 = accepted 前 0.25s 内 rel_x 命中 5e-4 且 |ht−(acc_t+dur)|<30ms 的预测消息
//   （x 与 ht 两把弱键联合足以圈出正确消息族；z 正是待标定量，不能进筛选）。
//   · z 偏移 = (acc_z−rel_z) 的众数（0.1mm 分辨率；同一抛多条候选投票，压掉误配尾巴）。
//   · C = median(ht−acc_t−dur)（只统计投给众数 z 的候选）= 提前量 − 状态发布开销
//     （开销实测 0.3~1.4ms）；提前量是整毫秒配置量，故 ADV = round(C+1ms)。
//   实测 0802/0803晨 C=+8.9→10ms、0803夜 +13.9→15ms、0804 −1.1→0ms，与 config.py 历史逐场对上。
// 票数 <3 时保留缺省值（页面注记标出），不拿噪声改合同。
const armConstCal = (()=>{
  const t0 = (RK && isNum(RK.t0)) ? RK.t0 : 0;
  const votes=new Map(), cands=[];
  ((ARM&&ARM.events)||[]).forEach(e=>{
    if(e.topic!=='/tennis/status') return;
    const m=/^accepted hit x=([\-0-9.]+) z=([\-0-9.]+) duration=([0-9.]+)/.exec(e.text);
    if(!m) return;
    const ax=Number(m[1]), az=Number(m[2]), dur=Number(m[3]), at=e.t+t0;
    armPreds.forEach(p=>{
      if(p.relSrc==='panel' || p.t>e.t || e.t-p.t>0.25) return;
      if(!isNum(p.rel_x)||!isNum(p.rel_z)||!isNum(p.ht)) return;
      if(Math.abs(p.rel_x-ax)>=5e-4) return;
      const c=p.ht-at-dur;
      if(Math.abs(c)>=3e-2) return;
      const key=Math.round((az-p.rel_z)*1e4)/1e4;
      votes.set(key,(votes.get(key)||0)+1);
      cands.push({key,c});
    });
  });
  let zOff=null, best=0, total=0;
  votes.forEach((n,k)=>{ total+=n; if(n>best){ best=n; zOff=k; } });
  if(best<3) return {zOff:null, adv:null, n:best, total, c:null};
  const cs=cands.filter(r=>r.key===zOff).map(r=>r.c).sort((a,b)=>a-b);
  const c=cs[cs.length>>1];
  const adv=Math.round(c*1000+1)/1000;
  return {zOff, adv:(adv>=0&&adv<=0.03)?adv:null, n:best, total, c};
})();
if(armConstCal.zOff!=null) ARM_HIT_Z_OFFSET=armConstCal.zOff;
if(armConstCal.adv!=null) HIT_TIME_ADVANCE_SEC=armConstCal.adv;
// [[arm-const-cal-core-end]]
// 臂数据整体失效的两种模式，以前都只在折叠备注里留一行，页面主体只剩满屏 "—"，
// 看上去像"这场没数据"而不是"报告读不了数据"。这里升成表头红条。
// [[arm-data-warn-core-begin]]
const armDataWarnHtml = (()=>{
  if(!ARM) return '';
  const accN=((ARM.events||[]).filter(e=>e.topic==='/tennis/status'&&
    /^accepted hit x=/.test(String(e.text||'')))).length;
  const msgs=[];
  if(armPredParseBad>0){
    msgs.push('/predict_hit_pos 载荷解析失败 '+armPredParseBad+'/'+armPredTotal+' 条'+
      '——多半是 extract_arm_bag 把事件文本截断了（RK 端加字段后 payload 变长会撞上限）。'+
      '重跑 test_src/extract_arm_bag.py 出 _arm.json 后再重生成本页。');
  }
  if(accN>0 && armConstCal.n<3){
    msgs.push('本场 '+accN+' 条 accepted hit 没能回配到原始 /predict_hit_pos'+
      '（臂端常量自标定只有 '+armConstCal.n+' 票 <3，沿用缺省 z 偏移/提前量）'+
      '——accepted 目标、击球真值、TCP 各列都会整列为 —。');
  }
  if(!msgs.length) return '';
  return '<div style="border:1px solid #e94560;background:rgba(233,69,96,0.12);color:#e94560;'+
    'font-weight:600;border-radius:8px;padding:8px 12px;margin:0 0 10px">⚠ 臂数据读取失败：'+
    msgs.join('　')+'</div>';
})();
// [[arm-data-warn-core-end]]
// /tennis/status 尾部的 `key=数值` 字段取值：字段随 arm_controller 版本增删（可变 pitch 0805、
// 拍速指定 0808…），老 bag 取不到就返回 null 让对应列显示 —。key 后必须紧跟 =，故取 'speed'
// 不会命中 'speed_req'。
const statusNum = (text,key) => {
  const m=new RegExp('(?:^|\\s)'+key+'=(-?[0-9]+(?:\\.[0-9]+)?)').exec(text||'');
  return m?Number(m[1]):null;
};
const _armHit = (()=>{
  if(!ARM) return {marks:[], zOff:null};
  const out=[], zOffs=[];
  // late ht saved 状态只写 duration、不写 ht，必须回配到原 /predict_hit_pos 才拿得到原始 ht。
  // on_hit_pos 里两者严格一对一（收一条必回一条），且两个 topic 各自的时间序在 bag 里都保序，
  // 但**跨 topic 的 bag 读出顺序会错位**（同一场实测出现过 S,S,P,P），所以不能按事件流 FIFO 配。
  // 做法：以 x/z 双 5e-4 精确回配成功的那条 accepted 为锚，按「hit_pos 派生状态的序号差 =
  // 预测序号差」平移——同一次挥拍里锚点与 late 只隔 ~300ms，中间不会丢消息。
  // 每条再自校验 now=ht−advance−duration 必须落在 [状态发布时刻−8ms, +0.6ms]。gap=发布时刻−now
  // 全在 arm_controller 进程内、同一单调钟（status 尾缀 t= 就是它发布时读的 perf_counter），
  // 不含 DDS 与 PC：status 写进 last_status 后要等下一个 100Hz tick 才 push，故 gap 是 sub-tick
  // 抖动，上限 = 一个 tick 10ms。旧上界 4ms 按"发布开销 0.3~1.4ms"取，0809 场实测到 4.90ms，
  // 把 #6/#13 两拍判成失配、盲区列静默退回 accepted（174ms 真值显示成 327ms），故放宽到 8ms。
  // 8ms 仍远小于错配一格的 32.5ms（视觉帧间隔），判别力不受影响。
  let si=-1;              // hit_pos 派生状态的序号
  let anchorSi=null, anchorPi=null;
  (ARM.events||[]).forEach(e=>{
    if(e.topic!=='/tennis/status') return;
    if(!/^(accepted hit |late ht saved|reject hit:|error hit_pos)/.test(e.text)) return;
    si++;
    let rec=null;
    let m=/^late ht saved: contact in ([0-9.]+)s/.exec(e.text);
    if(m){
      const cur=out[out.length-1];
      if(cur && cur.label==='hit' && e.t<=cur.done+0.05){
        const dur=Number(m[1]);
        const p=anchorSi!=null?armPreds[anchorPi+(si-anchorSi)]:null;
        const gap=p?(e.t+RK.t0-(p.ht-HIT_TIME_ADVANCE_SEC-dur)):null;
        const ok=!!(p && gap>=-0.0006 && gap<=0.008);
        const ht=ok?p.ht:null;
        cur.lates.push({t:e.t, dur, ht, ct:ok?p.ct:null,
                        hitTime:ht!=null?ht-RK.t0-HIT_TIME_ADVANCE_SEC:e.t+dur});
      }
      return;
    }
    m=/^accepted hit x=([\-0-9.]+) z=([\-0-9.]+) duration=([0-9.]+)(?:\s+hit_time=([0-9.]+))?/.exec(e.text);
    if(m){
      const dur=Number(m[3]), hitT=m[4]!=null?Number(m[4]):0.4;
      rec={cmd:e.t, lastAcceptT:e.t, tx:Number(m[1]), tz:Number(m[2]), start:e.t+dur-hitT, done:e.t+dur,
           label:'hit', n:1, lates:[], hitT,
           // 挥拍计划量（老场次没有这些字段就是 null，列显示 —）：
           //   pitch=     0805 起的可变拍面仰角目标（°，臂系≡世界系）
           //   speed=     0808 起恒带的计划触球拍速（m/s，拍心），已过各级钳位
           //   speed_req= 钳位改动了拍速时补的原始指令值；shortened= 引拍被夹掉的 rad
           //   face_yaw=  臂端锁面目标（rad，臂系）
           tgtPitch:statusNum(e.text,'pitch'), tgtSpeed:statusNum(e.text,'speed'),
           tgtSpeedReq:statusNum(e.text,'speed_req'), shortened:statusNum(e.text,'shortened'),
           tgtFaceYaw:statusNum(e.text,'face_yaw'),
           wx:null, wy:null, wz:null, wxw:null, wcarx:null, wct:null, wht:null, wpredT:null, wstage:null, wnFit:null};
      for(let i=armPreds.length-1;i>=0;i--){
        const p=armPreds[i];
        if(p.t>e.t) continue;
        if(e.t-p.t>0.25) break;
        if(armPredictionMatchesAccepted(p,e.t+RK.t0,rec.tx,rec.tz,dur)){
          rec.wx=p.rel_x; rec.wy=p.rel_y; rec.wz=p.rel_z; rec.wct=p.ct; rec.wht=p.ht;
          rec.wxw=isNum(p.xWorld)?p.xWorld:null; rec.wcarx=isNum(p.carPredX)?p.carPredX:null;
          rec.wpredT=p.t; rec.wstage=p.stage; rec.wnFit=p.nFit;
          anchorSi=si; anchorPi=i;   // late ht saved 的序号平移锚点
          if(isNum(p.rel_z)) zOffs.push(rec.tz-p.rel_z);
          break;
        }
      }
    } else {
      m=/^accepted arm_command (\w+) duration=([0-9.]+)/.exec(e.text);
      if(m) rec={cmd:e.t, lastAcceptT:e.t, start:null, done:e.t+Number(m[2]), label:m[1], n:1};
    }
    if(!rec) return;
    const cur=out[out.length-1];
    if(cur && cur.label===rec.label && Math.abs(rec.done-cur.done)<0.6){
      rec.cmd=cur.cmd; rec.n=cur.n+1;  // cmd 保留第一条，其余取最后一条
      if(cur.lates && cur.lates.length) rec.lates=cur.lates;
      out[out.length-1]=rec;
    } else {
      out.push(rec);
    }
  });
  // 逐拍重建 ht 重定相：lastUpdateT = 臂真正受理的最后一条 /predict_hit_pos 的到达时刻
  // （挥拍窗内的 late ht saved 也算受理，它就是重定相的养料）；finalHt/finalCt = 那条消息的
  // **原始 ht/ct**（不减臂内 10ms 提前量），ht−ct 即该拍最终命令的盲区时间（击球点 − 最晚
  // 那颗球的观测时刻）。重定相未生效（无 late / 剩余不足）时退回最后一条 accepted——那时臂本来
  // 就只用了 accepted 那条，退回是**正确值**。
  // ⚠ 唯一不能退回的是「重定相生效但回配失配」：此时臂用的是 late 那条的 ht/ct，但报告拿不到
  // 同源真值，退回 accepted 会把 ~174ms 的盲区冒充成 ~330ms（0809 场 #6/#13 就是这么错的）。
  // 这种情况置 finalMismatch，由盲区列显示 ⚠— 而不是给一个错的数。
  out.forEach(h=>{
    if(h.label!=='hit') return;
    h.lastUpdateT=h.lastAcceptT;
    h.finalHt=isNum(h.wht)?h.wht:null;
    h.finalCt=isNum(h.wct)?h.wct:null;
    h.finalDone=h.done;
    h.finalMismatch=false;
    h.reswing=null;
    const lates=h.lates||[];
    if(!lates.length) return;
    const last=lates[lates.length-1];   // 触发 tick 后一律 reject，故最后一条即被消费的那条
    const trig=h.done-SWING_HT_UPDATE_LEAD_SEC;
    const ok=last.hitTime-trig>=SWING_HT_UPDATE_MIN_REMAINING_SEC;
    h.lastUpdateT=last.t;
    h.reswing={trig, ok, n:lates.length, oldDone:h.done, newDone:last.hitTime,
               delta:(last.hitTime-h.done)*1000, remain:last.hitTime-trig,
               ht:last.ht, ct:last.ct};
    if(ok){
      h.finalDone=last.hitTime;
      if(last.ht!=null){ h.finalHt=last.ht; h.finalCt=last.ct; }  // ht/ct 必须同源
      else h.finalMismatch=true;                                   // 见上：不许拿 accepted 冒充
    }
  });
  zOffs.sort((a,b)=>a-b);
  return {marks:out, zOff:zOffs.length?zOffs[zOffs.length>>1]:null};
})();
// [[arm-swing-ht-core-end]]
const armHitMarks = _armHit.marks;
const armZOff = _armHit.zOff;  // 臂系 z − 世界系 z；FK 还原世界系 = tcp_z − armZOff
const armTcpRows = ARM ? ARM.states.filter(s=>Array.isArray(s.tcp)) : [];
// 在机械臂 state 时间轴按 HT 有界插值 FK TCP；相邻 state 超过 100ms 时不外推。
const tcpAt = t => {
  const s=interpRow(armTcpRows,t,0.1);
  if(!s || !Array.isArray(s.a.tcp) || !Array.isArray(s.b.tcp)) return null;
  return [0,1,2].map(k=>lerp(s.a.tcp[k],s.b.tcp[k],s.f));
};
// annotate_video 离线三角测量的拍心（世界系, m）：PC 报告轴，重投影 >30px 丢弃
const pcRacketRows = racket
  .map(r=>({t:isNum(r.rel_s)?r.rel_s:relTime(r.t), x:r.x, y:r.y, z:r.z,
            rp:isNum(r.reproj_err)?r.reproj_err:(isNum(r.reproj)?r.reproj:null)}))
  .filter(p=>isNum(p.t)&&isNum(p.x)&&(p.rp==null||p.rp<=30))
  .sort((a,b)=>a.t-b.t);
// [[racket-fit-core-begin]]
const RACKET_FIT_WINDOW_SEC=0.55;
const RACKET_MAX_GAP_SEC=0.18;
const RACKET_MAX_EXTRAP_SEC=0.25;
const RACKET_ACCEL_BOUND_MPS2=20;
const solveSystem = (a,b) => {
  const n=b.length, m=a.map((row,i)=>[...row,b[i]]);
  for(let col=0;col<n;col++){
    let pivot=col;
    for(let row=col+1;row<n;row++) if(Math.abs(m[row][col])>Math.abs(m[pivot][col])) pivot=row;
    if(Math.abs(m[pivot][col])<1e-12) return null;
    [m[col],m[pivot]]=[m[pivot],m[col]];
    const div=m[col][col];
    for(let j=col;j<=n;j++) m[col][j]/=div;
    for(let row=0;row<n;row++){
      if(row===col) continue;
      const f=m[row][col];
      for(let j=col;j<=n;j++) m[row][j]-=f*m[col][j];
    }
  }
  return m.map(row=>row[n]);
};
const fitMotion = (rows,t,key,degree) => {
  const n=degree+1, a=Array.from({length:n},()=>Array(n).fill(0)), b=Array(n).fill(0);
  rows.forEach(row=>{
    const dt=row.t-t, powers=[1,dt,dt*dt,dt*dt*dt,dt*dt*dt*dt];
    for(let i=0;i<n;i++){
      b[i]+=row[key]*powers[i];
      for(let j=0;j<n;j++) a[i][j]+=powers[i+j];
    }
  });
  const beta=solveSystem(a,b), invCol=solveSystem(a,[1,...Array(n-1).fill(0)]);
  if(!beta||!invCol) return null;
  let sse=0, resMax=0;
  rows.forEach(row=>{
    const dt=row.t-t, pred=beta[0]+beta[1]*dt+(degree===2?beta[2]*dt*dt:0);
    const res=Math.abs(row[key]-pred);
    sse+=res*res; resMax=Math.max(resMax,res);
  });
  const se=Math.sqrt(Math.max(0,sse/Math.max(1,rows.length-n)*invCol[0]));
  return {p:beta[0], v:beta[1], err:Math.max(resMax,1.96*se)};
};
const fitVisualRacketRows = (sourceRows,t) => {
  const candidates=sourceRows.filter(r=>Math.abs(r.t-t)<=RACKET_FIT_WINDOW_SEC&&isNum(r.x)&&isNum(r.y)&&isNum(r.z));
  if(candidates.length<4) return null;
  const segments=[];
  let seg=[];
  candidates.forEach(row=>{
    if(seg.length&&row.t-seg[seg.length-1].t>RACKET_MAX_GAP_SEC){segments.push(seg);seg=[];}
    seg.push(row);
  });
  if(seg.length) segments.push(seg);
  const distance=rows=>Math.min(...rows.map(row=>Math.abs(row.t-t)));
  let rows=segments.reduce((best,rows)=>distance(rows)<distance(best)?rows:best,segments[0]);
  rows=[...rows].sort((a,b)=>Math.abs(a.t-t)-Math.abs(b.t-t)).slice(0,10).sort((a,b)=>a.t-b.t);
  const dNear=distance(rows), span=rows[rows.length-1].t-rows[0].t;
  if(rows.length<4||span<0.1||dNear>RACKET_MAX_EXTRAP_SEC) return null;
  const bracketed=rows[0].t<=t&&rows[rows.length-1].t>=t;
  const degree=bracketed&&rows.length>=6?2:1;
  const fx=fitMotion(rows,t,'x',degree), fy=fitMotion(rows,t,'y',degree), fz=fitMotion(rows,t,'z',degree);
  if(!fx||!fy||!fz) return null;
  const modelErr=bracketed?0:0.5*RACKET_ACCEL_BOUND_MPS2*dNear*dNear;
  return {x:fx.p,y:fy.p,z:fz.p,vx:fx.v,vy:fy.v,vz:fz.v,
          err:Math.max(fx.err,fz.err,modelErr),n:rows.length,span,dNear,
          mode:bracketed?'interpolation':'extrapolation',degree};
};
// [[racket-fit-core-end]]
// 命中判定：触球后 0.8s 内球 y 是否由进(−)转出(+)——被拍打回才会反向；
// 地面反弹只弹 z，y 方向不变。窗口内无观测（遮挡/出视场）判"观测缺失"。
const strikeAfter = tPc => {
  const seg=pcRows.filter(p=>p.t>=tPc-0.05 && p.t<=tPc+0.8);
  if(seg.length<4) return {verdict:'观测缺失', hit:null};
  let run=0;
  for(let i=1;i<seg.length;i++){
    const dt=seg[i].t-seg[i-1].t;
    if(dt<=0||dt>0.2){ run=0; continue; }
    const vy=(seg[i].y-seg[i-1].y)/dt;
    run = vy>0.5 ? run+1 : 0;
    if(run>=2) return {verdict:'命中', hit:true};
  }
  return {verdict:'脱拍', hit:false};
};
// [[pc-return-core-begin]]
// PC 回球统计（All-in-One 新列；annotate_video.py 离线叠加同口径）：
// 1) 触球时刻 = 入弧/出弧 y(t) 二次拟合交点（来回交点法）：入弧 [−380,−25]ms、出弧 [+30,+330]ms，
//    入弧 vy<−1（真来球）、出弧 vy>0.15 才认定，交点限 [−100,+140]ms 取最小 |u|；
// 2) 回球速度 = 触球后 [+20,+400]ms 出弧连续段 x/y/z 拟合在触球时刻的导数（二次，<6点线性；
//    z 先扣 ½g·u² 再拟合，线性退化时 vz 不吃重力偏差）。
// 脏数据过滤：z<0.12 贴地/静止球剔除；观测按断档>150ms 或帧间位移速>20m/s 跳变（检测器接上
// 场上别的球；门槛在最快回球 ~15m/s 之上）切成连续段，按点数优先选首个通过"真出向"门槛的段——
// 静止球段 vy≈0 自然被拒，真回球被拍/臂遮挡断档后仍能用遮挡后的真弧（段首距触球 ≤300ms）；
// vz 由降(<−0.5)转升突增>3m/s 判地面反弹截断（防反弹污染俯仰角）；
// 出弧 vy≤0.5、水平速<1m/s、入弧<5点、出弧<5点或跨度<60ms 一律不认定回球。
const quadFitU = (pts, tc) => {
  const n=pts.length;
  if(n<3) return null;
  let su=0,suu=0,suuu=0,suuuu=0,sv=0,suv=0,suuv=0;
  for(const p of pts){
    const u=p.t-tc, v=p.v;
    su+=u; suu+=u*u; suuu+=u*u*u; suuuu+=u*u*u*u;
    sv+=v; suv+=u*v; suuv+=u*u*v;
  }
  if(n>=6){
    const sol=solve3([n,su,suu,suu,suuu,suuuu],[sv,suv,suuv]);
    if(sol) return {a:sol[0],b:sol[1],c:sol[2],n};
  }
  const den=n*suu-su*su;
  if(Math.abs(den)<1e-12) return null;
  return {a:(sv*suu-su*suv)/den, b:(n*suv-su*sv)/den, c:0, n};
};
// 把 [lo,hi] 内 z≥0.12 观测按 断档>150ms 或 帧间位移速>20m/s 跳变 切成连续段。
// 分段门 20m/s：快回球本身可达 12+m/s，更低的门（如 12）会把真出弧切碎
// （073646 实测 5 抛 12.2~12.4m/s 被误切）；隔帧跳到另一颗球的隐含速度通常 30m/s 以上。
// 真回球常在触球后被拍/臂遮挡断档，段选择（而非首段截断）保证遮挡后真弧仍可用；
// 静止球/换球段靠后续 vy 门槛拒绝，不会给出假值。
const RETURN_MAX_STEP_MPS=20;
const pcRuns = (lo, hi) => {
  const runs=[];
  let run=[];
  for(const p of pcRows){
    if(p.t<lo) continue;
    if(p.t>hi) break;
    if(p.z<0.12) continue;
    const last=run[run.length-1];
    if(last){
      const dt=p.t-last.t;
      if(dt>0.15 || Math.hypot(p.x-last.x,p.y-last.y,p.z-last.z)/Math.max(1e-9,dt)>RETURN_MAX_STEP_MPS){
        if(run.length) runs.push(run);
        run=[];
      }
    }
    run.push(p);
  }
  if(run.length) runs.push(run);
  return runs;
};
const RETURN_HALF_G=4.905;  // z 拟合先扣重力 ½g·u²，线性退化时 vz 不再吃重力偏差
const pcHitTimeAt = tApprox => {
  // 入弧取最贴近触球且 ≥5 点的连续段
  const yinRun=[...pcRuns(tApprox-0.38,tApprox-0.025)].reverse().find(r=>r.length>=5);
  if(!yinRun) return null;
  const fin=quadFitU(yinRun.map(p=>({t:p.t,v:p.y})),tApprox);
  if(!fin||fin.b>-1.0) return null;
  // 出弧段按点数从多到少试，取首个真出向（vy>0.15）且交点落窗的段
  const outRuns=pcRuns(tApprox+0.03,tApprox+0.33).filter(r=>r.length>=4)
    .sort((a,b)=>b.length-a.length);
  for(const run of outRuns){
    const fout=quadFitU(run.map(p=>({t:p.t,v:p.y})),tApprox);
    if(!fout||fout.b<0.15) continue;
    const a=fin.c-fout.c, b=fin.b-fout.b, c=fin.a-fout.a;
    let roots=[];
    if(Math.abs(a)<1e-9){ if(Math.abs(b)>1e-9) roots=[-c/b]; }
    else{
      const disc=b*b-4*a*c;
      if(disc>=0){ const r=Math.sqrt(disc); roots=[(-b+r)/(2*a),(-b-r)/(2*a)]; }
    }
    roots=roots.filter(u=>u>=-0.10&&u<=0.14);
    if(roots.length) return tApprox+roots.reduce((m,u)=>Math.abs(u)<Math.abs(m)?u:m,roots[0]);
  }
  return null;
};
// 段内地面反弹截断：vz 由降(<−0.5)转升突增>3m/s 的位置截断，防反弹污染俯仰角
const bounceCutRun = run => {
  for(let i=2;i<run.length;i++){
    const vzA=(run[i-1].z-run[i-2].z)/Math.max(1e-9,run[i-1].t-run[i-2].t);
    const vzB=(run[i].z-run[i-1].z)/Math.max(1e-9,run[i].t-run[i-1].t);
    if(vzA<-0.5 && vzB-vzA>3.0) return {seg:run.slice(0,i), bounceCut:true};
  }
  return {seg:run, bounceCut:false};
};
// 锚点（RK HT/accepted HT）有偏差时交点窗会混入对侧弧：小范围扫描锚点，
// 找到粗交点后再用它当锚点细化一次。
const pcHitTimeNear = tApprox => {
  if(!isNum(tApprox)) return null;
  let tHit=null;
  for(const d of [0,-0.06,0.06,-0.12,0.12]){
    tHit=pcHitTimeAt(tApprox+d);
    if(tHit!=null) break;
  }
  if(tHit==null) return null;
  const refined=pcHitTimeAt(tHit);
  return refined!=null ? refined : tHit;
};
const pcReturnAt = tApprox => {
  const tHit=pcHitTimeNear(tApprox);
  if(tHit==null) return null;
  // 出弧速度段：段首距触球 ≤300ms（限制回推外推量），按点数优先取首个过门槛的段
  const cands=pcRuns(tHit+0.02,tHit+0.40).filter(r=>r.length>=5 && r[0].t-tHit<=0.30)
    .sort((a,b)=>b.length-a.length);
  for(const run of cands){
    const {seg,bounceCut}=bounceCutRun(run);
    if(seg.length<5 || seg[seg.length-1].t-seg[0].t<0.06) continue;
    const fx=quadFitU(seg.map(p=>({t:p.t,v:p.x})),tHit);
    const fy=quadFitU(seg.map(p=>({t:p.t,v:p.y})),tHit);
    const fz=quadFitU(seg.map(p=>({t:p.t,v:p.z+RETURN_HALF_G*(p.t-tHit)*(p.t-tHit)})),tHit);
    if(!fx||!fy||!fz) continue;
    const vx=fx.b, vy=fy.b, vz=fz.b;
    if(vy<=0.5) continue;
    const vh=Math.hypot(vx,vy);
    if(vh<1.0) continue;
    return {tHit, vx, vy, vz,
      yaw:Math.atan2(vx,vy)*180/Math.PI,
      pitch:Math.atan2(vz,vh)*180/Math.PI,
      speed:Math.hypot(vx,vy,vz),
      n:seg.length, span:seg[seg.length-1].t-seg[0].t,
      start:seg[0].t-tHit, bounceCut};
  }
  return null;
};
// [[pc-return-core-end]]
// 拍面世界yaw,pitch@臂最后更新HT：臂系 face_yaw/face_pitch（Python _add_face_angles 逐帧 FK，
// link6 +X 法向）取冲击前窗 [−80,−6]ms 线性外推到该 HT——J5 冲击突跳（触始+13ms 机械传递，
// 本场 ≈ht−3ms）会污染跨冲击帧的直接插值（rebound 位姿采样同款约定）；车 yaw 取挥拍前
// [−450,−150]ms 窗圆均值（击球 ±0.4s 车位姿有挥拍塌陷伪迹，禁读）。ψ_world = fy − 车yaw，
// 口径与 PC回球 yaw 同为 atan2(x,y)（车体 CCW 正旋转使 ψ_world 变小）；纯 FK 值，不含 δ6 球侧偏置。
// **pitch 不减车 yaw**：J1/BASE_ROT 是纯 z 转、不动 n_z，故臂系 pitch ≡ 世界 pitch（车无 roll/pitch
// 前提），θ_world 比 ψ_world 少一个车侧误差源；且 θ̇≈0（挥拍是近水平圆弧，法向在 yaw 里飞扫、
// 在 pitch 里冻结），故 pitch 对时序误差天然免疫，两列之差应当接近 0。
const armFaceRows = ARM ? ARM.states.filter(s=>isNum(s.fy)&&isNum(s.fp)) : [];
const wrapDeg = d => ((d+180)%360+360)%360-180;
// 冲击前窗 [−80,−6]ms（相对该 HT）线性拟合 fy 与 fp，在 tEval 处取值：
// tEval=htRk 为 ≤6ms 外推；tEval=htRk−12ms 落在窗内，为纯插值。
const fitFaceAnglesTo = (accHtRk,tEval) => {
  if(accHtRk==null || !isNum(tEval) || !armFaceRows.length) return null;
  const seg=armFaceRows.filter(s=>s.t>=accHtRk-0.08 && s.t<=accHtRk-0.006);
  if(seg.length<2) return null;
  let st=0,sy=0,stt=0,sty=0,sp=0,stp=0;
  seg.forEach(s=>{const u=s.t-tEval; st+=u; stt+=u*u; sy+=s.fy; sty+=u*s.fy; sp+=s.fp; stp+=u*s.fp;});
  const n=seg.length, den=n*stt-st*st;
  if(Math.abs(den)<1e-12) return null;
  return {fy:(sy*stt-st*sty)/den, rate:(n*sty-st*sy)/den,
          fp:(sp*stt-st*stp)/den, pitchRate:(n*stp-st*sp)/den, n};
};
const faceAnglesWorldAt = accHtRk => {
  const fit=fitFaceAnglesTo(accHtRk,accHtRk);
  if(!fit) return null;
  const tPc=rkToPc(accHtRk);
  let ss=0,sc=0,m=0;
  pcCarRows.forEach(r=>{ if(r.t>=tPc-0.45 && r.t<=tPc-0.15 && isNum(r.yaw)){ ss+=Math.sin(r.yaw); sc+=Math.cos(r.yaw); m++; } });
  if(!m) return null;
  const carYaw=Math.atan2(ss,sc)*180/Math.PI;
  return {deg:wrapDeg(fit.fy-carYaw), fy:fit.fy, carYaw, n:fit.n, rate:fit.rate, m,
          pitch:fit.fp, pitchRate:fit.pitchRate};
};
// 拍面yaw,pitch,speed@臂最后更新HT−12ms（世界系）：fy/fp 同款冲击前窗拟合改在 HT−12ms 取值
// （拍速同样在该刻插值）；车 yaw
// 不用挥拍前圆均值，直接取该时刻 /bot_state yaw——车控接受 AprilTag 定位后 yaw 由
// IMU 连续更新，直到 HT 结束后才用 AprilTag 重定位更新 bot_state，因此 HT 前采样点
// 的 bot yaw 是无重定位台阶的瞬时值（挥拍塌陷伪迹只污染位置，yaw 走 IMU 不受累）。
const rkBotYawRows = (()=>{
  if(!RK || !RK.bot) return [];
  const T=ts(RK.bot), Y=ys(RK.bot,'yaw'), rows=[];
  for(let i=0;i<T.length;i++){
    const t=Number(T[i]), yaw=Number(Y[i]);
    if(isNum(t)&&isNum(yaw)) rows.push({t,yaw});
  }
  return rows.sort((a,b)=>a.t-b.t);
})();
const botYawDegAt = t => {
  const s=interpRow(rkBotYawRows,t,0.1);
  if(s){
    const dy=Math.atan2(Math.sin(s.b.yaw-s.a.yaw),Math.cos(s.b.yaw-s.a.yaw));
    return (s.a.yaw+dy*s.f)*180/Math.PI;
  }
  const n=nearest(rkBotYawRows,t);
  return (n && Math.abs(n.t-t)<=0.05) ? n.yaw*180/Math.PI : null;
};
// 车 yaw 角速度：取 /chassis_can/imu 的 yaw_speed（rad/s，零滞后陀螺原值）。bot_state yaw
// 自身有 0.3~0.5s 滤波滞后，绝不能对它数值求导当瞬时角速度；这里只做时序灵敏度提示。
const rkImuYawRateRows = (()=>{
  if(!RK || !RK.imu) return [];
  const T=ts(RK.imu), W=ys(RK.imu,'yaw_speed'), rows=[];
  for(let i=0;i<T.length;i++){
    const t=Number(T[i]), w=Number(W[i]);
    if(isNum(t)&&isNum(w)) rows.push({t,w});
  }
  return rows.sort((a,b)=>a.t-b.t);
})();
const imuYawRateDegAt = t => {
  const n=nearest(rkImuYawRateRows,t);
  return (n && Math.abs(n.t-t)<=0.06) ? n.w*180/Math.PI : null;
};
// 实测拍速@臂最后更新HT（m/s）：vt = Python 侧用同一次 FK 的解析 Jacobian 算出的完整拍心线
// 速度 |v_tcp|（逐帧，见 _add_face_angles），JS 只做插值；J1 分量 |q̇1|·r 与杠杆
// r=hypot(tcp_x,tcp_y)（J1 转轴是过臂基的铅垂线）用已有字段现算，用来和 status `speed=`
// 的口径（2·行程/hit_time·x，只算 J1）直接对齐。
// **拍速不外推**（与同列 yaw/pitch 不同）：挥拍段 J1 走 S 曲线，[−80,−6]ms 窗内拍速强非线性，
// 线性外推到 HT 实测会高估 40%+；HT 处两侧都有 100Hz 采样，直接插值即可。
// 触球锚不靠"实测峰值"找：实测 q̇1 全程叠着 ±0.5~1.4m/s 的伺服振荡（引拍段无球时同样存在，
// 见 osc），74ms 窗里取 argmax 只会落在某个振荡波峰上——0808 首版曾据此误判"峰值=触球、
// 触球早于 ht 39ms"，实为伪迹。真正的臂内触球用**指令自身**定位：HitTrajectory 触球后按恒 ω
// 巡航，故指令 J1 速度进入平台的第一帧就是 finish_hit_time（本场落在 HT−5~−12ms，与臂内
// 提前量对得上），该刻的实测/指令差才是伺服欠速。
const armSpeedRows = ARM ? ARM.states.filter(
  s=>isNum(s.vt)&&Array.isArray(s.tcp)&&Array.isArray(s.velocity)) : [];
const armCmdSpeedRows = ARM ? (ARM.commands||[]).filter(
  c=>Array.isArray(c.tcp)&&Array.isArray(c.velocity)&&isNum(c.velocity[0])) : [];
const j1SpeedOf = row => Math.abs(row.velocity[0])*Math.hypot(row.tcp[0],row.tcp[1]);
const RACKET_SPEED_MAX_GAP_S=0.05;
const cmdSpeedAt = t => {
  const c=interpRow(armCmdSpeedRows,t,RACKET_SPEED_MAX_GAP_S);
  return c?lerp(j1SpeedOf(c.a),j1SpeedOf(c.b),c.f):null;
};
// 任意时刻的实测拍速（HT 与 HT−12ms 两列共用）：|v_tcp| 插值 + J1 分量 + 同刻指令值。
const racketSpeedRawAt = t => {
  if(t==null || !armSpeedRows.length) return null;
  const s=interpRow(armSpeedRows,t,RACKET_SPEED_MAX_GAP_S);
  if(!s) return null;
  return {v:lerp(s.a.vt,s.b.vt,s.f),
          vj1:lerp(j1SpeedOf(s.a),j1SpeedOf(s.b),s.f),
          r:lerp(Math.hypot(s.a.tcp[0],s.a.tcp[1]),Math.hypot(s.b.tcp[0],s.b.tcp[1]),s.f),
          cmd:cmdSpeedAt(t)};
};
const racketSpeedAt = htRk => {
  const base=racketSpeedRawAt(htRk);
  if(!base) return null;
  // 臂内触球锚 = 指令 J1 速度平台的第一帧（挥拍段末端，之后是恒 ω 巡航）
  const seg=armCmdSpeedRows.filter(c=>c.t>=htRk-0.30&&c.t<=htRk+0.12).map(c=>({t:c.t,v:j1SpeedOf(c)}));
  let contactT=null, cmdContact=null, measContact=null;
  if(seg.length>10){
    const vmax=seg.reduce((m,c)=>Math.max(m,c.v),0);
    const hit=seg.find(c=>c.v>=vmax*0.995);
    if(hit){
      contactT=hit.t; cmdContact=vmax;
      const ms=interpRow(armSpeedRows,contactT,RACKET_SPEED_MAX_GAP_S);
      if(ms) measContact=lerp(ms.a.vt,ms.b.vt,ms.f);
    }
  }
  // 单点读数的抖动量级：引拍段[−250,−120]ms（肯定无球）实测−指令 J1 拍速残差的 std
  let sum=0,sum2=0,n=0;
  armSpeedRows.forEach(row=>{
    if(row.t<htRk-0.25||row.t>htRk-0.12) return;
    const c=cmdSpeedAt(row.t);
    if(c==null) return;
    const d=j1SpeedOf(row)-c; sum+=d; sum2+=d*d; n++;
  });
  const osc=n>=4?Math.sqrt(Math.max(0,sum2/n-(sum/n)*(sum/n))):null;
  return Object.assign({}, base,
    {contactDt:contactT!=null?(contactT-htRk)*1000:null, cmdContact, measContact, osc});
};
// 固定探针偏移：0808 起 12ms（此前 10ms）。本场用指令速度平台首帧定位的臂内触球锚落在
// HT−1.5~−18ms、中位 −11ms，故 −12ms 基本踩在真实触球上；与主列之差 = 角速度/加速度×12ms。
const FACE_YAW_PRE_S=0.012;
const faceAnglesWorldPreAt = accHtRk => {
  if(accHtRk==null) return null;
  const tEval=accHtRk-FACE_YAW_PRE_S;
  const fit=fitFaceAnglesTo(accHtRk,tEval);
  if(!fit) return null;
  const botYaw=botYawDegAt(tEval);
  if(botYaw==null) return null;
  return {deg:wrapDeg(fit.fy-botYaw), fy:fit.fy, botYaw, n:fit.n, rate:fit.rate,
          pitch:fit.fp, pitchRate:fit.pitchRate};
};
// RK≈300ms 主表与机械臂最后 accepted 分表：两个合同独立取值、独立对齐 PC 真值。
const reportThrows = rkThrows.filter(t=>(t.msgs||0)>=3).sort((a,b)=>a.ht-b.ht);
// 球面−车 (dx,dy,dz)(t)：RK 全量无污染观测重建的空间真值，可在任意时刻取值。
//   球 = 该抛 S1 期 RK 球世界观测（z≥0.15，[ht−0.45,ht−0.025] 窗）三轴二次拟合，3σ 剔污染按
//        主运动轴 y 判定，x/z 用同一批采样点（同进同出，不各剔各的）；
//   车 = 车实际 x/y 轨迹（bot 位姿，挥拍前 [ht−0.16,ht−0.028] 窗线性拟合外推——避开挥拍瞬间
//        bot_y 25~65mm 塌陷伪迹）；车中心 z ≡ 地面 0（/bot_state 无 z，实测 rel_z≡球世界z）。
//   三条曲线都只吃 RK 观测，不含任何预测量。世界轴，不随车体 yaw 旋转。
// dy = (球心 y − R球) − 车 y：车 y 面≡拍面（RK rel_y 无臂基 y 补偿，非近似），故 dy=0 即球面
// 刚够到拍面 = 真实触球；dy>0 = 该时刻球还没够到（评估时刻偏早），dy<0 = 已穿过（偏晚）。
// 等效时序 = dy/v_rel（v_rel<0 闭合），正=偏晚，与旧 HT err 列同向；球/ht 同轴，无需生成延迟修正。
// dx/dz = 球心 − 车中心：接触点在拍面上的落点（拍面法向≈y），扣球半径无意义故不扣；
// 量纲可直接对 RK 消息的 rel_x/rel_z（差 = 预测误差 + 车体系↔世界轴的 yaw 旋转）。
// [[ball-car-gap-core-begin]]
const ballCarGapForThrow = (()=>{
  const R_BALL=0.033;  // 网球半径：车 y 面≡拍面（RK rel_y 无臂基 y 补偿），故触球=球面够到车 y 面
  const rows=[];
  if(RK && RK.world){
    const T=ts(RK.world), X=ys(RK.world,'x'), Y=ys(RK.world,'y'), Z=ys(RK.world,'z');
    for(let i=0;i<T.length;i++){
      const t=Number(T[i]);
      if(isNum(t)&&isNum(X[i])&&isNum(Y[i])&&isNum(Z[i])) rows.push({t,x:X[i],y:Y[i],z:Z[i]});
    }
    rows.sort((a,b)=>a.t-b.t);
  }
  const botRows=[];
  if(RK && RK.bot){
    const BT=ts(RK.bot), BX=ys(RK.bot,'x'), BY=ys(RK.bot,'y');
    for(let i=0;i<BT.length;i++){
      const t=Number(BT[i]);
      if(isNum(t)&&isNum(BX[i])&&isNum(BY[i])) botRows.push({t,x:BX[i],y:BY[i]});
    }
    botRows.sort((a,b)=>a.t-b.t);
  }
  const polyfit=(pts,tc,order,key)=>{
    const n=pts.length;
    if(n<order+2) return null;
    const us=pts.map(p=>p.t-tc), vs=pts.map(p=>p[key]);
    let co;
    if(order===1){
      let su=0,suu=0,sy=0,suy=0;
      for(let i=0;i<n;i++){su+=us[i];suu+=us[i]*us[i];sy+=vs[i];suy+=us[i]*vs[i];}
      const den=n*suu-su*su;
      if(Math.abs(den)<1e-12) return null;
      const b=(n*suy-su*sy)/den;
      co=[(sy-b*su)/n,b,0];
    } else {
      const s=[0,0,0,0,0];
      let sy=0,suy=0,su2y=0;
      for(let i=0;i<n;i++){
        let p=1;
        for(let k=0;k<5;k++){s[k]+=p;p*=us[i];}
        sy+=vs[i];suy+=us[i]*vs[i];su2y+=us[i]*us[i]*vs[i];
      }
      const det3=m=>m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);
      const D0=det3([s[0],s[1],s[2],s[1],s[2],s[3],s[2],s[3],s[4]]);
      if(Math.abs(D0)<1e-12) return null;
      co=[det3([sy,s[1],s[2],suy,s[2],s[3],su2y,s[3],s[4]])/D0,
          det3([s[0],sy,s[2],s[1],suy,s[3],s[2],su2y,s[4]])/D0,
          det3([s[0],s[1],sy,s[1],s[2],suy,s[2],s[3],su2y])/D0];
    }
    let ss=0;
    for(let i=0;i<n;i++){
      const r=co[0]+co[1]*us[i]+co[2]*us[i]*us[i]-vs[i];
      ss+=r*r;
    }
    return {co,rms:Math.sqrt(ss/Math.max(n-(order+1),1)),n};
  };
  const carLine=ht=>{
    for(const lo of [0.16,0.30]){
      const pts=botRows.filter(p=>p.t>=ht-lo&&p.t<=ht-0.028);
      const n=pts.length;
      if(n<3) continue;
      const tm=pts.reduce((s,p)=>s+p.t,0)/n;
      let suu=0,suy=0,sy=0,sux=0,sx=0;
      for(const p of pts){
        const u=p.t-tm;
        suu+=u*u;suy+=u*p.y;sy+=p.y;sux+=u*p.x;sx+=p.x;
      }
      if(suu<1e-12) continue;
      const b=suy/suu, bx=sux/suu;
      // 车 y = a0 + b·u、车 x = ax0 + bx·u（u 相对 ht）
      return {a0:sy/n+b*(ht-tm), b, ax0:sx/n+bx*(ht-tm), bx, n};
    }
    return null;
  };
  const evalPoly=(fit,u)=>fit.co[0]+fit.co[1]*u+fit.co[2]*u*u;
  return (th,tEval)=>{
    if(!rows.length||!th||!isNum(th.ht)||!isNum(tEval)) return null;
    const ht=th.ht;
    const win=rows.filter(p=>p.z>=0.15&&p.t>=ht-0.45&&p.t<=ht-0.025);
    if(win.length<4) return null;
    let f=polyfit(win,ht,win.length>=6?2:1,'y');
    if(!f) return null;
    const thr=Math.max(3*f.rms,0.02);
    const kept=win.filter(p=>Math.abs(evalPoly(f,p.t-ht)-p.y)<=thr);
    // 剔污染按主运动轴 y 判定，x/z 复用同一批采样点（三轴同进同出）
    let used=win;
    if(kept.length<win.length&&kept.length>=4){
      const f2=polyfit(kept,ht,kept.length>=6?2:1,'y');
      if(f2){ f=f2; used=kept; }
    }
    const order=used.length>=6?2:1;
    const fx=polyfit(used,ht,order,'x'), fz=polyfit(used,ht,order,'z');
    if(!fx||!fz) return null;
    const car=carLine(ht);
    if(!car) return null;
    if(!((f.co[1]-car.b)<-1.0)) return null;
    // 评估点 u 相对该抛 ht；离拟合窗过远就不外推（拟合窗本身只有 ~450ms）。
    const u=tEval-ht;
    if(!(Math.abs(u)<=0.30)) return null;
    const ballX=evalPoly(fx,u), ballY=evalPoly(f,u), ballZ=evalPoly(fz,u);
    const carX=car.ax0+car.bx*u, carY=car.a0+car.b*u;
    const vy=f.co[1]+2*f.co[2]*u, vRel=vy-car.b;
    // dy>0=球面还没够到车 y 面（评估时刻偏早），<0=已穿过（偏晚）；
    // 等效时序 dtMs=dy/v_rel，正=偏晚（沿用旧 HT err 的符号）。
    // dx/dz 是拍面上的落点，不扣球半径；车中心 z ≡ 地面 0。
    return {dx:ballX-carX, dy:ballY-R_BALL-carY, dz:ballZ,
            ballX, ballY, ballZ, carX, carY, vy, vRel, u,
            dtMs:(vRel<-1e-6)?(ballY-R_BALL-carY)/vRel*1000:null,
            n:f.n, nWin:win.length, rms:f.rms, rmsX:fx.rms, rmsZ:fz.rms,
            carVx:car.bx, carVy:car.b, nCar:car.n,
            eA:isNum(th.lastS0Y)?car.a0-th.lastS0Y:null};
  };
})();
// [[ball-car-gap-core-end]]
const tableFmt = (v,d) => isNum(v) ? Number(v).toFixed(d) : '—';
// 0809 起机械臂两张表的距离量一律以 cm 呈现（单元格与悬停同口径）：内部一切照旧用米/秒，
// 只在显示层换算，入参永远是米，默认 1 位小数 = 毫米分辨率。
const cmFmt = (v,d=1) => isNum(v) ? (Number(v)*100).toFixed(d) : '—';
const cmSigned = (v,d=1) => isNum(v) ? (v>=0?'+':'')+(Number(v)*100).toFixed(d) : '—';
const tableXzCm = (x,z,d=1) => (isNum(x)&&isNum(z)) ? cmFmt(x,d)+'/'+cmFmt(z,d) : '—';
const tableSigned = v => isNum(v) ? (v>=0?'+':'')+Number(v).toFixed(1) : '—';
const tableXyzCm = (x,y,z,d=1) => (isNum(x)&&isNum(y)&&isNum(z))
  ? cmFmt(x,d)+'/'+cmFmt(y,d)+'/'+cmFmt(z,d) : '—';
const tableEsc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
// [[bot-run-end-core-begin]]
const botRunEndForThrow = th => {
  if(!RK || !th || !isNum(th.ht)) return null;
  const t=ts(RK.bot), phase=ys(RK.bot,'phase');
  const x=ys(RK.bot,'x'), y=ys(RK.bot,'y');
  const tx=ys(RK.bot,'target_x'), ty=ys(RK.bot,'target_y');
  let best=null;
  for(let i=0;i+1<t.length;i++){
    if(phase[i]!=='RUN' || phase[i+1]!=='BRAKE_IN_SWING' || !isNum(t[i])) continue;
    const dt=Math.abs(t[i]-th.ht);
    if(dt>0.1 || (best && dt>=best.dt)) continue;
    const owner=reportThrows.reduce((closest,candidate)=>
      !closest || Math.abs(candidate.ht-t[i])<Math.abs(closest.ht-t[i]) ? candidate : closest, null);
    if(owner!==th || ![x[i],y[i],tx[i],ty[i]].every(isNum)) continue;
    best={t:t[i],x:x[i],y:y[i],tx:tx[i],ty:ty[i],dt};
  }
  return best;
};
// [[bot-run-end-core-end]]
const pcTruthCell = (f,withY=false,tPc=null) => {
  if(!f) return pcTruthMissCell(tPc);
  // 两侧时距都显示：球是**外推**（拟合窗末点到目标时刻），车是**插值**（到前后最近一条
  // /pc_car_loc 的较大一侧）。0731 起 x/y 是「球世界−车世界」，只报球侧会把车侧的
  // 陈旧/宽插值藏起来——0809_122035 #13 就是这么用一个 ±0.8cm 的棒子盖住了 10cm 的车侧误差。
  const carMs=Math.round((f.carGap||0)*1000);
  const carTxt=carMs>0?'·车±'+carMs+'ms':'';
  return (withY?tableXyzCm(f.x,f.y,f.z):tableXzCm(f.x,f.z))+
    ' <span style="color:'+(carMs>150?'#f97316':'#fbbf24')+'" title="入弧拟合真值：x/y 线性、z 重力+阻力(λ=k_drag·水平速)+带界旋转曲率(|δ|≤2m/s²)；本行 δz='+((f.delta||0)>=0?'+':'')+(f.delta||0).toFixed(2)+'m/s²；表中 x/y=拟合球世界坐标−车世界坐标(世界轴不转yaw)；只用目标时刻20ms前观测，不跨目标时刻插值；max|残差| '+
    cmFmt(f.resMax)+'cm，ball_y−car_y='+cmSigned(f.y)+'cm'+
    '；球外推 '+Math.round(f.dNear*1000)+'ms（拟合窗末点→目标时刻）'+
    '；车 /pc_car_loc 前 '+Math.round((f.carGa||0)*1000)+'ms / 后 '+Math.round((f.carGb||0)*1000)+
    'ms 夹住插值（不外推、不冻结），按底盘 a_dec_max=3m/s² 折算插值误差 ±'+cmFmt(f.eCar||0)+
    'cm，已并进左边的总误差棒'+
    (f.carSingleTag?'；⚠ 夹住的 /pc_car_loc 里有单 tag 退化解（只剩一块 tag 可见，位置由冻结 yaw 经 0.42m 安装杠杆反解，已按 ±'+
      cmFmt(PC_TRUTH_SINGLE_TAG_ERR)+'cm 并进车侧误差）':'')+'">±'+
    cmFmt(Math.max(0.001,f.err))+'cm(球外推'+Math.round(f.dNear*1000)+'ms'+carTxt+')</span>';
};

// [[accepted-match-core-begin]]
const matchThrowByAcceptedCt = ct => {
  if(!isNum(ct)) return null;
  const candidates=reportThrows.filter(th=>ct>=th.firstT-0.02 && ct<=th.lastT+0.02);
  if(!candidates.length) return null;
  return candidates.reduce((best,th)=>{
    const center=isNum(th.ref300T)?th.ref300T:(th.firstT+th.lastT)/2;
    const bestCenter=isNum(best.ref300T)?best.ref300T:(best.firstT+best.lastT)/2;
    return Math.abs(ct-center)<Math.abs(ct-bestCenter)?th:best;
  },candidates[0]);
};
// [[accepted-match-core-end]]
const lastAcceptedForThrow = th => armAligned ? armHitMarks
  .filter(h=>h.label==='hit' && isNum(h.wct) && matchThrowByAcceptedCt(h.wct-RK.t0)===th)
  .reduce((last,h)=>!last||h.lastAcceptT>last.lastAcceptT?h:last,null) : null;
const rejectKind = reason => {
  let m;
  if((m=/^x [\-0-9.]+m is below min ([\-0-9.]+)m/.exec(reason))) return `x低于下限 ${m[1]}m`;
  if((m=/^effective duration [\-0-9.]+s < ([0-9.]+)s/.exec(reason))) return `effective duration < ${m[1]}s`;
  if((m=/^duration [\-0-9.]+s is <= hit_time ([0-9.]+)s/.exec(reason))) return `duration ≤ hit_time ${m[1]}s`;
  if(/^height adjust needs /.test(reason)) return '高度调整时间不足';
  if(/^hit phase in progress/.test(reason)) return '上一拍仍在执行';
  return reason;
};
const rejectNoteForThrow = th => {
  const counts=new Map();
  (ARM?ARM.events:[]).forEach(e=>{
    if(e.topic!=='/tennis/status' || e.t<th.firstT-0.02 || e.t>th.lastT+0.25) return;
    const m=/^reject hit: (.+)$/.exec(e.text);
    if(!m) return;
    const reason=rejectKind(m[1]);
    counts.set(reason,(counts.get(reason)||0)+1);
  });
  return counts.size
    ? [...counts].map(([reason,n])=>tableEsc(reason)+(n>1?' ×'+n:'')).join('<br>')
    : '未收到 accepted，且无可回配 reject 状态';
};

const rk300TableHtml = () => {
  if(!RK || !reportThrows.length) return '';
  const rows=reportThrows.map((th,idx)=>{
    const accepted=lastAcceptedForThrow(th);
    const runEnd=botRunEndForThrow(th);
    const runTargetError=runEnd
      ? tableSigned((runEnd.tx-runEnd.x)*100)+'/'+tableSigned((runEnd.ty-runEnd.y)*100)
      : '—';
    const rkPredCarError=runEnd&&isNum(th.ref300CarX)&&isNum(th.ref300CarY)
      ? tableSigned((th.ref300CarX-runEnd.x)*100)+'/'+tableSigned((th.ref300CarY-runEnd.y)*100)
      : '—';
    const accHt=accepted&&isNum(accepted.wht)?accepted.wht-RK.t0:null;
    // 臂最后更新 HT：臂真正受理的最后一条 /predict_hit_pos 的原始 ht——挥拍窗内 ht 重定相
    // 生效时就是那条 late ht saved 的 ht，否则退回最后一条 accepted 的 ht。全表空间量列
    // （PC真值/TCP/球面y−车y/击球点@300/车yaw/两列拍面 yaw,pitch 与实测拍速）统一锚在它上面
    // （不减臂内 HIT_TIME_ADVANCE_SEC，本场自标定值见表下注记；拍面右列固定再减 10ms 取值）。
    const finalHt=accepted&&isNum(accepted.finalHt)?accepted.finalHt-RK.t0:accHt;
    const rsw=accepted?accepted.reswing:null;
    const htSrcNote=rsw
      ? (rsw.ok
          ? '；HT源=挥拍中重定相后的最后一条late ht saved（老触球'+tableSigned(rsw.delta)+'ms）'
          : '；重定相因剩余'+(rsw.remain*1000).toFixed(0)+'ms<60ms放弃，HT源退回最后一条accepted')
      : '；HT源=最后一条accepted（本拍无挥拍窗内更新）';
    const acceptedTarget=accepted?tableXzCm(accepted.wx,accepted.wz):'—';
    const carYawAcc=finalHt!=null?botYawDegAt(finalHt):null;
    const carYawRate=finalHt!=null?imuYawRateDegAt(finalHt):null;
    // TCP−车心@臂最后更新HT（世界轴）：FK TCP 是臂系值，臂基≡车心（0标恒等式），臂系 x/y 按该
    // 时刻车 yaw（/bot_state 瞬时值，IMU 连续更新无重定位台阶）旋到世界轴、z−armZOff 还原世界
    // 高度，即拍心相对车中心的世界轴偏移——与左列 PC真值(球−车,世界轴) 同轴同基准，两列相减≈拍−球。
    // azimuth 口径 atan2(x,y)：世界θ=臂系θ−车yaw。
    const tcp=finalHt!=null?tcpAt(finalHt):null;
    const tcpYawRad=(carYawAcc!=null?carYawAcc:0)*Math.PI/180;
    const tcpWorld=tcp?[tcp[0]*Math.cos(tcpYawRad)-tcp[1]*Math.sin(tcpYawRad),
                        tcp[0]*Math.sin(tcpYawRad)+tcp[1]*Math.cos(tcpYawRad),
                        tcp[2]-(isNum(armZOff)?armZOff:0)]:null;
    const tcpCell=tcpWorld
      ? '<span title="'+tableEsc('臂系FK TCP=('+cmFmt(tcp[0])+', '+cmFmt(tcp[1])+', '
          +cmFmt(tcp[2])+')cm @臂最后更新HT '+rkToPc(finalHt).toFixed(3)+'s（原始ht，未减臂内提前量）'
          +'；按车yaw ψ='+(carYawAcc!=null?tableSigned(carYawAcc)+'°':'—（该时刻无/bot_state，按0°）')
          +' 旋到世界轴（xw=x·cosψ−y·sinψ、yw=x·sinψ+y·cosψ）、z−臂系z偏移'
          +(isNum(armZOff)?cmFmt(armZOff):'0')+'cm 还原世界高度'
          +'；臂基≡车心（0标恒等式），故本列=拍心TCP−车中心的世界轴偏移，与左列PC真值(球−车)'
          +'同轴同基准，两列相减≈拍−球'+htSrcNote)+'">'
          +tableXyzCm(tcpWorld[0],tcpWorld[1],tcpWorld[2])+'</span>'
      : '—';
    // TCP−accepted目标（世界轴）：目标侧 dx 取该 accepted 消息自带的 世界x−car_pred_x（与 lead 表
    // accepted−PC、击球点@300 列同一约定）、dz 取 rel_z(≡世界z)。
    // 两侧同为世界轴、各自相对自家车心；dz 与老口径臂系伺服差恒等（z 不随 yaw 转）。
    const tgtWx=accepted&&isNum(accepted.wxw)&&isNum(accepted.wcarx)?accepted.wxw-accepted.wcarx:null;
    const tcpAcceptedDx=(tcpWorld&&tgtWx!=null)?(tcpWorld[0]-tgtWx)*100:null;
    const tcpAcceptedDz=accepted&&tcpWorld&&isNum(accepted.wz)?(tcpWorld[2]-accepted.wz)*100:null;
    const tcpServoDx=(tcp&&accepted&&isNum(accepted.wx))?(tcp[0]-accepted.wx)*100:null;
    const tcpAcceptedError=(tcpAcceptedDx!=null||tcpAcceptedDz!=null)
      ? '<span title="'+tableEsc('世界轴，各自相对自家车心：dx=TCP世界x−(accepted消息世界x−car_pred_x'
          +(tgtWx!=null?('='+cmFmt(tgtWx)+'cm）'):'）——老bag缺世界x/car_pred_x字段，dx无值')
          +'、dz=TCP世界z−rel_z(≡世界z)；目标不随挥拍中重定相变（后续消息只存触球时刻不换目标），'
          +'TCP取臂最后更新HT，即臂真正击球时刻拍心是否到位'
          +(tcpServoDx!=null?('；老口径臂系伺服差 dx=tcp_x−rel_x='+tableSigned(tcpServoDx)
            +'cm（dz两口径恒等），世界差−伺服差=车yaw旋转泄漏'):'')
          +htSrcNote)+'">'
          +tableSigned(tcpAcceptedDx)+'/'+tableSigned(tcpAcceptedDz)+'</span>'
      : tableSigned(null)+'/'+tableSigned(null);
    // PC真值@臂最后更新HT：与左侧 @HT300 列同一套入弧拟合真值（x/y=拟合球世界−同时刻车世界、
    // z=世界高度），评估时刻=臂最后更新HT（原始值不减臂内提前量，与 TCP 列同锚）。两列之差 =
    // 真值在 HT300→臂最后更新HT 这段里真实移动了多少（球速≈10m/s 时 1ms≈1cm）。
    const truthAcc=finalHt!=null?pcTruthAt(rkToPc(finalHt)):null;
    // 目标挥拍速度/pitch：最后一条 accepted 状态自带的**计划值**（挥拍中 ht 重定相只改时间轴、
    // 不换目标，故用最后一条 accepted 即可）。speed 是 arm_controller 过完各级钳位后的计划触球
    // 拍速，口径 = 零起速梯形恒等式 2·|行程|/hit_time·x，只算 J1。
    // ⚠ 它是**受理时**的账：真正执行的挥拍段在挥拍首帧建、并在 ht 重定相触发点用当刻
    // (q_des,v_des) 重建，HitTrajectory 的触球速度是 2·Δ剩余/T剩余 − v0，起速非零就抬高
    //（本场 19/19 抛都重定相，指令侧比 speed= 高 14~48%）。要对实测拍速，看右列悬停的
    // 「同刻指令侧 |v_cmd1|·r」——那才是这条轨迹真正要求的触球拍速。
    const tgtSpeed=accepted&&isNum(accepted.tgtSpeed)?accepted.tgtSpeed:null;
    const tgtPitch=accepted&&isNum(accepted.tgtPitch)?accepted.tgtPitch:null;
    const tgtSpeedCell=(tgtSpeed!=null||tgtPitch!=null)
      ? '<span title="'+tableEsc('accepted 状态自带的挥拍计划量（最后一条 accepted；重定相不换目标）'
          +(tgtSpeed!=null?('；speed='+tgtSpeed.toFixed(2)+'m/s = 过完各级钳位后的计划触球拍速'
            +'（输入夹[1,12]→ω夹12rad/s→引拍窗装不下再夹短），口径 2·|行程|/hit_time('
            +(accepted&&isNum(accepted.hitT)?accepted.hitT.toFixed(3):'0.250')+'s)·x，只算 J1'):'')
          +(accepted&&isNum(accepted.tgtSpeedReq)
            ? ('；speed_req='+accepted.tgtSpeedReq.toFixed(2)+'m/s = 上游原始指令（本拍被钳位改动>5mm/s）')
            : '；无 speed_req ⇒ 计划值即上游指令')
          +(accepted&&isNum(accepted.shortened)
            ? ('；shortened='+accepted.shortened.toFixed(4)+'rad = 引拍起点被夹掉的角度：'
               +'窗口装不下就朝锚点（驻拍/上一拍收尾位）能引多少引多少，'
               +'锚点比目标引拍点更靠后时实际行程反而变长 ⇒ speed 高于 speed_req')
            : '')
          +(tgtPitch!=null?('；pitch='+tgtPitch.toFixed(2)+'°=目标拍面仰角（0805 起随来球俯冲角变，'
            +'臂系≡世界系不减车yaw，可直接减右列实测 pitch）'):'')
          +(accepted&&isNum(accepted.tgtFaceYaw)
            ? ('；face_yaw='+accepted.tgtFaceYaw.toFixed(4)+'rad='
               +(accepted.tgtFaceYaw*180/Math.PI).toFixed(2)+'°（臂端锁面目标）'):'')
          +htSrcNote)+'">'
          +(tgtSpeed!=null?tgtSpeed.toFixed(2):'—')+'/'
          +(tgtPitch!=null?tgtPitch.toFixed(1):'—')+'</span>'
      : '<span title="本抛无 accepted，或该场 arm_controller 早于可变 pitch(0805)/拍速逐拍指定(0808)，状态行不带这两个字段">—</span>';
    const faceYaw=faceAnglesWorldAt(finalHt);
    const swingSpeed=racketSpeedAt(finalHt);
    const speedNote=swingSpeed
      ? '。拍速 |v_tcp|='+swingSpeed.v.toFixed(2)+'m/s：完整 FK 解析 Jacobian 的拍心线速度，'
          +'在 HT 处**直接插值**/joint_states（不外推——S 曲线下窗内拍速强非线性，'
          +'线性外推会高估 40%+；速度大小与车 yaw 无关，臂系≡世界系）'
          +'；其中 J1 项 |q̇1|·r='+swingSpeed.vj1.toFixed(2)+'m/s（r='+cmFmt(swingSpeed.r)+'cm，'
          +'= status speed= 的口径）'
          +(swingSpeed.cmd!=null?('；同刻指令侧 /tennis/motor_command 的 |v_cmd1|·r='
            +swingSpeed.cmd.toFixed(2)+'m/s ⇒ 伺服差 '
            +tableSigned(swingSpeed.v-swingSpeed.cmd)+'m/s'):'；本刻无指令帧')
          +(swingSpeed.contactDt!=null?('；臂内触球锚@HT'+tableSigned(swingSpeed.contactDt)
            +'ms（= 指令 J1 速度进入恒 ω 巡航平台的第一帧 = HitTrajectory 的 finish_hit_time，'
            +'不靠实测峰值找）：指令 '+swingSpeed.cmdContact.toFixed(2)+'m/s'
            +(swingSpeed.measContact!=null?('、实测 '+swingSpeed.measContact.toFixed(2)
              +'m/s ⇒ 触球欠速 '+tableSigned(swingSpeed.measContact-swingSpeed.cmdContact)+'m/s'):'')):'')
          +(swingSpeed.osc!=null?('；单点抖动量级 σ='+swingSpeed.osc.toFixed(2)+'m/s'
            +'（引拍段[−250,−120]ms、肯定无球时的 实测−指令 残差 std：J1 实测速度全程叠着这个'
            +'量级的伺服振荡，所以别拿单帧极值当触球探测器）'):'')
          +(tgtSpeed!=null?('；目标(status speed=)='+tgtSpeed.toFixed(2)+'m/s，实测−目标 '
            +tableSigned(swingSpeed.v-tgtSpeed)+'m/s'):'')
      : '。本刻 50ms 内无 /joint_states 覆盖，拍速无值';
    const speedTxt=' <span style="color:#a0a0c0">'
      +(swingSpeed?swingSpeed.v.toFixed(2)+'m/s':'—m/s')+'</span>';
    const faceYawCell=(faceYaw||swingSpeed)
      ? '<span title="'+tableEsc((faceYaw
          ? ('臂系FK face_yaw='+tableSigned(faceYaw.fy)+'°（冲击前窗['
            +'−80,−6]ms '+faceYaw.n+'帧线性外推@臂最后更新HT，ψ̇='+tableSigned(faceYaw.rate)+'°/s）'
            +'；车yaw='+tableSigned(faceYaw.carYaw)+'°（挥拍前0.45~0.15s窗圆均值'+faceYaw.m+'帧）'
            +'；世界ψ=face_yaw−车yaw，口径同PC回球yaw=atan2(x,y)；纯FK不含δ6球侧偏置'
            +'。pitch='+tableSigned(faceYaw.pitch)+'°（同窗同帧拟合的 asin(n_z)，正=开面上仰，'
            +'θ̇='+tableSigned(faceYaw.pitchRate)+'°/s）——**不减车yaw**：J1/BASE_ROT 纯 z 转不动 n_z，'
            +'臂系pitch≡世界pitch；θ̇≈0 故 pitch 对时序免疫，与右列 pitch 之差 = θ̇×12ms')
          : '该 HT 无拍面角（冲击前窗内 <2 帧 FK state 或缺挥拍前车 yaw）')
          +speedNote+htSrcNote)+'">'
          +(faceYaw?tableSigned(faceYaw.deg)+'/'+tableSigned(faceYaw.pitch):'—/—')
          +speedTxt+'</span>'
      : '—';
    const faceYawPre=faceAnglesWorldPreAt(finalHt);
    // HT−12ms 探针上的实测拍速：只取该刻的插值值与同刻指令值（触球锚/振荡 σ 是整拍量，
    // 已在左列悬停里给过，不重复算）。与左列拍速之差 = 这 12ms 里 J1 加速掉的量。
    const swingSpeedPre=finalHt!=null?racketSpeedRawAt(finalHt-FACE_YAW_PRE_S):null;
    const speedPreNote=swingSpeedPre
      ? '。拍速 |v_tcp|='+swingSpeedPre.v.toFixed(2)+'m/s（同左列口径，在 HT−12ms 处插值）'
          +(swingSpeedPre.cmd!=null?('；同刻指令侧 |v_cmd1|·r='+swingSpeedPre.cmd.toFixed(2)
            +'m/s ⇒ 伺服差 '+tableSigned(swingSpeedPre.v-swingSpeedPre.cmd)+'m/s'):'')
          +(swingSpeed?('；与左列(@HT)之差 '+tableSigned(swingSpeedPre.v-swingSpeed.v)
            +'m/s（负=这 12ms 里仍在加速，正=实测反而掉了；单点抖动 σ 见左列悬停，本场'
            +' σ 与这个差同量级，逐拍符号别当结论）'):'')
      : '。HT−12ms 处 50ms 内无 /joint_states 覆盖，拍速无值';
    const speedPreTxt=' <span style="color:#a0a0c0">'
      +(swingSpeedPre?swingSpeedPre.v.toFixed(2)+'m/s':'—m/s')+'</span>';
    const faceYawPreCell=(faceYawPre||swingSpeedPre)
      ? '<span title="'+tableEsc((faceYawPre
          ? ('臂系FK face_yaw='+tableSigned(faceYawPre.fy)+'°（冲击前窗[−80,−6]ms '
            +faceYawPre.n+'帧线性拟合@臂最后更新HT−12ms取值，ψ̇='+tableSigned(faceYawPre.rate)+'°/s）'
            +'；车yaw='+tableSigned(faceYawPre.botYaw)+'°（/bot_state yaw@HT−12ms：AprilTag accept后由IMU连续更新，HT结束后才重定位）'
            +'；世界ψ=face_yaw−车yaw；HT−12ms 固定探针：臂内提前量本场='
            +(armConstCal.adv!=null?(armConstCal.adv*1000).toFixed(0):'?')+'ms，'
            +'而用指令速度平台首帧定位的臂内触球锚本场落在 HT−1.5~−18ms（中位 −11ms），'
            +'故本列基本踩在真实触球上，并兼作时序敏感度探针（与左列之差 = ψ̇×12ms）'
            +'。pitch='+tableSigned(faceYawPre.pitch)+'°（同窗拟合@HT−12ms，θ̇='
            +tableSigned(faceYawPre.pitchRate)+'°/s，不减车yaw：臂系pitch≡世界pitch）')
          : '该 HT−12ms 无拍面角（冲击前窗内 <2 帧 FK state 或该刻无 /bot_state）')
          +speedPreNote+htSrcNote)+'">'
          +(faceYawPre?tableSigned(faceYawPre.deg)+'/'+tableSigned(faceYawPre.pitch):'—/—')
          +speedPreTxt+'</span>'
      : '—';
    // 车yaw@臂最后更新HT：与右侧两列拍面yaw,pitch、左侧两列击球真值同锚（臂真正执行的击球时刻）。
    // 取 /bot_state yaw 瞬时值——车控 accept AprilTag 定位后 yaw 由 IMU 连续更新、
    // HT 结束后才用 AprilTag 重定位，故 HT 前采样点无重定位台阶；挥拍位姿伪迹只塌陷位置不动 yaw。
    // 悬停给 IMU yaw_speed（零滞后）换算的 10ms 时序灵敏度，以及 PC AprilTag 挥拍前窗圆均值对照
    // ——那正是右侧「拍面yaw,pitch@臂最后更新HT」列的 yaw 里减掉的车 yaw，两者之差即两列口径差的车侧来源。
    const carYawCell=carYawAcc!=null
      ? '<span title="'+tableEsc('/bot_state yaw@臂最后更新HT '+rkToPc(finalHt).toFixed(3)+'s（PC轴，'
          +'臂受理的最后一条预测的原始 ht，未减臂内提前量）：AprilTag accept 后由 IMU 连续更新，'
          +'HT 结束后才重定位，采样点无台阶'
          +(carYawRate!=null?('；IMU yaw_speed='+tableSigned(carYawRate)+'°/s（零滞后陀螺，'
            +'10ms 时序误差≈'+(carYawRate>=0?'+':'')+(carYawRate*0.01).toFixed(2)+'°车yaw）'):'；本场无 IMU yaw_speed')
          +(faceYaw?('；对照 PC AprilTag 挥拍前[−450,−150]ms 圆均值='+tableSigned(faceYaw.carYaw)
            +'°（右侧拍面yaw,pitch@臂最后更新HT 列的 yaw 减掉的车yaw），差 '
            +tableSigned(wrapDeg(carYawAcc-faceYaw.carYaw))+'°'):''))+'">'
          +tableSigned(carYawAcc)+'</span>'
      : '<span title="无 accepted/臂最后更新HT，或该时刻 50ms 内无 /bot_state">—</span>';
    // 最后更新→挥拍起：臂受理的最后一条预测的到达时刻 − 挥拍段起点(老触球−HIT_T)。
    // 正=更新落在挥拍开始之后（0803 起挥拍窗内不再拒收，这些就是 ht 重定相的养料）。
    const updT=accepted&&isNum(accepted.lastUpdateT)?accepted.lastUpdateT:null;
    const swingStart=accepted&&isNum(accepted.start)?accepted.start:null;
    const updGapCell=(updT!=null&&swingStart!=null)
      ? '<span title="'+tableEsc('最后受理更新 t='+rkToPc(updT).toFixed(3)+'s（PC轴）'
          +'；挥拍起 t='+rkToPc(swingStart).toFixed(3)+'s（=老触球−HIT_T 0.25s）'
          +'；最后一条 accepted t='+rkToPc(accepted.lastAcceptT).toFixed(3)+'s（其后 '
          +(rsw?rsw.n:0)+' 条只存触球时刻不换目标）'
          +(rsw
            ? '；重定相触发 t='+rkToPc(rsw.trig).toFixed(3)+'s（老触球−100ms）：老触球 '
              +rkToPc(rsw.oldDone).toFixed(3)+'s → 新触球 '+rkToPc(rsw.newDone).toFixed(3)+'s（'
              +tableSigned(rsw.delta)+'ms，剩余 '+(rsw.remain*1000).toFixed(0)+'ms）'
              +(rsw.ok?' ✓生效':' ✗剩余<60ms放弃')
            : '；本拍挥拍窗内无新预测，未触发重定相'))+'">'
          +tableSigned((updT-swingStart)*1000)+'</span>'
      : '—';
    // 盲区 ht−ct@臂最后更新：最终那条命令的「击球点时刻 − 它最晚看到的那颗球的观测时刻」。
    // 这段时间里预测纯外推、没有任何新观测进来，是本拍真正的信息盲区。
    const finalCt=accepted&&isNum(accepted.finalCt)?accepted.finalCt-RK.t0:null;
    const blindBad=!!(accepted&&accepted.finalMismatch);
    const blind=(finalHt!=null&&finalCt!=null&&!blindBad)?(finalHt-finalCt)*1000:null;
    const blindCell=blind!=null
      ? '<span title="'+tableEsc('臂最后更新消息 ct='+rkToPc(finalCt).toFixed(3)+'s（最晚一颗球的观测时刻）'
          +'、ht='+rkToPc(finalHt).toFixed(3)+'s（预测击球时刻，原始值未减臂内提前量'
          +(armConstCal.adv!=null?(armConstCal.adv*1000).toFixed(0)+'ms':'')+'）'
          +'；这段是纯外推、无新观测的盲区'+htSrcNote)+'">'+blind.toFixed(1)+'</span>'
      : (blindBad
        ? '<span style="color:#e0a24a" title="'+tableEsc('重定相已生效（臂用的是挥拍窗内最后一条 '
            +'late ht saved 的 ht），但该条 status 回配原 /predict_hit_pos 失败，拿不到同源的 ht/ct，'
            +'故本格不出数——退回最后一条 accepted 会把真实盲区（~170ms 量级）冒充成 ~330ms。'
            +'失配判据见 [[arm-swing-ht-core]] 的 gap 自校验窗')+'">⚠—</span>'
        : '—');
    // Δht 重定相：挥拍时用的 ht（最后一条 accepted）→ 更新后的 ht。二者都是原始 ht，
    // 同减臂内 10ms，故与臂内触球时刻之差完全等价。未采纳时显示候选值并置灰。
    const dHtCell=rsw
      ? '<span'+(rsw.ok?'':' style="color:#a0a0c0"')
          +' title="'+tableEsc('挥拍时 ht='+rkToPc(rsw.oldDone+HIT_TIME_ADVANCE_SEC).toFixed(3)
          +'s → 更新后 ht='+rkToPc(rsw.newDone+HIT_TIME_ADVANCE_SEC).toFixed(3)+'s'
          +'；正=新预测把击球点推晚。触球拍速是派生量，会随剩余时长一起变'
          +(rsw.ok?'':'；剩余'+(rsw.remain*1000).toFixed(0)+'ms<60ms，控制器放弃重定相，实际仍走老时间轴'))+'">'
          +tableSigned(rsw.delta)+(rsw.ok?'':'(未采纳)')+'</span>'
      : '<span title="本拍挥拍窗内没有新预测到达，ht 自挥拍起未变">—</span>';
    const rejectNote=accepted?'—':rejectNoteForThrow(th);
    const hitAnchorPc=accHt!=null?rkToPc(accHt):(isNum(th.ht)?rkToPc(th.ht):null);
    const ret=pcReturnAt(hitAnchorPc);
    const returnCell=ret
      ? '<span title="'+tableEsc('回球触球t='+ret.tHit.toFixed(3)+'s（来回交点法）；v=('+ret.vx.toFixed(2)+', '+ret.vy.toFixed(2)+', '+ret.vz.toFixed(2)+')m/s；出弧'+ret.n+'点/'+Math.round(ret.span*1000)+'ms，段首距触球+'+Math.round(ret.start*1000)+'ms'+(ret.bounceCut?'；地面反弹前截断':''))+'">'+
        tableSigned(ret.yaw)+'/'+tableSigned(ret.pitch)+' <span style="color:#a0a0c0">'+ret.speed.toFixed(1)+'m/s</span></span>'
      : '—';
    // RK 全量无污染观测（球世界三轴拟合 × 车实际 x/y 外推）在臂最后更新HT 上量出的真值，
    // 供右边两列共用：dy = 球面到车 y 面（≡拍面）的缺口 = 时序误差的空间形态；
    // dx/dz = 球心相对车中心的落点（拍面上的位置，不扣球半径）= 击球点真值。
    const gapFin=finalHt!=null?ballCarGapForThrow(th,finalHt):null;
    const truthNote=gapFin
      ? '真值取 RK 全量无污染观测@臂最后更新HT '+rkToPc(finalHt).toFixed(3)+'s（世界轴，不转车yaw）：'
        +'球世界 '+gapFin.n+'点三轴共用（窗[ht−450,−25]ms，z≥15cm'
        +(gapFin.n<gapFin.nWin?('，按y轴3σ剔'+(gapFin.nWin-gapFin.n)+'点'):'')+'，rms x/y/z='
        +cmFmt(gapFin.rmsX)+'/'+cmFmt(gapFin.rms)+'/'+cmFmt(gapFin.rmsZ)
        +'cm）× 车bot '+gapFin.nCar+'点（挥拍前[−160,−28]ms线性外推，vx/vy车='
        +gapFin.carVx.toFixed(2)+'/'+gapFin.carVy.toFixed(2)+'m/s）'
        +'；球心 x/y/z='+cmFmt(gapFin.ballX)+'/'+cmFmt(gapFin.ballY)+'/'+cmFmt(gapFin.ballZ)
        +'cm，车 x/y='+cmFmt(gapFin.carX)+'/'+cmFmt(gapFin.carY)+'cm（车中心z≡地面0）'
        +'；评估点距该抛RK最终ht '+(gapFin.u*1000).toFixed(1)+'ms'
      : '';
    const gapMiss='<span title="'+tableEsc(finalHt==null?'本拍无accepted/臂最后更新HT，无评估时刻'
      :'RK球观测或挥拍前车位姿不足（或臂最后更新HT离该抛拟合窗>300ms），无法量取')+'">—</span>';
    // 球面y−车y @臂最后更新HT：0=该 ht 正好是真实触球，正=球面还没够到拍面（ht 偏早），负=已穿过（偏晚）。
    const gapCell=gapFin
      ? '<span title="'+tableEsc('dy=(球心y−R球3.3cm)−车y='+cmSigned(gapFin.dy)+'cm；车y面≡拍面'
          +'（RK rel_y 无臂基y补偿），故 0=球面刚够到=真实触球'
          +'；闭合速度|v_rel|='+Math.abs(gapFin.vRel).toFixed(2)+'m/s（|vy球|='
          +Math.abs(gapFin.vy).toFixed(2)+'m/s）'
          +(isNum(gapFin.dtMs)?('，等效时序='+tableSigned(gapFin.dtMs)+'ms（正=该ht比真实触球晚）'):'')
          +'；'+truthNote
          +(isNum(gapFin.eA)?('；车@ht−冻结面lastS0Y='+cmFmt(gapFin.eA)+'cm'):''))+'">'
          +cmSigned(gapFin.dy)+'</span>'
      : gapMiss;
    // 击球点(x,z)@300预测 − 击球点(x,z)真值@臂最后更新HT，世界轴、两侧都相对各自的车：
    // 预测侧取该抛 S1@≈300ms 那条消息自带的世界击球点 x 减它同条给出的车 x（car_pred_x），
    // z 直接用 rel_z（车中心 z≡地面0，yaw 不转 z，故 rel_z 就是世界 z）——这样两侧同轴同基准，
    // 不掺车体系↔世界轴的 yaw 旋转。
    // 正 = 预测的击球点比真实球位更靠 +x / 更高，即臂被瞄到球的右侧/上方。
    const aimDx=(gapFin&&isNum(th.ref300Xw)&&isNum(th.ref300CarX))
      ? (th.ref300Xw-th.ref300CarX)-gapFin.dx : null;
    const aimDz=(gapFin&&isNum(th.ref300Z))?th.ref300Z-gapFin.dz:null;
    const aimCell=(aimDx!=null&&aimDz!=null)
      ? '<span title="'+tableEsc('dx=预测(消息世界x−car_pred_x)'
          +cmFmt(th.ref300Xw-th.ref300CarX)+'cm − 真值(球心x−车x)'
          +cmFmt(gapFin.dx)+'cm='+cmSigned(aimDx)+'cm'
          +'；dz=预测rel_z '+cmFmt(th.ref300Z)+'cm − 真值球心z'
          +cmFmt(gapFin.dz)+'cm='+cmSigned(aimDz)+'cm'
          +'；正=预测点在真实球位的 +x 侧/上方（臂瞄偏的方向）'
          +'；两侧都是世界轴、各自相对各自的车（预测用同条消息的 car_pred_x，真值用车实际x），'
          +'故不含 yaw 旋转项；预测取该抛 S1@≈300ms 那条消息（ht='
          +(isNum(th.ref300Ht)?rkToPc(th.ref300Ht).toFixed(3):'—')+'s）'
          +'；真值与左列「球面y−车y」是同一份 RK 全量拟合、同一个取值时刻（臂最后更新HT）'
          +'；'+truthNote)+'">'
          +cmSigned(aimDx)+'/'+cmSigned(aimDz)+'</span>'
      : (gapFin?'<span title="本抛无S1@≈300ms参考消息，无预测击球点">—</span>':gapMiss);
    if(!isNum(th.ref300T)||!isNum(th.ref300Ht)||!isNum(th.ref300X)||!isNum(th.ref300Z)){
      return '<tr><td>'+(idx+1)+'</td><td>—</td><td>—</td><td>—</td><td>'+runTargetError+'</td><td>'+rkPredCarError+'</td><td>'+acceptedTarget+'</td>'+
        '<td>—</td><td>'+pcTruthCell(truthAcc,true,finalHt!=null?rkToPc(finalHt):null)+'</td><td>'+tcpCell+'</td><td>'+tcpAcceptedError+'</td><td>'+updGapCell+'</td>'+
        '<td>'+blindCell+'</td><td>'+dHtCell+'</td><td>'+gapCell+'</td><td>'+aimCell+'</td>'+
        '<td>'+carYawCell+'</td><td>'+tgtSpeedCell+'</td>'+
        '<td>'+faceYawCell+'</td><td>'+faceYawPreCell+'</td>'+
        '<td>'+returnCell+'</td>'+
        '<td><span style="color:#fbbf24">无S1@300ms</span> / msgs='+(th.msgs||0)+'</td><td class="armTblNote"><div>'+rejectNote+'</div></td></tr>';
    }
    const ctPc=rkToPc(th.ref300T);
    const htPc=rkToPc(th.ref300Ht);
    const truth=pcTruthAt(htPc);
    const info='S'+th.ref300Stage+
      (isNum(th.ref300NFit)?' n_fit='+th.ref300NFit:'')+
      ' / msgs='+(th.msgs||0);
    return '<tr><td>'+(idx+1)+'</td>'+
      '<td>'+ctPc.toFixed(3)+'</td>'+
      '<td>'+(th.ref300Lead*1000).toFixed(1)+'</td>'+
      '<td>'+tableXzCm(th.ref300X,th.ref300Z)+'</td>'+
      '<td>'+runTargetError+'</td>'+
      '<td>'+rkPredCarError+'</td>'+
      '<td>'+acceptedTarget+'</td>'+
      '<td>'+pcTruthCell(truth,true,htPc)+'</td>'+
      '<td>'+pcTruthCell(truthAcc,true,finalHt!=null?rkToPc(finalHt):null)+'</td>'+
      '<td>'+tcpCell+'</td>'+
      '<td>'+tcpAcceptedError+'</td>'+
      '<td>'+updGapCell+'</td>'+
      '<td>'+blindCell+'</td>'+
      '<td>'+dHtCell+'</td>'+
      '<td>'+gapCell+'</td>'+
      '<td>'+aimCell+'</td>'+
      '<td>'+carYawCell+'</td>'+
      '<td>'+tgtSpeedCell+'</td>'+
      '<td>'+faceYawCell+'</td>'+
      '<td>'+faceYawPreCell+'</td>'+
      '<td>'+returnCell+'</td>'+
      '<td>'+info+'</td><td class="armTblNote"><div>'+rejectNote+'</div></td></tr>';
  });
  return '<div class="armTblWrap"><table class="armTbl"><thead><tr>'+
    '<th>#</th><th>RK ct@≈300ms<br>(s,PC轴)</th><th>lead(ms)</th>'+
    '<th>RK@≈300ms x/z(cm)</th><th>车RUN末帧 目标−实际 dx/dy(cm)<br>(RK世界系)</th>'+
    '<th>RK@≈300ms预测车@HT−RUN末实际 dx/dy(cm)<br>(RK世界系)</th><th>机械臂最后accepted目标 x/z(cm)</th>'+
    '<th>PC真值@HT300 x/y/z(cm)</th>'+
    '<th title="与左列同一套入弧拟合真值，只把评估时刻换成臂最后更新HT——臂真正执行的击球时刻'+
    '（重定相生效=被消费的那条 late ht saved 的原始 ht，否则=最后一条 accepted 的原始 ht；'+
    '未减臂内提前量，与右侧 TCP 列同锚）。两列之差=真值随 ht 变化移动的量'+
    '（球速≈10m/s 时 1ms≈1cm）">'+
    'PC真值@臂最后更新HT x/y/z(cm)</th>'+
    '<th title="FK TCP 臂系 x/y 按该时刻车yaw（/bot_state 瞬时值）旋到世界轴、z−armZOff 还原世界'+
    '高度；臂基≡车心（0标恒等式），即拍心−车中心的世界轴偏移，与左列 PC真值(球−车) 同轴同基准，'+
    '两列相减≈拍−球。悬停看臂系原值与旋转明细">TCP−车心@臂最后更新HT x/y/z(cm,世界轴)</th>'+
    '<th title="TCP(世界轴,rel实际车心)@臂最后更新HT − accepted目标(世界轴,rel car_pred)：'+
    'dx 目标侧=该消息世界x−car_pred_x，dz=rel_z(≡世界z)；'+
    '目标不随挥拍中重定相变。悬停对照老口径臂系伺服差">TCP−accepted dx/dz(cm,世界轴)</th>'+
    '<th title="臂受理的最后一条 /predict_hit_pos 到达时刻 − 挥拍段起点(老触球−HIT_T 0.25s)；'+
    '正=更新落在挥拍开始之后（0803 起挥拍窗内不再拒收，这些消息就是 ht 重定相的养料）。'+
    '悬停看两个绝对时刻与重定相是否生效">最后更新−挥拍起<br>(ms)</th>'+
    '<th title="定义：finalHt−finalCt，二者取自臂真正消费掉的最后一条 /predict_hit_pos 的原始 '+
    'ht/ct（同源、都不减臂内提前量）。重定相生效时=挥拍窗内最后一条 late ht saved 那条（它就是'+
    '重定相吃掉的那条，本场典型 160~190ms）；重定相未触发/剩余不足时=最后一条 accepted 那条'+
    '（典型 300~335ms）。这段时间预测纯外推、没有任何新观测进来，即本拍的信息盲区。'+
    '⚠— = 重定相生效但该条回配失败，拿不到同源 ht/ct，不出数（不退回 accepted 冒充）">'+
    '盲区 ht−ct@臂最后更新<br>(ms)</th>'+
    '<th title="挥拍时用的 ht（最后一条 accepted）→ 挥拍中重定相更新后的 ht，正=新预测把击球点推晚；'+
    '两者同为原始 ht，与臂内触球时刻(各减10ms)之差等价。灰字(未采纳)=剩余不足 60ms 控制器放弃了重定相">'+
    'Δht 重定相<br>(ms)</th>'+
    '<th title="在臂最后更新的那条 /predict_hit_pos 的 ht（臂真正执行的击球时刻）上，用 RK 全量无污染观测（球世界三轴二次拟合 × '+
    '车实际x/y挥拍前外推）量出的 (球心y−R球3.3cm)−车y，世界轴，单位cm。车y面≡拍面，'+
    '故 0=该 ht 正好是真实触球，正=那一刻球面还没够到拍面(ht 偏早)，负=球已穿过(ht 偏晚)。'+
    '悬停看闭合速度、等效时序(ms)与球/车两侧的拟合明细">'+
    '球面y−车y @臂最后更新HT<br>(cm, RK全量真值)</th>'+
    '<th title="该抛 S1@≈300ms 那条消息预测的击球点 (x,z) − 真值 (x,z)，单位cm。'+
    '这里的「真值」与左列同一份：RK 全量无污染观测（球世界三轴二次拟合 × 车实际x/y挥拍前外推，'+
    '不含任何预测量），取值时刻同样是臂最后更新HT（臂真正执行的击球时刻）。'+
    '两侧都是世界轴且各自相对各自的车：预测侧=消息世界x−同条的 car_pred_x、z 用 rel_z(车中心z≡地面0)；'+
    '真值侧=球心x−车实际x、球心z。故不含车体系↔世界轴的 yaw 旋转项。'+
    '正=预测点落在真实球位的 +x 侧/上方，即臂被瞄偏的方向">'+
    '击球点@300预测 − RK全量真值@臂最后更新HT<br>dx/dz(cm, 世界轴)</th>'+
    '<th title="车体 yaw@臂最后更新HT（臂受理的最后一条预测的原始 ht，未减臂内提前量；与左侧两列'+
    '击球真值、右侧两列拍面yaw,pitch 同锚）：取 /bot_state 瞬时值——车控 accept AprilTag 定位后 yaw 由 '+
    'IMU 连续更新、HT 结束后才重定位，故采样点无重定位台阶，挥拍位姿伪迹只塌陷位置不动 yaw。'+
    '悬停看 IMU yaw_speed 换算的 10ms 时序灵敏度，以及右侧拍面yaw,pitch@臂最后更新HT 列所减的 '+
    'PC AprilTag 挥拍前窗圆均值对照">'+
    '车yaw@臂最后更新HT(°)</th>'+
    '<th title="该次挥拍最后一条 accepted 状态自带的计划量：目标触球拍速（m/s，拍心，过完各级'+
    '钳位后的实际计划值，口径 2·|行程|/hit_time·x 且只算 J1）/ 目标拍面仰角（°，0805 起随来球'+
    '俯冲角变，臂系≡世界系）。悬停看 speed_req(原始指令)、shortened(引拍被夹 rad)、face_yaw 目标。'+
    '注意：speed 是受理时按零起速算的账，挥拍段在首帧建、并在 ht 重定相触发点用当刻(q,v)重建，'+
    '起速非零会抬高真实触球速度，与右列实测对照时以右列悬停的「同刻指令侧」为准">'+
    '目标挥拍速度/pitch<br>(m/s, °)</th>'+
    '<th title="拍面法向（FK link6 +X）的世界 yaw / pitch，同一份冲击前窗[−80,−6]ms 线性拟合；'+
    '灰字为实测拍心速度 |v_tcp|（m/s，完整 FK 解析 Jacobian，HT 处直接插值不外推）。'+
    'yaw=face_yaw−车yaw（车侧取挥拍前圆均值）；pitch=asin(n_z) 不减车yaw——J1/BASE_ROT 纯 z 转'+
    '不动 n_z，臂系 pitch≡世界 pitch，正=开面上仰。悬停看两个角速度（ψ̇ 几百°/s 时序敏感、'+
    'θ̇≈0 对时序免疫），以及拍速的 J1 分量、同刻指令值、臂内触球锚（指令速度平台首帧）上的'+
    '指令/实测与触球欠速、单点抖动 σ">'+
    '拍面yaw,pitch,speed@臂最后更新HT(°,°,m/s;世界系)</th>'+
    '<th title="同左列同一份拟合/同一套拍速口径，只把取值时刻挪到 HT−12ms（角度落在窗内为纯插值，'+
    '拍速为插值）。−12ms 是固定探针，且本场用指令速度平台首帧定位的臂内触球锚中位就在 HT−11ms，'+
    '故本列基本踩在真实触球上；与左列之差 = 12ms 时序误差的代价——yaw 上是 ψ̇×12ms（度级）、'+
    'pitch 上≈0、拍速上是这 12ms 里 J1 还在加速的量">'+
    '拍面yaw,pitch,speed@臂最后更新HT−12ms(°,世界系)</th>'+
    '<th>PC回球 yaw/俯仰(°)</th><th>消息</th><th>备注</th></tr></thead>'+
    '<tbody>'+rows.join('')+'</tbody></table></div>';
};

const armAcceptedTableHtml = () => {
  if(!ARM) return '';
  if(!armAligned) return '<div style="color:#f87171">机械臂数据未与RK单调钟对齐，无法可靠回配最后accepted原消息。</div>';
  const hits=armHitMarks.filter(h=>h.label==='hit');
  if(!hits.length) return '<div style="color:#a0a0c0">本场没有 accepted hit。</div>';
  const rows=hits.map(h=>{
    const recvPc=isNum(h.lastAcceptT)?rkToPc(h.lastAcceptT):null;
    const sourceOk=isNum(h.wct)&&isNum(h.wht)&&isNum(h.wx)&&isNum(h.wz);
    if(!sourceOk){
      return '<tr><td>—</td><td>'+tableFmt(recvPc,3)+'</td><td>—</td><td>—</td><td>—</td><td>无法回配</td>'+
        '<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>updates='+(h.n||1)+'</td><td>—</td></tr>';
    }
    const accCt=h.wct-RK.t0;
    const accHt=h.wht-RK.t0;
    const accCtPc=rkToPc(accCt);
    const accHtPc=rkToPc(accHt);
    const th=matchThrowByAcceptedCt(accCt);
    const throwNo=th?reportThrows.indexOf(th)+1:null;
    const truth=pcTruthAt(accHtPc);
    const dx=truth&&isNum(h.wxw)&&isNum(h.wcarx)?(h.wxw-h.wcarx)-truth.x:null;
    const dz=truth?h.wz-truth.z:null;
    const has300=!!(th&&isNum(th.ref300Ht)&&isNum(th.ref300X)&&isNum(th.ref300Z));
    const dHt=has300?(accHt-th.ref300Ht)*1000:null;
    const dX=has300?h.wx-th.ref300X:null;
    const dZ=has300?h.wz-th.ref300Z:null;
    const result=strikeAfter(accHtPc).verdict;
    const info='S'+tableFmt(h.wstage,0)+
      (isNum(h.wnFit)?' n_fit='+h.wnFit:'')+
      ' / updates='+(h.n||1);
    return '<tr><td>'+tableFmt(throwNo,0)+'</td>'+
      '<td>'+tableFmt(recvPc,3)+'</td>'+
      '<td>'+accCtPc.toFixed(3)+'</td>'+
      '<td>'+accHtPc.toFixed(3)+'</td>'+
      '<td>'+((accHt-accCt)*1000).toFixed(1)+'</td>'+
      '<td>'+tableXzCm(h.wx,h.wz)+'</td>'+
      '<td>'+pcTruthCell(truth,false,accHtPc)+'</td>'+
      '<td>'+cmSigned(dx)+'/'+cmSigned(dz)+'</td>'+
      '<td>'+(has300?rkToPc(th.ref300Ht).toFixed(3):'—')+'</td>'+
      '<td>'+(has300?tableXzCm(th.ref300X,th.ref300Z):'—')+'</td>'+
      '<td>'+tableSigned(dHt)+'</td>'+
      '<td>'+cmSigned(dX)+'/'+cmSigned(dZ)+'</td>'+
      '<td>'+info+'</td><td>'+result+'</td></tr>';
  });
  return '<div class="armTblWrap"><table class="armTbl"><thead><tr>'+
    '<th>RK抛#</th><th>最后accepted t<br>(s,PC轴)</th><th>接收消息ct<br>(s,PC轴)</th><th>接收消息HT<br>(s,PC轴)</th>'+
    '<th>lead(ms)</th><th>最后accepted RK x/z(cm)</th><th>PC真值@accepted HT(cm)</th><th>accepted−PC dx/dz(cm)</th>'+
    '<th>RK300 HT(s)</th><th>RK300 x/z(cm)</th><th>accepted−300 ΔHT(ms)</th><th>accepted−300 Δx/dz(cm)</th>'+
    '<th>消息</th><th>结果</th></tr></thead><tbody>'+rows.join('')+'</tbody></table></div>';
};

const renderRk300Table = () => {
  const el=document.getElementById('rk300Tbl');
  if(el) el.innerHTML=alignWarnHtml+armDataWarnHtml+rk300TableHtml();
};
// ================= 共享数据层结束 =================

buildPlots[0] = () => {
  const oT=obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t));
  const rT=racket.map(r=>isNum(r.rel_s) ? r.rel_s : relTime(r.t));
  const tr=[
    g2({x:oT, y:obs.map(o=>o.x), name:'Ball X', mode:'markers',
     marker:{color:'#7f8c8d',symbol:'circle',size:2,opacity:0.5},
     hovertemplate:'t=%{x:.3f}s<br>x=%{y:.3f} m<extra>Ball X</extra>',
     visible:'legendonly'}),
    g2({x:oT, y:obs.map(o=>o.y), name:'Ball Y', mode:'markers',
     marker:{color:'#95a5a6',symbol:'circle',size:2,opacity:0.5},
     hovertemplate:'t=%{x:.3f}s<br>y=%{y:.3f} m<extra>Ball Y</extra>',
     visible:'legendonly'}),
    g2({x:oT, y:obs.map(o=>o.z), name:'Ball Z', mode:'markers',
     marker:{color:'#bdc3c7',symbol:'circle',size:2.5,opacity:0.6},
     hovertemplate:'t=%{x:.3f}s<br>z=%{y:.3f} m<extra>Ball Z</extra>'}),

    ...(racket.length ? [
    g2({x:rT, y:racket.map(r=>r.x), name:'Racket X', mode:'markers',
     marker:{color:'#ff66cc',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket x=%{y:.3f} m<extra>Racket X</extra>',
     visible:'legendonly'}),
    g2({x:rT, y:racket.map(r=>r.y), name:'Racket Y', mode:'markers',
     marker:{color:'#ff33aa',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket y=%{y:.3f} m<extra>Racket Y</extra>',
     visible:'legendonly'}),
    g2({x:rT, y:racket.map(r=>r.z), name:'Racket Z', mode:'markers',
     marker:{color:'#cc00ff',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket z=%{y:.3f} m<extra>Racket Z</extra>'}),
    ] : []),

    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.x), name:'S0 X', mode:'markers',
     marker:{color:'#3498db',symbol:'triangle-up',size:5},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0 X</extra>'}),
    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.y), name:'S0 Y', mode:'markers',
     marker:{color:'#2980b9',symbol:'triangle-up',size:5},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0 Y</extra>'}),
    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.z), name:'S0 Z', mode:'markers',
     marker:{color:'#1abc9c',symbol:'triangle-up',size:5},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S0 Z</extra>'}),

    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.x), name:'S1@RK-S0 X', mode:'markers',
     marker:{color:'#e74c3c',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 X</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.y), name:'S1@RK-S0 Y', mode:'markers',
     marker:{color:'#c0392b',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 Y</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.z), name:'S1@RK-S0 Z', mode:'markers',
     marker:{color:'#e67e22',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 Z</extra>'}),

    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(predRemainingMs), name:'S0 remaining(ms)', mode:'markers',
     marker:{color:'#9b59b6',symbol:'triangle-up',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S0 remaining</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(predRemainingMs), name:'S1@RK-S0 remaining(ms)', mode:'markers',
     marker:{color:'#8e44ad',symbol:'square',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S1 remaining</extra>'}),

    // compute_latency = compute_t - ct（算完时刻 − 曝光时刻；tracker 内部耗时）
    // 旧 JSON 没有 compute_t 时过滤掉，避免 null 画成 0
    g2({x:s0.filter(p=>p.compute_t!=null).map(p=>relTime(p.ct)),
        y:s0.filter(p=>p.compute_t!=null).map(p=>(p.compute_t-p.ct)*1000),
        name:'S0 compute(ms)', mode:'markers',
        marker:{color:'#f39c12',symbol:'triangle-up',size:4}, yaxis:'y2',
        hovertemplate:'t=%{x:.3f}s<br>compute=%{y:.1f} ms<extra>S0 compute</extra>'}),
    g2({x:s1.filter(p=>p.compute_t!=null).map(p=>relTime(p.ct)),
        y:s1.filter(p=>p.compute_t!=null).map(p=>(p.compute_t-p.ct)*1000),
        name:'S1 compute(ms)', mode:'markers',
        marker:{color:'#d35400',symbol:'square',size:4}, yaxis:'y2',
        hovertemplate:'t=%{x:.3f}s<br>compute=%{y:.1f} ms<extra>S1 compute</extra>'}),

    ...(car.length ? [
    g2({x:car.map(c=>relTime(c.t)), y:car.map(c=>c.x), name:'Car X', mode:'markers',
     marker:{color:'#2ecc71',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car x=%{y:.3f} m<extra>Car X</extra>',
     visible:'legendonly'}),
    g2({x:car.map(c=>relTime(c.t)), y:car.map(c=>c.y), name:'Car Y', mode:'markers',
     marker:{color:'#27ae60',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car y=%{y:.3f} m<extra>Car Y</extra>',
     visible:'legendonly'}),
    g2({x:car.map(c=>relTime(c.t)), y:car.map(c=>c.z), name:'Car Z', mode:'markers',
     marker:{color:'#f1c40f',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car z=%{y:.3f} m<extra>Car Z</extra>',
     visible:'legendonly'}),
    ] : []),
  ];

  Plotly.newPlot('c0',tr,{
    ...DL,
    title:{text:'All Curves - click legend to toggle, scroll to zoom',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS}, yaxis:{title:'Value (m)',...GS},
    yaxis2:{title:'Remaining / compute (ms)',...GS,overlaying:'y',side:'right'},
  },PLOT_CONFIG).then(()=>{wl('c0','l0');wz('c0');});
};

// RK Car Move：每次抛球后的底盘移动，bot_state 原生 ~100Hz 逐帧回放。
// 分段 = phase 离开 WAIT（RK 收到该抛目标才 RUN，天然一抛一段），回到 WAIT 结束
// （含 BRAKE_IN_SWING / BRAKE_AFTER_SWING），前后各补 0.5s 上下文。
// 老 JSON（重提取前）无 phase/vx 等字段：退回 target_x 非空段分段，缺的量显示 —。
buildPlots[1] = () => {
  if(!RK) return;
  const numv = v => (typeof v==='number' && Number.isFinite(v)) ? v : null;
  const bT=ts(RK.bot), bY=k=>ys(RK.bot,k);
  const cols={x:bY('x'), y:bY('y'), yaw:bY('yaw'), vx:bY('vx'), vy:bY('vy'),
    phase:bY('phase'), steer:bY('steer_angle'), rem:bY('remaining'),
    tx:bY('target_x'), ty:bY('target_y')};
  const rows=[];
  for(let i=0;i<bT.length;i++){
    const t=numv(Number(bT[i]));
    if(t===null) continue;
    rows.push({t, x:numv(cols.x[i]), y:numv(cols.y[i]), yaw:numv(cols.yaw[i]),
      vx:numv(cols.vx[i]), vy:numv(cols.vy[i]),
      phase:cols.phase[i]!=null?String(cols.phase[i]):null,
      steer:numv(cols.steer[i]), rem:numv(cols.rem[i]),
      tx:numv(cols.tx[i]), ty:numv(cols.ty[i])});
  }
  rows.sort((a,b)=>a.t-b.t);
  const sel=document.getElementById('mvSel'), slider=document.getElementById('mvSlider'),
        clock=document.getElementById('mvClock'), note=document.getElementById('mvNote');
  const el=id=>document.getElementById(id);
  const V={frame:el('mvFrameV'), tRk:el('mvTRk'), tPc:el('mvTPc'), phase:el('mvPhase'),
    pos:el('mvPos'), spd:el('mvSpd'), vxy:el('mvVxy'), yaw:el('mvYaw'), imuW:el('mvImuW'),
    steer:el('mvSteer'), steerTgt:el('mvSteerTgt'), steerDir:el('mvSteerDir'), rem:el('mvRem'), tgt:el('mvTgt'), dist:el('mvDist')};
  if(!rows.length){ if(clock) clock.textContent='无 /bot_state 数据'; return; }
  // —— 分段 ——
  const hasPhase = rows.some(r=>r.phase!==null);
  const act = r => hasPhase ? (r.phase!==null && r.phase!=='WAIT') : (r.tx!==null);
  const spans=[];
  {
    let s=null;
    for(let i=0;i<rows.length;i++){
      if(act(rows[i])){ if(s===null) s=i; }
      else if(s!==null){ if(rows[i-1].t-rows[s].t>=0.3) spans.push([s,i-1]); s=null; }
    }
    if(s!==null && rows[rows.length-1].t-rows[s].t>=0.3) spans.push([s,rows.length-1]);
  }
  const PAD_B=0.5, PAD_A=hasPhase?0.5:2.0;
  const movements=spans.map(([a,b],k)=>{
    let i0=a; while(i0>0 && rows[a].t-rows[i0-1].t<=PAD_B) i0--;
    let i1=b; while(i1<rows.length-1 && rows[i1+1].t-rows[b].t<=PAD_A) i1++;
    let tx=null, ty=null, rem0=null;
    for(let i=a;i<=b;i++){
      if(rows[i].tx!==null){ tx=rows[i].tx; ty=rows[i].ty; }
      if(rem0===null && rows[i].rem!==null) rem0=rows[i].rem;
    }
    return {k, frames:rows.slice(i0,i1+1), runT0:rows[a].t, runT1:rows[b].t, tx, ty, rem0};
  });
  if(!movements.length){
    if(clock) clock.textContent='未检测到移动段（无 phase/target 活跃区间）';
    return;
  }
  // —— 舵轮转速 / IMU 角速度：最近邻查表（各自 ~100Hz，容差 60ms）——
  const mkLut=(series,key)=>{
    const T=ts(series), Y=ys(series,key);
    const out=[];
    for(let i=0;i<T.length;i++){
      const t=Number(T[i]), v=numv(Y[i]);
      if(Number.isFinite(t) && v!==null) out.push({t,v});
    }
    return out.sort((a,b)=>a.t-b.t);
  };
  const steerVelLut=mkLut(RK.steer_motor,'velocity');
  const steerCmdLut=RK.steer_cmd?mkLut(RK.steer_cmd,'position'):[];  // 老 JSON 无 steer_cmd → 显示 —
  const imuWLut=mkLut(RK.imu,'yaw_speed');
  const lutAt=(lut,t,tol)=>{
    if(!lut.length) return null;
    let lo=0,hi=lut.length;
    while(lo<hi){const m=(lo+hi)>>1; if(lut[m].t<t) lo=m+1; else hi=m;}
    let best=null;
    if(lo<lut.length) best=lut[lo];
    if(lo>0 && (best===null || Math.abs(lut[lo-1].t-t)<Math.abs(best.t-t))) best=lut[lo-1];
    return (best && Math.abs(best.t-t)<=tol) ? best.v : null;
  };
  // —— 播放状态 ——
  let seg=movements[0], cur=0, playing=false, raf=0, lastWall=null, acc=0;
  const speedSel=el('mvSpeed'), playBtn=el('mvPlay');
  const speed=()=>Number(speedSel && speedSel.value)||1;
  const deg=v=>v===null?'—':`${(v*180/Math.PI).toFixed(1)}° (${v.toFixed(3)} rad)`;
  // —— 2D 绘制：静态 2 条（全程路径/起点）+ 动态 6 条（目标/车→目标/轨迹/舵轮箭头/速度/车）——
  const ARROW_OFF=0.16, VEL_SCALE=0.4;
  let effTgt=[];   // 每帧生效目标：本帧激活的 target；未激活时沿用本段内上一次激活值
  const buildEffTgt=()=>{
    effTgt=new Array(seg.frames.length);
    let last=null;
    for(let i=0;i<seg.frames.length;i++){
      const f=seg.frames[i];
      if(f.tx!==null&&f.ty!==null) last={x:f.tx, y:f.ty};
      effTgt[i]=last?{x:last.x, y:last.y, live:f.tx!==null}:null;
    }
  };
  const dyn=f=>{
    const has=f.x!==null&&f.y!==null;
    const tgt=effTgt[cur]||null;
    // 舵轮箭头：方向 = yaw+steer；运动中（|v|>0.1）按速度符号消歧（舵轮可反向驱动）
    let arrow=null;
    if(has&&f.steer!==null){
      let a=(f.yaw!==null?f.yaw:0)+f.steer;
      if(f.vx!==null&&f.vy!==null&&Math.hypot(f.vx,f.vy)>0.1
         && Math.cos(a)*f.vx+Math.sin(a)*f.vy<0) a+=Math.PI;
      arrow={x:f.x+Math.cos(a)*ARROW_OFF, y:f.y+Math.sin(a)*ARROW_OFF, deg:90-a*180/Math.PI};
    }
    const vel=(has&&f.vx!==null&&f.vy!==null&&Math.hypot(f.vx,f.vy)>0.02)
      ? {x:[f.x, f.x+f.vx*VEL_SCALE], y:[f.y, f.y+f.vy*VEL_SCALE]} : {x:[],y:[]};
    const toTgt=(has&&tgt)?{x:[f.x,tgt.x],y:[f.y,tgt.y]}:{x:[],y:[]};
    const trailX=[], trailY=[];
    for(let i=0;i<=cur;i++){ trailX.push(seg.frames[i].x); trailY.push(seg.frames[i].y); }
    return {tgt, arrow, vel, toTgt, trailX, trailY, carX:has?[f.x]:[], carY:has?[f.y]:[]};
  };
  const traces=f=>{
    const d=dyn(f);
    const f0=seg.frames.find(r=>r.x!==null);
    return [
      {type:'scatter', x:seg.frames.map(r=>r.x), y:seg.frames.map(r=>r.y), name:'全程路径',
       mode:'lines', line:{color:'#34406b',width:1.5}, hoverinfo:'skip'},
      {type:'scatter', x:f0?[f0.x]:[], y:f0?[f0.y]:[], name:'起点', mode:'markers',
       marker:{color:'#94a3b8',symbol:'diamond',size:9}, hoverinfo:'skip'},
      {type:'scatter', x:d.tgt?[d.tgt.x]:[], y:d.tgt?[d.tgt.y]:[], name:'目标位置(每帧)',
       mode:'markers', marker:{color:'#fbbf24',symbol:'star',size:15,line:{color:'#fff',width:0.5}},
       hovertemplate:'target=(%{x:.3f}, %{y:.3f}) m<extra>目标</extra>'},
      {type:'scatter', x:d.toTgt.x, y:d.toTgt.y, name:'车→目标', mode:'lines',
       line:{color:'#fbbf24',width:1,dash:'dot'}, opacity:0.6, hoverinfo:'skip'},
      {type:'scatter', x:d.trailX, y:d.trailY, name:'已走轨迹', mode:'lines',
       line:{color:'#5cd0ff',width:2.5}, hoverinfo:'skip'},
      {type:'scatter', x:d.arrow?[d.arrow.x]:[], y:d.arrow?[d.arrow.y]:[], name:'舵轮方向',
       mode:'markers', marker:{color:'#fde047',symbol:'arrow-wide',size:15,
       angle:d.arrow?d.arrow.deg:0, line:{color:'#1a1a2e',width:0.5}}, hoverinfo:'skip'},
      {type:'scatter', x:d.vel.x, y:d.vel.y, name:`速度矢量(×${VEL_SCALE}s)`, mode:'lines',
       line:{color:'#2dd4bf',width:2}, hoverinfo:'skip'},
      {type:'scatter', x:d.carX, y:d.carY, name:'车', mode:'markers',
       marker:{color:'#e94560',symbol:'circle',size:11,line:{color:'#fff',width:1}},
       hoverinfo:'skip'},
    ];
  };
  // 等比坐标自己算（不用 scaleanchor：其约束求解器会把算过的范围当"用户编辑"，
  // react 切换移动段时旧范围粘住不更新）：按绘图区像素宽高取同一 m/px，居中放置。
  const MARGIN={l:60,r:20,t:40,b:50};
  const layout=()=>{
    const xs=[], ys2=[];
    seg.frames.forEach(r=>{
      if(r.x!==null&&r.y!==null){xs.push(r.x); ys2.push(r.y);}
      if(r.tx!==null&&r.ty!==null){xs.push(r.tx); ys2.push(r.ty);}   // 目标逐帧会移动，全部包进视野
    });
    if(!xs.length){ xs.push(0); ys2.push(0); }
    const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys2), y1=Math.max(...ys2);
    const padOf=(a,b)=>Math.max(0.45,(b-a)*0.18);
    const px=padOf(x0,x1), py=padOf(y0,y1);
    const div=document.getElementById('c1');
    const W=Math.max(200,((div&&div.clientWidth)||1100)-MARGIN.l-MARGIN.r);
    const H=Math.max(200,((div&&div.clientHeight)||680)-MARGIN.t-MARGIN.b);
    const cx=(x0+x1)/2, cy=(y0+y1)/2;
    const mpp=Math.max((x1-x0+2*px)/W,(y1-y0+2*py)/H);   // meters per pixel，取大者兜住两轴
    const sx=mpp*W/2, sy=mpp*H/2;
    return {
      ...DL,
      title:{text:`移动 #${seg.k+1} — bot_state ~100Hz 回放（RK 里程计世界系）`,font:{size:13,color:'#a0a0c0'}},
      margin:MARGIN,
      legend:{...DL.legend, orientation:'h', y:-0.08},
      xaxis:{title:'X (m)',...GS,range:[cx-sx,cx+sx]},
      yaxis:{title:'Y (m)',...GS,range:[cy-sy,cy+sy]},
    };
  };
  const DYN_IDX=[2,3,4,5,6,7];
  const setSide=f=>{
    V.frame.textContent=`${cur+1} / ${seg.frames.length}`;
    V.tRk.textContent=`${f.t.toFixed(3)} s`;
    V.tPc.textContent=`${rkToPc(f.t).toFixed(3)} s`;
    V.phase.textContent=f.phase!==null?f.phase:'—';
    V.pos.textContent=(f.x!==null&&f.y!==null)?`(${f.x.toFixed(3)}, ${f.y.toFixed(3)}) m`:'—';
    const spd=(f.vx!==null&&f.vy!==null)?Math.hypot(f.vx,f.vy):null;
    V.spd.textContent=spd===null?'—':`${spd.toFixed(3)} m/s`;
    V.vxy.textContent=f.vx===null?'—':`${f.vx.toFixed(3)} / ${f.vy.toFixed(3)} m/s`;
    V.yaw.textContent=deg(f.yaw);
    const w=lutAt(imuWLut,f.t,0.06);
    V.imuW.textContent=w===null?'—':`${w.toFixed(3)} rad/s`;
    V.steer.textContent=deg(f.steer);
    // 目标 steer：/chassis_can/steer_cmd MIT 位置设定点。BRAKE_IN/AFTER_SWING 不发 steer 帧，
    // 电机 MIT 自持上一帧设定点 → 显示最后一条并标"自持"。
    const sc=lutAt(steerCmdLut,f.t,0.06);
    if(sc!==null) V.steerTgt.textContent=deg(sc);
    else{
      let lo=0,hi=steerCmdLut.length;
      while(lo<hi){const m=(lo+hi)>>1; if(steerCmdLut[m].t<=f.t) lo=m+1; else hi=m;}
      const last=lo>0?steerCmdLut[lo-1]:null;
      V.steerTgt.textContent=last?`${deg(last.v)} 自持(${(f.t-last.t).toFixed(2)}s 无指令)`:'—';
    }
    let sv=lutAt(steerVelLut,f.t,0.06);
    if(sv===null && cur>0 && f.steer!==null){
      const p=seg.frames[cur-1];
      if(p.steer!==null && f.t>p.t) sv=(f.steer-p.steer)/(f.t-p.t);
    }
    V.steerDir.textContent=sv===null?'—':(Math.abs(sv)<0.05?`静止 (${sv.toFixed(2)} rad/s)`
      :(sv>0?`↺ 正转 +${sv.toFixed(2)} rad/s`:`↻ 反转 ${sv.toFixed(2)} rad/s`));
    V.rem.textContent=f.rem===null?'—（无激活目标）':`${f.rem.toFixed(3)} s`;
    const tgt=effTgt[cur]||null;
    V.tgt.textContent=!tgt?'—（尚未下发）':`(${tgt.x.toFixed(3)}, ${tgt.y.toFixed(3)}) m${tgt.live?'':' *'}`;
    V.dist.textContent=(tgt&&f.x!==null)?`${Math.hypot(tgt.x-f.x,tgt.y-f.y).toFixed(3)} m`:'—';
  };
  const render=()=>{
    const f=seg.frames[cur];
    slider.value=String(cur);
    clock.textContent=`帧 ${cur+1}/${seg.frames.length} · t(RK)=${f.t.toFixed(2)}s · PC=${rkToPc(f.t).toFixed(2)}s`;
    setSide(f);
    const d=dyn(f);
    Plotly.restyle('c1',{
      x:[d.tgt?[d.tgt.x]:[], d.toTgt.x, d.trailX, d.arrow?[d.arrow.x]:[], d.vel.x, d.carX],
      y:[d.tgt?[d.tgt.y]:[], d.toTgt.y, d.trailY, d.arrow?[d.arrow.y]:[], d.vel.y, d.carY],
      'marker.angle':[null,null,null,d.arrow?d.arrow.deg:0,null,null],
    },DYN_IDX);
  };
  const setFrame=i=>{
    cur=Math.max(0,Math.min(i,seg.frames.length-1));
    render();
  };
  let watchdog=0;
  const setPlaying=p=>{
    playing=p;
    if(playBtn) playBtn.textContent=p?'⏸ 暂停':'▶ 播放';
    if(p){
      lastWall=null; acc=0;
      if(!raf) raf=requestAnimationFrame(tick);
      // rAF 在被遮挡/后台标签页会被 Chrome 挂起：加 interval 看门狗兜底推进
      if(!watchdog) watchdog=setInterval(()=>{ if(playing) advance(performance.now()); },200);
    } else if(watchdog){
      clearInterval(watchdog); watchdog=0;
    }
  };
  const advance=wall=>{
    const pnl=document.getElementById('p1');
    if(!pnl || !pnl.classList.contains('on')){ setPlaying(false); return; }
    if(lastWall===null) lastWall=wall;
    acc+=(wall-lastWall)/1000*100*speed();   // 数据帧 ≈ 10ms 一帧
    lastWall=wall;
    const adv=Math.floor(acc);
    if(adv>0){
      acc-=adv;
      const nxt=cur+adv;
      if(nxt>=seg.frames.length-1){ setFrame(seg.frames.length-1); setPlaying(false); return; }
      setFrame(nxt);
    }
  };
  const tick=wall=>{
    raf=0;
    if(!playing) return;
    advance(wall);
    if(playing) raf=requestAnimationFrame(tick);
  };
  // 二次校正：首轮 react 后用 Plotly 实测的轴像素长（扣掉图例/标题后）把 m/px 拉齐
  const fixAspect=()=>{
    const p=document.getElementById('c1');
    const fx=p&&p._fullLayout&&p._fullLayout.xaxis, fy=p&&p._fullLayout&&p._fullLayout.yaxis;
    if(!fx||!fy||!fx._length||!fy._length||!Array.isArray(fx.range)) return;
    const rx=fx.range, ry=fy.range;
    const mppx=(rx[1]-rx[0])/fx._length, mppy=(ry[1]-ry[0])/fy._length;
    if(!(mppx>0)||!(mppy>0)||Math.abs(mppx/mppy-1)<0.01) return;
    const mpp=Math.max(mppx,mppy);
    const cx=(rx[0]+rx[1])/2, cy=(ry[0]+ry[1])/2;
    const sx=mpp*fx._length/2, sy=mpp*fy._length/2;
    Plotly.relayout('c1',{'xaxis.range':[cx-sx,cx+sx],'yaxis.range':[cy-sy,cy+sy]});
  };
  window.addEventListener('resize',()=>setTimeout(fixAspect,250));
  const setMovement=k=>{
    setPlaying(false);
    seg=movements[Math.max(0,Math.min(k,movements.length-1))];
    cur=0;
    buildEffTgt();
    slider.max=String(seg.frames.length-1);
    slider.value='0';
    Plotly.react('c1',traces(seg.frames[0]),layout(),{...PLOT_CONFIG,scrollZoom:true})
      .then(()=>{ fixAspect(); render(); });
  };
  // —— 控件 ——
  movements.forEach(m=>{
    const opt=document.createElement('option');
    opt.value=String(m.k);
    const tgt=m.tx!==null?` 末目标(${m.tx.toFixed(2)}, ${m.ty.toFixed(2)})`:'';
    const rem=m.rem0!==null?` 计划${m.rem0.toFixed(2)}s`:'';
    opt.textContent=`第 ${m.k+1} 次  RK ${m.runT0.toFixed(1)}→${m.runT1.toFixed(1)}s（PC ~${rkToPc(m.runT0).toFixed(1)}s）${tgt}${rem}`;
    sel.appendChild(opt);
  });
  sel.addEventListener('change',()=>setMovement(Number(sel.value)||0));
  slider.addEventListener('input',()=>{ setPlaying(false); setFrame(Number(slider.value)||0); });
  el('mvFirst').addEventListener('click',()=>{ setPlaying(false); setFrame(0); });
  el('mvPrev').addEventListener('click',()=>{ setPlaying(false); setFrame(cur-1); });
  el('mvNext').addEventListener('click',()=>{ setPlaying(false); setFrame(cur+1); });
  playBtn.addEventListener('click',()=>{
    if(!playing && cur>=seg.frames.length-1) setFrame(0);   // 播完再按播放从头开始
    setPlaying(!playing);
  });
  document.addEventListener('keydown',e=>{
    const pnl=document.getElementById('p1');
    if(!pnl || !pnl.classList.contains('on')) return;
    const tag=(e.target&&e.target.tagName)||'';
    if(/INPUT|SELECT|TEXTAREA/.test(tag)) return;
    if(e.key==='ArrowLeft'){ setPlaying(false); setFrame(cur-1); e.preventDefault(); }
    else if(e.key==='ArrowRight'){ setPlaying(false); setFrame(cur+1); e.preventDefault(); }
    else if(e.key===' '){ if(!playing&&cur>=seg.frames.length-1) setFrame(0); setPlaying(!playing); e.preventDefault(); }
  });
  if(note) note.innerHTML=
    `分段 = bot_state.phase 离开 WAIT（每抛 RK 下发目标后 RUN，含 BRAKE 两段），前后补 ${PAD_B}s 上下文，共 ${movements.length} 次移动。`+
    `<br>舵轮箭头方向 = yaw+steer，运动中按速度符号消歧（舵轮可反向驱动）；vx/vy 为世界系（与 dx/dt 中位差 0.02m/s）。`+
    `<br>目标 steer = /chassis_can/steer_cmd 的 MIT 位置设定点（SteerController 每拍限速斜坡，非最终朝向 theta_des）；BRAKE 两段不发 steer 帧，电机自持上一设定点（标"自持"）。`+
    `<br>目标星标与右栏逐帧刷新（RUN 中目标会随预测更新移动）；带 * = 该帧目标未激活，沿用本段上一次下发值。`+
    `<br>剩余到位时间 = bot_state.remaining（仅 target_active 时有值）。快捷键：←/→ 逐帧，空格 播放/暂停。`;
  setMovement(0);
};

buildPlots[2] = () => {
  Plotly.newPlot('c2',[
    {x:obs.map(o=>o.x),y:obs.map(o=>o.y),z:obs.map(o=>o.z),
     mode:'markers',type:'scatter3d',name:'Ball',
      marker:{color:obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t)),colorscale:'Viridis',size:2,opacity:0.5,
       colorbar:{title:'t(s)',len:0.5,tickfont:{color:'#e0e0e0'},titlefont:{color:'#e0e0e0'}}},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Ball</extra>',
      text:obs.map(o=>(isNum(o.rel_s) ? o.rel_s : relTime(o.t)).toFixed(3))},
    ...(racket.length ? [{
     x:racket.map(r=>r.x),y:racket.map(r=>r.y),z:racket.map(r=>r.z),
     mode:'markers',type:'scatter3d',name:'Racket',
     marker:{color:'#ff33aa',size:4,symbol:'diamond'},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Racket</extra>',
     text:racket.map(r=>(isNum(r.rel_s) ? r.rel_s : relTime(r.t)).toFixed(3))
    }] : []),
    {x:s0.map(p=>p.x),y:s0.map(p=>p.y),z:s0.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S0 pred',
     marker:{color:'#3498db',size:4,symbol:'diamond'},
     customdata:s0.map(predRemainingMs),
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>remaining=%{customdata:.1f} ms<extra>S0</extra>',
     text:s0.map(p=>relTime(p.ct).toFixed(3))},
    {x:s1.map(p=>p.x),y:s1.map(p=>p.y),z:s1.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S1@RK-S0 pred',
     marker:{color:'#e74c3c',size:4,symbol:'diamond',line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>remaining=%{customdata:.1f} ms<extra>S1</extra>',
      text:s1.map(p=>relTime(p.ct).toFixed(3))},
    ...(car.length ? [{x:car.map(c=>c.x),y:car.map(c=>c.y),z:car.map(c=>c.z),
     mode:'markers',type:'scatter3d',name:'Car',
     marker:{color:'#2ecc71',size:4,symbol:'square'},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Car</extra>',
      text:car.map(c=>relTime(c.t).toFixed(3))}] : []),
  ],{
    ...DL,
    title:{text:'3D Trajectory',font:{size:13,color:'#a0a0c0'}},
    scene:{xaxis:{title:'X(m)',...GS,backgroundcolor:'#16213e'},
           yaxis:{title:'Y(m)',...GS,backgroundcolor:'#16213e'},
           zaxis:{title:'Z(m)',...GS,backgroundcolor:'#16213e'},bgcolor:'#16213e'},
  },PLOT_CONFIG).then(()=>{wl('c2','l2');wz('c2');});
};

buildPlots[3] = () => {
  if(car.length <= 0) return;
  const cT=car.map(c=>relTime(c.t));
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push(g2({x:cT,y:car.map(c=>c[k]),name:`Car ${k.toUpperCase()}`,mode:'markers',
      marker:{color:['#2ecc71','#27ae60','#f1c40f'][i],size:2},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Car ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'}));
  });
  tr.push(g2({x:cT,y:car.map(c=>c.yaw),name:'Car Yaw',mode:'markers',
    marker:{color:'#e94560',size:2},
    hovertemplate:'t=%{x:.3f}s<br>yaw=%{y:.4f}rad<extra>Car Yaw</extra>',
    yaxis:'y4',xaxis:'x'}));
  tr.push(g2({x:cT,y:car.map(c=>c.reprojection_error),name:'Reproj Err',mode:'markers',
    marker:{color:'#e67e22',size:2},
    hovertemplate:'t=%{x:.3f}s<br>err=%{y:.2f} px<extra>Reproj</extra>',
    yaxis:'y5',xaxis:'x'}));

  Plotly.newPlot('c3',tr,{
    ...DL,
    title:{text:'Car Location (X / Y / Z / Yaw / Reproj)',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS,domain:[0,1],anchor:'y5'},
    yaxis:{title:'X (m)',...GS,domain:[0.82,1]},
    yaxis2:{title:'Y (m)',...GS,domain:[0.62,0.79]},
    yaxis3:{title:'Z (m)',...GS,domain:[0.42,0.59]},
    yaxis4:{title:'Yaw (rad)',...GS,domain:[0.22,0.39]},
    yaxis5:{title:'Reproj (px)',...GS,domain:[0.0,0.19]},
  },PLOT_CONFIG).then(()=>{wl('c3','l3');wz('c3');});
};

buildPlots[4] = () => {
  if(!ARM) return;
  const escA = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const J = ARM.joint_names || ['joint1','joint2','joint3','joint4','joint5','joint6'];
  const JC = ['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2'];
  const AXC = ['#2ecc71','#f1c40f','#5cd0ff'];
  const events = ARM.events || [];
  const states = ARM.states, cmds = ARM.commands || [];
  // 击打标记与 FK TCP 都在共享数据层；臂数据已在 RK 相对轴，统一走 rkToPc。
  const aligned = armAligned, hitMarks = armHitMarks;
  // 逐 trace 自适应抽稀：值变化 >1% 量程的点全保（挥拍段满分辨率），平稳段 0.3s 一点。
  // 不再按 hit 窗口区分。SVG scatter（非 scattergl）：软渲染 WebGL 画不动全量点。
  const thinRows = (rows, get) => {
    let mn=Infinity, mx=-Infinity;
    rows.forEach(r=>{const v=get(r); if(isNum(v)){if(v<mn)mn=v; if(v>mx)mx=v;}});
    const eps=(mx>mn)?(mx-mn)*0.01:1e-9;
    const T=[], V=[];
    let lastT=-1e9, lastV=null;
    rows.forEach(r=>{
      const v=get(r);
      if(!isNum(v)) return;
      if(r.t-lastT>=0.3 || Math.abs(v-lastV)>eps){ T.push(r.t); V.push(v); lastT=r.t; lastV=v; }
    });
    return {T,V};
  };
  // 命令流在两次动作之间有长间隔，插入 null 断开连线，避免斜拉直线
  const seriesXY = (rows, get, gapS) => {
    const d=thinRows(rows,get);
    const X=[], Y=[];
    let prev=null;
    for(let i=0;i<d.T.length;i++){
      if(prev!==null && d.T[i]-prev>gapS){ X.push(null); Y.push(null); }
      X.push(armToPc(d.T[i])); Y.push(d.V[i]); prev=d.T[i];
    }
    return {X,Y};
  };
  const sv = t => ({type:'scatter', ...t});
  const fieldSeries = field => {
    const tr=[];
    J.forEach((name,i)=>{
      const cs=seriesXY(cmds, r=>(r[field]&&r[field][i]!=null)?r[field][i]:null, 0.5);
      tr.push(sv({x:cs.X, y:cs.Y, legendgroup:`${name} target`,
        name:`${name} target`, mode:'lines', line:{color:JC[i%JC.length],width:2},
        hovertemplate:`t=%{x:.3f}s<br>%{y:.4f}<extra>${name} target</extra>`}));
      const ss=seriesXY(states, r=>(r[field]&&r[field][i]!=null)?r[field][i]:null, 1.5);
      tr.push(sv({x:ss.X, y:ss.Y, legendgroup:`${name} actual`,
        name:`${name} actual`, mode:'lines', line:{color:JC[i%JC.length],width:1,dash:'dot'},
        hovertemplate:`t=%{x:.3f}s<br>%{y:.4f}<extra>${name} actual</extra>`}));
    });
    return tr;
  };
  const tcpSeries = () => {
    const tr=[];
    ['x','y','z'].forEach((ax,i)=>{
      const cs=seriesXY(cmds, r=>(r.tcp&&r.tcp[i]!=null)?r.tcp[i]:null, 0.5);
      tr.push(sv({x:cs.X, y:cs.Y,
        name:`TCP ${ax} target`, mode:'lines', line:{color:AXC[i],width:2},
        hovertemplate:`t=%{x:.3f}s<br>${ax}=%{y:.4f} m<extra>TCP ${ax} target</extra>`}));
      const ss=seriesXY(states, r=>(r.tcp&&r.tcp[i]!=null)?r.tcp[i]:null, 1.5);
      tr.push(sv({x:ss.X, y:ss.Y,
        name:`TCP ${ax} actual`, mode:'lines', line:{color:AXC[i],width:1,dash:'dot'},
        hovertemplate:`t=%{x:.3f}s<br>${ax}=%{y:.4f} m<extra>TCP ${ax} actual</extra>`}));
    });
    return tr;
  };
  const markShapes = () => {
    const sh=[];
    hitMarks.forEach(h=>{
      const line=(x,color,dash,width)=>sh.push({type:'line',xref:'x',yref:'paper',
        x0:armToPc(x),x1:armToPc(x),y0:0,y1:1,line:{color,width,dash},opacity:0.85});
      line(h.cmd,'#94a3b8','dot',1);
      if(h.start!=null) line(h.start,'#e94560','solid',1.8);
      line(h.done,'#2dd4bf','solid',1.8);
    });
    return sh;
  };
  const markAnnotations = () => {
    const an=[];
    hitMarks.forEach(h=>{
      const add=(x,text,color,yy)=>an.push({x:armToPc(x),y:yy,xref:'x',yref:'paper',text,
        showarrow:false,font:{size:10,color},xanchor:'left',yanchor:'top'});
      add(h.cmd,`${h.label} cmd`,'#94a3b8',0.995);
      if(h.start!=null) add(h.start,'起拍','#e94560',0.973);
      add(h.done,h.label==='hit'?'触球':`${h.label} done`,'#2dd4bf',0.951);
    });
    return an;
  };
  const setArmEv = () => {
    const marks=hitMarks.map(h=>{
      const seg=[`<b>${h.label}</b> cmd ${armToPc(h.cmd).toFixed(2)}s`];
      if(h.start!=null) seg.push(`start <b>${armToPc(h.start).toFixed(2)}s</b>`);
      seg.push(`done <b>${armToPc(h.done).toFixed(2)}s</b>`);
      return seg.join(' → ');
    });
    const axisInfo=(aligned
      ? `Axis: PC report time（PC t = ${rkScale.toFixed(8)} × RK t + ${rkBias.toFixed(4)}s） &nbsp; `
      : 'Axis: arm data time（未对齐：需 v3 _arm.json + RK 数据） &nbsp; ') +
      (ARM.fk_source ? `FK: ${escA(ARM.fk_source)} &nbsp; ` : '') +
      (marks.length ? '| ' + marks.join(' &nbsp;|&nbsp; ') : '| no accepted commands');
    document.getElementById('armEv').innerHTML =
      alignWarnHtml + armDataWarnHtml + armAcceptedTableHtml() +
      `<div style="margin:8px 0 2px;color:#a0a0c0">${axisInfo}</div>`;
  };
  // 单 plot 四层 subplot（同 Car Location 模式）：只占一个渲染 context。
  const bindAxis=(traces,ya)=>traces.map(t=>({...t,xaxis:'x',yaxis:ya}));
  const build = first => {
    setArmEv();
    const tr=[
      ...bindAxis(fieldSeries('position'),'y'),
      ...bindAxis(fieldSeries('velocity'),'y2'),
      ...bindAxis(fieldSeries('effort'),'y3'),
      ...bindAxis(tcpSeries(),'y4'),
    ];
    const layout={
      ...DL,
      showlegend:false,
      title:{text:'Arm — target(solid) vs actual(dot): Position / Velocity / Effort / TCP(FK)',
        font:{size:13,color:'#a0a0c0'}},
      xaxis:{title:aligned?'PC report time (s)':'Arm bag time (s)',...GS,domain:[0,1],anchor:'y4'},
      yaxis:{title:'Position (rad)',...GS,domain:[0.79,1]},
      yaxis2:{title:'Velocity (rad/s)',...GS,domain:[0.53,0.76]},
      yaxis3:{title:'Effort (Nm)',...GS,domain:[0.27,0.50]},
      yaxis4:{title:'TCP (m)',...GS,domain:[0.0,0.24]},
      shapes:markShapes(),
      annotations:markAnnotations(),
    };
    if(!first && typeof ZSTATE==='object' && ZSTATE.c4) delete ZSTATE.c4.fullRange;
    return (first?Plotly.newPlot:Plotly.react)('c4',tr,layout,PLOT_CONFIG)
      .then(()=>{ if(first){wl('c4','l4');wz('c4');} else {tl('c4','l4');} });
  };
  window.__rebuildArm = () => build(false);
  build(true);
};

buildPlots[6] = () => {
  ensurePlot(5);
  if(typeof window.__buildRkSignals === 'function') window.__buildRkSignals();
};

buildPlots[5] = () => {
  if(!RK) return;
  const input = document.getElementById('rkOff');
  const info = document.getElementById('rkInfo');
  const shifted = xs => xs.map(rkToPc);
  const setInfo = () => {
    publishRkTimeMap();
    const errText = auto.err==null ? 'n/a' : `${auto.err.toFixed(3)}m / ${auto.n} pts`;
    const srcText = presetBias!=null ? `preset bias ${presetBias.toFixed(4)}s; ` : '';
    info.innerHTML = (alignBad ? '<span style="color:#e94560;font-weight:700">⚠ 对齐不可信</span> ' : '')
      + `PC t = ${rkScale.toFixed(8)} × RK t + ${rkBias.toFixed(4)}s; `+
        `drift ${driftPpm>=0?'+':''}${driftPpm.toFixed(1)}ppm; ${srcText}auto z-shape ${errText}`;
  };
  const tr = (series,key,name,axis,color,mode='markers',extra={}) => g2({
    x:shifted(ts(series)), y:ys(series,key), name, mode,
    marker:{color,size:3}, line:{color,width:1.4},
    yaxis:axis, xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}<extra>${name}</extra>`,
    ...extra,
  });
  const rkWorldTr = (key,name,color,extra={}) => tr(RK.world,key,name,'y',color,'markers',{
    customdata:ts(RK.world).map((_,i)=>[
      ys(RK.world,'result_t')[i],
      ys(RK.world,'latency_ms')[i],
    ]),
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>result_t=%{customdata[0]:.6f}s<br>latency=%{customdata[1]:.1f} ms<extra>${name}</extra>`,
    ...extra,
  });
  const rkPredTr = (key,name,color,extra={}) => tr(RK.pred,key,name,'y',color,'markers',{
    customdata:ys(RK.pred,'duration').map((v,i)=>[isNum(v)?v*1000:null, isNum(rkPredNFit[i])?rkPredNFit[i]:'']),
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>remaining=%{customdata[0]:.1f} ms n_fit=%{customdata[1]}<extra>${name}</extra>`,
    ...extra,
  });
  const rkRemainingTr = () => g2({
    x:shifted(ts(RK.pred)), y:ys(RK.pred,'duration').map(v=>isNum(v)?v*1000:null),
    name:'RK Predict remaining(ms)', mode:'markers',
    marker:{color:'#fde047',size:4,symbol:'triangle-up'}, yaxis:'y2', xaxis:'x',
    hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>RK Predict remaining</extra>',
  });
  // PC hit 预测 S0+S1 合成一条（按 ct 排序）；stage 用点形区分：S0=三角、S1=方块
  const sAll = [...s0, ...s1].sort((a,b)=>((a.ct||0)-(b.ct||0)));
  const stageSym = s => s===0 ? 'triangle-up' : 'square';
  const pcHitTr = (key,name,color,extra={}) => g2({
    x:sAll.map(p=>relTime(p.ct)), y:sAll.map(p=>p[key]), name, mode:'markers',
    customdata:sAll.map(p=>[predRemainingMs(p), p.stage]),
    marker:{color,size:5,symbol:sAll.map(p=>stageSym(p.stage))}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>S%{customdata[1]} remaining=%{customdata[0]:.1f} ms<extra>${name}</extra>`,
    ...extra,
  });
  // RK ref：/predict_hit_pos 的 rel_x/rel_y/rel_z（击球点相对「击球时刻预测车位姿」车体系，臂端消费的量）。
  // 每抛最后一条 ref 只画一次：作为 star 挪到最终 ht 处（击球时刻臂在执行的参考，
  // 与 PC 真值 star 同横坐标可垂直对比），不再在其原消息时刻重复画常规点。
  const rkRefRows = key => {
    const t=ts(RK.pred), val=ys(RK.pred,key);
    const finalIdx=new Set(rkThrows.map(th=>th.lastRelIdx).filter(i=>i!=null));
    const rows=[];
    for(let i=0;i<t.length;i++){
      if(finalIdx.has(i)) continue;
      const ti=Number(t[i]);
      if(!isNum(ti) || !isNum(val[i])) continue;
      rows.push({t:rkToPc(ti), v:val[i], stage:rkPredStage[i], sym:stageSym(rkPredStage[i]), size:5,
                 note:(isNum(rkPredDurMs[i])?`remaining=${rkPredDurMs[i].toFixed(1)} ms`:'')
                      +(isNum(rkPredNFit[i])?` n_fit=${rkPredNFit[i]}`:'')});
    }
    rkThrows.forEach(th=>{
      if(!isNum(th[key])) return;
      rows.push({t:rkToPc(th.ht), v:th[key], stage:th.stage, sym:'star', size:11, note:'@ht final ref'});
    });
    return rows.sort((a,b)=>a.t-b.t);
  };
  const rkRefTr = (key,name,color,extra={}) => {
    const rows=rkRefRows(key);
    return g2({
      x:rows.map(r=>r.t), y:rows.map(r=>r.v), name, mode:'markers',
      customdata:rows.map(r=>[r.note, r.stage]),
      marker:{color, size:rows.map(r=>r.size), symbol:rows.map(r=>r.sym)}, yaxis:'y', xaxis:'x',
      hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<br>S%{customdata[1]} %{customdata[0]}<extra>${name}</extra>`,
      ...extra,
    });
  };
  const pcTr = (key,name,color,extra={}) => g2({
    x:pcRows.map(p=>p.t), y:pcRows.map(p=>p[key]), name, mode:'markers',
    marker:{color,size:2.5}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const pcCarTr = (key,name,color) => g2({
    x:pcCarRows.map(c=>c.t), y:pcCarRows.map(c=>c[key]), name, mode:'markers',
    marker:{color,size:2.5,symbol:'diamond'}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
  });
  const traceData = () => [
    pcTr('x','PC Ball X','#7f8c8d',{visible:'legendonly'}),
    pcTr('y','PC Ball Y','#95a5a6',{visible:'legendonly'}),
    pcTr('z','PC Ball Z','#bdc3c7'),
    rkWorldTr('x','RK World X','#3498db',{visible:'legendonly'}),
    rkWorldTr('y','RK World Y','#2980b9',{visible:'legendonly'}),
    rkWorldTr('z','RK World Z','#5cd0ff'),
    rkPredTr('x','RK Predict X','#f97316',{visible:'legendonly',marker:{color:'#f97316',size:6,symbol:'triangle-up'}}),
    rkPredTr('y','RK Predict Y','#fb923c',{visible:'legendonly',marker:{color:'#fb923c',size:6,symbol:'triangle-up'}}),
    rkPredTr('z','RK Predict Z','#e94560',{marker:{color:'#e94560',size:6,symbol:'triangle-up'}}),
    // car_pred_* 是该条 /predict_hit_pos 对 HT 时刻小车世界系位置的预测；横轴仍是消息 ct。
    rkPredTr('car_pred_x','/predict_hit_pos car_pred_x','#d946ef'),
    rkPredTr('car_pred_y','/predict_hit_pos car_pred_y','#c084fc'),
    pcHitTr('x','PC Hit X','#fb7185',{visible:'legendonly'}),
    pcHitTr('y','PC Hit Y','#f43f5e',{visible:'legendonly'}),
    pcHitTr('z','PC Hit Z','#e11d48'),
    rkRefTr('rel_x','RK Ref X','#a3e635'),
    rkRefTr('rel_y','RK Ref Y','#4d7c0f',{visible:'legendonly'}),
    rkRefTr('rel_z','RK Ref Z','#84cc16'),
    pcCarTr('x','PC Car X','#d946ef'),
    pcCarTr('y','PC Car Y','#c084fc'),
    // 机械臂 FK TCP（与 RK 数据同轴，经 rkToPc 到 PC 轴；z 减 armZOff 还原世界系高度）
    ...(armAligned ? (()=>{
      const tArm=armTcpRows.map(s=>rkToPc(s.t));
      const val=k=>armTcpRows.map(s=>(k===2&&armZOff!=null)?s.tcp[2]-armZOff:s.tcp[k]);
      const mk=(k,name,color,extra={})=>g2({x:tArm, y:val(k), name, mode:'markers',
        marker:{color,size:2.5}, yaxis:'y', xaxis:'x',
        hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`, ...extra});
      return [mk(0,'Arm TCP X','#22d3ee',{visible:'legendonly'}),
              mk(1,'Arm TCP Y','#67e8f9',{visible:'legendonly'}),
              mk(2,'Arm TCP Z','#06b6d4')];
    })() : []),
    // 视觉拍心（annotate 离线三角测量，世界系→车体系，与 Arm TCP 同口径可直接叠比）
    ...(pcRacketRows.length ? (()=>{
      const rows=pcRacketRows.map(r=>{
        const rel=relToCar(r,carAt(r.t));       // 车位姿缺失或该刻无 yaw ⇒ null，丢点
        return rel ? {t:r.t, ...rel} : null;
      }).filter(Boolean);
      const mk=(k,name,color,extra={})=>g2({x:rows.map(r=>r.t), y:rows.map(r=>r[k]), name, mode:'markers',
        marker:{color,size:3.5,symbol:'circle-open'}, yaxis:'y', xaxis:'x',
        hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`, ...extra});
      return [mk('x','Vis Racket X','#f9a8d4',{visible:'legendonly'}),
              mk('y','Vis Racket Y','#f472b6',{visible:'legendonly'}),
              mk('z','Vis Racket Z','#ec4899')];
    })() : []),
    tr(RK.estimate,'x','RK Estimate X','y','#facc15','markers',{visible:'legendonly',marker:{color:'#facc15',size:4}}),
    tr(RK.estimate,'y','RK Estimate Y','y','#fde047','markers',{visible:'legendonly',marker:{color:'#fde047',size:4}}),
    tr(RK.estimate,'z','RK Estimate Z','y','#f1c40f','markers',{visible:'legendonly',marker:{color:'#f1c40f',size:4}}),
    tr(RK.bot,'x','Bot X','y','#67e8c3'),
    tr(RK.bot,'y','Bot Y','y','#9fffce'),
    g2({x:shifted(ts(RK.bot)), y:ys(RK.bot,'yaw').map(v=>isNum(v)?v*10:null), name:'Bot Yaw x10', mode:'markers',
      customdata:ys(RK.bot,'yaw'),
      marker:{color:'#5eead4',size:2.5,symbol:'diamond'}, yaxis:'y3', xaxis:'x',
      hovertemplate:'t=%{x:.3f}s<br>Bot Yaw=%{customdata:.4f}rad<br>display=%{y:.3f}<extra>Bot Yaw x10</extra>',
      visible:'legendonly'}),
    tr(RK.bot,'target_x','Bot Target X','y','#ffd27f','markers',{visible:'legendonly'}),
    tr(RK.bot,'target_y','Bot Target Y','y','#ff9f7f','markers',{visible:'legendonly'}),
    rkRemainingTr(),
  ];
  const layout = () => ({
    ...DL,
    title:{text:'RK Move ball and car positions aligned to PC timeline',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'PC report time (s)',...GS,domain:[0,1],anchor:'y'},
    yaxis:{title:'Ball + Car XYZ/XY (m)',...GS,domain:[0,1]},
    yaxis2:{title:'Remaining (ms)',...GS,overlaying:'y',side:'right'},
    yaxis3:{title:'Yaw rad x10',...GS,overlaying:'y',side:'right',position:0.94},
  });
  const redraw = () => {
    setInfo();
    syncSignalControls();
    const jobs=[Plotly.react('c5', traceData(), layout(), PLOT_CONFIG).then(()=>tl('c5','l5'))];
    if(document.getElementById('c6') && builtPlots.has(6)){
      jobs.push(Plotly.react('c6', signalTraceData(), signalLayout(), PLOT_CONFIG).then(()=>tl('c6','l6')));
    }
    if(builtPlots.has(4) && typeof window.__rebuildArm==='function'){
      jobs.push(window.__rebuildArm());
    }
    renderRk300Table();
    return Promise.all(jobs);
  };
  if(input) input.value = rkBias.toFixed(4);
  setInfo();
  renderRk300Table();
  Plotly.newPlot('c5', traceData(), layout(), PLOT_CONFIG).then(()=>{wl('c5','l5');wz('c5');});
  const apply = document.getElementById('rkApply');
  if(apply) apply.addEventListener('click',()=>{
    const v=Number(input.value);
    rkBias=isNum(v) ? v : 0;
    redraw();
  });
  const autoBtn = document.getElementById('rkAuto');
  if(autoBtn) autoBtn.addEventListener('click',()=>{
    rkScale=isNum(auto.scale)?auto.scale:1;
    rkBias=Math.round((isNum(auto.bias)?auto.bias:0)*10000)/10000;
    if(input) input.value=rkBias.toFixed(4);
    redraw();
  });

  const signalTraceData = () => [
    tr(RK.camera_cmd,'position','Camera Cmd Pos','y','#b197fc'),
    tr(RK.camera_motor,'position','Camera Motor Pos','y','#7fd1ff'),
    tr(RK.steer_cmd,'position','Steer Cmd Pos','y2','#f59e0b'),
    tr(RK.steer_motor,'position','Steer Motor Pos','y2','#facc15'),
    tr(RK.wheels_cmd,'current_avg','Wheel Current Avg','y3','#ff7f7f'),
    tr(RK.wheels_cmd,'speed_avg','Wheel Speed Avg','y3','#67e8c3','markers',{visible:'legendonly'}),
    tr(RK.wheels_pos_diff,'value_avg','Wheel PosDiff Avg','y3','#c084fc','markers',{visible:'legendonly'}),
    tr(RK.imu,'yaw_speed','IMU Yaw Speed','y4','#94a3b8','markers',{visible:'legendonly'}),
  ];
  const signalLayout = () => ({
    ...DL,
    title:{text:'RK move signals aligned to PC timeline',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'PC report time (s)',...GS,domain:[0,1],anchor:'y4'},
    yaxis:{title:'Camera pos',...GS,domain:[0.78,1]},
    yaxis2:{title:'Steer pos',...GS,domain:[0.52,0.74]},
    yaxis3:{title:'Wheels avg',...GS,domain:[0.26,0.48]},
    yaxis4:{title:'IMU',...GS,domain:[0.0,0.22]},
  });
  const sigInput = document.getElementById('rkSigOff');
  const sigInfo = document.getElementById('rkSigInfo');
  const syncSignalControls = () => {
    if(sigInput) sigInput.value = rkBias.toFixed(4);
    if(sigInfo) sigInfo.innerHTML = info.innerHTML;
  };
  syncSignalControls();
  const sigApply = document.getElementById('rkSigApply');
  if(sigApply) sigApply.addEventListener('click',()=>{
    const v=Number(sigInput.value);
    rkBias=isNum(v) ? v : 0;
    if(input) input.value=rkBias.toFixed(4);
    redraw().then(syncSignalControls);
  });
  const sigAuto = document.getElementById('rkSigAuto');
  if(sigAuto) sigAuto.addEventListener('click',()=>{
    rkScale=isNum(auto.scale)?auto.scale:1;
    rkBias=Math.round((isNum(auto.bias)?auto.bias:0)*10000)/10000;
    if(input) input.value=rkBias.toFixed(4);
    redraw().then(syncSignalControls);
  });

  window.__buildRkSignals = () => {
    syncSignalControls();
    Plotly.newPlot('c6', signalTraceData(), signalLayout(), PLOT_CONFIG).then(()=>{wl('c6','l6');wz('c6');});
  };
};
})();

function sw(i){
  ensurePlot(i);
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',Number(t.dataset.idx)===i));
  document.querySelectorAll('.pnl').forEach(p=>p.classList.toggle('on',p.id==='p'+i));
  window.dispatchEvent(new Event('resize'));
}

function tv(trace){return trace&&trace.visible!=='legendonly'&&trace.visible!==false}
function tc(trace){
  if(!trace) return '#d7d7eb';
  if(trace.line&&typeof trace.line.color==='string') return trace.line.color;
  if(trace.marker&&typeof trace.marker.color==='string') return trace.marker.color;
  return '#d7d7eb';
}
function tl(plotId,ctrlId){
  const plot=document.getElementById(plotId);
  const ctrl=document.getElementById(ctrlId);
  if(!plot||!ctrl||!plot.data) return;
  // legendgroup 相同的 trace 合成一个按钮整组开关（Arm 图同一 joint 出现在多层 subplot）
  const groups=[], byKey={};
  plot.data.forEach((trace,idx)=>{
    const key=trace&&trace.legendgroup?`g:${trace.legendgroup}`:`t:${idx}`;
    if(byKey[key]==null){
      byKey[key]=groups.length;
      groups.push({idxs:[], name:trace&&trace.name?trace.name:`trace ${idx+1}`, trace});
    }
    groups[byKey[key]].idxs.push(idx);
  });
  ctrl.innerHTML=groups.map(g=>{
    const on=g.idxs.some(i=>tv(plot.data[i]));
    return `<button type="button" class="lb${on?'':' off'}" data-plot="${plotId}" data-indices="${g.idxs.join(',')}" aria-pressed="${on?'true':'false'}"><span class="ls" style="background:${tc(g.trace)}"></span><span>${g.name}</span></button>`;
  }).join('');
}
function wl(plotId,ctrlId){
  const plot=document.getElementById(plotId);
  const ctrl=document.getElementById(ctrlId);
  if(!plot||!ctrl) return;
  tl(plotId,ctrlId);
  // 事件委托挂在容器上：单击后按钮条会整体重建，直接绑在按钮上 dblclick 收不到
  ctrl.addEventListener('click',ev=>{
    const btn=ev.target.closest('.lb');
    if(!btn) return;
    const idxs=btn.dataset.indices.split(',').map(Number);
    const next=!idxs.some(i=>tv(plot.data[i]));
    Plotly.restyle(plotId,{visible:next?true:'legendonly'},idxs).then(()=>tl(plotId,ctrlId));
  });
  // 双击一个系列：只显示它（整组）；已是 solo 时再双击恢复全部
  ctrl.addEventListener('dblclick',ev=>{
    const btn=ev.target.closest('.lb');
    if(!btn) return;
    const idxs=new Set(btn.dataset.indices.split(',').map(Number));
    const alreadySolo=plot.data.every((t,j)=>tv(t)===idxs.has(j));
    const vis=plot.data.map((t,j)=>(alreadySolo||idxs.has(j))?true:'legendonly');
    Plotly.restyle(plotId,{visible:vis}).then(()=>tl(plotId,ctrlId));
  });
  plot.on('plotly_restyle',()=>tl(plotId,ctrlId));
}

const ZSTEP=1.35, ZMAX=200.0;
let AP=null;
const ZSTATE={};
function gp(id){return document.getElementById(id)}
function sap(id){
  AP=id;
  document.querySelectorAll('.cc').forEach(cc=>{
    const plot=cc.querySelector('.cb');
    cc.classList.toggle('zoom-active',!!plot&&plot.id===id);
  });
}
function nx(plot){
  const xs=[];
  (plot?.data||[]).forEach(trace=>{
    (trace?.x||[]).forEach(v=>{
      if(typeof v==='number'&&Number.isFinite(v)) xs.push(v);
    });
  });
  return xs;
}
function fx(id){
  const cached=ZSTATE[id]?.fullRange;
  if(cached) return [...cached];
  const plot=gp(id);
  if(!plot) return null;
  let range=null;
  const axis=plot._fullLayout&&plot._fullLayout.xaxis;
  if(!axis) return null;
  if(axis&&Array.isArray(axis.range)&&axis.range.length===2){
    const a=Number(axis.range[0]), b=Number(axis.range[1]);
    if(Number.isFinite(a)&&Number.isFinite(b)&&b>a) range=[a,b];
  }
  if(!range){
    const xs=nx(plot);
    if(!xs.length) return null;
    range=[Math.min(...xs),Math.max(...xs)];
  }
  ZSTATE[id]={...(ZSTATE[id]||{}),fullRange:range};
  return [...range];
}
function cx(id){
  const plot=gp(id);
  if(!plot) return null;
  const axis=plot._fullLayout&&plot._fullLayout.xaxis;
  if(axis&&Array.isArray(axis.range)&&axis.range.length===2){
    const a=Number(axis.range[0]), b=Number(axis.range[1]);
    if(Number.isFinite(a)&&Number.isFinite(b)&&b>a) return [a,b];
  }
  return fx(id);
}
function qx(range,fullRange){
  const f0=fullRange[0], f1=fullRange[1], fs=f1-f0;
  if(!(fs>0)) return [f0,f1];
  let a=Number(range[0]), b=Number(range[1]);
  let span=b-a;
  const minSpan=Math.max(fs/ZMAX,1e-6);
  if(!(span>0)) span=minSpan;
  span=Math.max(minSpan,Math.min(fs,span));
  const center=(a+b)/2;
  a=center-span/2;
  b=center+span/2;
  if(a<f0){b+=f0-a;a=f0;}
  if(b>f1){a-=b-f1;b=f1;}
  if(a<f0)a=f0;
  if(b>f1)b=f1;
  return [a,b];
}
function ux(id){
  const full=fx(id), cur=cx(id), readout=document.getElementById(`${id}r`);
  if(!full||!cur){
    if(readout) readout.textContent='n/a';
    return;
  }
  const factor=Math.max(1,(full[1]-full[0])/Math.max(1e-9,cur[1]-cur[0]));
  if(readout) readout.textContent=`${factor.toFixed(2)}x`;
  document.querySelectorAll(`.zb[data-plot="${id}"]`).forEach(btn=>{
    btn.classList.toggle('on',btn.dataset.action==='reset'&&Math.abs(factor-1)<1e-3);
  });
}
function rx(id,range){
  return Plotly.relayout(id,{'xaxis.range':range,'xaxis.autorange':false}).then(()=>ux(id));
}
function mx(id,event){
  const plot=gp(id);
  const axis=plot?._fullLayout?.xaxis;
  const cur=cx(id);
  if(!plot||!axis||!cur) return null;
  const rect=plot.getBoundingClientRect();
  const axisOffset=Number(axis._offset), axisLength=Number(axis._length);
  if(!Number.isFinite(axisOffset)||!Number.isFinite(axisLength)||!(axisLength>0)) return null;
  const rawPixel=Number(event.clientX)-rect.left-axisOffset;
  const pixel=Math.max(0,Math.min(axisLength,rawPixel));
  if(typeof axis.p2l==='function'){
    const converted=Number(axis.p2l(pixel));
    if(Number.isFinite(converted)) return converted;
  }
  const ratio=pixel/axisLength;
  return cur[0]+ratio*(cur[1]-cur[0]);
}
function zx(id,spanFactor,centerX=null){
  const full=fx(id), cur=cx(id);
  if(!full||!cur) return Promise.resolve();
  const center=(typeof centerX==='number'&&Number.isFinite(centerX))?centerX:((cur[0]+cur[1])/2);
  const next=qx([center-((cur[1]-cur[0])*spanFactor)/2,center+((cur[1]-cur[0])*spanFactor)/2],full);
  return rx(id,next);
}
function zxReset(id){
  const full=fx(id);
  if(!full) return Promise.resolve();
  return rx(id,full);
}
function wz(id){
  const plot=gp(id);
  if(!plot) return;
  fx(id);
  ux(id);
  plot.addEventListener('pointerdown',()=>sap(id));
  plot.addEventListener('wheel',event=>{
    if(AP!==id) return;
    if(event.ctrlKey||event.metaKey) return;
    if(!fx(id)) return;
    event.preventDefault();
    zx(id,event.deltaY<0?(1/ZSTEP):ZSTEP,mx(id,event));
  },{passive:false});
  plot.on('plotly_relayout',()=>ux(id));
}
document.querySelectorAll('.zb[data-plot]').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const id=btn.dataset.plot;
    sap(id);
    if(btn.dataset.action==='in') zx(id,1/ZSTEP);
    else if(btn.dataset.action==='out') zx(id,ZSTEP);
    else zxReset(id);
  });
});
sw(window.__hasRK ? 5 : 0);
sap(window.__hasRK ? 'c5' : 'c0');
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="curve3_output/curve3_result.json")
    parser.add_argument("--racket-json", default=None)
    parser.add_argument(
        "--arm-json", default=None,
        help="extract_arm_bag.py 输出的机械臂 JSON；缺省时自动探测 <input>_arm.json",
    )
    parser.add_argument("--rk-tracking-json", default=None)
    parser.add_argument(
        "--rk-time-bias", type=float, default=None,
        help="预置仿射时间映射的 bias（秒）；scale 仍由共享小车位姿自动估计。"
             "页面 Auto align 按钮可恢复自动 bias。",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = os.path.splitext(args.input)[0]
    out = args.output or (base + ".html")
    arm_json = args.arm_json
    if arm_json is None:
        candidate = base + "_arm.json"
        if os.path.exists(candidate):
            arm_json = candidate
    rk_tracking_json = args.rk_tracking_json
    if rk_tracking_json is None:
        candidate = base + "_rk_tracking.json"
        if os.path.exists(candidate):
            rk_tracking_json = candidate
    generate_html(
        args.input,
        out,
        args.racket_json,
        arm_json,
        rk_tracking_json,
        args.rk_time_bias,
    )


if __name__ == "__main__":
    main()
