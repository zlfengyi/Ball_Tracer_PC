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


def generate_html(
    input_path: str,
    output_path: str,
    racket_json_path: str | None = None,
    arm_json_path: str | None = None,
    rk_tracking_json_path: str | None = None,
    rk_offset: float | None = None,
) -> None:
    data = _load_json(input_path)
    if racket_json_path:
        data = _merge_racket_json(data, _load_json(racket_json_path), racket_json_path)
    if arm_json_path:
        data["arm"] = _load_json(arm_json_path)
    if rk_tracking_json_path:
        data["rk_tracking"] = _load_json(rk_tracking_json_path)
    if rk_offset is not None:
        data["rk_offset_preset"] = rk_offset
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
.rkCtl{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 10px;font-size:12px;color:#a0a0c0}
.rkCtl input{width:92px;border:1px solid #0f3460;background:#16213e;color:#fff;border-radius:4px;padding:4px 6px;font:inherit}
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
    <div class="rkCtl"><span>RK offset(s)</span><input id="rkOff" type="number" step="0.0001" value="0"><button type="button" class="zb" id="rkApply">Apply</button><button type="button" class="zb" id="rkAuto">Auto align</button><span id="rkInfo"></span></div>
    <div class="armEv" id="rk300Tbl" style="padding:0 0 4px"></div>
    <div class="lc" id="l5"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c5" data-action="out">X-</button><button type="button" class="zb on" data-plot="c5" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c5" data-action="in">X+</button><span id="c5r" class="zr">1.00x</span></div><div class="zx"><div id="c5" class="cb"></div></div>
  </div>
</div>
<div id="p6" class="pnl">
  <div class="cc">
    <div class="rkCtl"><span>RK offset(s)</span><input id="rkSigOff" type="number" step="0.0001" value="0"><button type="button" class="zb" id="rkSigApply">Apply</button><button type="button" class="zb" id="rkSigAuto">Auto align</button><span id="rkSigInfo"></span></div>
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
const s1 = s1Full;
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
  stat('S1 preds', s1Full.length),
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
const pcCarRows = car.map(c=>({t:isNum(c.elapsed_s)?c.elapsed_s:relTime(c.t), x:c.x, y:c.y, z:c.z, yaw:c.yaw}))
  .filter(p=>isNum(p.t))
  .sort((a,b)=>a.t-b.t);
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
const pcFlights = (()=>{
  const groups=[];
  let rows=[];
  const flush=()=>{
    if(rows.length>=5 && rows[rows.length-1].t-rows[0].t>=0.2){
      const zs=rows.map(row=>row.z).filter(isNum);
      if(zs.length>=5 && Math.max(...zs)-Math.min(...zs)>=0.25) groups.push(rows);
    }
    rows=[];
  };
  for(const row of pcRows){
    if(rows.length && row.t-rows[rows.length-1].t>0.5) flush();
    rows.push(row);
  }
  flush();
  return groups;
})();
// Align only the ball height shape. Remove the median z bias at every candidate
// per visible flight, then weight flights equally so a long track cannot dominate.
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
const scoreOffset = (off, full=false) => {
  if(rkMovZ.length<30 || pcFlights.length<3) return null;
  const step=full ? 1 : Math.max(1,Math.floor(rkMovZ.length/400));
  const matches=pcFlights.map(()=>({dz:[], first:null, last:null}));
  let flightIdx=0;
  for(let i=0;i<rkMovZ.length;i+=step){
    const t=rkMovZ[i].t+off;
    while(flightIdx<pcFlights.length && t>pcFlights[flightIdx][pcFlights[flightIdx].length-1].t) flightIdx+=1;
    if(flightIdx>=pcFlights.length) break;
    const flight=pcFlights[flightIdx];
    if(t<flight[0].t) continue;
    const v=interpPcVal(flight, t, 'z', 0.08);
    if(v==null) continue;
    const match=matches[flightIdx];
    match.dz.push(v-rkMovZ[i].v);
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
  if(flightErrs.length<3) return null;
  return {err:median(flightErrs), n, flights:flightErrs.length};
};
// 混叠防护（0712 晚场教训）：抛球间隔相似时，轨迹匹配错位整数个抛也能出深谷；
// 且当 PC 观测只覆盖部分时段时，假谷只用到少量重叠样本反而 err 更低。
// Guard by matched flight count rather than raw frame count, then minimize z-shape error.
// 扫描界由数据重叠推导（0716 上午场教训：RK 节点晚入网 70s+，t0=全 bag 最小 payload
// 把 RK 相对轴锚偏，真实 offset=+62s 溢出旧 ±60s 硬编码界，自动对齐落假谷、全报告错轴）：
// off 须使 RK 运动段映射后与 PC 观测段有交集 → off ∈ [pc首-rk末, pc末-rk首]。
// 粗扫步长自适应封顶 ~18k 次（谷宽由插值门限 0.08s 定，粗步 ≤0.1s 仍稳落谷内），谷邻域再细扫。
const estimateOffset = () => {
  if(rkMovZ.length<30 || pcFlights.length<3) return {off:0, err:null, n:0};
  const lo=Math.floor(pcRows[0].t - rkMovZ[rkMovZ.length-1].t) - 1;
  const hi=Math.ceil(pcRows[pcRows.length-1].t - rkMovZ[0].t) + 1;
  const coarse=Math.max(0.02, (hi-lo)/18000);
  const cands=[];
  for(let off=lo; off<=hi+1e-4; off+=coarse){
    const s=scoreOffset(off);
    if(s) cands.push({off, ...s});
  }
  if(!cands.length) return {off:0, err:null, n:0};
  const flightMax=cands.reduce((m,c)=>Math.max(m,c.flights),0);
  const minFlights=Math.max(3,Math.ceil(0.6*flightMax));
  const ok=cands.filter(c=>c.flights>=minFlights);
  let best=ok.reduce((a,b)=>b.err<a.err?b:a, ok[0]);
  const win=coarse*2.5;
  for(let off=best.off-win; off<=best.off+win+1e-4; off+=0.0002){
    const s=scoreOffset(off,true);
    if(s && s.flights>=minFlights && s.err<best.err) best={off, ...s};
  }
  return best;
};
// [[align-core-end]]
const auto = RK ? estimateOffset() : {off:0, err:null, n:0};
// 对齐取证钩子（0717 定型）：控制台 __dbgAlign.scoreOffset(off) 可复查任意 offset 的
// z-shape match quality after removing the fixed height bias at each offset.
window.__dbgAlign={pcRows:()=>pcRows, rkMovZ:()=>rkMovZ, scoreOffset, auto:()=>auto};
// 预置 offset（--rk-offset）：PC/RK 无共同球观测窗的场次（轨迹自动对齐结构性不可行）
// 由外部锚（如车速互相关）离线求出后喂入，优先于 auto；Auto align 按钮仍可覆盖。
const presetOff = isNum(D.rk_offset_preset) ? Number(D.rk_offset_preset) : null;
let rkOffset = Math.round((presetOff!=null ? presetOff : auto.off)*10000)/10000;
window.__rkOffset = rkOffset;
// 对齐质量门（0716 早场教训：自动对齐落假谷时报告安静生成，全表错轴像"数据坏了"）：
// Good z-shape matches are below about 0.08m; n is the participating sample count.
// 有 preset 时不告警（正是自动对齐不可行场次的正解，rkOffset 已取 preset）。
const alignBad = !!RK && presetOff==null && (auto.err==null || auto.err>0.08 || auto.n<30);
const alignWarnHtml = alignBad
  ? `<div style="border:1px solid #e94560;background:rgba(233,69,96,0.12);color:#e94560;font-weight:600;border-radius:8px;padding:8px 12px;margin:0 0 10px">`+
    `⚠ PC↔RK 自动对齐不可信（中位 z 形状误差 ${auto.err==null?'n/a':auto.err.toFixed(3)+'m'} / ${auto.n} 点；门限 ≤0.08m 且 ≥30 点）`+
    `——本页所有跨轴内容（北极星表 / RK 轨迹叠加 / Arm 对齐）不可靠。`+
    `常见原因：两侧共同球观测太少（热身场、RK 晚入网、PC 未见球）。`+
    `处置：RK 页手动 Apply offset，或用 --rk-offset 预置外部锚。</div>`
  : (RK
    // 对齐质量常显（0717 凌晨场教训：RK 预测 ht 对多弹跳球偏早 0.3~0.6s，Δt 一大用户
    // 第一反应是"轴又没对齐"——把对齐证据亮在表上方，可信时表内偏差就该信为预测误差）。
    ? `<div style="font-size:11px;color:#7fbf9f;margin:0 0 6px">PC↔RK 对齐 ✓ offset ${rkOffset.toFixed(4)}s`+
      (presetOff!=null
        ? `（--rk-offset 外部锚预置）`
        : `（两侧球 z 形状互验：中位误差 ${auto.err.toFixed(3)}m / ${auto.n} 点，门限 ≤0.08m）`)+
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
  const relx=ys(RK.pred,'rel_x'), rely=ys(RK.pred,'rel_y'), relz=ys(RK.pred,'rel_z');
  const out=[];
  for(let i=0;i<t.length;i++){
    const ti=Number(t[i]);
    if(!isNum(ti) || !isNum(ht[i])) continue;
    const cur=out[out.length-1];
    const upd = th => {
      th.ht=ht[i]; th.lastT=ti; th.msgs=(th.msgs||0)+1;
      if(isNum(relx[i])&&isNum(rely[i])&&isNum(relz[i])){
        th.stage=rkPredStage[i]; th.rel_x=relx[i]; th.rel_y=rely[i]; th.rel_z=relz[i];
        th.lastRelIdx=i; th.refT=ti;  // 最终 ref 定格的计算时刻（该消息的 ct，RK 相对轴）
        const lead=ht[i]-ti;
        if(lead>0 && Number(rkPredStage[i])===1){
          const dev=Math.abs(lead-REF_LEAD_TARGET);
          if(th.ref300T==null || dev<th.ref300LeadDev){
            th.ref300Stage=rkPredStage[i]; th.ref300X=relx[i]; th.ref300Y=rely[i]; th.ref300Z=relz[i];
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
                ref300Lead:null, ref300LeadDev:null, ref300NFit:null, ref300Idx:null};
      upd(th);
      out.push(th);
    }
  }
  return out;
})();
// [[rk300-contract-core-end]]
const carAt = t => {
  const s=interpRow(pcCarRows,t,0.5);
  if(s && isNum(s.a.yaw) && isNum(s.b.yaw)){
    const dyaw=Math.atan2(Math.sin(s.b.yaw-s.a.yaw),Math.cos(s.b.yaw-s.a.yaw));
    return {x:lerp(s.a.x,s.b.x,s.f), y:lerp(s.a.y,s.b.y,s.f), yaw:s.a.yaw+dyaw*s.f};
  }
  const n=nearest(pcCarRows,t);
  return (n && Math.abs(n.t-t)<=0.3 && isNum(n.yaw)) ? {x:n.x, y:n.y, yaw:n.yaw} : null;
};
const relToCar = (b,c) => {
  const dx=b.x-c.x, dy=b.y-c.y, cy=Math.cos(c.yaw), sy=Math.sin(c.yaw);
  return {x:cy*dx+sy*dy, y:-sy*dx+cy*dy, z:b.z};
};
// PC 真值下"球与车在 y 向相会"的时刻（PC 报告轴）：ht 附近 ±0.6s 内
// rel_y 首次由正下穿 0 的线性插值。只取第一次下行穿越 → 击打后反弹（y 反向）
// 不会污染；要求来球速度 >1 m/s、穿越两侧观测间隔 ≤0.3s（遮挡断档过大不给）。
// cutHi（命中行传触球时刻）：穿越两点必须全在触球 20ms 之前——触球邻域的观测
// 可能已被击打改道（与 PC真值入弧口径一致），近触球穿越一律交给外推降级。
const truthMeetAt = (tPcApprox, cutHi) => {
  const lo=tPcApprox-0.6, hi=Math.min(tPcApprox+0.6, cutHi!=null?cutHi-0.02:Infinity);
  let prev=null;
  for(const p of pcRows){
    if(p.t<lo) continue;
    if(p.t>hi) break;
    const c=carAt(p.t);
    if(!c){ prev=null; continue; }
    const ry=relToCar(p,c).y;
    if(prev && prev.ry>0 && ry<=0){
      const dt=p.t-prev.t;
      if(dt>0 && dt<=0.3 && (prev.ry-ry)/dt>1.0){
        return prev.t + dt*prev.ry/(prev.ry-ry);
      }
      return null;  // 穿越处断档过大/速度异常，不硬给
    }
    prev={t:p.t, ry};
  }
  return null;
};
// pc相会t 外推降级（0716 用户定稿）：命中行球常在触球前 2~16cm 被拍截住打回（0716 早场 6/10 行
// 实测 minRy +0.022~+0.159m），rel_y 从未过 0——"相会"物理上没发生，直接插值必然给不出；
// 穿越处断档 >0.3s 时同样缺失。此时用触球/断档前最后一段逼近观测（ry>0、间隔 ≤0.2s，
// 取尾部 ≤8 点、≥3 点）对 rel_y(t) 线性最小二乘，解 ry=0 得"球本来会到达的时刻"：
// 外推跨度 ≤0.35s 才给（来球 >1 m/s 时对应 <35cm，单弧内 y 近匀速，线性外推安全），
// 黄标"±Mms(外推Nms)"注明 1σ 时刻误差与外推跨度。cutT = 触球时刻（命中行防击打后反弹
// 观测混入外推段；无触球传 null）。
const fitMeetAt = (tPcApprox, cutT) => {
  const lo=tPcApprox-0.6, hi=Math.min(tPcApprox+0.6, cutT!=null?cutT-0.02:Infinity);
  let run=[];
  for(const p of pcRows){
    if(p.t<lo) continue;
    if(p.t>hi) break;
    const c=carAt(p.t);
    if(!c) continue;
    const ry=relToCar(p,c).y;
    if(ry<=0) break;                                   // 逼近段到穿越为止（直接插值已判过失败）
    if(run.length && p.t-run[run.length-1].t>0.2) run=[];
    run.push({t:p.t, ry});
  }
  if(run.length<3) return null;
  const seg=run.slice(-8);
  const n=seg.length, t0=seg[0].t;
  let st=0,sv=0,stt=0,stv=0;
  seg.forEach(q=>{ const u=q.t-t0; st+=u; sv+=q.ry; stt+=u*u; stv+=u*q.ry; });
  const den=n*stt-st*st;
  if(Math.abs(den)<1e-9) return null;
  const b=(n*stv-st*sv)/den, a=(sv*stt-st*stv)/den;    // ry ≈ a + b·(t−t0)
  if(b>=-1.0) return null;                             // 与直接判同门限：来球 y 速 >1 m/s
  const tMeet=t0-a/b;
  const last=seg[n-1], ext=tMeet-last.t;
  if(ext<-0.02 || ext>0.35) return null;
  // 1σ 时刻误差（与 pcTruthAt 同口径，除以来球速折成时间）：
  // LS 均线在穿越点的预测标准误 ⊕ 未建模 y 减速（气动 ~1.5 m/s²）×外推跨度²/2，
  // 下限取 max|残差|（对外宣称不低于模型对数据的实际解释能力）。实测量级 ±2~8ms。
  const mean=st/n, sxx=stt-st*st/n;
  let ss=0, rmax=0;
  seg.forEach(q=>{ const r=a+b*(q.t-t0)-q.ry; ss+=r*r; rmax=Math.max(rmax,Math.abs(r)); });
  const uStar=tMeet-t0;
  const sePred=Math.sqrt(ss/Math.max(1,n-2)*(1/n+(uStar-mean)*(uStar-mean)/sxx));
  const eModel=0.75*ext*ext;                           // ½·(~1.5 m/s²)·ext²（m）
  const errT=Math.max(Math.sqrt(sePred*sePred+eModel*eModel), rmax)/(-b);
  return {t:tMeet, ext:Math.max(ext,0), err:errT, lastRy:last.ry, n};
};
// pc相会t 统一入口：先直接插值（无标），缺了走外推（黄标，悬停给依据）。
// cutDirect=true（命中行）：直接穿越也截到触球前——两点有一点落在触球邻域就不算直接，
// 走外推；false/缺省（未挥拍/降级）：直接穿越不截（球未被碰时触球后观测干净合法）。
const meetAt = (tPcApprox, cutT, cutDirect) => {
  const direct=truthMeetAt(tPcApprox, cutDirect?cutT:null);
  if(direct!=null) return {t:direct, tag:''};
  const f=fitMeetAt(tPcApprox, cutT);
  if(!f) return null;
  return {t:f.t, tag:` <span style="color:#fbbf24" title="外推相会：真值球在触球/断档前未过 rel_y=0（最后观测距车 y 向还有 ${Math.round(f.lastRy*1000)}mm），由此前 ${f.n} 点 rel_y 线性外推 ${Math.round(f.ext*1000)}ms 得到的到达时刻；± 为 1σ 时刻误差 =（LS 预测标准误 ⊕ 未建模气动减速×跨度²/2，下限 max|残差|）÷ 来球速">±${Math.max(1,Math.round(f.err*1000))}ms(外推${Math.round(f.ext*1000)}ms)</span>`};
};
// PC 坐标真值统一入口：只使用目标时刻 20ms 前的入弧观测拟合，不再跨目标时刻插值。
// 这样不依赖“y 是否反向”识别触球，擦拍、拍框接触或触球附近丢帧都不会把出弧混进入弧。
// x/y 线性、z 固定重力抛物线；至少 5 点、外推 ≤0.35s，并以 x/z 最大残差门控。
// 拟合失败返回 null，报告显示缺失，不退回可能跨越速度冲量的线性插值。
// 返回车体系 {x,y,z,gap,err,resMax,dNear}，z 为世界高度；gap/dNear 均为距最后入弧观测的时距。
// [[pc-truth-core-begin]]
const pcTruthAt = tPc => {
  const c=carAt(tPc);
  if(!c) return null;
  let win=pcRows.filter(p=>tPc-p.t>=0.02 && tPc-p.t<=0.75);
  let loT=-Infinity;
  win.forEach(p=>{
    if(p.z<0.12 && p.t>loT) loT=p.t;
  });
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
    const fz=reg(ts,win.map((p,i)=>p.z+4.905*ts[i]*ts[i]));
    if(!fx||!fy||!fz) return null;
    const zAt=t=>fz[0]+fz[1]*t-4.905*t*t;
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
      const lever=1/n+mean*mean/sxx;   // LS 在 τ=0 处的预测方差杠杆
      const se=res=>{
        let ss=0; res.forEach(r=>ss+=r*r);
        return Math.sqrt(ss/Math.max(1,n-2)*lever);
      };
      const seX=se(win.map((p,i)=>fx[0]+fx[1]*ts[i]-p.x));
      const seZ=se(win.map((p,i)=>zAt(ts[i])-p.z));
      const dNear=gap;
      const eModel=0.75*dNear*dNear;   // ½·(~1.5 m/s² 未建模气动)·d²
      const seMax=Math.max(seX,seZ);
      const resMax=Math.max(zMax,xMax);
      // 显示误差下限取 max|残差|：统计上均值估计可优于单点散布，但对外宣称
      // 不应低于模型对数据的实际解释能力（防"跨 0.3s 断档标 ±1mm"式过度自信）
      const err=Math.max(Math.sqrt(seMax*seMax+eModel*eModel), resMax);
      return {...relToCar({x:fx[0],y:fy[0],z:zAt(0)},c), gap, err, resMax, dNear};
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
// ---- 臂侧共享数据 ----
// v3 起臂数据与 RK 数据同钟（RK 单调钟绝对秒，extract_arm_bag.py 里由
// header.stamp 经常数换算）：减 RK.t0 落到 RK 相对轴后，与 RK 轨迹/预测
// 完全同轴 —— 全项目只有 PC/RK 两个时间轴，显示统一 + rkOffset 一个桥。
// v2 旧文件（bag 接收轴）不再桥接：显示未对齐提示，重跑 extract_arm_bag.py 即可。
const armAligned = !!(ARM && ARM.time_axis==='rk_mono_abs_s' && RK && isNum(RK.t0));
if(armAligned){
  ['states','commands','events'].forEach(k=>{
    (ARM[k]||[]).forEach(r=>{ if(isNum(r.t)) r.t -= RK.t0; });
  });
}
const armDispOff = () => armAligned ? rkOffset : 0;
// 击打时刻：accepted hit 状态自带 duration/hit_time，按 receive_hit 的公式
// start_hit = accept_t + duration − hit_time，finish_hit = accept_t + duration。
// 同一抛会连续 accept 多条更新：聚类为一组，cmd 取第一条（臂开始动作），
// start/done/目标取最后一条（最终执行的时序与目标）。
// 每条 accepted 用 x、机械臂当前 z 补偿和接触 HT 回配实际消费的 /predict_hit_pos，
// 并统计臂端 z 偏移 zOff = accepted_z − rel_z。
// [[arm-prediction-match-core-begin]]
const ARM_HIT_Z_OFFSET=-0.153;
const armPredictionMatchesAccepted = (p,acceptT,acceptX,acceptZ,acceptDuration) => {
  const zOffset=p.relSrc==='panel'?0:ARM_HIT_Z_OFFSET;
  return Math.abs(p.rel_x-acceptX)<5e-4 &&
    Math.abs(p.rel_z+zOffset-acceptZ)<5e-4 &&
    Math.abs(p.ht-(acceptT+acceptDuration))<5e-3;
};
// [[arm-prediction-match-core-end]]
const armPreds = (()=>{
  if(!ARM) return [];
  const out=[];
  (ARM.events||[]).forEach(e=>{
    if(e.topic!=='/predict_hit_pos') return;
    try{
      const p=JSON.parse(e.text);
      out.push({t:e.t, rel_x:Number(p.rel_x), rel_y:Number(p.rel_y), rel_z:Number(p.rel_z),
                duration:Number(p.duration), ht:Number(p.ht), ct:Number(p.ct), stage:Number(p.stage),
                nFit:Number(p.n_bounce_fit), relSrc:String(p.rel_src||'?')});
    }catch(err){}
  });
  return out;
})();
const _armHit = (()=>{
  if(!ARM) return {marks:[], zOff:null};
  const out=[], zOffs=[];
  (ARM.events||[]).forEach(e=>{
    if(e.topic!=='/tennis/status') return;
    let rec=null;
    let m=/^accepted hit x=([\-0-9.]+) z=([\-0-9.]+) duration=([0-9.]+)(?:\s+hit_time=([0-9.]+))?/.exec(e.text);
    if(m){
      const dur=Number(m[3]), hitT=m[4]!=null?Number(m[4]):0.4;
      rec={cmd:e.t, lastAcceptT:e.t, tx:Number(m[1]), tz:Number(m[2]), start:e.t+dur-hitT, done:e.t+dur,
           label:'hit', n:1,
           wx:null, wy:null, wz:null, wct:null, wht:null, wpredT:null, wstage:null, wnFit:null};
      for(let i=armPreds.length-1;i>=0;i--){
        const p=armPreds[i];
        if(p.t>e.t) continue;
        if(e.t-p.t>0.25) break;
        if(armPredictionMatchesAccepted(p,e.t+RK.t0,rec.tx,rec.tz,dur)){
          rec.wx=p.rel_x; rec.wy=p.rel_y; rec.wz=p.rel_z; rec.wct=p.ct; rec.wht=p.ht;
          rec.wpredT=p.t; rec.wstage=p.stage; rec.wnFit=p.nFit;
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
      out[out.length-1]=rec;
    } else {
      out.push(rec);
    }
  });
  zOffs.sort((a,b)=>a-b);
  return {marks:out, zOff:zOffs.length?zOffs[zOffs.length>>1]:null};
})();
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
const pcRacketRelRows = pcRacketRows.map(r=>{
  const c=carAt(r.t);
  return c ? {t:r.t, ...relToCar(r,c)} : null;
}).filter(Boolean);
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
const visualRacketAt = t => fitVisualRacketRows(pcRacketRelRows,t);
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
// RK≈300ms 主表与机械臂最后 accepted 分表：两个合同独立取值、独立对齐 PC 真值。
const reportThrows = rkThrows.filter(t=>(t.msgs||0)>=3).sort((a,b)=>a.ht-b.ht);
const tableFmt = (v,d) => isNum(v) ? Number(v).toFixed(d) : '—';
const tableXz = (x,z,d=4) => (isNum(x)&&isNum(z)) ? Number(x).toFixed(d)+'/'+Number(z).toFixed(d) : '—';
const tableSigned = v => isNum(v) ? (v>=0?'+':'')+Number(v).toFixed(1) : '—';
const tableXyz = (x,y,z,d=4) => (isNum(x)&&isNum(y)&&isNum(z))
  ? Number(x).toFixed(d)+'/'+Number(y).toFixed(d)+'/'+Number(z).toFixed(d) : tableFmt(null,d);
const tableEsc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const visualRacketCell = fit => {
  if(!fit) return '—';
  const errMm=Math.max(1,Math.round(fit.err*1000));
  const mode=fit.mode==='interpolation'?'双侧二次拟合':'单侧速度外推';
  const badge=fit.mode==='interpolation'?'拟合':'外推'+Math.round(fit.dNear*1000)+'ms';
  const title=mode+'；vx='+fit.vx.toFixed(3)+'m/s，vz='+fit.vz.toFixed(3)+'m/s；样本='+fit.n+
    '，跨度='+Math.round(fit.span*1000)+'ms，最近观测='+Math.round(fit.dNear*1000)+'ms；误差含拟合残差/95%回归误差'+
    (fit.mode==='extrapolation'?'及20m/s²未建模加速度上界':'');
  return tableXz(fit.x,fit.z)+' <span style="color:#fbbf24" title="'+tableEsc(title)+'">±'+errMm+'mm('+badge+')</span>';
};
const pcTruthCell = f => {
  if(!f) return '—';
  return tableXz(f.x,f.z,4)+
    ' <span style="color:#fbbf24" title="入弧拟合真值：x/y 线性、z 固定重力；只用目标时刻20ms前观测，不跨目标时刻插值；max|残差| '+
    Math.round(f.resMax*1000)+'mm，rel_y='+(f.y>=0?'+':'')+f.y.toFixed(3)+'m">±'+
    Math.max(1,Math.round(f.err*1000))+'mm(外推'+Math.round(f.dNear*1000)+'ms)</span>';
};
const pcTruthNote = 'PC真值严格取本栏HT时刻，并统一由此前入弧拟合；至少5点、外推不超过350ms，拟合不足显示—，不回退插值';

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
    const accHt=accepted&&isNum(accepted.wht)?accepted.wht-RK.t0:null;
    const tcp=accHt!=null?tcpAt(accHt):null;
    const acceptedTarget=accepted?tableXz(accepted.wx,accepted.wz):'—';
    const tcpWorld=tcp?[tcp[0],tcp[1],tcp[2]-(isNum(armZOff)?armZOff:0)]:null;
    const tcpCell=tcpWorld?tableXyz(tcpWorld[0],tcpWorld[1],tcpWorld[2]):tableFmt(null,4);
    const tcpAcceptedDx=accepted&&tcpWorld&&isNum(accepted.wx)?(tcpWorld[0]-accepted.wx)*100:null;
    const tcpAcceptedDz=accepted&&tcpWorld&&isNum(accepted.wz)?(tcpWorld[2]-accepted.wz)*100:null;
    const tcpAcceptedError=tableSigned(tcpAcceptedDx)+'/'+tableSigned(tcpAcceptedDz);
    const visualRacket=accHt!=null?visualRacketAt(accHt+rkOffset):null;
    const visualRacketValue=visualRacketCell(visualRacket);
    const rejectNote=accepted?'—':rejectNoteForThrow(th);
    if(!isNum(th.ref300T)||!isNum(th.ref300Ht)||!isNum(th.ref300X)||!isNum(th.ref300Z)){
      return '<tr><td>'+(idx+1)+'</td><td>—</td><td>—</td><td>—</td><td>—</td><td>'+acceptedTarget+'</td>'+
        '<td>—</td><td>'+tcpCell+'</td><td>'+tcpAcceptedError+'</td><td>'+visualRacketValue+'</td>'+
        '<td>—</td><td>—</td><td>—</td>'+
        '<td><span style="color:#fbbf24">无S1@300ms</span> / msgs='+(th.msgs||0)+'</td><td>'+rejectNote+'</td></tr>';
    }
    const ctPc=th.ref300T+rkOffset;
    const htPc=th.ref300Ht+rkOffset;
    const truth=pcTruthAt(htPc);
    const meet=meetAt(htPc,htPc,true);
    const dx=truth?(th.ref300X-truth.x)*1000:null;
    const dz=truth?(th.ref300Z-truth.z)*1000:null;
    const dt=meet?(htPc-meet.t)*1000:null;
    const info='S'+th.ref300Stage+
      (isNum(th.ref300NFit)?' n_fit='+th.ref300NFit:'')+
      ' / msgs='+(th.msgs||0);
    return '<tr><td>'+(idx+1)+'</td>'+
      '<td>'+ctPc.toFixed(3)+'</td>'+
      '<td>'+htPc.toFixed(3)+'</td>'+
      '<td>'+(th.ref300Lead*1000).toFixed(1)+'</td>'+
      '<td>'+tableXz(th.ref300X,th.ref300Z)+'</td>'+
      '<td>'+acceptedTarget+'</td>'+
      '<td>'+pcTruthCell(truth)+'</td>'+
      '<td>'+tcpCell+'</td>'+
      '<td>'+tcpAcceptedError+'</td>'+
      '<td>'+visualRacketValue+'</td>'+
      '<td>'+tableSigned(dx)+'/'+tableSigned(dz)+'</td>'+
      '<td>'+(meet?meet.t.toFixed(3)+meet.tag:'—')+'</td>'+
      '<td>'+tableSigned(dt)+'</td>'+
      '<td>'+info+'</td><td>'+rejectNote+'</td></tr>';
  });
  const notes=[
    '每行RK ct、HT、x/z、n_fit严格来自同一条S1消息：在该抛所有正提前量S1中选择 |(HT−ct)−300ms| 最小者',
    'RK≈300ms字段不回退到最终RK消息；机械臂两列独立回配该抛最后一条accepted原消息',
    '机械臂目标取该accepted消息的原始RK rel_x/rel_z；TCP取该消息自身HT时刻的FK state插值，z按机械臂偏移还原世界系；无accepted或HT无state覆盖时显示—',
    '视觉球拍取同一accepted HT加rkOffset后的PC时刻：每个观测先减小车中心并按yaw转到车体系，z保留世界高度；双侧观测用局部二次位置/速度拟合，遮挡时用单侧速度外推；黄色误差含拟合残差、95%回归误差，外推另含20m/s²未建模加速度上界',
    '备注仅在该抛没有机械臂accepted时显示回配到该抛时间窗的reject hit原因，并合并同类原因计数',
    'ct/HT已加同一个RK offset显示在PC报告轴；lead=HT−ct，不受offset影响',
    'RK−PC dx/dz使用PC拟合真值@本行HT；HT−相会为预测击球时刻与PC球车相会时刻之差',
    pcTruthNote,
  ];
  return '<div class="armTblWrap"><table class="armTbl"><thead><tr>'+
    '<th>#</th><th>RK ct@≈300ms<br>(s,PC轴)</th><th>RK HT@≈300ms<br>(s,PC轴)</th><th>lead(ms)</th>'+
    '<th>RK@≈300ms x/z(m)</th><th>机械臂最后accepted目标 x/z(m)</th>'+
    '<th>PC真值@RK HT x/z(m)</th>'+
    '<th>TCP@accepted HT x/y/z(m)</th><th>TCP−accepted dx/dz(cm)</th>'+
    '<th>视觉球拍@accepted HT x/z(m)<br>(相对小车)</th><th>RK−PC dx/dz(mm)</th>'+
    '<th>PC相会t(s)</th><th>HT−相会(ms)</th><th>消息</th><th>备注</th></tr></thead>'+
    '<tbody>'+rows.join('')+'</tbody></table>'+
    '<div style="font-size:11px;color:#a0a0c0;margin:2px 0 6px">'+notes.join('；')+'。</div></div>';
};

const armAcceptedTableHtml = () => {
  if(!ARM) return '';
  if(!armAligned) return '<div style="color:#f87171">机械臂数据未与RK单调钟对齐，无法可靠回配最后accepted原消息。</div>';
  const hits=armHitMarks.filter(h=>h.label==='hit');
  if(!hits.length) return '<div style="color:#a0a0c0">本场没有 accepted hit。</div>';
  const rows=hits.map(h=>{
    const recvPc=isNum(h.lastAcceptT)?h.lastAcceptT+rkOffset:null;
    const sourceOk=isNum(h.wct)&&isNum(h.wht)&&isNum(h.wx)&&isNum(h.wz);
    if(!sourceOk){
      return '<tr><td>—</td><td>'+tableFmt(recvPc,3)+'</td><td>—</td><td>—</td><td>—</td><td>无法回配</td>'+
        '<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>updates='+(h.n||1)+'</td><td>—</td></tr>';
    }
    const accCt=h.wct-RK.t0;
    const accHt=h.wht-RK.t0;
    const accCtPc=accCt+rkOffset;
    const accHtPc=accHt+rkOffset;
    const th=matchThrowByAcceptedCt(accCt);
    const throwNo=th?reportThrows.indexOf(th)+1:null;
    const truth=pcTruthAt(accHtPc);
    const dx=truth?(h.wx-truth.x)*1000:null;
    const dz=truth?(h.wz-truth.z)*1000:null;
    const has300=!!(th&&isNum(th.ref300Ht)&&isNum(th.ref300X)&&isNum(th.ref300Z));
    const dHt=has300?(accHt-th.ref300Ht)*1000:null;
    const dX=has300?(h.wx-th.ref300X)*1000:null;
    const dZ=has300?(h.wz-th.ref300Z)*1000:null;
    const result=strikeAfter(accHtPc).verdict;
    const info='S'+tableFmt(h.wstage,0)+
      (isNum(h.wnFit)?' n_fit='+h.wnFit:'')+
      ' / updates='+(h.n||1);
    return '<tr><td>'+tableFmt(throwNo,0)+'</td>'+
      '<td>'+tableFmt(recvPc,3)+'</td>'+
      '<td>'+accCtPc.toFixed(3)+'</td>'+
      '<td>'+accHtPc.toFixed(3)+'</td>'+
      '<td>'+((accHt-accCt)*1000).toFixed(1)+'</td>'+
      '<td>'+tableXz(h.wx,h.wz)+'</td>'+
      '<td>'+pcTruthCell(truth)+'</td>'+
      '<td>'+tableSigned(dx)+'/'+tableSigned(dz)+'</td>'+
      '<td>'+(has300?(th.ref300Ht+rkOffset).toFixed(3):'—')+'</td>'+
      '<td>'+(has300?tableXz(th.ref300X,th.ref300Z):'—')+'</td>'+
      '<td>'+tableSigned(dHt)+'</td>'+
      '<td>'+tableSigned(dX)+'/'+tableSigned(dZ)+'</td>'+
      '<td>'+info+'</td><td>'+result+'</td></tr>';
  });
  const notes=[
    '每行取该次挥拍最后一条accepted，并回配它实际收到的 /predict_hit_pos 原消息；ct/HT/x/z均来自这条原消息',
    '按原消息ct落入RK抛球消息窗口归抛，不使用 done≈HT 猜配；回配失败就明确显示，绝不借用300ms消息',
    'PC真值@accepted HT与主表的PC真值@RK300 HT分别计算，不能互用',
    'accepted−300 的ΔHT/Δx/Δz为机械臂最后accepted原消息减主表约300ms原消息',
    'accepted状态里的tz是臂坐标，表中x/z使用收到的原始RK rel_x/rel_z',
    pcTruthNote,
  ];
  return '<div class="armTblWrap"><table class="armTbl"><thead><tr>'+
    '<th>RK抛#</th><th>最后accepted t<br>(s,PC轴)</th><th>接收消息ct<br>(s,PC轴)</th><th>接收消息HT<br>(s,PC轴)</th>'+
    '<th>lead(ms)</th><th>最后accepted RK x/z(m)</th><th>PC真值@accepted HT</th><th>accepted−PC dx/dz(mm)</th>'+
    '<th>RK300 HT(s)</th><th>RK300 x/z(m)</th><th>accepted−300 ΔHT(ms)</th><th>accepted−300 Δx/dz(mm)</th>'+
    '<th>消息</th><th>结果</th></tr></thead><tbody>'+rows.join('')+'</tbody></table>'+
    '<div style="font-size:11px;color:#a0a0c0;margin:2px 0 6px">'+notes.join('；')+'。</div></div>';
};

const renderRk300Table = () => {
  const el=document.getElementById('rk300Tbl');
  if(el) el.innerHTML=alignWarnHtml+rk300TableHtml();
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

    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.x), name:'S1 X', mode:'markers',
     marker:{color:'#e74c3c',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 X</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.y), name:'S1 Y', mode:'markers',
     marker:{color:'#c0392b',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 Y</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.z), name:'S1 Z', mode:'markers',
     marker:{color:'#e67e22',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     customdata:s1.map(predRemainingMs),
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<br>remaining=%{customdata:.1f} ms<extra>S1 Z</extra>'}),

    g2({x:s0.map(p=>relTime(p.ct)), y:s0.map(predRemainingMs), name:'S0 remaining(ms)', mode:'markers',
     marker:{color:'#9b59b6',symbol:'triangle-up',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>remaining=%{y:.1f} ms<extra>S0 remaining</extra>'}),
    g2({x:s1.map(p=>relTime(p.ct)), y:s1.map(predRemainingMs), name:'S1 remaining(ms)', mode:'markers',
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
    const off=Number(window.__rkOffset)||0;
    V.frame.textContent=`${cur+1} / ${seg.frames.length}`;
    V.tRk.textContent=`${f.t.toFixed(3)} s`;
    V.tPc.textContent=`${(f.t+off).toFixed(3)} s`;
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
    const off=Number(window.__rkOffset)||0;
    clock.textContent=`帧 ${cur+1}/${seg.frames.length} · t(RK)=${f.t.toFixed(2)}s · PC=${(f.t+off).toFixed(2)}s`;
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
  const off0=Number(window.__rkOffset)||0;
  movements.forEach(m=>{
    const opt=document.createElement('option');
    opt.value=String(m.k);
    const tgt=m.tx!==null?` 末目标(${m.tx.toFixed(2)}, ${m.ty.toFixed(2)})`:'';
    const rem=m.rem0!==null?` 计划${m.rem0.toFixed(2)}s`:'';
    opt.textContent=`第 ${m.k+1} 次  RK ${m.runT0.toFixed(1)}→${m.runT1.toFixed(1)}s（PC ~${(m.runT0+off0).toFixed(1)}s）${tgt}${rem}`;
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
     mode:'markers',type:'scatter3d',name:'S1 pred',
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
  // 击打标记与 FK TCP 都在共享数据层；臂数据已在 RK 相对轴，显示只加 rkOffset
  const aligned = armAligned, dispOff = armDispOff, hitMarks = armHitMarks;
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
    const d=thinRows(rows,get), off=dispOff();
    const X=[], Y=[];
    let prev=null;
    for(let i=0;i<d.T.length;i++){
      if(prev!==null && d.T[i]-prev>gapS){ X.push(null); Y.push(null); }
      X.push(d.T[i]+off); Y.push(d.V[i]); prev=d.T[i];
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
    const off=dispOff(), sh=[];
    hitMarks.forEach(h=>{
      const line=(x,color,dash,width)=>sh.push({type:'line',xref:'x',yref:'paper',
        x0:x+off,x1:x+off,y0:0,y1:1,line:{color,width,dash},opacity:0.85});
      line(h.cmd,'#94a3b8','dot',1);
      if(h.start!=null) line(h.start,'#e94560','solid',1.8);
      line(h.done,'#2dd4bf','solid',1.8);
    });
    return sh;
  };
  const markAnnotations = () => {
    const off=dispOff(), an=[];
    hitMarks.forEach(h=>{
      const add=(x,text,color,yy)=>an.push({x:x+off,y:yy,xref:'x',yref:'paper',text,
        showarrow:false,font:{size:10,color},xanchor:'left',yanchor:'top'});
      add(h.cmd,`${h.label} cmd`,'#94a3b8',0.995);
      if(h.start!=null) add(h.start,'起拍','#e94560',0.973);
      add(h.done,h.label==='hit'?'触球':`${h.label} done`,'#2dd4bf',0.951);
    });
    return an;
  };
  const setArmEv = () => {
    const off=dispOff();
    const marks=hitMarks.map(h=>{
      const seg=[`<b>${h.label}</b> cmd ${(h.cmd+off).toFixed(2)}s`];
      if(h.start!=null) seg.push(`start <b>${(h.start+off).toFixed(2)}s</b>`);
      seg.push(`done <b>${(h.done+off).toFixed(2)}s</b>`);
      return seg.join(' → ');
    });
    const axisInfo=(aligned
      ? `Axis: PC report time（臂数据与 RK 同钟同轴，display = RK t + rkOffset ${(Number(window.__rkOffset)||0).toFixed(4)}s） &nbsp; `
      : 'Axis: arm data time（未对齐：需 v3 _arm.json + RK 数据） &nbsp; ') +
      (ARM.fk_source ? `FK: ${escA(ARM.fk_source)} &nbsp; ` : '') +
      (marks.length ? '| ' + marks.join(' &nbsp;|&nbsp; ') : '| no accepted commands');
    document.getElementById('armEv').innerHTML =
      alignWarnHtml + armAcceptedTableHtml() +
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
  const shifted = xs => xs.map(x=>isNum(Number(x)) ? Number(x)+rkOffset : null);
  const setInfo = () => {
    window.__rkOffset = rkOffset;  // Arm tab 同轴显示要用
    const errText = auto.err==null ? 'n/a' : `${auto.err.toFixed(3)}m / ${auto.n} pts`;
    const srcText = presetOff!=null ? `preset ${presetOff.toFixed(4)}s; ` : '';
    info.innerHTML = (alignBad ? '<span style="color:#e94560;font-weight:700">⚠ 对齐不可信</span> ' : '')
      + `display t = RK t + ${rkOffset.toFixed(4)}s; ${srcText}auto z-shape ${errText}`;
  };
  const tr = (series,key,name,axis,color,mode='markers',extra={}) => g2({
    x:shifted(ts(series)), y:ys(series,key), name, mode,
    marker:{color,size:3}, line:{color,width:1.4},
    yaxis:axis, xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}<extra>${name}</extra>`,
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
  const pcHitRemainingTr = () => g2({
    x:sAll.map(p=>relTime(p.ct)), y:sAll.map(predRemainingMs),
    name:'PC Hit remaining(ms)', mode:'markers',
    customdata:sAll.map(p=>p.stage),
    marker:{color:'#8e44ad',size:4,symbol:sAll.map(p=>stageSym(p.stage))}, yaxis:'y2', xaxis:'x',
    hovertemplate:'t=%{x:.3f}s<br>S%{customdata} remaining=%{y:.1f} ms<extra>PC Hit remaining</extra>',
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
      rows.push({t:ti+rkOffset, v:val[i], stage:rkPredStage[i], sym:stageSym(rkPredStage[i]), size:5,
                 note:(isNum(rkPredDurMs[i])?`remaining=${rkPredDurMs[i].toFixed(1)} ms`:'')
                      +(isNum(rkPredNFit[i])?` n_fit=${rkPredNFit[i]}`:'')});
    }
    rkThrows.forEach(th=>{
      if(!isNum(th[key])) return;
      rows.push({t:th.ht+rkOffset, v:th[key], stage:th.stage, sym:'star', size:11, note:'@ht final ref'});
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
  // PC 真值：球相对车体系位置，只画每抛最终 ht 处的 star；与表格一致，只用入弧拟合。
  // ht 前的过程真值与 PC Ball 曲线重复，不再展示。
  const truthRows = () => {
    const rows=[];
    rkThrows.forEach(th=>{
      const tHit=th.ht+rkOffset;
      const f=pcTruthAt(tHit);
      if(f) rows.push({t:tHit, x:f.x, y:f.y, z:f.z, fit:true,
                       tag:`（入弧拟合±${Math.max(1,Math.round(f.err*1000))}mm）`});
    });
    return rows.sort((a,b)=>a.t-b.t);
  };
  const truthTr = (rows,key,name,color,extra={}) => g2({
    x:rows.map(r=>r.t), y:rows.map(r=>r[key]), name, mode:'markers',
    customdata:rows.map(r=>r.tag||''),
    marker:{color, size:11, symbol:rows.map(r=>r.fit?'star-open':'star')},
    yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m @ht%{customdata}<extra>${name}</extra>`,
    ...extra,
  });
  const pcTr = (key,name,color,extra={}) => g2({
    x:pcRows.map(p=>p.t), y:pcRows.map(p=>p[key]), name, mode:'markers',
    marker:{color,size:2.5}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const pcCarTr = (key,name,color,extra={}) => g2({
    x:pcCarRows.map(c=>c.t), y:pcCarRows.map(c=>c[key]), name, mode:'markers',
    marker:{color,size:2.5,symbol:'diamond'}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const pcCarYawTr = () => g2({
    x:pcCarRows.map(c=>c.t), y:pcCarRows.map(c=>isNum(c.yaw)?c.yaw*10:null),
    name:'PC Car Yaw x10', mode:'markers',
    customdata:pcCarRows.map(c=>c.yaw),
    marker:{color:'#f472b6',size:2.5,symbol:'diamond'}, yaxis:'y3', xaxis:'x',
    hovertemplate:'t=%{x:.3f}s<br>PC Car Yaw=%{customdata:.4f}rad<br>display=%{y:.3f}<extra>PC Car Yaw x10</extra>',
    visible:'legendonly',
  });
  const makeRelSeries = (rows, carRows) => {
    const out={t:[], y:{dx:[], dy:[], dist:[]}};
    rows.forEach(row=>{
      const c=nearest(carRows, row.t);
      if(!c || Math.abs(c.t-row.t)>0.08 || !isNum(row.x) || !isNum(row.y) || !isNum(c.x) || !isNum(c.y)) return;
      const dx=row.x-c.x, dy=row.y-c.y;
      out.t.push(row.t);
      out.y.dx.push(dx);
      out.y.dy.push(dy);
      out.y.dist.push(Math.hypot(dx, dy));
    });
    return out;
  };
  const rkRel = {t:[], y:{dx:[], dy:[], dist:[]}};
  ts(RK.world).forEach((t,i)=>{
    const x=ys(RK.world,'x')[i], y=ys(RK.world,'y')[i];
    const cx=ys(RK.world,'bot_x')[i], cy=ys(RK.world,'bot_y')[i];
    if(!isNum(t) || !isNum(x) || !isNum(y) || !isNum(cx) || !isNum(cy)) return;
    const dx=x-cx, dy=y-cy;
    rkRel.t.push(t);
    rkRel.y.dx.push(dx);
    rkRel.y.dy.push(dy);
    rkRel.y.dist.push(Math.hypot(dx, dy));
  });
  const pcRel = makeRelSeries(pcRows, pcCarRows);
  const pcRelTr = (series,key,name,color,extra={}) => g2({
    x:series.t, y:series.y[key], name, mode:'markers',
    marker:{color,size:3,symbol:'x'}, yaxis:'y', xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`,
    ...extra,
  });
  const traceData = () => [
    pcTr('x','PC Ball X','#7f8c8d',{visible:'legendonly'}),
    pcTr('y','PC Ball Y','#95a5a6',{visible:'legendonly'}),
    pcTr('z','PC Ball Z','#bdc3c7'),
    tr(RK.world,'x','RK World X','y','#3498db','markers',{visible:'legendonly'}),
    tr(RK.world,'y','RK World Y','y','#2980b9','markers',{visible:'legendonly'}),
    tr(RK.world,'z','RK World Z','y','#5cd0ff'),
    rkPredTr('x','RK Predict X','#f97316',{visible:'legendonly',marker:{color:'#f97316',size:6,symbol:'triangle-up'}}),
    rkPredTr('y','RK Predict Y','#fb923c',{visible:'legendonly',marker:{color:'#fb923c',size:6,symbol:'triangle-up'}}),
    rkPredTr('z','RK Predict Z','#e94560',{marker:{color:'#e94560',size:6,symbol:'triangle-up'}}),
    pcHitTr('x','PC Hit X','#fb7185',{visible:'legendonly'}),
    pcHitTr('y','PC Hit Y','#f43f5e',{visible:'legendonly'}),
    pcHitTr('z','PC Hit Z','#e11d48'),
    rkRefTr('rel_x','RK Ref X','#a3e635'),
    rkRefTr('rel_y','RK Ref Y','#4d7c0f',{visible:'legendonly'}),
    rkRefTr('rel_z','RK Ref Z','#84cc16'),
    ...(rows=>[
      truthTr(rows,'x','PC Truth X','#e2e8f0'),
      truthTr(rows,'y','PC Truth Y','#94a3b8',{visible:'legendonly'}),
      truthTr(rows,'z','PC Truth Z','#ffffff'),
    ])(truthRows()),
    // 机械臂 FK TCP（与 RK 数据同轴，+rkOffset 到本轴；z 减 armZOff 还原世界系高度）
    ...(armAligned ? (()=>{
      const tArm=armTcpRows.map(s=>s.t+rkOffset);
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
        const c=carAt(r.t);
        return c ? {t:r.t, ...relToCar(r,c)} : null;
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
    pcCarTr('x','PC Car X','#d946ef'),
    pcCarTr('y','PC Car Y','#c084fc'),
    pcCarTr('z','PC Car Z','#a78bfa',{visible:'legendonly'}),
    pcCarYawTr(),
    tr(RK.bot,'x','Bot X','y','#67e8c3'),
    tr(RK.bot,'y','Bot Y','y','#9fffce'),
    g2({x:shifted(ts(RK.bot)), y:ys(RK.bot,'yaw').map(v=>isNum(v)?v*10:null), name:'Bot Yaw x10', mode:'markers',
      customdata:ys(RK.bot,'yaw'),
      marker:{color:'#5eead4',size:2.5,symbol:'diamond'}, yaxis:'y3', xaxis:'x',
      hovertemplate:'t=%{x:.3f}s<br>Bot Yaw=%{customdata:.4f}rad<br>display=%{y:.3f}<extra>Bot Yaw x10</extra>',
      visible:'legendonly'}),
    tr(RK.bot,'target_x','Bot Target X','y','#ffd27f','markers',{visible:'legendonly'}),
    tr(RK.bot,'target_y','Bot Target Y','y','#ff9f7f','markers',{visible:'legendonly'}),
    tr(rkRel,'dx','RK Ball-Car dX','y','#ef4444','markers',{visible:'legendonly',marker:{color:'#ef4444',size:3,symbol:'cross'}}),
    tr(rkRel,'dy','RK Ball-Car dY','y','#f97316','markers',{visible:'legendonly',marker:{color:'#f97316',size:3,symbol:'cross'}}),
    tr(rkRel,'dist','RK Ball-Car XY Dist','y','#facc15','markers',{marker:{color:'#facc15',size:3,symbol:'cross'}}),
    pcRelTr(pcRel,'dx','PC Ball-Car dX','#ec4899',{visible:'legendonly'}),
    pcRelTr(pcRel,'dy','PC Ball-Car dY','#d946ef',{visible:'legendonly'}),
    pcRelTr(pcRel,'dist','PC Ball-Car XY Dist','#a855f7'),
    pcHitRemainingTr(),
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
  if(input) input.value = rkOffset.toFixed(4);
  setInfo();
  renderRk300Table();
  Plotly.newPlot('c5', traceData(), layout(), PLOT_CONFIG).then(()=>{wl('c5','l5');wz('c5');});
  const apply = document.getElementById('rkApply');
  if(apply) apply.addEventListener('click',()=>{
    const v=Number(input.value);
    rkOffset=isNum(v) ? v : 0;
    redraw();
  });
  const autoBtn = document.getElementById('rkAuto');
  if(autoBtn) autoBtn.addEventListener('click',()=>{
    rkOffset=Math.round(auto.off*10000)/10000;
    if(input) input.value=rkOffset.toFixed(4);
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
    if(sigInput) sigInput.value = rkOffset.toFixed(4);
    if(sigInfo) sigInfo.innerHTML = info.innerHTML;
  };
  syncSignalControls();
  const sigApply = document.getElementById('rkSigApply');
  if(sigApply) sigApply.addEventListener('click',()=>{
    const v=Number(sigInput.value);
    rkOffset=isNum(v) ? v : 0;
    if(input) input.value=rkOffset.toFixed(4);
    redraw().then(syncSignalControls);
  });
  const sigAuto = document.getElementById('rkSigAuto');
  if(sigAuto) sigAuto.addEventListener('click',()=>{
    rkOffset=Math.round(auto.off*10000)/10000;
    if(input) input.value=rkOffset.toFixed(4);
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
        "--rk-offset", type=float, default=None,
        help="预置初始 RK offset（秒）。PC/RK 无共同球观测窗、轨迹自动对齐不可行的场次，"
             "用外部锚（如 PC AprilTag 车速 × RK bot_state 车速互相关）离线求出后喂入；"
             "页面 Auto align 按钮仍可重新估计覆盖。",
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
    generate_html(args.input, out, args.racket_json, arm_json, rk_tracking_json, args.rk_offset)


if __name__ == "__main__":
    main()
