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


def generate_html(input_path: str, output_path: str, racket_json_path: str | None = None) -> None:
    data = _load_json(input_path)
    if racket_json_path:
        data = _merge_racket_json(data, _load_json(racket_json_path), racket_json_path)
    data_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("%%DATA_JSON%%", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Interactive HTML saved: {output_path}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
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
.lb{appearance:none;display:inline-flex;align-items:center;gap:8px;border:1px solid #0f3460;background:#16213e;color:#d7d7eb;
    border-radius:999px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer;transition:background .18s ease,border-color .18s ease,opacity .18s ease,transform .18s ease}
.lb:hover{background:#1a1a3e;border-color:#e94560;transform:translateY(-1px)}
.lb.off{opacity:.45}
.ls{width:10px;height:10px;border-radius:999px;flex:0 0 10px;box-shadow:0 0 0 1px rgba(255,255,255,.15)}
.zx{overflow:hidden;padding-bottom:6px;border-radius:16px;transition:box-shadow .18s ease}
.cc.zoom-active .zx{box-shadow:0 0 0 1px rgba(92,208,255,.55),0 0 0 4px rgba(92,208,255,.10)}
.cb{width:100%;min-width:100%;height:780px;min-height:780px}
</style>
</head>
<body>
<div class="hdr">
  <h1>Tracker / Curve3 Interactive</h1>
  <div class="st" id="st"></div>
</div>
<div class="tabs">
  <div class="tab on" onclick="sw(0)">All-in-One</div>
  <div class="tab" onclick="sw(1)">X / Y / Z Subplots</div>
  <div class="tab" onclick="sw(2)">3D Trajectory</div>
  <div class="tab" onclick="sw(3)">Car Location</div>
</div>
<div id="p0" class="pnl on"><div class="cc"><div class="lc" id="l0"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c0" data-action="out">X-</button><button type="button" class="zb on" data-plot="c0" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c0" data-action="in">X+</button><span id="c0r" class="zr">1.00x</span></div><div class="zx"><div id="c0" class="cb"></div></div></div></div>
<div id="p1" class="pnl"><div class="cc"><div class="lc" id="l1"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c1" data-action="out">X-</button><button type="button" class="zb on" data-plot="c1" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c1" data-action="in">X+</button><span id="c1r" class="zr">1.00x</span></div><div class="zx"><div id="c1" class="cb"></div></div></div></div>
<div id="p2" class="pnl"><div class="cc"><div class="lc" id="l2"></div><div class="zt"><span class="ztl">X zoom</span><button type="button" class="zb" data-plot="c2" data-action="out">X-</button><button type="button" class="zb on" data-plot="c2" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c2" data-action="in">X+</button><span id="c2r" class="zr">n/a</span></div><div class="zx"><div id="c2" class="cb"></div></div></div></div>
<div id="p3" class="pnl"><div class="cc"><div class="lc" id="l3"></div><div class="zt"><span class="ztl">X zoom / click plot + wheel</span><button type="button" class="zb" data-plot="c3" data-action="out">X-</button><button type="button" class="zb on" data-plot="c3" data-action="reset">Reset</button><button type="button" class="zb" data-plot="c3" data-action="in">X+</button><span id="c3r" class="zr">1.00x</span></div><div class="zx"><div id="c3" class="cb"></div></div></div></div>

<script>
const D = %%DATA_JSON%%;
(function(){
const cfg = D.config || {};
const summary = D.summary || {};
const preds = D.predictions || [];
const frames = Array.isArray(D.frames) ? D.frames : [];
const writtenVideoFrameIds = Array.isArray(D.video_frame_indices) ? D.video_frame_indices : [];
const distanceScale = cfg.distance_unit === 'm' ? 1.0 : 0.001;
const scaleVec3 = p => ({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale});
const obsRaw = (D.observations || []).map(o=>scaleVec3(o));
const racketObsRaw = (D.racket_observations || []).map(o=>scaleVec3(o));
const car = (D.car_locs || []).map(c=>({...c, x:c.x*distanceScale, y:c.y*distanceScale, z:c.z*distanceScale}));
const s0 = preds.filter(p=>p.stage===0).map(p=>({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale}));
const s1 = preds.filter(p=>p.stage===1).map(p=>({...p, x:p.x*distanceScale, y:p.y*distanceScale, z:p.z*distanceScale}));
const resets = D.reset_times || summary.reset_times || [];
const throws = D.throws || [];
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
const fallbackT0 = [firstNumeric(obsRaw, 't'), firstNumeric(racketObsRaw, 't'), firstNumeric(car, 't'), firstNumeric(preds, 'ct')]
  .find(v => v !== null);
const t0 = firstFrameT0 !== null ? firstFrameT0 : (fallbackT0 !== null ? fallbackT0 : 0);
const relTime = v => isNum(v) ? (v - t0) : 0;
const frameSeries = frames
  .filter(f => f && isNum(f.exposure_pc))
  .map(f => ({
    ...f,
    rel_s: isNum(f.elapsed_s) ? f.elapsed_s : relTime(f.exposure_pc),
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
const obs = frameBallObs.length > 0
  ? frameBallObs
  : obsRaw.map(o => ({...o, rel_s: relTime(o.t), idx: null, video_frame_idx: null}));
const racket = frameRacketObs.length > 0
  ? frameRacketObs
  : racketObsRaw.map(o => ({
      ...o,
      rel_s: isNum(o.elapsed_s) ? o.elapsed_s : relTime(o.t),
      idx: Number.isInteger(o.frame_idx) ? o.frame_idx : null,
      video_frame_idx: Number.isInteger(o.video_frame_idx) ? o.video_frame_idx : null,
    }));
const pairedFrames = preferredFrames.filter(f => f.ball3d && f.racket3d);
const frameStartLabel =
  frames.length > 0 && isNum(frames[0].exposure_pc)
    ? `${Number(frames[0].exposure_pc).toFixed(6)}s`
    : null;
const ballSourceLabel = frameBallObs.length > 0 ? 'video-linked frames' : 'tracker observations';
const racketSourceLabel = frameRacketObs.length > 0 ? 'video-linked frames' : 'racket observations';

const stat=(k,v)=>`<span>${k}: <span class="v">${v!=null?v:'-'}</span></span>`;
document.getElementById('st').innerHTML=[
  stat('Source', sourceType),
  cfg.replay_source ? stat('Replay source', cfg.replay_source) : '',
  isNum(cfg.first_frame_exposure_pc) ? stat('t0 perf', Number(cfg.first_frame_exposure_pc).toFixed(6)+'s') : '',
  frameStartLabel ? stat('Frame0 perf', frameStartLabel) : '',
  cfg.video_frame_mapping_exact != null ? stat('Frame map', cfg.video_frame_mapping_exact ? 'exact' : 'fallback') : '',
  summary.video_frames_mapped ? stat('Mapped video frames', summary.video_frames_mapped) : '',
  stat('Ball 3D', obs.length),
  stat('Ball src', ballSourceLabel),
  racket.length ? stat('Racket 3D', racket.length) : '',
  racket.length ? stat('Racket src', racketSourceLabel) : '',
  videoLinkedFrames.length ? stat('Video-linked frames', videoLinkedFrames.length) : '',
  pairedFrames.length ? stat('Ball+Racket same-frame', pairedFrames.length) : '',
  stat('S0 preds', s0.length),
  stat('S1 preds', s1.length),
  car.length ? stat('Car locs', car.length) : '',
  stat('Resets', resets.length),
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

{
  const oT=obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t));
  const rT=racket.map(r=>isNum(r.rel_s) ? r.rel_s : relTime(r.t));
  const tr=[
    {x:oT, y:obs.map(o=>o.x), name:'Ball X', mode:'markers',
     marker:{color:'#7f8c8d',symbol:'circle',size:2,opacity:0.5},
     hovertemplate:'t=%{x:.3f}s<br>x=%{y:.3f} m<extra>Ball X</extra>',
     visible:'legendonly'},
    {x:oT, y:obs.map(o=>o.y), name:'Ball Y', mode:'markers',
     marker:{color:'#95a5a6',symbol:'circle',size:2,opacity:0.5},
     hovertemplate:'t=%{x:.3f}s<br>y=%{y:.3f} m<extra>Ball Y</extra>',
     visible:'legendonly'},
    {x:oT, y:obs.map(o=>o.z), name:'Ball Z', mode:'markers',
     marker:{color:'#bdc3c7',symbol:'circle',size:2.5,opacity:0.6},
     hovertemplate:'t=%{x:.3f}s<br>z=%{y:.3f} m<extra>Ball Z</extra>'},

    ...(racket.length ? [
    {x:rT, y:racket.map(r=>r.x), name:'Racket X', mode:'markers',
     marker:{color:'#ff66cc',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket x=%{y:.3f} m<extra>Racket X</extra>',
     visible:'legendonly'},
    {x:rT, y:racket.map(r=>r.y), name:'Racket Y', mode:'markers',
     marker:{color:'#ff33aa',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket y=%{y:.3f} m<extra>Racket Y</extra>',
     visible:'legendonly'},
    {x:rT, y:racket.map(r=>r.z), name:'Racket Z', mode:'markers',
     marker:{color:'#cc00ff',symbol:'x',size:5},
     hovertemplate:'t=%{x:.3f}s<br>racket z=%{y:.3f} m<extra>Racket Z</extra>'},
    ] : []),

    {x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.x), name:'S0 X', mode:'markers',
     marker:{color:'#3498db',symbol:'triangle-up',size:5},
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<extra>S0 X</extra>'},
    {x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.y), name:'S0 Y', mode:'markers',
     marker:{color:'#2980b9',symbol:'triangle-up',size:5},
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<extra>S0 Y</extra>'},
    {x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>p.z), name:'S0 Z', mode:'markers',
     marker:{color:'#1abc9c',symbol:'triangle-up',size:5},
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<extra>S0 Z</extra>'},

    {x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.x), name:'S1 X', mode:'markers',
     marker:{color:'#e74c3c',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<extra>S1 X</extra>'},
    {x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.y), name:'S1 Y', mode:'markers',
     marker:{color:'#c0392b',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<extra>S1 Y</extra>'},
    {x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>p.z), name:'S1 Z', mode:'markers',
     marker:{color:'#e67e22',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<extra>S1 Z</extra>'},

    {x:s0.map(p=>relTime(p.ct)), y:s0.map(p=>(p.ht-p.ct)*1000), name:'S0 lead(ms)', mode:'markers',
     marker:{color:'#9b59b6',symbol:'triangle-up',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S0 lead</extra>'},
    {x:s1.map(p=>relTime(p.ct)), y:s1.map(p=>(p.ht-p.ct)*1000), name:'S1 lead(ms)', mode:'markers',
     marker:{color:'#8e44ad',symbol:'square',size:4}, yaxis:'y2',
     hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S1 lead</extra>'},

    ...(car.length ? [
    {x:car.map(c=>relTime(c.t)), y:car.map(c=>c.x), name:'Car X', mode:'markers',
     marker:{color:'#2ecc71',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car x=%{y:.3f} m<extra>Car X</extra>',
     visible:'legendonly'},
    {x:car.map(c=>relTime(c.t)), y:car.map(c=>c.y), name:'Car Y', mode:'markers',
     marker:{color:'#27ae60',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car y=%{y:.3f} m<extra>Car Y</extra>',
     visible:'legendonly'},
    {x:car.map(c=>relTime(c.t)), y:car.map(c=>c.z), name:'Car Z', mode:'markers',
     marker:{color:'#f1c40f',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car z=%{y:.3f} m<extra>Car Z</extra>',
     visible:'legendonly'},
    ] : []),
  ];

  Plotly.newPlot('c0',tr,{
    ...DL,
    title:{text:'All Curves - click legend to toggle, scroll to zoom',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS}, yaxis:{title:'Value (m)',...GS},
    yaxis2:{title:'Lead (ms)',...GS,overlaying:'y',side:'right'},
  },{responsive:true}).then(()=>{wl('c0','l0');wz('c0');});
}

{
  const oT=obs.map(o=>isNum(o.rel_s) ? o.rel_s : relTime(o.t));
  const rT=racket.map(r=>isNum(r.rel_s) ? r.rel_s : relTime(r.t));
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push({x:oT,y:obs.map(o=>o[k]),name:`Ball ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#7f8c8d',symbol:'circle',size:2,opacity:0.4},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Ball ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'});
    if (racket.length) {
      tr.push({x:rT,y:racket.map(r=>r[k]),name:`Racket ${k.toUpperCase()}`,mode:'markers',
        marker:{color:['#ff66cc','#ff33aa','#cc00ff'][i],symbol:'x',size:4},
        hovertemplate:`t=%{x:.3f}s<br>racket ${k}=%{y:.3f} m<extra>Racket</extra>`,
        yaxis:ya,xaxis:'x'});
    }
    tr.push({x:s0.map(p=>relTime(p.ct)),y:s0.map(p=>p[k]),name:`S0 ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#3498db',symbol:'triangle-up',size:4},
      hovertemplate:`t=%{x:.3f}s<br>pred ${k}=%{y:.3f} m<extra>S0</extra>`,
      yaxis:ya,xaxis:'x'});
    tr.push({x:s1.map(p=>relTime(p.ct)),y:s1.map(p=>p[k]),name:`S1 ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#e74c3c',symbol:'square',size:4,line:{width:0.5,color:'#fff'}},
      hovertemplate:`t=%{x:.3f}s<br>pred ${k}=%{y:.3f} m<extra>S1</extra>`,
      yaxis:ya,xaxis:'x'});
  });
  tr.push({x:s0.map(p=>relTime(p.ct)),y:s0.map(p=>(p.ht-p.ct)*1000),name:'S0 lead',mode:'markers',
    marker:{color:'#9b59b6',symbol:'triangle-up',size:3},
    hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S0</extra>',
    yaxis:'y4',xaxis:'x'});
  tr.push({x:s1.map(p=>relTime(p.ct)),y:s1.map(p=>(p.ht-p.ct)*1000),name:'S1 lead',mode:'markers',
    marker:{color:'#8e44ad',symbol:'square',size:3},
    hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S1</extra>',
    yaxis:'y4',xaxis:'x'});

  Plotly.newPlot('c1',tr,{
    ...DL,
    title:{text:'X / Y / Z / Lead (shared time axis)',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS,domain:[0,1],anchor:'y4'},
    yaxis:{title:'X (m)',...GS,domain:[0.78,1]},
    yaxis2:{title:'Y (m)',...GS,domain:[0.53,0.75]},
    yaxis3:{title:'Z (m)',...GS,domain:[0.28,0.50]},
    yaxis4:{title:'Lead (ms)',...GS,domain:[0.0,0.25]},
  },{responsive:true}).then(()=>{wl('c1','l1');wz('c1');});
}

{
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
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>S0</extra>',
      text:s0.map(p=>relTime(p.ct).toFixed(3))},
    {x:s1.map(p=>p.x),y:s1.map(p=>p.y),z:s1.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S1 pred',
     marker:{color:'#e74c3c',size:4,symbol:'diamond',line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>S1</extra>',
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
  },{responsive:true}).then(()=>{wl('c2','l2');wz('c2');});
}

if(car.length > 0){
  const cT=car.map(c=>relTime(c.t));
  const yawDeg=car.map(c=>c.yaw*180/Math.PI);
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push({x:cT,y:car.map(c=>c[k]),name:`Car ${k.toUpperCase()}`,mode:'markers',
      marker:{color:['#2ecc71','#27ae60','#f1c40f'][i],size:2},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Car ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'});
  });
  tr.push({x:cT,y:yawDeg,name:'Car Yaw',mode:'markers',
    marker:{color:'#e94560',size:2},
    hovertemplate:'t=%{x:.3f}s<br>yaw=%{y:.1f}deg<extra>Car Yaw</extra>',
    yaxis:'y4',xaxis:'x'});
  tr.push({x:cT,y:car.map(c=>c.reprojection_error),name:'Reproj Err',mode:'markers',
    marker:{color:'#e67e22',size:2},
    hovertemplate:'t=%{x:.3f}s<br>err=%{y:.2f} px<extra>Reproj</extra>',
    yaxis:'y5',xaxis:'x'});

  Plotly.newPlot('c3',tr,{
    ...DL,
    title:{text:'Car Location (X / Y / Z / Yaw / Reproj)',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS,domain:[0,1],anchor:'y5'},
    yaxis:{title:'X (m)',...GS,domain:[0.82,1]},
    yaxis2:{title:'Y (m)',...GS,domain:[0.62,0.79]},
    yaxis3:{title:'Z (m)',...GS,domain:[0.42,0.59]},
    yaxis4:{title:'Yaw (deg)',...GS,domain:[0.22,0.39]},
    yaxis5:{title:'Reproj (px)',...GS,domain:[0.0,0.19]},
  },{responsive:true}).then(()=>{wl('c3','l3');wz('c3');});
}
})();

function sw(i){
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('on',j===i));
  document.querySelectorAll('.pnl').forEach((p,j)=>p.classList.toggle('on',j===i));
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
  ctrl.innerHTML=plot.data.map((trace,idx)=>{
    const on=tv(trace);
    const name=trace&&trace.name?trace.name:`trace ${idx+1}`;
    return `<button type="button" class="lb${on?'':' off'}" data-plot="${plotId}" data-index="${idx}" aria-pressed="${on?'true':'false'}"><span class="ls" style="background:${tc(trace)}"></span><span>${name}</span></button>`;
  }).join('');
  ctrl.querySelectorAll('.lb').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const idx=Number(btn.dataset.index);
      const next=!tv(plot.data[idx]);
      Plotly.restyle(plotId,{visible:next?true:'legendonly'},[idx]).then(()=>tl(plotId,ctrlId));
    });
  });
}
function wl(plotId,ctrlId){
  const plot=document.getElementById(plotId);
  if(!plot) return;
  tl(plotId,ctrlId);
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
document.querySelectorAll('.zb').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const id=btn.dataset.plot;
    sap(id);
    if(btn.dataset.action==='in') zx(id,1/ZSTEP);
    else if(btn.dataset.action==='out') zx(id,ZSTEP);
    else zxReset(id);
  });
});
sap('c0');
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="curve3_output/curve3_result.json")
    parser.add_argument("--racket-json", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = os.path.splitext(args.input)[0]
    out = args.output or (base + ".html")
    generate_html(args.input, out, args.racket_json)


if __name__ == "__main__":
    main()
