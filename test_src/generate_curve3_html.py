# -*- coding: utf-8 -*-
"""
从 curve3_result.json 生成可交互 HTML 可视化（共享曲线、连续时间轴）。
"""

import argparse
import json
import os


def generate_html(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    data_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("%%DATA_JSON%%", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Interactive HTML saved: {output_path}")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Curve3 Interactive</title>
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
.cb{width:100%;height:85vh;min-height:550px}
</style>
</head>
<body>
<div class="hdr">
  <h1>Curve3 — Continuous Multi-Throw</h1>
  <div class="st" id="st"></div>
</div>
<div class="tabs">
  <div class="tab on" onclick="sw(0)">All-in-One</div>
  <div class="tab" onclick="sw(1)">X / Y / Z Subplots</div>
  <div class="tab" onclick="sw(2)">3D Trajectory</div>
  <div class="tab" onclick="sw(3)">Car Location</div>
</div>
<div id="p0" class="pnl on"><div class="cc"><div id="c0" class="cb"></div></div></div>
<div id="p1" class="pnl"><div class="cc"><div id="c1" class="cb"></div></div></div>
<div id="p2" class="pnl"><div class="cc"><div id="c2" class="cb"></div></div></div>
<div id="p3" class="pnl"><div class="cc"><div id="c3" class="cb"></div></div></div>

<script>
const D = %%DATA_JSON%%;
(function(){
const cfg = D.config, preds = D.predictions;
// mm → m
const obs = D.observations.map(o=>({...o, x:o.x/1000, y:o.y/1000, z:o.z/1000}));
const car = (D.car_locs || []).map(c=>({...c, x:c.x/1000, y:c.y/1000, z:c.z/1000}));
const s0 = preds.filter(p=>p.stage===0).map(p=>({...p, x:p.x/1000, y:p.y/1000, z:p.z/1000}));
const s1 = preds.filter(p=>p.stage===1).map(p=>({...p, x:p.x/1000, y:p.y/1000, z:p.z/1000}));
// 兼容 test_curve3 和 run_tracker 两种 JSON 格式
const resets = D.reset_times || (D.summary && D.summary.reset_times) || [];
const throws = D.throws || [];

// 时间轴偏移：以第一个观测为 0s
const t0 = obs.length > 0 ? obs[0].t : 0;

// stats
const stat=(k,v)=>`<span>${k}: <span class="v">${v!=null?v:'-'}</span></span>`;
document.getElementById('st').innerHTML=[
  cfg.start_time ? stat('Start', cfg.start_time) : '',
  stat('Observations', obs.length),
  stat('S0 preds', s0.length),
  stat('S1 preds', s1.length),
  car.length ? stat('Car locs', car.length) : '',
  stat('Resets', resets.length),
  throws.length ? stat('Throws', throws.length) : '',
  cfg.fps ? stat('FPS', cfg.fps) : '',
  cfg.noise_mm!=null ? stat('Noise', cfg.noise_mm+'mm') : '',
  stat('COR', cfg.cor),
  stat('ideal_hit_z', (cfg.ideal_hit_z/1000).toFixed(2)+'m'),
  cfg.min_stage1_points ? stat('min_s1', cfg.min_stage1_points) : '',
  cfg.duration_s ? stat('Duration', cfg.duration_s.toFixed(1)+'s') : '',
].filter(Boolean).join('');

// layout
const DL={paper_bgcolor:'#1a1a2e',plot_bgcolor:'#16213e',font:{color:'#e0e0e0',size:11},
  legend:{bgcolor:'rgba(22,33,62,0.9)',bordercolor:'#0f3460',borderwidth:1,font:{size:10},itemsizing:'constant'},
  hovermode:'closest',margin:{l:60,r:30,t:40,b:50}};
const GS={gridcolor:'#0f3460',zerolinecolor:'#0f3460'};

// ═══ Chart 0: All-in-One ═══
{
  const oT=obs.map(o=>o.t-t0);
  const tr=[
    // Ball observations — each component independent
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

    // S0 predictions
    {x:s0.map(p=>p.ct-t0), y:s0.map(p=>p.x), name:'S0 X', mode:'markers',
     marker:{color:'#3498db',symbol:'triangle-up',size:5},
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<extra>S0 X</extra>'},
    {x:s0.map(p=>p.ct-t0), y:s0.map(p=>p.y), name:'S0 Y', mode:'markers',
     marker:{color:'#2980b9',symbol:'triangle-up',size:5},
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<extra>S0 Y</extra>'},
    {x:s0.map(p=>p.ct-t0), y:s0.map(p=>p.z), name:'S0 Z', mode:'markers',
     marker:{color:'#1abc9c',symbol:'triangle-up',size:5},
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<extra>S0 Z</extra>'},

    // S1 predictions
    {x:s1.map(p=>p.ct-t0), y:s1.map(p=>p.x), name:'S1 X', mode:'markers',
     marker:{color:'#e74c3c',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{x:.3f}s<br>pred x=%{y:.3f} m<extra>S1 X</extra>'},
    {x:s1.map(p=>p.ct-t0), y:s1.map(p=>p.y), name:'S1 Y', mode:'markers',
     marker:{color:'#c0392b',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{x:.3f}s<br>pred y=%{y:.3f} m<extra>S1 Y</extra>'},
    {x:s1.map(p=>p.ct-t0), y:s1.map(p=>p.z), name:'S1 Z', mode:'markers',
     marker:{color:'#e67e22',symbol:'square',size:5,line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{x:.3f}s<br>pred z=%{y:.3f} m<extra>S1 Z</extra>'},

    // Lead time
    {x:s0.map(p=>p.ct-t0), y:s0.map(p=>(p.ht-p.ct)*1000), name:'S0 lead(ms)', mode:'markers',
     marker:{color:'#9b59b6',symbol:'triangle-up',size:3},
     hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S0 lead</extra>',
     visible:'legendonly'},
    {x:s1.map(p=>p.ct-t0), y:s1.map(p=>(p.ht-p.ct)*1000), name:'S1 lead(ms)', mode:'markers',
     marker:{color:'#8e44ad',symbol:'square',size:3},
     hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S1 lead</extra>',
     visible:'legendonly'},

    // Car location
    ...(car.length ? [
    {x:car.map(c=>c.t-t0), y:car.map(c=>c.x), name:'Car X', mode:'markers',
     marker:{color:'#2ecc71',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car x=%{y:.3f} m<extra>Car X</extra>',
     visible:'legendonly'},
    {x:car.map(c=>c.t-t0), y:car.map(c=>c.y), name:'Car Y', mode:'markers',
     marker:{color:'#27ae60',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car y=%{y:.3f} m<extra>Car Y</extra>',
     visible:'legendonly'},
    {x:car.map(c=>c.t-t0), y:car.map(c=>c.z), name:'Car Z', mode:'markers',
     marker:{color:'#f1c40f',symbol:'circle',size:2},
     hovertemplate:'t=%{x:.3f}s<br>car z=%{y:.3f} m<extra>Car Z</extra>',
     visible:'legendonly'},
    ] : []),
  ];

  Plotly.newPlot('c0',tr,{
    ...DL,
    title:{text:'All Curves — click legend to toggle, scroll to zoom',font:{size:13,color:'#a0a0c0'}},
    xaxis:{title:'Time (s)',...GS}, yaxis:{title:'Value (m)',...GS},
  },{responsive:true});
}

// ═══ Chart 1: X / Y / Z / Lead subplots ═══
{
  const oT=obs.map(o=>o.t-t0);
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push({x:oT,y:obs.map(o=>o[k]),name:`Ball ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#7f8c8d',symbol:'circle',size:2,opacity:0.4},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Ball ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'});
    tr.push({x:s0.map(p=>p.ct-t0),y:s0.map(p=>p[k]),name:`S0 ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#3498db',symbol:'triangle-up',size:4},
      hovertemplate:`t=%{x:.3f}s<br>pred ${k}=%{y:.3f} m<extra>S0</extra>`,
      yaxis:ya,xaxis:'x'});
    tr.push({x:s1.map(p=>p.ct-t0),y:s1.map(p=>p[k]),name:`S1 ${k.toUpperCase()}`,mode:'markers',
      marker:{color:'#e74c3c',symbol:'square',size:4,line:{width:0.5,color:'#fff'}},
      hovertemplate:`t=%{x:.3f}s<br>pred ${k}=%{y:.3f} m<extra>S1</extra>`,
      yaxis:ya,xaxis:'x'});
  });
  // lead time
  tr.push({x:s0.map(p=>p.ct-t0),y:s0.map(p=>(p.ht-p.ct)*1000),name:'S0 lead',mode:'markers',
    marker:{color:'#9b59b6',symbol:'triangle-up',size:3},
    hovertemplate:'t=%{x:.3f}s<br>lead=%{y:.1f} ms<extra>S0</extra>',
    yaxis:'y4',xaxis:'x'});
  tr.push({x:s1.map(p=>p.ct-t0),y:s1.map(p=>(p.ht-p.ct)*1000),name:'S1 lead',mode:'markers',
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
  },{responsive:true});
}

// ═══ Chart 2: 3D ═══
{
  Plotly.newPlot('c2',[
    {x:obs.map(o=>o.x),y:obs.map(o=>o.y),z:obs.map(o=>o.z),
     mode:'markers',type:'scatter3d',name:'Ball',
     marker:{color:obs.map(o=>o.t-t0),colorscale:'Viridis',size:2,opacity:0.5,
       colorbar:{title:'t(s)',len:0.5,tickfont:{color:'#e0e0e0'},titlefont:{color:'#e0e0e0'}}},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Ball</extra>',
     text:obs.map(o=>(o.t-t0).toFixed(3))},
    {x:s0.map(p=>p.x),y:s0.map(p=>p.y),z:s0.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S0 pred',
     marker:{color:'#3498db',size:4,symbol:'diamond'},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>S0</extra>',
     text:s0.map(p=>(p.ct-t0).toFixed(3))},
    {x:s1.map(p=>p.x),y:s1.map(p=>p.y),z:s1.map(p=>p.z),
     mode:'markers',type:'scatter3d',name:'S1 pred',
     marker:{color:'#e74c3c',size:4,symbol:'diamond',line:{width:0.5,color:'#fff'}},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>S1</extra>',
     text:s1.map(p=>(p.ct-t0).toFixed(3))},
    ...(car.length ? [{x:car.map(c=>c.x),y:car.map(c=>c.y),z:car.map(c=>c.z),
     mode:'markers',type:'scatter3d',name:'Car',
     marker:{color:'#2ecc71',size:4,symbol:'square'},
     hovertemplate:'t=%{text}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Car</extra>',
     text:car.map(c=>(c.t-t0).toFixed(3))}] : []),
  ],{
    ...DL,
    title:{text:'3D Trajectory (all throws, color=time)',font:{size:13,color:'#a0a0c0'}},
    scene:{xaxis:{title:'X(m)',...GS,backgroundcolor:'#16213e'},
           yaxis:{title:'Y(m)',...GS,backgroundcolor:'#16213e'},
           zaxis:{title:'Z(m)',...GS,backgroundcolor:'#16213e'},bgcolor:'#16213e'},
  },{responsive:true});
}
// ═══ Chart 3: Car Location ═══
if(car.length > 0){
  const cT=car.map(c=>c.t-t0);
  const yawDeg=car.map(c=>c.yaw*180/Math.PI);
  const tr=[];
  ['x','y','z'].forEach((k,i)=>{
    const ya=i===0?'y':`y${i+1}`;
    tr.push({x:cT,y:car.map(c=>c[k]),name:`Car ${k.toUpperCase()}`,mode:'markers',
      marker:{color:['#2ecc71','#27ae60','#f1c40f'][i],size:2},
      hovertemplate:`t=%{x:.3f}s<br>${k}=%{y:.3f} m<extra>Car ${k.toUpperCase()}</extra>`,
      yaxis:ya,xaxis:'x'});
  });
  // yaw
  tr.push({x:cT,y:yawDeg,name:'Car Yaw',mode:'markers',
    marker:{color:'#e94560',size:2},
    hovertemplate:'t=%{x:.3f}s<br>yaw=%{y:.1f}°<extra>Car Yaw</extra>',
    yaxis:'y4',xaxis:'x'});
  // reproj error
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
    yaxis4:{title:'Yaw (°)',...GS,domain:[0.22,0.39]},
    yaxis5:{title:'Reproj (px)',...GS,domain:[0.0,0.19]},
  },{responsive:true});
}
})();
function sw(i){
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('on',j===i));
  document.querySelectorAll('.pnl').forEach((p,j)=>p.classList.toggle('on',j===i));
  window.dispatchEvent(new Event('resize'));
}
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="curve3_output/curve3_result.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    # 默认输出与输入同名但扩展名改为 .html
    base = os.path.splitext(args.input)[0]
    out = args.output or (base + ".html")
    generate_html(args.input, out)


if __name__ == "__main__":
    main()
