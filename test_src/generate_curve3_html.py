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


def _merge_racket_impact_json(
    base_data: dict,
    impact_data: dict,
    impact_json_path: str | None,
    tracker_json_path: str,
) -> dict:
    schema = impact_data.get("schema")
    if schema != "racket_impact/v3":
        raise ValueError(
            "racket impact sidecar schema mismatch: "
            f"expected 'racket_impact/v3', got {schema!r}"
        )
    measurements = impact_data.get("racket_impact")
    if not isinstance(measurements, list):
        raise ValueError("racket impact sidecar must contain a racket_impact array")
    expected_metadata = {
        "control_usage": "record_only",
        "frame_time_semantics": "mosaic_group_mean_exposure_center_pc_perf_counter",
        "vz_semantics": "racket_head_bbox_center_world_velocity_proxy",
    }
    for key, expected in expected_metadata.items():
        if impact_data.get(key) != expected:
            raise ValueError(
                f"racket impact sidecar {key} mismatch: "
                f"expected {expected!r}, got {impact_data.get(key)!r}"
            )
    source = impact_data.get("source")
    source_tracker = source.get("tracker_json") if isinstance(source, dict) else None
    if (
        not isinstance(source_tracker, str)
        or Path(source_tracker).resolve() != Path(tracker_json_path).resolve()
    ):
        raise ValueError("racket impact sidecar tracker source mismatch")
    merged = copy.deepcopy(base_data)
    merged["racket_impact"] = copy.deepcopy(measurements)
    if impact_json_path:
        merged.setdefault("config", {})["racket_impact_json_path"] = str(impact_json_path)
    return merged


def _add_face_angles(arm, *, tracker_json_path: str | None = None, car: str | None = None) -> None:
    """给 arm.states 逐帧附加 fy/fp/v_tcp_arm，并按本场车型复算 TCP。

    fy/fp = FK 拍面法向（该车 link6 系的法向轴，前向规范化）在臂系的
    yaw（°，atan2(x,y) 口径，与 PC回球 yaw 同式）与 pitch（°，asin(n_z)，正=开面上仰），
    v_tcp_arm = 拍心相对机械臂基座的三维线速度（m/s，臂系）。
    单源复用 extract_arm_bag.fk（0801 dz/yawrate 分析脚本同一公式），不在 JS 里抄第二份 FK 链。

    **车型**：v0.3 与 v0.4 是两台不同的臂，TCP 差几厘米（见 extract_arm_bag 文件头）。
    取值序：显式 car > arm JSON 的 "car"（新版 extract_arm_bag 写的）> 本场 tracker JSON 的
    config.car_config_path 推断。老 arm JSON 的 tcp 是按当年写死的 v0.3 链算的，与本场车型
    不符时**整表 tcp 就地按正确车型复算**（states 顺手、commands 补算），页面因此永远与
    所选车型自洽，不必为了改车型重跑 rosbag 提取。
    **pitch 不需要减车 yaw**：J1/BASE_ROT 都是纯 z（垂直）转，只搬 n 的水平分量、不动 n_z，
    故臂系 pitch ≡ 世界 pitch（车无 roll/pitch 前提），比 yaw 少一个误差源。
    **v_tcp_arm 走解析 Jacobian**：同一次 fk 已经给出每个关节的 joint_frames，转轴 a_j=R_j·axis、
    轴上一点 o_j=p_j 都是现成的，故 v_tcp = Σ_j q̇_j·(a_j×(p_tcp−o_j))——与 6 次数值差分
    逐分量一致到 1e-6，但只需一次 FK（差分要 7 次，整场多花 ~30s）。必须保留三维向量，
    后续才能与同刻车体平移及 yaw 刚体速度做世界系向量合成。缺 velocity 的帧只出 fy/fp。
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
        import extract_arm_bag as _eab
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
    # 车型：显式 > arm JSON 自述 > tracker JSON 推断（extract_arm_bag.car_for_session 单源）。
    # 三条都没有就整列留空——宁可空也不按错车算，那种错在页面上毫无征兆。
    arm_car = arm.get("car") if isinstance(arm, dict) else None
    try:
        car, car_source = _eab.car_for_session(arm, tracker_json_path, car)
    except Exception as exc:
        print(
            f"[report] 臂 FK 车型未知（{exc}）：TCP / 拍面角 / 拍速 全列留空。"
            "重跑 test_src/extract_arm_bag.py --car v03|v04 出 _arm.json 即可。",
            file=sys.stderr,
        )
        return
    model = _eab.use_car(car)
    fk, JOINTS = _eab.fk, _eab.JOINTS
    arm["fk_car"] = car
    arm["fk_car_source"] = car_source
    # 老 arm JSON（或换了车）里的 tcp 是别的车算的，commands 侧没人复算 → 这里补上。
    stale_tcp = arm_car != car
    arm["fk_source"] = f"extract_arm_bag.fk({car})"
    if stale_tcp:
        arm["fk_recomputed_from"] = arm_car or "未记车型（旧版 extract_arm_bag，即 v03 链）"
        arm["fk_source"] += f"（页面复算；arm JSON 原为 {arm['fk_recomputed_from']}）"
        n_cmd = _eab.recompute_tcp(arm.get("commands"))
        print(f"[report] arm JSON 的 TCP 按 {arm['fk_recomputed_from']} 算，"
              f"本场是 {car}（{car_source}）——就地按 {car} 复算 TCP"
              f"（commands {n_cmd} 行，states 随下面的逐帧 FK 一起）。")
    link6 = JOINTS[-1]["child"]
    face_axis = model.face_normal_in_link6
    for s in states:
        q = s.get("position") if isinstance(s, dict) else None
        if not (isinstance(q, list) and len(q) == 6
                and all(isinstance(v, (int, float)) for v in q)):
            continue
        res = fk(q)
        rot = res["link_transforms"][link6]
        # 拍面法向 = R·(该车 link6 系法向轴)；前向规范化要同时翻 n_z，否则 pitch 会跟着 n_y 变号
        n0, n1, n2 = (float(v) for v in rot[:3, :3] @ face_axis)
        if n1 < 0:
            n0, n1, n2 = -n0, -n1, -n2
        s["fy"] = round(_m.degrees(_m.atan2(n0, n1)), 2)
        s["fp"] = round(_m.degrees(_m.asin(max(-1.0, min(1.0, n2)))), 2)
        tcp = res["tcp"]
        if stale_tcp:
            s["tcp"] = [round(float(v), 4) for v in tcp]
        qd = s.get("velocity")
        if not (isinstance(qd, list) and len(qd) == 6
                and all(isinstance(v, (int, float)) for v in qd)):
            continue
        vel = _np.zeros(3)
        for rate, joint in zip(qd, JOINTS):
            frame = res["joint_frames"][joint["name"]]
            axis = frame[:3, :3] @ joint["axis"]
            vel += rate * _np.cross(axis, tcp - frame[:3, 3])
        s["v_tcp_arm"] = [round(float(v), 4) for v in vel]


def generate_html(
    input_path: str,
    output_path: str,
    racket_json_path: str | None = None,
    racket_impact_json_path: str | None = None,
    arm_json_path: str | None = None,
    rk_tracking_json_path: str | None = None,
    rk_time_bias: float | None = None,
    export_tables: bool = True,
) -> None:
    data = _load_json(input_path)
    if racket_json_path:
        data = _merge_racket_json(data, _load_json(racket_json_path), racket_json_path)
    impact_path = racket_impact_json_path
    if impact_path:
        data = _merge_racket_impact_json(
            data, _load_json(impact_path), impact_path, input_path
        )
    if arm_json_path:
        data["arm"] = _load_json(arm_json_path)
        _add_face_angles(data["arm"], tracker_json_path=input_path)
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
    if export_tables:
        _export_tables(output_path)


def _export_tables(output_path: str) -> None:
    """顺手导出一份纯文本/JSON 表格，给人和 AI 直接读，免得为了看几个数去起浏览器。

    走的是 export_report_tables.py（node 里跑同一份页面脚本），所以不会与页面分叉。
    这一步失败绝不能影响 HTML 本身，只打印原因。
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_report_tables import export as export_report_tables

        md_path, json_path = export_report_tables(Path(output_path))
        print(f"Report tables saved: {md_path}")
        print(f"Report tables saved: {json_path}")
    except Exception as exc:  # noqa: BLE001 — 导出是附加产物，不能拖垮报告生成
        print(f"Report tables export skipped: {exc}")


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
  <div class="tab on" id="tabRk" data-idx="5" onclick="sw(5)">末次Target / PC</div>
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
    <div class="rkCoordNote"><b>坐标说明：</b>PC真值采用世界坐标轴，不随车体 yaw 旋转。x = 拟合球心 world_x − 同时刻插值车体中心 world_x；y 默认是球心 world_y − 车体中心 world_y，仅“PC真值@对应预测HT”列的 y 显示球接触面，即 (球心 world_y − R球3.3cm) − 车体中心 world_y。PC S1 以同抛最后一条 RK S0 世界 y 为相会面，沿原 PC S1 drag 状态重求 x/z/HT/v；交点落到地面下时不显示。</div>
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
// RK 侧一次性配置公告（0817 起随 bag 自带；老 bag 为空）。不走 RK 变量的 world 门控。
const CA=(D.rk_tracking&&D.rk_tracking.config_announce)||{};
const gitLabel=c=>c&&c.git?((c.git_branch?c.git_branch+'@':'')+c.git):null;
const rkCarName=(CA.bot_center&&CA.bot_center.car_name)||(CA.chassis&&CA.chassis.car_name)||null;
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
  gitLabel(CA.bot_center) ? stat('RK git', gitLabel(CA.bot_center)) : '',
  rkCarName ? stat('RK car', rkCarName) : '',
  gitLabel(CA.arm) ? stat('Arm git', gitLabel(CA.arm)) : '',
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
// [[align-core-begin]] —— test_report_time_align.py 抽取此段在 node 里回归测试。
// 段内允许依赖的全局仅限：obs / car / RK / cfg / t0 / isNum / relTime；
// 新增依赖必须同步改 test_report_time_align.py 的 _run_estimate 桩，否则测试静默失真。
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

// PC 发布的小车位姿会进入 RK world 的 bot_x/bot_y/bot_yaw，但两侧记录的不是同一事件时刻：
// PC 行是相机曝光时刻，RK world.t 是球时刻，bot_* 还是经过视觉处理、网络与状态保持后的历史值。
// 因此这些共同值只配做 bias 粗定位，绝不能再拿来拟合时钟比例。报告合同固定 scale=1。
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
  if(anchors.length<8) return {scale:1,bias:null,anchors:anchors.length,mad:null};
  const off0=median(anchors.map(row=>row.off));
  const offMad=median(anchors.map(row=>Math.abs(row.off-off0)));
  // 位姿键重复时 rkFirst 会配到错的一条，这些错锚能偏出 ±14s；给偏移离群门硬上限。
  const maxDev=Math.min(0.5,Math.max(0.2,8*offMad));
  anchors=anchors.filter(row=>Math.abs(row.off-off0)<=maxDev).sort((a,b)=>a.rk-b.rk);
  if(anchors.length<8) return {scale:1,bias:null,anchors:anchors.length,mad:null};
  // 只交出常数偏移来收窄搜索窗；后面的逐抛 z 形状精锁按约 ±10ms 口径报质量。
  const bias=median(anchors.map(row=>row.off));
  const mad=median(anchors.map(row=>Math.abs(row.off-bias)));
  return mad>0.25
    ? {scale:1,bias:null,anchors:anchors.length,mad}
    : {scale:1,bias,anchors:anchors.length,mad};
})();

// ===== 粗定位来源 ①：录制时直接量出来的 PC↔RK 时钟桥（最强，且不依赖本场有没有球）=====
// tracker 运行时订阅一条 RK 高频 topic，记 median(perf_counter 收到时刻 − 载荷 RK 时刻)，
// 落在 config.rk_clock_bridge。它是**测量**不是推断：不看球、不看 tag，空场次照样成立。
// 精度受网络+载荷年龄影响（几 ms~几十 ms），只用来定窗，最终精度仍由 z 形状精锁给。
// 2026-08-12 之前的场次没有这个字段，自动退到下面的位姿形状锁。
const clockBridge = (()=>{
  const raw = cfg.rk_clock_bridge;
  if(!RK || !raw || !isNum(Number(raw.pc_minus_rk)) || !isNum(Number(RK.t0))) return {bias:null,mad:null,n:0};
  const n=Number(raw.n)||0, mad=isNum(Number(raw.mad))?Number(raw.mad):null;
  // 抖动过大说明那条 topic 当场断断续续，宁可不用
  if(n<20 || (mad!=null && mad>0.25)) return {bias:null,mad,n};
  // rel 轴换算：pc_rel − rk_rel = (pc_abs − rk_abs) + RK.t0 − t0
  return {bias:Number(raw.pc_minus_rk)+Number(RK.t0)-t0, mad, n};
})();

// 共享小车位姿的第二种用法：**形状锁**。不要求两侧数值相等（bot_state 多数时候发的是
// KF 传播过的位姿，实测 806 行里只有 33 行与 PC 值逐位相同 → 精确值锚天生稀疏、场次间
// 从 87 条到 0 条乱跳），直接把 RK 位姿序列在 PC 位姿序列上滑动，取中位偏差最小的 bias。
// 小车位姿全场不重复，所以这是唯一能做**全局粗定位**的信号；球 z 形状只能精锁不能粗定位
// （抛球每 10~20s 重复一次，全场搜索必混叠，见 [[v03-rk-pc-report-align]]）。
const rkPoseRows = (()=>{
  if(!RK) return [];
  const wx=ys(RK.world,'bot_x'), wy=ys(RK.world,'bot_y'), wyaw=ys(RK.world,'bot_yaw');
  return ts(RK.world).map((t,i)=>({
    t:Number(t), x:Number(wx[i]), y:Number(wy[i]), yaw:Number(wyaw[i]),
  })).filter(row=>isNum(row.t)&&isNum(row.x)&&isNum(row.y)&&isNum(row.yaw))
    .sort((a,b)=>a.t-b.t);
})();
// x/y 同属 pcCarRows，一次二分搜出线段两端一起用；yaw 行是另一套（null yaw 被剔过）。
// 这条路径要跑几千个 bias 候选 × 几百行，每省一次二分就是整页加载时间。
const interpPcPose = t => {
  const seg=interpRow(pcCarRows,t,0.3);
  const yawSeg=interpRow(pcCarYawRows,t,0.3);
  return {
    x:(seg&&isNum(seg.a.x)&&isNum(seg.b.x))?lerp(seg.a.x,seg.b.x,seg.f):null,
    y:(seg&&isNum(seg.a.y)&&isNum(seg.b.y))?lerp(seg.a.y,seg.b.y,seg.f):null,
    yaw:(yawSeg&&isNum(yawSeg.a.yaw)&&isNum(yawSeg.b.yaw))?lerp(yawSeg.a.yaw,yawSeg.b.yaw,yawSeg.f):null,
  };
};
// 位姿残差：x/y 用米、yaw 用弧度加权 2（≈半个车长的端点位移量级）。同源时是 mm 级。
const scorePose = (bias,rows) => {
  const dx=[], dy=[], dyaw=[];
  for(const row of rows){
    const pc=interpPcPose(row.t+bias);
    if(pc.x!=null) dx.push(Math.abs(pc.x-row.x));
    if(pc.y!=null) dy.push(Math.abs(pc.y-row.y));
    if(pc.yaw!=null) dyaw.push(Math.abs(pc.yaw-row.yaw));
  }
  if(dx.length<30||dy.length<30||dyaw.length<30) return null;
  return {err:median(dx)+median(dy)+2*median(dyaw), n:dx.length};
};
// 门槛 0.02 是跨 97 场实测切出来的：位姿真同源时残差 0.0002~0.02，此时形状锁与 z 精锁
// 的差恒在 0.29s 以内（就是 /pc_car_loc→bot_state 的管线延迟）；残差 ≥0.023 的场次
// （老场 RK 自算位姿 / 车全程静止）差值立刻跳到 0.5~6s，必须弃用。
const POSE_LOCK_MAX_ERR=0.02;
const POSE_LOCK_HALF_WIN=1.0;   // 实测最大 0.289s，留 3 倍余量；抛球间隔 ≥8s 不会混叠
const poseLock = (()=>{
  const empty={bias:null,err:null,n:0,span:null,usable:false};
  if(rkPoseRows.length<50||pcCarRows.length<50||pcCarYawRows.length<50) return empty;
  const step=Math.max(1,Math.round(rkPoseRows.length/200));
  const sub=rkPoseRows.filter((_,i)=>i%step===0);
  const lo=pcCarRows[0].t-rkPoseRows[rkPoseRows.length-1].t-1;
  const hi=pcCarRows[pcCarRows.length-1].t-rkPoseRows[0].t+1;
  const cands=[];
  for(let bias=lo;bias<=hi;bias+=0.1){
    const s=scorePose(bias,sub);
    if(s) cands.push({bias,...s});
  }
  if(!cands.length) return empty;
  let best=cands[0];
  for(const cand of cands) if(cand.err<best.err) best=cand;
  // 近优带宽度：车全程近静止时位姿在几十秒里都对得上，带会很宽。它不影响最优点的准头
  // （实测 span 20s 的场次形状锁照样准到 0.17s），只作为诊断量报出来。
  const band=cands.filter(cand=>cand.err<=Math.max(3*best.err,best.err+0.02)).map(cand=>cand.bias);
  const span=Math.max(...band)-Math.min(...band);
  for(const [width,stepSize] of [[0.1,0.01],[0.01,0.002]]){
    let local=best;
    for(let bias=best.bias-width;bias<=best.bias+width+1e-9;bias+=stepSize){
      const s=scorePose(bias,sub);
      if(s&&s.err<local.err) local={bias,...s};
    }
    best=local;
  }
  return {bias:best.bias,err:best.err,n:best.n,span,usable:best.err<=POSE_LOCK_MAX_ERR};
})();

// 给定固定比例时间映射 PC t = RK t + bias，只比较每抛 z 形状；
// 每抛独立去除固定高度偏差，再对各抛误差取中位数，长轨迹不会压过短轨迹。
const scoreTimeMap = (bias,rows) => {
  const matches=pcFlights.map(()=>({dz:[], first:null, last:null}));
  let flightIdx=0;
  for(const row of rows){
    const t=row.t+bias;
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
// 单抛 PC 取样相位：global bias 只负责把同一抛的 PC/RK 轨迹配成一对；z(t) 只在
// global baseline 上求一个 |delta|<100ms 的小修正。RK 字段不改；所有 PC 取样（含视觉）加 delta。
const THROW_PHASE_MAX_OFFSET_S=0.10;
const THROW_PHASE_RIVAL_GAP_S=0.01;
const scoreFlightPhase = (pcFlight,rkRows,pcMinusRk) => {
  const dz=[], times=[];
  for(const row of rkRows){
    const t=row.t+pcMinusRk;
    const z=interpPcVal(pcFlight,t,'z',0.08);
    if(z==null) continue;
    dz.push(z-row.v);
    times.push(t);
  }
  if(dz.length<5 || times[times.length-1]-times[0]<0.2) return null;
  const zOffset=median(dz);
  const residuals=dz.map(value=>Math.abs(value-zOffset)).sort((a,b)=>a-b);
  const trimN=Math.max(1,Math.ceil(0.9*residuals.length));
  const trimmed=residuals.slice(0,trimN);
  const p90=residuals[trimN-1];
  const trimmedRmse=Math.sqrt(trimmed.reduce((sum,value)=>sum+value*value,0)/trimmed.length);
  return {err:median(residuals),p90,trimmedRmse,n:dz.length,
          span:times[times.length-1]-times[0],zOffset};
};
const estimateFlightPhase = (pcFlight,rkRows,baselineBias) => {
  const empty={offsetS:null,err:null,p90:null,trimmedRmse:null,n:0,coverage:0,
               span:null,zOffset:null,margin:null,profileWidth:null,edge:false,usable:false};
  if(!pcFlight||!rkRows||!isNum(baselineBias)) return empty;
  const coarse=[];
  for(let offsetS=-THROW_PHASE_MAX_OFFSET_S;
      offsetS<=THROW_PHASE_MAX_OFFSET_S+1e-9;offsetS+=0.001){
    const s=scoreFlightPhase(pcFlight,rkRows,baselineBias+offsetS);
    if(s) coarse.push({offsetS,...s});
  }
  if(!coarse.length) return empty;
  let best=coarse.reduce((a,b)=>b.err<a.err?b:a,coarse[0]);
  const fineLo=Math.max(-THROW_PHASE_MAX_OFFSET_S,best.offsetS-0.003);
  const fineHi=Math.min(THROW_PHASE_MAX_OFFSET_S,best.offsetS+0.003);
  for(let offsetS=fineLo;offsetS<=fineHi+1e-9;offsetS+=0.0001){
    const s=scoreFlightPhase(pcFlight,rkRows,baselineBias+offsetS);
    if(s&&s.err<best.err) best={offsetS,...s};
  }
  const rival=coarse.filter(c=>Math.abs(c.offsetS-best.offsetS)>=THROW_PHASE_RIVAL_GAP_S)
    .reduce((m,c)=>(!m||c.err<m.err)?c:m,null);
  const margin=rival?rival.err/Math.max(1e-9,best.err):null;
  const profile=coarse.filter(c=>c.err<=best.err+0.005).map(c=>c.offsetS);
  const profileWidth=profile.length?Math.max(...profile)-Math.min(...profile):null;
  const maxN=coarse.reduce((value,c)=>Math.max(value,c.n),0);
  const coverage=maxN?best.n/maxN:0;
  const edge=Math.abs(best.offsetS)>=THROW_PHASE_MAX_OFFSET_S-0.00005;
  const usable=best.err<=0.025 && best.p90<=0.10 && best.trimmedRmse<=0.04 &&
    best.n>=8 && coverage>=0.60 && best.span>=0.25 && !edge &&
    margin!=null && margin>=1.50 && profileWidth!=null && profileWidth<=0.02;
  return {...best,coverage,margin,profileWidth,edge,usable};
};
// 搜索窗来源优先级：时钟桥 > 位姿形状锁 > 精确值锚 > 全场扫描。前三者都是独立于球的
// 证据，有它们时 z 只干自己擅长的事（精锁）；退到全场扫描时 z 必须自证唯一（见 margin）。
const windowSource = clockBridge.bias!=null ? 'bridge'
  : (poseLock.usable ? 'pose' : (clockAnchor.bias!=null ? 'anchor' : 'scan'));
const estimateTimeMap = () => {
  const requiredFlights=windowSource==='scan'?3:2;
  const empty={scale:1,bias:null,err:null,n:0,flights:0,
    anchors:clockAnchor.anchors,anchorMad:clockAnchor.mad,requiredFlights,
    windowSource,poseErr:poseLock.err,poseBias:poseLock.bias,
    bridgeBias:clockBridge.bias,margin:null};
  if(rkMovZ.length<30 || rkMovZCoarse.length<15 || pcFlights.length<requiredFlights) return empty;
  const center=windowSource==='bridge'?clockBridge.bias
    :(windowSource==='pose'?poseLock.bias:clockAnchor.bias);
  const half=windowSource==='scan'?0.75
    :(windowSource==='anchor'?0.75:POSE_LOCK_HALF_WIN);
  const lo=center==null
    ? Math.floor(pcRows[0].t-rkMovZ[rkMovZ.length-1].t)-1
    : center-half;
  const hi=center==null
    ? Math.ceil(pcRows[pcRows.length-1].t-rkMovZ[0].t)+1
    : center+half;
  const coarse=Math.max(0.005,Math.min(0.05,(hi-lo)/18000));
  // 窗由位姿/锚定死时用全部 RK 点粗扫（几百个候选，便宜）；只有退到全场 ±百秒扫描时
  // 才用每抛抽 16 点的 rkMovZCoarse。抽样会让「每抛匹配点数」变得很脏，见下面的覆盖门。
  const narrow=center!=null;
  const scanRows=narrow?rkMovZ:rkMovZCoarse;
  const cands=[];
  for(let bias=lo; bias<=hi+1e-4; bias+=coarse){
    const s=scoreTimeMap(bias,scanRows);
    if(s) cands.push({scale:1,bias,...s});
  }
  if(!cands.length) return empty;
  // 覆盖门（抛数 ≥0.6×最大）是防全场扫描里「只靠少量重叠样本打分」的假谷用的
  // （[[v03-rk-pc-report-align]] 坑3）。窗已经由独立证据定死时它只会帮倒忙：
  // 0812_052638 场真解 −11.75 在抽样粗扫下只凑够 3 抛（满打是 7 抛）而被这道门毙掉，
  // 冠军让给了窗边缘的 −10.95。有独立窗时只保留最低抛数要求。
  const flightMax=cands.reduce((m,c)=>Math.max(m,c.flights),0);
  const minFlights=narrow?requiredFlights:Math.max(requiredFlights,Math.ceil(0.6*flightMax));
  const ok=cands.filter(c=>c.flights>=minFlights);
  if(!ok.length) return empty;
  const coarseBest=ok.reduce((a,b)=>b.err<a.err?b:a,ok[0]);
  // 混叠自证：离冠军 ≥2s 之外的最好候选如果咬得很近，说明可能锁在「错一抛」上
  // （[[v03-rk-pc-report-align]] 坑3：假谷 22mm 比真谷 28mm 还低）。窗是位姿/锚给的
  // 时这个量没有意义（窗内本来就装不下另一抛），只在全场扫描时当质量门用。
  const rival=ok.filter(c=>Math.abs(c.bias-coarseBest.bias)>=2)
    .reduce((m,c)=>(!m||c.err<m.err)?c:m,null);
  const margin=rival?rival.err/Math.max(1e-9,coarseBest.err):null;
  const win=Math.max(0.04,coarse*3);
  let best=null;
  for(let bias=coarseBest.bias-win; bias<=coarseBest.bias+win+1e-4; bias+=0.0002){
    const s=scoreTimeMap(bias,rkMovZ);
    if(s && s.flights>=minFlights && (!best || s.err<best.err)) best={scale:1,bias,...s};
  }
  if(!best) return empty;
  return {...best,anchors:clockAnchor.anchors,anchorMad:clockAnchor.mad,requiredFlights,
    windowSource,poseErr:poseLock.err,poseBias:poseLock.bias,
    bridgeBias:clockBridge.bias,margin};
};
// [[align-core-end]]
const auto = RK ? estimateTimeMap() : {
  scale:1,bias:null,err:null,n:0,flights:0,anchors:0,anchorMad:null,requiredFlights:3,
  windowSource:'scan',poseErr:null,poseBias:null,bridgeBias:null,margin:null,
};
const presetBias = isNum(D.rk_time_bias_preset) ? Number(D.rk_time_bias_preset) : null;
// z 精锁失败时退到粗定位的 bias（时钟桥差几十 ms、位姿形状锁差 0.1~0.3s 管线延迟），
// 而不是退到 0——退 0 等于把 RK 整条轨迹平移几十秒，页面上全是无意义的曲线。
const fallbackBias = isNum(auto.bias) ? auto.bias
  : (!RK ? null
    : (clockBridge.bias!=null ? clockBridge.bias
      : (poseLock.usable ? poseLock.bias : null)));
let rkBias=Math.round((presetBias!=null?presetBias:(isNum(fallbackBias)?fallbackBias:0))*10000)/10000;
// global baseline 只负责统一页面轴；逐抛 zPhase 绝不改写 RK 原始事件。
const rkToPc = t => isNum(Number(t)) ? Number(t)+rkBias : null;
// RK 轴是否真的对齐到了 PC（预置/精锁/时钟桥/位姿锁任一给出 bias）。全缺时 rkBias=0 只是
// 「原样平移」，把 RK 数字标成 PC 是撒谎——显示侧须退回 RK 记法（0817 用户定：Car Move
// 面板时间统一 PC 主显，逐帧侧栏仍保留 t(RK) 行供对 bag）。
let rkAxisAligned = presetBias!=null || isNum(fallbackBias);
const publishRkTimeMap = () => {
  window.__rkTimeMap={scale:1,bias:rkBias};
};
publishRkTimeMap();
window.__dbgAlign={
  pcRows:()=>pcRows,
  rkMovZ:()=>rkMovZ,
  rkPoseRows:()=>rkPoseRows,
  clockBridge:()=>clockBridge,
  clockAnchor:()=>clockAnchor,
  poseLock:()=>poseLock,
  scorePose:(bias)=>scorePose(bias,rkPoseRows),
  scoreTimeMap:(bias)=>scoreTimeMap(bias,rkMovZ),
  auto:()=>auto,
};
// 有外部 bias 预置时由操作者承担锚点来源；否则必须同时通过轨迹形状、点数和抛数质量门。
// 另加混叠门：搜索窗来自全场扫描（既无位姿形状锁也无精确值锚）时，冠军必须明显优于
// 2s 外的次优，否则就是「错一抛」的假谷（坑3）。
// 点数下限同理分档：30 点是给全场扫描防「少量重叠样本凑出的假谷」用的，窗已由独立
// 证据定死时 15 点足够（0809_033823：位姿残差 0.0003、z 误差 0.0066m 却只有 24 点）。
const alignBad = !!RK && presetBias==null &&
  (auto.bias==null || auto.err==null || auto.err>0.08 ||
   auto.n<(auto.windowSource==='scan'?30:15) ||
   auto.flights<auto.requiredFlights ||
   (auto.windowSource==='scan' && auto.margin!=null && auto.margin<1.35));
// 逐抛配对必须先有可信 baseline 身份：外部/独立锚可直接定窗；纯全场 scan 则必须
// 完整通过全局质量门，不能因为返回了一个 finite bias 就放行同形状的错 flight。
let throwBaselineTrusted = presetBias!=null || clockBridge.bias!=null || poseLock.usable ||
  clockAnchor.bias!=null || !alignBad;
const windowSourceLabel = {bridge:'录制时时钟桥', pose:'位姿形状锁', anchor:'精确值锚', scan:'全场扫描'};
const alignWarnHtml = alignBad
  ? `<div style="border:1px solid #e94560;background:rgba(233,69,96,0.12);color:#e94560;font-weight:600;border-radius:8px;padding:8px 12px;margin:0 0 10px">`+
    `⚠ PC↔RK 全局自动对齐不可信——总览叠加不可靠；只有 baseline 身份可信的抛球才尝试PC取样 zPhase，质量不过门隐藏所有依赖该PC取样的格（含视觉），RK字段不受影响。<br>`+
    `<span style="font-weight:400">`+
    `① 粗定位：时钟桥 ${clockBridge.bias!=null?`✓ bias ${clockBridge.bias.toFixed(3)}s（${clockBridge.n} 样本）`
      :`✗ ${clockBridge.n?`抖动 ${clockBridge.mad==null?'n/a':clockBridge.mad.toFixed(3)}s 过大`:'本场未录（0812 前的场次没有这个字段）'}`}；`+
    `位姿形状锁 ${poseLock.usable?`✓ bias ${poseLock.bias.toFixed(3)}s（残差 ${poseLock.err.toFixed(4)}）`
      :`✗ 残差 ${poseLock.err==null?'n/a':poseLock.err.toFixed(4)}>${POSE_LOCK_MAX_ERR}（两侧 bot_* 不同源或小车全程静止）`}；`+
    `精确值锚 ${clockAnchor.bias!=null?`✓ ${clockAnchor.anchors} 条`:`✗ ${clockAnchor.anchors} 条`}；`+
    `实际用了「${windowSourceLabel[auto.windowSource]||auto.windowSource}」。<br>`+
    `② z 形状精锁：误差 ${auto.err==null?'n/a':auto.err.toFixed(3)+'m'} / ${auto.n} 点 / ${auto.flights||0} 抛`+
    `（需 ≤0.080m / ≥30 点 / ≥${auto.requiredFlights} 抛`+
    (auto.windowSource==='scan'&&auto.margin!=null?`；2s 外次优比 ${auto.margin.toFixed(2)}× 需 ≥1.35`:'')+
    (auto.flights===0?`；本场 PC 侧根本没有可用抛球，z 精锁无从谈起`:'')+`）。<br>`+
    `处置：`+(isNum(fallbackBias)
      ? `已按${clockBridge.bias!=null?'时钟桥':'位姿形状锁'}把时轴退到 ${rkBias.toFixed(4)}s`
        +`（准到 ${clockBridge.bias!=null?'几十 ms':'0.1~0.3s 管线延迟量级'}，够看轨迹`
        +`${clockBridge.bias!=null?'，北极星表的 ms 级口径仍建议人工复核':'不够做北极星表'}）；`
        +`RK 页 Apply time bias 在此基础上微调即可。`
      : `RK 页手动 Apply time bias，或用 --rk-time-bias 预置外部锚。`)+
    `</span></div>`
  : (RK
    ? `<div style="font-size:11px;color:#7fbf9f;margin:0 0 6px">PC↔RK 对齐 ✓ `+
      `PC t = RK t + ${rkBias.toFixed(4)}s（scale 固定为 1）`+
      (presetBias!=null
        ? `（--rk-time-bias 外部锚预置）`
        : `（粗定位 ${windowSourceLabel[auto.windowSource]||auto.windowSource}`+
          (auto.windowSource==='bridge'?` ${clockBridge.n} 样本/抖动 ${clockBridge.mad==null?'n/a':clockBridge.mad.toFixed(3)}s`:'')+
          (auto.windowSource==='pose'?` 残差 ${poseLock.err.toFixed(4)}`:'')+
          (auto.windowSource==='anchor'?` ${auto.anchors} 条`:'')+
          (auto.windowSource==='scan'&&auto.margin!=null?` 次优比 ${auto.margin.toFixed(2)}×`:'')+
          `；z 形状 ${auto.err.toFixed(3)}m / ${auto.n} 点 / ${auto.flights} 抛）`)+
      `——此全局 bias 供所有RK显示；每抛 zPhase 给所有PC取样（含视觉）加一个小offset。</div>`
    : '');
const rkPredStage = RK ? ys(RK.pred,'stage') : [];
const rkPredDurMs = (RK ? ys(RK.pred,'duration') : []).map(v=>isNum(v)?v*1000:null);
const rkPredNFit = RK ? ys(RK.pred,'n_bounce_fit') : [];
const rkPredRacketVz = RK ? ys(RK.pred,'rvz') : [];
const rkPredCorXyEff = RK ? ys(RK.pred,'cor_xy_eff') : [];
const rkPredCorEff = RK ? ys(RK.pred,'cor_eff') : [];
const rkPredCorMeasReplay = RK ? ys(RK.pred,'cor_meas_replay') : [];
const rkPredCxyMeasReplay = RK ? ys(RK.pred,'cxy_meas_replay') : [];
const rkPredCorMeasClosureMs = RK ? ys(RK.pred,'cor_meas_closure_ms') : [];
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
      th.ht=ht[i]; th.lastT=ti; th.lastIdx=i; th.msgs=(th.msgs||0)+1;
      if(Number(rkPredStage[i])===0 && isNum(worldY[i])){
        th.lastS0Y=worldY[i]; th.lastS0T=ti; th.lastS0Idx=i;
        th.lastS0Rvz=isNum(rkPredRacketVz[i])?rkPredRacketVz[i]:null;
        th.lastS0CorXyEff=isNum(rkPredCorXyEff[i])?rkPredCorXyEff[i]:null;
        th.lastS0CorEff=isNum(rkPredCorEff[i])?rkPredCorEff[i]:null;
      } else if(Number(rkPredStage[i])===1){
        th.hasS1=true;
        if(!isNum(th.corMeasReplay) && isNum(rkPredCorMeasReplay[i]) &&
           isNum(rkPredCxyMeasReplay[i])){
          th.corMeasReplay=rkPredCorMeasReplay[i];
          th.cxyMeasReplay=rkPredCxyMeasReplay[i];
          th.corMeasClosureMs=isNum(rkPredCorMeasClosureMs[i])?rkPredCorMeasClosureMs[i]:null;
          th.corMeasIdx=i;
        }
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
      const th={ht:ht[i], firstT:ti, lastT:ti, firstIdx:i, lastIdx:i, msgs:0, stage:null, rel_x:null, rel_y:null, rel_z:null, lastRelIdx:null, refT:null,
                ref300Stage:null, ref300X:null, ref300Y:null, ref300Z:null, ref300T:null, ref300Ht:null,
                ref300Xw:null, ref300CarX:null, ref300CarY:null,
                ref300Lead:null, ref300LeadDev:null, ref300NFit:null, ref300Idx:null,
                lastS0Y:null, lastS0T:null, lastS0Idx:null,
                lastS0Rvz:null, lastS0CorXyEff:null, lastS0CorEff:null,
                corMeasReplay:null,cxyMeasReplay:null,corMeasClosureMs:null,corMeasIdx:null,
                hasS1:false};
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
// 入弧拟合的有界离群剔除：一个坏 3D 点就把整抛判死太脆——0811 080158 实测 19 抛里
// 有 5 抛是这么丢的（#9/#14/#15/#17/#18），而剔掉 1~2 点后 max|残差| 落在 0.3~5cm，
// 不是勉强过门。坏点来源是多球关联失败（一台相机锁到场上另一颗球，三角化到两球之间）
// 和弹跳接触帧污染，两者都不是"球的测量"，剔掉是对的。
// 为了不变成"剔到过门为止"，三条硬约束：
//   ① 最多剔 2 个；② 剔完至少剩 8 点；③ 最坏点必须显著突出（≥2.5× 中位残差）——
//   同 car_localizer 的 _VIEW_OUTLIER_RATIO 口径。整体噪声顶在门边的抛剔不动，仍判失败。
// 剔过点的行在单元格里显式标 "剔N点"，不藏在 tooltip 里。
const PC_TRUTH_MAX_DROP=2;
const PC_TRUTH_MIN_KEEP=8;
const PC_TRUTH_OUTLIER_RATIO=2.5;
// 窗长稳定性自检用的短窗回溯长度 (s)：够拟合又明显短于 0.75s 全窗。
const PC_TRUTH_STAB_SPAN=0.40;
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
// winSpan：窗口回溯长度，默认 0.75s。内部会用 PC_TRUTH_STAB_SPAN 再算一次做
// 窗长稳定性自检（见下方 eStab），那一次递归传参进来，不再自检。
const pcTruthAt = (tPc, winSpan) => {
  const span=winSpan||0.75;
  const c=carAt(tPc);
  if(!c) return null;
  let win=pcRows.filter(p=>tPc-p.t>=0.02 && tPc-p.t<=span);
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
  // 贴地观测（z<0.12）无论如何都不该进入入弧拟合：可能是场上静止球或弹跳采样点。
  // ⚠ 但"连它之前的一起砍"只在它落在窗口**前段**时才成立。0811 080158 #17 是
  // 「球直接扎向地面」的抛（HT−31ms 时 z=0.06，臂也因 z 越下限拒了），贴地点落在
  // 窗尾，旧写法把 loT 推到 −31ms → 整窗 15 点被清空 → 真值判失败；实测只把那一个
  // 点剔掉、剩下 14 点拟合 max|残差| 仅 0.6cm。所以：砍完还剩得下才连前面一起砍，
  // 否则只剔贴地点本身。前段贴地（原本就想治的场景）两条路等价，行为不变。
  let loZ=-Infinity;
  win.forEach(p=>{ if(p.z<0.12 && p.t>loZ) loZ=p.t; });
  if(loZ>loT && win.filter(p=>p.t>loZ).length>=5) loT=loZ;
  win=win.filter(p=>p.t>loT && p.z>=0.12);
  const reg=(ts,vs)=>{
    const n=ts.length;
    let st=0,sv=0,stt=0,stv=0;
    for(let i=0;i<n;i++){ st+=ts[i]; sv+=vs[i]; stt+=ts[i]*ts[i]; stv+=ts[i]*vs[i]; }
    const den=n*stt-st*st;
    return Math.abs(den)<1e-9 ? null : [(sv*stt-st*stv)/den, (n*stv-st*sv)/den];
  };
  const dropped=[];          // 被剔掉的离群点 {dt, r}，进 tooltip 如实公示
  let nSplit=0;              // 断档切次数，预算与改动前一致（2 次）
  for(let attempt=0; attempt<3+PC_TRUTH_MAX_DROP; attempt++){
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
    // 逐点残差按各自门限归一（z/0.035、x/0.05），两轴才可比：>1 即该点单独顶破了门
    const rNorm=win.map((p,i)=>{
      const dz=Math.abs(zAt(ts[i])-p.z), dx=Math.abs(fx[0]+fx[1]*ts[i]-p.x);
      zMax=Math.max(zMax,dz); xMax=Math.max(xMax,dx);
      return Math.max(dz/0.035, dx/0.05);
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
      const out={x:fx[0]-c.x, y:fy[0]-c.y, z:zAt(0), gap, resMax, dNear, delta,
                 carGap, carGa, carGb, eCar, carSingleTag:!!c.single,
                 nPts:n, dropped, eStab:0};
      // 窗长稳定性自检：换成只回溯 PC_TRUTH_STAB_SPAN 的短窗重拟合，取三轴最大差
      // 当作模型失配的实测值并进误差棒。0811 两场实测：老行（长短窗都好拟）这项
      // 中位 1.1cm、最大 2.1cm，本来就落在原误差棒内；而弹跳后才有干净段的抛
      // （#14/#15/#17）这项能到 2.9~5.2cm，远超原来标的 ±0.8~2.4cm ——
      // eModel=0.75·d² 只算了到最后一点的外推，没算"用 0.7s 长窗拟合本身"的失配。
      // 不做这一项，新救回的行就会顶着一个盖不住的棒子对外发布。
      if(!winSpan){
        const s=pcTruthAt(tPc, PC_TRUTH_STAB_SPAN);
        if(s) out.eStab=Math.max(Math.abs(s.x-out.x),Math.abs(s.y-out.y),Math.abs(s.z-out.z));
      }
      // 显示误差下限取 max|残差|：统计上均值估计可优于单点散布，但对外宣称
      // 不应低于模型对数据的实际解释能力（防"跨 0.3s 断档标 ±1mm"式过度自信）
      out.err=Math.max(
        Math.sqrt(seMax*seMax+eModel*eModel+eCar*eCar+out.eStab*out.eStab), resMax);
      return out;
    }
    // ① 断档切优先（预算与改动前一致：2 次）：只保留最靠近目标时刻的连续入弧。
    // 顺序不能反 —— 大断档意味着更早那段很可能压根是另一颗球，整段丢弃比逐点剔
    // 更有物理依据；而且这样离群剔除纯粹是"老逻辑走到死路后"的追加手段，
    // 改动前能出值的行走的还是原来那条路，数值逐位不变（两场已验证）。
    let gi=-1, gmax=0;
    for(let i=1;i<win.length;i++){
      const g=win[i].t-win[i-1].t;
      if(g>gmax){ gmax=g; gi=i; }
    }
    if(gi>=0 && gmax>=0.1 && nSplit<2){ nSplit++; win=win.slice(gi); continue; }
    // ② 断档切用尽仍不过门，才试有界离群剔除：只剔一个**显著突出**的点，
    // 且剔完还得剩够点。三条约束都不满足就老老实实判失败。
    if(dropped.length<PC_TRUTH_MAX_DROP && win.length-1>=PC_TRUTH_MIN_KEEP){
      let wi=0;
      for(let i=1;i<rNorm.length;i++) if(rNorm[i]>rNorm[wi]) wi=i;
      const sorted=[...rNorm].sort((a,b)=>a-b);
      const med=sorted[sorted.length>>1];
      if(rNorm[wi]>1 && rNorm[wi]>=PC_TRUTH_OUTLIER_RATIO*Math.max(med,1e-9)){
        dropped.push({dt:win[wi].t-tPc, r:rNorm[wi]});
        win=win.slice(0,wi).concat(win.slice(wi+1));
        continue;
      }
    }
    return null;
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
  return '入弧拟合未过残差门（切掉弹跳/断档后点数不足，或 max|残差| 超 z 3.5cm / x 5cm）；'+
         '有界离群剔除已试过仍不过——说明不是一两个坏点，而是整段散布顶在门边或窗内混了两颗球';
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
// 完全同轴 —— 全项目只有 PC/RK 两个时间轴；RK显示与臂数据只走全局 rkToPc，
// 所有逐抛 PC 取样查询（含视觉）额外使用同抛 scale=1 zPhase offset。
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
// 每条 accepted 回配它实际消费的那条 /predict_hit_pos，并统计臂端 z 偏移 zOff = accepted_z − rel_z。
// [[arm-prediction-match-core-begin]]
// 回配三把键，从强到弱：
//   ① 序号键（结构键，值域变换免疫）：on_hit_pos 收一条必回一条，故 hit_pos 派生状态
//      （accepted / late ht saved / reject / error）与 /predict_hit_pos 严格一对一、各自保序，
//      序号差 δ=si−pi 全场恒定。实现见 [[arm-pred-align-core]]。
//   ② 时序键 |ht−(accept_t+duration)|<30ms：防跨抛误配（同抛内相邻消息 ht 只差 ms 级，
//      不具判别力）。系统差 = 臂内提前量 − 状态发布开销（0802/0803晨 ~+9ms、0803夜 ~+14ms、
//      0804 起 ~−1ms），5ms 容差会把全场回配打成 0（073646/064744 实测），故放宽到 30ms。
//   ③ 值键 x/z：**只作序号键不可用时的兜底与残差自检，不再当主键**。臂端会在 receive_hit
//      顶部对目标做单点变换，改一次就整场失配，且报告这边没有任何独立线索能察觉：
//      · 0811 11:37 场 kHitYawExtraRad=5° 起 x/=cos5°（+3.8mm@1m）→ 115/115 全灭；
//      · 0804 场 z 偏移 −0.153→−0.164 → 80 条命中 0。
//      两个量都由 armConstCal 逐场自标定，这里只留缺省值。
// ⚠ 值键还有个先天缺陷（正是它不配当主键的第二个理由）：同一抛内相邻两条预测的 rel_x/rel_z
//   常常只差 0.2mm，5e-4 容差分不开，就近搜会选中晚到的那条（0804/0808/0811 实测各 1~2 条）。
//   序号键没有这个模糊性。
let ARM_HIT_Z_OFFSET=-0.164;   // config.HIT_POS_Z_OFFSET：2026-08-04 起 −0.164（此前 −0.153）
let ARM_HIT_X_SCALE=1.0;       // receive_hit 顶部 x/=cos(kHitYawExtraRad)：2026-08-11 起 1/cos5°≈1.00382
const ARM_PRED_ARRIVE_MAX_SEC=0.25;   // 消息到达 → 状态发布的最大间隔（实测中位 ~50ms）
const ARM_PRED_HT_TOL_SEC=0.03;
const armPredictionMatchesAccepted = (p,acceptT,acceptX,acceptZ,acceptDuration) => {
  const zOffset=p.relSrc==='panel'?0:ARM_HIT_Z_OFFSET;
  return Math.abs(p.rel_x*ARM_HIT_X_SCALE-acceptX)<5e-4 &&
    Math.abs(p.rel_z+zOffset-acceptZ)<5e-4 &&
    Math.abs(p.ht-(acceptT+acceptDuration))<ARM_PRED_HT_TOL_SEC;
};
// [[arm-prediction-match-core-end]]
// arm_controller 常量（config.py）：臂内触球 = 原始 ht − HIT_TIME_ADVANCE_SEC。
// 2026-08-04 起归零——方向修正统一收敛到 SWING_J1_LEAD_SEC 的 J1 角度提前（不动时间轴），
// 该量不再兼职修拍面 yaw；历史值 10ms（0802~0803晨）/15ms（0803夜）。同样只作缺省，
// 实际值由 armConstCal 从本场 accepted 自标定。
let HIT_TIME_ADVANCE_SEC=0.0;
// 挥拍中 ht 重定相：当前连续 J1 sweep 的 `late ht saved ... sweep_w=... mode=...` 只表示
// update_ht() 收到了该 raw HT；每条内部只夹 ±50ms，下一控制 tick 才重解 profile。mode=2 明确
// 表示无可行解、继续骑旧 profile，因此 raw last-saved 只能用于预测/盲区账，不能当 TCP 执行接触锚。
// 旧控制器的无 sweep_w 状态只保存候选值，仍按老触球−100ms 的一次性触发门重建。
// 两种状态都与 /predict_hit_pos 一一对应。连续 sweep 可以合法地把触球推到老 done 之后，
// 因此它必须保留到当前 hit 的最后一条；无 sweep_w 的旧状态仍用 done+50ms 约束归属。
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
// [[arm-pred-align-core-begin]]
// 序号对齐：hit_pos 派生状态与 /predict_hit_pos 的序号差 δ=si−pi。
// 投票只用两把弱键（到达窗 + 时序键）：同一抛内相邻消息也会投票，但每条 accepted 的正确源
// 只有一个序号，正确 δ 每拍稳拿一票、错票按序号散开，众数即全场 δ。
// **投票里绝不能出现 x/z**——那正是臂端会改的量，见 [[arm-prediction-match-core]] 的两次事故。
// δ 恒定的前提是两个 topic 各自保序、且状态与消息不丢；跨 topic 的 bag 读出顺序错位不影响
//（同一场实测出现过 S,S,P,P）。真丢消息时 δ 会中途跳变，故每条配对还要过一遍弱键自校验
//（armPredForStatus），过不了就地失配、不硬套。
// 实测：0716~0811 共 60 场 bag，状态数≡消息数、δ≡0（唯一例外 0719 100823 那场早期 bag）。
const armHitStatuses = ((ARM&&ARM.events)||[])
  .filter(e=>e.topic==='/tennis/status' &&
    /^(accepted hit |late ht saved|reject hit:|error hit_pos)/.test(String(e.text||'')))
  .map((e,si)=>({si, t:e.t, text:String(e.text)}));
const armAcceptedHitRe=/^accepted hit x=([\-0-9.]+) z=([\-0-9.]+) duration=([0-9.]+)/;
const armPredAlign = (()=>{
  const t0=(RK&&isNum(RK.t0))?RK.t0:0;
  const votes=new Map(); let nAcc=0, total=0;
  armHitStatuses.forEach(s=>{
    const m=armAcceptedHitRe.exec(s.text);
    if(!m) return;
    nAcc+=1;
    const at=s.t+t0, dur=Number(m[3]);
    armPreds.forEach((p,pi)=>{
      if(!(p.t<=s.t && s.t-p.t<=ARM_PRED_ARRIVE_MAX_SEC)) return;
      if(!isNum(p.ht) || Math.abs(p.ht-at-dur)>=ARM_PRED_HT_TOL_SEC) return;
      votes.set(s.si-pi,(votes.get(s.si-pi)||0)+1);
      total+=1;
    });
  });
  let delta=null, n=0;
  votes.forEach((v,k)=>{
    if(v>n || (v===n && delta!=null && Math.abs(k)<Math.abs(delta))){ n=v; delta=k; }
  });
  // 认票门槛：≥3 票**且**≥6 成 accepted 投它。健康场是 100%（正确 δ 每条 accepted 都投），
  // 所以这道门只拦「1:1 不成立」的场——0719 100823 那场 bag 少收 11 条状态，δ=−7 只拿 4/9 票，
  // 却仍能骗过逐条弱键自校验（把 9 条回配打成 4 条、z 标定带歪 13mm）。这种场整场退回值键。
  const ok=(n>=3 && n>=0.6*nAcc);
  return {delta:ok?delta:null, n, total, nAcc};
})();
// 按序号取源消息，再过两把弱键自校验。dur=null 时只校验到达窗（late ht saved 自述的是
// 剩余触球时间不是 duration，它另有 gap 自校验，见 [[arm-swing-ht-core]]）。
const armPredForStatus = (s,dur) => {
  if(armPredAlign.delta==null) return null;
  const p=armPreds[s.si-armPredAlign.delta];
  if(!p || !(p.t<=s.t && s.t-p.t<=ARM_PRED_ARRIVE_MAX_SEC)) return null;
  if(dur!=null){
    const t0=(RK&&isNum(RK.t0))?RK.t0:0;
    if(!isNum(p.ht) || Math.abs(p.ht-(s.t+t0)-dur)>=ARM_PRED_HT_TOL_SEC) return null;
  }
  return p;
};
// [[arm-pred-align-core-end]]
// [[arm-const-cal-core-begin]]
// 臂端三个量逐场自标定（写死会随控制端改版整场失效）。样本 = 序号对齐配上的 accepted↔源消息对
//（旧版拿 x 值键当筛选来自举，臂端一改 x 变换就取不到样本 → 标定与回配一起归零，
//  0811 113734 场就是这么全表 — 的；现在筛选里没有任何值键）：
//   · z 偏移 zOff = 众数(acc_z − rel_z)（0.1mm 分辨率）= config 的 HIT_POS_Z_OFFSET。
//   · x 比例 xScale = 中位(acc_x / rel_x) = 1/cos(kHitYawExtraRad)。取比例不取差值：变换是
//     乘性的，差值随 x 变（0.7~1.1m 量程上差 0.3mm，会咬掉值键 5e-4 的判别力）；acc_x 只印
//     4 位小数 → 比值量化 1e-4，故取中位不取众数。⚠ 若哪天臂端改成加性偏置，这里会退化成
//     ±20%·偏置 的残差，值键兜底随之失灵——但主键是序号键，报告不会因此空表，
//     且残差会被 armDataWarnHtml 的 okN<accN 抓出来。
//   · 提前量 adv = round(median(ht−acc_t−dur)+1ms) = 臂内提前量 − 状态发布开销（实测 0.3~1.4ms）。
//     实测 0802/0803晨 C=+8.9→10ms、0803夜 +13.9→15ms、0804 起 −1.1→0ms，与 config 历史逐场对上。
// 票数 <3 时保留缺省值（页面红条标出），不拿噪声改合同。panel 手动目标不走 z 偏移，不参与投票。
const armConstCal = (()=>{
  const t0 = (RK && isNum(RK.t0)) ? RK.t0 : 0;
  const votes=new Map(), cands=[], ratios=[];
  const take=(p,ax,az,at,dur)=>{
    if(p.relSrc==='panel') return;
    if(!isNum(p.rel_x)||!isNum(p.rel_z)||!isNum(p.ht)) return;
    const key=Math.round((az-p.rel_z)*1e4)/1e4;
    votes.set(key,(votes.get(key)||0)+1);
    cands.push({key,c:p.ht-at-dur});
    if(Math.abs(p.rel_x)>0.1) ratios.push(ax/p.rel_x);
  };
  armHitStatuses.forEach(s=>{
    const m=armAcceptedHitRe.exec(s.text);
    if(!m) return;
    const ax=Number(m[1]), az=Number(m[2]), dur=Number(m[3]), at=s.t+t0;
    const p=armPredForStatus(s,dur);
    if(p){ take(p,ax,az,at,dur); return; }
    if(armPredAlign.delta!=null) return;   // 序号对齐立得住时只信它，不掺值键样本
    // 兜底自举（序号对齐不可用的早期 bag）：老办法用 x 值键 + 时序键圈候选族，
    // 同抛相邻消息会投错票，靠 z 众数压掉。臂端一改 x 变换这条路就取不到样本 → 见上。
    armPreds.forEach(q=>{
      if(!(q.t<=s.t && s.t-q.t<=ARM_PRED_ARRIVE_MAX_SEC)) return;
      if(!isNum(q.rel_x)||!isNum(q.ht)) return;
      if(Math.abs(q.rel_x*ARM_HIT_X_SCALE-ax)>=5e-4) return;
      if(Math.abs(q.ht-at-dur)>=ARM_PRED_HT_TOL_SEC) return;
      take(q,ax,az,at,dur);
    });
  });
  let zOff=null, best=0, total=0;
  votes.forEach((n,k)=>{ total+=n; if(n>best){ best=n; zOff=k; } });
  if(best<3) return {zOff:null, xScale:null, adv:null, n:best, total, c:null};
  const cs=cands.filter(r=>r.key===zOff).map(r=>r.c).sort((a,b)=>a-b);
  const c=cs[cs.length>>1];
  const adv=Math.round(c*1000+1)/1000;
  ratios.sort((a,b)=>a-b);
  const xScale=ratios.length>=3?ratios[ratios.length>>1]:null;
  return {zOff, xScale, adv:(adv>=0&&adv<=0.03)?adv:null, n:best, total, c, nx:ratios.length};
})();
if(armConstCal.zOff!=null) ARM_HIT_Z_OFFSET=armConstCal.zOff;
if(armConstCal.xScale!=null) ARM_HIT_X_SCALE=armConstCal.xScale;
if(armConstCal.adv!=null) HIT_TIME_ADVANCE_SEC=armConstCal.adv;
// [[arm-const-cal-core-end]]
// /tennis/status 尾部的 `key=数值` 字段取值：字段随 arm_controller 版本增删（可变 pitch 0805、
// 拍速指定 0808…），老 bag 取不到就返回 null 让对应列显示 —。key 后必须紧跟 =，故取 'speed'
// 不会命中 'speed_req'。
const statusNum = (text,key) => {
  const m=new RegExp('(?:^|\\s)'+key+'=(-?[0-9]+(?:\\.[0-9]+)?)').exec(text||'');
  return m?Number(m[1]):null;
};
const _armHit = (()=>{
  if(!ARM) return {marks:[], nAcc:0, nMatch:0};
  const out=[];
  let nAcc=0, nMatch=0;
  // late ht saved 状态只写 duration、不写 ht，必须回配到原 /predict_hit_pos 才拿得到原始 ht。
  // 回配一律走序号对齐（[[arm-pred-align-core]]）：两个 topic 各自保序但**跨 topic 的 bag
  // 读出顺序会错位**（同一场实测出现过 S,S,P,P），所以不能按事件流 FIFO 配，序号差才是不变量。
  // 序号对齐立不起来时（δ 票数 <3，多见于 2026-07 早期 bag）退回老路：以值键回配成功的那条
  // accepted 为锚按序号差平移——同一次挥拍里锚点与 late 只隔 ~300ms，中间不会丢消息。
  // 每条再自校验 now=ht−advance−duration。旧状态要求 gap=发布时刻−now 落在 [−0.6,+8]ms；
  // 连续 sweep 放宽负侧到 −3ms：逐场 advance 自标定是整数 ms，且不同部署版本取值会变，
  // 加上 duration 只打印 1ms 精度，健康配对实测会到 −2.8ms。sweep_w 状态证明消息进入 update_ht，
  // 且序号键与错配一格约 32.5ms 的间隔仍远大于 3ms，故不损失判别力。
  // 全在 arm_controller 进程内、同一单调钟（status 尾缀 t= 就是它发布时读的 perf_counter），
  // 不含 DDS 与 PC：status 写进 last_status 后要等下一个 100Hz tick 才 push，故 gap 是 sub-tick
  // 抖动，上限 = 一个 tick 10ms。旧上界 4ms 按"发布开销 0.3~1.4ms"取，0809 场实测到 4.90ms，
  // 把 #6/#13 两拍判成失配、盲区列静默退回 accepted（174ms 真值显示成 327ms），故放宽到 8ms。
  // 8ms 仍远小于错配一格的 32.5ms（视觉帧间隔），判别力不受影响。
  let anchorSi=null, anchorPi=null;   // 序号对齐不可用时的兜底锚
  armHitStatuses.forEach(s=>{
    const e={t:s.t, text:s.text};
    let rec=null;
    let m=/^late ht saved: contact in ([0-9.]+)s/.exec(e.text);
    if(m){
      const cur=out[out.length-1];
      const continuousSweep=/\bsweep_w=/.test(e.text);
      if(cur && cur.label==='hit' && (continuousSweep || e.t<=cur.done+0.05)){
        const dur=Number(m[1]);
        const p=armPredForStatus(s,null) ||
          (anchorSi!=null?armPreds[anchorPi+(s.si-anchorSi)]:null);
        const gap=p?(e.t+RK.t0-(p.ht-HIT_TIME_ADVANCE_SEC-dur)):null;
        const gapMin=continuousSweep?-0.003:-0.0006;
        const ok=!!(p && gap>=gapMin && gap<=0.008);
        const ht=ok?p.ht:null;
        cur.lates.push({t:e.t, dur, ht, ct:ok?p.ct:null,
                        hitTime:ht!=null?ht-RK.t0-HIT_TIME_ADVANCE_SEC:e.t+dur,
                        continuousSweep, mode:continuousSweep?statusNum(e.text,'mode'):null});
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
             tgtOutYaw:statusNum(e.text,'out_yaw'),
             tgtOutPitch:statusNum(e.text,'out_pitch'),
             tgtOutSpeed:statusNum(e.text,'out_speed'),
             tgtOutReplay:statusNum(e.text,'out_replay')===1,
           wx:null, wy:null, wz:null, wxw:null, wcarx:null, wct:null, wht:null, wpredT:null, wstage:null, wnFit:null};
      nAcc+=1;
      let hit=armPredForStatus(s,dur);                      // ① 序号键
      // ③ 值键：序号键拿不到时兜底；拿到了也复核一遍。
      // 复核是必要的：bag 少收一条状态会把其后所有配对整体带偏一格，而偏的那条同样过得了
      // 到达窗与时序键（相邻消息只差 ~30ms/~1ms），弱键自校验看不出来。此时窗内若另有一条
      // 严丝合缝命中本场自标定值键的消息，就改用它。
      // 反过来，臂端换了目标变换时窗内**没有任何一条**能命中值键 → 保留序号键的结果，
      // 报告不会因为"值键看不懂"而整表空白——这才是主键放在序号侧的意义。
      if(!hit || !armPredictionMatchesAccepted(hit,e.t+RK.t0,rec.tx,rec.tz,dur)){
        for(let i=armPreds.length-1;i>=0;i--){
          const p=armPreds[i];
          if(p.t>e.t) continue;
          if(e.t-p.t>ARM_PRED_ARRIVE_MAX_SEC) break;
          if(armPredictionMatchesAccepted(p,e.t+RK.t0,rec.tx,rec.tz,dur)){
            hit=p; anchorSi=s.si; anchorPi=i;   // late ht saved 的序号平移锚点
            break;
          }
        }
      }
      if(hit){
        rec.wx=hit.rel_x; rec.wy=hit.rel_y; rec.wz=hit.rel_z; rec.wct=hit.ct; rec.wht=hit.ht;
        rec.wxw=isNum(hit.xWorld)?hit.xWorld:null; rec.wcarx=isNum(hit.carPredX)?hit.carPredX:null;
        rec.wpredT=hit.t; rec.wstage=hit.stage; rec.wnFit=hit.nFit;
        nMatch+=1;
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
  // 逐拍保留最后进入 update_ht() 的 raw HT/CT，供预测盲区账使用；它不是 executed profile HT。
  // TCP 列严格在这个 raw HT 取 /joint_states FK；不另找过面时刻。
  // 旧版无 sweep_w 日志仍保留一次性触发的剩余时间门。raw HT/CT 回配失败时不退 accepted 冒充。
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
    const last=lates[lates.length-1];
    const trig=h.done-SWING_HT_UPDATE_LEAD_SEC;
    // 连续 sweep 的 saved 已在 receive_hit 内 update_ht()，无需再套旧版一次性触发门。
    const continuousSweep=!!last.continuousSweep;
    const ok=continuousSweep || last.hitTime-trig>=SWING_HT_UPDATE_MIN_REMAINING_SEC;
    h.lastUpdateT=last.t;
    const nCoast=continuousSweep?lates.filter(x=>x.mode===2).length:0;
    h.reswing={trig, ok, continuousSweep, n:lates.length, nCoast,
               oldDone:h.done, newDone:last.hitTime,
               delta:(last.hitTime-h.done)*1000, remain:last.hitTime-trig,
               ht:last.ht, ct:last.ct};
    if(ok){
      h.finalDone=last.hitTime;
      if(last.ht!=null){ h.finalHt=last.ht; h.finalCt=last.ct; }  // ht/ct 必须同源
      else {                                                       // 见上：不许拿 accepted 冒充
        h.finalHt=null; h.finalCt=null; h.finalMismatch=true;
      }
    }
  });
  return {marks:out, nAcc, nMatch};
})();
// [[arm-swing-ht-core-end]]
const armHitMarks = _armHit.marks;
// 本页 TCP 用的是哪台车的 FK 链（Python _add_face_angles 按 arm JSON 的 car / 本场 tracker
// JSON 的 car_config_path 定，并按它复算过 tcp）。两台车的臂不同，这条必须显示在 TCP 列上。
const armFkCarNote = ARM
  ? ('FK 车型 '+(ARM.fk_car||'未知')+
     (ARM.fk_recomputed_from?'（本页按 '+ARM.fk_car+' 复算，arm JSON 原为 '+ARM.fk_recomputed_from+'）':''))
  : 'FK 车型 —';
// 臂数据整体失效的两种模式，以前都只在折叠备注里留一行，页面主体只剩满屏 "—"，
// 看上去像"这场没数据"而不是"报告读不了数据"。这里升成表头红条（部分失配用橙条）。
// [[arm-data-warn-core-begin]]
const armDataWarnHtml = (()=>{
  if(!ARM) return '';
  const bad=[], warn=[];
  if(armPredParseBad>0){
    bad.push('/predict_hit_pos 载荷解析失败 '+armPredParseBad+'/'+armPredTotal+' 条'+
      '——多半是 extract_arm_bag 把事件文本截断了（RK 端加字段后 payload 变长会撞上限）。'+
      '重跑 test_src/extract_arm_bag.py 出 _arm.json 后再重生成本页。');
  }
  // 回配率是臂表所有列的总闸：失配的拍 accepted 目标/击球真值/TCP 全列 —。
  // 报出三把键各自的状态，下次再出事一眼能定位是结构变了（序号键）还是口径变了（值键）。
  if(_armHit.nAcc>0 && _armHit.nMatch<_armHit.nAcc){
    const keys='序号对齐 δ='+(armPredAlign.delta!=null?armPredAlign.delta:'—')+
      '（'+armPredAlign.n+'/'+armPredAlign.nAcc+' 票）；本场自标定 x×'+
      (armConstCal.xScale!=null?armConstCal.xScale.toFixed(5):'—')+'、z'+
      (armConstCal.zOff!=null?armConstCal.zOff.toFixed(4):'—')+'m、提前量'+
      (armConstCal.adv!=null?(armConstCal.adv*1000).toFixed(0)+'ms':'—');
    (_armHit.nMatch?warn:bad).push('本场 '+_armHit.nAcc+' 条 accepted hit 只回配上 '+
      _armHit.nMatch+' 条原始 /predict_hit_pos（'+keys+'）——失配那些拍的 accepted 目标、'+
      '击球真值、TCP 各列为 —。');
  }
  const box=(color,bg,head,msgs)=>msgs.length?('<div style="border:1px solid '+color+';'+
    'background:'+bg+';color:'+color+';font-weight:600;border-radius:8px;'+
    'padding:8px 12px;margin:0 0 10px">'+head+msgs.join('　')+'</div>'):'';
  return box('#e94560','rgba(233,69,96,0.12)','⚠ 臂数据读取失败：',bad)+
    box('#e0a24a','rgba(224,162,74,0.12)','⚠ 臂数据部分失配：',warn);
})();
// [[arm-data-warn-core-end]]
const armTcpRows = ARM ? ARM.states.filter(s=>Array.isArray(s.tcp)) : [];
// 在机械臂 state 时间轴按 HT 有界插值 FK TCP；相邻 state 超过 100ms 时不外推。
const tcpAt = t => {
  const s=interpRow(armTcpRows,t,0.1);
  if(!s || !Array.isArray(s.a.tcp) || !Array.isArray(s.b.tcp)) return null;
  return [0,1,2].map(k=>lerp(s.a.tcp[k],s.b.tcp[k],s.f));
};
// [[arm-point-world-core-begin]]
// x/y 按给定车 yaw 旋到世界轴；z 从 FK 安装面零点平移到机械臂中心地面点 z=0。
// zArmMinusWorld = 臂模型 z − 世界/地面 z（V04 为负安装高度）。
const rotateBodyVectorWorld = (v,yawDeg) => {
  if(!Array.isArray(v)||v.length!==3||!v.every(isNum)||!isNum(yawDeg)) return null;
  const a=yawDeg*Math.PI/180;
  return [v[0]*Math.cos(a)-v[1]*Math.sin(a),
          v[0]*Math.sin(a)+v[1]*Math.cos(a),v[2]];
};
const armPointWorld = (p,yawDeg,zArmMinusWorld) => {
  if(!Array.isArray(p)||!isNum(yawDeg)||!isNum(zArmMinusWorld)) return null;
  const rotated=rotateBodyVectorWorld(p,yawDeg);
  return rotated ? [rotated[0],rotated[1],p[2]-zArmMinusWorld] : null;
};
// [[arm-point-world-core-end]]
// [[racket-world-speed-core-begin]]
// 世界系真实拍心速度 = 六轴相对臂座速度 + 车心平移 + 车体 yaw 刚体转动。
// armForwardOffset 是录制时配置里的「臂座相对车心沿车前方偏移」；V04 当前为 0.045m。
const racketWorldVelocity = (vArm,tcp,vCar,yawDeg,yawRate,armForwardOffset) => {
  if(!Array.isArray(vArm)||vArm.length!==3||!vArm.every(isNum) ||
     !Array.isArray(tcp)||tcp.length!==3||!tcp.every(isNum) ||
     !Array.isArray(vCar)||vCar.length!==2||!vCar.every(isNum) ||
     !isNum(yawDeg)||!isNum(yawRate)||!isNum(armForwardOffset)) return null;
  const rBody=[tcp[0],tcp[1]+armForwardOffset,tcp[2]];
  const turnBody=[-yawRate*rBody[1],yawRate*rBody[0],0];
  const armWorld=rotateBodyVectorWorld(vArm,yawDeg);
  const turnWorld=rotateBodyVectorWorld(turnBody,yawDeg);
  const carWorld=[vCar[0],vCar[1],0];
  const world=[0,1,2].map(k=>armWorld[k]+carWorld[k]+turnWorld[k]);
  const norm=v=>Math.hypot(v[0],v[1],v[2]);
  return {world,armWorld,carWorld,turnWorld,
          speedWorld:norm(world),speedArm:norm(armWorld),
          speedCar:norm(carWorld),speedTurn:norm(turnWorld)};
};
// 拍面 yaw 口径是 atan2(nx,ny)，pitch=asin(nz)：还原世界系前向单位法向。
const faceNormalWorld = (yawDeg,pitchDeg) => {
  if(!isNum(yawDeg)||!isNum(pitchDeg)) return null;
  const y=yawDeg*Math.PI/180, p=pitchDeg*Math.PI/180, cp=Math.cos(p);
  return [Math.sin(y)*cp,Math.cos(y)*cp,Math.sin(p)];
};
// 球拍碰撞的拍面法向恢复系数：u=v_ball-v_racket，u_out,n=-e_n*u_in,n。
// 这是逐拍实测等效量；负值或 >1 保留作诊断，不钳位。只有法向闭合速度不足时 fail closed。
const racketNormalRestitution = (vIn,vOut,vRacket,normal) => {
  const vectors=[vIn,vOut,vRacket,normal];
  if(vectors.some(v=>!Array.isArray(v)||v.length!==3||!v.every(isNum))) return null;
  const nNorm=Math.hypot(normal[0],normal[1],normal[2]);
  if(nNorm<1e-9) return null;
  const n=normal.map(v=>v/nNorm);
  const relNormal=v=>[0,1,2].reduce((sum,k)=>sum+(v[k]-vRacket[k])*n[k],0);
  const uInN=relNormal(vIn), uOutN=relNormal(vOut), closing=-uInN;
  if(!(closing>1.0)) return null;
  return {en:uOutN/closing,uInN,uOutN,closing,normal:n};
};
// [[racket-world-speed-core-end]]
// 离线三角测量的拍心（世界系, m）：PC 报告轴，重投影 >30px 丢弃。
const pcRacketRows = racket
  .map(r=>({t:isNum(r.rel_s)?r.rel_s:relTime(r.t), x:r.x, y:r.y, z:r.z,
             rp:isNum(r.reproj_err)?r.reproj_err:(isNum(r.reproj)?r.reproj:null),
             rpMax:r.reproj_max_px, looMaxMm:r.loo_max_mm, heldoutMaxPx:r.heldout_max_px,
             n_cam:r.n_cam, pair_cm:r.pair_cm, d02_mm:r.d02_mm,
             blackMarker:r.black_marker===true}))
  .filter(p=>isNum(p.t)&&isNum(p.x)&&isNum(p.y)&&isNum(p.z)&&(p.rp==null||p.rp<=30))
  .sort((a,b)=>a.t-b.t);
// [[racket-bracket-core-begin]]
const RACKET_RAW_SIDE_MAX_SEC=0.035;
const bracketVisualRacketRows = (sourceRows,t,maxGap=RACKET_RAW_SIDE_MAX_SEC) => {
  let before=null, after=null;
  sourceRows.forEach(row=>{
    const dt=row.t-t;
    if(Math.abs(dt)>maxGap) return;
    if(dt<0){ if(!before||row.t>before.t) before=row; }
    else if(!after||row.t<after.t) after=row;
  });
  return {before,after};
};
// [[racket-bracket-core-end]]
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
// 出弧另有三道防污染门（0813_083521 抛28 定案，入弧不加——入弧窗内可含真地面反弹）：
// 1) 轨迹一致性切段：跳变门=位移/dt，断档越长放行的绝对位移越大（抛28：138ms 断档+1.99m
//    跳到另一颗静止球=14.4m/s 贴线过 20 门），故 run ≥3 点后由末 3 点线性外推，偏差 >0.5m
//    切段；快而直的真回球（073646 12.2~12.4m/s）外推偏差小不受累；
// 2) 段首前触地拒绝：拟合 z 加回重力后在 [0,段首] 低于 0.05m，说明该段在触球与段首之间
//    经过了地面反弹（bounceCutRun 只能截段内反弹），倒推跨反弹必错（091825 抛13 假 +74.8°）；
// 3) 残差拒绝：三轴拟合 max|残差|>0.12m 不出数——干净弧 ≤4cm、单帧深度噪点 ≤10cm、混轨
//    垃圾 ≥16cm；单个高杠杆野点会把二次拟合弯过去反把真点顶成最大残差，故不做剔点重拟合。
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
// 出弧轨迹一致性切段（防污染门 1）：run 内已有 ≥3 点时由末 3 点线性外推到新点时刻，
// 偏差 >RETURN_TRACK_DEV_M 视为检测器接上别的球。切出的野点子段（常为 1~2 点静止球）
// 随后被点数/vy 门自然拒绝。
const RETURN_TRACK_DEV_M=0.5;
const splitByTrackDev = runs => {
  const out=[];
  for(const run of runs){
    let cur=[];
    for(const p of run){
      if(cur.length>=3){
        const a=cur[cur.length-3], b=cur[cur.length-1];
        const dv=Math.max(1e-9,b.t-a.t), dt=p.t-b.t;
        const ex=b.x+(b.x-a.x)/dv*dt, ey=b.y+(b.y-a.y)/dv*dt, ez=b.z+(b.z-a.z)/dv*dt;
        if(Math.hypot(p.x-ex,p.y-ey,p.z-ez)>RETURN_TRACK_DEV_M){ out.push(cur); cur=[]; }
      }
      cur.push(p);
    }
    if(cur.length) out.push(cur);
  }
  return out;
};
const RETURN_HALF_G=4.905;  // z 拟合先扣重力 ½g·u²，线性退化时 vz 不再吃重力偏差
const RETURN_FIT_RESID_M=0.12;  // 防污染门 3：三轴拟合 max|残差| 上限
const RETURN_MIN_FIT_Z=0.05;    // 防污染门 2：[0,段首] 拟合 z（含重力）下限
const evalFit=(f,u)=>f.a+f.b*u+f.c*u*u;
const ballVelocityFitAt = (seg,tHit) => {
  if(seg.length<5 || seg[seg.length-1].t-seg[0].t<0.06) return null;
  const fx=quadFitU(seg.map(p=>({t:p.t,v:p.x})),tHit);
  const fy=quadFitU(seg.map(p=>({t:p.t,v:p.y})),tHit);
  const fz=quadFitU(seg.map(p=>({t:p.t,v:p.z+RETURN_HALF_G*(p.t-tHit)*(p.t-tHit)})),tHit);
  if(!fx||!fy||!fz) return null;
  let maxRes=0;
  for(const p of seg){
    const u=p.t-tHit;
    maxRes=Math.max(maxRes,
      Math.abs(p.x-evalFit(fx,u)),
      Math.abs(p.y-evalFit(fy,u)),
      Math.abs(p.z+RETURN_HALF_G*u*u-evalFit(fz,u)));
  }
  return {vx:fx.b,vy:fy.b,vz:fz.b,fx,fy,fz,maxRes,n:seg.length,
          span:seg[seg.length-1].t-seg[0].t};
};
// e_n 的来球速度只取触球前同一条连续入弧的末段；若窗内含落地反弹，切到最后一次反弹之后。
// 这条三轴拟合只服务 e_n，不改变既有 PC 回球列的识别结果；质量不过门时 incoming=null。
const pcIncomingAt = tHit => {
  const runs=splitByTrackDev(pcRuns(tHit-0.38,tHit-0.025))
    .filter(r=>r.length>=5 && tHit-r[r.length-1].t<=0.10)
    .sort((a,b)=>b[b.length-1].t-a[a.length-1].t);
  if(!runs.length) return null;
  const run=runs[0];
  let start=0;
  for(let i=2;i<run.length;i++){
    const vzA=(run[i-1].z-run[i-2].z)/Math.max(1e-9,run[i-1].t-run[i-2].t);
    const vzB=(run[i].z-run[i-1].z)/Math.max(1e-9,run[i].t-run[i-1].t);
    if(vzA<-0.5 && vzB-vzA>3.0) start=i;
  }
  const seg=run.slice(start).filter(p=>p.t>=tHit-0.30);
  const fit=ballVelocityFitAt(seg,tHit);
  return fit&&fit.vy<-1.0&&fit.maxRes<=RETURN_FIT_RESID_M ? fit : null;
};
const pcHitTimeAt = tApprox => {
  // 入弧取最贴近触球且 ≥5 点的连续段
  const yinRun=[...pcRuns(tApprox-0.38,tApprox-0.025)].reverse().find(r=>r.length>=5);
  if(!yinRun) return null;
  const fin=quadFitU(yinRun.map(p=>({t:p.t,v:p.y})),tApprox);
  if(!fin||fin.b>-1.0) return null;
  // 出弧段按点数从多到少试，取首个真出向（vy>0.15）且交点落窗的段
  const outRuns=splitByTrackDev(pcRuns(tApprox+0.03,tApprox+0.33)).filter(r=>r.length>=4)
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
  const incoming=pcIncomingAt(tHit);
  // 出弧速度段：段首距触球 ≤300ms（限制回推外推量），按点数优先取首个过门槛的段
  const cands=splitByTrackDev(pcRuns(tHit+0.02,tHit+0.40))
    .filter(r=>r.length>=5 && r[0].t-tHit<=0.30)
    .sort((a,b)=>b.length-a.length);
  for(const run of cands){
    const {seg,bounceCut}=bounceCutRun(run);
    const fit=ballVelocityFitAt(seg,tHit);
    if(!fit) continue;
    const {vx,vy,vz,fx,fy,fz,maxRes}=fit;
    if(vy<=0.5) continue;
    const vh=Math.hypot(vx,vy);
    if(vh<1.0) continue;
    const start=seg[0].t-tHit;
    // 防污染门 2：段首前拟合 z 触地即整段在反弹之后，倒推跨反弹必错
    let minZ=Infinity;
    for(let i=0;i<=16;i++){
      const u=start*i/16;
      minZ=Math.min(minZ,evalFit(fz,u)-RETURN_HALF_G*u*u);
    }
    if(minZ<RETURN_MIN_FIT_Z) continue;
    // 防污染门 3：混轨/乱拟合拒绝出数（不剔点重拟合——高杠杆野点会反把真点顶成最大残差）
    if(maxRes>RETURN_FIT_RESID_M) continue;
    return {tHit, vx, vy, vz,
      yaw:Math.atan2(vx,vy)*180/Math.PI,
      pitch:Math.atan2(vz,vh)*180/Math.PI,
      speed:Math.hypot(vx,vy,vz),
      n:seg.length, span:seg[seg.length-1].t-seg[0].t,
      start, bounceCut, maxRes, incoming};
  }
  return null;
};
// [[pc-return-core-end]]
// 拍面世界yaw,pitch@臂最后更新HT：臂系 face_yaw/face_pitch（Python _add_face_angles 逐帧 FK，
// 车型配置的拍面法向轴；V04 为 link6 +Y）取冲击前窗 [−80,−6]ms 线性外推到该 HT——J5 冲击突跳（触始+13ms 机械传递，
// 本场 ≈ht−3ms）会污染跨冲击帧的直接插值（rebound 位姿采样同款约定）；车 yaw 直接取同一 RK 时刻的
// /bot_state yaw（imu_t 轴）。ψ_world = fy − 车yaw，口径与 PC回球 yaw 同为 atan2(x,y)；纯 RK/Arm 值，
// 不含 PC zPhase，也不含 δ6 球侧偏置。
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
const faceAnglesWorldAt = (accHtRk,tEval=accHtRk) => {
  const fit=fitFaceAnglesTo(accHtRk,tEval);
  if(!fit) return null;
  const botYaw=botYawDegAt(tEval);
  return {deg:botYaw!=null?wrapDeg(fit.fy-botYaw):null, fy:fit.fy, botYaw,
          n:fit.n, rate:fit.rate,
          pitch:fit.fp, pitchRate:fit.pitchRate};
};
// 拍面yaw,pitch,speed@臂最后更新HT−12ms（世界系）：fy/fp 同款冲击前窗拟合改在 HT−12ms 取值
// （拍速同样在该刻插值）；主列与本列都直接取各自 RK 时刻的 /bot_state yaw——车控接受 AprilTag 定位后 yaw 由
// IMU 连续更新，直到 HT 结束后才用 AprilTag 重定位更新 bot_state，因此 HT 前采样点
// 的 bot yaw 是无重定位台阶的瞬时值（挥拍塌陷伪迹只污染位置，yaw 走 IMU 不受累）。
const rkBotYawRows = (()=>{
  if(!RK || !RK.bot) return [];
  const T=ys(RK.bot,'imu_t')||[], Y=ys(RK.bot,'yaw'), rows=[];
  for(let i=0;i<T.length;i++){
    const t=T[i], yaw=Y[i];
    if(isNum(t)&&isNum(yaw)) rows.push({t,yaw});
  }
  return rows.sort((a,b)=>a.t-b.t);
})();
const botYawDegAt = t => {
  const exact=nearest(rkBotYawRows,t);
  if(exact&&Math.abs(exact.t-t)<=1e-9) return exact.yaw*180/Math.PI;
  const s=interpRow(rkBotYawRows,t,0.05);
  if(s){
    const dy=Math.atan2(Math.sin(s.b.yaw-s.a.yaw),Math.cos(s.b.yaw-s.a.yaw));
    return (s.a.yaw+dy*s.f)*180/Math.PI;
  }
  return null;
};
// /bot_state 的 vx/vy 是车心世界系速度；位姿字段必须按同条 imu_t 物理轴取值，不能按发布 t。
const rkBotVelocityRows = (()=>{
  if(!RK || !RK.bot) return [];
  const T=ys(RK.bot,'imu_t')||[], VX=ys(RK.bot,'vx'), VY=ys(RK.bot,'vy'), rows=[];
  for(let i=0;i<T.length;i++){
    const t=T[i], vx=VX[i], vy=VY[i];
    if(isNum(t)&&isNum(vx)&&isNum(vy)) rows.push({t,vx,vy});
  }
  return rows.sort((a,b)=>a.t-b.t);
})();
const botVelocityAt = t => {
  const exact=nearest(rkBotVelocityRows,t);
  if(exact&&Math.abs(exact.t-t)<=1e-9) return [exact.vx,exact.vy];
  const s=interpRow(rkBotVelocityRows,t,0.05);
  return s ? [lerp(s.a.vx,s.b.vx,s.f),lerp(s.a.vy,s.b.vy,s.f)] : null;
};
// 车 yaw 角速度：取 /chassis_can/imu 的 yaw_speed（rad/s，零滞后陀螺原值）。bot_state yaw
// 自身有 0.3~0.5s 滤波滞后，绝不能对它数值求导当瞬时角速度。该原值未减 BotState 启动
// bias（bag 未记录 bias-corrected yaw_rate）；近期三场由纯 IMU 段反推的影响 <0.002m/s，低于 0.01m/s 显示分辨率。
const rkImuYawRateRows = (()=>{
  if(!RK || !RK.imu) return [];
  const T=ts(RK.imu), W=ys(RK.imu,'yaw_speed'), rows=[];
  for(let i=0;i<T.length;i++){
    const t=Number(T[i]), w=Number(W[i]);
    if(isNum(t)&&isNum(w)) rows.push({t,w});
  }
  return rows.sort((a,b)=>a.t-b.t);
})();
const imuYawRateAt = t => {
  const exact=nearest(rkImuYawRateRows,t);
  if(exact&&Math.abs(exact.t-t)<=1e-9) return exact.w;
  const s=interpRow(rkImuYawRateRows,t,0.05);
  return s ? lerp(s.a.w,s.b.w,s.f) : null;
};
const imuYawRateDegAt = t => {
  const w=imuYawRateAt(t);
  return w!=null ? w*180/Math.PI : null;
};
// 实测世界拍速@臂最后更新HT（m/s）：Python 侧保存六轴解析 Jacobian 的臂系三维速度，JS
// 在同一 RK 物理时刻叠加车心世界系 vx/vy 与 yaw 刚体速度，再取世界系合速度模长。
// J1 分量 |q̇1|·r 与杠杆 r=hypot(tcp_x,tcp_y) 只用于对齐 status `speed=` 的 J1 规划口径。
// **拍速不外推**（与同列 yaw/pitch 不同）：挥拍段 J1 走 S 曲线，[−80,−6]ms 窗内拍速强非线性，
// 线性外推到 HT 实测会高估 40%+；HT 处两侧都有 100Hz 采样，直接插值即可。
// 触球锚不靠"实测峰值"找：实测 q̇1 全程叠着 ±0.5~1.4m/s 的伺服振荡（引拍段无球时同样存在，
// 见 osc），74ms 窗里取 argmax 只会落在某个振荡波峰上——0808 首版曾据此误判"峰值=触球、
// 触球早于 ht 39ms"，实为伪迹。真正的臂内触球用**指令自身**定位：HitTrajectory 触球后按恒 ω
// 巡航，故指令 J1 速度进入平台的第一帧就是 finish_hit_time（本场落在 HT−5~−12ms，与臂内
// 提前量对得上），该刻的实测/指令差才是伺服欠速。
const armSpeedRows = ARM ? ARM.states.filter(
  s=>Array.isArray(s.v_tcp_arm)&&s.v_tcp_arm.length===3&&s.v_tcp_arm.every(isNum)&&
     Array.isArray(s.tcp)&&s.tcp.length===3&&s.tcp.every(isNum)&&Array.isArray(s.velocity)) : [];
const armCmdSpeedRows = ARM ? (ARM.commands||[]).filter(
  c=>Array.isArray(c.tcp)&&Array.isArray(c.velocity)&&isNum(c.velocity[0])) : [];
const j1SpeedOf = row => Math.abs(row.velocity[0])*Math.hypot(row.tcp[0],row.tcp[1]);
const RACKET_SPEED_MAX_GAP_S=0.05;
const armForwardOffsetM=CA.bot_center&&CA.bot_center.params&&
  isNum(CA.bot_center.params.arm_forward_offset_m)
    ? CA.bot_center.params.arm_forward_offset_m : null;
const cmdSpeedAt = t => {
  const c=interpRow(armCmdSpeedRows,t,RACKET_SPEED_MAX_GAP_S);
  return c?lerp(j1SpeedOf(c.a),j1SpeedOf(c.b),c.f):null;
};
// 任意时刻的实测拍速（HT 与 HT−12ms 两列共用）：所有向量分量先插值再合成；任一项缺失
// 都返回 null，绝不把 arm-only 或缺 yaw 项的数冒充世界拍速。
const racketSpeedRawAt = t => {
  if(t==null || !armSpeedRows.length) return null;
  const s=interpRow(armSpeedRows,t,RACKET_SPEED_MAX_GAP_S);
  if(!s) return null;
  const vArm=[0,1,2].map(k=>lerp(s.a.v_tcp_arm[k],s.b.v_tcp_arm[k],s.f));
  const tcp=[0,1,2].map(k=>lerp(s.a.tcp[k],s.b.tcp[k],s.f));
  const vCar=botVelocityAt(t), yawDeg=botYawDegAt(t), yawRate=imuYawRateAt(t);
  const total=racketWorldVelocity(vArm,tcp,vCar,yawDeg,yawRate,armForwardOffsetM);
  if(!total) return null;
  return Object.assign(total,
    {vJ1:lerp(j1SpeedOf(s.a),j1SpeedOf(s.b),s.f),
     radiusJ1:Math.hypot(tcp[0],tcp[1]),cmdJ1:cmdSpeedAt(t),
     yawDeg,yawRate,armForwardOffsetM});
};
const racketSpeedAt = htRk => {
  const base=racketSpeedRawAt(htRk);
  if(!base) return null;
  // 臂内触球锚 = 指令 J1 速度平台的第一帧（挥拍段末端，之后是恒 ω 巡航）
  const seg=armCmdSpeedRows.filter(c=>c.t>=htRk-0.30&&c.t<=htRk+0.12).map(c=>({t:c.t,v:j1SpeedOf(c)}));
  let contactT=null, cmdContactJ1=null, measContactJ1=null;
  if(seg.length>10){
    const vmax=seg.reduce((m,c)=>Math.max(m,c.v),0);
    const hit=seg.find(c=>c.v>=vmax*0.995);
    if(hit){
      contactT=hit.t; cmdContactJ1=vmax;
      const ms=interpRow(armSpeedRows,contactT,RACKET_SPEED_MAX_GAP_S);
      if(ms) measContactJ1=lerp(j1SpeedOf(ms.a),j1SpeedOf(ms.b),ms.f);
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
  const oscJ1=n>=4?Math.sqrt(Math.max(0,sum2/n-(sum/n)*(sum/n))):null;
  return Object.assign({}, base,
    {contactDt:contactT!=null?(contactT-htRk)*1000:null,
     cmdContactJ1,measContactJ1,oscJ1});
};
const speedVectorText = v => Array.isArray(v)
  ? '('+v.map(x=>(x>=0?'+':'')+x.toFixed(2)).join(', ')+')' : '—';
// 固定探针偏移：0808 起 12ms（此前 10ms）。本场用指令速度平台首帧定位的臂内触球锚落在
// HT−1.5~−18ms、中位 −11ms，故 −12ms 基本踩在真实触球上；与主列之差 = 角速度/加速度×12ms。
const FACE_YAW_PRE_S=0.012;
const faceAnglesWorldPreAt = accHtRk => {
  if(accHtRk==null) return null;
  const tEval=accHtRk-FACE_YAW_PRE_S;
  const face=faceAnglesWorldAt(accHtRk,tEval);
  return face&&face.deg!=null?face:null;
};
// 末次 target 主表与机械臂最后 accepted 分表：两个合同独立取值、独立对齐 PC 真值。
const reportThrows = rkThrows.filter(t=>(t.msgs||0)>=3).sort((a,b)=>a.ht-b.ht);
// 北极星/Accepted 表的 PC 取样相位：global bias 只做 flight 粗配对；每抛 z(t) 只给
// PC 查询时刻增加一个小 offset。HT 只用于选最近的 PC flight，不进入评分。
const THROW_ALIGN_MAX_ANCHOR_DIST_S=0.8;
const buildThrowPhase = th => {
  const empty={offsetS:null,err:null,p90:null,trimmedRmse:null,n:0,coverage:0,
               span:null,zOffset:null,margin:null,profileWidth:null,edge:false,usable:false,
               deltaMs:null,pcFlight:null,rkFlight:null,anchorDist:null};
  if(!rkAxisAligned||!throwBaselineTrusted||!isNum(th.ht)) return empty;
  let rkFlightIdx=null, maxOverlap=0;
  rkFlights.forEach((flight,idx)=>{
    const overlap=Math.max(0,Math.min(th.lastT,flight[flight.length-1].t)-Math.max(th.firstT,flight[0].t));
    if(overlap>maxOverlap){maxOverlap=overlap;rkFlightIdx=idx;}
  });
  if(rkFlightIdx==null||maxOverlap<0.2) return empty;
  const anchor=th.ht+rkBias;
  let pcFlightIdx=null,anchorDist=Infinity;
  pcFlights.forEach((flight,idx)=>{
    const dist=anchor<flight[0].t ? flight[0].t-anchor
      : (anchor>flight[flight.length-1].t ? anchor-flight[flight.length-1].t : 0);
    if(dist<anchorDist){anchorDist=dist;pcFlightIdx=idx;}
  });
  if(pcFlightIdx==null||anchorDist>THROW_ALIGN_MAX_ANCHOR_DIST_S){
    return {...empty,rkFlight:rkFlightIdx,anchorDist:isFinite(anchorDist)?anchorDist:null};
  }
  const pcFlight=pcFlights[pcFlightIdx];
  const pad=THROW_PHASE_MAX_OFFSET_S+0.03;
  const rkRows=rkFlights[rkFlightIdx].filter(row=>{
    const t=row.t+rkBias;
    return t>=pcFlight[0].t-pad&&t<=pcFlight[pcFlight.length-1].t+pad;
  });
  const fit=estimateFlightPhase(pcFlight,rkRows,rkBias);
  return {...fit,deltaMs:fit.offsetS==null?null:fit.offsetS*1000,
          pcFlight:pcFlightIdx,rkFlight:rkFlightIdx,anchorDist};
};
let throwPhases=[];
const rebuildThrowPhases = () => {
  throwPhases=reportThrows.map((th,idx)=>({reportRow:idx+1,...buildThrowPhase(th)}));
  window.__throwPhases=throwPhases;
};
rebuildThrowPhases();
const throwPhaseFor = th => {
  const idx=reportThrows.indexOf(th);
  return idx>=0?throwPhases[idx]:null;
};
const pcSampleTimeForThrow = (th,t) => {
  const phase=throwPhaseFor(th);
  const baseline=rkToPc(t);
  return phase&&phase.usable&&baseline!=null?baseline+phase.offsetS:null;
};
window.__dbgAlign.throwPhases=()=>throwPhases;
// PC球拍头 bbox 中心 bundle vz、RK当时S0系数、RK弹后实测是三条独立证据。
// vz_world_mps 是检测框几何中心的世界 z 速度代理，不是校准拍面速度，也不是球自转。
// v3 只按 contact_anchor_t_rk 回配；它是 RK ball_world 固定高度外推锚点。
// 拍头速度只拟合接触锚前125ms至后35ms，避免把准备下沉当成击球方向。
// 不是视频直接观测到的 impact 时刻。
// S0 系数和历史旧拍头vz只认同抛最后一条 Stage0 /predict_hit_pos；弹后实测只认 extractor
// 按 RK 已采纳 S1 帧复算出的 cor_meas_replay/cxy_meas_replay，不拿在线锚点冒充本抛实测。
// [[racket-cor-core-begin]]
const RACKET_IMPACT_MATCH_MAX_S=1.5;
const RACKET_NEAR_HORIZONTAL_VZ_MPS=0.30;
const racketMotionForVz = vz => !isNum(vz) ? null
  : (vz>=RACKET_NEAR_HORIZONTAL_VZ_MPS ? '拍头上行'
    : (vz<=-RACKET_NEAR_HORIZONTAL_VZ_MPS ? '拍头下行' : '近水平'));
const racketSpinTypeForVz = vz => !isNum(vz) ? null
  : (vz>=RACKET_NEAR_HORIZONTAL_VZ_MPS ? '上旋倾向'
    : (vz<=-RACKET_NEAR_HORIZONTAL_VZ_MPS ? '下旋倾向' : '旋转类型不判定'));
const racketImpactAvailable=()=>Array.isArray(D.racket_impact);
const racketImpactRows=(racketImpactAvailable()&&RK&&isNum(RK.t0)?D.racket_impact:[])
  .filter(row=>row&&(row.status==='accepted'||row.status==='rejected'))
  .map(row=>{
    const anchorTRk=isNum(row.contact_anchor_t_rk)?row.contact_anchor_t_rk:null;
    return {row,anchorTRk,matchedTRk:anchorTRk,timeSource:'contact_anchor',
            matchT:isNum(anchorTRk)?anchorTRk-RK.t0:null};
  })
  .filter(item=>isNum(item.matchT))
  .sort((a,b)=>a.matchT-b.matchT);
const racketImpactAssignments = () => {
  const assigned=new Map();
  if(!racketImpactAvailable()||!RK||!isNum(RK.t0)) return assigned;
  const temporal=[...reportThrows].sort((a,b)=>a.firstT-b.firstT);
  let throwIdx=0;
  racketImpactRows.forEach(item=>{
    while(throwIdx<temporal.length && temporal[throwIdx].firstT<item.matchT) throwIdx++;
    if(throwIdx>=temporal.length) return;
    const th=temporal[throwIdx], dt=th.firstT-item.matchT;
    if(dt<=RACKET_IMPACT_MATCH_MAX_S){
      assigned.set(th,{...item,dt});
      throwIdx++;
    }
  });
  return assigned;
};
const racketCorForThrow = (th,assignments) => {
  const match=assignments.get(th)||null;
  const rowStatus=match?match.row.status:null;
  const contactAnchorStatus=match?match.row.contact_anchor_status:null;
  const visionEvaluated=match?match.row.vision_evaluated:null;
  const measurement=match&&match.row.measurement&&typeof match.row.measurement==='object'
    ? match.row.measurement : null;
  const bundle=measurement&&measurement.bundle_diagnostics
    &&typeof measurement.bundle_diagnostics==='object'
    ? measurement.bundle_diagnostics : null;
  const bundleVelocity=bundle&&Array.isArray(bundle.bbox_center_velocity_world_mps)
    ? bundle.bbox_center_velocity_world_mps:null;
  const acceptedEvidence=measurement&&measurement.accepted===true
    &&measurement.reason==='accepted'
    &&measurement.observation_semantics==='racket_head_bbox_geometric_center_native_pixel'
    &&measurement.velocity_semantics==='racket_head_bbox_center_world_velocity_proxy'
    &&Array.isArray(measurement.raw_bbox_observations)
    &&measurement.raw_bbox_observations.length>=6
    &&bundle&&bundle.accepted===true&&bundle.reason==='accepted'
    &&bundle.observation_semantics==='racket_head_bbox_geometric_center'
    &&Array.isArray(bundle.supported_frames)&&bundle.supported_frames.length>=3
    &&isNum(bundle.fit_span_s)&&bundle.fit_span_s>=.055
    &&isNum(bundle.max_reprojection_error_px)&&bundle.max_reprojection_error_px<=8
    &&bundleVelocity&&bundleVelocity.length===3&&bundleVelocity.every(isNum)
    &&isNum(match.row.vz_world_mps)&&Math.abs(match.row.vz_world_mps)>=RACKET_NEAR_HORIZONTAL_VZ_MPS
    &&match.row.vz_semantics==='racket_head_bbox_center_world_velocity_proxy'
    &&isNum(measurement.bbox_center_vz_world_mps)
    &&isNum(bundle.bbox_center_vz_world_mps)
    &&Math.abs(match.row.vz_world_mps-measurement.bbox_center_vz_world_mps)<=1e-9
    &&Math.abs(match.row.vz_world_mps-bundle.bbox_center_vz_world_mps)<=1e-9
    &&Math.abs(match.row.vz_world_mps-bundleVelocity[2])<=1e-9;
  const acceptedState=rowStatus==='accepted'&&contactAnchorStatus==='accepted'
    &&visionEvaluated===true;
  const validAccepted=acceptedState&&acceptedEvidence;
  const validContactRejected=rowStatus==='rejected'&&contactAnchorStatus==='rejected'
    &&visionEvaluated===false;
  const validVisionNotEvaluated=rowStatus==='rejected'&&contactAnchorStatus==='accepted'
    &&visionEvaluated===false;
  const validVisionRejected=rowStatus==='rejected'&&contactAnchorStatus==='accepted'
    &&visionEvaluated===true;
  const acceptedVz=validAccepted&&isNum(match.row.vz_world_mps)
    ? match.row.vz_world_mps : null;
  const usedRvz=isNum(th.lastS0Rvz)?th.lastS0Rvz:null;
  const corXyEff=isNum(th.lastS0CorXyEff)?th.lastS0CorXyEff:null;
  const corEff=isNum(th.lastS0CorEff)?th.lastS0CorEff:null;
  const corMeasReplay=isNum(th.corMeasReplay)?th.corMeasReplay:null;
  const cxyMeasReplay=isNum(th.cxyMeasReplay)?th.cxyMeasReplay:null;
  const status=!racketImpactAvailable() ? 'legacy_untrusted'
    : (!match ? 'missing'
      : (validAccepted ? 'measured'
        : (acceptedState ? 'invalid_accepted_evidence'
        : (validContactRejected ? 'contact_rejected'
          : (validVisionNotEvaluated ? 'vision_not_evaluated'
            : (validVisionRejected ? 'vision_rejected':'invalid_v3_row'))))));
  const rejectedStatus=status==='contact_rejected'||status==='vision_not_evaluated'
    ||status==='vision_rejected';
  const failureReason=rejectedStatus&&typeof match.row.failure_reason==='string'
    &&match.row.failure_reason.trim() ? match.row.failure_reason.trim()
    : (rejectedStatus?'未提供failure_reason':null);
  return {status,motion:racketMotionForVz(acceptedVz),spinType:racketSpinTypeForVz(acceptedVz),
          measuredVz:acceptedVz,
          failureReason,
          rowStatus,contactAnchorStatus,visionEvaluated,
          acceptanceMode:match&&typeof match.row.acceptance_mode==='string'
            ? match.row.acceptance_mode:null,
          prefixSpread:match&&isNum(match.row.prefix_spread_s)
            ? match.row.prefix_spread_s:null,
          contactPointSpread:match&&isNum(match.row.contact_point_spread_m)
            ? match.row.contact_point_spread_m:null,
          supportedFrames:bundle&&Array.isArray(bundle.supported_frames)
            ? bundle.supported_frames:null,
          fitSpan:bundle&&isNum(bundle.fit_span_s)?bundle.fit_span_s:null,
          reprojectionMax:bundle&&isNum(bundle.max_reprojection_error_px)
            ? bundle.max_reprojection_error_px:null,
          leaveOneFrameVz:bundle&&Array.isArray(bundle.leave_one_frame_bbox_center_vz_mps)
            ? bundle.leave_one_frame_bbox_center_vz_mps:null,
          rejectionCounts:measurement?measurement.rejection_counts:null,
          usedRvz,corXyEff,corEff,
          corMeasReplay,cxyMeasReplay,corMeasClosureMs:th.corMeasClosureMs,
          corMeasIdx:th.corMeasIdx,
          matchedTRk:match?match.matchedTRk:null,
          anchorTRk:match?match.anchorTRk:null,
          timeSource:match?match.timeSource:null,dt:match?match.dt:null};
};
// [[racket-cor-core-end]]
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
const R_BALL=0.033;  // 网球半径；报告 y 接触面统一取球心 world_y − R_BALL
const ballCarGapForThrow = (()=>{
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
    const BT=ys(RK.bot,'imu_t')||[], BX=ys(RK.bot,'x'), BY=ys(RK.bot,'y');
    for(let i=0;i<BT.length;i++){
      const t=BT[i];
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
const BOT_POSE_MAX_GAP_S=0.05;
const rkBotPoseRows = (()=>{
  if(!RK || !RK.bot) return [];
  const T=ys(RK.bot,'imu_t')||[], X=ys(RK.bot,'x'), Y=ys(RK.bot,'y'), rows=[];
  for(let i=0;i<T.length;i++){
    const t=T[i], x=X[i], y=Y[i];
    if(isNum(t)&&isNum(x)&&isNum(y)) rows.push({t,x,y});
  }
  return rows.sort((a,b)=>a.t-b.t);
})();
const botPoseAtImuTime = t => {
  const exact=nearest(rkBotPoseRows,t);
  if(exact&&Math.abs(exact.t-t)<=1e-9) return {t,x:exact.x,y:exact.y};
  const s=interpRow(rkBotPoseRows,t,BOT_POSE_MAX_GAP_S);
  return s ? {t,x:lerp(s.a.x,s.b.x,s.f),y:lerp(s.a.y,s.b.y,s.f)} : null;
};
const botRunEndForThrow = th => {
  if(!RK || !th || !isNum(th.ht)) return null;
  const t=ts(RK.bot), phase=ys(RK.bot,'phase');
  const tx=ys(RK.bot,'target_x'), ty=ys(RK.bot,'target_y');
  const remaining=ys(RK.bot,'remaining')||[];
  let best=null;
  for(let i=0;i+1<t.length;i++){
    if(phase[i]!=='RUN' || phase[i+1]!=='BRAKE_IN_SWING' || !isNum(t[i])) continue;
    const dt=Math.abs(t[i]-th.ht);
    if(dt>0.1 || (best && dt>=best.dt)) continue;
    const owner=reportThrows.reduce((closest,candidate)=>
      !closest || Math.abs(candidate.ht-t[i])<Math.abs(closest.ht-t[i]) ? candidate : closest, null);
    if(owner!==th || ![tx[i],ty[i]].every(isNum)) continue;
    best={t:t[i],tx:tx[i],ty:ty[i],dt,endIdx:i};
  }
  if(!best) return null;
  let startIdx=best.endIdx;
  while(startIdx>0 && phase[startIdx-1]==='RUN') startIdx--;
  let prev=null, targetChange=null;
  for(let i=startIdx;i<=best.endIdx;i++){
    if(![t[i],tx[i],ty[i]].every(isNum)) continue;
    if(!prev || Math.abs(tx[i]-prev.x)>1e-6 || Math.abs(ty[i]-prev.y)>1e-6){
      targetChange={t:t[i],x:tx[i],y:ty[i],idx:i,
                    deadline:isNum(remaining[i])?t[i]+remaining[i]:null};
    }
    prev={x:tx[i],y:ty[i]};
  }
  best.startIdx=startIdx;
  best.targetChange=targetChange;
  return best;
};
// [[bot-run-end-core-end]]
// 最后一次可观察 raw target_x/y 坐标变化，对应哪条 /predict_hit_pos：
// set_travel_target 同拍写入 p.ht，bot_state.remaining=deadline-now；pred.t 则是 payload ct（观测时刻），
// 不是 callback 到达时刻。因此用 target 变化帧的 t+remaining 回配同抛 ht，不能按 ct 最近邻。
// [[last-target-pred-core-begin]]
const LAST_TARGET_HT_MATCH_MAX_S=0.001;
const lastTargetPredictionForThrow = (th,runEnd) => {
  const change=runEnd&&runEnd.targetChange;
  if(!change || !isNum(change.deadline) || !Number.isInteger(th.firstIdx) || !Number.isInteger(th.lastIdx)) return null;
  const t=ts(RK.pred), ht=ys(RK.pred,'ht_rel');
  const worldX=ys(RK.pred,'x');
  const relX=ys(RK.pred,'rel_x'), relZ=ys(RK.pred,'rel_z');
  const carX=ys(RK.pred,'car_pred_x'), carY=ys(RK.pred,'car_pred_y');
  let best=null;
  for(let i=th.firstIdx;i<=th.lastIdx;i++){
    if(!isNum(t[i]) || !isNum(ht[i]) || t[i]>change.t) continue;
    const htError=Math.abs(ht[i]-change.deadline);
    if(!best || htError<best.htError){
      best={idx:i,ct:t[i],ht:ht[i],lead:ht[i]-t[i],htError,
            stage:isNum(rkPredStage[i])?rkPredStage[i]:null,
            nFit:isNum(rkPredNFit[i])?rkPredNFit[i]:null,
            worldX:isNum(worldX[i])?worldX[i]:null,
            relX:isNum(relX[i])?relX[i]:null,relZ:isNum(relZ[i])?relZ[i]:null,
            carX:isNum(carX[i])?carX[i]:null,carY:isNum(carY[i])?carY[i]:null};
    }
  }
  return best&&best.htError<=LAST_TARGET_HT_MATCH_MAX_S?best:null;
};
// [[last-target-pred-core-end]]
const pcTruthCell = (f,withY=false,tPc=null,contactY=false) => {
  if(!f) return pcTruthMissCell(tPc);
  // 两侧时距都显示：球是**外推**（拟合窗末点到目标时刻），车是**插值**（到前后最近一条
  // /pc_car_loc 的较大一侧）。0731 起 x/y 是「球世界−车世界」，只报球侧会把车侧的
  // 陈旧/宽插值藏起来——0809_122035 #13 就是这么用一个 ±0.8cm 的棒子盖住了 10cm 的车侧误差。
  const carMs=Math.round((f.carGap||0)*1000);
  const carTxt=carMs>0?'·车±'+carMs+'ms':'';
  // 剔过点的行必须在格子里直接看得见，不能只藏在 tooltip：读者要能一眼分辨
  // "这格是全窗拟合" 还是 "剔掉坏点才凑出来的"
  const drops=Array.isArray(f.dropped)?f.dropped:[];
  const dropTxt=drops.length?'·剔'+drops.length+'点':'';
  const dropTitle=drops.length
    ? '；⚠ 有界离群剔除：剔掉 '+drops.length+' 个显著突出的点（'+
      drops.map(d=>(d.dt*1000).toFixed(0)+'ms/'+(d.r).toFixed(1)+'×门').join('、')+
      '），剩 '+f.nPts+' 点参与拟合。剔除条件=最坏点 ≥2.5× 中位残差且本身超门、剔完剩 ≥8 点、'+
      '最多剔 2 个；坏点来源通常是多球关联失败或弹跳接触帧，不是球的测量'
    : '';
  const yValue=contactY?f.y-R_BALL:f.y;
  const coordTitle=contactY
    ? '表中 x=球心world_x−车体中心world_x，y=(球心world_y−R球3.3cm)−车体中心world_y，z=球心world_z（世界轴不转yaw）；'
    : '表中 x/y=拟合球世界坐标−车世界坐标(世界轴不转yaw)；';
  const yTitle=contactY
    ? 'ball_surface_y−car_y='+cmSigned(yValue)+'cm'
    : 'ball_y−car_y='+cmSigned(f.y)+'cm';
  return (withY?tableXyzCm(f.x,yValue,f.z):tableXzCm(f.x,f.z))+
    ' <span style="color:'+(carMs>150?'#f97316':'#fbbf24')+'" title="入弧拟合真值：x/y 线性、z 重力+阻力(λ=k_drag·水平速)+带界旋转曲率(|δ|≤2m/s²)；本行 δz='+((f.delta||0)>=0?'+':'')+(f.delta||0).toFixed(2)+'m/s²；'+coordTitle+'只用目标时刻20ms前观测，不跨目标时刻插值；max|残差| '+
    cmFmt(f.resMax)+'cm，'+yTitle+
    '；球外推 '+Math.round(f.dNear*1000)+'ms（拟合窗末点→目标时刻）'+
    '；车 /pc_car_loc 前 '+Math.round((f.carGa||0)*1000)+'ms / 后 '+Math.round((f.carGb||0)*1000)+
    'ms 夹住插值（不外推、不冻结），按底盘 a_dec_max=3m/s² 折算插值误差 ±'+cmFmt(f.eCar||0)+
    'cm，已并进左边的总误差棒'+
    '；窗长稳定性自检（换 0.40s 短窗重拟合）三轴最大差 '+cmFmt(f.eStab||0)+
    'cm，同样并进误差棒——这一项大说明本抛只有弹跳后一小段干净弧，长窗模型失配明显'+
    (f.carSingleTag?'；⚠ 夹住的 /pc_car_loc 里有单 tag 退化解（只剩一块 tag 可见，位置由冻结 yaw 经 0.42m 安装杠杆反解，已按 ±'+
      cmFmt(PC_TRUTH_SINGLE_TAG_ERR)+'cm 并进车侧误差）':'')+dropTitle+'">±'+
    cmFmt(Math.max(0.001,f.err))+'cm(球外推'+Math.round(f.dNear*1000)+'ms'+carTxt+dropTxt+')</span>';
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

const racketCorCellHtml = (th,assignments) => {
  const r=racketCorForThrow(th,assignments);
  const signed3=v=>isNum(v)?(v>=0?'+':'')+v.toFixed(3):'—';
  const metaValue=v=>typeof v==='string'?v:JSON.stringify(v);
  const auditMeta=[
    r.acceptanceMode?'contact_acceptance_mode='+r.acceptanceMode:null,
    isNum(r.prefixSpread)?'prefix_spread_ms='+(1000*r.prefixSpread).toFixed(2):null,
    isNum(r.contactPointSpread)?'contact_point_spread_cm='+(100*r.contactPointSpread).toFixed(2):null,
    r.supportedFrames!=null?'supported_frames='+metaValue(r.supportedFrames):null,
    isNum(r.fitSpan)?'bundle_fit_span_ms='+(1000*r.fitSpan).toFixed(2):null,
    isNum(r.reprojectionMax)?'bundle_max_reprojection_error_px='+r.reprojectionMax.toFixed(2):null,
    r.leaveOneFrameVz!=null?'leave_one_frame_bbox_center_vz_mps='+metaValue(r.leaveOneFrameVz):null,
    r.rejectionCounts!=null?'rejection_counts='+metaValue(r.rejectionCounts):null,
  ].filter(Boolean).join(', ');
  const matchTimingNote=r.timeSource==='contact_anchor'
    ? '仅按 contact_anchor_t_rk='+r.anchorTRk.toFixed(6)+
      's 回配（RK ball_world 固定高度外推锚点，不是视频直接观测到的impact）'
    : '';
  const impactText=r.status==='measured'
    ? r.spinType+' '+signed3(r.measuredVz)
    : '—';
  const usedText=isNum(r.usedRvz)
    ? 'RK当时S0采用的旧拍头vz '+signed3(r.usedRvz)
    : 'RK当时S0未采用旧拍头vz（cor_xy走aMz）';
  const corXyText=isNum(r.corXyEff)?r.corXyEff.toFixed(4):'—';
  const corText=isNum(r.corEff)?r.corEff.toFixed(4):'—';
  const measuredCxyText=isNum(r.cxyMeasReplay)?r.cxyMeasReplay.toFixed(3):'—';
  const measuredCorText=isNum(r.corMeasReplay)?r.corMeasReplay.toFixed(3):'—';
  const usedValue=isNum(r.usedRvz)?signed3(r.usedRvz):'—';
  const sourceNote=r.status==='measured'
    ? '触球锚已通过；vision_evaluated=true；球拍头bbox中心bundle vz代理='+
      signed3(r.measuredVz)+'m/s（≥+0.30 拍头上行，≤−0.30 拍头下行，其余近水平）；'+
      matchTimingNote+'；拍头上行/下行→上旋/下旋倾向；不是直接观测球自转；'+
      '不是校准拍面速度（not calibrated racket-face speed）；未用于本场S0'+
      (auditMeta?'；审计字段 '+auditMeta:'')
    : (r.status==='contact_rejected'
      ? '触球锚阶段拒绝；vision_evaluated=false，视频检测与bundle没有运行；'+
        'failure_reason='+r.failureReason+(auditMeta?'；审计字段 '+auditMeta:'')
      : (r.status==='vision_not_evaluated'
        ? '触球锚已通过；vision_evaluated=false，视频检测与bundle没有运行；'+
          'failure_reason='+r.failureReason+(auditMeta?'；审计字段 '+auditMeta:'')
        : (r.status==='vision_rejected'
          ? '触球锚已通过；vision_evaluated=true；bundle拒绝；'+matchTimingNote+
            '；failure_reason='+r.failureReason+(auditMeta?'；审计字段 '+auditMeta:'')+
            '；拒绝记录的 vz_world_mps 无论是否存在都不读取、不显示'
      : (r.status==='legacy_untrusted'
        ? '旧session没有 D.racket_impact；旧运动质心流程已判定不可信，绝不再用于判型'
        : (r.status==='invalid_accepted_evidence'
            ? '本抛状态声称accepted，但measurement/bundle/vz语义或数值链不完整、不一致；'+
              matchTimingNote+'；vz_world_mps不读取、不显示'
          : (r.status==='invalid_v3_row'
            ? 'racket_impact/v3 状态组合无效（status='+metaValue(r.rowStatus)+
              ', contact_anchor_status='+metaValue(r.contactAnchorStatus)+
              ', vision_evaluated='+metaValue(r.visionEvaluated)+'）；vz_world_mps不读取、不显示'
            : '本抛没有可保序回配的 racket_impact/v3 记录'))))));
  const corNote=isNum(r.corXyEff)
    ? '水平cor_xy='+r.corXyEff.toFixed(4)+'（payload字段cor_xy_eff）直接取同抛最后一条Stage0 /predict_hit_pos（idx='+
      th.lastS0Idx+'）；Stage0用它计算弹后vx/vy，不取配置值，也不拿S1遥测冒充；'+usedText
    : '同抛最后Stage0缺cor_xy_eff，显示—，不从配置回填';
  const normalCorNote=isNum(r.corEff)
    ? '竖直cor_z='+r.corEff.toFixed(4)+'（payload字段cor_eff）取自同一条Stage0，Stage0按 vz_post=−cor_z·vz_pre 使用；'+
      '它是当前在线法向COR锚点，不是尚未发生的本抛反弹实测真值'
    : '同抛最后Stage0缺cor_eff，显示—，不从配置回填';
  const measuredNote=isNum(r.cxyMeasReplay)&&isNum(r.corMeasReplay)
    ? 'RK弹后实测复算：取已采纳S1帧 idx='+r.corMeasIdx+
      ' 对应的弹前/弹后 /ball_world_topic 拟合，在球心z=地面+3.3cm分别取速度；'+
      'cor_xy=vxy_out/vxy_in='+r.cxyMeasReplay.toFixed(4)+
      '，cor_z=−vz_out/vz_in='+r.corMeasReplay.toFixed(4)+
      (isNum(r.corMeasClosureMs)?'，两段触地时刻差='+r.corMeasClosureMs.toFixed(2)+'ms':'')+
      '。≈表示bag坐标只有4位小数、时间6位小数，非RK history原始double'
    : '本抛没有可复算的RK已采纳反弹样本（无S1、未通过反弹门或旧sidecar缺复算字段），显示—';
  return '<span style="white-space:nowrap" title="'+
    tableEsc(sourceNote+'；'+corNote+'；'+normalCorNote+'；'+measuredNote)+'">'+
    impactText+' / S0 cxy '+corXyText+', cor_z '+corText+
    ' / 实测反弹 cxy '+measuredCxyText+', cor_z '+measuredCorText+
    ' / RK旧拍头vz '+usedValue+'</span>';
};

const rk300TableHtml = () => {
  if(!RK || !reportThrows.length) return '';
  const racketAssignments=racketImpactAssignments();
  const visualTcpSameOrigin=ARM&&String(ARM.fk_car||'').toLowerCase()==='v04';
  const racketBlackMarker=pcRacketRows.some(r=>r.blackMarker);
  const armContractRows=[];
  const rows=reportThrows.map((th,idx)=>{
    const zPhase=throwPhaseFor(th);
    const zPhaseFailure=zPhase&&zPhase.usable?null
      : (!throwBaselineTrusted?'baseline_untrusted'
        :(!zPhase||zPhase.rkFlight==null?'rk_flight_not_found'
          :(zPhase.pcFlight==null?'pc_flight_not_found'
            :(zPhase.offsetS==null?'insufficient_overlap'
              :(zPhase.edge?'offset_at_100ms_boundary':'quality_gate')))));
    const rowPcFixed=(t,digits=3)=>{
      const value=rkToPc(t);
      return value==null?'—':value.toFixed(digits);
    };
    const zPhaseTitle=zPhase&&zPhase.usable
      ? '本抛 z(t) 只修正 PC 取样相位：tPC(sample)=tRK+global bias+'
        +tableSigned(zPhase.deltaMs)+'ms；RK字段和RK时间显示不变，所有PC取样（含视觉拍心）使用该修正。'
        +'z残差 median/p90/trim-RMSE='+(zPhase.err*100).toFixed(1)+'/'
        +(zPhase.p90*100).toFixed(1)+'/'+(zPhase.trimmedRmse*100).toFixed(1)+'cm；'
        +zPhase.n+'点/'+Math.round(zPhase.span*1000)+'ms/覆盖'
        +Math.round(zPhase.coverage*100)+'%；10ms外次优比 '+zPhase.margin.toFixed(2)
        +'×；5mm误差带宽 '+(zPhase.profileWidth*1000).toFixed(1)
        +'ms；PC/RK flight '+zPhase.pcFlight+'/'+zPhase.rkFlight
        +'。预测HT只用于粗配 flight，不进入评分，也未强制HT=实际触球。'
      : '本抛 z(t) 相位修正不可用（'+zPhaseFailure+'）；依赖本抛PC取样的格（含视觉）显示—，RK字段仍按global baseline显示。'
        +(zPhase&&zPhase.offsetS!=null
          ? '候选诊断：offset='+tableSigned(zPhase.offsetS*1000)+'ms，z median/p90='
            +(zPhase.err*100).toFixed(1)+'/'+(zPhase.p90*100).toFixed(1)+'cm，10ms外次优比 '
            +(zPhase.margin==null?'—':zPhase.margin.toFixed(2))+'×，5mm误差带宽 '
            +(zPhase.profileWidth==null?'—':(zPhase.profileWidth*1000).toFixed(1))+'ms。'
          : '没有可成对的同抛PC/RK运动轨迹。');
    const zPhaseCell='<span'+(zPhase&&zPhase.usable?'':' style="color:#e0a24a"')+
      ' title="'+tableEsc(zPhaseTitle)+'">'+(zPhase&&zPhase.usable
        ? tableSigned(zPhase.deltaMs):'⚠—')+'</span>';
    const accepted=lastAcceptedForThrow(th);
    const runEnd=botRunEndForThrow(th);
    const targetChange=runEnd&&runEnd.targetChange;
    const targetPred=lastTargetPredictionForThrow(th,runEnd);
    const targetUpdateCell=targetChange
      ? '<span title="'+tableEsc(targetPred
          ? '最后可观察 raw target 坐标变化 t='+rowPcFixed(targetChange.t,6)+'s（global PC轴），target=('
            +cmFmt(targetChange.x)+', '+cmFmt(targetChange.y)+')cm；对应预测 payload ct='
            +rowPcFixed(targetPred.ct,6)+'s，按 bot.t+remaining ↔ pred.ht 回配，deadline残差 '
            +(targetPred.htError*1000).toFixed(3)+'ms'
          : '已观察到最后 raw target 坐标变化，但 bot.t+remaining 无法在同抛内回配到 1ms 内的 pred.ht')+'">'
          +rowPcFixed(targetChange.t)
          +(targetPred?' <span style="color:#a0a0c0">(ct '+rowPcFixed(targetPred.ct)+')</span>':'')+'</span>'
      : '—';
    const targetPredHtBaseline=targetPred?rkToPc(targetPred.ht):null;
    const targetPredHt=targetPredHtBaseline!=null?targetPredHtBaseline.toFixed(3):'—';
    const targetPredSamplePc=targetPred?pcSampleTimeForThrow(th,targetPred.ht):null;
    const targetTruth=targetPredSamplePc!=null?pcTruthAt(targetPredSamplePc):null;
    const targetPredHit=targetPred?tableXzCm(targetPred.relX,targetPred.relZ):'—';
    const racketCorCell=racketCorCellHtml(th,racketAssignments);
    const actualAtTargetHt=targetChange&&isNum(targetChange.deadline)
      ? botPoseAtImuTime(targetChange.deadline) : null;
    const targetActualAtHtError=runEnd&&actualAtTargetHt
      ? tableSigned((runEnd.tx-actualAtTargetHt.x)*100)+'/'+tableSigned((runEnd.ty-actualAtTargetHt.y)*100)
      : '—';
    const actualAtPredHt=targetPred?botPoseAtImuTime(targetPred.ht):null;
    const predActualAtHtError=actualAtPredHt&&targetPred&&Number(targetPred.stage)===1&&isNum(targetPred.carX)&&isNum(targetPred.carY)
      ? tableSigned((targetPred.carX-actualAtPredHt.x)*100)+'/'+tableSigned((targetPred.carY-actualAtPredHt.y)*100)
      : (targetPred&&Number(targetPred.stage)===0
          ? '<span title="Stage0 的 car_pred_x/y 实际填的是 raw travel target，不是真实车轨迹预测">— (S0)</span>'
          : '—');
    const accHt=accepted&&isNum(accepted.wht)?accepted.wht-RK.t0:null;
    // raw 最后更新 HT：最后进入 update_ht() 的 /predict_hit_pos 原始 ht。它用于预测/盲区账；
    // mode=2 Coast 时旧 profile 继续执行，故不能把这个 raw HT 称为机械臂执行接触锚。
    // 本表所有 @臂最后更新HT 的量（包括 TCP）仍只在这个原始 HT 取值，不另找过面时刻。
    const finalHt=accepted
      ? (accepted.finalMismatch ? null : (isNum(accepted.finalHt)?accepted.finalHt-RK.t0:accHt))
      : null;
    const finalHtPcBaseline=finalHt!=null?rkToPc(finalHt):null;
    const finalHtPcSample=finalHt!=null?pcSampleTimeForThrow(th,finalHt):null;
    armContractRows.push({
      reportRow:idx+1,
      accepted:!!accepted,
      finalMismatch:!!(accepted&&accepted.finalMismatch),
      finalHtRkAbs:finalHt!=null?finalHt+RK.t0:null,
      finalHtPcBaselineElapsed:finalHtPcBaseline,
      finalHtPcSampleElapsed:finalHtPcSample,
      zPhase:zPhase?{
        usable:zPhase.usable,baselineTrusted:throwBaselineTrusted,
        failureReason:zPhaseFailure,deltaS:zPhase.usable?zPhase.offsetS:null,
        deltaMs:zPhase.usable?zPhase.deltaMs:null,
        err:zPhase.err,p90:zPhase.p90,trimmedRmse:zPhase.trimmedRmse,
        n:zPhase.n,coverage:zPhase.coverage,span:zPhase.span,
        margin:zPhase.margin,profileWidth:zPhase.profileWidth,edge:zPhase.edge,
        pcFlight:zPhase.pcFlight,rkFlight:zPhase.rkFlight,anchorDist:zPhase.anchorDist,
      }:null,
    });
    const rsw=accepted?accepted.reswing:null;
    const htSrcNote=rsw
      ? (rsw.ok
          ? (rsw.continuousSweep
              ? ('；raw HT源=连续sweep最后进入update_ht的late saved（老触球'
                 +tableSigned(rsw.delta)+'ms）'
                 +(rsw.nCoast?'；其中 '+rsw.nCoast+' 条状态为mode=2 Coast，raw ht_变化≠有效profile重解':''))
              : '；HT源=挥拍中重定相后的最后一条late ht saved（老触球'+tableSigned(rsw.delta)+'ms）')
          : '；重定相因剩余'+(rsw.remain*1000).toFixed(0)+'ms<60ms放弃，HT源退回最后一条accepted')
      : '；HT源=最后一条accepted（本拍无挥拍窗内更新）';
    const acceptedTarget=accepted?tableXzCm(accepted.wx,accepted.wz):'—';
    const carYawAcc=finalHt!=null?botYawDegAt(finalHt):null;
    const carYawRate=finalHt!=null?imuYawRateDegAt(finalHt):null;
    // TCP 只保留一个合同：在 raw 最后更新 HT 插值 /joint_states FK，x/y 以同刻车 yaw 旋到
    // 世界轴；z 用同车型刚性安装高度从 FK 安装面零点平移到机械臂中心地面点 z=0。
    const tcp=finalHt!=null?tcpAt(finalHt):null;
    const tcpYawDeg=carYawAcc;
    const tcpWorld=armPointWorld(tcp,tcpYawDeg,armConstCal.zOff);
    // last accepted 保持上游 /predict_hit_pos 的 rel_x/rel_z 原值；不按 face_yaw 或 HT yaw
    // 重建，也不加减任何位置偏置。这里只做新口径 TCP x/z − 上游原值的数值差。
    const tcpAccepted=accepted&&isNum(accepted.wx)&&isNum(accepted.wz)
      ? [accepted.wx,accepted.wz] : null;
    const tcpAcceptedDx=tcpWorld&&tcpAccepted
      ? (tcpWorld[0]-tcpAccepted[0])*100 : null;
    const tcpAcceptedDz=tcpWorld&&tcpAccepted
      ? (tcpWorld[2]-tcpAccepted[1])*100 : null;
    const tcpAcceptedErrorText=tcpAccepted
      ? tableSigned(tcpAcceptedDx)+'/'+tableSigned(tcpAcceptedDz) : '—/—';
    const tcpCell=tcpWorld
      ? '<span title="'+tableEsc('臂系FK TCP=('+cmFmt(tcp[0])+', '+cmFmt(tcp[1])+', '
          +cmFmt(tcp[2])+')cm @臂最后更新HT '+rowPcFixed(finalHt)+'s（global PC轴；原始ht，未减臂内提前量）'
          +'；按同刻车yaw ψ='+tableSigned(tcpYawDeg)+'° 旋到世界轴'
          +'（xw=x·cosψ−y·sinψ、yw=x·sinψ+y·cosψ）'
          +'；z_world=z_FK−zOffset='+cmFmt(tcp[2])+'−('+cmFmt(armConstCal.zOff)+')='
          +cmFmt(tcpWorld[2])+'cm，zOffset=臂模型z−世界z，原点=机械臂中心地面点z=0'
          +(tcpAccepted
            ? ('；last accepted 直接取上游 /predict_hit_pos rel_x/z=('
               +cmFmt(tcpAccepted[0])+', '+cmFmt(tcpAccepted[1])+')cm，不旋转、不加偏置'
               +'；tcp−last accepted dx/dz='+tcpAcceptedErrorText+'cm')
            : '；last accepted 未回配到上游 rel_x/z，误差不计算')
          +htSrcNote)+'">'
          +tableXyzCm(tcpWorld[0],tcpWorld[1],tcpWorld[2])+', '
          +tcpAcceptedErrorText+'</span>'
      : '—';
    // 保留离 HT 最近的前/后原始曝光，不再用宽窗多项式把原始观测拟合成一个值。
    // V04 的车底盘中心就是机械臂 base，故视觉(世界点−车心)可与同曝光 TCP 直接相减。
    const visPcT=finalHtPcSample;
    const visPair=visPcT!=null?bracketVisualRacketRows(pcRacketRows,visPcT):{before:null,after:null};
    const visRawEntry=(visSrc,sideLabel)=>{
      if(!visSrc) return null;
      const visCar=carAt(visSrc.t);
      if(!visCar||!isNum(visCar.x)||!isNum(visCar.y)) return null;
      const visRel=[visSrc.x-visCar.x,visSrc.y-visCar.y,visSrc.z];
      const visRkT=finalHt!=null&&visPcT!=null?finalHt+(visSrc.t-visPcT):null;
      const visTcp=visualTcpSameOrigin?tcpAt(visRkT):null;
      const visTcpWorld=visTcp?armPointWorld(visTcp,botYawDegAt(visRkT),armConstCal.zOff):null;
      const dx=visTcpWorld?(visRel[0]-visTcpWorld[0])*100:null;
      const dz=visTcpWorld?(visRel[2]-visTcpWorld[2])*100:null;
      const deltaText=visTcpWorld?tableSigned(dx)+'/'+tableSigned(dz):'—/—';
      const dtMs=(visSrc.t-visPcT)*1000;
      const qc=visSrc.blackMarker
        ? ('；四相机黑色拍心标记三角化，质检 max-reproj '
           +(isNum(visSrc.rpMax)?visSrc.rpMax.toFixed(2):'—')+'px / LOO '
           +(isNum(visSrc.looMaxMm)?visSrc.looMaxMm.toFixed(2):'—')+'mm / held-out '
           +(isNum(visSrc.heldoutMaxPx)?visSrc.heldoutMaxPx.toFixed(2):'—')+'px')
        : ('；多相机拍心三角化'+(isNum(visSrc.n_cam)?'，参与相机 '+visSrc.n_cam+' 台':''));
      return {
        html:sideLabel+tableSigned(dtMs)+'ms '+tableXzCm(visRel[0],visRel[2])
          +(visTcpWorld?'<span style="opacity:.65"> Δ'+deltaText+'</span>':''),
        title:sideLabel+'曝光 '+tableSigned(dtMs)+'ms：实测拍心(世界)=('
          +cmFmt(visSrc.x)+', '+cmFmt(visSrc.y)+', '+cmFmt(visSrc.z)+')cm，车心=('
          +cmFmt(visCar.x)+', '+cmFmt(visCar.y)+')cm，拍心−车心=('
          +cmFmt(visRel[0])+', '+cmFmt(visRel[1])+', '+cmFmt(visRel[2])+')cm'
          +(visTcpWorld
            ? '；同曝光 TCP x/z='+tableXzCm(visTcpWorld[0],visTcpWorld[2])
              +'cm，视觉−TCP dx/dz='+deltaText+'cm'
            : '；本车型未确认车心与机械臂 base 同原点，不计算视觉−TCP')
          +qc
      };
    };
    const visEntries=[visRawEntry(visPair.before,'前'),visRawEntry(visPair.after,'后')].filter(Boolean);
    const visCell=visEntries.length
      ? '<span title="'+tableEsc(visEntries.map(v=>v.title).join('；')+htSrcNote)+'">'
          +visEntries.map(v=>v.html).join('<br>')+'</span>'
      : '—';
    // PC真值@臂最后更新HT：与左侧 @对应预测HT 列同一套入弧拟合真值（x/y=拟合球世界−同时刻车世界、
    // z=世界高度），评估时刻=臂最后更新HT（原始值不减臂内提前量，与 TCP 列同锚）。两列之差 =
    // 真值在对应预测HT→臂最后更新HT 这段里真实移动了多少（球速≈10m/s 时 1ms≈1cm）。
    const truthAcc=finalHtPcSample!=null?pcTruthAt(finalHtPcSample):null;
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
    // 规划球出射三量必须来自最后一条 accepted 状态；不从拍面/拍速指令反推，也不把
    // yaw_extra（整拍补偿角）冒充目标出球 yaw。三个字段原子齐全才显示，防状态截断。
    const tgtOutYaw=accepted&&isNum(accepted.tgtOutYaw)?accepted.tgtOutYaw:null;
    const tgtOutPitch=accepted&&isNum(accepted.tgtOutPitch)?accepted.tgtOutPitch:null;
    const tgtOutSpeed=accepted&&isNum(accepted.tgtOutSpeed)?accepted.tgtOutSpeed:null;
    const tgtOutReplay=!!(accepted&&accepted.tgtOutReplay);
    const tgtOutComplete=tgtOutYaw!=null&&tgtOutPitch!=null&&tgtOutSpeed!=null;
    const tgtOutCell=tgtOutComplete
      ? '<span title="'+tableEsc((tgtOutReplay?'离线重放候选（075317 原日志未记录出球三元组）：':'最后一条 accepted 自带的规划球出射需求：')
          +'世界 yaw='+tgtOutYaw.toFixed(2)+'°（atan2(vx,vy)）；pitch='
          +tgtOutPitch.toFixed(2)+'°、speed='+tgtOutSpeed.toFixed(2)+'m/s（RacketPrediction '
          +'在拍面域钳位后保落点重解的球轨迹，不是拍面角/拍心速度，也不是实测回球）')+'">'
          +(tgtOutReplay?'<span style="color:#f3b64c">重放</span> ':'')
          +tgtOutYaw.toFixed(1)+'/'+tgtOutPitch.toFixed(1)+'/'+tgtOutSpeed.toFixed(2)+'</span>'
      : '<span title="该场 accepted 状态未同时记录 out_yaw/out_pitch/out_speed；禁止用 yaw_extra、拍面 pitch 或挥拍 speed 代填">—</span>';
    const faceYaw=faceAnglesWorldAt(finalHt);
    const swingSpeed=racketSpeedAt(finalHt);
    const speedNote=swingSpeed
      ? '。世界拍心速度 |v_world|='+swingSpeed.speedWorld.toFixed(2)+'m/s，v_world='
          +speedVectorText(swingSpeed.world)+'m/s：在同一 RK 物理时刻合成 R(yaw)·v_arm'
          +' + bot_state(vx,vy,0) + yaw_speed×r；所有量都只做 50ms 内有界插值，不外推'
          +'。六轴臂相对 |v_arm|='+swingSpeed.speedArm.toFixed(2)+'m/s，R·v_arm='
          +speedVectorText(swingSpeed.armWorld)+'m/s；车心平移 |v_car|='
          +swingSpeed.speedCar.toFixed(2)+'m/s，v_car='+speedVectorText(swingSpeed.carWorld)
          +'m/s；车体 yaw 刚体项 |ω×r|='+swingSpeed.speedTurn.toFixed(2)+'m/s，ω×r='
          +speedVectorText(swingSpeed.turnWorld)+'m/s（IMU raw yaw_speed='
          +(swingSpeed.yawRate*180/Math.PI).toFixed(2)+'°/s；录制时臂座前移='
          +cmFmt(swingSpeed.armForwardOffsetM)+'cm；未记录的启动 yaw bias 对近期两位小数结果无影响）'
          +'；其中实测 J1 项 |q̇1|·r='+swingSpeed.vJ1.toFixed(2)+'m/s（r='
          +cmFmt(swingSpeed.radiusJ1)+'cm，= status speed= 的口径）'
          +(swingSpeed.cmdJ1!=null?('；同刻指令 J1 项 |v_cmd1|·r='
            +swingSpeed.cmdJ1.toFixed(2)+'m/s ⇒ J1伺服差 '
            +tableSigned(swingSpeed.vJ1-swingSpeed.cmdJ1)+'m/s'):'；本刻无指令帧')
          +(swingSpeed.contactDt!=null?('；臂内触球锚@HT'+tableSigned(swingSpeed.contactDt)
            +'ms（= 指令 J1 速度进入恒 ω 巡航平台的第一帧 = HitTrajectory 的 finish_hit_time，'
            +'不靠实测峰值找）：指令J1 '+swingSpeed.cmdContactJ1.toFixed(2)+'m/s'
            +(swingSpeed.measContactJ1!=null?('、实测J1 '+swingSpeed.measContactJ1.toFixed(2)
              +'m/s ⇒ 触球J1欠速 '
              +tableSigned(swingSpeed.measContactJ1-swingSpeed.cmdContactJ1)+'m/s'):'')):'')
          +(swingSpeed.oscJ1!=null?('；J1单点抖动量级 σ='+swingSpeed.oscJ1.toFixed(2)+'m/s'
            +'（引拍段[−250,−120]ms、肯定无球时的 实测−指令 残差 std：J1 实测速度全程叠着这个'
            +'量级的伺服振荡，所以别拿单帧极值当触球探测器）'):'')
          +(tgtSpeed!=null?('；目标J1(status speed=)='+tgtSpeed.toFixed(2)
            +'m/s，实测J1−目标 '+tableSigned(swingSpeed.vJ1-tgtSpeed)+'m/s'):'')
      : '。本刻无法合成世界拍速：需要 50ms 内同时有六轴 /joint_states、bot_state '
          +'imu_t 上的 yaw/vx/vy、IMU yaw_speed，以及录制时 arm_forward_offset_m；不回退 arm-only';
    const speedTxt=' <span style="color:#a0a0c0">'
      +(swingSpeed?swingSpeed.speedWorld.toFixed(2)+'m/s':'—m/s')+'</span>';
    const faceYawCell=(faceYaw||swingSpeed)
      ? '<span title="'+tableEsc((faceYaw
          ? ('臂系FK face_yaw='+tableSigned(faceYaw.fy)+'°（冲击前窗['
            +'−80,−6]ms '+faceYaw.n+'帧线性外推@臂最后更新HT，ψ̇='+tableSigned(faceYaw.rate)+'°/s）'
            +(isNum(faceYaw.botYaw)
              ? ('；车yaw='+tableSigned(faceYaw.botYaw)+'°（RK /bot_state yaw@imu_t=raw HT）')
              : '；RK raw HT 处无有效 /bot_state yaw，世界yaw显示—')
            +'；世界ψ=face_yaw−车yaw，口径同PC回球yaw=atan2(x,y)；纯RK/Arm值，不受zPhase影响且不含δ6球侧偏置'
            +'。pitch='+tableSigned(faceYaw.pitch)+'°（同窗同帧拟合的 asin(n_z)，正=开面上仰，'
            +'θ̇='+tableSigned(faceYaw.pitchRate)+'°/s）——**不减车yaw**：J1/BASE_ROT 纯 z 转不动 n_z，'
            +'臂系pitch≡世界pitch；θ̇≈0 故 pitch 对时序免疫，与右列 pitch 之差 = θ̇×12ms')
          : '该 HT 冲击前窗内 <2 帧 FK state，拍面角不可用')
          +speedNote+htSrcNote)+'">'
          +(faceYaw?tableSigned(faceYaw.deg)+'/'+tableSigned(faceYaw.pitch):'—/—')
          +speedTxt+'</span>'
      : '—';
    const faceYawPre=faceAnglesWorldPreAt(finalHt);
    // HT−12ms 探针上的世界拍速：同一套三维向量合成与同刻 J1 指令；触球锚/J1振荡是整拍量，
    // 已在左列悬停里给过。与左列之差同时包含臂、车心平移和车 yaw 项的变化。
    const swingSpeedPre=finalHt!=null?racketSpeedRawAt(finalHt-FACE_YAW_PRE_S):null;
    const speedPreNote=swingSpeedPre
      ? '。世界拍心 |v_world|='+swingSpeedPre.speedWorld.toFixed(2)+'m/s，v_world='
          +speedVectorText(swingSpeedPre.world)+'m/s（HT−12ms 同刻向量合成）'
          +'；六轴臂相对='+swingSpeedPre.speedArm.toFixed(2)+'m/s；车心平移='
          +swingSpeedPre.speedCar.toFixed(2)+'m/s；车 yaw 刚体项='
          +swingSpeedPre.speedTurn.toFixed(2)+'m/s'
          +(swingSpeedPre.cmdJ1!=null?('；同刻指令J1='+swingSpeedPre.cmdJ1.toFixed(2)
            +'m/s，实测J1−指令J1='+tableSigned(swingSpeedPre.vJ1-swingSpeedPre.cmdJ1)+'m/s'):'')
          +(swingSpeed?('；世界拍速相对左列(@HT)之差 '
            +tableSigned(swingSpeedPre.speedWorld-swingSpeed.speedWorld)
            +'m/s（含臂、车平移、车 yaw 三项变化；J1单点抖动 σ 见左列悬停）'):'')
      : '。HT−12ms 无法合成世界拍速：六轴、车体或录制配置任一合同缺失即显示—';
    const speedPreTxt=' <span style="color:#a0a0c0">'
      +(swingSpeedPre?swingSpeedPre.speedWorld.toFixed(2)+'m/s':'—m/s')+'</span>';
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
    // 车yaw@raw臂最后更新HT：与右侧拍面/预测真值列同锚；mode=2 时不等于执行接触时刻。
    // 取 /bot_state yaw 瞬时值——车控 accept AprilTag 定位后 yaw 由 IMU 连续更新、
    // HT 结束后才用 AprilTag 重定位，故 HT 前采样点无重定位台阶；挥拍位姿伪迹只塌陷位置不动 yaw。
    // 悬停给 IMU yaw_speed（零滞后）换算的 10ms 时序灵敏度；右侧主拍面 yaw 直接减同一个 RK yaw。
    const carYawCell=carYawAcc!=null
      ? '<span title="'+tableEsc('/bot_state yaw@臂最后更新HT '+rowPcFixed(finalHt)+'s（global PC轴，'
          +'臂受理的最后一条预测的原始 ht，未减臂内提前量）：AprilTag accept 后由 IMU 连续更新，'
          +'HT 结束后才重定位，采样点无台阶'
          +(carYawRate!=null?('；IMU yaw_speed='+tableSigned(carYawRate)+'°/s（零滞后陀螺，'
            +'10ms 时序误差≈'+(carYawRate>=0?'+':'')+(carYawRate*0.01).toFixed(2)+'°车yaw）'):'；本场无 IMU yaw_speed'))+'">'
          +tableSigned(carYawAcc)+'</span>'
      : '<span title="无 accepted/臂最后更新HT，或该时刻 50ms 内无 /bot_state">—</span>';
    // 最后更新→挥拍起：臂受理的最后一条预测的到达时刻 − 挥拍段起点(老触球−HIT_T)。
    // 正=更新落在挥拍开始之后（0803 起挥拍窗内不再拒收，这些就是 ht 重定相的养料）。
    const updT=accepted&&isNum(accepted.lastUpdateT)?accepted.lastUpdateT:null;
    const swingStart=accepted&&isNum(accepted.start)?accepted.start:null;
    const updGapCell=(updT!=null&&swingStart!=null)
      ? '<span title="'+tableEsc('最后受理更新 t='+rowPcFixed(updT)+'s（global PC轴）'
          +'；挥拍起 t='+rowPcFixed(swingStart)+'s（=老触球−HIT_T 0.25s）'
          +'；最后一条 accepted t='+rowPcFixed(accepted.lastAcceptT)+'s（其后 '
          +(rsw?rsw.n:0)+' 条只存触球时刻不换目标）'
          +(rsw
            ? (rsw.continuousSweep
              ? '；连续sweep收到并写入ht_ '+rsw.n+' 条late raw HT：老触球 '
                +rowPcFixed(rsw.oldDone)+'s → 最后raw请求 '+rowPcFixed(rsw.newDone)
                +'s（'+tableSigned(rsw.delta)+'ms）'
                +(rsw.nCoast?('；'+rsw.nCoast+' 条回报mode=2 Coast=骑旧profile，不代表轨迹采用该raw HT'):'')
              : '；重定相触发 t='+rowPcFixed(rsw.trig)+'s（老触球−100ms）：老触球 '
                +rowPcFixed(rsw.oldDone)+'s → 新触球 '+rowPcFixed(rsw.newDone)+'s（'
                +tableSigned(rsw.delta)+'ms，剩余 '+(rsw.remain*1000).toFixed(0)+'ms）'
                +(rsw.ok?' ✓生效':' ✗剩余<60ms放弃'))
            : '；本拍挥拍窗内无新预测，未触发重定相'))+'">'
          +tableSigned((updT-swingStart)*1000)+'</span>'
      : '—';
    // 盲区 ht−ct@臂最后更新：最终那条命令的「击球点时刻 − 它最晚看到的那颗球的观测时刻」。
    // 这段时间里预测纯外推、没有任何新观测进来，是本拍真正的信息盲区。
    const finalCt=accepted&&isNum(accepted.finalCt)?accepted.finalCt-RK.t0:null;
    const blindBad=!!(accepted&&accepted.finalMismatch);
    const blind=(finalHt!=null&&finalCt!=null&&!blindBad)?(finalHt-finalCt)*1000:null;
    const blindCell=blind!=null
      ? '<span title="'+tableEsc('臂最后更新消息 ct='+rowPcFixed(finalCt)+'s（最晚一颗球的观测时刻）'
          +'、ht='+rowPcFixed(finalHt)+'s（global PC轴；预测击球时刻，原始值未减臂内提前量'
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
          +' title="'+tableEsc('挥拍时 ht='+rowPcFixed(rsw.oldDone+HIT_TIME_ADVANCE_SEC)
          +'s → 更新后 ht='+rowPcFixed(rsw.newDone+HIT_TIME_ADVANCE_SEC)+'s'
          +'；正=新预测把击球点推晚。触球拍速是派生量，会随剩余时长一起变'
          +(rsw.continuousSweep?'；连续sweep仅确认该raw HT进入update_ht，是否形成有效profile须看mode/执行轨迹':'')
          +(rsw.ok?'':'；剩余'+(rsw.remain*1000).toFixed(0)+'ms<60ms，控制器放弃重定相，实际仍走老时间轴'))+'">'
          +tableSigned(rsw.delta)+(rsw.ok?'':'(未采纳)')+'</span>'
      : '<span title="本拍挥拍窗内没有新预测到达，ht 自挥拍起未变">—</span>';
    const rejectNote=accepted?'—':rejectNoteForThrow(th);
    const hitAnchorPc=accHt!=null?pcSampleTimeForThrow(th,accHt)
      :(isNum(th.ht)?pcSampleTimeForThrow(th,th.ht):null);
    const ret=pcReturnAt(hitAnchorPc);
    const returnDtMs=ret&&finalHtPcSample!=null?(ret.tHit-finalHtPcSample)*1000:null;
    const returnCell=ret
      ? '<span title="'+tableEsc('回球触球t='+ret.tHit.toFixed(3)+'s（PC入/出弧交点法）'
        +(returnDtMs==null?'':'；实际触球−raw臂最后更新HT的PC取样锚='+tableSigned(returnDtMs)+'ms（只在PC查询侧加本抛zPhase）')
        +'；v=('+ret.vx.toFixed(2)+', '+ret.vy.toFixed(2)+', '+ret.vz.toFixed(2)+')m/s；出弧'+ret.n+'点/'+Math.round(ret.span*1000)+'ms，段首距触球+'+Math.round(ret.start*1000)+'ms；max|res|='+(ret.maxRes*100).toFixed(1)+'cm'+(ret.bounceCut?'；地面反弹前截断':''))+'">'+
        tableSigned(ret.yaw)+'/'+tableSigned(ret.pitch)+' <span style="color:#a0a0c0">'+ret.speed.toFixed(1)+'m/s'
        +(returnDtMs==null?'':' Δt'+tableSigned(returnDtMs)+'ms')+'</span></span>'
      : '—';
    // 实测等效 e_n：PC 入/出球世界速度与同一实际触球时刻的世界拍心速度都先投影到
    // 拍面前向法线，再按 u_out,n=-e_n*u_in,n 求解。PC/RK world 轴沿用本场共同场地标定合同；
    // tHit 通过本抛 zPhase 从 PC 取样轴反解回 RK，不能拿 raw HT 或固定 HT−12ms 代替。
    const enHitRk=ret&&finalHt!=null&&finalHtPcSample!=null
      ? finalHt+(ret.tHit-finalHtPcSample) : null;
    const enFaceTimeOk=enHitRk!=null&&enHitRk>=finalHt-0.08&&enHitRk<=finalHt+0.02;
    const enFace=enFaceTimeOk?faceAnglesWorldAt(finalHt,enHitRk):null;
    const enRacket=enFaceTimeOk?racketSpeedRawAt(enHitRk):null;
    const enNormal=enFace&&enFace.deg!=null?faceNormalWorld(enFace.deg,enFace.pitch):null;
    const enResult=ret&&ret.incoming&&enRacket&&enNormal
      ? racketNormalRestitution(
          [ret.incoming.vx,ret.incoming.vy,ret.incoming.vz],
          [ret.vx,ret.vy,ret.vz],enRacket.world,enNormal)
      : null;
    const enMissingReason=!ret
      ? 'PC 入/出弧未同时通过回球质量门'
      : (!ret.incoming
        ? 'PC 来球末段三轴拟合未通过：需 ≥5 点、跨度 ≥60ms、vy<−1m/s、max|残差|≤12cm'
        : (enHitRk==null
          ? '缺 raw臂最后更新HT 或本抛 zPhase，无法把实际触球时刻映回 RK'
          : (!enFaceTimeOk
            ? '实际触球距 raw HT 超出冲击前拍面拟合可信区间 [−80,+20]ms'
            : (!enFace||enFace.deg==null
              ? '实际触球时刻缺冲击前拍面法向或同刻车 yaw'
              : (!enRacket
                ? '实际触球时刻缺六轴+车体合成的世界拍心速度'
                : '法向闭合速度 ≤1m/s，分母不可信')))));
    const enCell=enResult
      ? '<span'+((enResult.en<0||enResult.en>1)?' style="color:#e0a24a"':'')
          +' title="'+tableEsc('实测 e_n,eff=−u_out,n/u_in,n='
            +enResult.en.toFixed(4)+'；u_in,n=(v_in−v_r)·n='
            +enResult.uInN.toFixed(3)+'m/s，u_out,n=(v_out−v_r)·n='
            +enResult.uOutN.toFixed(3)+'m/s，法向闭合速度='
            +enResult.closing.toFixed(3)+'m/s'
            +'；v_in='+speedVectorText([ret.incoming.vx,ret.incoming.vy,ret.incoming.vz])
            +'m/s，v_out='+speedVectorText([ret.vx,ret.vy,ret.vz])
            +'m/s，冲击前世界拍心 v_r='+speedVectorText(enRacket.world)+'m/s'
            +'；拍面前向 n='+speedVectorText(enResult.normal)+'，yaw/pitch='
            +tableSigned(enFace.deg)+'/'+tableSigned(enFace.pitch)+'°'
            +'；实际触球 tPC='+ret.tHit.toFixed(3)+'s → tRK='+enHitRk.toFixed(3)
            +'s（相对 raw HT '+tableSigned((enHitRk-finalHt)*1000)+'ms，本抛 zPhase 反解）'
            +'；来球拟合 '+ret.incoming.n+'点/'+Math.round(ret.incoming.span*1000)
            +'ms，max|残差|='+(ret.incoming.maxRes*100).toFixed(1)+'cm'
            +'；同一冲击前 v_r 同时用于入/出相对速度。这是拍心刚性速度近似下的逐拍等效值，'
            +'不是材料常数，也不是落地 cor_z/cxy 或世界y等效e；负值/>1保留诊断，不钳位')+'">'
          +enResult.en.toFixed(3)+'</span>'
      : '<span title="'+tableEsc(enMissingReason)+'">—</span>';
    // RK 全量无污染观测（球世界三轴拟合 × 车实际 x/y 外推）在臂最后更新HT 上量出的真值，
    // 供右边两列共用：dy = 球面到车 y 面的缺口 = 时序误差的空间形态；
    // dx/dz = 球心相对车中心的落点（拍面上的位置，不扣球半径）= 击球点真值。
    const gapFin=finalHt!=null?ballCarGapForThrow(th,finalHt):null;
    // [[hit-plane-shift-core-begin]]
    // 触球平面不再≡车 y 面：arm_controller 2026-08-11 起把整条挥拍弧绕竖直轴多转 δ
    //（kHitYawExtraRad，补出球 yaw 系统右偏），触球点在车系前移 目标x×sinδ；成对的
    // bot_center kHitPlaneStandoffM 让车 travel 目标 y 后退同量，世界系接触面才不动。
    // 用户 2026-08-12 拍板：**列的口径不动**（还是"球面y−车y"、还是锚臂最后更新HT），
    // 这里只把零点在悬停里写清楚——否则 dy=+4cm 会被按老读法当成"球还没够到、ht 偏早3ms"，
    // 而扣掉平面前移 9.1cm 后真相是"球已穿过拍面5cm、ht 偏晚4ms"，符号是反的。
    // δ 不写死：由本场自标定的 x 比例反解（臂端同一处变换 x/=cosδ ⇒ δ=acos(1/xScale)），
    // 下场臂端改 δ 这里自动跟上；δ=0（xScale=1 或没标定出来）时整段不出现，与改前逐字相同。
    const hitYawExtraRad=(armConstCal.xScale!=null&&armConstCal.xScale>1)
      ? Math.acos(1/armConstCal.xScale) : 0;
    // 前移量按该拍 accepted 目标 x 算（=挥拍半径 r，臂端已把它预除 cosδ 还原侧向），逐拍不同。
    const hitPlaneShift=(hitYawExtraRad>0&&accepted&&isNum(accepted.tx))
      ? accepted.tx*Math.sin(hitYawExtraRad) : 0;
    const hitPlaneNote=(hitPlaneShift>0&&gapFin)
      ? '；⚠本场臂端整体多转 δ='+(hitYawExtraRad*180/Math.PI).toFixed(1)+'°（由本场自标定 x 比例 '
        +armConstCal.xScale.toFixed(5)+'=1/cosδ 反解），触球平面在车系前移 目标x×sinδ='
        +cmFmt(hitPlaneShift)+'cm，故「本列零点是 +'+cmFmt(hitPlaneShift)+'cm，不是 0」：'
        +'扣掉后球面到拍面 '+cmSigned(gapFin.dy-hitPlaneShift)+'cm'
        +((isNum(gapFin.vRel)&&gapFin.vRel<-1e-6)
          ? '、等效时序 '+tableSigned((gapFin.dy-hitPlaneShift)/gapFin.vRel*1000)+'ms' : '')
        +'（成对的车侧 standoff 让车后退同量，世界系接触面不动；'
        +'这只是平面位置的几何项，不含 FK 模型偏差 F≈6cm）'
      : '';
    // [[hit-plane-shift-core-end]]
    const truthNote=gapFin
      ? '真值取 RK 全量无污染观测@臂最后更新HT '+rowPcFixed(finalHt)+'s（global PC轴；世界轴，不转车yaw）：'
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
    // 球面y−车y @臂最后更新HT：零点=该 ht 正好是真实触球，正=球面还没够到（ht 偏早），负=已穿过（偏晚）。
    // ⚠零点只在臂端 δ=0 时等于 0，δ≠0 时是 +目标x×sinδ，见 [[hit-plane-shift-core]]（口径不动，只标零点）。
    const gapCell=gapFin
      ? '<span title="'+tableEsc('dy=(球心y−R球3.3cm)−车y='+cmSigned(gapFin.dy)+'cm；车y面'
          +(hitPlaneShift>0?'+目标x×sinδ=拍面':'≡拍面')
          +'（RK rel_y 无臂基y补偿），故'+(hitPlaneShift>0?'零点':'0')+'=球面刚够到=真实触球'
          +hitPlaneNote
          +'；闭合速度|v_rel|='+Math.abs(gapFin.vRel).toFixed(2)+'m/s（|vy球|='
          +Math.abs(gapFin.vy).toFixed(2)+'m/s）'
          +(isNum(gapFin.dtMs)?('，等效时序='+tableSigned(gapFin.dtMs)+'ms（正=该ht比真实触球晚'
            +(hitPlaneShift>0?'，按零点0算的老口径值，δ 修正见上':'')+'）'):'')
          +'；'+truthNote
          +(isNum(gapFin.eA)?('；车@ht−冻结面lastS0Y='+cmFmt(gapFin.eA)+'cm'):''))+'">'
          +cmSigned(gapFin.dy)+'</span>'
      : gapMiss;
    // 末次target对应预测的击球点(x,z) − 击球点真值@臂最后更新HT，世界轴、两侧都相对各自的车：
    // 预测侧取末次target对应消息自带的世界击球点 x 减它同条给出的车 x（car_pred_x），
    // z 直接用 rel_z（车中心 z≡地面0，yaw 不转 z，故 rel_z 就是世界 z）——这样两侧同轴同基准，
    // 不掺车体系↔世界轴的 yaw 旋转。
    // 正 = 预测的击球点比真实球位更靠 +x / 更高，即臂被瞄到球的右侧/上方。
    const aimDx=(gapFin&&targetPred&&isNum(targetPred.worldX)&&isNum(targetPred.carX))
      ? (targetPred.worldX-targetPred.carX)-gapFin.dx : null;
    const aimDz=(gapFin&&targetPred&&isNum(targetPred.relZ))?targetPred.relZ-gapFin.dz:null;
    const aimCell=(aimDx!=null&&aimDz!=null)
      ? '<span title="'+tableEsc('dx=预测(消息世界x−car_pred_x)'
          +cmFmt(targetPred.worldX-targetPred.carX)+'cm − 真值(球心x−车x)'
          +cmFmt(gapFin.dx)+'cm='+cmSigned(aimDx)+'cm'
          +'；dz=预测rel_z '+cmFmt(targetPred.relZ)+'cm − 真值球心z'
          +cmFmt(gapFin.dz)+'cm='+cmSigned(aimDz)+'cm'
          +'；正=预测点在真实球位的 +x 侧/上方（臂瞄偏的方向）'
          +'；两侧都是世界轴、各自相对各自的车（预测用同条消息的 car_pred_x，真值用车实际x），'
          +'故不含 yaw 旋转项；预测取末次target对应消息（ht='
          +(targetPredHtBaseline!=null?targetPredHtBaseline.toFixed(3):'—')+'s，global PC轴）'
          +'；真值与左列「球面y−车y」是同一份 RK 全量拟合、同一个取值时刻（臂最后更新HT）'
          +'；'+truthNote)+'">'
          +cmSigned(aimDx)+'/'+cmSigned(aimDz)+'</span>'
      : (gapFin?'<span title="本抛无可回配的末次target对应预测消息，无预测击球点">—</span>':gapMiss);
    const targetInfo=targetPred
      ? '末次target S'+tableFmt(targetPred.stage,0)
        +(isNum(targetPred.nFit)?' n_fit='+targetPred.nFit:'')
        +' lead='+(targetPred.lead*1000).toFixed(1)+'ms / msgs='+(th.msgs||0)
      : '末次target无对应预测 / msgs='+(th.msgs||0);
    if(!targetPred||!isNum(targetPred.ht)||!isNum(targetPred.relX)||!isNum(targetPred.relZ)){
      return '<tr><td>'+(idx+1)+'</td><td>'+zPhaseCell+'</td><td>'+targetUpdateCell+'</td><td>'+targetPredHt+'</td><td>'+targetPredHit+'</td><td>'+targetActualAtHtError+'</td><td>'+predActualAtHtError+'</td><td>'+acceptedTarget+'</td>'+
        '<td>'+pcTruthCell(targetTruth,true,targetPredSamplePc,true)+'</td><td>'+pcTruthCell(truthAcc,true,finalHtPcSample)+'</td><td>'+tcpCell+'</td><td>'+visCell+'</td><td>'+updGapCell+'</td>'+
        '<td>'+blindCell+'</td><td>'+dHtCell+'</td><td>'+gapCell+'</td><td>'+aimCell+'</td>'+
        '<td>'+carYawCell+'</td><td>'+tgtSpeedCell+'</td><td>'+tgtOutCell+'</td>'+
        '<td>'+faceYawCell+'</td><td>'+faceYawPreCell+'</td>'+
        '<td>'+returnCell+'</td><td>'+enCell+'</td>'+
        '<td>'+targetInfo+'</td><td class="armTblNote"><div>'+rejectNote+'</div></td><td>'+racketCorCell+'</td></tr>';
    }
    return '<tr><td>'+(idx+1)+'</td>'+
      '<td>'+zPhaseCell+'</td>'+
      '<td>'+targetUpdateCell+'</td>'+
      '<td>'+targetPredHt+'</td>'+
      '<td>'+targetPredHit+'</td>'+
      '<td>'+targetActualAtHtError+'</td>'+
      '<td>'+predActualAtHtError+'</td>'+
      '<td>'+acceptedTarget+'</td>'+
      '<td>'+pcTruthCell(targetTruth,true,targetPredSamplePc,true)+'</td>'+
      '<td>'+pcTruthCell(truthAcc,true,finalHtPcSample)+'</td>'+
      '<td>'+tcpCell+'</td>'+
      '<td>'+visCell+'</td>'+
      '<td>'+updGapCell+'</td>'+
      '<td>'+blindCell+'</td>'+
      '<td>'+dHtCell+'</td>'+
      '<td>'+gapCell+'</td>'+
      '<td>'+aimCell+'</td>'+
      '<td>'+carYawCell+'</td>'+
      '<td>'+tgtSpeedCell+'</td>'+
      '<td>'+tgtOutCell+'</td>'+
      '<td>'+faceYawCell+'</td>'+
      '<td>'+faceYawPreCell+'</td>'+
      '<td>'+returnCell+'</td><td>'+enCell+'</td>'+
      '<td>'+targetInfo+'</td><td class="armTblNote"><div>'+rejectNote+'</div></td><td>'+racketCorCell+'</td></tr>';
  });
  window.__armHitContract={
    schema:'arm_final_ht/v4',
    rkT0:RK.t0,
    baselineTrusted:throwBaselineTrusted,
    unmappedReportRows:throwPhases.filter(p=>p.rkFlight==null||p.pcFlight==null)
      .map(p=>p.reportRow),
    zPhasePolicy:{maxAbsOffsetMs:THROW_PHASE_MAX_OFFSET_S*1000,
                  appliesTo:'all_pc_sampling',rkUse:'global_baseline'},
    calibration:{zOff:armConstCal.zOff,xScale:armConstCal.xScale,
                 advance:armConstCal.adv,n:armConstCal.n,total:armConstCal.total},
    rows:armContractRows,
  };
  const physicalPairMap=new Map();
  throwPhases.filter(p=>p.rkFlight!=null&&p.pcFlight!=null).forEach(p=>{
    const key=p.rkFlight+'/'+p.pcFlight;
    if(!physicalPairMap.has(key)) physicalPairMap.set(key,p);
  });
  const physicalPhases=[...physicalPairMap.values()];
  const usablePhysical=physicalPhases.filter(p=>p.usable);
  const alignedRows=throwPhases.filter(p=>p.usable).length;
  const phaseSummaryGood=reportThrows.length===0 ||
    (physicalPhases.length>0&&usablePhysical.length===physicalPhases.length);
  const throwAlignSummary='<div style="font-size:11px;color:'+
    (phaseSummaryGood?'#7fbf9f':'#e0a24a')+
    ';margin:0 0 6px">逐抛 PC 取样 zPhase：'+usablePhysical.length+'/'+physicalPhases.length+
    ' 个独立 PC/RK flight pair 通过（报告行 '+alignedRows+'/'+reportThrows.length+' 可用）；失败只隐藏依赖 local PC 取样的格（含视觉），'
    +'RK字段仍用global baseline。offset 范围 '+(usablePhysical.length
      ? tableSigned(Math.min(...usablePhysical.map(p=>p.deltaMs))):'—')+'～'+(usablePhysical.length
      ? tableSigned(Math.max(...usablePhysical.map(p=>p.deltaMs))):'—')+'ms。</div>';
  return throwAlignSummary+'<div class="armTblWrap"><table class="armTbl"><thead><tr>'+
    '<th>#</th><th title="每抛用同一颗球的 PC/RK z(t) 在global baseline上估一个不超过100ms的PC取样offset；HT只粗配flight，不进入评分。所有PC取样（含视觉）应用该offset，RK字段不应用">PC取样 zPhase<br>offset(ms)</th>'+
    '<th title="上行=RUN内最后一次可观察 raw target_x/y 坐标变化的 bot_state.t；下行=用该帧 t+remaining 回配到同抛 pred.ht 后，那条预测消息自己的 payload ct。两者只按global baseline映射，逐抛offset不改RK时间">最后改target t / 对应ct<br>(s,global PC轴)</th>'+
    '<th>对应预测 HT<br>(s,global PC轴)</th>'+
    '<th>对应预测击球 rel_x/z(cm)</th>'+
    '<th title="末次 target_x/y − /bot_state 实际 x/y；实际位置严格在 imu_t=target deadline 上有界插值，phase/target/deadline 仍按 bot_state.t 事件轴">末次target目标−实际车@HT<br>dx/dy(cm)<br>(RK世界系)</th>'+
    '<th title="同一条对应预测的 car_pred_x/y − /bot_state 实际 x/y；两侧都在 pred.ht，实际位置严格按 imu_t 有界插值。对应消息仍由最后 target 变化帧的 bot_state.t+remaining 与 pred.ht 回配；Stage0 的 car_pred 是 raw target，不冒充轨迹预测">末次target对应预测车@HT<br>−实际车@HT dx/dy(cm)<br>(RK世界系)</th>'+
    '<th title="'+tableEsc('最后一条 /tennis/status accepted 回配到它实际消费的上游 '+
      '/predict_hit_pos；本列直接显示 payload rel_x/rel_z 原值，不做 face_yaw、车 yaw 或位置偏置变换。'+
      '回配成功率 '+_armHit.nMatch+'/'+_armHit.nAcc+' 条（主键=序号对齐 δ='+
      (armPredAlign.delta!=null?armPredAlign.delta:'—')+'）')+'">'+
    '机械臂最后accepted目标<br>x/z(cm)</th>'+
    '<th title="x=球心world_x−车体中心world_x；y=(球心world_y−R球3.3cm)−车体中心world_y；z=球心world_z。三轴在global HT映射后只给PC取样时刻加本抛zPhase，世界轴不转车yaw">PC真值@对应预测HT+zPhase<br>x/y/z(cm)<br>(y为球接触面)</th>'+
    '<th title="与左列同一套入弧拟合真值，只把global评估锚换成raw臂最后更新HT，再给PC取样时刻加本抛zPhase。mode=2 Coast时'+
    '该raw HT不是机械臂执行接触锚；右侧 TCP 仍严格使用这个 raw HT，不另找过面时刻'+
    '（连续sweep时=最后进入update_ht的 late raw ht，否则=最后一条 accepted raw ht；'+
    '未减臂内提前量）。两列之差=真值随 ht 变化移动的量'+
    '（球速≈10m/s 时 1ms≈1cm）">'+
    'PC真值@臂最后更新HT+zPhase<br>x/y/z(cm)</th>'+
    '<th title="'+tableEsc('前三个数严格使用 raw 臂最后更新HT：TCP x/y 按同刻车yaw旋到世界轴，'+
    '不做车心平移；z 用车型刚性安装高度从 FK 安装面零点换到机械臂中心地面点 z=0。'+
    '后两个直接用该 TCP x/z 减上游 last accepted 的原始 rel_x/rel_z；accepted 侧不做 face_yaw、'+
    '车 yaw 或位置偏置变换。mode=2 Coast 也不换时刻。'+armFkCarNote+'。悬停看 yaw 与 z 高度换算')+'">'+
    'TCP@臂最后更新HT x/y/z(cm,世界轴)<br>相对机械臂中心地面点z=0<br>tcp−last accepted（dx，dz）</th>'+
    '<th title="'+tableEsc((racketBlackMarker
      ? '四相机黑色拍心标记中心三角化'
      : '离线多相机实测拍心三角化')+
    '−同曝光车心，世界轴。每格只保留距 raw 臂最后更新HT+zPhase 35ms 内最近的前、后原始曝光和 signed dt，'+
    '不做宽窗拟合。'+(visualTcpSameOrigin
      ? '本场 V04 的车底盘中心=机械臂 base，因此同时显示视觉−同曝光TCP dx/dz。'
      : '本车型未确认车心与机械臂 base 同原点，不计算视觉−TCP。')+
    '悬停看世界坐标、车心、同曝光 TCP 与每帧质检。')+'">'+
    '视觉拍心−车心@臂最后更新HT+zPhase前/后<br>x/z(cm,世界轴)<br>视觉−同曝光TCP（dx，dz）</th>'+
    '<th title="臂受理的最后一条 /predict_hit_pos 到达时刻 − 挥拍段起点(老触球−HIT_T 0.25s)；'+
    '正=更新落在挥拍开始之后（0803 起挥拍窗内不再拒收，这些消息就是 ht 重定相的养料）。'+
    '悬停看两个绝对时刻与重定相是否生效">最后更新−挥拍起<br>(ms)</th>'+
    '<th title="定义：finalHt−finalCt，二者取自最后进入 update_ht 的 /predict_hit_pos 原始 '+
    'ht/ct（同源、都不减臂内提前量）。连续sweep时=最后一条带 sweep_w 的 late ht saved（状态即代表'+
    '已被 update_ht 消费）；旧版重定相生效时=最后一条被采纳的 late ht saved；未触发/剩余不足时=最后一条 accepted'+
    '（典型 300~335ms）。这段时间预测纯外推、没有任何新观测进来，即本拍的信息盲区。'+
    '⚠— = 重定相生效但该条回配失败，拿不到同源 ht/ct，不出数（不退回 accepted 冒充）">'+
    '盲区 ht−ct<br>@臂最后更新HT<br>(ms)</th>'+
    '<th title="挥拍时用的 ht（最后一条 accepted）→ 挥拍中重定相更新后的 ht，正=新预测把击球点推晚；'+
    '两者同为原始 ht，与臂内触球时刻（各减同一个本场自标定提前量）之差等价。灰字(未采纳)=剩余不足 60ms 控制器放弃了重定相">'+
    'Δht 重定相<br>(ms)</th>'+
    '<th title="在最后进入 update_ht 的 raw /predict_hit_pos HT 上，用 RK 全量无污染观测（球世界三轴二次拟合 × '+
    '车实际x/y挥拍前外推）量出的 (球心y−R球3.3cm)−车y，世界轴，单位cm。车y面≡拍面，'+
    '故 0=该 ht 正好是真实触球，正=那一刻球面还没够到拍面(ht 偏早)，负=球已穿过(ht 偏晚)。'+
    '悬停看闭合速度、等效时序(ms)与球/车两侧的拟合明细">'+
    '球面y−车y<br>@臂最后更新HT<br>(cm, RK全量真值)</th>'+
    '<th title="末次target对应消息预测的击球点 (x,z) − 真值 (x,z)，单位cm。'+
    '这里的「真值」与左列同一份：RK 全量无污染观测（球世界三轴二次拟合 × 车实际x/y挥拍前外推，'+
    '不含任何预测量），取值时刻同样是raw臂最后更新HT；mode=2 Coast时它不是机械臂执行接触锚。'+
    '两侧都是世界轴且各自相对各自的车：预测侧=消息世界x−同条的 car_pred_x、z 用 rel_z(车中心z≡地面0)；'+
    '真值侧=球心x−车实际x、球心z。故不含车体系↔世界轴的 yaw 旋转项。'+
    '正=预测点落在真实球位的 +x 侧/上方，即臂被瞄偏的方向">'+
    '击球点@末次target预测<br>− RK全量真值@臂最后更新HT<br>dx/dz(cm, 世界轴)</th>'+
    '<th title="车体 yaw@臂最后更新HT（最后进入update_ht的预测原始ht，未减臂内提前量；与左侧两列'+
    '击球真值、右侧两列拍面yaw,pitch 同锚）：取 /bot_state 瞬时值——车控 accept AprilTag 定位后 yaw 由 '+
    'IMU 连续更新、HT 结束后才重定位，故采样点无重定位台阶，挥拍位姿伪迹只塌陷位置不动 yaw。'+
    '悬停看 IMU yaw_speed 换算的 10ms 时序灵敏度；右侧拍面yaw,pitch@臂最后更新HT 列直接减同一 RK yaw，不读取PC yaw">'+
    '车yaw@臂最后更新HT<br>(°)</th>'+
    '<th title="该次挥拍最后一条 accepted 状态自带的计划量：目标触球拍速（m/s，拍心，过完各级'+
    '钳位后的实际计划值，口径 2·|行程|/hit_time·x 且只算 J1）/ 目标拍面仰角（°，0805 起随来球'+
    '俯冲角变，臂系≡世界系）。悬停看 speed_req(原始指令)、shortened(引拍被夹 rad)、face_yaw 目标。'+
    '注意：speed 是受理时按零起速算的账，挥拍段在首帧建、并在 ht 重定相触发点用当刻(q,v)重建，'+
    '起速非零会抬高真实触球速度，与右列实测对照时以右列悬停的「同刻指令侧」为准">'+
    '目标挥拍速度/pitch<br>(m/s, °)</th>'+
    '<th title="最后一条 accepted 状态原子记录的规划球出射需求：世界 yaw=atan2(vx,vy)；pitch/speed 为 RacketPrediction 在拍面域钳位后、保落点重解得到的球轨迹。不是 yaw_extra、拍面角、拍心速度，也不是 PC 实测回球">'+
    '目标规划出球<br>yaw/pitch/speed<br>(°,°,m/s;世界系)</th>'+
    '<th title="拍面法向（车型配置轴；V04 为 FK link6 +Y）的世界 yaw / pitch，同一份冲击前窗[−80,−6]ms 线性拟合；'+
    '灰字为世界拍心速度 |v_world|：六轴 Jacobian 三维速度旋到世界系后，向量叠加同刻车心 vx/vy 与 '+
    'IMU yaw_speed×(车心到拍心杆臂)；车体位姿/速度按 bot_state imu_t，IMU t 已是物理采样时刻。'+
    '任一合同缺失即显示—，不退回 arm-only。'+
    'yaw=face_yaw−车yaw（车侧直接取 RK /bot_state yaw@imu_t=raw HT，不受PC zPhase影响）；'+
    'pitch=asin(n_z) 不减车yaw——J1/BASE_ROT 纯 z 转'+
    '不动 n_z，臂系 pitch≡世界 pitch，正=开面上仰。悬停看两个角速度（ψ̇ 几百°/s 时序敏感、'+
    'θ̇≈0 对时序免疫），以及六轴臂相对、车心平移、车 yaw 刚体项三个速度分量；J1 指令/实测、'+
    '臂内触球锚与抖动只按 J1 同口径比较">'+
    '拍面yaw,pitch / 世界拍心speed<br>@臂最后更新HT<br>(°,°,m/s;世界系)</th>'+
    '<th title="同左列同一份拟合/同一套拍速口径，只把取值时刻挪到 HT−12ms（角度落在窗内为纯插值，'+
    '拍速为插值）。−12ms 是固定探针，且本场用指令速度平台首帧定位的臂内触球锚中位就在 HT−11ms，'+
    '故本列基本踩在真实触球上；与左列之差 = 12ms 时序误差的代价——yaw 上是 ψ̇×12ms（度级）、'+
    'pitch 上≈0、世界拍速差同时含臂、车心平移和车 yaw 三项变化">'+
    '拍面yaw,pitch / 世界拍心speed<br>@臂最后更新HT−12ms<br>(°,°,m/s;世界系)</th>'+
    '<th title="Δt=PC入/出弧交点触球时刻−raw臂最后更新HT的PC取样锚；RK HT保持global baseline，只在PC查询/比较侧加本抛zPhase，质量不过门则本格—">PC回球<br>yaw/俯仰(°) / speed / Δt(ms)</th>'+
    '<th title="逐拍实测等效法向恢复系数：e_n=−((v_out−v_r)·n)/((v_in−v_r)·n)。PC 来/回球、完整六轴+车体世界拍心速度和拍面前向法线都取同一实际触球时刻；tHit 用本抛 zPhase 从 PC 轴反解到 RK。缺任一合同或法向闭合速度≤1m/s即—，不回退总速标量/J1；负值或>1保留诊断。它不是材料常数，也不是落地 cor_z/cxy 或世界y等效e。">实测 e_n,eff<br>(拍面法向)</th><th>消息</th><th>备注</th>'+
    '<th style="white-space:nowrap" title="racket_impact/v3 按 contact_anchor_t_rk 保序回配；该时刻是RK ball_world固定高度外推锚点，不是视频直接观测impact。拍头bbox中心速度拟合窗口为锚点前125ms至后35ms。触球锚与vision_evaluated分层显示；只有状态一致，且measurement、bundle、原始bbox观测、帧数/时跨/重投影证据齐全的accepted记录才显示vz。vz_world_mps是球拍头bbox几何中心bundle世界z速度代理，不是校准拍面速度，也不是直接观测球自转；拍头上行/下行只推导上旋/下旋倾向，且未用于本场S0。S0 cor_xy/cor_z、RK弹后实测复算、历史旧rvz是三条独立证据。">PC视频判型 / S0 cxy, cor_z / 实测反弹 cxy, cor_z / RK旧拍头vz</th></tr></thead>'+
    '<tbody>'+rows.join('')+'</tbody></table></div>';
};

const armAcceptedTableHtml = () => {
  if(!ARM) return '';
  if(!armAligned) return '<div style="color:#f87171">机械臂数据未与RK单调钟对齐，无法可靠回配最后accepted原消息。</div>';
  const hits=armHitMarks.filter(h=>h.label==='hit');
  if(!hits.length) return '<div style="color:#a0a0c0">本场没有 accepted hit。</div>';
  const rows=hits.map(h=>{
    const sourceOk=isNum(h.wct)&&isNum(h.wht)&&isNum(h.wx)&&isNum(h.wz);
    if(!sourceOk){
      return '<tr><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>无法回配</td>'+
        '<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>updates='+(h.n||1)+'</td><td>—</td></tr>';
    }
    const accCt=h.wct-RK.t0;
    const accHt=h.wht-RK.t0;
    const th=matchThrowByAcceptedCt(accCt);
    const recvPc=isNum(h.lastAcceptT)?rkToPc(h.lastAcceptT):null;
    const accCtPc=rkToPc(accCt);
    const accHtPcBaseline=rkToPc(accHt);
    const accHtPcSample=th?pcSampleTimeForThrow(th,accHt):null;
    const throwNo=th?reportThrows.indexOf(th)+1:null;
    const truth=accHtPcSample!=null?pcTruthAt(accHtPcSample):null;
    const dx=truth&&isNum(h.wxw)&&isNum(h.wcarx)?(h.wxw-h.wcarx)-truth.x:null;
    const dz=truth?h.wz-truth.z:null;
    const has300=!!(th&&isNum(th.ref300Ht)&&isNum(th.ref300X)&&isNum(th.ref300Z));
    const dHt=has300?(accHt-th.ref300Ht)*1000:null;
    const dX=has300?h.wx-th.ref300X:null;
    const dZ=has300?h.wz-th.ref300Z:null;
    const result=accHtPcSample!=null?strikeAfter(accHtPcSample).verdict:'—';
    const info='S'+tableFmt(h.wstage,0)+
      (isNum(h.wnFit)?' n_fit='+h.wnFit:'')+
      ' / updates='+(h.n||1);
    return '<tr><td>'+tableFmt(throwNo,0)+'</td>'+
      '<td>'+tableFmt(recvPc,3)+'</td>'+
      '<td>'+tableFmt(accCtPc,3)+'</td>'+
      '<td>'+tableFmt(accHtPcBaseline,3)+'</td>'+
      '<td>'+((accHt-accCt)*1000).toFixed(1)+'</td>'+
      '<td>'+tableXzCm(h.wx,h.wz)+'</td>'+
      '<td>'+pcTruthCell(truth,false,accHtPcSample)+'</td>'+
      '<td>'+cmSigned(dx)+'/'+cmSigned(dz)+'</td>'+
      '<td>'+(has300?tableFmt(rkToPc(th.ref300Ht),3):'—')+'</td>'+
      '<td>'+(has300?tableXzCm(th.ref300X,th.ref300Z):'—')+'</td>'+
      '<td>'+tableSigned(dHt)+'</td>'+
      '<td>'+cmSigned(dX)+'/'+cmSigned(dZ)+'</td>'+
      '<td>'+info+'</td><td>'+result+'</td></tr>';
  });
  return '<div class="armTblWrap"><table class="armTbl"><thead><tr>'+
    '<th>RK抛#</th><th>最后accepted t<br>(s,global PC轴)</th><th>接收消息ct<br>(s,global PC轴)</th><th>接收消息HT<br>(s,global PC轴)</th>'+
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
// phase/target/remaining 属于发布事件轴 bot_state.t；x/y/yaw/v 属于物理轴 imu_t。
// 老 JSON 缺 imu_t 时物理量显示 —，不回退到发布时刻冒充。
buildPlots[1] = () => {
  if(!RK) return;
  const numv = v => (typeof v==='number' && Number.isFinite(v)) ? v : null;
  const bT=ts(RK.bot), bY=k=>ys(RK.bot,k), bPoseT=bY('imu_t')||[];
  const cols={x:bY('x'), y:bY('y'), yaw:bY('yaw'), vx:bY('vx'), vy:bY('vy'),
    phase:bY('phase'), steer:bY('steer_angle'), rem:bY('remaining'),
    tx:bY('target_x'), ty:bY('target_y')};
  const rows=[];
  for(let i=0;i<bT.length;i++){
    const t=numv(bT[i]);
    if(t===null) continue;
    const poseT=numv(bPoseT[i]);
    rows.push({t, poseT,
      x:poseT===null?null:numv(cols.x[i]), y:poseT===null?null:numv(cols.y[i]),
      yaw:poseT===null?null:numv(cols.yaw[i]),
      vx:poseT===null?null:numv(cols.vx[i]), vy:poseT===null?null:numv(cols.vy[i]),
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
    // 0817 用户定：视口固定 x∈[-3,3]、y∈[2,6]（等比、按容器长边扩展），段间比例尺一致、
    // 跨段可比；冲刺假目标/越界星标允许出视野——固定比例尺本来就是为了这个。
    const x0=-3, x1=3, y0=2, y1=6;
    const div=document.getElementById('c1');
    const W=Math.max(200,((div&&div.clientWidth)||1100)-MARGIN.l-MARGIN.r);
    const H=Math.max(200,((div&&div.clientHeight)||680)-MARGIN.t-MARGIN.b);
    const cx=(x0+x1)/2, cy=(y0+y1)/2;
    const mpp=Math.max((x1-x0)/W,(y1-y0)/H);   // meters per pixel，取大者兜住两轴
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
    V.tRk.textContent=f.poseT===null?`— (发布 ${f.t.toFixed(3)} s)`
      :`${f.poseT.toFixed(3)} s (发布 ${f.t.toFixed(3)} s)`;
    V.tPc.textContent=f.poseT===null?'—':`${rkToPc(f.poseT).toFixed(3)} s`;
    V.phase.textContent=f.phase!==null?f.phase:'—';
    V.pos.textContent=(f.x!==null&&f.y!==null)?`(${f.x.toFixed(3)}, ${f.y.toFixed(3)}) m`:'—';
    const spd=(f.vx!==null&&f.vy!==null)?Math.hypot(f.vx,f.vy):null;
    V.spd.textContent=spd===null?'—':`${spd.toFixed(3)} m/s`;
    V.vxy.textContent=f.vx===null?'—':`${f.vx.toFixed(3)} / ${f.vy.toFixed(3)} m/s`;
    V.yaw.textContent=deg(f.yaw);
    const w=f.poseT===null?null:lutAt(imuWLut,f.poseT,0.06);
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
    clock.textContent = rkAxisAligned
      ? `帧 ${cur+1}/${seg.frames.length} · PC(imu)=${f.poseT===null?'—':rkToPc(f.poseT).toFixed(2)+'s'}`
      : `帧 ${cur+1}/${seg.frames.length} · imu_t=${f.poseT===null?'—':f.poseT.toFixed(2)+'s'}（未对齐）`;
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
    // 0817 用户定：时间统一 PC 主显（与北极星表/视频轴同轴，免心算）；未对齐场退回 RK 记法。
    opt.textContent = rkAxisAligned
      ? `第 ${m.k+1} 次  PC ${rkToPc(m.runT0).toFixed(1)}→${rkToPc(m.runT1).toFixed(1)}s${tgt}${rem}`
      : `第 ${m.k+1} 次  RK ${m.runT0.toFixed(1)}→${m.runT1.toFixed(1)}s（未对齐）${tgt}${rem}`;
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
      ? `Axis: PC report time（PC t = RK t + ${rkBias.toFixed(4)}s，scale=1） &nbsp; `
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
      + `PC t = RK t + ${rkBias.toFixed(4)}s; scale fixed 1; ${srcText}auto z-shape ${errText}`;
  };
  const tr = (series,key,name,axis,color,mode='markers',extra={}) => g2({
    x:shifted(ts(series)), y:ys(series,key), name, mode,
    marker:{color,size:3}, line:{color,width:1.4},
    yaxis:axis, xaxis:'x',
    hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}<extra>${name}</extra>`,
    ...extra,
  });
  const botPoseTr = (key,name,axis,color,extra={}) => g2({
    x:shifted(ys(RK.bot,'imu_t')||[]), y:ys(RK.bot,key), name, mode:'markers',
    marker:{color,size:3}, line:{color,width:1.4}, yaxis:axis, xaxis:'x',
    hovertemplate:`imu_t=%{x:.3f}s<br>${name}=%{y:.4f}<extra>${name}</extra>`,
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
    // 机械臂 FK TCP：每个 state 用同刻车 yaw 旋到世界轴，原点为机械臂中心且 z=0。
    ...(armAligned ? (()=>{
      const tArm=armTcpRows.map(s=>rkToPc(s.t));
      const tcpWorldRows=armTcpRows.map(s=>armPointWorld(s.tcp,botYawDegAt(s.t),armConstCal.zOff));
      const val=k=>tcpWorldRows.map(p=>p?p[k]:null);
      const mk=(k,name,color,extra={})=>g2({x:tArm, y:val(k), name, mode:'markers',
        marker:{color,size:2.5}, yaxis:'y', xaxis:'x',
        hovertemplate:`t=%{x:.3f}s<br>${name}=%{y:.4f}m<extra>${name}</extra>`, ...extra});
      return [mk(0,'Arm TCP world X@arm origin','#22d3ee',{visible:'legendonly'}),
              mk(1,'Arm TCP world Y@arm origin','#67e8f9',{visible:'legendonly'}),
              mk(2,'Arm TCP Z@arm origin','#06b6d4')];
    })() : []),
    // 视觉拍心保留车心原点，与 Arm TCP 的机械臂中心原点不同，不直接比较。
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
    botPoseTr('x','Bot X','y','#67e8c3'),
    botPoseTr('y','Bot Y','y','#9fffce'),
    g2({x:shifted(ys(RK.bot,'imu_t')||[]), y:ys(RK.bot,'yaw').map(v=>isNum(v)?v*10:null), name:'Bot Yaw x10', mode:'markers',
      customdata:ys(RK.bot,'yaw'),
      marker:{color:'#5eead4',size:2.5,symbol:'diamond'}, yaxis:'y3', xaxis:'x',
      hovertemplate:'imu_t=%{x:.3f}s<br>Bot Yaw=%{customdata:.4f}rad<br>display=%{y:.3f}<extra>Bot Yaw x10</extra>',
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
    rebuildThrowPhases();
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
    rkAxisAligned=isNum(v);
    throwBaselineTrusted=isNum(v);
    redraw();
  });
  const autoBtn = document.getElementById('rkAuto');
  if(autoBtn) autoBtn.addEventListener('click',()=>{
    rkBias=Math.round((isNum(auto.bias)?auto.bias:0)*10000)/10000;
    rkAxisAligned=isNum(auto.bias);
    throwBaselineTrusted=rkAxisAligned&&(presetBias!=null||clockBridge.bias!=null||
      poseLock.usable||clockAnchor.bias!=null||!alignBad);
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
    rkAxisAligned=isNum(v);
    throwBaselineTrusted=isNum(v);
    if(input) input.value=rkBias.toFixed(4);
    redraw().then(syncSignalControls);
  });
  const sigAuto = document.getElementById('rkSigAuto');
  if(sigAuto) sigAuto.addEventListener('click',()=>{
    rkBias=Math.round((isNum(auto.bias)?auto.bias:0)*10000)/10000;
    rkAxisAligned=isNum(auto.bias);
    throwBaselineTrusted=rkAxisAligned&&(presetBias!=null||clockBridge.bias!=null||
      poseLock.usable||clockAnchor.bias!=null||!alignBad);
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
        "--racket-impact-json", default=None,
        help="球拍头bbox中心bundle vz侧车；仅在显式提供时加载",
    )
    parser.add_argument(
        "--arm-json", default=None,
        help="extract_arm_bag.py 输出的机械臂 JSON；缺省时自动探测 <input>_arm.json",
    )
    parser.add_argument("--rk-tracking-json", default=None)
    parser.add_argument(
        "--rk-time-bias", type=float, default=None,
        help="预置固定 scale=1 时间映射的 bias（秒）；页面 Auto align 按钮可恢复自动 bias。",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--no-tables", action="store_true",
        help="不导出 <run>_tables.md/.json（默认导出，供人/AI 免浏览器直接读数）",
    )
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
        args.racket_impact_json,
        arm_json,
        rk_tracking_json,
        args.rk_time_bias,
        export_tables=not args.no_tables,
    )


if __name__ == "__main__":
    main()
