# -*- coding: utf-8 -*-
"""
离线标注脚本：读取拼接视频 + JSON 数据，生成带标注的视频。

画面标注刻意保持极简（用户反馈：框/数字/标记太多反而看不清）：panel 内只画两种框，
不带任何文字——网球检测框、车载 AprilTag 识别框（真实检测角点四边形）；回球期间另有
青色速度矢量箭头（无旁标文字）。球 3D/回球 yaw·pitch·速度仅保留在底部文字行，不写
车位姿、不画状态角标、不画 RK 目标点标记。

在原有球框/球 3D/curve3 标注之外，还会在离线阶段调用 ArmCalibration 同款的
`yolo_model/racket.onnx + yolo_model/racket_pose.onnx`，只使用关键点 0-3 的几何中心
做球拍 2D/3D 定位，并将结果补充回 JSON；视频画面只绘制球拍关键点与几何中心。

另外从 PC 球观测离线检测"击球回球"事件（与报告 All-in-One 表 PC回球列同口径，
含贴地球/断档/跳变/地面反弹脏数据过滤），在触球后 400ms 帧区间叠加回球速度矢量
（RETURN 文字行 + 各相机按标定投影的青色箭头），事件同时写回 JSON pc_return_events。

若同目录存在 <stem>_rk_tracking.json，会用共享小车位姿锚对齐 RK 时轴（同报告
clockAnchor 思路，锚不足时自动跳过）并把每次抛球的 target_active 时间段打印到控制台，
供核对 RK/PC 时轴对齐质量；不在视频画面上叠加目标点标记。

用法：
  python test_src/annotate_video.py --input tracker_output/tracker_20260311_193455/tracker_20260311_193455.json
  python test_src/annotate_video.py --input tracker_output/tracker_20260311_193455/tracker_20260311_193455.json ^
      --output tracker_output/tracker_20260311_193455/tracker_20260311_193455_annotated.avi
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ball_detector import BallDetection
from src.ball_localizer import Ball3D
from src.car_localizer import CarLoc, CarLocalizer
from src.racket_localizer import RacketDetection, RacketLoc, RacketLocalizer

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.1
FONT_THICKNESS = 2
BOX_COLORS = [
    (0, 255, 0),       # 绿色 - 1号相机
    (0, 165, 255),     # 橙色 - 2号相机
    (255, 100, 100),   # 蓝色 - 3号相机
    (255, 0, 255),     # 紫色 - 4号相机
]
RACKET_BOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_3D_COLOR = (0, 255, 255)
TEXT_RACKET_3D_COLOR = (255, 0, 255)
RETURN_COLOR = (255, 255, 0)  # 青色 - 回球速度矢量
CAR_TAG_BOX_COLOR = (0, 200, 255)  # 橙黄 - 车载 AprilTag 识别框
STATE_COLORS = {
    "idle":         (128, 128, 128),
    "tracking_s0":  (255, 200, 0),
    "in_landing":   (0, 165, 255),
    "tracking_s1":  (0, 255, 0),
    "done":         (0, 0, 255),
}

_DEFAULT_RACKET_MODEL = _PROJECT_ROOT / "yolo_model" / "racket.onnx"
_DEFAULT_RACKET_POSE_MODEL = _PROJECT_ROOT / "yolo_model" / "racket_pose.onnx"
_DEFAULT_TRACKER_CONFIG = _PROJECT_ROOT / "src" / "config" / "tracker.json"
_TRACKER_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def guess_tracker_video_path(json_path: Path, data: dict) -> Path:
    config = data.get("config", {}) if isinstance(data, dict) else {}
    video_output = config.get("video_output", {}) if isinstance(config, dict) else {}

    for key in ("artifact_path", "path"):
        candidate = video_output.get(key) if isinstance(video_output, dict) else None
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (json_path.parent / candidate_path).resolve()
        if candidate_path.exists():
            return candidate_path

    for suffix in _TRACKER_VIDEO_SUFFIXES:
        candidate_path = json_path.with_suffix(suffix)
        if candidate_path.exists():
            return candidate_path

    return json_path.with_suffix(".avi")


def _format_xyz_m(x: float, y: float, z: float) -> str:
    return f"({x:.3f}, {y:.3f}, {z:.3f}) m"


def _scale_xyz_entry(entry: dict, scale: float) -> None:
    for key in ("x", "y", "z"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            entry[key] = float(value) * scale


def normalize_tracker_json_to_m(data: dict) -> None:
    cfg = data.setdefault("config", {})
    if cfg.get("distance_unit") == "m":
        return

    scale = 1.0 / 1000.0
    if isinstance(cfg.get("ideal_hit_z"), (int, float)):
        cfg["ideal_hit_z"] = float(cfg["ideal_hit_z"]) * scale

    for seq_name in ("observations", "predictions", "car_locs", "racket_observations"):
        for entry in data.get(seq_name, []):
            if isinstance(entry, dict):
                _scale_xyz_entry(entry, scale)

    for frame_data in data.get("frames", []):
        if not isinstance(frame_data, dict):
            continue
        for key in ("ball3d", "prediction", "car_loc", "racket3d"):
            entry = frame_data.get(key)
            if isinstance(entry, dict):
                _scale_xyz_entry(entry, scale)

    cfg["distance_unit"] = "m"


@dataclass
class RacketPipeline:
    localizer: RacketLocalizer
    pose_model_path: str
    keypoint_score_threshold: float
    min_face_valid_keypoints: int


def sync_video_frame_metadata(
    data: dict,
    frame_mapping: list[int],
    has_exact_mapping: bool,
) -> None:
    """Attach video-frame indices onto JSON frame entries for exact frame linkage."""
    frames_data = data.get("frames", [])
    for frame_data in frames_data:
        if isinstance(frame_data, dict):
            frame_data.pop("video_frame_idx", None)
            frame_data.pop("video_mapping_exact", None)

    for video_frame_idx, json_frame_idx in enumerate(frame_mapping):
        if 0 <= json_frame_idx < len(frames_data):
            frame_data = frames_data[json_frame_idx]
            if isinstance(frame_data, dict):
                frame_data["video_frame_idx"] = video_frame_idx
                frame_data["video_mapping_exact"] = bool(has_exact_mapping)

    cfg = data.setdefault("config", {})
    summary = data.setdefault("summary", {})
    cfg["video_frame_mapping_exact"] = bool(has_exact_mapping)
    summary["video_frame_mapping_exact"] = bool(has_exact_mapping)
    summary["video_frames_mapped"] = len(frame_mapping)


def build_video_frame_mapping(data: dict, total_video_frames: int) -> tuple[list[int], bool]:
    """
    返回“视频第 i 帧 -> JSON frames[j]”的映射。
    新版 run_tracker 会在 JSON 中写入 video_frame_indices；
    旧版 JSON 没有这个字段时，只能退化为 1:1 顺序映射。
    """
    frames_data = data["frames"]
    mapping = data.get("video_frame_indices")
    if mapping is None:
        fallback = list(range(min(total_video_frames, len(frames_data))))
        return fallback, False

    valid_mapping = [
        int(idx) for idx in mapping
        if isinstance(idx, int) and 0 <= idx < len(frames_data)
    ]
    return valid_mapping, True


def grid_dimensions(n_panels: int, cols: int = 2) -> tuple[int, int]:
    cols = max(1, min(cols, n_panels))
    rows = max(1, math.ceil(n_panels / cols))
    return cols, rows


def infer_stitched_grid(n_panels: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    if n_panels <= 2:
        return grid_dimensions(n_panels, cols=n_panels)
    if frame_h > 0 and frame_w / frame_h >= 2.5:
        return grid_dimensions(n_panels, cols=n_panels)
    return grid_dimensions(n_panels, cols=2)


def grid_slot(
    index: int,
    panel_w: int,
    panel_h: int,
    *,
    cols: int = 2,
) -> tuple[int, int]:
    col = index % cols
    row = index // cols
    return col * panel_w, row * panel_h


def split_stitched_panels(
    img: np.ndarray,
    serials: list[str],
) -> tuple[dict[str, np.ndarray], int, int]:
    """将拼接视频帧按 2x2 row-major 相机顺序拆成 panel。"""
    h, w = img.shape[:2]
    n_cams = len(serials)
    cols, rows = infer_stitched_grid(n_cams, w, h)
    panel_w = w // cols
    panel_h = h // rows
    panels: dict[str, np.ndarray] = {}
    for i, sn in enumerate(serials):
        x1, y1 = grid_slot(i, panel_w, panel_h, cols=cols)
        x2 = w if (i % cols) == cols - 1 else x1 + panel_w
        y2 = h if (i // cols) == rows - 1 else y1 + panel_h
        panels[sn] = img[y1:y2, x1:x2]
    return panels, panel_w, panel_h


def load_tracker_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_panel_timestamp(frame_data: dict, frame_idx: int, fps: float) -> float:
    """优先使用 JSON 中的 exposure_pc，缺失时再退化到按帧时间。"""
    exposure_pc = frame_data.get("exposure_pc")
    if isinstance(exposure_pc, (int, float)):
        return float(exposure_pc)
    if fps <= 0:
        return float(frame_idx)
    return frame_idx / fps


def extract_time_reference(data: dict) -> float | None:
    """Return the tracker time used as t=0 for HTML/video overlays."""
    cfg = data.get("config", {})
    first_frame_exposure_pc = cfg.get("first_frame_exposure_pc")
    if isinstance(first_frame_exposure_pc, (int, float)):
        return float(first_frame_exposure_pc)

    frames = data.get("frames", [])
    if frames:
        exposure_pc = frames[0].get("exposure_pc")
        if isinstance(exposure_pc, (int, float)):
            return float(exposure_pc)

    candidates: list[float] = []
    for items, key in (
        (data.get("observations", []), "t"),
        (data.get("car_locs", []), "t"),
        (data.get("predictions", []), "ct"),
    ):
        for item in items:
            value = item.get(key)
            if isinstance(value, (int, float)):
                candidates.append(float(value))

    return min(candidates) if candidates else None


def build_relative_frame_time_s(
    frame_data: dict,
    frame_idx: int,
    fps: float,
    time_reference: float | None,
) -> float:
    """Frame time shown in annotated video; aligns with HTML t-axis."""
    timestamp = build_panel_timestamp(frame_data, frame_idx, fps)
    if time_reference is None:
        return float(timestamp)
    return max(0.0, float(timestamp) - float(time_reference))


# ---------------- PC 回球检测与速度矢量叠加 ----------------
# 与 generate_curve3_html 的 [[pc-return-core]]（All-in-One "PC回球" 列）同口径：
# 触球时刻 = 入弧/出弧 y(t) 二次拟合交点（来回交点法），回球速度 = 触球后 [+20,+400]ms
# 出弧连续段三轴拟合在触球时刻的导数（z 先扣 ½g·u²）。脏数据过滤：z<0.12 贴地/静止球
# 剔除；观测按断档>150ms 或帧间位移速>20m/s（检测器跳到别的球；门槛在最快回球 ~15m/s
# 之上，073646 实测 12.2~12.4m/s 回球会被 12 门误切）切成连续段，按点数优先选首个
# "真出向"段——静止球段 vy≈0 被拒，被拍/臂遮挡断档后仍能用遮挡后的真弧；
# vz 由降转升突增判地面反弹截断；出弧 vy≤0.5、水平速<1m/s 或点数不足不认定回球。
# 出弧另有三道防污染门（0813_083521 抛28 定案，入弧不加——入弧窗内可含真地面反弹）：
# 1) 轨迹一致性切段（>20m/s 跳变门在长断档下放行的绝对位移随 dt 变大，抛28：138ms 断档
#    +1.99m 跳到另一颗静止球=14.4m/s 贴线通过）；2) 段首前拟合 z 触地拒绝（整段在反弹之后，
#    倒推跨反弹必错）；3) 三轴拟合 max|残差|>0.12m 拒绝出数（混轨垃圾；不剔点重拟合——
#    高杠杆野点会把二次拟合弯过去反把真点顶成最大残差）。
RETURN_MIN_Z = 0.12
RETURN_MAX_STEP_MPS = 20.0
RETURN_TRACK_DEV_M = 0.5     # 防污染门 1：末 3 点线性外推偏差上限
RETURN_FIT_RESID_M = 0.12    # 防污染门 3：三轴拟合 max|残差| 上限
RETURN_MIN_FIT_Z = 0.05      # 防污染门 2：[0,段首] 拟合 z（含重力）下限
RETURN_OUT_WINDOW = (0.02, 0.40)
RETURN_MAX_SEG_START = 0.30  # 出弧段首距触球上限（限制回推外推量）
RETURN_HALF_G = 4.905
RETURN_ARROW_SECONDS = 0.15  # 箭头长度 = 速度矢量 × 该时长的位移


def _quad_fit_u(pts: list[tuple[float, float]], tc: float) -> dict | None:
    """v ≈ a + b·u + c·u²，u = t − tc；<6 点退线性。返回 {a,b,c,n}。"""
    n = len(pts)
    if n < 3:
        return None
    ts = np.array([p[0] - tc for p in pts], dtype=np.float64)
    vs = np.array([p[1] for p in pts], dtype=np.float64)
    deg = 2 if n >= 6 else 1
    try:
        coef = np.polyfit(ts, vs, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(coef)):
        return None
    if deg == 1:
        return {"a": float(coef[1]), "b": float(coef[0]), "c": 0.0, "n": n}
    return {"a": float(coef[2]), "b": float(coef[1]), "c": float(coef[0]), "n": n}


def _runs_in_window(
    rows: list[tuple[float, float, float, float]], lo: float, hi: float
) -> list[list[tuple[float, float, float, float]]]:
    """[lo,hi] 内 z≥0.12 观测按 断档>150ms 或帧间>20m/s 跳变 切成连续段。"""
    runs: list[list[tuple[float, float, float, float]]] = []
    run: list[tuple[float, float, float, float]] = []
    for row in rows:
        t, x, y, z = row
        if t < lo:
            continue
        if t > hi:
            break
        if z < RETURN_MIN_Z:
            continue
        if run:
            lt, lx, ly, lz = run[-1]
            dt = t - lt
            jump = math.hypot(x - lx, y - ly, z - lz) / max(1e-9, dt)
            if dt > 0.15 or jump > RETURN_MAX_STEP_MPS:
                runs.append(run)
                run = []
        run.append(row)
    if run:
        runs.append(run)
    return runs


def _split_by_track_dev(
    runs: list[list[tuple[float, float, float, float]]]
) -> list[list[tuple[float, float, float, float]]]:
    """防污染门 1：run 内已有 ≥3 点时由末 3 点线性外推到新点时刻，偏差 >0.5m 切段。
    只用于出弧（入弧窗内可含真地面反弹）；切出的野点子段随后被点数/vy 门拒绝。"""
    out: list[list[tuple[float, float, float, float]]] = []
    for run in runs:
        cur: list[tuple[float, float, float, float]] = []
        for p in run:
            if len(cur) >= 3:
                a, b = cur[-3], cur[-1]
                dv = max(1e-9, b[0] - a[0])
                dt = p[0] - b[0]
                pred = tuple(b[i] + (b[i] - a[i]) / dv * dt for i in (1, 2, 3))
                if math.hypot(p[1] - pred[0], p[2] - pred[1], p[3] - pred[2]) > RETURN_TRACK_DEV_M:
                    out.append(cur)
                    cur = []
            cur.append(p)
        if cur:
            out.append(cur)
    return out


def _pc_hit_time_at(rows: list[tuple[float, float, float, float]], t_approx: float) -> float | None:
    """入弧 [−380,−25]ms / 出弧 [+30,+330]ms y(t) 拟合交点；无有效交点返回 None。"""
    yin_run = next(
        (r for r in reversed(_runs_in_window(rows, t_approx - 0.38, t_approx - 0.025))
         if len(r) >= 5),
        None,
    )
    if yin_run is None:
        return None
    fin = _quad_fit_u([(t, y) for t, x, y, z in yin_run], t_approx)
    if fin is None or fin["b"] > -1.0:
        return None
    out_runs = [
        r for r in _split_by_track_dev(_runs_in_window(rows, t_approx + 0.03, t_approx + 0.33))
        if len(r) >= 4
    ]
    out_runs.sort(key=len, reverse=True)
    for run in out_runs:
        fout = _quad_fit_u([(t, y) for t, x, y, z in run], t_approx)
        if fout is None or fout["b"] < 0.15:
            continue
        a = fin["c"] - fout["c"]
        b = fin["b"] - fout["b"]
        c = fin["a"] - fout["a"]
        roots: list[float] = []
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                roots = [-c / b]
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                r = math.sqrt(disc)
                roots = [(-b + r) / (2 * a), (-b - r) / (2 * a)]
        roots = [u for u in roots if -0.10 <= u <= 0.14]
        if roots:
            return t_approx + min(roots, key=abs)
    return None


def _bounce_cut_run(
    run: list[tuple[float, float, float, float]]
) -> tuple[list[tuple[float, float, float, float]], bool]:
    """段内地面反弹截断：vz 由降(<−0.5)转升突增>3m/s 处截断。"""
    for i in range(2, len(run)):
        vz_a = (run[i - 1][3] - run[i - 2][3]) / max(1e-9, run[i - 1][0] - run[i - 2][0])
        vz_b = (run[i][3] - run[i - 1][3]) / max(1e-9, run[i][0] - run[i - 1][0])
        if vz_a < -0.5 and vz_b - vz_a > 3.0:
            return run[:i], True
    return run, False


def detect_return_events(observations: list[dict]) -> list[dict]:
    """从 PC 球观测检测击球回球事件（y 向由进转出），返回按时间排序的事件列表。"""
    rows: list[tuple[float, float, float, float]] = []
    for o in observations:
        if not isinstance(o, dict):
            continue
        vals = [o.get(k) for k in ("t", "x", "y", "z")]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            rows.append((float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])))
    rows.sort(key=lambda r: r[0])
    frows = [r for r in rows if r[3] >= RETURN_MIN_Z]

    # 候选锚点：帧间 vy 由强负（真来球）转正且连续两帧为正（防单帧噪声）。
    # 配对窗放宽到 0.65s：触球前后都可能被拍/臂遮挡，入弧尾与出弧头相距可达数百 ms，
    # 候选只管"疑似"，真伪由来回交点拟合与出弧门槛裁决。
    cands: list[float] = []
    last_in_t: float | None = None
    posrun = 0
    first_pos_tm: float | None = None
    for i in range(1, len(frows)):
        dt = frows[i][0] - frows[i - 1][0]
        if dt <= 0 or dt > 0.15:
            posrun = 0
            first_pos_tm = None
            if dt > 0.7:
                last_in_t = None
            continue
        jump = math.hypot(
            frows[i][1] - frows[i - 1][1],
            frows[i][2] - frows[i - 1][2],
            frows[i][3] - frows[i - 1][3],
        ) / dt
        if jump > RETURN_MAX_STEP_MPS:
            last_in_t = None
            posrun = 0
            first_pos_tm = None
            continue
        vy = (frows[i][2] - frows[i - 1][2]) / dt
        tm = 0.5 * (frows[i][0] + frows[i - 1][0])
        if vy < -1.0:
            last_in_t = tm
            posrun = 0
            first_pos_tm = None
        elif vy > 0.3:
            posrun += 1
            if first_pos_tm is None:
                first_pos_tm = tm
            if posrun >= 2 and last_in_t is not None and first_pos_tm - last_in_t <= 0.65:
                cands.append(0.5 * (last_in_t + first_pos_tm))
                last_in_t = None
                posrun = 0
                first_pos_tm = None

    events: list[dict] = []
    for tc in cands:
        if events and tc < events[-1]["t_hit"] + 0.5:
            continue
        t_hit = None
        # 候选锚点扫描：无 RK 锚，锚点=遮挡中点，可偏真触球 ±0.2s，扫描范围更宽
        for d in (0.0, -0.06, 0.06, -0.12, 0.12, -0.18, 0.18):
            t_hit = _pc_hit_time_at(rows, tc + d)
            if t_hit is not None:
                break
        if t_hit is None:
            continue
        refined = _pc_hit_time_at(rows, t_hit)  # 锚点偏差大时窗口混入对侧弧，再定一次
        if refined is not None:
            t_hit = refined
        if events and t_hit < events[-1]["t_hit"] + 0.5:
            continue
        cand_runs = [
            r for r in _split_by_track_dev(_runs_in_window(
                rows, t_hit + RETURN_OUT_WINDOW[0], t_hit + RETURN_OUT_WINDOW[1]
            ))
            if len(r) >= 5 and r[0][0] - t_hit <= RETURN_MAX_SEG_START
        ]
        cand_runs.sort(key=len, reverse=True)
        for run in cand_runs:
            seg, bounce_cut = _bounce_cut_run(run)
            if len(seg) < 5 or seg[-1][0] - seg[0][0] < 0.06:
                continue
            fx = _quad_fit_u([(p[0], p[1]) for p in seg], t_hit)
            fy = _quad_fit_u([(p[0], p[2]) for p in seg], t_hit)
            # z 先扣重力 ½g·u² 再拟合：fit_z 为无重力部分，还原位置时要减回去
            fz = _quad_fit_u(
                [(p[0], p[3] + RETURN_HALF_G * (p[0] - t_hit) ** 2) for p in seg], t_hit
            )
            if fx is None or fy is None or fz is None:
                continue
            vx, vy, vz = fx["b"], fy["b"], fz["b"]
            if vy <= 0.5:
                continue
            vh = math.hypot(vx, vy)
            if vh < 1.0:
                continue
            fit_x = (fx["a"], fx["b"], fx["c"])
            fit_y = (fy["a"], fy["b"], fy["c"])
            fit_z = (fz["a"], fz["b"], fz["c"])
            start = seg[0][0] - t_hit
            # 防污染门 2：段首前拟合 z 触地即整段在反弹之后，倒推跨反弹必错
            min_z = min(
                _eval_fit(fit_z, start * i / 16.0) - RETURN_HALF_G * (start * i / 16.0) ** 2
                for i in range(17)
            )
            if min_z < RETURN_MIN_FIT_Z:
                continue
            # 防污染门 3：混轨/乱拟合拒绝出数（不剔点重拟合）
            max_res = max(
                max(
                    abs(p[1] - _eval_fit(fit_x, p[0] - t_hit)),
                    abs(p[2] - _eval_fit(fit_y, p[0] - t_hit)),
                    abs(p[3] + RETURN_HALF_G * (p[0] - t_hit) ** 2 - _eval_fit(fit_z, p[0] - t_hit)),
                )
                for p in seg
            )
            if max_res > RETURN_FIT_RESID_M:
                continue
            events.append({
                "t_hit": t_hit,
                "t_end": t_hit + RETURN_OUT_WINDOW[1],
                "seg_t_end": seg[-1][0],
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "speed": math.hypot(vx, vy, vz),
                "yaw_deg": math.degrees(math.atan2(vx, vy)),
                "pitch_deg": math.degrees(math.atan2(vz, vh)),
                "n_points": len(seg),
                "span_s": seg[-1][0] - seg[0][0],
                "seg_start_s": seg[0][0] - t_hit,
                "max_res_m": max_res,
                "bounce_cut": bounce_cut,
                "fit_x": fit_x,
                "fit_y": fit_y,
                "fit_z": fit_z,
            })
            break
    return events


def serialize_return_events(events: list[dict], time_reference: float | None) -> list[dict]:
    """写回 JSON 的回球事件（去掉拟合系数，补相对时间）。"""
    out = []
    for ev in events:
        item = {k: v for k, v in ev.items() if not k.startswith("fit_")}
        item["elapsed_s"] = (
            round(ev["t_hit"] - time_reference, 3) if time_reference is not None else None
        )
        out.append(item)
    return out


def load_camera_projections(calib_config_path: str | Path) -> dict[str, dict]:
    """加载各相机 K/D/R_world/t_world（标定世界系为 mm），用于 3D→像素投影。"""
    with open(calib_config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cams: dict[str, dict] = {}
    for sn, cd in cfg.get("cameras", {}).items():
        cams[sn] = {
            "K": np.array(cd["K"], dtype=np.float64).reshape(3, 3),
            "D": np.array(cd["D"], dtype=np.float64).ravel(),
            "R": np.array(cd["R_world"], dtype=np.float64).reshape(3, 3),
            "t": np.array(cd["t_world"], dtype=np.float64).reshape(3),
        }
    return cams


def project_world_point_m(cam: dict, pt_m) -> tuple[float, float] | None:
    """世界系 3D 点（米）投影到该相机全分辨率像素；位于相机后方或数值异常返回 None。"""
    pt_mm = np.asarray(pt_m, dtype=np.float64) * 1000.0
    depth = cam["R"] @ pt_mm + cam["t"]
    if depth[2] <= 1e-6:
        return None
    rvec, _ = cv2.Rodrigues(cam["R"])
    px, _ = cv2.projectPoints(
        pt_mm.reshape(1, 1, 3), rvec, cam["t"].reshape(3, 1), cam["K"], cam["D"]
    )
    u, v = float(px[0, 0, 0]), float(px[0, 0, 1])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return u, v


def _eval_fit(fit: tuple[float, float, float], u: float) -> float:
    a, b, c = fit
    return a + b * u + c * u * u


def draw_return_vector(
    out: np.ndarray,
    event: dict,
    frame_time: float,
    ball3d: dict | None,
    serials: list[str],
    panel_w: int,
    panel_h: int,
    cols: int,
    cam_projections: dict[str, dict] | None,
) -> None:
    """在各相机 panel 上绘制回球速度矢量箭头（锚点=当前帧球 3D，缺则用出弧拟合位置）。"""
    if cam_projections is None:
        return
    u = frame_time - event["t_hit"]
    if ball3d is not None:
        pos = np.array([ball3d["x"], ball3d["y"], ball3d["z"]], dtype=np.float64)
    elif 0.0 <= u <= event["seg_t_end"] - event["t_hit"] + 0.05:
        pos = np.array([
            _eval_fit(event["fit_x"], u),
            _eval_fit(event["fit_y"], u),
            # fit_z 是扣过重力的序列，还原位置要减回 ½g·u²
            _eval_fit(event["fit_z"], u) - RETURN_HALF_G * u * u,
        ])
    else:
        return
    vel = np.array([event["vx"], event["vy"], event["vz"]], dtype=np.float64)
    tip = pos + vel * RETURN_ARROW_SECONDS
    scale = 0.5  # 拼接视频 panel 为半分辨率
    for cam_idx, sn in enumerate(serials):
        cam = cam_projections.get(sn)
        if cam is None:
            continue
        p0 = project_world_point_m(cam, pos)
        p1 = project_world_point_m(cam, tip)
        if p0 is None or p1 is None:
            continue
        x_offset, y_offset = grid_slot(cam_idx, panel_w, panel_h, cols=cols)
        x0 = p0[0] * scale
        y0 = p0[1] * scale
        if not (-40 <= x0 <= panel_w + 40 and -40 <= y0 <= panel_h + 40):
            continue
        dx = (p1[0] - p0[0]) * scale
        dy = (p1[1] - p0[1]) * scale
        length = math.hypot(dx, dy)
        max_len = 0.45 * panel_w
        if length > max_len > 0:
            dx *= max_len / length
            dy *= max_len / length
        pt0 = (int(round(x0 + x_offset)), int(round(y0 + y_offset)))
        pt1 = (int(round(x0 + dx + x_offset)), int(round(y0 + dy + y_offset)))
        ok, cpt0, cpt1 = cv2.clipLine((x_offset, y_offset, panel_w, panel_h), pt0, pt1)
        if not ok:
            continue
        cv2.arrowedLine(out, cpt0, cpt1, RETURN_COLOR, 3, cv2.LINE_AA, tipLength=0.22)


# ---------------- RK bot 目标点叠加（每次抛球：小车当前位置 → 要去的目标点） ----------------
# 目标点来自 <stem>_rk_tracking.json 的 bot_state（RK 相对时轴、与 PC 同名世界系）。
# 时间对齐用共享小车位姿锚（RK world 的 bot_x/bot_y/bot_yaw 是 PC 发布位姿的回显，
# 与报告的常数偏移合同一致）：bias = median(pc_elapsed − rk_t)，scale 固定为 1。
# 场内真实频差只积累到几毫秒量级，对视频叠加忽略。锚不足或残差过大则跳过叠加。
# 注意：锚匹配必须用 tracker 原始发布的 car_locs（RK 回显的就是它），
# 要在 clear_car_results() 全帧重标之前完成。

_FINITE = lambda v: isinstance(v, (int, float)) and math.isfinite(v)  # noqa: E731


def guess_rk_tracking_path(json_path: Path) -> Path | None:
    candidate = json_path.with_name(json_path.stem + "_rk_tracking.json")
    return candidate if candidate.exists() else None


def load_rk_bot_data(rk_path: Path) -> dict | None:
    """读取 bot_state 样本与 world 位姿回显行；结构异常返回 None。"""
    try:
        with open(rk_path, encoding="utf-8") as f:
            rk = json.load(f)
    except (OSError, ValueError):
        return None

    def cols(series: dict, keys: list[str]) -> list[list]:
        t = series.get("t") or []
        out = [t]
        for key in keys:
            v = (series.get("y") or {}).get(key) or []
            out.append(v + [None] * (len(t) - len(v)))
        return out

    bot = rk.get("bot") or {}
    t_arr, x_arr, y_arr, tx_arr, ty_arr, act_arr, phase_arr = cols(
        bot, ["x", "y", "target_x", "target_y", "target_active", "phase"]
    )
    samples = []
    for i, t in enumerate(t_arr):
        if not _FINITE(t):
            continue
        samples.append({
            "t": float(t),
            "x": x_arr[i] if _FINITE(x_arr[i]) else None,
            "y": y_arr[i] if _FINITE(y_arr[i]) else None,
            "tx": tx_arr[i] if _FINITE(tx_arr[i]) else None,
            "ty": ty_arr[i] if _FINITE(ty_arr[i]) else None,
            "active": bool(act_arr[i]),
            "phase": phase_arr[i] if isinstance(phase_arr[i], str) else "",
        })
    samples.sort(key=lambda s: s["t"])

    world = rk.get("world") or {}
    wt, wx, wy, wyaw = cols(world, ["bot_x", "bot_y", "bot_yaw"])
    pose_rows = [
        (float(t), float(x), float(y), float(yaw))
        for t, x, y, yaw in zip(wt, wx, wy, wyaw)
        if _FINITE(t) and _FINITE(x) and _FINITE(y) and _FINITE(yaw)
    ]

    # 各抛最终 ht（RK 相对轴）：与报告 rkThrows 同款聚类，供触球事件对齐兜底
    pred = rk.get("pred") or {}
    pt_arr, pht_arr = cols(pred, ["ht_rel"])
    throws: list[list[float]] = []
    for t, ht in zip(pt_arr, pht_arr):
        if not (_FINITE(t) and _FINITE(ht)):
            continue
        if throws and abs(ht - throws[-1][1]) < 0.8 and t - throws[-1][0] < 2.0:
            throws[-1] = [float(t), float(ht), throws[-1][2] + 1]
        else:
            throws.append([float(t), float(ht), 1])
    throw_hts = [ht for _t, ht, n in throws if n >= 3]

    if not samples:
        return None
    return {
        "samples": samples,
        "sample_ts": [s["t"] for s in samples],
        "pose_rows": pose_rows,
        "throw_hts": throw_hts,
    }


def estimate_rk_time_bias(
    rk_pose_rows: list[tuple[float, float, float, float]],
    pc_car_rows: list[tuple[float, float, float, float]],
) -> tuple[float, int, float] | None:
    """共享位姿锚估计 bias（pc = rk + bias）；返回 (bias, 锚数, MAD)，不可靠返回 None。"""
    key = lambda x, y, yaw: f"{x:.4f},{y:.4f},{yaw:.4f}"  # noqa: E731
    pc_unique: dict[str, float | None] = {}
    for t, x, y, yaw in pc_car_rows:
        k = key(x, y, yaw)
        pc_unique[k] = None if k in pc_unique else t
    rk_first: dict[str, float] = {}
    for t, x, y, yaw in rk_pose_rows:
        rk_first.setdefault(key(x, y, yaw), t)
    diffs = sorted(
        pc_unique[k] - rt
        for k, rt in rk_first.items()
        if pc_unique.get(k) is not None
    )
    if len(diffs) < 10:
        return None
    bias = diffs[len(diffs) // 2]
    mad = sorted(abs(d - bias) for d in diffs)[len(diffs) // 2]
    if mad > 0.08:
        return None
    return bias, len(diffs), mad


def estimate_rk_bias_from_hits(
    throw_hts: list[float], hit_elapsed: list[float]
) -> tuple[float, int, float] | None:
    """触球事件对齐兜底：PC 检出的回球触球时刻 ↔ RK 各抛最终 ht 的差值模式聚类。

    base json 的 car_locs 被 annotate 离线重标覆盖后不再是 RK 回显的原始位姿，
    位姿锚会失效；触球是两侧共同对准的物理事件（RK ht 预测误差 ±几十 ms），
    取所有 (事件−ht) 差值里 ±120ms 内最大簇的中位数为 bias。"""
    diffs = sorted(e - h for e in hit_elapsed for h in throw_hts)
    if len(diffs) < 3:
        return None
    best_i, best_j = 0, -1
    for i in range(len(diffs)):
        j = i
        while j + 1 < len(diffs) and diffs[j + 1] - diffs[i] <= 0.24:
            j += 1
        if j - i > best_j - best_i:
            best_i, best_j = i, j
    cluster = diffs[best_i:best_j + 1]
    if len(cluster) < 3:
        return None
    bias = cluster[len(cluster) // 2]
    mad = sorted(abs(d - bias) for d in cluster)[len(cluster) // 2]
    if mad > 0.12:
        return None
    return bias, len(cluster), mad


def target_episodes(rk_bot: dict, bias: float) -> list[tuple[float, float]]:
    """target_active 连续段（PC 报告轴），间隔 >1s 分段——每次抛球一段。"""
    episodes: list[tuple[float, float]] = []
    for s in rk_bot["samples"]:
        if not (s["active"] and s["tx"] is not None and s["ty"] is not None):
            continue
        t = s["t"] + bias
        if episodes and t - episodes[-1][1] <= 1.0:
            episodes[-1] = (episodes[-1][0], t)
        else:
            episodes.append((t, t))
    return episodes


def draw_car_tag_boxes(
    out: np.ndarray,
    car_loc: dict,
    sn: str,
    x_offset: int,
    y_offset: int,
    scale: float,
) -> None:
    """把该相机参与拟合的车载 AprilTag 检测角点画成四边形识别框（无文字）。"""
    tag_corners = car_loc.get("tag_corners")
    if not isinstance(tag_corners, dict):
        return
    for quad in (tag_corners.get(sn) or {}).values():
        pts = np.array(
            [[int(round(u * scale)) + x_offset, int(round(v * scale)) + y_offset]
             for u, v in quad],
            dtype=np.int32,
        )
        cv2.polylines(out, [pts], True, CAR_TAG_BOX_COLOR, 2, cv2.LINE_AA)


def convert_racket_loc_mm_to_m(loc: RacketLoc) -> RacketLoc:
    return RacketLoc(
        x=float(loc.x) / 1000.0,
        y=float(loc.y) / 1000.0,
        z=float(loc.z) / 1000.0,
        cameras_used=list(loc.cameras_used),
        pixels=dict(loc.pixels),
        reprojection_error=float(loc.reprojection_error),
        face_keypoint_score_min=float(loc.face_keypoint_score_min),
    )


def scale_panel_to_full(panel: np.ndarray) -> np.ndarray:
    """把半分辨率 panel 拉回原始坐标系大小，便于复用在线分片逻辑。"""
    h, w = panel.shape[:2]
    return cv2.resize(panel, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)


def extract_fullres_panels(
    img: np.ndarray,
    serials: list[str],
) -> dict[str, np.ndarray]:
    panels_half, _, _ = split_stitched_panels(img, serials)
    return {
        sn: scale_panel_to_full(panel)
        for sn, panel in panels_half.items()
    }


def init_racket_pipeline(
    first_frame: np.ndarray,
    serials: list[str],
    calib_config_path: str | Path,
    racket_model_path: str | Path,
    conf_threshold: float,
    pose_model_path: str | Path,
    keypoint_score_threshold: float,
    min_face_valid_keypoints: int,
) -> RacketPipeline:
    """Initialize the ArmCalibration racket-center pipeline for offline annotation."""
    racket_model_path = Path(racket_model_path)
    if not racket_model_path.exists():
        raise FileNotFoundError(f"找不到球拍模型: {racket_model_path}")
    pose_model_path = Path(pose_model_path)
    if not pose_model_path.exists():
        raise FileNotFoundError(f"找不到球拍关键点模型: {pose_model_path}")

    localizer = RacketLocalizer(
        calib_config_path=str(calib_config_path),
        racket_model_path=racket_model_path,
        pose_model_path=pose_model_path,
        bbox_conf=conf_threshold,
        keypoint_score_threshold=keypoint_score_threshold,
        min_valid_keypoints=min_face_valid_keypoints,
    )

    panels_full = extract_fullres_panels(first_frame, serials)
    try:
        localizer.locate(panels_full)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "球拍 ONNX 推理缺少依赖，请安装 onnx / onnxruntime 后重试。"
        ) from e

    return RacketPipeline(
        localizer=localizer,
        pose_model_path=str(pose_model_path),
        keypoint_score_threshold=float(keypoint_score_threshold),
        min_face_valid_keypoints=int(min_face_valid_keypoints),
    )


def detect_racket_frame(
    img: np.ndarray,
    serials: list[str],
    pipeline: RacketPipeline,
) -> tuple[dict[str, RacketDetection], Optional[RacketLoc]]:
    """Run the ArmCalibration racket detector on one stitched frame."""
    panels_full = extract_fullres_panels(img, serials)
    detections, loc = pipeline.localizer.locate(panels_full)
    if loc is not None:
        loc = convert_racket_loc_mm_to_m(loc)
    return detections, loc


def serialize_racket_detection(
    det: RacketDetection,
    *,
    keypoint_score_threshold: float,
) -> dict:
    payload = {
        "detected": bool(det.detected),
        "accepted": bool(det.accepted),
        "failure_reason": det.failure_reason,
    }
    if not det.detected:
        return payload

    if det.bbox_xyxy is not None:
        x1, y1, x2, y2 = det.bbox_xyxy
        payload["bbox"] = {
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "confidence": round(float(det.bbox_confidence), 4),
        }
        payload["x1"] = round(float(x1))
        payload["y1"] = round(float(y1))
        payload["x2"] = round(float(x2))
        payload["y2"] = round(float(y2))
        payload["conf"] = round(float(det.bbox_confidence), 3)

    if det.center_xy is not None:
        payload["x"] = round(float(det.center_xy[0]))
        payload["y"] = round(float(det.center_xy[1]))
        payload["center_xy"] = [
            round(float(det.center_xy[0]), 2),
            round(float(det.center_xy[1]), 2),
        ]

    if det.keypoints_xy is not None and len(det.keypoints_xy) > 0:
        keypoints_xy = np.asarray(det.keypoints_xy, dtype=np.float64)
        score_arr = np.asarray(det.keypoint_scores, dtype=np.float64)
        valid_mask = score_arr >= float(keypoint_score_threshold)
        payload["keypoints"] = [
            {
                "id": int(idx),
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "score": round(float(score_arr[idx]), 3),
                "valid": bool(valid_mask[idx]),
                "used_for_center": bool(idx in (0, 1, 2, 3)),
            }
            for idx, point in enumerate(keypoints_xy)
        ]
        payload["all_keypoints_center_xy"] = [
            round(float(keypoints_xy[:, 0].mean()), 2),
            round(float(keypoints_xy[:, 1].mean()), 2),
        ]
        payload["keypoint_score_mean"] = round(float(score_arr.mean()), 3)
        payload["keypoint_score_min"] = round(float(score_arr.min()), 3)
        payload["keypoint_score_max"] = round(float(score_arr.max()), 3)
        payload["valid_keypoint_count"] = int(np.sum(valid_mask))

    payload["center_keypoint_ids"] = [0, 1, 2, 3]
    payload["face_keypoint_score_min"] = round(float(det.face_keypoint_score_min), 3)
    payload["face_valid_keypoint_count"] = int(det.face_valid_keypoint_count)
    return payload


def serialize_racket_detections(
    detections: dict[str, RacketDetection],
    *,
    keypoint_score_threshold: float,
) -> dict[str, list[dict]]:
    serialized: dict[str, list[dict]] = {}
    for sn, det in detections.items():
        if not det.detected:
            continue
        serialized[sn] = [
            serialize_racket_detection(
                det,
                keypoint_score_threshold=keypoint_score_threshold,
            )
        ]
    return serialized


def serialize_3d(obj3d: Ball3D) -> dict:
    return {
        "x": round(obj3d.x, 4),
        "y": round(obj3d.y, 4),
        "z": round(obj3d.z, 4),
        "reproj": round(obj3d.reprojection_error, 1),
        "conf": round(obj3d.confidence, 3),
        "cameras": obj3d.cameras_used,
    }


def serialize_racket_3d(obj3d: RacketLoc) -> dict:
    return {
        "x": round(obj3d.x, 4),
        "y": round(obj3d.y, 4),
        "z": round(obj3d.z, 4),
        "reproj": round(obj3d.reprojection_error, 1),
        "conf": round(obj3d.face_keypoint_score_min, 3),
        "face_min": round(obj3d.face_keypoint_score_min, 3),
        "cameras": obj3d.cameras_used,
        "pixels": {
            sn: [round(float(px), 2), round(float(py), 2)]
            for sn, (px, py) in obj3d.pixels.items()
        },
    }


def serialize_car_loc(obj3d: CarLoc, *, elapsed_s: float | None) -> dict:
    return {
        "x": round(obj3d.x, 4),
        "y": round(obj3d.y, 4),
        "z": round(obj3d.z, 4),
        # 单 tag 退化帧 yaw 是 None（本帧无可信 yaw），如实写 null
        "yaw": None if obj3d.yaw is None else round(obj3d.yaw, 4),
        "yaw_valid": obj3d.yaw_valid,
        "t": obj3d.t,
        "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
        "tag_id": obj3d.tag_id,
        "reference": "car_base",
        "cameras_used": obj3d.cameras_used,
        "reprojection_error": round(obj3d.reprojection_error, 2),
        "pixels": {
            sn: [round(float(px)), round(float(py))]
            for sn, (px, py) in obj3d.pixels.items()
        },
        "tag_corners": {
            sn: {
                str(tag_id): [
                    [round(float(u), 1), round(float(v), 1)] for u, v in quad
                ]
                for tag_id, quad in tags.items()
            }
            for sn, tags in obj3d.corners_px.items()
        },
    }


def apply_racket_results(
    frame_data: dict,
    detections: dict[str, RacketDetection],
    racket3d: Optional[RacketLoc],
    *,
    keypoint_score_threshold: float,
) -> None:
    """把当前帧的球拍结果写回 JSON frame entry。"""
    frame_data.pop("racket_detections", None)
    frame_data.pop("racket3d", None)

    serialized_dets = serialize_racket_detections(
        detections,
        keypoint_score_threshold=keypoint_score_threshold,
    )
    if serialized_dets:
        frame_data["racket_detections"] = serialized_dets

    if racket3d is not None:
        frame_data["racket3d"] = serialize_racket_3d(racket3d)


def apply_car_result(
    frame_data: dict,
    car_loc: Optional[CarLoc],
    *,
    elapsed_s: float | None,
) -> None:
    frame_data.pop("car_loc", None)
    frame_data["car_loc_sampled"] = True
    frame_data["car_loc_status"] = "miss"
    if car_loc is not None:
        frame_data["car_loc"] = serialize_car_loc(car_loc, elapsed_s=elapsed_s)
        frame_data["car_loc_status"] = "hit"


def clear_car_results(data: dict) -> None:
    for frame_data in data.get("frames", []):
        if not isinstance(frame_data, dict):
            continue
        frame_data.pop("car_loc", None)
        frame_data.pop("car_loc_sampled", None)
        frame_data.pop("car_loc_status", None)
    data["car_locs"] = []

    summary = data.get("summary")
    if isinstance(summary, dict):
        summary.pop("car_locs", None)
        summary.pop("car_loc_sampled_frames", None)
        summary.pop("car_loc_misses", None)
        summary.pop("car_loc_dropped_frames", None)


def clear_racket_results(data: dict) -> None:
    """清理旧的球拍结果，避免重复运行时留下脏数据。"""
    for frame_data in data.get("frames", []):
        frame_data.pop("racket_detections", None)
        frame_data.pop("racket3d", None)
    data.pop("racket_observations", None)

    summary = data.get("summary")
    if isinstance(summary, dict):
        summary.pop("racket_observations_3d", None)
        summary.pop("racket_frames_processed", None)


def build_racket_json_payload(
    data: dict,
    *,
    source_json_path: Path,
    source_video_path: Path,
) -> dict:
    """Build a racket-only JSON payload that stays frame-aligned with the saved video."""
    cfg = data.get("config", {})
    summary = data.get("summary", {})
    racket_frames: list[dict] = []

    for frame_data in data.get("frames", []):
        if not isinstance(frame_data, dict):
            continue
        frame_payload: dict = {}
        for key in (
            "idx",
            "video_frame_idx",
            "video_mapping_exact",
            "exposure_pc",
            "elapsed_s",
        ):
            if key in frame_data:
                frame_payload[key] = frame_data[key]

        for key in ("racket_detections", "racket3d"):
            if key in frame_data:
                frame_payload[key] = frame_data[key]

        if frame_payload:
            racket_frames.append(frame_payload)

    return {
        "config": {
            "source_tracker_json": str(source_json_path),
            "source_video_path": str(source_video_path),
            "distance_unit": cfg.get("distance_unit", "m"),
            "serials": cfg.get("serials", []),
            "first_frame_exposure_pc": cfg.get("first_frame_exposure_pc"),
            "video_frame_mapping_exact": cfg.get("video_frame_mapping_exact"),
            "racket_model_path": cfg.get("racket_model_path"),
            "racket_pose_model_path": cfg.get("racket_pose_model_path"),
            "racket_conf_threshold": cfg.get("racket_conf_threshold"),
            "racket_keypoint_score_threshold": cfg.get("racket_keypoint_score_threshold"),
            "racket_min_face_valid_keypoints": cfg.get("racket_min_face_valid_keypoints"),
        },
        "summary": {
            "video_frame_mapping_exact": summary.get(
                "video_frame_mapping_exact",
                cfg.get("video_frame_mapping_exact"),
            ),
            "video_frames_mapped": summary.get("video_frames_mapped"),
            "racket_observations_3d": summary.get(
                "racket_observations_3d",
                len(data.get("racket_observations", [])),
            ),
            "racket_frames_processed": summary.get("racket_frames_processed"),
        },
        "frames": racket_frames,
        "racket_observations": data.get("racket_observations", []),
    }


def draw_scaled_detections(
    out: np.ndarray,
    detections: list[dict],
    x_offset: int,
    y_offset: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """把全分辨率检测框按缩放比例绘制到 annotated 视频（只画框，不写文字）。"""
    for det in detections:
        x1 = int(det["x1"] * scale) + x_offset
        y1 = int(det["y1"] * scale) + y_offset
        x2 = int(det["x2"] * scale) + x_offset
        y2 = int(det["y2"] * scale) + y_offset
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)


def draw_racket_detections(
    out: np.ndarray,
    detections: list[dict],
    x_offset: int,
    y_offset: int,
    scale: float,
) -> None:
    """Draw only racket keypoints and their face-center marker."""
    for det in detections:
        accepted = bool(det.get("accepted", False))
        for keypoint in det.get("keypoints", []):
            kp_x = int(keypoint["x"] * scale) + x_offset
            kp_y = int(keypoint["y"] * scale) + y_offset
            if keypoint.get("used_for_center"):
                kp_color = (0, 255, 0) if keypoint.get("valid") else (0, 165, 255)
            else:
                kp_color = (255, 200, 0)
            cv2.circle(out, (kp_x, kp_y), 5, kp_color, -1)

        if "x" in det and "y" in det:
            cx = int(det["x"] * scale) + x_offset
            cy = int(det["y"] * scale) + y_offset
            cv2.drawMarker(
                out,
                (cx, cy),
                (0, 0, 255) if accepted else (0, 165, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=2,
            )


def annotate_frame(
    img: np.ndarray,
    frame_data: dict,
    serials: list[str],
    n_cams: int,
    panel_w: int,
    panel_h: int,
    layout_cols: int,
    *,
    show_racket: bool = False,
    relative_time_s: float | None = None,
    return_event: dict | None = None,
    cam_projections: dict[str, dict] | None = None,
    frame_time: float | None = None,
) -> np.ndarray:
    """在拼接画面上绘制球/球拍/3D/状态等离线标注。"""
    h, w = img.shape[:2]
    out = img.copy()
    cols = layout_cols
    rows = max(1, math.ceil(n_cams / max(1, cols)))
    scale = 0.5

    detections = frame_data.get("detections", {})
    racket_detections = frame_data.get("racket_detections", {})
    frame_car_loc = frame_data.get("car_loc")

    for cam_idx, sn in enumerate(serials):
        color = BOX_COLORS[cam_idx % len(BOX_COLORS)]
        x_offset, y_offset = grid_slot(cam_idx, panel_w, panel_h, cols=cols)

        draw_scaled_detections(
            out,
            [
                det for det in detections.get(sn, [])
                if det.get("label", "tennis_ball") == "tennis_ball"
            ],
            x_offset,
            y_offset,
            scale,
            color,
        )

        if show_racket:
            draw_racket_detections(
                out,
                racket_detections.get(sn, []),
                x_offset,
                y_offset,
                scale,
            )

        if frame_car_loc:
            draw_car_tag_boxes(out, frame_car_loc, sn, x_offset, y_offset, scale)

    for col in range(1, cols):
        x = panel_w * col
        cv2.line(out, (x, 0), (x, h), (100, 100, 100), 1)
    for row in range(1, rows):
        y = panel_h * row
        cv2.line(out, (0, y), (w, y), (100, 100, 100), 1)

    line_h = 40
    lines: list[tuple[str, tuple[int, int, int]]] = []

    lines.append((
        f"#{frame_data['idx']}  "
        f"{('t=' + format(relative_time_s, '.3f') + 's  ') if relative_time_s is not None else ''}"
        f"perf={frame_data.get('exposure_pc', 0):.6f}s  "
        f"lat={frame_data.get('latency_ms', 0):.0f}ms",
        TEXT_COLOR,
    ))

    if show_racket:
        racket_parts = []
        for sn in serials:
            dets = racket_detections.get(sn, [])
            accepted_count = sum(1 for det in dets if det.get("accepted"))
            detected_count = sum(1 for det in dets if det.get("detected", True))
            racket_parts.append(f"{sn[-3:]}={accepted_count}/{detected_count}")
        lines.append((f"racket: {'  '.join(racket_parts)}", RACKET_BOX_COLOR))

    ball3d = frame_data.get("ball3d")
    if ball3d:
        cams = "+".join(s[-3:] for s in ball3d["cameras"])
        lines.append((
            f"3D: {_format_xyz_m(ball3d['x'], ball3d['y'], ball3d['z'])}  "
            f"reproj={ball3d['reproj']:.1f}px  cams={cams}  conf={ball3d['conf']:.2f}",
            TEXT_3D_COLOR,
        ))

    if return_event is not None:
        lines.append((
            f"RETURN yaw={return_event['yaw_deg']:+.1f}deg "
            f"pitch={return_event['pitch_deg']:+.1f}deg  "
            f"v=({return_event['vx']:+.2f},{return_event['vy']:+.2f},{return_event['vz']:+.2f})m/s  "
            f"|v|={return_event['speed']:.2f}",
            RETURN_COLOR,
        ))

    racket3d = frame_data.get("racket3d")
    if show_racket and racket3d:
        cams = "+".join(s[-3:] for s in racket3d["cameras"])
        lines.append((
            f"R3D: {_format_xyz_m(racket3d['x'], racket3d['y'], racket3d['z'])}  "
            f"reproj={racket3d['reproj']:.1f}px  cams={cams}  face_min={racket3d['face_min']:.1f}",
            TEXT_RACKET_3D_COLOR,
        ))

    state = frame_data.get("state", "idle")
    state_color = STATE_COLORS.get(state, TEXT_COLOR)
    state_str = f"curve3: {state}"
    pred = frame_data.get("prediction")
    if pred:
        state_str += (
            f"  hit={_format_xyz_m(pred['x'], pred['y'], pred['z'])} "
            f"stage={pred['stage']} lead={pred['lead_ms']}ms"
        )
    lines.append((state_str, state_color))

    if return_event is not None and frame_time is not None:
        draw_return_vector(
            out,
            return_event,
            frame_time,
            ball3d,
            serials,
            panel_w,
            panel_h,
            cols,
            cam_projections,
        )

    y = h - 15
    for text, color in reversed(lines):
        cv2.putText(out, text, (10, y), FONT, FONT_SCALE, color, FONT_THICKNESS)
        y -= line_h

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="离线标注视频，并补充球拍 2D/3D 结果")
    parser.add_argument("--input", required=True, help="输入 tracker JSON 路径")
    parser.add_argument("--video", default=None, help="原始拼接视频路径，默认自动查找同名 .mp4/.avi")
    parser.add_argument("--output", default=None, help="输出 annotated 视频路径，默认同目录 _annotated.avi")
    parser.add_argument("--no-output-video", action="store_true", help="只更新 JSON/HTML 所需结果，不写 annotated 视频")
    parser.add_argument("--json-output", default=None, help="补充后的 merged JSON 输出路径，默认覆写输入 JSON")
    parser.add_argument("--racket-json-output", default=None, help="单独输出球拍 2D/3D 与帧映射 JSON")
    parser.add_argument("--tracker-config", default=str(_DEFAULT_TRACKER_CONFIG), help="tracker.json 路径")
    parser.add_argument("--racket-model", default=str(_DEFAULT_RACKET_MODEL), help="球拍 bbox 模型路径")
    parser.add_argument("--racket-pose-model", default=str(_DEFAULT_RACKET_POSE_MODEL), help="球拍关键点模型路径")
    parser.add_argument("--racket-conf", type=float, default=0.25, help="球拍 bbox 置信度阈值")
    parser.add_argument("--racket-keypoint-threshold", type=float, default=40.0, help="ArmCalibration 同款 0-3 关键点分数阈值")
    parser.add_argument("--racket-min-face-valid-keypoints", type=int, default=4, help="ArmCalibration 同款中心关键点最少有效个数")
    parser.add_argument("--no-racket", action="store_true", help="只做旧标注，不补充球拍结果")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧，便于快速验证")
    args = parser.parse_args()

    json_path = Path(args.input)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    normalize_tracker_json_to_m(data)

    video_path = (
        Path(args.video)
        if args.video
        else guess_tracker_video_path(json_path, data)
    )
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return

    output_path = args.output or str(json_path.with_name(json_path.stem + "_annotated.avi"))
    json_output_path = Path(args.json_output) if args.json_output else json_path
    racket_json_output_path = (
        Path(args.racket_json_output)
        if args.racket_json_output
        else None
    )

    serials = data["config"]["serials"]
    calib_config_path = data["config"]["calib_config_path"]
    # 车体 AprilTag 布局跟着录制那场走：run_tracker 从 2026-08-15 起把 --car-config
    # 记进 session json。更早的 session 没这个字段，那批全是 v0.3 车。
    car_config_path = data["config"].get("car_config_path") or str(
        _PROJECT_ROOT / "src" / "config" / "arm_poe_racket_center.json"
    )
    n_cams = len(serials)
    frames_data = data["frames"]
    racket_enabled = not args.no_racket

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    car_cfg = data.get("config", {}).get("car_localizer", {})
    car_enabled = bool(car_cfg.get("enabled", True))
    tracker_car_sample_every_frames = None
    if isinstance(car_cfg.get("sample_every_frames"), int):
        tracker_car_sample_every_frames = max(
            int(car_cfg["sample_every_frames"]),
            1,
        )
    cols, rows = infer_stitched_grid(n_cams, w, h)
    panel_w = w // cols
    panel_h = h // rows
    frame_mapping, has_exact_mapping = build_video_frame_mapping(data, total)
    time_reference = extract_time_reference(data)
    sync_video_frame_metadata(data, frame_mapping, has_exact_mapping)

    return_events = detect_return_events(data.get("observations", []))

    # RK bot 目标点：必须在 clear_car_results() 之前用 tracker 原始 car_locs 做共享位姿锚
    rk_bot = None
    rk_bias = None
    rk_anchor_info: tuple[float, int, float] | None = None
    rk_path = guess_rk_tracking_path(json_path)
    if rk_path is not None:
        rk_bot = load_rk_bot_data(rk_path)
    if rk_bot is not None:
        pc_pose_rows = []
        for c in data.get("car_locs") or []:
            if not isinstance(c, dict):
                continue
            vals = [c.get(k) for k in ("t", "x", "y", "yaw")]
            if all(_FINITE(v) for v in vals):
                elapsed = (
                    float(vals[0]) - time_reference
                    if time_reference is not None
                    else float(vals[0])
                )
                pc_pose_rows.append((elapsed, float(vals[1]), float(vals[2]), float(vals[3])))
        rk_anchor_method = "共享位姿锚"
        rk_anchor_info = estimate_rk_time_bias(rk_bot["pose_rows"], pc_pose_rows)
        if rk_anchor_info is None:
            rk_anchor_method = "触球事件锚"
            rk_anchor_info = estimate_rk_bias_from_hits(
                rk_bot.get("throw_hts") or [],
                [
                    ev["t_hit"] - time_reference if time_reference is not None else ev["t_hit"]
                    for ev in return_events
                ],
            )
        if rk_anchor_info is None:
            rk_bot = None
        else:
            rk_bias = rk_anchor_info[0]

    cam_projections = None
    if not args.no_output_video:
        try:
            cam_projections = load_camera_projections(calib_config_path)
        except (OSError, KeyError, ValueError) as e:
            print(f"警告：标定加载失败，车框/球3D投影/回球矢量/目标点只画文字行: {e}")

    if car_enabled:
        clear_car_results(data)
    if racket_enabled:
        clear_racket_results(data)

    car_localizer = (
        CarLocalizer(
            calib_config_path=calib_config_path,
            vehicle_config_path=car_config_path,
        )
        if car_enabled
        else None
    )
    if car_localizer is not None:
        print(f"车体配置: {car_config_path}")

    writer = None
    if not args.no_output_video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"输入视频: {video_path} ({w}x{h}, {fps:.0f}fps, {total} 帧)")
    print(f"JSON 帧数: {len(frames_data)}")
    if has_exact_mapping:
        print(f"视频映射: 使用 JSON video_frame_indices（{len(frame_mapping)} 帧精确对齐）")
    else:
        print("视频映射: JSON 不含 video_frame_indices，退化为按帧号 1:1 对齐（若录制时丢帧，标注可能漂移）")
    if args.no_output_video:
        print("输出视频: disabled (--no-output-video)")
    else:
        print(f"输出视频: {output_path}")
    if car_enabled:
        print("AprilTag 离线补标: every frame")
        if tracker_car_sample_every_frames is not None:
            print(
                "Tracker 原始 AprilTag 采样: "
                f"1/{tracker_car_sample_every_frames}"
            )
    if racket_enabled:
        print(f"球拍模型: {args.racket_model}")
        print(f"球拍关键点模型: {args.racket_pose_model}")
        print(f"JSON 输出: {json_output_path}")
        if racket_json_output_path is not None:
            print(f"球拍 JSON 输出: {racket_json_output_path}")
    if return_events:
        print(f"PC 回球检测: {len(return_events)} 次（对应帧区间将叠加速度矢量）")
        for ev in return_events:
            rel = ev["t_hit"] - time_reference if time_reference is not None else ev["t_hit"]
            print(
                f"  t={rel:.3f}s yaw={ev['yaw_deg']:+.1f}deg "
                f"pitch={ev['pitch_deg']:+.1f}deg |v|={ev['speed']:.2f}m/s "
                f"n={ev['n_points']}" + ("（地面反弹前截断）" if ev["bounce_cut"] else "")
            )
    else:
        print("PC 回球检测: 未检出")
    if rk_bot is not None and rk_bias is not None:
        episodes = target_episodes(rk_bot, rk_bias)
        print(
            f"RK 目标点叠加: 时轴对齐 bias={rk_bias:+.4f}s"
            f"（{rk_anchor_method} ×{rk_anchor_info[1]}, MAD {rk_anchor_info[2] * 1000:.0f}ms），"
            f"target_active 段 {len(episodes)} 个:"
        )
        for lo, hi in episodes:
            print(f"  t={lo:.3f}s..{hi:.3f}s")
    elif rk_path is None:
        print("RK 目标点叠加: 未找到 *_rk_tracking.json，跳过")
    else:
        print("RK 目标点叠加: 共享位姿锚与触球事件锚均不可用，无法对齐 RK 时轴，跳过")

    frame_idx = 0
    n_annotated = 0
    car_observations: list[dict] = []
    car_frames_processed = 0
    racket_pipeline: Optional[RacketPipeline] = None
    racket_observations: list[dict] = []
    racket_frames_processed = 0

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ret, img = cap.read()
        if not ret:
            break

        if frame_idx < len(frame_mapping):
            fd = frames_data[frame_mapping[frame_idx]]
            frame_time = build_panel_timestamp(fd, frame_idx, fps)
            relative_time_s = build_relative_frame_time_s(
                fd, frame_idx, fps, time_reference
            )

            if car_localizer is not None:
                panels = extract_fullres_panels(img, serials)
                car_loc = car_localizer.locate(panels, t=frame_time)
                apply_car_result(fd, car_loc, elapsed_s=relative_time_s)
                car_frames_processed += 1
                if car_loc is not None:
                    car_observations.append({
                        "frame_idx": fd.get("idx", frame_mapping[frame_idx]),
                        "video_frame_idx": frame_idx,
                        **serialize_car_loc(car_loc, elapsed_s=relative_time_s),
                    })

            if racket_enabled:
                if racket_pipeline is None:
                    racket_pipeline = init_racket_pipeline(
                        first_frame=img,
                        serials=serials,
                        calib_config_path=calib_config_path,
                        racket_model_path=args.racket_model,
                        conf_threshold=args.racket_conf,
                        pose_model_path=args.racket_pose_model,
                        keypoint_score_threshold=args.racket_keypoint_threshold,
                        min_face_valid_keypoints=args.racket_min_face_valid_keypoints,
                    )
                    print(
                        "球拍关键点预热完成: "
                        f"pose={racket_pipeline.pose_model_path}, "
                        f"thr={racket_pipeline.keypoint_score_threshold:.1f}, "
                        f"min_face_valid={racket_pipeline.min_face_valid_keypoints}"
                    )

                racket_dets, racket3d = detect_racket_frame(
                    img, serials, racket_pipeline
                )
                apply_racket_results(
                    fd,
                    racket_dets,
                    racket3d,
                    keypoint_score_threshold=racket_pipeline.keypoint_score_threshold,
                )
                racket_frames_processed += 1

                if racket3d is not None:
                    racket_observations.append({
                        "frame_idx": fd.get("idx", frame_mapping[frame_idx]),
                        "video_frame_idx": frame_idx,
                        "x": racket3d.x,
                        "y": racket3d.y,
                        "z": racket3d.z,
                        "t": frame_time,
                        "elapsed_s": relative_time_s,
                        "reproj_err": racket3d.reprojection_error,
                        "confidence": racket3d.face_keypoint_score_min,
                        "face_keypoint_score_min": racket3d.face_keypoint_score_min,
                        "cameras_used": racket3d.cameras_used,
                    })

            active_return = None
            for ev in return_events:
                if ev["t_hit"] <= frame_time <= ev["t_end"]:
                    active_return = ev
                    break

            annotated = annotate_frame(
                img,
                fd,
                serials,
                n_cams,
                panel_w,
                panel_h,
                cols,
                show_racket=racket_enabled,
                relative_time_s=relative_time_s,
                return_event=active_return,
                cam_projections=cam_projections,
                frame_time=frame_time,
            )
            n_annotated += 1
        else:
            annotated = img

        if writer is not None:
            writer.write(annotated)
        frame_idx += 1

        if frame_idx % 200 == 0:
            if racket_enabled and car_enabled:
                print(
                    f"  {frame_idx}/{total} 帧... "
                    f"car_3d={len(car_observations)}  "
                    f"racket_3d={len(racket_observations)}"
                )
            elif racket_enabled:
                print(
                    f"  {frame_idx}/{total} 帧... "
                    f"racket_3d={len(racket_observations)}"
                )
            elif car_enabled:
                print(
                    f"  {frame_idx}/{total} 帧... "
                    f"car_3d={len(car_observations)}"
                )
            else:
                print(f"  {frame_idx}/{total} 帧...")

    cap.release()
    if writer is not None:
        writer.release()

    if car_enabled:
        config = data.setdefault("config", {})
        summary = data.setdefault("summary", {})
        car_cfg_out = config.setdefault("car_localizer", {})
        if tracker_car_sample_every_frames is not None:
            car_cfg_out["tracker_sample_every_frames"] = tracker_car_sample_every_frames
        car_cfg_out["sample_every_frames"] = 1
        car_cfg_out["result_source"] = "annotate_video_full_frames"
        data["car_locs"] = car_observations
        summary["car_locs"] = len(car_observations)
        summary["car_loc_sampled_frames"] = car_frames_processed
        summary["car_loc_misses"] = max(
            car_frames_processed - len(car_observations),
            0,
        )

    if racket_enabled:
        config = data.setdefault("config", {})
        summary = data.setdefault("summary", {})
        config["racket_model_path"] = str(
            Path(args.racket_model)
        )
        config["racket_pose_model_path"] = str(
            Path(args.racket_pose_model)
        )
        config["racket_conf_threshold"] = args.racket_conf
        config["racket_keypoint_score_threshold"] = args.racket_keypoint_threshold
        config["racket_min_face_valid_keypoints"] = args.racket_min_face_valid_keypoints
        data["racket_observations"] = racket_observations
        summary["racket_observations_3d"] = len(racket_observations)
        summary["racket_frames_processed"] = racket_frames_processed

        if racket_json_output_path is not None:
            racket_payload = build_racket_json_payload(
                data,
                source_json_path=json_path,
                source_video_path=video_path,
            )
            with open(racket_json_output_path, "w", encoding="utf-8") as f:
                json.dump(racket_payload, f, ensure_ascii=False, indent=2)
            print(f"球拍 JSON 已输出: {racket_json_output_path}")

    data["pc_return_events"] = serialize_return_events(return_events, time_reference)
    if car_enabled or racket_enabled or return_events:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON 已更新: {json_output_path}")

    if racket_enabled:
        print(f"球拍 3D 观测数: {len(racket_observations)}")

    if writer is not None:
        print(f"完成：{n_annotated} 帧已标注，输出到 {output_path}")
    else:
        print(f"完成：{n_annotated} 帧已处理，未写出 annotated 视频")


if __name__ == "__main__":
    main()
