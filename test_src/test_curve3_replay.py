# -*- coding: utf-8 -*-
"""
离线回放验证脚本：从 tracker JSON 读取 3D observations，
逐帧喂给 Curve3Tracker，生成相同格式的 JSON 并调用 HTML 可视化。

用法：
  python test_curve3_replay.py --input tracker_output/tracker_20260227_124452.json
                               [--ideal-hit-z 500] [--cor 0.75] [--cor-xy 0.45]
                               [--motion-window 0.2] [--motion-min-y 500]
                               [--no-html]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# 确保 src 可以导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.curve3 import BallObservation, Curve3Tracker, TrackerState

# 少于此数量的 group 视为噪声，不作为有效 throw
MIN_THROW_OBS = 10


def replay(input_path: str, **overrides) -> dict:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_obs = data["observations"]
    orig_config = data.get("config", {})

    tracker = Curve3Tracker(prediction_time_mode="observation", **overrides)

    log_observations = []
    log_predictions = []
    log_state_transitions = []
    prev_state = TrackerState.IDLE

    for i, o in enumerate(raw_obs):
        obs = BallObservation(x=o["x"], y=o["y"], z=o["z"], t=o["t"])
        result = tracker.update(obs)

        log_observations.append({
            "x": obs.x, "y": obs.y, "z": obs.z, "t": obs.t,
        })

        if result.prediction is not None:
            p = result.prediction
            log_predictions.append({
                "x": p.x, "y": p.y, "z": p.z,
                "stage": p.stage, "ct": p.ct, "ht": p.ht,
            })

        if result.state != prev_state:
            log_state_transitions.append({
                "obs_idx": i, "t": obs.t,
                "from": prev_state.value, "to": result.state.value,
            })
            prev_state = result.state

    # 统计每组（throw）的观测数
    resets = tracker.reset_times
    groups = []
    obs_times = [o["t"] for o in log_observations]
    start = 0
    for rt in resets:
        count = sum(1 for t in obs_times[start:] if t < rt)
        groups.append(count)
        start += count
    groups.append(len(obs_times) - start)

    s0_preds = [p for p in log_predictions if p["stage"] == 0]
    s1_preds = [p for p in log_predictions if p["stage"] == 1]

    result_json = {
        "config": {
            "start_time": orig_config.get("start_time", ""),
            "serial_left": orig_config.get("serial_left", ""),
            "serial_right": orig_config.get("serial_right", ""),
            "duration_s": (obs_times[-1] - obs_times[0]) if obs_times else 0,
            "ideal_hit_z": tracker.ideal_hit_z,
            "cor": tracker.cor,
            "cor_xy": tracker.cor_xy,
            "replay_source": os.path.basename(input_path),
            "motion_window_s": tracker.motion_window_s,
            "motion_min_y": tracker.motion_min_y,
            "prediction_time_mode": tracker.prediction_time_mode,
        },
        "summary": {
            "total_observations": len(raw_obs),
            "predictions": len(log_predictions),
            "s0_predictions": len(s0_preds),
            "s1_predictions": len(s1_preds),
            "state_transitions": len(log_state_transitions),
            "reset_times": resets,
            "throw_obs_counts": groups,
        },
        "observations": log_observations,
        "predictions": log_predictions,
        "state_transitions": log_state_transitions,
    }

    return result_json


def _get_throws(obs, resets):
    """按 reset 边界划分 throw，过滤掉过短的噪声 group，返回连续编号。"""
    boundaries = [obs[0]["t"]] + resets + [obs[-1]["t"] + 1]
    throws = []
    for gi in range(len(boundaries) - 1):
        t_start, t_end = boundaries[gi], boundaries[gi + 1]
        g_obs = [o for o in obs if t_start <= o["t"] < t_end]
        if len(g_obs) >= MIN_THROW_OBS:
            throws.append((t_start, t_end, g_obs))
    return throws


def print_summary(result: dict) -> None:
    summary = result["summary"]
    config = result["config"]
    preds = result["predictions"]
    obs = result["observations"]
    resets = summary["reset_times"]

    print(f"\n{'=' * 60}")
    print(f"Replay Summary  (source: {config['replay_source']})")
    print(f"{'=' * 60}")
    print(f"  Motion filter:  window={config.get('motion_window_s', '?')}s, "
          f"min_y={config.get('motion_min_y', '?')}mm")
    print(f"  COR: z={config['cor']}, xy={config.get('cor_xy', '?')}")
    print(f"  Total observations:    {summary['total_observations']}")
    print(f"  Predictions:           {summary['predictions']} "
          f"(S0={summary['s0_predictions']}, S1={summary['s1_predictions']})")
    print(f"  Resets:                {len(resets)}")
    print(f"  Raw group obs counts:  {summary['throw_obs_counts']}")

    throws = _get_throws(obs, resets)
    for ti, (t_start, t_end, g_obs) in enumerate(throws):
        g_s0 = [p for p in preds
                if p["stage"] == 0 and t_start <= p["ct"] < t_end]
        g_s1 = [p for p in preds
                if p["stage"] == 1 and t_start <= p["ct"] < t_end]

        print(f"\n  -- Throw {ti} ({len(g_obs)} obs) --")
        if g_s0:
            ys = [p["y"] for p in g_s0]
            xs = [p["x"] for p in g_s0]
            print(f"    S0: {len(g_s0)} preds, "
                  f"y=[{min(ys):.0f}, {max(ys):.0f}], "
                  f"x=[{min(xs):.0f}, {max(xs):.0f}]")
        else:
            print(f"    S0: no predictions")
        if g_s1:
            zs = [p["z"] for p in g_s1]
            ys = [p["y"] for p in g_s1]
            print(f"    S1: {len(g_s1)} preds, "
                  f"z=[{min(zs):.0f}, {max(zs):.0f}], "
                  f"y=[{min(ys):.0f}, {max(ys):.0f}]")
        else:
            print(f"    S1: no predictions")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay tracker JSON through Curve3Tracker offline")
    parser.add_argument("--input", required=True,
                        help="Path to tracker JSON file")
    parser.add_argument("--ideal-hit-z", type=float, default=None)
    parser.add_argument("--cor", type=float, default=None)
    parser.add_argument("--cor-xy", type=float, default=None)
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML generation")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto)")
    args = parser.parse_args()

    # CLI 参数覆盖默认值
    overrides = {}
    if args.ideal_hit_z is not None:
        overrides["ideal_hit_z"] = args.ideal_hit_z
    if args.cor is not None:
        overrides["cor"] = args.cor
    if args.cor_xy is not None:
        overrides["cor_xy"] = args.cor_xy

    result = replay(args.input, **overrides)

    # 输出 JSON
    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input).stem
        out_dir = Path(args.input).parent
        out_path = str(out_dir / f"{stem}_replay.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Replay JSON saved: {out_path}")

    print_summary(result)

    # 生成 HTML
    if not args.no_html:
        html_script = Path(__file__).parent / "generate_curve3_html.py"
        if html_script.exists():
            html_path = out_path.replace(".json", ".html")
            subprocess.run([
                sys.executable, str(html_script),
                "--input", out_path,
                "--output", html_path,
            ], check=True)
        else:
            print(f"Warning: {html_script} not found, skip HTML generation")


if __name__ == "__main__":
    main()
