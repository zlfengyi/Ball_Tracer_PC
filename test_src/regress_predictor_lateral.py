# -*- coding: utf-8 -*-
"""V04 historical A/B for the predictor's lateral-velocity model.

This is deliberately an execution-conditioned model replay, not a full
BotTrajectoryPredictor/TravelController replay.  For every published Stage1
snapshot it matches the target transition that authorized that prediction,
uses the later RK execution's measured wheel axis and longitudinal velocity
from that event to HT, changes only the treatment of wheel-frame lateral
velocity, and compares endpoints with event pose plus RK vx/vy integration.
The direct RK pose at HT is retained only as a pose-correction diagnostic.

old: discard v_perp at every 10 ms step.
new: retain world vx/vy and decay v_perp with a time constant plus the V04
     max-lateral-acceleration cap.

All business timestamps come from the RK payload perf_counter axis.  Bot pose
belongs to bot.imu_t, not bot.t.  Truth is accepted only by two-sided linear
interpolation with a bounded gap; there is no nearest-neighbour, freeze, or
extrapolation fallback.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


STEP_SEC = 0.01
SEGMENT_GAP_SEC = 0.70
ANCHOR_MIN_DURATION_SEC = 0.30
TARGET_MATCH_MAX_LATENCY_SEC = 0.25
TARGET_MATCH_X_TOL_M = 0.03
TARGET_MATCH_DEADLINE_TOL_SEC = 0.001
DEFAULT_TAU_GRID = (0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30)
DEFAULT_RUN_NAMES = (
    "tracker_20260826_134922",
    "tracker_20260825_041629",
    "tracker_20260825_040224",
    "tracker_20260825_031528",
    "tracker_20260825_025455",
    "tracker_20260825_023918",
    "tracker_20260825_021910",
    "tracker_20260825_020939",
    "tracker_20260824_120301",
    "tracker_20260824_075317",
    "tracker_20260824_032859",
    "tracker_20260824_030024",
    "tracker_20260822_145905",
    "tracker_20260822_124829",
    "tracker_20260822_113456",
    "tracker_20260822_112835",
    "tracker_20260822_094941",
    "tracker_20260822_071624",
    "tracker_20260822_070331",
    "tracker_20260822_054103",
)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True)
class BotSample:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float


@dataclass(frozen=True)
class TargetTransition:
    t: float
    x: float
    deadline: float
    delta_x: float


class BotTrace:
    def __init__(self, payload: dict, max_gap_sec: float):
        bot = payload["bot"]
        values = bot["y"]
        rows: dict[float, BotSample] = {}
        for t, x, y, yaw, vx, vy in zip(
            values["imu_t"],
            values["x"],
            values["y"],
            values["yaw"],
            values["vx"],
            values["vy"],
        ):
            if all(finite(v) for v in (t, x, y, yaw, vx, vy)):
                rows[float(t)] = BotSample(
                    float(x), float(y), float(yaw), float(vx), float(vy)
                )
        if len(rows) < 2:
            raise ValueError("bot.imu_t has fewer than two finite pose samples")
        self.times = sorted(rows)
        self.rows = [rows[t] for t in self.times]
        self.max_gap_sec = max_gap_sec

        steer = payload["steer_motor"]
        steer_rows: dict[float, float] = {}
        for t, angle in zip(steer["t"], steer["y"]["position"]):
            if finite(t) and finite(angle):
                steer_rows[float(t)] = float(angle)
        if len(steer_rows) < 2:
            raise ValueError("steer_motor has fewer than two finite samples")
        self.steer_times = sorted(steer_rows)
        self.steer_angles = [steer_rows[t] for t in self.steer_times]

        target_rows: list[TargetTransition] = []
        previous: TargetTransition | None = None
        for t, active, target_x, remaining in zip(
            bot["t"],
            values["target_active"],
            values["target_x"],
            values["remaining"],
        ):
            if not active:
                previous = None
                continue
            if not all(finite(v) for v in (t, target_x, remaining)):
                continue
            x_value = float(target_x)
            current = TargetTransition(
                float(t),
                x_value,
                float(t) + float(remaining),
                0.0 if previous is None else x_value - previous.x,
            )
            if (
                previous is None
                or abs(current.x - previous.x) > 5e-5
                or abs(current.deadline - previous.deadline) > 5e-4
            ):
                target_rows.append(current)
            previous = current
        self.target_transitions = target_rows

    def state_at(self, t: float) -> BotSample | None:
        i = bisect.bisect_left(self.times, t)
        if i < len(self.times) and abs(self.times[i] - t) <= 1e-9:
            return self.rows[i]
        if i == 0 or i == len(self.times):
            return None
        ta, tb = self.times[i - 1], self.times[i]
        if tb - ta > self.max_gap_sec:
            return None
        f = (t - ta) / (tb - ta)
        a, b = self.rows[i - 1], self.rows[i]
        dyaw = math.remainder(b.yaw - a.yaw, 2.0 * math.pi)
        return BotSample(
            a.x + f * (b.x - a.x),
            a.y + f * (b.y - a.y),
            a.yaw + f * dyaw,
            a.vx + f * (b.vx - a.vx),
            a.vy + f * (b.vy - a.vy),
        )

    def steer_at(self, t: float) -> float | None:
        i = bisect.bisect_left(self.steer_times, t)
        if i < len(self.steer_times) and abs(self.steer_times[i] - t) <= 1e-9:
            return self.steer_angles[i]
        if i == 0 or i == len(self.steer_times):
            return None
        ta, tb = self.steer_times[i - 1], self.steer_times[i]
        if tb - ta > self.max_gap_sec:
            return None
        f = (t - ta) / (tb - ta)
        return self.steer_angles[i - 1] + f * (
            self.steer_angles[i] - self.steer_angles[i - 1]
        )

    def target_x_jump(self, start: float, end: float) -> float:
        jumps = [
            abs(row.delta_x)
            for row in self.target_transitions
            if start <= row.t <= end
        ]
        return max(jumps, default=0.0)

    def match_target_transition(
        self, pred: "Prediction", target_x_offset: float, start_index: int
    ) -> tuple[float | None, int]:
        expected_x = pred.hit_x + target_x_offset
        candidates: list[tuple[float, int, TargetTransition]] = []
        for i in range(start_index, len(self.target_transitions)):
            row = self.target_transitions[i]
            if row.t < pred.ct:
                continue
            if row.t > pred.ct + TARGET_MATCH_MAX_LATENCY_SEC:
                break
            x_error = abs(row.x - expected_x)
            deadline_error = abs(row.deadline - pred.ht)
            if (
                x_error <= TARGET_MATCH_X_TOL_M
                and deadline_error <= TARGET_MATCH_DEADLINE_TOL_SEC
            ):
                score = x_error / TARGET_MATCH_X_TOL_M + deadline_error / TARGET_MATCH_DEADLINE_TOL_SEC
                candidates.append((score, i, row))
        if not candidates:
            return None, start_index
        _score, index, row = min(candidates, key=lambda item: item[0])
        return row.t, index + 1


@dataclass(frozen=True)
class Prediction:
    ct: float
    ht: float
    duration: float
    hit_x: float
    car_pred_x: float
    car_pred_y: float
    event_t: float | None = None


@dataclass
class ReplayRow:
    run: str
    segment: int
    ct: float
    event_t: float
    ht: float
    duration: float
    horizon: float
    init_vperp: float
    init_slip_deg: float
    target_x_jump: float
    logged_ex: float
    logged_ey: float
    old_ex: float
    old_ey: float
    new_ex: float
    new_ey: float
    pose_correction_ex: float
    pose_correction_ey: float


@dataclass(frozen=True)
class RunContext:
    name: str
    git_version: str
    max_lateral_accel: float
    trace: BotTrace
    segments: list[list[Prediction]]


@dataclass
class Evaluation:
    all_rows: list[ReplayRow]
    anchor_rows: list[ReplayRow]
    per_run: dict[str, list[ReplayRow]]
    skips: Counter[str]


def prediction_segments(payload: dict) -> list[list[Prediction]]:
    pred = payload.get("pred")
    if not isinstance(pred, dict) or not isinstance(pred.get("y"), dict):
        return []
    values = pred["y"]
    required = ("stage", "ht_rel", "duration", "x", "car_pred_x", "car_pred_y")
    if not isinstance(pred.get("t"), list) or any(
        not isinstance(values.get(key), list) for key in required
    ):
        return []
    segments: list[list[Prediction]] = []
    current: list[Prediction] = []
    for ct, stage, ht, duration, hit_x, car_x, car_y in zip(
        pred["t"],
        values["stage"],
        values["ht_rel"],
        values["duration"],
        values["x"],
        values["car_pred_x"],
        values["car_pred_y"],
    ):
        if stage != 1:
            if current:
                segments.append(current)
                current = []
            continue
        if not all(finite(v) for v in (ct, ht, duration, hit_x, car_x, car_y)):
            continue
        row = Prediction(
            float(ct),
            float(ht),
            float(duration),
            float(hit_x),
            float(car_x),
            float(car_y),
        )
        if current and row.ct - current[-1].ct > SEGMENT_GAP_SEC:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def attach_event_times(
    segments: list[list[Prediction]], trace: BotTrace, target_x_offset: float
) -> tuple[list[list[Prediction]], int]:
    matched: list[list[Prediction]] = []
    transition_index = 0
    unmatched = 0
    for segment in segments:
        output_segment = []
        for pred in segment:
            event_t, next_index = trace.match_target_transition(
                pred, target_x_offset, transition_index
            )
            if event_t is None:
                unmatched += 1
            else:
                transition_index = next_index
            output_segment.append(
                Prediction(
                    pred.ct,
                    pred.ht,
                    pred.duration,
                    pred.hit_x,
                    pred.car_pred_x,
                    pred.car_pred_y,
                    event_t,
                )
            )
        matched.append(output_segment)
    return matched, unmatched


def replay_prediction(
    run: str,
    segment: int,
    pred: Prediction,
    trace: BotTrace,
    tau_sec: float,
    max_lateral_accel: float,
    target_x_jump: float,
) -> tuple[ReplayRow | None, str | None]:
    if pred.event_t is None:
        return None, "target_transition_unmatched"
    if pred.ht <= pred.event_t:
        return None, "nonpositive_horizon"
    seed = trace.state_at(pred.event_t)
    pose_at_ht = trace.state_at(pred.ht)
    steer0 = trace.steer_at(pred.event_t)
    if seed is None:
        return None, "seed_pose_gap"
    if pose_at_ht is None:
        return None, "truth_pose_gap"
    if steer0 is None:
        return None, "seed_steer_gap"

    axis0 = seed.yaw + steer0
    c0, s0 = math.cos(axis0), math.sin(axis0)
    init_vparallel = seed.vx * c0 + seed.vy * s0
    init_vperp = -seed.vx * s0 + seed.vy * c0
    init_slip = math.degrees(math.atan2(abs(init_vperp), abs(init_vparallel)))

    old_x = new_x = integral_x = seed.x
    old_y = new_y = integral_y = seed.y
    new_vx, new_vy = seed.vx, seed.vy
    t = pred.event_t
    while t < pred.ht - 1e-12:
        dt = min(STEP_SEC, pred.ht - t)
        actual = trace.state_at(t)
        actual_next = trace.state_at(t + dt)
        steer = trace.steer_at(t)
        if actual is None or actual_next is None:
            return None, "path_pose_gap"
        if steer is None:
            return None, "path_steer_gap"
        axis = actual.yaw + steer
        c, s = math.cos(axis), math.sin(axis)
        vparallel = actual.vx * c + actual.vy * s

        old_vx, old_vy = vparallel * c, vparallel * s
        old_x += old_vx * dt
        old_y += old_vy * dt

        vperp = -new_vx * s + new_vy * c
        decay = -math.expm1(-dt / tau_sec)
        delta = min(abs(vperp) * decay, max_lateral_accel * dt)
        vperp -= math.copysign(delta, vperp)
        new_vx = vparallel * c - vperp * s
        new_vy = vparallel * s + vperp * c
        new_x += new_vx * dt
        new_y += new_vy * dt

        integral_x += 0.5 * (actual.vx + actual_next.vx) * dt
        integral_y += 0.5 * (actual.vy + actual_next.vy) * dt
        t += dt

    return (
        ReplayRow(
            run=run,
            segment=segment,
            ct=pred.ct,
            event_t=pred.event_t,
            ht=pred.ht,
            duration=pred.duration,
            horizon=pred.ht - pred.event_t,
            init_vperp=init_vperp,
            init_slip_deg=init_slip,
            target_x_jump=target_x_jump,
            logged_ex=pred.car_pred_x - integral_x,
            logged_ey=pred.car_pred_y - integral_y,
            old_ex=old_x - integral_x,
            old_ey=old_y - integral_y,
            new_ex=new_x - integral_x,
            new_ey=new_y - integral_y,
            pose_correction_ex=pose_at_ht.x - integral_x,
            pose_correction_ey=pose_at_ht.y - integral_y,
        ),
        None,
    )


def model_errors(row: ReplayRow, model: str) -> tuple[float, float]:
    return getattr(row, f"{model}_ex"), getattr(row, f"{model}_ey")


def metrics(rows: list[ReplayRow], model: str) -> dict[str, float | int]:
    errors = [model_errors(row, model) for row in rows]
    ex = [x * 100.0 for x, _ in errors]
    ey = [y * 100.0 for _, y in errors]
    e2 = [math.hypot(x, y) for x, y in zip(ex, ey)]
    if not errors:
        return {"n": 0}
    return {
        "n": len(errors),
        "xbias": statistics.fmean(ex),
        "xmae": statistics.fmean(abs(x) for x in ex),
        "xrmse": math.sqrt(statistics.fmean(x * x for x in ex)),
        "ybias": statistics.fmean(ey),
        "ymae": statistics.fmean(abs(y) for y in ey),
        "yrmse": math.sqrt(statistics.fmean(y * y for y in ey)),
        "e2mae": statistics.fmean(e2),
        "e2rmse": math.sqrt(statistics.fmean(x * x for x in e2)),
        "x5": sum(abs(x) > 5.0 for x in ex),
        "x7": sum(abs(x) > 7.0 for x in ex),
        "e25": sum(x > 5.0 for x in e2),
        "e27": sum(x > 7.0 for x in e2),
    }


def print_metric_block(title: str, rows: list[ReplayRow]) -> None:
    print(f"\n{title}")
    print(
        "model       n   xbias  xMAE xRMSE   ybias  yMAE yRMSE  "
        "2DMAE 2DRMSE  |x|>5 |x|>7 2D>5 2D>7   (cm)"
    )
    for model in ("logged", "old", "new"):
        m = metrics(rows, model)
        if not m["n"]:
            print(f"{model:<10} 0")
            continue
        print(
            f"{model:<10} {m['n']:4d} {m['xbias']:+7.2f} {m['xmae']:5.2f} "
            f"{m['xrmse']:5.2f} {m['ybias']:+7.2f} {m['ymae']:5.2f} "
            f"{m['yrmse']:5.2f} {m['e2mae']:6.2f} {m['e2rmse']:6.2f} "
            f"{m['x5']:6d} {m['x7']:6d} {m['e25']:5d} {m['e27']:5d}"
        )


def select_runs(root: Path) -> list[tuple[Path, dict]]:
    selected: list[tuple[Path, dict]] = []
    for run_name in DEFAULT_RUN_NAMES:
        run_dir = root / run_name
        path = run_dir / f"{run_dir.name}_rk_tracking.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing required RK tracking JSON: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        announce = payload.get("config_announce", {})
        car_name = announce.get("bot_center", {}).get("car_name") or announce.get(
            "chassis", {}
        ).get("car_name")
        if car_name != "v04":
            raise ValueError(f"{run_name}: expected car_name=v04, got {car_name!r}")
        if not prediction_segments(payload):
            raise ValueError(f"{run_name}: no usable Stage1 prediction segment")
        bot = payload.get("bot", {})
        if len(bot.get("t", [])) < 2:
            raise ValueError(f"{run_name}: no usable /bot_state series")
        selected.append((run_dir, payload))
    return selected


def prepare_contexts(
    selected: list[tuple[Path, dict]],
    max_gap_sec: float,
    max_lateral_accel_override: float | None,
) -> list[RunContext]:
    contexts = []
    for run_dir, payload in selected:
        trace = BotTrace(payload, max_gap_sec)
        announce = payload["config_announce"]["bot_center"]
        params = announce["params"]
        max_lateral_accel = (
            max_lateral_accel_override
            if max_lateral_accel_override is not None
            else float(params["max_lateral_accel"])
        )
        target_x_offset = float(params["travel_target_x_offset"])
        segments, _unmatched = attach_event_times(
            prediction_segments(payload), trace, target_x_offset
        )
        contexts.append(
            RunContext(
                run_dir.name,
                announce.get("git", "unknown"),
                max_lateral_accel,
                trace,
                segments,
            )
        )
    return contexts


def evaluate_contexts(contexts: list[RunContext], tau_sec: float) -> Evaluation:
    all_rows: list[ReplayRow] = []
    anchor_rows: list[ReplayRow] = []
    per_run: dict[str, list[ReplayRow]] = {}
    skips: Counter[str] = Counter()
    for context in contexts:
        run_anchors: list[ReplayRow] = []
        for segment_index, segment in enumerate(context.segments, 1):
            event_times = [p.event_t for p in segment if p.event_t is not None]
            target_jump = (
                context.trace.target_x_jump(min(event_times), max(event_times) + 0.02)
                if event_times
                else 0.0
            )
            valid_by_ct: dict[float, ReplayRow] = {}
            for pred in segment:
                row, reason = replay_prediction(
                    context.name,
                    segment_index,
                    pred,
                    context.trace,
                    tau_sec,
                    context.max_lateral_accel,
                    target_jump,
                )
                if row is None:
                    skips[reason or "unknown"] += 1
                    continue
                valid_by_ct[pred.ct] = row
                all_rows.append(row)
            candidates = [
                p
                for p in segment
                if p.event_t is not None
                and p.ht - p.event_t > ANCHOR_MIN_DURATION_SEC
            ]
            if candidates:
                anchor = valid_by_ct.get(candidates[-1].ct)
                if anchor is not None:
                    anchor_rows.append(anchor)
                    run_anchors.append(anchor)
                else:
                    skips["anchor_invalid"] += 1
        per_run[context.name] = run_anchors
    return Evaluation(all_rows, anchor_rows, per_run, skips)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tracker_output",
    )
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument(
        "--tau-grid",
        default=",".join(f"{value:g}" for value in DEFAULT_TAU_GRID),
        help="comma-separated whole-run holdout candidates",
    )
    parser.add_argument("--max-gap-ms", type=float, default=30.0)
    parser.add_argument(
        "--max-lateral-accel",
        type=float,
        default=None,
        help="override per-run bot_center max_lateral_accel",
    )
    args = parser.parse_args()
    if args.tau <= 0.0 or args.max_gap_ms <= 0.0:
        parser.error("tau and max-gap-ms must be positive")
    if args.max_lateral_accel is not None and args.max_lateral_accel <= 0.0:
        parser.error("max-lateral-accel must be positive")

    try:
        tau_grid = tuple(sorted({float(v) for v in args.tau_grid.split(",")} | {args.tau, 0.10}))
    except ValueError:
        parser.error("tau-grid must be comma-separated numbers")
    if not tau_grid or any(value <= 0.0 for value in tau_grid):
        parser.error("every tau-grid value must be positive")

    runs = select_runs(args.root.resolve())
    max_gap_sec = args.max_gap_ms / 1000.0
    contexts = prepare_contexts(runs, max_gap_sec, args.max_lateral_accel)
    evaluations = {tau: evaluate_contexts(contexts, tau) for tau in tau_grid}
    fixed = evaluations[args.tau]
    all_rows = fixed.all_rows
    anchor_rows = fixed.anchor_rows
    skips = fixed.skips
    total_snapshots = sum(len(segment) for c in contexts for segment in c.segments)
    total_segments = sum(len(c.segments) for c in contexts)

    print("Execution-conditioned V04 predictor lateral A/B")
    print(
        "SEMANTICS: old/new use the later RK execution's actual wheel axis and "
        "v_parallel from the matched target transition to the same HT. This isolates lateral velocity handling; "
        "it is NOT a full closed-loop BotTrajectoryPredictor replay."
    )
    print(
        f"new: tau={args.tau:.3f}s, dt={STEP_SEC:.3f}s, "
        "max_lateral_accel=per-run config; primary truth=event pose + trapezoidal "
        "integration of RK /bot_state vx/vy on imu_t, "
        f"two-sided interpolation gap<={args.max_gap_ms:.1f}ms."
    )
    print(
        "event_t contract: target_x~=pred.x+travel_target_x_offset and "
        "bot.t+remaining~=pred.ht; "
        f"x_tol={TARGET_MATCH_X_TOL_M*100:.1f}cm, "
        f"deadline_tol={TARGET_MATCH_DEADLINE_TOL_SEC*1000:.0f}ms, "
        f"max_latency={TARGET_MATCH_MAX_LATENCY_SEC*1000:.0f}ms; "
        "unmatched snapshots fail closed."
    )
    print(
        f"operational anchor: last matched Stage1 row per segment with "
        f"HT-event_t>{ANCHOR_MIN_DURATION_SEC:.2f}s."
    )
    print(
        f"coverage: runs={len(runs)} segments={total_segments} "
        f"Stage1 snapshots={total_snapshots} valid={len(all_rows)} "
        f"anchors={len(anchor_rows)} skips={dict(sorted(skips.items()))}"
    )
    print("runs:")
    for context in contexts:
        print(
            f"  {context.name}  git={context.git_version}  "
            f"max_lat={context.max_lateral_accel:.3f}m/s^2"
        )

    print_metric_block("ALL VALID STAGE1 SNAPSHOTS", all_rows)
    print_metric_block("OPERATIONAL ANCHORS (one per S1 segment)", anchor_rows)

    pose_correction = metrics(anchor_rows, "pose_correction")
    print(
        "\nDIRECT POSE minus velocity-integrated truth on anchors: "
        f"n={pose_correction['n']} "
        f"xMAE={pose_correction.get('xmae', float('nan')):.2f}cm "
        f"yMAE={pose_correction.get('ymae', float('nan')):.2f}cm "
        f"2DRMSE={pose_correction.get('e2rmse', float('nan')):.2f}cm. "
        "This is a pose-correction integrity diagnostic, not predictor error."
    )

    print("\nTAU SWEEP ON ALL 20 RUNS (descriptive only; objective=anchor new 2D RMSE)")
    print("tau(s)   n   xMAE  xRMSE  2DMAE  2DRMSE  |x|>5 |x|>7")
    sweep_scores = {}
    for tau in tau_grid:
        summary = metrics(evaluations[tau].anchor_rows, "new")
        sweep_scores[tau] = float(summary["e2rmse"])
        print(
            f"{tau:6.3f} {summary['n']:4d} {summary['xmae']:7.2f} "
            f"{summary['xrmse']:7.2f} {summary['e2mae']:7.2f} "
            f"{summary['e2rmse']:8.2f} {summary['x5']:6d} {summary['x7']:6d}"
        )
    global_best_tau = min(tau_grid, key=lambda tau: sweep_scores[tau])
    fixed_010_score = sweep_scores[0.10]
    print(
        f"all-run descriptive best tau={global_best_tau:.3f}s "
        f"2DRMSE={sweep_scores[global_best_tau]:.2f}cm; "
        f"fixed 0.100s={fixed_010_score:.2f}cm."
    )
    if global_best_tau != 0.10:
        print("NOTICE: fixed tau=0.100s is not the minimum of the requested grid.")

    print("\nLEAVE-ONE-RUN-OUT TAU SELECTION")
    print("held-out run               selected_tau  train_2DRMSE  test_n  test_2DRMSE")
    oof_rows: list[ReplayRow] = []
    chosen_taus: Counter[float] = Counter()
    for held_out in contexts:
        train_scores = {}
        for tau in tau_grid:
            train_rows = [
                row
                for context in contexts
                if context.name != held_out.name
                for row in evaluations[tau].per_run[context.name]
            ]
            train_scores[tau] = float(metrics(train_rows, "new")["e2rmse"])
        selected_tau = min(tau_grid, key=lambda tau: train_scores[tau])
        test_rows = evaluations[selected_tau].per_run[held_out.name]
        test_summary = metrics(test_rows, "new")
        oof_rows.extend(test_rows)
        chosen_taus[selected_tau] += 1
        print(
            f"{held_out.name:<27} {selected_tau:10.3f} "
            f"{train_scores[selected_tau]:13.2f} {test_summary['n']:7d} "
            f"{test_summary.get('e2rmse', float('nan')):12.2f}"
        )
    print(f"LOO selected tau counts: {dict(sorted(chosen_taus.items()))}")
    print_metric_block("LEAVE-ONE-RUN-OUT OOF OPERATIONAL ANCHORS", oof_rows)

    print("\nPER-RUN OPERATIONAL ANCHORS (cm)")
    print("run                       n   logged xMAE/2DRMSE   old xMAE/2DRMSE   new xMAE/2DRMSE")
    for context in contexts:
        rows = fixed.per_run[context.name]
        logged = metrics(rows, "logged")
        old = metrics(rows, "old")
        new = metrics(rows, "new")
        print(
            f"{context.name:<26} {logged['n']:3d}   "
            f"{logged.get('xmae', float('nan')):6.2f}/{logged.get('e2rmse', float('nan')):6.2f}   "
            f"{old.get('xmae', float('nan')):6.2f}/{old.get('e2rmse', float('nan')):6.2f}   "
            f"{new.get('xmae', float('nan')):6.2f}/{new.get('e2rmse', float('nan')):6.2f}"
        )

    subsets = (
        ("|v_perp0| < 0.10m/s", lambda r: abs(r.init_vperp) < 0.10),
        (
            "0.10 <= |v_perp0| < 0.20m/s",
            lambda r: 0.10 <= abs(r.init_vperp) < 0.20,
        ),
        ("|v_perp0| >= 0.20m/s", lambda r: abs(r.init_vperp) >= 0.20),
        ("target_x jump < 0.05m", lambda r: r.target_x_jump < 0.05),
        (
            "0.05 <= target_x jump < 0.15m",
            lambda r: 0.05 <= r.target_x_jump < 0.15,
        ),
        ("target_x jump >= 0.15m", lambda r: r.target_x_jump >= 0.15),
        (
            "HIGH-RISK: |v_perp0|>=0.15m/s and target_x jump>=0.15m",
            lambda r: abs(r.init_vperp) >= 0.15 and r.target_x_jump >= 0.15,
        ),
    )
    for title, predicate in subsets:
        print_metric_block(f"ANCHOR SUBSET: {title}", [r for r in anchor_rows if predicate(r)])

    focus_run = "tracker_20260826_134922"
    print(f"\nFOCUS {focus_run} segments #10/#25")
    print(
        "shot kind      event_t horizon vperp0 targetJump   logged(ex,ey)   "
        "old(ex,ey)      new(ex,ey) cm"
    )
    for shot in (10, 25):
        segment_rows = [r for r in all_rows if r.run == focus_run and r.segment == shot]
        if not segment_rows:
            continue
        anchor = next(
            (r for r in anchor_rows if r.run == focus_run and r.segment == shot), None
        )
        worst = max(segment_rows, key=lambda r: abs(r.logged_ex))
        for kind, row in (("anchor", anchor), ("worst-log", worst)):
            if row is None:
                continue
            print(
                f"{shot:4d} {kind:<9} {row.event_t:8.3f} {row.horizon:7.3f} "
                f"{row.init_vperp:+7.3f} {row.target_x_jump:10.3f}   "
                f"({row.logged_ex*100:+6.2f},{row.logged_ey*100:+6.2f}) "
                f"({row.old_ex*100:+6.2f},{row.old_ey*100:+6.2f}) "
                f"({row.new_ex*100:+6.2f},{row.new_ey*100:+6.2f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
