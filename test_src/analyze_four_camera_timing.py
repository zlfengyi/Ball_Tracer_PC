from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import src.mvs_env as _env  # noqa: F401

from CameraParams_header import MVCC_INTVALUE_EX
from src.ball_grabber import SyncCapture, _TICK_FREQ


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def _sample_latch(cam, attempts: int) -> dict:
    best: dict | None = None
    all_samples: list[dict] = []

    for _ in range(attempts):
        perf_before = time.perf_counter()
        t_before = time.perf_counter()
        cam.MV_CC_SetCommandValue("GevTimestampControlLatch")
        t_after = time.perf_counter()
        perf_after = time.perf_counter()

        val = MVCC_INTVALUE_EX()
        ret = cam.MV_CC_GetIntValueEx("GevTimestampValue", val)
        if ret != 0:
            continue

        dev_time_s = int(val.nCurValue) / _TICK_FREQ
        pc_mid_s = (t_before + t_after) / 2.0
        offset_s = pc_mid_s - dev_time_s
        rtt_s = perf_after - perf_before

        sample = {
            "pc_mid_s": pc_mid_s,
            "dev_time_s": dev_time_s,
            "offset_s": offset_s,
            "offset_ms": offset_s * 1000.0,
            "rtt_s": rtt_s,
            "rtt_ms": rtt_s * 1000.0,
            "uncertainty_ms": rtt_s * 500.0,
        }
        all_samples.append(sample)
        if best is None or sample["rtt_s"] < best["rtt_s"]:
            best = sample

    if best is None:
        raise RuntimeError("Failed to read GevTimestampValue during latch probe.")

    return {
        "best": best,
        "all": all_samples,
    }


def _probe_offsets(cap: SyncCapture, serials: list[str], rounds: int, attempts: int) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {sn: [] for sn in serials}
    for round_idx in range(rounds):
        for sn in serials:
            sample = _sample_latch(cap._cameras[sn], attempts)["best"]
            entry = dict(sample)
            entry["round_idx"] = round_idx
            results[sn].append(entry)
    return results


def _summarize_probe(results: dict[str, list[dict]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for sn, rows in results.items():
        offsets_ms = [row["offset_ms"] for row in rows]
        rtt_ms = [row["rtt_ms"] for row in rows]
        uncertainty_ms = [row["uncertainty_ms"] for row in rows]
        drift_ms = offsets_ms[-1] - offsets_ms[0] if len(offsets_ms) >= 2 else 0.0
        summary[sn] = {
            "offset_ms": _stats(offsets_ms),
            "best_rtt_ms": _stats(rtt_ms),
            "midpoint_uncertainty_ms": _stats(uncertainty_ms),
            "offset_drift_ms": drift_ms,
        }
    return summary


def _frame_offset_ms(frame) -> float:
    return (frame.exposure_start_pc - frame.dev_timestamp / _TICK_FREQ) * 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 4-camera timing, offset, RTT, and master/slave skew.",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Capture duration in seconds.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup time before capture.")
    parser.add_argument("--probe-rounds", type=int, default=40, help="Offset probe rounds before and after capture.")
    parser.add_argument("--probe-attempts", type=int, default=3, help="Latch attempts per probe round; best RTT sample is kept.")
    parser.add_argument(
        "--recalib-every",
        type=int,
        default=10,
        help="Pass-through recalibration interval for SyncCapture; 0 disables background recalibration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "capture_test_output"),
        help="Directory for analysis JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"four_camera_timing_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_cfg = json.loads((root / "src" / "config" / "camera.json").read_text(encoding="utf-8"))
    master_serial = camera_cfg["master_serial"]

    print("=" * 60)
    print("Four Camera Timing Analysis")
    print("=" * 60)

    with SyncCapture.from_config(recalib_every=args.recalib_every) as cap:
        serials = cap.sync_serials
        if master_serial not in serials:
            raise RuntimeError(f"Configured master {master_serial} not found in sync serials {serials}")
        slave_serials = [sn for sn in serials if sn != master_serial]

        print(f"Master: {master_serial}")
        print(f"Slaves: {slave_serials}")
        print(f"Configured FPS: {cap.fps:.1f}")

        print(f"\n[1/4] Warmup {args.warmup:.1f}s")
        time.sleep(args.warmup)

        print(f"[2/4] Probing offset/RTT before capture ({args.probe_rounds} rounds)")
        probe_pre_raw = _probe_offsets(cap, serials, args.probe_rounds, args.probe_attempts)
        probe_pre = _summarize_probe(probe_pre_raw)

        print(f"[3/4] Capturing frame groups for {args.duration:.1f}s")
        captured_groups = 0
        timeout_count = 0
        all_spread_ms: list[float] = []
        slave_spread_ms: list[float] = []
        group_latency_ms: list[float] = []
        frame_interval_ms: list[float] = []
        arrival_latency_ms: dict[str, list[float]] = {sn: [] for sn in serials}
        frame_offset_ms: dict[str, list[float]] = {sn: [] for sn in serials}
        master_delta_ms: dict[str, list[float]] = {sn: [] for sn in slave_serials}
        earliest_counts = {sn: 0 for sn in serials}
        latest_counts = {sn: 0 for sn in serials}
        frame_records: list[dict] = []
        prev_group_exposure_pc: float | None = None

        t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.duration:
            frames = cap.get_frames(timeout_s=1.0)
            if frames is None:
                timeout_count += 1
                continue

            captured_groups += 1
            exposure_pc = {sn: frames[sn].exposure_start_pc for sn in serials}
            arrival_pc = {sn: frames[sn].arrival_perf for sn in serials}
            master_exp = exposure_pc[master_serial]

            ordered = sorted(exposure_pc.items(), key=lambda item: item[1])
            earliest_counts[ordered[0][0]] += 1
            latest_counts[ordered[-1][0]] += 1

            group_exposure_pc = statistics.mean(exposure_pc.values())
            if prev_group_exposure_pc is not None:
                frame_interval_ms.append((group_exposure_pc - prev_group_exposure_pc) * 1000.0)
            prev_group_exposure_pc = group_exposure_pc

            all_spread_ms.append((max(exposure_pc.values()) - min(exposure_pc.values())) * 1000.0)
            if slave_serials:
                slave_times = [exposure_pc[sn] for sn in slave_serials]
                slave_spread_ms.append((max(slave_times) - min(slave_times)) * 1000.0)
            group_latency_ms.append((max(arrival_pc.values()) - group_exposure_pc) * 1000.0)

            deltas_this_frame = {}
            for sn in serials:
                arrival_latency_ms[sn].append((arrival_pc[sn] - exposure_pc[sn]) * 1000.0)
                frame_offset_ms[sn].append(_frame_offset_ms(frames[sn]))
                if sn != master_serial:
                    delta_ms = (exposure_pc[sn] - master_exp) * 1000.0
                    master_delta_ms[sn].append(delta_ms)
                    deltas_this_frame[sn] = delta_ms

            frame_records.append({
                "idx": captured_groups - 1,
                "all_spread_ms": all_spread_ms[-1],
                "slave_spread_ms": slave_spread_ms[-1] if slave_spread_ms else 0.0,
                "group_latency_ms": group_latency_ms[-1],
                "master_deltas_ms": deltas_this_frame,
            })

        print(f"[4/4] Probing offset/RTT after capture ({args.probe_rounds} rounds)")
        probe_post_raw = _probe_offsets(cap, serials, args.probe_rounds, args.probe_attempts)
        probe_post = _summarize_probe(probe_post_raw)

    bias_ms = {master_serial: 0.0}
    for sn in slave_serials:
        values = master_delta_ms[sn]
        bias_ms[sn] = float(np.median(values)) if values else 0.0

    residual_all_spread_ms: list[float] = []
    residual_slave_spread_ms: list[float] = []
    for record in frame_records:
        corrected = [0.0]
        for sn in slave_serials:
            corrected.append(record["master_deltas_ms"][sn] - bias_ms[sn])
        residual_all_spread_ms.append(max(corrected) - min(corrected))
        if slave_serials:
            slave_corrected = [record["master_deltas_ms"][sn] - bias_ms[sn] for sn in slave_serials]
            residual_slave_spread_ms.append(max(slave_corrected) - min(slave_corrected))

    result = {
        "config": {
            "master_serial": master_serial,
            "slave_serials": slave_serials,
            "all_serials": serials,
            "configured_fps": cap.fps if "cap" in locals() else camera_cfg["fps"],
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "probe_rounds": args.probe_rounds,
            "probe_attempts": args.probe_attempts,
            "recalib_every": args.recalib_every,
        },
        "probe_pre": probe_pre,
        "probe_post": probe_post,
        "capture": {
            "captured_groups": captured_groups,
            "timeout_count": timeout_count,
            "all_spread_ms": _stats(all_spread_ms),
            "slave_spread_ms": _stats(slave_spread_ms),
            "residual_all_spread_ms": _stats(residual_all_spread_ms),
            "residual_slave_spread_ms": _stats(residual_slave_spread_ms),
            "group_latency_ms": _stats(group_latency_ms),
            "frame_interval_ms": _stats(frame_interval_ms),
            "frame_interval_fps": (
                1000.0 / _stats(frame_interval_ms)["mean"]
                if frame_interval_ms
                else 0.0
            ),
            "arrival_latency_ms": {
                sn: _stats(values) for sn, values in arrival_latency_ms.items()
            },
            "frame_offset_ms": {
                sn: _stats(values) for sn, values in frame_offset_ms.items()
            },
            "master_delta_ms": {
                sn: _stats(values) for sn, values in master_delta_ms.items()
            },
            "master_delta_bias_ms": bias_ms,
            "earliest_counts": earliest_counts,
            "latest_counts": latest_counts,
        },
    }

    output_path = output_dir / "analysis.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary")
    print(f"  Captured groups: {captured_groups}")
    print(f"  Timeouts: {timeout_count}")
    print(
        "  All-camera spread: "
        f"mean={result['capture']['all_spread_ms']['mean']:.3f}ms "
        f"p95={result['capture']['all_spread_ms']['p95']:.3f}ms"
    )
    print(
        "  Slave-only spread: "
        f"mean={result['capture']['slave_spread_ms']['mean']:.3f}ms "
        f"p95={result['capture']['slave_spread_ms']['p95']:.3f}ms"
    )
    print(
        "  Residual all-camera spread: "
        f"mean={result['capture']['residual_all_spread_ms']['mean']:.3f}ms "
        f"p95={result['capture']['residual_all_spread_ms']['p95']:.3f}ms"
    )
    for sn in slave_serials:
        delta = result["capture"]["master_delta_ms"][sn]
        print(
            f"  {sn} - master: mean={delta['mean']:.3f}ms "
            f"std={delta['std']:.3f}ms p95={delta['p95']:.3f}ms"
        )
    for sn in serials:
        rtt = result["probe_pre"][sn]["best_rtt_ms"]
        off = result["capture"]["frame_offset_ms"][sn]
        print(
            f"  {sn}: latch RTT mean={rtt['mean']:.3f}ms, "
            f"frame-offset std={off['std']:.3f}ms"
        )
    print(f"  Analysis JSON: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
