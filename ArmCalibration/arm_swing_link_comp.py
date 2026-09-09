#!/usr/bin/env python3
"""Arm-only swing driver: synthetic /predict_hit_pos targets while the car stands still.

Why: to verify the J1 link (spring-twist) compensation of arm_controller_cpp against the four-camera
black marker without a ball.  The tracker records the video + rosbag as usual; afterwards
    python test_src/extract_arm_bag.py ...          (arm json, now with /tennis/link_command)
    python test_src/arm_swing_black_marker.py --session <s>
    python test_src/analyze_arm_swing_marker.py --session <s>
Run one session per configuration and compare: rule / RL  ×  --link-comp / --no-link-comp.

RK side (operator):  bash scripts/run_arm_cpp_ready.sh --car v04 [--rl] [--link-comp|--no-link-comp]
                     (bot_center should be running too: the tracker's PC/RK clock bridge listens to /bot_state)
PC side (operator):  .\\run_tracker.ps1  (recording), then this script.

⚠ This script makes the arm swing and requires the user's explicit authorization.
   It refuses to start unless the ROS graph is the expected one (controller alone on /joint_states and
   /tennis/status, no web-panel stream, no /predict_hit_pos publisher other than this script / an idle
   bot_center), waits for the arm to be still before every target, and asks once before the first swing.

Each target is fed the way bot_center does: a 30 Hz stream of stage-1 messages with a fixed ht
(first message `--lead` s before ht, last one 50 ms before), n_points growing from 6.  The rule path
accepts on the first message (duration > 0.25 s), the RL path starts its READY→swing episode from it.
Targets default to the FinalHT (rel_x, rel_z) set of a real session (`--from-session`), speeds to the
controller default (omit `speed` → kSwingSpeedDefault); widen with `--speeds 3,4,5.5,7` for a torque sweep.

`--mode wiggle` (rule controller only, refused when rl_swing_mode != off): no swing at all — stage-0 messages with
car_yaw = ±θ and n_points = 0 make the READY repark move J1 by ±θ (quintic, ≤1 rad/s, ≤12 rad/s², ≥0.3 s; the
controller clips |car_yaw| ≤ 0.35 rad).  That is a clean low-torque J1 reversal (≈5~10 N·m) — the most sensitive
probe for backlash vs spring: a gap shows up as a ±g step at every reversal, a spring as κ·τ that is tiny here.
Each amplitude is repeated `--wiggle-cycles` times; the black-marker pipeline picks these up as windows too.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "arm_controller_data" / "arm_swing_link_comp"

NODE_NAME = "arm_swing_link_comp"
HIT_TOPIC = "/predict_hit_pos"
STATUS_TOPIC = "/tennis/status"
JOINT_STATE_TOPIC = "/joint_states"
PANEL_STREAM_TOPIC = "/tennis/panel_stream"
ARM_COMMAND_TOPIC = "/tennis/arm_command"

STREAM_HZ = 30.0
STREAM_STOP_BEFORE_HT_S = 0.05
N_POINTS_START = 6
STILL_V_RAD_S = 0.5           # Lingzu joints report ~0.2 rad/s velocity noise at rest; stillness = position span
STILL_SPAN_RAD = 0.01         # max joint position span over STILL_WINDOW_S (0.57 deg)
STILL_WINDOW_S = 0.6
JOINT_MAX_AGE_S = 0.5
STATUS_WATCH_S = 1.2
FAILED_SWING_STATUS = ("sweep degrade", "sweep infeasible", "sweep hold", "adjust partial",
                       "feedback hold", "aborted", "error")
WIGGLE_V = 1.0            # cfg::kReadyYawReparkV
WIGGLE_MIN_T = 0.3        # cfg::kReadyYawReparkMinT
WIGGLE_YAW_MAX = 0.35     # cfg::kReadyYawMaxAbs


class ExperimentError(RuntimeError):
    pass


def swing_status_failure(statuses: list[str]) -> str | None:
    """Stop future publishing on failure; this does not cancel an already accepted swing."""
    return next((s for s in statuses if any(k in s.lower() for k in FAILED_SWING_STATUS)), None)


def rl_swing_result(statuses: list[str], requested_ht: float) -> dict:
    """RL policy-end report is not acceptance, successful motion, or return-to-ready."""
    accepted = [s for s in statuses if s.startswith("accepted hit ") and re.search(r"\brl=1\b", s)]
    completed = [s for s in statuses if s.startswith("rl_swing done mode=active ")]
    final = re.search(r"\bht=([-+\d.eE]+)", completed[-1]) if completed else None
    final_ht = float(final.group(1)) if final else None
    error = None
    if not accepted:
        error = "no active RL accepted hit"
    elif not completed:
        error = "active RL policy-end report missing"
    elif final_ht is None or not math.isfinite(final_ht) or abs(final_ht - requested_ht) > 2e-6:
        error = "active RL final HT differs from fixed requested HT"
    for status in accepted:
        target = re.search(r"\bht=([-+\d.eE]+)", status)
        if not target or not math.isfinite(float(target.group(1))) or abs(float(target.group(1)) - requested_ht) > 2e-6:
            error = "active RL accepted HT differs from fixed requested HT"
    return {"accepted": bool(accepted), "completion_observed": bool(completed),
            "final_ht_rk": final_ht, "contract_valid": error is None, "contract_error": error}


def parse_targets(text: str) -> list[tuple[float, float]]:
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        x, z = (float(v) for v in item.split(","))
        out.append((x, z))
    return out


def targets_from_session(session: str, max_targets: int) -> list[tuple[float, float]]:
    """FinalHT (rel_x, rel_z) of a real session's accepted throws (report _tables.json + _rk_tracking.json)."""
    base = PROJECT_ROOT / "tracker_output" / session / session
    tables = json.loads(Path(f"{base}_tables.json").read_text(encoding="utf-8"))
    rk = json.loads(Path(f"{base}_rk_tracking.json").read_text(encoding="utf-8"))
    t0 = float(rk["t0"]); times = [float(t) for t in rk["pred"]["t"]]; pl = rk["pred"]["y"]
    seen: dict[tuple[float, float], int] = {}
    for row in tables["arm_contract"]["rows"]:
        if row.get("accepted") is not True or not isinstance(row.get("finalCtRkAbs"), (int, float)):
            continue
        ct_rel = float(row["finalCtRkAbs"]) - t0
        k = min(range(len(times)), key=lambda i: abs(times[i] - ct_rel))
        if abs(times[k] - ct_rel) > 0.002:
            continue
        key = (round(float(pl["rel_x"][k]), 2), round(float(pl["rel_z"][k]), 2))
        seen[key] = seen.get(key, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])
    return [key for key, _ in ranked[:max_targets]]


def build_plan(targets, speeds, repeats) -> list[dict[str, Any]]:
    plan = []
    for r in range(repeats):
        for x, z in targets:
            for speed in speeds:
                plan.append({"index": len(plan), "repeat": r, "rel_x": x, "rel_z": z, "speed": speed})
    return plan


def payload(rel_x: float, rel_z: float, ct_rk: float, ht_rk: float, n_points: int, speed: float | None) -> dict:
    """Same keys the controller parser (node/messages.hpp parse_hit_pos) reads; rel_src marks the source."""
    p = {"rel_x": rel_x, "rel_y": 0.0, "rel_z": rel_z, "stage": 1, "ct": ct_rk, "ht": ht_rk,
         "duration": ht_rk - ct_rk, "car_yaw": 0.0, "rel_src": NODE_NAME, "hit_yaw_extra": 0.0,
         "n_points": n_points, "spin_rv": None}
    if speed is not None:
        p["speed"] = speed
    return p


def payload_wiggle(rel_x: float, rel_z: float, ct_rk: float, car_yaw: float) -> dict:
    """stage 0 + n_points 0: the rule path only runs update_car_yaw (READY repark of J1 by −car_yaw); no hit is accepted."""
    return {"rel_x": rel_x, "rel_y": 0.0, "rel_z": rel_z, "stage": 0, "ct": ct_rk, "ht": ct_rk + 2.0, "duration": 2.0,
            "car_yaw": car_yaw, "rel_src": NODE_NAME + "_wiggle", "hit_yaw_extra": 0.0, "n_points": 0, "spin_rv": None}


def build_wiggle_plan(target, amplitudes_deg, cycles) -> list[dict[str, Any]]:
    plan = []
    x, z = target
    for amp in amplitudes_deg:
        theta = min(math.radians(abs(amp)), WIGGLE_YAW_MAX)
        seq = []
        for _ in range(cycles):
            seq += [theta, -theta]
        seq.append(0.0)
        prev = 0.0
        for yaw in seq:
            span = abs(yaw - prev)
            plan.append({"index": len(plan), "amp_deg": amp, "rel_x": x, "rel_z": z, "car_yaw": yaw,
                         "move_s": max(WIGGLE_MIN_T, span / WIGGLE_V)})
            prev = yaw
    return plan


def execute(plan: list[dict[str, Any]], args) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch  # noqa: F401  (DLL preload order, same as the tracker / capture script)
    from src.ros2_support import ensure_ros2_environment

    ensure_ros2_environment()
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)

    class Driver(Node):
        def __init__(self) -> None:
            super().__init__(NODE_NAME)
            self._lock = threading.Lock()
            self._joints: deque[dict[str, Any]] = deque(maxlen=4000)
            self._statuses: deque[tuple[float, str]] = deque(maxlen=1000)
            self._clock_samples: list[tuple[float, float]] = []
            self.hit_pub = self.create_publisher(String, HIT_TOPIC, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
            self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint, reliable)
            self.create_subscription(String, STATUS_TOPIC, self._on_status, reliable)
            self._params = AsyncParameterClient(self, "/arm_controller_cpp")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

        def _spin(self) -> None:
            while rclpy.ok() and not self._stop.is_set():
                self._executor.spin_once(timeout_sec=0.05)

        def _on_joint(self, msg: JointState) -> None:
            try:
                v = [float(x) for x in msg.velocity]
                q = [float(x) for x in msg.position]
            except (TypeError, ValueError):
                return
            if len(v) < 6 or len(q) < 6:
                return
            t_recv = time.perf_counter()
            t_rk = int(msg.header.stamp.sec) + int(msg.header.stamp.nanosec) * 1e-9
            with self._lock:
                self._clock_samples.append((t_recv, t_rk))
                self._joints.append({"perf": t_recv, "rk": t_rk,
                                     "vmax": max(abs(x) for x in v[:6]), "q": q[:6]})

        def _on_status(self, msg: String) -> None:
            with self._lock:
                self._statuses.append((time.perf_counter(), str(msg.data).strip()))

        # ── RK clock from the controller's own stamps ──
        def rk_now(self) -> float:
            with self._lock:
                if not self._joints:
                    raise ExperimentError("no /joint_states yet")
                row = self._joints[-1]
            age = time.perf_counter() - row["perf"]
            if age > JOINT_MAX_AGE_S:
                raise ExperimentError(f"/joint_states stale ({age:.2f}s)")
            return row["rk"] + age

        def wait_still(self, timeout_s: float) -> None:
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                with self._lock:
                    rows = [r for r in self._joints if r["perf"] >= now - STILL_WINDOW_S]
                span = max(max(r["q"][j] for r in rows) - min(r["q"][j] for r in rows) for j in range(6)) if rows else 1e9
                if rows and now - rows[-1]["perf"] <= JOINT_MAX_AGE_S and len(rows) >= 6 \
                        and max(r["vmax"] for r in rows) < STILL_V_RAD_S and span < STILL_SPAN_RAD:
                    return
                time.sleep(0.05)
            raise ExperimentError("arm did not come to rest")

        def statuses_since(self, t_perf: float) -> list[str]:
            with self._lock:
                return [s for t, s in self._statuses if t >= t_perf]

        def controller_params(self, timeout_s: float = 3.0) -> dict[str, str]:
            """mode/config_path/rl_swing_mode are always declared; link_compensation only exists on builds that
            carry the link compensation (an undeclared name makes the whole GetParameters reply empty)."""
            if not self._params.wait_for_services(timeout_sec=timeout_s):
                raise ExperimentError("arm_controller_cpp parameter services are unavailable")

            def fetch(names: list[str]) -> list[str] | None:
                future = self._params.get_parameters(names)
                deadline = time.perf_counter() + timeout_s
                while not future.done() and time.perf_counter() < deadline:
                    time.sleep(0.02)
                response = future.result() if future.done() else None
                if response is None or len(response.values) != len(names):
                    return None
                return [v.string_value for v in response.values]

            base = fetch(["mode", "config_path", "rl_swing_mode"])
            if base is None:
                raise ExperimentError("could not read arm_controller_cpp parameters (mode/config_path/rl_swing_mode)")
            out = dict(zip(["mode", "config_path", "rl_swing_mode"], base))
            lc = fetch(["link_compensation"])
            out["link_compensation"] = lc[0] if lc else "absent(build without link compensation)"
            return out

        @staticmethod
        def _endpoint_name(endpoint: Any) -> str:
            namespace = str(endpoint.node_namespace).rstrip("/")
            return f"{namespace}/{endpoint.node_name}" if namespace else f"/{endpoint.node_name}"

        def graph(self) -> dict[str, list[str]]:
            pubs = lambda topic: sorted(self._endpoint_name(e) for e in self.get_publishers_info_by_topic(topic))  # noqa: E731
            return {"hit_publishers": pubs(HIT_TOPIC), "panel_stream_publishers": pubs(PANEL_STREAM_TOPIC),
                    "joint_state_publishers": pubs(JOINT_STATE_TOPIC), "status_publishers": pubs(STATUS_TOPIC),
                    "arm_command_publishers": pubs(ARM_COMMAND_TOPIC)}

        def check_graph(self) -> dict[str, list[str]]:
            g = self.graph()
            others = [p for p in g["hit_publishers"] if p != f"/{NODE_NAME}"]
            foreign = [p for p in others if "bot_center" not in p]
            if foreign:
                raise ExperimentError(f"another {HIT_TOPIC} publisher is alive: {foreign} (close the panel / other scripts)")
            if others:
                print(f"  note: bot_center publishes {HIT_TOPIC} too ({others}); it only speaks when it sees a ball — keep the court empty")
            if g["panel_stream_publishers"]:
                raise ExperimentError(f"web panel stream is alive: {g['panel_stream_publishers']} — close the panel")
            if g["joint_state_publishers"] != ["/arm_controller_cpp"] or g["status_publishers"] != ["/arm_controller_cpp"]:
                raise ExperimentError(f"unexpected controller graph: {g}")
            return g

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Driver()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_ROOT / f"arm_swing_link_comp_{stamp}.json"
    log: dict[str, Any] = {"created": stamp, "args": vars(args), "plan": plan, "triggers": []}
    try:
        time.sleep(1.0)
        graph = node.check_graph()
        node.wait_still(10.0)
        params = node.controller_params()
        log["controller"] = params
        if params.get("mode") != "active" or not params.get("config_path", "").replace("\\", "/").endswith("/config/cars/v04.yaml"):
            raise ExperimentError(f"controller is not the active V04 node: {params}")
        print(f"graph ok: {graph}")
        print(f"controller: link_compensation={params['link_compensation']} rl_swing_mode={params['rl_swing_mode']}")
        if args.mode == "wiggle":
            if params.get("rl_swing_mode", "off") != "off":
                raise ExperimentError("wiggle needs the rule controller (stage-0 targets would start RL prep); restart without --rl")
            print(f"RK now ≈ {node.rk_now():.3f}s; {len(plan)} yaw reparks (wiggle), amplitudes {sorted({p['amp_deg'] for p in plan})} deg")
            if not args.yes:
                input("J1 will repark back and forth on Enter (no swing; Ctrl+C to abort) … ")
            for item in plan:
                node.check_graph()
                node.wait_still(10.0)
                t_send0 = time.perf_counter()
                ct = node.rk_now()
                msg = String()
                msg.data = json.dumps(payload_wiggle(item["rel_x"], item["rel_z"], ct, item["car_yaw"]))
                node.hit_pub.publish(msg)
                time.sleep(item["move_s"] + args.wiggle_hold)
                statuses = node.statuses_since(t_send0)
                interesting = [s for s in statuses if any(k in s for k in ("repark", "reject", "accepted", "error", "hold"))]
                row = {**item, "ct_rk": ct, "t_send0_perf": t_send0, "statuses": interesting}
                log["triggers"].append(row)
                log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
                ok = any("ready repark" in s for s in interesting)
                print(f"[{item['index'] + 1}/{len(plan)}] amp {item['amp_deg']:g}° car_yaw {math.degrees(item['car_yaw']):+.1f}°: "
                      f"{'REPARK' if ok else 'no repark status'}; " + " | ".join(interesting[:2]))
                if not ok and args.strict:
                    raise ExperimentError("repark not acknowledged; stopping (--strict)")
            return log_path
        print(f"RK now ≈ {node.rk_now():.3f}s; {len(plan)} swings, lead {args.lead}s, gap {args.gap}s")
        if not args.yes:
            input("Arm will swing on Enter (Ctrl+C to abort) … ")
        for item in plan:
            node.check_graph()
            node.wait_still(args.gap + 10.0)
            if log["triggers"]:
                previous = log["triggers"][-1]
                previous["statuses"] = node.statuses_since(previous["t_send0_perf"])
                failure = swing_status_failure(previous["statuses"])
                if failure:
                    raise ExperimentError(f"late failure: {failure}; no next target sent (current swing is not cancelled)")
            t_send0 = time.perf_counter()
            ct0 = node.rk_now()
            ht = ct0 + args.lead
            n_sent = 0
            row = {**item, "ct0_rk": ct0, "ht_rk": ht, "n_sent": 0, "t_send0_perf": t_send0, "statuses": []}
            log["triggers"].append(row)
            while True:
                ct = node.rk_now()
                if ct > ht - STREAM_STOP_BEFORE_HT_S:
                    break
                if swing_status_failure(node.statuses_since(t_send0)):
                    break
                msg = String()
                msg.data = json.dumps(payload(item["rel_x"], item["rel_z"], ct, ht, N_POINTS_START + n_sent, item["speed"]))
                node.hit_pub.publish(msg)
                n_sent += 1
                row["n_sent"] = n_sent
                time.sleep(1.0 / STREAM_HZ)
            time.sleep(STATUS_WATCH_S)
            statuses = node.statuses_since(t_send0)
            row["statuses"] = statuses
            log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
            accepted = any("accepted hit " in s for s in statuses)
            if params["rl_swing_mode"] == "active":
                row["rl_result"] = rl_swing_result(statuses, ht)
                accepted = row["rl_result"]["accepted"]
            print(f"[{item['index'] + 1}/{len(plan)}] rel_x {item['rel_x']:.2f} rel_z {item['rel_z']:.2f} speed {item['speed']} "
                  f"ht {ht:.3f}: {'ACCEPTED' if accepted else 'NOT ACCEPTED'}; " + " | ".join(statuses[:3]))
            failure = swing_status_failure(statuses)
            if failure:
                raise ExperimentError(f"{failure}; further publishing stopped (current swing is not cancelled)")
            if not accepted and args.strict:
                raise ExperimentError("target not accepted; stopping (--strict)")
            if params["rl_swing_mode"] == "active" and args.strict and not row["rl_result"]["contract_valid"]:
                raise ExperimentError(row["rl_result"]["contract_error"] + "; stopping (--strict; current swing is not cancelled)")
            time.sleep(max(0.0, args.gap - STATUS_WATCH_S))
            row["statuses"] = node.statuses_since(t_send0)
            failure = swing_status_failure(row["statuses"])
            if failure:
                raise ExperimentError(f"late failure: {failure}; no next target sent (current swing is not cancelled)")
    finally:
        if log["triggers"]:
            previous = log["triggers"][-1]
            previous["statuses"] = node.statuses_since(previous["t_send0_perf"])
        node._stop.set()
        node._thread.join(timeout=1.0)
        with node._lock:
            samples = list(node._clock_samples)
        if samples:
            offsets = sorted(pc - rk for pc, rk in samples)
            offset = statistics.median(offsets)
            log["clock_bridge"] = {
                "source": "/joint_states header -> PC callback perf_counter",
                "pc_minus_rk": offset, "n": len(samples),
                "mad": statistics.median(abs(x - offset) for x in offsets),
                "p05": offsets[int(0.05 * (len(offsets) - 1))],
                "p95": offsets[int(0.95 * (len(offsets) - 1))],
                "note": "Receive bridge includes one-way transport and callback delay; not exposure synchronization.",
                "sample_columns": ["pc_recv_perf", "rk_header"], "samples": samples,
            }
        node.destroy_node()
        rclpy.shutdown()
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="", help='"rel_x,rel_z;rel_x,rel_z" (m, ground-relative rel_z as bot_center publishes)')
    parser.add_argument("--from-session", default="tracker_20260905_213942",
                        help="take the most frequent FinalHT (rel_x, rel_z) pairs of a real session (needs its _tables.json)")
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--speeds", default="", help="comma list of ht racket speeds (m/s); empty = controller default")
    parser.add_argument("--mode", choices=["swing", "wiggle"], default="swing",
                        help="swing = synthetic hit targets; wiggle = low-torque J1 reparks via car_yaw (rule controller only)")
    parser.add_argument("--wiggle-deg", default="5,10,20", help="wiggle amplitudes in degrees (|car_yaw| ≤ 20°)")
    parser.add_argument("--wiggle-cycles", type=int, default=3, help="±θ cycles per amplitude")
    parser.add_argument("--wiggle-hold", type=float, default=0.8, help="extra rest after each repark move (s)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--lead", type=float, default=0.65, help="first message → ht (s); real sessions accept ~0.6 s ahead")
    parser.add_argument("--gap", type=float, default=5.0, help="pause after each ht before the next target (s)")
    parser.add_argument("--strict", action="store_true", help="also stop if no hit was accepted; motion failure always stops publishing")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and one payload, no ROS")
    args = parser.parse_args(argv)
    targets = parse_targets(args.targets) if args.targets else targets_from_session(args.from_session, args.max_targets)
    speeds = [float(v) for v in args.speeds.split(",") if v.strip()] or [None]
    if args.mode == "wiggle":
        amps = [float(v) for v in args.wiggle_deg.split(",") if v.strip()]
        plan = build_wiggle_plan(targets[0], amps, args.wiggle_cycles)
        print(f"wiggle around target {targets[0]}; amplitudes {amps} deg × {args.wiggle_cycles} cycles → {len(plan)} reparks, "
              f"~{sum(p['move_s'] + args.wiggle_hold for p in plan):.0f} s")
        if args.dry_run:
            print(json.dumps(payload_wiggle(targets[0][0], targets[0][1], 1000.0, plan[0]["car_yaw"]), indent=1))
            return 0
    else:
        plan = build_plan(targets, speeds, args.repeats)
        print(f"targets {targets}; speeds {speeds}; {len(plan)} swings")
        if args.dry_run:
            print(json.dumps(payload(targets[0][0], targets[0][1], 1000.0, 1000.0 + args.lead, N_POINTS_START, speeds[0]), indent=1))
            return 0
    try:
        log_path = execute(plan, args)
    except ExperimentError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
