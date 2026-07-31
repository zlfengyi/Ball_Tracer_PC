# -*- coding: utf-8 -*-
"""Extract RK ball-tracking/move topics from tracker rosbag for HTML report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MOVE_TOPICS = (
    "/ball_loc_topic",
    "/ball_world_topic",
    "/predict_hit_pos",
    "/estimate_loc_topic",
    "/bot_state",
    "/chassis_can/camera_motor",
    "/chassis_can/camera_cmd",
    "/chassis_can/imu",
    "/chassis_can/steer_motor",
    "/chassis_can/steer_cmd",
    "/chassis_can/wheels_cmd",
    "/chassis_can/wheels_pos_diff",
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _topic_key(topic: str) -> str:
    return topic[1:] if topic.startswith("/") else topic


def _payload_time(topic: str, payload: dict) -> float | None:
    if topic == "/predict_hit_pos":
        value = payload.get("ct")
    else:
        value = payload.get("t")
    return float(value) if _finite(value) else None


def _new_series() -> dict:
    return {"t": [], "y": {}}


def _add(series: dict, t: float, **values) -> None:
    series["t"].append(round(float(t), 6))
    for key, value in values.items():
        series["y"].setdefault(key, []).append(value)


def _append_xyz(target: dict, t: float, payload: dict, *, stage=None) -> None:
    x, y, z = payload.get("x"), payload.get("y"), payload.get("z")
    if not (_finite(x) and _finite(y) and _finite(z)):
        return
    target["t"].append(round(float(t), 6))
    target["x"].append(float(x))
    target["y"].append(float(y))
    target["z"].append(float(z))
    if stage is not None:
        target.setdefault("stage", []).append(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag directory")
    parser.add_argument("--output", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    import rosbag2_py  # noqa: E402
    from rclpy.serialization import deserialize_message  # noqa: E402
    from std_msgs.msg import String  # noqa: E402

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    rows: list[tuple[str, int, dict]] = []
    counts: dict[str, int] = {}
    t0_candidates: list[float] = []

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if topic not in MOVE_TOPICS:
            continue
        counts[_topic_key(topic)] = counts.get(_topic_key(topic), 0) + 1
        try:
            payload = json.loads(deserialize_message(data, String).data)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows.append((topic, int(stamp_ns), payload))
        payload_t = _payload_time(topic, payload)
        if payload_t is not None:
            t0_candidates.append(payload_t)

    if not t0_candidates:
        raise RuntimeError("no RK payload time field t/ct found in bag")
    t0 = min(t0_candidates)

    ball_timing_by_shot_t: dict[float, tuple[float, float]] = {}
    for topic, _stamp_ns, payload in rows:
        if topic != "/ball_loc_topic":
            continue
        shot_t = _payload_time(topic, payload)
        result_t = payload.get("result_t")
        if shot_t is not None and _finite(result_t):
            result_t = float(result_t)
            ball_timing_by_shot_t[round(shot_t, 6)] = (
                result_t,
                (result_t - shot_t) * 1000.0,
            )

    ball = _new_series()
    world = _new_series()
    pred = _new_series()
    bot = _new_series()
    camera_cmd = _new_series()
    camera_motor = _new_series()
    steer_cmd = _new_series()
    steer_motor = _new_series()
    wheels_cmd = _new_series()
    wheels_pos_diff = _new_series()
    imu = _new_series()
    estimate = _new_series()
    xy_ball = {"t": [], "x": [], "y": [], "z": []}
    xy_world = {"t": [], "x": [], "y": [], "z": []}
    xy_pred = {"t": [], "x": [], "y": [], "z": [], "stage": []}

    for topic, _stamp_ns, payload in rows:
        payload_t = _payload_time(topic, payload)
        if payload_t is None:
            continue
        t = payload_t - t0
        key = _topic_key(topic)

        if key == "ball_loc_topic":
            result_t, latency_ms = ball_timing_by_shot_t.get(
                round(payload_t, 6), (None, None)
            )
            _add(
                ball,
                t,
                x=payload.get("x"),
                y=payload.get("y"),
                z=payload.get("z"),
                result_t=result_t,
                latency_ms=latency_ms,
            )
            _append_xyz(xy_ball, t, payload)
        elif key == "ball_world_topic":
            result_t, latency_ms = ball_timing_by_shot_t.get(
                round(payload_t, 6), (None, None)
            )
            _add(
                world,
                t,
                x=payload.get("x"),
                y=payload.get("y"),
                z=payload.get("z"),
                camera_yaw=payload.get("camera_yaw"),
                bot_x=payload.get("bot_x"),
                bot_y=payload.get("bot_y"),
                bot_yaw=payload.get("bot_yaw"),
                result_t=result_t,
                latency_ms=latency_ms,
            )
            _append_xyz(xy_world, t, payload)
        elif key == "predict_hit_pos":
            ht = payload.get("ht")
            duration = (ht - payload_t) if _finite(ht) else payload.get("duration")
            stage = payload.get("stage")
            _add(
                pred,
                t,
                x=payload.get("x"),
                y=payload.get("y"),
                z=payload.get("z"),
                stage=stage,
                duration=duration,
                ht_rel=(ht - t0) if _finite(ht) else None,
                rel_x=payload.get("rel_x"),
                rel_y=payload.get("rel_y"),
                rel_z=payload.get("rel_z"),
                n_bounce_fit=payload.get("n_bounce_fit"),
            )
            _append_xyz(xy_pred, t, payload, stage=stage)
        elif key == "estimate_loc_topic":
            loc = payload.get("ball_loc") if isinstance(payload.get("ball_loc"), list) else []
            _add(
                estimate,
                t,
                x=loc[0] if len(loc) > 0 else payload.get("x"),
                y=loc[1] if len(loc) > 1 else payload.get("y"),
                z=loc[2] if len(loc) > 2 else payload.get("z"),
            )
        elif key == "bot_state":
            target_active = bool(payload.get("target_active"))
            _add(
                bot,
                t,
                x=payload.get("x"),
                y=payload.get("y"),
                yaw=payload.get("yaw"),
                vx=payload.get("vx"),
                vy=payload.get("vy"),
                imu_t=(payload.get("imu_t") - t0) if _finite(payload.get("imu_t")) else None,
                phase=payload.get("phase"),
                steer_angle=payload.get("steer_angle"),
                remaining=payload.get("remaining") if target_active else None,
                v_target=payload.get("v_target") if target_active else None,
                target_x=payload.get("target_x") if target_active else None,
                target_y=payload.get("target_y") if target_active else None,
                target_active=1 if target_active else 0,
            )
        elif key == "chassis_can/camera_cmd":
            if payload.get("cmd") == "mit":
                _add(
                    camera_cmd,
                    t,
                    position=payload.get("position"),
                    velocity=payload.get("velocity"),
                    torque_ff=payload.get("torque_ff"),
                )
        elif key == "chassis_can/camera_motor":
            _add(
                camera_motor,
                t,
                position=payload.get("position"),
                velocity=payload.get("velocity"),
                torque=payload.get("torque"),
                enabled=1 if payload.get("enabled") else 0,
            )
        elif key == "chassis_can/steer_cmd":
            if payload.get("cmd") == "mit":
                _add(
                    steer_cmd,
                    t,
                    position=payload.get("position"),
                    velocity=payload.get("velocity"),
                    torque_ff=payload.get("torque_ff"),
                )
        elif key == "chassis_can/steer_motor":
            _add(
                steer_motor,
                t,
                position=payload.get("position"),
                velocity=payload.get("velocity"),
                torque=payload.get("torque"),
                enabled=1 if payload.get("enabled") else 0,
            )
        elif key == "chassis_can/wheels_cmd":
            currents = payload.get("current")
            speeds = payload.get("speed")
            if not isinstance(currents, list):
                currents = []
            if not isinstance(speeds, list):
                speeds = []
            _add(
                wheels_cmd,
                t,
                current_avg=(
                    sum(float(v) for v in currents if _finite(v)) / len(currents)
                    if currents else None
                ),
                speed_avg=(
                    sum(float(v) for v in speeds if _finite(v)) / len(speeds)
                    if speeds else None
                ),
            )
        elif key == "chassis_can/wheels_pos_diff":
            values = payload.get("pos_diff") or payload.get("position_diff")
            if not isinstance(values, list):
                values = []
            _add(
                wheels_pos_diff,
                t,
                value_avg=(
                    sum(float(v) for v in values if _finite(v)) / len(values)
                    if values else None
                ),
            )
        elif key == "chassis_can/imu":
            _add(
                imu,
                t,
                yaw_speed=payload.get("yaw_speed"),
                ax=payload.get("ax"),
                ay=payload.get("ay"),
            )

    output = {
        "source": "tracker_rosbag_rk_topics",
        "bag_dir": str(args.bag.resolve()),
        "time_axis": "rk_payload_time_relative_s",
        "t0": t0,
        "counts": counts,
        "ball": ball,
        "world": world,
        "pred": pred,
        "estimate": estimate,
        "bot": bot,
        "camera_cmd": camera_cmd,
        "camera_motor": camera_motor,
        "steer_cmd": steer_cmd,
        "steer_motor": steer_motor,
        "wheels_cmd": wheels_cmd,
        "wheels_pos_diff": wheels_pos_diff,
        "imu": imu,
        "xy_ball": xy_ball,
        "xy_world": xy_world,
        "xy_pred": xy_pred,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"RK tracking JSON saved: {args.output}")
    print("topics: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
