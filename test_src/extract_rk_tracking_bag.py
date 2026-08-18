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

# 一次性配置公告（latched，每 topic 1 条，无 payload t）：RK 各节点自报 git 版本 +
# 生效参数实值（2026-08-17 起；老 bag 没有 → config_announce 为空对象）。值 = 输出别名。
CONFIG_TOPICS = {
    "/chassis_can/motor_config": "chassis",
    "/bot_center/config": "bot_center",
    "/tennis/config": "arm",
}


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
    config_announce: dict[str, dict] = {}

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if topic in CONFIG_TOPICS:
            try:
                payload = json.loads(deserialize_message(data, String).data)
            except Exception:
                continue
            if isinstance(payload, dict):
                config_announce[CONFIG_TOPICS[topic]] = payload
            continue
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
                car_pred_x=payload.get("car_pred_x"),
                car_pred_y=payload.get("car_pred_y"),
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
            # 载荷键是 raw（int16 裸 LSB ×4，[FL,BL,FR,BR]）+ cnt（物理帧序号）。
            # 旧名 pos_diff/position_diff 早已废弃，留作读老 bag 的回退——只认旧名会让
            # 这一路恒空（此前 value_avg 全场 None，轮速链离线完全不可见）。
            values = payload.get("raw")
            if not isinstance(values, list):
                values = payload.get("pos_diff") or payload.get("position_diff")
            if not isinstance(values, list):
                values = []
            _add(
                wheels_pos_diff,
                t,
                # 不做 LSB→m 换算：与 RK 侧口径一致，量纲留给消费方（见 bot_state.hpp）。
                raw=[float(v) for v in values] if values else None,
                cnt=payload.get("cnt"),
            )
        elif key == "chassis_can/imu":
            _add(
                imu,
                t,
                yaw_speed=payload.get("yaw_speed"),
                ax=payload.get("ax"),
                ay=payload.get("ay"),
                az=payload.get("az"),
                # cnt/ovfl 是 2026-08-16 起才有的丢帧定位字段（老 bag 为 None）：
                #   cnt 跳 ≥2        → 该消息 publish 之后丢的（DDS/录制），车上无影响
                #   cnt +1 且 ovfl 涨 → socket 接收队列溢出，帧到了内核没读走
                #   cnt +1 且 ovfl 平 → 帧根本没到内核（总线电气 / MCU 没发）
                # 只看 t 的跳变无法区分这三者，别再据此下结论。
                cnt=payload.get("cnt"),
                ovfl=payload.get("ovfl"),
                rate_hz=payload.get("rate_hz"),
            )

    output = {
        "source": "tracker_rosbag_rk_topics",
        "bag_dir": str(args.bag.resolve()),
        "time_axis": "rk_payload_time_relative_s",
        "t0": t0,
        "config_announce": config_announce,
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
