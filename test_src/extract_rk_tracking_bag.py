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

_TENNIS_BALL_RADIUS_M = 0.033
_GRAVITY_MPS2 = 9.8
_STAGE1_LAMBDA_MIN = 0.02
_STAGE1_LAMBDA_MAX = 0.40


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


def _solve2(m00, m01, m11, r0, r1) -> tuple[float, float]:
    det = m00 * m11 - m01 * m01
    if abs(det) < 1e-18:
        raise ValueError("singular 2x2 bounce fit")
    return (
        (r0 * m11 - m01 * r1) / det,
        (m00 * r1 - m01 * r0) / det,
    )


def _solve3(m00, m01, m02, m11, m12, m22, r0, r1, r2):
    det = (
        m00 * (m11 * m22 - m12 * m12)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02)
    )
    if abs(det) < 1e-24:
        raise ValueError("singular 3x3 bounce fit")
    da = (
        r0 * (m11 * m22 - m12 * m12)
        - m01 * (r1 * m22 - m12 * r2)
        + m02 * (r1 * m12 - m11 * r2)
    )
    db = (
        m00 * (r1 * m22 - m12 * r2)
        - r0 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * r2 - r1 * m02)
    )
    dc = (
        m00 * (m11 * r2 - r1 * m12)
        - m01 * (m01 * r2 - r1 * m02)
        + r0 * (m01 * m12 - m11 * m02)
    )
    return da / det, db / det, dc / det


def _fit_free_curve(obs: list[dict]) -> dict:
    if len(obs) < 3:
        raise ValueError("too few free-curve points")
    t_ref = float(obs[0]["t"])
    s0 = s1 = s2 = s3 = s4 = 0.0
    bx0 = bx1 = by0 = by1 = bz0 = bz1 = bz2 = 0.0
    for point in obs:
        dt = float(point["t"]) - t_ref
        dt2 = dt * dt
        x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
        s0 += 1.0
        s1 += dt
        s2 += dt2
        s3 += dt2 * dt
        s4 += dt2 * dt2
        bx0 += x
        bx1 += dt * x
        by0 += y
        by1 += dt * y
        bz0 += z
        bz1 += dt * z
        bz2 += dt2 * z
    ax, vx = _solve2(s0, s1, s2, bx0, bx1)
    ay, vy = _solve2(s0, s1, s2, by0, by1)
    az, vz, cz = _solve3(s0, s1, s2, s2, s3, s4, bz0, bz1, bz2)
    return {
        "t_ref": t_ref,
        "ax": ax,
        "vx": vx,
        "ay": ay,
        "vy": vy,
        "az": az,
        "vz": vz,
        "cz": cz,
        "lam": 0.0,
        "g": 0.0,
    }


def _fit_coupled_curve(obs: list[dict], lam: float, g_eff: float) -> dict:
    if len(obs) < 3 or lam <= 1e-6 or g_eff <= 0.0:
        raise ValueError("invalid coupled-curve inputs")
    t_ref = float(obs[0]["t"])
    p0 = p1 = p2 = 0.0
    rx0 = rx1 = ry0 = ry1 = rz0 = rz1 = 0.0
    for point in obs:
        dt = float(point["t"]) - t_ref
        phi = -math.expm1(-lam * dt) / lam
        x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
        adjusted_z = z + (g_eff / lam) * dt
        p0 += 1.0
        p1 += phi
        p2 += phi * phi
        rx0 += x
        rx1 += phi * x
        ry0 += y
        ry1 += phi * y
        rz0 += adjusted_z
        rz1 += phi * adjusted_z
    ax, vx = _solve2(p0, p1, p2, rx0, rx1)
    ay, vy = _solve2(p0, p1, p2, ry0, ry1)
    az, wz = _solve2(p0, p1, p2, rz0, rz1)
    return {
        "t_ref": t_ref,
        "ax": ax,
        "vx": vx,
        "ay": ay,
        "vy": vy,
        "az": az,
        "vz": wz - g_eff / lam,
        "cz": 0.0,
        "lam": lam,
        "g": g_eff,
    }


def _curve_velocity(curve: dict, t: float) -> tuple[float, float, float]:
    dt = t - curve["t_ref"]
    lam = curve["lam"]
    if lam > 0.0:
        decay = math.exp(-lam * dt)
        g_over_lam = curve["g"] / lam
        return (
            curve["vx"] * decay,
            curve["vy"] * decay,
            (curve["vz"] + g_over_lam) * decay - g_over_lam,
        )
    return curve["vx"], curve["vy"], curve["vz"] + 2.0 * curve["cz"] * dt


def _curve_z(curve: dict, t: float) -> float:
    dt = t - curve["t_ref"]
    lam = curve["lam"]
    if lam > 0.0:
        phi = -math.expm1(-lam * dt) / lam
        g_over_lam = curve["g"] / lam
        return curve["az"] + (curve["vz"] + g_over_lam) * phi - g_over_lam * dt
    return curve["az"] + curve["vz"] * dt + curve["cz"] * dt * dt


def _fit_stage1_curve(obs: list[dict], config: dict, g_eff: float) -> dict:
    fixed_lam = float(config["stage1_drag_lambda"])
    if fixed_lam > 1e-6:
        return _fit_coupled_curve(obs, fixed_lam, g_eff)
    drag_k = float(config["stage1_drag_k"])
    if drag_k <= 1e-9:
        raise ValueError("stage1 adaptive drag is disabled")
    curve = _fit_free_curve(obs)
    for _ in range(2):
        speed = math.sqrt(sum(v * v for v in _curve_velocity(curve, float(obs[-1]["t"]))))
        lam = min(_STAGE1_LAMBDA_MAX, max(_STAGE1_LAMBDA_MIN, drag_k * speed))
        curve = _fit_coupled_curve(obs, lam, g_eff)
    return curve


def _free_contact_time(curve: dict, contact_z: float) -> float:
    a, b, c = curve["cz"], curve["vz"], curve["az"] - contact_z
    if abs(a) < 1e-12:
        raise ValueError("free curve has no quadratic contact root")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError("free curve misses contact height")
    root = math.sqrt(disc)
    candidates = [
        (-b - root) / (2.0 * a) + curve["t_ref"],
        (-b + root) / (2.0 * a) + curve["t_ref"],
    ]
    valid = [t for t in candidates if t >= curve["t_ref"] - 1e-6]
    if not valid:
        raise ValueError("free contact root precedes fit")
    return max(valid)


def _coupled_contact_time(curve: dict, contact_z: float) -> float:
    lo, hi = curve["t_ref"] - 0.5, curve["t_ref"]
    if not (_curve_z(curve, lo) < contact_z and _curve_z(curve, hi) >= contact_z):
        raise ValueError("coupled curve misses rising contact height")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _curve_z(curve, mid) < contact_z:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _replay_bounce_sample(pre: list[dict], post: list[dict], config: dict, az_eff: float):
    # 这条历史回填只支持本次两场实际使用的 S0 无水平阻力路径；配置变了就不伪造值。
    if len(pre) < 6 or len(post) < 6:
        raise ValueError("too few gated bounce points")
    if abs(float(config["s0_drag_k"])) > 1e-9:
        raise ValueError("S0 drag replay is unsupported")
    g_eff = _GRAVITY_MPS2 + float(config["s1_az_blend"]) * (
        float(az_eff) - _GRAVITY_MPS2
    )
    curve0 = _fit_free_curve(pre)
    curve1 = _fit_stage1_curve(post, config, g_eff)
    contact_z = float(config["ground_z"]) + _TENNIS_BALL_RADIUS_M
    t_pre = _free_contact_time(curve0, contact_z)
    t_post = _coupled_contact_time(curve1, contact_z)
    vx_in, vy_in, vz_in = _curve_velocity(curve0, t_pre)
    vx_out, vy_out, vz_out = _curve_velocity(curve1, t_post)
    vxy_in, vxy_out = math.hypot(vx_in, vy_in), math.hypot(vx_out, vy_out)
    closure_ms = (t_post - t_pre) * 1000.0
    if not (
        vz_in < 0.0
        and vz_out > 0.0
        and vxy_in > 1e-6
        and vxy_out >= 0.0
        and abs(closure_ms) <= 25.0
    ):
        raise ValueError("quantized bounce replay fails physical/closure gates")
    cor, cxy = -vz_out / vz_in, vxy_out / vxy_in
    if not (0.70 <= cor <= 0.95 and 0.35 <= cxy <= 0.90):
        raise ValueError("quantized bounce replay is outside RK coefficient gates")
    return cor, cxy, closure_ms


def _replay_bounce_measurements(
    rows: list[tuple[str, int, dict]], config_announce: dict
) -> dict[int, tuple[float, float, float]]:
    """Return values only on accepted S1 rows.

    Values are recomputed from the quantized bag payloads; they are not direct RK
    BounceSample/history fields.
    """
    bot_config = config_announce.get("bot_center", {}).get("params", {})
    required = (
        "ground_z",
        "s0_drag_k",
        "stage1_drag_k",
        "stage1_drag_lambda",
        "s1_az_blend",
    )
    if not all(_finite(bot_config.get(key)) for key in required):
        return {}

    world: list[dict] = []
    world_index_by_t: dict[float, int] = {}
    last_s0 = None
    last_online_n = None
    replay: dict[int, tuple[float, float, float]] = {}
    for row_index, (topic, _stamp_ns, payload) in enumerate(rows):
        if topic == "/ball_world_topic":
            if all(_finite(payload.get(key)) for key in ("t", "x", "y", "z")):
                world_index_by_t[round(float(payload["t"]), 6)] = len(world)
                world.append(payload)
            continue
        if topic != "/predict_hit_pos":
            continue

        stage = payload.get("stage")
        online_n = payload.get("online_n")
        if stage == 0:
            last_s0 = payload
        if not _finite(online_n):
            continue
        online_n = int(online_n)
        increments_once = last_online_n is not None and online_n == last_online_n + 1
        last_online_n = online_n
        if not (
            increments_once
            and stage == 1
            and payload.get("bounce_rej") == "ok"
            and payload.get("n_bounce_fit") == 6
            and last_s0 is not None
            and _finite(payload.get("ct"))
            and _finite(payload.get("n_points"))
            and _finite(last_s0.get("ct"))
            and _finite(last_s0.get("n_points"))
            and _finite(last_s0.get("az_eff"))
        ):
            continue

        n_post = 6
        n_pre = int(payload["n_points"]) - n_post
        if n_pre != int(last_s0["n_points"]):
            continue
        current_world = world_index_by_t.get(round(float(payload["ct"]), 6))
        last_s0_world = world_index_by_t.get(round(float(last_s0["ct"]), 6))
        if (
            current_world is None
            or last_s0_world is None
            or n_pre < 3
            or last_s0_world - n_pre + 1 < 0
            or current_world - n_post + 1 <= last_s0_world
        ):
            continue
        pre = world[last_s0_world - n_pre + 1 : last_s0_world + 1]
        post = world[current_world - n_post + 1 : current_world + 1]
        try:
            replay[row_index] = _replay_bounce_sample(
                pre, post, bot_config, float(last_s0["az_eff"])
            )
        except (ArithmeticError, OverflowError, ValueError):
            continue
    return replay


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

    bounce_replay_by_row = _replay_bounce_measurements(rows, config_announce)

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

    for row_index, (topic, _stamp_ns, payload) in enumerate(rows):
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
            cor_meas, cxy_meas, closure_ms = bounce_replay_by_row.get(
                row_index, (None, None, None)
            )
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
                rvz=payload.get("rvz"),
                cor_xy_eff=payload.get("cor_xy_eff"),
                cor_eff=payload.get("cor_eff"),
                cor_meas_replay=cor_meas,
                cxy_meas_replay=cxy_meas,
                cor_meas_closure_ms=closure_ms,
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
        "bounce_measurement_replay": {
            "semantics": "recomputed from quantized bag /ball_world_topic; not direct RK BounceSample/history",
            "selection": "first S1 row with online_n +1, bounce_rej=ok, n_bounce_fit=6",
        },
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
