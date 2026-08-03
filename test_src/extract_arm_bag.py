# -*- coding: utf-8 -*-
"""从 tracker rosbag 提取机械臂数据为 {run_id}_arm.json。

必须在 ROS2 环境中运行（经 ros2/run_ros2.bat 启动），依赖 rosbag2_py。
TCP 正解使用本文件内置的 FK，不依赖 tennis-man/arm_controller 源码路径。

输出供 test_src/generate_curve3_html.py 的 Arm tab 使用：
  states   — /joint_states 实际关节位置/速度/力矩 + FK TCP
  commands — /tennis/motor_command 目标（首个轨迹点）+ FK TCP
  events   — status / arm_command / hit_pos / predict_hit_pos 文本事件

时间轴（全项目只有两个时间轴）：
  所有 t 一律为 RK 单调钟（CLOCK_MONOTONIC）绝对秒 —— 与 /predict_hit_pos
  的 ct/ht、rk_tracking 的 payload t 同一个钟。报告端复用同一个 RK→PC
  仿射时间映射，臂数据没有任何独立的桥。

  新固件（damiao 驱动 + arm_controller 单调钟版）：
    /joint_states、/tennis/motor_command 的 header.stamp 就是 RK 单调钟，
    /tennis/status 文本尾缀 " t=<单调秒>"，全部原生直读、零换算。
  旧 bag 兼容（stamp 为 epoch 系统钟、status 无 t= 的场次）：
    t = stamp − median(stamp − recv) + median(bot_state.t − recv)，
    两个中位数里的传输延迟互相抵消，残差 ~ms 级；status 等无时间事件
    退回 recv + median(bot_state.t − recv)。clock_sync 里明示所用模式。

  clock_sync 自检随文件输出：RK 单调钟 vs PC 收钟漂移率（ms/min）、
  joint_states stamp 时钟域、ht 锚点残差（新调度触球≡ht 时 done−ht 中位，
  验证时间链）。漂移率若超 ~2ms/min，单 offset 假设需复查。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np


# Copied from arm_controller.compact_arm_kinematics at a266857.  Keep the
# report extractor self-contained: generating an Arm JSON must not depend on a
# neighbouring tennis-man checkout.
SHORT_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
ROOT_LINK = "/tennis_arm_j5j6_7_6/Geometry/base_link"


def _pose(pos, quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    out = np.eye(4)
    out[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    out[:3, 3] = pos
    return out


_AXIS = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}

# physics:localPos/localRot/axis copied verbatim from the USD PhysicsRevoluteJoints.
JOINTS = (
    {
        "name": "J1_joint",
        "parent": ROOT_LINK,
        "child": ROOT_LINK + "/J1_Link",
        "axis": _AXIS["Y"],
        "local0": _pose((0.0, 0.0, 0.4385), (-0.4999999, -0.5, 0.49999994, 0.5)),
        "local1": _pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    },
    {
        "name": "J2_JOINT",
        "parent": ROOT_LINK + "/J1_Link",
        "child": ROOT_LINK + "/J1_Link/J2_Link",
        "axis": _AXIS["Z"],
        "local0": _pose((5e-05, 0.1719001, -0.07355), (0.0, -1.0, 0.0, 0.0)),
        "local1": _pose((0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0)),
    },
    {
        "name": "J3_joint",
        "parent": ROOT_LINK + "/J1_Link/J2_Link",
        "child": ROOT_LINK + "/J1_Link/J2_Link/J3_Link",
        "axis": _AXIS["Z"],
        "local0": _pose((-0.000125, 0.44997022, 0.02125), (0.0, -1.0, 0.0, 0.0)),
        "local1": _pose((0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0)),
    },
    {
        "name": "J4_JOINT",
        "parent": ROOT_LINK + "/J1_Link/J2_Link/J3_Link",
        "child": ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link",
        "axis": _AXIS["Y"],
        "local0": _pose((0.0, 0.3, 0.08575), (0.70710677, 0.70710677, 0.0, 0.0)),
        "local1": _pose((0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
    },
    {
        "name": "J5_JOINT",
        "parent": ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link",
        "child": ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link/J5_Link",
        "axis": _AXIS["Y"],
        "local0": _pose((0.0, 0.03973, 0.095), (0.4999999, 0.5, -0.49999994, -0.5)),
        "local1": _pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    },
    {
        "name": "J6_JOINT",
        "parent": ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link/J5_Link",
        "child": ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link/J5_Link/J6_Link",
        "axis": _AXIS["Z"],
        "local0": _pose(
            (0.00042069482, 0.047, 0.031753197),
            (1.6081226e-16, 1.6155446e-15, -1.0, -1.6653345e-15),
        ),
        "local1": _pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    },
)

TOOL_AXIS_IN_LINK6 = np.array([0.0, 0.0, 1.0])

# USD base_link -> hit convention: rotate +90 deg about Z so -Y_base becomes +X.
BASE_ROT = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    out = np.eye(4)
    out[:3, :3] = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )
    return out


def _q6(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(q), dtype=float)
    if q.shape == (5,):
        q = np.concatenate((q, [0.0]))
    if q.shape != (6,):
        raise ValueError(f"expected 5 or 6 joint values, got shape {q.shape}")
    return q


def fk(q: Iterable[float], *, tcp_distance: float = 0.62) -> dict[str, np.ndarray]:
    """Exact forward kinematics from the USD physics joints, in hit convention."""
    q = _q6(q)
    link_transforms = {ROOT_LINK: BASE_ROT.copy()}
    joint_frames = {}

    for angle, joint in zip(q, JOINTS):
        joint_t = link_transforms[joint["parent"]] @ joint["local0"]
        child_t = joint_t @ _axis_rotation(joint["axis"], angle) @ np.linalg.inv(joint["local1"])
        joint_frames[joint["name"]] = joint_t
        link_transforms[joint["child"]] = child_t

    link6 = link_transforms[JOINTS[-1]["child"]]
    handle_axis = link6[:3, :3] @ TOOL_AXIS_IN_LINK6
    handle_axis = handle_axis / np.linalg.norm(handle_axis)
    tcp = link6[:3, 3] + tcp_distance * handle_axis

    return {
        "q": q,
        "tcp": tcp,
        "handle_axis": handle_axis,
        "joint_frames": joint_frames,
        "link_transforms": link_transforms,
    }

EVENT_TOPICS = (
    "/tennis/status",
    "/tennis/arm_command",
    "/tennis/hit_pos",
    "/predict_hit_pos",
    "/arm_controller/status",
)

# RK 单调钟参考话题（按优先级）：payload 带发布时刻的单调钟 t，发布延迟接近 0。
# 仅旧 bag（epoch stamp / 无 t= 的 status）需要；新固件全部原生直读。
# /predict_hit_pos 的 ct 是球观测时刻（比发布早一个 RK 管线时延，0716 实测
# ~70ms），只配当保底，用到时在 clock_sync 里明示 biased。
MONO_REF_TOPICS = ("/bot_state", "/chassis_can/imu")

# 单调钟量级上限：RK 开机秒（连续运行数月也 <1e8）；epoch 秒 ~1.7e9。
MONO_MAX_SEC = 1e8

# status 文本尾缀发布时刻（arm_controller 单调钟版追加）："... t=9203.123456"
STATUS_T_RE = re.compile(r"\s+t=([0-9]+\.[0-9]+)$")


def _ordered(values: list[float], names: list[str], joint_names: tuple[str, ...]) -> list[float | None]:
    """按 joint_names 顺序重排（与 session_viewer._ordered 一致）。"""
    by_name = {name: idx for idx, name in enumerate(names)}
    ordered: list[float | None] = []
    for idx, name in enumerate(joint_names):
        src = by_name.get(name, idx if not names else None)
        ordered.append(float(values[src]) if src is not None and src < len(values) else None)
    return ordered


def _round_list(values: list[float | None], digits: int) -> list[float | None]:
    return [None if v is None else round(v, digits) for v in values]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _thirds_drift_ms_per_min(samples: list[tuple[float, float]]) -> dict | None:
    """(recv_s, diff_s) 样本按首/末三分之一中位差估漂移率。中位数抗离群，O(n)。"""
    if len(samples) < 30:
        return None
    samples = sorted(samples)
    span = samples[-1][0] - samples[0][0]
    if span < 10.0:
        return None
    third = span / 3.0
    lo = [d for t, d in samples if t <= samples[0][0] + third]
    hi = [d for t, d in samples if t >= samples[-1][0] - third]
    t_lo = statistics.median([t for t, _ in samples if t <= samples[0][0] + third])
    t_hi = statistics.median([t for t, _ in samples if t >= samples[-1][0] - third])
    if t_hi - t_lo < 1.0:
        return None
    rate = (statistics.median(hi) - statistics.median(lo)) / (t_hi - t_lo)
    return {
        "ms_per_min": round(rate * 60_000, 3),
        "span_s": round(span, 1),
        "n": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag 目录（含 metadata.yaml）")
    parser.add_argument("--output", type=Path, required=True, help="输出 arm JSON 路径")
    args = parser.parse_args()

    import rosbag2_py  # noqa: E402
    from rclpy.serialization import deserialize_message  # noqa: E402
    from rosidl_runtime_py.utilities import get_message  # noqa: E402

    joint_names = tuple(SHORT_JOINT_NAMES)

    def tcp_of(positions: list[float | None]) -> list[float] | None:
        if any(v is None for v in positions):
            return None
        try:
            return [round(float(v), 4) for v in fk(positions)["tcp"]]
        except Exception:
            return None

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    type_by_topic = {item.name: item.type for item in reader.get_all_topics_and_types()}
    msg_types = {}
    for topic, type_name in type_by_topic.items():
        try:
            msg_types[topic] = get_message(type_name)
        except Exception:
            pass  # 无法解析的类型只计数，不采样

    states: list[dict] = []
    commands: list[dict] = []
    events: list[dict] = []
    state_diffs: list[tuple[float, float]] = []    # (recv_s, stamp − recv) 全部有效 stamp
    command_diffs: list[tuple[float, float]] = []
    mono_diffs: dict[str, list[tuple[float, float]]] = {t: [] for t in MONO_REF_TOPICS}
    predict_ct_diffs: list[tuple[float, float]] = []
    counts: dict[str, int] = {}
    seen_state_names: list[str] = []
    seen_command_names: list[str] = []
    start_ns: int | None = None
    end_ns: int | None = None

    def _header_stamp_sec(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        if start_ns is None:
            start_ns = timestamp
        end_ns = timestamp
        msg_type = msg_types.get(topic)
        if msg_type is None:
            continue
        recv = timestamp / 1e9

        if topic == "/joint_states":
            msg = deserialize_message(data, msg_type)
            names = list(msg.name)
            if not seen_state_names and names:
                seen_state_names = names
            positions = _ordered(list(msg.position), names, joint_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 0.0:
                state_diffs.append((recv, stamp - recv))
            states.append(
                {
                    "stamp": stamp,
                    "recv": recv,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(msg.velocity), names, joint_names), 5),
                    "effort": _round_list(_ordered(list(msg.effort), names, joint_names), 5),
                    "tcp": tcp_of(positions),
                }
            )
        elif topic == "/tennis/motor_command":
            msg = deserialize_message(data, msg_type)
            if not msg.points:
                continue
            point = msg.points[0]
            names = list(msg.joint_names)
            if not seen_command_names and names:
                seen_command_names = names
            positions = _ordered(list(point.positions), names, joint_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 0.0:
                command_diffs.append((recv, stamp - recv))
            commands.append(
                {
                    "stamp": stamp,
                    "recv": recv,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(point.velocities), names, joint_names), 5),
                    "effort": _round_list(_ordered(list(point.effort), names, joint_names), 5),
                    "tcp": tcp_of(positions),
                }
            )
        elif topic in EVENT_TOPICS or topic in MONO_REF_TOPICS:
            msg = deserialize_message(data, msg_type)
            raw = msg.data if hasattr(msg, "data") else None
            if topic in MONO_REF_TOPICS:
                try:
                    payload = json.loads(raw)
                    t_mono = payload.get("t")
                    if isinstance(t_mono, (int, float)):
                        mono_diffs[topic].append((recv, float(t_mono) - recv))
                except Exception:
                    pass
                continue
            if raw is not None:
                text = (
                    " ".join(f"{float(v):.4g}" for v in raw)
                    if isinstance(raw, (list, tuple)) or type(raw).__name__ == "array"
                    else str(raw)
                )
            else:
                text = str(msg)
            event = {"recv": recv, "topic": topic, "text": text[:500], "t_payload": None}
            if topic == "/predict_hit_pos":
                # payload 自带 ct（RK 单调钟，球观测时刻）——事件直接用它，
                # 与 rk_tracking 的 pred 序列同源同值。
                try:
                    ct = json.loads(text).get("ct")
                    if isinstance(ct, (int, float)):
                        predict_ct_diffs.append((recv, float(ct) - recv))
                        event["t_payload"] = float(ct)
                except Exception:
                    pass
            else:
                # 新固件 status 尾缀 " t=<单调秒>" = 发布时刻，解析后从文本剥离
                m = STATUS_T_RE.search(event["text"])
                if m:
                    event["t_payload"] = float(m.group(1))
                    event["text"] = event["text"][: m.start()]
            events.append(event)

    if start_ns is None:
        raise RuntimeError(f"bag has no messages: {args.bag}")

    # ---- stamp 时钟域判定（按话题多数）----
    def _stamp_domain(rows: list[dict]) -> str | None:
        vals = [row["stamp"] for row in rows if row["stamp"] > 0.0]
        if not vals:
            return None
        mono_n = sum(1 for v in vals if v < MONO_MAX_SEC)
        return "rk_mono_native" if mono_n * 2 >= len(vals) else "rk_sys_converted_legacy"

    state_domain = _stamp_domain(states)
    command_domain = _stamp_domain(commands)
    # legacy = 旧固件的 epoch stamp；recv 映射 = 无自带时间的事件（PC 发的
    # hit_pos/arm_command、旧固件 status）。后者在新 bag 里也存在，不算 legacy。
    legacy_stamps = "rk_sys_converted_legacy" in (state_domain, command_domain)
    needs_c_mono = legacy_stamps or any(e["t_payload"] is None for e in events)

    # ---- RK 单调钟参考 C_mono = median(payload t − recv) ----
    mono_ref_topic = None
    mono_ref_biased = False
    c_mono = None
    if needs_c_mono:
        for topic in MONO_REF_TOPICS:
            if len(mono_diffs[topic]) >= 30:
                mono_ref_topic = topic
                c_mono = _median([d for _, d in mono_diffs[topic]])
                break
        if c_mono is None and len(predict_ct_diffs) >= 10:
            # 保底：ct 是观测时刻，比发布早一个 RK 管线时延 → 相关事件整体偏早同量
            mono_ref_topic = "/predict_hit_pos(ct, biased)"
            mono_ref_biased = True
            c_mono = _median([d for _, d in predict_ct_diffs])
        if c_mono is None:
            raise RuntimeError(
                "no RK mono reference in bag (/bot_state, /chassis_can/imu or "
                "/predict_hit_pos payload) — cannot map stamp-less/legacy data onto the RK axis"
            )

    c_sys_js = _median([d for _, d in state_diffs])
    c_sys_mc = _median([d for _, d in command_diffs])

    # ---- 统一落到 RK 单调钟绝对秒 ----
    def _finish(rows: list[dict], c_sys: float | None) -> None:
        for row in rows:
            stamp = row.pop("stamp")
            recv = row.pop("recv")
            if 0.0 < stamp < MONO_MAX_SEC:
                row["t"] = round(stamp, 5)                    # 新固件：原生单调钟直读
            elif stamp >= MONO_MAX_SEC and c_sys is not None and c_mono is not None:
                row["t"] = round(stamp - c_sys + c_mono, 5)   # 旧 bag：epoch → 单调钟
            elif c_mono is not None:
                row["t"] = round(recv + c_mono, 5)            # 缺 stamp：接收时刻映射
            else:
                row["t"] = round(recv, 5)                     # 不可达（needs_legacy 已保证 c_mono）

    _finish(states, c_sys_js)
    _finish(commands, c_sys_mc)
    status_payload_n = 0
    for e in events:
        if e["t_payload"] is not None:
            e["t"] = round(e["t_payload"], 5)
            status_payload_n += 1
        elif c_mono is not None:
            e["t"] = round(e["recv"] + c_mono, 5)
        else:
            e["t"] = round(e["recv"], 5)
    # dict 键序整理：t 放行首，便于肉眼查文件
    states = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort", "tcp")}} for r in states]
    commands = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort", "tcp")}} for r in commands]
    events = [{"t": e["t"], "topic": e["topic"], "text": e["text"]} for e in events]

    # ---- ht 锚点残差：新调度触球≡ht，done(=accepted.t+duration) − ht 应 ≈0 ----
    # 全链自检：stamp 域 + status 发布时刻 + predict ct 任一有偏都会体现在这里。
    predicts: list[dict] = []
    for e in events:
        if e["topic"] != "/predict_hit_pos":
            continue
        try:
            p = json.loads(e["text"])
            predicts.append({"t": e["t"], "duration": p.get("duration"),
                             "rel_x": p.get("rel_x"), "ht": p.get("ht")})
        except Exception:
            pass
    ht_residuals: list[float] = []
    acc_re = re.compile(r"^accepted hit x=([\-0-9.]+) z=[\-0-9.]+ duration=([0-9.]+)")
    for e in events:
        if e["topic"] != "/tennis/status":
            continue
        m = acc_re.match(e["text"])
        if not m:
            continue
        x, dur = float(m.group(1)), float(m.group(2))
        for p in reversed(predicts):
            if p["t"] > e["t"]:
                continue
            if e["t"] - p["t"] > 0.35:
                break
            d_ok = isinstance(p["duration"], (int, float)) and abs(p["duration"] - dur) < 2e-3
            x_ok = isinstance(p["rel_x"], (int, float)) and abs(p["rel_x"] - x) < 5e-4
            if (d_ok or x_ok) and isinstance(p["ht"], (int, float)):
                ht_residuals.append((e["t"] + dur) - p["ht"])
                break

    mono_drift_samples = (
        mono_diffs[mono_ref_topic] if mono_ref_topic in mono_diffs
        else (state_diffs if state_domain == "rk_mono_native" else predict_ct_diffs)
    )
    clock_sync = {
        "joint_states_stamp_domain": state_domain,
        "motor_command_stamp_domain": command_domain,
        "events_with_payload_t": status_payload_n,
        "events_total": len(events),
        "legacy_stamps": legacy_stamps,
        "mono_ref_topic": mono_ref_topic,
        "mono_ref_biased": mono_ref_biased,
        "mono_minus_recv_median_s": None if c_mono is None else round(c_mono, 6),
        "joint_states_stamp_minus_recv_median_s": None if c_sys_js is None else round(c_sys_js, 6),
        "motor_command_stamp_minus_recv_median_s": None if c_sys_mc is None else round(c_sys_mc, 6),
        # 漂移自检：该钟 − PC 收钟的速率（原生模式下 joint_states stamp 本身就是单调钟）
        "mono_vs_pc_drift": _thirds_drift_ms_per_min(mono_drift_samples),
        "joint_states_stamp_vs_pc_drift": _thirds_drift_ms_per_min(state_diffs),
        "ht_anchor_residual_ms": (
            None if not ht_residuals
            else round(statistics.median(ht_residuals) * 1000, 1)
        ),
        "ht_anchor_n": len(ht_residuals),
    }

    result = {
        "schema": "tracker_arm_bag_v3",
        "time_axis": "rk_mono_abs_s",
        "time_sources": {
            "states": (
                "header.stamp（RK 单调钟原生）" if state_domain == "rk_mono_native"
                else "header.stamp(RK系统钟) − median(stamp−recv) + median(bot_state.t−recv)（旧 bag 兼容）"
            ),
            "commands": (
                "header.stamp（RK 单调钟原生）" if command_domain == "rk_mono_native"
                else "同 states 换算（用 motor_command 自己的 stamp−recv 中位）"
            ),
            "events": "status 尾缀 t= / predict 的 ct（原生）；无时间的旧事件 = recv + median(bot_state.t−recv)",
            "note": "与 /predict_hit_pos ct/ht、rk_tracking payload t 同钟；报告端复用 RK→PC 仿射时间映射",
        },
        "clock_sync": clock_sync,
        "bag_dir": str(args.bag.resolve()),
        "fk_source": "extract_arm_bag.fk",
        "start_ns": start_ns,
        "duration_sec": round((end_ns - start_ns) / 1e9, 4),
        "joint_names": list(joint_names),
        "state_joint_names_raw": seen_state_names,
        "command_joint_names_raw": seen_command_names,
        "topics": [
            {"name": name, "type": type_by_topic.get(name, ""), "count": counts[name]}
            for name in sorted(counts)
        ],
        "states": states,
        "commands": commands,
        "events": events,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        "arm json saved: %s (states=%d commands=%d events=%d duration=%.1fs)"
        % (args.output, len(states), len(commands), len(events), result["duration_sec"])
    )
    print("state joint names: %s" % (seen_state_names,))
    print("command joint names: %s" % (seen_command_names,))
    print("clock_sync: %s" % json.dumps(clock_sync, ensure_ascii=False))
    if legacy_stamps:
        print("NOTE: legacy epoch stamps converted (old firmware bag) — upgrade RK side for native mono stamps")
    d_mono = clock_sync["mono_vs_pc_drift"]
    if d_mono and abs(d_mono["ms_per_min"]) > 2.0:
        print("WARNING: RK mono vs PC drift %.2f ms/min > 2 — single-offset assumption needs review"
              % d_mono["ms_per_min"])
    if clock_sync["ht_anchor_residual_ms"] is not None and abs(clock_sync["ht_anchor_residual_ms"]) > 20.0:
        print("WARNING: |median(done - ht)| = %.1f ms — check clock chain or scheduler execution"
              % clock_sync["ht_anchor_residual_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
