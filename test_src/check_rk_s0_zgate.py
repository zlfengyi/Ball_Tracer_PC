# -*- coding: utf-8 -*-
"""离线评估 RK/bot_center 的 S0 发布门：拒绝 z<0 样本入拟合窗前后对比。

背景（0811 场 tracker_20260811_053055 实证）：地面静止球（world≈(1.54,22.3,0.04)）
被 RK 当真目标检出，正好落在某抛出手瞬间，导致 bot_center 来球滤波起轨的头两个样本
z<0（在地面以下）。S0 的拟合窗锚在起轨点且逐点增长，这两个坏样本永不老化，把整条
抛物线的 vz 拽歪 → predY 被摁在移动围栏下限以下 → 整抛零 /predict_hit_pos
（越界丢弃整条预测，不发消息，所以连 reject 行都没有）。

本脚本用 bot_center 自己发布的 estimate 位置复算 S0，对每条轨迹给出：
  baseline —— 全窗（含 z<0 样本）
  z-gate   —— z<0 样本不入窗（拟修改）
两者各自的尝试次数、落在围栏内的次数、首个合法 predY。
baseline 会与该轨迹实际发布的 stage-0 消息对照，用来校验复算方法本身。

只读 bag，不需要 ROS：mcap + 手工 CDR 解析（std_msgs/String = 4 字节封装头
+ uint32 长度 + UTF-8 字节含结尾 NUL），可直接用 .venv_clean 跑。

用法：
    python test_src/check_rk_s0_zgate.py --bag tracker_output/<session>/<session>_rosbag
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# bot_center 启动横幅：S0 发布门 y∈[0.50, 8.00]（移动围栏）+ 弧长≥0.35s + 点数≥10
FENCE_Y_MIN = 0.50
FENCE_Y_MAX = 8.00
MIN_POINTS = 10
MIN_SPAN_S = 0.35
# 兜底反弹系数；实际优先从本场 /predict_hit_pos 的 cor_eff / cor_xy_eff 中位数取
DEFAULT_COR = 0.8133
DEFAULT_COR_XY = 0.5141
DEFAULT_HIT_Z = 0.80
TRACK_GAP_S = 0.5
G = 9.81


def _decode_string_msg(data: bytes) -> dict:
    length = struct.unpack_from("<I", data, 4)[0]
    return json.loads(data[8 : 8 + length - 1].decode("utf-8", errors="replace"))


def _read_bag(bag_dir: Path) -> tuple[list, list, list]:
    from mcap.reader import make_reader

    files = sorted(bag_dir.glob("*.mcap"))
    if not files:
        raise SystemExit(f"no .mcap under {bag_dir}")
    est: list[tuple[float, list]] = []
    pred: list[dict] = []
    bot: list[tuple[float, float]] = []
    topics = ["/estimate_loc_topic", "/predict_hit_pos", "/bot_state"]
    for path in files:
        with open(path, "rb") as fh:
            for _schema, channel, message in make_reader(fh).iter_messages(topics=topics):
                try:
                    payload = _decode_string_msg(message.data)
                except Exception:
                    continue
                if channel.topic == "/estimate_loc_topic":
                    # source 区分来球滤波(return_ball_predict)与挥拍后找球(fire_ball_*)
                    if payload.get("source") == "return_ball_predict":
                        # ⚠ 该 topic 的时间字段是 ts，不是 t
                        est.append((float(payload["ts"]), payload["ball_loc"]))
                elif channel.topic == "/predict_hit_pos":
                    pred.append(payload)
                elif channel.topic == "/bot_state":
                    if payload.get("imu_t") is not None and payload.get("y") is not None:
                        bot.append((float(payload["imu_t"]), float(payload["y"])))
    est.sort(key=lambda r: r[0])
    bot.sort(key=lambda r: r[0])
    return est, pred, bot


def _split_tracks(est: list[tuple[float, list]]) -> list[list[tuple[float, list]]]:
    tracks: list[list] = []
    current: list = [est[0]]
    for prev, cur in zip(est, est[1:]):
        if cur[0] - prev[0] > TRACK_GAP_S:
            tracks.append(current)
            current = []
        current.append(cur)
    tracks.append(current)
    return tracks


def _pred_y(ts, ys, zs, car_y, cor, cor_xy, hit_z) -> float:
    """按 bot_center 的模型复算击球点 world y：自由飞行落地 → 反弹 → 下降穿 hit_z。"""
    if len(ts) < 5:
        return float("nan")
    t = np.asarray(ts) - ts[-1]
    vy, y_end = np.polyfit(t, ys, 1)
    cz = np.polyfit(t, zs, 2)
    z_end, vz = cz[2], cz[1]
    if z_end <= 0:
        return float("nan")
    a = -G / 2.0
    disc = vz * vz - 4 * a * z_end
    if disc < 0:
        return float("nan")
    t_land = (-vz - np.sqrt(disc)) / (2 * a)
    y_land = y_end + vy * t_land
    vz_impact = vz + 2 * a * t_land
    vy_out, vz_out = vy * cor_xy, -vz_impact * cor
    disc2 = vz_out * vz_out - 2 * G * hit_z
    if disc2 < 0:
        return float("nan")
    t_hit = (vz_out + np.sqrt(disc2)) / G  # 下降支
    return float(y_land + vy_out * t_hit + car_y)


def _scan(track, car_y, cor, cor_xy, hit_z, drop_below_zero: bool) -> dict:
    ts, ys, zs = [], [], []
    attempts = 0
    in_fence = 0
    first: tuple[float, float] | None = None
    dropped = 0
    for t, (_x, y, z) in track:
        if drop_below_zero and z < 0:
            dropped += 1
            continue
        ts.append(t)
        ys.append(y)
        zs.append(z)
        if len(ts) < MIN_POINTS or ts[-1] - ts[0] < MIN_SPAN_S:
            continue
        attempts += 1
        value = _pred_y(ts, ys, zs, car_y, cor, cor_xy, hit_z)
        if np.isfinite(value) and FENCE_Y_MIN <= value <= FENCE_Y_MAX:
            in_fence += 1
            if first is None:
                first = (t, value)
    return {"attempts": attempts, "in_fence": in_fence, "first": first, "dropped": dropped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag 目录")
    parser.add_argument("--cor", type=float, default=None, help="覆盖法向反弹系数")
    parser.add_argument("--cor-xy", type=float, default=None, help="覆盖切向反弹系数")
    parser.add_argument("--hit-z", type=float, default=DEFAULT_HIT_Z)
    args = parser.parse_args()

    # Windows 控制台默认 cp1252/gbk，中文表头会炸
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    est, pred, bot = _read_bag(args.bag)
    if not est:
        raise SystemExit("bag 内没有 source=return_ball_predict 的 estimate 样本")

    cor = args.cor if args.cor is not None else float(
        np.median([p["cor_eff"] for p in pred if p.get("cor_eff") is not None]) if pred else DEFAULT_COR
    )
    cor_xy = args.cor_xy if args.cor_xy is not None else float(
        np.median([p["cor_xy_eff"] for p in pred if p.get("cor_xy_eff") is not None]) if pred else DEFAULT_COR_XY
    )
    bot_t = np.asarray([r[0] for r in bot]) if bot else None
    bot_y = np.asarray([r[1] for r in bot]) if bot else None

    t0 = min([est[0][0]] + [p["ct"] for p in pred if p.get("ct") is not None])
    tracks = _split_tracks(est)
    pred_t = np.asarray([p["ct"] for p in pred])
    pred_y0 = [p["y"] for p in pred]
    pred_stage = [p["stage"] for p in pred]

    print(f"bag        : {args.bag}")
    print(f"参数       : cor={cor:.4f} cor_xy={cor_xy:.4f} hit_z={args.hit_z:.2f} "
          f"围栏 y∈[{FENCE_Y_MIN:.2f}, {FENCE_Y_MAX:.2f}] 点数≥{MIN_POINTS} 弧长≥{MIN_SPAN_S:.2f}s")
    print(f"轨迹       : {len(tracks)} 条 (source=return_ball_predict, 间隔>{TRACK_GAP_S}s 断开)")
    print()
    header = (f'{"起轨t":>9} {"n":>4} {"实发S0":>7} {"实发首predY":>11} | '
              f'{"base尝试":>8} {"base围栏内":>10} {"base首predY":>11} | '
              f'{"gate尝试":>8} {"gate围栏内":>10} {"gate首predY":>11} | {"丢弃":>4}')
    print(header)
    print("-" * len(header))

    rescued = 0
    touched = 0
    for track in tracks:
        start, end = track[0][0], track[-1][0]
        car_y = float(np.interp(start, bot_t, bot_y)) if bot_t is not None else 0.0
        mask = (pred_t >= start - 0.6) & (pred_t <= end + 0.8)
        pub = [(t, y) for t, y, s in zip(pred_t[mask], np.asarray(pred_y0)[mask],
                                         np.asarray(pred_stage)[mask]) if s == 0]
        base = _scan(track, car_y, cor, cor_xy, args.hit_z, drop_below_zero=False)
        gate = _scan(track, car_y, cor, cor_xy, args.hit_z, drop_below_zero=True)
        if gate["dropped"]:
            touched += 1
        if base["in_fence"] == 0 and gate["in_fence"] > 0:
            rescued += 1

        def fmt(entry) -> str:
            return "—" if entry["first"] is None else f'{entry["first"][1]:.2f}'

        print(f'{start - t0:9.2f} {len(track):4d} {len(pub):7d} '
              f'{(f"{pub[0][1]:.2f}" if pub else "—"):>11} | '
              f'{base["attempts"]:8d} {base["in_fence"]:10d} {fmt(base):>11} | '
              f'{gate["attempts"]:8d} {gate["in_fence"]:10d} {fmt(gate):>11} | '
              f'{gate["dropped"]:4d}')

    print()
    print(f"z<0 样本影响到的轨迹: {touched} 条；由 0 次合法尝试变为 >0 的: {rescued} 条")
    print("注：base 列应与「实发S0」量级/趋势一致，用于校验复算方法；"
          "gate 列是「S0 拒绝 z<0 样本入窗」后的预期结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
