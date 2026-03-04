# -*- coding: utf-8 -*-
"""
性能基准测试：使用 SyncCapture 连续同步取图并统计各项性能指标。

从 config/camera.json 加载相机配置（主从相机、曝光、ROI 等）。

监测指标：
  1) exposure_start_pc spread — 硬件同步精度（μs 级）
  2) 曝光→接收延迟 (exposure_start_pc → arrival_perf) — GigE 传输延迟（~32ms）
  3) 帧间间隔稳定性、丢包/丢帧统计、时间漂移分析

用法：
  python -m src.benchmark [--duration 300] [--save]
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import SyncCapture, frame_to_numpy

REPORT_DIR = _root / "benchmark_reports"
PROGRESS_INTERVAL = 30.0


def _percentile(data: list[float], p: float) -> float:
    """简易百分位数计算。"""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def _stats(data: list[float]) -> dict:
    """计算基本统计量。"""
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p1": 0.0, "p99": 0.0}
    s = sorted(data)
    mean = sum(s) / len(s)
    var = sum((x - mean) ** 2 for x in s) / len(s)
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": s[0],
        "max": s[-1],
        "p1": _percentile(s, 1),
        "p99": _percentile(s, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="Ball Tracer 性能基准测试")
    parser.add_argument("--duration", type=float, default=300.0, help="测试时长（秒，默认300）")
    parser.add_argument("--save", action="store_true", help="保存第一组同步图片")
    args = parser.parse_args()

    print("正在加载配置 (config/camera.json)...")
    with SyncCapture.from_config() as cap:
        all_serials = cap.sync_serials
        print(f"=== Ball Tracer 性能基准测试 ===")
        print(f"同步相机: {', '.join(all_serials)}  测试时长: {args.duration}s")
        print()
        # 等待相机稳定
        print("等待相机稳定 (2s)...")
        time.sleep(2.0)

        # ── 统计容器 ──
        sync_count = 0
        timeout_count = 0
        inter_frame_intervals_ms: list[float] = []
        arrival_spreads_ms: list[float] = []
        lost_packets: dict[str, int] = {sn: 0 for sn in all_serials}
        frame_nums: dict[str, int] = {sn: -1 for sn in all_serials}
        frame_drops: dict[str, int] = {sn: 0 for sn in all_serials}
        resolutions: dict[str, str] = {}
        prev_master_ts: int = 0
        saved = False

        # 曝光时间分析专用容器
        exposure_start_spreads_ms: list[float] = []  # 组内 exposure_start_pc max-min（ms）
        receive_latencies_ms: dict[str, list[float]] = {sn: [] for sn in all_serials}  # 每台相机 arrival - exposure_start_pc（ms）

        # 分时段统计（每 PROGRESS_INTERVAL 秒一段）
        segment_start_time = 0.0
        segment_exposure_spreads: list[float] = []
        segment_receive_latencies: dict[str, list[float]] = {sn: [] for sn in all_serials}
        segment_count = 0
        segment_idx = 0

        print(f"\n开始测试（{args.duration}s）...\n")
        t_start = time.monotonic()
        next_progress = t_start + PROGRESS_INTERVAL
        segment_start_time = t_start

        while time.monotonic() - t_start < args.duration:
            frames = cap.get_frames(timeout_s=1.0)
            if frames is None:
                timeout_count += 1
                continue

            sync_count += 1
            segment_count += 1

            # ── 收集基础指标 ──
            arrivals = []
            exposure_starts = []
            for sn in all_serials:
                f = frames[sn]
                arrivals.append(f.arrival_mono)
                exposure_starts.append(f.exposure_start_pc)
                lost_packets[sn] += f.lost_packet

                # 接收延迟：arrival_perf - exposure_start_pc（均为 perf_counter 时间轴）
                latency_ms = (f.arrival_perf - f.exposure_start_pc) * 1000.0
                receive_latencies_ms[sn].append(latency_ms)
                segment_receive_latencies[sn].append(latency_ms)

                # 分辨率
                if sn not in resolutions:
                    resolutions[sn] = f"{f.width}x{f.height}"

                # 丢帧检测
                prev_fn = frame_nums[sn]
                cur_fn = f.frame_num
                if prev_fn >= 0 and cur_fn != prev_fn + 1:
                    frame_drops[sn] += abs(cur_fn - prev_fn - 1)
                frame_nums[sn] = cur_fn

            # 组内 arrival_mono 差（ms）
            arrival_spread = (max(arrivals) - min(arrivals)) * 1000.0
            arrival_spreads_ms.append(arrival_spread)

            # 组内 exposure_start_pc 差（ms）— 衡量硬件同步精度
            exp_spread = (max(exposure_starts) - min(exposure_starts)) * 1000.0
            exposure_start_spreads_ms.append(exp_spread)
            segment_exposure_spreads.append(exp_spread)

            # 帧间间隔（第一台同步相机 host_timestamp, 单位 ms）
            master_ts = frames[all_serials[0]].host_timestamp
            if prev_master_ts > 0:
                interval = master_ts - prev_master_ts
                inter_frame_intervals_ms.append(float(interval))
            prev_master_ts = master_ts

            # 保存第一组图片
            if args.save and not saved:
                import cv2
                save_dir = _root / "benchmark_frames"
                save_dir.mkdir(parents=True, exist_ok=True)
                for sn, f in frames.items():
                    img = frame_to_numpy(f)
                    cv2.imwrite(str(save_dir / f"{sn}.png"), img)
                print(f"  已保存一组图片到 {save_dir}")
                saved = True

            # ── 分时段进度输出 ──
            now = time.monotonic()
            if now >= next_progress:
                elapsed = now - t_start
                seg_elapsed = now - segment_start_time
                segment_idx += 1

                seg_exp_stats = _stats(segment_exposure_spreads)
                print(f"  ── {elapsed:.0f}s / 段{segment_idx} ({segment_count} 组, {segment_count/seg_elapsed:.1f} fps) ──")
                print(f"     曝光时间差(exposure_start_pc gap):  均值={seg_exp_stats['mean']:.3f}ms  最大={seg_exp_stats['max']:.3f}ms  P99={seg_exp_stats['p99']:.3f}ms")
                for sn in all_serials:
                    tag = sn[-3:]
                    lat_stats = _stats(segment_receive_latencies[sn])
                    print(f"     [{tag}] {sn} 接收延迟: 均值={lat_stats['mean']:.1f}ms  最大={lat_stats['max']:.1f}ms")
                print()

                # 重置段统计
                segment_exposure_spreads = []
                segment_receive_latencies = {sn: [] for sn in all_serials}
                segment_count = 0
                segment_start_time = now
                next_progress = now + PROGRESS_INTERVAL

        t_elapsed = time.monotonic() - t_start

    # ── 生成报告内容 ──
    lines: list[str] = []
    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    out()
    out("=" * 60)
    out("       Ball Tracer 性能报告（曝光时间分析）")
    out("=" * 60)
    out()
    if t_elapsed > 0:
        out(f"运行时长:       {t_elapsed:.1f}s")
        out(f"同步取图次数:   {sync_count} ({sync_count/t_elapsed:.1f} fps)")
    out(f"超时次数:       {timeout_count}")
    out()

    # 分辨率
    out("相机分辨率:")
    for sn in all_serials:
        tag = sn[-3:]
        out(f"  [{tag}] {sn}: {resolutions.get(sn, '未知')}")
    out()

    # 帧间间隔
    if inter_frame_intervals_ms:
        iv_stats = _stats(inter_frame_intervals_ms)
        out("帧间间隔 (主相机 host_timestamp, ms):")
        out(f"  均值:    {iv_stats['mean']:.2f}")
        out(f"  标准差:  {iv_stats['std']:.2f}")
        out(f"  最小:    {iv_stats['min']:.2f}")
        out(f"  最大:    {iv_stats['max']:.2f}")
        out(f"  P1-P99:  [{iv_stats['p1']:.2f}, {iv_stats['p99']:.2f}]")
        out()

    # ★ 曝光开始时间差（exposure_start_pc gap）— 硬件同步精度
    if exposure_start_spreads_ms:
        exp_stats = _stats(exposure_start_spreads_ms)
        out("★ 曝光开始时间差 (exposure_start_pc max-min, ms):")
        out(f"  均值:    {exp_stats['mean']:.3f}")
        out(f"  标准差:  {exp_stats['std']:.3f}")
        out(f"  最小:    {exp_stats['min']:.3f}")
        out(f"  最大:    {exp_stats['max']:.3f}")
        out(f"  P1-P99:  [{exp_stats['p1']:.3f}, {exp_stats['p99']:.3f}]")
        out()

    # 主机到达抖动（arrival_mono spread）
    if arrival_spreads_ms:
        arr_stats = _stats(arrival_spreads_ms)
        out("主机到达抖动 (arrival_mono max-min, ms):")
        out(f"  均值:    {arr_stats['mean']:.3f}")
        out(f"  标准差:  {arr_stats['std']:.3f}")
        out(f"  最大:    {arr_stats['max']:.3f}")
        out(f"  P99:     {arr_stats['p99']:.3f}")
        out()

    # ★ 接收延迟（exposure_start_pc → arrival_perf）
    out("★ 接收延迟 (exposure_start_pc → arrival_perf, ms):")
    for sn in all_serials:
        tag = sn[-3:]
        lat_stats = _stats(receive_latencies_ms[sn])
        out(f"  [{tag}] {sn}:")
        out(f"    均值={lat_stats['mean']:.1f}  标准差={lat_stats['std']:.1f}  最小={lat_stats['min']:.1f}  最大={lat_stats['max']:.1f}")
        out(f"    P1-P99: [{lat_stats['p1']:.1f}, {lat_stats['p99']:.1f}]")
    out()

    # ★ 时间漂移分析：对比前10%和后10%的 exposure_start_pc spread
    n = len(exposure_start_spreads_ms)
    if n >= 20:
        n10 = max(n // 10, 1)
        early = exposure_start_spreads_ms[:n10]
        late = exposure_start_spreads_ms[-n10:]
        early_mean = sum(early) / len(early)
        late_mean = sum(late) / len(late)
        drift = late_mean - early_mean
        out("★ 时间漂移分析 (前10% vs 后10% exposure_start_pc gap):")
        out(f"  前10%均值:  {early_mean:.3f} ms")
        out(f"  后10%均值:  {late_mean:.3f} ms")
        out(f"  漂移:       {drift:+.3f} ms ({'稳定' if abs(drift) < 0.5 else '有漂移'})")
        out()

        # 接收延迟漂移
        out("★ 接收延迟漂移 (前10% vs 后10%):")
        for sn in all_serials:
            tag = sn[-3:]
            lats = receive_latencies_ms[sn]
            if len(lats) >= 20:
                n10_l = max(len(lats) // 10, 1)
                e_mean = sum(lats[:n10_l]) / n10_l
                l_mean = sum(lats[-n10_l:]) / n10_l
                d = l_mean - e_mean
                out(f"  [{tag}] {sn}: 前={e_mean:.1f}ms → 后={l_mean:.1f}ms  漂移={d:+.1f}ms")
        out()

    # 丢包
    out("丢包统计:")
    for sn in all_serials:
        tag = sn[-3:]
        out(f"  [{tag}] {sn}: {lost_packets[sn]}")
    out()

    # 丢帧
    total_drops = sum(frame_drops.values())
    out("丢帧检测 (frame_num 不连续):")
    for sn in all_serials:
        tag = sn[-3:]
        out(f"  [{tag}] {sn}: {frame_drops[sn]}")
    out()

    if total_drops == 0 and timeout_count == 0:
        out("所有指标正常，无丢帧无超时。")
    elif total_drops > 0:
        out(f"*** 警告: 检测到 {total_drops} 帧丢失 ***")

    # ── 保存报告到项目目录 ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"benchmark_{ts}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已保存到: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
