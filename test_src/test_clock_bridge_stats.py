# -*- coding: utf-8 -*-
"""tracker 侧 PC↔RK 时钟桥的统计口径。

桥是报告对齐里唯一**不依赖本场有没有球**的粗定位手段（见 memory v03-rk-pc-report-align），
所以它宁可交白卷也不能交个不准的数：样本不够 → None；有尖峰 → 中位数扛住。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_tracker import clock_bridge_stats


def test_returns_none_below_twenty_samples():
    assert clock_bridge_stats([1.0] * 19, source="/bot_state") is None
    assert clock_bridge_stats([], source="/bot_state") is None


def test_median_survives_scheduling_spikes():
    offsets = [10.000 + 0.001 * (i % 5) for i in range(200)]
    offsets += [10.9, 11.4, 12.0, 9.1]  # 偶发重传/调度尖峰
    stats = clock_bridge_stats(offsets, source="/bot_state")
    assert stats is not None
    assert abs(stats["pc_minus_rk"] - 10.002) <= 0.003, stats
    assert stats["mad"] <= 0.01, stats
    assert stats["n"] == 204
    assert stats["source"] == "/bot_state"


def test_mad_reports_real_jitter():
    offsets = [10.0 + (0.5 if i % 2 else -0.5) for i in range(100)]
    stats = clock_bridge_stats(offsets, source="/bot_state")
    assert stats is not None
    # 抖动 0.5s 会被报告端的 mad>0.25 门挡掉，正是这里要暴露的信息
    assert stats["mad"] >= 0.25, stats
