"""/joint_states 关节时刻对齐（generate_curve3_html._time_align_joint_states）。

合成一段 arm_controller_cpp 两相调度的 /joint_states：tick 5ms，A 相 J1,J2,J3,J5,J4、B 相 J1,J5,J6，
帧间 g=0.52ms，每帧反馈 0.15ms 后到达，header.stamp=最新一帧到达。六轴各按匀速运动生成真值，
快照里非本相轴保留旧值（正如节点的 B 相快照里 J2-J4 是 5ms 前的值）。
对齐后每行 position 应等于该行戳时刻的真值；识别不出两相结构时必须原样不动。
"""
import copy
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent


def _load():
    sys.path.insert(0, str(SRC))
    try:
        from generate_curve3_html import _time_align_joint_states, _TX_PHASE_ORDER, _REPLY_FRAME_MS
    finally:
        sys.path.pop(0)
    return _time_align_joint_states, _TX_PHASE_ORDER, _REPLY_FRAME_MS


def _synth_two_phase(n_ticks=400, g_ms=0.52, reply_ms=0.15, tick_ms=5.0, t0=1000.0):
    _, order, _ = _load()
    w = [3.0, -0.8, 0.6, 0.4, -2.5, 1.7]          # rad/s，六轴各自匀速
    a = [-0.5, 0.3, 1.5, 1.8, 0.0, 0.3]
    wd = [0.2, -0.1, 0.05, 0.0, -0.3, 0.1]        # velocity 也给个线性变化，验证同样被插值
    eff = [10.0, -5.0, 2.0, 1.0, 0.5, 0.1]
    truth = lambda j, t: a[j] + w[j] * (t - t0)
    vel_truth = lambda j, t: w[j] + wd[j] * (t - t0)
    last = [truth(j, t0) for j in range(6)]
    last_v = [vel_truth(j, t0) for j in range(6)]
    rows = []
    for k in range(n_ticks):
        phase = "A" if k % 2 == 0 else "B"
        seq = order[phase]
        tick = t0 + k * tick_ms * 1e-3
        stamp = None
        for i, j in enumerate(seq):
            ts = tick + i * g_ms * 1e-3            # 命令发出即取样（简化），反馈 reply_ms 后到达
            last[j] = truth(j, ts)
            last_v[j] = vel_truth(j, ts)
            stamp = ts + reply_ms * 1e-3
        rows.append({"t": stamp, "position": [round(v, 5) for v in last],
                     "velocity": [round(v, 5) for v in last_v], "effort": list(eff)})
    return rows, truth, vel_truth


def test_two_phase_rows_are_aligned_to_stamp():
    align_fn, _, _ = _load()
    rows, truth, vel_truth = _synth_two_phase()
    arm = {"states": rows}
    info = align_fn(arm)
    assert info is not None and info["schedule"] == "two_phase_j1_j5"
    assert info["tx_gap_ms"] == pytest.approx(0.52, abs=0.01)
    assert info["phase_rows"]["unknown"] <= 1
    # 中段（首尾各留几行给插值找邻居）逐行逐轴对齐到真值：J1 200Hz、J2-J4 100Hz（B 相原本是旧值）、J6 100Hz
    for row in rows[5:-5]:
        for j in range(6):
            assert row["position"][j] == pytest.approx(truth(j, row["t"]), abs=2e-4), (j, row["t"])
            assert row["velocity"][j] == pytest.approx(vel_truth(j, row["t"]), abs=2e-4), (j, row["t"])
        assert row["effort"] == [10.0, -5.0, 2.0, 1.0, 0.5, 0.1]
    lag = info["sample_lag_before_stamp_ms"]
    assert lag["A"]["J1"] == pytest.approx(4 * 0.52 + 0.15, abs=0.05)
    assert lag["B"]["J1"] == pytest.approx(2 * 0.52 + 0.15, abs=0.05)
    assert lag["A"]["J4"] == pytest.approx(0.15, abs=0.01) and lag["B"]["J6"] == pytest.approx(0.15, abs=0.01)


def test_raw_rows_were_actually_wrong_before_alignment():
    """对照：不对齐时 B 相快照里的 J2 与真值差一整拍（w2·~6ms），J1 差 ~2ms——这就是本函数存在的理由。"""
    _, _, _ = _load()
    rows, truth, _ = _synth_two_phase()
    worst_j2 = max(abs(r["position"][1] - truth(1, r["t"])) for r in rows[5:-5])
    worst_j1 = max(abs(r["position"][0] - truth(0, r["t"])) for r in rows[5:-5])
    assert worst_j2 > 0.8 * 5.5e-3      # |w2|=0.8 rad/s × ≥5.5ms
    assert worst_j1 > 3.0 * 2.0e-3      # |w1|=3.0 rad/s × ≥2ms


def test_non_two_phase_data_is_left_untouched():
    align_fn, _, _ = _load()
    # 单相 100Hz 节点：每 10ms 六轴全刷新
    rows = [{"t": 5.0 + 0.01 * k, "position": [0.1 * k] * 6, "velocity": [1.0] * 6, "effort": [0.0] * 6}
            for k in range(600)]
    arm = {"states": copy.deepcopy(rows)}
    assert align_fn(arm) is None
    assert arm["states"] == rows
    # 行数太少
    arm2 = {"states": copy.deepcopy(rows[:50])}
    assert align_fn(arm2) is None
    assert arm2["states"] == rows[:50]
    # 关节残缺的行不参与也不报错
    rows_bad, _, _ = _synth_two_phase(n_ticks=300)
    rows_bad[10]["position"] = [0.1, None, 0.3, 0.15, -0.4, 0.25]
    rows_bad[11]["position"] = rows_bad[11]["position"][:5]
    rows_bad[12].pop("velocity")
    arm3 = {"states": rows_bad}
    info = align_fn(arm3)
    assert info is not None
    assert arm3["states"][10]["position"] == [0.1, None, 0.3, 0.15, -0.4, 0.25]
    assert len(arm3["states"][11]["position"]) == 5
    assert "velocity" not in arm3["states"][12]
