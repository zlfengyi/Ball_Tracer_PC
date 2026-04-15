"""
Windows time-sync responder for RK <-> PC synchronization.

Protocol:
  1. RK publishes ping: {"seq": N, "t1": <rk_perf_counter>} -> /time_sync/ping
  2. Windows receives and immediately publishes:
     {
       "seq": N,
       "t1": <rk_perf_counter>,
       "t2": <win_perf_counter>
     } -> /time_sync/pong
  3. RK uses t1/t2/t3 to estimate offset and RTT.

This responder also prints a 5-second rolling summary of ping arrival
intervals and one-way delay jitter inferred from:

    delta_delay ~= (recv_i - recv_{i-1}) - (t1_i - t1_{i-1})

The absolute clock offset cancels in that difference, so it is useful for
evaluating network jitter, packet bunching, and missed pings. It is not a
direct measurement of absolute one-way delay or RTT on the Windows side.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ros2_support import ensure_ros2_environment, make_best_effort_qos

ensure_ros2_environment()

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ModuleNotFoundError:
    rclpy = None

    class Node:  # type: ignore[override]
        pass

    String = None  # type: ignore[assignment]

PING_TOPIC = "/time_sync/ping"
PONG_TOPIC = "/time_sync/pong"
REPORT_PERIOD_S = 5.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = max(0.0, min(1.0, p)) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _to_optional_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_optional_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


@dataclass
class TimeSyncWindowStats:
    period_s: float = REPORT_PERIOD_S
    total_count: int = 0
    total_parse_errors: int = 0
    total_reports: int = 0
    prev_seq: Optional[int] = None
    prev_t1: Optional[float] = None
    prev_recv_mono: Optional[float] = None
    window_count: int = 0
    window_parse_errors: int = 0
    first_seq: Optional[int] = None
    last_seq: Optional[int] = None
    seq_gap_count: int = 0
    out_of_order_count: int = 0
    missing_t1_count: int = 0
    first_recv_mono: Optional[float] = None
    last_recv_mono: Optional[float] = None
    local_dt_ms: list[float] = field(default_factory=list)
    remote_dt_ms: list[float] = field(default_factory=list)
    delta_delay_ms: list[float] = field(default_factory=list)
    jitter_abs_ms: list[float] = field(default_factory=list)
    callback_ms: list[float] = field(default_factory=list)

    def mark_parse_error(self) -> None:
        self.total_parse_errors += 1
        self.window_parse_errors += 1

    def record(
        self,
        *,
        seq: Optional[int],
        t1: Optional[float],
        recv_mono: float,
        callback_ms: float,
    ) -> None:
        self.total_count += 1
        self.window_count += 1
        if self.first_recv_mono is None:
            self.first_recv_mono = recv_mono
        self.last_recv_mono = recv_mono
        self.callback_ms.append(callback_ms)

        if seq is not None:
            if self.first_seq is None:
                self.first_seq = seq
            self.last_seq = seq
            if self.prev_seq is not None:
                gap = seq - self.prev_seq - 1
                if gap > 0:
                    self.seq_gap_count += gap
                elif seq <= self.prev_seq:
                    self.out_of_order_count += 1
            self.prev_seq = seq

        if t1 is None:
            self.missing_t1_count += 1

        if self.prev_recv_mono is not None:
            local_dt_ms = (recv_mono - self.prev_recv_mono) * 1000.0
            self.local_dt_ms.append(local_dt_ms)

            if t1 is not None and self.prev_t1 is not None:
                remote_dt_ms = (t1 - self.prev_t1) * 1000.0
                if remote_dt_ms > 0.0:
                    delay_delta_ms = local_dt_ms - remote_dt_ms
                    self.remote_dt_ms.append(remote_dt_ms)
                    self.delta_delay_ms.append(delay_delta_ms)
                    self.jitter_abs_ms.append(abs(delay_delta_ms))

        self.prev_recv_mono = recv_mono
        if t1 is not None:
            self.prev_t1 = t1

    def build_report_and_reset(self, *, now_mono: float) -> str:
        self.total_reports += 1
        if self.window_count == 0:
            report = (
                f"time_sync[{self.period_s:.0f}s] no ping "
                f"(parse_err={self.window_parse_errors}, total_rx={self.total_count})"
            )
            self._reset_window()
            return report

        span_s = 0.0
        if self.first_recv_mono is not None and self.last_recv_mono is not None:
            span_s = max(0.0, self.last_recv_mono - self.first_recv_mono)
        effective_span_s = max(span_s, self.period_s)
        rate_hz = self.window_count / effective_span_s

        parts = [
            f"time_sync[{self.period_s:.0f}s]",
            f"rx={self.window_count}",
            f"rate={rate_hz:.1f} Hz",
        ]
        if self.first_seq is not None and self.last_seq is not None:
            parts.append(f"seq={self.first_seq}->{self.last_seq}")
        if self.seq_gap_count > 0:
            parts.append(f"gap={self.seq_gap_count}")
        if self.out_of_order_count > 0:
            parts.append(f"ooo={self.out_of_order_count}")
        if self.window_parse_errors > 0:
            parts.append(f"parse_err={self.window_parse_errors}")

        if self.local_dt_ms:
            parts.append(
                "dt_local="
                f"{_mean(self.local_dt_ms):.2f} +/- {_std(self.local_dt_ms):.2f} ms"
            )
        if self.remote_dt_ms:
            parts.append(
                "dt_rk="
                f"{_mean(self.remote_dt_ms):.2f} +/- {_std(self.remote_dt_ms):.2f} ms"
            )
        elif self.missing_t1_count > 0:
            parts.append(f"rk_t1_missing={self.missing_t1_count}")

        if self.jitter_abs_ms:
            parts.append(
                "jitter="
                f"avg={_mean(self.jitter_abs_ms):.3f} ms "
                f"p95={_percentile(self.jitter_abs_ms, 0.95):.3f} ms "
                f"max={max(self.jitter_abs_ms):.3f} ms"
            )
            parts.append(
                "delay_delta="
                f"avg={_mean(self.delta_delay_ms):+.3f} ms"
            )
        else:
            parts.append("jitter=n/a")

        if self.callback_ms:
            parts.append(
                "cb="
                f"avg={_mean(self.callback_ms):.3f} ms "
                f"max={max(self.callback_ms):.3f} ms"
            )

        parts.append(f"total_rx={self.total_count}")
        report = "  ".join(parts)
        self._reset_window()
        return report

    def _reset_window(self) -> None:
        self.window_count = 0
        self.window_parse_errors = 0
        self.first_seq = None
        self.last_seq = None
        self.seq_gap_count = 0
        self.out_of_order_count = 0
        self.missing_t1_count = 0
        self.first_recv_mono = None
        self.last_recv_mono = None
        self.local_dt_ms.clear()
        self.remote_dt_ms.clear()
        self.delta_delay_ms.clear()
        self.jitter_abs_ms.clear()
        self.callback_ms.clear()


class TimeSyncResponder(Node):
    def __init__(self):
        super().__init__("win_time_sync_responder")
        self._stats = TimeSyncWindowStats(period_s=REPORT_PERIOD_S)
        qos = make_best_effort_qos()
        self.sub = self.create_subscription(
            String, PING_TOPIC, self._on_ping, qos
        )
        self.pub = self.create_publisher(String, PONG_TOPIC, qos)
        self._report_timer = self.create_timer(REPORT_PERIOD_S, self._report_stats)
        self.get_logger().info(
            f"time_sync responder started: listen {PING_TOPIC}, publish {PONG_TOPIC}"
        )

    def _on_ping(self, msg: String):
        recv_mono = time.perf_counter()
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self._stats.mark_parse_error()
            return

        seq = _to_optional_int(data.get("seq"))
        t1 = _to_optional_float(data.get("t1"))

        data["t2"] = recv_mono
        pong = String()
        pong.data = json.dumps(data)
        self.pub.publish(pong)

        callback_ms = (time.perf_counter() - recv_mono) * 1000.0
        self._stats.record(
            seq=seq,
            t1=t1,
            recv_mono=recv_mono,
            callback_ms=callback_ms,
        )

    def _report_stats(self) -> None:
        self.get_logger().info(
            self._stats.build_report_and_reset(now_mono=time.perf_counter())
        )


def main():
    if rclpy is None or String is None:
        raise RuntimeError("rclpy/std_msgs is required to run win_time_sync.py")
    rclpy.init()
    node = TimeSyncResponder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
