# -*- coding: utf-8 -*-

"""软触发循环（TriggerSource=Software 时使用）。

该模块负责“发命令”的侧写：
- 每个触发周期记录一次 send 事件（host monotonic + wall time），用于离线统计 send-fps。
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional, Sequence, Tuple

from mvs.core.events import MvsEvent


class SoftwareTriggerLoop(threading.Thread):
    """按固定频率下发 TriggerSoftware。

    说明：
    - 该方式用于“先跑起来/先压测链路”。
    - 由于触发命令需要按相机逐个下发，曝光并非真正同一时刻；严格同步仍推荐硬件外触发。
    """

    def __init__(
        self,
        *,
        targets: Sequence[Tuple[str, Any]],
        stop_event: threading.Event,
        fps: float,
        out_q: Optional["queue.Queue[MvsEvent]"] = None,
    ) -> None:
        """创建一个软触发线程。

        Args:
            targets: (serial, cam_handle) 列表。
            stop_event: 外部停止信号。
            fps: 目标触发频率。
            out_q: 可选事件队列；若提供，会写入 type=soft_trigger_send 的 JSON 记录。
        """

        super().__init__(daemon=True)
        self._targets = list(targets)
        self._stop_event = stop_event
        self._period = 1.0 / max(float(fps), 0.001)
        self._out_q = out_q

    def run(self) -> None:
        """主循环：按频率下发 TriggerSoftware。"""
        next_t = time.perf_counter()
        seq = 0
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue

            send_monotonic = time.monotonic()
            send_wall_time = time.time()
            targets_serial = []
            for serial, cam in self._targets:
                try:
                    cam.MV_CC_SetCommandValue("TriggerSoftware")
                    targets_serial.append(str(serial))
                except Exception:
                    pass

            if self._out_q is not None:
                try:
                    self._out_q.put_nowait(
                        {
                            "type": "soft_trigger_send",
                            "seq": seq,
                            "created_at": send_wall_time,
                            "host_monotonic": send_monotonic,
                            "targets": targets_serial,
                        }
                    )
                except queue.Full:
                    # 事件队列满：丢弃发送事件（不影响采集主流程）
                    pass

            seq += 1

            next_t += self._period
            if now - next_t > 1.0:
                next_t = now + self._period
