# -*- coding: utf-8 -*-

"""取流线程封装（MV_CC_GetImageBuffer）。"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from mvs.sdk.binding import MvsBinding


@dataclass(frozen=True, slots=True)
class FramePacket:
    """一帧图像与其元信息。

    该数据结构尽量保持“纯数据”：不绑定相机句柄，便于跨线程传递、落盘或后续处理。
    """

    cam_index: int
    serial: str
    frame_num: int
    dev_timestamp: int
    host_timestamp: int
    width: int
    height: int
    pixel_type: int
    frame_len: int
    lost_packet: int
    arrival_monotonic: float
    data: bytes


class Grabber(threading.Thread):
    """单相机抓取线程，把帧放入队列。"""

    def __init__(
        self,
        *,
        binding: MvsBinding,
        cam_index: int,
        serial: str,
        cam: Any,
        out_q: "queue.Queue[FramePacket]",
        stop_event: threading.Event,
        timeout_ms: int,
    ) -> None:
        """创建一个抓取线程。

        Args:
            binding: 已加载的 MVS 绑定。
            cam_index: 相机在管线中的下标（0..N-1）。
            serial: 相机序列号（用于日志/诊断）。
            cam: SDK 相机句柄（MvCamera 实例）。
            out_q: 输出队列（跨线程传递 FramePacket）。
            stop_event: 外部停止信号。
            timeout_ms: 单次 MV_CC_GetImageBuffer 等待超时（毫秒）。
        """

        super().__init__(daemon=True, name=f"Grabber-{serial}")
        self._binding = binding
        self._cam_index = cam_index
        self._serial = serial
        self._cam = cam
        self._out_q = out_q
        self._stop_event = stop_event
        self._timeout_ms = int(timeout_ms)

    def run(self) -> None:
        """循环取流并把帧送入队列。

        关键点：
        - 这里会把 SDK 缓冲区拷贝成 Python `bytes`（`ctypes.string_at`），随后立刻 `FreeImageBuffer`。
          这样下游跨线程处理/落盘时不会引用到已被 SDK 复用的内存。
        - 代价是一次内存拷贝；如果你追求极致性能并能严格管理生命周期，可以做零拷贝，
          但复杂度会显著上升（更容易踩到“缓冲复用导致数据被覆盖”的坑）。
        """

        st_frame = self._binding.params.MV_FRAME_OUT()
        while not self._stop_event.is_set():
            ret = self._cam.MV_CC_GetImageBuffer(st_frame, int(self._timeout_ms))
            if int(ret) != int(self._binding.MV_OK):
                continue

            pkt: Optional[FramePacket] = None
            try:
                arrival = time.monotonic()
                info = st_frame.stFrameInfo

                width = int(info.nExtendWidth) if int(info.nExtendWidth) else int(info.nWidth)
                height = int(info.nExtendHeight) if int(info.nExtendHeight) else int(info.nHeight)
                frame_len = int(info.nFrameLenEx) if int(info.nFrameLenEx) else int(info.nFrameLen)

                dev_ts = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
                host_ts = int(info.nHostTimeStamp)

                # 防御性：极端情况下 SDK 可能返回 0 长度。
                data = b"" if frame_len <= 0 else ctypes.string_at(st_frame.pBufAddr, frame_len)

                pkt = FramePacket(
                    cam_index=self._cam_index,
                    serial=self._serial,
                    frame_num=int(info.nFrameNum),
                    dev_timestamp=dev_ts,
                    host_timestamp=host_ts,
                    width=width,
                    height=height,
                    pixel_type=int(info.enPixelType),
                    frame_len=frame_len,
                    lost_packet=int(info.nLostPacket),
                    arrival_monotonic=arrival,
                    data=data,
                )
            finally:
                # 无论后续处理是否异常，都必须归还 SDK 缓冲。
                try:
                    self._cam.MV_CC_FreeImageBuffer(st_frame)
                except Exception:
                    pass

            if pkt is None:
                # 解析失败（例如内存不足/异常像素格式等）：本帧丢弃。
                continue

            try:
                self._out_q.put(pkt, timeout=0.5)
            except queue.Full:
                # 下游处理跟不上：丢帧（不阻塞取流线程，避免反压导致 SDK 堵塞）。
                pass
