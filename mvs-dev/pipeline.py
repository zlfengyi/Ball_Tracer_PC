# -*- coding: utf-8 -*-

"""一个最小可复用的“多相机采集管线”封装。

这里不做过度设计：只把常用步骤串起来，方便业务层调用。
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from mvs.core._cleanup import best_effort, join_quietly
from mvs.core.events import MvsEvent
from mvs.sdk.binding import MvsBinding
from mvs.sdk.camera import MvsCamera, MvsSdk
from mvs.sdk.devices import enumerate_devices

from .grab import FramePacket, Grabber
from .grouping import TriggerGroupAssembler
from .soft_trigger import SoftwareTriggerLoop


@dataclass
class QuadCapture:
    """多相机采集：打开->取流->按 group_by 组包。

    该对象的定位是“管线句柄”：
    - 内部持有相机句柄、抓取线程、分组器、可选软触发线程；
    - 对外提供 `get_next_group()` 作为最小消费接口；
    - 事件（相机事件/软触发发送记录等）会写入 `event_queue`，更适合做诊断/离线统计；
    - 使用 `close()` 或上下文管理器负责资源释放。
    """

    binding: MvsBinding
    sdk: MvsSdk
    cameras: List[MvsCamera]
    stop_event: threading.Event
    frame_queue: "queue.Queue[FramePacket]"
    event_queue: "queue.Queue[MvsEvent]"
    grabbers: List[Grabber]
    assembler: TriggerGroupAssembler
    soft_trigger: Optional[SoftwareTriggerLoop]

    def get_next_group(self, timeout_s: float = 0.5) -> Optional[List[FramePacket]]:
        """获取下一组 N 张图（按 group_by 凑齐）。

        Args:
            timeout_s: 等待帧的超时时间。

        Returns:
            组满则返回 N 张 FramePacket；超时返回 None。
        """

        end_t = None if timeout_s <= 0 else (time.monotonic() + float(timeout_s))
        while not self.stop_event.is_set():
            remaining = 0.1
            if end_t is not None:
                remaining = max(0.0, end_t - time.monotonic())
                if remaining <= 0:
                    return None

            try:
                pkt = self.frame_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue

            group = self.assembler.add(pkt)
            if group is not None:
                return group

        return None

    def get_next_event(self, timeout_s: float = 0.0) -> Optional[MvsEvent]:
        """获取下一条事件（用于诊断/统计）。

        Notes:
            - 事件队列是“可选诊断通道”，不应影响主采集流程。
            - timeout_s<=0 时为非阻塞尝试。

        Args:
            timeout_s: 等待事件的超时时间（秒）。

        Returns:
            获取到事件则返回；无事件返回 None。
        """
        

        try:
            if float(timeout_s) <= 0:
                return self.event_queue.get_nowait()
            return self.event_queue.get(timeout=float(timeout_s))
        except queue.Empty:
            return None

    def drain_events(self, max_items: int = 1000) -> list[MvsEvent]:
        """尽力批量取出队列中的事件。

        Args:
            max_items: 最多取出的事件数量（用于限制一次性内存占用）。

        Returns:
            事件列表（可能为空）。
        """

        items: list[MvsEvent] = []
        n = max(0, int(max_items))
        for _ in range(n):
            ev = self.get_next_event(timeout_s=0.0)
            if ev is None:
                break
            items.append(ev)
        return items

    def close(self) -> None:
        """停止线程并释放 SDK 资源。

        Notes:
            - 该方法设计为“尽力清理”：清理过程中遇到的异常会被吞掉。
            - Grabber 线程的退出依赖 `timeout_ms`：若超时较大，join 可能需要更久。
        """

        self.stop_event.set()
        if self.soft_trigger is not None:
            join_quietly(self.soft_trigger, timeout_s=1.0)

        for g in self.grabbers:
            join_quietly(g, timeout_s=1.0)

        for c in self.cameras:
            best_effort(c.close)

        best_effort(self.sdk.finalize)

    def __enter__(self) -> "QuadCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_quad_capture(
    *,
    binding: MvsBinding,
    serials: Sequence[str],
    trigger_sources: Sequence[str],
    trigger_activation: str,
    trigger_cache_enable: bool,
    timeout_ms: int = 1000,
    group_timeout_ms: int = 200,
    max_pending_groups: int = 256,
    group_by: Literal["frame_num", "sequence"] = "frame_num",
    enable_soft_trigger_fps: float = 0.0,
    soft_trigger_serials: Sequence[str] = (),
    camera_event_names: Sequence[str] = (),
    master_serial: str = "",
    master_line_output: str = "",
    master_line_source: str = "",
    master_line_mode: str = "Output",
    pixel_format: str = "",
    image_width: int | None = None,
    image_height: int | None = None,
    image_offset_x: int = 0,
    image_offset_y: int = 0,
    exposure_auto: str = "",
    exposure_time_us: float | None = None,
    gain_auto: str = "",
    gain: float | None = None,
) -> QuadCapture:
    """打开多相机采集并启动抓取线程。

    约定：`serials` 的顺序就是 cam0..cam(N-1)。

    Args:
        binding: 已加载的 MVS 绑定。
        serials: 相机序列号列表。
        trigger_sources: 每台相机的触发源（长度必须与 serials 相同）。
        trigger_activation: 触发沿（软触发时通常不生效/不可写）。
        trigger_cache_enable: 是否启用触发缓存（部分机型不支持）。
        timeout_ms: 单次取流等待超时（影响 Grabber 线程的响应速度）。
        group_timeout_ms: 分组等待超时。
        max_pending_groups: 最大待凑齐分组缓存数。
        group_by: 分组键（默认 frame_num）。可选：frame_num/sequence。
        enable_soft_trigger_fps: >0 时启用软触发循环（用于链路验证，不保证同曝光）。
        soft_trigger_serials: 指定哪些相机参与软触发；为空时会在“全部为 Software 触发”时默认全选。
        pixel_format: 可选像素格式（PixelFormat）。空字符串表示不设置。
        image_width: 可选输出宽度（ROI）。None 表示不设置。
        image_height: 可选输出高度（ROI）。None 表示不设置。
        image_offset_x: ROI 左上角 X 偏移（默认 0）。
        image_offset_y: ROI 左上角 Y 偏移（默认 0）。
        exposure_auto: 自动曝光模式。空字符串表示不设置。
        exposure_time_us: 曝光时间（微秒）。None 表示不设置。
        gain_auto: 自动增益模式。空字符串表示不设置。
        gain: 增益值。None 表示不设置。

    Returns:
        一个已启动的 `QuadCapture`。

    Raises:
        ValueError: 参数不合法。
        RuntimeError: 找不到指定序列号的相机。
        Exception: 打开/配置/启动过程中底层 SDK 抛出的任意异常。
    """

    if len(serials) <= 0:
        raise ValueError("serials 不能为空")

    if len(trigger_sources) != len(serials):
        raise ValueError("trigger_sources 的长度必须与 serials 一致")

    num_cameras = len(serials)

    sdk = MvsSdk(binding)
    sdk.initialize()

    st_dev_list, descs = enumerate_devices(binding)
    serial_to_desc = {d.serial: d for d in descs}

    stop_event = threading.Event()
    frame_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=4096)
    event_queue: "queue.Queue[MvsEvent]" = queue.Queue(maxsize=65536)
    cameras: List[MvsCamera] = []
    grabbers: List[Grabber] = []
    soft_trigger: Optional[SoftwareTriggerLoop] = None
    try:
        for s, trig_src in zip(serials, trigger_sources):
            dev = serial_to_desc.get(s)
            if dev is None:
                raise RuntimeError(f"找不到序列号为 {s} 的相机")

            is_master = bool(master_serial) and (str(s) == str(master_serial))
            cameras.append(
                MvsCamera.open_from_device_list(
                    binding=binding,
                    st_dev_list=st_dev_list,
                    dev_index=dev.index,
                    serial=dev.serial,
                    tlayer_type=dev.tlayer_type,
                    trigger_source=str(trig_src),
                    trigger_activation=trigger_activation,
                    trigger_cache_enable=trigger_cache_enable,
                    event_queue=event_queue,
                    event_names=camera_event_names,
                    line_output_selector=(str(master_line_output) if is_master else ""),
                    line_output_source=(str(master_line_source) if is_master else ""),
                    line_output_mode=str(master_line_mode or "Output"),
                    pixel_format=str(pixel_format),
                    image_width=image_width,
                    image_height=image_height,
                    image_offset_x=int(image_offset_x),
                    image_offset_y=int(image_offset_y),
                    exposure_auto=str(exposure_auto),
                    exposure_time_us=exposure_time_us,
                    gain_auto=str(gain_auto),
                    gain=gain,
                )
            )

        for cam_index, c in enumerate(cameras):
            g = Grabber(
                binding=binding,
                cam_index=cam_index,
                serial=c.serial,
                cam=c.cam,
                out_q=frame_queue,
                stop_event=stop_event,
                timeout_ms=timeout_ms,
            )
            grabbers.append(g)
            g.start()

        if float(enable_soft_trigger_fps) > 0:
            wanted = {s for s in soft_trigger_serials if str(s).strip()}
            if not wanted:
                # 兼容默认行为：如果所有相机都是 Software 触发，则软触发下发给全部相机。
                if all(str(x).lower() == "software" for x in trigger_sources):
                    wanted = {c.serial for c in cameras}

            targets = [(c.serial, c.cam) for c in cameras if c.serial in wanted]
            if targets:
                soft_trigger = SoftwareTriggerLoop(
                    targets=targets,
                    stop_event=stop_event,
                    fps=float(enable_soft_trigger_fps),
                    out_q=event_queue,
                )
                soft_trigger.start()

        assembler = TriggerGroupAssembler(
            num_cameras=num_cameras,
            group_timeout_s=float(group_timeout_ms) / 1000.0,
            max_pending_groups=max_pending_groups,
            group_by=group_by,
        )

        return QuadCapture(
            binding=binding,
            sdk=sdk,
            cameras=cameras,
            stop_event=stop_event,
            frame_queue=frame_queue,
            event_queue=event_queue,
            grabbers=grabbers,
            assembler=assembler,
            soft_trigger=soft_trigger,
        )
    except Exception:
        stop_event.set()
        if soft_trigger is not None:
            join_quietly(soft_trigger, timeout_s=1.0)
        for g in grabbers:
            join_quietly(g, timeout_s=1.0)
        for c in cameras:
            best_effort(c.close)
        best_effort(sdk.finalize)
        raise
