# -*- coding: utf-8 -*-
"""
海康威视多相机同步采集模块 (ball_grabber)。

═══════════════════════════  系统方案概述  ═══════════════════════════

硬件：
  - 相机型号：海康 MV-CS050-10GC (GigE Vision, 500万像素, BayerRG8)
  - 主相机 DA7403103：free-run 35fps，Line1 输出 ExposureStartActive 信号
  - 从相机 DA8571029 / DA7403087 / DA8474746：Line0 FallingEdge 硬触发（物理接线接收主相机信号）
  - 当前默认配置下主相机也参与图像输出，共返回 4 路同步图像

相机参数 (config/camera.json)：
  - 曝光：3000 μs | 增益：23.5 dB | 帧率：35 fps | GigE 丢包重传已开启
  - 4 台相机默认均为全画幅 2048×1536

时间同步：
  - 每台相机的 ImageGrabber 启动时校准 PC↔设备时钟偏移 (GevTimestampControlLatch)
  - 后台线程周期性重校准（~0.5s 间隔），偏移跳变 >5ms 自动忽略
  - 每帧计算 exposure_start_pc = dev_timestamp / 100MHz + offset (perf_counter 时间轴)

同步取图流程 (SyncCapture.get_frames)：
  1. 各 ImageGrabber 后台线程持续取帧入队（bounded deque, max=10）
  2. get_frames() 排空所有 grabber 队列，仅保留各相机最新帧
  3. 检查 arrival_perf spread ≤ 200ms（粗同步）
  4. 硬触发模式额外检查 exposure_start_pc spread ≤ 10ms（确保同一触发周期）
  5. 返回 {serial: Frame} 字典，或超时返回 None

像素格式：
  - 海康 Bayer 命名与 OpenCV 相反：BayerRG8 → cv2.COLOR_BayerBG2BGR
  - frame_to_numpy() 自动处理 Mono8 / BayerXX8 / RGB8 / BGR8

模块结构：
  - open_camera()    打开单台相机并 StartGrabbing
  - close_camera()   停止取流、关闭设备、销毁句柄
  - frame_to_numpy() Frame 原始数据 → numpy BGR/Mono（默认 180° 旋转）
  - ImageGrabber     后台取帧线程（带时间偏移校准）
  - SyncCapture      多相机同步采集上下文管理器
  - SyncCapture.from_config()  从 config/camera.json 创建
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# 环境初始化（MvImport + DLL 路径），必须在 import SDK 之前
import src.mvs_env as _env  # noqa: F401

from MvCameraControl_class import MvCamera
from CameraParams_header import (
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_DEVICE_INFO,
    MV_FRAME_OUT,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
)
from PixelType_header import (
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_RGB8_Packed,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerGB8,
)
from ctypes import cast, POINTER, byref, sizeof, memset


# ───────────────── 工具函数 ─────────────────


def _env_optional_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} 不是合法布尔值")


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_optional_bool(name)
    return default if raw is None else raw


_ENV_CAMERA_REVERSE_180 = _env_bool("BALL_TRACER_CAMERA_REVERSE_180", True)
_ENV_CAMERA_REVERSE_X = _env_optional_bool("BALL_TRACER_CAMERA_REVERSE_X")
_ENV_CAMERA_REVERSE_Y = _env_optional_bool("BALL_TRACER_CAMERA_REVERSE_Y")
_ENV_SOFTWARE_ROTATE_180 = _env_bool("BALL_TRACER_SOFTWARE_ROTATE_180", False)


def _decode(buf) -> str:
    try:
        b = memoryview(buf).tobytes()
        i = b.find(b"\x00")
        if i >= 0:
            b = b[:i]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _serial_of(dev_info, tlayer: int) -> str:
    if tlayer == MV_GIGE_DEVICE:
        return _decode(dev_info.SpecialInfo.stGigEInfo.chSerialNumber)
    if tlayer == MV_USB_DEVICE:
        return _decode(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
    return ""


def _model_of(dev_info, tlayer: int) -> str:
    if tlayer == MV_GIGE_DEVICE:
        return _decode(dev_info.SpecialInfo.stGigEInfo.chModelName)
    if tlayer == MV_USB_DEVICE:
        return _decode(dev_info.SpecialInfo.stUsb3VInfo.chModelName)
    return "Unknown"


def _check(ret: int, what: str) -> None:
    if int(ret) != 0:
        raise RuntimeError(f"{what} failed, ret=0x{int(ret):08X}")


# ───────────────── 数据结构 ─────────────────


@dataclass
class Frame:
    """单帧图像数据。"""
    data: bytes
    width: int
    height: int
    frame_num: int
    pixel_type: int
    dev_timestamp: int = 0       # 设备硬件时间戳 tick（nDevTimeStampHigh<<32 | nDevTimeStampLow）
    host_timestamp: int = 0      # SDK 主机时间戳（nHostTimeStamp, ms）
    exposure_time: float = 0.0   # 曝光时间 μs（fExposureTime）
    lost_packet: int = 0         # 本帧丢包数（nLostPacket）
    arrival_perf: float = 0.0    # time.perf_counter() 到达时刻
    exposure_start_pc: float = 0.0  # 曝光开始时刻（PC perf_counter 时间轴，秒）


# ───────────────── 时间校准 ─────────────────


# GevTimestampTickFrequency，MV-CS050-10GC 为 100 MHz
_TICK_FREQ = 100_000_000


def calibrate_time_offset(cam: Any) -> float:
    """
    校准 PC perf_counter 时间与相机设备时间的偏移量。

    通过 GevTimestampControlLatch 锁存设备时间戳并与 PC 时间比对。
    返回 offset（秒），使得:
        曝光时刻_perf = dev_timestamp / TICK_FREQ + offset

    为减少网络延迟误差，取 3 次中往返最短的一次。
    """
    from CameraParams_header import MVCC_INTVALUE_EX

    best_offset = 0.0
    best_rtt = float("inf")

    for _ in range(3):
        t_before = time.perf_counter()
        cam.MV_CC_SetCommandValue("GevTimestampControlLatch")
        t_after = time.perf_counter()

        val = MVCC_INTVALUE_EX()
        ret = cam.MV_CC_GetIntValueEx("GevTimestampValue", val)
        if ret != 0:
            continue

        rtt = t_after - t_before
        pc_time = (t_before + t_after) / 2.0
        dev_time = int(val.nCurValue) / _TICK_FREQ
        offset = pc_time - dev_time

        if rtt < best_rtt:
            best_rtt = rtt
            best_offset = offset

    return best_offset


# ───────────────── 像素转换 ─────────────────


def frame_to_numpy(frame: Frame, *, rotate_180: Optional[bool] = None):
    """将 Frame 转为 numpy BGR / 灰度图像。默认做 180° 旋转，支持 Mono8、BayerXX8、RGB8、BGR8。"""
    import cv2
    import numpy as np

    if rotate_180 is None:
        rotate_180 = _ENV_SOFTWARE_ROTATE_180

    pt = frame.pixel_type
    w, h = frame.width, frame.height

    # Mono8
    if pt == PixelType_Gvsp_Mono8:
        img = np.frombuffer(frame.data, dtype=np.uint8, count=w * h).reshape(h, w)
        return cv2.rotate(img, cv2.ROTATE_180) if rotate_180 else img

    # BGR8 — 直接 reshape
    if pt == PixelType_Gvsp_BGR8_Packed:
        img = np.frombuffer(frame.data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        return cv2.rotate(img, cv2.ROTATE_180) if rotate_180 else img

    # RGB8 → BGR
    if pt == PixelType_Gvsp_RGB8_Packed:
        rgb = np.frombuffer(frame.data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.rotate(img, cv2.ROTATE_180) if rotate_180 else img

    # Bayer 8-bit → BGR
    # 海康 GigE Vision 的 Bayer 命名与 OpenCV 相反（RG↔BG, GR↔GB）。
    # 若最终需要 180° 旋转，先旋转 raw Bayer 再解码，可避免对 3 通道 BGR 再做一次大内存旋转。
    bayer_codes = {
        PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
        PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
        PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
        PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
    }
    rotated_bayer_codes = {
        PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerRG2BGR,
        PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerBG2BGR,
        PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGR2BGR,
        PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGB2BGR,
    }
    if pt in bayer_codes:
        raw = np.frombuffer(frame.data, dtype=np.uint8, count=w * h).reshape(h, w)
        if rotate_180:
            raw = cv2.rotate(raw, cv2.ROTATE_180)
            return cv2.cvtColor(raw, rotated_bayer_codes[pt])
        return cv2.cvtColor(raw, bayer_codes[pt])

    raise RuntimeError(f"不支持的像素格式 0x{pt:08X}，请在 MVS Client 中切换为 Mono8 或 BayerXX8")


# ───────────────── 设备枚举 ─────────────────


def _enum_devices():
    """返回 (st_dev_list, [(index, serial, model, tlayer), ...])。不做 Initialize/Finalize。"""
    st_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_list)
    if ret != 0:
        raise RuntimeError(f"枚举设备失败 ret=0x{ret:08X}")
    devs = []
    for i in range(int(st_list.nDeviceNum)):
        info = cast(st_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        t = int(info.nTLayerType)
        devs.append((i, _serial_of(info, t), _model_of(info, t), t))
    return st_list, devs


def list_devices() -> list[tuple[int, str, str]]:
    """枚举设备，返回 [(index, serial, model), ...]。"""
    MvCamera.MV_CC_Initialize()
    try:
        _, devs = _enum_devices()
        return [(i, s, m) for i, s, m, _t in devs]
    finally:
        MvCamera.MV_CC_Finalize()


# ───────────────── 相机打开 / 关闭 ─────────────────


def open_camera(
    serial: str,
    *,
    trigger_source: Optional[str] = "Software",
    trigger_activation: str = "FallingEdge",
    line_output: str = "",
    line_source: str = "",
    frame_rate: float = 0.0,
    full_frame: bool = False,
    exposure_us: float = 0.0,
    gain_db: float = -1.0,
    pixel_format: str = "",
    roi_offset_y: int = 0,
    roi_height: int = 0,
    roi_width: int = 0,
    binning: int = 1,
    reverse_x: Optional[bool] = None,
    reverse_y: Optional[bool] = None,
    _st_dev_list=None,
) -> Any:
    """
    打开指定序列号的相机并 StartGrabbing。

    Args:
        serial: 相机序列号。
        trigger_source: 触发源 — "Software"/"Line0" 等；None 表示 free-run 连续采集。
        trigger_activation: 硬触发沿 — "FallingEdge" / "RisingEdge"（软触发时忽略）。
        line_output: 主相机输出线（如 "Line1"），用于输出曝光信号触发从机。空字符串跳过。
        line_source: 输出源（如 "ExposureStartActive"）。与 line_output 配合使用。
        frame_rate: 帧率（fps），>0 时设置 AcquisitionFrameRate（仅 free-run 有效）。
        full_frame: True 时重置 ROI 为传感器最大分辨率。
        _st_dev_list: 内部复用，避免重复枚举。

    Returns:
        MvCamera 实例（已 StartGrabbing）。
    """
    serial = str(serial).strip()

    if reverse_x is None:
        reverse_x = _ENV_CAMERA_REVERSE_X
        if reverse_x is None:
            reverse_x = _ENV_CAMERA_REVERSE_180
    if reverse_y is None:
        reverse_y = _ENV_CAMERA_REVERSE_Y
        if reverse_y is None:
            reverse_y = _ENV_CAMERA_REVERSE_180

    if _st_dev_list is None:
        st_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_list)
        _check(ret, "EnumDevices")
    else:
        st_list = _st_dev_list

    for i in range(int(st_list.nDeviceNum)):
        dev_info = cast(st_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        tlayer = int(dev_info.nTLayerType)
        if _serial_of(dev_info, tlayer) != serial:
            continue

        cam = MvCamera()
        _check(cam.MV_CC_CreateHandle(dev_info), f"CreateHandle({serial})")

        ret = cam.MV_CC_OpenDevice(3, 0)  # MV_ACCESS_Control
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"OpenDevice({serial}) 失败 ret=0x{ret:08X}")

        # GigE 网络优化
        if tlayer == MV_GIGE_DEVICE:
            try:
                ps = cam.MV_CC_GetOptimalPacketSize()
                if int(ps) > 0:
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", ps)
            except Exception:
                pass
            # 丢包重传（参考 mvs-dev/camera.py _best_effort_gige_network_tuning）
            try:
                cam.MV_GIGE_SetResend(True, 100, 50)
            except Exception:
                pass

        # ── Binning（像素合并，降低分辨率但保持 FOV）──
        # 必须在 full_frame 之前设置，因为 binning 会改变 WidthMax/HeightMax
        if binning > 1:
            # 先缩小 Width/Height 以免超出 binning 后的最大值
            try:
                cam.MV_CC_SetIntValue("OffsetX", 0)
                cam.MV_CC_SetIntValue("OffsetY", 0)
                cam.MV_CC_SetIntValue("Width", 16)
                cam.MV_CC_SetIntValue("Height", 16)
            except Exception:
                pass
            try:
                cam.MV_CC_SetEnumValueByString("BinningHorizontalMode", "Average")
            except Exception:
                pass
            try:
                cam.MV_CC_SetEnumValueByString("BinningVerticalMode", "Average")
            except Exception:
                pass
            try:
                cam.MV_CC_SetIntValue("BinningHorizontal", binning)
            except Exception:
                pass
            try:
                cam.MV_CC_SetIntValue("BinningVertical", binning)
            except Exception:
                pass

        # ── 全画幅 ROI 重置（binning 后的最大分辨率）──
        if full_frame:
            try:
                cam.MV_CC_SetIntValue("OffsetX", 0)
                cam.MV_CC_SetIntValue("OffsetY", 0)
                from CameraParams_header import MVCC_INTVALUE
                w_param = MVCC_INTVALUE()
                cam.MV_CC_GetIntValue("WidthMax", w_param)
                cam.MV_CC_SetIntValue("Width", int(w_param.nCurValue))
                h_param = MVCC_INTVALUE()
                cam.MV_CC_GetIntValue("HeightMax", h_param)
                cam.MV_CC_SetIntValue("Height", int(h_param.nCurValue))
            except Exception:
                pass

        # ── 触发配置 ──
        if trigger_source is None:
            # Free-run 连续采集（关闭触发模式）
            _check(cam.MV_CC_SetEnumValueByString("TriggerMode", "Off"), "TriggerMode=Off")
            if frame_rate > 0:
                try:
                    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                except Exception:
                    pass
                try:
                    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate))
                except Exception:
                    pass
        else:
            # 触发模式（Software / Line0 等）
            _check(cam.MV_CC_SetEnumValueByString("TriggerMode", "On"), "TriggerMode")
            _check(cam.MV_CC_SetEnumValueByString("TriggerSource", trigger_source), f"TriggerSource={trigger_source}")

            # 硬触发才需要设置触发沿 + 输入线配置
            if trigger_source.lower() not in ("software", "triggersoftware"):
                try:
                    cam.MV_CC_SetEnumValueByString("TriggerActivation", trigger_activation)
                except Exception:
                    pass
                # 选择输入线并设置滤波时间（防误触发）
                try:
                    cam.MV_CC_SetEnumValueByString("LineSelector", trigger_source)
                    cam.MV_CC_SetIntValueEx("LineDebouncerTime", 50)
                except Exception:
                    pass

            try:
                cam.MV_CC_SetFloatValue("TriggerDelay", 0.0)
            except Exception:
                pass
            try:
                cam.MV_CC_SetBoolValue("TriggerCacheEnable", False)
            except Exception:
                pass

        # ── 主相机 Line 输出（参考 MVS SDK ParametrizeCamera_AreaScanIOSettings）──
        if line_output and line_source:
            try:
                cam.MV_CC_SetEnumValueByString("LineSelector", line_output)
                # 尝试 Output / Strobe
                ret2 = cam.MV_CC_SetEnumValueByString("LineMode", "Output")
                if int(ret2) != 0:
                    cam.MV_CC_SetEnumValueByString("LineMode", "Strobe")
                # 显式清零残留 strobe 参数，避免机内持久化状态引入不确定输出。
                try:
                    cam.MV_CC_SetBoolValue("LineInverter", False)
                except Exception:
                    pass
                cam.MV_CC_SetEnumValueByString("LineSource", line_source)
                for node_name in ("StrobeLineDuration", "StrobeLineDelay", "StrobeLinePreDelay"):
                    try:
                        cam.MV_CC_SetIntValueEx(node_name, 0)
                    except Exception:
                        pass
                # 开启输出使能（StrobeEnable）
                cam.MV_CC_SetBoolValue("StrobeEnable", True)
            except Exception:
                pass

        if pixel_format:
            _check(
                cam.MV_CC_SetEnumValueByString("PixelFormat", pixel_format),
                f"PixelFormat={pixel_format}",
            )

        # ── 曝光 / 增益 ──
        if exposure_us > 0:
            try:
                cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            except Exception:
                pass
            try:
                cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
            except Exception:
                pass

        if gain_db >= 0:
            try:
                cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            except Exception:
                pass
            try:
                cam.MV_CC_SetFloatValue("Gain", float(gain_db))
            except Exception:
                pass

        # ── 自定义 ROI（X 轴：宽度裁剪）──
        if roi_width > 0:
            try:
                cam.MV_CC_SetIntValue("OffsetX", 0)
                cam.MV_CC_SetIntValue("Width", roi_width)
            except Exception:
                pass

        # ── 自定义 ROI（Y 轴）──
        if roi_offset_y > 0 or roi_height > 0:
            try:
                cam.MV_CC_SetIntValue("OffsetY", 0)
                if roi_height > 0:
                    cam.MV_CC_SetIntValue("Height", roi_height)
                elif roi_offset_y > 0:
                    from CameraParams_header import MVCC_INTVALUE as _IV
                    h_max = _IV()
                    cam.MV_CC_GetIntValue("HeightMax", h_max)
                    cam.MV_CC_SetIntValue("Height", int(h_max.nCurValue) - roi_offset_y)
                if roi_offset_y > 0:
                    cam.MV_CC_SetIntValue("OffsetY", roi_offset_y)
            except Exception:
                pass

        # 相机侧 180° 反转必须在 StartGrabbing 之前设置。
        if reverse_x is not None:
            _check(cam.MV_CC_SetBoolValue("ReverseX", bool(reverse_x)), f"ReverseX={reverse_x}")
        if reverse_y is not None:
            _check(cam.MV_CC_SetBoolValue("ReverseY", bool(reverse_y)), f"ReverseY={reverse_y}")

        _check(cam.MV_CC_StartGrabbing(), f"StartGrabbing({serial})")
        return cam

    raise RuntimeError(f"未找到序列号为 {serial} 的相机")


def close_camera(cam: Any) -> None:
    """停止取流、关闭设备、销毁句柄。"""
    try:
        cam.MV_CC_StopGrabbing()
    except Exception:
        pass
    try:
        cam.MV_CC_CloseDevice()
    except Exception:
        pass
    try:
        cam.MV_CC_DestroyHandle()
    except Exception:
        pass


# ───────────────── Grabber ─────────────────


class ImageGrabber(threading.Thread):
    """
    单相机抓取器：后台取流入队（最大 10 张），提供 get_frame() 取图出队。

    自动校准 PC 与设备时钟偏移，每帧计算 exposure_start_pc（PC perf_counter 时间轴）。
    周期性校准在独立后台线程执行，不阻塞取帧热循环。
    """

    MAX_QUEUE = 10

    def __init__(
        self,
        cam: Any,
        serial: str = "",
        *,
        timeout_ms: int = 1000,
        recalib_every: int = 10,
        fps: float = 0.0,
    ):
        super().__init__(daemon=True, name=f"Grabber-{serial}")
        self._cam = cam
        self.serial = serial
        self._timeout_ms = int(timeout_ms)
        self._queue: deque[Frame] = deque(maxlen=self.MAX_QUEUE)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recalib_every = recalib_every
        self._fps = float(fps)
        self._time_offset: float = 0.0       # PC perf_counter - dev_time（秒）
        self._calib_thread: Optional[threading.Thread] = None

    def run(self) -> None:
        # 初始校准（取帧前同步执行）
        self._time_offset = calibrate_time_offset(self._cam)

        # 启动后台校准线程
        if self._recalib_every > 0:
            self._calib_thread = threading.Thread(
                target=self._recalib_loop, daemon=True,
                name=f"Calib-{self.serial}",
            )
            self._calib_thread.start()

        st_frame = MV_FRAME_OUT()
        while not self._stop_event.is_set():
            memset(byref(st_frame), 0, sizeof(st_frame))
            ret = self._cam.MV_CC_GetImageBuffer(st_frame, self._timeout_ms)
            if ret != 0:
                continue

            try:
                arrival_perf = time.perf_counter()
                info = st_frame.stFrameInfo
                w = int(info.nExtendWidth) if info.nExtendWidth else int(info.nWidth)
                h = int(info.nExtendHeight) if info.nExtendHeight else int(info.nHeight)
                flen = int(info.nFrameLenEx) if info.nFrameLenEx else int(info.nFrameLen)
                data = ctypes.string_at(st_frame.pBufAddr, flen) if flen > 0 else b""
                dev_ts = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)

                # 计算 PC 时间轴上的曝光开始时刻（读取 offset 是原子操作，无需加锁）
                exposure_start_pc = dev_ts / _TICK_FREQ + self._time_offset

                frame = Frame(
                    data=data, width=w, height=h,
                    frame_num=int(info.nFrameNum),
                    pixel_type=int(info.enPixelType),
                    dev_timestamp=dev_ts,
                    host_timestamp=int(info.nHostTimeStamp),
                    exposure_time=float(info.fExposureTime),
                    lost_packet=int(info.nLostPacket),
                    arrival_perf=arrival_perf,
                    exposure_start_pc=exposure_start_pc,
                )
                with self._lock:
                    self._queue.append(frame)
            finally:
                try:
                    self._cam.MV_CC_FreeImageBuffer(st_frame)
                except Exception:
                    pass

    def _recalib_loop(self) -> None:
        """后台线程：按帧间隔周期性重新校准 PC-设备时钟偏移。"""
        # 按真实帧率把“每多少帧重校准一次”换算成秒，避免魔法常数带来的抖动。
        if self._fps > 0:
            interval = max(self._recalib_every / self._fps, 0.5)
        else:
            interval = max(self._recalib_every * 0.043, 0.5)
        while not self._stop_event.is_set():
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break
            new_offset = calibrate_time_offset(self._cam)
            # 平滑更新：如果新旧 offset 差异过大（>5ms），可能是测量异常，跳过
            if abs(new_offset - self._time_offset) < 0.005:
                self._time_offset = new_offset

    def get_frame(self, timeout_s: float = 0.0) -> Optional[Frame]:
        """从队列取出一张图并出队。timeout_s=0 非阻塞。"""
        if timeout_s <= 0:
            with self._lock:
                return self._queue.popleft() if self._queue else None

        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            with self._lock:
                if self._queue:
                    return self._queue.popleft()
            time.sleep(0.005)
        return None

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stop(self) -> None:
        self._stop_event.set()

    def _drain_keep_latest(self) -> Optional[Frame]:
        """清空队列，仅返回最新帧。"""
        with self._lock:
            if not self._queue:
                return None
            latest = self._queue[-1]
            self._queue.clear()
            return latest


# ───────────────── 同步采集 ─────────────────


class SyncCapture:
    """
    多相机硬触发同步采集上下文管理器。

    主相机 free-run + Line 输出 ExposureStartActive → 从相机 Line0 FallingEdge 硬触发。
    每台从相机的 ROI 参数通过 slave_params 独立配置。

    推荐用法（从配置文件创建）::

        with SyncCapture.from_config() as cap:
            frames = cap.get_frames(timeout_s=1.0)  # {serial: Frame}
            for sn, f in frames.items():
                img = frame_to_numpy(f)  # numpy BGR
    """

    def __init__(
        self,
        master_serial: str,
        slave_serials: list[str],
        *,
        fps: float = 30.0,
        sync_threshold_ms: float = 200.0,
        master_line_output: str = "Line1",
        master_line_source: str = "ExposureStartActive",
        slave_trigger_source: str = "Line0",
        slave_trigger_activation: str = "FallingEdge",
        recalib_every: int = 10,
        exposure_us: float = 0.0,
        gain_db: float = -1.0,
        pixel_format: str = "",
        master_min_bandwidth: bool = False,
        slave_params: Optional[dict[str, dict]] = None,
    ):
        self._master = master_serial
        self._slaves = list(slave_serials)
        self._all_serials = [master_serial] + self._slaves
        # 同步帧集合：master_min_bandwidth 时主机仅做触发源，不参与同步输出
        if master_min_bandwidth:
            self._sync_serials = list(slave_serials)
        else:
            self._sync_serials = [master_serial] + list(slave_serials)
        self._sync_set = set(self._sync_serials)
        self._fps = fps
        self._recalib_every = recalib_every
        self._sync_threshold_s = sync_threshold_ms / 1000.0

        self._cameras: dict[str, Any] = {}
        self._grabbers: dict[str, ImageGrabber] = {}
        self._stop = threading.Event()

        MvCamera.MV_CC_Initialize()

        st_dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_dev_list)
        _check(ret, "EnumDevices")

        try:
            # 主相机 free-run + Line 输出触发信号
            self._cameras[master_serial] = open_camera(
                master_serial,
                trigger_source=None,
                frame_rate=fps,
                full_frame=True,
                line_output=master_line_output,
                line_source=master_line_source,
                exposure_us=exposure_us,
                gain_db=gain_db,
                pixel_format=pixel_format,
                roi_height=16 if master_min_bandwidth else 0,
                _st_dev_list=st_dev_list,
            )
            # 从机硬触发（每台相机独立 ROI 参数）
            for sn in self._slaves:
                params = (slave_params or {}).get(sn, {})
                self._cameras[sn] = open_camera(
                    sn,
                    trigger_source=slave_trigger_source,
                    trigger_activation=slave_trigger_activation,
                    full_frame=True,
                    exposure_us=exposure_us,
                    gain_db=gain_db,
                    pixel_format=pixel_format,
                    roi_offset_y=params.get("roi_offset_y", 0),
                    roi_height=params.get("roi_height", 0),
                    roi_width=params.get("roi_width", 0),
                    binning=params.get("binning", 1),
                    _st_dev_list=st_dev_list,
                )

            # 启动 grabber
            for sn, cam in self._cameras.items():
                g = ImageGrabber(cam, sn, timeout_ms=1000,
                                 recalib_every=self._recalib_every,
                                 fps=self._fps)
                self._grabbers[sn] = g
                g.start()
        except Exception:
            self.close()
            raise

    def get_frames(self, timeout_s: float = 1.0) -> Optional[dict[str, Frame]]:
        """
        获取最新一组同步帧。

        仅返回 sync_serials 中的相机帧（master_min_bandwidth 时不含主机）。
        用 exposure_start_pc 判断是否属于同一触发周期（阈值 10ms）。

        返回 {serial: Frame} 字典，或超时返回 None。
        """
        deadline = time.perf_counter() + timeout_s
        latest: dict[str, Frame] = {}

        while time.perf_counter() < deadline:
            for sn, g in self._grabbers.items():
                f = g._drain_keep_latest()
                # 仅收集 sync_serials 的帧；其他 grabber 照常排空
                if f is not None and sn in self._sync_set:
                    latest[sn] = f

            if len(latest) == len(self._sync_serials):
                # 粗同步检查：arrival_perf spread
                arrivals = [f.arrival_perf for f in latest.values()]
                spread = max(arrivals) - min(arrivals)
                if spread <= self._sync_threshold_s:
                    # 精同步检查：exposure_start_pc 是否属于同一触发周期
                    exp_starts = [f.exposure_start_pc for f in latest.values()]
                    exp_spread = max(exp_starts) - min(exp_starts)
                    if exp_spread > 0.010:  # >10ms 说明不是同一触发周期
                        oldest_sn = min(latest, key=lambda sn: latest[sn].exposure_start_pc)
                        del latest[oldest_sn]
                        time.sleep(0.001)
                        continue
                    return latest
                # 帧不同步，丢弃最老的重新收集
                oldest_sn = min(latest, key=lambda sn: latest[sn].arrival_perf)
                del latest[oldest_sn]

            time.sleep(0.001)

        return None

    @classmethod
    def from_config(cls, config_path: str = "", **overrides):
        """从 JSON 配置文件创建 SyncCapture。默认路径 config/camera.json。"""
        import json
        if not config_path:
            config_path = str(Path(__file__).resolve().parent / "config" / "camera.json")
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(overrides)
        master = cfg.pop("master_serial")
        slaves = cfg.pop("slave_serials")
        return cls(master, slaves, **cfg)

    @property
    def serials(self) -> list[str]:
        """所有打开的相机序列号列表（主机在前）。"""
        return list(self._all_serials)

    @property
    def sync_serials(self) -> list[str]:
        """参与同步输出的相机序列号列表（get_frames 返回的相机）。"""
        return list(self._sync_serials)

    @property
    def fps(self) -> float:
        return float(self._fps)

    def close(self) -> None:
        """停止所有 grabber 并关闭所有相机。"""
        self._stop.set()
        for g in self._grabbers.values():
            g.stop()
        for g in self._grabbers.values():
            g.join(timeout=2.0)
        for cam in self._cameras.values():
            close_camera(cam)
        self._cameras.clear()
        self._grabbers.clear()
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
