# -*- coding: utf-8 -*-

"""设备枚举与基本信息抽取。

该模块的定位是“设备目录”：
- 只负责把 SDK 的枚举结果转成可读的 Python 数据；
- 不负责打开相机、不做任何节点配置；
- 这样可以避免枚举逻辑与相机生命周期强耦合。
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from mvs.core.text import decode_c_string

from .binding import MvsBinding


def _bytes_to_str(buf: Iterable[int]) -> str:
    """把 SDK 的定长 byte 数组安全转成字符串。"""

    # 统一走 decode_c_string，保持编码策略一致。
    return decode_c_string(buf)


def _ip_int_to_str(ip: int) -> str:
    """把 GigE 的 32-bit IPv4（大端）转成点分十进制字符串。"""
    parts = [str((ip >> shift) & 0xFF) for shift in (24, 16, 8, 0)]
    return ".".join(parts)


@dataclass(frozen=True, slots=True)
class DeviceDesc:
    """设备的简化描述。

    字段说明：
    - index: 该设备在 SDK 枚举列表里的下标（后续 CreateHandle 需要）。
    - tlayer_type: 设备类型（GigE/USB 等，来自 nTLayerType）。
    - model: 型号名称（例如 MV-CS050-10GC）。
    - serial: 序列号（常用于定位设备，建议作为业务侧主键）。
    - user_name: 用户自定义名称（MVS Client 可设置）。
    - ip: GigE 设备的当前 IP（USB 设备为 None）。
    """

    index: int
    tlayer_type: int
    model: str
    serial: str
    user_name: str
    ip: Optional[str]


def enumerate_devices(binding: MvsBinding) -> tuple[Any, List[DeviceDesc]]:
    """枚举设备并返回 SDK 的 device list 与简化描述。

    上层在打开相机时，需要把 `st_dev_list` 传回 SDK 的 CreateHandle。

    Notes:
        - 一般需要在调用前先执行 SDK 初始化（`MvsSdk.initialize()` / `MV_CC_Initialize()`）。
          若未初始化，不同版本的 SDK 可能会返回错误码或表现不一致。

    Args:
        binding: 已加载的 MVS 绑定。

    Returns:
        一个二元组：
        - st_dev_list: SDK 原始设备列表结构体（后续 CreateHandle 需要）。
        - descs: 对设备信息的简化描述列表。

    Raises:
        RuntimeError: 枚举失败。
    """

    st_dev_list = binding.params.MV_CC_DEVICE_INFO_LIST()
    ret = binding.MvCamera.MV_CC_EnumDevices(
        binding.MV_GIGE_DEVICE | binding.MV_USB_DEVICE, st_dev_list
    )
    if int(ret) != binding.MV_OK:
        raise RuntimeError(f"MV_CC_EnumDevices failed, ret=0x{int(ret):08X}")

    descs: List[DeviceDesc] = []
    for i in range(int(st_dev_list.nDeviceNum)):
        dev_info = ctypes.cast(
            st_dev_list.pDeviceInfo[i], ctypes.POINTER(binding.params.MV_CC_DEVICE_INFO)
        ).contents

        tlayer = int(dev_info.nTLayerType)
        model = ""
        serial = ""
        user_name = ""
        ip: Optional[str] = None

        if tlayer == binding.params.MV_GIGE_DEVICE:
            gigE = dev_info.SpecialInfo.stGigEInfo
            model = _bytes_to_str(gigE.chModelName)
            serial = _bytes_to_str(gigE.chSerialNumber)
            user_name = _bytes_to_str(gigE.chUserDefinedName)
            ip = _ip_int_to_str(int(gigE.nCurrentIp))
        elif tlayer == binding.params.MV_USB_DEVICE:
            usb = dev_info.SpecialInfo.stUsb3VInfo
            model = _bytes_to_str(usb.chModelName)
            serial = _bytes_to_str(usb.chSerialNumber)
            user_name = _bytes_to_str(usb.chUserDefinedName)
        else:
            model = "UNKNOWN"

        descs.append(
            DeviceDesc(
                index=i,
                tlayer_type=tlayer,
                model=model,
                serial=serial,
                user_name=user_name,
                ip=ip,
            )
        )

    return st_dev_list, descs
