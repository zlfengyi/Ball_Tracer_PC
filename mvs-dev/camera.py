# -*- coding: utf-8 -*-

"""相机生命周期封装：创建句柄、打开、配置、取流开关、关闭回收。

该模块只做“相机层”的封装：
- 负责把 SDK 的句柄创建/打开/关闭/销毁串起来；
- 提供触发相关的最小配置函数；
- 对外暴露可复用的数据结构（`MvsCamera`）和错误类型（`MvsError`）。
"""

from __future__ import annotations

import ctypes
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from mvs.core._cleanup import best_effort
from mvs.core.events import MvsEvent
from mvs.core.text import decode_c_string

from .binding import MvsBinding


class MvsError(RuntimeError):
    pass


def _check(ret: int, ok: int, what: str) -> None:
    if int(ret) != int(ok):
        raise MvsError(f"{what} failed, ret=0x{int(ret):08X}")


def _get_int_node_info(*, binding: MvsBinding, cam: Any, key: str) -> Optional[tuple[int, int, int, int]]:
    """读取 int 节点的当前值/最小/最大/步进。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄。
        key: 节点名，例如 "Width" / "Height"。

    Returns:
        (cur, min, max, inc)。读取失败返回 None。
    """

    try:
        st = binding.params.MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue(str(key), st)
        if int(ret) != int(binding.MV_OK):
            return None
        cur = int(getattr(st, "nCurValue"))
        vmin = int(getattr(st, "nMin"))
        vmax = int(getattr(st, "nMax"))
        inc = int(getattr(st, "nInc"))
        return cur, vmin, vmax, inc
    except Exception:
        return None


def _align_down(value: int, *, vmin: int, vmax: int, inc: int) -> int:
    """把目标值裁剪到[min,max]并按 inc 向下对齐。

    Notes:
        - 很多 MVS 节点都有步进（inc），例如 WidthInc=4/8/16。
        - 为了稳定降低带宽，这里默认“向下对齐”，避免因为对齐导致分辨率反而变大。
    """

    if inc <= 0:
        inc = 1

    v = int(value)
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax

    # 对齐到以 vmin 为起点的步进格。
    steps = (v - vmin) // inc
    return int(vmin + steps * inc)


def configure_resolution(
    *,
    binding: MvsBinding,
    cam: Any,
    width: int,
    height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    """配置相机输出分辨率（ROI）。

    这里的“下调分辨率”通常指 ROI 裁剪（改变传输/取流的图像宽高），不是把图像做缩放。
    ROI 能显著降低带宽与 CPU 压力，但视场(FOV)会变小。

    Notes:
        - 建议在 StartGrabbing 之前设置（采集开始后很多节点会被锁定为不可写）。
        - Width/Height/OffsetX/OffsetY 往往有步进限制（Inc），比如 8 的倍数。
          本函数会按步进“向下对齐”，尽量不超过你给的目标值。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        width: 目标宽度，例如 1920。
        height: 目标高度，例如 1080。
        offset_x: ROI 左上角 X 偏移（默认 0）。
        offset_y: ROI 左上角 Y 偏移（默认 0）。

    Raises:
        MvsError: 读取节点信息失败或写入失败。
        ValueError: width/height 非正。
    """

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width/height 必须为正")

    def _try_set_int_node(key: str, value: int) -> bool:
        """尽力设置 int 节点。

        说明：
            - MVS 的 SetIntValue 通常用返回码表示失败，并不一定抛异常。
            - ROI 场景下我们更希望“尽力修正历史状态”，而不是因为某个节点不可写就中断流程。
        """

        try:
            ret = int(cam.MV_CC_SetIntValue(str(key), int(value)))
        except Exception:
            return False
        return int(ret) == int(binding.MV_OK)

    # 关键点：Width/Height 的最大值经常受当前 OffsetX/OffsetY 影响。
    # 如果相机之前被配置过非零 offset，那么在读取 Width/Height 范围时会看到被“缩小”的 max。
    # 因此先尽力把 Offset 归零，再读取 Width/Height 的范围来计算目标值。
    _try_set_int_node("OffsetX", 0)
    _try_set_int_node("OffsetY", 0)

    w_info = _get_int_node_info(binding=binding, cam=cam, key="Width")
    h_info = _get_int_node_info(binding=binding, cam=cam, key="Height")
    if w_info is None or h_info is None:
        raise MvsError("无法读取 Width/Height 节点信息：该机型可能不支持 ROI，或当前节点不可读。")

    w_cur, w_min, w_max, w_inc = w_info
    h_cur, h_min, h_max, h_inc = h_info

    target_w = _align_down(int(width), vmin=w_min, vmax=w_max, inc=w_inc)
    target_h = _align_down(int(height), vmin=h_min, vmax=h_max, inc=h_inc)

    def _set_or_accept_readonly_if_already(key: str, *, target: int, cur_before: int) -> None:
        """设置 int 节点；若节点不可写但值已是目标值，则允许继续。

        背景：
            部分机型在固定 AOI（不可改 Width/Height，只能改 OffsetX/OffsetY）时，
            对 Width/Height 写入会返回失败码，但读取到的 cur 值已满足目标。
            在这种情况下，继续执行后续 Offset 设置比直接报错更符合工程预期。

        约束：
            仅当 *写前* 与 *写后* 的 cur 都等于 target 时才吞掉错误，避免掩盖真实配置失败。
        """

        ret = int(cam.MV_CC_SetIntValue(str(key), int(target)))
        if int(ret) == int(binding.MV_OK):
            return

        after = _get_int_node_info(binding=binding, cam=cam, key=str(key))
        cur_after = int(after[0]) if after is not None else None

        if int(cur_before) == int(target) and (cur_after is not None) and int(cur_after) == int(target):
            return

        _check(ret, binding.MV_OK, f"SetIntValue({key}={target})")

    _set_or_accept_readonly_if_already("Width", target=int(target_w), cur_before=int(w_cur))
    _set_or_accept_readonly_if_already("Height", target=int(target_h), cur_before=int(h_cur))

    # 最后再设置用户请求的偏移（如果节点存在）。偏移同样可能有步进限制。
    ox_info = _get_int_node_info(binding=binding, cam=cam, key="OffsetX")
    oy_info = _get_int_node_info(binding=binding, cam=cam, key="OffsetY")

    if ox_info is not None:
        _, ox_min, ox_max, ox_inc = ox_info
        ox = _align_down(int(offset_x), vmin=ox_min, vmax=ox_max, inc=ox_inc)
        ret = cam.MV_CC_SetIntValue("OffsetX", int(ox))
        _check(ret, binding.MV_OK, f"SetIntValue(OffsetX={ox})")

    if oy_info is not None:
        _, oy_min, oy_max, oy_inc = oy_info
        oy = _align_down(int(offset_y), vmin=oy_min, vmax=oy_max, inc=oy_inc)
        ret = cam.MV_CC_SetIntValue("OffsetY", int(oy))
        _check(ret, binding.MV_OK, f"SetIntValue(OffsetY={oy})")

    # 二次尝试：某些机型在 offset 调整后，Width/Height 的 max 会变化。
    # 这一步用于修复“历史 offset 影响 max，导致宽高被误 clamp”的情况。
    # 若二次设置失败则保持现状（避免改变既有错误处理行为）。
    w2 = _get_int_node_info(binding=binding, cam=cam, key="Width")
    h2 = _get_int_node_info(binding=binding, cam=cam, key="Height")
    if w2 is not None and h2 is not None:
        cur_w, w_min2, w_max2, w_inc2 = w2
        cur_h, h_min2, h_max2, h_inc2 = h2
        target_w2 = _align_down(int(width), vmin=w_min2, vmax=w_max2, inc=w_inc2)
        target_h2 = _align_down(int(height), vmin=h_min2, vmax=h_max2, inc=h_inc2)

        if int(cur_w) != int(target_w2):
            _try_set_int_node("Width", int(target_w2))
        if int(cur_h) != int(target_h2):
            _try_set_int_node("Height", int(target_h2))


def configure_pixel_format(*, binding: MvsBinding, cam: Any, pixel_format: str) -> str:
    """配置相机输出像素格式（用于降低带宽/算力）。

    常见经验：
        - `RGB8Packed/BGR8Packed` 通常是 24bpp，带宽占用很高。
        - `Mono8` 或 `BayerRG8/BayerBG8/...` 通常是 8bpp，带宽大幅下降。

    Notes:
        - 不同机型/固件支持的 PixelFormat 枚举不同。
        - 建议在 StartGrabbing 之前设置（采集开始后很多节点会锁定不可写）。
        - 为了兼容不同彩色相机的 Bayer 排列差异，本函数支持传入候选列表，
          例如："BayerRG8,BayerBG8"，会按顺序尝试，直到成功。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄。
        pixel_format: 目标像素格式（或候选列表）。支持用逗号/竖线分隔多个候选。

    Returns:
        实际成功设置的像素格式名称。

    Raises:
        MvsError: 设置失败（所有候选都不支持/不可写）。
        ValueError: pixel_format 为空。
    """

    raw = str(pixel_format or "").strip()
    if not raw:
        raise ValueError("pixel_format 不能为空")

    # 支持："A,B" 或 "A|B" 或混用。
    seps = [",", "|"]
    candidates: list[str] = [raw]
    for sep in seps:
        if sep in raw:
            candidates = [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
            break

    last_ret: Optional[int] = None
    for cand in candidates:
        try:
            ret = int(cam.MV_CC_SetEnumValueByString("PixelFormat", str(cand)))
        except Exception as exc:
            # 直接异常通常意味着节点不存在/不可写/SDK 状态异常。
            raise MvsError(f"PixelFormat 设置异常：{exc}") from exc

        last_ret = ret
        if int(ret) == int(binding.MV_OK):
            return str(cand)

    tried = "/".join(candidates)
    raise MvsError(
        f"PixelFormat 设置失败：尝试了 {tried} 均失败，ret=0x{int(last_ret or 0):08X}。\n"
        "建议：用 MVS Client 打开相机，在 PixelFormat 下拉框确认该机型支持的枚举值；\n"
        "常见可尝试：Mono8、BayerRG8/BayerBG8/BayerGR8/BayerGB8、RGB8Packed。"
    )


def _platform_functype() -> Any:
    """返回当前平台的 ctypes 回调函数类型工厂。

    说明：MVS SDK 回调在 Windows 上使用 stdcall，需用 WINFUNCTYPE。
    """

    if sys.platform.startswith("win") and hasattr(ctypes, "WINFUNCTYPE"):
        return ctypes.WINFUNCTYPE
    return ctypes.CFUNCTYPE


def _enable_device_events(
    *,
    binding: MvsBinding,
    cam: Any,
    serial: str,
    event_names: Sequence[str],
    out_q: Optional["queue.Queue[MvsEvent]"],
    callback_keepalive: Dict[str, Any],
) -> List[str]:
    """尽力开启设备事件，并把事件记录写入队列。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        serial: 序列号。
        event_names: 需要订阅的事件名称列表，例如 ["ExposureStart", "ExposureEnd"].
        out_q: 事件队列（可选）。为 None 时不启用。
        callback_keepalive: 用于保活回调对象，避免被 GC。

    Returns:
        实际成功启用的事件名列表。
    """

    if out_q is None:
        return []

    enabled: List[str] = []
    fun_ctype = _platform_functype()
    event_info_ptr = ctypes.POINTER(binding.params.MV_EVENT_OUT_INFO)
    cb_type = fun_ctype(None, event_info_ptr, ctypes.c_void_p)

    for requested in [str(x).strip() for x in event_names if str(x).strip()]:
        # 事件回调：尽量轻量，只做解析与入队。
        def _on_event(p_event_info: Any, p_user: Any, _requested: str = requested) -> None:
            try:
                info = ctypes.cast(p_event_info, event_info_ptr).contents
                event_name = decode_c_string(info.EventName) or _requested
                ts = (int(info.nTimestampHigh) << 32) | int(info.nTimestampLow)
                blk = (int(info.nBlockIdHigh) << 32) | int(info.nBlockIdLow)
                out_q.put_nowait(
                    {
                        "type": "camera_event",
                        "created_at": time.time(),
                        "host_monotonic": time.monotonic(),
                        "serial": str(serial),
                        "event_name": event_name,
                        "requested_event_name": _requested,
                        "event_id": int(info.nEventID),
                        "stream_channel": int(info.nStreamChannel),
                        "block_id": int(blk),
                        "event_timestamp": int(ts),
                    }
                )
            except queue.Full:
                # 事件队列满：丢弃事件（不影响采集主流程）
                return
            except Exception:
                return

        cb = cb_type(_on_event)
        callback_keepalive[requested] = cb

        try:
            ret = cam.MV_CC_EventNotificationOn(requested)
            if int(ret) != int(binding.MV_OK):
                continue
            ret = cam.MV_CC_RegisterEventCallBackEx(requested, cb, None)
            if int(ret) != int(binding.MV_OK):
                continue
        except Exception:
            continue

        enabled.append(requested)

    return enabled


def _best_effort_gige_network_tuning(cam: Any) -> None:
    # 最佳 packet size（仅 GigE 支持）
    try:
        packet_size = int(cam.MV_CC_GetOptimalPacketSize())
        if packet_size > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
    except Exception:
        pass

    try:
        cam.MV_GIGE_SetResend(True, 100, 50)
    except Exception:
        pass


def _is_access_denied(binding: MvsBinding, ret: int) -> bool:
    try:
        return int(ret) == int(binding.err.MV_E_ACCESS_DENIED)
    except Exception:
        return int(ret) == 0x80000203


def configure_trigger(
    *,
    binding: MvsBinding,
    cam: Any,
    trigger_source: str,
    trigger_activation: str,
    trigger_cache_enable: bool,
) -> None:
    """配置触发相关参数。

    参考官方示例 `ParametrizeCamera_AreaScanIOSettings.py`。

    Notes:
        - 软触发（Software）场景下通常不需要/不支持设置 `TriggerActivation`。
        - 不同机型/固件可写属性不一致：本函数对部分属性采用“尽力设置”。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        trigger_source: 触发源，例如 "Line0" / "Software"。
        trigger_activation: 触发沿，例如 "FallingEdge"/"RisingEdge"。
        trigger_cache_enable: 是否启用触发缓存（部分机型不支持）。

    Raises:
        MvsError: 关键参数设置失败（由 `_check()` 抛出）。
    """

    # 参考官方示例 ParametrizeCamera_AreaScanIOSettings.py
    _check(cam.MV_CC_SetEnumValueByString("TriggerMode", "On"), binding.MV_OK, "TriggerMode")
    _check(
        cam.MV_CC_SetEnumValueByString("TriggerSource", trigger_source),
        binding.MV_OK,
        "TriggerSource",
    )

    # 软触发场景下通常不需要/不支持设置 TriggerActivation（会返回 MV_E_GC_ACCESS=0x80000106）。
    if str(trigger_source).lower() not in {"software", "triggersoftware"}:
        _check(
            cam.MV_CC_SetEnumValueByString("TriggerActivation", trigger_activation),
            binding.MV_OK,
            "TriggerActivation",
        )

    try:
        cam.MV_CC_SetFloatValue("TriggerDelay", 0.0)
    except Exception:
        pass

    # 有些机型/固件不支持 TriggerCacheEnable，失败忽略。
    try:
        ret = cam.MV_CC_SetBoolValue("TriggerCacheEnable", bool(trigger_cache_enable))
        if int(ret) != binding.MV_OK:
            return
    except Exception:
        return


def configure_line_output(
    *,
    binding: MvsBinding,
    cam: Any,
    line_selector: str,
    line_source: str,
    line_mode: str = "Output",
) -> None:
    """配置相机某一路 IO 作为输出，并选择输出源。

    典型用途：master/slave 触发链路里，让 master 的 Line1 输出曝光有效信号，去触发 slave 的 Line0。

    Notes:
        - 不同机型/固件支持的枚举值可能不同；如果你遇到失败，请用 MVS Client 先确认可选值。
        - 一般顺序是：LineSelector -> LineMode -> LineSource。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄。
        line_selector: 例如 "Line1"。
        line_source: 例如 "ExposureStartActive"（在 MVS Client UI 中可能显示为 "Exposure Start Active"）。
        line_mode: 通常为 "Output"。

    Raises:
        MvsError: 关键节点设置失败。
    """

    _check(
        cam.MV_CC_SetEnumValueByString("LineSelector", str(line_selector)),
        binding.MV_OK,
        "LineSelector",
    )

    # 机型差异提示：部分相机在 MVS Client 里把“输出模式”叫作 Strobe，
    # 此时 LineMode=Output 会返回 MV_E_PARAMETER(0x80000004)。
    requested_mode = str(line_mode).strip() or "Output"
    mode_candidates: List[str] = [requested_mode]
    if requested_mode.lower() == "output":
        mode_candidates.append("Strobe")
    elif requested_mode.lower() == "strobe":
        mode_candidates.append("Output")

    last_ret = None
    selected_mode: Optional[str] = None
    for cand in mode_candidates:
        try:
            ret = int(cam.MV_CC_SetEnumValueByString("LineMode", str(cand)))
        except Exception as exc:
            raise MvsError(f"LineMode 设置失败：{exc}") from exc

        last_ret = ret
        if int(ret) == int(binding.MV_OK):
            selected_mode = str(cand)
            break

    if selected_mode is None:
        # 只对最常见的错误码做解释，避免引入复杂映射。
        extra = ""
        try:
            if last_ret is not None and int(last_ret) == int(getattr(binding.err, "MV_E_PARAMETER")):
                extra = (
                    "（MV_E_PARAMETER：参数错误。很可能该机型不支持该 LineMode 枚举值；"
                    "可在 MVS Client 的 IO Output 查看 Line Mode 可选值，"
                    "例如尝试 --master-line-mode Strobe）"
                )
        except Exception:
            extra = ""

        tried = "/".join(mode_candidates)
        raise MvsError(
            f"LineMode 设置失败：尝试了 {tried} 均失败，ret=0x{int(last_ret or 0):08X}{extra}"
        )

    # 设置输出源。不同固件在 Strobe 模式下可能使用 StrobeSource 节点。
    source_node_candidates = ["LineSource"]
    if selected_mode.lower() == "strobe":
        source_node_candidates.append("StrobeSource")

    last_ret = None
    ok_source = False
    for node_name in source_node_candidates:
        try:
            ret = int(cam.MV_CC_SetEnumValueByString(str(node_name), str(line_source)))
        except Exception:
            continue
        last_ret = ret
        if int(ret) == int(binding.MV_OK):
            ok_source = True
            break

    if not ok_source:
        tried = "/".join(source_node_candidates)
        raise MvsError(
            f"输出源设置失败：{tried} 均无法设置为 {line_source}，ret=0x{int(last_ret or 0):08X}。"
            "请在 MVS Client 的 IO Output 页面确认该模式下可选的 Source 节点/枚举值。"
        )

    # Strobe 模式下通常需要显式开启 StrobeEnable（有的机型节点名就叫这个）。
    if selected_mode.lower() == "strobe":
        try:
            cam.MV_CC_SetBoolValue("StrobeEnable", True)
        except Exception:
            pass

    # 某些机型支持反相节点；默认尽力关闭反相，避免边沿误判。
    try:
        cam.MV_CC_SetBoolValue("LineInverter", False)
    except Exception:
        pass


def configure_exposure(
    *,
    binding: MvsBinding,
    cam: Any,
    exposure_auto: str = "Off",
    exposure_time_us: Optional[float] = None,
    gain_auto: str = "Off",
    gain: Optional[float] = None,
) -> None:
    """配置曝光/增益。

    该函数刻意保持“最小能力集”：只覆盖工程中最常用的 4 个节点。
    - ExposureAuto / ExposureTime
    - GainAuto / Gain

    Notes:
        - 一般建议先关 Auto，再设置手动值，避免被自动策略覆盖或节点不可写。
        - 不同机型/固件的可写性和单位可能不同：ExposureTime 通常是微秒(us)，Gain 常见是 dB。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        exposure_auto: 自动曝光模式："Off"/"Once"/"Continuous"。
        exposure_time_us: 曝光时间（微秒）。为 None 时不设置。
        gain_auto: 自动增益模式："Off"/"Continuous"（常见）。
        gain: 增益值（常见单位 dB）。为 None 时不设置。

    Raises:
        MvsError: 关键节点设置失败。
    """

    _check(
        cam.MV_CC_SetEnumValueByString("ExposureAuto", str(exposure_auto)),
        binding.MV_OK,
        "ExposureAuto",
    )
    if exposure_time_us is not None:
        _check(
            cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time_us)),
            binding.MV_OK,
            "ExposureTime",
        )

    # 有些机型没有 GainAuto 或不可写：这里仍用 _check，让问题尽早暴露。
    _check(
        cam.MV_CC_SetEnumValueByString("GainAuto", str(gain_auto)),
        binding.MV_OK,
        "GainAuto",
    )
    if gain is not None:
        _check(cam.MV_CC_SetFloatValue("Gain", float(gain)), binding.MV_OK, "Gain")


@dataclass
class MvsCamera:
    """一个已打开并可取流的相机。"""

    binding: MvsBinding
    cam: Any
    serial: str
    tlayer_type: int
    event_names_enabled: List[str]
    _event_callbacks_keepalive: Dict[str, Any]

    @classmethod
    def open_from_device_list(
        cls,
        *,
        binding: MvsBinding,
        st_dev_list: Any,
        dev_index: int,
        serial: str,
        tlayer_type: int,
        trigger_source: str,
        trigger_activation: str,
        trigger_cache_enable: bool,
        event_queue: Optional["queue.Queue[MvsEvent]"] = None,
        event_names: Sequence[str] = (),
        line_output_selector: str = "",
        line_output_source: str = "",
        line_output_mode: str = "Output",
        pixel_format: str = "",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        image_offset_x: int = 0,
        image_offset_y: int = 0,
        exposure_auto: str = "",
        exposure_time_us: Optional[float] = None,
        gain_auto: str = "",
        gain: Optional[float] = None,
    ) -> "MvsCamera":
        """从枚举结果打开相机并启动取流。

        Args:
            binding: 已加载的 MVS 绑定。
            st_dev_list: `MV_CC_EnumDevices` 的输出（SDK 结构体）。
            dev_index: 设备在列表中的下标。
            serial: 设备序列号（用于报错信息和对外标识）。
            tlayer_type: 设备类型（GigE/USB）。
            trigger_source: 触发源。
            trigger_activation: 触发沿（对软触发无效/可能不可写）。
            trigger_cache_enable: 是否启用触发缓存。
            image_width: 可选输出宽度（ROI）。None 表示不设置。
            image_height: 可选输出高度（ROI）。None 表示不设置。
            image_offset_x: ROI 左上角 X 偏移（默认 0）。
            image_offset_y: ROI 左上角 Y 偏移（默认 0）。
            exposure_auto: 自动曝光模式。空字符串表示不设置。
            exposure_time_us: 曝光时间（微秒）。None 表示不设置。
            gain_auto: 自动增益模式。空字符串表示不设置。
            gain: 增益值。None 表示不设置。

        Returns:
            已打开并开始抓取的 `MvsCamera`。

        Raises:
            MvsError: 创建句柄/打开设备/启动取流失败。
        """

        cam = binding.MvCamera()

        dev_info = ctypes.cast(
            st_dev_list.pDeviceInfo[dev_index],
            ctypes.POINTER(binding.params.MV_CC_DEVICE_INFO),
        ).contents

        _check(cam.MV_CC_CreateHandle(dev_info), binding.MV_OK, f"CreateHandle({serial})")
        try:
            # GigE 默认用 Control 权限打开：比 Exclusive 更兼容（允许其它监控连接），也更贴近实际工程使用。
            mv_access_control = 3
            mv_access_monitor = 7
            open_ret = cam.MV_CC_OpenDevice(mv_access_control, 0)  # MV_ACCESS_Control
            if int(open_ret) != int(binding.MV_OK):
                if _is_access_denied(binding, int(open_ret)):
                    # 尝试用 Monitor 打开做诊断：若能打开，说明设备大概率被其它程序以控制权限占用。
                    diag_ret = cam.MV_CC_OpenDevice(mv_access_monitor, 0)  # MV_ACCESS_Monitor
                    if int(diag_ret) == int(binding.MV_OK):
                        try:
                            cam.MV_CC_CloseDevice()
                        except Exception:
                            pass
                        raise MvsError(
                            f"OpenDevice({serial}) failed, ret=0x{int(open_ret):08X} (MV_E_ACCESS_DENIED). "
                            "相机无访问权限：很可能已被其它程序占用（例如 MVS Client / 另一个脚本）。\n"
                            "请关闭/断开占用相机的程序，等待 2~5 秒后重试；必要时给相机断电重启。"
                        )

                    raise MvsError(
                        f"OpenDevice({serial}) failed, ret=0x{int(open_ret):08X} (MV_E_ACCESS_DENIED). "
                        "相机无访问权限：很可能已被其它程序占用（例如 MVS Client / 另一个脚本），"
                        "或相机当前不允许本机控制。\n"
                        "请关闭/断开占用相机的程序，等待 2~5 秒后重试；必要时给相机断电重启。"
                    )

                _check(int(open_ret), binding.MV_OK, f"OpenDevice({serial})")

            if tlayer_type == binding.params.MV_GIGE_DEVICE:
                _best_effort_gige_network_tuning(cam)

            # 可选：设置像素格式（通常是降带宽最立竿见影的开关之一）。
            if str(pixel_format).strip():
                configure_pixel_format(binding=binding, cam=cam, pixel_format=str(pixel_format))

            configure_trigger(
                binding=binding,
                cam=cam,
                trigger_source=trigger_source,
                trigger_activation=trigger_activation,
                trigger_cache_enable=trigger_cache_enable,
            )

            # 可选：配置某一路输出线（用于 master 输出触发脉冲）。
            if str(line_output_selector).strip() and str(line_output_source).strip():
                configure_line_output(
                    binding=binding,
                    cam=cam,
                    line_selector=str(line_output_selector),
                    line_source=str(line_output_source),
                    line_mode=str(line_output_mode or "Output"),
                )

            # 可选：设置 ROI 分辨率（用于降低带宽/算力）。
            if (image_width is not None) or (image_height is not None):
                if image_width is None or image_height is None:
                    raise ValueError("image_width 与 image_height 必须同时提供")
                configure_resolution(
                    binding=binding,
                    cam=cam,
                    width=int(image_width),
                    height=int(image_height),
                    offset_x=int(image_offset_x),
                    offset_y=int(image_offset_y),
                )

            # 可选曝光配置：保持“默认不干预”，只有显式传参才会设置。
            if str(exposure_auto).strip() or (exposure_time_us is not None):
                configure_exposure(
                    binding=binding,
                    cam=cam,
                    exposure_auto=str(exposure_auto or "Off"),
                    exposure_time_us=exposure_time_us,
                    gain_auto=str(gain_auto or "Off"),
                    gain=gain,
                )

            callbacks_keepalive: Dict[str, Any] = {}
            enabled_events = _enable_device_events(
                binding=binding,
                cam=cam,
                serial=serial,
                event_names=event_names,
                out_q=event_queue,
                callback_keepalive=callbacks_keepalive,
            )

            _check(cam.MV_CC_StartGrabbing(), binding.MV_OK, f"StartGrabbing({serial})")
            return cls(
                binding=binding,
                cam=cam,
                serial=serial,
                tlayer_type=tlayer_type,
                event_names_enabled=enabled_events,
                _event_callbacks_keepalive=callbacks_keepalive,
            )
        except Exception:
            best_effort(cam.MV_CC_CloseDevice)
            best_effort(cam.MV_CC_DestroyHandle)
            raise

    def close(self) -> None:
        best_effort(self.cam.MV_CC_StopGrabbing)

        # 关闭事件并取消订阅（尽力而为）。
        # 注意：MVS 文档建议通过 RegisterEventCallBackEx(event, NULL) 来取消订阅。
        for ev in list(self.event_names_enabled or []):
            best_effort(self.cam.MV_CC_EventNotificationOff, str(ev))
            best_effort(self.cam.MV_CC_RegisterEventCallBackEx, str(ev), None, None)

        best_effort(self._event_callbacks_keepalive.clear)

        best_effort(self.cam.MV_CC_CloseDevice)
        best_effort(self.cam.MV_CC_DestroyHandle)

    def __enter__(self) -> "MvsCamera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MvsSdk:
    """SDK 初始化/反初始化封装。"""

    def __init__(self, binding: MvsBinding):
        self._binding = binding
        self._inited = False

    def initialize(self) -> None:
        if self._inited:
            return
        ret = self._binding.MvCamera.MV_CC_Initialize()
        _check(ret, self._binding.MV_OK, "MV_CC_Initialize")
        self._inited = True

    def finalize(self) -> None:
        if not self._inited:
            return
        try:
            self._binding.MvCamera.MV_CC_Finalize()
        finally:
            self._inited = False

    def __enter__(self) -> "MvsSdk":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()
