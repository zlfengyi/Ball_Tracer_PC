# -*- coding: utf-8 -*-

"""最小示例：演示如何使用 mvs 包进行四相机同步采集。

前置条件：
- 海康 MVS SDK 已安装，或 MvCameraControl.dll 已通过 MVS_DLL_DIR 指定
- MVS 官方 Python 示例绑定目录（MvImport）可被找到（建议设置 MVS_MVIMPORT_DIR）
- 4 台 GigE 相机已连接网络并配置好 IP
"""

import sys
import time
from pathlib import Path

# 添加包根目录到 sys.path
pkg_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pkg_root))

from mvs import MvsDllNotFoundError, MvsSdk, enumerate_devices, load_mvs_binding, open_quad_capture


def example_list_devices():
    """示例 1：列举设备."""
    print("\n=== Example 1: List Devices ===")

    try:
        binding = load_mvs_binding()
    except MvsDllNotFoundError as e:
        print(f"Failed to load MVS binding: {e}")
        return

    sdk = MvsSdk(binding)
    sdk.initialize()

    try:
        _, descs = enumerate_devices(binding)
        print(f"Found {len(descs)} device(s):")
        for d in descs:
            ip = d.ip or "-"
            print(f"  [{d.index}] {d.model} (sn={d.serial}) @ {ip}")
    finally:
        sdk.finalize()


def example_quad_capture_one_group():
    """示例 2：采集一组（4 张图）。

    注意：需要 4 台相机已连接。可改 serials 为实际的序列号。
    """
    print("\n=== Example 2: Capture One Group ===")

    # === 配置这里 ===
    serials = [
        "DA8199285",
        "DA8199303",
        "DA8199402",
        "DA8199???",  # 改为实际的第 4 台序列号
    ]
    trigger_source = "Software"  # 用软触发（先测试链路）
    soft_trigger_fps = 5.0  # 5fps 触发
    group_by = "frame_num"  # 组包键：frame_num 或 sequence
    # === 配置结束 ===

    def _group_key(group_by: str, group) -> int | None:
        if not group:
            return None
        if str(group_by).strip() == "sequence":
            return int(getattr(group[0], "sequence", -1))
        return int(getattr(group[0], "frame_num", -1))

    try:
        binding = load_mvs_binding()
    except MvsDllNotFoundError as e:
        print(f"Failed to load MVS binding: {e}")
        return

    try:
        with open_quad_capture(
            binding=binding,
            serials=serials,
            trigger_sources=[trigger_source] * len(serials),
            trigger_activation="FallingEdge",
            trigger_cache_enable=False,
            timeout_ms=1000,
            group_timeout_ms=500,  # 给充足时间凑齐 4 张
            max_pending_groups=256,
            group_by=group_by,
            enable_soft_trigger_fps=soft_trigger_fps,
            soft_trigger_serials=serials,
        ) as cap:
            print("Waiting for first group...")
            group = cap.get_next_group(timeout_s=5.0)

            if group is None:
                print("Timeout: no group received")
                return

            key = _group_key(group_by, group)
            print(f"Got group: group_by={group_by} key={key}")
            for frame in group:
                print(
                    f"  cam{frame.cam_index}: {frame.width}x{frame.height}, "
                    f"frame_num={frame.frame_num}, dev_ts={frame.dev_timestamp}, "
                    f"lost_packet={frame.lost_packet}"
                )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_batch_capture():
    """示例 3：采集一批（多个组）并保存到文件。"""
    print("\n=== Example 3: Batch Capture ===")

    serials = [
        "DA8199285",
        "DA8199303",
        "DA8199402",
        "DA8199???",
    ]
    num_groups = 5
    output_dir = Path(__file__).resolve().parent.parent / "example_captures"

    try:
        binding = load_mvs_binding()
    except MvsDllNotFoundError as e:
        print(f"Failed to load MVS binding: {e}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open_quad_capture(
            binding=binding,
            serials=serials,
            trigger_sources=["Software"] * len(serials),
            trigger_activation="FallingEdge",
            trigger_cache_enable=False,
            timeout_ms=1000,
            group_timeout_ms=500,
            max_pending_groups=256,
            group_by="frame_num",
            enable_soft_trigger_fps=5.0,
            soft_trigger_serials=serials,
        ) as cap:
            for i in range(num_groups):
                group = cap.get_next_group(timeout_s=2.0)
                if group is None:
                    print(f"Group {i}: timeout")
                    continue

                group_key = int(getattr(group[0], "frame_num", -1)) if group else -1
                group_dir = output_dir / f"group_{i:010d}_key_{group_key:010d}"
                group_dir.mkdir(parents=True, exist_ok=True)

                for frame in group:
                    frame_file = group_dir / f"cam{frame.cam_index}.bin"
                    frame_file.write_bytes(frame.data)
                    print(f"Group {i} cam{frame.cam_index}: saved {frame_file}")

                # 打印统计信息
                dropped = cap.assembler.dropped_groups
                print(f"Group {i}: key={group_key}, assembler_dropped={dropped}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("MVS Package Examples")
    print("=" * 60)

    # 根据需要运行不同示例
    example_list_devices()

    # example_quad_capture_one_group()
    example_batch_capture()

    print("\n" + "=" * 60)
    print("Done.")

