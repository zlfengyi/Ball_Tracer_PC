# -*- coding: utf-8 -*-
"""
采集 DA8199285 和 DA8199402 两台相机的同步图片。

触发逻辑：
  主相机 DA8199303  → free-run 30fps, Line1 输出 ExposureStartActive
  从机 DA8199402    → Line0 FallingEdge 硬触发
  从机 DA8199285    → Line0 FallingEdge 硬触发
  从机 DA8199243    → 不打开（节省带宽）

相机设置：
  曝光 1800μs, 增益 20dB, ROI Y[700:] (高度 1348)

主相机 ROI 设为最小（仅做触发源，不传输图像数据）。
"""

import sys
import time
import datetime
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src import SyncCapture, frame_to_numpy

MASTER_SERIAL = "DA8199303"
SYNC_SLAVES = ["DA8199402", "DA8199285"]

save_dir = root / "capture_test_output"
save_dir.mkdir(parents=True, exist_ok=True)

print("初始化相机...")
print(f"  主相机: {MASTER_SERIAL} (仅触发，最小 ROI)")
print(f"  同步从机: {', '.join(SYNC_SLAVES)}")
print(f"  曝光: 1800μs, 增益: 20dB, ROI: Y[700:]")
print()

with SyncCapture(
    MASTER_SERIAL,
    SYNC_SLAVES,
    fps=30.0,
    trigger_mode="hardware",
    exposure_us=1800.0,
    gain_db=20.0,
    roi_offset_y=700,
    master_min_bandwidth=True,
) as cap:
    print("等待稳定 (2s)...")
    time.sleep(2.0)

    print("采集同步帧...")
    frames = cap.get_frames(timeout_s=3.0)
    if frames is None:
        print("ERROR: 超时，未获取到同步帧")
        sys.exit(1)

    import cv2
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    for sn, f in frames.items():
        img = frame_to_numpy(f)
        filename = f"{sn}_{ts}.png"
        path = save_dir / filename
        cv2.imwrite(str(path), img)
        print(f"  {sn}: {f.width}x{f.height}, exposure={f.exposure_time:.0f}μs, "
              f"exposure_start_pc={f.exposure_start_pc:.6f}")

    exp_starts = [f.exposure_start_pc for f in frames.values()]
    spread_ms = (max(exp_starts) - min(exp_starts)) * 1000.0
    print(f"\n返回相机: {list(frames.keys())}")
    print(f"曝光时间差: {spread_ms:.3f} ms")
    print(f"图片已保存到: {save_dir}")
