# -*- coding: utf-8 -*-
"""
快速演示：从 config/camera.json 加载配置，同步取一组图片并保存。

用法：
  python -m src.demo
"""

import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import SyncCapture, frame_to_numpy


def main():
    save_dir = _root / "captured_frames"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("加载配置 (config/camera.json)...")
    with SyncCapture.from_config() as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")
        print("等待稳定 (2s)...")
        time.sleep(2.0)

        import cv2
        frames = cap.get_frames(timeout_s=3.0)
        if frames is None:
            print("*** 未取到同步帧 ***")
            return 1

        for sn, f in frames.items():
            img = frame_to_numpy(f)
            path = save_dir / f"{sn}.png"
            cv2.imwrite(str(path), img)
            print(f"  [{sn}] {f.width}x{f.height}  frame_num={f.frame_num}  => {path.name}")

        print(f"\n保存完成！目录: {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
