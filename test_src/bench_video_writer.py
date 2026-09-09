# -*- coding: utf-8 -*-
"""不接相机，直接按固定帧率喂真实 VideoWriterThread，判断这台机器能不能录到那个帧率。

    python test_src/bench_video_writer.py --fps 60 --w 2048 --h 928 --load 6

0906 的 G 组曾量到「写视频 25.3ms/帧」并据此认为 60Hz 录像不可能，那是错的：当时相机网卡在
打滑、整台机器被拖住，写线程是被饿出来的。本台架单独量：2048x928 avc1/MSMF 在 60Hz 空载
2.4ms、带 6 个背景占核线程 11.0ms，70Hz 带 6 线程 11.7ms，全部零丢帧。判据看 keeps_up 与
dropped，别只看 avg_ms——队列（maxsize 30）能吸收单次尖峰，真正的失败是 dropped>0。
"""
import sys, time, json, argparse, statistics
from pathlib import Path
import numpy as np, cv2
ROOT = Path(r"D:\Ball_Tracer_PC"); sys.path.insert(0, str(ROOT))
from src.run_tracker import VideoWriterThread, WriteJob

ap = argparse.ArgumentParser()
ap.add_argument("--w", type=int, default=2048)
ap.add_argument("--h", type=int, default=928)
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--codec", default="avc1")
ap.add_argument("--backend", default="auto")
ap.add_argument("--hw", type=int, default=1)
ap.add_argument("--out", default=None)
ap.add_argument("--load", type=int, default=0, help="背景负载线程数，模拟管线抢 CPU")
args = ap.parse_args()

SERIALS = ["A", "B", "C", "D"]
# Realistic content: a real court frame beats noise (noise is worst case for any codec)
src = None
for run in sorted((ROOT / "tracker_output").glob("tracker_2026090*/*.mp4")):
    cap = cv2.VideoCapture(str(run)); ok, img = cap.read(); cap.release()
    if ok: src = img; break
base = (cv2.resize(src, (args.w, args.h)) if src is not None
        else np.random.randint(0, 255, (args.h, args.w, 3), np.uint8))
frames = [np.roll(base, k * 7, axis=1) for k in range(12)]   # 12 distinct frames, cycled

# 背景负载：模拟主循环(tile 解码/YOLO 前后处理)+全图线抢核
import threading
_stop_load = threading.Event()
def _burn():
    a = np.random.randint(0, 255, (928, 2048), np.uint8)
    while not _stop_load.is_set():
        cv2.resize(cv2.cvtColor(a, cv2.COLOR_BayerBG2BGR), (416, 416))
load_threads = [threading.Thread(target=_burn, daemon=True) for _ in range(args.load)]
for t in load_threads: t.start()

out = Path(args.out or (ROOT / "tracker_output" / "_writer_bench.mp4"))
out.parent.mkdir(parents=True, exist_ok=True)
w = VideoWriterThread(str(out), args.w, args.h, n_cams=4, fps=args.fps,
                      codec=args.codec, backend=args.backend,
                      prefer_hw_accel=bool(args.hw), display=False,
                      full_res=False, cam_serials=SERIALS)
period = 1.0 / args.fps
n = int(args.seconds * args.fps)
t0 = time.perf_counter()
late = 0
for i in range(n):
    deadline = t0 + i * period
    now = time.perf_counter()
    if now < deadline:
        time.sleep(deadline - now)
    else:
        late += 1
    img = frames[i % len(frames)]
    w.submit(WriteJob(images={s: img for s in SERIALS}, serials=SERIALS,
                      exposure_perf=deadline, elapsed_s=i * period, frame_idx=i))
submit_elapsed = time.perf_counter() - t0
dropped = w.stop()
_stop_load.set()
total = time.perf_counter() - t0
st = w.stats()
size_mb = out.stat().st_size / 1e6 if out.exists() else 0.0
cap = cv2.VideoCapture(str(out)); written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
budget = period * 1000.0
print(json.dumps({
    "grid": f"{args.w//2*2}x{args.h//2*2}", "fps": args.fps,
    "codec_req": args.codec, "codec": st["codec"], "backend": st["backend"],
    "hw_accel_requested": st["hw_accel_requested"],
    "submitted": n, "written_in_file": written, "dropped": dropped,
    "avg_ms": st["avg_process_ms"], "max_ms": st["max_process_ms"],
    "budget_ms": round(budget, 2),
    "keeps_up": st["avg_process_ms"] < budget and dropped == 0,
    "queue_max": st["queue_max_size"], "bg_load_threads": args.load,
    "submit_wall_s": round(submit_elapsed, 1), "drain_wall_s": round(total, 1),
    "file_MB": round(size_mb, 1),
}, ensure_ascii=False))
out.unlink(missing_ok=True)
