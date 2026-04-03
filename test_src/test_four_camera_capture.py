from __future__ import annotations

import argparse
import datetime
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src import SyncCapture, frame_to_numpy


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    data = np.array(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "min": float(data.min()),
        "max": float(data.max()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


class CameraVideoWriter:
    def __init__(
        self,
        video_path: Path,
        *,
        frame_size: tuple[int, int],
        fps: float,
        codec: str,
        queue_size: int = 128,
    ) -> None:
        self.video_path = video_path
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._submitted = 0
        self._written = 0
        self._dropped = 0

        codec_candidates = [codec]
        if codec != "MJPG":
            codec_candidates.append("MJPG")

        self._writer = None
        self._actual_codec = None
        for codec_name in codec_candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
            if writer.isOpened():
                self._writer = writer
                self._actual_codec = codec_name
                break
            writer.release()

        if self._writer is None:
            raise RuntimeError(
                f"Failed to open VideoWriter for {video_path} with codecs {codec_candidates}"
            )

        self._thread = threading.Thread(
            target=self._run,
            name=f"Writer-{video_path.stem}",
            daemon=True,
        )
        self._thread.start()

    @property
    def actual_codec(self) -> str:
        return self._actual_codec or "UNKNOWN"

    @property
    def submitted(self) -> int:
        return self._submitted

    @property
    def written(self) -> int:
        return self._written

    @property
    def dropped(self) -> int:
        return self._dropped

    def submit(self, image: np.ndarray) -> None:
        self._submitted += 1
        try:
            self._queue.put_nowait(image.copy())
        except queue.Full:
            self._dropped += 1

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        self._writer.release()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._writer.write(item)
            self._written += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record synchronized video from 4 hardware-triggered cameras and report latency/FPS.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Record duration in seconds.")
    parser.add_argument("--warmup", type=float, default=1.5, help="Warmup time before capture.")
    parser.add_argument(
        "--output-dir",
        default=str(root / "four_camera_data"),
        help="Directory for videos and stats.",
    )
    parser.add_argument("--codec", default="MJPG", help="Preferred output codec, e.g. MJPG or XVID.")
    parser.add_argument("--display", action="store_true", help="Show a stitched preview window.")
    parser.add_argument(
        "--no-rotate-180",
        action="store_true",
        help="Disable the default 180-degree rotation before saving and previewing frames.",
    )
    return parser.parse_args()


def _process_frame_group(
    *,
    frame_idx: int,
    frames: dict[str, object],
    serials: list[str],
    writers: dict[str, CameraVideoWriter],
    per_camera_latency_ms: dict[str, list[float]],
    group_latency_ms: list[float],
    sync_spread_ms: list[float],
    decode_times_ms: list[float],
    frame_intervals_ms: list[float],
    prev_exposure_pc: float | None,
    show_preview: bool,
    rotate_180: bool,
) -> float:
    exposure_starts = [frames[sn].exposure_start_pc for sn in serials]
    arrivals = [frames[sn].arrival_perf for sn in serials]
    exposure_pc = sum(exposure_starts) / len(exposure_starts)

    for sn in serials:
        per_camera_latency_ms[sn].append(
            (frames[sn].arrival_perf - frames[sn].exposure_start_pc) * 1000.0
        )

    group_latency_ms.append((max(arrivals) - exposure_pc) * 1000.0)
    sync_spread_ms.append((max(exposure_starts) - min(exposure_starts)) * 1000.0)
    if prev_exposure_pc is not None:
        frame_intervals_ms.append((exposure_pc - prev_exposure_pc) * 1000.0)

    t0 = time.perf_counter()
    images = {sn: frame_to_numpy(frames[sn], rotate_180=rotate_180) for sn in serials}
    decode_times_ms.append((time.perf_counter() - t0) * 1000.0)

    for sn in serials:
        writers[sn].submit(images[sn])

    if show_preview:
        preview_panels = [
            cv2.resize(
                images[sn],
                (images[sn].shape[1] // 2, images[sn].shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
            for sn in serials
        ]
        preview = np.hstack(preview_panels)
        cv2.putText(
            preview,
            f"frame={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.imshow("FourCameraCapture", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    return exposure_pc


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    rotate_180 = not args.no_rotate_180
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / f"four_camera_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Four Camera Capture Test")
    print("=" * 60)
    print(f"Output root: {output_root}")
    print(f"Rotate 180: {'ON' if rotate_180 else 'OFF'}")

    timeout_count = 0
    frame_idx = 0
    per_camera_latency_ms: dict[str, list[float]] = {}
    group_latency_ms: list[float] = []
    sync_spread_ms: list[float] = []
    decode_times_ms: list[float] = []
    frame_intervals_ms: list[float] = []
    prev_exposure_pc: float | None = None
    writers: dict[str, CameraVideoWriter] = {}
    serials: list[str] = []
    configured_fps = 0.0
    t_start = 0.0

    try:
        with SyncCapture.from_config() as cap:
            serials = cap.sync_serials
            configured_fps = cap.fps
            per_camera_latency_ms = {sn: [] for sn in serials}

            print(f"Configured serials: {serials}")
            print(f"Configured FPS: {configured_fps:.1f}")
            if len(serials) != 4:
                print(f"WARNING: expected 4 synchronized cameras, got {len(serials)}")

            print(f"Warmup: {args.warmup:.1f}s")
            time.sleep(args.warmup)

            first_frames = cap.get_frames(timeout_s=3.0)
            if first_frames is None:
                print("ERROR: failed to acquire the first synchronized frame group.")
                return 1

            for sn in serials:
                image = frame_to_numpy(first_frames[sn], rotate_180=rotate_180)
                video_path = session_dir / f"{sn}.avi"
                writers[sn] = CameraVideoWriter(
                    video_path,
                    frame_size=(image.shape[1], image.shape[0]),
                    fps=configured_fps,
                    codec=args.codec.upper(),
                )
                print(
                    f"  {sn}: {image.shape[1]}x{image.shape[0]} -> {video_path.name} "
                    f"(codec={writers[sn].actual_codec})"
                )

            t_start = time.perf_counter()
            prev_exposure_pc = _process_frame_group(
                frame_idx=frame_idx,
                frames=first_frames,
                serials=serials,
                writers=writers,
                per_camera_latency_ms=per_camera_latency_ms,
                group_latency_ms=group_latency_ms,
                sync_spread_ms=sync_spread_ms,
                decode_times_ms=decode_times_ms,
                frame_intervals_ms=frame_intervals_ms,
                prev_exposure_pc=prev_exposure_pc,
                show_preview=args.display,
                rotate_180=rotate_180,
            )
            frame_idx += 1

            while time.perf_counter() - t_start < args.duration:
                frames = cap.get_frames(timeout_s=1.0)
                if frames is None:
                    timeout_count += 1
                    continue

                prev_exposure_pc = _process_frame_group(
                    frame_idx=frame_idx,
                    frames=frames,
                    serials=serials,
                    writers=writers,
                    per_camera_latency_ms=per_camera_latency_ms,
                    group_latency_ms=group_latency_ms,
                    sync_spread_ms=sync_spread_ms,
                    decode_times_ms=decode_times_ms,
                    frame_intervals_ms=frame_intervals_ms,
                    prev_exposure_pc=prev_exposure_pc,
                    show_preview=args.display,
                    rotate_180=rotate_180,
                )
                frame_idx += 1

                if frame_idx % 30 == 0:
                    elapsed = time.perf_counter() - t_start
                    actual_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  {frame_idx} groups  "
                        f"actual_fps={actual_fps:.1f}  "
                        f"group_latency={_stats(group_latency_ms)['mean']:.1f}ms  "
                        f"sync_spread={_stats(sync_spread_ms)['mean']:.3f}ms"
                    )
    except KeyboardInterrupt:
        print("\nCapture interrupted.")
    finally:
        if args.display:
            cv2.destroyAllWindows()
        for writer in writers.values():
            writer.close()

    elapsed = time.perf_counter() - t_start if t_start > 0 else 0.0
    actual_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    interval_stats = _stats(frame_intervals_ms)
    per_camera_stats = {
        sn: _stats(values) for sn, values in per_camera_latency_ms.items()
    }

    summary = {
        "config": {
            "serials": serials,
            "configured_fps": configured_fps,
            "duration_s": elapsed,
            "codec": args.codec.upper(),
            "rotate_180": rotate_180,
            "output_dir": str(session_dir),
        },
        "summary": {
            "captured_groups": frame_idx,
            "actual_fps": actual_fps,
            "timeout_count": timeout_count,
            "group_latency_ms": _stats(group_latency_ms),
            "sync_spread_ms": _stats(sync_spread_ms),
            "decode_time_ms": _stats(decode_times_ms),
            "frame_interval_ms": interval_stats,
            "frame_interval_fps": (
                1000.0 / interval_stats["mean"] if interval_stats["mean"] > 0 else 0.0
            ),
            "per_camera_latency_ms": per_camera_stats,
        },
        "videos": {
            sn: {
                "path": str(writers[sn].video_path),
                "codec": writers[sn].actual_codec,
                "submitted": writers[sn].submitted,
                "written": writers[sn].written,
                "dropped": writers[sn].dropped,
            }
            for sn in serials
        },
    }

    stats_path = session_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Captured groups: {frame_idx}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Actual FPS: {actual_fps:.2f}")
    print(
        "Group latency (ms): "
        f"mean={summary['summary']['group_latency_ms']['mean']:.1f} "
        f"p95={summary['summary']['group_latency_ms']['p95']:.1f} "
        f"max={summary['summary']['group_latency_ms']['max']:.1f}"
    )
    print(
        "Sync spread (ms): "
        f"mean={summary['summary']['sync_spread_ms']['mean']:.3f} "
        f"p95={summary['summary']['sync_spread_ms']['p95']:.3f} "
        f"max={summary['summary']['sync_spread_ms']['max']:.3f}"
    )
    for sn in serials:
        stats = per_camera_stats[sn]
        video = summary["videos"][sn]
        print(
            f"{sn}: latency_mean={stats['mean']:.1f}ms  "
            f"latency_p95={stats['p95']:.1f}ms  "
            f"written={video['written']}  dropped={video['dropped']}"
        )
    print(f"Stats JSON: {stats_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
