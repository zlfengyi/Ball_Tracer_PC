# -*- coding: utf-8 -*-
"""
网球定位与实验视频保存 (DEVELOP_LIST 步骤 4.5)。

完整管线流程：
  1. SyncCapture 双目同步拍摄（硬件触发，30fps）
  2. BallDetector YOLO 批量检测左右图像中的网球（全分辨率）
  3. 若左右各检测到 1 个网球 → BallLocalizer.triangulate() 三角测量得到 3D 位置
  4. 将 3D 位置送入 Curve3Tracker 进行轨迹追踪与击球点预测
  5. 左右图像 + 检测/追踪结果 交给后台写入线程：
     - 缩小到半分辨率
     - 标注（检测框、曝光时间、3D 坐标、curve3 状态）
     - VideoWriter 编码写入
     主线程不等待写入完成，立刻处理下一帧。
  6. JSON 结果日志在结束后保存

性能设计：
  - YOLO 推理在全分辨率图上运行，不降分辨率
  - 图像缩放、标注绘制、MJPG 编码全部在后台线程完成
  - 主线程只做：取帧 → Bayer解码 → YOLO → 三角测量 → curve3 → 入队

用法：
  python run_tracker.py [--duration 60] [--no-video] [--output-dir tracker_output]
                        [--display] [--ideal-hit-z 800]

输出文件（存放在 tracker_output/ 下）：
  tracker_YYYYMMDD_HHMMSS.avi   — 标注拼接视频（半分辨率，MJPG）
  tracker_YYYYMMDD_HHMMSS.json  — 观测、预测、状态变化等完整日志
"""

from __future__ import annotations

import argparse
import datetime
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import (
    SyncCapture,
    frame_to_numpy,
    BallDetector,
    BallDetection,
    BallLocalizer,
    Ball3D,
)
from src.curve3 import (
    BallObservation,
    Curve3Tracker,
    TrackerState,
    TrackerResult,
)


# ══════════════════════════════════════════════════════════════════════════
#  标注参数
# ══════════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
BOX_COLOR_LEFT = (0, 255, 0)       # 绿色 — 左相机检测框
BOX_COLOR_RIGHT = (0, 165, 255)    # 橙色 — 右相机检测框
TEXT_COLOR = (255, 255, 255)        # 白色文字
TEXT_3D_COLOR = (0, 255, 255)       # 黄色 — 3D 坐标
STATE_COLORS = {
    TrackerState.IDLE:         (128, 128, 128),   # 灰色
    TrackerState.TRACKING_S0:  (255, 200, 0),     # 青色
    TrackerState.IN_LANDING:   (0, 165, 255),     # 橙色
    TrackerState.TRACKING_S1:  (0, 255, 0),       # 绿色
    TrackerState.DONE:         (0, 0, 255),        # 红色
}


# ══════════════════════════════════════════════════════════════════════════
#  后台写入线程需要的数据包
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class WriteJob:
    """主线程投递给写入线程的工作包。"""
    img_left: np.ndarray           # 全分辨率左图
    img_right: np.ndarray          # 全分辨率右图
    det_left: list[BallDetection]  # 左图检测结果（全分辨率坐标）
    det_right: list[BallDetection] # 右图检测结果（全分辨率坐标）
    exposure_wall: float             # wall clock 时间 (time.time() 轴)
    ball3d: Optional[Ball3D]
    tracker_result: TrackerResult
    frame_idx: int
    latency_ms: float                # 从曝光到 YOLO 出结果的延迟 (ms)


# ══════════════════════════════════════════════════════════════════════════
#  图像标注（在写入线程中调用）
# ══════════════════════════════════════════════════════════════════════════


def _draw_detections(
    img: np.ndarray,
    detections: list[BallDetection],
    color: tuple[int, int, int],
) -> None:
    """在图像上绘制 YOLO 检测框和置信度。"""
    for det in detections:
        cv2.rectangle(
            img,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            color, 2,
        )
        cv2.putText(
            img, f"{det.confidence:.2f}",
            (int(det.x1), int(det.y1) - 5),
            FONT, FONT_SCALE, color, FONT_THICKNESS,
        )


def annotate_frame(
    img_left: np.ndarray,
    img_right: np.ndarray,
    det_left: list[BallDetection],
    det_right: list[BallDetection],
    exposure_wall: float,
    ball3d: Ball3D | None,
    tracker_result: TrackerResult,
    frame_idx: int,
    latency_ms: float = 0.0,
) -> np.ndarray:
    """
    拼接左右图像并标注检测结果和追踪状态。

    接收的图像已经是半分辨率，检测框坐标也已缩放。

    标注布局（从上到下）：
      行1: 帧号 + 曝光开始时间（PC perf_counter 轴，秒）
      行2: 左右检测数量
      行3: 3D 坐标 + 重投影误差（仅当三角测量成功时显示，黄色）
      行4: Curve3 追踪状态 + predict-hit-pos 详情（颜色随状态变化）
      底部: "LEFT" / "RIGHT" 标识 + 分隔线
    """
    left = img_left.copy()
    right = img_right.copy()

    _draw_detections(left, det_left, BOX_COLOR_LEFT)
    _draw_detections(right, det_right, BOX_COLOR_RIGHT)

    stitched = np.hstack([left, right])
    h, w = stitched.shape[:2]
    left_w = img_left.shape[1]

    y = 22

    cv2.putText(
        stitched,
        f"#{frame_idx}  {datetime.datetime.fromtimestamp(exposure_wall).strftime('%H:%M:%S.%f')[:-3]}"
        f"  lat={latency_ms:.0f}ms",
        (10, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS,
    )
    y += 22

    cv2.putText(
        stitched,
        f"det: L={len(det_left)} R={len(det_right)}",
        (10, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS,
    )
    y += 22

    if ball3d is not None:
        cv2.putText(
            stitched,
            f"3D: ({ball3d.x:.0f}, {ball3d.y:.0f}, {ball3d.z:.0f}) mm  "
            f"reproj={ball3d.reprojection_error:.1f}px  "
            f"conf={ball3d.confidence:.2f}",
            (10, y), FONT, FONT_SCALE, TEXT_3D_COLOR, FONT_THICKNESS,
        )
        y += 22

    state = tracker_result.state
    state_color = STATE_COLORS.get(state, TEXT_COLOR)
    state_str = f"curve3: {state.value}"

    pred = tracker_result.prediction
    if pred is not None:
        lead_ms = (pred.ht - pred.ct) * 1000
        state_str += (
            f"  hit=({pred.x:.0f}, {pred.y:.0f}, {pred.z:.0f}) "
            f"stage={pred.stage} lead={lead_ms:.0f}ms"
        )

    cv2.putText(
        stitched, state_str,
        (10, y), FONT, FONT_SCALE, state_color, FONT_THICKNESS,
    )

    cv2.putText(stitched, "LEFT", (10, h - 10), FONT, 0.5, BOX_COLOR_LEFT, 1)
    cv2.putText(stitched, "RIGHT", (left_w + 10, h - 10), FONT, 0.5, BOX_COLOR_RIGHT, 1)
    cv2.line(stitched, (left_w, 0), (left_w, h), (100, 100, 100), 1)

    return stitched


# ══════════════════════════════════════════════════════════════════════════
#  后台写入线程
# ══════════════════════════════════════════════════════════════════════════


class VideoWriterThread:
    """
    后台线程：接收 WriteJob → 缩放到半分辨率 → 标注 → 编码写入视频。

    主线程通过 submit(job) 投递工作，不阻塞。
    写入线程从队列中取出工作，顺序处理。
    队列满时（maxsize=30，即 1 秒缓冲），丢弃最旧的帧以避免主线程阻塞。
    """

    def __init__(
        self,
        video_path: str,
        frame_w: int,
        frame_h: int,
        fps: float = 30.0,
        display: bool = False,
    ):
        self._video_path = video_path
        self._half_w = frame_w // 2
        self._half_h = frame_h // 2
        self._display = display
        self._queue: queue.Queue[WriteJob | None] = queue.Queue(maxsize=30)
        self._stopped = False
        self._drop_count = 0

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(
            video_path, fourcc, fps,
            (self._half_w * 2, self._half_h),
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: WriteJob) -> None:
        """投递工作包（非阻塞）。队列满时丢弃最旧帧。"""
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # 丢弃队列头部最旧的帧，腾出空间
            try:
                self._queue.get_nowait()
                self._drop_count += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                pass

    def stop(self) -> int:
        """通知线程停止，等待队列排空，释放资源。返回丢弃帧数。"""
        self._queue.put(None)  # 哨兵
        self._thread.join(timeout=10.0)
        self._writer.release()
        if self._display:
            cv2.destroyAllWindows()
        return self._drop_count

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            self._process(job)

    def _process(self, job: WriteJob) -> None:
        """缩放 → 缩放检测坐标 → 标注 → 写入视频。"""
        half_left = cv2.resize(job.img_left, (self._half_w, self._half_h))
        half_right = cv2.resize(job.img_right, (self._half_w, self._half_h))

        det_left_h = [
            BallDetection(
                x=d.x / 2, y=d.y / 2, confidence=d.confidence,
                x1=d.x1 / 2, y1=d.y1 / 2, x2=d.x2 / 2, y2=d.y2 / 2,
            )
            for d in job.det_left
        ]
        det_right_h = [
            BallDetection(
                x=d.x / 2, y=d.y / 2, confidence=d.confidence,
                x1=d.x1 / 2, y1=d.y1 / 2, x2=d.x2 / 2, y2=d.y2 / 2,
            )
            for d in job.det_right
        ]

        annotated = annotate_frame(
            half_left, half_right,
            det_left_h, det_right_h,
            job.exposure_wall, job.ball3d,
            job.tracker_result, job.frame_idx,
            latency_ms=job.latency_ms,
        )

        self._writer.write(annotated)

        if self._display:
            cv2.imshow("Tracker", annotated)
            cv2.waitKey(1)


# ══════════════════════════════════════════════════════════════════════════
#  主管线
# ══════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="网球定位与实验视频保存 (Step 4.5)")
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="录制时长（秒，默认 60）")
    parser.add_argument(
        "--no-video", action="store_true",
        help="不保存视频（仅 JSON）")
    parser.add_argument(
        "--output-dir", type=str, default="tracker_output",
        help="输出目录（默认 tracker_output/）")
    parser.add_argument(
        "--display", action="store_true",
        help="实时显示拼接画面（按 q 退出）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 初始化组件 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 4.5 — 网球定位与实验视频保存")
    print("=" * 60)

    print("\n[1/4] 初始化 BallDetector (YOLO)...")
    detector = BallDetector()
    print(f"  模型: {detector.model_path}")

    print("[2/4] 初始化 BallLocalizer (立体标定)...")
    localizer = BallLocalizer(detector=detector)
    serial_left = localizer.serial_left
    serial_right = localizer.serial_right
    print(f"  左相机: {serial_left}, 右相机: {serial_right}")

    print("[3/4] 初始化 Curve3Tracker...")
    tracker = Curve3Tracker()
    print(f"  ideal_hit_z={tracker.ideal_hit_z}mm, cor={tracker.cor}, "
          f"cor_xy={tracker.cor_xy}")

    # ── 日志容器 ────────────────────────────────────────────────────────
    log_observations: list[dict] = []
    log_predictions: list[dict] = []
    log_frames: list[dict] = []
    log_state_transitions: list[dict] = []

    # ── 打开同步相机 ────────────────────────────────────────────────────
    print("[4/4] 打开同步相机...")
    with SyncCapture.from_config() as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")

        if serial_left not in sync_sns or serial_right not in sync_sns:
            print(f"*** 错误: 标定相机 {serial_left}/{serial_right} "
                  f"不在同步列表 {sync_sns} 中 ***")
            return 1

        print("  等待相机稳定 (1s)...")
        time.sleep(1.0)

        first_frames = cap.get_frames(timeout_s=3.0)
        if first_frames is None:
            print("*** 错误: 无法获取初始帧 ***")
            return 1

        img_sample = frame_to_numpy(first_frames[serial_left])
        frame_h, frame_w = img_sample.shape[:2]
        print(f"  单帧分辨率: {frame_w}x{frame_h}, "
              f"视频输出分辨率: {frame_w}x{frame_h // 2}")

        # ── YOLO 预热（TensorRT 首次推理很慢，先跑几次让引擎稳定）──
        print("  YOLO 预热中...")
        warmup_img = frame_to_numpy(first_frames[serial_left])
        for _ in range(5):
            detector.detect_batch([warmup_img, warmup_img])
        print("  预热完成")

        # ── 后台写入线程 ───────────────────────────────────────────────
        writer_thread: VideoWriterThread | None = None
        video_path = output_dir / f"tracker_{ts}.avi"
        if not args.no_video:
            writer_thread = VideoWriterThread(
                str(video_path), frame_w, frame_h,
                fps=30.0, display=args.display,
            )
            print(f"  视频输出: {video_path}")

        # ── 主循环 ────────────────────────────────────────────────────
        #   主线程只做: 取帧 → Bayer解码 → YOLO推理 → 三角测量 → curve3
        #   标注/缩放/视频编码全部交给后台线程
        # ──────────────────────────────────────────────────────────────
        frame_idx = 0
        t_start = time.monotonic()
        # perf_counter → wall clock 转换基准
        _pc_epoch = time.perf_counter()
        _wall_epoch = time.time()
        prev_state = TrackerState.IDLE
        timeout_count = 0
        _t_decode_sum = 0.0
        _t_yolo_sum = 0.0

        print(f"\n{'*' * 60}")
        print(f"  ✓ 预热完成，开始追踪！（{args.duration}s）按 Ctrl+C 提前结束")
        print(f"{'*' * 60}\n")
        print("\a", end="", flush=True)  # 蜂鸣提示音

        try:
            while time.monotonic() - t_start < args.duration:
                frames = cap.get_frames(timeout_s=1.0)
                if frames is None:
                    timeout_count += 1
                    continue

                frame_left = frames[serial_left]
                frame_right = frames[serial_right]

                # ── Bayer 解码（全分辨率）──
                _t0 = time.perf_counter()
                img_left = frame_to_numpy(frame_left)
                img_right = frame_to_numpy(frame_right)
                _t1 = time.perf_counter()
                _t_decode_sum += _t1 - _t0

                exposure_pc = (
                    frame_left.exposure_start_pc
                    + frame_right.exposure_start_pc
                ) / 2.0
                # perf_counter → wall clock 转换
                exposure_wall = _wall_epoch + (exposure_pc - _pc_epoch)

                # ── YOLO 批量检测（全分辨率）──
                det_left, det_right = detector.detect_batch(
                    [img_left, img_right])
                _t2 = time.perf_counter()
                _t_yolo_sum += _t2 - _t1

                # 延迟 = YOLO 出结果时间 - 曝光时间（都在 perf_counter 轴）
                latency_ms = (_t2 - exposure_pc) * 1000.0

                # ── 三角测量 + Curve3 更新 ──
                ball3d: Ball3D | None = None
                tracker_result: TrackerResult

                if len(det_left) == 1 and len(det_right) == 1:
                    ball3d = localizer.triangulate(det_left[0], det_right[0])

                    obs = BallObservation(
                        x=ball3d.x, y=ball3d.y,
                        z=ball3d.z, t=exposure_pc,
                    )
                    tracker_result = tracker.update(obs)

                    log_observations.append({
                        "x": ball3d.x, "y": ball3d.y, "z": ball3d.z,
                        "t": exposure_pc,
                        "reproj_err": ball3d.reprojection_error,
                        "confidence": ball3d.confidence,
                    })

                    if tracker_result.prediction is not None:
                        p = tracker_result.prediction
                        log_predictions.append({
                            "x": p.x, "y": p.y, "z": p.z,
                            "stage": p.stage,
                            "ct": p.ct, "ht": p.ht,
                        })
                else:
                    tracker_result = TrackerResult(
                        prediction=None,
                        state=tracker.tracker_state,
                    )

                # ── 状态变化记录 ──
                if tracker_result.state != prev_state:
                    log_state_transitions.append({
                        "frame": frame_idx,
                        "t": exposure_pc,
                        "from": prev_state.value,
                        "to": tracker_result.state.value,
                    })
                    prev_state = tracker_result.state

                # ── 投递给后台写入线程（非阻塞）──
                if writer_thread is not None:
                    writer_thread.submit(WriteJob(
                        img_left=img_left,
                        img_right=img_right,
                        det_left=det_left,
                        det_right=det_right,
                        exposure_wall=exposure_wall,
                        ball3d=ball3d,
                        tracker_result=tracker_result,
                        frame_idx=frame_idx,
                        latency_ms=latency_ms,
                    ))

                frame_entry = {
                    "idx": frame_idx,
                    "exposure_pc": exposure_pc,
                    "exposure_time": datetime.datetime.fromtimestamp(
                        exposure_wall).strftime('%H:%M:%S.%f')[:-3],
                    "det_left": len(det_left),
                    "det_right": len(det_right),
                    "has_3d": ball3d is not None,
                    "state": tracker_result.state.value,
                    "latency_ms": round(latency_ms, 1),
                }
                if det_left:
                    frame_entry["det_left_detail"] = [
                        {"x": round(d.x), "y": round(d.y),
                         "conf": round(d.confidence, 3)}
                        for d in det_left
                    ]
                if det_right:
                    frame_entry["det_right_detail"] = [
                        {"x": round(d.x), "y": round(d.y),
                         "conf": round(d.confidence, 3)}
                        for d in det_right
                    ]
                log_frames.append(frame_entry)

                frame_idx += 1

                if frame_idx % 100 == 0:
                    elapsed = time.monotonic() - t_start
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    n = max(frame_idx, 1)
                    print(
                        f"  [{elapsed:.1f}s] {frame_idx} frames "
                        f"({fps:.1f} fps)  "
                        f"3D={len(log_observations)}  "
                        f"preds={len(log_predictions)}  "
                        f"state={tracker_result.state.value}  "
                        f"avg: decode={_t_decode_sum/n*1000:.1f}ms "
                        f"yolo={_t_yolo_sum/n*1000:.1f}ms"
                    )

        except KeyboardInterrupt:
            print("\n手动中断")

        # ── 清理 ──
        drop_count = 0
        if writer_thread is not None:
            print("  等待视频写入完成...")
            drop_count = writer_thread.stop()

    # ── 保存 JSON 结果 ──────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start

    latencies = [f["latency_ms"] for f in log_frames]
    lat_avg = sum(latencies) / len(latencies) if latencies else 0
    lat_min = min(latencies) if latencies else 0
    lat_max = max(latencies) if latencies else 0

    result = {
        "config": {
            "start_time": datetime.datetime.fromtimestamp(
                _wall_epoch).strftime("%y-%m-%d %H:%M"),
            "serial_left": serial_left,
            "serial_right": serial_right,
            "duration_s": elapsed,
            "ideal_hit_z": tracker.ideal_hit_z,
            "cor": tracker.cor,
            "cor_xy": tracker.cor_xy,
            "model_path": str(detector.model_path),
        },
        "summary": {
            "total_frames": frame_idx,
            "actual_fps": frame_idx / elapsed if elapsed > 0 else 0,
            "timeouts": timeout_count,
            "observations_3d": len(log_observations),
            "predictions": len(log_predictions),
            "state_transitions": len(log_state_transitions),
            "reset_times": tracker.reset_times,
            "video_frames_dropped": drop_count,
            "latency_ms_avg": round(lat_avg, 1),
            "latency_ms_min": round(lat_min, 1),
            "latency_ms_max": round(lat_max, 1),
        },
        "observations": log_observations,
        "predictions": log_predictions,
        "frames": log_frames,
        "state_transitions": log_state_transitions,
    }

    json_path = output_dir / f"tracker_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── 最终统计 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  录制时长:   {elapsed:.1f}s")
    print(f"  总帧数:     {frame_idx}")
    print(f"  实际帧率:   {frame_idx / elapsed if elapsed > 0 else 0:.1f} fps")
    print(f"  3D 观测:    {len(log_observations)}")
    print(f"  预测数:     {len(log_predictions)}")
    print(f"  状态转换:   {len(log_state_transitions)}")
    print(f"  超时次数:   {timeout_count}")
    print(f"  延迟(ms):   avg={lat_avg:.0f}  min={lat_min:.0f}  max={lat_max:.0f}")
    if not args.no_video:
        print(f"  视频:       {video_path}")
        if drop_count > 0:
            print(f"  视频丢帧:   {drop_count}")
    print(f"  JSON:       {json_path}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
