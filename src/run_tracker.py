# -*- coding: utf-8 -*-
"""
网球定位与实验视频保存 (DEVELOP_LIST 步骤 4.5)。

完整管线流程：
  1. SyncCapture 三目同步拍摄（硬件触发，30fps）
  2. TileManager 为每台相机选择 800x800 切片（跟踪/搜索模式）
  3. BallDetector YOLO 批量检测切片中的网球
  4. 若 ≥2 台相机各检测到 1 个网球 → BallLocalizer.triangulate() 多视图三角测量得到 3D 位置
  5. 将 3D 位置送入 Curve3Tracker 进行轨迹追踪与击球点预测
  6. 图像 + 检测/追踪结果 交给后台写入线程：
     - 缩小到半分辨率
     - 标注（检测框、切片区域框、曝光时间、3D 坐标、curve3 状态）
     - VideoWriter 编码写入
     主线程不等待写入完成，立刻处理下一帧。
  7. JSON 结果日志在结束后保存

性能设计：
  - YOLO 推理在 800x800 切片上运行（跟踪模式：追踪球位置；搜索模式：轮询预定义区域）
  - 图像缩放、标注绘制、MJPG 编码全部在后台线程完成
  - 主线程只做：取帧 → Bayer解码 → 分片 → YOLO → 三角测量 → curve3 → 入队

用法：
  python run_tracker.py [--duration 60] [--no-video] [--output-dir tracker_output]
                        [--display] [--ideal-hit-z 800]

输出文件（存放在 tracker_output/ 下）：
  tracker_YYYYMMDD_HHMMSS.avi   — 标注拼接视频（半分辨率，MJPG）
  tracker_YYYYMMDD_HHMMSS.json  — 观测、预测、状态变化等完整日志
"""

from __future__ import annotations

import argparse
import signal
import datetime
import json
import math
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
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
    CarLocalizer,
    CarLoc,
)
from src.curve3 import (
    BallObservation,
    Curve3Tracker,
    TrackerState,
    TrackerResult,
)
from src.tile_manager import TileManager, TileRect


# ══════════════════════════════════════════════════════════════════════════
#  标注参数
# ══════════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.1
FONT_THICKNESS = 2
BOX_COLORS = [
    (0, 255, 0),       # 绿色 — 1号相机
    (0, 165, 255),     # 橙色 — 2号相机
    (255, 100, 100),   # 蓝色 — 3号相机
]
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
    images: dict[str, np.ndarray]              # {序列号: 全分辨率图像}
    detections: dict[str, list[BallDetection]] # {序列号: 检测结果列表}
    serials: list[str]                         # 相机序列号顺序
    exposure_wall: float                       # wall clock 时间
    ball3d: Optional[Ball3D]
    tracker_result: TrackerResult
    frame_idx: int
    latency_ms: float
    tiles: dict[str, TileRect] = field(default_factory=dict)  # 当前帧切片区域
    car_loc: Optional[CarLoc] = None


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
    images: dict[str, np.ndarray],
    detections: dict[str, list[BallDetection]],
    serials: list[str],
    exposure_wall: float,
    ball3d: Ball3D | None,
    tracker_result: TrackerResult,
    frame_idx: int,
    latency_ms: float = 0.0,
    tiles: dict[str, TileRect] | None = None,
    car_loc: CarLoc | None = None,
    cam_scales: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    拼接多台相机图像并标注检测结果和追踪状态。

    接收的图像已经是半分辨率，检测框坐标也已缩放。
    """
    panels = []
    for i, sn in enumerate(serials):
        if sn not in images:
            continue
        img = images[sn].copy()
        color = BOX_COLORS[i % len(BOX_COLORS)]
        dets = detections.get(sn, [])
        _draw_detections(img, dets, color)
        # 切片区域框
        if tiles and sn in tiles:
            t = tiles[sn]
            cv2.rectangle(img, (t.x, t.y), (t.x + t.w, t.y + t.h),
                          (255, 255, 0), 2)
        # AprilTag 标记（像素坐标按相机实际缩放比例转换）
        if car_loc and sn in car_loc.pixels:
            px, py = car_loc.pixels[sn]
            sx, sy = cam_scales.get(sn, (0.5, 0.5)) if cam_scales else (0.5, 0.5)
            cx, cy = int(px * sx), int(py * sy)
            cv2.drawMarker(img, (cx, cy), (0, 200, 255),
                           cv2.MARKER_DIAMOND, 20, 2)
        # 相机标签
        cv2.putText(img, sn[-3:], (10, img.shape[0] - 15),
                    FONT, 1.0, color, 2)
        panels.append(img)

    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    stitched = np.hstack(panels)
    h, w = stitched.shape[:2]

    # 画分隔线
    panel_w = panels[0].shape[1]
    for i in range(1, len(panels)):
        x = panel_w * i
        cv2.line(stitched, (x, 0), (x, h), (100, 100, 100), 1)

    # 从底部向上绘制文字信息
    line_h = 40
    # 先收集所有要绘制的行（从上到下的逻辑顺序）
    lines: list[tuple[str, tuple[int, int, int]]] = []

    lines.append((
        f"#{frame_idx}  {datetime.datetime.fromtimestamp(exposure_wall).strftime('%H:%M:%S.%f')[:-3]}"
        f"  lat={latency_ms:.0f}ms",
        TEXT_COLOR,
    ))

    det_str = "  ".join(
        f"{sn[-3:]}={len(detections.get(sn, []))}"
        for sn in serials
    )
    lines.append((f"det: {det_str}", TEXT_COLOR))

    if ball3d is not None:
        cams = "+".join(s[-3:] for s in ball3d.cameras_used)
        lines.append((
            f"3D: ({ball3d.x:.0f}, {ball3d.y:.0f}, {ball3d.z:.0f}) mm  "
            f"reproj={ball3d.reprojection_error:.1f}px  "
            f"cams={cams}  conf={ball3d.confidence:.2f}",
            TEXT_3D_COLOR,
        ))

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
    lines.append((state_str, state_color))

    if car_loc is not None:
        cams = "+".join(s[-3:] for s in car_loc.cameras_used)
        lines.append((
            f"car: ({car_loc.x:.0f}, {car_loc.y:.0f}, {car_loc.z:.0f}) mm  "
            f"yaw={math.degrees(car_loc.yaw):.1f}deg  "
            f"reproj={car_loc.reprojection_error:.1f}px  cams={cams}",
            (0, 200, 255),
        ))

    # 从底部向上绘制
    y = h - 15
    for text, color in reversed(lines):
        cv2.putText(stitched, text, (10, y), FONT, FONT_SCALE, color, FONT_THICKNESS)
        y -= line_h

    return stitched


# ══════════════════════════════════════════════════════════════════════════
#  YOLO 批量检测（处理 engine batch size 限制）
# ══════════════════════════════════════════════════════════════════════════


def _yolo_detect_n(detector, img_list, engine_batch):
    """按 engine 支持的 batch size 拆分调用 YOLO。"""
    if len(img_list) <= engine_batch:
        padded = img_list[:]
        while len(padded) < engine_batch:
            padded.append(padded[-1])
        results = detector.detect_batch(padded)
        return results[:len(img_list)]

    detections_list = []
    for i in range(0, len(img_list), engine_batch):
        batch = img_list[i:i + engine_batch]
        actual_n = len(batch)
        while len(batch) < engine_batch:
            batch.append(batch[-1])
        r = detector.detect_batch(batch)
        detections_list.extend(r[:actual_n])
    return detections_list


# ══════════════════════════════════════════════════════════════════════════
#  后台写入线程
# ══════════════════════════════════════════════════════════════════════════


class VideoWriterThread:
    """
    后台线程：接收 WriteJob → 缩放到半分辨率 → 标注 → 编码写入视频。

    队列满时（maxsize=30，即 1 秒缓冲），丢弃最旧的帧以避免主线程阻塞。
    """

    def __init__(
        self,
        video_path: str,
        frame_w: int,
        frame_h: int,
        n_cams: int,
        fps: float = 30.0,
        display: bool = False,
    ):
        self._video_path = video_path
        self._half_w = frame_w // 2
        self._half_h = frame_h // 2
        self._n_cams = n_cams
        self._display = display
        self._queue: queue.Queue[WriteJob | None] = queue.Queue(maxsize=30)
        self._stopped = False
        self._drop_count = 0

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(
            video_path, fourcc, fps,
            (self._half_w * n_cams, self._half_h),
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: WriteJob) -> None:
        """投递工作包（非阻塞）。队列满时丢弃最旧帧。"""
        try:
            self._queue.put_nowait(job)
        except queue.Full:
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
        self._queue.put(None)
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
        """缩放 → 拼接 → 写入原始视频（不标注）。"""
        panels = []
        for sn in job.serials:
            if sn in job.images:
                panels.append(cv2.resize(
                    job.images[sn], (self._half_w, self._half_h)))
        if not panels:
            return
        stitched = np.hstack(panels)
        self._writer.write(stitched)

        if self._display:
            cv2.imshow("Tracker", stitched)
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

    # ── 加载追踪配置 ──────────────────────────────────────────────────────
    _config_dir = Path(__file__).resolve().parent / "config"
    _tracker_config_path = _config_dir / "tracker.json"
    with open(_tracker_config_path, encoding="utf-8") as _f:
        tracker_cfg = json.load(_f)

    tile_size = tracker_cfg.get("tile_size", 1280)
    track_timeout_s = tracker_cfg.get("track_timeout_s", 0.3)
    search_hold_frames = tracker_cfg.get("search_hold_frames", 4)
    min_cameras_for_3d = tracker_cfg.get("min_cameras_for_3d", 2)
    max_reproj_error_px = tracker_cfg.get("max_reproj_error_px", 15.0)
    curve3_cfg = tracker_cfg.get("curve3", {})


    # ── 初始化组件 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("网球定位与实验视频保存")
    print("=" * 60)

    print("\n[1/5] 初始化 BallDetector (YOLO)...")
    detector = BallDetector()
    print(f"  模型: {detector.model_path}")

    print("[2/5] 初始化 BallLocalizer (多目标定)...")
    localizer = BallLocalizer(detector=detector)
    # 面板顺序：243（俯视）在最左，285、402 在右
    _display_order = ["DA8199243", "DA8199285", "DA8199402"]
    cam_serials = sorted(
        localizer.serials,
        key=lambda sn: _display_order.index(sn)
        if sn in _display_order else 999,
    )
    print(f"  相机: {cam_serials}")

    print("[3/5] 初始化 Curve3Tracker...")
    tracker = Curve3Tracker(**curve3_cfg)
    print(f"  ideal_hit_z={tracker.ideal_hit_z}mm, cor={tracker.cor}, "
          f"cor_xy={tracker.cor_xy}")

    print("[4/5] 初始化 CarLocalizer (AprilTag)...")
    car_localizer = CarLocalizer()
    print(f"  相机: {car_localizer.serials}")

    # ── ROS2 桥接子进程（UDP → /pc_car_loc topic）──
    _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _udp_addr = ("127.0.0.1", 5858)
    _udp_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _udp_addr2 = ("127.0.0.1", 5859)
    _ros2_proc: subprocess.Popen | None = None
    _ros2_proc2: subprocess.Popen | None = None
    _ros2_bat = _ROOT / "ros2" / "run_car_loc.bat"
    if _ros2_bat.exists():
        try:
            _ros2_proc = subprocess.Popen(
                [str(_ros2_bat)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            print(f"  ROS2 桥接已启动 (PID={_ros2_proc.pid})")
        except Exception as e:
            print(f"  ROS2 桥接启动失败: {e}")
    else:
        print(f"  ROS2 桥接脚本不存在，跳过: {_ros2_bat}")
    _ros2_bat2 = _ROOT / "ros2" / "run_predict_hit.bat"
    if _ros2_bat2.exists():
        try:
            _ros2_proc2 = subprocess.Popen(
                [str(_ros2_bat2)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            print(f"  ROS2 predict_hit 桥接已启动 (PID={_ros2_proc2.pid})")
        except Exception as e:
            print(f"  ROS2 predict_hit 桥接启动失败: {e}")

    # ── 日志容器 ────────────────────────────────────────────────────────
    log_observations: list[dict] = []
    log_predictions: list[dict] = []
    log_car_locs: list[dict] = []
    log_frames: list[dict] = []
    log_state_transitions: list[dict] = []

    # ── 打开同步相机 ────────────────────────────────────────────────────
    print("[5/5] 打开同步相机...")
    with SyncCapture.from_config() as cap:
        sync_sns = cap.sync_serials
        print(f"  同步相机: {sync_sns}")

        # 确认标定相机在同步列表中
        missing = [sn for sn in cam_serials if sn not in sync_sns]
        if missing:
            print(f"*** 错误: 标定相机 {missing} 不在同步列表 {sync_sns} 中 ***")
            return 1

        print("  等待相机稳定 (1s)...")
        time.sleep(1.0)

        first_frames = cap.get_frames(timeout_s=3.0)
        if first_frames is None:
            print("*** 错误: 无法获取初始帧 ***")
            return 1

        # 获取每台相机的分辨率
        camera_sizes = {}
        for sn in cam_serials:
            if sn in first_frames:
                img_tmp = frame_to_numpy(first_frames[sn])
                camera_sizes[sn] = (img_tmp.shape[1], img_tmp.shape[0])
        img_sample = frame_to_numpy(first_frames[cam_serials[0]])
        frame_h, frame_w = img_sample.shape[:2]
        n_cams = len(cam_serials)
        print(f"  单帧分辨率: {frame_w}x{frame_h}, "
              f"视频输出分辨率: {frame_w // 2 * n_cams}x{frame_h // 2}")

        # 初始化分片管理器
        tile_mgr = TileManager(
            camera_sizes,
            tile_size=tile_size,
            track_timeout=track_timeout_s,
            search_hold_frames=search_hold_frames,
        )
        for sn in cam_serials:
            n_tiles = tile_mgr.get_search_tile_count(sn)
            sz = camera_sizes.get(sn, (0, 0))
            print(f"  {sn[-3:]}: {sz[0]}x{sz[1]} → {n_tiles} 搜索切片")

        # ── YOLO 预热 + 自动检测 batch size ──
        print("  YOLO 预热中...")
        # 用切片大小的图像预热（匹配实际推理输入）
        _ts = tile_mgr._tile_size
        warmup_img = img_sample[:_ts, :_ts] if (
            img_sample.shape[0] >= _ts and img_sample.shape[1] >= _ts
        ) else img_sample

        engine_batch = n_cams
        for try_batch in [n_cams, n_cams - 1, 2, 1]:
            try:
                detector.detect_batch([warmup_img] * try_batch)
                engine_batch = try_batch
                break
            except Exception:
                continue

        for _ in range(4):
            _yolo_detect_n(detector, [warmup_img] * n_cams, engine_batch)

        if engine_batch >= n_cams:
            print(f"  预热完成（batch={engine_batch}，单次推理）")
        else:
            n_calls = math.ceil(n_cams / engine_batch)
            print(f"  预热完成（engine batch={engine_batch}，"
                  f"需 {n_calls} 次推理处理 {n_cams} 张图）")

        # ── 后台写入线程 ───────────────────────────────────────────────
        writer_thread: VideoWriterThread | None = None
        video_path = output_dir / f"tracker_{ts}.avi"
        if not args.no_video:
            writer_thread = VideoWriterThread(
                str(video_path), frame_w, frame_h,
                n_cams=n_cams, fps=30.0, display=args.display,
            )
            print(f"  视频输出: {video_path}")

        # ── 并行解码线程池 ──
        from concurrent.futures import ThreadPoolExecutor
        _decode_pool = ThreadPoolExecutor(max_workers=len(cam_serials))

        # ── 小车定位后台线程（不阻塞主循环）──
        _car_lock = threading.Lock()
        _car_latest: CarLoc | None = None
        _car_images: dict[str, np.ndarray] | None = None
        _car_event = threading.Event()
        _car_stop = threading.Event()

        def _car_worker():
            nonlocal _car_latest
            while not _car_stop.is_set():
                _car_event.wait(timeout=0.5)
                if _car_stop.is_set():
                    break
                _car_event.clear()
                with _car_lock:
                    imgs = _car_images
                if imgs is None:
                    continue
                t_car = time.time()
                result = car_localizer.locate(imgs, t=t_car)
                with _car_lock:
                    _car_latest = result

        _car_thread = threading.Thread(target=_car_worker, daemon=True)
        _car_thread.start()

        # ── 信号处理（确保被终止时也能保存数据）──
        _shutdown = threading.Event()
        _stop_file = output_dir / '.stop_tracker'
        _stop_file.unlink(missing_ok=True)  # 清除上次残留

        def _signal_handler(sig, frame):
            _shutdown.set()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, _signal_handler)

        # ── 主循环 ────────────────────────────────────────────────────
        frame_idx = 0
        t_start = time.monotonic()
        _wall_epoch = time.time()  # 记录启动时间（用于 JSON 日志）
        prev_state = TrackerState.IDLE
        timeout_count = 0
        _t_decode_sum = 0.0
        _t_yolo_sum = 0.0

        print(f"\n{'*' * 60}")
        print(f"  预热完成，开始追踪！（{args.duration}s）按 Ctrl+C 提前结束")
        print(f"{'*' * 60}\n")
        print("\a", end="", flush=True)

        try:
            while not _shutdown.is_set() and time.monotonic() - t_start < args.duration:
                # 检测停止文件
                if _stop_file.exists():
                    print("检测到停止文件，优雅退出...")
                    _stop_file.unlink(missing_ok=True)
                    break
                frames = cap.get_frames(timeout_s=1.0)
                if frames is None:
                    timeout_count += 1
                    continue

                # ── Bayer 解码（全分辨率，并行）──
                _t0 = time.perf_counter()
                _decode_sns = [sn for sn in cam_serials if sn in frames]
                _decode_futs = {sn: _decode_pool.submit(frame_to_numpy, frames[sn]) for sn in _decode_sns}
                images = {sn: fut.result() for sn, fut in _decode_futs.items()}
                _t1 = time.perf_counter()
                _t_decode_sum += _t1 - _t0

                exp_starts = [
                    frames[sn].exposure_start_pc
                    for sn in cam_serials if sn in frames
                ]
                exposure_pc = sum(exp_starts) / len(exp_starts)
                exposure_wall = exposure_pc  # exposure_pc 已是 epoch 时间

                # ── YOLO 分片检测 ──
                img_sns = [sn for sn in cam_serials if sn in images]
                frame_tiles: dict[str, TileRect] = {}
                tile_imgs = []
                for sn in img_sns:
                    crop, tile_rect = tile_mgr.get_tile(
                        sn, images[sn], exposure_pc)
                    tile_imgs.append(crop)
                    frame_tiles[sn] = tile_rect

                det_results = _yolo_detect_n(
                    detector, tile_imgs, engine_batch)
                _t2 = time.perf_counter()
                _t_yolo_sum += _t2 - _t1

                latency_ms = (time.time() - exposure_pc) * 1000.0

                # 映射检测坐标回全图 + 整理
                all_detections: dict[str, list[BallDetection]] = {}
                for i, sn in enumerate(img_sns):
                    all_detections[sn] = [
                        TileManager.map_detection_to_full(
                            d, frame_tiles[sn])
                        for d in det_results[i]
                    ]

                # ── 三角测量 + Curve3 更新 ──
                ball3d: Ball3D | None = None
                tracker_result: TrackerResult

                # 收集恰好检测到 1 个网球的相机
                good_dets = {
                    sn: dets[0]
                    for sn, dets in all_detections.items()
                    if len(dets) == 1
                }

                # 尝试三角测量并检查重投影误差
                if len(good_dets) >= min_cameras_for_3d:
                    candidate = localizer.triangulate(good_dets)
                    if candidate.reprojection_error <= max_reproj_error_px:
                        ball3d = candidate

                        # 3D 定位成功 → 跟踪模式
                        for sn, det in good_dets.items():
                            tile_mgr.on_3d_located(
                                sn, det.x, det.y, exposure_pc)

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
                            "cameras_used": ball3d.cameras_used,
                        })

                        if tracker_result.prediction is not None:
                            p = tracker_result.prediction
                            log_predictions.append({
                                "x": p.x, "y": p.y, "z": p.z,
                                "stage": p.stage,
                                "ct": p.ct, "ht": p.ht,
                            })
                    else:
                        # 重投影误差过大 → 当作 2D 检测
                        for sn in good_dets:
                            tile_mgr.on_2d_detected(sn, frame_tiles[sn])
                        tracker_result = TrackerResult(
                            prediction=None,
                            state=tracker.tracker_state,
                        )
                else:
                    # 不足相机数 → 检测到球的相机进入持续模式
                    for sn in good_dets:
                        tile_mgr.on_2d_detected(sn, frame_tiles[sn])
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

                # ── 小车 AprilTag 定位（后台线程异步执行）──
                with _car_lock:
                    _car_images = images
                    car_loc = _car_latest
                _car_event.set()

                if car_loc is not None:
                    log_car_locs.append({
                        "x": round(car_loc.x, 1),
                        "y": round(car_loc.y, 1),
                        "z": round(car_loc.z, 1),
                        "yaw": round(car_loc.yaw, 4),
                        "t": car_loc.t,
                        "tag_id": car_loc.tag_id,
                        "reprojection_error": round(
                            car_loc.reprojection_error, 2),
                    })

                # ── UDP 发送给 ROS2 桥接 ──
                if car_loc is not None:
                    try:
                        _udp_sock.sendto(json.dumps({
                            "topic": "car_loc",
                            "x": round(car_loc.x / 1000, 4),
                            "y": round(car_loc.y / 1000, 4),
                            "z": round(car_loc.z / 1000, 4),
                            "yaw": round(car_loc.yaw, 4),
                            "t": round(car_loc.t, 6),
                            "tag_id": car_loc.tag_id,
                        }).encode(), _udp_addr)
                    except OSError:
                        pass
                if tracker_result.prediction is not None:
                    p = tracker_result.prediction
                    try:
                        _udp_sock2.sendto(json.dumps({
                            "x": round(p.x / 1000, 4),
                            "y": round(p.y / 1000, 4),
                            "z": round(p.z / 1000, 4),
                            "stage": p.stage,
                            "ct": round(p.ct, 6),
                            "ht": round(p.ht, 6),
                            "duration": round(p.ht - p.ct, 4),
                        }).encode(), _udp_addr2)
                    except OSError:
                        pass

                # ── 投递给后台写入线程（非阻塞）──
                if writer_thread is not None:
                    writer_thread.submit(WriteJob(
                        images=images,
                        detections=all_detections,
                        serials=cam_serials,
                        exposure_wall=exposure_wall,
                        ball3d=ball3d,
                        tracker_result=tracker_result,
                        frame_idx=frame_idx,
                        latency_ms=latency_ms,
                        tiles=frame_tiles,
                        car_loc=car_loc,
                    ))

                # ── 日志（含完整标注数据）──
                frame_entry = {
                    "idx": frame_idx,
                    "exposure_pc": exposure_pc,
                    "exposure_time": datetime.datetime.fromtimestamp(
                        exposure_wall).strftime('%H:%M:%S.%f')[:-3],
                    "has_3d": ball3d is not None,
                    "state": tracker_result.state.value,
                    "latency_ms": round(latency_ms, 1),
                }
                # 检测框（含 bbox）
                frame_dets = {}
                for sn in cam_serials:
                    dets = all_detections.get(sn, [])
                    if dets:
                        frame_dets[sn] = [
                            {"x": round(d.x), "y": round(d.y),
                             "x1": round(d.x1), "y1": round(d.y1),
                             "x2": round(d.x2), "y2": round(d.y2),
                             "conf": round(d.confidence, 3)}
                            for d in dets
                        ]
                if frame_dets:
                    frame_entry["detections"] = frame_dets
                # 切片区域
                if frame_tiles:
                    frame_entry["tiles"] = {
                        sn: {"x": t.x, "y": t.y, "w": t.w, "h": t.h}
                        for sn, t in frame_tiles.items()
                    }
                # 3D 球位置
                if ball3d is not None:
                    frame_entry["ball3d"] = {
                        "x": round(ball3d.x, 1),
                        "y": round(ball3d.y, 1),
                        "z": round(ball3d.z, 1),
                        "reproj": round(ball3d.reprojection_error, 1),
                        "conf": round(ball3d.confidence, 3),
                        "cameras": ball3d.cameras_used,
                    }
                # 预测
                pred = tracker_result.prediction
                if pred is not None:
                    frame_entry["prediction"] = {
                        "x": round(pred.x, 1),
                        "y": round(pred.y, 1),
                        "z": round(pred.z, 1),
                        "stage": pred.stage,
                        "lead_ms": round((pred.ht - pred.ct) * 1000),
                    }
                # 小车位置
                if car_loc is not None:
                    frame_entry["car_loc"] = {
                        "x": round(car_loc.x),
                        "y": round(car_loc.y),
                        "z": round(car_loc.z),
                        "yaw": round(car_loc.yaw, 4),
                        "tag_id": car_loc.tag_id,
                        "cameras_used": car_loc.cameras_used,
                        "pixels": {sn: [round(u), round(v)]
                                   for sn, (u, v) in car_loc.pixels.items()},
                    }
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

    # ── 关闭小车定位线程 ──
    _car_stop.set()
    _car_event.set()
    _car_thread.join(timeout=2.0)

    # ── 关闭 ROS2 桥接子进程 ──
    _udp_sock.close()
    _udp_sock2.close()
    if _ros2_proc is not None and _ros2_proc.poll() is None:
        _ros2_proc.terminate()
        try:
            _ros2_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _ros2_proc.kill()
    if _ros2_proc2 is not None and _ros2_proc2.poll() is None:
        _ros2_proc2.terminate()
        try:
            _ros2_proc2.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _ros2_proc2.kill()
    print("  ROS2 桥接已关闭")

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
            "serials": cam_serials,
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
            "car_locs": len(log_car_locs),
            "state_transitions": len(log_state_transitions),
            "reset_times": tracker.reset_times,
            "video_frames_dropped": drop_count,
            "latency_ms_avg": round(lat_avg, 1),
            "latency_ms_min": round(lat_min, 1),
            "latency_ms_max": round(lat_max, 1),
        },
        "observations": log_observations,
        "predictions": log_predictions,
        "car_locs": log_car_locs,
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
    print(f"  小车定位:   {len(log_car_locs)}")
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
