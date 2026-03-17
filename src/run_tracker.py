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
    StationaryObjectFilter,
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
STATIONARY_BOX_COLOR = (180, 180, 180)
STATE_COLORS = {
    TrackerState.IDLE:         (128, 128, 128),   # 灰色
    TrackerState.TRACKING_S0:  (255, 200, 0),     # 青色
    TrackerState.IN_LANDING:   (0, 165, 255),     # 橙色
    TrackerState.TRACKING_S1:  (0, 255, 0),       # 绿色
    TrackerState.DONE:         (0, 0, 255),        # 红色
}


def _format_detection_counts(
    detections: dict[str, list[BallDetection]],
    serials: list[str],
) -> str:
    parts = []
    for sn in serials:
        dets = detections.get(sn, [])
        tennis_ball_count = sum(det.is_tennis_ball for det in dets)
        stationary_count = sum(det.is_stationary_object for det in dets)
        if stationary_count > 0:
            parts.append(f"{sn[-3:]}=b{tennis_ball_count}/s{stationary_count}")
        else:
            parts.append(f"{sn[-3:]}=b{tennis_ball_count}")
    return "  ".join(parts)


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


class NullRos2Sink:
    mode = "off"

    def publish_car_loc(self, payload: dict) -> None:
        return None

    def publish_predict_hit(self, payload: dict) -> None:
        return None

    def close(self) -> None:
        return None


class UdpBridgeRos2Sink:
    mode = "bridge"

    def __init__(self) -> None:
        self._sock_car = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_hit = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr_car = ("127.0.0.1", 5858)
        self._addr_hit = ("127.0.0.1", 5859)
        self._proc_car: subprocess.Popen | None = None
        self._proc_hit: subprocess.Popen | None = None

        bat_car = _ROOT / "ros2" / "run_car_loc.bat"
        if bat_car.exists():
            try:
                self._proc_car = subprocess.Popen(
                    [str(bat_car)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                print(f"  ROS2 桥接已启动 (PID={self._proc_car.pid})")
            except Exception as e:
                print(f"  ROS2 桥接启动失败: {e}")
        else:
            print(f"  ROS2 桥接脚本不存在，跳过: {bat_car}")

        bat_hit = _ROOT / "ros2" / "run_predict_hit.bat"
        if bat_hit.exists():
            try:
                self._proc_hit = subprocess.Popen(
                    [str(bat_hit)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                print(f"  ROS2 predict_hit 桥接已启动 (PID={self._proc_hit.pid})")
            except Exception as e:
                print(f"  ROS2 predict_hit 桥接启动失败: {e}")
        else:
            print(f"  ROS2 predict_hit 桥接脚本不存在，跳过: {bat_hit}")

    def publish_car_loc(self, payload: dict) -> None:
        try:
            self._sock_car.sendto(json.dumps(payload).encode(), self._addr_car)
        except OSError:
            pass

    def publish_predict_hit(self, payload: dict) -> None:
        try:
            self._sock_hit.sendto(json.dumps(payload).encode(), self._addr_hit)
        except OSError:
            pass

    def close(self) -> None:
        self._sock_car.close()
        self._sock_hit.close()
        for proc in (self._proc_car, self._proc_hit):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("  ROS2 桥接已关闭")


class DirectRos2Sink:
    mode = "direct"

    def __init__(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import String

        self._rclpy = rclpy
        self._executor_type = SingleThreadedExecutor
        self._msg_type = String
        self._pub_lock = threading.Lock()
        self._log_interval_s = 2.0
        self._ping_count = 0
        self._car_count = 0
        self._predict_count = 0
        self._last_ping_log_t = 0.0
        self._last_car_log_t = 0.0
        self._spin_stop = threading.Event()
        self._rclpy.init(args=None)
        self._node = Node("ball_tracer_tracker")
        self._car_pub = self._node.create_publisher(String, "/pc_car_loc", 10)
        self._hit_pub = self._node.create_publisher(String, "/predict_hit_pos", 10)
        self._pong_pub = self._node.create_publisher(String, "/time_sync/pong", 10)
        self._ping_sub = self._node.create_subscription(
            String, "/time_sync/ping", self._on_time_sync_ping, 10
        )
        self._executor = self._executor_type()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name="Ros2DirectSpin",
            daemon=True,
        )
        self._spin_thread.start()
        print("  ROS2 直连已启用（进程内发布）")

    def _spin_loop(self) -> None:
        while not self._spin_stop.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def _should_log(self, last_log_t: float) -> bool:
        return (time.monotonic() - last_log_t) >= self._log_interval_s

    def _on_time_sync_ping(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        payload["t2"] = time.time()
        pong = self._msg_type()
        pong.data = json.dumps(payload)
        with self._pub_lock:
            self._pong_pub.publish(pong)

        self._ping_count += 1
        if self._should_log(self._last_ping_log_t):
            self._last_ping_log_t = time.monotonic()
            self._node.get_logger().info(
                f"time_sync pong #{payload.get('seq', '?')}: replied={self._ping_count}"
            )

    def _publish(self, publisher, payload: dict) -> None:
        msg = self._msg_type()
        msg.data = json.dumps(payload)
        with self._pub_lock:
            publisher.publish(msg)

    def publish_car_loc(self, payload: dict) -> None:
        self._car_count += 1
        self._publish(self._car_pub, payload)
        if self._should_log(self._last_car_log_t):
            self._last_car_log_t = time.monotonic()
            self._node.get_logger().info(
                "/pc_car_loc "
                f"#{self._car_count}: "
                f"x={payload.get('x')} y={payload.get('y')} z={payload.get('z')} "
                f"yaw={payload.get('yaw')} tag_id={payload.get('tag_id')}"
            )

    def publish_predict_hit(self, payload: dict) -> None:
        self._predict_count += 1
        self._publish(self._hit_pub, payload)
        self._node.get_logger().info(
            "/predict_hit_pos "
            f"#{self._predict_count}: "
            f"stage={payload.get('stage')} "
            f"x={payload.get('x')} y={payload.get('y')} z={payload.get('z')} "
            f"ct={payload.get('ct')} ht={payload.get('ht')} "
            f"duration={payload.get('duration')}"
        )

    def close(self) -> None:
        self._spin_stop.set()
        self._spin_thread.join(timeout=2.0)
        self._executor.shutdown()
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        self._rclpy.shutdown()
        print("  ROS2 直连已关闭")


def _create_ros2_sink(mode: str):
    direct_error = None
    if mode in ("auto", "direct"):
        try:
            return DirectRos2Sink()
        except Exception as e:
            direct_error = e
            if mode == "direct":
                raise

    if mode in ("auto", "bridge"):
        if direct_error is not None:
            print(f"  ROS2 直连不可用，回退 bridge: {direct_error}")
        return UdpBridgeRos2Sink()

    return NullRos2Sink()


def _run_postprocess_command(
    description: str,
    command: list[str],
) -> bool:
    print(f"\n[post] {description}...")
    try:
        subprocess.run(command, cwd=str(_ROOT), check=True)
    except Exception as e:
        print(f"[post] {description} failed: {e}")
        return False
    print(f"[post] {description} done")
    return True


def _generate_post_run_artifacts(
    *,
    json_path: Path,
    video_path: Path | None,
    generate_html: bool,
    generate_annotated_video: bool,
    annotated_video_no_racket: bool,
) -> dict[str, Path]:
    generated: dict[str, Path] = {}
    python_exe = sys.executable

    if generate_html:
        html_path = json_path.with_suffix(".html")
        if _run_postprocess_command(
            "Generate HTML",
            [
                python_exe,
                str(_ROOT / "test_src" / "generate_curve3_html.py"),
                "--input",
                str(json_path),
                "--output",
                str(html_path),
            ],
        ):
            generated["html"] = html_path

    if generate_annotated_video:
        if video_path is None or not video_path.exists():
            print("[post] Skip annotated video: source video missing")
        else:
            annotated_path = json_path.with_name(f"{json_path.stem}_annotated.avi")
            command = [
                python_exe,
                str(_ROOT / "test_src" / "annotate_video.py"),
                "--input",
                str(json_path),
                "--video",
                str(video_path),
                "--output",
                str(annotated_path),
            ]
            if annotated_video_no_racket:
                command.append("--no-racket")
            if _run_postprocess_command("Generate annotated video", command):
                generated["annotated_video"] = annotated_path

    return generated


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
        draw_color = color if det.is_tennis_ball else STATIONARY_BOX_COLOR
        label = "ball" if det.is_tennis_ball else "static"
        cv2.rectangle(
            img,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            draw_color, 2,
        )
        cv2.putText(
            img, f"{label} {det.confidence:.2f}",
            (int(det.x1), int(det.y1) - 5),
            FONT, FONT_SCALE, draw_color, FONT_THICKNESS,
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

    det_str = _format_detection_counts(detections, serials)
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
        codec: str = "MJPG",
        display: bool = False,
    ):
        self._video_path = video_path
        self._half_w = frame_w // 2
        self._half_h = frame_h // 2
        self._n_cams = n_cams
        self._display = display
        self._fps = fps
        self._codec = codec
        self._queue: queue.Queue[WriteJob | None] = queue.Queue(maxsize=30)
        self._stopped = False
        self._drop_count = 0
        self._written_frame_indices: list[int] = []

        output_size = (self._half_w * n_cams, self._half_h)
        codec_candidates = [codec]
        if codec != "MJPG":
            codec_candidates.append("MJPG")

        self._writer = None
        self._actual_codec = None
        for codec_name in codec_candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(video_path, fourcc, fps, output_size)
            if writer.isOpened():
                self._writer = writer
                self._actual_codec = codec_name
                break
            writer.release()

        if self._writer is None:
            raise RuntimeError(
                f"Failed to open VideoWriter for {video_path} with codecs {codec_candidates}"
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

    def written_frame_indices(self) -> list[int]:
        """返回实际写入视频的主线程 frame_idx 顺序。"""
        return list(self._written_frame_indices)

    @property
    def actual_codec(self) -> str:
        return self._actual_codec or self._codec

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
        self._written_frame_indices.append(job.frame_idx)

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
    parser.add_argument(
        "--ros2-mode",
        choices=("auto", "direct", "bridge", "off"),
        default="auto",
        help="ROS2 output mode",
    )
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
    detection_cfg = tracker_cfg.get("detection_postprocess", {})
    detection_duplicate_iou_threshold = detection_cfg.get(
        "duplicate_iou_threshold", 0.95
    )
    detection_max_box_aspect_ratio = detection_cfg.get(
        "max_box_aspect_ratio", 1.2
    )
    stationary_cfg = tracker_cfg.get("stationary_filter", {})
    stationary_enabled = stationary_cfg.get("enabled", True)
    stationary_window_s = stationary_cfg.get("window_s", 15.0)
    stationary_radius_px = stationary_cfg.get("radius_px", 2.0)
    stationary_min_occurrences = stationary_cfg.get("min_occurrences", 6)
    car_loc_cfg = tracker_cfg.get("car_localizer", {})
    car_loc_enabled = car_loc_cfg.get("enabled", True)
    car_loc_active_interval_s = car_loc_cfg.get("active_interval_s", 0.1)
    car_loc_idle_interval_s = car_loc_cfg.get("idle_interval_s", 0.5)
    car_loc_idle_after_misses = car_loc_cfg.get("idle_after_misses", 3)
    video_output_cfg = tracker_cfg.get("video_output", {})
    video_output_codec = str(video_output_cfg.get("codec", "XVID")).upper()
    post_run_cfg = tracker_cfg.get("post_run", {})
    post_run_enabled = post_run_cfg.get("enabled", True)
    post_run_generate_html = post_run_cfg.get("generate_html", True)
    post_run_generate_annotated_video = post_run_cfg.get(
        "generate_annotated_video", True
    )
    post_run_annotated_video_no_racket = post_run_cfg.get(
        "annotated_video_no_racket", True
    )


    # ── 初始化组件 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("网球定位与实验视频保存")
    print("=" * 60)

    print("\n[1/5] 初始化 BallDetector (YOLO)...")
    detector = BallDetector(
        duplicate_iou_threshold=detection_duplicate_iou_threshold,
        max_box_aspect_ratio=detection_max_box_aspect_ratio,
    )
    print(f"  模型: {detector.model_path}")
    duplicate_iou_text = (
        "off"
        if detection_duplicate_iou_threshold is None
        else f">={detection_duplicate_iou_threshold:.2f}"
    )
    aspect_ratio_text = (
        "off"
        if detection_max_box_aspect_ratio is None
        else f"{detection_max_box_aspect_ratio:.2f}"
    )
    print(
        "  检测后处理: "
        f"duplicate_iou={duplicate_iou_text}, "
        f"max_box_aspect_ratio={aspect_ratio_text}"
    )

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
    stationary_filter = None
    if stationary_enabled:
        stationary_filter = StationaryObjectFilter(
            window_s=stationary_window_s,
            radius_px=stationary_radius_px,
            min_occurrences=stationary_min_occurrences,
        )
        print(
            "  静止过滤: "
            f"window={stationary_window_s:.1f}s, "
            f"radius={stationary_radius_px:.1f}px, "
            f"min_occurrences={stationary_min_occurrences}"
        )
    else:
        print("  静止过滤: disabled")

    print("[4/5] 初始化 CarLocalizer (AprilTag)...")
    car_localizer = CarLocalizer() if car_loc_enabled else None
    if car_localizer is not None:
        print(f"  相机: {car_localizer.serials}")
        print(
            "  小车定位节流: "
            f"active_interval={car_loc_active_interval_s:.2f}s, "
            f"idle_interval={car_loc_idle_interval_s:.2f}s, "
            f"idle_after_misses={car_loc_idle_after_misses}"
        )
    else:
        print("  小车定位: disabled")

    # ── ROS2 桥接子进程（UDP → /pc_car_loc topic）──
    _ros2_sink = _create_ros2_sink(args.ros2_mode)

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
                n_cams=n_cams, fps=30.0,
                codec=video_output_codec,
                display=args.display,
            )
            print(
                f"  视频输出: {video_path}"
                f" (codec={writer_thread.actual_codec})"
            )

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
            miss_count = 0
            next_attempt_t = 0.0
            while not _car_stop.is_set():
                _car_event.wait(timeout=0.5)
                if _car_stop.is_set():
                    break
                _car_event.clear()
                if car_localizer is None:
                    continue

                now_t = time.monotonic()
                if now_t < next_attempt_t:
                    continue

                with _car_lock:
                    imgs = _car_images
                if imgs is None:
                    continue

                t_car = time.time()
                result = car_localizer.locate(imgs, t=t_car)
                if result is None:
                    miss_count += 1
                    interval_s = (
                        car_loc_idle_interval_s
                        if miss_count >= car_loc_idle_after_misses
                        else car_loc_active_interval_s
                    )
                else:
                    miss_count = 0
                    interval_s = car_loc_active_interval_s

                next_attempt_t = time.monotonic() + max(interval_s, 0.0)
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
                    mapped_detections = [
                        TileManager.map_detection_to_full(
                            d, frame_tiles[sn])
                        for d in det_results[i]
                    ]
                    if stationary_filter is not None:
                        mapped_detections = stationary_filter.classify(
                            sn, mapped_detections, exposure_pc
                        )
                    all_detections[sn] = mapped_detections

                # ── 三角测量 + Curve3 更新 ──
                ball3d: Ball3D | None = None
                tracker_result: TrackerResult

                # 收集恰好检测到 1 个网球的相机
                good_dets: dict[str, BallDetection] = {}
                for sn, dets in all_detections.items():
                    ball_dets = [det for det in dets if det.is_tennis_ball]
                    if len(ball_dets) == 1:
                        good_dets[sn] = ball_dets[0]

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
                    _ros2_sink.publish_car_loc({
                        "topic": "car_loc",
                        "x": round(car_loc.x / 1000, 4),
                        "y": round(car_loc.y / 1000, 4),
                        "z": round(car_loc.z / 1000, 4),
                        "yaw": round(car_loc.yaw, 4),
                        "t": round(car_loc.t, 6),
                        "tag_id": car_loc.tag_id,
                    })
                if tracker_result.prediction is not None:
                    p = tracker_result.prediction
                    _ros2_sink.publish_predict_hit({
                        "x": round(p.x / 1000, 4),
                        "y": round(p.y / 1000, 4),
                        "z": round(p.z / 1000, 4),
                        "stage": p.stage,
                        "ct": round(p.ct, 6),
                        "ht": round(p.ht, 6),
                        "duration": round(p.ht - p.ct, 4),
                    })

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
                frame_det_counts = {}
                for sn in cam_serials:
                    dets = all_detections.get(sn, [])
                    if dets:
                        frame_det_counts[sn] = {
                            "tennis_ball": sum(d.is_tennis_ball for d in dets),
                            "stationary_object": sum(
                                d.is_stationary_object for d in dets
                            ),
                        }
                        frame_dets[sn] = [
                            {"x": round(d.x), "y": round(d.y),
                             "x1": round(d.x1), "y1": round(d.y1),
                             "x2": round(d.x2), "y2": round(d.y2),
                             "conf": round(d.confidence, 3),
                             "label": d.label}
                            for d in dets
                        ]
                if frame_dets:
                    frame_entry["detections"] = frame_dets
                    frame_entry["detection_counts"] = frame_det_counts
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
        written_frame_indices: list[int] = []
        if writer_thread is not None:
            print("  等待视频写入完成...")
            drop_count = writer_thread.stop()
            written_frame_indices = writer_thread.written_frame_indices()

    # ── 关闭小车定位线程 ──
    _car_stop.set()
    _car_event.set()
    _car_thread.join(timeout=2.0)

    # ── 关闭 ROS2 桥接子进程 ──
    _ros2_sink.close()

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
            "ros2_mode": _ros2_sink.mode,
            "detection_postprocess": {
                "duplicate_iou_threshold": detection_duplicate_iou_threshold,
                "max_box_aspect_ratio": detection_max_box_aspect_ratio,
            },
            "stationary_filter": {
                "enabled": stationary_enabled,
                "window_s": stationary_window_s,
                "radius_px": stationary_radius_px,
                "min_occurrences": stationary_min_occurrences,
            },
            "car_localizer": {
                "enabled": car_loc_enabled,
                "active_interval_s": car_loc_active_interval_s,
                "idle_interval_s": car_loc_idle_interval_s,
                "idle_after_misses": car_loc_idle_after_misses,
            },
            "video_output": {
                "codec": (
                    writer_thread.actual_codec
                    if writer_thread is not None
                    else video_output_codec
                ),
            },
            "post_run": {
                "enabled": post_run_enabled,
                "generate_html": post_run_generate_html,
                "generate_annotated_video": post_run_generate_annotated_video,
                "annotated_video_no_racket": post_run_annotated_video_no_racket,
            },
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
            "video_frames_written": len(written_frame_indices),
            "latency_ms_avg": round(lat_avg, 1),
            "latency_ms_min": round(lat_min, 1),
            "latency_ms_max": round(lat_max, 1),
        },
        "observations": log_observations,
        "predictions": log_predictions,
        "car_locs": log_car_locs,
        "frames": log_frames,
        "video_frame_indices": written_frame_indices,
        "state_transitions": log_state_transitions,
    }

    json_path = output_dir / f"tracker_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    generated_artifacts: dict[str, Path] = {}
    if post_run_enabled:
        generated_artifacts = _generate_post_run_artifacts(
            json_path=json_path,
            video_path=video_path if not args.no_video else None,
            generate_html=post_run_generate_html,
            generate_annotated_video=(
                post_run_generate_annotated_video and not args.no_video
            ),
            annotated_video_no_racket=post_run_annotated_video_no_racket,
        )

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
    if "html" in generated_artifacts:
        print(f"  HTML:       {generated_artifacts['html']}")
    if "annotated_video" in generated_artifacts:
        print(f"  标注视频:   {generated_artifacts['annotated_video']}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

