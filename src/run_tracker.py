# -*- coding: utf-8 -*-
"""
网球定位与实验视频保存 (DEVELOP_LIST 步骤 4.5)。

完整管线流程：
  1. SyncCapture 四相机同步拍摄（硬件触发，默认 35fps）
  2. TileManager 为每台相机选择 1000x1000 切片（跟踪/搜索模式）
  3. BallDetector YOLO 批量检测切片中的网球
  4. 若 ≥2 台相机各检测到 1 个网球 → BallLocalizer.triangulate() 多视图三角测量得到 3D 位置
  5. 将 3D 位置送入 Curve4Tracker 进行轨迹追踪与击球点预测（drag-aware 模型）
  6. 原始图像交给后台写入线程：
     - 缩小到半分辨率
     - 拼接为单路原始视频
     - VideoWriter 编码写入
     主线程不等待写入完成，立刻处理下一帧。
  7. JSON 结果日志在结束后保存；可选后处理生成 HTML 和离线标注视频

性能设计：
  - YOLO 推理在 1000x1000 切片上运行（跟踪模式：追踪球位置；搜索模式：轮询预定义区域）
  - YOLO 主线只解码 Bayer ROI；全图解码、AprilTag 和视频写入在后台流水线完成
  - 主线程只做：取帧 → ROI 解码/缩放 → YOLO → 三角测量 → curve4 → 入队

用法：
  python run_tracker.py [--duration 60] [--no-video] [--output-dir tracker_output]
                        [--display] [--ros2-mode auto|direct|bridge|off]

输出文件（每个 session 一个子目录 tracker_output/tracker_YYYYMMDD_HHMMSS/，
run 产物与后处理产物全部落在该目录内）：
  tracker_YYYYMMDD_HHMMSS.avi     — 原始拼接视频（半分辨率）
  tracker_YYYYMMDD_HHMMSS.json    — 观测、预测、状态变化等完整日志
  tracker_YYYYMMDD_HHMMSS_rosbag/ — ros2 bag（局域网全部 ROS topic 录制）
停止文件 .stop_tracker 仍固定在 tracker_output/ 根目录。
"""

from __future__ import annotations

import argparse
import atexit
import signal
import datetime
import json
import math
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Force UTF-8 logs on Windows even when stdout/stderr are redirected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
from src.ball_grabber import frame_bayer_roi_to_numpy
from src.curve4 import (
    BallObservation,
    Curve4Tracker,
    TrackerState,
    TrackerResult,
)
from src.ros2_support import (
    CYCLONEDDS_XML_PATH,
    DEFAULT_ROS_DOMAIN_ID,
    ROS2_RELIABLE_TOPICS,
    ROS2_ROOT,
    ROS2_TRACKER_PEERS,
    TRACKER_PC_IP,
    cyclonedds_file_uri,
    ensure_ros2_environment,
    make_best_effort_qos,
    make_topic_qos,
)
from src.tile_manager import TileManager, TileRect


def _terminate_process_tree(proc: subprocess.Popen | None, *, timeout: float = 3.0) -> None:
    if proc is None or proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=timeout,
            )
        except Exception:
            pass
        finally:
            try:
                proc.wait(timeout=timeout)
            except Exception:
                pass
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _ros2_python_executable() -> str:
    configured = os.environ.get("BALL_TRACER_ROS2_PYTHON", "").strip()
    if configured:
        return configured
    ros2_root = os.environ.get("BALL_TRACER_ROS2_ROOT", "").strip()
    if ros2_root:
        python_name = "python.exe" if os.name == "nt" else "bin/python"
        candidate = Path(ros2_root) / python_name
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _ros_runtime_config() -> dict[str, str]:
    domain_id = (
        os.environ.get("ROS_DOMAIN_ID")
        or os.environ.get("BALL_TRACER_ROS_DOMAIN_ID")
        or DEFAULT_ROS_DOMAIN_ID
    )
    rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    cyclone_uri = os.environ.get(
        "CYCLONEDDS_URI",
        cyclonedds_file_uri(CYCLONEDDS_XML_PATH),
    )
    return {
        "domain_id": str(domain_id),
        "rmw": str(rmw),
        "cyclone_uri": str(cyclone_uri),
        "local_ip": TRACKER_PC_IP,
        "peers": ",".join(ROS2_TRACKER_PEERS),
    }


def _topic_qos_label(topic: str, *, depth: int = 1) -> str:
    reliability = "reliable" if topic in ROS2_RELIABLE_TOPICS else "best_effort"
    return f"{reliability}(depth={int(depth)})"


def _print_ros_comm_config(prefix: str, topic_specs: list[tuple[str, int]]) -> None:
    cfg = _ros_runtime_config()
    topics_text = ", ".join(
        f"{topic}={_topic_qos_label(topic, depth=depth)}"
        for topic, depth in topic_specs
    )
    print(
        f"  {prefix}: "
        f"domain={cfg['domain_id']} "
        f"rmw={cfg['rmw']} "
        f"local_ip={cfg['local_ip']} "
        f"peers={cfg['peers']}"
    )
    print(f"  {prefix}: cyclonedds={cfg['cyclone_uri']}")
    print(f"  {prefix}: topics={topics_text}")



# ══════════════════════════════════════════════════════════════════════════
#  标注参数（仅供 VideoWriterThread 的时间戳 badge 使用）
# ══════════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_SIMPLEX
STITCHED_SERIAL_ORDER = [
    "DA7403103",  # 103
    "DA8474746",  # 746
    "DA7403087",  # 087
    "DA8571029",  # 029
]


def _format_xyz_m(x: float, y: float, z: float) -> str:
    return f"({x:.3f}, {y:.3f}, {z:.3f}) m"


def _serial_matches_selector(serial: str, selector: str) -> bool:
    selector = str(selector).strip()
    if not selector:
        return False
    return serial == selector or serial.endswith(selector)


# ══════════════════════════════════════════════════════════════════════════
#  后台写入线程需要的数据包
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class WriteJob:
    """主线程投递给视频写入线程的工作包（仅含视频编码所需字段）。"""
    images: dict[str, np.ndarray]              # {序列号: 全分辨率图像}
    serials: list[str]                         # 相机序列号顺序
    exposure_perf: float                       # perf_counter 时间
    elapsed_s: float | None
    frame_idx: int


@dataclass
class FullFrameJob:
    """全图解码线的工作包；原始 Bayer 帧不经过 YOLO 主线。"""
    frames: dict
    frame_idx: int
    exposure_pc: float
    elapsed_s: float | None
    car_loc_sampled: bool


@dataclass
class CarLocJob:
    """主线程投递给 car_loc 工作线程的工作包。"""
    frame_idx: int
    exposure_pc: float
    elapsed_s: float | None
    images: dict[str, np.ndarray]


@dataclass
class ArchiveJob:
    """主线程投递给归档线程的每帧业务数据（不含 images，JSON 构造用）。"""
    frame_idx: int
    exposure_pc: float
    elapsed_s: float | None
    latency_ms: float
    all_detections: dict[str, list[BallDetection]]
    frame_tiles: dict[str, TileRect]
    ball3d: Optional[Ball3D]
    tracker_result: TrackerResult
    car_loc_sampled: bool
    # tracker.update() 返回时的 perf_counter 时间戳（tracker_result.prediction 生成完的
    # 时刻）；用于计算 compute latency = compute_t - ct（ct 为曝光时刻）。None 表示该帧
    # 没有 prediction 产出或 tracker 根本没被调用。
    compute_done_t: Optional[float] = None


@dataclass
class CarLocEvent:
    """car_loc 工作线程或主线程（drop 时）发给归档线程的事件。"""
    frame_idx: int
    exposure_pc: float
    elapsed_s: float | None
    status: str  # "hit" / "miss" / "dropped"
    car_loc: Optional[CarLoc] = None


def _car_submit_latest(
    job_queue: queue.Queue[CarLocJob | None],
    job: CarLocJob,
) -> Optional[CarLocJob]:
    """向 maxsize=1 的 car_loc 队列投递任务；队列满时弹出旧任务并返回。"""
    while True:
        try:
            job_queue.put_nowait(job)
            return None
        except queue.Full:
            try:
                stale = job_queue.get_nowait()
            except queue.Empty:
                continue
            if stale is None:
                raise RuntimeError("car_loc queue entered shutdown while submitting")
            try:
                job_queue.put_nowait(job)
                return stale
            except queue.Full:
                continue


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
            _terminate_process_tree(proc)
        print("  ROS2 桥接已关闭")


class RosbagRecorderProcess:
    """独立 ros2 bag 录制进程：录制局域网内全部 ROS topic 到 {run_id}_rosbag/。

    通过 ros2/run_ros2.bat 启动 src/rosbag_recorder.py，ROS2 环境由 bat 自带，
    与 tracker 进程的 ros2_mode 无关。停止协议：close() 创建 stop 文件，
    子进程轮询到后调用 Recorder.stop() 并写出 bag 元数据。
    """

    def __init__(self, *, bag_dir: Path) -> None:
        self.bag_dir = bag_dir.resolve()
        self.stop_file = self.bag_dir.with_name(self.bag_dir.name + ".stop")
        self._proc: subprocess.Popen | None = None

        launcher = _ROOT / "ros2" / "run_ros2.bat"
        script = _ROOT / "src" / "rosbag_recorder.py"
        missing = [str(p) for p in (launcher, script) if not p.exists()]
        if missing:
            print(f"  rosbag 录制启动文件不存在，跳过: {', '.join(missing)}")
            return

        try:
            self.stop_file.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            self._proc = subprocess.Popen(
                [
                    str(launcher),
                    str(script),
                    "--output",
                    str(self.bag_dir),
                    "--stop-file",
                    str(self.stop_file),
                ],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            print(f"  rosbag 录制进程已启动 (PID={self._proc.pid}) -> {self.bag_dir}")
        except Exception as e:
            print(f"  rosbag 录制进程启动失败: {e}")
            self._proc = None

    def was_started(self) -> bool:
        return self._proc is not None

    def close(self, *, timeout_sec: float = 10.0) -> None:
        if self._proc is None:
            return
        try:
            self.stop_file.write_text("stop", encoding="utf-8")
        except OSError:
            pass
        try:
            self._proc.wait(timeout=max(float(timeout_sec), 0.1))
        except subprocess.TimeoutExpired:
            _terminate_process_tree(self._proc)
        try:
            self.stop_file.unlink(missing_ok=True)
        except OSError:
            pass
        print("  rosbag 录制进程已关闭")


class DirectRos2Sink:
    mode = "direct"

    def __init__(self) -> None:
        ensure_ros2_environment()
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import String

        self._rclpy = rclpy
        self._executor_type = SingleThreadedExecutor
        self._msg_type = String
        self._pub_lock = threading.Lock()
        self._car_count = 0
        self._spin_stop = threading.Event()
        self._rclpy.init(args=None)
        self._node = Node("ball_tracer_tracker")
        self._car_pub = self._node.create_publisher(
            String, "/pc_car_loc", make_best_effort_qos()
        )
        self._executor = self._executor_type()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name="Ros2DirectSpin",
            daemon=True,
        )
        self._spin_thread.start()
        print("  ROS2 car_loc publish enabled (/pc_car_loc only)")
        _print_ros_comm_config(
            "ROS2 car_loc",
            [
                ("/pc_car_loc", 1),
            ],
        )

    def _spin_loop(self) -> None:
        while not self._spin_stop.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def _publish(self, publisher, payload: dict) -> None:
        msg = self._msg_type()
        msg.data = json.dumps(payload)
        with self._pub_lock:
            publisher.publish(msg)

    def publish_car_loc(self, payload: dict) -> None:
        self._car_count += 1
        self._publish(self._car_pub, payload)
        print(
            "/pc_car_loc "
            f"#{self._car_count}: "
            f"x={payload.get('x')} y={payload.get('y')} z={payload.get('z')} "
            f"yaw={payload.get('yaw')} yaw_valid={payload.get('yaw_valid')} "
            f"tag_id={payload.get('tag_id')}",
            flush=True,
        )

    def publish_predict_hit(self, payload: dict) -> None:
        return None

    def close(self) -> None:
        self._spin_stop.set()
        self._spin_thread.join(timeout=2.0)
        self._executor.shutdown()
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
        print("  ROS2 直连已关闭")


def _create_ros2_sink(mode: str):
    if mode == "off":
        return NullRos2Sink()
    return DirectRos2Sink()


def _report_tool_env() -> dict[str, str]:
    """报告工具子进程的环境：把 ROS2 pixi 环境从 PYTHONPATH 里摘掉。

    本进程为了 import rclpy 已经把 ROS2 site-packages 前插进 PYTHONPATH
    （ros2_support.ensure_ros2_environment）。generate_curve3_html / annotate_video
    用 tracker venv 的解释器跑，继承这份 PYTHONPATH 后 `import numpy` 会先命中
    conda-forge 那一份、在 venv 解释器里 C 扩展加载失败——2026-08-05 之前的报告
    因此整场缺「拍面yaw,pitch@臂最后更新HT」两列（_add_face_angles 吞掉 ImportError 静默显示
    —），手工重跑（无此 PYTHONPATH）才有。这两个工具不碰 ROS2，按 ROS2 根前缀剔除。
    """
    env = os.environ.copy()
    raw = env.get("PYTHONPATH", "")
    if not raw:
        return env
    ros2_root = os.path.normcase(str(ROS2_ROOT))
    kept = [
        entry
        for entry in raw.split(os.pathsep)
        if entry
        and not os.path.normcase(os.path.abspath(entry)).startswith(ros2_root)
    ]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    return env


def _run_postprocess_command(
    description: str,
    command: list[str],
    env: dict[str, str] | None = None,
) -> bool:
    print(f"\n[post] {description}...")
    try:
        subprocess.run(command, cwd=str(_ROOT), check=True, env=env)
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
    # 纯 Python 报告工具（非 ROS2）用干净 PYTHONPATH，见 _report_tool_env
    report_env = _report_tool_env()
    json_path = json_path.resolve()
    video_path = video_path.resolve() if video_path is not None else None

    # ── rosbag → 机械臂 JSON（供 HTML Arm tab；bag 不存在时跳过）──
    arm_json_path: Path | None = None
    rk_tracking_json_path: Path | None = None
    bag_dir = json_path.with_name(f"{json_path.stem}_rosbag")
    if (bag_dir / "metadata.yaml").exists():
        rk_candidate = json_path.with_name(f"{json_path.stem}_rk_tracking.json")
        if _run_postprocess_command(
            "Extract RK tracking rosbag",
            [
                str(_ROOT / "ros2" / "run_ros2.bat"),
                str(_ROOT / "test_src" / "extract_rk_tracking_bag.py"),
                "--bag",
                str(bag_dir),
                "--output",
                str(rk_candidate),
            ],
        ) and rk_candidate.exists():
            rk_tracking_json_path = rk_candidate
            generated["rk_tracking_json"] = rk_candidate

        candidate = json_path.with_name(f"{json_path.stem}_arm.json")
        if _run_postprocess_command(
            "Extract arm rosbag",
            [
                str(_ROOT / "ros2" / "run_ros2.bat"),
                str(_ROOT / "test_src" / "extract_arm_bag.py"),
                "--bag",
                str(bag_dir),
                "--output",
                str(candidate),
            ],
        ) and candidate.exists():
            arm_json_path = candidate
            generated["arm_json"] = candidate

    if generate_html:
        html_path = json_path.with_suffix(".html")
        html_command = [
            python_exe,
            str(_ROOT / "test_src" / "generate_curve3_html.py"),
            "--input",
            str(json_path),
            "--output",
            str(html_path),
        ]
        if arm_json_path is not None:
            html_command.extend(["--arm-json", str(arm_json_path)])
        if rk_tracking_json_path is not None:
            html_command.extend(["--rk-tracking-json", str(rk_tracking_json_path)])
        if _run_postprocess_command("Generate HTML", html_command, env=report_env):
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
            if _run_postprocess_command(
                "Generate annotated video", command, env=report_env
            ):
                generated["annotated_video"] = annotated_path

    return generated


# ══════════════════════════════════════════════════════════════════════════
#  视频时间戳 badge（仅在 VideoWriterThread 中调用）
# ══════════════════════════════════════════════════════════════════════════


def _draw_badge(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    font_scale: float = 1.2,
    thickness: int = 3,
) -> int:
    """Draw a high-contrast text badge and return the box right edge."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, FONT, font_scale, thickness
    )
    pad_x = 12
    pad_y = 10
    box_tl = (x, y)
    box_br = (x + text_w + pad_x * 2, y + text_h + baseline + pad_y * 2)
    cv2.rectangle(img, box_tl, box_br, (0, 0, 0), -1)
    cv2.rectangle(img, box_tl, box_br, color, 2)
    cv2.putText(
        img,
        text,
        (x + pad_x, y + pad_y + text_h),
        FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return box_br[0]


def _build_video_time_text(
    frame_idx: int,
    exposure_perf: float,
    elapsed_s: float | None,
) -> str:
    if elapsed_s is None:
        return f"#{frame_idx}  perf={exposure_perf:.6f}s"
    return f"#{frame_idx}  t={elapsed_s:.3f}s  perf={exposure_perf:.6f}s"


def _shape_gate_summary() -> dict[str, float]:
    """把 BallDetector 的形状门计数整理成 session json 里的一段。

    只看 3D 观测无法区分「相机没看见球」和「看见了但框被拦掉」。拒绝率高
    通常是曝光过长把球拉成条（拖影 ≈ 像面速度 × 曝光），回球段像面速度是来球的
    2~3 倍，最先在那里成片漏检 —— 对症是降曝光，不是无脑放宽 max_box_aspect_ratio
    （放宽会让一台相机冒出第 2 个球框，反而在 "恰好 1 个检测" 那步把整台相机废掉）。
    """
    stats = BallDetector.shape_gate_stats
    kept = int(stats.get("ball_kept", 0))
    rejected = int(stats.get("ball_rejected", 0))
    total = kept + rejected
    return {
        "ball_kept": kept,
        "ball_rejected": rejected,
        "rejected_frac": round(rejected / total, 4) if total else 0.0,
        "rejected_aspect_mean": (
            round(stats.get("rejected_aspect_sum", 0.0) / rejected, 3) if rejected else 0.0
        ),
        "rejected_aspect_max": round(stats.get("rejected_aspect_max", 0.0), 3),
    }


def _grid_dimensions(n_panels: int, cols: int = 2) -> tuple[int, int]:
    cols = max(1, min(cols, n_panels))
    rows = max(1, math.ceil(n_panels / cols))
    return cols, rows


def _grid_slot(
    index: int,
    panel_w: int,
    panel_h: int,
    *,
    cols: int = 2,
) -> tuple[int, int]:
    col = index % cols
    row = index // cols
    return col * panel_w, row * panel_h


# ══════════════════════════════════════════════════════════════════════════
#  YOLO 批量检测（处理 engine batch size 限制）
# ══════════════════════════════════════════════════════════════════════════


def _yolo_detect_n(detector, img_list, engine_batch):
    """按 engine 支持的 batch size 拆分调用 YOLO。"""
    if not img_list or engine_batch <= 0:
        return []
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


def _normalize_video_codec(codec: str) -> str:
    codec_text = str(codec or "").strip()
    if not codec_text:
        return "MJPG"
    if codec_text.lower() == "avc1":
        return "avc1"
    return codec_text.upper()


def _video_container_suffix(codec: str) -> str:
    normalized = _normalize_video_codec(codec)
    if normalized.lower() in {"avc1", "h264", "mp4v"}:
        return ".mp4"
    return ".avi"


def _candidate_video_codecs(codec: str) -> list[str]:
    normalized = _normalize_video_codec(codec)
    if _video_container_suffix(normalized) == ".mp4":
        ordered = [normalized]
        for alt in ("avc1", "H264", "mp4v"):
            if alt.lower() != normalized.lower():
                ordered.append(alt)
        return ordered

    ordered = [normalized]
    for alt in ("XVID", "MJPG"):
        if alt.upper() != normalized.upper():
            ordered.append(alt)
    return ordered


def _candidate_video_backends(
    backend: str,
    *,
    prefer_mp4: bool,
) -> list[tuple[str, int | None]]:
    normalized = str(backend or "auto").strip().lower()
    cap_ffmpeg = getattr(cv2, "CAP_FFMPEG", None)
    cap_msmf = getattr(cv2, "CAP_MSMF", None)

    if normalized == "ffmpeg":
        ordered = [("FFMPEG", cap_ffmpeg), ("DEFAULT", None)]
    elif normalized == "msmf":
        ordered = [("MSMF", cap_msmf), ("DEFAULT", None)]
    elif normalized == "default":
        ordered = [("DEFAULT", None)]
    elif os.name == "nt" and prefer_mp4:
        ordered = [("MSMF", cap_msmf), ("FFMPEG", cap_ffmpeg), ("DEFAULT", None)]
    else:
        ordered = [("FFMPEG", cap_ffmpeg), ("MSMF", cap_msmf), ("DEFAULT", None)]

    deduped: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for name, api in ordered:
        key = (name, api)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, api))
    return deduped


def _infer_engine_batch_from_model_path(model_path: Path) -> int | None:
    if model_path.suffix.lower() != ".engine":
        return None
    name = model_path.name.lower()
    marker = "_b"
    idx = name.find(marker)
    while idx >= 0:
        start = idx + len(marker)
        end = start
        while end < len(name) and name[end].isdigit():
            end += 1
        if end > start and (end == len(name) or name[end] in {"_", "."}):
            return int(name[start:end])
        idx = name.find(marker, idx + 1)
    return None


def _infer_model_input_size_from_model_path(model_path: Path) -> int | None:
    match = re.search(r"_b\d+_(\d+)(?:_|$)", model_path.stem)
    if not match:
        return None
    size = int(match.group(1))
    return size if size > 0 else None


def _select_detector_model_for_active_cams(
    model_path: Path,
    active_yolo_cams: int,
    *,
    target_input_size: int | None = None,
) -> Path:
    target_batch = max(int(active_yolo_cams), 1)
    current_batch = _infer_engine_batch_from_model_path(model_path)
    current_input_size = _infer_model_input_size_from_model_path(model_path)
    normalized_target_input_size = (
        max(int(target_input_size), 1)
        if target_input_size is not None
        else None
    )
    if (
        current_batch == target_batch
        and (
            normalized_target_input_size is None
            or current_input_size == normalized_target_input_size
        )
    ):
        return model_path

    candidate_paths: list[Path] = []
    if current_batch is not None:
        candidate_name = model_path.stem
        candidate_name = re.sub(
            r"_b\d+(?=_|$)",
            f"_b{target_batch}",
            candidate_name,
            count=1,
        )
        if normalized_target_input_size is not None and current_input_size is not None:
            candidate_name = re.sub(
                r"_\d+$",
                f"_{normalized_target_input_size}",
                candidate_name,
                count=1,
            )
        candidate_path = model_path.with_name(f"{candidate_name}{model_path.suffix}")
        candidate_paths.append(candidate_path)

    if model_path.parent.exists():
        candidate_paths.extend(model_path.parent.glob(f"*{model_path.suffix}"))

    best_match = model_path
    best_score: tuple[int, int, int, int, str] | None = None
    seen_candidates: set[Path] = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen_candidates:
            continue
        seen_candidates.add(candidate_path)
        if not candidate_path.exists():
            continue
        candidate_batch = _infer_engine_batch_from_model_path(candidate_path)
        candidate_input_size = _infer_model_input_size_from_model_path(candidate_path)
        size_match = (
            normalized_target_input_size is None
            or candidate_input_size == normalized_target_input_size
        )
        size_penalty = 0
        if (
            normalized_target_input_size is not None
            and candidate_input_size is not None
            and candidate_input_size != normalized_target_input_size
        ):
            size_penalty = abs(candidate_input_size - normalized_target_input_size)
        score = (
            int(candidate_batch == target_batch),
            int(size_match),
            -size_penalty,
            int(candidate_path == model_path),
            candidate_path.name,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_match = candidate_path

    return best_match


def _resolve_engine_batch(
    detector,
    warmup_img,
    n_ball_detect_cams: int,
    n_cams: int,
) -> int:
    last_error: Exception | None = None
    warmup_shape = getattr(warmup_img, "shape", None)
    warmup_dtype = getattr(warmup_img, "dtype", None)
    fixed_engine_batch = _infer_engine_batch_from_model_path(detector.model_path)
    if fixed_engine_batch is not None and fixed_engine_batch >= 1:
        try:
            detector.detect_batch([warmup_img] * fixed_engine_batch)
            return fixed_engine_batch
        except Exception as exc:
            last_error = exc

    try_batches: list[int] = []
    for try_batch in [n_ball_detect_cams, n_cams, n_ball_detect_cams - 1, 2, 1]:
        if try_batch >= 1 and try_batch not in try_batches:
            try_batches.append(try_batch)
    for try_batch in try_batches:
        try:
            detector.detect_batch([warmup_img] * try_batch)
            return try_batch
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            "detector warmup failed for all candidate batch sizes "
            f"{try_batches} on {detector.model_path.name} "
            f"(shape={warmup_shape}, dtype={warmup_dtype}): {last_error}"
        ) from last_error

    raise RuntimeError(
        "detector warmup failed before trying any TensorRT batch size "
        f"for {detector.model_path.name} "
        f"(shape={warmup_shape}, dtype={warmup_dtype})"
    )


# ══════════════════════════════════════════════════════════════════════════
#  后台写入线程
# ══════════════════════════════════════════════════════════════════════════


class VideoWriterThread:
    """
    后台线程：接收 WriteJob → 编码写入视频。

    队列满时（maxsize=30，即 1 秒缓冲），丢弃最旧的帧以避免主线程阻塞。

    两种模式：
      - 默认（grid）：每相机缩到 1/2 分辨率，2x2 拼接 + 时间戳 badge，单文件输出
      - full_res：每相机独立全分辨率 mp4，无拼接、无 badge（用于训练数据采集）
    """

    def __init__(
        self,
        video_path: str,
        frame_w: int,
        frame_h: int,
        n_cams: int,
        fps: float = 30.0,
        codec: str = "MJPG",
        backend: str = "auto",
        prefer_hw_accel: bool = True,
        display: bool = False,
        full_res: bool = False,
        cam_serials: list[str] | None = None,
    ):
        self._video_path = video_path
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._half_w = frame_w // 2
        self._half_h = frame_h // 2
        self._n_cams = n_cams
        self._grid_cols, self._grid_rows = _grid_dimensions(n_cams, cols=2)
        self._display = display and not full_res
        self._fps = fps
        self._codec = _normalize_video_codec(codec)
        self._backend = str(backend or "auto").strip().lower()
        self._prefer_hw_accel = bool(prefer_hw_accel)
        self._full_res = bool(full_res)
        self._cam_serials = list(cam_serials or [])
        if self._full_res and not self._cam_serials:
            raise ValueError("full_res mode requires cam_serials")
        self._queue: queue.Queue[WriteJob | None] = queue.Queue(maxsize=30)
        self._stopped = False
        self._drop_count = 0
        self._written_frame_indices: list[int] = []
        self._queue_max_size = 0
        self._process_count = 0
        self._process_time_sum = 0.0
        self._process_time_max = 0.0

        if self._full_res:
            base = Path(video_path)
            self._video_paths = [
                str(base.with_name(f"{base.stem}_{sn[-3:]}{base.suffix}"))
                for sn in self._cam_serials
            ]
            output_size = (frame_w, frame_h)
            self._stitched = None
            self._panel_views = []
        else:
            self._video_paths = [video_path]
            output_size = (
                self._half_w * self._grid_cols,
                self._half_h * self._grid_rows,
            )
            self._stitched = np.zeros(
                (output_size[1], output_size[0], 3),
                dtype=np.uint8,
            )
            self._panel_views = []
            for idx in range(self._n_cams):
                x, y = _grid_slot(
                    idx,
                    self._half_w,
                    self._half_h,
                    cols=self._grid_cols,
                )
                self._panel_views.append(
                    self._stitched[y:y + self._half_h, x:x + self._half_w]
                )

        self._writers: list[cv2.VideoWriter] = []
        self._actual_codec = None
        self._backend_name = "DEFAULT"
        self._hw_accel_requested = False
        for path in self._video_paths:
            writer, codec_name, backend_name, hw_accel = self._open_writer(
                path, output_size
            )
            self._writers.append(writer)
            self._actual_codec = codec_name
            self._backend_name = backend_name
            self._hw_accel_requested = hw_accel

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open_writer(
        self, path: str, output_size: tuple[int, int]
    ) -> tuple[cv2.VideoWriter, str, str, bool]:
        prefer_mp4 = _video_container_suffix(self._codec) == ".mp4"
        for backend_name, api_preference in _candidate_video_backends(
            self._backend,
            prefer_mp4=prefer_mp4,
        ):
            for codec_name in _candidate_video_codecs(self._codec):
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                writer = None
                hw_accel_requested = False
                try:
                    if api_preference is None:
                        writer = cv2.VideoWriter(
                            path, fourcc, self._fps, output_size,
                        )
                    else:
                        params = None
                        if backend_name == "FFMPEG" and self._prefer_hw_accel:
                            hw_prop = getattr(
                                cv2, "VIDEOWRITER_PROP_HW_ACCELERATION", None,
                            )
                            accel_any = getattr(
                                cv2, "VIDEO_ACCELERATION_ANY", None,
                            )
                            if hw_prop is not None and accel_any is not None:
                                params = [int(hw_prop), int(accel_any)]
                                hw_accel_requested = True
                        if params is not None:
                            writer = cv2.VideoWriter(
                                path, api_preference, fourcc,
                                self._fps, output_size, params=params,
                            )
                        else:
                            writer = cv2.VideoWriter(
                                path, api_preference, fourcc,
                                self._fps, output_size,
                            )
                except Exception:
                    writer = None
                if writer is None:
                    continue
                if writer.isOpened():
                    try:
                        actual_backend = writer.getBackendName()
                    except Exception:
                        actual_backend = backend_name
                    return writer, codec_name, actual_backend, hw_accel_requested
                writer.release()
        raise RuntimeError(
            "Failed to open VideoWriter for "
            f"{path} with codec={self._codec} backend={self._backend}"
        )

    def submit(self, job: WriteJob) -> None:
        """投递工作包（非阻塞）。队列满时丢弃最旧帧。"""
        try:
            self._queue.put_nowait(job)
            self._queue_max_size = max(self._queue_max_size, self._queue.qsize())
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._drop_count += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
                self._queue_max_size = max(self._queue_max_size, self._queue.qsize())
            except queue.Full:
                pass

    def stop(self) -> int:
        """通知线程停止，等待队列排空，释放资源。返回丢弃帧数。"""
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        for w in self._writers:
            w.release()
        if self._display:
            cv2.destroyAllWindows()
        return self._drop_count

    @property
    def video_paths(self) -> list[str]:
        return list(self._video_paths)

    @property
    def full_res(self) -> bool:
        return self._full_res

    def written_frame_indices(self) -> list[int]:
        """返回实际写入视频的主线程 frame_idx 顺序。"""
        return list(self._written_frame_indices)

    def stats(self) -> dict[str, float | int | bool]:
        avg_process_ms = (
            self._process_time_sum / self._process_count * 1000.0
            if self._process_count > 0
            else 0.0
        )
        return {
            "enabled": True,
            "queue_max_size": int(self._queue_max_size),
            "process_count": int(self._process_count),
            "avg_process_ms": round(avg_process_ms, 1),
            "max_process_ms": round(self._process_time_max * 1000.0, 1),
            "codec": self.actual_codec,
            "backend": self.backend_name,
            "hw_accel_requested": bool(self._hw_accel_requested),
        }

    @property
    def actual_codec(self) -> str:
        return self._actual_codec or self._codec

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def hw_accel_requested(self) -> bool:
        return self._hw_accel_requested

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            self._process(job)

    def _process(self, job: WriteJob) -> None:
        """编码写入原始视频（无标注）。"""
        t0 = time.perf_counter()
        try:
            if not job.serials:
                return
            if self._full_res:
                for idx, sn in enumerate(self._cam_serials):
                    if idx >= len(self._writers):
                        break
                    img = job.images.get(sn)
                    if img is not None:
                        self._writers[idx].write(img)
                self._written_frame_indices.append(job.frame_idx)
                return

            self._stitched.fill(0)
            for idx, sn in enumerate(job.serials):
                if idx >= len(self._panel_views):
                    break
                panel = self._panel_views[idx]
                if sn in job.images:
                    cv2.resize(
                        job.images[sn],
                        (self._half_w, self._half_h),
                        dst=panel,
                    )
                else:
                    panel.fill(0)
            for col in range(1, self._grid_cols):
                x = self._half_w * col
                cv2.line(
                    self._stitched,
                    (x, 0),
                    (x, self._stitched.shape[0]),
                    (100, 100, 100),
                    1,
                )
            for row in range(1, self._grid_rows):
                y = self._half_h * row
                cv2.line(
                    self._stitched,
                    (0, y),
                    (self._stitched.shape[1], y),
                    (100, 100, 100),
                    1,
                )
            _draw_badge(
                self._stitched,
                _build_video_time_text(
                    job.frame_idx, job.exposure_perf, job.elapsed_s
                ),
                10,
                10,
                (255, 255, 255),
                font_scale=0.9,
                thickness=2,
            )
            self._writers[0].write(self._stitched)
            self._written_frame_indices.append(job.frame_idx)

            if self._display:
                cv2.imshow("Tracker", self._stitched)
                cv2.waitKey(1)
        finally:
            dt = time.perf_counter() - t0
            self._process_count += 1
            self._process_time_sum += dt
            self._process_time_max = max(self._process_time_max, dt)


# ══════════════════════════════════════════════════════════════════════════
#  归档线程（构造 frame_entry / 维护 log 列表 / 合并 car_loc 结果）
# ══════════════════════════════════════════════════════════════════════════


class ArchiveThread:
    """
    独立线程：串行消费 ArchiveJob 和 CarLocEvent。

    职责：
      - 构造每帧 frame_entry dict，追加到 log_frames；
      - 根据 ball3d / prediction / state 变化维护 log_observations /
        log_predictions / log_state_transitions；
      - 处理 car_loc 事件（hit / miss / dropped），更新 frame_entry 并追加
        log_car_locs。

    主线程退出时调用 stop()，拿到 result() 中的 5 个 log 列表、
    frame_entry_by_idx（给 video_frame_idx 回填用）、以及 car_loc_missed 计数。
    """

    def __init__(
        self,
        cam_serials: list[str],
        has_car_localizer: bool,
    ) -> None:
        self._cam_serials = list(cam_serials)
        self._has_car_localizer = bool(has_car_localizer)
        self._queue: queue.Queue[ArchiveJob | CarLocEvent | None] = queue.Queue()
        self._log_frames: list[dict] = []
        self._log_observations: list[dict] = []
        self._log_predictions: list[dict] = []
        self._log_car_locs: list[dict] = []
        self._log_state_transitions: list[dict] = []
        self._frame_entry_by_idx: dict[int, dict] = {}
        # 归档线程消费 ArchiveJob 和 CarLocEvent 两种消息。car_loc worker
        # 处理较快时，CarLocEvent 可能先于对应的 ArchiveJob 到达队列（主线程
        # 先派发 CarLocJob、后派发 ArchiveJob，但 worker 端是否先完成不确定）。
        # 此时暂存到 _pending_events，待 ArchiveJob 创建 entry 后再应用。
        self._pending_events: dict[int, CarLocEvent] = {}
        self._car_loc_missed = 0
        self._prev_state = TrackerState.IDLE
        self._thread = threading.Thread(
            target=self._run, name="ArchiveThread", daemon=True,
        )
        self._thread.start()

    def submit(self, msg: ArchiveJob | CarLocEvent) -> None:
        self._queue.put_nowait(msg)

    def stop(self, *, timeout_s: float = 10.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("archive thread did not finish before shutdown")

    def result(self) -> dict:
        return {
            "log_frames": self._log_frames,
            "log_observations": self._log_observations,
            "log_predictions": self._log_predictions,
            "log_car_locs": self._log_car_locs,
            "log_state_transitions": self._log_state_transitions,
            "frame_entry_by_idx": self._frame_entry_by_idx,
            "car_loc_missed_frames": self._car_loc_missed,
        }

    def _run(self) -> None:
        while True:
            msg = self._queue.get()
            if msg is None:
                return
            if isinstance(msg, ArchiveJob):
                self._process_archive(msg)
            else:
                self._process_car_loc_event(msg)

    def _process_archive(self, job: ArchiveJob) -> None:
        state = job.tracker_result.state
        if state != self._prev_state:
            self._log_state_transitions.append({
                "frame": job.frame_idx,
                "t": job.exposure_pc,
                "from": self._prev_state.value,
                "to": state.value,
            })
            self._prev_state = state

        entry: dict = {
            "idx": job.frame_idx,
            "exposure_pc": job.exposure_pc,
            "elapsed_s": (
                round(job.elapsed_s, 3) if job.elapsed_s is not None else None
            ),
            "has_3d": job.ball3d is not None,
            "state": state.value,
            "latency_ms": round(job.latency_ms, 1),
        }
        if self._has_car_localizer:
            entry["car_loc_sampled"] = job.car_loc_sampled
            entry["car_loc_status"] = (
                "pending" if job.car_loc_sampled else "skipped"
            )

        frame_dets: dict[str, list[dict]] = {}
        frame_det_counts: dict[str, dict[str, int]] = {}
        for sn in self._cam_serials:
            dets = job.all_detections.get(sn, [])
            if not dets:
                continue
            frame_det_counts[sn] = {
                "tennis_ball": sum(d.is_tennis_ball for d in dets),
                "stationary_object": sum(
                    d.is_stationary_object for d in dets
                ),
            }
            frame_dets[sn] = [
                {
                    "x": round(d.x), "y": round(d.y),
                    "x1": round(d.x1), "y1": round(d.y1),
                    "x2": round(d.x2), "y2": round(d.y2),
                    "conf": round(d.confidence, 3),
                    "label": d.label,
                }
                for d in dets
            ]
        if frame_dets:
            entry["detections"] = frame_dets
            entry["detection_counts"] = frame_det_counts

        if job.frame_tiles:
            entry["tiles"] = {
                sn: {"x": t.x, "y": t.y, "w": t.w, "h": t.h}
                for sn, t in job.frame_tiles.items()
            }

        if job.ball3d is not None:
            b = job.ball3d
            entry["ball3d"] = {
                "x": round(b.x, 4), "y": round(b.y, 4), "z": round(b.z, 4),
                "reproj": round(b.reprojection_error, 1),
                "conf": round(b.confidence, 3),
                "cameras": b.cameras_used,
            }
            self._log_observations.append({
                "x": b.x, "y": b.y, "z": b.z,
                "t": job.exposure_pc,
                "reproj_err": b.reprojection_error,
                "confidence": b.confidence,
                "cameras_used": b.cameras_used,
            })

        pred = job.tracker_result.prediction
        if pred is not None:
            compute_t = job.compute_done_t
            compute_latency_ms = (
                round((compute_t - pred.ct) * 1000)
                if compute_t is not None else None
            )
            entry["prediction"] = {
                "x": round(pred.x, 4),
                "y": round(pred.y, 4),
                "z": round(pred.z, 4),
                "vx": round(pred.vx, 4),
                "vy": round(pred.vy, 4),
                "vz": round(pred.vz, 4),
                "stage": pred.stage,
                "lead_ms": round((pred.ht - pred.ct) * 1000),
                "compute_latency_ms": compute_latency_ms,
            }
            log_pred = {
                "x": pred.x, "y": pred.y, "z": pred.z,
                "vx": pred.vx, "vy": pred.vy, "vz": pred.vz,
                "stage": pred.stage, "ct": pred.ct, "ht": pred.ht,
            }
            if compute_t is not None:
                # perf_counter 时刻，tracker.update() 返回那一刻；报表 latency = compute_t - ct
                log_pred["compute_t"] = compute_t
            self._log_predictions.append(log_pred)

        self._log_frames.append(entry)
        self._frame_entry_by_idx[job.frame_idx] = entry
        # 若该帧的 car_loc 结果先到，现在回填
        pending = self._pending_events.pop(job.frame_idx, None)
        if pending is not None:
            self._apply_car_loc(entry, pending)

    def _process_car_loc_event(self, ev: CarLocEvent) -> None:
        entry = self._frame_entry_by_idx.get(ev.frame_idx)
        if entry is None:
            # ArchiveJob 还没到，先缓存，后面创建 entry 时再应用
            self._pending_events[ev.frame_idx] = ev
            return
        self._apply_car_loc(entry, ev)

    def _apply_car_loc(self, entry: dict, ev: CarLocEvent) -> None:
        if ev.status == "dropped":
            if entry.get("car_loc_status") == "pending":
                entry["car_loc_status"] = "dropped"
            return
        if ev.status == "miss":
            entry["car_loc_status"] = "miss"
            self._car_loc_missed += 1
            return
        loc = ev.car_loc
        if loc is None:
            return
        entry["car_loc_status"] = "hit"
        entry["car_loc"] = {
            "x": round(loc.x, 4),
            "y": round(loc.y, 4),
            "z": round(loc.z, 4),
            "yaw": round(loc.yaw, 4),
            "yaw_valid": loc.yaw_valid,
            "t": ev.exposure_pc,
            "elapsed_s": (
                round(ev.elapsed_s, 3)
                if ev.elapsed_s is not None else None
            ),
            "tag_id": loc.tag_id,
            "tag_ids": loc.tag_ids,
            "reference": "car_base",
            "cameras_used": loc.cameras_used,
            "pixels": {
                sn: [round(u), round(v)]
                for sn, (u, v) in loc.pixels.items()
            },
        }
        self._log_car_locs.append({
            "frame_idx": ev.frame_idx,
            "x": round(loc.x, 4),
            "y": round(loc.y, 4),
            "z": round(loc.z, 4),
            "yaw": round(loc.yaw, 4),
            "yaw_valid": loc.yaw_valid,
            "t": ev.exposure_pc,
            "elapsed_s": (
                round(ev.elapsed_s, 3) if ev.elapsed_s is not None else None
            ),
            "tag_id": loc.tag_id,
            "tag_ids": loc.tag_ids,
            "reference": "car_base",
            "cameras_used": loc.cameras_used,
            "reprojection_error": round(loc.reprojection_error, 2),
        })


# ══════════════════════════════════════════════════════════════════════════
#  主管线
# ══════════════════════════════════════════════════════════════════════════


def main() -> int:
    config_dir = Path(__file__).resolve().parent / "config"
    parser = argparse.ArgumentParser(
        description="网球定位与实验视频保存 (Step 4.5)")
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="录制时长（秒，默认 60）")
    parser.add_argument(
        "--no-video", action="store_true",
        help="不保存视频")
    parser.add_argument(
        "--no-log", action="store_true",
        help="退出时不保存 JSON/HTML/PC logger 等日志文件")
    parser.add_argument(
        "--output-dir", type=str, default="tracker_output",
        help="输出目录（默认 tracker_output/）")
    parser.add_argument(
        "--display", action="store_true",
        help="实时显示拼接画面（按 q 退出）")
    parser.add_argument(
        "--ros2-mode",
        choices=("auto", "direct", "bridge", "off"),
        default="direct",
        help="ROS2 output mode",
    )
    parser.add_argument(
        "--full-res-video", action="store_true",
        help="保存每相机全分辨率视频（多个 mp4，无拼接、无 badge；编码慢、丢帧多，但保留原图细节供训练数据用）"
    )
    parser.add_argument(
        "--camera-config",
        default=str(config_dir / "camera.json"),
        help="相机采集配置",
    )
    # 曝光/增益按当场光照临时覆盖，不用改配置文件。
    # 曝光直接决定运动拖影：blur_px = v_img(px/s) × T。整条弹道里回球段像面速度最高
    # （出拍后向上横切，~1400px/s，是来球段 200~900px/s 的 2~3 倍），所以最先在那里
    # 成片漏检。2026-08-09 场实测：T=9ms 时回球段拖影 11~13px，与球本身直径 13~18px
    # 同量级；击球后连丢 4 帧（138ms）导致出弧只剩 3~4 点，回球统计整场为空。
    # 具体是被形状门还是重投影门拦下的，看 summary.detection_shape_gate。
    parser.add_argument(
        "--exposure-us", type=float, default=None,
        help="覆盖相机曝光（μs）；不给则用 --camera-config 里的值。"
             "光线足时调小可减运动拖影（回球段拖影 ∝ 曝光）",
    )
    parser.add_argument(
        "--gain-db", type=float, default=None,
        help="覆盖相机增益（dB）；配合 --exposure-us 补回亮度（+6dB ≈ 亮度×2）",
    )
    parser.add_argument(
        "--calib-config",
        default=str(config_dir / "four_camera_calib.json"),
        help="相机标定与外参配置",
    )
    args = parser.parse_args()
    save_logs = not args.no_log

    output_root = Path(args.output_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"tracker_{ts}"
    # 每个 session 一个子目录：tracker_output/tracker_YYYYMMDD_HHMMSS/
    output_dir = output_root / run_id
    if save_logs or not args.no_video:
        output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"

    # ── 加载追踪配置 ──────────────────────────────────────────────────────
    _tracker_config_path = config_dir / "tracker.json"
    with open(_tracker_config_path, encoding="utf-8") as _f:
        tracker_cfg = json.load(_f)

    tile_size = tracker_cfg.get("tile_size", 1280)
    tile_resize = max(int(tracker_cfg.get("tile_resize", TileManager.RESIZE_TO)), 1)
    track_timeout_s = tracker_cfg.get("track_timeout_s", 0.3)
    search_hold_frames = tracker_cfg.get("search_hold_frames", 4)
    min_cameras_for_3d = tracker_cfg.get("min_cameras_for_3d", 2)
    max_reproj_error_px = tracker_cfg.get("max_reproj_error_px", 15.0)
    curve4_cfg = tracker_cfg.get("curve4", {})
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
    ball_detection_disabled_serial_selectors = [
        str(selector).strip()
        for selector in tracker_cfg.get("ball_detection_disabled_serials", [])
        if str(selector).strip()
    ]
    car_loc_cfg = tracker_cfg.get("car_localizer", {})
    car_loc_enabled = car_loc_cfg.get("enabled", True)
    car_loc_sample_every_frames = max(
        int(car_loc_cfg.get("sample_every_frames", 3)),
        1,
    )
    video_output_cfg = tracker_cfg.get("video_output", {})
    video_output_codec = _normalize_video_codec(
        video_output_cfg.get("codec", "avc1")
    )
    video_output_backend = str(
        video_output_cfg.get("backend", "auto")
    ).strip().lower()
    video_output_hw_accel = bool(video_output_cfg.get("hw_accel", True))
    video_path = output_dir / f"{run_id}{_video_container_suffix(video_output_codec)}"
    post_run_cfg = tracker_cfg.get("post_run", {})
    post_run_enabled = bool(post_run_cfg.get("enabled", True)) and save_logs
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
    if not save_logs:
        print("  日志保存: disabled (--no-log)")
    if args.no_video:
        print("  视频保存: disabled (--no-video)")

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

    print("[2/5] 初始化 BallLocalizer (四相机标定)...")
    localizer = BallLocalizer(
        calib_config_path=args.calib_config,
        detector=detector,
    )
    calib_serials = list(localizer.serials)
    print(f"  标定相机: {calib_serials}")

    print("[3/5] 初始化 Curve4Tracker...")
    tracker = Curve4Tracker(**curve4_cfg)
    print(f"  ideal_hit_z={tracker.ideal_hit_z:.3f}m, cor={tracker.cor}, "
          f"cor_xy={tracker.cor_xy}, k_drag={tracker.k_drag:.4f}, "
          f"weight_ratio={tracker.weight_ratio}")
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
    car_localizer = (
        CarLocalizer(calib_config_path=args.calib_config)
        if car_loc_enabled
        else None
    )
    if car_localizer is not None:
        print(f"  相机: {car_localizer.serials}")
        for tag_id, center in sorted(car_localizer.tag_layout_m.items()):
            print(f"  车载 tag id{tag_id} 车体系中心: {_format_xyz_m(*center)}")
        print(
            "  小车定位采样: "
            f"every {car_loc_sample_every_frames} frame(s)"
        )
    else:
        print("  小车定位: disabled")

    # ── ROS2 边车子进程（rosbag 录制 / 发送 sink）──
    _ros2_sink = _create_ros2_sink(args.ros2_mode)
    if save_logs:
        _rosbag_proc = RosbagRecorderProcess(bag_dir=output_dir / f"{run_id}_rosbag")
    else:
        _rosbag_proc = None
        print("  rosbag 录制已禁用：--no-log")
    _sidecars_closed = False
    _sidecar_lock = threading.Lock()

    def _close_sidecars(*, timeout_sec: float) -> None:
        nonlocal _sidecars_closed
        with _sidecar_lock:
            if _sidecars_closed:
                return
            _sidecars_closed = True
        try:
            if _rosbag_proc is not None:
                _rosbag_proc.close(timeout_sec=timeout_sec)
        except Exception:
            pass

    atexit.register(lambda: _close_sidecars(timeout_sec=0.5))

    # log_frames / log_observations / log_predictions / log_car_locs /
    # log_state_transitions / frame_entry_by_idx / car_loc_missed_frames
    # 都由 ArchiveThread 维护，shutdown 后再回收。
    car_loc_sampled_frames = 0
    car_loc_dropped_frames = 0
    capture_fps = 0.0

    # ── 打开同步相机 ────────────────────────────────────────────────────
    print("[5/5] 打开同步相机...")
    capture_overrides: dict[str, float] = {}
    if args.exposure_us is not None:
        capture_overrides["exposure_us"] = float(args.exposure_us)
    if args.gain_db is not None:
        capture_overrides["gain_db"] = float(args.gain_db)
    with SyncCapture.from_config(args.camera_config, **capture_overrides) as cap:
        sync_sns = cap.sync_serials
        capture_fps = cap.fps
        # 相机实际生效的曝光/增益（读回，不是配置里写的值——越界写入会被相机拒绝）。
        # 曝光是回球段能不能连续检出的决定因素：拖影 px ≈ 像面速度 × 曝光，
        # 回球段像面速度 ~1400px/s，球本身才 13~18px 宽。
        camera_settings = cap.camera_settings()
        print(f"  同步相机: {sync_sns}")
        print(f"  配置帧率: {capture_fps:.1f} fps")
        for _sn in sync_sns:
            _s = camera_settings.get(_sn)
            if _s:
                print(f"    {_sn}: 曝光 {_s.get('exposure_us', float('nan')):.0f}μs  "
                      f"增益 {_s.get('gain_db', float('nan')):.2f}dB")

        # 确认当前采集相机与标定文件一致
        missing = [sn for sn in calib_serials if sn not in sync_sns]
        extra = [sn for sn in sync_sns if sn not in calib_serials]
        if missing or extra:
            print(f"*** 错误: 当前采集相机与 {args.calib_config} 不一致 ***")
            print(f"  标定相机: {calib_serials}")
            print(f"  采集相机: {sync_sns}")
            if missing:
                print(f"  缺少标定: {missing}")
            if extra:
                print(f"  未标定相机: {extra}")
            print("  请先使用当前 4 相机重新标定后再运行 tracker。")
            return 1
        order_rank = {sn: idx for idx, sn in enumerate(STITCHED_SERIAL_ORDER)}
        cam_serials = sorted(
            [sn for sn in sync_sns if sn in calib_serials],
            key=lambda sn: order_rank.get(sn, len(order_rank)),
        )
        print(f"  运行相机顺序: {cam_serials}")
        ball_detection_disabled_serials = {
            sn for sn in cam_serials
            if any(
                _serial_matches_selector(sn, selector)
                for selector in ball_detection_disabled_serial_selectors
            )
        }
        ball_detect_serials = [
            sn for sn in cam_serials
            if sn not in ball_detection_disabled_serials
        ]
        preferred_detector_model = _select_detector_model_for_active_cams(
            detector.model_path,
            len(ball_detect_serials),
            target_input_size=tile_resize,
        )
        if preferred_detector_model != detector.model_path:
            print(
                "  YOLO engine 鎵归噺鍖归厤: "
                f"{detector.model_path.name} -> {preferred_detector_model.name}"
            )
            detector = BallDetector(
                model_path=preferred_detector_model,
                duplicate_iou_threshold=detection_duplicate_iou_threshold,
                max_box_aspect_ratio=detection_max_box_aspect_ratio,
            )
            localizer._detector = detector
        actual_tile_resize = (
            _infer_model_input_size_from_model_path(detector.model_path)
            or tile_resize
        )
        if actual_tile_resize != tile_resize:
            print(
                "  YOLO 输入尺寸回退: "
                f"requested={tile_resize} -> actual={actual_tile_resize} "
                f"(match {detector.model_path.name})"
            )
        else:
            print(f"  YOLO 输入尺寸: {actual_tile_resize}")
        if len(ball_detect_serials) < min_cameras_for_3d:
            print("*** 错误: 可用于球检测/3D 的相机数量不足 ***")
            print(f"  已禁用球检测: {sorted(ball_detection_disabled_serials)}")
            print(f"  剩余球检测相机: {ball_detect_serials}")
            return 1
        if ball_detection_disabled_serials:
            print(
                "  跳过球检测/3D 的相机: "
                f"{[sn[-3:] for sn in sorted(ball_detection_disabled_serials)]}"
            )
        print(f"  球检测相机: {ball_detect_serials}")

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
            resize_to=actual_tile_resize,
            track_timeout=track_timeout_s,
            search_hold_frames=search_hold_frames,
        )
        for sn in cam_serials:
            n_tiles = tile_mgr.get_search_tile_count(sn)
            sz = camera_sizes.get(sn, (0, 0))
            print(
                f"  {sn[-3:]}: {sz[0]}x{sz[1]} → {n_tiles} 搜索切片 "
                f"(tile={tile_size}, resize={tile_mgr.resize_to})"
            )

        # ── YOLO 预热 + 自动检测 batch size ──
        print("  YOLO 预热中...")
        # 用切片大小的图像预热（匹配实际推理输入）
        _ts = tile_mgr._tile_size
        warmup_crop = img_sample[:_ts, :_ts] if (
            img_sample.shape[0] >= _ts and img_sample.shape[1] >= _ts
        ) else img_sample
        warmup_img = cv2.resize(
            warmup_crop,
            (tile_mgr.resize_to, tile_mgr.resize_to),
        )

        n_ball_detect_cams = len(ball_detect_serials)
        engine_batch = _resolve_engine_batch(
            detector,
            warmup_img,
            n_ball_detect_cams,
            n_cams,
        )

        for _ in range(4):
            _yolo_detect_n(
                detector,
                [warmup_img] * n_ball_detect_cams,
                engine_batch,
            )

        if engine_batch >= n_ball_detect_cams:
            print(f"  预热完成（batch={engine_batch}，单次推理）")
        else:
            n_calls = math.ceil(n_ball_detect_cams / engine_batch)
            print(f"  预热完成（engine batch={engine_batch}，"
                  f"需 {n_calls} 次推理处理 {n_ball_detect_cams} 张图）")

        # ── 后台写入线程 ───────────────────────────────────────────────
        writer_thread: VideoWriterThread | None = None
        if not args.no_video:
            writer_thread = VideoWriterThread(
                str(video_path), frame_w, frame_h,
                n_cams=n_cams, fps=capture_fps,
                codec=video_output_codec,
                backend=video_output_backend,
                prefer_hw_accel=video_output_hw_accel,
                display=args.display,
                full_res=args.full_res_video,
                cam_serials=cam_serials,
            )
            if writer_thread.full_res:
                print("  视频输出 (全分辨率, 每相机一个文件):")
                for path in writer_thread.video_paths:
                    print(f"    {path}")
                print(
                    f"  (codec={writer_thread.actual_codec}, "
                    f"backend={writer_thread.backend_name}, "
                    f"hw_accel={'on' if writer_thread.hw_accel_requested else 'off'})"
                )
            else:
                print(
                    f"  视频输出: {video_path}"
                    f" (codec={writer_thread.actual_codec}, "
                    f"backend={writer_thread.backend_name}, "
                    f"hw_accel={'on' if writer_thread.hw_accel_requested else 'off'})"
                )

        # ── YOLO 原始 Bayer ROI 解码线程池 ──
        from concurrent.futures import ThreadPoolExecutor
        _tile_decode_pool = ThreadPoolExecutor(max_workers=len(cam_serials))

        # ── 归档线程（线3）：视频以外的 JSON/log 构造 ──
        archive_thread = ArchiveThread(
            cam_serials=cam_serials,
            has_car_localizer=car_localizer is not None,
        )

        # ── 小车定位后台线程（线2）：直接 publish /pc_car_loc 并通知归档 ──
        _car_job_queue: queue.Queue[CarLocJob | None] | None = None
        _car_thread: threading.Thread | None = None
        if car_localizer is not None:
            _car_job_queue = queue.Queue(maxsize=1)

            def _car_worker():
                while True:
                    job = _car_job_queue.get()
                    if job is None:
                        break
                    loc = car_localizer.locate(job.images, t=job.exposure_pc)
                    if loc is not None:
                        _ros2_sink.publish_car_loc({
                            "topic": "car_loc",
                            "x": round(loc.x, 4),
                            "y": round(loc.y, 4),
                            "z": round(loc.z, 4),
                            "yaw": round(loc.yaw, 4),
                            "yaw_valid": loc.yaw_valid,
                            "t": round(job.exposure_pc, 6),
                            "tag_id": loc.tag_id,
                            "tag_ids": loc.tag_ids,
                        })
                    archive_thread.submit(CarLocEvent(
                        frame_idx=job.frame_idx,
                        exposure_pc=job.exposure_pc,
                        elapsed_s=job.elapsed_s,
                        status="hit" if loc is not None else "miss",
                        car_loc=loc,
                    ))

            _car_thread = threading.Thread(
                target=_car_worker, name="CarLocWorker", daemon=True,
            )
            _car_thread.start()

        # ── 全图线：与 YOLO 并行解码，再交给 AprilTag / 视频线程 ──
        _full_job_queue: queue.Queue[FullFrameJob | None] | None = None
        _full_thread: threading.Thread | None = None
        _full_decode_time_sum = 0.0
        _full_decode_count = 0
        _full_queue_max_size = 0
        _full_drop_count = 0
        if writer_thread is not None or car_localizer is not None:
            _full_job_queue = queue.Queue(maxsize=2)

            def _full_frame_worker():
                nonlocal _full_decode_time_sum, _full_decode_count
                nonlocal car_loc_dropped_frames
                with ThreadPoolExecutor(max_workers=len(cam_serials)) as pool:
                    while True:
                        job = _full_job_queue.get()
                        if job is None:
                            break
                        t0 = time.perf_counter()
                        sns = [sn for sn in cam_serials if sn in job.frames]
                        futures = {
                            sn: pool.submit(frame_to_numpy, job.frames[sn])
                            for sn in sns
                        }
                        images = {
                            sn: future.result()
                            for sn, future in futures.items()
                        }
                        _full_decode_time_sum += time.perf_counter() - t0
                        _full_decode_count += 1

                        if job.car_loc_sampled:
                            assert _car_job_queue is not None
                            stale = _car_submit_latest(
                                _car_job_queue,
                                CarLocJob(
                                    frame_idx=job.frame_idx,
                                    exposure_pc=job.exposure_pc,
                                    elapsed_s=job.elapsed_s,
                                    images=images,
                                ),
                            )
                            if stale is not None:
                                car_loc_dropped_frames += 1
                                archive_thread.submit(CarLocEvent(
                                    frame_idx=stale.frame_idx,
                                    exposure_pc=stale.exposure_pc,
                                    elapsed_s=stale.elapsed_s,
                                    status="dropped",
                                ))

                        if writer_thread is not None:
                            writer_thread.submit(WriteJob(
                                images=images,
                                serials=cam_serials,
                                exposure_perf=job.exposure_pc,
                                elapsed_s=job.elapsed_s,
                                frame_idx=job.frame_idx,
                            ))

            _full_thread = threading.Thread(
                target=_full_frame_worker,
                name="FullFrameWorker",
                daemon=True,
            )
            _full_thread.start()

        # ── 信号处理（确保被终止时也能保存数据）──
        _shutdown = threading.Event()
        # stop 文件固定在输出根目录（外部工具约定路径，不随 session 子目录走）
        _stop_file = output_root / '.stop_tracker'
        _stop_file.unlink(missing_ok=True)  # 清除上次残留

        def _signal_handler(sig, frame):
            _shutdown.set()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, _signal_handler)

        # ── 主循环 ────────────────────────────────────────────────────
        frame_idx = 0
        first_frame_exposure_pc: float | None = None
        t_start = time.perf_counter()
        timeout_count = 0
        _t_capture_sum = 0.0
        _t_decode_sum = 0.0
        _t_yolo_sum = 0.0
        _t_other_sum = 0.0

        # 预热期的推理不计入形状门统计
        BallDetector.reset_shape_gate_stats()

        print(f"\n{'*' * 60}")
        print(f"  预热完成，开始追踪！（{args.duration}s）按 Ctrl+C 提前结束")
        print(f"{'*' * 60}\n")
        print("\a", end="", flush=True)

        try:
            while not _shutdown.is_set() and time.perf_counter() - t_start < args.duration:
                # 检测停止文件
                if _stop_file.exists():
                    print("检测到停止文件，优雅退出...")
                    _stop_file.unlink(missing_ok=True)
                    break
                _t_loop0 = time.perf_counter()
                frames = cap.get_frames(timeout_s=1.0)
                _t_capture_done = time.perf_counter()
                if frames is None:
                    timeout_count += 1
                    continue
                _t_capture_sum += _t_capture_done - _t_loop0

                exp_starts = [
                    frames[sn].exposure_start_pc
                    for sn in cam_serials if sn in frames
                ]
                exposure_pc = sum(exp_starts) / len(exp_starts)
                if first_frame_exposure_pc is None:
                    first_frame_exposure_pc = exposure_pc
                frame_elapsed_s = exposure_pc - first_frame_exposure_pc

                # ── 标记本帧是否需要后台车辆定位 ──
                car_loc_sampled = (
                    car_localizer is not None
                    and (frame_idx % car_loc_sample_every_frames) == 0
                )
                if car_loc_sampled:
                    car_loc_sampled_frames += 1

                # ── YOLO 线：原始 Bayer 先裁 ROI，再解码/缩放 ──
                img_sns = [sn for sn in cam_serials if sn in frames]
                ball_detect_sns = [
                    sn for sn in img_sns
                    if sn not in ball_detection_disabled_serials
                ]
                frame_tiles: dict[str, TileRect] = {}
                _t0 = time.perf_counter()
                tile_futures = {}
                for sn in ball_detect_sns:
                    frame = frames[sn]
                    tile_rect = tile_mgr.select_tile(
                        sn,
                        frame.width,
                        frame.height,
                        exposure_pc,
                    )
                    frame_tiles[sn] = tile_rect
                    tile_futures[sn] = _tile_decode_pool.submit(
                        frame_bayer_roi_to_numpy,
                        frame,
                        x=tile_rect.x,
                        y=tile_rect.y,
                        width=tile_rect.w,
                        height=tile_rect.h,
                        resize_to=tile_mgr.resize_to,
                    )
                tile_imgs = [tile_futures[sn].result() for sn in ball_detect_sns]
                _t1 = time.perf_counter()
                _t_decode_sum += _t1 - _t0

                # 全图线与 GPU 推理并行，不进入 YOLO 关键路径。
                if _full_job_queue is not None and (
                    writer_thread is not None or car_loc_sampled
                ):
                    full_job = FullFrameJob(
                        frames=frames,
                        frame_idx=frame_idx,
                        exposure_pc=exposure_pc,
                        elapsed_s=frame_elapsed_s,
                        car_loc_sampled=car_loc_sampled,
                    )
                    try:
                        _full_job_queue.put_nowait(full_job)
                        _full_queue_max_size = max(
                            _full_queue_max_size,
                            _full_job_queue.qsize(),
                        )
                    except queue.Full:
                        _full_drop_count += 1
                        if car_loc_sampled:
                            car_loc_dropped_frames += 1
                            archive_thread.submit(CarLocEvent(
                                frame_idx=frame_idx,
                                exposure_pc=exposure_pc,
                                elapsed_s=frame_elapsed_s,
                                status="dropped",
                            ))

                det_results = _yolo_detect_n(
                    detector, tile_imgs, engine_batch)
                _t2 = time.perf_counter()
                _t_yolo_sum += _t2 - _t1

                latency_ms = (time.perf_counter() - exposure_pc) * 1000.0

                # 映射检测坐标回全图 + 整理
                all_detections: dict[str, list[BallDetection]] = {
                    sn: [] for sn in img_sns
                }
                for i, sn in enumerate(ball_detect_sns):
                    mapped_detections = [
                        TileManager.map_detection_to_full(
                            d,
                            frame_tiles[sn],
                            resize_to=tile_mgr.resize_to,
                        )
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
                compute_done_t: Optional[float] = None
                if len(good_dets) >= min_cameras_for_3d:
                    candidate = localizer.triangulate(good_dets)
                    if candidate.reprojection_error <= max_reproj_error_px:
                        ball3d = candidate
                        for sn, det in good_dets.items():
                            tile_mgr.on_3d_located(
                                sn, det.x, det.y, exposure_pc)
                        tracker_result = tracker.update(BallObservation(
                            x=ball3d.x, y=ball3d.y,
                            z=ball3d.z, t=exposure_pc,
                        ))
                        # 在 tracker.update() 刚返回的时刻抓 perf_counter，作为本预测的
                        # 算完时间。compute_latency = compute_done_t - ct 即算几 ms 延迟。
                        compute_done_t = time.perf_counter()
                    else:
                        for sn in good_dets:
                            tile_mgr.on_2d_detected(sn, frame_tiles[sn])
                        tracker_result = TrackerResult(
                            prediction=None,
                            state=tracker.tracker_state,
                        )
                else:
                    for sn in good_dets:
                        tile_mgr.on_2d_detected(sn, frame_tiles[sn])
                    tracker_result = TrackerResult(
                        prediction=None,
                        state=tracker.tracker_state,
                    )

                # ── 发布 /predict_hit_pos（线1 延迟敏感出口）──
                if tracker_result.prediction is not None:
                    p = tracker_result.prediction
                    _ros2_sink.publish_predict_hit({
                        "x": round(p.x, 4),
                        "y": round(p.y, 4),
                        "z": round(p.z, 4),
                        "vx": round(p.vx, 4),
                        "vy": round(p.vy, 4),
                        "vz": round(p.vz, 4),
                        "stage": p.stage,
                        "ct": round(p.ct, 6),
                        "ht": round(p.ht, 6),
                        "duration": round(p.ht - p.ct, 4),
                    })

                # ── 投递归档线程（线3，非阻塞，只传业务数据）──
                archive_thread.submit(ArchiveJob(
                    frame_idx=frame_idx,
                    exposure_pc=exposure_pc,
                    elapsed_s=frame_elapsed_s,
                    latency_ms=latency_ms,
                    all_detections=all_detections,
                    frame_tiles=frame_tiles,
                    ball3d=ball3d,
                    tracker_result=tracker_result,
                    car_loc_sampled=car_loc_sampled,
                    compute_done_t=compute_done_t,
                ))

                _t_other_sum += time.perf_counter() - _t2

                frame_idx += 1

                if frame_idx % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    n = max(frame_idx, 1)
                    writer_note = ""
                    if writer_thread is not None:
                        _writer_stats = writer_thread.stats()
                        writer_note = (
                            f"  video={_writer_stats['avg_process_ms']:.1f}ms"
                            f" qmax={_writer_stats['queue_max_size']}"
                        )
                    print(
                        f"  [{elapsed:.1f}s] {frame_idx} frames "
                        f"({fps:.1f} fps)  "
                        f"state={tracker_result.state.value}  "
                        f"avg: cap={_t_capture_sum/n*1000:.1f}ms "
                        f"tile={_t_decode_sum/n*1000:.1f}ms "
                        f"yolo={_t_yolo_sum/n*1000:.1f}ms "
                        f"full={_full_decode_time_sum/max(_full_decode_count, 1)*1000:.1f}ms "
                        f"other={_t_other_sum/n*1000:.1f}ms"
                        f"{writer_note}"
                    )

        except KeyboardInterrupt:
            print("\n手动中断")

        # ── 清理 ──
        processing_elapsed = time.perf_counter() - t_start
        _tile_decode_pool.shutdown(wait=True)
        # 先排空全图解码线，再关闭 car_loc / 视频消费者。
        if _full_job_queue is not None:
            _full_job_queue.put(None)
        if _full_thread is not None:
            _full_thread.join(timeout=10.0)
            if _full_thread.is_alive():
                raise RuntimeError(
                    "full-frame worker did not finish before shutdown"
                )
        # 关闭 car_loc（确保把最后的 CarLocEvent 送到归档线程）
        if _car_job_queue is not None:
            _car_job_queue.put(None)
        if _car_thread is not None:
            _car_thread.join(timeout=10.0)
            if _car_thread.is_alive():
                raise RuntimeError(
                    "car_loc worker did not finish before shutdown"
                )
        # 再关视频写入线程
        drop_count = 0
        written_frame_indices: list[int] = []
        writer_stats: dict[str, float | int | bool] = {"enabled": False}
        if writer_thread is not None:
            print("  等待视频写入完成...")
            drop_count = writer_thread.stop() + _full_drop_count
            written_frame_indices = writer_thread.written_frame_indices()
            writer_stats = writer_thread.stats()
        # 最后关归档线程
        archive_thread.stop()
        archive_result = archive_thread.result()
        log_frames = archive_result["log_frames"]
        log_observations = archive_result["log_observations"]
        log_predictions = archive_result["log_predictions"]
        log_car_locs = archive_result["log_car_locs"]
        log_state_transitions = archive_result["log_state_transitions"]
        frame_entry_by_idx = archive_result["frame_entry_by_idx"]
        car_loc_missed_frames = archive_result["car_loc_missed_frames"]
        # 将视频帧序号回填到对应 frame_entry
        for video_frame_idx, frame_id in enumerate(written_frame_indices):
            frame_data = frame_entry_by_idx.get(frame_id)
            if frame_data is not None:
                frame_data["video_frame_idx"] = video_frame_idx
                frame_data["video_mapping_exact"] = True

    # ── 关闭 ROS2 边车子进程（rosbag / 发送 sink）──
    try:
        _ros2_sink.close()
    finally:
        _close_sidecars(timeout_sec=5.0)

    total_elapsed = time.perf_counter() - t_start
    elapsed = processing_elapsed

    latencies = [f["latency_ms"] for f in log_frames]
    lat_avg = sum(latencies) / len(latencies) if latencies else 0
    lat_min = min(latencies) if latencies else 0
    lat_max = max(latencies) if latencies else 0
    n_timing = max(frame_idx, 1)

    timing_summary = {
        "capture_avg": round(_t_capture_sum / n_timing * 1000.0, 1),
        "decode_avg": round(_t_decode_sum / n_timing * 1000.0, 1),
        "full_decode_avg": round(
            _full_decode_time_sum / max(_full_decode_count, 1) * 1000.0,
            1,
        ),
        "yolo_avg": round(_t_yolo_sum / n_timing * 1000.0, 1),
        "other_avg": round(_t_other_sum / n_timing * 1000.0, 1),
    }

    result = {
        "config": {
            "first_frame_exposure_pc": first_frame_exposure_pc,
            "serials": cam_serials,
            "camera_config_path": str(Path(args.camera_config).resolve()),
            # 逐相机读回的实际曝光/增益。拖影 px ≈ v_img(px/s) × 曝光(s)，
            # 回球段 v_img ~1400px/s 是全弹道最高，最先在那里成片漏检
            "camera_settings": camera_settings,
            "camera_capture_overrides": capture_overrides,
            "calib_config_path": str(Path(args.calib_config).resolve()),
            "duration_s": processing_elapsed,
            "end_to_end_duration_s": total_elapsed,
            "fps": capture_fps,
            "distance_unit": "m",
            "ideal_hit_z": tracker.ideal_hit_z,
            "cor": tracker.cor,
            "cor_xy": tracker.cor_xy,
            "k_drag": tracker.k_drag,
            "weight_ratio": tracker.weight_ratio,
            "model_path": str(detector.model_path),
            "ball_detection_serials": ball_detect_serials,
            "ball_detection_disabled_serials": sorted(
                ball_detection_disabled_serials
            ),
            "tile_size": tile_size,
            "tile_resize": tile_mgr.resize_to,
            "engine_batch": engine_batch,
            "yolo_inference_calls_per_frame": (
                math.ceil(len(ball_detect_serials) / engine_batch)
                if engine_batch > 0 else 0
            ),
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
                "sample_every_frames": car_loc_sample_every_frames,
                "position_reference": "car_base",
                "apriltag_layout_car_m": (
                    {
                        str(tag_id): [round(float(v), 4) for v in center]
                        for tag_id, center in sorted(
                            car_localizer.tag_layout_m.items()
                        )
                    }
                    if car_localizer is not None
                    else None
                ),
            },
            "video_output": {
                "artifact_path": (
                    str(video_path.resolve())
                    if not args.no_video and not args.full_res_video
                    else None
                ),
                "artifact_paths": (
                    [str(Path(p).resolve()) for p in writer_thread.video_paths]
                    if writer_thread is not None and writer_thread.full_res
                    else None
                ),
                "requested_codec": video_output_codec,
                "requested_backend": video_output_backend,
                "requested_hw_accel": video_output_hw_accel,
                "codec": (
                    writer_thread.actual_codec
                    if writer_thread is not None
                    else video_output_codec
                ),
                "backend": (
                    writer_thread.backend_name
                    if writer_thread is not None
                    else video_output_backend
                ),
                "layout": (
                    "per_camera_full_res"
                    if writer_thread is not None and writer_thread.full_res
                    else "grid"
                ),
                "grid_cols": _grid_dimensions(len(cam_serials), cols=2)[0],
                "grid_rows": _grid_dimensions(len(cam_serials), cols=2)[1],
                "serial_order": cam_serials,
            },
            "post_run": {
                "enabled": post_run_enabled,
                "generate_html": post_run_generate_html,
                "generate_annotated_video": post_run_generate_annotated_video,
                "annotated_video_no_racket": post_run_annotated_video_no_racket,
            },
            "rosbag": {
                "enabled": _rosbag_proc is not None and _rosbag_proc.was_started(),
                "bag_dir": str(_rosbag_proc.bag_dir)
                if _rosbag_proc is not None and _rosbag_proc.was_started()
                else None,
                "bag_exists": _rosbag_proc.bag_dir.exists()
                if _rosbag_proc is not None and _rosbag_proc.was_started()
                else False,
            },
            "video_frame_mapping_exact": bool(written_frame_indices),
        },
        "summary": {
            "total_frames": frame_idx,
            "actual_fps": (
                frame_idx / processing_elapsed if processing_elapsed > 0 else 0
            ),
            "end_to_end_fps": (
                frame_idx / total_elapsed if total_elapsed > 0 else 0
            ),
            "processing_duration_s": round(processing_elapsed, 3),
            "end_to_end_duration_s": round(total_elapsed, 3),
            "timeouts": timeout_count,
            "observations_3d": len(log_observations),
            "predictions": len(log_predictions),
            "car_locs": len(log_car_locs),
            "car_loc_sampled_frames": car_loc_sampled_frames,
            "car_loc_misses": car_loc_missed_frames,
            # 暗光 tag 救援：raw 检测全空 → CLAHE 拉对比度重试的次数 / 救回来的次数。
            # retries 长期不为 0 说明现场照度已经在 AprilTag 的边缘上，该补光了。
            "car_loc_low_light_retries": (
                getattr(car_localizer, "low_light_retries", 0)
                if car_localizer is not None else 0
            ),
            "car_loc_low_light_recovered": (
                getattr(car_localizer, "low_light_recovered", 0)
                if car_localizer is not None else 0
            ),
            "car_loc_dropped_frames": car_loc_dropped_frames,
            "state_transitions": len(log_state_transitions),
            # 形状门统计：被 max_box_aspect_ratio 拦掉的网球框有多少、平均/最大长宽比多少。
            # rejected 占比高 = 曝光太长、球被拖影拉长（回球段最严重），该降曝光而不是放宽门。
            "detection_shape_gate": _shape_gate_summary(),
            "reset_times": tracker.reset_times,
            "video_frames_dropped": drop_count,
            "video_frames_written": len(written_frame_indices),
            "video_frames_mapped": len(written_frame_indices),
            "video_frame_mapping_exact": bool(written_frame_indices),
            "latency_ms_avg": round(lat_avg, 1),
            "latency_ms_min": round(lat_min, 1),
            "latency_ms_max": round(lat_max, 1),
            "timing_ms": timing_summary,
            "video_writer": writer_stats,
        },
        "observations": log_observations,
        "predictions": log_predictions,
        "car_locs": log_car_locs,
        "frames": log_frames,
        "video_frame_indices": written_frame_indices,
        "state_transitions": log_state_transitions,
    }

    generated_artifacts: dict[str, Path] = {}
    if save_logs:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if save_logs and post_run_enabled:
        # full_res 模式下产出多个独立 mp4，标注视频脚本只识别拼接 grid，所以跳过它
        post_run_video_path = (
            video_path
            if not args.no_video and not args.full_res_video
            else None
        )
        generated_artifacts = _generate_post_run_artifacts(
            json_path=json_path,
            video_path=post_run_video_path,
            generate_html=post_run_generate_html,
            generate_annotated_video=(
                post_run_generate_annotated_video
                and post_run_video_path is not None
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
    print(
        "  小车定位:   "
        f"hits={len(log_car_locs)}  "
        f"sampled={car_loc_sampled_frames}  "
        f"misses={car_loc_missed_frames}  "
        f"dropped={car_loc_dropped_frames}"
    )
    print(f"  状态转换:   {len(log_state_transitions)}")
    print(f"  超时次数:   {timeout_count}")
    print(f"  延迟(ms):   avg={lat_avg:.0f}  min={lat_min:.0f}  max={lat_max:.0f}")
    print(
        "  时序(ms):   "
        f"cap={timing_summary['capture_avg']:.1f}  "
        f"tile={timing_summary['decode_avg']:.1f}  "
        f"yolo={timing_summary['yolo_avg']:.1f}  "
        f"full={timing_summary['full_decode_avg']:.1f}(async)  "
        f"other={timing_summary['other_avg']:.1f}"
    )
    print(
        "  全图异步线: "
        f"decoded={_full_decode_count}  "
        f"dropped={_full_drop_count}  "
        f"qmax={_full_queue_max_size}"
    )
    print(
        "  YOLO批量:   "
        f"engine_batch={engine_batch}  "
        f"calls/frame={math.ceil(len(ball_detect_serials) / engine_batch) if engine_batch > 0 else 0}"
    )
    if writer_stats.get("enabled"):
        print(
            "  写视频(ms): "
            f"avg={writer_stats['avg_process_ms']:.1f}  "
            f"max={writer_stats['max_process_ms']:.1f}  "
            f"qmax={writer_stats['queue_max_size']}"
        )
    if not args.no_video:
        if writer_thread is not None and writer_thread.full_res:
            print("  视频(全分辨率):")
            for path in writer_thread.video_paths:
                print(f"    {path}")
        else:
            print(f"  视频:       {video_path}")
        if drop_count > 0:
            print(f"  视频丢帧:   {drop_count}")
    if save_logs:
        print(f"  JSON:       {json_path}")
        if _rosbag_proc is not None and _rosbag_proc.was_started():
            if _rosbag_proc.bag_dir.exists():
                print(f"  ROSBAG:     {_rosbag_proc.bag_dir}")
            else:
                print(
                    "  ROSBAG:     missing"
                    f" (target={_rosbag_proc.bag_dir})"
                )
        if "html" in generated_artifacts:
            print(f"  HTML:       {generated_artifacts['html']}")
        if "annotated_video" in generated_artifacts:
            print(f"  标注视频:   {generated_artifacts['annotated_video']}")
    else:
        print("  日志:       disabled (--no-log)")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

