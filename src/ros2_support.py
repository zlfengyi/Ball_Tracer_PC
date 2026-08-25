from __future__ import annotations

import os
import sys
from pathlib import Path

TRACKER_PC_IP = os.environ.get("BALL_TRACER_PC_IP", "192.168.50.230")
# 一台车一个 RK IP（车上静态配置，换路由器/换楼层都不变，2026-08-18 定）。
# 旧的 ARM_RK_IP/CHASSIS_RK_IP 两个字段其实是两台车各自的 IP，概念已废。
CAR_RK_IPS = {
    "v03": "192.168.50.143",
    "v04": "192.168.50.68",
}
# run_tracker.ps1 按 -Car 设置；没设时列出两台车（本值只用于启动打印，
# 实际 DDS 发现走 CYCLONEDDS_URI 指到的 xml 里的 Peers）
RK_CAR_IP = os.environ.get("BALL_TRACER_RK_CAR_IP", "")
DEFAULT_ROS_DOMAIN_ID = "2"

ROS2_ROOT = Path(
    os.environ.get("BALL_TRACER_ROS2_ROOT", r"C:\dev\ros2_jazzy")
).resolve()
ROS2_SITE_PACKAGES = ROS2_ROOT / "Lib" / "site-packages"
ROS2_SCRIPTS_DIR = ROS2_ROOT / "Scripts"
ROS2_BIN_DIR = ROS2_ROOT / "bin"
CYCLONEDDS_XML_PATH = Path(__file__).resolve().parent.parent / "ros2" / "cyclonedds.xml"
ROS2_DLL_DIRS = (
    ROS2_ROOT,
    ROS2_ROOT / "Library" / "mingw-w64" / "bin",
    ROS2_ROOT / "Library" / "usr" / "bin",
    ROS2_ROOT / "Library" / "bin",
    ROS2_SCRIPTS_DIR,
    ROS2_BIN_DIR,
)
_DLL_HANDLES = []

ROS2_TRACKER_PEERS = (RK_CAR_IP,) if RK_CAR_IP else tuple(CAR_RK_IPS.values())
ROS2_BEST_EFFORT_DEPTH = 1
ROS2_RELIABLE_TOPICS = frozenset(
    {
        "/predict_hit_pos",
    }
)


def _prepend_unique_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _prepend_unique_env_path(name: str, value: Path | str) -> None:
    value_str = str(value)
    if not value_str:
        return

    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if value_str in entries:
        return

    if current:
        os.environ[name] = value_str + os.pathsep + current
    else:
        os.environ[name] = value_str


def _add_dll_directory(path: Path) -> None:
    if not hasattr(os, "add_dll_directory") or not path.exists():
        return
    path_str = str(path)
    if any(item[0] == path_str for item in _DLL_HANDLES):
        return
    _DLL_HANDLES.append((path_str, os.add_dll_directory(path_str)))


def ensure_ros2_environment() -> None:
    """Prepare the current process to load ROS 2 Jazzy with Cyclone DDS."""
    if ROS2_SITE_PACKAGES.exists():
        _prepend_unique_sys_path(ROS2_SITE_PACKAGES)
        _prepend_unique_env_path("PYTHONPATH", ROS2_SITE_PACKAGES)

    # rcutils 在 C 层用 LoadLibrary 按文件名加载 RMW DLL（rmw_cyclonedds_cpp.dll），
    # 该方式只搜索 PATH、不认 os.add_dll_directory，所以 DLL 目录必须同时进 PATH。
    for dll_dir in reversed(ROS2_DLL_DIRS):
        _add_dll_directory(dll_dir)
        if dll_dir.exists():
            _prepend_unique_env_path("PATH", dll_dir)

    _prepend_unique_env_path("AMENT_PREFIX_PATH", ROS2_ROOT)
    _prepend_unique_env_path("COLCON_PREFIX_PATH", ROS2_ROOT)
    _prepend_unique_env_path("CMAKE_PREFIX_PATH", ROS2_ROOT)

    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ.setdefault("ROS_PYTHON_VERSION", "3")
    os.environ.setdefault("ROS_VERSION", "2")
    os.environ.setdefault(
        "ROS_DOMAIN_ID",
        os.environ.get("BALL_TRACER_ROS_DOMAIN_ID", DEFAULT_ROS_DOMAIN_ID),
    )
    os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    os.environ.setdefault("CYCLONEDDS_URI", cyclonedds_file_uri(CYCLONEDDS_XML_PATH))

    for env_name in (
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "FASTDDS_DEFAULT_PROFILES_FILE",
    ):
        os.environ.pop(env_name, None)


def make_best_effort_qos(depth: int = ROS2_BEST_EFFORT_DEPTH):
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def make_reliable_qos(depth: int = ROS2_BEST_EFFORT_DEPTH):
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def make_topic_qos(topic: str, depth: int = ROS2_BEST_EFFORT_DEPTH):
    if topic in ROS2_RELIABLE_TOPICS:
        return make_reliable_qos(depth=depth)
    return make_best_effort_qos(depth=depth)


def cyclonedds_file_uri(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return f"file://{resolved.as_posix()}"
    return resolved.as_uri()
