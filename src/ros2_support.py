from __future__ import annotations

import os
import sys
from pathlib import Path

TRACKER_PC_IP = "192.168.50.18"
ARM_RK_IP = "192.168.50.17"
CHASSIS_RK_IP = "192.168.50.68"
DEFAULT_ROS_DOMAIN_ID = "2"

ROS2_ROOT = Path(
    os.environ.get("BALL_TRACER_ROS2_ROOT", r"C:\dev\ros2_jazzy")
).resolve()
ROS2_SITE_PACKAGES = ROS2_ROOT / "Lib" / "site-packages"
ROS2_SCRIPTS_DIR = ROS2_ROOT / "Scripts"
ROS2_BIN_DIR = ROS2_ROOT / "bin"
CYCLONEDDS_XML_PATH = Path(__file__).resolve().parent.parent / "ros2" / "cyclonedds.xml"

ROS2_TRACKER_PEERS = (
    ARM_RK_IP,
    CHASSIS_RK_IP,
)
ROS2_BEST_EFFORT_DEPTH = 1
ROS2_RELIABLE_TOPICS = frozenset(
    {
        "/arm_controller/hit_command",
        "/predict_hit_pos",
        "/arm_logger/control",
        "/arm_controller/status",
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


def ensure_ros2_environment() -> None:
    """Prepare the current process to load ROS 2 Jazzy with Cyclone DDS."""
    if ROS2_SITE_PACKAGES.exists():
        _prepend_unique_sys_path(ROS2_SITE_PACKAGES)
        _prepend_unique_env_path("PYTHONPATH", ROS2_SITE_PACKAGES)

    if ROS2_SCRIPTS_DIR.exists():
        _prepend_unique_env_path("PATH", ROS2_SCRIPTS_DIR)
    if ROS2_BIN_DIR.exists():
        _prepend_unique_env_path("PATH", ROS2_BIN_DIR)

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
    os.environ["CYCLONEDDS_URI"] = cyclonedds_file_uri(CYCLONEDDS_XML_PATH)

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
