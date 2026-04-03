# -*- coding: utf-8 -*-
"""
UDP → ROS2 桥接节点：接收 run_tracker 通过 UDP 发送的数据，
按 topic 字段路由到对应的 ROS2 topic。

支持的 topic：
  - /pc_car_loc       小车位置 + yaw
  - /predict_hit_pos  预测击球位置

run_tracker.py (Python 3.13) 通过 UDP sendto 127.0.0.1:5858 发送 JSON，
JSON 中包含 "topic" 字段指定目标 ROS2 topic。

用法：
    ros2/run_ros2.bat ros2/car_loc_bridge.py
    或由 run_tracker.py 自动作为子进程启动
"""

import json
import socket
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ros2_support import ensure_ros2_environment, make_topic_qos

ensure_ros2_environment()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

UDP_PORT = 5858
PUBLISH_HZ = 30

TOPICS = {
    "car_loc": "/pc_car_loc",
    "predict_hit": "/predict_hit_pos",
}


class TrackerBridge(Node):
    def __init__(self):
        super().__init__("tracker_bridge")

        self._pubs = {}
        for key, topic in TOPICS.items():
            self._pubs[key] = self.create_publisher(
                String, topic, make_topic_qos(topic)
            )

        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._on_timer)

        # 每个 topic 缓存最新消息
        self._latest: dict[str, str] = {}
        self._lock = threading.Lock()
        self._count = 0

        # 后台 UDP 接收线程
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", UDP_PORT))
        self._sock.settimeout(1.0)
        self._udp_thread = threading.Thread(
            target=self._udp_recv_loop, daemon=True)
        self._udp_thread.start()

        topics_str = ", ".join(TOPICS.values())
        self.get_logger().info(
            f"TrackerBridge 已启动：UDP:{UDP_PORT} → {topics_str}")

    def _udp_recv_loop(self):
        while rclpy.ok():
            try:
                data, _ = self._sock.recvfrom(4096)
                text = data.decode("utf-8")
                d = json.loads(text)
                key = d.pop("topic", None)
                if key and key in self._pubs:
                    payload = json.dumps(d)
                    with self._lock:
                        self._latest[key] = payload
            except socket.timeout:
                continue
            except (OSError, json.JSONDecodeError):
                continue

    def _on_timer(self):
        with self._lock:
            pending = dict(self._latest)
            self._latest.clear()

        for key, payload in pending.items():
            pub = self._pubs.get(key)
            if pub is None:
                continue
            msg = String()
            msg.data = payload
            pub.publish(msg)

            self._count += 1
            if self._count % 300 == 1:
                self.get_logger().info(
                    f"#{self._count} {TOPICS[key]}: {payload[:80]}")


def main():
    rclpy.init()
    node = TrackerBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
