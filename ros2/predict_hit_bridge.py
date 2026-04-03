# -*- coding: utf-8 -*-
"""
UDP → ROS2 桥接：接收 predict_hit UDP 数据，发布到 /predict_hit_pos topic。
独立于 car_loc_bridge，使用 UDP 端口 5859。
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

UDP_PORT = 5859


class PredictHitBridge(Node):
    def __init__(self):
        super().__init__("predict_hit_bridge")
        self._pub = self.create_publisher(
            String, "/predict_hit_pos", make_topic_qos("/predict_hit_pos")
        )
        self._timer = self.create_timer(1.0 / 30, self._on_timer)

        self._latest = None
        self._lock = threading.Lock()
        self._count = 0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", UDP_PORT))
        self._sock.settimeout(1.0)
        self._udp_thread = threading.Thread(
            target=self._udp_recv_loop, daemon=True)
        self._udp_thread.start()

        self.get_logger().info(
            f"PredictHitBridge 已启动：UDP:{UDP_PORT} → /predict_hit_pos")

    def _udp_recv_loop(self):
        while rclpy.ok():
            try:
                data, _ = self._sock.recvfrom(4096)
                payload = data.decode("utf-8")
                with self._lock:
                    self._latest = payload
            except socket.timeout:
                continue
            except OSError:
                continue

    def _on_timer(self):
        with self._lock:
            payload = self._latest
            self._latest = None

        if payload is None:
            return

        msg = String()
        msg.data = payload
        self._pub.publish(msg)

        self._count += 1
        if self._count % 100 == 1:
            self.get_logger().info(
                f"#{self._count} /predict_hit_pos: {payload[:80]}")


def main():
    rclpy.init()
    node = PredictHitBridge()
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
