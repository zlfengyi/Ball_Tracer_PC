# -*- coding: utf-8 -*-
"""
UDP → ROS2 桥接节点：接收 run_tracker 通过 UDP 发送的小车位置，
发布到 ROS2 topic /pc_car_loc。

run_tracker.py (Python 3.13) 通过 UDP sendto 127.0.0.1:5858 发送 JSON，
本节点在后台线程接收，ROS2 定时器 30Hz 发布最新数据。

用法：
    ros2/run_ros2.bat ros2/car_loc_bridge.py
    或由 run_tracker.py 自动作为子进程启动
"""

import json
import socket
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

UDP_PORT = 5858
TOPIC = "/pc_car_loc"
PUBLISH_HZ = 30


class CarLocBridge(Node):
    def __init__(self):
        super().__init__("car_loc_bridge")

        self._pub = self.create_publisher(String, TOPIC, 10)
        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._on_timer)

        self._latest: str | None = None
        self._lock = threading.Lock()
        self._count = 0

        # 后台 UDP 接收线程
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", UDP_PORT))
        self._sock.settimeout(1.0)
        self._udp_thread = threading.Thread(
            target=self._udp_recv_loop, daemon=True)
        self._udp_thread.start()

        self.get_logger().info(
            f"CarLocBridge 已启动：UDP:{UDP_PORT} → {TOPIC}")

    def _udp_recv_loop(self):
        while rclpy.ok():
            try:
                data, _ = self._sock.recvfrom(4096)
                with self._lock:
                    self._latest = data.decode("utf-8")
            except socket.timeout:
                continue
            except OSError:
                break

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
        if self._count % 300 == 1:
            try:
                d = json.loads(payload)
                self.get_logger().info(
                    f"#{self._count} car=({d.get('x', '?')}, "
                    f"{d.get('y', '?')}, {d.get('z', '?')}) m")
            except json.JSONDecodeError:
                pass


def main():
    rclpy.init()
    node = CarLocBridge()
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
