"""
Windows 端时间同步应答器 -- 配合 RK 端 WinRKTimeSync 使用。

协议：
  1. RK 发布 ping: {"seq": N, "t1": <rk_epoch>}   -> /time_sync/ping
  2. Win 收到后立即回复 pong: {"seq": N, "t1": <rk_epoch>, "t2": <win_epoch>} -> /time_sync/pong
  3. RK 根据 t1/t2/t3 计算 offset 和 rtt

用法：
    ros2/run_ros2.bat src/win_time_sync.py
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PING_TOPIC = "/time_sync/ping"
PONG_TOPIC = "/time_sync/pong"


class TimeSyncResponder(Node):
    def __init__(self):
        super().__init__("win_time_sync_responder")
        self.sub = self.create_subscription(String, PING_TOPIC, self._on_ping, 10)
        self.pub = self.create_publisher(String, PONG_TOPIC, 10)
        self.count = 0
        self.get_logger().info(
            f"时间同步应答器已启动，监听 {PING_TOPIC}，回复 {PONG_TOPIC}"
        )

    def _on_ping(self, msg: String):
        t2 = time.time()
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        data["t2"] = t2
        pong = String()
        pong.data = json.dumps(data)
        self.pub.publish(pong)

        self.count += 1
        if self.count % 10 == 1:
            self.get_logger().info(
                f"pong #{data.get('seq', '?')}: t2={t2:.3f}, 已回复 {self.count} 次"
            )


def main():
    rclpy.init()
    node = TimeSyncResponder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
