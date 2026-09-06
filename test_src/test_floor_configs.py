from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "src" / "config"
ROS2_DIR = ROOT_DIR / "ros2"


class SiteConfigTest(unittest.TestCase):
    """18F 是唯一在用的场地（2026-09-06），16F 那套配置已删除。"""

    def test_camera_and_calib_agree(self) -> None:
        camera = json.loads(
            (CONFIG_DIR / "camera_18.json").read_text(encoding="utf-8"))
        calib = json.loads(
            (CONFIG_DIR / "four_camera_calib_18.json").read_text(encoding="utf-8"))

        camera_serials = {camera["master_serial"], *camera["slave_serials"]}
        self.assertEqual(camera["trigger_mode"], "action")
        self.assertEqual(camera["master_serial"], calib["reference_serial"])
        self.assertEqual(camera_serials, set(calib["cameras"]))

    def test_ros2_addresses(self) -> None:
        # 2026-08-18 换路由器后 PC 一律绑 192.168.50.230（Wi-Fi，路由器保留），
        # Peers 是两台车各自唯一的 IP（v03=.143 / v04=.68，车上静态配置）。
        car_ips = {"192.168.50.143", "192.168.50.68"}
        for filename in ("cyclonedds_18.xml",):
            root = ET.parse(ROS2_DIR / filename).getroot()
            self.assertEqual(
                root.find("./Domain/General/Interfaces/NetworkInterface").attrib["address"],
                "192.168.50.230",
            )
            peers = {
                peer.attrib["Address"]
                for peer in root.findall("./Domain/Discovery/Peers/Peer")
            }
            self.assertEqual(peers, car_ips)


if __name__ == "__main__":
    unittest.main()
