from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "src" / "config"
ROS2_DIR = ROOT_DIR / "ros2"


class FloorConfigTest(unittest.TestCase):
    def assert_floor_config(self, suffix: str, trigger_mode: str) -> set[str]:
        camera_path = CONFIG_DIR / f"camera{suffix}.json"
        calib_path = CONFIG_DIR / f"four_camera_calib{suffix}.json"
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        calib = json.loads(calib_path.read_text(encoding="utf-8"))

        camera_serials = {camera["master_serial"], *camera["slave_serials"]}
        calib_serials = set(calib["cameras"])
        self.assertEqual(camera["trigger_mode"], trigger_mode)
        self.assertEqual(camera["master_serial"], calib["reference_serial"])
        self.assertEqual(camera_serials, calib_serials)
        return camera_serials

    def test_floor_configs_are_complete_and_separate(self) -> None:
        floor_16 = self.assert_floor_config("", "line")
        floor_18 = self.assert_floor_config("_18", "action")
        self.assertTrue(floor_16.isdisjoint(floor_18))

    def test_floor_ros2_addresses(self) -> None:
        # 2026-08-18 换路由器后两层同网段：PC 一律绑 192.168.50.230（Wi-Fi，
        # 路由器保留），Peers 是两台车各自唯一的 IP（v03=.143 / v04=.68，
        # 车上静态配置，跟楼层无关）。楼层差异只剩相机/标定，不再分网段。
        car_ips = {"192.168.50.143", "192.168.50.68"}
        for filename in ("cyclonedds.xml", "cyclonedds_18.xml"):
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
