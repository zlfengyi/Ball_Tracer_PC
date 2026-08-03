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

    def test_floor_ros2_addresses_are_separate(self) -> None:
        expected = {
            "cyclonedds.xml": ("192.168.50.230", "192.168.50.17"),
            "cyclonedds_18.xml": ("192.168.31.78", "192.168.31.23"),
        }
        for filename, (local_ip, arm_rk_ip) in expected.items():
            root = ET.parse(ROS2_DIR / filename).getroot()
            self.assertEqual(
                root.find("./Domain/General/Interfaces/NetworkInterface").attrib["address"],
                local_ip,
            )
            peers = {
                peer.attrib["Address"]
                for peer in root.findall("./Domain/Discovery/Peers/Peer")
            }
            self.assertIn(arm_rk_ip, peers)


if __name__ == "__main__":
    unittest.main()
