from __future__ import annotations

import http.client
import json
import time
import unittest

import cv2
import numpy as np

from calibration.phone_preview import PhonePreviewServer


class PhonePreviewServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preview = PhonePreviewServer(["CAM_A", "CAM_B"], target_frames=25, port=0)
        self.preview.start(host="127.0.0.1")

    def tearDown(self) -> None:
        self.preview.close()

    def request(self, path: str) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.preview.port, timeout=2.0)
        connection.request("GET", path)
        response = connection.getresponse()
        status = response.status
        headers = {key.lower(): value for key, value in response.getheaders()}
        body = response.read()
        connection.close()
        return status, headers, body

    def test_page_status_and_resized_jpeg(self) -> None:
        self.assertFalse(self.preview.should_publish(time.perf_counter()))
        status, headers, page = self.request("/")
        self.assertEqual(200, status)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn(b"CAM_A", page)
        self.assertIn(b"CAM_B", page)

        self.preview.update_status(captured=7, elapsed_s=3.5)
        status, _, body = self.request("/status.json")
        self.assertEqual(200, status)
        self.assertEqual(
            {"captured": 7, "target": 25, "elapsed_s": 3.5},
            json.loads(body),
        )

        now = time.perf_counter()
        self.assertTrue(self.preview.should_publish(now))
        image = np.full((1536, 2048, 3), (20, 120, 220), dtype=np.uint8)
        self.preview.publish({"CAM_A": image, "CAM_B": image}, now)
        status, headers, jpeg = self.request("/frame/CAM_A.jpg")
        self.assertEqual(200, status)
        self.assertEqual("image/jpeg", headers["content-type"])
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((540, 720), decoded.shape[:2])


if __name__ == "__main__":
    unittest.main()
