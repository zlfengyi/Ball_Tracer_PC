import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration.four_camera_intrinsics_calibrate import (
    _calibrate_one_camera,
    _make_obj_points,
)
from calibration.four_camera_calib_common import create_session_dir, latest_session_dir


class FourCameraIntrinsicsCalibrationTests(unittest.TestCase):
    def test_session_selection_uses_sequence_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "001_first").mkdir()
            (root / "003_third").mkdir()
            (root / "manual").mkdir()

            self.assertEqual(latest_session_dir(root).name, "003_third")
            session_dir = create_session_dir(root, "checker board")
            self.assertEqual(session_dir.name, "004_checker_board")
            self.assertTrue(session_dir.is_dir())
            with self.assertRaises(FileExistsError):
                create_session_dir(root, "ignored", "003_third")

    def test_rejects_non_rigid_bad_frame(self):
        rng = np.random.default_rng(7)
        obj_pts = _make_obj_points(8, 11, 45.0)
        image_size = [1920, 1200]
        K = np.array(
            [[1350.0, 0.0, 960.0], [0.0, 1340.0, 600.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        detections = {}
        for index in range(1, 13):
            rvec = np.array(
                [0.04 * (index - 6), 0.025 * ((index % 4) - 1.5), 0.01 * index],
                dtype=np.float64,
            )
            tvec = np.array(
                [-220.0 + 35.0 * index, -150.0 + 24.0 * index, 1250.0 + 18.0 * index],
                dtype=np.float64,
            )
            corners, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
            corners = corners.reshape(-1, 2)
            corners += rng.normal(0.0, 0.08, corners.shape)
            if index == 6:
                corners += rng.normal(0.0, 8.0, corners.shape)
            detections[str(index)] = {
                "corners": corners.tolist(),
                "image_size": image_size,
                "score": 1.0,
            }

        result = _calibrate_one_camera(
            "camera", detections, obj_pts, score_threshold=0.8, max_frames=0
        )

        rejected = {item["index"] for item in result["diagnostics"]["rejected_frames"]}
        self.assertIn(6, rejected)
        self.assertEqual(result["frames_used"], 11)
        self.assertTrue(np.all(np.isfinite(np.asarray(result["K"]))))
        self.assertGreater(result["K"][0][0], 0.0)
        self.assertGreater(result["K"][1][1], 0.0)

    def test_refuses_sampling_below_minimum_frame_count(self):
        detection = {
            "corners": np.zeros((88, 2), dtype=np.float32).tolist(),
            "image_size": [1920, 1200],
            "score": 1.0,
        }
        detections = {str(index): detection for index in range(1, 9)}

        with self.assertRaisesRegex(RuntimeError, "max_frames leaves too few"):
            _calibrate_one_camera(
                "camera",
                detections,
                _make_obj_points(8, 11, 45.0),
                score_threshold=0.8,
                max_frames=7,
            )


if __name__ == "__main__":
    unittest.main()
