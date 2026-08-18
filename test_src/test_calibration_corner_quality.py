import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from calibration.detect_corners_v2 import (
    canonicalize_corner_order,
    checker_parity_sign,
    detect_corners,
    grid_quality_score,
    process_image,
)


class CalibrationCornerQualityTests(unittest.TestCase):
    def test_strong_perspective_grid_keeps_high_score(self):
        cols, rows = 8, 11
        ideal = np.array(
            [[col, row] for row in range(rows) for col in range(cols)],
            dtype=np.float32,
        )
        homography = np.array(
            [[43.0, 9.0, 180.0], [4.0, 35.0, 120.0], [0.012, 0.018, 1.0]],
            dtype=np.float64,
        )
        projected = cv2.perspectiveTransform(
            ideal.reshape(-1, 1, 2), homography
        )

        self.assertGreater(grid_quality_score(projected, cols, rows), 0.99)

    def test_non_projective_corner_errors_are_rejected(self):
        cols, rows = 8, 11
        corners = np.array(
            [[[60.0 + 30.0 * col, 80.0 + 30.0 * row]]
             for row in range(rows) for col in range(cols)],
            dtype=np.float32,
        )
        corners[::5, 0, 0] += 10.0
        corners[2::7, 0, 1] -= 9.0

        self.assertLess(grid_quality_score(corners, cols, rows), 0.8)

    def test_checker_parity_canonicalizes_reversed_order(self):
        cols, rows, square = 8, 11, 40
        gray = np.zeros(((rows + 1) * square, (cols + 1) * square), np.uint8)
        for row in range(rows + 1):
            for col in range(cols + 1):
                if (row + col) % 2 == 0:
                    gray[
                        row * square:(row + 1) * square,
                        col * square:(col + 1) * square,
                    ] = 255
        corners = np.array(
            [[[(col + 1) * square, (row + 1) * square]]
             for row in range(rows) for col in range(cols)],
            dtype=np.float32,
        )

        self.assertGreater(checker_parity_sign(gray, corners, cols, rows), 0.9)
        canonical, parity, reversed_order = canonicalize_corner_order(
            gray, corners[::-1], cols, rows
        )

        self.assertTrue(reversed_order)
        self.assertGreater(parity, 0.9)
        np.testing.assert_allclose(canonical, corners)

    def test_rejects_detection_that_remains_negative_after_reversal(self):
        corners = np.zeros((88, 1, 2), dtype=np.float32)
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.png"
            cv2.imwrite(str(image_path), np.zeros((120, 160, 3), np.uint8))
            with (
                patch(
                    "calibration.detect_corners_v2.detect_corners",
                    return_value=(corners, "sb"),
                ),
                patch(
                    "calibration.detect_corners_v2.canonicalize_corner_order",
                    return_value=(corners, -0.8, True),
                ),
            ):
                detection = process_image(
                    image_path, (8, 11), None, 8, 11, save_failed=False
                )

        self.assertIsNone(detection)

    def test_sb_corners_are_not_refined_a_second_time(self):
        corners = np.arange(176, dtype=np.float32).reshape(88, 1, 2)
        gray = np.zeros((120, 160), dtype=np.uint8)
        with (
            patch(
                "calibration.detect_corners_v2.cv2.findChessboardCornersSB",
                return_value=(True, corners),
            ),
            patch("calibration.detect_corners_v2.cv2.cornerSubPix") as refine,
        ):
            detected, method = detect_corners(gray, (8, 11))

        self.assertEqual(method, "sb")
        np.testing.assert_array_equal(detected, corners)
        refine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
