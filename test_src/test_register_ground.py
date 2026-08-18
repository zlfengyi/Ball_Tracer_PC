import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration.register_ground import (
    _calibrate_control_scale,
    _load_ground_controls,
    aggregate_stable_corners,
    make_ground_checkerboard_points,
    reproj_rms,
    solve_ground_pose,
)


class RegisterGroundTests(unittest.TestCase):
    def test_validates_triangulated_ground_control_baseline(self):
        K = np.array(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        relative = {
            "cam_a": np.zeros((3, 1), dtype=np.float64),
            "cam_b": np.array([[-1000.0], [0.0], [0.0]], dtype=np.float64),
        }
        calib = {
            "cameras": {
                serial: {
                    "K": K.tolist(),
                    "D": np.zeros(5).tolist(),
                    "R_ref_to_camera": np.eye(3).tolist(),
                    "t_ref_to_camera": translation.tolist(),
                }
                for serial, translation in relative.items()
            }
        }
        world = np.array([[-4115.0, 5480.0, 0.0], [4115.0, 5480.0, 0.0]])
        reference_points = np.array([[-4115.0, 0.0, 9000.0], [4115.0, 0.0, 9000.0]])
        controls = []
        for serial, translation in relative.items():
            projected, _ = cv2.projectPoints(
                reference_points,
                np.zeros(3),
                translation,
                K,
                np.zeros(5),
            )
            controls.append(
                {
                    "serial": serial,
                    "world_mm": world,
                    "image": projected.reshape(-1, 2),
                }
            )

        baseline = _calibrate_control_scale(calib, controls)

        self.assertAlmostEqual(baseline["expected_mm"], 8230.0)
        self.assertAlmostEqual(baseline["triangulated_mm"], 8230.0, places=5)
        self.assertAlmostEqual(baseline["error_mm"], 0.0, places=5)
        self.assertAlmostEqual(baseline["applied_scale_factor"], 1.0, places=8)

    def test_uses_ground_control_baseline_to_correct_translation_scale(self):
        K = np.array(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        true_translations = {
            "cam_a": np.zeros((3, 1), dtype=np.float64),
            "cam_b": np.array([[-1000.0], [0.0], [0.0]], dtype=np.float64),
        }
        calib = {
            "cameras": {
                serial: {
                    "K": K.tolist(),
                    "D": np.zeros(5).tolist(),
                    "R_ref_to_camera": np.eye(3).tolist(),
                    "t_ref_to_camera": (translation * 1.005).tolist(),
                }
                for serial, translation in true_translations.items()
            },
            "diagnostics": {
                "epipolar_residual": {
                    "pairs": {"cam_a|cam_b": {"baseline_mm": 1005.0}}
                }
            },
        }
        world = np.array([[-4115.0, 5480.0, 0.0], [4115.0, 5480.0, 0.0]])
        reference_points = np.array(
            [[-4115.0, 0.0, 9000.0], [4115.0, 0.0, 9000.0]]
        )
        controls = []
        for serial, translation in true_translations.items():
            projected, _ = cv2.projectPoints(
                reference_points, np.zeros(3), translation, K, np.zeros(5)
            )
            controls.append(
                {
                    "serial": serial,
                    "world_mm": world,
                    "image": projected.reshape(-1, 2),
                }
            )

        baseline = _calibrate_control_scale(calib, controls)

        self.assertAlmostEqual(
            baseline["input_triangulated_mm"], 8230.0 * 1.005, places=4
        )
        self.assertAlmostEqual(baseline["triangulated_mm"], 8230.0, places=5)
        np.testing.assert_allclose(
            calib["cameras"]["cam_b"]["t_ref_to_camera"],
            true_translations["cam_b"],
            atol=1e-7,
        )
        self.assertAlmostEqual(
            calib["diagnostics"]["epipolar_residual"]["pairs"]
            ["cam_a|cam_b"]["baseline_mm"],
            1000.0,
            places=6,
        )

        calib["diagnostics"]["ground_registration"] = {
            "ground_controls": {"baseline_scale_calibration": baseline}
        }
        repeated = _calibrate_control_scale(calib, controls)

        self.assertAlmostEqual(
            repeated["initial_triangulated_mm"], 8230.0 * 1.005, places=4
        )
        self.assertAlmostEqual(repeated["applied_scale_factor"], 1.0, places=7)
        self.assertAlmostEqual(
            repeated["cumulative_scale_factor"], 1.0 / 1.005, places=7
        )

    def test_loads_ground_controls_in_millimeters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            values = {
                "cam_a": {"1": [[-4.115, 5.48, 0.0], [61, 821]]},
                "cam_b": {"1": [[4.115, 5.48, 0.0], [1854, 641]]},
            }
            for serial, data in values.items():
                camera_dir = session / serial
                camera_dir.mkdir()
                (camera_dir / "0636_annotations.json").write_text(
                    json.dumps(data), encoding="utf-8"
                )

            controls, paths = _load_ground_controls(
                session, "0636", ["cam_a", "cam_b"]
            )

        np.testing.assert_allclose(controls[0]["world_mm"][0], [-4115, 5480, 0])
        np.testing.assert_allclose(controls[1]["image"][0], [1854, 641])
        self.assertEqual(set(paths), {"cam_a", "cam_b"})

    def test_reprojection_rms_uses_two_dimensional_distance(self):
        world = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        K = np.array(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        projected, _ = cv2.projectPoints(world, rvec, tvec, K, None)
        observed = projected.reshape(-1, 2) + np.array([3.0, 4.0])

        self.assertAlmostEqual(
            reproj_rms(world, observed, rvec, tvec, K, np.zeros(5)),
            5.0,
        )

    def test_world_points_follow_measured_top_left_corner(self):
        points = make_ground_checkerboard_points(0.0, 2.0)

        self.assertEqual(points.shape, (88, 3))
        np.testing.assert_allclose(points[0], [45.0, 1955.0, 0.0])
        np.testing.assert_allclose(points[10], [495.0, 1955.0, 0.0])
        np.testing.assert_allclose(points[-1], [495.0, 1640.0, 0.0])

    def test_repeated_frames_use_median_and_reject_motion(self):
        base = np.arange(176, dtype=np.float64).reshape(88, 2)
        median, worst = aggregate_stable_corners(
            [base - 0.02, base, base + 0.02]
        )
        np.testing.assert_allclose(median, base)
        self.assertLess(worst, 0.03)

        with self.assertRaisesRegex(RuntimeError, "moved between captures"):
            aggregate_stable_corners([base, base, base + 1.0])

    def test_two_boards_resolve_corner_reversal(self):
        K = np.array(
            [[1200.0, 0.0, 1024.0], [0.0, 1200.0, 768.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        camera_position = np.array([250.0, -1000.0, 2500.0])
        forward = np.array([0.0, 3500.0, -2500.0])
        forward /= np.linalg.norm(forward)
        right = np.array([1.0, 0.0, 0.0])
        down = np.cross(forward, right)
        R_reference = np.stack([right, down, forward])
        t_reference = (-R_reference @ camera_position).reshape(3, 1)

        relative_translation = np.array([-500.0, 0.0, 0.0])
        calib = {
            "reference_serial": "cam_a",
            "cameras": {
                "cam_a": {
                    "K": K.tolist(),
                    "D": np.zeros(5).tolist(),
                    "R_ref_to_camera": np.eye(3).tolist(),
                    "t_ref_to_camera": np.zeros((3, 1)).tolist(),
                },
                "cam_b": {
                    "K": K.tolist(),
                    "D": np.zeros(5).tolist(),
                    "R_ref_to_camera": np.eye(3).tolist(),
                    "t_ref_to_camera": relative_translation.reshape(3, 1).tolist(),
                },
            },
        }

        observations = []
        for session, outer_y in (("near", 2.0), ("far", 3.0)):
            world = make_ground_checkerboard_points(0.0, outer_y)
            for serial, relative_t in (
                ("cam_a", np.zeros((3, 1))),
                ("cam_b", relative_translation.reshape(3, 1)),
            ):
                t_camera = t_reference + relative_t
                rvec, _ = cv2.Rodrigues(R_reference)
                projected, _ = cv2.projectPoints(world, rvec, t_camera, K, None)
                observations.append(
                    {
                        "session": session,
                        "serial": serial,
                        "world_mm": world,
                        "image": projected.reshape(-1, 2)[::-1],
                    }
                )

        _, flips, scores = solve_ground_pose(
            calib, observations, ["near", "far"], []
        )

        self.assertEqual(flips, {"near": True, "far": True})
        self.assertLess(scores[0][0], 1e-5)
        self.assertGreater(scores[1][0], 1.0)


if __name__ == "__main__":
    unittest.main()
