import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.detect_corners_v2 import (
    DETECTION_QUALITY_METRIC,
    MIN_PARITY_CONFIDENCE,
)
from calibration.multi_calibrator import (
    BoardConfig,
    CameraCalib,
    MultiCalibResult,
    MultiCalibrator,
    _detect_checkerboard,
    _validate_epipolar_residual,
    _make_obj_points,
)


def _transform(rotation_xyz, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rotation_xyz).as_matrix()
    transform[:3, 3] = translation
    return transform


def _project_transform(points, transform, K, D):
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    projected, _ = cv2.projectPoints(points, rvec, transform[:3, 3], K, D)
    return projected.astype(np.float32)


class MultiCalibratorTests(unittest.TestCase):
    def test_live_detection_uses_shared_corner_detector(self):
        board = BoardConfig(inner_cols=8, inner_rows=11, square_size=45.0)
        gray = np.zeros((120, 160), dtype=np.uint8)
        corners = np.zeros((88, 1, 2), dtype=np.float32)
        with (
            patch(
                "calibration.multi_calibrator.detect_corners",
                return_value=(corners, "sb"),
            ) as detect,
            patch(
                "calibration.multi_calibrator.canonicalize_corner_order",
                return_value=(corners, 0.9, False),
            ),
        ):
            result = _detect_checkerboard(gray, board)

        np.testing.assert_array_equal(result, corners)
        detect.assert_called_once_with(gray, (8, 11))

    def test_result_schema_names_reference_to_camera_direction(self):
        camera = CameraCalib(
            serial="A",
            K=np.eye(3),
            D=np.zeros(5),
            image_size=(100, 80),
            R_ref_to_camera=np.eye(3),
            t_ref_to_camera=np.array([[1.0], [2.0], [3.0]]),
        )
        result = MultiCalibResult(
            reference_serial="A",
            cameras={"A": camera},
            board_config=BoardConfig(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "calibration.json"
            result.save(output)
            camera_json = json.loads(output.read_text(encoding="utf-8"))["cameras"]["A"]
            self.assertIn("R_ref_to_camera", camera_json)
            self.assertIn("t_ref_to_camera", camera_json)
            self.assertNotIn("R_to_ref", camera_json)
            self.assertNotIn("t_to_ref", camera_json)
            loaded = MultiCalibResult.load(output)
            self.assertEqual(loaded.epipolar_residual, {})
            np.testing.assert_array_equal(
                loaded.cameras["A"].t_ref_to_camera,
                np.array([1.0, 2.0, 3.0]),
            )

    def test_epipolar_validation_accepts_consistent_horizontal_baseline(self):
        serials, detections, cameras = self._make_epipolar_case()

        diagnostics = _validate_epipolar_residual(
            serials, "A", detections, cameras
        )

        pair = diagnostics["pairs"]["A|B"]
        self.assertTrue(pair["validated"])
        self.assertEqual(pair["residual_axis"], "y")
        self.assertEqual(pair["frames"], 12)
        self.assertEqual(diagnostics["validated_pairs"], 1)
        self.assertLess(pair["rms_px"], 1e-4)

    def test_epipolar_validation_accepts_consistent_vertical_baseline(self):
        vertical_pose = _transform((0.01, 0.02, -0.01), (20.0, -350.0, 10.0))
        serials, detections, cameras = self._make_epipolar_case(vertical_pose)

        diagnostics = _validate_epipolar_residual(
            serials, "A", detections, cameras
        )

        pair = diagnostics["pairs"]["A|B"]
        self.assertEqual(pair["residual_axis"], "x")
        self.assertLess(pair["rms_px"], 1e-4)

    def test_epipolar_validation_rejects_orthogonal_mismatch(self):
        serials, detections, cameras = self._make_epipolar_case()
        for frame in detections.values():
            frame["B"] = frame["B"].copy()
            frame["B"][:, :, 1] += 3.0

        with self.assertRaisesRegex(RuntimeError, "Epipolar y residual too high"):
            _validate_epipolar_residual(serials, "A", detections, cameras)

    def test_epipolar_validation_requires_ten_frame_supported_path(self):
        serials, detections, cameras = self._make_epipolar_case()
        detections = {idx: frame for idx, frame in detections.items() if idx <= 9}

        with self.assertRaisesRegex(RuntimeError, "graph is disconnected"):
            _validate_epipolar_residual(serials, "A", detections, cameras)

    @staticmethod
    def _make_epipolar_case(camera_b_pose=None):
        serials = ["A", "B"]
        board = BoardConfig(inner_cols=8, inner_rows=11, square_size=45.0)
        points = _make_obj_points(board)
        K = np.array(
            [[1400.0, 0.0, 1024.0], [0.0, 1400.0, 768.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        if camera_b_pose is None:
            camera_b_pose = _transform(
                (0.01, 0.02, -0.01), (-350.0, 20.0, 10.0)
            )
        camera_poses = {"A": np.eye(4), "B": camera_b_pose}
        detections = {}
        for index in range(1, 13):
            board_pose = _transform(
                (0.01 * index, -0.008 * index, 0.004 * index),
                (-150.0 + 20.0 * index, -100.0 + 10.0 * index, 1800.0),
            )
            detections[index] = {
                serial: _project_transform(
                    points, camera_pose @ board_pose, K, distortion
                )
                for serial, camera_pose in camera_poses.items()
            }
        cameras = {
            serial: CameraCalib(
                serial=serial,
                K=K.copy(),
                D=distortion.copy(),
                image_size=(2048, 1536),
                R_ref_to_camera=pose[:3, :3].copy(),
                t_ref_to_camera=pose[:3, 3].reshape(3, 1).copy(),
            )
            for serial, pose in camera_poses.items()
        }
        return serials, detections, cameras

    def test_cache_rejects_wrong_image_size_and_out_of_bounds_corners(self):
        board = BoardConfig(inner_cols=8, inner_rows=11, square_size=45.0)
        corners = np.column_stack(
            np.meshgrid(np.arange(8, dtype=np.float32), np.arange(11, dtype=np.float32))
        ).reshape(-1, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            camera_dir = image_dir / "A"
            camera_dir.mkdir()
            self.assertTrue(
                cv2.imwrite(str(camera_dir / "0001.png"), np.zeros((80, 100), np.uint8))
            )
            cache_path = image_dir / "corner_detections.json"
            calibrator = MultiCalibrator(
                serials=["A"],
                image_dir=image_dir,
                reference_serial="A",
                board=board,
                image_range=(1, 1),
            )

            def write_cache(image_size, cached_corners):
                cache_path.write_text(
                    json.dumps(
                        {
                            "board": {"inner_cols": 8, "inner_rows": 11},
                            "quality_metric": DETECTION_QUALITY_METRIC,
                            "min_parity_confidence": MIN_PARITY_CONFIDENCE,
                            "cameras": {
                                "A": {
                                    "1": {
                                        "corners": cached_corners.tolist(),
                                        "image_size": image_size,
                                        "score": 1.0,
                                        "parity_sign": 1.0,
                                        "reversed_to_canonical": False,
                                    }
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            write_cache([101, 80], corners)
            with self.assertRaisesRegex(RuntimeError, "cache="):
                calibrator._detect_all(board)

            corners_outside = corners.copy()
            corners_outside[0] = [100.0, 10.0]
            write_cache([100, 80], corners_outside)
            with self.assertRaisesRegex(RuntimeError, "A/1"):
                calibrator._detect_all(board)

    def test_pairwise_filter_removes_bad_observation_from_ba_input(self):
        serials = ["A", "B", "C"]
        board = BoardConfig(inner_cols=8, inner_rows=11, square_size=45.0)
        points = _make_obj_points(board)
        K = np.array(
            [[1400.0, 0.0, 1024.0], [0.0, 1400.0, 768.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        camera_poses = {
            "A": np.eye(4),
            "B": _transform((0.01, 0.02, -0.01), (-350.0, 20.0, 10.0)),
            "C": _transform((-0.01, -0.015, 0.01), (350.0, -10.0, 20.0)),
        }
        detections = {}
        for index in range(1, 8):
            board_pose = _transform(
                (0.025 * index, -0.018 * index, 0.01 * index),
                (-120.0 + 35.0 * index, -80.0 + 20.0 * index, 1800.0 + 25.0 * index),
            )
            detections[index] = {}
            for serial in serials:
                camera_board = camera_poses[serial] @ board_pose
                if index == 4 and serial == "C":
                    camera_board = camera_board.copy()
                    camera_board[0, 3] += 500.0
                detections[index][serial] = _project_transform(
                    points, camera_board, K, distortion
                )
        detections[8] = {
            "A": _project_transform(
                points,
                _transform((0.03, -0.02, 0.01), (50.0, 20.0, 1900.0)),
                K,
                distortion,
            )
        }

        calibrator = MultiCalibrator(
            serials=serials,
            image_dir=Path("."),
            reference_serial="A",
            board=board,
            fix_intrinsics=True,
            min_cameras_per_board=2,
        )
        intrinsics = {serial: K for serial in serials}
        distortions = {serial: distortion for serial in serials}

        calibrator._init_extrinsics_pairwise(
            detections, intrinsics, distortions, points
        )

        self.assertEqual(set(detections[4]), {"A", "B"})
        self.assertEqual(detections[8], {})

    def test_bundle_adjust_rejects_high_reprojection_rms(self):
        serials = ["A", "B"]
        board = BoardConfig(inner_cols=8, inner_rows=11, square_size=45.0)
        points = _make_obj_points(board)
        K = np.array(
            [[1400.0, 0.0, 1024.0], [0.0, 1400.0, 768.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        camera_poses = {
            "A": np.eye(4),
            "B": _transform((0.01, 0.02, -0.01), (-350.0, 20.0, 10.0)),
        }
        board_pose = _transform((0.05, -0.04, 0.02), (-100.0, -80.0, 1800.0))
        detections = {
            1: {
                serial: _project_transform(
                    points, camera_pose @ board_pose, K, distortion
                ) + 2.0
                for serial, camera_pose in camera_poses.items()
            }
        }
        calibrator = MultiCalibrator(
            serials=serials,
            image_dir=Path("."),
            reference_serial="A",
            board=board,
            fix_intrinsics=True,
            min_cameras_per_board=2,
        )

        def unchanged_result(residuals, x0, **_kwargs):
            fun = residuals(x0)
            return SimpleNamespace(
                x=x0.copy(),
                fun=fun,
                success=True,
                status=1,
                message="test result",
                nfev=1,
                cost=float(np.dot(fun, fun) / 2.0),
            )

        with patch(
            "calibration.multi_calibrator.least_squares",
            side_effect=unchanged_result,
        ), self.assertRaisesRegex(RuntimeError, r"2\.828px > 1\.000px"):
            calibrator._bundle_adjust(
                detections=detections,
                valid_images=[1],
                image_sizes={serial: (2048, 1536) for serial in serials},
                init_K={serial: K for serial in serials},
                init_D={serial: distortion for serial in serials},
                init_cam_poses=camera_poses,
                init_board_poses={1: board_pose},
                board=board,
                obj_pts=points,
            )


if __name__ == "__main__":
    unittest.main()
