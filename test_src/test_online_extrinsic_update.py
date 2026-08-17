import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration.online_extrinsic_update import (
    COURT_LINES_MM,
    _serialize_like_source,
    build_line_reference,
    evaluate_holdout_decision,
    promote_candidate,
    refine_camera_pose,
    update_relative_extrinsics,
    validate_calibration_geometry,
    validate_candidate_changes,
)


def _look_at_pose(camera_center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack((right, down, forward))
    return R, (-R @ camera_center).reshape(3, 1)


def _render_court(
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int] = (1280, 960),
) -> np.ndarray:
    width, height = image_size
    image = np.full((height, width, 3), (45, 30, 20), dtype=np.uint8)
    rvec, _ = cv2.Rodrigues(R)
    for start, end, _ in COURT_LINES_MM.values():
        alpha = np.linspace(0.0, 1.0, 300)
        world = start[None, :] + alpha[:, None] * (end - start)[None, :]
        projected, _ = cv2.projectPoints(world, rvec, t, K, D)
        cv2.polylines(
            image,
            [np.rint(projected[:, 0]).astype(np.int32)],
            False,
            (210, 210, 210),
            5,
            cv2.LINE_AA,
        )
    return image


def _add_line_impostor(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    alpha_start: float = 0.25,
    alpha_end: float = 0.50,
    offset_px: float = 5.0,
) -> np.ndarray:
    altered = image.copy()
    start, end, _ = COURT_LINES_MM["center_service"]
    alpha = np.linspace(alpha_start, alpha_end, 80)
    world = start[None, :] + alpha[:, None] * (end - start)[None, :]
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(world, rvec, t, K, D)
    projected = projected[:, 0]
    tangent = np.gradient(projected, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    shifted = projected + offset_px * normal
    cv2.polylines(
        altered,
        [np.rint(projected).astype(np.int32)],
        False,
        (45, 30, 20),
        11,
        cv2.LINE_AA,
    )
    cv2.polylines(
        altered,
        [np.rint(shifted).astype(np.int32)],
        False,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return altered


def _calibration_fixture() -> dict:
    ref_R, _ = cv2.Rodrigues(np.array([0.02, -0.01, 0.03]))
    cam_R, _ = cv2.Rodrigues(np.array([-0.03, 0.04, -0.02]))
    ref_t = np.array([[100.0], [200.0], [300.0]])
    cam_t = np.array([[-400.0], [50.0], [600.0]])

    def camera(R: np.ndarray, t: np.ndarray) -> dict:
        return {
            "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]],
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [1280, 960],
            "R_ref_to_camera": np.eye(3).tolist(),
            "t_ref_to_camera": np.zeros((3, 1)).tolist(),
            "R_world": R.tolist(),
            "t_world": t.tolist(),
            "pos_world": np.zeros((3, 1)).tolist(),
        }

    calibration = {
        "reference_serial": "ref",
        "cameras": {
            "ref": camera(ref_R, ref_t),
            "cam": camera(cam_R, cam_t),
        },
        "diagnostics": {"keep": {"rms_px": 0.5}},
    }
    update_relative_extrinsics(calibration)
    return calibration


class CourtLineRefinerTests(unittest.TestCase):
    def test_recovers_larger_drift_at_production_focal_length(self):
        K = np.array(
            [[2380.0, 0.0, 1024.0], [0.0, 2377.0, 768.0], [0.0, 0.0, 1.0]]
        )
        D = np.zeros(5)
        initial_center = np.array([0.0, -3500.0, 3200.0])
        initial_R, initial_t = _look_at_pose(
            initial_center, np.array([0.0, 9000.0, 0.0])
        )
        reference_image = _render_court(
            K, D, initial_R, initial_t, (2048, 1536)
        )
        delta_R, _ = cv2.Rodrigues(np.array([0.0030, -0.0040, 0.0010]))
        moved_R = delta_R @ initial_R
        moved_center = initial_center + np.array([15.0, -10.0, 7.0])
        moved_t = (-moved_R @ moved_center).reshape(3, 1)
        moved_image = _render_court(K, D, moved_R, moved_t, (2048, 1536))
        camera = {
            "K": K.tolist(),
            "D": D.tolist(),
            "image_size": [2048, 1536],
            "R_world": initial_R.tolist(),
            "t_world": initial_t.tolist(),
        }

        line_reference, _ = build_line_reference(
            [reference_image, reference_image, reference_image], camera
        )
        refined_R, refined_t, metrics, _ = refine_camera_pose(
            [moved_image, moved_image, moved_image], camera, line_reference
        )

        rotation_error, _ = cv2.Rodrigues(refined_R @ moved_R.T)
        refined_center = (-refined_R.T @ refined_t).ravel()
        self.assertTrue(metrics["accepted"])
        self.assertLess(np.linalg.norm(rotation_error) * 180.0 / np.pi, 0.04)
        self.assertLess(np.linalg.norm(refined_center - moved_center), 6.0)

    def test_rejects_images_with_uncalibrated_resolution(self):
        K = np.eye(3)
        camera = {
            "K": K.tolist(),
            "D": np.zeros(5).tolist(),
            "image_size": [1280, 960],
            "R_world": np.eye(3).tolist(),
            "t_world": np.zeros((3, 1)).tolist(),
        }
        wrong_size = np.zeros((720, 1280, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "calibrated image_size"):
            build_line_reference([wrong_size, wrong_size, wrong_size], camera)

    def test_rejects_reference_profile_poisoned_by_one_parallel_line(self):
        K = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]]
        )
        D = np.zeros(5)
        center = np.array([0.0, -3500.0, 3200.0])
        R, t = _look_at_pose(center, np.array([0.0, 9000.0, 0.0]))
        clean_image = _render_court(K, D, R, t)
        poisoned_reference = _add_line_impostor(
            clean_image,
            K,
            D,
            R,
            t,
            alpha_start=0.04,
            alpha_end=0.96,
            offset_px=2.0,
        )
        camera = {
            "K": K.tolist(),
            "D": D.tolist(),
            "image_size": [1280, 960],
            "R_world": R.tolist(),
            "t_world": t.tolist(),
        }

        line_reference, _ = build_line_reference(
            [poisoned_reference, poisoned_reference, poisoned_reference], camera
        )
        _, _, metrics, _ = refine_camera_pose(
            [clean_image, clean_image, clean_image], camera, line_reference
        )

        self.assertFalse(metrics["accepted"])
        self.assertEqual(metrics["rotation_change_deg"], 0.0)
        self.assertEqual(metrics["position_change_mm"], 0.0)

    def test_rejects_local_false_line_without_pose_update(self):
        K = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]]
        )
        D = np.zeros(5)
        center = np.array([0.0, -3500.0, 3200.0])
        R, t = _look_at_pose(center, np.array([0.0, 9000.0, 0.0]))
        reference_image = _render_court(K, D, R, t)
        obstructed_image = _add_line_impostor(reference_image, K, D, R, t)
        camera = {
            "K": K.tolist(),
            "D": D.tolist(),
            "image_size": [1280, 960],
            "R_world": R.tolist(),
            "t_world": t.tolist(),
        }

        line_reference, _ = build_line_reference(
            [reference_image, reference_image, reference_image], camera
        )
        _, _, metrics, _ = refine_camera_pose(
            [obstructed_image, obstructed_image, obstructed_image],
            camera,
            line_reference,
        )

        self.assertFalse(metrics["accepted"])
        self.assertEqual(metrics["rotation_change_deg"], 0.0)
        self.assertEqual(metrics["position_change_mm"], 0.0)

    def test_recovers_small_pose_drift(self):
        K = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]]
        )
        D = np.zeros(5)
        initial_center = np.array([0.0, -3500.0, 3200.0])
        initial_R, initial_t = _look_at_pose(
            initial_center, np.array([0.0, 9000.0, 0.0])
        )
        reference_image = _render_court(K, D, initial_R, initial_t)

        delta_R, _ = cv2.Rodrigues(np.array([0.0015, -0.0020, 0.0008]))
        moved_R = delta_R @ initial_R
        moved_center = initial_center + np.array([18.0, -12.0, 9.0])
        moved_t = (-moved_R @ moved_center).reshape(3, 1)
        moved_image = _render_court(K, D, moved_R, moved_t)
        camera = {
            "K": K.tolist(),
            "D": D.tolist(),
            "image_size": [1280, 960],
            "R_world": initial_R.tolist(),
            "t_world": initial_t.tolist(),
        }

        line_reference, _ = build_line_reference(
            [reference_image, reference_image, reference_image], camera
        )
        refined_R, refined_t, metrics, _ = refine_camera_pose(
            [moved_image, moved_image, moved_image], camera, line_reference
        )

        rotation_error, _ = cv2.Rodrigues(refined_R @ moved_R.T)
        refined_center = (-refined_R.T @ refined_t).ravel()
        self.assertTrue(metrics["accepted"])
        self.assertLess(np.linalg.norm(rotation_error) * 180.0 / np.pi, 0.04)
        self.assertLess(np.linalg.norm(refined_center - moved_center), 6.0)
        self.assertLess(metrics["final_rms_px"], metrics["initial_rms_px"] * 0.25)

    def test_recomputes_relative_extrinsics_from_world_poses(self):
        ref_R, _ = cv2.Rodrigues(np.array([0.1, -0.05, 0.02]))
        cam_R, _ = cv2.Rodrigues(np.array([-0.03, 0.07, -0.04]))
        ref_t = np.array([[100.0], [200.0], [300.0]])
        cam_t = np.array([[-400.0], [50.0], [600.0]])
        calib = {
            "reference_serial": "ref",
            "cameras": {
                "ref": {"R_world": ref_R.tolist(), "t_world": ref_t.tolist()},
                "cam": {"R_world": cam_R.tolist(), "t_world": cam_t.tolist()},
            },
        }

        update_relative_extrinsics(calib)

        expected_R = cam_R @ ref_R.T
        expected_t = cam_t - expected_R @ ref_t
        np.testing.assert_allclose(
            calib["cameras"]["cam"]["R_ref_to_camera"], expected_R, atol=1e-12
        )
        np.testing.assert_allclose(
            calib["cameras"]["cam"]["t_ref_to_camera"], expected_t, atol=1e-12
        )
        np.testing.assert_allclose(
            calib["cameras"]["ref"]["R_ref_to_camera"], np.eye(3), atol=1e-12
        )

    def test_selective_update_preserves_unaffected_camera_and_diagnostics(self):
        baseline = _calibration_fixture()
        candidate = copy.deepcopy(baseline)
        untouched_reference = copy.deepcopy(candidate["cameras"]["ref"])
        delta_R, _ = cv2.Rodrigues(np.array([0.001, -0.002, 0.0005]))
        old_R = np.asarray(candidate["cameras"]["cam"]["R_world"])
        candidate["cameras"]["cam"]["R_world"] = (delta_R @ old_R).tolist()
        old_t = np.asarray(candidate["cameras"]["cam"]["t_world"])
        candidate["cameras"]["cam"]["t_world"] = (
            old_t + np.array([[2.0], [-3.0], [4.0]])
        ).tolist()

        changed = update_relative_extrinsics(candidate, "cam")

        self.assertEqual(candidate["cameras"]["ref"], untouched_reference)
        self.assertEqual(candidate["diagnostics"], baseline["diagnostics"])
        self.assertEqual(
            set(changed["cam"]),
            {"pos_world", "R_ref_to_camera", "t_ref_to_camera"},
        )
        validated = validate_candidate_changes(baseline, candidate, "cam")
        self.assertEqual(
            set(validated["cam"]),
            {
                "R_world",
                "t_world",
                "pos_world",
                "R_ref_to_camera",
                "t_ref_to_camera",
            },
        )
        validate_calibration_geometry(candidate)

    def test_candidate_rejects_intrinsic_or_diagnostic_changes(self):
        baseline = _calibration_fixture()
        candidate = copy.deepcopy(baseline)
        candidate["cameras"]["cam"]["K"][0][0] += 1.0
        with self.assertRaisesRegex(ValueError, "illegally changes K"):
            validate_candidate_changes(baseline, candidate, "cam")

        candidate = copy.deepcopy(baseline)
        candidate["diagnostics"]["keep"]["rms_px"] = 99.0
        with self.assertRaisesRegex(ValueError, "top-level field diagnostics"):
            validate_candidate_changes(baseline, candidate, "cam")

    def test_json_serialization_preserves_source_format(self):
        calibration = _calibration_fixture()
        source = json.dumps(calibration, indent=4, ensure_ascii=False).replace(
            "\n", "\r\n"
        )
        self.assertFalse(source.endswith("\r\n"))
        self.assertEqual(_serialize_like_source(calibration, source), source)

        changed = copy.deepcopy(calibration)
        changed["cameras"]["cam"]["t_world"][0][0] += 1.0
        rendered = _serialize_like_source(changed, source)
        self.assertIn("\r\n", rendered)
        self.assertFalse(rendered.endswith("\r\n"))
        self.assertEqual(json.loads(rendered), changed)

    def test_flying_ball_gate_requires_enough_paired_samples(self):
        insufficient = evaluate_holdout_decision([10.0] * 19, [5.0] * 19)
        self.assertFalse(insufficient["supported"])

        supported = evaluate_holdout_decision([10.0] * 20, [5.0] * 20)
        self.assertTrue(supported["supported"])
        self.assertEqual(supported["baseline_px"]["n"], 20)
        self.assertEqual(supported["candidate_better_fraction"], 1.0)

    def test_flying_ball_gate_rejects_p95_regression(self):
        baseline = [10.0] * 20
        candidate = [5.0] * 18 + [30.0, 30.0]
        decision = evaluate_holdout_decision(baseline, candidate)
        self.assertLess(decision["candidate_px"]["median"], 8.0)
        self.assertGreater(decision["candidate_px"]["p95"], 10.0)
        self.assertFalse(decision["supported"])

    def test_promotion_is_atomic_and_keeps_verified_backup(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            calibration_path = root / "calibration.json"
            backup_path = root / "backup.json"
            baseline_payload = b'{"version": 1}\n'
            candidate_payload = b'{"version": 2}\n'
            calibration_path.write_bytes(baseline_payload)
            expected_hash = hashlib.sha256(baseline_payload).hexdigest()

            promote_candidate(
                calibration_path,
                expected_hash,
                candidate_payload,
                backup_path,
            )

            self.assertEqual(calibration_path.read_bytes(), candidate_payload)
            self.assertEqual(backup_path.read_bytes(), baseline_payload)

    def test_promotion_rejects_changed_baseline(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            calibration_path = root / "calibration.json"
            backup_path = root / "backup.json"
            baseline_payload = b'{"version": 1}\n'
            expected_hash = hashlib.sha256(baseline_payload).hexdigest()
            calibration_path.write_bytes(b'{"version": 9}\n')

            with self.assertRaisesRegex(RuntimeError, "changed during validation"):
                promote_candidate(
                    calibration_path,
                    expected_hash,
                    b'{"version": 2}\n',
                    backup_path,
                )
            self.assertFalse(backup_path.exists())
            self.assertEqual(calibration_path.read_bytes(), b'{"version": 9}\n')


if __name__ == "__main__":
    unittest.main()
