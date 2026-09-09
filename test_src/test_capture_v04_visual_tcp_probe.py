from __future__ import annotations

from collections import Counter
import unittest

from ArmCalibration import capture_v04_visual_tcp_probe as probe


class VisualTcpStaticCaptureTest(unittest.TestCase):
    def test_plan_order_and_counts(self) -> None:
        plan = probe.build_plan()
        self.assertEqual(len(plan), 16)
        self.assertEqual(
            [trial.point.point_id for trial in plan],
            [point_id for point_id, _phase in probe.PLAN_SPEC],
        )
        self.assertEqual(
            Counter(trial.point.point_id for trial in plan),
            {"C0": 4, "X1": 3, "X2": 3, "Cross": 3, "Z1": 3},
        )
        self.assertEqual(plan[0].phase, "anchor_start")
        self.assertEqual(plan[-1].phase, "anchor_end")

    def test_inspect_command_uses_model_z(self) -> None:
        point = probe.POINTS["C0"]
        self.assertEqual(probe.inspect_command(point), "inspect 0.8245 0.9253")
        self.assertNotIn("1.0956", probe.inspect_command(point))

    def test_acceptance_is_exact(self) -> None:
        point = probe.POINTS["C0"]
        accepted = probe.parse_inspect_status(
            "accepted arm_command inspect x=0.8245 z=0.9253 "
            "duration=1.000 t=123.456789",
            point,
        )
        self.assertEqual(accepted["duration"], 1.0)
        for text in (
            "accepted arm_command inspect x=0.8245 z=0.9753 duration=1.000 t=1.0",
            "accepted arm_command inspect x=0.8245 z=0.9253 duration=1.000 extra=1 t=1.0",
            "accepted arm_command ready duration=1.000 t=1.0",
        ):
            with self.subTest(text=text), self.assertRaises(probe.ExperimentError):
                probe.parse_inspect_status(text, point)

    def test_preset_acceptance(self) -> None:
        ready = probe.parse_preset_status(
            "accepted arm_command inspect ready duration=1.000 t=8.000000",
            "inspect ready",
        )
        self.assertEqual(ready["duration"], 1.0)
        droop = probe.parse_preset_status(
            "accepted arm_command droop duration=8.000 t=9.000000",
            "droop",
        )
        self.assertEqual(droop["duration"], 8.0)


if __name__ == "__main__":
    unittest.main()
