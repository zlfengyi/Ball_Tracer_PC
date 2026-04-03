from __future__ import annotations

from src import ball_grabber


def test_rotation_defaults_prefer_camera_hardware():
    assert ball_grabber._ENV_CAMERA_REVERSE_180 is True
    assert ball_grabber._ENV_SOFTWARE_ROTATE_180 is False
