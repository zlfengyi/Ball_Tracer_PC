from __future__ import annotations

from src import ros2_support


def test_ensure_ros2_environment_defaults_ros_domain_id_to_2(monkeypatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.delenv("BALL_TRACER_ROS_DOMAIN_ID", raising=False)

    ros2_support.ensure_ros2_environment()

    assert ros2_support.os.environ["ROS_DOMAIN_ID"] == "2"


def test_ensure_ros2_environment_prefers_ball_tracer_ros_domain_id(monkeypatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    monkeypatch.setenv("BALL_TRACER_ROS_DOMAIN_ID", "7")

    ros2_support.ensure_ros2_environment()

    assert ros2_support.os.environ["ROS_DOMAIN_ID"] == "7"
