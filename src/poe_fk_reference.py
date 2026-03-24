# -*- coding: utf-8 -*-
"""Minimal FK reference for arm_poe_racket_center.json.

This file is intentionally standalone so another agent can understand the
runtime FK without needing the rest of the repo context.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "arm_poe_racket_center.json"


def load_poe_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as inp:
        return json.load(inp)


def normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("zero-length axis direction")
    return arr / norm


def skew(vec: Sequence[float]) -> np.ndarray:
    x, y, z = np.array(vec, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def exp_revolute(omega: Sequence[float], q_point_mm: Sequence[float], theta_rad: float) -> np.ndarray:
    """exp([S] * theta) for a revolute joint defined by axis direction omega and one point q."""
    omega = normalize(omega)
    q_point_mm = np.array(q_point_mm, dtype=np.float64).reshape(3)
    omega_hat = skew(omega)
    theta = float(theta_rad)

    R = (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * omega_hat
        + (1.0 - math.cos(theta)) * (omega_hat @ omega_hat)
    )
    t = (np.eye(3, dtype=np.float64) - R) @ q_point_mm

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def forward_base_mm(joint_values_rad: Sequence[float], config: dict) -> np.ndarray:
    """Return racket center in base coordinates, unit mm."""
    poe_cfg = config["poe_model_position_only"]
    axes = poe_cfg["space_axes_base"]
    offsets = np.array(poe_cfg["joint_angle_offsets_rad"], dtype=np.float64)
    home_point_base_mm = np.array(poe_cfg["home_point_base_mm"], dtype=np.float64).reshape(3)

    q = np.array(joint_values_rad[: len(offsets)], dtype=np.float64)
    q_used = q + offsets

    T = np.eye(4, dtype=np.float64)
    for axis_cfg, theta in zip(axes, q_used):
        T = T @ exp_revolute(axis_cfg["omega"], axis_cfg["q_mm"], float(theta))

    home_h = np.array(
        [home_point_base_mm[0], home_point_base_mm[1], home_point_base_mm[2], 1.0],
        dtype=np.float64,
    )
    point_h = T @ home_h
    return point_h[:3]


def forward_rel_world_mm(joint_values_rad: Sequence[float], config: dict) -> np.ndarray:
    """Return p_racket_rel_base_in_world(by poe), unit mm."""
    point_base_mm = forward_base_mm(joint_values_rad, config)
    R_base_in_world = np.array(config["T_base_in_world"]["R"], dtype=np.float64).reshape(3, 3)
    return R_base_in_world @ point_base_mm


def forward_world_mm(joint_values_rad: Sequence[float], config: dict) -> np.ndarray:
    """Return absolute racket center in world coordinates, unit mm."""
    point_rel_world_mm = forward_rel_world_mm(joint_values_rad, config)
    t_base_in_world_mm = np.array(config["T_base_in_world"]["t_mm"], dtype=np.float64).reshape(3)
    return point_rel_world_mm + t_base_in_world_mm


if __name__ == "__main__":
    cfg = load_poe_config()

    # Example q = [joint_1, joint_2, joint_3, joint_4] in rad.
    q_example = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    p_base_mm = forward_base_mm(q_example, cfg)
    p_rel_world_mm = forward_rel_world_mm(q_example, cfg)
    p_world_mm = forward_world_mm(q_example, cfg)

    np.set_printoptions(precision=3, suppress=True)
    print("q_rad =", q_example.tolist())
    print("p_racket_base_mm =", p_base_mm.tolist())
    print("p_racket_rel_base_in_world_mm =", p_rel_world_mm.tolist())
    print("p_racket_world_mm =", p_world_mm.tolist())
