# -*- coding: utf-8 -*-
"""Forward kinematics helpers for the exported racket-center POE config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _SRC_DIR / "config" / "arm_poe_racket_center.json"


@dataclass(frozen=True)
class PoeAxis:
    joint_index: int
    omega: np.ndarray
    q_mm: np.ndarray
    v_mm: np.ndarray


@dataclass(frozen=True)
class PoeForwardResult:
    joint_values_rad: list[float]
    joint_values_used_rad: list[float]
    point_base_mm: np.ndarray
    point_world_mm: np.ndarray


class ArmPoePositionModel:
    """Position-only POE model for the racket center."""

    def __init__(
        self,
        *,
        config_path: str | Path = _DEFAULT_CONFIG,
    ) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path, "r", encoding="utf-8") as inp:
            cfg = json.load(inp)

        poe_cfg = cfg["poe_model_position_only"]
        t_cfg = cfg["T_base_in_world"]
        vehicle_cfg = cfg.get("vehicle_reference", {})
        self._joint_count = int(poe_cfg["joint_count"])
        self._joint_offsets = np.array(poe_cfg["joint_angle_offsets_rad"], dtype=np.float64)
        self._home_point_base = np.array(poe_cfg["home_point_base_mm"], dtype=np.float64)
        self._R_base_in_world = np.array(t_cfg["R"], dtype=np.float64).reshape(3, 3)
        self._t_base_in_world = np.array(t_cfg["t_mm"], dtype=np.float64).reshape(3)
        self._apriltag_to_car_base_offset_mm = np.array(
            vehicle_cfg.get("apriltag_center_to_car_base_offset_mm", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        self._axes: list[PoeAxis] = []
        for axis_cfg in poe_cfg["space_axes_base"]:
            self._axes.append(
                PoeAxis(
                    joint_index=int(axis_cfg["joint_index"]),
                    omega=self._normalize(np.array(axis_cfg["omega"], dtype=np.float64)),
                    q_mm=np.array(axis_cfg["q_mm"], dtype=np.float64),
                    v_mm=np.array(axis_cfg["v_mm"], dtype=np.float64),
                )
            )

        expected_names = [f"joint_{idx}" for idx in range(1, self._joint_count + 1)]
        self._expected_joint_names = expected_names

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            raise ValueError("invalid zero-length axis direction in POE config")
        return vec / norm

    @staticmethod
    def _exp_revolute(axis: PoeAxis, theta: float) -> np.ndarray:
        R, _ = cv2.Rodrigues((axis.omega * float(theta)).reshape(3, 1))
        t = (np.eye(3, dtype=np.float64) - R) @ axis.q_mm
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def joint_count(self) -> int:
        return self._joint_count

    @property
    def joint_offsets_rad(self) -> list[float]:
        return self._joint_offsets.tolist()

    @property
    def expected_joint_names(self) -> list[str]:
        return list(self._expected_joint_names)

    @property
    def t_base_in_world_mm(self) -> np.ndarray:
        return self._t_base_in_world.copy()

    @property
    def R_base_in_world(self) -> np.ndarray:
        return self._R_base_in_world.copy()

    @property
    def apriltag_to_car_base_offset_mm(self) -> np.ndarray:
        return self._apriltag_to_car_base_offset_mm.copy()

    def extract_joint_vector(
        self,
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
    ) -> np.ndarray:
        name_to_pos = {str(name): float(pos) for name, pos in zip(joint_names, joint_positions)}
        values = []
        for name in self._expected_joint_names:
            if name not in name_to_pos:
                raise KeyError(f"missing required joint in JointState: {name}")
            values.append(name_to_pos[name])
        return np.array(values, dtype=np.float64)

    def extract_joint_vector_from_snapshot(self, snapshot: Mapping[str, object]) -> np.ndarray:
        names = snapshot.get("name")
        positions = snapshot.get("position")
        if not isinstance(names, Sequence) or not isinstance(positions, Sequence):
            raise KeyError("joint snapshot missing name/position fields")
        return self.extract_joint_vector(names, positions)

    def forward_base(self, joint_values_rad: Sequence[float]) -> np.ndarray:
        q = np.array(joint_values_rad[: self._joint_count], dtype=np.float64)
        q_used = q + self._joint_offsets
        T = np.eye(4, dtype=np.float64)
        for axis, theta in zip(self._axes, q_used):
            T = T @ self._exp_revolute(axis, float(theta))
        ph = T @ np.array(
            [self._home_point_base[0], self._home_point_base[1], self._home_point_base[2], 1.0],
            dtype=np.float64,
        )
        return ph[:3]

    def forward_world(self, joint_values_rad: Sequence[float]) -> np.ndarray:
        point_base = self.forward_base(joint_values_rad)
        return self._R_base_in_world @ point_base + self._t_base_in_world

    def forward(self, joint_values_rad: Sequence[float]) -> PoeForwardResult:
        q = np.array(joint_values_rad[: self._joint_count], dtype=np.float64)
        point_base = self.forward_base(q)
        point_world = self._R_base_in_world @ point_base + self._t_base_in_world
        return PoeForwardResult(
            joint_values_rad=q.tolist(),
            joint_values_used_rad=(q + self._joint_offsets).tolist(),
            point_base_mm=point_base,
            point_world_mm=point_world,
        )
