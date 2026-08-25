"""Stable opponent-contact anchor from outgoing RK ``ball_world`` points.

The solver keeps the original strict 6/7/8-prefix contract and adds one
bounded physical-consensus acceptance mode.  It is record-only and never
feeds robot control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_GRAVITY_MPS2 = 9.8
_CONTACT_MODEL = "rk_ball_z_crossing_fixed_height"
_MAX_WINDOW_SHIFT = 4


@dataclass(frozen=True)
class BallTrajectory:
    """Outgoing RK ``ball_world`` fit on the RK clock and world frame."""

    reference_t_rk: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    fit_rms_m: float

    def position_at(self, t_rk: float) -> tuple[float, float, float]:
        dt = float(t_rk) - self.reference_t_rk
        return (
            self.x_m + self.vx_mps * dt,
            self.y_m + self.vy_mps * dt,
            self.z_m + self.vz_mps * dt - 0.5 * _GRAVITY_MPS2 * dt * dt,
        )


@dataclass(frozen=True)
class RacketContactEstimate:
    valid: bool
    failure_reason: str
    acceptance_mode: str
    contact_anchor_t_rk: float | None
    contact_anchor_world_m: tuple[float, float, float] | None
    contact_model: str
    contact_height_m: float
    prefix_anchor_t_rk: tuple[float, ...]
    prefix_spread_s: float | None
    contact_point_spread_m: float | None
    ball_fit_rms_m: float | None
    first_observation_lead_s: float | None
    approach_speed_mps: float | None
    n_points: int
    window_shift: int
    trajectory: BallTrajectory | None


class StableRacketContactSolver:
    """Accept a strict fit or a bounded physical 6/7/8-prefix consensus."""

    def __init__(
        self,
        *,
        contact_height_m: float = 1.05,
        min_approach_mps: float = 3.0,
        max_gap_s: float = 0.25,
        cooldown_s: float = 1.5,
        max_prefix_spread_s: float = 0.015,
        max_ball_fit_rms_m: float = 0.15,
        max_step_speed_mps: float = 40.0,
        consensus_max_prefix_spread_s: float = 0.060,
        consensus_max_contact_spread_m: float = 0.15,
        consensus_max_ball_fit_rms_m: float = 0.40,
        contact_x_range_m: tuple[float, float] = (-3.0, 3.0),
        contact_y_range_m: tuple[float, float] = (13.5, 20.5),
        contact_reach_margin_m: float = 0.25,
    ) -> None:
        self._contact_height_m = float(contact_height_m)
        self._min_approach_mps = float(min_approach_mps)
        self._max_gap_s = float(max_gap_s)
        self._cooldown_s = float(cooldown_s)
        self._max_prefix_spread_s = float(max_prefix_spread_s)
        self._max_ball_fit_rms_m = float(max_ball_fit_rms_m)
        self._max_step_speed_mps = float(max_step_speed_mps)
        self._consensus_max_prefix_spread_s = float(
            consensus_max_prefix_spread_s
        )
        self._consensus_max_contact_spread_m = float(
            consensus_max_contact_spread_m
        )
        self._consensus_max_ball_fit_rms_m = float(
            consensus_max_ball_fit_rms_m
        )
        self._contact_x_range_m = tuple(float(value) for value in contact_x_range_m)
        self._contact_y_range_m = tuple(float(value) for value in contact_y_range_m)
        self._contact_reach_margin_m = float(contact_reach_margin_m)
        if self._contact_reach_margin_m < 0.0:
            raise ValueError("contact_reach_margin_m must be nonnegative")
        self._leg: list[tuple[float, float, float, float]] = []
        self._resolved = False
        self._last_resolve_t_rk = float("-inf")
        self._last_contact_anchor_t_rk = float("-inf")
        self._pending_rejection: RacketContactEstimate | None = None

    def add(self, payload: dict) -> RacketContactEstimate | None:
        try:
            point = tuple(float(payload[key]) for key in ("t", "x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in point):
            return None

        t_rk, x_m, y_m, z_m = point
        boundary_rejection = None
        boundary_resolve_t_rk = None
        if self._leg:
            last_t, last_x, last_y, last_z = self._leg[-1]
            dt = t_rk - last_t
            approaching = dt > 0.0 and (last_y - y_m) / dt > 0.0
            step_speed_mps = (
                math.dist((last_x, last_y, last_z), (x_m, y_m, z_m)) / dt
                if dt > 0.0
                else float("inf")
            )
            if (
                dt <= 0.0
                or dt > self._max_gap_s
                or not approaching
                or step_speed_mps > self._max_step_speed_mps
            ):
                boundary_rejection = self._pending_rejection
                boundary_resolve_t_rk = last_t
                self._leg = []
                self._resolved = False
                self._pending_rejection = None

        self._leg.append(point)
        if boundary_rejection is not None:
            self._last_resolve_t_rk = float(boundary_resolve_t_rk)
            return boundary_rejection
        if len(self._leg) > 40:
            self._leg.pop(0)
        if self._resolved or len(self._leg) < 8:
            return None
        if t_rk - self._last_resolve_t_rk < self._cooldown_s:
            return None

        first_shift = (
            0
            if self._pending_rejection is None
            else self._pending_rejection.window_shift + 1
        )
        last_shift = min(_MAX_WINDOW_SHIFT, len(self._leg) - 8)
        for window_shift in range(first_shift, last_shift + 1):
            result = self._evaluate_window(
                self._leg[window_shift:window_shift + 8],
                window_shift,
            )
            if result.valid:
                return self._finalize(result, t_rk)
            self._pending_rejection = result
        if last_shift == _MAX_WINDOW_SHIFT:
            return self._finalize(self._pending_rejection, t_rk)
        return None

    def finish(self) -> RacketContactEstimate | None:
        """Finalize one pending failed leg at end of the RK stream."""

        if self._pending_rejection is None:
            return None
        return self._finalize(self._pending_rejection, self._leg[-1][0])

    def _finalize(
        self,
        result: RacketContactEstimate,
        resolve_t_rk: float,
    ) -> RacketContactEstimate | None:
        self._pending_rejection = None
        self._resolved = True
        self._last_resolve_t_rk = float(resolve_t_rk)
        if result.valid:
            anchor = float(result.contact_anchor_t_rk)
            if abs(anchor - self._last_contact_anchor_t_rk) < self._cooldown_s:
                return None
            self._last_contact_anchor_t_rk = anchor
        return result

    def _evaluate_window(
        self,
        points: list[tuple[float, float, float, float]],
        window_shift: int,
    ) -> RacketContactEstimate:
        prefix_results = [self._fit_prefix(points[:count]) for count in (6, 7, 8)]
        if any(result is None for result in prefix_results):
            return self._rejected(
                "invalid_ball_contact_prefix",
                window_shift=window_shift,
                prefix_anchor_t_rk=(),
                prefix_spread_s=None,
                contact_point_spread_m=None,
                ball_fit_rms_m=None,
            )

        fits = [result for result in prefix_results if result is not None]
        contact_times = np.array([fit[0] for fit in fits], dtype=np.float64)
        contact_points = np.array(
            [fit[1].position_at(fit[0]) for fit in fits],
            dtype=np.float64,
        )
        time_spread_s = float(np.ptp(contact_times))
        point_spread_m = float(
            max(
                np.linalg.norm(left - right)
                for left in contact_points
                for right in contact_points
            )
        )
        max_rms_m = float(max(fit[1].fit_rms_m for fit in fits))

        reach_margin_m = self._contact_reach_margin_m
        x_lo, x_hi = (
            self._contact_x_range_m[0] - reach_margin_m,
            self._contact_x_range_m[1] + reach_margin_m,
        )
        y_lo, y_hi = (
            self._contact_y_range_m[0] - reach_margin_m,
            self._contact_y_range_m[1] + reach_margin_m,
        )
        points_in_reach_volume = all(
            x_lo <= point[0] <= x_hi and y_lo <= point[1] <= y_hi
            for point in contact_points
        )
        strict_quality = (
            time_spread_s <= self._max_prefix_spread_s
            and max_rms_m <= self._max_ball_fit_rms_m
        )
        strict = strict_quality and points_in_reach_volume
        physical_consensus_quality = (
            time_spread_s <= self._consensus_max_prefix_spread_s
            and point_spread_m <= self._consensus_max_contact_spread_m
            and max_rms_m <= self._consensus_max_ball_fit_rms_m
        )
        physical_consensus = physical_consensus_quality and points_in_reach_volume
        if not strict and not physical_consensus:
            failure_reason = (
                "contact_outside_reach_volume"
                if (
                    not points_in_reach_volume
                    and (strict_quality or physical_consensus_quality)
                )
                else (
                    "unstable_ball_contact_time"
                    if time_spread_s > self._max_prefix_spread_s
                    else "ball_contact_fit_residual"
                )
            )
            return self._rejected(
                failure_reason,
                window_shift=window_shift,
                prefix_anchor_t_rk=tuple(float(value) for value in contact_times),
                prefix_spread_s=time_spread_s,
                contact_point_spread_m=point_spread_m,
                ball_fit_rms_m=max_rms_m,
            )

        contact_anchor_t_rk = float(np.median(contact_times))
        contact_x, contact_y, _ = np.median(contact_points, axis=0)
        trajectory = fits[-1][1]
        return RacketContactEstimate(
            valid=True,
            failure_reason="",
            acceptance_mode="strict" if strict else "physical_consensus",
            contact_anchor_t_rk=contact_anchor_t_rk,
            contact_anchor_world_m=(
                float(contact_x),
                float(contact_y),
                self._contact_height_m,
            ),
            contact_model=_CONTACT_MODEL,
            contact_height_m=self._contact_height_m,
            prefix_anchor_t_rk=tuple(float(value) for value in contact_times),
            prefix_spread_s=time_spread_s,
            contact_point_spread_m=point_spread_m,
            ball_fit_rms_m=max_rms_m,
            first_observation_lead_s=float(points[0][0] - contact_anchor_t_rk),
            approach_speed_mps=float(-trajectory.vy_mps),
            n_points=8,
            window_shift=window_shift,
            trajectory=trajectory,
        )

    def _rejected(
        self,
        failure_reason: str,
        *,
        window_shift: int,
        prefix_anchor_t_rk: tuple[float, ...],
        prefix_spread_s: float | None,
        contact_point_spread_m: float | None,
        ball_fit_rms_m: float | None,
    ) -> RacketContactEstimate:
        return RacketContactEstimate(
            valid=False,
            failure_reason=failure_reason,
            acceptance_mode="",
            contact_anchor_t_rk=None,
            contact_anchor_world_m=None,
            contact_model=_CONTACT_MODEL,
            contact_height_m=self._contact_height_m,
            prefix_anchor_t_rk=prefix_anchor_t_rk,
            prefix_spread_s=prefix_spread_s,
            contact_point_spread_m=contact_point_spread_m,
            ball_fit_rms_m=ball_fit_rms_m,
            first_observation_lead_s=None,
            approach_speed_mps=None,
            n_points=8,
            window_shift=window_shift,
            trajectory=None,
        )

    def _fit_prefix(
        self,
        points: list[tuple[float, float, float, float]],
    ) -> tuple[float, BallTrajectory] | None:
        times = np.array([point[0] for point in points], dtype=np.float64)
        xs = np.array([point[1] for point in points], dtype=np.float64)
        ys = np.array([point[2] for point in points], dtype=np.float64)
        zs = np.array([point[3] for point in points], dtype=np.float64)
        span = float(times[-1] - times[0])
        if span <= 0.0 or (ys[0] - ys[-1]) / span < self._min_approach_mps:
            return None

        mean_t = float(times.mean())
        relative_t = times - mean_t
        design = np.c_[np.ones(len(times)), relative_t]
        x_coeff = np.linalg.lstsq(design, xs, rcond=None)[0]
        y_coeff = np.linalg.lstsq(design, ys, rcond=None)[0]
        z_coeff = np.linalg.lstsq(
            design,
            zs + 0.5 * _GRAVITY_MPS2 * relative_t**2,
            rcond=None,
        )[0]
        z0, vz0 = (float(value) for value in z_coeff)
        discriminant = vz0 * vz0 + 2.0 * _GRAVITY_MPS2 * (
            z0 - self._contact_height_m
        )
        if discriminant < 0.0:
            return None
        contact_t_rk = mean_t + (vz0 - math.sqrt(discriminant)) / _GRAVITY_MPS2
        lead_s = float(times[0] - contact_t_rk)
        if not (-0.05 < lead_s < 0.80):
            return None

        fitted_x = design @ x_coeff
        fitted_y = design @ y_coeff
        fitted_z = design @ z_coeff - 0.5 * _GRAVITY_MPS2 * relative_t**2
        residuals = np.sqrt(
            (fitted_x - xs) ** 2
            + (fitted_y - ys) ** 2
            + (fitted_z - zs) ** 2
        )
        trajectory = BallTrajectory(
            reference_t_rk=mean_t,
            x_m=float(x_coeff[0]),
            y_m=float(y_coeff[0]),
            z_m=z0,
            vx_mps=float(x_coeff[1]),
            vy_mps=float(y_coeff[1]),
            vz_mps=vz0,
            fit_rms_m=float(np.sqrt(np.mean(residuals**2))),
        )
        return float(contact_t_rk), trajectory


__all__ = [
    "BallTrajectory",
    "RacketContactEstimate",
    "StableRacketContactSolver",
]
