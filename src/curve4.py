# -*- coding: utf-8 -*-
"""
网球轨迹追踪与击球点预测模块 (curve4) — drag-aware 版本。

与 curve3 完全相同的公开接口（BallObservation / PredictHitPos /
TrackerState / TrackerResult / Curve4Tracker）。唯一区别：球轨迹
物理模型由 "线性 xy + 抛物线 z" 改为 "6-DOF 带二次型空气阻力的
ODE"。

物理模型：
  a_x = -k · |v| · v_x
  a_y = -k · |v| · v_y
  a_z = -g - k · |v| · v_z

其中 k = 0.030 (1/m) 是经 23 个 clean throws 全局优化得到的经验值
（见 data/_analysis/stage1_z_accuracy/drag_k_sweep.py）。k 吸收了
纯 drag + 典型网球上旋的 Magnus 贡献，实测比教科书 k=0.019 好。

拟合：
  6-DOF LSQ on (x0, y0, z0, vx0, vy0, vz0) at t_ref，scipy
  least_squares + odeint 数值积分。初值用 curve3 的线性/二次
  最小二乘作为种子。

预测：
  Stage 0 / Stage 1 接口与 curve3 一致。二次反弹（stage0）与
  y=target_y 求解（stage1）均用 drag ODE 正向积分 + 数值求根。

坐标系约定：
  z 轴向上为正 (m)，地面 z=0，y 为球前进方向，x 为侧向。
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from scipy.integrate import odeint
from scipy.optimize import least_squares

from .cv_linalg import solve_least_squares

GRAVITY = 9.8  # m/s²
K_DRAG = 0.024  # 1/m，在 56 clean obs-segmented throws × 6 time-buckets (pre 0.5/0.75/1s
#               # + post 0.3/0.4/0.5s) 上最小化 drag ODE 拟合 RMSE 得出，median 11.6mm
_ODE_RTOL = 1e-7
_ODE_ATOL = 1e-9
_ODE_MXSTEP = 500


def _drag_rhs(state, t, k, g):
    """6-DOF quadratic-drag RHS for scipy.odeint."""
    x, y, z, vx, vy, vz = state
    v = math.sqrt(vx * vx + vy * vy + vz * vz)
    return [vx, vy, vz, -k * v * vx, -k * v * vy, -g - k * v * vz]


# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class BallObservation:
    """单次观测到的球位置。单位: 米 / 秒。"""
    x: float
    y: float
    z: float
    t: float


@dataclass
class PredictHitPos:
    """预测击球位置。"""
    x: float
    y: float
    z: float
    stage: int
    ct: float
    ht: float


class TrackerState(str, Enum):
    """
    Curve4 追踪器状态，供外部管线（如 run_tracker.py）显示。

    状态转移流程（一次抛球周期内）：
      IDLE → TRACKING_S0 → IN_LANDING → TRACKING_S1 → DONE → IDLE
                                                        ↑
                          超时/位置跳变也会直接触发 ──────┘
    """
    IDLE = "idle"
    TRACKING_S0 = "tracking_s0"
    IN_LANDING = "in_landing"
    TRACKING_S1 = "tracking_s1"
    DONE = "done"


@dataclass
class TrackerResult:
    """Curve4Tracker.update() 的返回值。"""
    prediction: Optional[PredictHitPos]
    state: TrackerState


# ── 轨迹拟合（drag ODE） ──────────────────────────────────────────────────


@dataclass
class DragFit:
    """
    drag-aware 轨迹拟合结果。

    state0: 长度 6 的初始状态 [x, y, z, vx, vy, vz]，对应时刻 t_ref。
    k:      drag 系数 (1/m)。
    """
    state0: np.ndarray
    t_ref: float
    k: float

    def _integrate(self, dt_array: np.ndarray) -> np.ndarray:
        """Integrate the drag ODE at the given dt offsets from t_ref.
        dt_array must be monotonically non-decreasing and start at >= 0.
        Returns (N, 6) array of [x, y, z, vx, vy, vz]."""
        if dt_array[0] != 0.0:
            ts = np.concatenate(([0.0], dt_array))
            sol = odeint(_drag_rhs, self.state0, ts, args=(self.k, GRAVITY),
                         mxstep=_ODE_MXSTEP, rtol=_ODE_RTOL, atol=_ODE_ATOL)
            return sol[1:]
        return odeint(_drag_rhs, self.state0, dt_array,
                      args=(self.k, GRAVITY),
                      mxstep=_ODE_MXSTEP, rtol=_ODE_RTOL, atol=_ODE_ATOL)

    def predict(self, t: float) -> Tuple[float, float, float]:
        """返回 t 时刻的 (x, y, z)。"""
        dt = t - self.t_ref
        if dt < 0:
            dt = 0.0
        if dt == 0.0:
            return float(self.state0[0]), float(self.state0[1]), float(self.state0[2])
        sol = self._integrate(np.array([0.0, dt]))
        return float(sol[-1, 0]), float(sol[-1, 1]), float(sol[-1, 2])

    def velocity_at(self, t: float) -> Tuple[float, float, float]:
        """返回 t 时刻的 (vx, vy, vz)。"""
        dt = t - self.t_ref
        if dt < 0:
            dt = 0.0
        if dt == 0.0:
            return float(self.state0[3]), float(self.state0[4]), float(self.state0[5])
        sol = self._integrate(np.array([0.0, dt]))
        return float(sol[-1, 3]), float(sol[-1, 4]), float(sol[-1, 5])

    def _find_crossing(
        self,
        axis: int,
        target: float,
        after_time: Optional[float],
        latest: bool,
        search_seconds: float = 2.0,
        n_pts: int = 201,
    ) -> Optional[float]:
        """Integrate forward and find t where state[axis] == target.
        axis: 0=x, 1=y, 2=z.
        Returns absolute time in same frame as t_ref."""
        min_t = after_time if after_time is not None else self.t_ref
        dt_min = max(0.0, min_t - self.t_ref)
        dt_max = dt_min + search_seconds
        t_grid = np.linspace(dt_min, dt_max, n_pts)
        sol = self._integrate(t_grid)
        vals = sol[:, axis]
        delta = vals - target
        sign = np.sign(delta)
        # 处理恰好等于 target 的情况
        sign[sign == 0] = 1
        cross = np.where(np.diff(sign) != 0)[0]
        if len(cross) == 0:
            return None
        idx = cross[-1] if latest else cross[0]
        denom = vals[idx + 1] - vals[idx]
        if abs(denom) < 1e-12:
            dt_hit = t_grid[idx]
        else:
            frac = (target - vals[idx]) / denom
            dt_hit = t_grid[idx] + frac * (t_grid[idx + 1] - t_grid[idx])
        return float(dt_hit + self.t_ref)

    def solve_t_for_z(self, target_z: float,
                      after_time: Optional[float] = None,
                      latest: bool = False) -> Optional[float]:
        """求解 z(t) = target_z。latest=True 取最晚解（下降阶段）。"""
        return self._find_crossing(2, target_z, after_time, latest)

    def solve_t_for_y(self, target_y: float,
                      after_time: Optional[float] = None) -> Optional[float]:
        """求解 y(t) = target_y，取最晚解（与 curve3 语义一致：
        y 在 drag 作用下单调接近渐近值，单次穿越即可）。"""
        # y 在 drag 下仍然单调（同向速度 drag 永远减速不反向），
        # 所以 first/latest 等价；用 latest=True 与 curve3 post-bounce
        # 行为一致（晚解）。
        return self._find_crossing(1, target_y, after_time, latest=True)


def fit_curve(observations: List[BallObservation],
              k: float = K_DRAG,
              weight_ratio: float = 1.0) -> Optional[DragFit]:
    """
    6-DOF drag-aware LSQ 拟合。

    初值用线性（xy）/ 二次（z）最小二乘作为 warm start，然后用
    scipy.least_squares(method='lm') 最小化 ODE 积分残差。

    weight_ratio: 时间几何权重系数。观测按时间升序，权重
        w_i = weight_ratio^i 再归一化到最新点 = 1。1.0 等价于
        等权；>1.0 时 bounce-附近观测权重更大，能缓解 k_drag
        misspecification 在长窗口下对 v_b 的偏置。
    """
    if len(observations) < 3:
        return None

    t_ref = observations[0].t
    ts = np.array([obs.t - t_ref for obs in observations], dtype=np.float64)
    xs = np.array([obs.x for obs in observations], dtype=np.float64)
    ys = np.array([obs.y for obs in observations], dtype=np.float64)
    zs = np.array([obs.z for obs in observations], dtype=np.float64)
    n = len(ts)

    weighted = weight_ratio != 1.0
    if weighted:
        w = weight_ratio ** np.arange(n, dtype=np.float64)
        w /= w[-1]
        sw = np.sqrt(w)
    else:
        sw = None

    A1 = np.column_stack([np.ones_like(ts), ts])
    A2 = np.column_stack([np.ones_like(ts), ts, ts * ts])
    if weighted:
        A1w = A1 * sw[:, None]
        A2w = A2 * sw[:, None]
        ax, bx = solve_least_squares(A1w, xs * sw)
        ay, by = solve_least_squares(A1w, ys * sw)
        az, bz, _cz = solve_least_squares(A2w, zs * sw)
    else:
        ax, bx = solve_least_squares(A1, xs)
        ay, by = solve_least_squares(A1, ys)
        az, bz, _cz = solve_least_squares(A2, zs)
    state0_init = np.array([ax, ay, az, bx, by, bz], dtype=np.float64)

    def residuals(state0: np.ndarray) -> np.ndarray:
        try:
            sol = odeint(_drag_rhs, state0, ts, args=(k, GRAVITY),
                         mxstep=_ODE_MXSTEP, rtol=_ODE_RTOL, atol=_ODE_ATOL)
        except Exception:
            return np.full(3 * n, 1e6)
        rx = sol[:, 0] - xs
        ry = sol[:, 1] - ys
        rz = sol[:, 2] - zs
        if weighted:
            rx = rx * sw
            ry = ry * sw
            rz = rz * sw
        return np.concatenate([rx, ry, rz])

    try:
        res = least_squares(residuals, state0_init, method='lm', max_nfev=200)
    except Exception:
        return None
    if not res.success and res.status <= 0:
        return None
    return DragFit(state0=np.asarray(res.x, dtype=np.float64),
                   t_ref=t_ref, k=k)


# ── 主追踪器 ──────────────────────────────────────────────────────────────


class Curve4Tracker:
    """
    网球轨迹追踪器（drag-aware 版本）。

    接口与 Curve3Tracker 完全一致，唯一区别是内部用 6-DOF drag ODE
    拟合代替线性+二次拟合。适用于需要更准 stage-1 z 预测的场景。
    """

    def __init__(
        self,
        ideal_hit_z: float = 0.8,
        cor: float = 0.7914,
        cor_xy: float = 0.4466,
        ground_z: float = 0.0,
        min_points: int = 5,
        min_stage1_points: int = 5,
        fit_rmse_max: float = 0.4,
        reset_timeout: float = 0.5,
        motion_window_s: float = 0.2,
        motion_min_y: float = 0.5,
        land_skip_time: float = 0.05,
        k_drag: float = K_DRAG,
        weight_ratio: float = 1.0,
    ):
        """
        默认参数通过"纯物理拟合"在 56 clean obs-segmented throws × 6 time-
        buckets 上得出（见 memory/project_curve4_cor_fit.md）：

        - k_drag=0.024: 最小化 pre+post drag ODE 拟合 RMSE
        - cor=0.7914, cor_xy=0.4466: 固定 k 后对每球独立 pre/post fit 读
          |v_post/v_b| 的 median，std 分别 0.014 / 0.025

        不要用下游代理指标（|pred_z-0.8| 或 Δht）重拟 cor/cor_xy——会把
        cor_xy 推到 0.62+ 破坏 Δht 连续性（见 memory）。

        Args 与 Curve3Tracker 相同，额外两个：
            k_drag: 二次型空气阻力系数 (1/m)。
            weight_ratio: fit_curve 的时间几何权重（w_i = weight_ratio^i，
                最新观测权重最大，归一化到 newest=1）。1.0 = 等权（默认）。
                >1.0 让 bounce-附近观测权重更大，但如果用了就必须同时重
                拟 cor/cor_xy，否则会放大 stage-0 ht 偏置。
        """
        self.ideal_hit_z = ideal_hit_z
        self.cor = cor
        self.cor_xy = cor_xy
        self.ground_z = ground_z
        self.min_points = min_points
        self.min_stage1_points = min_stage1_points
        self.fit_rmse_max = fit_rmse_max
        self.reset_timeout = reset_timeout
        self.motion_window_s = motion_window_s
        self.motion_min_y = motion_min_y
        self.land_skip_time = land_skip_time
        self.k_drag = k_drag
        self.weight_ratio = weight_ratio

        self.predictions: List[PredictHitPos] = []
        self.reset_times: List[float] = []

        self._last_obs_time: Optional[float] = None
        self._reset_throw()

    def _reset_throw(self) -> None:
        self._obs: List[BallObservation] = []
        self._stage: int = 0
        self._bounce_index: Optional[int] = None
        self._bounce_time: Optional[float] = None
        self._curve0: Optional[DragFit] = None
        self._curve1: Optional[DragFit] = None
        self._predicted_land_time: Optional[float] = None
        self._t_land: Optional[float] = None
        self._t_land2: Optional[float] = None
        self._n_post_fit: int = 0
        self._last_s0_pred: Optional[PredictHitPos] = None
        self._post_bounce_max_z: float = 0.0
        self._state: TrackerState = TrackerState.IDLE
        self._pending_obs: List[BallObservation] = []
        self._motion_confirmed: bool = False
        self._jump_count: int = 0
        self._skip_obs: bool = False

    # ── 公开接口 ──

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def bounce_time(self) -> Optional[float]:
        return self._bounce_time

    @property
    def tracker_state(self) -> TrackerState:
        return self._state

    def update(self, obs: BallObservation) -> TrackerResult:
        self._skip_obs = False
        if self._check_reset(obs):
            self.reset_times.append(obs.t)
            self._reset_throw()

        self._last_obs_time = obs.t

        if self._skip_obs:
            return TrackerResult(prediction=None, state=self._state)

        if not self._motion_confirmed:
            self._pending_obs.append(obs)
            if self._check_motion():
                self._motion_confirmed = True
                self._flush_pending_to_obs()
            else:
                return TrackerResult(prediction=None, state=TrackerState.IDLE)
        else:
            if self._stage == 0 and self._predicted_land_time is not None:
                if abs(obs.t - self._predicted_land_time) < self.land_skip_time:
                    self._state = TrackerState.IN_LANDING
                    return TrackerResult(prediction=None, state=self._state)
                if obs.t > self._predicted_land_time:
                    self._transition_to_s1()

            self._obs.append(obs)

        if self._stage == 0:
            if self._fit_s0():
                return TrackerResult(prediction=None, state=TrackerState.DONE)

        if self._stage == 1:
            self._update_post_bounce(obs)
            if self._detect_throw_complete():
                self.reset_times.append(obs.t)
                self._reset_throw()
                return TrackerResult(prediction=None, state=TrackerState.DONE)

            if self._fit_s1():
                return TrackerResult(prediction=None, state=TrackerState.DONE)

            if self._curve1 is not None:
                self._update_land2_time()
                if self._t_land2 is not None and obs.t >= self._t_land2:
                    self.reset_times.append(obs.t)
                    self._reset_throw()
                    return TrackerResult(prediction=None, state=TrackerState.DONE)

        if self._stage == 0:
            if len(self._obs) >= self.min_points and self._curve0 is not None:
                self._state = TrackerState.TRACKING_S0
            else:
                self._state = TrackerState.IDLE
        else:
            n_post = self._n_post_fit
            if n_post < self.min_stage1_points:
                self._state = TrackerState.IN_LANDING
            else:
                self._state = TrackerState.TRACKING_S1

        pred = self._predict(obs.t)
        if pred is not None:
            self.predictions.append(pred)
            if pred.stage == 0:
                self._last_s0_pred = pred

        return TrackerResult(prediction=pred, state=self._state)

    # ── 运动过滤 ──

    def _check_motion(self) -> bool:
        if len(self._pending_obs) < 2:
            return False
        latest = self._pending_obs[-1]
        for earlier in reversed(self._pending_obs[:-1]):
            dt = latest.t - earlier.t
            if dt <= 0:
                continue
            if dt > self.motion_window_s:
                break
            if latest.y - earlier.y <= -self.motion_min_y:
                return True
        return False

    def _flush_pending_to_obs(self) -> None:
        if not self._pending_obs:
            return
        motion_start = 0
        for k in range(1, len(self._pending_obs)):
            cur = self._pending_obs[k]
            for j in range(k - 1, -1, -1):
                dt = cur.t - self._pending_obs[j].t
                if dt <= 0:
                    continue
                if dt > self.motion_window_s:
                    break
                if cur.y - self._pending_obs[j].y <= -self.motion_min_y:
                    motion_start = j + 1
                    break
            if motion_start > 0:
                break
        self._obs.extend(self._pending_obs[motion_start:])
        self._pending_obs.clear()

    # ── 重置检测 ──

    _VELOCITY_JUMP_THRESHOLD = 50.0
    _JUMP_RESET_COUNT = 3

    def _check_reset(self, obs: BallObservation) -> bool:
        if self._last_obs_time is not None:
            dt = obs.t - self._last_obs_time
            if dt > self.reset_timeout:
                self._jump_count = 0
                return True
        prev = self._obs[-1] if self._obs else (
            self._pending_obs[-1] if self._pending_obs else None)
        if prev is not None:
            dt = obs.t - prev.t
            if dt > 0:
                dist = math.sqrt((obs.x - prev.x) ** 2
                                 + (obs.y - prev.y) ** 2
                                 + (obs.z - prev.z) ** 2)
                if dist / dt > self._VELOCITY_JUMP_THRESHOLD:
                    self._jump_count += 1
                    if self._jump_count >= self._JUMP_RESET_COUNT:
                        self._jump_count = 0
                        return True
                    self._skip_obs = True
                    return False
                else:
                    self._jump_count = 0
        return False

    def _update_post_bounce(self, obs: BallObservation) -> None:
        if obs.z > self._post_bounce_max_z:
            self._post_bounce_max_z = obs.z

    def _detect_throw_complete(self) -> bool:
        if self._t_land is None:
            return False
        post = [o for o in self._obs
                if o.t > self._t_land + self.land_skip_time]
        if len(post) < 10:
            return False
        if self._post_bounce_max_z < 0.3:
            return False
        return self._obs[-1].z < 0.1

    def _update_land2_time(self) -> None:
        if self._curve1 is None or self._bounce_time is None:
            return
        t_land = self._curve1.solve_t_for_z(
            self.ground_z, after_time=self._bounce_time + 0.1)
        if t_land is not None:
            self._t_land2 = t_land

    # ── 拟合 ──

    def _compute_fit_rmse(
        self, curve: DragFit, obs: List[BallObservation],
    ) -> Tuple[float, float, float]:
        """用 drag ODE 一次性积分所有 obs 时刻，计算 RMSE。"""
        if not obs:
            return 0.0, 0.0, 0.0
        ts = np.array([o.t - curve.t_ref for o in obs], dtype=np.float64)
        xs = np.array([o.x for o in obs], dtype=np.float64)
        ys = np.array([o.y for o in obs], dtype=np.float64)
        zs = np.array([o.z for o in obs], dtype=np.float64)
        try:
            sol = curve._integrate(ts)
        except Exception:
            return float('inf'), float('inf'), float('inf')
        rmse_x = float(np.sqrt(np.mean((sol[:, 0] - xs) ** 2)))
        rmse_y = float(np.sqrt(np.mean((sol[:, 1] - ys) ** 2)))
        rmse_z = float(np.sqrt(np.mean((sol[:, 2] - zs) ** 2)))
        return rmse_x, rmse_y, rmse_z

    def _fit_s0(self) -> bool:
        n = len(self._obs)
        if n < 3:
            return False

        curve = fit_curve(self._obs, k=self.k_drag,
                          weight_ratio=self.weight_ratio)
        if curve is None:
            return False

        self._curve0 = curve

        t_land = curve.solve_t_for_z(
            self.ground_z, after_time=curve.t_ref, latest=True)
        if t_land is not None:
            self._predicted_land_time = t_land

        if n >= self.min_points:
            rx, ry, rz = self._compute_fit_rmse(curve, self._obs)
            if max(rx, ry, rz) > self.fit_rmse_max:
                self.reset_times.append(self._obs[-1].t)
                self._reset_throw()
                return True

        return False

    def _transition_to_s1(self) -> None:
        self._stage = 1
        self._bounce_time = self._predicted_land_time
        self._t_land = self._predicted_land_time
        self._bounce_index = len(self._obs)

    def _fit_s1(self) -> bool:
        if self._t_land is None:
            return False
        t_hi = self._t_land + self.land_skip_time
        post = [o for o in self._obs if o.t > t_hi]
        self._n_post_fit = len(post)
        if len(post) < 3:
            return False
        curve = fit_curve(post, k=self.k_drag,
                          weight_ratio=self.weight_ratio)
        if curve is None:
            return False
        self._curve1 = curve

        if len(post) >= self.min_stage1_points:
            rx, ry, rz = self._compute_fit_rmse(curve, post)
            if max(rx, ry, rz) > self.fit_rmse_max:
                self.reset_times.append(self._obs[-1].t)
                self._reset_throw()
                return True
        return False

    # ── 击球预测 ──

    def _predict(self, ct: float) -> Optional[PredictHitPos]:
        if self._stage == 0:
            return self._predict_stage0(ct)
        return self._predict_stage1(ct)

    def _predict_stage0(self, ct: float) -> Optional[PredictHitPos]:
        """
        Stage 0: 用 drag ODE 积分 → 求落地点 → COR 反弹 →
        再用 drag ODE 正向积分反弹后轨迹，找 z=ideal_hit_z 最晚穿越。
        """
        curve = self._curve0
        if curve is None:
            return None

        t_bounce = curve.solve_t_for_z(self.ground_z, after_time=ct)
        if t_bounce is None:
            t_bounce = curve.solve_t_for_z(self.ground_z,
                                           after_time=curve.t_ref)
            if t_bounce is None or t_bounce < ct - 0.5:
                return None

        x_b, y_b, _ = curve.predict(t_bounce)
        vx_b, vy_b, vz_b = curve.velocity_at(t_bounce)
        vz_post = -self.cor * vz_b
        vx_post = self.cor_xy * vx_b
        vy_post = self.cor_xy * vy_b

        # 反弹后仍用 drag ODE 积分，不再退化为纯重力抛物线
        post_state = np.array(
            [x_b, y_b, self.ground_z, vx_post, vy_post, vz_post],
            dtype=np.float64,
        )
        post_fit = DragFit(state0=post_state, t_ref=t_bounce, k=self.k_drag)

        t_hit = post_fit.solve_t_for_z(
            self.ideal_hit_z, after_time=t_bounce, latest=True)
        if t_hit is None:
            return None

        x_hit, y_hit, _ = post_fit.predict(t_hit)
        return PredictHitPos(
            x=x_hit, y=y_hit, z=self.ideal_hit_z,
            stage=0, ct=ct, ht=t_hit,
        )

    def _predict_stage1(self, ct: float) -> Optional[PredictHitPos]:
        """
        Stage 1: 用 drag ODE 求反弹后 y(t) = target_y，取该时刻 (x, z)。
        """
        if self._n_post_fit < self.min_stage1_points:
            return None

        curve = self._curve1
        if curve is None:
            return None

        if self._last_s0_pred is None:
            return None

        target_y = self._last_s0_pred.y

        t_hit = curve.solve_t_for_y(target_y, after_time=curve.t_ref)
        if t_hit is None:
            return None

        x_hit, _, z_hit = curve.predict(t_hit)
        return PredictHitPos(
            x=x_hit, y=target_y, z=z_hit,
            stage=1, ct=ct, ht=t_hit,
        )
