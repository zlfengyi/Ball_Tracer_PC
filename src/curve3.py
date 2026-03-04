# -*- coding: utf-8 -*-
"""
网球轨迹追踪与击球点预测模块 (curve3)。

物理模型：
  - 空中阶段：x(t), y(t) 线性，z(t) 抛物线（重力加速度）
  - 落地反弹：z 方向速度乘以恢复系数 (COR) 并反向
  - 两阶段拟合：
      stage 0 (落地前) → 预测反弹后下降阶段到达 ideal_hit_z 的 (x, y, ht)
      stage 1 (落地后) → 拟合反弹后曲线 → 在小车已到达的 y 位置求 (x, z, ht)

坐标系约定：
  z 轴向上为正 (mm)，地面 z=0，y 为球前进方向，x 为侧向。

══════════════════════════════════════════════════════════════════════
Stage 0 / Stage 1 预测逻辑说明（对应 DEVELOP_LIST 步骤 4、5）
══════════════════════════════════════════════════════════════════════

Stage 0（落地前）：
  拟合空中轨迹 → 推算落地点 → 应用 COR 计算反弹后速度 →
  解析求球在下降阶段到达 ideal_hit_z 的时刻 ht 和位置 (x, y)。
  输出: PredictHitPos(x, y, z=ideal_hit_z, stage=0, ct, ht)
  小车根据此预测持续向 (x, y) 移动。

Stage 1（落地后）：
  球已落地反弹，小车此时已到达（或正在到达）stage 0 最后一次预测
  给出的 y 坐标，且小车不再改变 y 方向的运动。因此 stage 1 需要
  回答的问题是：

    "球到达小车所在的 y 位置时，x 和 z 分别是多少？"

  具体计算：
    1. target_y = 最后一次 stage 0 预测的 y（即小车停留的 y 位置）
    2. 用 stage 1 拟合曲线求解 y(t) = target_y → 得到 t_hit
    3. 在 t_hit 时刻求 x(t_hit) 和 z(t_hit)
    4. 输出 PredictHitPos(x=x(t_hit), y=target_y, z=z(t_hit),
                          stage=1, ct, ht=t_hit)

  其中 z(t_hit) 即 DEVELOP_LIST 中的 "stage1z_in_stage0_car_loc"，
  它与 ideal_hit_z 的差值为 "bounce_z_error"，理想应 < 300mm。
  x(t_hit) 即 "stage1x_in_stage0_car_loc"。

自动重置机制：
  tracker 持续监控输入。当一次抛球结束后（检测到超时、位置跳变、
  或球在反弹后回到地面），自动重置内部状态，准备接收下一次抛球。
  全局的 predictions 列表跨抛球累积，不会被重置。
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

GRAVITY = 9800.0  # mm/s²


# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class BallObservation:
    """单次观测到的球位置。"""
    x: float  # mm
    y: float  # mm
    z: float  # mm
    t: float  # 秒


@dataclass
class PredictHitPos:
    """预测击球位置。"""
    x: float      # 击球位置 x (mm)
    y: float      # 击球位置 y (mm)
    z: float      # 击球位置 z (mm)
    stage: int    # 0 = 落地前预测, 1 = 落地后预测
    ct: float     # 给出预测时的当前时刻 (s)
    ht: float     # 预测击球时刻 (s)


class TrackerState(str, Enum):
    """
    Curve3 追踪器状态，供外部管线（如 run_tracker.py）显示。

    状态转移流程（一次抛球周期内）：
      IDLE → TRACKING_S0 → IN_LANDING → TRACKING_S1 → DONE → IDLE
                                                        ↑
                          超时/位置跳变也会直接触发 ──────┘

    IDLE:          无数据或刚重置，等待新抛球（观测点不足以拟合）
    TRACKING_S0:   Stage 0 追踪中，输出 predict-hit-pos(stage=0)
    IN_LANDING:    球已落地反弹，但反弹后采样点不足 min_stage1_points，
                   此阶段不输出预测，仅告知外部"球正在落地/反弹中"
    TRACKING_S1:   Stage 1 追踪中，输出 predict-hit-pos(stage=1)
    DONE:          本次抛球结束（球二次落地 / 超时 / 位置跳变），即将重置
    """
    IDLE = "idle"
    TRACKING_S0 = "tracking_s0"
    IN_LANDING = "in_landing"
    TRACKING_S1 = "tracking_s1"
    DONE = "done"


@dataclass
class TrackerResult:
    """
    Curve3Tracker.update() 的返回值。

    将预测结果和追踪器状态打包在一起返回，避免调用者需要
    额外查询 tracker_state（因为 DONE 状态在 update 返回后
    已被重置为 IDLE，直接返回才能让调用者捕获到 DONE）。
    """
    prediction: Optional[PredictHitPos]  # 击球预测，可能为 None
    state: TrackerState                  # 当前帧的追踪状态


# ── 曲线拟合 ──────────────────────────────────────────────────────────────


@dataclass
class FittedCurve:
    """
    拟合的抛物线参数。

    x(t) = ax + bx * dt      (线性)
    y(t) = ay + by * dt      (线性)
    z(t) = az + bz * dt + cz * dt²  (二次)
    其中 dt = t - t_ref
    """
    ax: float
    bx: float
    ay: float
    by: float
    az: float
    bz: float
    cz: float
    t_ref: float

    def predict(self, t: float) -> Tuple[float, float, float]:
        dt = t - self.t_ref
        x = self.ax + self.bx * dt
        y = self.ay + self.by * dt
        z = self.az + self.bz * dt + self.cz * dt * dt
        return x, y, z

    def velocity_at(self, t: float) -> Tuple[float, float, float]:
        dt = t - self.t_ref
        return self.bx, self.by, self.bz + 2.0 * self.cz * dt

    def solve_t_for_z(self, target_z: float,
                      after_time: Optional[float] = None,
                      latest: bool = False) -> Optional[float]:
        """求解 z(t) = target_z。latest=True 取最晚解（下降阶段）。"""
        a, b, c = self.cz, self.bz, self.az - target_z
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return None
            t = -c / b + self.t_ref
            min_t = after_time if after_time is not None else self.t_ref
            return t if t >= min_t - 1e-6 else None
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        t1 = (-b - sq) / (2.0 * a) + self.t_ref
        t2 = (-b + sq) / (2.0 * a) + self.t_ref
        min_t = after_time if after_time is not None else self.t_ref
        valid = sorted(t for t in [t1, t2] if t >= min_t - 1e-6)
        if not valid:
            return None
        return valid[-1] if latest else valid[0]

    def solve_t_for_y(self, target_y: float,
                      after_time: Optional[float] = None) -> Optional[float]:
        """
        求解 y(t) = target_y。y 是线性的，最多一个解。
        """
        if abs(self.by) < 1e-12:
            return None
        dt = (target_y - self.ay) / self.by
        t = dt + self.t_ref
        min_t = after_time if after_time is not None else self.t_ref
        return t if t >= min_t - 1e-6 else None


def fit_curve(observations: List[BallObservation]) -> Optional[FittedCurve]:
    if len(observations) < 3:
        return None
    t_ref = observations[0].t
    ts = np.array([obs.t - t_ref for obs in observations])
    xs = np.array([obs.x for obs in observations])
    ys = np.array([obs.y for obs in observations])
    zs = np.array([obs.z for obs in observations])
    A_lin = np.column_stack([np.ones_like(ts), ts])
    ax, bx = np.linalg.lstsq(A_lin, xs, rcond=None)[0]
    ay, by = np.linalg.lstsq(A_lin, ys, rcond=None)[0]
    A_quad = np.column_stack([np.ones_like(ts), ts, ts ** 2])
    az, bz, cz = np.linalg.lstsq(A_quad, zs, rcond=None)[0]
    return FittedCurve(ax=ax, bx=bx, ay=ay, by=by,
                       az=az, bz=bz, cz=cz, t_ref=t_ref)


# ── 主追踪器 ──────────────────────────────────────────────────────────────


class Curve3Tracker:
    """
    网球轨迹追踪器 — 自动检测反弹、两阶段预测、自动重置。

    支持连续输入多次抛球的观测流，自动在每次抛球结束后重置，
    所有预测结果累积在 self.predictions 中。
    """

    def __init__(
        self,
        ideal_hit_z: float = 800.0,
        cor: float = 0.78,
        cor_xy: float = 0.42,
        ground_z: float = 0.0,
        min_points: int = 3,
        min_stage1_points: int = 5,
        bounce_z_threshold: float = 200.0,
        reset_timeout: float = 0.5,
        motion_window_s: float = 0.2,
        motion_min_y: float = 500.0,
        land_skip_time: float = 0.05,
    ):
        """
        Args:
            ideal_hit_z: 期望击球高度 (mm)。
            cor: 法向恢复系数（z 方向），反弹后 vz_post = -cor * vz_pre。
                实测三次抛球（能量守恒推算）：0.773, 0.774, 0.814，
                平均 0.787，取 0.78。
            cor_xy: 切向恢复系数（x/y 方向），反弹后水平速度衰减因子。
                球落地时摩擦和形变导致水平速度大幅降低，实测三次抛球：
                  vy_post / vy_pre = 0.424, 0.469, 0.432，平均 0.442，取 0.42。
                与法向 cor=0.78 不同，切向恢复系数显著更低。
            ground_z: 地面高度 (mm)。
            min_points: 拟合所需最少观测点。
            min_stage1_points: stage 1 至少多少点才输出预测。
            bounce_z_threshold: 反弹检测 z 阈值 (mm)。
            reset_timeout: 两帧间隔超过此值 (s) 则重置。
            motion_window_s: 运动检测滑动窗口时长 (s)。在此窗口内
                |Δy| 必须超过 motion_min_y 才认为球在飞行中。
                实测依据：30fps 时 0.2s ≈ 6 帧，足以区分静止/飞行。
            motion_min_y: 窗口内 |Δy| 最小阈值 (mm)。
                实测依据：人持球静止 ~1.4mm / 行走 ~74mm / 飞行 ~1340mm，
                500mm 阈值可安全分离静止+行走 vs 飞行。
            land_skip_time: 落地排除窗口半宽 (s)。用 S0 曲线预测落地
                时间 t_land，排除 [t_land - skip, t_land + skip] 内的
                观测，避免地面接触帧（形变/摩擦）污染 S0 和 S1 拟合。
                基于时间而非帧数，不受帧率影响。
        """
        self.ideal_hit_z = ideal_hit_z
        self.cor = cor
        self.cor_xy = cor_xy
        self.ground_z = ground_z
        self.min_points = min_points
        self.min_stage1_points = min_stage1_points
        self.bounce_z_threshold = bounce_z_threshold
        self.reset_timeout = reset_timeout
        self.motion_window_s = motion_window_s
        self.motion_min_y = motion_min_y
        self.land_skip_time = land_skip_time

        # ── 全局累积（不随重置清除） ──
        self.predictions: List[PredictHitPos] = []
        self.reset_times: List[float] = []

        # ── 当前抛球状态 ──
        self._last_obs_time: Optional[float] = None
        self._reset_throw()

    def _reset_throw(self) -> None:
        """重置当前抛球的内部状态，不清除全局 predictions。"""
        self._obs: List[BallObservation] = []
        self._stage: int = 0
        self._bounce_index: Optional[int] = None
        self._bounce_time: Optional[float] = None
        self._curve0: Optional[FittedCurve] = None
        self._curve1: Optional[FittedCurve] = None
        self._t_land: Optional[float] = None   # S0 预测的一次落地时间
        self._t_land2: Optional[float] = None  # S1 预测的二次落地时间
        self._n_post_fit: int = 0  # 参与 S1 拟合的观测数
        self._last_s0_pred: Optional[PredictHitPos] = None
        self._post_bounce_max_z: float = 0.0
        self._state: TrackerState = TrackerState.IDLE
        # ── 运动过滤状态 ──
        self._pending_obs: List[BallObservation] = []
        self._motion_confirmed: bool = False
        self._jump_count: int = 0   # 连续跳变帧计数
        self._skip_obs: bool = False  # 当前帧是否因跳变被跳过

    # ── 公开接口 ──

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def bounce_time(self) -> Optional[float]:
        return self._bounce_time

    @property
    def tracker_state(self) -> TrackerState:
        """当前追踪器状态，用于外部显示。"""
        return self._state

    def update(self, obs: BallObservation) -> TrackerResult:
        """
        输入一个新观测，返回 TrackerResult（包含预测和状态）。

        返回的 state 字段反映本帧的追踪状态：
          - DONE: 检测到超时/跳变/二次落地，本次抛球结束
          - IDLE: 观测点不足以拟合
          - TRACKING_S0: 正在 stage 0 追踪
          - IN_LANDING: 球已反弹但 post-bounce 点数不足
          - TRACKING_S1: 正在 stage 1 追踪
        """
        # ── 检查是否需要重置（超时 / 位置跳变）──
        self._skip_obs = False
        if self._check_reset(obs):
            self.reset_times.append(obs.t)
            self._reset_throw()
            # 注意：重置后继续处理当前 obs（它属于新一轮抛球的第一帧）

        # _last_obs_time 在运动过滤前更新（保持超时检测正常）
        self._last_obs_time = obs.t

        # 跳变帧：丢弃此帧，不进入拟合，保持当前追踪状态
        if self._skip_obs:
            return TrackerResult(prediction=None, state=self._state)

        # ── 运动过滤：剔除抛球前的静止观测 ──
        if not self._motion_confirmed:
            self._pending_obs.append(obs)
            if self._check_motion():
                self._motion_confirmed = True
                self._flush_pending_to_obs()
            else:
                return TrackerResult(prediction=None, state=TrackerState.IDLE)

        else:
            self._obs.append(obs)

        # ── 反弹检测 ──
        if self._stage == 0:
            self._detect_bounce()

        # ── 反弹后完成检测（球二次落地 → DONE） ──
        if self._stage == 1:
            self._update_post_bounce(obs)
            if self._detect_throw_complete():
                self.reset_times.append(obs.t)
                self._reset_throw()
                return TrackerResult(prediction=None, state=TrackerState.DONE)

        # ── 拟合 ──
        self._fit()

        # ── 二次落地检测（通过 S1 拟合曲线预测） ──
        if self._stage == 1 and self._curve1 is not None:
            self._update_land2_time()
            if self._t_land2 is not None and obs.t >= self._t_land2:
                self.reset_times.append(obs.t)
                self._reset_throw()
                return TrackerResult(prediction=None, state=TrackerState.DONE)

        # ── 确定当前状态 ──
        if self._stage == 0:
            if len(self._obs) >= self.min_points:
                self._state = TrackerState.TRACKING_S0
            else:
                self._state = TrackerState.IDLE
        else:
            # stage == 1：反弹后
            n_post = len(self._obs) - (self._bounce_index or 0)
            if n_post < self.min_stage1_points:
                # 反弹后采样不足，处于"落地中"阶段，不输出预测
                self._state = TrackerState.IN_LANDING
            else:
                self._state = TrackerState.TRACKING_S1

        # ── 预测 ──
        pred = self._predict(obs.t)
        if pred is not None:
            self.predictions.append(pred)
            if pred.stage == 0:
                self._last_s0_pred = pred

        return TrackerResult(prediction=pred, state=self._state)

    # ── 运动过滤 ──

    def _check_motion(self) -> bool:
        """
        检查 _pending_obs 中最新观测是否处于飞行状态。

        算法：从最新观测向前扫描，找到最近的、时间差刚好 >= motion_window_s
        的观测（最紧窗口），检查 |Δy| 是否 >= motion_min_y。

        用最紧窗口（而非最早观测）确保阈值对应固定时间段：
          飞行速度 ~6700mm/s × 0.2s = ~1340mm → 远超 500mm 阈值
          行走速度 ~370mm/s × 0.2s = ~74mm → 远低于 500mm 阈值
        若用最早观测（dt 可能 >> 0.2s），行走也能累积超过 500mm。
        """
        if len(self._pending_obs) < 2:
            return False
        latest = self._pending_obs[-1]
        # 从新到旧扫描，找第一个 dt >= window 的观测（最紧窗口）
        for earlier in reversed(self._pending_obs[:-1]):
            dt = latest.t - earlier.t
            if dt >= self.motion_window_s:
                return abs(latest.y - earlier.y) >= self.motion_min_y
        return False

    def _flush_pending_to_obs(self) -> None:
        """
        运动确认后，将 pending 中飞行阶段的观测转入 _obs，丢弃静止部分。

        策略：从前向后扫描 pending，对每个观测 k 找其最紧的 0.2s 窗口
        参考点，第一个 |Δy| >= motion_min_y 的 k 即为飞行起始点。
        """
        if not self._pending_obs:
            return
        motion_start = 0  # fallback: include all
        for k in range(1, len(self._pending_obs)):
            cur = self._pending_obs[k]
            # 从 k-1 向前找最近的 dt >= window 的观测
            for j in range(k - 1, -1, -1):
                if cur.t - self._pending_obs[j].t >= self.motion_window_s:
                    if abs(cur.y - self._pending_obs[j].y) >= self.motion_min_y:
                        motion_start = k
                    break
            if motion_start > 0:
                break
        self._obs.extend(self._pending_obs[motion_start:])
        self._pending_obs.clear()

    # ── 重置检测 ──

    _VELOCITY_JUMP_THRESHOLD = 50000  # 50 m/s — 超过此速度视为异常跳变
    _JUMP_RESET_COUNT = 3  # 连续跳变帧数达此值才真正重置

    def _check_reset(self, obs: BallObservation) -> bool:
        """超时 → 重置；位置跳变 → 跳过该帧（不重置）。

        单帧三角测量噪声（如一帧坐标偏移 40m）不应打断追踪。
        策略：
          - 超时 (>reset_timeout): 立即重置
          - 速度异常 (>50 m/s): 计数 +1，跳过该帧（返回 'skip'）
          - 连续 3 帧异常: 真正重置（可能是新抛球）
          - 正常帧: 计数归零
        返回值: True=重置, False=正常, 内部通过 _skip_obs 标记跳过。
        """
        if self._last_obs_time is not None:
            dt = obs.t - self._last_obs_time
            if dt > self.reset_timeout:
                self._jump_count = 0
                return True
        # 取最近一个已知观测（优先 _obs，否则 _pending_obs）
        prev = self._obs[-1] if self._obs else (
            self._pending_obs[-1] if self._pending_obs else None)
        if prev is not None:
            dt = obs.t - prev.t
            if dt > 0:
                dist = np.sqrt((obs.x - prev.x)**2
                               + (obs.y - prev.y)**2
                               + (obs.z - prev.z)**2)
                if dist / dt > self._VELOCITY_JUMP_THRESHOLD:
                    self._jump_count += 1
                    if self._jump_count >= self._JUMP_RESET_COUNT:
                        self._jump_count = 0
                        return True
                    self._skip_obs = True  # 跳过此帧，不进入拟合
                    return False
                else:
                    self._jump_count = 0
        return False

    def _update_post_bounce(self, obs: BallObservation) -> None:
        if obs.z > self._post_bounce_max_z:
            self._post_bounce_max_z = obs.z

    def _detect_throw_complete(self) -> bool:
        """反弹后球回到地面 → 本次抛球结束。"""
        if self._bounce_index is None:
            return False
        n_post = len(self._obs) - self._bounce_index
        if n_post < 10:
            return False
        if self._post_bounce_max_z < self.bounce_z_threshold * 2:
            return False
        return self._obs[-1].z < self.bounce_z_threshold

    def _update_land2_time(self) -> None:
        """用 S1 拟合曲线预测二次落地时间。超过此时间后不再给出预测。"""
        if self._curve1 is None or self._bounce_time is None:
            return
        # solve_t_for_z 返回最早的解；我们要反弹后的落地，即 after_time > bounce
        t_land = self._curve1.solve_t_for_z(
            self.ground_z, after_time=self._bounce_time + 0.1)
        if t_land is not None:
            self._t_land2 = t_land

    # ── 反弹检测 ──

    def _detect_bounce(self) -> None:
        n = len(self._obs)
        if n < 4:
            return
        for i in range(max(2, n - 5), n):
            p2, p1, cur = self._obs[i-2], self._obs[i-1], self._obs[i]
            if (p2.z > p1.z and cur.z > p1.z
                    and p1.z < self.bounce_z_threshold):
                self._stage = 1
                self._bounce_index = i - 1
                self._bounce_time = p1.t
                # 用 S0 曲线预测落地时间（比观测点更精确）
                if self._curve0 is not None:
                    t_pred = self._curve0.solve_t_for_z(
                        self.ground_z, latest=True)
                    self._t_land = t_pred if t_pred is not None else p1.t
                else:
                    self._t_land = p1.t
                return

    # ── 曲线拟合 ──

    def _fit(self) -> None:
        if self._stage == 0:
            if len(self._obs) >= self.min_points:
                self._curve0 = fit_curve(self._obs)
        else:
            if self._t_land is not None:
                t_lo = self._t_land - self.land_skip_time
                t_hi = self._t_land + self.land_skip_time
                pre = [o for o in self._obs if o.t < t_lo]
                post = [o for o in self._obs if o.t > t_hi]
                self._n_post_fit = len(post)
                if len(pre) >= self.min_points:
                    self._curve0 = fit_curve(pre)
                if len(post) >= self.min_points:
                    self._curve1 = fit_curve(post)

    # ── 击球预测 ──

    def _predict(self, ct: float) -> Optional[PredictHitPos]:
        if self._stage == 0:
            return self._predict_stage0(ct)
        else:
            return self._predict_stage1(ct)

    def _predict_stage0(self, ct: float) -> Optional[PredictHitPos]:
        """
        Stage 0: 拟合空中曲线 → 推算落地 → COR 反弹 →
        求下降阶段到达 ideal_hit_z 的时刻和位置。
        小车根据此 (x, y) 移动。
        """
        curve = self._curve0
        if curve is None:
            return None

        # 求落地时间
        t_bounce = curve.solve_t_for_z(self.ground_z, after_time=ct)
        if t_bounce is None:
            t_bounce = curve.solve_t_for_z(self.ground_z,
                                           after_time=curve.t_ref)
            if t_bounce is None or t_bounce < ct - 0.5:
                return None

        x_b, y_b, _ = curve.predict(t_bounce)
        vx_b, vy_b, vz_b = curve.velocity_at(t_bounce)
        # 反弹后速度：z 反向 × cor，x/y 衰减 × cor_xy
        vz_post = -self.cor * vz_b
        vx_post = self.cor_xy * vx_b
        vy_post = self.cor_xy * vy_b

        # z(dt) = vz_post*dt - 0.5*g*dt² = ideal_hit_z
        a = -0.5 * GRAVITY
        b = vz_post
        c = self.ground_z - self.ideal_hit_z
        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        dts = sorted(d for d in [(-b-sq)/(2*a), (-b+sq)/(2*a)] if d > 1e-6)
        if not dts:
            return None
        dt_hit = dts[-1]  # 下降阶段

        return PredictHitPos(
            x=x_b + vx_post * dt_hit,
            y=y_b + vy_post * dt_hit,
            z=self.ideal_hit_z,
            stage=0, ct=ct, ht=t_bounce + dt_hit,
        )

    def _predict_stage1(self, ct: float) -> Optional[PredictHitPos]:
        """
        Stage 1: 小车已到达 stage 0 最后预测的 y 位置（不再移动），
        用反弹后拟合曲线求解球到达该 y 时的 x 和 z。

        具体步骤：
          1. target_y = last_s0_pred.y （小车的 y 位置）
          2. 求解 curve1 的 y(t) = target_y → t_hit
          3. 在 t_hit 取 x(t_hit) 和 z(t_hit)
          4. z(t_hit) 即 stage1z_in_stage0_car_loc
        """
        # 需要足够多的反弹后采样点（land_skip_time 排除区间外的有效点数）
        if self._n_post_fit < self.min_stage1_points:
            return None

        curve = self._curve1
        if curve is None:
            return None

        # 需要 stage 0 的最后预测来确定小车的 y 位置
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
