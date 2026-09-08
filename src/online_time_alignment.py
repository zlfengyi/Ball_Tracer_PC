"""Online per-throw PC/RK ball-trajectory time alignment.

The fitted clock contract is always::

    pc_exposure_time = rk_payload_time + pc_minus_rk

Only complete incoming throws are eligible: each side must observe the ball
cross an incoming gate line toward the car, descend into a bounce, and then
rise in stage 1.  The gate sits at y=8 m, but a throw whose first bounce lands
beyond that line is judged against a line ``_CROSS_BOUNCE_MARGIN_M`` above its
own bounce instead -- what the gate is really asserting is "this side watched
the ball come in and go down", which a fixed line cannot express for a short
feed.  Each side uses its own bounce, so the car's pose-belief error (see
``xy_shift`` below) cancels out of the comparison.  A completed PC/RK pair is
consumed exactly once.  Failed fits never replace the last accepted offset.

Spatial agreement is judged translation-invariant: the median XY
displacement between the two world-frame tracks is the car's pose belief
error (which stays large until the first accepted offset unblocks
/pc_car_loc) and is reported as ``xy_shift``, never gated on.
"""

from __future__ import annotations

import bisect
import copy
import math
import threading
from collections import Counter
from dataclasses import dataclass, field


_CROSS_Y_M = 8.0
# Floor only.  A feed that first bounces past _CROSS_Y_M never crosses it while
# airborne, so gating on the fixed line rejects the whole session (0907 evening:
# first bounces at 8.1-11.8 m, every attempt "bounce_before_cross").  Raising the
# constant instead is not an option -- the 0907 morning session only acquired the
# ball at y~9.4-10.8, and any line high enough for the deep feeds is above where
# those tracks start.  The margin is what both cases actually have in common.
_CROSS_BOUNCE_MARGIN_M = 1.0
_MAX_POINT_GAP_S = 0.25
_MAX_TIMESTAMP_ARRIVAL_SKEW_S = 1.0
_MIN_CLOCK_ROLLBACK_S = 5.0
_MAX_FIT_GAP_S = 0.08
_MAX_FLIGHT_SPAN_S = 3.0
_MAX_SPEED_MPS = 45.0
_MAX_BOUNCE_Z_M = 0.35
_FIT_PRE_BOUNCE_S = 0.55
_FIT_POST_BOUNCE_S = 0.30
_MIN_STAGE1_SPAN_S = 0.20
_MIN_SHORT_STAGE1_POINTS = 6

_MAX_JOINT_ARRIVAL_GAP_S = 0.10
_MAX_JOINT_SPATIAL_GAP_M = 3.3
_MAX_CONSECUTIVE_SPATIAL_REJECTS = 3
_MAX_JOINT_GAP_S = 0.35
_BOUNCE_JOINT_PAYLOAD_WINDOW_S = 0.12
_MAX_BOUNCE_SAMPLE_ARRIVAL_GAP_S = 0.16
_RECENT_SAMPLE_SPAN_S = 0.70
_CONSUMED_TAIL_S = 1.25

_MAX_OFFSET_DELTA_S = 0.10
_MAX_ACCEPTED_OFFSET_JUMP_S = 0.030
_RIVAL_GAP_S = 0.01
_MAX_XY_RES_MEDIAN_M = 0.30
_MAX_XY_RES_P90_M = 0.60


@dataclass(frozen=True)
class _Point:
    t: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class _Sample:
    point: _Point
    arrival_pc: float
    tracker_state: str | None = None
    bounce_time: float | None = None


@dataclass
class _SourceClock:
    last_t: float = float("-inf")
    last_arrival_pc: float = float("-inf")
    last_point: _Point | None = None


@dataclass
class _ThrowCandidate:
    sequence: int
    start_arrival_pc: float
    last_joint_arrival_pc: float
    samples: dict[str, list[_Sample]] = field(
        default_factory=lambda: {"pc": [], "rk": []}
    )
    joint_pairs: list[tuple[_Sample, _Sample]] = field(default_factory=list)
    bounce_index: dict[str, int | None] = field(
        default_factory=lambda: {"pc": None, "rk": None}
    )
    max_arrival_gap_s: float = 0.0
    max_spatial_gap_m: float = 0.0
    spatial_rejects: int = 0
    spatial_reject_streak: int = 0
    confirmed: bool = False


def _slope(points: list[_Point] | tuple[_Point, ...], key: str) -> float:
    if len(points) < 2:
        return 0.0
    ts = [point.t for point in points]
    values = [getattr(point, key) for point in points]
    t_mean = sum(ts) / len(ts)
    v_mean = sum(values) / len(values)
    denom = sum((value - t_mean) ** 2 for value in ts)
    if denom <= 1e-12:
        return 0.0
    return sum(
        (point.t - t_mean) * (getattr(point, key) - v_mean)
        for point in points
    ) / denom


def _upper_median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _valid_point(point: _Point) -> bool:
    values = (point.t, point.x, point.y, point.z)
    return (
        all(math.isfinite(value) for value in values)
        and -20.0 <= point.x <= 20.0
        and -5.0 <= point.y <= 25.0
        and -0.10 <= point.z <= 5.0
    )


def _distance(left: _Point, right: _Point) -> float:
    return math.sqrt(
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    )


def _curve_is_clean(points: list[_Point]) -> bool:
    if len(points) < 10:
        return False
    falling_steps = sum(
        b.y <= a.y + 0.04 for a, b in zip(points, points[1:])
    )
    if falling_steps / max(1, len(points) - 1) < 0.85:
        return False
    for left, middle, right in zip(points, points[1:], points[2:]):
        span = right.t - left.t
        if span <= 0.0:
            return False
        fraction = (middle.t - left.t) / span
        for key, limit in (("x", 0.30), ("y", 0.30)):
            expected = getattr(left, key) + fraction * (
                getattr(right, key) - getattr(left, key)
            )
            if abs(getattr(middle, key) - expected) > limit:
                return False
    return True


class OnlineThrowTimeAligner:
    """Maintain one accepted PC-minus-RK offset from complete throws."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._source_clock = {"pc": _SourceClock(), "rk": _SourceClock()}
        self._rk_rollback_probe: _Sample | None = None
        self._recent: dict[str, list[_Sample]] = {"pc": [], "rk": []}
        self._unmatched: dict[str, list[_Sample]] = {"pc": [], "rk": []}
        self._candidate: _ThrowCandidate | None = None
        self._candidate_sequence = 0
        self._blocked_until_arrival_pc = float("-inf")
        self._pc_minus_rk: float | None = None
        self._updated_pc: float | None = None
        self._attempts = 0
        self._accepted = 0
        self._rejected = 0
        self._pc_flights = 0
        self._rk_flights = 0
        self._history: list[dict] = []
        self._reason_counts: Counter[str] = Counter()

    def add_pc(
        self,
        *,
        t: float,
        x: float,
        y: float,
        z: float,
        arrival_pc: float,
        tracker_state=None,
        bounce_time: float | None = None,
    ) -> dict | None:
        state = getattr(tracker_state, "value", tracker_state)
        state = str(state).lower() if state is not None else None
        hint = None
        if bounce_time is not None:
            try:
                hint = float(bounce_time)
            except (TypeError, ValueError, OverflowError):
                hint = None
            if hint is not None and not math.isfinite(hint):
                hint = None
        try:
            point = _Point(float(t), float(x), float(y), float(z))
            arrival = float(arrival_pc)
        except (TypeError, ValueError, OverflowError):
            return None
        return self._add(
            "pc",
            point,
            arrival_pc=arrival,
            tracker_state=state,
            bounce_time=hint,
        )

    def add_rk(self, payload: dict, *, arrival_pc: float) -> dict | None:
        try:
            point = _Point(
                float(payload["t"]),
                float(payload["x"]),
                float(payload["y"]),
                float(payload["z"]),
            )
            arrival = float(arrival_pc)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return self._add("rk", point, arrival_pc=arrival)

    def expire(self, *, now_arrival_pc: float) -> dict | None:
        try:
            now_arrival_pc = float(now_arrival_pc)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(now_arrival_pc):
            return None
        with self._lock:
            candidate = self._candidate
            if (
                candidate is None
                or now_arrival_pc - candidate.last_joint_arrival_pc
                <= _MAX_JOINT_GAP_S
            ):
                return None
            if candidate.confirmed and self._candidate_is_committed(candidate):
                return self._finish_incomplete(
                    now_arrival_pc=now_arrival_pc,
                    joint_gap=True,
                )
            self._discard_seed()
            return None

    def _add(
        self,
        source: str,
        point: _Point,
        *,
        arrival_pc: float,
        tracker_state: str | None = None,
        bounce_time: float | None = None,
    ) -> dict | None:
        if not math.isfinite(arrival_pc) or not _valid_point(point):
            return None
        with self._lock:
            sample = self._validated_sample(
                source,
                point,
                arrival_pc=arrival_pc,
                tracker_state=tracker_state,
                bounce_time=bounce_time,
            )
            if sample is None:
                return None
            terminal = None
            candidate = self._candidate
            if (
                candidate is not None
                and arrival_pc - candidate.last_joint_arrival_pc
                > _MAX_JOINT_GAP_S
            ):
                if candidate.confirmed and self._candidate_is_committed(candidate):
                    terminal = self._finish_incomplete(
                        now_arrival_pc=arrival_pc,
                        joint_gap=True,
                    )
                else:
                    self._discard_seed()

            if arrival_pc < self._blocked_until_arrival_pc:
                return terminal
            if math.isfinite(self._blocked_until_arrival_pc):
                self._blocked_until_arrival_pc = float("-inf")
                self._clear_pair_buffers()

            self._prune_buffers(arrival_pc)
            self._recent[source].append(sample)

            opposite = "rk" if source == "pc" else "pc"
            spatial_rejects = []
            matches = []
            for index, other in enumerate(self._unmatched[opposite]):
                arrival_gap = abs(sample.arrival_pc - other.arrival_pc)
                if arrival_gap > _MAX_JOINT_ARRIVAL_GAP_S:
                    continue
                spatial_gap = _distance(sample.point, other.point)
                if spatial_gap > _MAX_JOINT_SPATIAL_GAP_M:
                    spatial_rejects.append(
                        (arrival_gap, spatial_gap, index, other)
                    )
                    continue
                matches.append((arrival_gap, spatial_gap, index, other))

            if not matches:
                self._unmatched[source].append(sample)
                if spatial_rejects:
                    if self._candidate is not None:
                        evidence = {
                            side: self._source_evidence(self._candidate, side)
                            for side in ("pc", "rk")
                        }
                        if (
                            self._candidate.confirmed
                            and all(
                                evidence[side]["state"]
                                in {"stage1_ready", "complete"}
                                for side in ("pc", "rk")
                            )
                        ):
                            completed = self._fit_candidate(
                                self._candidate,
                                evidence=evidence,
                                now_arrival_pc=arrival_pc,
                                defer_rejection=True,
                            )
                            if completed is not None:
                                return terminal if terminal is not None else completed
                    arrival_gap, spatial_gap, _, other = min(spatial_rejects)
                    joint_arrival = max(
                        sample.arrival_pc, other.arrival_pc
                    )
                    if self._candidate is not None:
                        self._candidate.spatial_rejects += 1
                        self._candidate.spatial_reject_streak += 1
                        self._candidate.max_arrival_gap_s = max(
                            self._candidate.max_arrival_gap_s, arrival_gap
                        )
                        self._candidate.max_spatial_gap_m = max(
                            self._candidate.max_spatial_gap_m, spatial_gap
                        )
                        self._candidate.last_joint_arrival_pc = joint_arrival
                elif self._candidate is not None:
                    self._append_candidate_sample(source, sample)
                completed = None
                if self._candidate is not None:
                    self._lock_first_bounces(self._candidate)
                    completed = self._evaluate_candidate(
                        now_arrival_pc=arrival_pc
                    )
                return terminal if terminal is not None else completed

            arrival_gap, spatial_gap, match_index, other = min(matches)
            self._unmatched[opposite].pop(match_index)
            pc_sample, rk_sample = (
                (sample, other) if source == "pc" else (other, sample)
            )
            joint_arrival = max(pc_sample.arrival_pc, rk_sample.arrival_pc)

            if self._candidate is None and self._joint_can_start(joint_arrival):
                self._start_candidate(
                    pc_sample, rk_sample, joint_arrival_pc=joint_arrival
                )

            candidate = self._candidate
            if candidate is None:
                return terminal
            candidate.spatial_reject_streak = 0
            self._append_candidate_sample("pc", pc_sample)
            self._append_candidate_sample("rk", rk_sample)
            candidate.joint_pairs.append((pc_sample, rk_sample))
            candidate.last_joint_arrival_pc = joint_arrival
            candidate.max_arrival_gap_s = max(
                candidate.max_arrival_gap_s, arrival_gap
            )
            candidate.max_spatial_gap_m = max(
                candidate.max_spatial_gap_m, spatial_gap
            )
            self._lock_first_bounces(candidate)
            completed = self._evaluate_candidate(now_arrival_pc=joint_arrival)
            return terminal if terminal is not None else completed

    def _validated_sample(
        self,
        source: str,
        point: _Point,
        *,
        arrival_pc: float,
        tracker_state: str | None,
        bounce_time: float | None,
    ) -> _Sample | None:
        clock = self._source_clock[source]
        if point.t <= clock.last_t:
            if (
                source == "rk"
                and point.t < clock.last_t - _MIN_CLOCK_ROLLBACK_S
                and arrival_pc > clock.last_arrival_pc
            ):
                probe = self._rk_rollback_probe
                if probe is not None:
                    payload_dt = point.t - probe.point.t
                    arrival_dt = arrival_pc - probe.arrival_pc
                    if (
                        payload_dt > 0.0
                        and arrival_dt > 0.0
                        and abs(payload_dt - arrival_dt)
                        <= _MAX_TIMESTAMP_ARRIVAL_SKEW_S
                    ):
                        # Two coherent samples on a substantially lower RK
                        # perf_counter axis prove a restart.  A single late or
                        # corrupt packet must never invalidate a good offset.
                        self._source_clock["rk"] = _SourceClock(
                            last_t=point.t,
                            last_arrival_pc=arrival_pc,
                            last_point=point,
                        )
                        self._rk_rollback_probe = None
                        self._candidate = None
                        self._clear_pair_buffers()
                        self._blocked_until_arrival_pc = float("-inf")
                        self._pc_minus_rk = None
                        self._updated_pc = None
                        return _Sample(
                            point=point,
                            arrival_pc=arrival_pc,
                            tracker_state=tracker_state,
                            bounce_time=bounce_time,
                        )
                self._rk_rollback_probe = _Sample(
                    point=point,
                    arrival_pc=arrival_pc,
                )
            elif source == "rk":
                self._rk_rollback_probe = None
            return None
        if source == "rk":
            self._rk_rollback_probe = None
        if math.isfinite(clock.last_t):
            payload_dt = point.t - clock.last_t
            arrival_dt = arrival_pc - clock.last_arrival_pc
            if (
                arrival_dt < 0.0
                or abs(payload_dt - arrival_dt)
                > _MAX_TIMESTAMP_ARRIVAL_SKEW_S
            ):
                return None
            if (
                clock.last_point is not None
                and _distance(clock.last_point, point) / payload_dt
                > _MAX_SPEED_MPS
            ):
                return None
        clock.last_t = point.t
        clock.last_arrival_pc = arrival_pc
        clock.last_point = point
        return _Sample(
            point=point,
            arrival_pc=arrival_pc,
            tracker_state=tracker_state,
            bounce_time=bounce_time,
        )

    def _prune_buffers(self, now_arrival_pc: float) -> None:
        for source in ("pc", "rk"):
            self._recent[source] = [
                sample for sample in self._recent[source]
                if now_arrival_pc - sample.arrival_pc <= _RECENT_SAMPLE_SPAN_S
            ]
            self._unmatched[source] = [
                sample for sample in self._unmatched[source]
                if now_arrival_pc - sample.arrival_pc
                <= _MAX_JOINT_ARRIVAL_GAP_S
            ]

    def _clear_pair_buffers(self) -> None:
        self._recent = {"pc": [], "rk": []}
        self._unmatched = {"pc": [], "rk": []}

    def _discard_seed(self) -> None:
        self._candidate = None
        self._clear_pair_buffers()

    @staticmethod
    def _candidate_is_committed(candidate: _ThrowCandidate) -> bool:
        if any(index is not None for index in candidate.bounce_index.values()):
            return True
        return any(
            sample.tracker_state in {"in_landing", "tracking_s1"}
            for sample in candidate.samples["pc"]
        )

    def _joint_can_start(self, joint_arrival_pc: float) -> bool:
        return all(
            any(
                sample.point.y >= _CROSS_Y_M
                and joint_arrival_pc - sample.arrival_pc <= _MAX_JOINT_GAP_S
                for sample in self._recent[source]
            )
            for source in ("pc", "rk")
        )

    @staticmethod
    def _joint_incoming_prefix_start(
        candidate: _ThrowCandidate,
    ) -> float | None:
        if len(candidate.joint_pairs) < 3:
            return None
        latest_arrival = max(
            max(pc.arrival_pc, rk.arrival_pc)
            for pc, rk in candidate.joint_pairs
        )
        pairs = [
            pair for pair in candidate.joint_pairs
            if latest_arrival - max(pair[0].arrival_pc, pair[1].arrival_pc)
            <= _MAX_JOINT_GAP_S
        ]
        if len(pairs) < 3:
            return None
        for source in ("pc", "rk"):
            sample_index = 0 if source == "pc" else 1
            prefix = [pair[sample_index].point for pair in pairs]
            if (
                len(prefix) < 3
                or not any(point.y >= _CROSS_Y_M for point in prefix)
                or prefix[-1].t - prefix[0].t < 0.04
                or max(point.y for point in prefix) - prefix[-1].y < 0.10
                or _slope(prefix, "y") > -1.0
            ):
                return None
        return min(pairs[0][0].arrival_pc, pairs[0][1].arrival_pc)

    def _start_candidate(
        self,
        pc_sample: _Sample,
        rk_sample: _Sample,
        *,
        joint_arrival_pc: float,
    ) -> None:
        self._candidate_sequence += 1
        self._candidate = _ThrowCandidate(
            sequence=self._candidate_sequence,
            start_arrival_pc=min(
                pc_sample.arrival_pc, rk_sample.arrival_pc
            ),
            last_joint_arrival_pc=joint_arrival_pc,
        )
        for source in ("pc", "rk"):
            for recent in self._recent[source]:
                if (
                    joint_arrival_pc - recent.arrival_pc
                    <= _RECENT_SAMPLE_SPAN_S
                ):
                    self._append_candidate_sample(source, recent)

    def _append_candidate_sample(self, source: str, sample: _Sample) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        samples = candidate.samples[source]
        if samples and sample.point.t <= samples[-1].point.t:
            return
        samples.append(sample)

    @staticmethod
    def _find_bounce_index(samples: list[_Sample]) -> int | None:
        points = [sample.point for sample in samples]
        for index in range(2, len(points) - 2):
            bounce = points[index]
            if bounce.z > _MAX_BOUNCE_Z_M + 0.10:
                continue
            pre = points[index - 2:index + 1]
            post = points[index:index + 3]
            if bounce.z > min(point.z for point in pre + post) + 0.025:
                continue
            if _slope(pre, "z") >= -0.30 or _slope(post, "z") <= 0.30:
                continue
            return index
        return None

    @classmethod
    def _find_pc_hint_bounce_index(
        cls, samples: list[_Sample]
    ) -> int | None:
        hints = [
            sample.bounce_time for sample in samples
            if sample.bounce_time is not None
            and sample.tracker_state in {"in_landing", "tracking_s1"}
        ]
        if not hints:
            return None
        hint = hints[0]
        points = [sample.point for sample in samples]
        if not points or points[-1].t < hint + 0.06:
            return None
        index = min(range(len(points)), key=lambda i: abs(points[i].t - hint))
        if index < 2 or index + 2 >= len(points):
            return None
        if points[index].z > _MAX_BOUNCE_Z_M + 0.10:
            return None
        pre = points[index - 2:index + 1]
        post = points[index:index + 3]
        if _slope(pre, "z") >= -0.30 or _slope(post, "z") <= 0.30:
            return None
        return index

    def _lock_first_bounces(self, candidate: _ThrowCandidate) -> None:
        if not candidate.confirmed:
            return
        for source in ("pc", "rk"):
            if candidate.bounce_index[source] is not None:
                continue
            samples = candidate.samples[source]
            index = self._find_bounce_index(samples)
            if index is None and source == "pc":
                index = self._find_pc_hint_bounce_index(samples)
            if index is not None:
                candidate.bounce_index[source] = index

    @staticmethod
    def _cross_time(
        points: list[_Point], end_index: int, cross_y: float = _CROSS_Y_M
    ) -> float | None:
        for left, right in zip(points[:end_index], points[1:end_index + 1]):
            if left.y < cross_y or right.y > cross_y:
                continue
            if right.y >= left.y or right.t <= left.t:
                continue
            fraction = (left.y - cross_y) / (left.y - right.y)
            return left.t + fraction * (right.t - left.t)
        return None

    @staticmethod
    def _cross_line(bounce_y: float) -> float:
        return max(_CROSS_Y_M, bounce_y + _CROSS_BOUNCE_MARGIN_M)

    def _source_evidence(
        self, candidate: _ThrowCandidate, source: str
    ) -> dict:
        samples = candidate.samples[source]
        points = [sample.point for sample in samples]
        span = points[-1].t - points[0].t if len(points) >= 2 else 0.0
        bounce_index = candidate.bounce_index[source]
        generic_cross = self._cross_time(points, len(points) - 1) if points else None
        evidence = {
            "state": "first_bounce_incomplete",
            "crossed_8": generic_cross is not None,
            "cross_t": generic_cross,
            "cross_y": _CROSS_Y_M,
            "pre_points": 0,
            "post_points": 0,
            "bounce_y": None,
            "bounce_t": None,
            "bounce_arrival_pc": None,
            "span_ms": span * 1000.0,
            "post_window_complete": False,
            "fit_points": (),
        }
        if bounce_index is None:
            return evidence

        bounce_sample = samples[bounce_index]
        bounce = bounce_sample.point
        cross_y = self._cross_line(bounce.y)
        cross_t = self._cross_time(points, bounce_index, cross_y)
        evidence.update({
            "crossed_8": cross_t is not None,
            "cross_t": cross_t,
            "cross_y": cross_y,
            "bounce_y": bounce.y,
            "bounce_t": bounce.t,
            "bounce_arrival_pc": bounce_sample.arrival_pc,
            "post_window_complete": (
                points[-1].t - bounce.t >= _FIT_POST_BOUNCE_S
            ),
        })
        if cross_t is None:
            evidence["state"] = "bounce_before_cross"
            return evidence
        if bounce.z > _MAX_BOUNCE_Z_M:
            evidence["state"] = "bounce_invalid"
            return evidence

        fit_start = bounce.t - _FIT_PRE_BOUNCE_S
        fit_end = bounce.t + _FIT_POST_BOUNCE_S
        pre = [point for point in points if fit_start <= point.t <= bounce.t]
        post = [point for point in points if bounce.t <= point.t <= fit_end]
        fit_points = [point for point in points if fit_start <= point.t <= fit_end]
        evidence["pre_points"] = len(pre)
        evidence["post_points"] = len(post)
        evidence["fit_points"] = tuple(fit_points)

        local_pre = points[bounce_index - 2:bounce_index + 1]
        local_post = points[bounce_index:bounce_index + 3]
        if len(local_pre) < 3 or len(local_post) < 3:
            return evidence
        if any(b.z > a.z + 0.03 for a, b in zip(local_pre, local_pre[1:])):
            evidence["state"] = "bounce_invalid"
            return evidence
        if any(b.z < a.z - 0.03 for a, b in zip(local_post, local_post[1:])):
            evidence["state"] = "bounce_invalid"
            return evidence
        if any(
            abs(b.z - a.z) / (b.t - a.t) > 12.0
            for segment in (local_pre, local_post)
            for a, b in zip(segment, segment[1:])
        ):
            evidence["state"] = "bounce_invalid"
            return evidence

        short_stage1_proven = len(post) >= _MIN_SHORT_STAGE1_POINTS
        if (
            points[-1].t - bounce.t < _MIN_STAGE1_SPAN_S
            and not short_stage1_proven
        ):
            evidence["state"] = "stage1_incomplete"
            return evidence
        if len(pre) < 3 or len(post) < 5:
            evidence["state"] = "stage1_invalid"
            return evidence
        if (
            pre[-1].t - pre[0].t < 0.06
            or (
                post[-1].t - post[0].t < _MIN_STAGE1_SPAN_S
                and not short_stage1_proven
            )
        ):
            evidence["state"] = "stage1_invalid"
            return evidence
        if _slope(pre, "z") > -0.5 or _slope(post, "z") < 0.5:
            evidence["state"] = "stage1_invalid"
            return evidence
        if pre[0].z - bounce.z < 0.10:
            evidence["state"] = "bounce_invalid"
            return evidence
        if max(point.z for point in post) - bounce.z < 0.12:
            evidence["state"] = "stage1_invalid"
            return evidence
        rising_steps = sum(
            right.z >= left.z - 0.03 for left, right in zip(post, post[1:])
        )
        if rising_steps / max(1, len(post) - 1) < 0.75:
            evidence["state"] = "stage1_invalid"
            return evidence
        if fit_points[-1].t - fit_points[0].t < 0.36:
            evidence["state"] = "flight_curve_dirty"
            return evidence
        if any(
            right.t - left.t > _MAX_POINT_GAP_S
            for left, right in zip(fit_points, fit_points[1:])
        ):
            evidence["state"] = "flight_curve_dirty"
            return evidence
        for left, middle, right in zip(
            fit_points, fit_points[1:], fit_points[2:]
        ):
            if abs(middle.t - bounce.t) <= 0.08:
                continue
            fraction = (middle.t - left.t) / (right.t - left.t)
            expected_z = left.z + fraction * (right.z - left.z)
            if abs(middle.z - expected_z) > 0.30:
                evidence["state"] = "flight_curve_dirty"
                return evidence
        if _slope(fit_points, "y") > -1.0 or not _curve_is_clean(fit_points):
            evidence["state"] = "flight_curve_dirty"
            return evidence
        evidence["state"] = (
            "complete"
            if points[-1].t - bounce.t >= _FIT_POST_BOUNCE_S
            else "stage1_ready"
        )
        return evidence

    def _candidate_features(
        self,
        candidate: _ThrowCandidate,
        evidence: dict[str, dict] | None = None,
    ) -> dict:
        evidence = evidence or {
            source: self._source_evidence(candidate, source)
            for source in ("pc", "rk")
        }
        features = {
            "joint_confirmed": candidate.confirmed,
            "joint_pair_count": len(candidate.joint_pairs),
            "max_arrival_gap_ms": candidate.max_arrival_gap_s * 1000.0,
            "max_spatial_gap_m": candidate.max_spatial_gap_m,
            "spatial_reject_count": candidate.spatial_rejects,
            "spatial_reject_streak": candidate.spatial_reject_streak,
        }
        for source in ("pc", "rk"):
            row = evidence[source]
            features.update({
                f"{source}_state": row["state"],
                f"{source}_crossed_8": bool(row["crossed_8"]),
                f"{source}_cross_y": row["cross_y"],
                f"{source}_pre_points": int(row["pre_points"]),
                f"{source}_post_points": int(row["post_points"]),
                f"{source}_bounce_y": row["bounce_y"],
                f"{source}_span_ms": row["span_ms"],
            })
        return features

    @staticmethod
    def _incomplete_reason(
        candidate: _ThrowCandidate, evidence: dict[str, dict], *, joint_gap: bool
    ) -> str:
        if (
            candidate.spatial_reject_streak
            >= _MAX_CONSECUTIVE_SPATIAL_REJECTS
            and joint_gap
        ):
            return "joint_identity_lost"
        states = {evidence[source]["state"] for source in ("pc", "rk")}
        if "bounce_before_cross" in states:
            return "bounce_before_cross"
        if "bounce_invalid" in states:
            return "bounce_invalid"
        if not all(evidence[source]["crossed_8"] for source in ("pc", "rk")):
            return "cross_incomplete"
        if "first_bounce_incomplete" in states:
            return "first_bounce_incomplete"
        if states & {"stage1_incomplete", "stage1_invalid"}:
            return "stage1_incomplete"
        if "flight_curve_dirty" in states:
            return "flight_curve_dirty"
        return "joint_gap" if joint_gap else "quality_gate"

    def _evaluate_candidate(self, *, now_arrival_pc: float) -> dict | None:
        candidate = self._candidate
        if candidate is None:
            return None
        if not candidate.confirmed:
            prefix_start = self._joint_incoming_prefix_start(candidate)
            if prefix_start is not None:
                candidate.confirmed = True
                keep_from = prefix_start - _MAX_JOINT_ARRIVAL_GAP_S
                for source in ("pc", "rk"):
                    candidate.samples[source] = [
                        sample for sample in candidate.samples[source]
                        if sample.arrival_pc >= keep_from
                    ]
                candidate.bounce_index = {"pc": None, "rk": None}
                self._lock_first_bounces(candidate)
        evidence = {
            source: self._source_evidence(candidate, source)
            for source in ("pc", "rk")
        }
        states = {evidence[source]["state"] for source in ("pc", "rk")}
        if not candidate.confirmed:
            if now_arrival_pc - candidate.start_arrival_pc > _MAX_FLIGHT_SPAN_S:
                self._discard_seed()
            return None
        if (
            candidate.spatial_reject_streak
            >= _MAX_CONSECUTIVE_SPATIAL_REJECTS
        ):
            return self._record_terminal(
                "joint_identity_lost",
                features=self._candidate_features(candidate, evidence),
                now_arrival_pc=now_arrival_pc,
                consume_tail=True,
            )
        if states & {"bounce_before_cross", "bounce_invalid"}:
            reason = self._incomplete_reason(
                candidate, evidence, joint_gap=False
            )
            return self._record_terminal(
                reason,
                features=self._candidate_features(candidate, evidence),
                now_arrival_pc=now_arrival_pc,
                consume_tail=True,
            )
        if "flight_curve_dirty" in states or (
            "stage1_invalid" in states
            and all(
                evidence[source]["post_window_complete"]
                for source in ("pc", "rk")
            )
        ):
            reason = self._incomplete_reason(
                candidate, evidence, joint_gap=False
            )
            return self._record_terminal(
                reason,
                features=self._candidate_features(candidate, evidence),
                now_arrival_pc=now_arrival_pc,
                consume_tail=True,
            )
        if not all(
            evidence[source]["state"] == "complete"
            for source in ("pc", "rk")
        ):
            if now_arrival_pc - candidate.start_arrival_pc > _MAX_FLIGHT_SPAN_S:
                return self._finish_incomplete(
                    now_arrival_pc=now_arrival_pc,
                    joint_gap=False,
                )
            return None
        return self._fit_candidate(
            candidate,
            evidence=evidence,
            now_arrival_pc=now_arrival_pc,
        )

    def _finish_incomplete(
        self, *, now_arrival_pc: float, joint_gap: bool
    ) -> dict | None:
        candidate = self._candidate
        if candidate is None:
            return None
        evidence = {
            source: self._source_evidence(candidate, source)
            for source in ("pc", "rk")
        }
        if (
            candidate.confirmed
            and candidate.spatial_reject_streak
            < _MAX_CONSECUTIVE_SPATIAL_REJECTS
            and all(
                evidence[source]["state"] in {"stage1_ready", "complete"}
                for source in ("pc", "rk")
            )
        ):
            return self._fit_candidate(
                candidate,
                evidence=evidence,
                now_arrival_pc=now_arrival_pc,
            )
        reason = self._incomplete_reason(candidate, evidence, joint_gap=joint_gap)
        return self._record_terminal(
            reason,
            features=self._candidate_features(candidate, evidence),
            now_arrival_pc=now_arrival_pc,
            consume_tail=candidate.confirmed,
        )

    def _fit_candidate(
        self,
        candidate: _ThrowCandidate,
        *,
        evidence: dict[str, dict],
        now_arrival_pc: float,
        defer_rejection: bool = False,
    ) -> dict | None:
        features = self._candidate_features(candidate, evidence)
        pc = evidence["pc"]
        rk = evidence["rk"]
        bounce_arrival_gap = abs(
            pc["bounce_arrival_pc"] - rk["bounce_arrival_pc"]
        )
        features["bounce_arrival_gap_ms"] = bounce_arrival_gap * 1000.0
        bounce_joint_pair_count = sum(
            abs(pc_sample.point.t - pc["bounce_t"])
            <= _BOUNCE_JOINT_PAYLOAD_WINDOW_S
            and abs(rk_sample.point.t - rk["bounce_t"])
            <= _BOUNCE_JOINT_PAYLOAD_WINDOW_S
            and abs(pc_sample.arrival_pc - rk_sample.arrival_pc)
            <= _MAX_JOINT_ARRIVAL_GAP_S
            and _distance(pc_sample.point, rk_sample.point)
            <= _MAX_JOINT_SPATIAL_GAP_M
            for pc_sample in candidate.samples["pc"]
            for rk_sample in candidate.samples["rk"]
        )
        features["bounce_joint_pair_count"] = bounce_joint_pair_count
        if (
            bounce_joint_pair_count == 0
            or bounce_arrival_gap > _MAX_BOUNCE_SAMPLE_ARRIVAL_GAP_S
        ):
            if defer_rejection:
                return None
            return self._record_terminal(
                "bounce_arrival_mismatch",
                features=features,
                now_arrival_pc=now_arrival_pc,
                consume_tail=True,
            )

        pc_points = pc["fit_points"]
        rk_points = rk["fit_points"]
        if self._pc_minus_rk is None:
            baseline = pc["bounce_t"] - rk["bounce_t"]
            baseline_source = "paired_bounce"
        else:
            baseline = self._pc_minus_rk
            baseline_source = "previous_throw"
        extra = {
            "pc_bounce_t": pc["bounce_t"],
            "rk_bounce_t": rk["bounce_t"],
            "baseline_pc_minus_rk": baseline,
            "baseline_source": baseline_source,
        }
        baseline_bounce_error = pc["bounce_t"] - (rk["bounce_t"] + baseline)
        extra["baseline_bounce_error_ms"] = baseline_bounce_error * 1000.0
        if abs(baseline_bounce_error) > 0.15:
            if defer_rejection:
                return None
            return self._record_terminal(
                "pair_time_mismatch",
                features=features,
                extra=extra,
                now_arrival_pc=now_arrival_pc,
                consume_tail=True,
            )

        rk_moving = tuple(
            point
            for previous, point in zip(rk_points, rk_points[1:])
            if 0.0 < point.t - previous.t < 0.2
            and abs(point.z - previous.z) / (point.t - previous.t) > 0.4
        )
        fit = self._estimate(tuple(pc_points), rk_moving, baseline)
        extra.update(fit)
        reason = self._rejection_reason(fit)
        fitted_offset = fit.get("pc_minus_rk")
        if reason is None and isinstance(fitted_offset, float):
            bounce_error = pc["bounce_t"] - (rk["bounce_t"] + fitted_offset)
            extra["bounce_error_ms"] = bounce_error * 1000.0
            if abs(bounce_error) > 0.06:
                reason = "bounce_mismatch"
            xy = self._score_xy(
                tuple(pc_points), tuple(rk_points), fitted_offset
            )
            extra.update(xy)
            features.update({
                "xy_n": xy["xy_n"],
                "xy_shift": xy["xy_shift"],
                "xy_res_median": xy["xy_res_median"],
                "xy_res_p90": xy["xy_res_p90"],
            })
            if (
                reason is None
                and (
                    xy["xy_n"] < 8
                    or xy["xy_res_median"] is None
                    or xy["xy_res_median"] > _MAX_XY_RES_MEDIAN_M
                    or xy["xy_res_p90"] is None
                    or xy["xy_res_p90"] > _MAX_XY_RES_P90_M
                )
            ):
                reason = "spatial_residual"
            if self._pc_minus_rk is not None:
                offset_jump = fitted_offset - self._pc_minus_rk
                features["offset_jump_ms"] = offset_jump * 1000.0
                if (
                    reason is None
                    and abs(offset_jump) > _MAX_ACCEPTED_OFFSET_JUMP_S
                ):
                    reason = "offset_jump"
        if reason is not None:
            if defer_rejection:
                return None
            return self._record_terminal(
                reason,
                features=features,
                extra=extra,
                now_arrival_pc=now_arrival_pc,
                consume_tail=True,
            )

        self._pc_minus_rk = float(fitted_offset)
        self._updated_pc = pc_points[-1].t
        return self._record_terminal(
            "accepted",
            accepted=True,
            features=features,
            extra=extra,
            now_arrival_pc=now_arrival_pc,
            consume_tail=True,
        )

    def _record_terminal(
        self,
        reason: str,
        *,
        features: dict,
        now_arrival_pc: float,
        accepted: bool = False,
        extra: dict | None = None,
        consume_tail: bool,
    ) -> dict:
        candidate = self._candidate
        if candidate is None:
            raise RuntimeError("terminal without active joint throw")
        self._attempts += 1
        result = {
            "attempt": self._attempts,
            "pc_flight": candidate.sequence,
            "rk_flight": candidate.sequence,
            "accepted": bool(accepted),
            "reason": reason,
            "features": copy.deepcopy(features),
        }
        if extra:
            result.update(extra)
        if accepted:
            self._accepted += 1
            result["update"] = self._accepted
        else:
            self._rejected += 1
        if features.get("pc_state") == "complete":
            self._pc_flights += 1
        if features.get("rk_state") == "complete":
            self._rk_flights += 1
        if not accepted:
            self._reason_counts[reason] += 1
        self._history.append(copy.deepcopy(result))
        self._candidate = None
        self._clear_pair_buffers()
        if consume_tail:
            self._blocked_until_arrival_pc = (
                now_arrival_pc + _CONSUMED_TAIL_S
            )
        return copy.deepcopy(result)

    def current_offset(self) -> float | None:
        with self._lock:
            return self._pc_minus_rk

    @staticmethod
    def _interp_axis(
        points: tuple[_Point, ...],
        times: list[float],
        t: float,
        key: str,
    ) -> float | None:
        right = bisect.bisect_left(times, t)
        if right <= 0 or right >= len(points):
            return None
        left_point = points[right - 1]
        right_point = points[right]
        gap = right_point.t - left_point.t
        if gap <= 0.0 or gap > _MAX_FIT_GAP_S:
            return None
        fraction = (t - left_point.t) / gap
        left_value = getattr(left_point, key)
        return left_value + fraction * (getattr(right_point, key) - left_value)

    @classmethod
    def _interp_z(
        cls, points: tuple[_Point, ...], times: list[float], t: float
    ) -> float | None:
        return cls._interp_axis(points, times, t, "z")

    @classmethod
    def _score_xy(
        cls,
        pc_points: tuple[_Point, ...],
        rk_points: tuple[_Point, ...],
        pc_minus_rk: float,
    ) -> dict:
        pc_times = [point.t for point in pc_points]
        deltas = []
        matched_times = []
        for point in rk_points:
            t_pc = point.t + pc_minus_rk
            pc_x = cls._interp_axis(pc_points, pc_times, t_pc, "x")
            pc_y = cls._interp_axis(pc_points, pc_times, t_pc, "y")
            if pc_x is None or pc_y is None:
                continue
            deltas.append((pc_x - point.x, pc_y - point.y))
            matched_times.append(t_pc)
        if not deltas:
            return {
                "xy_n": 0,
                "xy_shift_x": None,
                "xy_shift_y": None,
                "xy_shift": None,
                "xy_res_median": None,
                "xy_res_p90": None,
            }
        # A constant PC-minus-RK displacement is the car's world-pose belief
        # error (RK world ball = onboard observation + bot pose), not wrong-
        # throw evidence; before the first accepted offset unblocks
        # /pc_car_loc that error has no way to shrink, so gate only the
        # de-translated residual shape.
        shift_x = _upper_median([dx for dx, _ in deltas])
        shift_y = _upper_median([dy for _, dy in deltas])
        residuals = sorted(
            math.hypot(dx - shift_x, dy - shift_y) for dx, dy in deltas
        )
        p90_index = min(
            len(residuals) - 1, math.ceil(0.9 * len(residuals)) - 1
        )
        return {
            "xy_n": len(residuals),
            "xy_shift_x": shift_x,
            "xy_shift_y": shift_y,
            "xy_shift": math.hypot(shift_x, shift_y),
            "xy_res_median": _upper_median(residuals),
            "xy_res_p90": residuals[p90_index],
            "xy_span": matched_times[-1] - matched_times[0],
        }

    @classmethod
    def _score(
        cls,
        pc_points: tuple[_Point, ...],
        rk_points: tuple[_Point, ...],
        pc_minus_rk: float,
    ) -> dict | None:
        pc_times = [point.t for point in pc_points]
        dz: list[float] = []
        matched_times: list[float] = []
        for point in rk_points:
            t_pc = point.t + pc_minus_rk
            pc_z = cls._interp_z(pc_points, pc_times, t_pc)
            if pc_z is None:
                continue
            dz.append(pc_z - point.z)
            matched_times.append(t_pc)
        if len(dz) < 5 or matched_times[-1] - matched_times[0] < 0.2:
            return None
        z_offset = _upper_median(dz)
        residuals = sorted(abs(value - z_offset) for value in dz)
        trim_n = max(1, math.ceil(0.9 * len(residuals)))
        trimmed = residuals[:trim_n]
        return {
            "err": _upper_median(residuals),
            "p90": residuals[trim_n - 1],
            "trimmed_rmse": math.sqrt(
                sum(value * value for value in trimmed) / len(trimmed)
            ),
            "n": len(dz),
            "span": matched_times[-1] - matched_times[0],
            "z_offset": z_offset,
        }

    @classmethod
    def _estimate(
        cls,
        pc_points: tuple[_Point, ...],
        rk_points: tuple[_Point, ...],
        baseline: float,
    ) -> dict:
        coarse: list[dict] = []
        for millisecond in range(-100, 101):
            delta = millisecond / 1000.0
            score = cls._score(pc_points, rk_points, baseline + delta)
            if score is not None:
                coarse.append({"delta": delta, **score})
        if not coarse:
            return {
                "pc_minus_rk": None,
                "delta_ms": None,
                "err": None,
                "p90": None,
                "trimmed_rmse": None,
                "n": 0,
                "coverage": 0.0,
                "span": None,
                "z_offset": None,
                "margin": None,
                "profile_width_ms": None,
                "edge": False,
                "usable": False,
            }

        best = min(coarse, key=lambda row: row["err"])
        fine_lo = max(-_MAX_OFFSET_DELTA_S, best["delta"] - 0.003)
        fine_hi = min(_MAX_OFFSET_DELTA_S, best["delta"] + 0.003)
        fine_start = math.ceil(fine_lo * 10000.0 - 1e-9)
        fine_end = math.floor(fine_hi * 10000.0 + 1e-9)
        for tick in range(fine_start, fine_end + 1):
            delta = tick / 10000.0
            score = cls._score(pc_points, rk_points, baseline + delta)
            if score is not None and score["err"] < best["err"]:
                best = {"delta": delta, **score}

        rivals = [
            row for row in coarse
            if abs(row["delta"] - best["delta"]) >= _RIVAL_GAP_S - 1e-12
        ]
        rival = min(rivals, key=lambda row: row["err"]) if rivals else None
        margin = (
            rival["err"] / max(1e-9, best["err"])
            if rival is not None else None
        )
        profile = [
            row["delta"] for row in coarse
            if row["err"] <= best["err"] + 0.005
        ]
        profile_width = max(profile) - min(profile) if profile else None
        max_n = max(row["n"] for row in coarse)
        coverage = best["n"] / max_n if max_n else 0.0
        edge = abs(best["delta"]) >= _MAX_OFFSET_DELTA_S - 0.00005
        usable = (
            best["err"] <= 0.025
            and best["p90"] <= 0.10
            and best["trimmed_rmse"] <= 0.04
            and best["n"] >= 8
            and coverage >= 0.60
            and best["span"] >= 0.25
            and not edge
            and margin is not None
            and margin >= 1.50
            and profile_width is not None
            and profile_width <= 0.02
        )
        return {
            "pc_minus_rk": baseline + best["delta"],
            "delta_ms": best["delta"] * 1000.0,
            "err": best["err"],
            "p90": best["p90"],
            "trimmed_rmse": best["trimmed_rmse"],
            "n": best["n"],
            "coverage": coverage,
            "span": best["span"],
            "z_offset": best["z_offset"],
            "margin": margin,
            "profile_width_ms": (
                profile_width * 1000.0 if profile_width is not None else None
            ),
            "edge": edge,
            "usable": usable,
        }

    @staticmethod
    def _rejection_reason(fit: dict) -> str | None:
        if fit.get("usable"):
            return None
        if fit.get("pc_minus_rk") is None or fit.get("n", 0) < 8:
            return "insufficient_overlap"
        if fit.get("edge"):
            return "search_edge"
        if (
            fit.get("err") is None
            or fit["err"] > 0.025
            or fit.get("p90") is None
            or fit["p90"] > 0.10
            or fit.get("trimmed_rmse") is None
            or fit["trimmed_rmse"] > 0.04
        ):
            return "shape_residual"
        if fit.get("coverage", 0.0) < 0.60 or (fit.get("span") or 0.0) < 0.25:
            return "insufficient_coverage"
        if fit.get("margin") is None or fit["margin"] < 1.50:
            return "ambiguous_rival"
        if (
            fit.get("profile_width_ms") is None
            or fit["profile_width_ms"] > 20.0
        ):
            return "wide_profile"
        return "quality_gate"

    def snapshot(self) -> dict:
        with self._lock:
            active_features = None
            if self._candidate is not None:
                active_features = self._candidate_features(self._candidate)
            return {
                "contract": "pc_exposure_t = rk_ball_world_t + pc_minus_rk_s",
                "topic": "/ball_world_topic",
                "incoming_cross_y_m": _CROSS_Y_M,
                "incoming_cross_bounce_margin_m": _CROSS_BOUNCE_MARGIN_M,
                "joint_arrival_gap_ms": _MAX_JOINT_ARRIVAL_GAP_S * 1000.0,
                "joint_spatial_gap_m": _MAX_JOINT_SPATIAL_GAP_M,
                "joint_gap_ms": _MAX_JOINT_GAP_S * 1000.0,
                "pc_minus_rk_s": self._pc_minus_rk,
                "updated_pc": self._updated_pc,
                "updates": self._accepted,
                "attempts": self._attempts,
                "rejected": self._rejected,
                "pc_complete_flights": self._pc_flights,
                "rk_complete_flights": self._rk_flights,
                "pending_pc_flights": int(
                    self._candidate is not None
                    and bool(self._candidate.samples["pc"])
                ),
                "pending_rk_flights": int(
                    self._candidate is not None
                    and bool(self._candidate.samples["rk"])
                ),
                "active_throw_features": active_features,
                "reason_counts": dict(self._reason_counts),
                "history": copy.deepcopy(self._history),
            }
