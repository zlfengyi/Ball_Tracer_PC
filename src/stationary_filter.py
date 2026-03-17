from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace


TENNIS_BALL_LABEL = "tennis_ball"
STATIONARY_OBJECT_LABEL = "stationary_object"


@dataclass
class _DetectionSample:
    timestamp_s: float
    x: float
    y: float


class StationaryObjectFilter:
    """Classify per-camera detections as tennis balls or stationary objects."""

    def __init__(
        self,
        window_s: float = 15.0,
        radius_px: float = 2.0,
        min_occurrences: int = 6,
    ) -> None:
        self._window_s = window_s
        self._radius_sq = radius_px * radius_px
        self._min_occurrences = min_occurrences
        self._history: dict[str, deque[_DetectionSample]] = defaultdict(deque)

    def classify(
        self,
        serial: str,
        detections: list[object],
        timestamp_s: float,
    ) -> list[object]:
        """Return copies of detections with a `label` field set."""
        history = self._history[serial]
        self._prune(history, timestamp_s)

        classified: list[object] = []
        for det in detections:
            occurrences = 1 + self._count_matches(history, det.x, det.y)
            label = (
                STATIONARY_OBJECT_LABEL
                if occurrences >= self._min_occurrences
                else TENNIS_BALL_LABEL
            )
            classified.append(replace(det, label=label))
            history.append(_DetectionSample(timestamp_s=timestamp_s, x=det.x, y=det.y))

        return classified

    def _prune(self, history: deque[_DetectionSample], now_s: float) -> None:
        cutoff = now_s - self._window_s
        while history and history[0].timestamp_s < cutoff:
            history.popleft()

    def _count_matches(
        self,
        history: deque[_DetectionSample],
        x: float,
        y: float,
    ) -> int:
        matches = 0
        for sample in history:
            dx = sample.x - x
            dy = sample.y - y
            if dx * dx + dy * dy <= self._radius_sq:
                matches += 1
        return matches
