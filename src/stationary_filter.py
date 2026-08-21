from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import math


TENNIS_BALL_LABEL = "tennis_ball"
STATIONARY_OBJECT_LABEL = "stationary_object"


@dataclass
class _DetectionSample:
    timestamp_s: float
    x: float
    y: float
    cell: tuple[int, int]


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
        self._cell_size = max(radius_px, 1.0)
        self._min_occurrences = min_occurrences
        self._history: dict[str, deque[_DetectionSample]] = defaultdict(deque)
        self._grid: dict[
            str,
            defaultdict[tuple[int, int], deque[_DetectionSample]],
        ] = defaultdict(lambda: defaultdict(deque))

    def classify(
        self,
        serial: str,
        detections: list[object],
        timestamp_s: float,
    ) -> list[object]:
        """Return copies of detections with a `label` field set."""
        history = self._history[serial]
        grid = self._grid[serial]
        self._prune(history, grid, timestamp_s)

        classified: list[object] = []
        for det in detections:
            cell = self._cell(det.x, det.y)
            matches_needed = max(self._min_occurrences - 1, 0)
            label = (
                STATIONARY_OBJECT_LABEL
                if matches_needed == 0
                or self._count_matches(grid, cell, det.x, det.y, matches_needed)
                >= matches_needed
                else TENNIS_BALL_LABEL
            )
            classified.append(replace(det, label=label))
            sample = _DetectionSample(
                timestamp_s=timestamp_s,
                x=det.x,
                y=det.y,
                cell=cell,
            )
            history.append(sample)
            grid[cell].append(sample)

        return classified

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor(x / self._cell_size),
            math.floor(y / self._cell_size),
        )

    def _prune(
        self,
        history: deque[_DetectionSample],
        grid: defaultdict[tuple[int, int], deque[_DetectionSample]],
        now_s: float,
    ) -> None:
        cutoff = now_s - self._window_s
        while history and history[0].timestamp_s < cutoff:
            sample = history.popleft()
            bucket = grid[sample.cell]
            bucket.popleft()
            if not bucket:
                del grid[sample.cell]

    def _count_matches(
        self,
        grid: defaultdict[tuple[int, int], deque[_DetectionSample]],
        cell: tuple[int, int],
        x: float,
        y: float,
        limit: int,
    ) -> int:
        matches = 0
        cell_x, cell_y = cell
        for grid_y in range(cell_y - 1, cell_y + 2):
            for grid_x in range(cell_x - 1, cell_x + 2):
                for sample in grid.get((grid_x, grid_y), ()):
                    dx = sample.x - x
                    dy = sample.y - y
                    if dx * dx + dy * dy <= self._radius_sq:
                        matches += 1
                        if matches >= limit:
                            return matches
        return matches
