"""Pure local-costmap readiness checks used by the episode goal gate."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any


class LocalCostmapReadiness:
    """Validate that a fresh rolling costmap contains the robot sensor origins."""

    def __init__(
        self,
        *,
        sensor_frames: Iterable[str],
        lookup_xy: Callable[[str, str], tuple[float, float] | None],
    ):
        self._sensor_frames = tuple(sensor_frames)
        self._lookup_xy = lookup_xy
        self._reset_monotonic = 0.0
        self._costmap = None
        self._received_monotonic = 0.0
        self.reason = "no-costmap"

    def reset(self) -> None:
        # Simulation time may rewind on reset; use a monotonic process clock for arrival order.
        self._reset_monotonic = time.monotonic()
        self._costmap = None
        self._received_monotonic = 0.0
        self.reason = "no-fresh-local-costmap"

    def receive(self, costmap: Any) -> None:
        self._costmap = costmap
        self._received_monotonic = time.monotonic()

    def ready(self) -> bool:
        costmap = self._costmap
        if costmap is None:
            self.reason = "no-fresh-local-costmap"
            return False

        if self._received_monotonic <= self._reset_monotonic:
            self.reason = "stale-local-costmap(callback predates reset boundary)"
            return False

        metadata = costmap.metadata
        frame_id = str(costmap.header.frame_id)
        if not frame_id:
            self.reason = "local-costmap-has-no-frame"
            return False

        min_x = float(metadata.origin.position.x)
        min_y = float(metadata.origin.position.y)
        max_x = min_x + float(metadata.size_x) * float(metadata.resolution)
        max_y = min_y + float(metadata.size_y) * float(metadata.resolution)
        margin = max(float(metadata.resolution), 1.0e-6)

        for sensor_frame in self._sensor_frames:
            xy = self._lookup_xy(frame_id, sensor_frame)
            if xy is None:
                self.reason = f"no-TF:{frame_id}->{sensor_frame}"
                return False
            x, y = xy
            if not (
                min_x + margin <= x <= max_x - margin
                and min_y + margin <= y <= max_y - margin
            ):
                self.reason = (
                    f"sensor-outside-local-costmap:{sensor_frame}="
                    f"({x:.2f},{y:.2f}) bounds=({min_x:.2f},{min_y:.2f})"
                    f"-({max_x:.2f},{max_y:.2f})"
                )
                return False

        self.reason = "ready"
        return True
