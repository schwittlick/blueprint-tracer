"""Core geometry types.

Convention: image arrays are indexed ``[row, col]`` == ``[y, x]``. Path points
are stored as ``(x, y)`` float32 pairs so they map directly onto SVG/JSON output.
The (row, col) -> (x, y) flip happens once, at the tracing boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)  # identity equality: points are numpy arrays, and paths
class Path:            # move through list/set membership by object identity
    """A single traced stroke as an ordered polyline of ``(x, y)`` points."""

    points: np.ndarray  # (N, 2) float32, (x, y)
    stroke_width: float = 1.0
    closed: bool = False
    id: int = -1
    region_id: int = -1  # detected text region this stroke belongs to, or -1

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)

    @property
    def length(self) -> float:
        """Total polyline length in pixels (pen-down travel)."""
        if len(self.points) < 2:
            return 0.0
        d = np.diff(self.points, axis=0)
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def reversed(self) -> "Path":
        return Path(self.points[::-1].copy(), self.stroke_width, self.closed,
                    self.id, self.region_id)


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    d = np.diff(points, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())
