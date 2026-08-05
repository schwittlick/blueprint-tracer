"""Polyline simplification (Douglas-Peucker via OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np


def simplify_points(points: np.ndarray, epsilon: float, closed: bool = False) -> np.ndarray:
    """Reduce vertex count while keeping the shape within ``epsilon`` pixels.

    Endpoints are always preserved. A ``closed`` ring is simplified in closed mode
    and re-closed afterwards, so ``points[0] == points[-1]`` still holds -- running
    the open algorithm over a ring drops the duplicated closing vertex and leaves a
    visible gap that grows with radius.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if epsilon <= 0 or len(points) < 3:
        return points

    if closed:
        ring = points[:-1] if np.array_equal(points[0], points[-1]) else points
        if len(ring) < 3:
            return points
        approx = cv2.approxPolyDP(ring.reshape(-1, 1, 2), float(epsilon), True)
        out = approx.reshape(-1, 2).astype(np.float32)
        if len(out) < 3:
            return points
        return np.vstack([out, out[:1]])  # restore the closing vertex

    approx = cv2.approxPolyDP(points.reshape(-1, 1, 2), float(epsilon), False)
    out = approx.reshape(-1, 2).astype(np.float32)
    if len(out) < 2:  # degenerate guard: keep first and last
        out = points[[0, -1]]
    return out
