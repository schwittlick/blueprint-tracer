"""Rasterize traced paths to a PNG-able array for visual QA (no SVG renderer dep)."""

from __future__ import annotations

import cv2
import numpy as np

from blueprint_tracer.core.pipeline import TraceResult


def render_paths(result: TraceResult, thickness: int = 1, scale: float = 1.0) -> np.ndarray:
    """Draw every path in black on white. Returns a uint8 grayscale image."""
    w = max(1, int(round(result.width * scale)))
    h = max(1, int(round(result.height * scale)))
    img = np.full((h, w), 255, dtype=np.uint8)
    for p in result.paths:
        if len(p.points) < 2:
            continue
        pts = (p.points * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], p.closed, 0, thickness, cv2.LINE_AA)
    return img
