"""Separating solid filled regions from stroke line-work.

Skeletonizing a solid region yields its medial axis -- a branching squiggle that
looks nothing like the shape. A filled arrowhead, a blacked-out label or an ink
blot is described by its *boundary*, not its centerline, so those regions are
contoured instead and only the genuine strokes are skeletonized.
"""

from __future__ import annotations

import cv2
import numpy as np

from blueprint_tracer.core.geometry import Path


def split_solid(mask: np.ndarray, min_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Split an ink mask into (stroke_mask, solid_mask).

    A morphological opening keeps exactly the area covered by disks of radius
    ``min_width / 2`` that fit inside the ink, which is empty for a thin stroke and
    close to the full shape for a blob. Solid pixels are then removed from the
    stroke mask, so an arrow keeps a centerlined shaft and an outlined head.
    """
    radius = max(1, int(round(min_width / 2.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    solid = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    if not solid.any():
        return mask.astype(bool), np.zeros_like(mask, dtype=bool)

    # The opening erodes the rim away; grow it back, clipped to the real ink.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    solid = cv2.dilate(solid, grow) & mask.astype(np.uint8)

    solid_bool = solid.astype(bool)
    return (mask.astype(bool) & ~solid_bool), solid_bool


def contour_paths(solid: np.ndarray, epsilon: float = 0.0) -> list[Path]:
    """Trace the outlines (outer boundaries and holes) of solid regions."""
    if not solid.any():
        return []
    contours, _ = cv2.findContours(
        solid.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    paths: list[Path] = []
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float32)  # OpenCV gives (x, y)
        if len(pts) < 3:
            continue
        if epsilon > 0:
            approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), float(epsilon), True)
            pts = approx.reshape(-1, 2).astype(np.float32)
            if len(pts) < 3:
                continue
        pts = np.vstack([pts, pts[:1]])  # close the ring
        paths.append(Path(points=pts, stroke_width=1.0, closed=True))
    return paths
