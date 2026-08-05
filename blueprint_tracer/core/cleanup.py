"""Mask cleanup: gap closing, scan-border removal, despeckle (all via OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from blueprint_tracer.core.config import Config


def cleanup(mask: np.ndarray, cfg: Config) -> np.ndarray:
    m = mask.astype(np.uint8)
    if cfg.close_gaps > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.close_gaps, cfg.close_gaps)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if cfg.remove_border:
        m = remove_edge_components(m)
    if cfg.despeckle_min_area > 0:
        m = despeckle(m, cfg.despeckle_min_area)
    return m.astype(bool)


def despeckle(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # component 0 is the background
    return keep[labels].astype(np.uint8)


def remove_edge_components(mask: np.ndarray) -> np.ndarray:
    """Drop connected components whose bounding box touches the image edge.

    Scanner black borders hug the extreme edge, while a drawing's own frame sits
    inside a white margin, so this separates them without harming real content.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    h, w = mask.shape
    keep = np.ones(num, dtype=bool)
    keep[0] = False
    for i in range(1, num):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        right = left + stats[i, cv2.CC_STAT_WIDTH]
        bottom = top + stats[i, cv2.CC_STAT_HEIGHT]
        if left == 0 or top == 0 or right == w or bottom == h:
            keep[i] = False
    return keep[labels].astype(np.uint8)
