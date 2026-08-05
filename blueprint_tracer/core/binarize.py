"""Binarization: grayscale (ink-dark) -> boolean ink mask (True == ink)."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from blueprint_tracer.core.config import Config


def binarize(gray: np.ndarray, cfg: Config) -> np.ndarray:
    method = cfg.threshold_method
    if method == "sauvola":
        window = max(3, (cfg.sauvola_window or 25) | 1)
        t = threshold_sauvola(gray, window_size=window, k=cfg.sauvola_k)
        mask = gray < t
    elif method == "adaptive":
        block = max(3, (cfg.adaptive_block or 35) | 1)
        binimg = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, cfg.adaptive_c,
        )
        mask = binimg > 0
    elif method == "otsu":
        _, binimg = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        mask = binimg > 0
    else:
        raise ValueError(f"unknown threshold_method: {method!r}")

    if cfg.fill_solid and method != "otsu":
        mask = fill_dark_interiors(mask, gray)
    return mask.astype(bool)


def fill_dark_interiors(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Restore the insides of ink regions wider than the threshold window.

    A local threshold compares each pixel to its surroundings, so the middle of a
    region wider than the window looks like its own background and drops out --
    filled arrowheads, solid bars and bold text come back as hollow outlines.

    Only background areas that are *enclosed* by detected ink and *dark* in the
    original image are filled. Enclosure rules out gray paper and shadows, which
    connect to the border; the darkness test rules out the counters of letters
    like O and P, which are enclosed but light and must stay open.
    """
    background = (~mask).astype(np.uint8)
    num, labels = cv2.connectedComponents(background, connectivity=4)
    if num <= 1:
        return mask

    border = np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ])
    open_regions = np.zeros(num, dtype=bool)
    open_regions[np.unique(border)] = True
    open_regions[0] = True  # label 0 is the ink itself

    enclosed = ~open_regions[labels] & ~mask
    if not enclosed.any():
        return mask

    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask | (enclosed & (gray <= thresh))
