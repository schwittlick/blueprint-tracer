"""Skeletonization: ink mask -> 1px centerline + a stroke-width (distance) map."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (skeleton_bool, distance_map).

    ``distance_map[y, x]`` is the distance from an ink pixel to the nearest
    background pixel, i.e. roughly half the local stroke width. Sampling it along
    the skeleton gives per-stroke width.
    """
    skel = skeletonize(mask)
    dist = distance_transform_edt(mask)
    return skel.astype(bool), dist.astype(np.float32)
