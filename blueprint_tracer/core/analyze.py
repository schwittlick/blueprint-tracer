"""Image analysis used to auto-scale pipeline parameters.

Every length-like parameter in the pipeline (threshold window, spur length, RDP
epsilon, ...) is really a multiple of the drawing's stroke width. Scans differ
wildly in resolution, so measuring that width and deriving the parameters from it
is far more robust than fixed pixel constants.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Largest area we bother analyzing; bigger images are sampled down for speed.
_ANALYZE_MAX_PIXELS = 4_000_000


def estimate_stroke_width(gray: np.ndarray, default: float = 2.0) -> float:
    """Estimate the median ink stroke width (px) of a dark-on-light drawing.

    Otsu-binarizes, skeletonizes, and reads the distance transform along the
    skeleton: at a centerline pixel that distance is half the local stroke width.
    """
    g = gray
    scale = 1.0
    if g.size > _ANALYZE_MAX_PIXELS:
        scale = float(np.sqrt(_ANALYZE_MAX_PIXELS / g.size))
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _, binimg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = binimg > 0
    if not mask.any():
        return default

    dist = distance_transform_edt(mask)
    skel = skeletonize(mask)
    # Overestimates odd widths by 1px. Erring high is deliberate: an underestimate
    # shrinks the threshold window below the stroke and hollows out solid ink.
    widths = 2.0 * dist[skel]
    if widths.size == 0:
        return default

    w = float(np.median(widths)) / scale
    if not np.isfinite(w) or w <= 0:
        return default
    return float(np.clip(w, 1.0, 64.0))


def ink_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / max(1, mask.size)
