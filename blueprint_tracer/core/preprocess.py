"""Preprocessing: downscale/supersample, polarity normalization, flat-field, deskew,
ink shaping.

Output is always a uint8 grayscale image where ink is dark on a light background.
"""

from __future__ import annotations

import cv2
import numpy as np

from blueprint_tracer.core.config import Config


def preprocess(gray: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    """Return (processed_gray, applied_angle_degrees)."""
    g = gray
    if cfg.max_dim and max(g.shape) > cfg.max_dim:
        s = cfg.max_dim / max(g.shape)
        g = cv2.resize(
            g, (round(g.shape[1] * s), round(g.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )

    if cfg.supersample and cfg.supersample != 1.0:
        s = float(cfg.supersample)
        g = cv2.resize(
            g, (round(g.shape[1] * s), round(g.shape[0] * s)),
            interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA,
        )

    if cfg.invert_auto and is_inverted(g):
        g = 255 - g

    if cfg.flatfield:
        g = flatfield(g, cfg.flatfield_size or 25)

    if cfg.manual_angle is not None:
        angle = float(cfg.manual_angle)
    elif cfg.deskew:
        angle = estimate_skew(g, cfg.deskew_max_angle)
    else:
        angle = 0.0

    # Rotating resamples the image, which blurs thin ink; skip negligible skews.
    if abs(angle) >= cfg.deskew_min_angle:
        g = rotate_image(g, angle)
    else:
        angle = 0.0

    # Shaping runs last, on the image the binarizer will actually see: the flat-field
    # background estimate and the skew search both want undistorted ink, and putting
    # the smoothing after the rotation lets it absorb the resampling blur as well.
    if cfg.ink_morph or cfg.pre_smooth:
        g = shape_ink(g, cfg.ink_morph, cfg.pre_smooth)

    return g, angle


def shape_ink(gray: np.ndarray, morph_px: int, sigma: float) -> np.ndarray:
    """Thicken (``morph_px > 0``) or thin (``< 0``) the ink, then soften its edges.

    Thickening is what rescues a crisply drawn but thin original. A 1-2 px
    anti-aliased line has no choice but to binarize into a 1 px staircase, and the
    skeleton then faithfully traces every step of it -- the traced curve visibly
    wobbles and burns points on jitter. Widen the same line into a ribbon several
    pixels across and its medial axis averages those steps out. Raise ``supersample``
    first: without the extra resolution there is nothing to widen into, and the
    thickening only fuses neighbouring detail.

    Thinning is the opposite case -- over-inked photocopies and fax scans, where
    adjacent lines bleed into each other and would otherwise skeletonize as one.

    Ink is dark, so the grayscale operations are inverted relative to the mask-space
    ones in :mod:`~blueprint_tracer.core.cleanup`: eroding takes the local minimum
    and therefore *grows* dark ink.

    Both steps trade detail for smoothness, and the small stuff goes first: too much
    of either closes the counters of small lettering and merges tightly spaced
    features (bolt circles, hatching) into blobs.
    """
    g = gray
    k = abs(int(morph_px))
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        g = cv2.erode(g, kernel) if morph_px > 0 else cv2.dilate(g, kernel)
    if sigma and sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(sigma))
    return g


def is_inverted(gray: np.ndarray) -> bool:
    """True if this looks like light ink on a dark ground (a true cyanotype blueprint).

    Splits the histogram with Otsu and asks which side is the minority: ink is the
    sparse class on a drawing. A plain median test misfires on any scan with heavy
    dark borders or large filled regions, flipping a normal drawing and erasing it.
    """
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_fraction = float(np.count_nonzero(gray <= thresh)) / max(1, gray.size)
    return dark_fraction > 0.5


def flatfield(gray: np.ndarray, size: int) -> np.ndarray:
    """Divide out low-frequency illumination (uneven lighting, aged paper shading).

    The background is estimated with a grayscale morphological close, which fills
    in strokes thinner than the kernel. Features *wider* than the kernel survive
    into that estimate, and dividing an image by itself yields pure white -- which
    would silently erase thick borders, filled arrowheads and shaded title blocks.
    To prevent that, the estimate is smoothed and then floored at a fraction of the
    paper level, so the correction can only ever brighten toward the paper tone and
    never cancels real ink.
    """
    k = max(3, int(size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Illumination is low-frequency; blurring keeps any leaked-in ink from acting
    # as its own background.
    blur = max(3, (k * 2 + 1) | 1)
    bg = cv2.GaussianBlur(bg, (blur, blur), 0)

    paper = float(np.percentile(gray, 90))
    floor = max(1.0, 0.6 * paper)
    bg = np.maximum(bg.astype(np.float32), floor)

    out = cv2.divide(gray.astype(np.float32), bg, scale=255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


# Backwards-compatible alias for the previous private name.
_flatfield = flatfield


def estimate_skew(gray: np.ndarray, max_angle: float) -> float:
    """Estimate small rotation from the dominant near-axis line directions.

    Returns the angle (degrees) to pass to :func:`rotate_image` to level the drawing.
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    threshold = int(max(80, 0.25 * min(h, w)))
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold)
    if lines is None:
        return 0.0

    devs = []
    for rho, theta in lines[:, 0]:
        deg = np.degrees(theta) % 90.0
        if deg > 45.0:
            deg -= 90.0
        if abs(deg) <= max_angle:
            devs.append(deg)
    if not devs:
        return 0.0
    return float(np.median(devs))


def rotate_image(gray: np.ndarray, angle_deg: float, border: int = 255) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h),
        flags=cv2.INTER_CUBIC,  # cubic preserves thin strokes better than linear
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
