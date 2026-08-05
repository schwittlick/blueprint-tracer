"""Image loading. Handles TIFF/JPG/PNG, RGBA/palette flattening, and DPI probing."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def load_gray(path: str) -> tuple[np.ndarray, Optional[float]]:
    """Load an image as a uint8 grayscale array, flattening transparency over white.

    Returns (gray, dpi). ``dpi`` is the horizontal DPI if the file records one.
    """
    im = Image.open(path)
    dpi = _read_dpi(im)

    has_alpha = im.mode in ("RGBA", "LA", "PA") or (
        im.mode == "P" and "transparency" in im.info
    )
    if has_alpha:
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        im = bg

    gray = np.asarray(im.convert("L"), dtype=np.uint8)
    return gray, dpi


def _read_dpi(im: Image.Image) -> Optional[float]:
    dpi = im.info.get("dpi")
    if dpi:
        try:
            x = float(dpi[0])
            if x > 1:  # PIL sometimes reports (1, 1) or (0, 0) when unknown
                return x
        except (TypeError, ValueError, IndexError):
            pass
    return None


def list_images(path: str) -> list[str]:
    """Expand a file or directory into a sorted list of image paths."""
    if os.path.isdir(path):
        out = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        return out
    return [path]
