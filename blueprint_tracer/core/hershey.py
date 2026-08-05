"""Single-stroke (Hershey) text rendering.

Hershey fonts are defined as pen strokes rather than filled outlines, so a plotter
draws each glyph in one pass. That makes them the natural replacement for traced
lettering: a centerline trace of small type is always a wobbly approximation of
the shape, whereas a Hershey glyph is exactly what a pen can draw.

Glyphs are fitted to the region they replace by mapping the rendered text's own
bounding box onto it, so the result stays put whatever the font's internal metrics.
"""

from __future__ import annotations

import numpy as np

from blueprint_tracer.core.geometry import Path

DEFAULT_FONT = "futural"      # Hershey Simplex: the classic single-stroke roman
_FONT_CACHE: dict = {}


class HersheyUnavailable(RuntimeError):
    """The hershey-fonts package is not installed."""


def is_available() -> bool:
    try:
        import HersheyFonts  # noqa: F401
    except ImportError:
        return False
    return True


def available_fonts() -> list[str]:
    try:
        from HersheyFonts import HersheyFonts as _HF
    except ImportError:
        return []
    return sorted(_HF().default_font_names)


def _load(font: str):
    if font in _FONT_CACHE:
        return _FONT_CACHE[font]
    try:
        from HersheyFonts import HersheyFonts as _HF
    except ImportError as exc:
        raise HersheyUnavailable(
            "The hershey-fonts package is missing.\n"
            "Install the OCR extra:  uv pip install -e '.[ocr]'"
        ) from exc
    engine = _HF()
    engine.load_default_font(font)
    engine.normalize_rendering(100)
    _FONT_CACHE[font] = engine
    return engine


def text_strokes(text: str, font: str = DEFAULT_FONT) -> list[np.ndarray]:
    """Raw strokes for a string, in the font's own coordinate space."""
    if not text.strip():
        return []
    engine = _load(font)
    strokes = []
    for stroke in engine.strokes_for_text(text):
        pts = np.asarray(stroke, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes


def render_into(
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    font: str = DEFAULT_FONT,
    orientation: str = "horizontal",
    stroke_width: float = 1.0,
) -> list[Path]:
    """Draw ``text`` as single-stroke paths fitted into the given box.

    The glyphs keep their aspect ratio and are centred in the box, so replacing a
    traced label does not stretch the lettering to match an imperfect bounding box.
    """
    strokes = text_strokes(text, font)
    if not strokes:
        return []

    every = np.vstack(strokes)
    x0, y0 = float(every[:, 0].min()), float(every[:, 1].min())
    x1, y1 = float(every[:, 0].max()), float(every[:, 1].max())
    src_w = max(x1 - x0, 1e-6)
    src_h = max(y1 - y0, 1e-6)

    box_w, box_h = (height, width) if orientation == "vertical" else (width, height)
    scale = min(box_w / src_w, box_h / src_h)
    draw_w, draw_h = src_w * scale, src_h * scale
    off_x = (box_w - draw_w) / 2.0
    off_y = (box_h - draw_h) / 2.0

    paths: list[Path] = []
    for stroke in strokes:
        local = np.empty_like(stroke)
        local[:, 0] = (stroke[:, 0] - x0) * scale + off_x
        # Hershey glyphs are defined y-up; image space is y-down, so flip or every
        # letter comes out mirrored.
        local[:, 1] = (y1 - stroke[:, 1]) * scale + off_y

        if orientation == "vertical":
            # Rotate a quarter turn anticlockwise about the box, reading bottom-up.
            rotated = np.empty_like(local)
            rotated[:, 0] = local[:, 1]
            rotated[:, 1] = box_w - local[:, 0]
            local = rotated

        local[:, 0] += x
        local[:, 1] += y
        paths.append(Path(points=local.astype(np.float32), stroke_width=stroke_width))
    return paths


def render_region(region, font: str = DEFAULT_FONT, stroke_width: float = 1.0) -> list[Path]:
    """Render a :class:`TextRegion`'s text into its own bounding box."""
    paths = render_into(
        region.text, region.x, region.y, region.width, region.height,
        font=font, orientation=getattr(region, "orientation", "horizontal"),
        stroke_width=stroke_width,
    )
    for p in paths:
        p.region_id = region.id
    return paths
