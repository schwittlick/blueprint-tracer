"""Pipeline configuration. Every tunable parameter the GUI will expose lives here.

Length-like parameters default to ``None`` meaning *derive from the measured
stroke width* (see :meth:`Config.resolve`). Scans range from crisp large-format
line-work to low-resolution photocopies, and a pixel constant that suits one
destroys the other -- 2 px text is faithfully traced only when the threshold
window and spur length are sized relative to the stroke.

Set any field explicitly to pin it; explicit values always win over auto-scaling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional


def _odd(n: float) -> int:
    """Round to the nearest odd integer >= 3 (OpenCV/skimage window sizes)."""
    i = int(round(n))
    if i % 2 == 0:
        i += 1
    return max(3, i)


@dataclass
class Config:
    # --- scale ---
    max_dim: int = 0                 # longest-edge cap in px; 0 = full resolution
    supersample: float = 1.0         # upscale before tracing (helps low-res text)
    auto_scale: bool = True          # derive None-valued params from stroke width

    # --- preprocess ---
    invert_auto: bool = True         # auto-detect white-on-dark blueprints and invert
    flatfield: bool = True           # correct uneven illumination / aged paper
    flatfield_size: Optional[int] = None   # background-estimate kernel; auto ~= 12x stroke
    deskew: bool = True
    deskew_max_angle: float = 10.0   # only correct skews within +/- this many degrees
    deskew_min_angle: float = 0.15   # below this, skip rotation (resampling blurs thin ink)
    manual_angle: Optional[float] = None   # override auto-deskew (degrees)

    # --- binarize ---
    threshold_method: str = "sauvola"      # "sauvola" | "adaptive" | "otsu"
    sauvola_window: Optional[int] = None   # auto ~= 4x stroke width
    sauvola_k: float = 0.08          # low keeps faint/thin strokes solid; high erodes them
    adaptive_block: Optional[int] = None
    adaptive_c: int = 10
    fill_solid: bool = True          # restore interiors of ink wider than the window

    # --- solid regions ---
    # Filled shapes (arrowheads, blacked-out labels, ink blots) are described by
    # their boundary; skeletonizing them yields a meaningless medial-axis squiggle.
    solid_mode: str = "outline"      # "outline" | "skeleton" | "ignore"
    solid_min_width: Optional[float] = None  # auto ~= 4x stroke width, min 8 px

    # --- text layer ---
    detect_text: bool = True         # locate lettering so it can be handled separately
    text_min_glyphs: int = 2         # a lone mark is not a word
    text_max_glyph_height: int = 0   # cap on character height in px; 0 = auto
    text_gap_ratio: float = 1.1      # reach along the line, in character heights
    text_line_ratio: float = 0.22    # reach across lines; keeps stacked lines apart

    # --- cleanup ---
    despeckle_min_area: Optional[int] = None  # auto ~= half a stroke-square
    close_gaps: int = 0              # morphological close kernel to bridge gaps; 0 = off
    remove_border: bool = False      # strip components touching the image edge (scan cruft)

    # --- trace / prune ---
    spur_length: Optional[float] = None   # auto ~= 1.5x stroke width
    prune_iterations: int = 3
    min_path_length: Optional[float] = None  # auto ~= 0.75x stroke width

    # --- simplify ---
    rdp_epsilon: Optional[float] = None   # auto ~= 0.25x stroke width

    # --- optimize (plotting) ---
    join_paths: bool = True
    join_max_angle: float = 20.0     # max bend (deg) to fuse two collinear strokes
    plot_order: bool = True          # reorder paths to minimize pen-up travel

    # --- output ---
    dpi: Optional[float] = None      # overrides DPI read from the image, if set

    def copy(self) -> "Config":
        return Config(**asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def resolve(self, stroke_width: float, image_shape: Optional[tuple] = None) -> "Config":
        """Return a copy with every ``None`` parameter filled in for this stroke width.

        Multipliers are chosen so a 4 px stroke reproduces the original hand-tuned
        constants, while a 2 px stroke (small text, low-res scans) gets
        proportionally tighter values instead of being pruned away.

        ``image_shape`` bounds the solid-region threshold against the page: a scan
        dominated by filled areas inflates the stroke estimate toward those areas,
        and without the bound nothing would ever be classified as solid.
        """
        c = self.copy()
        w = max(1.0, float(stroke_width))

        if not c.auto_scale:
            # Manual mode: fall back to the historical fixed defaults.
            defaults = {
                "sauvola_window": 25, "adaptive_block": 35, "despeckle_min_area": 8,
                "spur_length": 6.0, "min_path_length": 2.0, "rdp_epsilon": 1.0,
                "flatfield_size": 25, "solid_min_width": 12.0,
            }
            for key, val in defaults.items():
                if getattr(c, key) is None:
                    setattr(c, key, val)
            return c

        if c.sauvola_window is None:
            # Window must span ink plus surrounding paper; ~4x stroke, kept modest
            # because an oversized window washes out local contrast in dense text.
            c.sauvola_window = _odd(min(max(4.0 * w + 1.0, 9), 51))
        if c.adaptive_block is None:
            c.adaptive_block = _odd(min(max(6.0 * w + 1.0, 11), 61))
        if c.flatfield_size is None:
            # Must exceed the widest stroke so the background estimate erases ink.
            c.flatfield_size = _odd(min(max(12.0 * w, 15), 151))
        if c.despeckle_min_area is None:
            # A real mark covers at least about half a stroke-square.
            c.despeckle_min_area = int(max(3, round(0.5 * w * w)))
        if c.spur_length is None:
            # Skeleton artifacts (junction forks, arrowhead spikes) run ~1 stroke long.
            c.spur_length = round(1.5 * w, 3)
        if c.min_path_length is None:
            c.min_path_length = round(0.75 * w, 3)
        if c.rdp_epsilon is None:
            c.rdp_epsilon = round(0.25 * w, 3)
        if c.solid_min_width is None:
            # Well clear of bold lettering, which must stay centerlined...
            value = max(4.0 * w, 8.0)
            if image_shape:
                # ...but ink spanning a noticeable share of the page is a filled
                # area, whatever the stroke estimate says.
                value = min(value, 0.05 * min(image_shape[0], image_shape[1]))
            c.solid_min_width = round(max(value, 3.0), 3)
        return c
