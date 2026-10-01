"""Text region detection (no recognition).

Finding *where* the lettering is pays off on its own: text becomes a layer you can
toggle, and each region can be handled differently from the surrounding line-work.
It is also the only workable basis for OCR here -- running an engine over a whole
blueprint returns nothing, because page-layout analysis expects a document rather
than labels scattered through line art. Cropping a detected region and recognizing
it alone works far better.

The detector is deliberately conservative: a missed label is traced normally,
whereas a false positive would invite replacing real geometry with lettering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from blueprint_tracer.core.geometry import Path


@dataclass
class TextRegion:
    """A detected line of lettering, and what the user decided to do with it."""

    id: int
    x: int
    y: int
    width: int
    height: int
    char_height: float
    n_glyphs: int
    orientation: str = "horizontal"     # "horizontal" | "vertical"

    # Filled in by later phases; carried through export either way.
    text: str = ""
    confidence: float = -1.0
    mode: str = "trace"                 # "trace" | "hershey" | "hide"
    source: str = "detected"            # "detected" | "manual" (drawn by the user)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (self.x - margin <= px <= self.x + self.width + margin
                and self.y - margin <= py <= self.y + self.height + margin)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TextRegion":
        """Tolerant of both unknown and missing keys.

        A project file is the user's review work; one malformed region entry must
        not make the whole file — geometry and all — unopenable.
        """
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in d.items() if k in known}
        for required, default in (("id", -1), ("x", 0), ("y", 0),
                                  ("width", 0), ("height", 0),
                                  ("char_height", 0.0), ("n_glyphs", 0)):
            values.setdefault(required, default)
        return cls(**values)


def detect_text_regions(
    mask: np.ndarray,
    stroke_width: float = 2.0,
    min_glyphs: int = 2,
    max_glyph_height: int = 0,
    gap_ratio: float = 1.1,
    line_ratio: float = 0.22,
) -> list[TextRegion]:
    """Locate lines of lettering in a binary ink mask.

    Individual glyph-like components are found first, then merged into lines. Both
    steps key off the *measured* glyph height rather than fixed pixel sizes, so the
    same thresholds work across scan resolutions.
    """
    if mask is None or not mask.any():
        return []

    glyphs, heights = _glyph_candidates(mask, stroke_width, max_glyph_height)
    if len(heights) < min_glyphs:
        return []

    # Lettering on one sheet is near enough one size; drop anything far off the
    # dominant height so that dashes, bolt circles and hatch fragments fall away.
    median_h = float(np.median(heights))
    keep = [g for g, h in zip(glyphs, heights) if 0.55 * median_h <= h <= 1.9 * median_h]
    if len(keep) < min_glyphs:
        return []

    regions: list[TextRegion] = []
    for orientation in ("horizontal", "vertical"):
        regions.extend(
            _group_into_lines(keep, median_h, mask.shape, orientation, min_glyphs,
                              gap_ratio, line_ratio)
        )

    regions = _drop_overlaps(regions)
    for i, r in enumerate(regions):
        r.id = i
    return regions


def manual_region(x: float, y: float, width: float, height: float,
                  region_id: int = -1) -> TextRegion:
    """Build a region from a box the user drew around lettering the detector missed.

    The detector is deliberately conservative (see the module docstring), so the
    labels it declines to claim -- an isolated word, lettering wound into the
    line-work, a size far off the page's dominant one -- are exactly the ones worth
    marking by hand. Such a region behaves like any other from here on: it can be
    recognized, lettered in Hershey, or hidden.

    Orientation is read from the box's shape, since that is the only evidence
    available: a taller-than-wide box is a vertical label. The character height
    follows from the same reading, and is what fits Hershey lettering into the box.
    """
    x, y = int(round(x)), int(round(y))
    width, height = max(1, int(round(width))), max(1, int(round(height)))
    orientation = "vertical" if height > width else "horizontal"
    char_height = float(width if orientation == "vertical" else height)
    return TextRegion(
        id=region_id, x=x, y=y, width=width, height=height,
        char_height=char_height, n_glyphs=0, orientation=orientation,
        source="manual",
    )


def merge_regions(detected: list[TextRegion], manual: list[TextRegion],
                  scale: float = 1.0) -> list[TextRegion]:
    """Combine a fresh detection with the user's hand-drawn regions.

    Re-tracing rebuilds the detected regions from nothing, so the manual ones have
    to be carried across explicitly or every parameter tweak would erase them.
    ``scale`` converts them into the new page's pixels: supersampling and preview
    downscaling both change how many pixels the page has, and a box kept verbatim
    would land somewhere else on the drawing.

    Where the detector has since found the same lettering, the hand-drawn box wins
    -- it is the user's explicit statement about that ink, and keeping both would
    describe it twice.

    The manual regions are rescaled and renumbered in place: the caller holds the
    very objects the review panel is bound to, so replacing them with copies would
    strand the panel on regions no longer in the drawing.
    """
    kept: list[TextRegion] = []
    for r in manual:
        if scale != 1.0:
            r.x = int(round(r.x * scale))
            r.y = int(round(r.y * scale))
            r.width = max(1, int(round(r.width * scale)))
            r.height = max(1, int(round(r.height * scale)))
            r.char_height = r.char_height * scale
        kept.append(r)

    for r in detected:
        if any(_iou(r, m) > 0.35 or _containment(r, m) > 0.7 for m in kept):
            continue
        kept.append(r)

    kept.sort(key=lambda r: (r.y, r.x))
    for i, r in enumerate(kept):
        r.id = i
    return kept


def _glyph_candidates(
    mask: np.ndarray, stroke_width: float, max_glyph_height: int
) -> tuple[list[tuple], list[float]]:
    """Connected components whose shape is plausible for a single character."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    h_img, w_img = mask.shape

    lo = max(3.0, 2.0 * stroke_width)
    if max_glyph_height:
        hi = float(max_glyph_height)     # explicit settings win over auto-scaling
    else:
        hi = max(24.0, 18.0 * stroke_width)
        hi = min(hi, 0.15 * min(h_img, w_img))
    hi = max(hi, lo + 1.0)               # never let the band invert

    glyphs, heights = [], []
    for i in range(1, num):
        # Plain ints: OpenCV hands back numpy int32, which json cannot serialize
        # and which would leak all the way into exports and project files.
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if not (lo <= h <= hi):
            continue
        if w > 4.0 * h or h > 8.0 * w:       # rules out rules, leaders and dashes
            continue
        if w > hi * 2.5:
            continue
        fill = area / float(max(1, w * h))
        if fill < 0.08 or fill > 0.95:        # hollow boxes and solid blobs alike
            continue
        if x == 0 or y == 0 or x + w >= w_img or y + h >= h_img:
            continue
        glyphs.append((x, y, w, h))
        heights.append(float(h))
    return glyphs, heights


def _group_into_lines(
    glyphs: list[tuple],
    char_height: float,
    shape: tuple,
    orientation: str,
    min_glyphs: int,
    gap_ratio: float = 1.1,
    line_ratio: float = 0.22,
) -> list[TextRegion]:
    """Merge neighbouring glyphs into text lines by directional dilation.

    ``gap_ratio`` is how far, in character heights, the dilation reaches along the
    reading direction. Too small and a part code like ``C-294-127`` fragments at its
    widest gap; too large and neighbouring labels fuse, which costs more at the
    recognizer than the fragments did. The default suits well-separated labels;
    raise it only for sheets whose lettering is loosely spaced.
    """
    canvas = np.zeros(shape, dtype=np.uint8)
    for x, y, w, h in glyphs:
        canvas[y:y + h, x:x + w] = 1

    # Wide enough to bridge inter-character gaps, shallow enough that separate
    # lines of a multi-line label stay separate.
    span = max(2, int(round(char_height * gap_ratio)))
    thin = max(1, int(round(char_height * line_ratio)))
    ksize = (span, thin) if orientation == "horizontal" else (thin, span)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    merged = cv2.dilate(canvas, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    centres = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in glyphs]

    regions: list[TextRegion] = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        members = [
            g for g, (cx, cy) in zip(glyphs, centres)
            if labels[int(min(max(cy, 0), shape[0] - 1)), int(min(max(cx, 0), shape[1] - 1))] == i
        ]
        if len(members) < min_glyphs:
            continue

        if orientation == "horizontal" and w < h * 1.2:
            continue        # a stack of glyphs is not a horizontal line
        if orientation == "vertical" and h < w * 1.2:
            continue

        # Trim the dilation padding back to the glyphs themselves.
        gx0 = min(g[0] for g in members)
        gy0 = min(g[1] for g in members)
        gx1 = max(g[0] + g[2] for g in members)
        gy1 = max(g[1] + g[3] for g in members)
        regions.append(TextRegion(
            id=-1, x=int(gx0), y=int(gy0), width=int(gx1 - gx0), height=int(gy1 - gy0),
            char_height=float(char_height), n_glyphs=len(members),
            orientation=orientation,
        ))
    return regions


def _drop_overlaps(regions: list[TextRegion]) -> list[TextRegion]:
    """Keep the richer region when horizontal and vertical passes claim the same ink.

    Containment is tested as well as IoU: a small region nested inside a large one
    has a low IoU yet describes the same lettering twice, which would then make a
    path's owning region depend on iteration order.
    """
    ordered = sorted(regions, key=lambda r: (-r.n_glyphs, r.x, r.y))
    kept: list[TextRegion] = []
    for r in ordered:
        if any(_iou(r, k) > 0.35 or _containment(r, k) > 0.7 for k in kept):
            continue
        kept.append(r)
    return sorted(kept, key=lambda r: (r.y, r.x))


def _containment(a: TextRegion, b: TextRegion) -> float:
    """Overlap as a fraction of the smaller region."""
    inter = _intersection(a, b)
    smaller = min(a.width * a.height, b.width * b.height)
    return inter / float(smaller or 1)


def _intersection(a: TextRegion, b: TextRegion) -> int:
    ix = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    return ix * iy


def _iou(a: TextRegion, b: TextRegion) -> float:
    inter = _intersection(a, b)
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / float(union or 1)


def assign_paths(paths: list[Path], regions: list[TextRegion]) -> None:
    """Tag each path with the region holding most of its points (or -1).

    Done after all path processing, so joining and reordering cannot invalidate it.
    """
    for p in paths:
        p.region_id = -1
    if not regions:
        return

    for p in paths:
        pts = p.points
        if len(pts) == 0:
            continue
        lo_x, hi_x = float(pts[:, 0].min()), float(pts[:, 0].max())
        lo_y, hi_y = float(pts[:, 1].min()), float(pts[:, 1].max())

        best_id, best_score, best_area = -1, 0.6, None
        for r in regions:
            # Cheap bbox rejection. Testing the stroke's own extent (rather than its
            # centroid) keeps this consistent with the inside-fraction rule below.
            if hi_x < r.x or lo_x > r.x + r.width:
                continue
            if hi_y < r.y or lo_y > r.y + r.height:
                continue
            inside = (
                (pts[:, 0] >= r.x) & (pts[:, 0] <= r.x + r.width)
                & (pts[:, 1] >= r.y) & (pts[:, 1] <= r.y + r.height)
            )
            score = float(inside.mean())
            area = r.width * r.height
            # Prefer the region holding most of the stroke; on a tie the tighter one.
            if score > best_score or (
                score == best_score and best_area is not None and area < best_area
            ):
                best_id, best_score, best_area = r.id, score, area
        p.region_id = best_id
