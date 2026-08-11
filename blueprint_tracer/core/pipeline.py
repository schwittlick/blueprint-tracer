"""End-to-end pipeline orchestration: grayscale image -> traced paths."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from blueprint_tracer.core import analyze as _analyze
from blueprint_tracer.core import binarize as _binarize
from blueprint_tracer.core import cleanup as _cleanup
from blueprint_tracer.core import optimize as _opt
from blueprint_tracer.core import preprocess as _preprocess
from blueprint_tracer.core import skeleton as _skeleton
from blueprint_tracer.core import solids as _solids
from blueprint_tracer.core import text as _text
from blueprint_tracer.core import trace as _trace
from blueprint_tracer.core.config import Config
from blueprint_tracer.core.geometry import Path
from blueprint_tracer.core.simplify import simplify_points


@dataclass
class TraceResult:
    """Geometry lives in *processed* image space.

    Preprocessing may downscale, supersample and rotate the page, so processed
    pixels -- not source pixels -- are the only frame in which the paths, the
    reported ``width``/``height`` and ``gray`` all agree. ``dpi`` is rescaled to
    match, so physical (mm) measurements stay correct at any internal scale.
    """

    paths: list[Path]
    width: int
    height: int
    dpi: Optional[float] = None
    angle: float = 0.0
    stats: dict = field(default_factory=dict)
    gray: Optional[np.ndarray] = None  # the preprocessed image the paths align to
    text_regions: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)  # intermediate images when requested

    @property
    def px_per_mm(self) -> Optional[float]:
        return self.dpi / 25.4 if self.dpi else None


def run(
    gray: np.ndarray,
    cfg: Config,
    dpi: Optional[float] = None,
    debug: bool = False,
) -> TraceResult:
    t0 = time.perf_counter()

    # Length-like parameters are multiples of the drawing's stroke width, so measure
    # it first and resolve the config against it (see Config.resolve).
    stroke_width = _analyze.estimate_stroke_width(gray)
    scale = float(cfg.supersample or 1.0)
    cfg = cfg.resolve(stroke_width * scale, image_shape=gray.shape)

    proc, angle = _preprocess.preprocess(gray, cfg)
    mask = _binarize.binarize(proc, cfg)
    mask = _cleanup.cleanup(mask, cfg)

    # Solid shapes are described by their boundary, not a centerline; peel them
    # off so only genuine strokes reach the skeletonizer.
    solid_mask = None
    if cfg.solid_mode in ("outline", "ignore"):
        stroke_mask, solid_mask = _solids.split_solid(mask, cfg.solid_min_width or 12.0)
        if solid_mask.any():
            mask = stroke_mask

    regions: list = []
    if cfg.detect_text:
        # The mask is in processed pixels, which max_dim may have shrunk relative to
        # the source the stroke width was measured on. Scale to match, or the glyph
        # size band is wrong for every preview.
        px_scale = mask.shape[1] / gray.shape[1] if gray.shape[1] else 1.0
        regions = _text.detect_text_regions(
            mask, stroke_width * px_scale,
            min_glyphs=cfg.text_min_glyphs,
            max_glyph_height=cfg.text_max_glyph_height,
            gap_ratio=cfg.text_gap_ratio,
            line_ratio=cfg.text_line_ratio,
        )

    skel, dist = _skeleton.skeletonize_mask(mask)

    h, w = proc.shape
    chains = _trace.trace_skeleton(skel)
    paths = _build_paths(chains, dist, w, h)

    solid_paths = 0
    if cfg.solid_mode == "outline" and solid_mask is not None and solid_mask.any():
        outlines = _solids.contour_paths(solid_mask)
        solid_paths = len(outlines)
        paths.extend(outlines)

    paths = _opt.prune_spurs(paths, cfg.spur_length, cfg.prune_iterations)
    if cfg.join_paths:
        paths = _opt.join_collinear(paths, cfg.join_max_angle)

    for p in paths:
        p.points = simplify_points(p.points, cfg.rdp_epsilon, closed=p.closed)

    if cfg.min_path_length > 0:
        paths = [p for p in paths if p.length >= cfg.min_path_length or p.closed]

    # Identify the strokes *before* the plotting permutation. Numbering them
    # afterwards would just restate their array position, leaving the exported id
    # and plot_order as the same 0..n-1 run and losing the trace order entirely.
    for i, p in enumerate(paths):
        p.id = i

    pen_up_before = _opt.pen_up_travel(paths)
    if cfg.plot_order:
        paths = _opt.order_for_plotting(paths)
    pen_up_after = _opt.pen_up_travel(paths)

    # Assign after ordering and joining, so neither can invalidate the mapping.
    if regions:
        _text.assign_paths(paths, regions)

    stats = {
        "n_paths": len(paths),
        "n_points": int(sum(len(p.points) for p in paths)),
        "ink_length_px": float(sum(p.length for p in paths)),
        "pen_up_before_px": float(pen_up_before),
        "pen_up_after_px": float(pen_up_after),
        # Says outright whether paths are in plotting sequence, rather than leaving
        # a consumer to infer it from the travel figures.
        "plot_ordered": bool(cfg.plot_order),
        "stroke_width_px": round(stroke_width, 2),
        "resolved": {
            "sauvola_window": cfg.sauvola_window,
            "sauvola_k": cfg.sauvola_k,
            "spur_length": cfg.spur_length,
            "rdp_epsilon": cfg.rdp_epsilon,
            "despeckle_min_area": cfg.despeckle_min_area,
            "min_path_length": cfg.min_path_length,
            "solid_min_width": cfg.solid_min_width,
            "supersample": scale,
        },
        "solid_mode": cfg.solid_mode,
        "n_solid_regions": int(solid_paths),
        "n_text_regions": len(regions),
        "elapsed_s": time.perf_counter() - t0,
    }

    # Geometry is in processed pixels, so scale DPI by the same factor to keep
    # px_per_mm (and therefore any millimetre readout) physically correct.
    source_dpi = cfg.dpi or dpi
    effective_dpi = None
    if source_dpi:
        px_scale = w / gray.shape[1] if gray.shape[1] else 1.0
        effective_dpi = source_dpi * px_scale

    result = TraceResult(
        paths=paths,
        width=w,
        height=h,
        dpi=effective_dpi,
        angle=angle,
        stats=stats,
        gray=proc,
        text_regions=regions,
    )
    if debug:
        result.debug = {"gray": proc, "mask": mask, "skeleton": skel}
    return result


def geometry_stats(paths: list[Path]) -> dict:
    """Measure the geometry as it stands.

    Anything that recomposes the path list -- dropping hidden regions, adding
    Hershey lettering -- must refresh these, or the figures describe a drawing that
    was never written and cannot be checked against the file.
    """
    return {
        "n_paths": len(paths),
        "n_points": int(sum(len(p.points) for p in paths)),
        "ink_length_px": float(sum(p.length for p in paths)),
        "pen_up_after_px": float(_opt.pen_up_travel(paths)),
    }


def _build_paths(chains, dist: np.ndarray, w: int, h: int) -> list[Path]:
    """Turn (row, col) chains into (x, y) Paths, sampling stroke width from dist."""
    paths: list[Path] = []
    for ch in chains:
        if len(ch) < 2:
            continue
        rows = np.clip(ch[:, 0], 0, h - 1)
        cols = np.clip(ch[:, 1], 0, w - 1)
        width = 2.0 * float(np.median(dist[rows, cols]))
        xy = np.column_stack([cols, rows]).astype(np.float32)  # (row,col) -> (x,y)
        closed = bool(np.array_equal(ch[0], ch[-1]))
        paths.append(Path(points=xy, stroke_width=max(width, 0.5), closed=closed))
    return paths
