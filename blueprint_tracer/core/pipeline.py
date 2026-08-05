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
from blueprint_tracer.core import trace as _trace
from blueprint_tracer.core.config import Config
from blueprint_tracer.core.geometry import Path
from blueprint_tracer.core.simplify import simplify_points


@dataclass
class TraceResult:
    paths: list[Path]
    width: int
    height: int
    dpi: Optional[float] = None
    angle: float = 0.0
    stats: dict = field(default_factory=dict)
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
    cfg = cfg.resolve(stroke_width * scale)

    proc, angle = _preprocess.preprocess(gray, cfg)
    mask = _binarize.binarize(proc, cfg)
    mask = _cleanup.cleanup(mask, cfg)
    skel, dist = _skeleton.skeletonize_mask(mask)

    h, w = proc.shape
    chains = _trace.trace_skeleton(skel)
    paths = _build_paths(chains, dist, w, h)

    paths = _opt.prune_spurs(paths, cfg.spur_length, cfg.prune_iterations)
    if cfg.join_paths:
        paths = _opt.join_collinear(paths, cfg.join_max_angle)

    for p in paths:
        p.points = simplify_points(p.points, cfg.rdp_epsilon, closed=p.closed)

    if cfg.min_path_length > 0:
        paths = [p for p in paths if p.length >= cfg.min_path_length or p.closed]

    # Supersampling is an internal fidelity trick: report geometry in source pixels.
    if scale != 1.0:
        inv = 1.0 / scale
        for p in paths:
            p.points = (p.points * inv).astype(np.float32)
            p.stroke_width *= inv
        w = int(round(w * inv))
        h = int(round(h * inv))

    pen_up_before = _opt.pen_up_travel(paths)
    if cfg.plot_order:
        paths = _opt.order_for_plotting(paths)
    pen_up_after = _opt.pen_up_travel(paths)

    for i, p in enumerate(paths):
        p.id = i

    stats = {
        "n_paths": len(paths),
        "n_points": int(sum(len(p.points) for p in paths)),
        "ink_length_px": float(sum(p.length for p in paths)),
        "pen_up_before_px": float(pen_up_before),
        "pen_up_after_px": float(pen_up_after),
        "stroke_width_px": round(stroke_width, 2),
        "resolved": {
            "sauvola_window": cfg.sauvola_window,
            "sauvola_k": cfg.sauvola_k,
            "spur_length": cfg.spur_length,
            "rdp_epsilon": cfg.rdp_epsilon,
            "despeckle_min_area": cfg.despeckle_min_area,
            "min_path_length": cfg.min_path_length,
            "supersample": scale,
        },
        "elapsed_s": time.perf_counter() - t0,
    }

    result = TraceResult(
        paths=paths,
        width=w,
        height=h,
        dpi=cfg.dpi or dpi,
        angle=angle,
        stats=stats,
    )
    if debug:
        result.debug = {"gray": proc, "mask": mask, "skeleton": skel}
    return result


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
