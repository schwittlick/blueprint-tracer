"""SVG export. One polyline per stroke, no fills (plotter-friendly)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from blueprint_tracer.core.pipeline import TraceResult


def to_svg(result: TraceResult, stroke: str = "#000000", uniform_width: Optional[float] = None) -> str:
    w, h = result.width, result.height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<g fill="none" stroke="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">',
    ]
    for p in result.paths:
        if len(p.points) < 2:
            continue
        pts = p.points
        # A closed ring must return to its start; emit the closing vertex if the
        # simplified geometry no longer carries it.
        if p.closed and not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[:1]])
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        sw = uniform_width if uniform_width is not None else max(p.stroke_width, 0.5)
        parts.append(f'<polyline points="{coords}" stroke-width="{sw:.2f}"/>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def write_svg(result: TraceResult, path: str, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_svg(result, **kwargs))
