"""Raw-geometry JSON export (the canonical machine-readable output)."""

from __future__ import annotations

import json
import os
from typing import Optional

from blueprint_tracer.core.pipeline import TraceResult


def to_json(result: TraceResult, image_name: Optional[str] = None, round_to: int = 2) -> dict:
    doc = {
        "image": image_name,
        "width_px": result.width,
        "height_px": result.height,
        "dpi": result.dpi,
        "px_per_mm": result.px_per_mm,
        "deskew_angle_deg": round(result.angle, 4),
        "stats": result.stats,
        "paths": [],
    }
    for p in result.paths:
        doc["paths"].append(
            {
                "id": p.id,
                "points": [[round(float(x), round_to), round(float(y), round_to)] for x, y in p.points],
                "stroke_width_px": round(float(p.stroke_width), 3),
                "length_px": round(float(p.length), 3),
                "closed": bool(p.closed),
                "plot_order": p.id,
            }
        )
    return doc


def write_json(result: TraceResult, path: str, image_name: Optional[str] = None) -> None:
    if image_name is None:
        image_name = os.path.basename(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(result, image_name=image_name), f, indent=1)
