"""Project files: image reference, parameters, and manual vector edits.

Saving the edited geometry (not just the parameters) is what makes a session
resumable -- re-running the pipeline would discard hand corrections.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.geometry import Path
from blueprint_tracer.core.text import TextRegion

FORMAT_VERSION = 1


def _plain(value):
    """numpy scalars are not JSON serializable; coerce them to built-ins."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_project(path: str, image_path: str, cfg: Config, paths: list[Path],
                 regions: Optional[list] = None) -> None:
    doc = {
        "format": "blueprint-tracer-project",
        "version": FORMAT_VERSION,
        "image": os.path.abspath(image_path) if image_path else None,
        "config": asdict(cfg),
        # Region decisions are part of the review work, not derivable from a re-trace.
        "text_regions": [r.to_dict() for r in (regions or [])],
        "paths": [
            {
                "points": [[round(float(x), 3), round(float(y), 3)] for x, y in p.points],
                "stroke_width": round(float(p.stroke_width), 3),
                "closed": bool(p.closed),
                "region_id": int(getattr(p, "region_id", -1)),
            }
            for p in paths
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, default=_plain)


def load_project(path: str) -> tuple[str, Config, list[Path], list[TextRegion]]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != "blueprint-tracer-project":
        raise ValueError("not a blueprint-tracer project file")

    known = set(Config.field_names())
    cfg = Config(**{k: v for k, v in (doc.get("config") or {}).items() if k in known})

    paths = [
        Path(
            points=np.asarray(p["points"], dtype=np.float32),
            stroke_width=float(p.get("stroke_width", 1.0)),
            closed=bool(p.get("closed", False)),
            region_id=int(p.get("region_id", -1)),
        )
        for p in doc.get("paths", [])
    ]
    regions = []
    for entry in doc.get("text_regions", []):
        try:
            regions.append(TextRegion.from_dict(entry))
        except (TypeError, ValueError):
            continue   # skip a corrupt region rather than lose the whole project
    return doc.get("image") or "", cfg, paths, regions
