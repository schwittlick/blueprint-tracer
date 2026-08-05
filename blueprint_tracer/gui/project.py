"""Project files: image reference, parameters, and manual vector edits.

Saving the edited geometry (not just the parameters) is what makes a session
resumable -- re-running the pipeline would discard hand corrections.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.geometry import Path

FORMAT_VERSION = 1


def save_project(path: str, image_path: str, cfg: Config, paths: list[Path]) -> None:
    doc = {
        "format": "blueprint-tracer-project",
        "version": FORMAT_VERSION,
        "image": os.path.abspath(image_path) if image_path else None,
        "config": asdict(cfg),
        "paths": [
            {
                "points": [[round(float(x), 3), round(float(y), 3)] for x, y in p.points],
                "stroke_width": round(float(p.stroke_width), 3),
                "closed": bool(p.closed),
            }
            for p in paths
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def load_project(path: str) -> tuple[str, Config, list[Path]]:
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
        )
        for p in doc.get("paths", [])
    ]
    return doc.get("image") or "", cfg, paths
