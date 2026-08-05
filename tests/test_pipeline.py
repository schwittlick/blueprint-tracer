"""Smoke test: a synthetic drawing survives the whole pipeline into SVG + JSON."""

import numpy as np

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.pipeline import run
from blueprint_tracer.export.json_export import to_json
from blueprint_tracer.export.svg import to_svg


def _synthetic_drawing():
    img = np.full((200, 300), 255, dtype=np.uint8)
    img[50, 20:280] = 0            # horizontal line
    img[150, 20:280] = 0           # horizontal line
    img[50:150, 20] = 0            # vertical line
    img[50:150, 280] = 0           # vertical line
    import cv2

    cv2.circle(img, (150, 100), 30, 0, 2)  # a circle inside
    return img


def test_pipeline_runs_and_exports():
    img = _synthetic_drawing()
    cfg = Config(flatfield=False, deskew=False, rdp_epsilon=1.0)
    result = run(img, cfg, dpi=300.0)

    assert result.width == 300 and result.height == 200
    assert result.stats["n_paths"] > 0
    assert result.px_per_mm is not None

    svg = to_svg(result)
    assert svg.startswith("<svg") and "polyline" in svg

    doc = to_json(result, image_name="synthetic.png")
    assert doc["width_px"] == 300
    assert len(doc["paths"]) == result.stats["n_paths"]
    assert all("points" in p and "stroke_width_px" in p for p in doc["paths"])


def test_pipeline_deskew_sign():
    # A near-horizontal line rotated by +3 deg should come back close to horizontal.
    import cv2

    img = np.full((200, 400), 255, dtype=np.uint8)
    img[100, 20:380] = 0
    m = cv2.getRotationMatrix2D((200, 100), 3.0, 1.0)
    rotated = cv2.warpAffine(img, m, (400, 200), borderValue=255)

    cfg = Config(flatfield=False, deskew=True, plot_order=False)
    result = run(rotated, cfg)
    # Deskew should have detected a rotation of roughly +/-3 degrees.
    assert 1.0 < abs(result.angle) < 6.0
