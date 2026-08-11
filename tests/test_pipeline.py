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


def _plate():
    import cv2
    img = np.full((260, 380), 255, dtype=np.uint8)
    # Scattered strokes, so a good plotting order differs from the trace order.
    for i in range(9):
        x = 20 + (i * 137) % 330
        y = 20 + (i * 91) % 210
        cv2.line(img, (x, y), (x + 28, y + 18), 0, 2)
    return img


def test_id_is_a_permutation_not_the_array_index():
    """id must identify the stroke, plot_order its position. Numbering paths after
    the plotting permutation collapses both onto the array index and throws the
    trace order away."""
    from blueprint_tracer.export.json_export import to_json

    result = run(_plate(), Config(flatfield=False, deskew=False, plot_order=True))
    doc = to_json(result, image_name="p.png")
    ids = [p["id"] for p in doc["paths"]]
    orders = [p["plot_order"] for p in doc["paths"]]
    n = len(doc["paths"])
    assert n > 3, "need several strokes for ordering to matter"

    assert orders == list(range(n)), "plot_order must be the position in the sequence"
    assert sorted(ids) == list(range(n)), "id must be a permutation of 0..n-1"
    assert ids != orders, "id and plot_order collapsed onto the array index"


def test_exported_order_actually_reduces_pen_up_travel():
    """The permutation has to encode the optimisation, not merely differ from it."""
    import math

    from blueprint_tracer.export.json_export import to_json

    result = run(_plate(), Config(flatfield=False, deskew=False, plot_order=True))
    doc = to_json(result, image_name="p.png")

    def travel(sequence):
        total, cur = 0.0, (0.0, 0.0)
        for path in sequence:
            pts = path["points"]
            total += math.dist(cur, pts[0])
            cur = tuple(pts[-1])
        return total

    as_exported = travel(doc["paths"])
    as_traced = travel(sorted(doc["paths"], key=lambda p: p["id"]))
    assert as_exported < as_traced, "exported order is no better than trace order"
    assert doc["stats"]["plot_ordered"] is True


def test_unordered_export_reports_itself_as_unordered():
    from blueprint_tracer.export.json_export import to_json

    result = run(_plate(), Config(flatfield=False, deskew=False, plot_order=False))
    doc = to_json(result, image_name="p.png")
    assert doc["stats"]["plot_ordered"] is False
    # With no permutation applied the two fields legitimately coincide.
    assert [p["id"] for p in doc["paths"]] == [p["plot_order"] for p in doc["paths"]]
