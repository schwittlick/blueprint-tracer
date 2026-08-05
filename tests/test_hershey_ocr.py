"""Single-stroke text rendering, and OCR when Tesseract is available."""

import numpy as np
import pytest

from blueprint_tracer.core import hershey, ocr
from blueprint_tracer.core.text import TextRegion

pytestmark = pytest.mark.skipif(
    not hershey.is_available(), reason="hershey-fonts not installed"
)


def _extent(paths):
    pts = np.vstack([p.points for p in paths])
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def test_renders_single_stroke_paths_inside_the_box():
    paths = hershey.render_into("GROOVED CASING", 100, 50, 200, 20)
    assert paths, "no strokes rendered"
    assert all(not p.closed for p in paths), "single-stroke text must not be filled"
    x0, x1, y0, y1 = _extent(paths)
    assert 100 <= x0 and x1 <= 300
    assert 50 <= y0 and y1 <= 70


def test_glyphs_are_not_mirrored():
    """Hershey space is y-up and image space is y-down; forgetting the flip turns
    every letter upside down."""
    # 'L' is asymmetric vertically: its bar sits at the bottom in image space, so
    # the lower half must carry more ink than the upper half.
    paths = hershey.render_into("L", 0, 0, 40, 60)
    pts = np.vstack([p.points for p in paths])
    mid = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    assert (pts[:, 1] > mid).sum() > (pts[:, 1] < mid).sum(), "glyph is upside down"


def test_vertical_text_is_rotated_into_a_tall_box():
    paths = hershey.render_into("VERTICAL", 10, 10, 20, 200, orientation="vertical")
    x0, x1, y0, y1 = _extent(paths)
    assert x1 - x0 <= 20 + 1e-3, "vertical text overflows its narrow box"
    assert y1 - y0 > 40, "vertical text was not rotated"


def test_empty_text_renders_nothing():
    assert hershey.render_into("   ", 0, 0, 100, 20) == []


def test_render_region_tags_paths_with_their_region():
    region = TextRegion(id=7, x=5, y=5, width=120, height=18,
                        char_height=14.0, n_glyphs=6, text="CASING")
    paths = hershey.render_region(region)
    assert paths and all(p.region_id == 7 for p in paths)


def test_scaling_preserves_aspect_ratio():
    """A wide box must not stretch the lettering to fill it."""
    narrow = hershey.render_into("AB", 0, 0, 40, 20)
    wide = hershey.render_into("AB", 0, 0, 400, 20)
    nx0, nx1, ny0, ny1 = _extent(narrow)
    wx0, wx1, wy0, wy1 = _extent(wide)
    narrow_ratio = (nx1 - nx0) / max(ny1 - ny0, 1e-6)
    wide_ratio = (wx1 - wx0) / max(wy1 - wy0, 1e-6)
    assert narrow_ratio == pytest.approx(wide_ratio, rel=0.05)


@pytest.mark.skipif(not ocr.is_available(), reason="tesseract not installed")
def test_ocr_reads_rendered_text():
    import cv2

    if "eng" not in ocr.available_languages():
        pytest.skip("no English language data")

    page = np.full((120, 460), 255, dtype=np.uint8)
    cv2.putText(page, "GROOVED", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    region = TextRegion(id=0, x=15, y=30, width=250, height=55,
                        char_height=40.0, n_glyphs=7)

    result = ocr.recognize_region(page, region, lang="eng")
    assert "GROOVED" in result.text.upper()
    assert result.confidence > 0


def test_ocr_missing_language_reports_actionable_error():
    with pytest.raises(ocr.OcrUnavailable) as excinfo:
        ocr.check_ready("definitely_not_a_language")
    assert "definitely_not_a_language" in str(excinfo.value)
