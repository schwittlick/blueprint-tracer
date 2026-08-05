"""Text region detection.

The detector is deliberately conservative: missing a label is harmless (it is
traced normally), while a false positive on real geometry would invite replacing
that geometry with lettering later on.
"""

import cv2
import numpy as np

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.geometry import Path
from blueprint_tracer.core.pipeline import run
from blueprint_tracer.core.text import TextRegion, assign_paths, detect_text_regions


def _sheet_with_labels():
    img = np.full((320, 520), 255, dtype=np.uint8)
    cv2.putText(img, "GROOVED CASING", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(img, "BASE PLUG", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return img


def test_detects_label_lines():
    regions = detect_text_regions(_sheet_with_labels() < 128, stroke_width=2.0)
    assert len(regions) >= 2, f"expected two label lines, got {len(regions)}"
    tops = sorted(r.y for r in regions)
    assert tops[0] < 70 and tops[-1] > 90, "regions do not match the two text rows"
    assert all(r.mode == "trace" for r in regions), "regions must default to trace"


def test_no_false_positives_on_plain_line_art():
    """Rules, borders and hatching must not be mistaken for lettering."""
    img = np.full((320, 520), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (490, 290), 0, 2)
    for x in range(60, 460, 18):                     # cross-hatching
        cv2.line(img, (x, 60), (x + 40, 260), 0, 1)
    cv2.line(img, (60, 300), (460, 300), 0, 1)       # a dimension rule

    regions = detect_text_regions(img < 128, stroke_width=2.0)
    assert not regions, f"line art produced {len(regions)} phantom text regions"


def test_regions_land_on_the_ink():
    """A bbox in the wrong coordinate frame would sit away from its glyphs."""
    img = _sheet_with_labels()
    mask = img < 128
    for region in detect_text_regions(mask, stroke_width=2.0):
        window = mask[region.y:region.y + region.height,
                      region.x:region.x + region.width]
        assert window.mean() > 0.02, "region contains almost no ink"


def test_assign_paths_tags_only_enclosed_strokes():
    region = TextRegion(id=0, x=0, y=0, width=50, height=20,
                        char_height=14.0, n_glyphs=4)
    inside = Path(points=np.array([[10, 5], [20, 12]], dtype=np.float32))
    outside = Path(points=np.array([[200, 200], [220, 210]], dtype=np.float32))
    straddling = Path(points=np.array([[40, 10], [300, 10]], dtype=np.float32))

    assign_paths([inside, outside, straddling], [region])
    assert inside.region_id == 0
    assert outside.region_id == -1
    assert straddling.region_id == -1, "a stroke leaving the box is not text"


def test_pipeline_exposes_regions_and_tags_paths():
    result = run(_sheet_with_labels(), Config(flatfield=False, deskew=False))
    assert result.stats["n_text_regions"] == len(result.text_regions)
    assert result.text_regions
    assert any(p.region_id >= 0 for p in result.paths), "no path tagged to a region"
    # Tagged paths must actually fall inside their region.
    for p in result.paths:
        if p.region_id < 0:
            continue
        region = next(r for r in result.text_regions if r.id == p.region_id)
        cx, cy = float(p.points[:, 0].mean()), float(p.points[:, 1].mean())
        assert region.contains(cx, cy, margin=2.0)


def test_detection_can_be_disabled():
    result = run(_sheet_with_labels(), Config(flatfield=False, deskew=False,
                                              detect_text=False))
    assert result.text_regions == []
    assert all(p.region_id == -1 for p in result.paths)


def test_region_serialization_roundtrip():
    region = TextRegion(id=3, x=10, y=20, width=80, height=16, char_height=14.0,
                        n_glyphs=7, orientation="vertical", text="CASING",
                        confidence=87.5, mode="hide")
    restored = TextRegion.from_dict(region.to_dict())
    assert restored == region
    # Unknown keys from a newer file must not break loading.
    payload = region.to_dict()
    payload["future_field"] = 1
    assert TextRegion.from_dict(payload).text == "CASING"


def test_edits_keep_region_id_so_strokes_cannot_escape_hiding():
    """An edited stroke must stay tagged, or it escapes its region and reappears."""
    from blueprint_tracer.gui.edit_tools import EditState

    tagged = Path(points=np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float32),
                  region_id=2)
    other = Path(points=np.array([[10, 0], [20, 0]], dtype=np.float32), region_id=2)

    st = EditState()
    st.set_paths([tagged, other])
    st.straighten({0})
    assert st.paths[-1].region_id == 2, "straighten dropped the region tag"

    st.set_paths([Path(points=np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float32),
                       region_id=2)])
    st.split(0, 1)
    assert all(p.region_id == 2 for p in st.paths), "split dropped the region tag"


def test_join_only_keeps_region_when_both_parts_agree():
    from blueprint_tracer.gui.edit_tools import EditState

    st = EditState()
    st.set_paths([
        Path(points=np.array([[0, 0], [10, 0]], dtype=np.float32), region_id=2),
        Path(points=np.array([[10, 0], [20, 0]], dtype=np.float32), region_id=2),
    ])
    st.join({0, 1})
    assert st.paths[-1].region_id == 2

    st.set_paths([
        Path(points=np.array([[0, 0], [10, 0]], dtype=np.float32), region_id=2),
        Path(points=np.array([[10, 0], [20, 0]], dtype=np.float32), region_id=-1),
    ])
    st.join({0, 1})
    assert st.paths[-1].region_id == -1, "merged stroke smuggled geometry into a region"


def test_assign_paths_picks_the_best_region_not_the_first():
    big = TextRegion(id=0, x=0, y=0, width=200, height=100,
                     char_height=14.0, n_glyphs=8)
    tight = TextRegion(id=1, x=10, y=10, width=40, height=20,
                       char_height=14.0, n_glyphs=3)
    stroke = Path(points=np.array([[15, 15], [30, 18], [45, 20]], dtype=np.float32))
    assign_paths([stroke], [big, tight])
    assert stroke.region_id == 1, "path went to the first match, not the tightest"


def test_region_from_dict_tolerates_a_truncated_entry():
    restored = TextRegion.from_dict({"id": 4, "text": "PLUG", "mode": "hide"})
    assert restored.id == 4 and restored.text == "PLUG" and restored.mode == "hide"


def test_explicit_max_glyph_height_is_not_clamped_away():
    from blueprint_tracer.core.text import _glyph_candidates

    img = np.full((200, 200), 255, dtype=np.uint8)
    cv2.putText(img, "A", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)  # ~40px tall
    mask = img < 128
    tall = max(h for _, _, _, h in _glyph_candidates(mask, 2.0, max_glyph_height=80)[0]
               or [(0, 0, 0, 0)])
    assert tall > 24, "an explicitly allowed glyph height was clamped away"
