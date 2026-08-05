"""Regression tests for defects found in adversarial review.

Each test encodes a bug that was reproduced against the real code, so a future
refactor cannot silently reintroduce it.
"""

import cv2
import numpy as np

from blueprint_tracer.core.analyze import estimate_stroke_width
from blueprint_tracer.core.config import Config
from blueprint_tracer.core.pipeline import run
from blueprint_tracer.core.preprocess import flatfield, is_inverted
from blueprint_tracer.core.simplify import simplify_points
from blueprint_tracer.core.trace import trace_skeleton
from blueprint_tracer.export.svg import to_svg


def test_flatfield_keeps_features_wider_than_kernel():
    """A wide faint feature must survive; morphological close copies it into the
    background estimate, and a naive divide would erase it to paper white."""
    img = np.full((80, 120), 245, dtype=np.uint8)
    img[20:60, 30:61] = 90  # 31px wide, wider than the 25px kernel
    out = flatfield(img, 25)
    assert out[40, 45] < 220, f"wide feature erased to {out[40, 45]}"
    assert out[5, 5] > 230, "paper should stay bright"


def test_flatfield_still_corrects_illumination():
    grad = np.tile(np.linspace(150, 255, 200).astype(np.uint8), (100, 1))
    img = grad.copy()
    img[40:60, 20:23] = 40  # a thin dark stroke in the dim region
    out = flatfield(img, 25)
    # Paper should flatten out across the gradient...
    assert abs(int(out[5, 5]) - int(out[5, 190])) < 40
    # ...while the stroke stays clearly darker than its surroundings.
    assert int(out[50, 21]) < int(out[50, 60]) - 40


def test_polarity_detection_survives_dim_paper():
    """A dark-ink drawing on dim/gray paper must not be flipped. A plain
    ``median < 110`` test inverts it and erases every stroke."""
    dim = np.full((200, 200), 95, dtype=np.uint8)
    dim[100, 20:180] = 0
    dim[20:180, 100] = 0
    assert np.median(dim) < 110, "precondition: this fools a plain median test"
    assert not is_inverted(dim)


def test_polarity_detection_ignores_heavy_dark_border():
    img = np.full((200, 200), 240, dtype=np.uint8)
    img[:20, :] = 0
    img[-20:, :] = 0
    img[:, :20] = 0
    img[:, -20:] = 0
    img[100, 40:160] = 0
    assert not is_inverted(img)


def test_polarity_detection_flips_true_blueprint():
    blueprint = np.full((200, 200), 40, dtype=np.uint8)  # light ink on dark ground
    blueprint[100, 20:180] = 230
    assert is_inverted(blueprint)


def test_rdp_keeps_closed_rings_closed():
    """Simplifying a ring in open mode drops the closing vertex and leaves a gap
    that grows with radius."""
    for radius in (5, 12, 30, 80):
        t = np.linspace(0, 2 * np.pi, 200)
        ring = np.column_stack([
            radius * np.cos(t) + 100, radius * np.sin(t) + 100
        ]).astype(np.float32)
        ring[-1] = ring[0]
        out = simplify_points(ring, epsilon=1.0, closed=True)
        gap = float(np.hypot(*(out[0] - out[-1])))
        assert gap < 1e-4, f"radius {radius} left a {gap:.2f}px gap"
        assert len(out) >= 4


def test_svg_closes_closed_paths():
    img = np.full((160, 160), 255, dtype=np.uint8)
    cv2.circle(img, (80, 80), 40, 0, 3)
    result = run(img, Config(flatfield=False, deskew=False))
    closed = [p for p in result.paths if p.closed]
    assert closed, "expected at least one closed ring from a circle"

    svg = to_svg(result)
    for line in svg.splitlines():
        if "polyline" not in line:
            continue
        coords = line.split('points="')[1].split('"')[0].split()
        if len(coords) > 3 and coords[0] == coords[-1]:
            break
    else:
        raise AssertionError("no closed polyline emitted for the traced circle")


def test_trace_emits_no_spurious_micro_loops_at_corners():
    """Junction-internal chains must not be emitted as tiny closed loops."""
    skel = np.zeros((40, 40), dtype=bool)
    skel[20, 5:35] = True   # horizontal
    skel[5:35, 20] = True   # vertical
    chains = trace_skeleton(skel)
    tiny_loops = [
        c for c in chains
        if len(c) > 2 and tuple(c[0]) == tuple(c[-1]) and len(c) < 5
    ]
    assert not tiny_loops, f"emitted {len(tiny_loops)} spurious micro-loops"
    assert len(chains) == 4, f"a '+' should yield 4 arms, got {len(chains)}"


def test_solid_regions_are_not_hollowed_out():
    """A filled arrowhead/bar wider than the threshold window must stay solid.
    Plain Sauvola sees its interior as its own background and returns an outline."""
    from blueprint_tracer.core.binarize import binarize

    img = np.full((160, 160), 250, dtype=np.uint8)
    img[40:120, 40:120] = 25          # a solid block far wider than the window
    img[140, 10:150] = 25             # plus a thin faint stroke

    cfg = Config(sauvola_window=11, sauvola_k=0.08, fill_solid=True)
    mask = binarize(img, cfg)
    interior = mask[70:90, 70:90]
    assert interior.all(), "solid block interior was hollowed out"
    assert mask[140, 80], "thin stroke lost"

    hollow = binarize(img, Config(sauvola_window=11, sauvola_k=0.08, fill_solid=False))
    assert not hollow[70:90, 70:90].all(), "precondition: plain Sauvola hollows this"


def test_fill_solid_keeps_letter_counters_open():
    """The enclosed-but-light inside of an 'O' must not be filled, or the ring
    collapses to a blob and the letter is destroyed."""
    from blueprint_tracer.core.binarize import binarize

    img = np.full((120, 120), 250, dtype=np.uint8)
    cv2.circle(img, (60, 60), 30, 30, 4)  # a ring: dark stroke, light interior
    mask = binarize(img, Config(sauvola_window=11, sauvola_k=0.08, fill_solid=True))
    assert not mask[55:65, 55:65].any(), "letter counter was filled in"


def test_fill_solid_ignores_gray_paper():
    """A gray background patch touches the border and must never become ink."""
    from blueprint_tracer.core.binarize import binarize

    img = np.full((150, 150), 250, dtype=np.uint8)
    img[:70, :] = 190          # a gray smudge running to the image edge
    img[100, 20:130] = 20      # real ink
    mask = binarize(img, Config(sauvola_window=11, sauvola_k=0.08, fill_solid=True))
    assert mask[:60, :].mean() < 0.05, "gray paper was marked as ink"
    assert mask[100, 75], "real ink lost"


def test_solid_region_is_outlined_not_skeletonized():
    """A filled blob has no meaningful centerline: skeletonizing it produces a
    branching medial axis that looks nothing like the shape."""
    img = np.full((240, 300), 250, dtype=np.uint8)
    cv2.rectangle(img, (70, 90), (230, 150), 20, -1)  # a solid bar, 160 x 60

    outlined = run(img, Config(flatfield=False, deskew=False, solid_mode="outline"))
    rings = [p for p in outlined.paths if p.closed]
    assert rings, "solid bar produced no outline"
    # The outline must hug the bar's edges rather than run down its middle.
    ring = max(rings, key=lambda p: p.length)
    on_edge = (np.abs(ring.points[:, 1] - 90) < 6) | (np.abs(ring.points[:, 1] - 150) < 6)
    assert on_edge.mean() > 0.5, "outline does not follow the bar's boundary"

    skeletonized = run(img, Config(flatfield=False, deskew=False, solid_mode="skeleton"))
    midline = [
        p for p in skeletonized.paths
        if abs(float(p.points[:, 1].mean()) - 120) < 12 and p.length > 40
    ]
    assert midline, "precondition: skeleton mode collapses the bar to a medial axis"


def test_solid_mode_ignore_drops_blobs_but_keeps_strokes():
    img = np.full((220, 260), 250, dtype=np.uint8)
    cv2.circle(img, (60, 110), 40, 20, -1)   # solid blob
    cv2.line(img, (130, 40), (240, 40), 20, 2)  # thin stroke

    result = run(img, Config(flatfield=False, deskew=False, solid_mode="ignore"))
    near_blob = [
        p for p in result.paths
        if np.hypot(p.points[:, 0] - 60, p.points[:, 1] - 110).mean() < 50
    ]
    near_line = [p for p in result.paths if p.points[:, 1].mean() < 60]
    assert not near_blob, "blob should be dropped in ignore mode"
    assert near_line, "thin stroke must survive"


def test_thin_strokes_are_never_treated_as_solid():
    """Ordinary line-work and lettering must keep their centerlines."""
    img = np.full((200, 300), 250, dtype=np.uint8)
    cv2.line(img, (20, 100), (280, 100), 20, 3)
    cv2.line(img, (150, 20), (150, 180), 20, 3)

    result = run(img, Config(flatfield=False, deskew=False, solid_mode="outline"))
    assert result.stats["n_solid_regions"] == 0
    # A centerlined cross is open paths, not rings.
    assert not [p for p in result.paths if p.closed]


def test_auto_scale_adapts_to_stroke_width():
    thin = Config().resolve(2.0)
    thick = Config().resolve(8.0)
    assert thin.spur_length < thick.spur_length
    assert thin.sauvola_window < thick.sauvola_window
    assert thin.rdp_epsilon < thick.rdp_epsilon
    # A 2px stroke must not be pruned by a spur length tuned for heavy line-work.
    assert thin.spur_length <= 4.0


def test_auto_scale_disabled_uses_fixed_defaults():
    cfg = Config(auto_scale=False).resolve(2.0)
    assert cfg.sauvola_window == 25
    assert cfg.spur_length == 6.0
    assert cfg.rdp_epsilon == 1.0


def test_estimate_stroke_width_matches_drawn_width():
    """Compare against the width actually rasterized, which OpenCV rounds up from
    the nominal thickness."""
    for nominal in (2, 4, 9):
        img = np.full((200, 200), 255, dtype=np.uint8)
        cv2.line(img, (20, 100), (180, 100), 0, nominal)
        cv2.line(img, (100, 20), (100, 180), 0, nominal)
        actual = int((img[:, 60] == 0).sum())
        est = estimate_stroke_width(img)
        # Erring high is deliberate (see estimate_stroke_width): never underestimate.
        assert actual <= est <= actual + 1.5, f"drew {actual}px, estimated {est}"
