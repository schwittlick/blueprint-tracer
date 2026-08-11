import numpy as np
import pytest

from blueprint_tracer.core.geometry import Path, polyline_length
from blueprint_tracer.core.optimize import (
    join_collinear,
    order_for_plotting,
    pen_up_travel,
    prune_spurs,
)
from blueprint_tracer.core.simplify import simplify_points


def test_path_length_and_reverse():
    p = Path(points=np.array([[0, 0], [3, 4]], dtype=np.float32))
    assert abs(p.length - 5.0) < 1e-5
    r = p.reversed()
    assert np.allclose(r.points[0], [3, 4])
    assert abs(r.length - 5.0) < 1e-5


def test_rdp_collapses_collinear_points():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32)
    out = simplify_points(pts, epsilon=0.5)
    assert len(out) == 2
    assert np.allclose(out[0], [0, 0]) and np.allclose(out[-1], [4, 0])


def test_join_collinear_fuses_two_straight_segments():
    a = Path(points=np.array([[0, 0], [5, 0]], dtype=np.float32))
    b = Path(points=np.array([[5, 0], [10, 0]], dtype=np.float32))
    out = join_collinear([a, b], max_angle_deg=20.0)
    assert len(out) == 1
    assert abs(out[0].length - 10.0) < 1e-4


def test_join_keeps_sharp_corner_split():
    a = Path(points=np.array([[0, 0], [5, 0]], dtype=np.float32))
    b = Path(points=np.array([[5, 0], [5, 5]], dtype=np.float32))  # 90 deg turn
    out = join_collinear([a, b], max_angle_deg=20.0)
    assert len(out) == 2


def test_prune_spurs_removes_short_leaf_branch():
    trunk = Path(points=np.array([[0, 0], [20, 0]], dtype=np.float32))
    branch2 = Path(points=np.array([[20, 0], [40, 0]], dtype=np.float32))
    spur = Path(points=np.array([[20, 0], [22, 2]], dtype=np.float32))  # short leaf
    out = prune_spurs([trunk, branch2, spur], spur_length=6.0, iterations=3)
    assert spur not in out
    assert trunk in out and branch2 in out


def test_ordering_reduces_pen_up_travel():
    # Three segments deliberately given in a bad order.
    a = Path(points=np.array([[0, 0], [1, 0]], dtype=np.float32))
    b = Path(points=np.array([[100, 0], [101, 0]], dtype=np.float32))
    c = Path(points=np.array([[2, 0], [3, 0]], dtype=np.float32))
    before = pen_up_travel([a, b, c])
    ordered = order_for_plotting([a, b, c])
    after = pen_up_travel(ordered)
    assert after <= before


def test_pen_up_travel_counts_only_moves_between_paths():
    """The reported figure must be recomputable from the exported geometry alone,
    so an approach from an unrecorded home position is not counted."""
    a = Path(points=np.array([[100, 0], [110, 0]], dtype=np.float32))
    b = Path(points=np.array([[150, 0], [160, 0]], dtype=np.float32))
    assert pen_up_travel([a, b]) == pytest.approx(40.0)      # 110 -> 150 only
    assert pen_up_travel([a, b], start=(0.0, 0.0)) == pytest.approx(140.0)


def test_order_groups_keeps_groups_contiguous():
    """Groups map to different pens, so they must not interleave however much
    travel a global tour would save."""
    from blueprint_tracer.core.optimize import order_groups_for_plotting

    # Two families deliberately interleaved in space.
    group_a = [Path(points=np.array([[i * 100, 0], [i * 100 + 10, 0]], dtype=np.float32),
                    stroke_width=3.0) for i in range(5)]
    group_b = [Path(points=np.array([[i * 100 + 50, 5], [i * 100 + 60, 5]], dtype=np.float32),
                    stroke_width=1.0) for i in range(5)]

    ordered = order_groups_for_plotting([group_a, group_b])
    assert len(ordered) == 10
    widths = [p.stroke_width for p in ordered]
    assert widths == sorted(widths, reverse=True), "groups interleaved"
    # Exactly one transition between the groups == one pen change.
    changes = sum(1 for x, y in zip(widths, widths[1:]) if x != y)
    assert changes == 1


def test_order_groups_beats_naive_concatenation():
    from blueprint_tracer.core.optimize import order_groups_for_plotting

    scattered = [
        Path(points=np.array([[x, y], [x + 5, y]], dtype=np.float32), stroke_width=1.0)
        for x, y in [(0, 0), (500, 400), (10, 5), (490, 380), (20, 0), (480, 400)]
    ]
    naive = pen_up_travel(scattered)
    grouped = pen_up_travel(order_groups_for_plotting([scattered]))
    assert grouped < naive
