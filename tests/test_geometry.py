import numpy as np

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
