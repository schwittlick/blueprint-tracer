"""Topology tests for the skeleton tracer on hand-built skeletons."""

import numpy as np

from blueprint_tracer.core.trace import trace_skeleton


def _blank(h=20, w=20):
    return np.zeros((h, w), dtype=bool)


def test_single_horizontal_line():
    skel = _blank()
    skel[10, 3:15] = True
    chains = trace_skeleton(skel)
    assert len(chains) == 1
    ch = chains[0]
    # endpoints at (row=10, col=3) and (row=10, col=14)
    cols = sorted([ch[0][1], ch[-1][1]])
    assert cols == [3, 14]
    assert all(pt[0] == 10 for pt in ch)


def test_plus_junction_has_four_arms():
    skel = _blank()
    skel[10, 4:17] = True   # horizontal
    skel[4:17, 10] = True   # vertical
    chains = trace_skeleton(skel)
    # A '+' has one degree-4 node and four arms.
    assert len(chains) == 4
    for ch in chains:
        assert len(ch) >= 2


def test_closed_loop_is_detected():
    skel = _blank(24, 24)
    # a rectangle ring
    skel[6, 6:18] = True
    skel[17, 6:18] = True
    skel[6:18, 6] = True
    skel[6:18, 17] = True
    chains = trace_skeleton(skel)
    # Ring collapses to chain(s); the union of points covers the whole ring.
    covered = {tuple(p) for ch in chains for p in ch}
    ring = {tuple(p) for p in np.argwhere(skel)}
    assert ring.issubset(covered)


def test_isolated_pixel_is_dropped():
    skel = _blank()
    skel[5, 5] = True
    chains = trace_skeleton(skel)
    # A lone pixel has no edge; nothing >= 2 points should be produced.
    assert all(len(ch) >= 2 for ch in chains) or chains == []
