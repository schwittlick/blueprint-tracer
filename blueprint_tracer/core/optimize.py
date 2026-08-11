"""Post-trace optimization for pen plotting: spur pruning, collinear joining,
and pen-up travel ordering.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from blueprint_tracer.core.geometry import Path


def prune_spurs(paths: list[Path], spur_length: float, iterations: int) -> list[Path]:
    """Remove short dangling branches (leaf edges) attached at a junction.

    A path is a spur if one endpoint is a leaf (touched by only this path), the
    other is shared with something else, and it is shorter than ``spur_length``.
    Isolated short segments (both ends free) are kept.
    """
    if spur_length <= 0:
        return paths
    for _ in range(max(1, iterations)):
        counts: dict[tuple, int] = defaultdict(int)
        for p in paths:
            counts[_key(p.points[0])] += 1
            counts[_key(p.points[-1])] += 1

        kept, removed = [], False
        for p in paths:
            a, b = counts[_key(p.points[0])], counts[_key(p.points[-1])]
            leaf = a == 1 or b == 1
            isolated = a == 1 and b == 1
            if leaf and not isolated and p.length < spur_length:
                removed = True
                continue
            kept.append(p)
        paths = kept
        if not removed:
            break
    return paths


def join_collinear(paths: list[Path], max_angle_deg: float) -> list[Path]:
    """Fuse pairs of paths that meet at a shared endpoint and continue nearly
    straight, reducing the number of separate strokes (pen lifts).

    Only endpoints where exactly two path-ends meet are fused; genuine junctions
    (3+ ends) are left split.
    """
    paths = list(paths)
    min_dot = -np.cos(np.radians(max_angle_deg))  # dot of into-path tangents ~ -1 when straight
    changed = True
    while changed:
        changed = False
        ends: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for i, p in enumerate(paths):
            ends[_key(p.points[0])].append((i, 0))
            ends[_key(p.points[-1])].append((i, 1))

        consumed = [False] * len(paths)
        merged: list[Path] = []
        for _, lst in ends.items():
            live = [(i, e) for (i, e) in lst if not consumed[i]]
            if len(live) != 2:
                continue
            (i, ei), (j, ej) = live
            if i == j:
                continue
            ti = _end_tangent(paths[i].points, ei)
            tj = _end_tangent(paths[j].points, ej)
            if ti is None or tj is None:
                continue
            if float(np.dot(ti, tj)) > min_dot:
                continue  # too sharp a bend to be one stroke
            consumed[i] = consumed[j] = True
            merged.append(_merge(paths[i], ei, paths[j], ej))
            changed = True

        paths = [p for k, p in enumerate(paths) if not consumed[k]] + merged
    return paths


def order_for_plotting(paths: list[Path], start=(0.0, 0.0)) -> list[Path]:
    """Greedy nearest-neighbour ordering over path endpoints to cut pen-up travel.

    Paths are reversed as needed so the nearer end is drawn first. Runs in roughly
    O(n log n) using a KD-tree with an escalating-k lookup that skips used ends.
    """
    n = len(paths)
    if n < 2:
        return list(paths)

    endpoints = np.array([[p.points[0], p.points[-1]] for p in paths], dtype=np.float64)
    tree = cKDTree(endpoints.reshape(2 * n, 2))
    used = np.zeros(n, dtype=bool)

    order: list[Path] = []
    cur = np.asarray(start, dtype=np.float64)
    for _ in range(n):
        found, flip = None, False
        k = 1
        while found is None:
            k = min(k, 2 * n)
            _, idxs = tree.query(cur, k=k)
            for idx in np.atleast_1d(idxs):
                pi = int(idx) // 2
                if not used[pi]:
                    found, flip = pi, (int(idx) % 2 == 1)
                    break
            if found is None:
                if k >= 2 * n:
                    break
                k *= 2
        if found is None:
            break
        used[found] = True
        p = paths[found].reversed() if flip else paths[found]
        order.append(p)
        cur = p.points[-1].astype(np.float64)

    for i in range(n):  # safety: append anything the loop missed
        if not used[i]:
            order.append(paths[i])
    return order


def order_groups_for_plotting(groups: list[list[Path]], start=(0.0, 0.0)) -> list[Path]:
    """Order each group internally, keeping the groups themselves contiguous.

    A globally optimal tour interleaves everything, which is wrong when groups map
    to different pens: the plotter would swap pens at the carousel on almost every
    path and lose far more time than the shorter travel saves. Ordering within each
    group -- and starting each group from where the previous one ended, so the seam
    is not wasted -- keeps one pen change per group while still recovering most of
    the travel saving.
    """
    out: list[Path] = []
    cur = np.asarray(start, dtype=np.float64)
    for group in groups:
        if not group:
            continue
        ordered = order_for_plotting(group, start=cur)
        out.extend(ordered)
        cur = ordered[-1].points[-1].astype(np.float64)
    return out


def pen_up_travel(paths: list[Path], start=None) -> float:
    """Total pen-up (non-drawing) travel for the given path order.

    Only the moves *between* consecutive paths are counted, which is exactly what a
    consumer can recompute from the exported geometry. The approach from a home
    position is excluded unless ``start`` is given: no home position is recorded in
    the file, so including one would make the reported figure unverifiable.
    """
    if not paths:
        return 0.0
    total = 0.0
    if start is not None:
        total += float(np.hypot(*(paths[0].points[0] - np.asarray(start))))
    for a, b in zip(paths, paths[1:]):
        total += float(np.hypot(*(b.points[0] - a.points[-1])))
    return total


# --- helpers ---

def _key(pt) -> tuple[int, int]:
    return (int(round(float(pt[0]))), int(round(float(pt[1]))))


def _end_tangent(points: np.ndarray, end: int, span: int = 4):
    """Unit vector pointing from the given end into the path (None if degenerate)."""
    if len(points) < 2:
        return None
    if end == 0:
        v = points[min(span, len(points) - 1)] - points[0]
    else:
        v = points[-min(span, len(points) - 1) - 1] - points[-1]
    norm = np.hypot(v[0], v[1])
    if norm < 1e-6:
        return None
    return (v / norm).astype(np.float64)


def _merge(pi: Path, ei: int, pj: Path, ej: int) -> Path:
    """Concatenate two paths at a shared endpoint into one continuous stroke."""
    seq_i = pi.points if ei == 1 else pi.points[::-1]  # joint at tail
    seq_j = pj.points if ej == 0 else pj.points[::-1]  # joint at head
    pts = np.vstack([seq_i, seq_j[1:]])
    width = (pi.stroke_width * len(pi.points) + pj.stroke_width * len(pj.points)) / (
        len(pi.points) + len(pj.points)
    )
    return Path(points=pts, stroke_width=width, closed=False)
