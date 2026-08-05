"""Skeleton -> polylines.

Walks an 8-connected 1px skeleton into a set of polyline chains, splitting at
endpoints (degree 1) and junctions (degree >= 3).

8-connectivity inflates the neighbor count of pixels next to a crossing, so a
single logical junction shows up as a small blob of high-degree pixels. We label
those blobs into clusters and collapse each to one representative point, so a
'+' yields four arms rather than a spray of tiny junction edges. The skeleton is
padded by one pixel so the neighborhood lookup needs no bounds checks.

Chains are returned in (row, col) pixel space; the caller flips them to (x, y).
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import convolve

# 8-neighborhood offsets.
_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def trace_skeleton(skel: np.ndarray) -> list[np.ndarray]:
    """Return a list of (N, 2) int arrays of (row, col) points, one per chain."""
    skel = np.ascontiguousarray(skel.astype(bool))
    padded = np.pad(skel, 1)

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    degree = convolve(padded.astype(np.uint8), kernel, mode="constant") * padded
    node_mask = padded & (degree != 2)

    # Cluster adjacent junction/endpoint pixels; one representative point each.
    _, labels = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    reps = _cluster_reps(node_mask, labels)
    node_set = set(map(tuple, np.argwhere(node_mask)))

    def neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in _OFFSETS:
            if padded[r + dr, c + dc]:
                out.append((r + dr, c + dc))
        return out

    chains: list[list[tuple[int, int]]] = []
    stepped: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    emitted_from_cluster: set[int] = set()
    covered_nodes: set[tuple[int, int]] = set()
    guard_max = padded.size + 8

    # 1) Walk every edge leaving a node until it reaches another node.
    for node in node_set:
        node_label = labels[node]
        for nb in neighbors(*node):
            if (node, nb) in stepped:
                continue
            if nb in node_set and labels[nb] == node_label:
                continue  # same junction blob, not a real edge
            stepped.add((node, nb))
            chain = [node, nb]
            prev, cur = node, nb
            guard = 0
            while cur not in node_set:
                nxts = [p for p in neighbors(*cur) if p != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                stepped.add((cur, nxt))
                chain.append(nxt)
                prev, cur = cur, nxt
                guard += 1
                if guard > guard_max:
                    break
            stepped.add((chain[-1], chain[-2]))
            # A chain that never leaves one junction blob is internal to that
            # junction, not a stroke; emitting it would add a spurious micro-loop
            # and double-draw the junction.
            if all(labels[p] == node_label for p in chain):
                emitted_from_cluster.add(node_label)
                continue
            # Extend to the cluster representatives so arms of one junction share an
            # exact coordinate, without dropping the actual node pixels we walked.
            r0 = reps[labels[chain[0]]]
            r1 = reps[labels[chain[-1]]]
            if chain[0] != r0:
                chain.insert(0, r0)
            if chain[-1] != r1:
                chain.append(r1)
            emitted_from_cluster.add(labels[chain[0]])
            emitted_from_cluster.add(labels[chain[-1]])
            covered_nodes.update(chain)
            chains.append(chain)

    # 1b) Stubs shorter than the junction blob itself have no degree-2 interior, so
    # the walk above never emits them. Reconnect each stranded endpoint to its
    # cluster representative rather than silently dropping real ink.
    for endpoint in np.argwhere(padded & (degree == 1)):
        pt = (int(endpoint[0]), int(endpoint[1]))
        if pt in covered_nodes:
            continue
        rep = reps[labels[pt]]
        if rep != pt:
            chains.append([rep, pt])
            covered_nodes.add(pt)

    # 2) Pure loops (rings of only degree-2 pixels, no node to seed from).
    used: set[tuple[int, int]] = set()
    for ch in chains:
        used.update(ch)
    loop_set = set(map(tuple, np.argwhere(padded & (degree == 2)))) - used
    while loop_set:
        start = next(iter(loop_set))
        chain = [start]
        prev, cur = None, start
        guard = 0
        while True:
            cand = [p for p in neighbors(*cur) if p != prev]
            if not cand:
                break
            nxt = cand[0]
            if nxt == start:
                chain.append(start)
                break
            chain.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if cur not in loop_set or guard > len(loop_set) + 2:
                break
        loop_set.difference_update(chain)
        chains.append(chain)

    # Back to unpadded (row, col).
    out = []
    for ch in chains:
        arr = np.array(ch, dtype=np.int32) - 1
        out.append(arr)
    return out


def _cluster_reps(node_mask: np.ndarray, labels: np.ndarray) -> dict[int, tuple[int, int]]:
    """Representative pixel per node cluster: the member nearest its centroid."""
    coords_by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ys, xs = np.nonzero(node_mask)
    for r, c in zip(ys.tolist(), xs.tolist()):
        coords_by_label[int(labels[r, c])].append((r, c))
    reps: dict[int, tuple[int, int]] = {}
    for lab, pix in coords_by_label.items():
        arr = np.array(pix)
        centroid = arr.mean(axis=0)
        nearest = arr[np.argmin(((arr - centroid) ** 2).sum(axis=1))]
        reps[lab] = (int(nearest[0]), int(nearest[1]))
    return reps
