"""Vector editing operations plus a delta-based undo stack.

Undo stores only the paths an operation removed and added, not a snapshot of the
whole drawing -- a 15k-path sheet would otherwise cost megabytes per step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from blueprint_tracer.core.geometry import Path


@dataclass
class EditState:
    """The editable set of paths and its undo/redo history."""

    paths: list[Path] = field(default_factory=list)
    _undo: list[tuple[str, list, list]] = field(default_factory=list, repr=False)
    _redo: list[tuple[str, list, list]] = field(default_factory=list, repr=False)
    max_depth: int = 100

    def set_paths(self, paths: list[Path]) -> None:
        self.paths = paths
        self._undo.clear()
        self._redo.clear()

    # --- history ---

    def _record(self, label: str, removed: list[tuple[int, Path]], added: list[Path]) -> None:
        self._undo.append((label, removed, added))
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str:
        if not self._undo:
            return ""
        label, removed, added = self._undo.pop()
        for p in added:
            if p in self.paths:
                self.paths.remove(p)
        for idx, p in sorted(removed, key=lambda t: t[0]):
            self.paths.insert(min(idx, len(self.paths)), p)
        self._redo.append((label, removed, added))
        return label

    def redo(self) -> str:
        if not self._redo:
            return ""
        label, removed, added = self._redo.pop()
        for _, p in removed:
            if p in self.paths:
                self.paths.remove(p)
        self.paths.extend(added)
        self._undo.append((label, removed, added))
        return label

    def snapshot_for_edit(self, indices: list[int], label: str) -> None:
        """Record an in-place mutation (e.g. node drag) as remove+add of copies."""
        removed = [(i, copy.deepcopy(self.paths[i])) for i in indices if 0 <= i < len(self.paths)]
        added = [self.paths[i] for i in indices if 0 <= i < len(self.paths)]
        # Reverting restores the copies; redoing restores the mutated originals.
        self._undo.append((label, removed, added))
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
        self._redo.clear()

    # --- operations ---

    def delete(self, indices: set[int]) -> str:
        removed = [(i, self.paths[i]) for i in sorted(indices) if 0 <= i < len(self.paths)]
        if not removed:
            return ""
        for _, p in removed:
            self.paths.remove(p)
        self._record("delete", removed, [])
        return f"deleted {len(removed)} path(s)"

    def join(self, indices: set[int], tolerance: float = 12.0) -> str:
        """Join selected paths end-to-end, nearest endpoints first."""
        sel = [i for i in sorted(indices) if 0 <= i < len(self.paths)]
        if len(sel) < 2:
            return "select at least two paths to join"
        removed = [(i, self.paths[i]) for i in sel]
        chain = [self.paths[i] for i in sel]

        merged = chain.pop(0)
        while chain:
            best, best_d, best_cfg = None, None, None
            for j, cand in enumerate(chain):
                for a_end, a_pt in ((0, merged.points[0]), (1, merged.points[-1])):
                    for b_end, b_pt in ((0, cand.points[0]), (1, cand.points[-1])):
                        d = float(np.hypot(*(a_pt - b_pt)))
                        if best_d is None or d < best_d:
                            best, best_d, best_cfg = j, d, (a_end, b_end)
            cand = chain.pop(best)
            a_end, b_end = best_cfg
            first = merged.points if a_end == 1 else merged.points[::-1]
            second = cand.points if b_end == 0 else cand.points[::-1]
            # Drop the shared vertex, or the join leaves a zero-length segment that
            # confuses tangent maths and wastes a plotter move.
            if float(np.hypot(*(first[-1] - second[0]))) <= 1e-3:
                second = second[1:]
            pts = np.vstack([first, second])
            # Keep the region only while every part agrees, so a merged stroke
            # cannot smuggle unrelated geometry into a hidden text region.
            region = merged.region_id if merged.region_id == cand.region_id else -1
            merged = Path(points=pts, stroke_width=merged.stroke_width,
                          region_id=region)

        for _, p in removed:
            self.paths.remove(p)
        self.paths.append(merged)
        self._record("join", removed, [merged])
        return f"joined {len(sel)} paths"

    def split(self, index: int, point_index: int) -> str:
        if not (0 <= index < len(self.paths)):
            return ""
        p = self.paths[index]
        if not (0 < point_index < len(p.points) - 1):
            return "pick an interior node to split at"
        a = Path(points=p.points[: point_index + 1].copy(),
                 stroke_width=p.stroke_width, region_id=p.region_id)
        b = Path(points=p.points[point_index:].copy(),
                 stroke_width=p.stroke_width, region_id=p.region_id)
        removed = [(index, p)]
        self.paths.remove(p)
        self.paths.extend([a, b])
        self._record("split", removed, [a, b])
        return "split path"

    def straighten(self, indices: set[int]) -> str:
        """Replace each selected path with a straight segment between its ends."""
        sel = [i for i in sorted(indices) if 0 <= i < len(self.paths)]
        if not sel:
            return ""
        removed, added = [], []
        for i in sel:
            p = self.paths[i]
            if len(p.points) < 3:
                continue
            removed.append((i, p))
            added.append(Path(
                points=np.vstack([p.points[0], p.points[-1]]),
                stroke_width=p.stroke_width,
                region_id=p.region_id,
            ))
        if not removed:
            return "nothing to straighten"
        for _, p in removed:
            self.paths.remove(p)
        self.paths.extend(added)
        self._record("straighten", removed, added)
        return f"straightened {len(added)} path(s)"

    def delete_node(self, index: int, point_index: int) -> str:
        if not (0 <= index < len(self.paths)):
            return ""
        p = self.paths[index]
        if len(p.points) <= 2:
            return "a path needs at least two nodes"
        self.snapshot_for_edit([index], "delete node")
        p.points = np.delete(p.points, point_index, axis=0)
        return "deleted node"
