"""Canvas: source image underlay plus the traced vector overlay.

Drawings reach tens of thousands of paths, so all vectors are painted by a single
QGraphicsItem rather than one item per path -- 15k QGraphicsPathItems make panning
crawl. Hit-testing is done with numpy against the path point arrays instead of
relying on the scene's item index.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QRubberBand,
)

from blueprint_tracer.core.geometry import Path

INK = QColor("#101010")
SELECTED = QColor("#e2231a")
NODE = QColor("#0a7bd6")
NODE_EDGE = QColor("#ffffff")


class PathsItem(QGraphicsItem):
    """Paints every traced path in one go."""

    def __init__(self) -> None:
        super().__init__()
        self.paths: list[Path] = []
        self.selected: set[int] = set()
        self.show_nodes = False
        self._rect = QRectF(0, 0, 1, 1)
        self._cache: list[QPainterPath] = []
        self.setZValue(10)

    def set_paths(self, paths: list[Path], width: int, height: int) -> None:
        self.prepareGeometryChange()
        self.paths = paths
        self._rect = QRectF(0, 0, max(1, width), max(1, height))
        self.rebuild()

    def rebuild(self) -> None:
        self._cache = []
        for p in self.paths:
            pts = p.points
            qp = QPainterPath()
            if len(pts) >= 2:
                qp.moveTo(float(pts[0][0]), float(pts[0][1]))
                for x, y in pts[1:]:
                    qp.lineTo(float(x), float(y))
                if p.closed and not np.array_equal(pts[0], pts[-1]):
                    qp.closeSubpath()
            self._cache.append(qp)
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        painter.setRenderHint(QPainter.Antialiasing, lod > 0.35)

        # Hairlines: cosmetic pens keep strokes readable at every zoom level.
        pen = QPen(INK)
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        for i, qp in enumerate(self._cache):
            if i in self.selected:
                continue
            painter.drawPath(qp)

        if self.selected:
            sel = QPen(SELECTED)
            sel.setCosmetic(True)
            sel.setWidthF(2.0)
            painter.setPen(sel)
            for i in self.selected:
                if 0 <= i < len(self._cache):
                    painter.drawPath(self._cache[i])

        if self.show_nodes and self.selected and lod > 0.15:
            r = 3.0 / max(lod, 1e-6)
            painter.setPen(QPen(NODE_EDGE, 0))
            painter.setBrush(NODE)
            for i in self.selected:
                if not (0 <= i < len(self.paths)):
                    continue
                for x, y in self.paths[i].points:
                    painter.drawEllipse(QPointF(float(x), float(y)), r, r)


class Canvas(QGraphicsView):
    selection_changed = Signal()
    edit_started = Signal(int)          # path index about to be mutated in place
    paths_edited = Signal(str)          # description for the undo stack
    viewport_changed = Signal()         # pan/zoom, for syncing paired views
    status = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#f4f4f2"))
        self.setMouseTracking(True)

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self._scene.addItem(self.image_item)
        self.paths_item = PathsItem()
        self._scene.addItem(self.paths_item)

        self.mode = "select"          # select | node | pan
        self._panning = False
        self._pan_start = QPointF()
        self._drag_node: Optional[tuple[int, int]] = None
        self._rubber: Optional[QRubberBand] = None
        self._rubber_origin = None

        self.horizontalScrollBar().valueChanged.connect(self.viewport_changed)
        self.verticalScrollBar().valueChanged.connect(self.viewport_changed)

        self._caption = ""

    def set_caption(self, text: str) -> None:
        """Label drawn in the corner, so paired panes are tellable apart."""
        self._caption = text
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        if not self._caption:
            return
        painter.save()
        painter.resetTransform()
        painter.setPen(QColor("#8a8a86"))
        font = painter.font()
        font.setPointSizeF(max(8.0, font.pointSizeF()))
        painter.setFont(font)
        painter.drawText(10, 20, self._caption)
        painter.restore()

    # --- content ---

    def set_image(self, gray: np.ndarray) -> None:
        h, w = gray.shape
        img = QImage(np.ascontiguousarray(gray).data, w, h, w, QImage.Format_Grayscale8)
        self.image_item.setPixmap(QPixmap.fromImage(img.copy()))
        self.set_page_size(w, h)

    def set_page_size(self, width: int, height: int) -> None:
        """Define the page extent so paired views scroll over the same area."""
        self._scene.setSceneRect(0, 0, max(1, width), max(1, height))

    def set_paths(self, paths: list[Path], width: int, height: int) -> None:
        self.paths_item.set_paths(paths, width, height)

    def set_image_visible(self, visible: bool) -> None:
        self.image_item.setVisible(visible)

    def set_vectors_visible(self, visible: bool) -> None:
        self.paths_item.setVisible(visible)

    def fit(self) -> None:
        rect = self._scene.sceneRect()
        if rect.width() > 1:
            self.fitInView(rect, Qt.KeepAspectRatio)
            self.viewport_changed.emit()

    def sync_viewport_from(self, other: "Canvas") -> None:
        """Mirror another view's zoom and scroll position."""
        self.setTransform(other.transform())
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())

    # --- interaction ---

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.0015 ** event.angleDelta().y()
        self.scale(factor, factor)
        self.viewport_changed.emit()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MiddleButton or self.mode == "pan":
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            if self.mode == "node":
                hit = self._hit_node(pos)
                if hit is not None:
                    self._drag_node = hit
                    # Snapshot before the in-place mutation so undo can restore it.
                    self.edit_started.emit(hit[0])
                    return
            idx = self._hit_path(pos)
            if idx is None:
                self._rubber_origin = event.position().toPoint()
                self._rubber = QRubberBand(QRubberBand.Rectangle, self)
                self._rubber.setGeometry(self._rubber_origin.x(), self._rubber_origin.y(), 0, 0)
                self._rubber.show()
                if not (event.modifiers() & Qt.ShiftModifier):
                    self.paths_item.selected.clear()
                    self.paths_item.update()
                    self.selection_changed.emit()
                return
            self._toggle_select(idx, extend=bool(event.modifiers() & Qt.ShiftModifier))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return
        if self._drag_node is not None:
            pos = self.mapToScene(event.position().toPoint())
            pi, ni = self._drag_node
            self.paths_item.paths[pi].points[ni] = (pos.x(), pos.y())
            self.paths_item.rebuild()
            return
        if self._rubber is not None and self._rubber_origin is not None:
            cur = event.position().toPoint()
            x0, y0 = self._rubber_origin.x(), self._rubber_origin.y()
            self._rubber.setGeometry(
                min(x0, cur.x()), min(y0, cur.y()),
                abs(cur.x() - x0), abs(cur.y() - y0),
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._drag_node is not None:
            self._drag_node = None
            self.paths_edited.emit("move node")
            return
        if self._rubber is not None:
            rect = self._rubber.geometry()
            self._rubber.hide()
            self._rubber = None
            self._rubber_origin = None
            if rect.width() > 3 and rect.height() > 3:
                scene_rect = self.mapToScene(rect).boundingRect()
                self._select_in_rect(scene_rect, extend=bool(event.modifiers() & Qt.ShiftModifier))
            return
        super().mouseReleaseEvent(event)

    # --- hit testing ---

    def _tolerance(self) -> float:
        """Pick a scene-space click radius that feels constant on screen."""
        scale = self.transform().m11() or 1.0
        return max(1.5, 6.0 / scale)

    def _hit_path(self, pos: QPointF) -> Optional[int]:
        tol = self._tolerance()
        target = np.array([pos.x(), pos.y()], dtype=np.float32)
        best, best_d = None, tol
        for i, p in enumerate(self.paths_item.paths):
            pts = p.points
            if len(pts) == 0:
                continue
            # Cheap bbox reject before the per-segment distance test.
            if (pts[:, 0].min() - tol > target[0] or pts[:, 0].max() + tol < target[0]
                    or pts[:, 1].min() - tol > target[1] or pts[:, 1].max() + tol < target[1]):
                continue
            d = _point_polyline_distance(target, pts)
            if d < best_d:
                best, best_d = i, d
        return best

    def _hit_node(self, pos: QPointF) -> Optional[tuple[int, int]]:
        tol = self._tolerance()
        target = np.array([pos.x(), pos.y()], dtype=np.float32)
        for i in self.paths_item.selected:
            if not (0 <= i < len(self.paths_item.paths)):
                continue
            pts = self.paths_item.paths[i].points
            d = np.hypot(pts[:, 0] - target[0], pts[:, 1] - target[1])
            j = int(np.argmin(d))
            if d[j] <= tol:
                return (i, j)
        return None

    def _toggle_select(self, idx: int, extend: bool) -> None:
        sel = self.paths_item.selected
        if extend:
            sel.symmetric_difference_update({idx})
        else:
            sel.clear()
            sel.add(idx)
        self.paths_item.update()
        self.selection_changed.emit()

    def _select_in_rect(self, rect: QRectF, extend: bool) -> None:
        sel = self.paths_item.selected
        if not extend:
            sel.clear()
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        for i, p in enumerate(self.paths_item.paths):
            pts = p.points
            if len(pts) == 0:
                continue
            inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                      & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
            if inside.any():
                sel.add(i)
        self.paths_item.update()
        self.selection_changed.emit()


def _point_polyline_distance(pt: np.ndarray, pts: np.ndarray) -> float:
    """Shortest distance from a point to a polyline (vectorized over segments)."""
    if len(pts) == 1:
        return float(np.hypot(*(pts[0] - pt)))
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    denom = (ab * ab).sum(axis=1)
    denom[denom == 0] = 1e-9
    t = (((pt - a) * ab).sum(axis=1) / denom).clip(0.0, 1.0)
    proj = a + ab * t[:, None]
    return float(np.hypot(proj[:, 0] - pt[0], proj[:, 1] - pt[1]).min())
