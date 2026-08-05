"""Main window: image + parameters + live preview + vector editing + export."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.io_utils import load_gray
from blueprint_tracer.core.pipeline import TraceResult
from blueprint_tracer.export.json_export import write_json
from blueprint_tracer.export.svg import write_svg
from blueprint_tracer.gui.canvas import Canvas
from blueprint_tracer.gui.edit_tools import EditState
from blueprint_tracer.gui.params_panel import ParamsPanel
from blueprint_tracer.gui.project import load_project, save_project
from blueprint_tracer.gui.worker import TraceWorker

PREVIEW_MAX_DIM = 1600
DEBOUNCE_MS = 250


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("blueprint-tracer")
        self.resize(1400, 900)

        self.image_path: str = ""
        self.gray: Optional[np.ndarray] = None
        self.dpi: Optional[float] = None
        self.result: Optional[TraceResult] = None
        self.edits = EditState()
        self.dirty_edits = False

        # Left view shows the scan alone, the main view carries the vectors. In
        # side-by-side mode neither view holds the other's layer, so the heavy
        # path cache and the page pixmap are each built only once.
        self.source_canvas = Canvas()
        self.source_canvas.set_vectors_visible(False)
        self.source_canvas.set_caption("Scan")
        self.canvas = Canvas()

        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.source_canvas)
        self.split.addWidget(self.canvas)
        self.split.setCollapsible(0, False)
        self.split.setCollapsible(1, False)
        self.source_canvas.hide()
        self.setCentralWidget(self.split)

        self._syncing = False
        self.source_canvas.viewport_changed.connect(
            lambda: self._sync_views(self.source_canvas, self.canvas))
        self.canvas.viewport_changed.connect(
            lambda: self._sync_views(self.canvas, self.source_canvas))

        self.params = ParamsPanel()
        dock = QDockWidget("Parameters", self)
        dock.setWidget(self.params)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.params_dock = dock

        self.worker = TraceWorker(self)
        self.worker.finished_trace.connect(self._on_traced)
        self.worker.failed.connect(self._on_failed)
        self.worker.started_trace.connect(lambda: self._set_busy(True))
        self.worker.start()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._retrace)

        self.setStatusBar(QStatusBar())
        self.stats_label = QLabel("no image loaded")
        self.statusBar().addPermanentWidget(self.stats_label)

        self._build_actions()
        self._build_toolbar()

        self.params.changed.connect(self._on_params_changed)
        self.canvas.selection_changed.connect(self._update_selection_status)
        self.canvas.edit_started.connect(self._on_edit_started)
        self.canvas.paths_edited.connect(self._on_paths_edited)

    # --- setup ---

    def _build_actions(self) -> None:
        self.act_open = QAction("&Open image…", self, shortcut=QKeySequence.Open)
        self.act_open.triggered.connect(self.open_image)
        self.act_open_project = QAction("Open &project…", self)
        self.act_open_project.triggered.connect(self.open_project)
        self.act_save_project = QAction("&Save project…", self, shortcut=QKeySequence.Save)
        self.act_save_project.triggered.connect(self.save_project_as)
        self.act_export = QAction("&Export SVG + JSON…", self)
        self.act_export.setShortcut("Ctrl+E")
        self.act_export.triggered.connect(self.export)
        self.act_quit = QAction("&Quit", self, shortcut=QKeySequence.Quit)
        self.act_quit.triggered.connect(self.close)

        self.act_undo = QAction("&Undo", self, shortcut=QKeySequence.Undo)
        self.act_undo.triggered.connect(self.undo)
        self.act_redo = QAction("&Redo", self, shortcut=QKeySequence.Redo)
        self.act_redo.triggered.connect(self.redo)
        self.act_delete = QAction("&Delete selected", self, shortcut=QKeySequence.Delete)
        self.act_delete.triggered.connect(lambda: self._edit_op("delete"))
        self.act_join = QAction("&Join selected", self, shortcut="J")
        self.act_join.triggered.connect(lambda: self._edit_op("join"))
        self.act_straighten = QAction("S&traighten", self, shortcut="T")
        self.act_straighten.triggered.connect(lambda: self._edit_op("straighten"))
        self.act_select_all = QAction("Select &all", self, shortcut=QKeySequence.SelectAll)
        self.act_select_all.triggered.connect(self.select_all)
        self.act_retrace = QAction("&Re-trace", self, shortcut="Ctrl+R")
        self.act_retrace.triggered.connect(lambda: self._retrace(full=True))
        self.act_fit = QAction("&Fit view", self, shortcut="Ctrl+0")
        self.act_fit.triggered.connect(self.fit_views)
        self.act_side_by_side = QAction("Side by side", self, checkable=True)
        self.act_side_by_side.setShortcut("Ctrl+B")
        self.act_side_by_side.setToolTip(
            "Show the scan and the trace in synchronized panes (Ctrl+B)")
        self.act_side_by_side.toggled.connect(self.set_side_by_side)

        m = self.menuBar().addMenu("&File")
        for a in (self.act_open, self.act_open_project, self.act_save_project,
                  self.act_export, self.act_quit):
            m.addAction(a)
        m = self.menuBar().addMenu("&Edit")
        for a in (self.act_undo, self.act_redo, self.act_delete, self.act_join,
                  self.act_straighten, self.act_select_all):
            m.addAction(a)
        m = self.menuBar().addMenu("&View")
        m.addAction(self.act_side_by_side)
        m.addAction(self.act_fit)
        m.addAction(self.act_retrace)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self.act_open)
        tb.addAction(self.act_export)
        tb.addSeparator()

        self.chk_image = QCheckBox("Image")
        self.chk_image.setChecked(True)
        self.chk_image.toggled.connect(self.canvas.set_image_visible)
        self.chk_vectors = QCheckBox("Vectors")
        self.chk_vectors.setChecked(True)
        self.chk_vectors.toggled.connect(self.canvas.set_vectors_visible)
        self.chk_nodes = QCheckBox("Nodes")
        self.chk_nodes.toggled.connect(self._toggle_node_mode)
        for w in (self.chk_image, self.chk_vectors, self.chk_nodes):
            tb.addWidget(w)

        tb.addSeparator()
        tb.addAction(self.act_side_by_side)
        tb.addSeparator()
        tb.addAction(self.act_delete)
        tb.addAction(self.act_join)
        tb.addAction(self.act_straighten)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()
        tb.addAction(self.act_fit)

    # --- view modes ---

    def set_side_by_side(self, on: bool) -> None:
        """Split into scan | trace, or fold back to a single overlaid view."""
        self.source_canvas.setVisible(on)
        # The layer toggles only mean something when both layers share one view.
        for w in (self.chk_image, self.chk_vectors):
            w.setEnabled(not on)
        self.canvas.set_image_visible(not on and self.chk_image.isChecked())
        self.canvas.set_vectors_visible(on or self.chk_vectors.isChecked())
        self.canvas.set_caption("Trace" if on else "")

        if on:
            self._refresh_source_view()
            width = max(self.split.width(), 2)
            self.split.setSizes([width // 2, width - width // 2])
            # Fit once the splitter has actually resized its panes, otherwise the
            # views are fitted to stale geometry.
            QTimer.singleShot(0, self.fit_views)
        else:
            self.fit_views()

    def _refresh_source_view(self) -> None:
        """Give the left pane the same page the vectors were traced from."""
        page = self.result.gray if self.result is not None else self.gray
        if page is None:
            return
        self.source_canvas.set_image(page)
        h, w = page.shape
        self.canvas.set_page_size(w, h)

    def _sync_views(self, src: Canvas, dst: Canvas) -> None:
        if self._syncing or not self.source_canvas.isVisible():
            return
        self._syncing = True
        try:
            dst.sync_viewport_from(src)
        finally:
            self._syncing = False

    def fit_views(self) -> None:
        self.canvas.fit()
        if self.source_canvas.isVisible():
            self.source_canvas.fit()
            self._sync_views(self.canvas, self.source_canvas)

    # --- loading ---

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open blueprint scan", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        try:
            gray, dpi = load_gray(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open image", str(exc))
            return
        self.image_path = path
        self.gray = gray
        self.dpi = dpi
        self.setWindowTitle(f"blueprint-tracer — {os.path.basename(path)}")
        self.canvas.set_image(gray)  # replaced by the preprocessed page once traced
        self.source_canvas.set_image(gray)
        self.fit_views()
        self.statusBar().showMessage(f"loaded {gray.shape[1]}x{gray.shape[0]}", 4000)
        self._retrace()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "Project (*.btproj *.json);;All files (*)")
        if not path:
            return
        try:
            image_path, cfg, paths = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))
            return
        self.params.from_config(cfg)
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        if paths:
            # Restore hand-edited geometry instead of the freshly traced result.
            self.edits.set_paths(paths)
            self._show_paths()
            self.statusBar().showMessage(f"restored {len(paths)} edited paths", 5000)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "", "Project (*.btproj);;All files (*)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".btproj"
        try:
            save_project(path, self.image_path, self.params.to_config(), self.edits.paths)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))
            return
        self.statusBar().showMessage(f"saved {os.path.basename(path)}", 4000)

    def export(self) -> None:
        if not self.result or not self.edits.paths:
            QMessageBox.information(self, "Nothing to export", "Trace an image first.")
            return
        base, _ = QFileDialog.getSaveFileName(
            self, "Export SVG + JSON", os.path.splitext(self.image_path)[0],
            "SVG (*.svg);;All files (*)")
        if not base:
            return
        base = os.path.splitext(base)[0]

        # Export exactly what is on screen, including manual edits.
        out = TraceResult(
            paths=self.edits.paths, width=self.result.width, height=self.result.height,
            dpi=self.result.dpi, angle=self.result.angle, stats=dict(self.result.stats),
        )
        for i, p in enumerate(out.paths):
            p.id = i
        out.stats["n_paths"] = len(out.paths)
        out.stats["n_points"] = int(sum(len(p.points) for p in out.paths))
        try:
            write_svg(out, base + ".svg")
            write_json(out, base + ".json", image_name=os.path.basename(self.image_path))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"exported {os.path.basename(base)}.svg and .json ({len(out.paths)} paths)", 6000)

    # --- tracing ---

    def _on_params_changed(self) -> None:
        self._debounce.start()

    def _retrace(self, full: bool = False) -> None:
        if self.gray is None:
            return
        if self.dirty_edits and not self._confirm_discard_edits():
            return
        cfg = self.params.to_config()
        preview = not full
        if preview:
            longest = max(self.gray.shape)
            if longest > PREVIEW_MAX_DIM:
                cfg.max_dim = PREVIEW_MAX_DIM
        self.worker.request(self.gray, cfg, self.dpi, preview)

    def _confirm_discard_edits(self) -> bool:
        answer = QMessageBox.question(
            self, "Discard manual edits?",
            "Re-tracing replaces the geometry and discards your manual edits.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.dirty_edits = False
            return True
        return False

    def _on_traced(self, result: TraceResult, preview: bool) -> None:
        self._set_busy(False)
        self.result = result
        self.edits.set_paths(list(result.paths))
        self.dirty_edits = False
        # Show the preprocessed page: deskew and preview downscaling mean the
        # source image no longer shares a coordinate frame with the paths.
        if result.gray is not None:
            first = self.canvas.image_item.pixmap().isNull()
            self.canvas.set_image(result.gray)
            self.canvas.set_image_visible(
                not self.source_canvas.isVisible() and self.chk_image.isChecked()
            )
            self._refresh_source_view()
            if first:
                self.fit_views()
        self._show_paths()
        self.params.show_resolved(result.stats)

        s = result.stats
        scope = "preview" if preview else "full"
        self.stats_label.setText(
            f"{scope}: {s['n_paths']} paths · {s['n_points']} pts · "
            f"stroke {s['stroke_width_px']}px · pen-up {s['pen_up_after_px']:.0f}px · "
            f"{s['elapsed_s']:.1f}s"
        )
        self._update_actions()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("trace failed — see dialog", 5000)
        QMessageBox.critical(self, "Trace failed", message)

    def _set_busy(self, busy: bool) -> None:
        QApplication.setOverrideCursor(Qt.BusyCursor) if busy else QApplication.restoreOverrideCursor()
        if busy:
            self.statusBar().showMessage("tracing…")
        else:
            self.statusBar().clearMessage()

    def _show_paths(self) -> None:
        w = self.result.width if self.result else (self.gray.shape[1] if self.gray is not None else 1)
        h = self.result.height if self.result else (self.gray.shape[0] if self.gray is not None else 1)
        self.canvas.paths_item.selected.clear()
        self.canvas.set_paths(self.edits.paths, w, h)
        self._update_selection_status()

    # --- editing ---

    def _toggle_node_mode(self, on: bool) -> None:
        self.canvas.mode = "node" if on else "select"
        self.canvas.paths_item.show_nodes = on
        self.canvas.paths_item.update()

    def _on_edit_started(self, index: int) -> None:
        self.edits.snapshot_for_edit([index], "move node")
        self.dirty_edits = True

    def _on_paths_edited(self, label: str) -> None:
        self.dirty_edits = True
        self._update_actions()
        self.statusBar().showMessage(label, 2500)

    def _edit_op(self, op: str) -> None:
        sel = set(self.canvas.paths_item.selected)
        if not sel:
            self.statusBar().showMessage("nothing selected", 2500)
            return
        if op == "delete":
            message = self.edits.delete(sel)
        elif op == "join":
            message = self.edits.join(sel)
        elif op == "straighten":
            message = self.edits.straighten(sel)
        else:
            return
        self.dirty_edits = True
        self._show_paths()
        self.canvas.paths_item.rebuild()
        self.statusBar().showMessage(message or "no change", 3000)
        self._update_actions()

    def undo(self) -> None:
        label = self.edits.undo()
        if label:
            self._show_paths()
            self.statusBar().showMessage(f"undo {label}", 2500)
        self._update_actions()

    def redo(self) -> None:
        label = self.edits.redo()
        if label:
            self._show_paths()
            self.statusBar().showMessage(f"redo {label}", 2500)
        self._update_actions()

    def select_all(self) -> None:
        self.canvas.paths_item.selected = set(range(len(self.edits.paths)))
        self.canvas.paths_item.update()
        self._update_selection_status()

    def _update_selection_status(self) -> None:
        n = len(self.canvas.paths_item.selected)
        if n:
            self.statusBar().showMessage(f"{n} path(s) selected", 2000)
        self._update_actions()

    def _update_actions(self) -> None:
        has_sel = bool(self.canvas.paths_item.selected)
        self.act_delete.setEnabled(has_sel)
        self.act_join.setEnabled(len(self.canvas.paths_item.selected) >= 2)
        self.act_straighten.setEnabled(has_sel)
        self.act_undo.setEnabled(self.edits.can_undo)
        self.act_redo.setEnabled(self.edits.can_redo)
        self.act_export.setEnabled(bool(self.edits.paths))

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop()
        super().closeEvent(event)
