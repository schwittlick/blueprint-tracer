"""Main window: image + parameters + live preview + vector editing + export."""

from __future__ import annotations

import os
from typing import Optional

import cv2
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
from blueprint_tracer.core import optimize
from blueprint_tracer.core.pipeline import TraceResult, geometry_stats
from blueprint_tracer.export.json_export import write_json
from blueprint_tracer.export.render import render_paths
from blueprint_tracer.export.svg import write_svg
from blueprint_tracer.gui.canvas import Canvas
from blueprint_tracer.gui.edit_tools import EditState
from blueprint_tracer.gui.params_panel import ParamsPanel
from blueprint_tracer.gui.project import load_project, save_project
from blueprint_tracer.core import hershey, ocr, text as text_core
from blueprint_tracer.gui.text_panel import TextPanel
from blueprint_tracer.gui.worker import OcrWorker, TraceWorker

PREVIEW_MAX_DIM = 1600
DEBOUNCE_MS = 250

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_dir(name: str) -> str:
    """Locate the project's data/ or out/ folder for file dialogs to start in.

    The working directory wins so a copy of the project elsewhere still opens its
    own scans; the source tree is the fallback when the app is launched from
    somewhere unrelated. An empty string leaves the dialog at its default.
    """
    for base in (os.getcwd(), _REPO_ROOT):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    return ""


def _out_path(name: str) -> str:
    """Suggest ``name`` inside out/, falling back to a bare name if there is none."""
    return os.path.join(_project_dir("out"), name)


def _carry_over_decisions(old_regions: list, new_regions: list, scale: float = 1.0) -> None:
    """Copy review decisions from a previous detection onto a fresh one.

    Re-tracing rebuilds the regions from scratch, so without this every parameter
    tweak would quietly reset the text, confidence and mode the user had settled on.
    Regions are matched by overlapping position, which survives small shifts.

    ``scale`` maps the old boxes into the new page's pixels. Switching between the
    downscaled preview and a full-resolution trace changes the page size outright,
    and unscaled boxes would then overlap nothing and match nothing.
    """
    reviewed = [r for r in old_regions if r.mode != "trace" or r.text]
    if not reviewed or not new_regions:
        return
    for fresh in new_regions:
        best, best_score = None, 0.0
        for old in reviewed:
            ox, oy = old.x * scale, old.y * scale
            ow, oh = old.width * scale, old.height * scale
            ix = max(0.0, min(fresh.x + fresh.width, ox + ow) - max(fresh.x, ox))
            iy = max(0.0, min(fresh.y + fresh.height, oy + oh) - max(fresh.y, oy))
            inter = ix * iy
            if not inter:
                continue
            score = inter / float(min(fresh.width * fresh.height, ow * oh) or 1)
            if score > best_score:
                best, best_score = old, score
        if best is not None and best_score > 0.6:
            fresh.text = best.text
            fresh.confidence = best.confidence
            fresh.mode = best.mode


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
        self.text_paths: list = []   # Hershey lettering standing in for regions
        self._pending_restore = None  # project work waiting for its trace to land
        # Page width the current region boxes are measured in, so hand-drawn ones
        # can be moved onto a re-traced page of a different size.
        self._region_page_width = 0

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

        self.text_panel = TextPanel()
        text_dock = QDockWidget("Text regions", self)
        text_dock.setWidget(self.text_panel)
        text_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, text_dock)
        self.text_dock = text_dock
        self.tabifyDockWidget(dock, text_dock)
        dock.raise_()

        self.text_panel.region_selected.connect(self._focus_region)
        self.text_panel.region_removed.connect(self._remove_region)
        self.text_panel.modes_changed.connect(self._apply_region_modes)
        self.text_panel.text_edited.connect(lambda _id: self._apply_region_modes())
        self.text_panel.ocr_requested.connect(self.run_ocr)
        self.text_panel.set_languages(ocr.available_languages())
        self.ocr_worker: Optional[OcrWorker] = None

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
        self.canvas.region_drawn.connect(self._add_region)
        self.canvas.region_mode_exited.connect(
            lambda: self.act_add_region.setChecked(False))

    # --- setup ---

    def _build_actions(self) -> None:
        self.act_open = QAction("&Open image…", self, shortcut=QKeySequence.Open)
        self.act_open.triggered.connect(self.open_image)
        self.act_open_project = QAction("Open &project…", self)
        self.act_open_project.triggered.connect(self.open_project)
        self.act_save_project = QAction("&Save project…", self, shortcut=QKeySequence.Save)
        self.act_save_project.triggered.connect(self.save_project_as)
        self.act_export = QAction("&Export SVG + JSON + PNG…", self)
        self.act_export.setShortcut("Ctrl+E")
        self.act_export.triggered.connect(self.export)
        self.act_quit = QAction("&Quit", self, shortcut=QKeySequence.Quit)
        self.act_quit.triggered.connect(self.close)

        self.act_undo = QAction("&Undo", self, shortcut=QKeySequence.Undo)
        self.act_undo.triggered.connect(self.undo)
        self.act_redo = QAction("&Redo", self, shortcut=QKeySequence.Redo)
        self.act_redo.triggered.connect(self.redo)
        self.act_delete = QAction("&Delete selected", self)
        # Backspace as well as Delete: it is what the hand reaches for, and laptop
        # keyboards do not all carry a Del key.
        self.act_delete.setShortcuts([QKeySequence.Delete, QKeySequence(Qt.Key_Backspace)])
        self.act_delete.setToolTip("Delete the selected strokes (Del or Backspace)")
        self.act_delete.triggered.connect(lambda: self._edit_op("delete"))
        self.act_join = QAction("&Join selected", self, shortcut="J")
        self.act_join.setToolTip("Join the selected strokes end to end (J)")
        self.act_join.triggered.connect(lambda: self._edit_op("join"))
        self.act_straighten = QAction("S&traighten", self, shortcut="T")
        self.act_straighten.setToolTip("Reduce the selected strokes to straight lines (T)")
        self.act_straighten.triggered.connect(lambda: self._edit_op("straighten"))
        self.act_select_all = QAction("Select &all", self, shortcut=QKeySequence.SelectAll)
        self.act_select_all.triggered.connect(self.select_all)
        self.act_add_region = QAction("Add text &region", self, checkable=True)
        self.act_add_region.setShortcut("R")
        self.act_add_region.setToolTip(
            "Drag a box around lettering the detector missed (R).\n"
            "It then behaves like any other region: recognize it, letter it in\n"
            "Hershey, or hide it — without deleting the strokes underneath."
        )
        self.act_add_region.toggled.connect(self._toggle_region_mode)
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
                  self.act_straighten, self.act_select_all, self.act_add_region):
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
        self.chk_text = QCheckBox("Text boxes")
        self.chk_text.setToolTip("Outline the detected lettering regions")
        self.chk_text.toggled.connect(self._toggle_text_boxes)
        for w in (self.chk_image, self.chk_vectors, self.chk_nodes, self.chk_text):
            tb.addWidget(w)

        tb.addSeparator()
        tb.addAction(self.act_side_by_side)
        tb.addSeparator()
        tb.addAction(self.act_add_region)
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

    def _apply_restore(self) -> None:
        """Put a loaded project's geometry and review decisions in place.

        Called after the trace it must override has landed, never before.
        """
        if self._pending_restore is None:
            return
        paths, regions = self._pending_restore
        self._pending_restore = None
        self.edits.set_paths(paths)
        if regions and self.result is not None:
            # The saved region tags describe hand-edited paths, so keep them rather
            # than re-deriving tags for geometry the detector never saw.
            self._set_regions(regions, reassign=False)
        self._show_paths()
        self._apply_region_modes()
        self.statusBar().showMessage(
            f"restored {len(paths)} edited path(s) and {len(regions)} text region(s)",
            6000)

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
            self, "Open blueprint scan", _project_dir("data"),
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
            self, "Open project", _project_dir("out"),
            "Project (*.btproj *.json);;All files (*)")
        if not path:
            return
        try:
            image_path, cfg, paths, regions = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))
            return
        self.params.from_config(cfg)
        # load_image queues a background trace; applying the saved work now would be
        # overwritten the moment that trace lands. Stash it and let _on_traced apply it.
        self._pending_restore = (paths, regions) if paths else None
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        elif paths:
            self._apply_restore()

    def save_project_as(self) -> None:
        stem = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", _out_path(stem + ".btproj" if stem else ""),
            "Project (*.btproj);;All files (*)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".btproj"
        try:
            save_project(path, self.image_path, self.params.to_config(),
                         self.edits.paths, self.text_panel.regions)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))
            return
        self.statusBar().showMessage(f"saved {os.path.basename(path)}", 4000)

    def export(self) -> None:
        if not self.result or not self.edits.paths:
            QMessageBox.information(self, "Nothing to export", "Trace an image first.")
            return
        base, _ = QFileDialog.getSaveFileName(
            self, "Export SVG + JSON + traced PNG",
            _out_path(os.path.splitext(os.path.basename(self.image_path))[0]),
            "SVG (*.svg);;All files (*)")
        if not base:
            return
        base = os.path.splitext(base)[0]

        # Export exactly what is on screen: manual edits included, hidden regions
        # left out, and Hershey lettering standing in where it was chosen.
        hidden = self.text_panel.hidden_region_ids()
        drawing = [p for p in self.edits.paths if p.region_id not in hidden]
        lettering = list(self.text_paths)

        # Removing regions and appending generated lettering makes a different
        # drawing from the one the pipeline optimized, so order it again. Drawing
        # and lettering stay in separate blocks because they map to separate pens:
        # interleaving them would cost a carousel swap per path.
        if self.params.plot_order.isChecked():
            paths = optimize.order_groups_for_plotting([drawing, lettering])
        else:
            paths = drawing + lettering

        out = TraceResult(
            paths=paths, width=self.result.width, height=self.result.height,
            dpi=self.result.dpi, angle=self.result.angle,
            stats=dict(self.result.stats), text_regions=self.result.text_regions,
        )
        # Keep each traced stroke's identity; renumbering here would flatten id
        # back onto the array position and destroy the plot-order permutation.
        # Hershey lettering is generated, so it gets fresh ids past the last one.
        next_id = max((p.id for p in self.edits.paths if p.id >= 0), default=-1) + 1
        for p in out.paths:
            if p.id < 0:
                p.id = next_id
                next_id += 1
        # Describe the geometry actually written, not the pipeline's earlier run.
        out.stats.update(geometry_stats(out.paths))
        out.stats["plot_ordered"] = self.params.plot_order.isChecked()
        out.stats["plot_order_scope"] = "grouped: drawing, then lettering"
        try:
            write_svg(out, base + ".svg")
            write_json(out, base + ".json", image_name=os.path.basename(self.image_path))
            cv2.imwrite(base + "_trace.png", render_paths(out))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"exported {os.path.basename(base)} .svg / .json / _trace.png "
            f"({len(out.paths)} paths)", 6000)

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
            # Parameters like supersample change the page's pixel size, so hold on
            # to the area being looked at and put the view back on it afterwards.
            view = None if first else self.canvas.view_state()
            self.canvas.set_image(result.gray)
            self.canvas.set_image_visible(
                not self.source_canvas.isVisible() and self.chk_image.isChecked()
            )
            self._refresh_source_view()
            if first:
                self.fit_views()
            else:
                self.canvas.restore_view_state(view)
                self._sync_views(self.canvas, self.source_canvas)
        # A re-trace produces brand new regions; carry the review decisions over so
        # nudging a parameter does not silently discard them, and re-place the
        # hand-drawn boxes, which no amount of detection will ever reproduce.
        scale = (result.width / self._region_page_width) if self._region_page_width else 1.0
        old_regions = list(self.text_panel.regions)
        _carry_over_decisions(old_regions, result.text_regions, scale)
        manual = [r for r in old_regions if getattr(r, "source", "detected") == "manual"]
        self._show_paths()
        self._set_regions(
            text_core.merge_regions(result.text_regions, manual, scale)
        )
        if self._pending_restore is not None:
            self._apply_restore()
        self._apply_region_modes()
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

    def _toggle_text_boxes(self, on: bool) -> None:
        self.canvas.set_regions_visible(on)
        if on:
            self.text_dock.raise_()

    def _sync_canvas_mode(self) -> None:
        """Three jobs share the left button, so derive the mode from the toggles."""
        if self.act_add_region.isChecked():
            self.canvas.mode = "region"
        elif self.chk_nodes.isChecked():
            self.canvas.mode = "node"
        else:
            self.canvas.mode = "select"
        self.canvas.setCursor(
            Qt.CrossCursor if self.canvas.mode == "region" else Qt.ArrowCursor)

    def _toggle_region_mode(self, on: bool) -> None:
        if on:
            self.chk_nodes.setChecked(False)
            # Drawing boxes you cannot see is guesswork.
            self.chk_text.setChecked(True)
            self.text_dock.raise_()
            self.statusBar().showMessage(
                "drawing text regions — drag a box around the lettering · "
                "Esc or R to go back to selecting strokes", 10000)
        else:
            self.statusBar().showMessage("selecting strokes", 3000)
        self._sync_canvas_mode()

    def _set_regions(self, regions: list, reassign: bool = True) -> None:
        """Publish a changed region set to the canvas, the panel and the paths.

        Ids are renumbered whenever the set changes, so the region tags on the paths
        must be recomputed with it -- otherwise hiding a region acts on whichever
        strokes happen to hold the old number. ``reassign`` is off only when
        restoring a project, whose saved tags already describe hand-edited paths
        that no re-derivation could reproduce.
        """
        if self.result is not None:
            self.result.text_regions = regions
            self._region_page_width = self.result.width
        if reassign:
            text_core.assign_paths(self.edits.paths, regions)
        width = self.result.width if self.result else 1
        height = self.result.height if self.result else 1
        page = self.result.gray if self.result else None
        self.canvas.set_regions(regions, width, height)
        self.text_panel.set_regions(regions, page)

    def _add_region(self, rect) -> None:
        """Turn a box the user dragged out into a text region."""
        if self.result is None:
            return
        # Clamp to the page: a box hanging off the edge describes ink that is not
        # there, and its OCR crop would be part empty strip.
        x0 = max(0.0, min(rect.left(), self.result.width - 2.0))
        y0 = max(0.0, min(rect.top(), self.result.height - 2.0))
        x1 = min(float(self.result.width), max(rect.right(), x0 + 1.0))
        y1 = min(float(self.result.height), max(rect.bottom(), y0 + 1.0))
        if x1 - x0 < 2.0 or y1 - y0 < 2.0:
            self.statusBar().showMessage("box too small — drag across the lettering", 3000)
            return

        region = text_core.manual_region(x0, y0, x1 - x0, y1 - y0)
        self._set_regions(text_core.merge_regions(self.result.text_regions, [region]))
        self._apply_region_modes()
        # Land the caret in the Text cell: typing the label is the whole point of
        # having drawn the box.
        self.text_panel.begin_edit(region.id)
        self.statusBar().showMessage(
            f"added region {region.id} ({region.orientation}) — type the label, "
            f"then set its mode to hershey", 6000)

    def _remove_region(self, region_id: int) -> None:
        regions = [r for r in (self.result.text_regions if self.result else [])
                   if r.id != region_id]
        for i, r in enumerate(regions):
            r.id = i
        self._set_regions(regions)
        self._apply_region_modes()
        self.statusBar().showMessage(f"removed region {region_id}", 3000)

    def _focus_region(self, region_id: int) -> None:
        for region in self.result.text_regions if self.result else []:
            if region.id == region_id:
                self.canvas.regions_item.highlight = region_id
                self.canvas.set_regions_visible(True)
                self.chk_text.setChecked(True)
                self.canvas.focus_rect(region.x, region.y, region.width, region.height)
                if self.source_canvas.isVisible():
                    self._sync_views(self.canvas, self.source_canvas)
                return

    def run_ocr(self, lang: str) -> None:
        """Recognize every detected region, leaving all of them still traced.

        Reading the text never changes what is drawn; switching a region to
        hershey is a separate, per-region decision.
        """
        if self.result is None or not self.text_panel.regions:
            self.statusBar().showMessage("no text regions to recognize", 3000)
            return
        if self.ocr_worker is not None and self.ocr_worker.isRunning():
            return
        try:
            ocr.check_ready(lang)
        except ocr.OcrUnavailable as exc:
            QMessageBox.warning(self, "OCR unavailable", str(exc))
            return

        self.text_panel.btn_ocr.setEnabled(False)
        self.ocr_worker = OcrWorker(self.result.gray, self.text_panel.regions, lang, self)
        self.ocr_worker.progress.connect(
            lambda done, total: self.statusBar().showMessage(
                f"recognizing… {done}/{total}"))
        self.ocr_worker.failed.connect(self._on_ocr_failed)
        self.ocr_worker.finished_ocr.connect(self._on_ocr_done)
        self.ocr_worker.start()

    def _on_ocr_done(self, found: int) -> None:
        self.text_panel.btn_ocr.setEnabled(True)
        self.text_panel.refresh_results()
        self._apply_region_modes()
        total = len(self.text_panel.regions)
        self.statusBar().showMessage(
            f"recognized {found} of {total} region(s) — review the text, then switch "
            f"regions to hershey", 8000)

    def _on_ocr_failed(self, message: str) -> None:
        self.text_panel.btn_ocr.setEnabled(True)
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "OCR failed", message)

    def _apply_region_modes(self) -> None:
        """Recompose what is drawn from the region decisions.

        Traced strokes are never destroyed: a hidden or Hershey-substituted region
        is only skipped at paint and export time, so switching back restores it.
        """
        hidden = self.text_panel.hidden_region_ids()
        self.canvas.paths_item.hidden_regions = hidden
        self.canvas.paths_item.update()

        text_paths: list = []
        for region in self.text_panel.hershey_regions():
            text_paths.extend(hershey.render_region(region))
        self.text_paths = text_paths
        self.canvas.set_text_paths(text_paths, self.canvas.paths_item._rect.width(),
                                  self.canvas.paths_item._rect.height())
        self.canvas.regions_item.update()

        replaced = sum(1 for p in self.edits.paths if p.region_id in hidden)
        n_hershey = len(self.text_panel.hershey_regions())
        self.statusBar().showMessage(
            f"{n_hershey} region(s) as hershey text · {replaced} traced path(s) "
            f"replaced or hidden", 4000)

    def _toggle_node_mode(self, on: bool) -> None:
        if on:
            self.act_add_region.setChecked(False)
        self.canvas.paths_item.show_nodes = on
        self.canvas.paths_item.update()
        self._sync_canvas_mode()

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

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Escape leaves region mode wherever the focus happens to be.

        Adding a region parks the caret in the dock's text cell, so the canvas is
        not the widget hearing the key. Handling it here catches the leftovers of
        the chain: a cell editor claims the first Escape to cancel itself, and the
        next one reaches the window and puts the left button back to selecting.
        """
        if event.key() == Qt.Key_Escape and self.act_add_region.isChecked():
            self.act_add_region.setChecked(False)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop()
        super().closeEvent(event)
