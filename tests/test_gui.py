"""Headless GUI tests. Skipped if PySide6 is not installed."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from blueprint_tracer.core.config import Config  # noqa: E402
from blueprint_tracer.core.geometry import Path  # noqa: E402
from blueprint_tracer.gui.edit_tools import EditState  # noqa: E402
from blueprint_tracer.gui.project import load_project, save_project  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """Answer every message box instead of showing it.

    A modal dialog offscreen blocks forever, and it need not belong to the test
    that hangs: windows outlive the test that made them, and a stray timer firing
    inside the *next* test's event loop is enough. That turns a plain assertion
    failure into a suite that never finishes, so nothing here is allowed to block.
    """
    from PySide6.QtWidgets import QMessageBox

    for name, answer in (("question", QMessageBox.Yes), ("critical", QMessageBox.Ok),
                         ("warning", QMessageBox.Ok), ("information", QMessageBox.Ok)):
        monkeypatch.setattr(QMessageBox, name, staticmethod(
            lambda *a, _answer=answer, **k: _answer))


def _paths():
    return [
        Path(points=np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)),
        Path(points=np.array([[20, 0], [30, 0]], dtype=np.float32)),
        Path(points=np.array([[0, 50], [10, 55], [20, 50]], dtype=np.float32)),
    ]


def test_delete_undo_redo_roundtrip():
    st = EditState()
    st.set_paths(_paths())
    assert st.delete({0})
    assert len(st.paths) == 2
    st.undo()
    assert len(st.paths) == 3
    st.redo()
    assert len(st.paths) == 2


def test_join_merges_selected_paths():
    st = EditState()
    st.set_paths(_paths())
    st.join({0, 1})
    assert len(st.paths) == 2
    joined = st.paths[-1]
    assert len(joined.points) == 4
    st.undo()
    assert len(st.paths) == 3


def test_straighten_reduces_to_two_points():
    st = EditState()
    st.set_paths(_paths())
    st.straighten({2})
    assert len(st.paths[-1].points) == 2
    st.undo()
    assert len(st.paths[2].points) == 3


def test_node_edit_undo_restores_original_position():
    st = EditState()
    st.set_paths(_paths())
    original = st.paths[0].points[0].copy()
    st.snapshot_for_edit([0], "move node")
    st.paths[0].points[0] = (99.0, 99.0)
    st.undo()
    assert np.allclose(st.paths[0].points[0], original)


def test_project_roundtrip(tmp_path):
    cfg = Config(sauvola_k=0.11, supersample=2.0)
    paths = _paths()
    target = tmp_path / "p.btproj"
    save_project(str(target), "/some/image.tif", cfg, paths)

    image, loaded_cfg, loaded_paths, loaded_regions = load_project(str(target))
    assert image.endswith("image.tif")
    assert loaded_cfg.sauvola_k == pytest.approx(0.11)
    assert loaded_cfg.supersample == pytest.approx(2.0)
    assert len(loaded_paths) == len(paths)
    assert np.allclose(loaded_paths[0].points, paths[0].points)
    assert loaded_regions == []


def test_params_panel_carries_ink_shaping_both_ways(qapp):
    """A control that does not survive the config bridge silently does nothing."""
    from blueprint_tracer.gui.params_panel import ParamsPanel

    panel = ParamsPanel()
    panel.from_config(Config(ink_morph=2, pre_smooth=1.5))
    assert panel.ink_morph.value() == 2
    assert panel.pre_smooth.value() == pytest.approx(1.5)

    cfg = panel.to_config()
    assert cfg.ink_morph == 2
    assert cfg.pre_smooth == pytest.approx(1.5)


def test_window_traces_and_aligns_overlay(qapp, tmp_path):
    """The canvas must show the preprocessed page, so paths and image share a frame."""
    import cv2

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((300, 400), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (340, 240), 0, 2)
    cv2.line(img, (60, 150), (340, 150), 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    win.load_image(str(src))

    loop = QEventLoop()
    got = {}

    def done(result, _preview):
        got["result"] = result
        QTimer.singleShot(50, loop.quit)

    win.worker.finished_trace.connect(done)
    QTimer.singleShot(30000, loop.quit)
    loop.exec()

    assert "result" in got, "worker produced no trace"
    result = got["result"]
    assert result.stats["n_paths"] > 0
    assert result.gray is not None

    pixmap = win.canvas.image_item.pixmap()
    assert pixmap.width() == result.width
    assert pixmap.height() == result.height

    win.select_all()
    assert win.canvas.paths_item.selected
    win.worker.stop()
    win.close()


def test_export_writes_svg_json_and_traced_png(qapp, tmp_path):
    import cv2
    from PySide6.QtWidgets import QFileDialog

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((200, 260), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (220, 160), 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    win.load_image(str(src))
    loop = QEventLoop()
    win.worker.finished_trace.connect(lambda *_: QTimer.singleShot(50, loop.quit))
    QTimer.singleShot(30000, loop.quit)
    loop.exec()

    base = tmp_path / "out"
    original = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(base) + ".svg", ""))
    try:
        win.export()
    finally:
        QFileDialog.getSaveFileName = original

    assert (tmp_path / "out.svg").exists()
    assert (tmp_path / "out.json").exists()
    png_path = tmp_path / "out_trace.png"
    assert png_path.exists(), "traced PNG was not written"

    png = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    assert png.shape == (win.result.height, win.result.width)
    assert (png < 128).any(), "traced PNG contains no ink"

    win.worker.stop()
    win.close()


def test_hiding_text_regions_is_non_destructive(qapp, tmp_path):
    """Hiding a region must exclude it from view and export without destroying
    path data, so restoring it brings the strokes back exactly."""
    import cv2
    from PySide6.QtWidgets import QFileDialog

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((320, 520), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 200), (490, 300), 0, 2)
    cv2.putText(img, "GROOVED CASING", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    win.load_image(str(src))
    loop = QEventLoop()
    win.worker.finished_trace.connect(lambda *_: QTimer.singleShot(50, loop.quit))
    QTimer.singleShot(30000, loop.quit)
    loop.exec()

    assert win.text_panel.regions, "no text regions detected"
    total = len(win.edits.paths)
    tagged = [p for p in win.edits.paths if p.region_id >= 0]
    assert tagged, "no path was tagged to a region"

    win.text_panel.set_all("hide")
    qapp.processEvents()
    assert len(win.edits.paths) == total, "hiding destroyed path data"
    assert win.canvas.paths_item.hidden_regions

    base = tmp_path / "hidden"
    original = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(base) + ".svg", ""))
    try:
        win.export()
        hidden_json = json.loads((tmp_path / "hidden.json").read_text())
        assert len(hidden_json["paths"]) == total - len(tagged)

        win.text_panel.set_all("trace")
        qapp.processEvents()
        base = tmp_path / "restored"
        QFileDialog.getSaveFileName = staticmethod(
            lambda *a, **k: (str(base) + ".svg", ""))
        win.export()
        restored_json = json.loads((tmp_path / "restored.json").read_text())
    finally:
        QFileDialog.getSaveFileName = original

    assert len(restored_json["paths"]) == total, "restoring did not bring paths back"
    win.worker.stop()
    win.close()


def _shutdown(win):
    """Leave nothing behind that can fire inside the next test's event loop."""
    win._debounce.stop()
    win.dirty_edits = False
    win.worker.stop()
    win.close()


def _trace_and_wait(win, path):
    win.load_image(str(path))
    loop = QEventLoop()
    win.worker.finished_trace.connect(lambda *_: QTimer.singleShot(50, loop.quit))
    QTimer.singleShot(30000, loop.quit)
    loop.exec()


def test_hand_drawn_region_letters_text_the_detector_missed(qapp, tmp_path):
    """The whole point of drawing a box: lettering the detector will not claim can
    still be replaced with Hershey text, without deleting its strokes by hand."""
    import cv2
    from PySide6.QtWidgets import QFileDialog

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((320, 520), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 200), (490, 300), 0, 2)
    cv2.putText(img, "A1", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    win.params.detect_text.setChecked(False)     # stand in for a missed label
    _trace_and_wait(win, src)
    assert not win.text_panel.regions

    total = len(win.edits.paths)
    win.act_add_region.setChecked(True)
    assert win.canvas.mode == "region"
    win._add_region(QRectF(30, 30, 60, 40))
    qapp.processEvents()

    assert len(win.text_panel.regions) == 1
    region = win.text_panel.regions[0]
    assert region.source == "manual"
    assert [p for p in win.edits.paths if p.region_id == region.id], \
        "the strokes inside the box were not tagged to it"

    # Empty Hershey text must not hide anything: that would blank the label.
    region.mode = "hershey"
    win._apply_region_modes()
    assert not win.canvas.paths_item.hidden_regions
    assert not win.text_paths

    region.text = "A1"
    win._apply_region_modes()
    qapp.processEvents()
    assert win.canvas.paths_item.hidden_regions == {region.id}
    assert win.text_paths, "hershey lettering was not generated"

    base = tmp_path / "lettered"
    original = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(base) + ".svg", ""))
    try:
        win.export()
    finally:
        QFileDialog.getSaveFileName = original

    doc = json.loads((tmp_path / "lettered.json").read_text())
    replaced = sum(1 for p in win.edits.paths if p.region_id == region.id)
    assert len(doc["paths"]) == total - replaced + len(win.text_paths)
    assert len(win.edits.paths) == total, "lettering destroyed the traced strokes"
    _shutdown(win)


def test_dragging_in_region_mode_reports_a_box_and_selects_nothing(qapp):
    """In region mode the left button draws; it must not also rubber-band select,
    or every box would silently change the selection under the user."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest

    from blueprint_tracer.gui.canvas import Canvas

    canvas = Canvas()
    canvas.resize(400, 300)
    canvas.set_paths(_paths(), 400, 300)
    canvas.show()
    QTest.qWaitForWindowExposed(canvas)
    canvas.fit()

    boxes = []
    canvas.region_drawn.connect(boxes.append)
    canvas.mode = "region"

    viewport = canvas.viewport()
    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, QPoint(40, 40))
    QTest.mouseMove(viewport, QPoint(160, 120))
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, QPoint(160, 120))

    assert len(boxes) == 1, "the drag did not report a region"
    assert boxes[0].width() > 0 and boxes[0].height() > 0
    assert not canvas.paths_item.selected, "drawing a box changed the selection"
    canvas.close()


def test_edit_shortcuts_act_on_the_selection_from_the_canvas(qapp, tmp_path):
    """Del, Backspace and J must edit the selection when the canvas has focus —
    that is the only place a selection can be made in the first place."""
    import cv2
    from PySide6.QtTest import QTest

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((300, 400), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (340, 240), 0, 2)
    cv2.line(img, (60, 150), (340, 150), 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    _trace_and_wait(win, src)
    win.show()
    QTest.qWaitForWindowExposed(win)

    keys = win.act_delete.shortcuts()
    assert Qt.Key_Backspace in [k[0].key() for k in keys], "Backspace is not bound"
    assert win.act_join.shortcut().toString() == "J"

    total = len(win.edits.paths)
    assert total >= 2, "need at least two strokes to exercise join"
    for key in (Qt.Key_Delete, Qt.Key_Backspace):
        win.canvas.paths_item.selected = {0}
        win._update_selection_status()
        win.canvas.setFocus()
        QTest.keyClick(win.canvas, key)
        qapp.processEvents()
        assert len(win.edits.paths) == total - 1, f"{key} did not delete"
        win.undo()
        assert len(win.edits.paths) == total

    win.canvas.paths_item.selected = {0, 1}
    win._update_selection_status()
    win.canvas.setFocus()
    QTest.keyClick(win.canvas, Qt.Key_J)
    qapp.processEvents()
    assert len(win.edits.paths) == total - 1, "J did not join"
    _shutdown(win)


def test_escape_leaves_region_mode_from_canvas_or_dock(qapp, tmp_path):
    """A drawing mode you cannot get out of strands the user. Escape must work
    from the dock too: adding a region parks the caret in its text cell."""
    import cv2
    from PySide6.QtTest import QTest

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((320, 520), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 200), (490, 300), 0, 2)
    cv2.putText(img, "A1", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    win.params.detect_text.setChecked(False)
    _trace_and_wait(win, src)

    win.act_add_region.setChecked(True)
    QTest.keyClick(win.canvas, Qt.Key_Escape)
    assert not win.act_add_region.isChecked()
    assert win.canvas.mode == "select", "strokes cannot be selected again"

    # Now the path that actually bites: focus sits in the dock after a box.
    win.act_add_region.setChecked(True)
    win._add_region(QRectF(30, 30, 60, 40))
    qapp.processEvents()
    win.text_panel.table.closePersistentEditor(
        win.text_panel.table.item(0, 2))     # first Escape cancels the cell edit
    QTest.keyClick(win.text_panel.table, Qt.Key_Escape)
    qapp.processEvents()
    assert not win.act_add_region.isChecked()
    assert win.canvas.mode == "select"

    win.canvas.paths_item.selected = {0}
    win._edit_op("delete")
    assert len(win.edits.paths) < len(win.result.paths), "delete did nothing"
    _shutdown(win)


def test_hand_drawn_region_survives_a_rescaled_retrace(qapp, tmp_path):
    """Re-tracing rebuilds the detected regions from nothing. A hand-drawn box has
    no detector to bring it back, and supersample changes the page it sits on."""
    import cv2

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((320, 520), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 200), (490, 300), 0, 2)
    cv2.putText(img, "A1", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.params.deskew.setChecked(False)
    win.params.detect_text.setChecked(False)
    _trace_and_wait(win, src)

    page_width = win.result.width
    win._add_region(QRectF(30, 30, 60, 40))
    region = win.text_panel.regions[-1]
    region.text = "PLATE"
    box = (region.x, region.y, region.width, region.height)

    loop = QEventLoop()
    win.worker.finished_trace.connect(lambda *_: QTimer.singleShot(50, loop.quit))
    win.params.supersample.setValue(2.0)
    win._retrace(full=True)
    QTimer.singleShot(30000, loop.quit)
    loop.exec()

    kept = [r for r in win.text_panel.regions if r.source == "manual"]
    assert len(kept) == 1, "the hand-drawn region did not survive the re-trace"
    assert kept[0].text == "PLATE"
    scale = win.result.width / page_width
    assert kept[0].x == pytest.approx(box[0] * scale, abs=2)
    assert kept[0].width == pytest.approx(box[2] * scale, abs=2)
    assert [p for p in win.edits.paths if p.region_id == kept[0].id], \
        "strokes were not re-tagged on the new page"
    _shutdown(win)


def test_retrace_keeps_the_view_on_the_same_part_of_the_page(qapp, tmp_path):
    """Supersample rescales the page; the user must not be thrown somewhere else."""
    import cv2

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((300, 400), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (340, 240), 0, 2)
    cv2.line(img, (60, 150), (340, 150), 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    win.params.deskew.setChecked(False)
    win.load_image(str(src))

    def wait_for_trace():
        loop = QEventLoop()
        conn = win.worker.finished_trace.connect(lambda *_: QTimer.singleShot(50, loop.quit))
        QTimer.singleShot(30000, loop.quit)
        loop.exec()
        win.worker.finished_trace.disconnect(conn)

    wait_for_trace()

    # Zoom into the lower right quadrant of the page.
    page = win.canvas.sceneRect()
    win.canvas.fitInView(
        QRectF(page.width() * 0.5, page.height() * 0.5,
               page.width() * 0.25, page.height() * 0.25),
        Qt.KeepAspectRatio,
    )
    qapp.processEvents()
    before = win.canvas.view_state()

    win.params.supersample.setValue(2.0)
    wait_for_trace()
    qapp.processEvents()

    assert win.canvas.sceneRect().width() != page.width(), "page size did not change"
    after = win.canvas.view_state()
    assert after is not None and before is not None
    for got, want in zip(after, before):
        assert got == pytest.approx(want, rel=0.02)

    win.worker.stop()
    win.close()


def test_side_by_side_splits_layers_and_syncs_views(qapp, tmp_path):
    """Scan and trace get one layer each, and the two panes stay locked together."""
    import cv2

    from blueprint_tracer.gui.main_window import MainWindow

    img = np.full((300, 400), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (340, 240), 0, 2)
    cv2.line(img, (60, 150), (340, 150), 0, 2)
    src = tmp_path / "sheet.png"
    cv2.imwrite(str(src), img)

    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    win.params.deskew.setChecked(False)
    win.load_image(str(src))

    loop = QEventLoop()
    win.worker.finished_trace.connect(lambda *_: QTimer.singleShot(50, loop.quit))
    QTimer.singleShot(30000, loop.quit)
    loop.exec()

    win.act_side_by_side.setChecked(True)
    qapp.processEvents()

    assert win.source_canvas.isVisible()
    # Each pane carries exactly one layer, so neither builds the other's cache.
    assert not win.source_canvas.paths_item.isVisible()
    assert not win.canvas.image_item.isVisible()
    assert win.canvas.paths_item.isVisible()
    # Layer toggles are meaningless when the layers live in separate panes.
    assert not win.chk_image.isEnabled()

    win.canvas.fitInView(QRectF(50, 50, 200, 150), Qt.KeepAspectRatio)
    qapp.processEvents()
    assert win.canvas.transform().m11() == pytest.approx(
        win.source_canvas.transform().m11(), rel=1e-6)
    assert (win.canvas.horizontalScrollBar().value()
            == win.source_canvas.horizontalScrollBar().value())

    win.act_side_by_side.setChecked(False)
    qapp.processEvents()
    assert not win.source_canvas.isVisible()
    assert win.canvas.image_item.isVisible()
    assert win.chk_image.isEnabled()
    win.worker.stop()
    win.close()
