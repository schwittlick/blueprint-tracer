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
