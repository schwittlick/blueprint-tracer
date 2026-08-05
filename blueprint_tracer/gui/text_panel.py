"""Text region review panel.

Every region starts as ``trace`` -- the faithful centerline trace -- and any other
treatment is opted into per region. Nothing is ever silently replaced, because a
misread label on an engineering drawing looks authoritative in a way a wobbly
trace does not. Low-confidence rows are tinted so review effort lands where the
recognizer was least sure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

MODES = ("trace", "hershey", "hide")
LOW_CONFIDENCE = 75.0
THUMB_HEIGHT = 24
THUMB_MAX_WIDTH = 132

COL_ID, COL_PREVIEW, COL_TEXT, COL_CONF, COL_MODE = range(5)


class TextPanel(QWidget):
    region_selected = Signal(int)       # region id, to focus on the canvas
    modes_changed = Signal()
    text_edited = Signal(int)           # region id whose text the user changed
    ocr_requested = Signal(str)         # language code

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.regions: list = []
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
        self.lang = QComboBox()
        self.lang.setToolTip("Tesseract language data to recognize with")
        self.btn_ocr = QPushButton("Run OCR")
        self.btn_ocr.setToolTip(
            "Recognize each detected region. Text stays editable, and regions keep\n"
            "their traced strokes until you switch them to hershey."
        )
        self.btn_ocr.clicked.connect(
            lambda: self.ocr_requested.emit(self.lang.currentText()))
        controls.addWidget(QLabel("Lang"))
        controls.addWidget(self.lang, 1)
        controls.addWidget(self.btn_ocr)
        root.addLayout(controls)

        self.summary = QLabel("no text detected")
        self.summary.setStyleSheet("color:#666;font-size:11px;")
        root.addWidget(self.summary)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Preview", "Text", "Conf", "Mode"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_ID, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_PREVIEW, QHeaderView.Interactive)
        header.setSectionResizeMode(COL_TEXT, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_CONF, QHeaderView.Fixed)
        header.setSectionResizeMode(COL_MODE, QHeaderView.Fixed)
        self.table.setColumnWidth(COL_PREVIEW, 138)
        self.table.setColumnWidth(COL_CONF, 48)
        self.table.setColumnWidth(COL_MODE, 96)
        self.setMinimumWidth(520)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_trace_all = QPushButton("All to trace")
        self.btn_hershey_ok = QPushButton("Hershey if confident")
        self.btn_hershey_ok.setToolTip(
            f"Switch regions recognized above {LOW_CONFIDENCE:.0f}% to hershey,\n"
            "leaving the rest traced.")
        self.btn_hide_all = QPushButton("Hide all")
        self.btn_trace_all.clicked.connect(lambda: self.set_all("trace"))
        self.btn_hershey_ok.clicked.connect(self.accept_confident)
        self.btn_hide_all.clicked.connect(lambda: self.set_all("hide"))
        for b in (self.btn_trace_all, self.btn_hershey_ok, self.btn_hide_all):
            buttons.addWidget(b)
        root.addLayout(buttons)

    # --- content ---

    def set_languages(self, languages: list[str], preferred: str = "eng") -> None:
        current = self.lang.currentText()
        self.lang.clear()
        self.lang.addItems(languages or ["eng"])
        for candidate in (current, preferred):
            if candidate and candidate in languages:
                self.lang.setCurrentText(candidate)
                break
        self.btn_ocr.setEnabled(bool(languages))
        if not languages:
            self.btn_ocr.setToolTip("Tesseract or its language data is not installed")

    def set_regions(self, regions: list, page: Optional[np.ndarray]) -> None:
        self._loading = True
        self.regions = list(regions)
        self.table.setRowCount(len(self.regions))

        for row, region in enumerate(self.regions):
            index = QTableWidgetItem(str(region.id))
            index.setFlags(index.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, COL_ID, index)

            preview = QTableWidgetItem()
            preview.setFlags(preview.flags() & ~Qt.ItemIsEditable)
            pixmap = _crop_pixmap(page, region)
            if pixmap is not None:
                preview.setData(Qt.DecorationRole, pixmap)
            self.table.setItem(row, COL_PREVIEW, preview)

            text = QTableWidgetItem(region.text)
            text.setToolTip("Recognized text — edit to correct it")
            self.table.setItem(row, COL_TEXT, text)

            self.table.setItem(row, COL_CONF, _confidence_item(region.confidence))

            combo = QComboBox()
            combo.addItems(MODES)
            combo.setCurrentText(region.mode if region.mode in MODES else "trace")
            combo.currentTextChanged.connect(
                lambda value, r=region: self._on_mode_changed(r, value))
            self.table.setCellWidget(row, COL_MODE, combo)
            self._tint_row(row, region)

        self.table.resizeRowsToContents()
        self._loading = False
        self._update_summary()

    def refresh_results(self) -> None:
        """Re-read text and confidence from the regions after a recognition pass."""
        self._loading = True
        for row, region in enumerate(self.regions):
            item = self.table.item(row, COL_TEXT)
            if item is not None:
                item.setText(region.text)
            self.table.setItem(row, COL_CONF, _confidence_item(region.confidence))
            self._tint_row(row, region)
        self._loading = False
        self._update_summary()

    def hidden_region_ids(self) -> set[int]:
        """Regions whose traced strokes must not be drawn or exported."""
        return {r.id for r in self.regions if r.mode in ("hide", "hershey")}

    def hershey_regions(self) -> list:
        return [r for r in self.regions if r.mode == "hershey" and r.text.strip()]

    def set_all(self, mode: str) -> None:
        if not self.regions:
            return
        self._loading = True
        for row, region in enumerate(self.regions):
            region.mode = mode
            widget = self.table.cellWidget(row, COL_MODE)
            if widget is not None:
                widget.setCurrentText(mode)
            self._tint_row(row, region)
        self._loading = False
        self._update_summary()
        self.modes_changed.emit()

    def accept_confident(self) -> None:
        """Opt the confidently-read regions into Hershey, leaving the rest traced."""
        if not self.regions:
            return
        self._loading = True
        switched = 0
        for row, region in enumerate(self.regions):
            good = region.confidence >= LOW_CONFIDENCE and bool(region.text.strip())
            region.mode = "hershey" if good else "trace"
            switched += int(good)
            widget = self.table.cellWidget(row, COL_MODE)
            if widget is not None:
                widget.setCurrentText(region.mode)
            self._tint_row(row, region)
        self._loading = False
        self._update_summary()
        self.modes_changed.emit()

    def focus_region(self, region_id: int) -> None:
        for row, region in enumerate(self.regions):
            if region.id == region_id:
                self.table.selectRow(row)
                return

    # --- internals ---

    def _on_mode_changed(self, region, value: str) -> None:
        region.mode = value
        for row, candidate in enumerate(self.regions):
            if candidate is region:
                self._tint_row(row, region)
                break
        if not self._loading:
            self._update_summary()
            self.modes_changed.emit()

    def _on_item_changed(self, item) -> None:
        if self._loading or item.column() != COL_TEXT:
            return
        row = item.row()
        if not (0 <= row < len(self.regions)):
            return
        region = self.regions[row]
        region.text = item.text()
        self.text_edited.emit(region.id)

    def _on_row_selected(self) -> None:
        if self._loading:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.regions):
            self.region_selected.emit(self.regions[row].id)

    def _tint_row(self, row: int, region) -> None:
        """Amber where the recognizer was unsure, so review lands there first."""
        unsure = 0 <= region.confidence < LOW_CONFIDENCE
        brush = QBrush(QColor("#fdf1dc")) if unsure else QBrush(Qt.NoBrush)
        for col in (COL_ID, COL_PREVIEW, COL_TEXT, COL_CONF):
            item = self.table.item(row, col)
            if item is not None:
                item.setBackground(brush)

    def _update_summary(self) -> None:
        if not self.regions:
            self.summary.setText("no text detected")
            return
        hidden = sum(1 for r in self.regions if r.mode == "hide")
        hershey = sum(1 for r in self.regions if r.mode == "hershey")
        read = sum(1 for r in self.regions if r.text.strip())
        unsure = sum(1 for r in self.regions if 0 <= r.confidence < LOW_CONFIDENCE)
        self.summary.setText(
            f"{len(self.regions)} region(s) · {read} read · {unsure} low confidence · "
            f"{hershey} hershey · {hidden} hidden"
        )


def _confidence_item(confidence: float) -> QTableWidgetItem:
    item = QTableWidgetItem("—" if confidence < 0 else f"{confidence:.0f}")
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    item.setTextAlignment(Qt.AlignCenter)
    return item


def _crop_pixmap(page: Optional[np.ndarray], region) -> Optional[QPixmap]:
    """Thumbnail of the region straight from the page, for eyeballing the match."""
    if page is None:
        return None
    h, w = page.shape[:2]
    pad = 2
    y0 = max(0, region.y - pad)
    y1 = min(h, region.y + region.height + pad)
    x0 = max(0, region.x - pad)
    x1 = min(w, region.x + region.width + pad)
    if y1 <= y0 or x1 <= x0:
        return None

    crop = np.ascontiguousarray(page[y0:y1, x0:x1])
    image = QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1],
                   QImage.Format_Grayscale8).copy()
    pixmap = QPixmap.fromImage(image)
    if pixmap.height() > THUMB_HEIGHT:
        pixmap = pixmap.scaledToHeight(THUMB_HEIGHT, Qt.SmoothTransformation)
    if pixmap.width() > THUMB_MAX_WIDTH:
        pixmap = pixmap.scaledToWidth(THUMB_MAX_WIDTH, Qt.SmoothTransformation)
    return pixmap
