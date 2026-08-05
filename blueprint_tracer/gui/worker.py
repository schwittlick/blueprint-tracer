"""Background tracing so parameter edits never block the UI.

A single worker thread owns the pipeline. Requests supersede one another: while a
trace is running, newer requests replace the queued one, so dragging a slider
runs at most one extra trace rather than a backlog of them.
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, QWaitCondition, Signal

from blueprint_tracer.core.config import Config
from blueprint_tracer.core.pipeline import TraceResult, run


class OcrWorker(QThread):
    """Recognizes regions off the UI thread; a dense sheet has hundreds of them."""

    progress = Signal(int, int)         # done, total
    finished_ocr = Signal(int)          # regions with text
    failed = Signal(str)

    def __init__(self, page, regions, lang: str, parent=None) -> None:
        super().__init__(parent)
        self._page = page
        self._regions = regions
        self._lang = lang
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        from blueprint_tracer.core import ocr

        try:
            ocr.check_ready(self._lang)
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(str(exc))
            return

        found = 0
        total = len(self._regions)
        for i, region in enumerate(self._regions, 1):
            if self._cancel:
                break
            try:
                result = ocr.recognize_region(self._page, region, lang=self._lang)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))
                return
            region.text = result.text
            region.confidence = result.confidence
            found += bool(result.text)
            if i % 5 == 0 or i == total:
                self.progress.emit(i, total)
        self.finished_ocr.emit(found)


class TraceWorker(QThread):
    finished_trace = Signal(object, bool)   # TraceResult, is_preview
    failed = Signal(str)
    started_trace = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._wake = QWaitCondition()
        self._pending: Optional[tuple] = None
        self._abort = False

    def request(self, gray: np.ndarray, cfg: Config, dpi, preview: bool) -> None:
        with QMutexLocker(self._mutex):
            self._pending = (gray, cfg, dpi, preview)
            self._wake.wakeAll()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._abort = True
            self._wake.wakeAll()
        self.wait(3000)

    def run(self) -> None:  # noqa: D102 - QThread entry point
        while True:
            with QMutexLocker(self._mutex):
                while self._pending is None and not self._abort:
                    self._wake.wait(self._mutex)
                if self._abort:
                    return
                job = self._pending
                self._pending = None

            gray, cfg, dpi, preview = job
            self.started_trace.emit()
            try:
                result = run(gray, cfg, dpi=dpi)
                self.finished_trace.emit(result, preview)
            except Exception:
                self.failed.emit(traceback.format_exc(limit=3))
