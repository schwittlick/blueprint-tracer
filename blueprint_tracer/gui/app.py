"""GUI entry point: ``blueprint-tracer-gui [image]``."""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from blueprint_tracer.gui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("blueprint-tracer")

    window = MainWindow()
    window.show()

    for arg in argv[1:]:
        if not arg.startswith("-") and os.path.exists(arg):
            window.load_image(arg)
            break

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
