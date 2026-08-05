"""Parameter dock: grouped controls that emit a debounced change signal.

Auto-scaled parameters show an "Auto" checkbox; unticking it pins the value.
That mirrors Config, where ``None`` means "derive from the measured stroke width".
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from blueprint_tracer.core.config import Config


class AutoField(QWidget):
    """A spin box paired with an Auto toggle; Auto means the config value is None."""

    changed = Signal()

    def __init__(self, minimum, maximum, step, decimals=0, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if decimals:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
        else:
            self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.auto = QCheckBox("Auto")
        self.auto.setChecked(True)
        self.spin.setEnabled(False)

        layout.addWidget(self.spin, 1)
        layout.addWidget(self.auto)

        self.auto.toggled.connect(self._on_auto)
        self.spin.valueChanged.connect(lambda _: self.changed.emit())

    def _on_auto(self, checked: bool) -> None:
        self.spin.setEnabled(not checked)
        self.changed.emit()

    def value(self):
        return None if self.auto.isChecked() else self.spin.value()

    def set_value(self, value) -> None:
        blocked = self.blockSignals(True)
        if value is None:
            self.auto.setChecked(True)
            self.spin.setEnabled(False)
        else:
            self.auto.setChecked(False)
            self.spin.setEnabled(True)
            self.spin.setValue(value)
        self.blockSignals(blocked)

    def show_resolved(self, value) -> None:
        """Display the auto-derived value without making it look user-set."""
        if self.auto.isChecked() and value is not None:
            blocked = self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(blocked)


class ParamsPanel(QScrollArea):
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        inner = QWidget()
        self.setWidget(inner)
        root = QVBoxLayout(inner)
        root.setSpacing(10)

        # --- scale ---
        box = QGroupBox("Scale")
        form = QFormLayout(box)
        self.supersample = QDoubleSpinBox()
        self.supersample.setRange(0.25, 4.0)
        self.supersample.setSingleStep(0.5)
        self.supersample.setValue(1.0)
        self.supersample.setToolTip(
            "Upscale before tracing. Values above 1 recover small text on\n"
            "low-resolution scans at the cost of speed."
        )
        form.addRow("Supersample", self.supersample)
        self.auto_scale = QCheckBox("Scale parameters to stroke width")
        self.auto_scale.setChecked(True)
        self.auto_scale.setToolTip(
            "Derive window/spur/epsilon from the measured stroke width.\n"
            "Fixed pixel values destroy small text on low-resolution scans."
        )
        form.addRow(self.auto_scale)
        root.addWidget(box)

        # --- preprocess ---
        box = QGroupBox("Preprocess")
        form = QFormLayout(box)
        self.invert_auto = QCheckBox("Auto-detect polarity")
        self.invert_auto.setChecked(True)
        self.flatfield = QCheckBox("Flat-field (even out lighting)")
        self.flatfield.setChecked(True)
        self.deskew = QCheckBox("Auto deskew")
        self.deskew.setChecked(True)
        self.manual_angle = QDoubleSpinBox()
        self.manual_angle.setRange(-45.0, 45.0)
        self.manual_angle.setDecimals(2)
        self.manual_angle.setSingleStep(0.25)
        self.use_manual_angle = QCheckBox("Manual angle")
        form.addRow(self.invert_auto)
        form.addRow(self.flatfield)
        form.addRow(self.deskew)
        form.addRow(self.use_manual_angle, self.manual_angle)
        root.addWidget(box)

        # --- binarize ---
        box = QGroupBox("Binarize")
        form = QFormLayout(box)
        self.method = QComboBox()
        self.method.addItems(["sauvola", "adaptive", "otsu"])
        self.sauvola_window = AutoField(3, 201, 2)
        self.sauvola_k = QDoubleSpinBox()
        self.sauvola_k.setRange(0.0, 0.6)
        self.sauvola_k.setSingleStep(0.01)
        self.sauvola_k.setDecimals(3)
        self.sauvola_k.setValue(0.08)
        self.sauvola_k.setToolTip(
            "Lower keeps faint and thin strokes solid; higher erodes them."
        )
        self.fill_solid = QCheckBox("Fill solid ink (arrowheads, bars)")
        self.fill_solid.setChecked(True)
        form.addRow("Method", self.method)
        form.addRow("Window", self.sauvola_window)
        form.addRow("k", self.sauvola_k)
        form.addRow(self.fill_solid)
        root.addWidget(box)

        # --- solid regions ---
        box = QGroupBox("Solid regions")
        form = QFormLayout(box)
        self.solid_mode = QComboBox()
        self.solid_mode.addItems(["outline", "skeleton", "ignore"])
        self.solid_mode.setToolTip(
            "How to handle filled shapes (arrowheads, blacked-out labels, ink blots):\n"
            "  outline   – trace their boundary (a solid shape has no meaningful centerline)\n"
            "  skeleton  – reduce them to a medial axis\n"
            "  ignore    – drop them entirely"
        )
        self.solid_min_width = AutoField(2.0, 400.0, 1.0, decimals=1)
        self.solid_min_width.setToolTip("Ink at least this wide counts as solid, not a stroke.")
        form.addRow("Mode", self.solid_mode)
        form.addRow("Min width", self.solid_min_width)
        root.addWidget(box)

        # --- cleanup ---
        box = QGroupBox("Cleanup")
        form = QFormLayout(box)
        self.despeckle = AutoField(0, 500, 1)
        self.close_gaps = QSpinBox()
        self.close_gaps.setRange(0, 15)
        self.remove_border = QCheckBox("Remove edge-touching scan cruft")
        form.addRow("Despeckle area", self.despeckle)
        form.addRow("Close gaps", self.close_gaps)
        form.addRow(self.remove_border)
        root.addWidget(box)

        # --- trace ---
        box = QGroupBox("Trace and simplify")
        form = QFormLayout(box)
        self.spur = AutoField(0.0, 60.0, 0.5, decimals=2)
        self.min_len = AutoField(0.0, 60.0, 0.5, decimals=2)
        self.rdp = AutoField(0.0, 20.0, 0.1, decimals=2)
        form.addRow("Spur prune", self.spur)
        form.addRow("Min length", self.min_len)
        form.addRow("RDP epsilon", self.rdp)
        root.addWidget(box)

        # --- plotting ---
        box = QGroupBox("Plotter output")
        form = QFormLayout(box)
        self.join_paths = QCheckBox("Join collinear strokes")
        self.join_paths.setChecked(True)
        self.join_angle = QDoubleSpinBox()
        self.join_angle.setRange(0.0, 90.0)
        self.join_angle.setValue(20.0)
        self.plot_order = QCheckBox("Optimize pen-up travel")
        self.plot_order.setChecked(True)
        form.addRow(self.join_paths)
        form.addRow("Join angle", self.join_angle)
        form.addRow(self.plot_order)
        root.addWidget(box)

        self.resolved_label = QLabel("")
        self.resolved_label.setWordWrap(True)
        self.resolved_label.setStyleSheet("color:#666;font-size:11px;")
        root.addWidget(self.resolved_label)
        root.addStretch(1)

        for w in (self.invert_auto, self.flatfield, self.deskew, self.use_manual_angle,
                  self.fill_solid, self.remove_border, self.join_paths, self.plot_order,
                  self.auto_scale):
            w.toggled.connect(self.changed)
        for w in (self.sauvola_k, self.manual_angle, self.join_angle, self.supersample):
            w.valueChanged.connect(self.changed)
        for w in (self.close_gaps,):
            w.valueChanged.connect(self.changed)
        self.method.currentTextChanged.connect(self.changed)
        self.solid_mode.currentTextChanged.connect(self.changed)
        for w in (self.sauvola_window, self.despeckle, self.spur, self.min_len,
                  self.rdp, self.solid_min_width):
            w.changed.connect(self.changed)

    # --- config bridge ---

    def to_config(self, base: Optional[Config] = None) -> Config:
        cfg = base.copy() if base else Config()
        cfg.supersample = float(self.supersample.value())
        cfg.auto_scale = self.auto_scale.isChecked()
        cfg.invert_auto = self.invert_auto.isChecked()
        cfg.flatfield = self.flatfield.isChecked()
        cfg.deskew = self.deskew.isChecked()
        cfg.manual_angle = (
            float(self.manual_angle.value()) if self.use_manual_angle.isChecked() else None
        )
        cfg.threshold_method = self.method.currentText()
        cfg.sauvola_window = self.sauvola_window.value()
        cfg.sauvola_k = float(self.sauvola_k.value())
        cfg.fill_solid = self.fill_solid.isChecked()
        cfg.solid_mode = self.solid_mode.currentText()
        cfg.solid_min_width = self.solid_min_width.value()
        cfg.despeckle_min_area = self.despeckle.value()
        cfg.close_gaps = int(self.close_gaps.value())
        cfg.remove_border = self.remove_border.isChecked()
        cfg.spur_length = self.spur.value()
        cfg.min_path_length = self.min_len.value()
        cfg.rdp_epsilon = self.rdp.value()
        cfg.join_paths = self.join_paths.isChecked()
        cfg.join_max_angle = float(self.join_angle.value())
        cfg.plot_order = self.plot_order.isChecked()
        return cfg

    def from_config(self, cfg: Config) -> None:
        blocked = self.blockSignals(True)
        self.supersample.setValue(cfg.supersample or 1.0)
        self.auto_scale.setChecked(cfg.auto_scale)
        self.invert_auto.setChecked(cfg.invert_auto)
        self.flatfield.setChecked(cfg.flatfield)
        self.deskew.setChecked(cfg.deskew)
        self.use_manual_angle.setChecked(cfg.manual_angle is not None)
        if cfg.manual_angle is not None:
            self.manual_angle.setValue(cfg.manual_angle)
        self.method.setCurrentText(cfg.threshold_method)
        self.sauvola_window.set_value(cfg.sauvola_window)
        self.sauvola_k.setValue(cfg.sauvola_k)
        self.fill_solid.setChecked(cfg.fill_solid)
        self.solid_mode.setCurrentText(cfg.solid_mode)
        self.solid_min_width.set_value(cfg.solid_min_width)
        self.despeckle.set_value(cfg.despeckle_min_area)
        self.close_gaps.setValue(cfg.close_gaps)
        self.remove_border.setChecked(cfg.remove_border)
        self.spur.set_value(cfg.spur_length)
        self.min_len.set_value(cfg.min_path_length)
        self.rdp.set_value(cfg.rdp_epsilon)
        self.join_paths.setChecked(cfg.join_paths)
        self.join_angle.setValue(cfg.join_max_angle)
        self.plot_order.setChecked(cfg.plot_order)
        self.blockSignals(blocked)

    def show_resolved(self, stats: dict) -> None:
        r = (stats or {}).get("resolved") or {}
        if not r:
            self.resolved_label.setText("")
            return
        self.sauvola_window.show_resolved(r.get("sauvola_window"))
        self.despeckle.show_resolved(r.get("despeckle_min_area"))
        self.spur.show_resolved(r.get("spur_length"))
        self.min_len.show_resolved(r.get("min_path_length"))
        self.rdp.show_resolved(r.get("rdp_epsilon"))
        self.solid_min_width.show_resolved(r.get("solid_min_width"))
        self.resolved_label.setText(
            f"Measured stroke width {stats.get('stroke_width_px')} px — "
            f"auto values derived from it are shown greyed."
        )
