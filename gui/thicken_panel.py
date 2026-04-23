"""
gui/thicken_panel.py

ThickenPanel — floating non-modal panel for the thicken operation.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QKeyEvent

from gui.expr_spinbox import ExprSpinBox
from cad.prefs import prefs


_PANEL_STYLE = """
QWidget#ThickenPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel {
    color: #d4d4d4;
    font-size: 12px;
}
QLabel#title {
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
}
QLabel#section {
    color: #888;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QPushButton {
    background: #2a2a2a;
    color: #d4d4d4;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton:pressed { background: #222; }
QPushButton#ok {
    background: #1a4a7a;
    border-color: #4a90d9;
    color: #fff;
    padding: 5px 18px;
    font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
QRadioButton {
    color: #d4d4d4;
    font-size: 12px;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 13px; height: 13px;
    border-radius: 7px;
    border: 1px solid #555;
    background: #2a2a2a;
}
QRadioButton::indicator:checked {
    background: #4a90d9;
    border-color: #4a90d9;
}
"""

_SEP_STYLE = "background: #333;"


class ThickenPanel(QWidget):
    thicken_requested = pyqtSignal(float)   # thickness_mm (signed: positive=grow, negative=cut)
    cancelled         = pyqtSignal()
    preview_changed   = pyqtSignal(float)   # thickness_mm for live preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ThickenPanel")
        self.setStyleSheet(_PANEL_STYLE)
        self.setFixedWidth(220)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Thicken / Cut")
        title.setObjectName("title")
        layout.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(_SEP_STYLE); sep.setFixedHeight(1)
        layout.addWidget(sep)

        mode_lbl = QLabel("MODE")
        mode_lbl.setObjectName("section")
        layout.addWidget(mode_lbl)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._radio_thicken = QRadioButton("Thicken")
        self._radio_cut     = QRadioButton("Cut")
        self._radio_thicken.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_thicken, 0)
        self._mode_group.addButton(self._radio_cut,     1)
        self._mode_group.idClicked.connect(self._schedule_preview)
        mode_row.addWidget(self._radio_thicken)
        mode_row.addWidget(self._radio_cut)
        layout.addLayout(mode_row)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(_SEP_STYLE); sep2.setFixedHeight(1)
        layout.addWidget(sep2)

        thick_lbl = QLabel("THICKNESS")
        thick_lbl.setObjectName("section")
        layout.addWidget(thick_lbl)

        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(1.0)
        layout.addWidget(self._spinbox)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(_SEP_STYLE); sep3.setFixedHeight(1)
        layout.addWidget(sep3)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("ok")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self._fire_preview)

        self._spinbox.value_changed.connect(self._schedule_preview)

    def _signed_value(self) -> float | None:
        val = self._spinbox.mm_value()
        if val is None:
            return None
        val = abs(val)
        if self._mode_group.checkedId() == 1:  # Cut
            val = -val
        return val

    def _on_ok(self):
        val = self._signed_value()
        if val is not None and val != 0:
            self.thicken_requested.emit(val)

    def _schedule_preview(self, _val=None):
        self._preview_timer.start()

    def _fire_preview(self):
        val = self._signed_value()
        if val is not None:
            self.preview_changed.emit(val)

    def _emit_preview(self, _val=None):
        self._preview_timer.start()

    def set_thickness(self, thickness_mm: float):
        """Restore panel state from a saved thickness (signed)."""
        if thickness_mm < 0:
            self._radio_cut.setChecked(True)
        else:
            self._radio_thicken.setChecked(True)
        self._spinbox.set_mm(abs(thickness_mm))

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)
