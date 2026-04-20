"""
gui/thicken_panel.py

ThickenPanel — floating non-modal panel for the thicken operation.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
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
QLabel#hint {
    color: #888;
    font-size: 11px;
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
"""


class ThickenPanel(QWidget):
    thicken_requested = pyqtSignal(float)   # thickness_mm (signed)
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

        title = QLabel("Thicken")
        title.setObjectName("title")
        layout.addWidget(title)

        hint = QLabel("positive = grow  ·  negative = shrink")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(1.0)
        layout.addWidget(self._spinbox)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("ok")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self._fire_preview)

        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.cancelled)
        self._spinbox.value_changed.connect(self._schedule_preview)

    def _on_ok(self):
        val = self._spinbox.mm_value()
        if val is not None and val != 0:
            self.thicken_requested.emit(val)

    def _schedule_preview(self, _val=None):
        self._preview_timer.start()

    def _fire_preview(self):
        val = self._spinbox.mm_value()
        if val is not None:
            self.preview_changed.emit(val)

    def _emit_preview(self, _val=None):
        self._preview_timer.start()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)
