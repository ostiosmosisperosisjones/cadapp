"""
gui/fillet3d_panel.py

Fillet3DPanel — floating panel for 3D edge fillet operations.
Modelled after ExtrudePanel / ThickenPanel.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#Fillet3DPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel { color: #d4d4d4; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 13px; font-weight: bold; }
QLabel#section {
    color: #888; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px;
}
QPushButton {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 4px 10px; font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton:pressed { background: #222; }
QPushButton#ok {
    background: #1a4a7a; border-color: #4a90d9;
    color: #fff; padding: 5px 18px; font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
QPushButton#pick_face[active=true] {
    background: #2a1e1e; border-color: #d96a4a; color: #ff9977;
}
"""

_SEP_STYLE = "background: #333;"


class Fillet3DPanel(QWidget):
    confirmed           = pyqtSignal(float)
    cancelled           = pyqtSignal()
    preview_changed     = pyqtSignal(float)
    face_entry_removed  = pyqtSignal(int)
    picking_face_changed = pyqtSignal(bool)

    def __init__(self, workspace, parent=None):
        super().__init__(None)   # top-level so it's never clipped by the viewport
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint)
        self._viewport = parent
        self.setObjectName("Fillet3DPanel")
        self.setStyleSheet(_PANEL_STYLE)
        self.setFixedWidth(240)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace     = workspace
        self._picking_face  = False

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._fire_preview)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox
        from cad.prefs import prefs

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Fillet")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Faces ─────────────────────────────────────────────────────
        face_header = QHBoxLayout()
        face_header.setSpacing(6)
        face_header.addWidget(self._section_label("Faces"))
        face_header.addStretch()
        self._pick_face_btn = QPushButton("+ Add")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_header.addWidget(self._pick_face_btn)
        root.addLayout(face_header)

        self._face_list = SelectionList(empty_text="No faces selected")
        self._face_list.entry_removed.connect(self.face_entry_removed)
        root.addWidget(self._face_list)

        root.addWidget(self._separator())

        # ── Radius ────────────────────────────────────────────────────
        root.addWidget(self._section_label("Radius"))
        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(1.0)
        self._spinbox.value_changed.connect(self._on_radius_changed)
        root.addWidget(self._spinbox)

        root.addWidget(self._separator())

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setObjectName("ok")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(self._ok_btn)
        root.addLayout(btn_row)

    def _separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(_SEP_STYLE)
        sep.setFixedHeight(1)
        return sep

    def _section_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("section")
        return lbl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_radius(self, radius: float):
        self._spinbox.set_mm(radius)

    def add_face_entry(self, body_id: str | None, face_idx: int | None, label: str):
        key = (body_id, face_idx)
        self._face_list.add(key, label)

    def remove_face_entry(self, index: int):
        self._face_list.remove_at(index)

    def clear_face_entries(self):
        self._face_list.clear()

    @property
    def _has_faces(self) -> bool:
        return len(self._face_list) > 0

    def end_pick_face(self):
        self._picking_face = False
        self._pick_face_btn.setChecked(False)
        self._pick_face_btn.setProperty("active", False)
        self._pick_face_btn.style().unpolish(self._pick_face_btn)
        self._pick_face_btn.style().polish(self._pick_face_btn)
        self.picking_face_changed.emit(False)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _fire_preview(self):
        v = self._spinbox.mm_value()
        if v is not None and v > 0:
            self.preview_changed.emit(v)

    def _emit_preview(self):
        self._preview_timer.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_radius_changed(self, _):
        self._emit_preview()

    def _on_pick_face_toggle(self, checked: bool):
        if checked:
            self._picking_face = True
            self._pick_face_btn.setProperty("active", True)
            self._pick_face_btn.style().unpolish(self._pick_face_btn)
            self._pick_face_btn.style().polish(self._pick_face_btn)
            self.picking_face_changed.emit(True)
        else:
            self.end_pick_face()

    def _on_ok(self):
        if not self._has_faces:
            return
        v = self._spinbox.mm_value()
        if v is not None and v > 0:
            self.confirmed.emit(v)

    # ------------------------------------------------------------------
    # Keyboard / window
    # ------------------------------------------------------------------

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_face:
                self.end_pick_face()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
