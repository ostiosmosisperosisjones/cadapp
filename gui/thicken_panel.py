"""
gui/thicken_panel.py

ThickenPanel — floating non-modal panel for the thicken operation.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame, QSizePolicy,
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
QPushButton#pick_face[active=true] {
    background: #2a1e1e;
    border-color: #d96a4a;
    color: #ff9977;
}
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
    face_entry_removed = pyqtSignal(int)    # index of removed face entry

    def __init__(self, parent=None):
        super().__init__(None)   # top-level so it's never clipped by the viewport
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint)
        self._viewport = parent
        self.setObjectName("ThickenPanel")
        self.setStyleSheet(_PANEL_STYLE)
        self.setFixedWidth(240)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._has_face     = False
        self._face_entries : list = []   # list of (body_id, face_idx, label, valid)
        self._picking_face : bool = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Thicken / Cut")
        title.setObjectName("title")
        layout.addWidget(title)

        layout.addWidget(self._separator())

        # ── Faces ──────────────────────────────────────────────────────
        face_header = QHBoxLayout()
        face_header.setSpacing(6)
        face_header.addWidget(self._section_label("Faces"))
        face_header.addStretch()
        self._pick_face_btn = QPushButton("+ Add Face")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_header.addWidget(self._pick_face_btn)
        layout.addLayout(face_header)

        self._face_list_widget = QWidget()
        self._face_list_layout = QVBoxLayout(self._face_list_widget)
        self._face_list_layout.setContentsMargins(0, 0, 0, 0)
        self._face_list_layout.setSpacing(2)
        self._face_empty_label = QLabel("No faces selected")
        self._face_empty_label.setStyleSheet(
            "color: #555; font-size: 11px; font-family: monospace;")
        self._face_list_layout.addWidget(self._face_empty_label)
        layout.addWidget(self._face_list_widget)

        layout.addWidget(self._separator())

        # ── Mode ───────────────────────────────────────────────────────
        mode_lbl = self._section_label("Mode")
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

        layout.addWidget(self._separator())

        # ── Thickness ──────────────────────────────────────────────────
        thick_lbl = self._section_label("Thickness")
        layout.addWidget(thick_lbl)

        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(1.0)
        layout.addWidget(self._spinbox)

        layout.addWidget(self._separator())

        # ── Buttons ────────────────────────────────────────────────────
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    # Face list API
    # ------------------------------------------------------------------

    def add_face_entry(self, body_id: str, face_idx: int, label: str, valid: bool = True):
        for entry in self._face_entries:
            if entry[0] == body_id and entry[1] == face_idx:
                return
        self._face_entries.append((body_id, face_idx, label, valid, ""))
        self._rebuild_face_list()

    def remove_face_entry(self, index: int):
        if 0 <= index < len(self._face_entries):
            self._face_entries.pop(index)
            self._rebuild_face_list()
            self.face_entry_removed.emit(index)

    def clear_face_entries(self):
        self._face_entries.clear()
        self._rebuild_face_list()

    def set_face_entry_error(self, index: int, error_msg: str):
        """Mark a face entry as invalid with a tooltip error message."""
        if 0 <= index < len(self._face_entries):
            body_id, face_idx, label, _valid, _tip = self._face_entries[index]
            self._face_entries[index] = (body_id, face_idx, label, False, error_msg)
            self._rebuild_face_list()

    def clear_face_errors(self):
        """Reset all face entries to valid state."""
        changed = False
        for i, (body_id, face_idx, label, valid, tip) in enumerate(self._face_entries):
            if not valid:
                self._face_entries[i] = (body_id, face_idx, label, True, "")
                changed = True
        if changed:
            self._rebuild_face_list()

    def _rebuild_face_list(self):
        while self._face_list_layout.count():
            item = self._face_list_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._face_empty_label:
                w.deleteLater()

        self._has_face = bool(self._face_entries) and all(e[3] for e in self._face_entries)

        if not self._face_entries:
            self._face_list_layout.addWidget(self._face_empty_label)
            self._face_empty_label.show()
            return

        self._face_empty_label.hide()
        for i, (body_id, face_idx, label, valid, tooltip) in enumerate(self._face_entries):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            lbl = QLabel(label)
            color = "#d4d4d4" if valid else "#cc4444"
            lbl.setStyleSheet(
                f"color: {color}; font-size: 11px; font-family: monospace;")
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            if tooltip:
                lbl.setToolTip(tooltip)
            row_layout.addWidget(lbl)
            rm_btn = QPushButton("✕")
            rm_btn.setFixedWidth(22)
            rm_btn.setStyleSheet(
                "QPushButton { background: #2a2a2a; color: #666; border: none; "
                "font-size: 10px; padding: 1px; } "
                "QPushButton:hover { color: #cc4444; }")
            idx_capture = i
            rm_btn.clicked.connect(lambda _, idx=idx_capture: self.remove_face_entry(idx))
            row_layout.addWidget(rm_btn)
            self._face_list_layout.addWidget(row)

    # ------------------------------------------------------------------
    # Pick face toggle
    # ------------------------------------------------------------------

    def _on_pick_face_toggle(self, checked: bool):
        if checked:
            self._picking_face = True
            self._pick_face_btn.setProperty("active", True)
            self._pick_face_btn.style().unpolish(self._pick_face_btn)
            self._pick_face_btn.style().polish(self._pick_face_btn)
        else:
            self._end_pick_face()

    def end_pick_face(self):
        self._end_pick_face()

    def _end_pick_face(self):
        self._picking_face = False
        self._pick_face_btn.setChecked(False)
        self._pick_face_btn.setProperty("active", False)
        self._pick_face_btn.style().unpolish(self._pick_face_btn)
        self._pick_face_btn.style().polish(self._pick_face_btn)

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------

    def _signed_value(self) -> float | None:
        val = self._spinbox.mm_value()
        if val is None:
            return None
        val = abs(val)
        if self._mode_group.checkedId() == 1:  # Cut
            val = -val
        return val

    def _on_ok(self):
        if not self._has_face:
            return
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

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_face:
                self._end_pick_face()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
