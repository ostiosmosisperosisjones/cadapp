"""
gui/parts_panel.py

PartsPanel — lists every body in the workspace with a visibility toggle.

Signals
-------
body_selected(body_id: str | None)
    Emitted when the user clicks a body row. None if deselected.
body_visibility_changed(body_id: str, visible: bool)
    Emitted when the eye toggle is clicked.
"""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from cad.workspace import Workspace


_BG         = "#1e1e1e"
_BG_ROW     = "#2a2a2a"
_BG_SEL     = "#2f3f52"
_TEXT       = "#d4d4d4"
_TEXT_DIM   = "#555"
_BORDER_SEL = "#4a90d9"
_BORDER_ROW = "#333"

_EYE_ON  = "👁"
_EYE_OFF = "◌"
_BODY_ICON = "⬡"


class _BodyRow(QFrame):
    clicked            = pyqtSignal(str)
    visibility_toggled = pyqtSignal(str, bool)

    def __init__(self, body_id: str, name: str, visible: bool, selected: bool):
        super().__init__()
        self._body_id = body_id
        self._visible = visible
        self._selected = selected
        self._build(name)
        self._apply_style()

    def _build(self, name: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Visibility toggle
        self._eye_btn = QPushButton(_EYE_ON if self._visible else _EYE_OFF)
        self._eye_btn.setFixedSize(20, 20)
        self._eye_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #666;
                font-size: 13px;
                padding: 0;
            }
            QPushButton:hover { color: #aaa; }
        """)
        self._eye_btn.clicked.connect(self._toggle_visibility)
        layout.addWidget(self._eye_btn)

        # Body icon
        icon = QLabel(_BODY_ICON)
        icon.setStyleSheet(
            "color: #4a90d9; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(icon)

        # Name
        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet(
            f"color: {_TEXT if self._visible else _TEXT_DIM}; "
            f"font-size: 12px; background: transparent; border: none;")
        self._name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Preferred)
        layout.addWidget(self._name_lbl)

    def _apply_style(self):
        border = _BORDER_SEL if self._selected else _BORDER_ROW
        bg     = _BG_SEL     if self._selected else _BG_ROW
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border-left: 3px solid {border};
                border-radius: 3px;
                margin: 1px 4px;
            }}
        """)

    def _toggle_visibility(self):
        self._visible = not self._visible
        self._eye_btn.setText(_EYE_ON if self._visible else _EYE_OFF)
        self._name_lbl.setStyleSheet(
            f"color: {_TEXT if self._visible else _TEXT_DIM}; "
            f"font-size: 12px; background: transparent; border: none;")
        self.visibility_toggled.emit(self._body_id, self._visible)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._body_id)
        super().mousePressEvent(e)


class PartsPanel(QWidget):
    body_selected           = pyqtSignal(object)   # str | None
    body_visibility_changed = pyqtSignal(str, bool)

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self._workspace   = workspace
        self._selected_id: str | None = None
        self._visible:     dict[str, bool] = {}
        self._rows:        dict[str, _BodyRow] = {}
        self._setup()
        self.refresh()

    def _setup(self):
        self.setStyleSheet(f"background: {_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QLabel("  Parts")
        header.setFixedHeight(28)
        header.setStyleSheet("""
            background: #161616;
            color: #555;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            border-bottom: 1px solid #2a2a2a;
        """)
        root.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("border: none; background: transparent;")

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        root.addWidget(self._scroll, stretch=1)

    def refresh(self):
        """Rebuild rows from current workspace bodies."""
        # Clear existing rows
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        for body_id, body in self._workspace.bodies.items():
            visible = self._visible.get(body_id, True)
            selected = (body_id == self._selected_id)
            row = _BodyRow(body_id, body.name, visible, selected)
            row.clicked.connect(self._on_row_clicked)
            row.visibility_toggled.connect(self._on_visibility_toggled)
            self._rows[body_id] = row
            self._list_layout.insertWidget(
                self._list_layout.count() - 1, row)

    def _on_row_clicked(self, body_id: str):
        # Toggle deselect if clicking the already-selected body
        if self._selected_id == body_id:
            self._selected_id = None
            self.body_selected.emit(None)
        else:
            self._selected_id = body_id
            self.body_selected.emit(body_id)
        # Update selection highlight on all rows
        for bid, row in self._rows.items():
            row.set_selected(bid == self._selected_id)

    def _on_visibility_toggled(self, body_id: str, visible: bool):
        self._visible[body_id] = visible
        self.body_visibility_changed.emit(body_id, visible)

    def selected_body_id(self) -> str | None:
        return self._selected_id

    def set_selected_body(self, body_id: str | None):
        """Driven externally (e.g. from viewport click) — updates highlight
        without emitting body_selected to avoid signal loops."""
        self._selected_id = body_id
        for bid, row in self._rows.items():
            row.set_selected(bid == self._selected_id)

    def is_body_visible(self, body_id: str) -> bool:
        return self._visible.get(body_id, True)
