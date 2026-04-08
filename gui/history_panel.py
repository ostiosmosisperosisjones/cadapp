"""
gui/history_panel.py

Sidebar showing the flat workspace history, grouped visually by body.
Double-click an entry to edit its parameters (parametric replay).
"""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QDialog, QFormLayout,
    QDoubleSpinBox, QDialogButtonBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from cad.history import History, HistoryEntry
from cad.workspace import Workspace
from cad.operations import EDIT_SCHEMA

_BG          = "#1e1e1e"
_BG_ENTRY    = "#2a2a2a"
_BG_CURRENT  = "#2f3f52"
_BG_FUTURE   = "#212121"
_TEXT        = "#d4d4d4"
_TEXT_FUTURE = "#484848"
_BORDER_SEL  = "#4a90d9"
_BORDER_PAST = "#404040"
_BORDER_FUTURE = "#2a2a2a"

# Per-operation accent colours (left border only, subtle)
_OP_ACCENT = {
    "import":  "#505050",
    "extrude": "#3a6e44",
    "cut":     "#6e3a3a",
}


def _op_icon(op: str) -> str:
    return {"import": "⬡", "extrude": "▲", "cut": "▼"}.get(op, "•")


# ---------------------------------------------------------------------------
# Generic edit dialog — driven entirely by EDIT_SCHEMA
# ---------------------------------------------------------------------------

class _EditDialog(QDialog):
    def __init__(self, entry: HistoryEntry, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit — {entry.label}")
        self.setMinimumWidth(300)
        self.result_params = None

        schema = EDIT_SCHEMA.get(entry.operation, [])
        if not schema:
            self.result_params = entry.params.copy()
            return

        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(10)

        self._spinboxes: dict[str, QDoubleSpinBox] = {}

        # Reconstruct signed value
        signed_val = float(entry.params.get("distance", 0))
        if entry.operation == "cut":
            signed_val = -abs(signed_val)

        for (key, label, typ, mn, mx, decimals) in schema:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setMinimum(-mx)
            spin.setMaximum(mx)
            spin.setSuffix(" mm")
            spin.setValue(signed_val if key == "distance"
                          else float(entry.params.get(key, 0)))
            self._spinboxes[key] = spin
            layout.addRow(label + "  (±):", spin)

        note = QLabel("Positive = add material   /   Negative = cut")
        note.setStyleSheet("color: #555; font-size: 11px;")
        layout.addRow(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_accept(self):
        self.result_params = {k: s.value() for k, s in self._spinboxes.items()}
        self.accept()


# ---------------------------------------------------------------------------
# Entry widget
# ---------------------------------------------------------------------------

class _EntryWidget(QFrame):
    clicked        = pyqtSignal(int)
    double_clicked = pyqtSignal(int)

    def __init__(self, entry: HistoryEntry, index: int,
                 is_current: bool, is_future: bool, body_name: str):
        super().__init__()
        self._index = index
        editable = entry.operation in EDIT_SCHEMA

        if is_current:
            border = _BORDER_SEL
            bg     = _BG_CURRENT
        elif is_future:
            border = _BORDER_FUTURE
            bg     = _BG_FUTURE
        else:
            border = _OP_ACCENT.get(entry.operation, _BORDER_PAST)
            bg     = _BG_ENTRY

        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border-left: 3px solid {border};
                border-radius: 3px;
                margin: 1px 4px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 5, 10, 5)
        outer.setSpacing(1)

        # Body name (dim, small)
        if entry.operation != "import":
            body_lbl = QLabel(body_name)
            body_lbl.setStyleSheet(
                f"color: #484848; font-size: 10px; border: none; background: transparent;")
            outer.addWidget(body_lbl)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.setContentsMargins(0, 0, 0, 0)

        icon_color = border if not is_future else "#333"
        tag = QLabel(_op_icon(entry.operation))
        tag.setFixedWidth(16)
        tag.setStyleSheet(
            f"color: {icon_color}; font-size: 11px; border: none; background: transparent;")
        row.addWidget(tag)

        text_color = _TEXT_FUTURE if is_future else _TEXT
        lbl = QLabel(entry.label)
        lbl.setStyleSheet(
            f"color: {text_color}; font-size: 12px; border: none; background: transparent;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                          QSizePolicy.Policy.Preferred)
        row.addWidget(lbl)

        if editable and not is_future:
            hint = QLabel("✎")
            hint.setStyleSheet("color: #383838; font-size: 10px; border: none; background: transparent;")
            hint.setToolTip("Double-click to edit")
            row.addWidget(hint)

        outer.addLayout(row)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self._index)
        super().mouseDoubleClickEvent(e)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class HistoryPanel(QWidget):
    seek_requested   = pyqtSignal(int)
    replay_requested = pyqtSignal(int)

    def __init__(self, workspace: Workspace, history: History, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._history   = history
        self._setup()
        self.refresh()

    def _setup(self):
        self.setMinimumWidth(190)
        self.setMaximumWidth(270)
        self.setStyleSheet(f"background: {_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QLabel("  History")
        header.setFixedHeight(32)
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

        btn_style = """
            QPushButton {
                background: #222;
                color: #666;
                border: none;
                border-top: 1px solid #2a2a2a;
                font-size: 12px;
                padding: 8px 0;
            }
            QPushButton:hover   { background: #2a2a2a; color: #d4d4d4; }
            QPushButton:pressed { background: #1a1a1a; }
            QPushButton:disabled { color: #333; }
        """

        btn_row = QHBoxLayout()
        btn_row.setSpacing(0)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self._undo_btn = QPushButton("↩ Undo")
        self._undo_btn.setStyleSheet(
            btn_style + "QPushButton { border-right: 1px solid #2a2a2a; }")
        self._undo_btn.clicked.connect(
            lambda: self.seek_requested.emit(self._history.cursor - 1))
        btn_row.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("Redo ↪")
        self._redo_btn.setStyleSheet(btn_style)
        self._redo_btn.clicked.connect(
            lambda: self.seek_requested.emit(self._history.cursor + 1))
        btn_row.addWidget(self._redo_btn)

        bc = QWidget()
        bc.setLayout(btn_row)
        root.addWidget(bc)

    # ------------------------------------------------------------------

    def refresh(self):
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = self._history.entries
        cursor  = self._history.cursor

        for i, entry in enumerate(entries):
            body = self._workspace.bodies.get(entry.body_id)
            body_name = body.name if body else "?"
            w = _EntryWidget(entry, i,
                             is_current=(i == cursor),
                             is_future=(i > cursor),
                             body_name=body_name)
            w.clicked.connect(self._on_entry_clicked)
            w.double_clicked.connect(self._on_entry_double_clicked)
            self._list_layout.insertWidget(i, w)

        self._undo_btn.setEnabled(self._history.can_undo)
        self._redo_btn.setEnabled(self._history.can_redo)
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum())

    # ------------------------------------------------------------------

    def _on_entry_clicked(self, index: int):
        self.seek_requested.emit(index)

    def _on_entry_double_clicked(self, index: int):
        entries = self._history.entries
        if index >= len(entries):
            return
        entry = entries[index]

        if entry.operation not in EDIT_SCHEMA:
            return

        if index > self._history.cursor:
            QMessageBox.information(
                self, "Can't edit future entry",
                "Single-click to seek here first, then double-click to edit.")
            return

        dlg = _EditDialog(entry, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_params is None:
            return

        dist = dlg.result_params.get("distance", 0)
        if dist < 0:
            entry.operation = "cut"
            entry.params["distance"] = abs(dist)
        elif dist > 0:
            entry.operation = "extrude"
            entry.params["distance"] = abs(dist)
        else:
            return

        self.replay_requested.emit(index)
