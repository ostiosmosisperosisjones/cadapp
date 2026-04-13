"""
gui/extrude_panel.py

ExtrudePanel — floating non-modal panel for the extrude/cut operation.
"""

from __future__ import annotations
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QComboBox, QFrame, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent


_PANEL_STYLE = """
QWidget#ExtrudePanel {
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
QPushButton#pick_edge {
    color: #aaa;
    font-size: 11px;
}
QPushButton#pick_edge[active=true] {
    background: #2a3a1e;
    border-color: #6aaa44;
    color: #8dcc5a;
}
QPushButton#pick_vertex {
    color: #aaa;
    font-size: 11px;
}
QPushButton#pick_vertex[active=true] {
    background: #2a1e3a;
    border-color: #aa6aee;
    color: #cc8dff;
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
QComboBox {
    background: #2a2a2a;
    color: #d4d4d4;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 12px;
}
QComboBox:disabled { color: #555; border-color: #333; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #252525;
    color: #d4d4d4;
    selection-background-color: #2a4a6a;
}
"""

_SEP_STYLE = "background: #333;"


class ExtrudePanel(QWidget):
    extrude_requested      = pyqtSignal(float, object, object)  # (dist_mm, dir|None, body_id|None)
    cancelled              = pyqtSignal()
    picking_edge_changed   = pyqtSignal(bool)
    picking_vertex_changed = pyqtSignal(bool)
    preview_changed        = pyqtSignal(float, object)          # (signed_dist_mm, dir|None)

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.setObjectName("ExtrudePanel")
        self.setStyleSheet(_PANEL_STYLE)
        self.setFixedWidth(260)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace      = workspace
        self._direction      : np.ndarray | None = None
        self._picking_edge   : bool = False
        self._picking_vertex : bool = False

        # Vertex target state
        self._target_vertex  : np.ndarray | None = None
        self._face_origin    : np.ndarray | None = None
        self._face_normal    : np.ndarray | None = None
        self._vertex_dist_mm : float | None = None

        self._build_ui()
        self._refresh_bodies()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox
        from cad.prefs import prefs

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        # Title
        title = QLabel("Extrude / Cut")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Mode: Extrude / Cut ───────────────────────────────────────
        root.addWidget(self._section_label("Mode"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._radio_extrude = QRadioButton("Extrude")
        self._radio_cut     = QRadioButton("Cut")
        self._radio_extrude.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_extrude, 0)
        self._mode_group.addButton(self._radio_cut,     1)
        self._mode_group.idClicked.connect(self._emit_preview)
        mode_row.addWidget(self._radio_extrude)
        mode_row.addWidget(self._radio_cut)
        root.addLayout(mode_row)

        root.addWidget(self._separator())

        # ── Target: Distance / Up to Vertex ───────────────────────────
        root.addWidget(self._section_label("Target"))

        target_row = QHBoxLayout()
        target_row.setSpacing(6)
        self._radio_manual = QRadioButton("Distance")
        self._radio_vertex = QRadioButton("Up to Vertex")
        self._radio_manual.setChecked(True)
        self._target_group = QButtonGroup(self)
        self._target_group.addButton(self._radio_manual, 0)
        self._target_group.addButton(self._radio_vertex, 1)
        self._target_group.idClicked.connect(self._on_target_mode_changed)
        target_row.addWidget(self._radio_manual)
        target_row.addWidget(self._radio_vertex)
        root.addLayout(target_row)

        # Distance input (manual mode)
        self._dist_widget = QWidget()
        dist_layout = QVBoxLayout(self._dist_widget)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(2)
        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(10.0)
        self._spinbox.value_changed.connect(self._emit_preview)
        dist_layout.addWidget(self._spinbox)
        root.addWidget(self._dist_widget)

        # Vertex target input (vertex mode)
        self._vertex_widget = QWidget()
        vtx_layout = QVBoxLayout(self._vertex_widget)
        vtx_layout.setContentsMargins(0, 0, 0, 0)
        vtx_layout.setSpacing(4)

        vtx_row = QHBoxLayout()
        vtx_row.setSpacing(6)
        self._vtx_label = QLabel("No vertex selected")
        self._vtx_label.setStyleSheet(
            "color: #888; font-size: 11px; font-family: monospace;")
        self._vtx_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        vtx_row.addWidget(self._vtx_label)

        self._pick_vtx_btn = QPushButton("Pick")
        self._pick_vtx_btn.setObjectName("pick_vertex")
        self._pick_vtx_btn.setToolTip("Click a vertex on the model")
        self._pick_vtx_btn.setCheckable(True)
        self._pick_vtx_btn.clicked.connect(self._on_pick_vertex_toggle)
        vtx_row.addWidget(self._pick_vtx_btn)

        reset_vtx_btn = QPushButton("✕")
        reset_vtx_btn.setFixedWidth(24)
        reset_vtx_btn.setToolTip("Clear vertex target")
        reset_vtx_btn.clicked.connect(self._reset_vertex)
        vtx_row.addWidget(reset_vtx_btn)
        vtx_layout.addLayout(vtx_row)

        self._vtx_dist_label = QLabel("—")
        self._vtx_dist_label.setStyleSheet(
            "color: #666; font-size: 11px; font-family: monospace;")
        vtx_layout.addWidget(self._vtx_dist_label)

        root.addWidget(self._vertex_widget)
        self._vertex_widget.hide()

        # ── Offsets ───────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Offsets"))

        offset_grid = QHBoxLayout()
        offset_grid.setSpacing(8)

        start_col = QVBoxLayout()
        start_col.setSpacing(2)
        start_col.addWidget(QLabel("Start"))
        self._start_offset = ExprSpinBox(unit=prefs.default_unit)
        self._start_offset.set_mm(0.0)
        self._start_offset.value_changed.connect(self._emit_preview)
        start_col.addWidget(self._start_offset)
        offset_grid.addLayout(start_col)

        end_col = QVBoxLayout()
        end_col.setSpacing(2)
        end_col.addWidget(QLabel("End"))
        self._end_offset = ExprSpinBox(unit=prefs.default_unit)
        self._end_offset.set_mm(0.0)
        self._end_offset.value_changed.connect(self._emit_preview)
        end_col.addWidget(self._end_offset)
        offset_grid.addLayout(end_col)

        root.addLayout(offset_grid)

        # ── Direction ─────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Direction"))

        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)

        self._dir_label = QLabel("Normal")
        self._dir_label.setStyleSheet(
            "color: #888; font-size: 11px; font-family: monospace;")
        self._dir_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        dir_row.addWidget(self._dir_label)

        self._flip_btn = QPushButton("⇅")
        self._flip_btn.setObjectName("pick_edge")  # reuse style
        self._flip_btn.setToolTip("Flip direction")
        self._flip_btn.setCheckable(True)
        self._flip_btn.clicked.connect(self._on_flip)
        dir_row.addWidget(self._flip_btn)

        self._pick_btn = QPushButton("Pick Edge")
        self._pick_btn.setObjectName("pick_edge")
        self._pick_btn.setToolTip("Click an edge on the model to use its direction")
        self._pick_btn.setCheckable(True)
        self._pick_btn.clicked.connect(self._on_pick_edge_toggle)
        dir_row.addWidget(self._pick_btn)

        reset_dir_btn = QPushButton("✕")
        reset_dir_btn.setFixedWidth(24)
        reset_dir_btn.setToolTip("Reset to face normal")
        reset_dir_btn.clicked.connect(self._reset_direction)
        dir_row.addWidget(reset_dir_btn)

        root.addLayout(dir_row)

        # ── Operation ─────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Operation"))

        self._radio_new   = QRadioButton("New Body")
        self._radio_merge = QRadioButton("Merge into")
        self._radio_new.setChecked(True)
        self._op_group = QButtonGroup(self)
        self._op_group.addButton(self._radio_new,   0)
        self._op_group.addButton(self._radio_merge, 1)
        self._op_group.idClicked.connect(self._on_op_changed)
        root.addWidget(self._radio_new)

        merge_row = QHBoxLayout()
        merge_row.setSpacing(6)
        merge_row.addWidget(self._radio_merge)
        self._body_combo = QComboBox()
        self._body_combo.setEnabled(False)
        self._body_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        merge_row.addWidget(self._body_combo)
        root.addLayout(merge_row)

        root.addWidget(self._separator())

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("ok")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
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

    def set_face_origin(self, origin: np.ndarray):
        self._face_origin = origin.copy()

    def set_face_normal(self, normal: np.ndarray):
        self._face_normal = normal / np.linalg.norm(normal)
        self._recompute_vertex_dist()
        self._emit_preview()

    def set_direction(self, vec: np.ndarray):
        """Called by viewport when the user picks an edge direction."""
        self._direction = vec / np.linalg.norm(vec)
        x, y, z = self._direction
        self._dir_label.setText(f"{x:+.2f}, {y:+.2f}, {z:+.2f}")
        self._dir_label.setStyleSheet(
            "color: #8dcc5a; font-size: 11px; font-family: monospace;")
        self._end_pick_edge()
        self._recompute_vertex_dist()
        self._emit_preview()

    def set_vertex_target(self, vertex: np.ndarray):
        """Called by viewport when the user picks a vertex."""
        self._target_vertex = vertex.copy()
        x, y, z = vertex
        self._vtx_label.setText(f"{x:.2f}, {y:.2f}, {z:.2f}")
        self._vtx_label.setStyleSheet(
            "color: #cc8dff; font-size: 11px; font-family: monospace;")
        self._end_pick_vertex()
        self._recompute_vertex_dist()
        self._emit_preview()

    def refresh_bodies(self, exclude_body_id: str | None = None):
        self._exclude_body = exclude_body_id
        self._refresh_bodies()

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _recompute_vertex_dist(self):
        if self._target_vertex is None or self._face_origin is None:
            self._vertex_dist_mm = None
            self._vtx_dist_label.setText("—")
            return
        direction = self._direction if self._direction is not None else self._face_normal
        if direction is None:
            self._vertex_dist_mm = None
            self._vtx_dist_label.setText("no direction")
            return
        # Flip applied to direction for projection so the readout matches
        if getattr(self, '_flip', False):
            direction = -direction
        dist = abs(float(np.dot(self._target_vertex - self._face_origin, direction)))
        self._vertex_dist_mm = dist
        from cad.units import format_value
        from cad.prefs import prefs
        self._vtx_dist_label.setText(f"→ {format_value(dist, prefs.default_unit)}")

    def _dist_and_dir(self):
        """
        Return (distance_mm, direction_or_None).
        distance_mm is always positive — mode toggle (extrude/cut) determines sign.
        """
        start_off = self._start_offset.mm_value() or 0.0
        end_off   = self._end_offset.mm_value()   or 0.0

        if self._target_group.checkedId() == 1:
            if self._vertex_dist_mm is None:
                return None, None
            dist = self._vertex_dist_mm + end_off - start_off
        else:
            dist = self._spinbox.mm_value()
            if dist is None:
                return None, None
            dist = abs(dist) + end_off - start_off  # always positive from spinbox

        # Apply direction flip
        flip = getattr(self, '_flip', False)
        direction = None
        if self._direction is not None:
            direction = -self._direction if flip else self._direction
        elif flip:
            # No custom direction — flip means go the other way, negate dist
            dist = -dist

        return dist, direction

    def _signed_dist_and_dir(self):
        """
        Return (signed_distance_mm, direction_or_None) with cut mode applied.
        Positive = extrude (add), negative = cut (remove).
        """
        dist, direction = self._dist_and_dir()
        if dist is None:
            return None, None
        is_cut = self._mode_group.checkedId() == 1
        return (-abs(dist) if is_cut else abs(dist)), direction

    def _emit_preview(self, _=None):
        dist, direction = self._signed_dist_and_dir()
        if dist is None:
            return
        self.preview_changed.emit(dist, direction)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_target_mode_changed(self, btn_id: int):
        if btn_id == 0:
            self._dist_widget.show()
            self._vertex_widget.hide()
            self._end_pick_vertex()
        else:
            self._dist_widget.hide()
            self._vertex_widget.show()
        self._emit_preview()

    def _on_flip(self, checked: bool):
        self._flip = checked
        self._flip_btn.setText("⇅" if not checked else "⇵")
        self._recompute_vertex_dist()
        self._emit_preview()

    def _on_pick_edge_toggle(self, checked: bool):
        if checked:
            self._picking_edge = True
            self._pick_btn.setProperty("active", True)
            self._pick_btn.style().unpolish(self._pick_btn)
            self._pick_btn.style().polish(self._pick_btn)
            self.picking_edge_changed.emit(True)
        else:
            self._end_pick_edge()

    def _end_pick_edge(self):
        self._picking_edge = False
        self._pick_btn.setChecked(False)
        self._pick_btn.setProperty("active", False)
        self._pick_btn.style().unpolish(self._pick_btn)
        self._pick_btn.style().polish(self._pick_btn)
        self.picking_edge_changed.emit(False)

    def _on_pick_vertex_toggle(self, checked: bool):
        if checked:
            self._picking_vertex = True
            self._pick_vtx_btn.setProperty("active", True)
            self._pick_vtx_btn.style().unpolish(self._pick_vtx_btn)
            self._pick_vtx_btn.style().polish(self._pick_vtx_btn)
            self.picking_vertex_changed.emit(True)
        else:
            self._end_pick_vertex()

    def _end_pick_vertex(self):
        self._picking_vertex = False
        self._pick_vtx_btn.setChecked(False)
        self._pick_vtx_btn.setProperty("active", False)
        self._pick_vtx_btn.style().unpolish(self._pick_vtx_btn)
        self._pick_vtx_btn.style().polish(self._pick_vtx_btn)
        self.picking_vertex_changed.emit(False)

    def _reset_vertex(self):
        self._target_vertex  = None
        self._vertex_dist_mm = None
        self._vtx_label.setText("No vertex selected")
        self._vtx_label.setStyleSheet(
            "color: #888; font-size: 11px; font-family: monospace;")
        self._vtx_dist_label.setText("—")
        self._end_pick_vertex()
        self._emit_preview()

    def _reset_direction(self):
        self._direction = None
        self._dir_label.setText("Normal")
        self._dir_label.setStyleSheet(
            "color: #888; font-size: 11px; font-family: monospace;")
        self._end_pick_edge()
        self._recompute_vertex_dist()
        self._emit_preview()

    def _on_op_changed(self, btn_id: int):
        self._body_combo.setEnabled(btn_id == 1)

    def _refresh_bodies(self):
        self._body_combo.clear()
        exclude = getattr(self, "_exclude_body", None)
        for body in self._workspace.bodies.values():
            if body.id != exclude:
                self._body_combo.addItem(body.name, userData=body.id)

    def _on_ok(self):
        self._spinbox._on_commit()
        dist, direction = self._signed_dist_and_dir()
        if dist is None:
            return

        op_id = self._op_group.checkedId()
        if op_id == 1:
            merge_body_id = self._body_combo.currentData()
        else:
            merge_body_id = "__new_body__"

        self.extrude_requested.emit(dist, direction, merge_body_id)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_vertex:
                self._end_pick_vertex()
            elif self._picking_edge:
                self._end_pick_edge()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
