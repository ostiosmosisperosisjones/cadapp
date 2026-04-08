"""
gui/prefs_dialog.py

PrefsDialog — a simple color/settings editor for user preferences.

Each preference is shown as a labelled row.  Color values get a clickable
swatch that opens QColorDialog.  Bool values get a checkbox.  Float values
get a spinbox.  Changes are written to the prefs singleton immediately and
saved to disk on OK.
"""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QCheckBox, QDoubleSpinBox,
    QDialogButtonBox, QFrame, QSizePolicy
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from cad.prefs import prefs


# Human-readable labels for each pref key
_LABELS = {
    'background_color':        ('Viewport',  'Background color'),
    'body_color':              ('Bodies',    'Body color'),
    'body_color_active':       ('Bodies',    'Active body color'),
    'edge_color':              ('Edges',     'Edge color'),
    'edge_width':              ('Edges',     'Edge width'),
    'show_edges':              ('Edges',     'Show edges'),
    'face_selected_color':     ('Selection', 'Selected face color'),
    'edge_hovered_color':      ('Selection', 'Hovered edge color'),
    'edge_selected_color':     ('Selection', 'Selected edge color'),
    'vertex_hovered_color':    ('Selection', 'Hovered vertex color'),
    'vertex_selected_color':   ('Selection', 'Selected vertex color'),
    'sketch_line_color':       ('Sketch',    'Sketch line color'),
    'sketch_reference_color':  ('Sketch',    'Reference geometry color'),
    'sketch_preview_color':    ('Sketch',    'Preview line color'),
    'sketch_cursor_color':     ('Sketch',    'Cursor color'),
    'sketch_axis_x_color':     ('Sketch',    'X axis color'),
    'sketch_axis_y_color':     ('Sketch',    'Y axis color'),
    'sketch_grid_major_color': ('Sketch',    'Grid major color'),
    'sketch_grid_minor_color': ('Sketch',    'Grid minor color'),
}


def _color_to_qcolor(c) -> QColor:
    r, g, b = c
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def _qcolor_to_tuple(q: QColor) -> tuple:
    return (q.red() / 255, q.green() / 255, q.blue() / 255)


def _swatch_style(color) -> str:
    r, g, b = color
    hr = int(r * 255)
    hg = int(g * 255)
    hb = int(b * 255)
    return (f"background: rgb({hr},{hg},{hb}); "
            f"border: 1px solid #555; border-radius: 3px;")


class PrefsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(420)
        self.setMinimumHeight(500)
        self._widgets: dict[str, QWidget] = {}
        self._setup()

    def _setup(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        # Group rows by section
        current_section = None
        for key, (section, label) in _LABELS.items():
            if section != current_section:
                current_section = section
                sec_lbl = QLabel(section.upper())
                sec_lbl.setStyleSheet(
                    "color: #555; font-size: 10px; font-weight: bold; "
                    "letter-spacing: 1px; padding-top: 12px; padding-bottom: 4px;")
                layout.addWidget(sec_lbl)
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #333;")
                layout.addWidget(line)

            val = getattr(prefs, key)
            row = QHBoxLayout()
            row.setContentsMargins(0, 3, 0, 3)

            lbl = QLabel(label)
            lbl.setStyleSheet("color: #ccc; font-size: 12px;")
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Preferred)
            row.addWidget(lbl)

            if isinstance(val, tuple):             # Color
                btn = QPushButton()
                btn.setFixedSize(48, 22)
                btn.setStyleSheet(_swatch_style(val))
                btn.setProperty('pref_key', key)
                btn.clicked.connect(self._pick_color)
                self._widgets[key] = btn
                row.addWidget(btn)

            elif isinstance(val, bool):            # Bool
                cb = QCheckBox()
                cb.setChecked(val)
                cb.setProperty('pref_key', key)
                self._widgets[key] = cb
                row.addWidget(cb)

            elif isinstance(val, float):           # Float
                sp = QDoubleSpinBox()
                sp.setDecimals(2)
                sp.setMinimum(0.1)
                sp.setMaximum(10.0)
                sp.setSingleStep(0.1)
                sp.setValue(val)
                sp.setFixedWidth(80)
                sp.setProperty('pref_key', key)
                self._widgets[key] = sp
                row.addWidget(sp)

            row_w = QWidget()
            row_w.setLayout(row)
            layout.addWidget(row_w)

        layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

        # Reset + OK/Cancel
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(12, 8, 12, 8)
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)

        btn_w = QWidget()
        btn_w.setLayout(btn_layout)
        root.addWidget(btn_w)

    def _pick_color(self):
        btn = self.sender()
        key = btn.property('pref_key')
        current = getattr(prefs, key)
        qc = _color_to_qcolor(current)
        from PyQt6.QtWidgets import QColorDialog
        new_qc = QColorDialog.getColor(qc, self, f"Choose color — {key}")
        if new_qc.isValid():
            new_color = _qcolor_to_tuple(new_qc)
            btn.setStyleSheet(_swatch_style(new_color))
            btn.setProperty('current_color', new_color)

    def _apply_and_accept(self):
        for key, widget in self._widgets.items():
            if isinstance(widget, QPushButton):
                # Color swatch — read from property if changed
                c = widget.property('current_color')
                if c is not None:
                    setattr(prefs, key, tuple(c))
            elif isinstance(widget, QCheckBox):
                setattr(prefs, key, widget.isChecked())
            elif isinstance(widget, QDoubleSpinBox):
                setattr(prefs, key, widget.value())
        prefs.save()
        self.accept()

    def _reset(self):
        prefs.reset()
        # Rebuild dialog contents
        self.close()
        dlg = PrefsDialog(self.parent())
        dlg.exec()
