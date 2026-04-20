"""
gui/toolbar.py

Top toolbar with SVG-drawn operation buttons.
"""

from __future__ import annotations
from PyQt6.QtWidgets import (QToolBar, QToolButton, QFrame, QWidget,
                              QMenu, QWidgetAction, QButtonGroup, QHBoxLayout,
                              QLabel)
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QByteArray


_ICON_SIZE = 36
_BTN_SIZE  = 58   # fixed button width

_STYLE = """
QToolBar {
    background: #1e1e1e;
    border-bottom: 1px solid #2a2a2a;
    spacing: 2px;
    padding: 4px 6px;
}
QToolButton {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 4px 2px 2px 2px;
    color: #ccc;
    font-size: 10px;
    min-width: 52px;
}
QToolButton:hover {
    background: #383838;
    border-color: #555;
    color: #eee;
}
QToolButton:pressed {
    background: #1a1a1a;
}
QToolButton:disabled {
    background: #222;
    color: #3a3a3a;
    border-color: #2a2a2a;
}
"""

# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _svg_icon(svg: str) -> QIcon:
    data = QByteArray(svg.encode())
    renderer = QSvgRenderer(data)
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


def _stub_icon(letter: str, color: str = "#666") -> QIcon:
    """Minimal placeholder: a faint rounded square with a letter."""
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <rect x="4" y="4" width="28" height="28" rx="5"
        fill="none" stroke="{color}" stroke-width="1.2" stroke-dasharray="3,2" opacity="0.5"/>
  <text x="18" y="23" text-anchor="middle" font-family="monospace"
        font-size="13" fill="{color}" opacity="0.6">{letter}</text>
</svg>"""
    return _svg_icon(svg)


# ---------------------------------------------------------------------------
# SVG icon definitions  (implemented operations)
# ---------------------------------------------------------------------------

# Extrude: face rectangle + upward arrow
_SVG_EXTRUDE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <rect x="6" y="22" width="20" height="8" rx="1"
        fill="#4fc3f7" fill-opacity="0.25" stroke="#4fc3f7" stroke-width="1.2"/>
  <rect x="11" y="12" width="10" height="10"
        fill="#4fc3f7" fill-opacity="0.15" stroke="#4fc3f7" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="16" y1="20" x2="16" y2="6" stroke="#4fc3f7" stroke-width="2" stroke-linecap="round"/>
  <polyline points="11,11 16,5 21,11" fill="none" stroke="#4fc3f7" stroke-width="2"
            stroke-linejoin="round" stroke-linecap="round"/>
</svg>
"""

# Thicken: flat surface + three offset arrows
_SVG_THICKEN = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <rect x="5" y="15" width="26" height="5" rx="1"
        fill="#81c784" fill-opacity="0.2" stroke="#81c784" stroke-width="1.4"/>
  <rect x="5" y="8" width="26" height="4" rx="1"
        fill="#81c784" fill-opacity="0.1" stroke="#81c784" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="10" y1="14" x2="10" y2="9"  stroke="#81c784" stroke-width="1.6" stroke-linecap="round"/>
  <polyline points="7,12 10,8 13,12" fill="none" stroke="#81c784" stroke-width="1.6"
            stroke-linejoin="round" stroke-linecap="round"/>
  <line x1="18" y1="14" x2="18" y2="9"  stroke="#81c784" stroke-width="1.6" stroke-linecap="round"/>
  <polyline points="15,12 18,8 21,12" fill="none" stroke="#81c784" stroke-width="1.6"
            stroke-linejoin="round" stroke-linecap="round"/>
  <line x1="26" y1="14" x2="26" y2="9"  stroke="#81c784" stroke-width="1.6" stroke-linecap="round"/>
  <polyline points="23,12 26,8 29,12" fill="none" stroke="#81c784" stroke-width="1.6"
            stroke-linejoin="round" stroke-linecap="round"/>
</svg>
"""

# Sketch: pencil on a plane grid
_SVG_SKETCH = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <!-- grid plane -->
  <line x1="4" y1="28" x2="32" y2="28" stroke="#ffb74d" stroke-width="1" opacity="0.4"/>
  <line x1="4" y1="23" x2="32" y2="23" stroke="#ffb74d" stroke-width="0.7" opacity="0.25"/>
  <line x1="10" y1="28" x2="10" y2="18" stroke="#ffb74d" stroke-width="0.7" opacity="0.25"/>
  <line x1="18" y1="28" x2="18" y2="18" stroke="#ffb74d" stroke-width="0.7" opacity="0.25"/>
  <line x1="26" y1="28" x2="26" y2="18" stroke="#ffb74d" stroke-width="0.7" opacity="0.25"/>
  <!-- pencil body -->
  <rect x="19" y="7" width="5" height="14" rx="1" transform="rotate(-35 21.5 14)"
        fill="#ffb74d" fill-opacity="0.3" stroke="#ffb74d" stroke-width="1.2"/>
  <!-- pencil tip -->
  <polygon points="14,26 17,20 20,23" fill="#ffb74d" opacity="0.7"/>
  <!-- pencil eraser end -->
  <rect x="21" y="6" width="5" height="3" rx="1" transform="rotate(-35 23.5 7.5)"
        fill="#ef9a9a" opacity="0.6"/>
</svg>
"""

# ---------------------------------------------------------------------------
# Stub SVG icons  (operations not yet implemented — greyed dashed outlines)
# ---------------------------------------------------------------------------

# Revolve: profile arc rotating around an axis
_SVG_REVOLVE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <line x1="18" y1="4" x2="18" y2="32" stroke="#888" stroke-width="1.2"
        stroke-dasharray="2,2" opacity="0.5"/>
  <path d="M18,8 Q30,18 18,28" fill="none" stroke="#888" stroke-width="1.4"
        stroke-dasharray="3,2" opacity="0.55"/>
  <path d="M18,8 Q6,18 18,28" fill="#888" fill-opacity="0.08" stroke="#888"
        stroke-width="1.4" stroke-dasharray="3,2" opacity="0.55"/>
  <polyline points="13,12 18,8 23,12" fill="none" stroke="#888" stroke-width="1.4"
            stroke-linejoin="round" stroke-linecap="round" opacity="0.5"/>
</svg>
"""

# Loft: two profiles connected by smooth ruled surface
_SVG_LOFT = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <ellipse cx="18" cy="27" rx="10" ry="3.5" fill="#888" fill-opacity="0.1"
           stroke="#888" stroke-width="1.3" stroke-dasharray="3,2" opacity="0.55"/>
  <ellipse cx="18" cy="11" rx="5" ry="2" fill="#888" fill-opacity="0.1"
           stroke="#888" stroke-width="1.3" stroke-dasharray="3,2" opacity="0.55"/>
  <line x1="8" y1="27" x2="13" y2="11" stroke="#888" stroke-width="1" opacity="0.4"/>
  <line x1="28" y1="27" x2="23" y2="11" stroke="#888" stroke-width="1" opacity="0.4"/>
</svg>
"""

# Draft: a box face with an angled taper arrow
_SVG_DRAFT = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <polygon points="7,28 29,28 24,10 12,10" fill="#888" fill-opacity="0.08"
           stroke="#888" stroke-width="1.3" stroke-dasharray="3,2" opacity="0.55"/>
  <line x1="18" y1="28" x2="18" y2="10" stroke="#888" stroke-width="1"
        stroke-dasharray="2,2" opacity="0.4"/>
  <line x1="18" y1="10" x2="24" y2="28" stroke="#888" stroke-width="1" opacity="0.3"/>
  <text x="18" y="34" text-anchor="middle" font-family="monospace"
        font-size="7" fill="#888" opacity="0.5">α</text>
</svg>
"""

# Fillet: a sharp corner becoming rounded
_SVG_FILLET = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <!-- sharp corner (grey, faint) -->
  <polyline points="7,29 7,7 29,7" fill="none" stroke="#888" stroke-width="1"
            opacity="0.3" stroke-dasharray="2,2"/>
  <!-- rounded fillet result -->
  <path d="M7,29 L7,16 Q7,7 16,7 L29,7" fill="none" stroke="#888" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" opacity="0.55"/>
  <!-- radius indicator arc -->
  <path d="M7,16 Q7,7 16,7" fill="none" stroke="#aaa" stroke-width="1"
        stroke-dasharray="2,2" opacity="0.5"/>
</svg>
"""

# Chamfer: a sharp corner with a diagonal bevel
_SVG_CHAMFER = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <!-- original corner (faint) -->
  <polyline points="7,29 7,7 29,7" fill="none" stroke="#888" stroke-width="1"
            opacity="0.3" stroke-dasharray="2,2"/>
  <!-- chamfered result -->
  <path d="M7,29 L7,15 L15,7 L29,7" fill="none" stroke="#888" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" opacity="0.55"/>
  <!-- bevel line -->
  <line x1="7" y1="15" x2="15" y2="7" stroke="#aaa" stroke-width="1.5"
        opacity="0.6"/>
</svg>
"""

# Boolean: two overlapping circles (union/subtract/intersect)
_SVG_BOOLEAN = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <circle cx="14" cy="18" r="9" fill="#888" fill-opacity="0.1"
          stroke="#888" stroke-width="1.3" stroke-dasharray="3,2" opacity="0.55"/>
  <circle cx="22" cy="18" r="9" fill="#888" fill-opacity="0.1"
          stroke="#888" stroke-width="1.3" stroke-dasharray="3,2" opacity="0.55"/>
  <!-- intersection highlight -->
  <path d="M18,9.3 Q26,14 26,18 Q26,22 18,26.7 Q10,22 10,18 Q10,14 18,9.3 Z"
        fill="#888" fill-opacity="0.18"/>
</svg>
"""


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------

class OpsToolbar(QToolBar):
    extrude_requested = pyqtSignal()
    thicken_requested = pyqtSignal()
    sketch_requested  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Operations", parent)
        self.setMovable(False)
        self.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self.setStyleSheet(_STYLE)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        # --- sketch (wired) ---
        self._btn_sketch = self._add_op_button(
            "Sketch", _SVG_SKETCH, self.sketch_requested,
            "Start a sketch on the selected face")

        self._add_separator()

        # --- implemented ops ---
        self._btn_extrude = self._add_op_button(
            "Extrude", _SVG_EXTRUDE, self.extrude_requested,
            "Extrude or cut a selected face / sketch profile")
        self._btn_thicken = self._add_op_button(
            "Thicken", _SVG_THICKEN, self.thicken_requested,
            "Offset selected face(s) outward to add material")

        self._add_separator()

        # --- stubs ---
        self._btn_revolve  = self._add_stub("Revolve",  _SVG_REVOLVE,  "Revolve a profile around an axis  [not yet implemented]")
        self._btn_loft     = self._add_stub("Loft",     _SVG_LOFT,     "Loft between two or more profiles  [not yet implemented]")
        self._btn_draft    = self._add_stub("Draft",    _SVG_DRAFT,    "Apply a draft angle to faces  [not yet implemented]")

        self._add_separator()

        self._btn_fillet   = self._add_stub("Fillet",   _SVG_FILLET,   "Round selected edges  [not yet implemented]")
        self._btn_chamfer  = self._add_stub("Chamfer",  _SVG_CHAMFER,  "Bevel selected edges  [not yet implemented]")

        self._add_separator()

        self._btn_boolean  = self._add_stub("Boolean",  _SVG_BOOLEAN,  "Boolean union / subtract / intersect  [not yet implemented]")

        self.set_enabled(False)

    def _add_op_button(self, label: str, svg: str, signal, tooltip: str) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(label)
        btn.setIcon(_svg_icon(svg))
        btn.setToolTip(tooltip)
        btn.clicked.connect(signal.emit)
        self.addWidget(btn)
        return btn

    def _add_stub(self, label: str, svg: str, tooltip: str) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(label)
        btn.setIcon(_svg_icon(svg))
        btn.setToolTip(tooltip)
        btn.setEnabled(False)
        self.addWidget(btn)
        return btn

    def _add_separator(self):
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet("background: #2a2a2a; margin: 6px 4px;")
        self.addWidget(sep)

    def set_enabled(self, enabled: bool):
        self._btn_sketch.setEnabled(enabled)
        self._btn_extrude.setEnabled(enabled)
        self._btn_thicken.setEnabled(enabled)
        # stubs stay disabled always


# ---------------------------------------------------------------------------
# Sketch tool SVG icons
# ---------------------------------------------------------------------------

_SVG_LINE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <line x1="6" y1="30" x2="30" y2="6" stroke="#ffb74d" stroke-width="2.2"
        stroke-linecap="round"/>
  <circle cx="6"  cy="30" r="2.5" fill="#ffb74d" opacity="0.8"/>
  <circle cx="30" cy="6"  r="2.5" fill="#ffb74d" opacity="0.8"/>
</svg>"""

_SVG_ARC = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <path d="M6,30 Q8,8 30,6" fill="none" stroke="#ffb74d" stroke-width="2.2"
        stroke-linecap="round"/>
  <circle cx="6"  cy="30" r="2.5" fill="#ffb74d" opacity="0.8"/>
  <circle cx="30" cy="6"  r="2.5" fill="#ffb74d" opacity="0.8"/>
  <circle cx="17" cy="11" r="2"   fill="#ffb74d" opacity="0.55"/>
</svg>"""

_SVG_CIRCLE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <circle cx="18" cy="18" r="12" fill="none" stroke="#ffb74d" stroke-width="2.2"/>
  <circle cx="18" cy="18" r="2"  fill="#ffb74d" opacity="0.7"/>
  <line x1="18" y1="18" x2="30" y2="18" stroke="#ffb74d" stroke-width="1.2"
        stroke-dasharray="2,2" opacity="0.6"/>
</svg>"""

_SVG_TRIM = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <line x1="6" y1="18" x2="30" y2="18" stroke="#ef9a9a" stroke-width="2"
        stroke-linecap="round" stroke-dasharray="4,3"/>
  <line x1="6" y1="10" x2="30" y2="28" stroke="#ffb74d" stroke-width="2"
        stroke-linecap="round"/>
  <circle cx="18" cy="18" r="3.5" fill="none" stroke="#ef9a9a" stroke-width="1.5"/>
</svg>"""

_SVG_OFFSET = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <rect x="8" y="10" width="16" height="16" rx="2"
        fill="none" stroke="#ffb74d" stroke-width="1.8"/>
  <rect x="4" y="6"  width="24" height="24" rx="4"
        fill="none" stroke="#ffb74d" stroke-width="1.2" stroke-dasharray="3,2"
        opacity="0.55"/>
  <line x1="24" y1="10" x2="28" y2="6"  stroke="#ffb74d" stroke-width="1"
        opacity="0.5"/>
  <line x1="24" y1="26" x2="28" y2="30" stroke="#ffb74d" stroke-width="1"
        opacity="0.5"/>
</svg>"""

_SVG_DIVIDE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <line x1="6" y1="18" x2="30" y2="18" stroke="#ffb74d" stroke-width="2"
        stroke-linecap="round"/>
  <circle cx="18" cy="18" r="2.5" fill="#ffb74d"/>
  <line x1="12" y1="8" x2="12" y2="28" stroke="#ffb74d" stroke-width="1.4"
        stroke-linecap="round" opacity="0.6" stroke-dasharray="2,2"/>
  <line x1="24" y1="8" x2="24" y2="28" stroke="#ffb74d" stroke-width="1.4"
        stroke-linecap="round" opacity="0.6" stroke-dasharray="2,2"/>
</svg>"""

_SVG_INCLUDE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <path d="M8,28 L18,8 L28,28" fill="none" stroke="#81c784" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="3,2"
        opacity="0.7"/>
  <path d="M8,28 L18,8 L28,28" fill="none" stroke="#81c784" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="0"
        opacity="0.4"/>
  <polyline points="12,24 18,10 24,24" fill="none" stroke="#81c784"
            stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

_SVG_COMMIT = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <polyline points="6,18 14,27 30,9" fill="none" stroke="#66bb6a"
            stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

_SKETCH_TOOLBAR_STYLE = """
QToolBar {
    background: #1a2a1a;
    border-bottom: 1px solid #2a3a2a;
    spacing: 2px;
    padding: 4px 6px;
}
QToolButton {
    background: #223322;
    border: 1px solid #3a4a3a;
    border-radius: 6px;
    padding: 4px 2px 2px 2px;
    color: #ccc;
    font-size: 10px;
    min-width: 52px;
}
QToolButton:hover {
    background: #2a4a2a;
    border-color: #556655;
    color: #eee;
}
QToolButton:pressed, QToolButton:checked {
    background: #1a3a1a;
    border-color: #66bb6a;
    color: #66bb6a;
}
QToolButton#commit {
    background: #1a3a1a;
    border-color: #4a7a4a;
    color: #66bb6a;
    font-weight: bold;
}
QToolButton#commit:hover {
    background: #224422;
    border-color: #66bb6a;
}
QMenu {
    background: #1e2e1e;
    border: 1px solid #3a4a3a;
    color: #ccc;
    font-size: 11px;
}
QMenu::item:selected { background: #2a4a2a; }
QMenu::item:checked  { color: #66bb6a; }
"""


class SketchToolbar(QToolBar):
    """
    Toolbar shown while in sketch mode.  Replaces OpsToolbar.
    Signals map to viewport sketch actions.
    """
    tool_line_requested    = pyqtSignal()
    tool_arc_requested     = pyqtSignal()
    tool_circle_requested  = pyqtSignal(str)   # emits CircleMode value
    tool_trim_requested    = pyqtSignal()
    tool_divide_requested  = pyqtSignal()
    tool_fillet_requested  = pyqtSignal()
    tool_offset_requested  = pyqtSignal()
    tool_include_requested = pyqtSignal()
    commit_requested       = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Sketch Tools", parent)
        self.setMovable(False)
        self.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self.setStyleSheet(_SKETCH_TOOLBAR_STYLE)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self._circle_mode = "center_radius"
        self._build_ui()

    def _build_ui(self):
        from cad.sketch_tools.circle import CircleMode

        self._btn_line = self._add_btn(
            "Line  L", _SVG_LINE, self.tool_line_requested, "Line tool  (L)")
        self._btn_arc  = self._add_btn(
            "Arc  A",  _SVG_ARC,  self.tool_arc_requested,  "3-point arc  (A)")

        # Circle button with drop-down corner menu for sub-mode
        self._btn_circle = self._make_circle_btn()
        self.addWidget(self._btn_circle)

        self._add_separator()

        self._btn_trim   = self._add_btn(
            "Trim  T",   _SVG_TRIM,   self.tool_trim_requested,   "Trim  (T)")
        self._btn_divide = self._add_btn(
            "Divide  D", _SVG_DIVIDE, self.tool_divide_requested, "Divide  (D)")
        self._btn_fillet = self._add_btn(
            "Fillet  F", _SVG_FILLET, self.tool_fillet_requested, "Fillet  (F)")
        self._btn_offset = self._add_btn(
            "Offset  O", _SVG_OFFSET, self.tool_offset_requested, "Offset  (O)")

        self._add_separator()

        self._btn_include = self._add_btn(
            "Include  I", _SVG_INCLUDE, self.tool_include_requested,
            "Include geometry  (I)")

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(
            spacer.sizePolicy().horizontalPolicy(),
            spacer.sizePolicy().verticalPolicy())
        from PyQt6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        self._btn_commit = self._add_btn(
            "Commit  ↵", _SVG_COMMIT, self.commit_requested,
            "Commit sketch  (Return)")
        self._btn_commit.setObjectName("commit")

    def _add_btn(self, label, svg, signal, tooltip) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(label)
        btn.setIcon(_svg_icon(svg))
        btn.setToolTip(tooltip)
        btn.clicked.connect(signal.emit)
        self.addWidget(btn)
        return btn

    def _make_circle_btn(self) -> QToolButton:
        from cad.sketch_tools.circle import CircleMode
        btn = QToolButton(self)
        btn.setText("Circle  C")
        btn.setIcon(_svg_icon(_SVG_CIRCLE))
        btn.setToolTip("Circle — click arrow for sub-mode  (C)")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        menu = QMenu(btn)
        self._circle_actions = {}
        for mode in CircleMode.ALL:
            act = menu.addAction(CircleMode.LABELS[mode])
            act.setCheckable(True)
            act.setData(mode)
            act.triggered.connect(lambda checked, m=mode: self._set_circle_mode(m))
            self._circle_actions[mode] = act
        self._circle_actions[self._circle_mode].setChecked(True)

        btn.setMenu(menu)
        btn.clicked.connect(
            lambda: self.tool_circle_requested.emit(self._circle_mode))
        return btn

    def _set_circle_mode(self, mode: str):
        for m, act in self._circle_actions.items():
            act.setChecked(m == mode)
        self._circle_mode = mode
        self.tool_circle_requested.emit(mode)

    def _add_separator(self):
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet("background: #2a3a2a; margin: 6px 4px;")
        self.addWidget(sep)

    def set_active_tool(self, tool_name: str | None):
        """Highlight the button matching the active tool."""
        mapping = {
            "LINE":   self._btn_line,
            "ARC3":   self._btn_arc,
            "CIRCLE": self._btn_circle,
            "TRIM":   self._btn_trim,
            "DIVIDE": self._btn_divide,
            "FILLET": self._btn_fillet,
            "OFFSET": self._btn_offset,
            "NONE":   None,
        }
        for btn in [self._btn_line, self._btn_arc, self._btn_circle,
                    self._btn_trim, self._btn_divide, self._btn_fillet,
                    self._btn_offset, self._btn_include]:
            btn.setChecked(False)
        active = mapping.get(tool_name or "NONE")
        if active is not None:
            active.setChecked(True)
