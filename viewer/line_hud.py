"""
viewer/line_hud.py

LineHUD — floating input widget shown while the LineTool has a start point.

Fields show live length + angle. Typing overrides the live value.
Mouse movement updates both fields (overwriting any typed value).
Tab cycles between fields. Enter draws the line using whatever is in the fields.
Escape hides without drawing.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent


class _Field(QLineEdit):
    enter_pressed = pyqtSignal()
    tab_pressed   = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.enter_pressed.emit()
        elif e.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            self.tab_pressed.emit()
        else:
            super().keyPressEvent(e)


class LineHUD(QWidget):
    """
    Signals
    -------
    draw_requested  — Enter pressed; caller should place the point
    """

    draw_requested = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.setStyleSheet("""
            QWidget {
                background: rgba(18, 30, 55, 210);
                border-radius: 6px;
                border: 1px solid rgba(80, 130, 220, 160);
            }
            QLabel {
                color: rgba(140, 180, 255, 200);
                font-size: 10px;
                background: transparent;
                border: none;
            }
            QLineEdit {
                color: #e8f0ff;
                background: rgba(40, 60, 100, 180);
                border: 1px solid rgba(80, 130, 220, 120);
                border-radius: 3px;
                padding: 1px 4px;
                font-family: monospace;
                font-size: 10px;
                min-width: 72px;
                max-width: 72px;
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 160, 255, 220);
                background: rgba(50, 80, 140, 200);
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(5)

        self._len_field = _Field()
        self._len_field.setPlaceholderText("length")
        self._ang_field = _Field()
        self._ang_field.setPlaceholderText("angle°")

        layout.addWidget(QLabel("L"))
        layout.addWidget(self._len_field)
        layout.addWidget(QLabel("∠"))
        layout.addWidget(self._ang_field)

        self.adjustSize()
        self.hide()

        self._len_field.tab_pressed.connect(
            lambda: self._ang_field.setFocus() or self._ang_field.selectAll())
        self._ang_field.tab_pressed.connect(
            lambda: self._len_field.setFocus() or self._len_field.selectAll())
        self._len_field.enter_pressed.connect(self.draw_requested)
        self._ang_field.enter_pressed.connect(self.draw_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_live(self, length_mm: float, angle_deg: float,
                    unit: str, decimals: int):
        """Overwrite both fields with current live values."""
        from cad.units import from_mm
        self._len_field.setText(f"{from_mm(length_mm, unit):.{decimals}f}")
        self._ang_field.setText(f"{angle_deg:.2f}")

    def read_length_mm(self) -> float | None:
        """Return parsed length in mm from field, or None if invalid."""
        from cad.prefs import prefs
        from cad.units import to_mm
        try:
            return to_mm(float(self._len_field.text()), prefs.default_unit)
        except ValueError:
            return None

    def read_angle_deg(self) -> float | None:
        """Return parsed angle in degrees from field, or None if invalid."""
        try:
            return float(self._ang_field.text())
        except ValueError:
            return None

    def move_to_cursor(self, cx: int, cy: int, vw: int, vh: int):
        self.adjustSize()
        w, h = self.width(), self.height()
        off = 20
        x = cx + off if cx + off + w <= vw else cx - w - off
        y = cy + off if cy + off + h <= vh else cy - h - off
        self.move(max(0, x), max(0, y))
