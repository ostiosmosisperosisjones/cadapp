"""
cad/sketch_tools/line.py

LineTool — single-shot line segment (click start, click end).

For a continuous polyline chain, see continuous_line.py (future tool).
Each click pair produces one LineEntity.  The pending start point is
held until the second click; ESC clears it without leaving sketch mode.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class LineTool(BaseTool):

    def __init__(self):
        self._start:      np.ndarray | None = None
        self._cursor_2d:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = snap_result.point.copy() \
            if snap_result.point is not None else None

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import LineEntity
        pt = snap_result.point
        if pt is None:
            return False
        if self._start is None:
            self._start = pt.copy()
        else:
            sketch.push_undo_snapshot()
            sketch.entities.append(LineEntity(self._start, pt))
            self._start = pt.copy()   # chain: end of this becomes start of next
        return True

    def cancel(self) -> None:
        """ESC within line tool — drop the pending start point."""
        self._start     = None
        self._cursor_2d = None

    # ------------------------------------------------------------------
    # Read by sketch_overlay for the preview line
    # ------------------------------------------------------------------

    @property
    def line_start(self) -> np.ndarray | None:
        return self._start
