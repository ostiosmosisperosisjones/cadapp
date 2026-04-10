"""
cad/sketch_tools/line.py

LineTool — click to place connected line segments.

First click  : sets the start point.
Each subsequent click : completes one segment and starts the next.
ESC (cancel) : abandons the in-progress segment, start point cleared.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class LineTool(BaseTool):

    def __init__(self):
        self._start:     np.ndarray | None = None
        self._cursor_2d: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, pt2d: np.ndarray, sketch) -> None:
        self._cursor_2d = pt2d.copy() if pt2d is not None else None

    def handle_click(self, pt2d: np.ndarray, sketch) -> bool:
        from cad.sketch import LineEntity
        if pt2d is None:
            return False
        if self._start is None:
            self._start = pt2d.copy()
        else:
            sketch.entities.append(LineEntity(self._start, pt2d))
            self._start = pt2d.copy()
        return True

    def cancel(self) -> None:
        """ESC within line tool — drop the pending start point."""
        self._start     = None
        self._cursor_2d = None

    # ------------------------------------------------------------------
    # Line-tool-specific state (read by sketch_overlay for preview)
    # ------------------------------------------------------------------

    @property
    def line_start(self) -> np.ndarray | None:
        return self._start
