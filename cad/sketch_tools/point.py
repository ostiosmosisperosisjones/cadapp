"""
cad/sketch_tools/point.py

PointTool — place a construction point with full snap support.
Press P, then any snap declaration (E/M/C/I/T/N), then click.
The point participates in all snap types as an endpoint.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class PointTool(BaseTool):

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.point.copy()
                           if snap_result.point is not None else None)

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import PointEntity
        pt = snap_result.point
        if pt is None:
            return False
        sketch.push_undo_snapshot()
        sketch.entities.append(PointEntity(pt))
        return True

    def cancel(self) -> None:
        self._cursor_2d = None
