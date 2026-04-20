"""
cad/sketch_tools/dimension.py

DimensionTool — click a line to apply a length constraint.

Workflow:
  1. Shift+D activates the tool.
  2. Click a LineEntity — a dialog appears with the current length.
  3. User types a value; constraint is stored and solver runs immediately.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class DimensionTool(BaseTool):

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None
        self._hover_idx: int | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    @property
    def hover_entity_idx(self) -> int | None:
        return self._hover_idx

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pt = snap_result.point
        if pt is None:
            self._cursor_2d = None
            self._hover_idx = None
            return
        self._cursor_2d = pt.copy()
        self._hover_idx = _nearest_line(pt, sketch.entities)

    def handle_click(self, snap_result, sketch) -> bool:
        pt = snap_result.point
        if pt is None:
            return False
        idx = _nearest_line(pt, sketch.entities)
        if idx is None:
            return False
        if hasattr(sketch, '_dimension_callback') and sketch._dimension_callback:
            from cad.sketch import LineEntity
            ent = sketch.entities[idx]
            current_len = float(np.linalg.norm(ent.p1 - ent.p0))
            sketch._dimension_callback(idx, current_len)
        return True

    def cancel(self) -> None:
        self._hover_idx = None
        self._cursor_2d = None


def _nearest_line(pt: np.ndarray, entities, tol: float = 5.0) -> int | None:
    """Return index of the closest LineEntity to pt within tol mm, or None."""
    from cad.sketch import LineEntity
    best_idx  = None
    best_dist = tol
    for i, ent in enumerate(entities):
        if not isinstance(ent, LineEntity):
            continue
        d = _pt_to_seg_dist(pt, ent.p0, ent.p1)
        if d < best_dist:
            best_dist = d
            best_idx  = i
    return best_idx


def _pt_to_seg_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab  = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab)) / ab2
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(pt - (a + t * ab)))
