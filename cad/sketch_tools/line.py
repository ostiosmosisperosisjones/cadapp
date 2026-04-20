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
from cad.sketch_tools.snap import SnapType


_HLINE_EXTENT = 100_000.0   # mm — large enough to behave as infinite


def _snap_meta(snap_result) -> tuple | None:
    """Extract (entity_idx, snap_type) for snaps that need a coincidence constraint.

    ENDPOINT snaps are handled by the solver's canonicalization (points within
    1mm tolerance share a solver variable), so we skip them here. FREE and GRID
    have no source entity. Everything else — MIDPOINT, CENTER, NEAREST, TANGENT —
    lands on a position that isn't another endpoint, so the solver needs an
    explicit coincidence constraint to preserve it.
    """
    if snap_result is None:
        return None
    skip = {SnapType.FREE, SnapType.GRID, SnapType.ENDPOINT, SnapType.INTERSECTION}
    if snap_result.type in skip or snap_result.entity_idx is None:
        return None
    return (snap_result.entity_idx, snap_result.type)


class LineTool(BaseTool):

    def __init__(self, constrain: str | None = None):
        self._start:      np.ndarray | None = None
        self._cursor_2d:  np.ndarray | None = None
        self._constrain:  str | None = constrain  # 'H', 'V', or None
        self._start_snap: object | None = None    # SnapResult from first click

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pt = snap_result.point
        if pt is None:
            self._cursor_2d = None
            return
        self._cursor_2d = pt.copy()
        # Keep snap engine informed of anchor so tangent snap is stable
        sketch.snap.anchor_pt = self._start

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import LineEntity
        pt = snap_result.point
        if pt is None:
            return False

        if self._constrain == 'H':
            sketch.push_undo_snapshot()
            sketch.entities.append(LineEntity(
                np.array([pt[0] - _HLINE_EXTENT, pt[1]]),
                np.array([pt[0] + _HLINE_EXTENT, pt[1]]),
            ))
            return True

        if self._constrain == 'V':
            sketch.push_undo_snapshot()
            sketch.entities.append(LineEntity(
                np.array([pt[0], pt[1] - _HLINE_EXTENT]),
                np.array([pt[0], pt[1] + _HLINE_EXTENT]),
            ))
            return True

        if self._start is None:
            self._start = pt.copy()
            self._start_snap = snap_result
        else:
            sketch.push_undo_snapshot()
            ent = LineEntity(self._start, pt)
            ent.p0_snap = _snap_meta(self._start_snap)
            ent.p1_snap = _snap_meta(snap_result)
            sketch.entities.append(ent)
            self._start = pt.copy()   # chain: end of this becomes start of next
            self._start_snap = snap_result
        return True

    def cancel(self) -> None:
        """ESC within line tool — drop the pending start point."""
        self._start      = None
        self._cursor_2d  = None
        self._start_snap = None

    # ------------------------------------------------------------------
    # Read by sketch_overlay for the preview line
    # ------------------------------------------------------------------

    @property
    def line_start(self) -> np.ndarray | None:
        return self._start
