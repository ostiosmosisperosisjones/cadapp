"""
cad/sketch_tools/geometric.py

GeometricConstraintTool — apply parallel, perpendicular, horizontal, or
vertical constraints.

Tab cycles through modes.  Horizontal/vertical are single-click; parallel
and perpendicular require two clicks (reference line, then target line).
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool
from cad.sketch_tools.dimension import _nearest_line


MODES = ['parallel', 'perpendicular', 'horizontal', 'vertical', 'equal']


def _writeback_entities(sketch, tmp):
    """Copy solved line and arc coordinates from tmp back to sketch."""
    from cad.sketch import LineEntity, ArcEntity
    for i, ent in enumerate(tmp.entities):
        if isinstance(ent, LineEntity):
            sketch.entities[i].p0 = ent.p0.copy()
            sketch.entities[i].p1 = ent.p1.copy()
        elif isinstance(ent, ArcEntity):
            sketch.entities[i].center      = ent.center.copy()
            sketch.entities[i].radius      = ent.radius
            sketch.entities[i].start_angle = ent.start_angle
            sketch.entities[i].end_angle   = ent.end_angle
MODE_LABELS = {
    'parallel':      'PARALLEL  (Tab→)',
    'perpendicular': 'PERP  (Tab→)',
    'horizontal':    'HORIZONTAL  (Tab→)',
    'vertical':      'VERTICAL  (Tab→)',
    'equal':         'EQUAL  (Tab→)',
}


class GeometricConstraintTool(BaseTool):

    def __init__(self, mode: str = 'parallel'):
        self._mode      = mode if mode in MODES else 'parallel'
        self._first_idx: int | None = None
        self._hover_idx: int | None = None
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def first_idx(self) -> int | None:
        return self._first_idx

    @property
    def hover_entity_idx(self) -> int | None:
        return self._hover_idx

    def cycle_mode(self):
        idx = MODES.index(self._mode)
        self._mode = MODES[(idx + 1) % len(MODES)]
        self._first_idx = None

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

        if self._mode in ('horizontal', 'vertical'):
            from cad.sketch import SketchConstraint, SketchEntry
            sketch.push_undo_snapshot()
            sketch.constraints.append(SketchConstraint(self._mode, (idx,), 0.0))
            tmp = SketchEntry.from_sketch_mode(sketch)
            if tmp.apply_all_constraints():
                _writeback_entities(sketch, tmp)
            else:
                sketch.constraints.pop()
                sketch._entity_snapshots.pop()
            return True

        if self._first_idx is None:
            self._first_idx = idx
            return True

        if idx == self._first_idx:
            return True

        ref_idx, mov_idx = self._first_idx, idx
        self._first_idx = None

        from cad.sketch import SketchConstraint, SketchEntry
        sketch.push_undo_snapshot()
        sketch.constraints.append(
            SketchConstraint(self._mode, (ref_idx, mov_idx), 0.0)
        )
        tmp = SketchEntry.from_sketch_mode(sketch)
        if tmp.apply_all_constraints():
            _writeback_entities(sketch, tmp)
        else:
            sketch.constraints.pop()
            sketch._entity_snapshots.pop()
        return True

    def cancel(self) -> None:
        self._first_idx = None
        self._hover_idx = None
        self._cursor_2d = None
