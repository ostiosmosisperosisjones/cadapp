"""
cad/sketch_tools/offset.py

OffsetTool — click an entity or closed loop, enter a distance, get a new
independent offset copy.

Workflow:
  1. User presses O → OffsetTool activates
  2. Hover highlights the nearest entity (or whole loop if loop detected)
  3. Click to select → panel opens for distance input
  4. Enter confirms → offset entities appended to sketch, tool resets

Offset rules:
  - LineEntity     → parallel line, same length, offset perpendicular
  - ArcEntity      → concentric arc, radius ± distance (side determined by
                     which side of the arc the click landed on)
  - Closed loop    → all entities offset together; side from click position
                     vs loop interior
"""

from __future__ import annotations
import math
import numpy as np
from cad.sketch_tools.base import BaseTool


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def _find_loop_for_entity(entity, entities: list, tol: float = 1e-3) -> list | None:
    """
    If entity is part of a closed chain, return the ordered list of entities
    in that chain.  Returns None if entity is not in a closed loop.
    """
    from cad.sketch import LineEntity, ArcEntity

    drawn = [e for e in entities if isinstance(e, (LineEntity, ArcEntity))]

    def ep(e):
        return e.p0, e.p1

    # Build adjacency by chaining endpoints
    remaining = list(drawn)
    while remaining:
        if entity not in remaining:
            break
        start = entity
        remaining.remove(start)
        chain = [start]
        changed = True
        while changed:
            changed = False
            tail = chain[-1].p1
            head = chain[0].p0
            for e in list(remaining):
                if np.linalg.norm(e.p0 - tail) < tol:
                    chain.append(e); remaining.remove(e); changed = True; break
                if np.linalg.norm(e.p1 - tail) < tol:
                    # reverse: swap p0/p1 representation via a wrapper
                    chain.append(_Reversed(e)); remaining.remove(e); changed = True; break
                if np.linalg.norm(e.p1 - head) < tol:
                    chain.insert(0, e); remaining.remove(e); changed = True; break
                if np.linalg.norm(e.p0 - head) < tol:
                    chain.insert(0, _Reversed(e)); remaining.remove(e); changed = True; break

        if np.linalg.norm(chain[-1].p1 - chain[0].p0) < tol:
            # Closed — return the original (non-reversed) entities
            return [e._e if isinstance(e, _Reversed) else e for e in chain]
        # Not closed — put back and break
        remaining.extend(chain[1:])
        break

    return None


class _Reversed:
    """Temporary wrapper that swaps p0/p1 for chaining purposes."""
    def __init__(self, e):
        self._e = e

    @property
    def p0(self):
        return self._e.p1

    @property
    def p1(self):
        return self._e.p0


# ---------------------------------------------------------------------------
# Signed-distance helpers (which side of an entity is the click on)
# ---------------------------------------------------------------------------

def _point_side_of_line(pt: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    """Positive = left of p0→p1, negative = right."""
    d = p1 - p0
    return float(d[0] * (pt[1] - p0[1]) - d[1] * (pt[0] - p0[0]))


def _point_side_of_arc(pt: np.ndarray, arc) -> float:
    """Positive = outside circle, negative = inside."""
    return float(np.linalg.norm(pt - arc.center)) - arc.radius


def _loop_contains(pt: np.ndarray, loop_entities: list, tol: float = 1e-3) -> bool:
    """Ray-casting test against a tessellated loop polygon."""
    from cad.sketch import LineEntity, ArcEntity
    pts = []
    for e in loop_entities:
        if isinstance(e, LineEntity):
            pts.append(e.p0)
        elif isinstance(e, ArcEntity):
            pts.extend(e.tessellate(32)[:-1])
    if not pts:
        return False
    x, y = float(pt[0]), float(pt[1])
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(pts[i][0]), float(pts[i][1])
        xj, yj = float(pts[j][0]), float(pts[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Offset geometry
# ---------------------------------------------------------------------------

def offset_line(line, dist: float):
    """Return a new LineEntity offset by dist (positive = left of direction)."""
    from cad.sketch import LineEntity
    d = line.p1 - line.p0
    length = float(np.linalg.norm(d))
    if length < 1e-10:
        return None
    n = np.array([-d[1], d[0]]) / length * dist
    return LineEntity(line.p0 + n, line.p1 + n)


def offset_arc(arc, dist: float):
    """
    Return a new ArcEntity offset by dist (positive = outward from center).
    Returns None if the resulting radius would be <= 0.
    """
    from cad.sketch import ArcEntity
    new_r = arc.radius + dist
    if new_r <= 1e-6:
        return None
    return ArcEntity(arc.center.copy(), new_r, arc.start_angle, arc.end_angle)


def offset_entities(entities: list, dist: float, click_pt: np.ndarray,
                    is_loop: bool) -> list:
    """
    Offset a list of entities by dist.  The sign of dist is determined by
    the click position relative to the geometry.
    Returns a list of new offset entities (may be empty on failure).
    """
    from cad.sketch import LineEntity, ArcEntity

    if not entities:
        return []

    if is_loop:
        inside = _loop_contains(click_pt, entities)
        sign = 1.0 if inside else -1.0
    else:
        e = entities[0]
        if isinstance(e, LineEntity):
            sign = 1.0 if _point_side_of_line(click_pt, e.p0, e.p1) >= 0 else -1.0
        else:
            sign = 1.0 if _point_side_of_arc(click_pt, e) >= 0 else -1.0

    # Offset each entity individually
    offset = []
    for e in entities:
        if isinstance(e, LineEntity):
            o = offset_line(e, sign * dist)
        elif isinstance(e, ArcEntity):
            o = offset_arc(e, sign * dist)
        else:
            continue
        if o is None:
            return []
        offset.append(o)

    if not is_loop or len(offset) < 2:
        return [o for o in offset if o is not None]

    # For a closed loop, reconnect corners by intersecting adjacent offset entities
    return _reconnect_loop(offset)


# ---------------------------------------------------------------------------
# Loop reconnection
# ---------------------------------------------------------------------------

def _intersect_offset_pair(a, b):
    """
    Find the intersection of two offset entities (infinite extensions).
    Returns the intersection point, or None if parallel/no intersection.
    a, b: LineEntity or ArcEntity
    """
    from cad.sketch import LineEntity, ArcEntity

    if isinstance(a, LineEntity) and isinstance(b, LineEntity):
        # Extend both lines infinitely and intersect
        d1 = a.p1 - a.p0
        d2 = b.p1 - b.p0
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) < 1e-10:
            return None   # parallel
        f = b.p0 - a.p0
        t = (f[0] * d2[1] - f[1] * d2[0]) / cross
        return a.p0 + t * d1

    if isinstance(a, ArcEntity) and isinstance(b, LineEntity):
        # Arc endpoint is fixed; project onto line
        return _project_onto_line_infinite(a.p1, b)

    if isinstance(a, LineEntity) and isinstance(b, ArcEntity):
        return _project_onto_line_infinite(b.p0, a)

    if isinstance(a, ArcEntity) and isinstance(b, ArcEntity):
        # Concentric arcs share same center — endpoints connect directly
        return None

    return None


def _project_onto_line_infinite(pt: np.ndarray, line) -> np.ndarray:
    """Project pt onto the infinite extension of line, return the foot point."""
    d = line.p1 - line.p0
    t = float(np.dot(pt - line.p0, d)) / float(np.dot(d, d))
    return line.p0 + t * d


def _reconnect_loop(offset: list) -> list:
    """
    Fix up the corners of an offset loop by intersecting adjacent entity pairs
    and trimming/extending each entity to the intersection point.
    """
    from cad.sketch import LineEntity, ArcEntity
    n = len(offset)
    # Compute corner intersection points
    corners = []
    for i in range(n):
        a = offset[i]
        b = offset[(i + 1) % n]
        pt = _intersect_offset_pair(a, b)
        if pt is None:
            # Fallback: midpoint of the gap
            pt = (a.p1 + b.p0) * 0.5
        corners.append(pt)

    # Rebuild entities with corrected endpoints
    result = []
    for i in range(n):
        e = offset[i]
        new_p0 = corners[(i - 1) % n]
        new_p1 = corners[i]
        if isinstance(e, LineEntity):
            if np.linalg.norm(new_p1 - new_p0) > 1e-6:
                result.append(LineEntity(new_p0, new_p1))
        elif isinstance(e, ArcEntity):
            # Recompute start/end angles from the new corner points
            a0 = math.atan2(float(new_p0[1] - e.center[1]),
                            float(new_p0[0] - e.center[0]))
            a1 = math.atan2(float(new_p1[1] - e.center[1]),
                            float(new_p1[0] - e.center[0]))
            span = (a1 - a0) % (2 * math.pi)
            if span > 1e-6:
                result.append(ArcEntity(e.center.copy(), e.radius, a0, a0 + span))
    return result


# ---------------------------------------------------------------------------
# OffsetTool
# ---------------------------------------------------------------------------

class OffsetTool(BaseTool):
    """
    State machine:
      HOVER   — mouse moves highlight entity/loop under cursor
      SELECTED — entity/loop chosen, waiting for panel distance input
    """

    STATE_HOVER    = 'hover'
    STATE_SELECTED = 'selected'

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None
        self._state = self.STATE_HOVER

        # Set during HOVER by mouse move
        self.hovered_entities: list = []
        self.hovered_is_loop:  bool = False

        # Set when user clicks
        self.selected_entities: list = []
        self.selected_is_loop:  bool = False
        self.selected_click_pt: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.cursor_raw.copy()
                           if snap_result.cursor_raw is not None
                           else snap_result.point)
        if self._state != self.STATE_HOVER:
            return
        self._update_hover(sketch)

    def _update_hover(self, sketch):
        from cad.sketch import LineEntity, ArcEntity
        from cad.sketch_tools.snap import _nearest_on_segment, _nearest_on_arc

        if self._cursor_2d is None:
            self.hovered_entities = []
            return

        drawn = [e for e in sketch.entities
                 if isinstance(e, (LineEntity, ArcEntity))]
        if not drawn:
            self.hovered_entities = []
            return

        # Find closest entity
        best_d = np.inf
        best_e = None
        for e in drawn:
            if isinstance(e, LineEntity):
                p = _nearest_on_segment(self._cursor_2d, e.p0, e.p1)
            else:
                p, _ = _nearest_on_arc(self._cursor_2d, e)
            d = float(np.linalg.norm(self._cursor_2d - p))
            if d < best_d:
                best_d = d
                best_e = e

        if best_e is None:
            self.hovered_entities = []
            return

        # Check if part of a closed loop
        loop = _find_loop_for_entity(best_e, sketch.entities)
        if loop is not None:
            self.hovered_entities = loop
            self.hovered_is_loop  = True
        else:
            self.hovered_entities = [best_e]
            self.hovered_is_loop  = False

    def handle_click(self, snap_result, sketch) -> bool:
        if self._state != self.STATE_HOVER:
            return False
        if not self.hovered_entities:
            return False

        self.selected_entities = list(self.hovered_entities)
        self.selected_is_loop  = self.hovered_is_loop
        self.selected_click_pt = (snap_result.cursor_raw.copy()
                                  if snap_result.cursor_raw is not None
                                  else snap_result.point.copy())
        self._state = self.STATE_SELECTED
        return True

    def apply_offset(self, dist_mm: float, sketch) -> bool:
        """Called by the panel when the user confirms a distance."""
        if not self.selected_entities or self.selected_click_pt is None:
            return False

        result = offset_entities(self.selected_entities, dist_mm,
                                 self.selected_click_pt, self.selected_is_loop)
        if not result:
            return False

        sketch.push_undo_snapshot()
        sketch.entities.extend(result)

        # Reset to hover for another offset
        self._state = self.STATE_HOVER
        self.selected_entities = []
        self.selected_click_pt = None
        return True

    def cancel(self) -> None:
        self._state            = self.STATE_HOVER
        self._cursor_2d        = None
        self.hovered_entities  = []
        self.selected_entities = []
        self.selected_click_pt = None
