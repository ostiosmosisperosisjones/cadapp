"""
cad/sketch_tools/arc.py

Arc3Tool — 3-point arc (click p1, p2, p3 → fit circle through all three).

Workflow:
  click 1 : set start point
  click 2 : set a point on the arc (mid-constraint)
  click 3 : set end point  →  commit ArcEntity, chain: end becomes new start

ESC after click 1 or 2 drops in-progress state without leaving sketch mode.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


def _circle_from_3pts(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Fit a circle through three 2-D points.
    Returns (center, radius) or None if points are collinear.
    """
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None   # collinear

    ux = ((ax*ax + ay*ay) * (by - cy) +
          (bx*bx + by*by) * (cy - ay) +
          (cx*cx + cy*cy) * (ay - by)) / d
    uy = ((ax*ax + ay*ay) * (cx - bx) +
          (bx*bx + by*by) * (ax - cx) +
          (cx*cx + cy*cy) * (bx - ax)) / d

    center = np.array([ux, uy], dtype=np.float64)
    radius = float(np.linalg.norm(p1 - center))
    return center, radius


def _arc_angles(center: np.ndarray, p_start: np.ndarray,
                p_mid: np.ndarray, p_end: np.ndarray
                ) -> tuple[float, float]:
    """
    Return (start_angle, end_angle) in radians such that the arc sweeps
    CCW from p_start through p_mid to p_end.
    """
    def angle(p):
        return np.arctan2(float(p[1] - center[1]), float(p[0] - center[0]))

    a0 = angle(p_start)
    am = angle(p_mid)
    a1 = angle(p_end)

    # Normalize am and a1 relative to a0 in [0, 2π)
    def norm(a, ref):
        d = (a - ref) % (2 * np.pi)
        return d

    dm = norm(am, a0)
    d1 = norm(a1, a0)

    if dm <= d1:
        # mid is between start and end CCW — arc goes CCW
        return a0, a0 + d1
    else:
        # mid is past end CCW — arc goes CW, flip to CCW representation
        return a0 + d1, a0 + 2 * np.pi


class Arc3Tool(BaseTool):

    def __init__(self):
        self._p1:         np.ndarray | None = None   # start
        self._p2:         np.ndarray | None = None   # on-arc
        self._cursor_2d:  np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    # preview geometry read by overlay
    @property
    def arc_p1(self) -> np.ndarray | None:
        return self._p1

    @property
    def arc_p2(self) -> np.ndarray | None:
        return self._p2

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = snap_result.point.copy() if snap_result.point is not None else None
        # Last confirmed point is the anchor for tangent snap
        sketch.snap.anchor_pt = self._p2 if self._p2 is not None else self._p1

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import ArcEntity
        pt = snap_result.point
        if pt is None:
            return False

        if self._p1 is None:
            self._p1 = pt.copy()
            return True

        if self._p2 is None:
            if np.linalg.norm(pt - self._p1) < 1e-6:
                return False   # same point, ignore
            self._p2 = pt.copy()
            return True

        # Third click — commit
        p3 = pt
        if np.linalg.norm(p3 - self._p1) < 1e-6 or np.linalg.norm(p3 - self._p2) < 1e-6:
            return False

        result = _circle_from_3pts(self._p1, self._p2, p3)
        if result is None:
            # Collinear — fall back to line
            from cad.sketch import LineEntity
            sketch.push_undo_snapshot()
            sketch.entities.append(LineEntity(self._p1, p3))
            self._p1 = p3.copy()
            self._p2 = None
            return True

        center, radius = result
        start_angle, end_angle = _arc_angles(center, self._p1, self._p2, p3)

        sketch.push_undo_snapshot()
        sketch.entities.append(ArcEntity(center, radius, start_angle, end_angle))
        # Chain: end of this arc becomes start of next
        self._p1 = p3.copy()
        self._p2 = None
        return True

    def cancel(self) -> None:
        if self._p2 is not None:
            # Drop back to just having p1
            self._p2 = None
        else:
            self._p1 = None
        self._cursor_2d = None
