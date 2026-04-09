"""
viewer/sketch_overlay.py

Renders the sketch grid, axes, entities, and cursor.
All colors read from cad.prefs.  All coordinates in world mm.
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *
from cad.sketch import SketchMode, SketchTool, LineEntity, ReferenceEntity
from cad.prefs import prefs


# ---------------------------------------------------------------------------
# Adaptive grid spacing
# ---------------------------------------------------------------------------

_GRID_STEPS = [
    0.01, 0.02, 0.05,
    0.1,  0.2,  0.5,
    1.0,  2.0,  5.0,
    10.0, 20.0, 50.0,
    100.0, 200.0, 500.0,
]

def _choose_spacing(camera_distance: float) -> float:
    target = camera_distance / 15.0
    for s in _GRID_STEPS:
        if s >= target:
            return s
    return _GRID_STEPS[-1]


# ---------------------------------------------------------------------------
# SketchOverlay
# ---------------------------------------------------------------------------

class SketchOverlay:
    """Stateless helper — call draw() every paintGL when in sketch mode."""

    def draw(self, sketch: SketchMode, camera_distance: float):
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        spacing = _choose_spacing(camera_distance)
        extent  = camera_distance * 2.0

        self._draw_grid(sketch.plane, spacing, extent)
        self._draw_axes(sketch.plane, extent)
        self._draw_references(sketch)
        self._draw_entities(sketch)
        self._draw_preview(sketch)
        self._draw_cursor(sketch, camera_distance)

        glPopAttrib()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pt(self, plane, u, v):
        p = plane.to_3d(u, v)
        return (float(p[0]), float(p[1]), float(p[2]))

    def _draw_grid(self, plane, spacing, extent):
        n  = int(extent / spacing) + 1
        nm = int(extent / (spacing * 5)) + 1

        glEnable(GL_BLEND)
        glBegin(GL_LINES)

        r, g, b = prefs.sketch_grid_minor_color
        glColor4f(r, g, b, 0.35)
        for i in range(-n, n + 1):
            t = i * spacing
            if abs(t) < spacing * 0.01:
                continue
            glVertex3f(*self._pt(plane,  t,      -extent))
            glVertex3f(*self._pt(plane,  t,       extent))
            glVertex3f(*self._pt(plane, -extent,  t))
            glVertex3f(*self._pt(plane,  extent,  t))

        major = spacing * 5.0
        r, g, b = prefs.sketch_grid_major_color
        glColor4f(r, g, b, 0.70)
        for i in range(-nm, nm + 1):
            t = i * major
            if abs(t) < major * 0.01:
                continue
            glVertex3f(*self._pt(plane,  t,      -extent))
            glVertex3f(*self._pt(plane,  t,       extent))
            glVertex3f(*self._pt(plane, -extent,  t))
            glVertex3f(*self._pt(plane,  extent,  t))

        glEnd()
        glDisable(GL_BLEND)

    def _draw_axes(self, plane, extent):
        glLineWidth(1.8)
        glBegin(GL_LINES)
        glColor3f(*prefs.sketch_axis_x_color)
        glVertex3f(*self._pt(plane, -extent, 0))
        glVertex3f(*self._pt(plane,  extent, 0))
        glColor3f(*prefs.sketch_axis_y_color)
        glVertex3f(*self._pt(plane, 0, -extent))
        glVertex3f(*self._pt(plane, 0,  extent))
        glEnd()
        glLineWidth(1.0)

    def _draw_references(self, sketch: SketchMode):
        refs = [e for e in sketch.entities if isinstance(e, ReferenceEntity)]
        if not refs:
            return
        r, g, b = prefs.sketch_reference_color
        glEnable(GL_BLEND)
        glColor4f(r, g, b, 0.75)
        glLineWidth(prefs.sketch_reference_width)
        for ref in refs:
            if len(ref.points) == 1:
                p  = ref.points[0]
                cr = 0.8
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, p[0] - cr, p[1]))
                glVertex3f(*self._pt(sketch.plane, p[0] + cr, p[1]))
                glVertex3f(*self._pt(sketch.plane, p[0], p[1] - cr))
                glVertex3f(*self._pt(sketch.plane, p[0], p[1] + cr))
                glEnd()
            else:
                glBegin(GL_LINE_STRIP)
                for p in ref.points:
                    glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
                glEnd()
        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_entities(self, sketch: SketchMode):
        lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
        if not lines:
            return
        glColor3f(*prefs.sketch_line_color)
        glLineWidth(prefs.sketch_line_width)
        glBegin(GL_LINES)
        for ent in lines:
            glVertex3f(*self._pt(sketch.plane, ent.p0[0], ent.p0[1]))
            glVertex3f(*self._pt(sketch.plane, ent.p1[0], ent.p1[1]))
        glEnd()
        glLineWidth(1.0)

    def _draw_preview(self, sketch: SketchMode):
        if sketch.tool != SketchTool.LINE:
            return
        if sketch._line_start is None or sketch._cursor_2d is None:
            return
        r, g, b = prefs.sketch_preview_color
        glColor3f(r, g, b)
        glLineWidth(1.6)
        glBegin(GL_LINES)
        glVertex3f(*self._pt(sketch.plane,
                             sketch._line_start[0], sketch._line_start[1]))
        glVertex3f(*self._pt(sketch.plane,
                             sketch._cursor_2d[0],  sketch._cursor_2d[1]))
        glEnd()
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glVertex3f(*self._pt(sketch.plane,
                             sketch._line_start[0], sketch._line_start[1]))
        glEnd()
        glPointSize(1.0)
        glLineWidth(1.0)

    def _draw_cursor(self, sketch: SketchMode, camera_distance: float):
        if sketch._cursor_2d is None:
            return
        u, v = sketch._cursor_2d
        cr   = camera_distance * 0.008
        glColor3f(*prefs.sketch_cursor_color)
        glLineWidth(1.2)
        glBegin(GL_LINES)
        glVertex3f(*self._pt(sketch.plane, u - cr, v))
        glVertex3f(*self._pt(sketch.plane, u + cr, v))
        glVertex3f(*self._pt(sketch.plane, u,      v - cr))
        glVertex3f(*self._pt(sketch.plane, u,      v + cr))
        glEnd()
        glLineWidth(1.0)
