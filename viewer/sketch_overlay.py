"""
viewer/sketch_overlay.py
Renders the sketch grid and entities into the existing OpenGL context.

All drawing happens in normalised mesh-space (same space as the mesh VBOs).
Uses immediate-mode GL lines — good enough for an overlay; no extra VBOs needed
since sketch geometry changes every frame while editing.
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *
from cad.sketch import SketchMode, SketchTool, LineEntity


# ---------------------------------------------------------------------------
# Adaptive grid spacing
# ---------------------------------------------------------------------------

_GRID_STEPS = [0.001, 0.002, 0.005,
               0.01,  0.02,  0.05,
               0.1,   0.2,   0.5,
               1.0,   2.0,   5.0,
               10.0,  20.0,  50.0]

def _choose_spacing(ortho_scale: float) -> float:
    """Pick a grid line spacing that puts roughly 8–20 lines on screen."""
    target = ortho_scale / 10.0
    for s in _GRID_STEPS:
        if s >= target:
            return s
    return _GRID_STEPS[-1]


# ---------------------------------------------------------------------------
# SketchOverlay
# ---------------------------------------------------------------------------

class SketchOverlay:
    """
    Stateless helper — call draw() every paintGL when in SKETCH mode.

    Parameters passed at draw time so there is nothing to keep in sync.
    """

    # Grid colours
    GRID_MAJOR_COLOR  = (0.45, 0.55, 0.70, 0.85)   # blueish, opaque enough
    GRID_MINOR_COLOR  = (0.30, 0.38, 0.50, 0.45)
    AXIS_X_COLOR      = (0.85, 0.25, 0.25, 1.0)    # red  — sketch X
    AXIS_Y_COLOR      = (0.25, 0.75, 0.35, 1.0)    # green — sketch Y
    ENTITY_COLOR      = (0.15, 0.75, 1.00, 1.0)    # cyan — committed lines
    PREVIEW_COLOR     = (1.00, 0.85, 0.20, 1.0)    # yellow — in-progress line
    CURSOR_COLOR      = (1.00, 1.00, 1.00, 0.90)   # white crosshair

    GRID_HALF_EXTENT  = 2.5   # how far the grid extends in normalised space
    AXIS_HALF_LENGTH  = 2.5
    CURSOR_RADIUS     = 0.012  # crosshair arm length

    def draw(self, sketch: SketchMode, ortho_scale: float):
        """
        Draw the full overlay.  Must be called while the GL context is current,
        after the mesh has been drawn (overlay is rendered on top).

        sketch      — active SketchMode instance
        ortho_scale — Camera.ortho_scale (drives grid spacing)
        """
        plane = sketch.plane

        # Save all relevant GL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)

        spacing = _choose_spacing(ortho_scale)
        self._draw_grid(plane, spacing)
        self._draw_axes(plane)
        self._draw_entities(sketch, plane)
        self._draw_preview(sketch, plane)
        self._draw_cursor(sketch, plane)

        glPopAttrib()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pt(self, plane, u, v):
        """Plane-space → 3-D (returns a flat Python list for glVertex3f)."""
        p = plane.to_3d(u, v)
        return (float(p[0]), float(p[1]), float(p[2]))

    def _draw_grid(self, plane, spacing):
        half = self.GRID_HALF_EXTENT
        n = int(half / spacing) + 1

        glBegin(GL_LINES)

        # Minor grid — every 'spacing'
        r, g, b, a = self.GRID_MINOR_COLOR
        glColor4f(r, g, b, a)
        for i in range(-n, n + 1):
            t = i * spacing
            if abs(t) < 1e-9:
                continue   # skip axes (drawn separately)
            p0 = self._pt(plane, t, -half)
            p1 = self._pt(plane, t,  half)
            glVertex3f(*p0); glVertex3f(*p1)
            p0 = self._pt(plane, -half, t)
            p1 = self._pt(plane,  half, t)
            glVertex3f(*p0); glVertex3f(*p1)

        # Major grid — every 5× spacing
        major = spacing * 5.0
        nm = int(half / major) + 1
        r, g, b, a = self.GRID_MAJOR_COLOR
        glColor4f(r, g, b, a)
        for i in range(-nm, nm + 1):
            t = i * major
            if abs(t) < 1e-9:
                continue
            p0 = self._pt(plane, t, -half)
            p1 = self._pt(plane, t,  half)
            glVertex3f(*p0); glVertex3f(*p1)
            p0 = self._pt(plane, -half, t)
            p1 = self._pt(plane,  half, t)
            glVertex3f(*p0); glVertex3f(*p1)

        glEnd()

    def _draw_axes(self, plane):
        half = self.AXIS_HALF_LENGTH
        glLineWidth(1.8)
        glBegin(GL_LINES)

        # X axis (red)
        r, g, b, a = self.AXIS_X_COLOR
        glColor4f(r, g, b, a)
        glVertex3f(*self._pt(plane, -half, 0))
        glVertex3f(*self._pt(plane,  half, 0))

        # Y axis (green)
        r, g, b, a = self.AXIS_Y_COLOR
        glColor4f(r, g, b, a)
        glVertex3f(*self._pt(plane, 0, -half))
        glVertex3f(*self._pt(plane, 0,  half))

        glEnd()
        glLineWidth(1.0)

    def _draw_entities(self, sketch, plane):
        r, g, b, a = self.ENTITY_COLOR
        glColor4f(r, g, b, a)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for ent in sketch.entities:
            if isinstance(ent, LineEntity):
                glVertex3f(*self._pt(plane, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt(plane, ent.p1[0], ent.p1[1]))
        glEnd()
        glLineWidth(1.0)

    def _draw_preview(self, sketch, plane):
        """Draw the in-progress line from the first click to the cursor."""
        if sketch.tool != SketchTool.LINE:
            return
        if sketch._line_start is None or sketch._cursor_2d is None:
            return

        r, g, b, a = self.PREVIEW_COLOR
        glColor4f(r, g, b, a)
        glLineWidth(1.6)
        glBegin(GL_LINES)
        glVertex3f(*self._pt(plane, sketch._line_start[0], sketch._line_start[1]))
        glVertex3f(*self._pt(plane, sketch._cursor_2d[0],  sketch._cursor_2d[1]))
        glEnd()

        # Small dot at the anchor point
        self._draw_point(plane, sketch._line_start[0], sketch._line_start[1],
                         r, g, b, a, size=5.0)
        glLineWidth(1.0)

    def _draw_cursor(self, sketch, plane):
        """Small crosshair at current cursor position."""
        if sketch._cursor_2d is None:
            return
        u, v = sketch._cursor_2d
        cr = self.CURSOR_RADIUS
        r, g, b, a = self.CURSOR_COLOR
        glColor4f(r, g, b, a)
        glLineWidth(1.2)
        glBegin(GL_LINES)
        glVertex3f(*self._pt(plane, u - cr, v))
        glVertex3f(*self._pt(plane, u + cr, v))
        glVertex3f(*self._pt(plane, u, v - cr))
        glVertex3f(*self._pt(plane, u, v + cr))
        glEnd()
        glLineWidth(1.0)

    def _draw_point(self, plane, u, v, r, g, b, a, size=4.0):
        glPointSize(size)
        glColor4f(r, g, b, a)
        glBegin(GL_POINTS)
        glVertex3f(*self._pt(plane, u, v))
        glEnd()
        glPointSize(1.0)
