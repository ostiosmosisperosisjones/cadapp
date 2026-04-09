"""
viewer/view_cube.py

ViewCube — orientation cube in the top-right corner of the viewport.

- Semi-transparent faces (prefs body color, low alpha)
- Hovered face brightens to active body color
- RGB axis lines inside (X=red, Y=green, Z=blue)
- Wireframe edges
- Click a face to snap camera to that normal
"""

from __future__ import annotations
import numpy as np
from math import sqrt
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
from cad.prefs import prefs

_SIZE_PX  = 140   # logical pixels
_PADDING  = 16    # logical pixels from window edge
_CAM_Z    = 6.5   # camera pullback

_FACES = [
    ('right',  ( 1, 0, 0), [( 1,-1,-1),( 1, 1,-1),( 1, 1, 1),( 1,-1, 1)]),
    ('left',   (-1, 0, 0), [(-1,-1, 1),(-1, 1, 1),(-1, 1,-1),(-1,-1,-1)]),
    ('top',    ( 0, 1, 0), [(-1, 1,-1),( 1, 1,-1),( 1, 1, 1),(-1, 1, 1)]),
    ('bottom', ( 0,-1, 0), [(-1,-1, 1),( 1,-1, 1),( 1,-1,-1),(-1,-1,-1)]),
    ('front',  ( 0, 0, 1), [(-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1)]),
    ('back',   ( 0, 0,-1), [( 1,-1,-1),(-1,-1,-1),(-1, 1,-1),( 1, 1,-1)]),
]

_EDGES = [
    ((-1,-1,-1),( 1,-1,-1)), (( 1,-1,-1),( 1, 1,-1)), (( 1, 1,-1),(-1, 1,-1)), ((-1, 1,-1),(-1,-1,-1)),
    ((-1,-1, 1),( 1,-1, 1)), (( 1,-1, 1),( 1, 1, 1)), (( 1, 1, 1),(-1, 1, 1)), ((-1, 1, 1),(-1,-1, 1)),
    ((-1,-1,-1),(-1,-1, 1)), (( 1,-1,-1),( 1,-1, 1)), (( 1, 1,-1),( 1, 1, 1)), ((-1, 1,-1),(-1, 1, 1)),
]


class ViewCube:
    def __init__(self):
        self._hovered_face: int | None = None

    def draw(self, rotation_matrix: np.ndarray, vw: int, vh: int, dpr: float = 1.0):
        px  = int(_SIZE_PX * dpr)
        pad = int(_PADDING * dpr)
        vp_x = int(vw * dpr) - px - pad
        vp_y = int(vh * dpr) - px - pad

        glViewport(vp_x, vp_y, px, px)

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(28.0, 1.0, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        R = rotation_matrix
        m = [
            R[0,0], R[1,0], R[2,0], 0.0,
            R[0,1], R[1,1], R[2,1], 0.0,
            R[0,2], R[1,2], R[2,2], 0.0,
            0.0,    0.0,    0.0,    1.0,
        ]
        glTranslatef(0.0, 0.0, -_CAM_Z)
        glMultMatrixf(m)

        self._draw_axes(R)
        self._draw_faces()
        self._draw_edges()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

        glViewport(0, 0, int(vw * dpr), int(vh * dpr))

    def handle_mouse_move(self, mx: int, my: int, vw: int, vh: int,
                          rotation_matrix: np.ndarray, dpr: float = 1.0) -> bool:
        old = self._hovered_face
        hit = self._hit_test(mx, my, vw, vh, rotation_matrix, dpr)
        self._hovered_face = hit[1] if hit else None
        return self._hovered_face != old

    def handle_mouse_press(self, mx: int, my: int, vw: int, vh: int,
                           rotation_matrix: np.ndarray, dpr: float = 1.0):
        hit = self._hit_test(mx, my, vw, vh, rotation_matrix, dpr)
        if hit is None:
            return None
        _, normal, _ = _FACES[hit[1]]
        return (normal, False)

    def is_over_cube(self, mx: int, my: int, vw: int, vh: int, dpr: float = 1.0) -> bool:
        return self._in_subviewport(mx, my, vw, vh, dpr) is not None

    def _draw_axes(self, R: np.ndarray):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(0.85, 0.20, 0.20, 1.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(1.1, 0.0, 0.0)
        glColor4f(0.20, 0.75, 0.30, 1.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 1.1, 0.0)
        glColor4f(0.25, 0.50, 0.90, 1.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 1.1)
        glEnd()
        glLineWidth(1.0)

    def _draw_faces(self):
        r0, g0, b0 = prefs.body_color
        r1, g1, b1 = prefs.body_color_active
        for i, (name, normal, verts) in enumerate(_FACES):
            if self._hovered_face == i:
                glColor4f(r1, g1, b1, 0.70)
            else:
                glColor4f(r0, g0, b0, 0.18)
            glBegin(GL_QUADS)
            for v in verts:
                glVertex3f(*v)
            glEnd()

    def _draw_edges(self):
        r, g, b = prefs.edge_color
        glColor4f(max(r, 0.35), max(g, 0.35), max(b, 0.35), 0.8)
        glLineWidth(1.2)
        glBegin(GL_LINES)
        for a, b_ in _EDGES:
            glVertex3f(*a)
            glVertex3f(*b_)
        glEnd()
        glLineWidth(1.0)

    def _in_subviewport(self, mx, my, vw, vh, dpr):
        x0 = vw - _SIZE_PX - _PADDING
        y0 = _PADDING
        x1 = x0 + _SIZE_PX
        y1 = y0 + _SIZE_PX
        if not (x0 <= mx <= x1 and y0 <= my <= y1):
            return None
        nx = (mx - x0) / _SIZE_PX * 2 - 1
        ny = (my - y0) / _SIZE_PX * 2 - 1
        return (nx, ny)

    def _hit_test(self, mx, my, vw, vh, R, dpr):
        ndc = self._in_subviewport(mx, my, vw, vh, dpr)
        if ndc is None:
            return None

        mnx, mny = ndc
        mny = -mny

        import math
        fov_half_tan = math.tan(math.radians(14.0))
        cam_z = _CAM_Z
        RT = R.T

        best_dist = 999.0
        best_idx  = None

        for i, (name, normal, verts) in enumerate(_FACES):
            p  = np.array(normal, dtype=float)
            pc = RT @ p
            # Cull back faces: normal pointing away from camera has positive Z
            if pc[2] > 0:
                continue
            cam_cz = pc[2] - cam_z
            if cam_cz >= -0.01:
                continue
            sx = -pc[0] / (cam_cz * fov_half_tan)
            sy = -pc[1] / (cam_cz * fov_half_tan)
            d = (sx - mnx)**2 + (sy - mny)**2
            if d < 0.55 and d < best_dist:
                best_dist = d
                best_idx  = i

        if best_idx is not None:
            return ('face', best_idx)
        return None
