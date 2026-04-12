"""
viewer/view_cube.py

ViewCube — orientation cube in the top-right corner of the viewport.

- Opaque front faces only (back-face culled via dot-product with camera)
- Isolated depth buffer (scissor-clear trick) — never z-fights with the scene
- Hovered face brightens
- RGB axis lines, wireframe edges
- Click a face to snap camera to that normal
"""

from __future__ import annotations
import numpy as np
from math import sqrt, radians, tan, inf
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
from cad.prefs import prefs

_SIZE_PX  = 140   # logical pixels
_PADDING  = 16    # logical pixels from window edge
_CAM_Z    = 6.5   # camera pullback distance

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

# Pre-computed face normals as numpy arrays
_NORMALS = [np.array(n, dtype=float) for _, n, _ in _FACES]


class ViewCube:
    def __init__(self):
        self._hovered_face: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, rotation_matrix: np.ndarray, vw: int, vh: int,
             dpr: float = 1.0):
        px  = int(_SIZE_PX * dpr)
        pad = int(_PADDING * dpr)
        vp_x = int(vw * dpr) - px - pad
        vp_y = int(vh * dpr) - px - pad

        # Isolate depth buffer — clear only this sub-viewport region
        glEnable(GL_SCISSOR_TEST)
        glScissor(vp_x, vp_y, px, px)
        glClear(GL_DEPTH_BUFFER_BIT)
        glDisable(GL_SCISSOR_TEST)

        glViewport(vp_x, vp_y, px, px)

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
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
        # The view-cube camera is fixed at (0,0,_CAM_Z).  We need to rotate
        # the cube by the VIEW rotation (world→camera = R.T), not the
        # camera's local→world rotation (R).  In OpenGL column-major the
        # column-major encoding of R.T is just R with row/col swapped:
        m = [
            R[0,0], R[0,1], R[0,2], 0.0,
            R[1,0], R[1,1], R[1,2], 0.0,
            R[2,0], R[2,1], R[2,2], 0.0,
            0.0,    0.0,    0.0,    1.0,
        ]
        glTranslatef(0.0, 0.0, -_CAM_Z)
        glMultMatrixf(m)

        # Sort faces back-to-front so semi-transparent front faces blend right
        RT = R.T
        face_order = sorted(range(len(_FACES)),
                            key=lambda i: float((RT @ _NORMALS[i])[2]))

        self._draw_faces(RT, face_order)
        self._draw_edges()
        self._draw_axes()

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

    def is_over_cube(self, mx: int, my: int, vw: int, vh: int,
                     dpr: float = 1.0) -> bool:
        return self._in_subviewport(mx, my, vw, vh, dpr) is not None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_faces(self, RT: np.ndarray, face_order: list):
        r0, g0, b0 = prefs.body_color
        r1, g1, b1 = prefs.body_color_active

        for i in face_order:
            name, normal, verts = _FACES[i]
            cam_z = float((RT @ _NORMALS[i])[2])
            hovered = (self._hovered_face == i)

            # Skip back-facing faces unless they're hovered — then draw faintly
            # so edge-on / back faces still respond visually to the mouse.
            if cam_z < 0.0 and not hovered:
                continue

            if hovered:
                glColor4f(r1, g1, b1, 1.0)
            else:
                glColor4f(r0, g0, b0, 1.0)

            glBegin(GL_QUADS)
            for v in verts:
                glVertex3f(*v)
            glEnd()

    def _draw_edges(self):
        r, g, b = prefs.edge_color
        glColor4f(max(r, 0.45), max(g, 0.45), max(b, 0.45), 1.0)
        glLineWidth(1.2)
        glBegin(GL_LINES)
        for a, b_ in _EDGES:
            glVertex3f(*a)
            glVertex3f(*b_)
        glEnd()
        glLineWidth(1.0)

    def _draw_axes(self):
        # Disable depth test so axis lines always appear on top of faces
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(0.85, 0.20, 0.20, 1.0); glVertex3f(0,0,0); glVertex3f(1.1,0,0)
        glColor4f(0.20, 0.75, 0.30, 1.0); glVertex3f(0,0,0); glVertex3f(0,1.1,0)
        glColor4f(0.25, 0.50, 0.90, 1.0); glVertex3f(0,0,0); glVertex3f(0,0,1.1)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

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
        mny = -mny   # screen Y is flipped vs NDC Y

        fov_half_tan = tan(radians(14.0))
        RT = R.T

        # Ray from eye (origin) through the NDC pixel, in eye space.
        # Model matrix is R.T; cube centre is at (0, 0, -_CAM_Z) in eye space.
        ray_o = np.zeros(3)
        ray_d = np.array([mnx * fov_half_tan, mny * fov_half_tan, -1.0])
        ray_d = ray_d / np.linalg.norm(ray_d)

        best_t   = inf
        best_idx = None

        for i, (name, normal, verts) in enumerate(_FACES):
            cam_z_i = float((RT @ _NORMALS[i])[2])
            if cam_z_i < 0.0:
                continue  # skip back-facing faces

            # Transform quad vertices into eye space
            eye_verts = [RT @ np.array(v, dtype=float) +
                         np.array([0.0, 0.0, -_CAM_Z]) for v in verts]

            # Ray–quad intersection (two triangles)
            t = _ray_quad(ray_o, ray_d, eye_verts)
            if t is not None and t < best_t:
                best_t   = t
                best_idx = i

        if best_idx is not None:
            return ('face', best_idx)

        # Fallback: face-centre proximity for edge-on / nearly-edge-on faces.
        # Project each visible (or edge-on) face centre to screen NDC and pick
        # the closest one within a generous radius.
        best_d   = 0.65   # NDC radius — about half the cube's apparent size
        best_idx = None
        for i in range(len(_FACES)):
            cam_z_i = float((RT @ _NORMALS[i])[2])
            if cam_z_i < 0.0:          # skip faces clearly pointing away
                continue
            c_eye = RT @ _NORMALS[i] + np.array([0.0, 0.0, -_CAM_Z])
            sx = c_eye[0] / (-c_eye[2]) / fov_half_tan
            sy = c_eye[1] / (-c_eye[2]) / fov_half_tan
            d = sqrt((sx - mnx) ** 2 + (sy - mny) ** 2)
            if d < best_d:
                best_d   = d
                best_idx = i

        return ('face', best_idx) if best_idx is not None else None


# ---------------------------------------------------------------------------
# Ray helpers
# ---------------------------------------------------------------------------

def _ray_tri(ro, rd, v0, v1, v2):
    """Möller–Trumbore. Returns t > 0 on hit, else None."""
    EPS = 1e-8
    e1 = v1 - v0
    e2 = v2 - v0
    h  = np.cross(rd, e2)
    a  = float(np.dot(e1, h))
    if abs(a) < EPS:
        return None
    f = 1.0 / a
    s = ro - v0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * float(np.dot(rd, q))
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * float(np.dot(e2, q))
    return t if t > EPS else None


def _ray_quad(ro, rd, verts):
    """Ray–quad intersection (quad split into two triangles). Returns t or None."""
    v0, v1, v2, v3 = verts
    t = _ray_tri(ro, rd, v0, v1, v2)
    if t is None:
        t = _ray_tri(ro, rd, v0, v2, v3)
    return t
