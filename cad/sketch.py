"""
cad/sketch.py
Sketch mode data model.

SketchPlane  — a build123d Plane transformed into normalised mesh-space,
               providing the axes / origin used by the overlay and picker.
SketchMode   — holds the active plane, sketch entities, and current tool.
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from build123d import Plane


# ---------------------------------------------------------------------------
# Tool enum
# ---------------------------------------------------------------------------

class SketchTool(Enum):
    NONE   = auto()
    LINE   = auto()
    CIRCLE = auto()
    ARC    = auto()


# ---------------------------------------------------------------------------
# Entity types (minimal, extend as needed)
# ---------------------------------------------------------------------------

class LineEntity:
    """A 2-D line segment in sketch coordinates."""
    def __init__(self, p0: tuple[float, float], p1: tuple[float, float]):
        self.p0 = np.array(p0, dtype=np.float64)
        self.p1 = np.array(p1, dtype=np.float64)


# ---------------------------------------------------------------------------
# SketchPlane
# ---------------------------------------------------------------------------

class SketchPlane:
    """
    Wraps a build123d Plane and exposes the origin / axes already transformed
    into the normalised mesh-space used by the OpenGL viewport.

    Parameters
    ----------
    b3d_plane : build123d.Plane
    mesh_center : array-like, shape (3,)   — Mesh.center
    mesh_scale  : float                    — Mesh.scale
    """

    def __init__(self, b3d_plane: Plane, mesh_center, mesh_scale: float):
        self._plane      = b3d_plane
        self.mesh_center = np.array(mesh_center, dtype=np.float64)
        self.mesh_scale  = float(mesh_scale)

        wo = b3d_plane.origin  # build123d Vector
        # Transform origin into normalised GL space
        self.origin = np.array([
            (wo.X - mesh_center[0]) / mesh_scale,
            (wo.Y - mesh_center[1]) / mesh_scale,
            (wo.Z - mesh_center[2]) / mesh_scale,
        ], dtype=np.float64)

        # Axes — pure directions, no translation needed
        xd = b3d_plane.x_dir
        yd = b3d_plane.y_dir   # build123d calls this the in-plane Y
        zd = b3d_plane.z_dir   # face normal

        self.x_axis  = np.array([xd.X, xd.Y, xd.Z], dtype=np.float64)
        self.y_axis  = np.array([yd.X, yd.Y, yd.Z], dtype=np.float64)
        self.normal  = np.array([zd.X, zd.Y, zd.Z], dtype=np.float64)

        # Normalise (should already be unit, but floating-point safety)
        self.x_axis /= np.linalg.norm(self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)
        self.normal  /= np.linalg.norm(self.normal)

    # ------------------------------------------------------------------
    # Ray → 2-D sketch coords
    # ------------------------------------------------------------------

    def ray_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Intersect a ray with this plane (all in normalised GL space).

        Returns
        -------
        pt3d  : (3,) world-space intersection, or None if parallel
        pt2d  : (2,) sketch-space (u, v) coords, or None if parallel
        """
        denom = float(np.dot(ray_dir, self.normal))
        if abs(denom) < 1e-9:
            return None, None

        t = float(np.dot(self.origin - ray_origin, self.normal)) / denom
        if t < 0:
            return None, None

        pt3d = ray_origin + t * ray_dir
        delta = pt3d - self.origin
        u = float(np.dot(delta, self.x_axis))
        v = float(np.dot(delta, self.y_axis))
        return pt3d, np.array([u, v], dtype=np.float64)

    # ------------------------------------------------------------------
    # 2-D → 3-D (for rendering entities)
    # ------------------------------------------------------------------

    def to_3d(self, u: float, v: float) -> np.ndarray:
        """Convert sketch-space coords to normalised GL-space 3-D point."""
        return self.origin + u * self.x_axis + v * self.y_axis


# ---------------------------------------------------------------------------
# SketchMode
# ---------------------------------------------------------------------------

class SketchMode:
    """
    Holds everything about the active sketch session.

    Create one when the user double-clicks a planar face; discard on Escape.
    """

    def __init__(self, sketch_plane: SketchPlane, face_idx: int):
        self.plane     = sketch_plane
        self.face_idx  = face_idx
        self.entities: list = []         # LineEntity, etc.
        self.tool      = SketchTool.NONE

        # In-progress line tool state
        self._line_start: np.ndarray | None = None
        self._cursor_2d:  np.ndarray | None = None  # live cursor position

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def set_tool(self, tool: SketchTool):
        self.tool = tool
        self._line_start = None
        self._cursor_2d  = None

    def handle_mouse_move(self, ray_origin, ray_dir):
        """Update live cursor. Call from Viewport.mouseMoveEvent in SKETCH mode."""
        _, pt2d = self.plane.ray_intersect(
            np.array(ray_origin), np.array(ray_dir)
        )
        self._cursor_2d = pt2d  # may be None if ray is parallel to plane

    def handle_click(self, ray_origin, ray_dir) -> bool:
        """
        Handle a left-click in sketch mode.
        Returns True if a repaint is needed.
        """
        _, pt2d = self.plane.ray_intersect(
            np.array(ray_origin), np.array(ray_dir)
        )
        if pt2d is None:
            return False

        if self.tool == SketchTool.LINE:
            if self._line_start is None:
                self._line_start = pt2d.copy()
                return True
            else:
                self.entities.append(LineEntity(self._line_start, pt2d))
                self._line_start = pt2d.copy()   # chain: next segment starts here
                return True

        return False
