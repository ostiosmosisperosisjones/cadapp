"""
viewer/hover.py

HoverState — screen-space projection cache and hover queries.

Occlusion testing uses ray casting (same Möller–Trumbore code as face
picking) rather than depth buffer sampling.  The depth buffer approach
fails when the far clip plane is very large (all depths compress to ~0.999),
making the per-pixel tolerance meaningless.  Ray casting is exact regardless
of clip plane range.
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *


VERTEX_HOVER_RADIUS = 12   # screen pixels
EDGE_HOVER_RADIUS   = 6    # screen pixels


def _ray_hits_anything(eye: np.ndarray, target: np.ndarray,
                        meshes: dict, workspace) -> bool:
    """
    Return True if any mesh triangle blocks the line of sight from eye
    to target.

    Ray is cast FROM target TOWARD eye so the origin is near the surface
    being tested.  We offset the origin a tiny bit along the ray to avoid
    the surface immediately self-intersecting.  Any hit with t < (dist - epsilon)
    means something is in the way.
    """
    from cad.picker import pick_face

    direction = eye - target
    dist      = float(np.linalg.norm(direction))
    if dist < 1e-9:
        return False
    ray_dir = direction / dist

    # Offset origin slightly toward eye so we don't hit the surface itself
    origin = target + ray_dir * 0.05

    for body_id, mesh in meshes.items():
        body = workspace.bodies.get(body_id)
        if body and not body.visible:
            continue
        result = pick_face(mesh, origin, ray_dir, return_t=True)
        if result is None:
            continue
        _, t = result
        # If hit is before we reach the eye, something is occluding
        if t < dist - 0.05:
            return True
    return False


class HoverState:
    """
    Holds cached screen-space projections and answers hover queries.

    Usage
    -----
    After paintGL (geometry drawn, matrices captured):
        hover.rebuild(meshes, workspace, modelview, projection, viewport, dpr)

    On mouseMoveEvent:
        body_id, vert_idx = hover.vertex_at(x, y)
        body_id, edge_idx = hover.edge_at(x, y)
    """

    def __init__(self):
        self._sv:    dict[str, np.ndarray]       = {}  # screen verts  (N,2)
        self._sv3d:  dict[str, np.ndarray]       = {}  # world verts   (N,3)
        self._se:    dict[str, list[np.ndarray]] = {}  # screen edges  list(M,2)
        self._se3d:  dict[str, list[np.ndarray]] = {}  # world edges   list(M,3)
        self._eye:   np.ndarray | None           = None
        self._meshes  = None
        self._workspace = None
        self._ready   = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def rebuild(self, meshes, workspace, modelview, projection, viewport, dpr,
                camera_eye: np.ndarray = None):
        """Project all topo verts/edges to screen space.  Call once per paintGL."""
        self._ready = False
        if modelview is None:
            return

        mv  = np.array(modelview,  dtype=np.float64).reshape(4, 4)
        prj = np.array(projection, dtype=np.float64).reshape(4, 4)
        mvp = mv @ prj
        vw  = float(viewport[2])
        vh  = float(viewport[3])

        self._eye       = camera_eye.copy() if camera_eye is not None \
                          else np.zeros(3)
        self._meshes    = meshes
        self._workspace = workspace

        def _project(pts: np.ndarray) -> np.ndarray:
            """(N,3) world → (N,2) logical widget pixels."""
            n      = len(pts)
            ones   = np.ones((n, 1), dtype=np.float64)
            clip   = np.hstack([pts.astype(np.float64), ones]) @ mvp
            w      = clip[:, 3]
            safe_w = np.where(np.abs(w) > 1e-9, w, 1e-9)
            ndcx   = clip[:, 0] / safe_w
            ndcy   = clip[:, 1] / safe_w
            sx     = (ndcx * 0.5 + 0.5) * vw
            sy     = (ndcy * 0.5 + 0.5) * vh
            wx     = sx / dpr
            wy     = (vh - sy) / dpr
            return np.stack([wx, wy], axis=1)

        self._sv.clear();   self._sv3d.clear()
        self._se.clear();   self._se3d.clear()

        for body_id, mesh in meshes.items():
            body = workspace.bodies.get(body_id)
            if body and not body.visible:
                continue

            tv = mesh.topo_verts
            self._sv[body_id]   = _project(tv) if len(tv) > 0 \
                                   else np.zeros((0, 2))
            self._sv3d[body_id] = tv

            se_list, se3d_list = [], []
            for edge_pts in mesh.topo_edges:
                se_list.append(_project(edge_pts))
                se3d_list.append(edge_pts)
            self._se[body_id]   = se_list
            self._se3d[body_id] = se3d_list

        self._ready = True

    # ------------------------------------------------------------------
    # Occlusion
    # ------------------------------------------------------------------

    def _visible(self, world_pt: np.ndarray) -> bool:
        """True if world_pt has clear line-of-sight from the camera eye."""
        if self._eye is None or self._meshes is None:
            return True
        return not _ray_hits_anything(self._eye, world_pt,
                                      self._meshes, self._workspace)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def vertex_at(self, x: float, y: float) -> tuple[str | None, int | None]:
        """Closest visible topo vertex within VERTEX_HOVER_RADIUS pixels."""
        if not self._ready:
            return None, None

        # First pass: find candidates within screen radius (cheap)
        candidates: list[tuple[float, str, int]] = []
        for body_id, sv in self._sv.items():
            if len(sv) == 0:
                continue
            dists = np.hypot(sv[:, 0] - x, sv[:, 1] - y)
            mask  = dists < VERTEX_HOVER_RADIUS
            for i in np.where(mask)[0]:
                candidates.append((float(dists[i]), body_id, int(i)))

        if not candidates:
            return None, None

        # Second pass: ray-cast only the candidates, closest screen first
        candidates.sort()
        for dist, body_id, i in candidates:
            wp = self._sv3d[body_id][i].astype(np.float64)
            if self._visible(wp):
                return body_id, i

        return None, None

    def edge_at(self, x: float, y: float) -> tuple[str | None, int | None]:
        """Closest visible topo edge within EDGE_HOVER_RADIUS pixels."""
        if not self._ready:
            return None, None

        # First pass: screen-space distance to each edge polyline
        candidates: list[tuple[float, str, int, np.ndarray]] = []

        for body_id, sedges in self._se.items():
            for ei, se in enumerate(sedges):
                if len(se) < 2:
                    continue
                a      = se[:-1]
                b      = se[1:]
                ab     = b - a
                len_sq = (ab * ab).sum(axis=1)
                t      = ((np.array([[x, y]]) - a) * ab).sum(axis=1) / \
                         np.where(len_sq > 1e-9, len_sq, 1e-9)
                t       = np.clip(t, 0.0, 1.0)
                closest = a + t[:, np.newaxis] * ab
                dists   = np.hypot(closest[:, 0] - x, closest[:, 1] - y)
                d       = float(dists.min())
                if d < EDGE_HOVER_RADIUS:
                    # Store the 3D point at the closest location on the edge
                    best_seg = int(np.argmin(dists))
                    pts3d    = self._se3d[body_id][ei]
                    # Interpolate world-space point at parameter t
                    t_val = float(t[best_seg])
                    if best_seg + 1 < len(pts3d):
                        wp3d = (pts3d[best_seg].astype(np.float64) * (1 - t_val) +
                                pts3d[best_seg + 1].astype(np.float64) * t_val)
                    else:
                        wp3d = pts3d[best_seg].astype(np.float64)
                    candidates.append((d, body_id, ei, wp3d))

        if not candidates:
            return None, None

        # Second pass: ray-cast candidates, closest screen dist first
        candidates.sort(key=lambda c: c[0])
        for d, body_id, ei, wp3d in candidates:
            if self._visible(wp3d):
                return body_id, ei

        return None, None

    def clear(self):
        self._sv.clear();   self._sv3d.clear()
        self._se.clear();   self._se3d.clear()
        self._eye     = None
        self._meshes  = None
        self._workspace = None
        self._ready   = False
