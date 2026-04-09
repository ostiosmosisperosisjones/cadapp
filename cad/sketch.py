"""
cad/sketch.py

Sketch mode data model.  All coordinates in world millimetres — no
normalisation, matching the rest of the pipeline since mesh coordinates
were moved to world space.

SketchPlane  — wraps a build123d Plane, provides world-space axes/origin
               and ray intersection / coordinate conversion helpers.
SketchMode   — holds the active plane, entities, and tool state.
SketchEntry  — immutable snapshot of a committed sketch, stored in history.
"""

from __future__ import annotations
import copy
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
# Entity types
# ---------------------------------------------------------------------------

class LineEntity:
    """A 2-D line segment in sketch (u, v) coordinates (world mm)."""
    def __init__(self, p0: tuple[float, float], p1: tuple[float, float]):
        self.p0 = np.array(p0, dtype=np.float64)
        self.p1 = np.array(p1, dtype=np.float64)


class ReferenceEntity:
    """
    A projected reference polyline — created by the Include tool.

    Stores a list of 2-D (u, v) points forming a polyline projected from
    a selected 3-D edge or vertex onto the sketch plane.  Rendered in a
    distinct dim style and used as snap targets; not part of the sketch
    profile for extrude.

    source_type : 'edge' | 'vertex'
    """
    def __init__(self, points: list[np.ndarray], source_type: str = 'edge'):
        self.points      = [np.array(p, dtype=np.float64) for p in points]
        self.source_type = source_type


# ---------------------------------------------------------------------------
# SketchPlane
# ---------------------------------------------------------------------------

class SketchPlane:
    """
    Wraps a build123d Plane.  Exposes origin and axes in world mm space,
    matching the coordinate system used by mesh VBOs and the camera.
    """

    def __init__(self, b3d_plane: Plane):
        self._plane = b3d_plane

        wo = b3d_plane.origin
        self.origin = np.array([wo.X, wo.Y, wo.Z], dtype=np.float64)

        xd = b3d_plane.x_dir
        yd = b3d_plane.y_dir
        zd = b3d_plane.z_dir

        self.x_axis = np.array([xd.X, xd.Y, xd.Z], dtype=np.float64)
        self.y_axis = np.array([yd.X, yd.Y, yd.Z], dtype=np.float64)
        self.normal = np.array([zd.X, zd.Y, zd.Z], dtype=np.float64)

        self.x_axis /= np.linalg.norm(self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)
        self.normal  /= np.linalg.norm(self.normal)

    # ------------------------------------------------------------------
    # Ray → 2-D sketch coords
    # ------------------------------------------------------------------

    def ray_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Intersect a ray with this plane (world mm).
        Returns (pt3d, pt2d) or (None, None) if parallel / behind.
        """
        denom = float(np.dot(ray_dir, self.normal))
        if abs(denom) < 1e-9:
            return None, None
        t = float(np.dot(self.origin - ray_origin, self.normal)) / denom
        if t < 0:
            return None, None
        pt3d  = ray_origin + t * ray_dir
        delta = pt3d - self.origin
        u = float(np.dot(delta, self.x_axis))
        v = float(np.dot(delta, self.y_axis))
        return pt3d, np.array([u, v], dtype=np.float64)

    # ------------------------------------------------------------------
    # 2-D → 3-D
    # ------------------------------------------------------------------

    def to_3d(self, u: float, v: float) -> np.ndarray:
        """Sketch (u, v) in mm → world 3-D point."""
        return self.origin + u * self.x_axis + v * self.y_axis

    def project_point(self, world_pt: np.ndarray) -> np.ndarray:
        """
        Orthographically project a world 3-D point onto this plane.
        Returns (u, v) sketch coordinates in mm.
        """
        delta = np.array(world_pt, dtype=np.float64) - self.origin
        u = float(np.dot(delta, self.x_axis))
        v = float(np.dot(delta, self.y_axis))
        return np.array([u, v], dtype=np.float64)


# ---------------------------------------------------------------------------
# SketchMode
# ---------------------------------------------------------------------------

class SketchMode:
    """
    Holds everything about the active sketch session.

    Created when the user double-clicks a planar face.
    Discarded on ESC or when the sketch is committed.

    Parameters
    ----------
    b3d_plane : build123d.Plane   — the face plane in world mm
    body_id   : str               — which body this sketch is on
    face_idx  : int               — face index within that body
    """

    def __init__(self, b3d_plane: Plane, body_id: str, face_idx: int):
        self.plane    = SketchPlane(b3d_plane)
        self.body_id  = body_id
        self.face_idx = face_idx
        self.entities: list = []
        self.tool = SketchTool.NONE

        self._line_start: np.ndarray | None = None
        self._cursor_2d:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def set_tool(self, tool: SketchTool):
        self.tool        = tool
        self._line_start = None
        self._cursor_2d  = None

    def handle_mouse_move(self, ray_origin: np.ndarray, ray_dir: np.ndarray):
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        self._cursor_2d = pt2d

    def handle_click(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> bool:
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if pt2d is None:
            return False
        if self.tool == SketchTool.LINE:
            if self._line_start is None:
                self._line_start = pt2d.copy()
            else:
                self.entities.append(LineEntity(self._line_start, pt2d))
                self._line_start = pt2d.copy()
            return True
        return False

    @property
    def is_empty(self) -> bool:
        return len(self.entities) == 0

    # ------------------------------------------------------------------
    # Include (reference geometry projection)
    # ------------------------------------------------------------------

    def include_selection(self, selection, meshes: dict) -> int:
        """
        Project all selected edges and vertices onto the sketch plane as
        ReferenceEntity instances and append them to self.entities.

        Parameters
        ----------
        selection : SelectionSet
        meshes    : dict[body_id, Mesh]

        Returns the number of reference entities added.
        """
        added = 0

        # Selected edges → project each polyline point
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue
            world_pts = mesh.topo_edges[es.edge_idx]   # (N, 3) float32
            uv_pts = [self.plane.project_point(p) for p in world_pts]
            if len(uv_pts) >= 2:
                self.entities.append(ReferenceEntity(uv_pts, source_type='edge'))
                added += 1

        # Selected vertices → each becomes a single-point reference
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            world_pt = mesh.topo_verts[vs.vertex_idx]
            uv_pt    = self.plane.project_point(world_pt)
            self.entities.append(ReferenceEntity([uv_pt], source_type='vertex'))
            added += 1

        return added


# ---------------------------------------------------------------------------
# SketchEntry  — immutable snapshot of a committed sketch
# ---------------------------------------------------------------------------

class SketchEntry:
    """
    Immutable record of a committed sketch.  Stored on HistoryEntry.params
    under the key "sketch_entry".  shape_before == shape_after on the
    HistoryEntry (sketches don't mutate geometry).

    Attributes
    ----------
    plane_origin  : (3,) float64   — world-space plane origin
    plane_x_axis  : (3,) float64   — world-space X axis of sketch plane
    plane_y_axis  : (3,) float64   — world-space Y axis of sketch plane
    plane_normal  : (3,) float64   — world-space normal
    entities      : list           — deep copy of LineEntity / ReferenceEntity
    body_id       : str
    face_idx      : int
    visible       : bool           — overlay visibility toggle
    """

    def __init__(self, plane_origin, plane_x_axis, plane_y_axis, plane_normal,
                 entities, body_id, face_idx, visible=True):
        self.plane_origin = np.array(plane_origin, dtype=np.float64)
        self.plane_x_axis = np.array(plane_x_axis, dtype=np.float64)
        self.plane_y_axis = np.array(plane_y_axis, dtype=np.float64)
        self.plane_normal = np.array(plane_normal, dtype=np.float64)
        self.entities     = entities
        self.body_id      = body_id
        self.face_idx     = face_idx
        self.visible      = visible

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_sketch_mode(cls, sketch: SketchMode) -> SketchEntry:
        return cls(
            plane_origin = sketch.plane.origin.copy(),
            plane_x_axis = sketch.plane.x_axis.copy(),
            plane_y_axis = sketch.plane.y_axis.copy(),
            plane_normal = sketch.plane.normal.copy(),
            entities     = copy.deepcopy(sketch.entities),
            body_id      = sketch.body_id,
            face_idx     = sketch.face_idx,
        )

    # ------------------------------------------------------------------
    # Derived geometry — built on demand, never cached
    # ------------------------------------------------------------------

    def line_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return all segments (p0_uv, p1_uv) from LineEntity instances.
        Coordinates are 2-D sketch (u, v) in world mm.
        """
        segs = []
        for e in self.entities:
            if isinstance(e, LineEntity):
                segs.append((e.p0.copy(), e.p1.copy()))
        return segs

    def build_wire(self, tol: float = 1e-3):
        """
        Convert all LineEntity segments into a build123d Wire.

        Chains connected segments into polylines, then builds each as a
        build123d Polyline on the sketch plane.  Returns a Wire (single
        chain) or a Compound of wires if the sketch has multiple loops.

        Raises ValueError if there are no line segments.
        """
        from build123d import Wire, Polyline, Vector, Compound

        segs = self.line_segments()
        if not segs:
            raise ValueError("Sketch has no line segments to build a wire from.")

        chains = _chain_segments(segs, tol=tol)

        wires = []
        for chain in chains:
            pts_3d = [
                Vector(*(
                    self.plane_origin
                    + u * self.plane_x_axis
                    + v * self.plane_y_axis
                ).tolist())
                for u, v in chain
            ]
            if len(pts_3d) < 2:
                continue
            wire = Polyline(*pts_3d)
            wires.append(wire)

        if not wires:
            raise ValueError("Could not build any wires from sketch segments.")
        if len(wires) == 1:
            return wires[0]
        return Compound(children=wires)

    def closed_loops(self, tol: float = 1e-3) -> list[list[tuple[float, float]]]:
        """
        Return only chains whose first and last point are within *tol* of
        each other — i.e. closed loops suitable for extrusion.
        """
        segs = self.line_segments()
        chains = _chain_segments(segs, tol=tol)
        closed = []
        for chain in chains:
            p0 = np.array(chain[0])
            p1 = np.array(chain[-1])
            if np.linalg.norm(p1 - p0) < tol:
                closed.append(chain)
        return closed

    def has_closed_loop(self, tol: float = 1e-3) -> bool:
        return len(self.closed_loops(tol=tol)) > 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chain_segments(
    segs: list[tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-3,
) -> list[list[tuple[float, float]]]:
    """
    Greedily chain (p0, p1) segments into ordered polylines.
    Returns a list of point chains, each as [(u, v), …].
    """
    remaining = [(p0.copy(), p1.copy()) for p0, p1 in segs]
    chains: list[list[tuple[float, float]]] = []

    while remaining:
        p0, p1 = remaining.pop(0)
        chain = [(float(p0[0]), float(p0[1])),
                 (float(p1[0]), float(p1[1]))]
        changed = True
        while changed:
            changed = False
            for i, (a, b) in enumerate(remaining):
                tail = np.array(chain[-1])
                head = np.array(chain[0])
                if np.linalg.norm(a - tail) < tol:
                    chain.append((float(b[0]), float(b[1])))
                    remaining.pop(i); changed = True; break
                if np.linalg.norm(b - tail) < tol:
                    chain.append((float(a[0]), float(a[1])))
                    remaining.pop(i); changed = True; break
                if np.linalg.norm(b - head) < tol:
                    chain.insert(0, (float(a[0]), float(a[1])))
                    remaining.pop(i); changed = True; break
                if np.linalg.norm(a - head) < tol:
                    chain.insert(0, (float(b[0]), float(b[1])))
                    remaining.pop(i); changed = True; break
        chains.append(chain)

    return chains
