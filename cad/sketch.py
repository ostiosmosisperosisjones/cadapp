"""
cad/sketch.py

Sketch data model.  All coordinates in world millimetres.

Classes
-------
SketchTool      — enum of available tools (add new values here when adding tools)
LineEntity      — a committed 2-D line segment
ReferenceEntity — a projected reference edge/vertex from existing geometry
SketchPlane     — wraps a build123d Plane; ray intersection + coord conversion
SketchMode      — the live sketch session (plane, entities, active tool instance)
SketchEntry     — immutable committed snapshot stored in history

Tool logic lives in cad/sketch_tools/.
"""

from __future__ import annotations
import copy
import numpy as np
from enum import Enum, auto
from build123d import Plane


# ---------------------------------------------------------------------------
# Tool enum  — add a value here when adding a new tool
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

    points      : list of (u, v) np.ndarray — projected onto sketch plane
    source_type : 'edge' | 'vertex'
    occ_edges   : list of TopoDS_Edge | None
                  Original OCCT edges preserved for exact face construction.
    """
    def __init__(self, points: list[np.ndarray], source_type: str = 'edge',
                 occ_edges=None):
        self.points      = [np.array(p, dtype=np.float64) for p in points]
        self.source_type = source_type
        self.occ_edges   = occ_edges


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

    def to_3d(self, u: float, v: float) -> np.ndarray:
        """Sketch (u, v) in mm → world 3-D point."""
        return self.origin + u * self.x_axis + v * self.y_axis

    def project_point(self, world_pt: np.ndarray) -> np.ndarray:
        """Orthographic projection of a world point → (u, v) sketch coords."""
        delta = np.array(world_pt, dtype=np.float64) - self.origin
        u = float(np.dot(delta, self.x_axis))
        v = float(np.dot(delta, self.y_axis))
        return np.array([u, v], dtype=np.float64)


# ---------------------------------------------------------------------------
# SketchMode
# ---------------------------------------------------------------------------

class SketchMode:
    """
    Live sketch session.  Created on double-click, discarded on ESC/commit.

    The active tool is a BaseTool instance (from cad/sketch_tools/).
    set_tool() creates a fresh instance; the old one is discarded.
    """

    def __init__(self, b3d_plane: Plane, body_id: str, face_idx: int):
        self.plane    = SketchPlane(b3d_plane)
        self.body_id  = body_id
        self.face_idx = face_idx
        self.entities: list = []

        # Active tool enum value (for status bar / overlay checks)
        self.tool = SketchTool.NONE
        # Active tool instance — None when no tool selected
        self._active_tool = None

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def set_tool(self, tool_enum: SketchTool):
        """Activate a tool by enum value.  Creates a fresh tool instance."""
        from cad.sketch_tools import TOOLS
        self.tool = tool_enum
        cls = TOOLS.get(tool_enum)
        self._active_tool = cls() if cls is not None else None

    def cancel_tool(self):
        """ESC within a tool — reset in-progress state, stay in the tool."""
        if self._active_tool is not None:
            self._active_tool.cancel()

    # ------------------------------------------------------------------
    # Input forwarding — viewport calls these every frame
    # ------------------------------------------------------------------

    def handle_mouse_move(self, ray_origin: np.ndarray, ray_dir: np.ndarray):
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if self._active_tool is not None:
            self._active_tool.handle_mouse_move(pt2d, self)

    def handle_click(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> bool:
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if pt2d is None or self._active_tool is None:
            return False
        return self._active_tool.handle_click(pt2d, self)

    # ------------------------------------------------------------------
    # Overlay helpers — read by sketch_overlay.py
    # ------------------------------------------------------------------

    @property
    def _cursor_2d(self) -> np.ndarray | None:
        """Current cursor position for the overlay."""
        if self._active_tool is not None:
            return self._active_tool.cursor_2d
        return None

    @property
    def _line_start(self) -> np.ndarray | None:
        """Line preview start point — only valid when LINE tool is active."""
        from cad.sketch_tools.line import LineTool
        if isinstance(self._active_tool, LineTool):
            return self._active_tool.line_start
        return None

    @property
    def is_empty(self) -> bool:
        return len(self.entities) == 0


# ---------------------------------------------------------------------------
# SketchEntry  — immutable snapshot of a committed sketch
# ---------------------------------------------------------------------------

class SketchEntry:
    """
    Immutable record of a committed sketch stored on a HistoryEntry.
    shape_before == shape_after (sketches don't mutate geometry).
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
        return [(e.p0.copy(), e.p1.copy())
                for e in self.entities if isinstance(e, LineEntity)]

    def reference_chains(self) -> list[list[tuple[float, float]]]:
        return [[(float(p[0]), float(p[1])) for p in e.points]
                for e in self.entities
                if isinstance(e, ReferenceEntity) and len(e.points) >= 2]

    def closed_loops(self, tol: float = 1e-3) -> list[list[tuple[float, float]]]:
        """All closed chains from LineEntity segments and ReferenceEntity polylines."""
        closed = []
        segs = self.line_segments()
        if segs:
            for chain in _chain_segments(segs, tol=tol):
                if np.linalg.norm(np.array(chain[-1]) - np.array(chain[0])) < tol:
                    closed.append(chain)
        for chain in self.reference_chains():
            if np.linalg.norm(np.array(chain[-1]) - np.array(chain[0])) < tol:
                closed.append(chain)
        return closed

    def has_closed_loop(self, tol: float = 1e-3) -> bool:
        return len(self.closed_loops(tol=tol)) > 0

    def build_wire(self, tol: float = 1e-3):
        """Convert LineEntity segments into a build123d Wire (for DXF export etc.)."""
        from build123d import Polyline, Vector, Compound
        segs = self.line_segments()
        if not segs:
            raise ValueError("Sketch has no line segments to build a wire from.")
        wires = []
        for chain in _chain_segments(segs, tol=tol):
            pts = [Vector(*(self.plane_origin
                            + u * self.plane_x_axis
                            + v * self.plane_y_axis).tolist())
                   for u, v in chain]
            if len(pts) >= 2:
                wires.append(Polyline(*pts))
        if not wires:
            raise ValueError("Could not build any wires.")
        return wires[0] if len(wires) == 1 else Compound(children=wires)

    def build_faces(self, tol: float = 1e-3) -> list:
        """
        Build a planar Face for each closed loop.

        Reference loops: projects original OCCT edges onto the sketch plane
        using GeomProjLib.ProjectOnPlane (preserves curve type — circle stays
        a circle regardless of offset distance).
        Line loops: built from (u,v) points which are already on the plane.
        Falls back to tessellated polyline if projection fails.
        """
        from build123d import Face, Polyline, Vector
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                        BRepBuilderAPI_MakeWire,
                                        BRepBuilderAPI_MakeEdge)
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomProjLib import GeomProjLib
        from OCP.Geom import Geom_Plane
        from OCP.gp import gp_Ax3, gp_Pnt, gp_Dir

        try:
            ax3 = gp_Ax3(
                gp_Pnt(float(self.plane_origin[0]),
                       float(self.plane_origin[1]),
                       float(self.plane_origin[2])),
                gp_Dir(float(self.plane_normal[0]),
                       float(self.plane_normal[1]),
                       float(self.plane_normal[2])),
            )
            geom_plane = Geom_Plane(ax3)
            proj_dir   = gp_Dir(float(self.plane_normal[0]),
                                float(self.plane_normal[1]),
                                float(self.plane_normal[2]))
        except Exception as ex:
            print(f"[Sketch] build_faces: could not build plane — {ex}")
            geom_plane = None
            proj_dir   = None

        faces = []

        # --- Reference entity loops ---
        for ref in self.entities:
            if not isinstance(ref, ReferenceEntity):
                continue
            if len(ref.points) < 2:
                continue
            if np.linalg.norm(ref.points[-1] - ref.points[0]) > tol:
                continue

            built = False
            if ref.occ_edges and geom_plane is not None:
                try:
                    wm = BRepBuilderAPI_MakeWire()
                    for occ_edge in ref.occ_edges:
                        adaptor    = BRepAdaptor_Curve(occ_edge)
                        u0         = adaptor.FirstParameter()
                        u1         = adaptor.LastParameter()
                        proj_curve = GeomProjLib.ProjectOnPlane_s(
                            adaptor.Curve().Curve(), geom_plane, proj_dir, True)
                        wm.Add(BRepBuilderAPI_MakeEdge(proj_curve, u0, u1).Edge())
                    if wm.IsDone():
                        fm = BRepBuilderAPI_MakeFace(wm.Wire(), True)
                        if fm.IsDone():
                            faces.append(Face(fm.Face()))
                            built = True
                        else:
                            print(f"[Sketch] build_faces: MakeFace error {fm.Error()}")
                    else:
                        print("[Sketch] build_faces: MakeWire failed")
                except Exception as ex:
                    print(f"[Sketch] build_faces: projection failed — {ex}")

            if not built:
                pts = [Vector(*(self.plane_origin
                                + float(p[0]) * self.plane_x_axis
                                + float(p[1]) * self.plane_y_axis).tolist())
                       for p in ref.points]
                if len(pts) >= 3:
                    try:
                        fm = BRepBuilderAPI_MakeFace(
                            Polyline(*pts).wrapped, True)
                        if fm.IsDone():
                            faces.append(Face(fm.Face()))
                            print("[Sketch] build_faces: used polyline fallback")
                        else:
                            print(f"[Sketch] build_faces: polyline fallback "
                                  f"error {fm.Error()}")
                    except Exception as ex:
                        print(f"[Sketch] build_faces: polyline fallback — {ex}")

        # --- Line segment loops ---
        segs = self.line_segments()
        if segs:
            for loop in _chain_segments(segs, tol=tol):
                if np.linalg.norm(np.array(loop[-1]) - np.array(loop[0])) > tol:
                    continue
                pts = [Vector(*(self.plane_origin
                                + u * self.plane_x_axis
                                + v * self.plane_y_axis).tolist())
                       for u, v in loop]
                if len(pts) < 3:
                    continue
                try:
                    fm = BRepBuilderAPI_MakeFace(Polyline(*pts).wrapped, True)
                    if fm.IsDone():
                        faces.append(Face(fm.Face()))
                    else:
                        print(f"[Sketch] build_faces: line loop error {fm.Error()}")
                except Exception as ex:
                    print(f"[Sketch] build_faces: line loop failed — {ex}")

        return faces


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chain_segments(
    segs: list[tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-3,
) -> list[list[tuple[float, float]]]:
    """Greedily chain (p0, p1) segments into ordered polylines."""
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
