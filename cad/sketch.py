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
                  This is a baked cache; updated during parametric replay.
    source_type : 'edge' | 'vertex' | 'sketch_edge'
    occ_edges   : list of TopoDS_Edge | None
                  Original OCCT edges preserved for exact face construction.
    source      : EdgeSource | None
                  Stable parametric reference used to re-derive points and
                  occ_edges during replay.  None for vertices and legacy data.
    """
    def __init__(self, points: list[np.ndarray], source_type: str = 'edge',
                 occ_edges=None, source=None):
        self.points      = [np.array(p, dtype=np.float64) for p in points]
        self.source_type = source_type
        self.occ_edges   = occ_edges
        self.source      = source          # EdgeSource | None

    def __deepcopy__(self, memo):
        # TopoDS_Edge is not picklable — share by reference (OCCT shapes are
        # immutable value types so aliasing is safe).
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.points      = copy.deepcopy(self.points, memo)
        result.source_type = self.source_type
        result.occ_edges   = self.occ_edges   # shared reference, not copied
        result.source      = copy.deepcopy(self.source, memo)
        return result


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

    Owns a SnapEngine — snap resolution happens here so every tool gets
    it automatically.  Tools receive SnapResult, never raw pt2d.
    """

    def __init__(self, b3d_plane: Plane, body_id: str, face_idx: int,
                 plane_source=None):
        self.plane        = SketchPlane(b3d_plane)
        self.body_id      = body_id
        self.face_idx     = face_idx
        self.plane_source = plane_source   # SketchPlaneSource | None
        self.entities: list = []

        # Per-sketch undo stack — each entry is a deep-copy of self.entities
        # taken before an entity is committed.  Ctrl+Z pops and restores.
        self._entity_snapshots: list = []

        # Active tool enum value (for status bar / overlay checks)
        self.tool = SketchTool.NONE
        # Active tool instance — None when no tool selected
        self._active_tool = None

        # Snap engine — shared across all tools in this session
        from cad.sketch_tools.snap import SnapEngine
        import numpy as np
        self.snap = SnapEngine()
        # World-plane sketches: origin (0,0) in UV space is always snappable
        if isinstance(plane_source, __import__('cad.plane_ref',
                fromlist=['WorldPlaneSource']).WorldPlaneSource):
            self.snap.extra_points = [np.array([0.0, 0.0])]

        # Last resolved snap result — read by overlay for indicator drawing
        self.last_snap: object = None   # SnapResult | None

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
        """ESC within a tool — discard in-progress state and return to NONE.
        Stays in sketch mode so the user can select geometry, measure, etc."""
        if self._active_tool is not None:
            self._active_tool.cancel()
        self.tool = SketchTool.NONE
        self._active_tool = None
        self.last_snap = None

    # ------------------------------------------------------------------
    # Per-sketch undo
    # ------------------------------------------------------------------

    def push_undo_snapshot(self):
        """
        Save the current entity list before a commit.
        Call this immediately before appending or removing entities so
        Ctrl+Z can restore the previous state.
        """
        import copy
        self._entity_snapshots.append(copy.deepcopy(self.entities))

    def undo_entity(self) -> bool:
        """
        Restore the most recent entity snapshot.
        Returns True if anything was undone, False if the stack is empty.
        """
        if not self._entity_snapshots:
            return False
        self.entities = self._entity_snapshots.pop()
        # Cancel any in-progress tool state so it doesn't reference removed pts
        if self._active_tool is not None:
            self._active_tool.cancel()
        return True

    # ------------------------------------------------------------------
    # Input forwarding — viewport calls these every frame
    # ------------------------------------------------------------------

    def handle_mouse_move(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                          camera_distance: float = 100.0):
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if pt2d is None:
            self.last_snap = None
            return
        snap_result = self.snap.resolve(pt2d, self.entities,
                                        self.plane, camera_distance)
        self.last_snap = snap_result
        if self._active_tool is not None:
            self._active_tool.handle_mouse_move(snap_result, self)

    def handle_click(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                     camera_distance: float = 100.0) -> bool:
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if pt2d is None or self._active_tool is None:
            return False
        if (self.last_snap is not None and
                np.linalg.norm(self.last_snap.cursor_raw - pt2d) < 1e-6):
            snap_result = self.last_snap
        else:
            snap_result = self.snap.resolve(pt2d, self.entities,
                                            self.plane, camera_distance)
            self.last_snap = snap_result
        return self._active_tool.handle_click(snap_result, self)

    # ------------------------------------------------------------------
    # Overlay helpers — read by sketch_overlay.py
    # ------------------------------------------------------------------

    @property
    def _cursor_2d(self) -> np.ndarray | None:
        """Snapped cursor position for the overlay."""
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
# UV-space helpers
# ---------------------------------------------------------------------------

def _uv_point_in_loop(pt: tuple[float, float],
                      loop: list[tuple[float, float]]) -> bool:
    """
    2-D ray-casting point-in-polygon test.
    Returns True if `pt` is strictly inside the polygon defined by `loop`.
    """
    x, y = pt
    n = len(loop)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = loop[i]
        xj, yj = loop[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# SketchEntry  — immutable snapshot of a committed sketch
# ---------------------------------------------------------------------------

class SketchEntry:
    """
    Immutable record of a committed sketch stored on a HistoryEntry.
    shape_before == shape_after (sketches don't mutate geometry).
    """

    def __init__(self, plane_origin, plane_x_axis, plane_y_axis, plane_normal,
                 entities, body_id, face_idx, visible=True, plane_source=None):
        self.plane_origin = np.array(plane_origin, dtype=np.float64)
        self.plane_x_axis = np.array(plane_x_axis, dtype=np.float64)
        self.plane_y_axis = np.array(plane_y_axis, dtype=np.float64)
        self.plane_normal = np.array(plane_normal, dtype=np.float64)
        self.entities     = entities
        self.body_id      = body_id
        self.face_idx     = face_idx
        self.visible      = visible
        self.plane_source = plane_source   # SketchPlaneSource | None

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
            plane_source = sketch.plane_source,
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

    def face_regions(self, tol: float = 1e-3):
        """
        Return UV region descriptors parallel to build_faces().
        Each entry is (outer_uvs, [hole_uvs, ...]) — the outer polygon and
        the direct-child polygons that are punched as holes.
        Used by the picker for exact point-in-region tests.
        """
        uv_loops = self._collect_uv_loops(tol)
        if not uv_loops:
            return []

        def _area(uvs):
            a = 0.0
            for k in range(len(uvs)):
                x0, y0 = uvs[k]; x1, y1 = uvs[(k + 1) % len(uvs)]
                a += x0 * y1 - x1 * y0
            return abs(a) * 0.5

        areas = [_area(uv) for uv in uv_loops]
        n = len(uv_loops)

        def _loop_contains(j, i):
            if areas[j] <= areas[i]:
                return False
            return all(_uv_point_in_loop(pt, uv_loops[j]) for pt in uv_loops[i])

        parent = [None] * n
        for i in range(n):
            containers = [j for j in range(n) if _loop_contains(j, i)]
            if containers:
                parent[i] = min(containers, key=lambda j: areas[j])

        return [
            (uv_loops[i], [uv_loops[j] for j in range(n) if parent[j] == i])
            for i in range(n)
        ]

    def _collect_uv_loops(self, tol: float = 1e-3):
        """Collect closed UV polygon lists from all entities (shared by build_faces and face_regions)."""
        uv_loops = []
        for ref in self.entities:
            if not isinstance(ref, ReferenceEntity) or len(ref.points) < 3:
                continue
            pts_close = np.linalg.norm(ref.points[-1] - ref.points[0]) < tol
            edge_closed = False
            if not pts_close and ref.occ_edges:
                try:
                    from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
                    for _e in ref.occ_edges:
                        _a = _BAC(_e)
                        if _a.IsClosed() or _a.IsPeriodic():
                            edge_closed = True
                            break
                except Exception:
                    pass
            if not pts_close and not edge_closed:
                continue
            raw = ref.points[:-1] if pts_close else ref.points
            uv = [(float(p[0]), float(p[1])) for p in raw]
            if len(uv) >= 3:
                uv_loops.append(uv)

        segs = self.line_segments()
        if segs:
            for loop in _chain_segments(segs, tol=tol):
                if np.linalg.norm(np.array(loop[-1]) - np.array(loop[0])) > tol:
                    continue
                open_loop = [(float(u), float(v)) for u, v in loop[:-1]]
                if len(open_loop) >= 3:
                    uv_loops.append(open_loop)
        return uv_loops

    def _collect_uv_loops_drawn(self, tol: float = 1e-3):
        """Collect closed UV polygon loops from drawn LineEntity segments only."""
        uv_loops = []
        segs = self.line_segments()
        if segs:
            for loop in _chain_segments(segs, tol=tol):
                if np.linalg.norm(np.array(loop[-1]) - np.array(loop[0])) > tol:
                    continue
                open_loop = [(float(u), float(v)) for u, v in loop[:-1]]
                if len(open_loop) >= 3:
                    uv_loops.append(open_loop)
        return uv_loops

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
        Build one planar Face per closed loop (includes and drawn lines).

        Uses BRepAlgoAPI_Splitter as a "cookie cutter": every loop face is
        passed as both an argument and a tool so OCCT computes all planar
        intersections at once and returns naturally non-overlapping regions.
        Ring-shaped regions automatically get the correct topology with an
        outer boundary and one or more inner holes.

        Returns (faces, regions) where regions[k] = (outer_uvs, [hole_uvs])
        parallel to faces[k].  outer_uvs is the vertex list of the largest
        (outer) wire; hole_uvs is a list of vertex lists for inner wires.
        """
        from build123d import Face
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                        BRepBuilderAPI_MakeWire,
                                        BRepBuilderAPI_MakeEdge)
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
        from OCP.BRepTools import BRepTools_WireExplorer
        from OCP.Geom import Geom_Plane
        from OCP.gp import gp_Ax3, gp_Pnt, gp_Dir
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCP.TopTools import TopTools_ListOfShape
        from OCP.TopoDS import TopoDS
        from OCP.BRep import BRep_Tool

        _geom_plane = Geom_Plane(gp_Ax3(
            gp_Pnt(float(self.plane_origin[0]),
                   float(self.plane_origin[1]),
                   float(self.plane_origin[2])),
            gp_Dir(float(self.plane_normal[0]),
                   float(self.plane_normal[1]),
                   float(self.plane_normal[2])),
            gp_Dir(float(self.plane_x_axis[0]),
                   float(self.plane_x_axis[1]),
                   float(self.plane_x_axis[2])),
        ))

        def _uv_to_wire(uv_pts):
            try:
                wm = BRepBuilderAPI_MakeWire()
                n = len(uv_pts)
                for k in range(n):
                    u0, v0 = uv_pts[k]
                    u1, v1 = uv_pts[(k + 1) % n]
                    p0 = gp_Pnt(*(self.plane_origin
                                  + float(u0) * self.plane_x_axis
                                  + float(v0) * self.plane_y_axis).tolist())
                    p1 = gp_Pnt(*(self.plane_origin
                                  + float(u1) * self.plane_x_axis
                                  + float(v1) * self.plane_y_axis).tolist())
                    wm.Add(BRepBuilderAPI_MakeEdge(p0, p1).Edge())
                if wm.IsDone():
                    return wm.Wire()
            except Exception as ex:
                print(f"[Sketch] build_faces: MakeWire error — {ex}")
            return None

        def _project_edge_to_plane(edge):
            """
            Re-create an edge whose 3-D curve is projected onto the sketch plane.
            Handles circles/arcs exactly; falls back to None for other curve types.
            """
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC2
            from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
            from OCP.Geom import Geom_Circle as _GCircle
            from OCP.gp import gp_Ax2 as _gAx2
            adp = _BAC2(edge)
            if adp.GetType() == GeomAbs_Circle:
                circ = adp.Circle()
                # Project centre onto sketch plane
                c3d = circ.Location()
                cw  = np.array([c3d.X(), c3d.Y(), c3d.Z()])
                delta = cw - self.plane_origin
                cu   = float(np.dot(delta, self.plane_x_axis))
                cv   = float(np.dot(delta, self.plane_y_axis))
                c_on_plane = (self.plane_origin
                              + cu * self.plane_x_axis
                              + cv * self.plane_y_axis)
                ax2 = _gAx2(
                    gp_Pnt(*c_on_plane.tolist()),
                    gp_Dir(float(self.plane_normal[0]),
                           float(self.plane_normal[1]),
                           float(self.plane_normal[2])),
                    gp_Dir(float(self.plane_x_axis[0]),
                           float(self.plane_x_axis[1]),
                           float(self.plane_x_axis[2])),
                )
                new_circ = _GCircle(ax2, circ.Radius())
                u1, u2 = adp.FirstParameter(), adp.LastParameter()
                if abs(u2 - u1 - 2 * np.pi) < 1e-9:
                    return BRepBuilderAPI_MakeEdge(new_circ).Edge()
                else:
                    return BRepBuilderAPI_MakeEdge(new_circ, u1, u2).Edge()
            return None  # unsupported type — caller falls back to polyline

        def _occ_edges_to_face(occ_edges, uv_fallback):
            """
            Build a planar face from exact OCCT edges projected onto the sketch plane.
            Falls back to polyline wire if anything goes wrong.
            """
            try:
                wm = BRepBuilderAPI_MakeWire()
                all_projected = True
                for edge in occ_edges:
                    projected = _project_edge_to_plane(edge)
                    if projected is None:
                        all_projected = False
                        break
                    wm.Add(projected)
                if not all_projected or not wm.IsDone():
                    raise RuntimeError(f"projection/wire failed")
                wire = wm.Wire()
                fm = BRepBuilderAPI_MakeFace(_geom_plane, wire)
                if fm.IsDone():
                    return fm.Face()
                raise RuntimeError(f"MakeFace failed: {fm.Error()}")
            except Exception as ex:
                print(f"[Sketch] build_faces: occ_edges face error — {ex}")
            # Fallback to polyline wire
            w = _uv_to_wire(uv_fallback)
            if w is None:
                return None
            fm = BRepBuilderAPI_MakeFace(_geom_plane, w)
            return fm.Face() if fm.IsDone() else None

        # ── Collect loops: exact edges for ReferenceEntity, polylines for drawn segments ──

        loop_faces = []
        loop_uvs   = []

        for ref in self.entities:
            if not isinstance(ref, ReferenceEntity) or len(ref.points) < 3:
                continue
            pts_close = np.linalg.norm(ref.points[-1] - ref.points[0]) < tol
            edge_closed = False
            if not pts_close and ref.occ_edges:
                try:
                    from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
                    for _e in ref.occ_edges:
                        _a = _BAC(_e)
                        if _a.IsClosed() or _a.IsPeriodic():
                            edge_closed = True
                            break
                except Exception:
                    pass
            if not pts_close and not edge_closed:
                continue
            raw = ref.points[:-1] if pts_close else ref.points
            uv = [(float(p[0]), float(p[1])) for p in raw]
            if len(uv) < 3:
                continue
            valid_occ_edges = [e for e in (ref.occ_edges or []) if e is not None]
            if valid_occ_edges:
                f = _occ_edges_to_face(valid_occ_edges, uv)
                if f is not None:
                    loop_faces.append(f)
                    loop_uvs.append(uv)
                    continue
            # Fallback: polyline wire
            w = _uv_to_wire(uv)
            if w is not None:
                fm = BRepBuilderAPI_MakeFace(_geom_plane, w)
                if fm.IsDone():
                    loop_faces.append(fm.Face())
                    loop_uvs.append(uv)
                else:
                    print(f"[Sketch] build_faces: MakeFace error {fm.Error()}")

        for uv in self._collect_uv_loops_drawn(tol):
            w = _uv_to_wire(uv)
            if w is not None:
                fm = BRepBuilderAPI_MakeFace(_geom_plane, w)
                if fm.IsDone():
                    loop_faces.append(fm.Face())
                    loop_uvs.append(uv)
                else:
                    print(f"[Sketch] build_faces: MakeFace error {fm.Error()}")

        if not loop_faces:
            return [], []

        if len(loop_faces) == 1:
            # Single loop — no splitting needed
            faces   = [Face(loop_faces[0])]
            regions = [(loop_uvs[0], [])]
            return faces, regions

        # ── Splitter: cookie-cutter all loops simultaneously ──────────────
        splitter = BRepAlgoAPI_Splitter()
        shape_list = TopTools_ListOfShape()
        for f in loop_faces:
            shape_list.Append(f)
        splitter.SetArguments(shape_list)
        splitter.SetTools(shape_list)
        splitter.Build()

        if not splitter.IsDone():
            print("[Sketch] build_faces: Splitter failed")
            return [], []

        # ── Extract output faces and per-face UV wire polygons ────────────
        def _wire_uv_pts(wire):
            """Sample UV points along a wire (handles polyline and curved edges)."""
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC2
            from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
            from OCP.TopExp import TopExp_Explorer as _TExp
            from OCP.TopAbs import TopAbs_EDGE as _EDGE
            from OCP.TopoDS import TopoDS as _TDS
            pts = []
            we = BRepTools_WireExplorer(wire)
            while we.More():
                edge = we.Current()
                adp = _BAC2(edge)
                # 32 points per edge is plenty for area comparison; skip last to avoid dup
                sampler = GCPnts_QuasiUniformAbscissa()
                sampler.Initialize(adp, 33)
                npts = sampler.NbPoints()
                for j in range(1, npts):
                    p3d = adp.Value(sampler.Parameter(j))
                    world = np.array([p3d.X(), p3d.Y(), p3d.Z()])
                    delta = world - self.plane_origin
                    u = float(np.dot(delta, self.plane_x_axis))
                    v_coord = float(np.dot(delta, self.plane_y_axis))
                    pts.append((u, v_coord))
                we.Next()
            return pts

        def _uv_area(uvs):
            a = 0.0
            for k in range(len(uvs)):
                x0, y0 = uvs[k]; x1, y1 = uvs[(k + 1) % len(uvs)]
                a += x0 * y1 - x1 * y0
            return abs(a) * 0.5

        faces   = []
        regions = []
        exp = TopExp_Explorer(splitter.Shape(), TopAbs_FACE)
        while exp.More():
            try:
                occ_face = TopoDS.Face_s(exp.Current())
                # Collect all wires on this face
                wire_uvs = []
                wexp = TopExp_Explorer(occ_face, TopAbs_WIRE)
                while wexp.More():
                    wire = TopoDS.Wire_s(wexp.Current())
                    uvs = _wire_uv_pts(wire)
                    if len(uvs) >= 3:
                        wire_uvs.append(uvs)
                    wexp.Next()
                if not wire_uvs:
                    exp.Next()
                    continue
                # Largest wire = outer boundary; rest = holes
                wire_uvs.sort(key=_uv_area, reverse=True)
                outer_uvs = wire_uvs[0]
                hole_uvs  = wire_uvs[1:]
                faces.append(Face(occ_face))
                regions.append((outer_uvs, hole_uvs))
            except Exception as ex:
                print(f"[Sketch] build_faces: face extraction failed — {ex}")
            exp.Next()

        return faces, regions


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
