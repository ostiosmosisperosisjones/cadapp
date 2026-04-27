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
    NONE      = auto()
    LINE      = auto()
    CIRCLE    = auto()
    ARC       = auto()
    ARC3      = auto()   # 3-point arc
    TRIM      = auto()
    DIVIDE    = auto()
    POINT     = auto()
    OFFSET    = auto()
    FILLET    = auto()
    DIMENSION = auto()
    GEOMETRIC = auto()


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

class LineEntity:
    """A 2-D line segment in sketch (u, v) coordinates (world mm)."""
    def __init__(self, p0: tuple[float, float], p1: tuple[float, float]):
        self.p0 = np.array(p0, dtype=np.float64)
        self.p1 = np.array(p1, dtype=np.float64)
        # Snap metadata: (entity_idx, SnapType) if endpoint was snapped to a
        # non-endpoint position (midpoint, center, nearest, tangent). Used by
        # the constraint solver to emit coincidence constraints.
        self.p0_snap: tuple[int, object] | None = None
        self.p1_snap: tuple[int, object] | None = None


class PointEntity:
    """A 2-D construction point in sketch (u, v) coordinates."""
    def __init__(self, pos: tuple[float, float]):
        self.pos = np.array(pos, dtype=np.float64)

    @property
    def p0(self) -> np.ndarray:
        return self.pos

    @property
    def p1(self) -> np.ndarray:
        return self.pos


class ArcEntity:
    """
    A 2-D circular arc in sketch (u, v) coordinates (world mm).

    Stored as center + radius + angles so the solver can own these params later.
    Angles are in radians, CCW from the U-axis.  start_angle < end_angle always
    (the arc sweeps CCW from start_angle to end_angle).
    """
    def __init__(self, center: tuple[float, float], radius: float,
                 start_angle: float, end_angle: float):
        self.center      = np.array(center, dtype=np.float64)
        self.radius      = float(radius)
        self.start_angle = float(start_angle)
        self.end_angle   = float(end_angle)

    @property
    def p0(self) -> np.ndarray:
        """Start point of the arc."""
        return self.center + self.radius * np.array(
            [np.cos(self.start_angle), np.sin(self.start_angle)])

    @property
    def p1(self) -> np.ndarray:
        """End point of the arc."""
        return self.center + self.radius * np.array(
            [np.cos(self.end_angle), np.sin(self.end_angle)])

    @property
    def mid_point(self) -> np.ndarray:
        """True mid-arc point."""
        mid = (self.start_angle + self.end_angle) * 0.5
        return self.center + self.radius * np.array([np.cos(mid), np.sin(mid)])

    def tessellate(self, n: int = 64) -> list[np.ndarray]:
        """Return n+1 points evenly distributed along the arc."""
        angles = np.linspace(self.start_angle, self.end_angle, n + 1)
        return [self.center + self.radius * np.array([np.cos(a), np.sin(a)])
                for a in angles]


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
        self.constraints: list = []

        # Per-sketch undo stack — each entry is a deep-copy of (entities, constraints)
        # taken before a mutation.  Ctrl+Z pops and restores.
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
        self.snap.anchor_pt = None

    # ------------------------------------------------------------------
    # Per-sketch undo
    # ------------------------------------------------------------------

    def push_undo_snapshot(self):
        """Save entities + constraints before a mutation so Ctrl+Z can restore."""
        import copy
        self._entity_snapshots.append(
            (copy.deepcopy(self.entities), copy.deepcopy(self.constraints))
        )

    def undo_entity(self) -> bool:
        """Restore the most recent snapshot. Returns True if anything was undone."""
        if not self._entity_snapshots:
            return False
        snapshot = self._entity_snapshots.pop()
        if isinstance(snapshot, tuple):
            self.entities, self.constraints = snapshot
        else:
            self.entities = snapshot  # backwards compat with old single-list snapshots
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
        result = self._active_tool.handle_click(snap_result, self)
        self.snap.consume_declared()
        return result

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
# SketchConstraint
# ---------------------------------------------------------------------------

class SketchConstraint:
    """
    One parametric constraint on sketch entities.

    type    : 'distance' | 'parallel' | 'perpendicular' | 'horizontal' | 'vertical'
    indices : tuple of entity indices the constraint applies to
    value   : numeric value (mm for distance; ignored for geometric constraints)
    label_offset : perpendicular offset of the dimension label in mm (None = auto)
    """
    def __init__(self, type: str, indices: tuple, value: float,
                 label_offset: float | None = None):
        self.type         = type
        self.indices      = tuple(indices)
        self.value        = float(value)
        self.label_offset: float | None = label_offset


# ---------------------------------------------------------------------------
# SketchEntry  — immutable snapshot of a committed sketch
# ---------------------------------------------------------------------------

class SketchEntry:
    """
    Immutable record of a committed sketch stored on a HistoryEntry.
    shape_before == shape_after (sketches don't mutate geometry).
    """

    def __init__(self, plane_origin, plane_x_axis, plane_y_axis, plane_normal,
                 entities, body_id, face_idx, visible=True, plane_source=None,
                 constraints=None, undo_snapshots=None):
        self.plane_origin    = np.array(plane_origin, dtype=np.float64)
        self.plane_x_axis    = np.array(plane_x_axis, dtype=np.float64)
        self.plane_y_axis    = np.array(plane_y_axis, dtype=np.float64)
        self.plane_normal    = np.array(plane_normal, dtype=np.float64)
        self.entities        = entities
        self.body_id         = body_id
        self.face_idx        = face_idx
        self.visible         = visible
        self.plane_source    = plane_source   # SketchPlaneSource | None
        self.constraints     = constraints or []
        # Per-sketch undo history — list of (entities, constraints) snapshots.
        # Populated from SketchMode on commit; restored on re-entry so the user
        # can undo individual sketch operations after re-opening the sketch.
        self.undo_snapshots  = undo_snapshots or []

    @classmethod
    def from_sketch_mode(cls, sketch: SketchMode) -> SketchEntry:
        return cls(
            plane_origin    = sketch.plane.origin.copy(),
            plane_x_axis    = sketch.plane.x_axis.copy(),
            plane_y_axis    = sketch.plane.y_axis.copy(),
            plane_normal    = sketch.plane.normal.copy(),
            entities        = copy.deepcopy(sketch.entities),
            body_id         = sketch.body_id,
            face_idx        = sketch.face_idx,
            plane_source    = sketch.plane_source,
            constraints     = copy.deepcopy(getattr(sketch, 'constraints', [])),
            undo_snapshots  = copy.deepcopy(sketch._entity_snapshots),
        )

    # ------------------------------------------------------------------
    # Constraint solver
    # ------------------------------------------------------------------

    def compute_constraint_status(self) -> tuple[str, int]:
        """
        Return (status, user_dof) without touching entity coordinates.

        Builds a fresh solver system with all constraints but zero dragged
        points, solves, and reads dof().  In SolveSpace a fully shape-
        constrained 2D sketch always has exactly 3 residual DOFs (rigid-body
        translation × 2 + rotation × 1), so:

            user_dof = dof() - 3

        status values: 'none' | 'under' | 'fully' | 'over'
        user_dof      : meaningful DOFs remaining (0 = fully constrained).
                        -1 when status is 'none' or 'over'.
        """
        if not self.constraints:
            return ('none', -1)

        lines = [e for e in self.entities if isinstance(e, LineEntity)]
        if not lines:
            return ('none', -1)

        try:
            from python_solvespace import SolverSystem
        except ImportError:
            return ('none', -1)

        sys = SolverSystem()
        wp  = sys.create_2d_base()

        # Deduplicate coincident endpoints so the solver sees shared points.
        tol = 1e-3
        canon_map: dict[tuple, tuple] = {}
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                key = (i, which)
                pt  = ent.p0 if which == 'p0' else ent.p1
                found = None
                for k, ck in canon_map.items():
                    if ck != k:
                        continue
                    other = self.entities[k[0]]
                    op = other.p0 if k[1] == 'p0' else other.p1
                    if np.linalg.norm(pt - op) < tol:
                        found = k
                        break
                canon_map[key] = found if found is not None else key

        slvs_pts: dict[tuple, object] = {}
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                ck = canon_map[(i, which)]
                if ck not in slvs_pts:
                    pt = ent.p0 if which == 'p0' else ent.p1
                    slvs_pts[ck] = sys.add_point_2d(float(pt[0]), float(pt[1]), wp)

        def _pt(ei, which):
            return slvs_pts[canon_map[(ei, which)]]

        slvs_lines: dict[int, object] = {}
        for i, ent in enumerate(self.entities):
            if isinstance(ent, LineEntity):
                slvs_lines[i] = sys.add_line_2d(_pt(i, 'p0'), _pt(i, 'p1'), wp)

        # Apply constraints — no dragged points at all.
        for con in self.constraints:
            if con.type == 'distance':
                i = con.indices[0]
                if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                    sys.distance(_pt(i, 'p0'), _pt(i, 'p1'), con.value, wp)
            elif con.type == 'parallel':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    sys.parallel(slvs_lines[ref], slvs_lines[mov], wp)
            elif con.type == 'perpendicular':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    sys.perpendicular(slvs_lines[ref], slvs_lines[mov], wp)
            elif con.type == 'horizontal':
                i = con.indices[0]
                if i in slvs_lines:
                    sys.horizontal(slvs_lines[i], wp)
            elif con.type == 'vertical':
                i = con.indices[0]
                if i in slvs_lines:
                    sys.vertical(slvs_lines[i], wp)

        result = sys.solve()
        if result != 0:
            return ('over', -1)

        # SolveSpace dof() includes the workplane overhead (6) plus the 2D
        # rigid-body residual (translation×2 + rotation×1 = 3).  Baseline = 9.
        user_dof = sys.dof() - 9
        if user_dof <= 0:
            return ('fully', 0)
        return ('under', user_dof)

    def _add_snap_coincidences(self, sys, wp, slvs_pts, slvs_lines, canon_map):
        """Add geometric constraints for non-endpoint snap anchors.

        MIDPOINT snap on a line  → sys.midpoint(ep, line)      — parametric, no pinning
        NEAREST snap on a line   → sys.coincident(ep, line)    — point-on-line, no pinning
        CENTER/NEAREST/TANGENT on an arc → pin endpoint to stored coordinate (arcs are
            not yet registered as solver entities, so we fall back to a fixed point)
        """
        from cad.sketch_tools.snap import SnapType

        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which, snap_meta in (('p0', getattr(ent, 'p0_snap', None)),
                                     ('p1', getattr(ent, 'p1_snap', None))):
                if snap_meta is None:
                    continue
                src_idx, snap_type = snap_meta
                if src_idx >= len(self.entities):
                    continue
                src = self.entities[src_idx]
                ep_sp = slvs_pts.get(canon_map.get((i, which)))
                if ep_sp is None:
                    continue

                if snap_type == SnapType.MIDPOINT and isinstance(src, LineEntity):
                    if src_idx in slvs_lines:
                        sys.midpoint(ep_sp, slvs_lines[src_idx], wp)

                elif snap_type == SnapType.NEAREST and isinstance(src, LineEntity):
                    if src_idx in slvs_lines:
                        sys.coincident(ep_sp, slvs_lines[src_idx], wp)

                elif snap_type in (SnapType.CENTER, SnapType.NEAREST,
                                   SnapType.TANGENT, SnapType.MIDPOINT):
                    # Arc/reference entity — pin to stored snapped coordinate.
                    ref_uv = (ent.p0 if which == 'p0' else ent.p1).copy()
                    ref_sp = sys.add_point_2d(float(ref_uv[0]), float(ref_uv[1]), wp)
                    sys.dragged(ref_sp, wp)
                    sys.distance(ep_sp, ref_sp, 0.0, wp)

    def apply_last_constraint(self) -> bool:
        """Solve all constraints with no pinning. Alias for apply_all_constraints."""
        return self.apply_all_constraints()

    def apply_all_constraints(self) -> bool:
        """Solve all constraints simultaneously with no pinning."""
        if not self.constraints:
            return True
        return self.apply_constraints(free_for_last=False, no_pin=True)

    def _free_cks_for_constraint(self, con, canon_map) -> set:
        """Return canonical keys that must be FREE when applying constraint `con`."""
        free = set()
        if con.type == 'distance':
            i = con.indices[0]
            if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                free.add(canon_map[(i, 'p1')])
        elif con.type in ('parallel', 'perpendicular'):
            mov = con.indices[1]
            if mov < len(self.entities) and isinstance(self.entities[mov], LineEntity):
                free.add(canon_map[(mov, 'p0')])
                free.add(canon_map[(mov, 'p1')])
        elif con.type in ('horizontal', 'vertical'):
            i = con.indices[0]
            if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                free.add(canon_map[(i, 'p1')])
        return free

    def apply_constraints(self, free_for_last: bool = False,
                          no_pin: bool = False) -> bool:
        """
        Run the SolveSpace solver over all entities + constraints.
        Updates entity coordinates in-place.  Returns True on success.

        free_for_last: pin everything except what the last constraint needs to move.
        no_pin: skip all pinning — solver finds minimum motion (use when editing).
        """
        if not self.constraints:
            return True
        try:
            from python_solvespace import SolverSystem
        except ImportError:
            return False

        sys = SolverSystem()
        wp  = sys.create_2d_base()

        tol = 1e-3
        canon_map: dict[tuple, tuple] = {}
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                key = (i, which)
                pt  = ent.p0 if which == 'p0' else ent.p1
                found = None
                for k in canon_map:
                    if canon_map[k] != k:
                        continue
                    other = self.entities[k[0]]
                    op = other.p0 if k[1] == 'p0' else other.p1
                    if np.linalg.norm(pt - op) < tol:
                        found = k
                        break
                canon_map[key] = found if found is not None else key

        slvs_pts: dict[tuple, object] = {}
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                ck = canon_map[(i, which)]
                if ck not in slvs_pts:
                    pt = ent.p0 if which == 'p0' else ent.p1
                    slvs_pts[ck] = sys.add_point_2d(float(pt[0]), float(pt[1]), wp)

        def _pt(ent_idx, which):
            return slvs_pts[canon_map[(ent_idx, which)]]

        slvs_lines: dict[int, object] = {}
        for i, ent in enumerate(self.entities):
            if isinstance(ent, LineEntity):
                slvs_lines[i] = sys.add_line_2d(_pt(i, 'p0'), _pt(i, 'p1'), wp)

        self._add_snap_coincidences(sys, wp, slvs_pts, slvs_lines, canon_map)

        if free_for_last and self.constraints:
            free_cks = self._free_cks_for_constraint(self.constraints[-1], canon_map)

            # Also free p1 of any previously distance-constrained line — dragging
            # it would conflict with the distance constraint itself.
            for con in self.constraints[:-1]:
                if con.type == 'distance':
                    i = con.indices[0]
                    if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                        free_cks.add(canon_map[(i, 'p1')])

            # Propagate: points connected by geometric constraints to a free
            # point must also be free.
            changed = True
            while changed:
                changed = False
                for con in self.constraints[:-1]:
                    if con.type in ('parallel', 'perpendicular'):
                        ref, mov = con.indices
                        involved = {canon_map.get((ref, 'p0')), canon_map.get((ref, 'p1')),
                                    canon_map.get((mov, 'p0')), canon_map.get((mov, 'p1'))} - {None}
                        if involved & free_cks:
                            before = len(free_cks)
                            free_cks |= involved
                            changed = changed or len(free_cks) > before
                    elif con.type in ('horizontal', 'vertical'):
                        i = con.indices[0]
                        p0 = canon_map.get((i, 'p0'))
                        p1 = canon_map.get((i, 'p1'))
                        if p0 in free_cks or p1 in free_cks:
                            before = len(free_cks)
                            free_cks |= {p0, p1} - {None}
                            changed = changed or len(free_cks) > before

            for ck, sp in slvs_pts.items():
                if ck not in free_cks:
                    sys.dragged(sp, wp)
        elif not no_pin:
            # Pin p0 of distance-constrained lines to prevent positional drift.
            for con in self.constraints:
                if con.type == 'distance':
                    i = con.indices[0]
                    if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                        sys.dragged(slvs_pts[canon_map[(i, 'p0')]], wp)

        for con in self.constraints:
            if con.type == 'distance':
                i = con.indices[0]
                if not isinstance(self.entities[i], LineEntity): continue
                sys.distance(_pt(i, 'p0'), _pt(i, 'p1'), con.value, wp)
            elif con.type == 'parallel':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    sys.parallel(slvs_lines[ref], slvs_lines[mov], wp)
            elif con.type == 'perpendicular':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    sys.perpendicular(slvs_lines[ref], slvs_lines[mov], wp)
            elif con.type == 'horizontal':
                i = con.indices[0]
                if i in slvs_lines: sys.horizontal(slvs_lines[i], wp)
            elif con.type == 'vertical':
                i = con.indices[0]
                if i in slvs_lines: sys.vertical(slvs_lines[i], wp)

        if sys.solve() != 0:
            return False

        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                ck = canon_map[(i, which)]
                sp = slvs_pts.get(ck)
                if sp is None:
                    continue
                uv = sys.params(sp.params)
                if which == 'p0':
                    ent.p0 = np.array([uv[0], uv[1]], dtype=np.float64)
                else:
                    ent.p1 = np.array([uv[0], uv[1]], dtype=np.float64)

        return True

    # ------------------------------------------------------------------
    # Derived geometry — built on demand, never cached
    # ------------------------------------------------------------------

    def line_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(e.p0.copy(), e.p1.copy())
                for e in self.entities if isinstance(e, LineEntity)]

    def arc_entities(self) -> list:
        return [e for e in self.entities if isinstance(e, ArcEntity)]

    def reference_chains(self) -> list[list[tuple[float, float]]]:
        return [[(float(p[0]), float(p[1])) for p in e.points]
                for e in self.entities
                if isinstance(e, ReferenceEntity) and len(e.points) >= 2]

    def closed_loops(self, tol: float = 1e-3) -> list[list[tuple[float, float]]]:
        """All closed chains from LineEntity/ArcEntity segments and ReferenceEntity polylines."""
        closed = []
        for _chain, uv_pts in self._collect_drawn_loops_with_arcs(tol=tol):
            closed.append(uv_pts)
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

        for _chain, uv_pts in self._collect_drawn_loops_with_arcs(tol=tol):
            if len(uv_pts) >= 3:
                uv_loops.append(uv_pts)
        return uv_loops

    def _collect_drawn_loops_with_arcs(self, tol: float = 1e-3):
        """
        Chain LineEntity and ArcEntity into closed loops, preserving entity refs.
        Returns list of (seg_loop, uv_pts) where:
          seg_loop : [(entity, p0_uv, p1_uv), ...]  in chained order
          uv_pts   : tessellated [(u,v), ...] for UV area tests

        Arcs are pre-split at any line endpoint that lands on their circumference
        (tangent/nearest snaps), so the chainer sees proper endpoint connections.
        """
        import math as _math

        # Collect all line endpoints that land on each arc's circumference.
        # These become split points that break the arc into sub-arcs.
        def _angle_on_arc(pt, arc):
            """Return the angle of pt on arc if within tol, else None."""
            d = pt - arc.center
            if abs(float(np.linalg.norm(d)) - arc.radius) > tol:
                return None
            angle = float(_math.atan2(float(d[1]), float(d[0])))
            # Normalise into [start_angle, end_angle]
            span = arc.end_angle - arc.start_angle
            rel  = (angle - arc.start_angle) % (2 * _math.pi)
            if rel > span + tol:
                return None
            return arc.start_angle + rel

        # Build (entity, p0, p1) list, splitting arcs where lines touch them.
        segs = []
        for e in self.entities:
            if isinstance(e, LineEntity):
                segs.append((e, e.p0.copy(), e.p1.copy()))

        # For each arc, collect all split angles from line endpoints
        for arc in self.entities:
            if not isinstance(arc, ArcEntity):
                continue
            split_angles = set()
            for e in self.entities:
                if not isinstance(e, LineEntity):
                    continue
                for pt in (e.p0, e.p1):
                    ang = _angle_on_arc(pt, arc)
                    if ang is not None:
                        # Avoid duplicating the arc's own endpoints
                        if (abs(ang - arc.start_angle) > tol and
                                abs(ang - arc.end_angle) > tol):
                            split_angles.add(ang)

            # Build ordered list of boundary angles: start → splits → end
            angles = sorted([arc.start_angle] + list(split_angles) + [arc.end_angle])

            for k in range(len(angles) - 1):
                a0 = angles[k]
                a1 = angles[k + 1]
                if abs(a1 - a0) < tol:
                    continue
                p0 = arc.center + arc.radius * np.array([_math.cos(a0), _math.sin(a0)])
                p1 = arc.center + arc.radius * np.array([_math.cos(a1), _math.sin(a1)])
                segs.append((arc, p0, p1))

        if not segs:
            return []

        remaining = list(segs)
        loops = []

        while remaining:
            ent0, a0, b0 = remaining.pop(0)
            chain = [(ent0, a0, b0)]
            changed = True
            while changed:
                changed = False
                tail = chain[-1][2]
                head = chain[0][1]
                for i, (ent, a, b) in enumerate(remaining):
                    if np.linalg.norm(a - tail) < tol:
                        chain.append((ent, a, b)); remaining.pop(i); changed = True; break
                    if np.linalg.norm(b - tail) < tol:
                        chain.append((ent, b, a) if isinstance(ent, LineEntity)
                                     else (ent, a, b)); remaining.pop(i); changed = True; break
                    if np.linalg.norm(b - head) < tol:
                        chain.insert(0, (ent, a, b)); remaining.pop(i); changed = True; break
                    if np.linalg.norm(a - head) < tol:
                        chain.insert(0, (ent, b, a) if isinstance(ent, LineEntity)
                                     else (ent, a, b)); remaining.pop(i); changed = True; break

            # Only keep closed loops
            if np.linalg.norm(chain[-1][2] - chain[0][1]) > tol:
                continue

            # Tessellate for UV area/containment tests
            uv_pts = []
            for ent, ep0, ep1 in chain:
                if isinstance(ent, ArcEntity):
                    # Tessellate the specific sub-arc span (ep0→ep1 angles)
                    a0 = float(_math.atan2(float(ep0[1] - ent.center[1]),
                                           float(ep0[0] - ent.center[0])))
                    a1 = float(_math.atan2(float(ep1[1] - ent.center[1]),
                                           float(ep1[0] - ent.center[0])))
                    span = (a1 - a0) % (2 * _math.pi)
                    if span < tol:
                        span = 2 * _math.pi
                    angles = np.linspace(a0, a0 + span, 17)
                    uv_pts.extend(
                        (float(ent.center[0] + ent.radius * _math.cos(ang)),
                         float(ent.center[1] + ent.radius * _math.sin(ang)))
                        for ang in angles[:-1]
                    )
                else:
                    uv_pts.append((float(ep0[0]), float(ep0[1])))
            if len(uv_pts) >= 3:
                loops.append((chain, uv_pts))

        return loops

    def _collect_uv_loops_drawn(self, tol: float = 1e-3):
        """Collect closed UV polygon loops from drawn LineEntity/ArcEntity segments."""
        uv_loops = []
        segs = self.line_segments()
        # Add arc endpoint pairs so the chainer can close loops containing arcs
        for arc in self.arc_entities():
            segs.append((arc.p0.copy(), arc.p1.copy()))
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
        Build one planar Face per enclosed region in the sketch.

        Strategy: feed every drawn edge individually as a wire tool into
        BRepAlgoAPI_Splitter against a large planar face.  OCCT figures out
        all topology — no Python loop detection needed.  ReferenceEntity closed
        loops are also added as wire tools.  The largest output face (the
        unbounded remainder of the big plane) is discarded; all others are kept.

        Returns (faces, regions) where regions[k] = (outer_uvs, [hole_uvs])
        parallel to faces[k].
        """
        import math as _math
        from build123d import Face
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                        BRepBuilderAPI_MakeWire,
                                        BRepBuilderAPI_MakeEdge)
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
        from OCP.BRepTools import BRepTools_WireExplorer
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.Geom import Geom_Circle as _GCircle
        from OCP.gp import gp_Pln, gp_Ax3, gp_Ax2, gp_Pnt, gp_Dir
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCP.TopTools import TopTools_ListOfShape
        from OCP.TopoDS import TopoDS

        plane_pln = gp_Pln(gp_Ax3(
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

        # Big planar face — large enough to contain all sketch geometry
        _BIG = 1e5
        big_face = BRepBuilderAPI_MakeFace(plane_pln, -_BIG, _BIG, -_BIG, _BIG).Face()

        def _uv3d(u, v):
            return gp_Pnt(*(self.plane_origin
                            + float(u) * self.plane_x_axis
                            + float(v) * self.plane_y_axis).tolist())

        def _single_edge_wire(edge):
            wm = BRepBuilderAPI_MakeWire()
            wm.Add(edge)
            return wm.Wire() if wm.IsDone() else None

        tools_list = TopTools_ListOfShape()

        # ── Drawn LineEntity edges — each edge is its own wire tool ──────
        for ent in self.entities:
            if isinstance(ent, LineEntity):
                e = BRepBuilderAPI_MakeEdge(_uv3d(ent.p0[0], ent.p0[1]),
                                            _uv3d(ent.p1[0], ent.p1[1])).Edge()
                w = _single_edge_wire(e)
                if w is not None:
                    tools_list.Append(w)

            elif isinstance(ent, ArcEntity):
                try:
                    cx, cy = float(ent.center[0]), float(ent.center[1])
                    c3d = (self.plane_origin
                           + cx * self.plane_x_axis
                           + cy * self.plane_y_axis)
                    ax2 = gp_Ax2(
                        gp_Pnt(*c3d.tolist()),
                        gp_Dir(float(self.plane_normal[0]),
                               float(self.plane_normal[1]),
                               float(self.plane_normal[2])),
                        gp_Dir(float(self.plane_x_axis[0]),
                               float(self.plane_x_axis[1]),
                               float(self.plane_x_axis[2])),
                    )
                    geom_circ = _GCircle(ax2, ent.radius)
                    span = ent.end_angle - ent.start_angle
                    if abs(span - 2 * _math.pi) < 1e-9 or abs(span) < 1e-9:
                        arc_edge = BRepBuilderAPI_MakeEdge(geom_circ).Edge()
                    else:
                        arc_edge = BRepBuilderAPI_MakeEdge(
                            geom_circ, ent.start_angle, ent.end_angle).Edge()
                    w = _single_edge_wire(arc_edge)
                    if w is not None:
                        tools_list.Append(w)
                except Exception as ex:
                    print(f"[Sketch] build_faces: arc edge error — {ex}")

        # ── ReferenceEntity closed loops — add as wire tools ─────────────
        def _project_ref_edge(edge):
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
            from OCP.GeomAbs import GeomAbs_Circle
            adp = _BAC(edge)
            if adp.GetType() == GeomAbs_Circle:
                circ = adp.Circle()
                # Check if circle is viewed edge-on (its normal is perpendicular
                # to the sketch normal) — if so, project as a line segment.
                circ_ax = circ.Axis().Direction()
                circ_normal = np.array([circ_ax.X(), circ_ax.Y(), circ_ax.Z()])
                dot_normals = abs(float(np.dot(circ_normal, self.plane_normal)))
                if dot_normals < 1e-3:
                    # Edge-on: the circle projects to a line of length 2*radius.
                    # The axis of the projected line is the cross product of
                    # circ_normal and sketch_normal, normalised to sketch UV.
                    line_dir_3d = np.cross(circ_normal, self.plane_normal)
                    ld_len = float(np.linalg.norm(line_dir_3d))
                    if ld_len < 1e-9:
                        return None
                    line_dir_3d /= ld_len
                    c3d = circ.Location()
                    cw  = np.array([c3d.X(), c3d.Y(), c3d.Z()])
                    delta = cw - self.plane_origin
                    cu = float(np.dot(delta, self.plane_x_axis))
                    cv = float(np.dot(delta, self.plane_y_axis))
                    # line_dir in UV space
                    ldu = float(np.dot(line_dir_3d, self.plane_x_axis))
                    ldv = float(np.dot(line_dir_3d, self.plane_y_axis))
                    r = circ.Radius()
                    p0 = _uv3d(cu - r * ldu, cv - r * ldv)
                    p1 = _uv3d(cu + r * ldu, cv + r * ldv)
                    em = BRepBuilderAPI_MakeEdge(p0, p1)
                    return em.Edge() if em.IsDone() else None

                c3d  = circ.Location()
                cw   = np.array([c3d.X(), c3d.Y(), c3d.Z()])
                delta = cw - self.plane_origin
                cu = float(np.dot(delta, self.plane_x_axis))
                cv = float(np.dot(delta, self.plane_y_axis))
                c_on = (self.plane_origin
                        + cu * self.plane_x_axis
                        + cv * self.plane_y_axis)
                ax2 = gp_Ax2(
                    gp_Pnt(*c_on.tolist()),
                    gp_Dir(float(self.plane_normal[0]),
                           float(self.plane_normal[1]),
                           float(self.plane_normal[2])),
                    gp_Dir(float(self.plane_x_axis[0]),
                           float(self.plane_x_axis[1]),
                           float(self.plane_x_axis[2])),
                )
                new_circ = _GCircle(ax2, circ.Radius())
                u1, u2 = adp.FirstParameter(), adp.LastParameter()
                if abs(u2 - u1 - 2 * _math.pi) < 1e-9:
                    return BRepBuilderAPI_MakeEdge(new_circ).Edge()
                return BRepBuilderAPI_MakeEdge(new_circ, u1, u2).Edge()
            return None

        for ref in self.entities:
            if not isinstance(ref, ReferenceEntity):
                continue
            valid_occ = [e for e in (ref.occ_edges or []) if e is not None]
            if valid_occ:
                for occ_edge in valid_occ:
                    projected = _project_ref_edge(occ_edge)
                    if projected is not None:
                        w = _single_edge_wire(projected)
                        if w is not None:
                            tools_list.Append(w)
                    else:
                        # Polyline fallback for non-circle reference edges
                        try:
                            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC2
                            from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
                            adp = _BAC2(occ_edge)
                            sampler = GCPnts_QuasiUniformAbscissa()
                            sampler.Initialize(adp, 33)
                            pts_3d = []
                            for j in range(1, sampler.NbPoints() + 1):
                                p = adp.Value(sampler.Parameter(j))
                                pts_3d.append(gp_Pnt(p.X(), p.Y(), p.Z()))
                            for j in range(len(pts_3d) - 1):
                                em = BRepBuilderAPI_MakeEdge(pts_3d[j], pts_3d[j+1])
                                if em.IsDone():
                                    w = _single_edge_wire(em.Edge())
                                    if w is not None:
                                        tools_list.Append(w)
                        except Exception:
                            pass
            elif len(ref.points) >= 2:
                # Polyline from tessellated points
                pts = ref.points
                for j in range(len(pts) - 1):
                    em = BRepBuilderAPI_MakeEdge(_uv3d(pts[j][0], pts[j][1]),
                                                 _uv3d(pts[j+1][0], pts[j+1][1]))
                    if em.IsDone():
                        w = _single_edge_wire(em.Edge())
                        if w is not None:
                            tools_list.Append(w)

        if tools_list.IsEmpty():
            return [], []

        # ── Splitter: big face cut by all edge-wires ──────────────────────
        args_list = TopTools_ListOfShape()
        args_list.Append(big_face)
        splitter = BRepAlgoAPI_Splitter()
        splitter.SetArguments(args_list)
        splitter.SetTools(tools_list)
        splitter.Build()

        if not splitter.IsDone():
            print("[Sketch] build_faces: Splitter failed")
            return [], []

        # ── Extract faces, discard the unbounded remainder ────────────────
        def _wire_uv_pts(wire):
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC2
            from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
            pts = []
            we = BRepTools_WireExplorer(wire)
            while we.More():
                adp = _BAC2(we.Current())
                sampler = GCPnts_QuasiUniformAbscissa()
                sampler.Initialize(adp, 33)
                for j in range(1, sampler.NbPoints()):
                    p3d = adp.Value(sampler.Parameter(j))
                    world = np.array([p3d.X(), p3d.Y(), p3d.Z()])
                    delta = world - self.plane_origin
                    pts.append((float(np.dot(delta, self.plane_x_axis)),
                                float(np.dot(delta, self.plane_y_axis))))
                we.Next()
            return pts

        def _face_area(occ_face):
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(occ_face, props)
            return props.Mass()

        def _uv_area(uvs):
            a = 0.0
            for k in range(len(uvs)):
                x0, y0 = uvs[k]; x1, y1 = uvs[(k + 1) % len(uvs)]
                a += x0 * y1 - x1 * y0
            return abs(a) * 0.5

        # Collect all output faces with their areas
        raw_faces = []
        exp = TopExp_Explorer(splitter.Shape(), TopAbs_FACE)
        while exp.More():
            try:
                occ_face = TopoDS.Face_s(exp.Current())
                raw_faces.append((occ_face, _face_area(occ_face)))
            except Exception as ex:
                print(f"[Sketch] build_faces: face extraction error — {ex}")
            exp.Next()

        if not raw_faces:
            return [], []

        # Discard the largest face (unbounded big-plane remainder)
        raw_faces.sort(key=lambda x: x[1], reverse=True)
        bounded_faces = raw_faces[1:]  # drop the biggest

        faces   = []
        regions = []
        for occ_face, _ in bounded_faces:
            try:
                wire_uvs = []
                wexp = TopExp_Explorer(occ_face, TopAbs_WIRE)
                while wexp.More():
                    uvs = _wire_uv_pts(TopoDS.Wire_s(wexp.Current()))
                    if len(uvs) >= 3:
                        wire_uvs.append(uvs)
                    wexp.Next()
                if not wire_uvs:
                    continue
                wire_uvs.sort(key=_uv_area, reverse=True)
                faces.append(Face(occ_face))
                regions.append((wire_uvs[0], wire_uvs[1:]))
            except Exception as ex:
                print(f"[Sketch] build_faces: region extraction error — {ex}")

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
