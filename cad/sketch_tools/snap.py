"""
cad/sketch_tools/snap.py

SnapEngine — resolves a raw cursor (u,v) to a snapped point.

For ReferenceEntity instances with original OCCT edges, snap queries the
true curve geometry rather than the tessellated point array:
  - Circles / arcs  → exact center, exact endpoint, exact mid-parameter pt
  - Nearest-on-curve → GeomAPI_ProjectPointOnCurve (exact)
  - No false midpoints on tessellation vertices

For LineEntity and ReferenceEntity without occ_edges, uses the polyline.

The sketch plane is passed into resolve() so world↔UV conversion is exact.

Snap priority: forced → endpoint → midpoint → center → nearest → grid → free
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class SnapType(Enum):
    FREE         = auto()
    GRID         = auto()
    ENDPOINT     = auto()
    MIDPOINT     = auto()
    CENTER       = auto()
    NEAREST      = auto()
    INTERSECTION = auto()   # future


@dataclass
class SnapResult:
    point      : np.ndarray   # resolved (u, v) in sketch mm
    type       : SnapType
    cursor_raw : np.ndarray   # original unsnapped cursor

    @property
    def snapped(self) -> bool:
        return self.type != SnapType.FREE


# ---------------------------------------------------------------------------
# SnapEngine
# ---------------------------------------------------------------------------

class SnapEngine:

    def __init__(self, snap_radius_mm: float = 2.0, grid_size: float = 1.0):
        self.snap_radius_mm  = snap_radius_mm
        self.grid_size       = grid_size
        self.forced_type: SnapType | None = None
        self.enabled: set[SnapType] = {
            SnapType.ENDPOINT,
            SnapType.MIDPOINT,
            SnapType.CENTER,
            SnapType.NEAREST,
        }
        # Extra fixed UV snap points (e.g. world origin on a world-plane sketch)
        self.extra_points: list[np.ndarray] = []

    def set_grid_snap(self, active: bool):
        if active:
            self.enabled.add(SnapType.GRID)
        else:
            self.enabled.discard(SnapType.GRID)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def resolve(self, cursor_pt2d: np.ndarray,
                sketch_entities: list,
                plane,                        # SketchPlane
                camera_distance: float = 100.0) -> SnapResult:
        """
        cursor_pt2d     : (2,) raw cursor in sketch UV mm
        sketch_entities : list of LineEntity / ReferenceEntity
        plane           : SketchPlane — needed for world↔UV conversion
        """
        cursor = np.array(cursor_pt2d, dtype=np.float64)
        radius = self.snap_radius_mm

        forced = self.forced_type
        self.forced_type = None

        if forced is not None:
            p = self._try_snap(forced, cursor, sketch_entities, radius, plane)
            if p is not None:
                return SnapResult(point=p, type=forced, cursor_raw=cursor)
            return SnapResult(point=cursor.copy(), type=SnapType.FREE,
                              cursor_raw=cursor)

        for snap_type in [SnapType.ENDPOINT, SnapType.MIDPOINT,
                          SnapType.CENTER, SnapType.NEAREST, SnapType.GRID]:
            if snap_type not in self.enabled:
                continue
            p = self._try_snap(snap_type, cursor, sketch_entities, radius, plane)
            if p is not None:
                return SnapResult(point=p, type=snap_type, cursor_raw=cursor)

        return SnapResult(point=cursor.copy(), type=SnapType.FREE,
                          cursor_raw=cursor)

    def _try_snap(self, snap_type, cursor, entities, radius, plane):
        if snap_type == SnapType.ENDPOINT:
            return self._snap_endpoint(cursor, entities, radius, plane)
        if snap_type == SnapType.MIDPOINT:
            return self._snap_midpoint(cursor, entities, radius, plane)
        if snap_type == SnapType.CENTER:
            return self._snap_center(cursor, entities, radius, plane)
        if snap_type == SnapType.NEAREST:
            return self._snap_nearest(cursor, entities, radius, plane)
        if snap_type == SnapType.GRID:
            return self._snap_grid(cursor, radius)
        return None

    # ------------------------------------------------------------------
    # World ↔ UV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _world_to_uv(world_pt: np.ndarray, plane) -> np.ndarray:
        return plane.project_point(world_pt)

    @staticmethod
    def _uv_to_world(uv: np.ndarray, plane) -> np.ndarray:
        return plane.to_3d(float(uv[0]), float(uv[1]))

    # ------------------------------------------------------------------
    # OCC curve helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _occ_adaptor(occ_edge):
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        return BRepAdaptor_Curve(occ_edge)

    def _curve_endpoints_uv(self, ref, plane) -> list[np.ndarray]:
        """True start/end UV points from OCC curve parameters."""
        results = []
        try:
            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                for t in (a.FirstParameter(), a.LastParameter()):
                    p = a.Value(t)
                    w = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
                    results.append(self._world_to_uv(w, plane))
        except Exception:
            pass
        return results

    def _curve_midpoint_uv(self, ref, plane) -> list[np.ndarray]:
        """True mid-parameter UV points — NOT tessellation midpoints."""
        results = []
        try:
            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                t = (a.FirstParameter() + a.LastParameter()) * 0.5
                p = a.Value(t)
                w = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
                results.append(self._world_to_uv(w, plane))
        except Exception:
            pass
        return results

    def _curve_centers_uv(self, ref, plane) -> list[np.ndarray]:
        """Center UV for circle/arc/ellipse OCC edges."""
        results = []
        try:
            from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse
            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                gt = a.GetType()
                if gt == GeomAbs_Circle:
                    loc = a.Circle().Location()
                elif gt == GeomAbs_Ellipse:
                    loc = a.Ellipse().Location()
                else:
                    continue
                w = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
                results.append(self._world_to_uv(w, plane))
        except Exception:
            pass
        return results

    def _curve_nearest_uv(self, cursor_uv: np.ndarray, ref,
                           plane) -> tuple[np.ndarray | None, float]:
        """
        Nearest point on OCC curve(s) using GeomAPI_ProjectPointOnCurve.
        Returns (uv, distance_in_uv_space) or (None, inf).
        """
        best_d  = np.inf
        best_uv = None
        try:
            from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
            from OCP.gp import gp_Pnt
            # Convert cursor UV to world 3D for the projection query
            cursor_world = self._uv_to_world(cursor_uv, plane)
            query = gp_Pnt(float(cursor_world[0]),
                           float(cursor_world[1]),
                           float(cursor_world[2]))

            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                geom_curve = a.Curve().Curve()
                u0, u1 = a.FirstParameter(), a.LastParameter()
                proj = GeomAPI_ProjectPointOnCurve(query, geom_curve, u0, u1)
                if proj.NbPoints() > 0:
                    nearest = proj.NearestPoint()
                    w = np.array([nearest.X(), nearest.Y(), nearest.Z()],
                                 dtype=np.float64)
                    uv = self._world_to_uv(w, plane)
                    d  = float(np.linalg.norm(cursor_uv - uv))
                    if d < best_d:
                        best_d  = d
                        best_uv = uv
        except Exception:
            pass
        return best_uv, best_d

    # ------------------------------------------------------------------
    # Snap implementations
    # ------------------------------------------------------------------

    def _snap_endpoint(self, cursor, entities, radius, plane):
        from cad.sketch import LineEntity, ReferenceEntity
        best_d = radius
        best_p = None

        # Extra fixed points (world origin etc.)
        for ep in self.extra_points:
            d = float(np.linalg.norm(cursor - ep))
            if d < best_d:
                best_d = d
                best_p = np.array(ep, dtype=np.float64)

        for ent in entities:
            candidates = []
            if isinstance(ent, LineEntity):
                candidates = [ent.p0, ent.p1]
            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    # Skip endpoints of closed curves (circles, closed splines)
                    # — center snap handles those more usefully.
                    if _all_edges_closed(ent.occ_edges):
                        continue
                    candidates = self._curve_endpoints_uv(ent, plane)
                elif len(ent.points) >= 2:
                    # Polyline fallback — skip if first==last (closed loop)
                    p0, p1 = ent.points[0], ent.points[-1]
                    if np.linalg.norm(p1 - p0) > 1e-6:
                        candidates = [p0, p1]

            for p in candidates:
                d = float(np.linalg.norm(cursor - p))
                if d < best_d:
                    best_d = d
                    best_p = np.array(p, dtype=np.float64)
        return best_p

    def _snap_midpoint(self, cursor, entities, radius, plane):
        """
        For OCC edges: true mid-parameter point (no tessellation contamination).
        For LineEntity: arithmetic midpoint.
        For ReferenceEntity without occ_edges: midpoints of polyline segments.
        """
        from cad.sketch import LineEntity, ReferenceEntity
        best_d = radius
        best_p = None

        for ent in entities:
            mids = []
            if isinstance(ent, LineEntity):
                mids = [(ent.p0 + ent.p1) * 0.5]
            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    mids = self._curve_midpoint_uv(ent, plane)
                else:
                    pts = ent.points
                    mids = [(pts[i] + pts[i+1]) * 0.5
                            for i in range(len(pts) - 1)]

            for m in mids:
                d = float(np.linalg.norm(cursor - m))
                if d < best_d:
                    best_d = d
                    best_p = np.array(m, dtype=np.float64)
        return best_p

    def _snap_center(self, cursor, entities, radius, plane):
        from cad.sketch import ReferenceEntity
        best_d = radius
        best_p = None

        for ent in entities:
            if not isinstance(ent, ReferenceEntity):
                continue
            centers = []
            if ent.occ_edges:
                centers = self._curve_centers_uv(ent, plane)
            else:
                # Fallback: centroid of closed polyline
                if len(ent.points) >= 3:
                    p0, p1 = ent.points[0], ent.points[-1]
                    if np.linalg.norm(p1 - p0) < radius * 4:
                        centers = [np.mean(
                            [p.astype(np.float64) for p in ent.points],
                            axis=0)]

            for c in centers:
                d = float(np.linalg.norm(cursor - c))
                if d < best_d:
                    best_d = d
                    best_p = np.array(c, dtype=np.float64)
        return best_p

    def _snap_nearest(self, cursor, entities, radius, plane):
        from cad.sketch import LineEntity, ReferenceEntity
        best_d = radius
        best_p = None

        for ent in entities:
            if isinstance(ent, LineEntity):
                p = _nearest_on_segment(cursor, ent.p0, ent.p1)
                d = float(np.linalg.norm(cursor - p))
                if d < best_d:
                    best_d = d
                    best_p = p

            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    uv, d = self._curve_nearest_uv(cursor, ent, plane)
                    if uv is not None and d < best_d:
                        best_d = d
                        best_p = uv
                else:
                    pts = ent.points
                    for i in range(len(pts) - 1):
                        p = _nearest_on_segment(cursor, pts[i], pts[i+1])
                        d = float(np.linalg.norm(cursor - p))
                        if d < best_d:
                            best_d = d
                            best_p = p
        return best_p

    def _snap_grid(self, cursor, radius):
        if self.grid_size <= 0:
            return None
        gs = self.grid_size
        snapped = np.array([
            round(cursor[0] / gs) * gs,
            round(cursor[1] / gs) * gs,
        ], dtype=np.float64)
        return snapped if np.linalg.norm(cursor - snapped) < radius else None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _nearest_on_segment(p: np.ndarray, a: np.ndarray,
                         b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return np.array(a, dtype=np.float64)
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_len_sq))
    return a + t * ab


def _all_edges_closed(occ_edges: list) -> bool:
    """
    Return True if all OCC edges in the list are closed curves
    (i.e. the curve's first and last parameter meet at the same point).
    Used to suppress endpoint snap on circles so center snap fires instead.
    """
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        for edge in occ_edges:
            a = BRepAdaptor_Curve(edge)
            p0 = a.Value(a.FirstParameter())
            p1 = a.Value(a.LastParameter())
            dist = ((p0.X()-p1.X())**2 +
                    (p0.Y()-p1.Y())**2 +
                    (p0.Z()-p1.Z())**2) ** 0.5
            if dist > 1e-4:
                return False
        return True
    except Exception:
        return False
