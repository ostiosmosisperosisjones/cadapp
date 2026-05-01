"""
cad/edge_ref.py

Stable edge references and abstract edge sources for parametric include replay.

EdgeRef          — geometry fingerprint that re-finds an edge after topology changes
EdgeSource       — abstract base: resolve → (world_pts, occ_edges)
BodyEdgeSource   — edge on a specific body (most common include case)
SketchEdgeSource — line entity from a previously committed sketch
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# EdgeRef — stable fingerprint
# ---------------------------------------------------------------------------

@dataclass
class EdgeRef:
    """
    Identifies an edge by midpoint, arc length, and tangent direction at
    the midpoint.  The tangent is canonicalised (first non-zero component
    is positive) so the same physical edge always produces the same ref
    regardless of OCCT orientation.

    Tolerances in find_in() are intentionally generous to survive boolean
    operations that may slightly perturb edge positions.
    """
    midpoint: tuple    # (x, y, z) world mm
    length:   float    # arc length mm
    tangent:  tuple    # (tx, ty, tz) unit vector, canonicalised

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_occ_edge(cls, occ_edge) -> "EdgeRef | None":
        """Build an EdgeRef from a raw TopoDS_Edge. Returns None on failure."""
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GCPnts import GCPnts_AbscissaPoint

            adaptor = BRepAdaptor_Curve(occ_edge)
            u0   = adaptor.FirstParameter()
            u1   = adaptor.LastParameter()
            umid = (u0 + u1) * 0.5

            p = adaptor.Value(umid)
            midpoint = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))

            length = GCPnts_AbscissaPoint.Length_s(adaptor, u0, u1, 1e-6)

            tv  = adaptor.DN(umid, 1)
            t   = np.array([tv.X(), tv.Y(), tv.Z()], dtype=np.float64)
            tn  = np.linalg.norm(t)
            if tn < 1e-10:
                return None
            t /= tn
            # Canonicalise: flip if first significant component is negative
            for c in t:
                if abs(c) > 1e-6:
                    if c < 0:
                        t = -t
                    break
            tangent = tuple(np.round(t, 8).tolist())

            return cls(midpoint=midpoint,
                       length=round(float(length), 6),
                       tangent=tangent)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_in(self, shape,
                mid_tol: float = 0.1,
                len_tol: float = 0.5,
                tan_tol: float = 0.02,
                ) -> tuple[int, object, list] | tuple[None, None, None]:
        """
        Find the matching edge in a build123d shape.

        Returns (edge_index, TopoDS_Edge, world_pts) where world_pts is a
        list of [x, y, z] floats sampled along the edge.
        Returns (None, None, None) if no match is found.
        """
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_AbscissaPoint

        ref_mid = np.array(self.midpoint)
        ref_tan = np.array(self.tangent)
        best_dist = float("inf")
        best = (None, None, None)

        for idx, edge in enumerate(shape.edges()):
            occ = edge.wrapped
            try:
                adaptor = BRepAdaptor_Curve(occ)
                u0   = adaptor.FirstParameter()
                u1   = adaptor.LastParameter()
                umid = (u0 + u1) * 0.5

                p   = adaptor.Value(umid)
                mid = np.array([p.X(), p.Y(), p.Z()])
                dist = float(np.linalg.norm(mid - ref_mid))
                if dist > mid_tol:
                    continue

                length = GCPnts_AbscissaPoint.Length_s(adaptor, u0, u1, 1e-6)
                if abs(length - self.length) > len_tol:
                    continue

                tv = adaptor.DN(umid, 1)
                t  = np.array([tv.X(), tv.Y(), tv.Z()])
                tn = np.linalg.norm(t)
                if tn > 1e-10:
                    t /= tn
                    for c in t:
                        if abs(c) > 1e-6:
                            if c < 0:
                                t = -t
                            break
                    if float(np.linalg.norm(t - ref_tan)) > tan_tol:
                        continue

                if dist < best_dist:
                    best_dist = dist
                    best = (idx, occ, _sample_edge(adaptor, u0, u1))

            except Exception:
                continue

        return best


def _sample_edge(adaptor, u0: float, u1: float, n: int = 32) -> list:
    """Sample n+1 evenly-spaced world points along an edge adaptor."""
    pts = []
    for i in range(n + 1):
        u = u0 + (u1 - u0) * i / n
        p = adaptor.Value(u)
        pts.append([p.X(), p.Y(), p.Z()])
    return pts


# ---------------------------------------------------------------------------
# EdgeSource — abstract
# ---------------------------------------------------------------------------

class EdgeSource(ABC):
    """
    Abstract reference to a geometric edge that can be re-resolved at any
    point in the replay timeline.

    resolve(history, before_index) → (world_pts, occ_edges)
      world_pts  : list of [x, y, z]  — used for UV projection
      occ_edges  : list of TopoDS_Edge | None  — for exact face construction
    Raises RuntimeError on failure so the caller can surface the error.
    """

    @abstractmethod
    def resolve(self, history, before_index: int) -> tuple[list, list | None]:
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...


# ---------------------------------------------------------------------------
# BodyEdgeSource
# ---------------------------------------------------------------------------

class BodyEdgeSource(EdgeSource):
    """An edge on a specific body, re-found each replay via EdgeRef matching."""

    def __init__(self, body_id: str, edge_ref: EdgeRef):
        self.body_id  = body_id
        self.edge_ref = edge_ref

    def resolve(self, history, before_index: int) -> tuple[list, list | None]:
        shape = history._shape_for_body_at(self.body_id, before_index)
        if shape is None:
            raise RuntimeError(
                f"BodyEdgeSource: no shape for body '{self.body_id}' "
                f"before history index {before_index}")
        _, occ_edge, world_pts = self.edge_ref.find_in(shape)
        if occ_edge is None:
            raise RuntimeError(
                f"BodyEdgeSource: could not relocate edge in body '{self.body_id}'")
        return world_pts, [occ_edge]

    def to_dict(self) -> dict:
        return {"type": "body_edge", "body_id": self.body_id}


# ---------------------------------------------------------------------------
# SketchEdgeSource
# ---------------------------------------------------------------------------

class SketchEdgeSource(EdgeSource):
    """
    A LineEntity from a previously committed sketch entry.

    On resolve(), converts the UV-space line back to world space using the
    source SketchEntry's (possibly already-updated) plane cache, so it
    automatically inherits upstream sketch plane changes.
    """

    def __init__(self, history_idx: int, entity_idx: int):
        self.history_idx = history_idx
        self.entity_idx  = entity_idx

    def resolve(self, history, before_index: int) -> tuple[list, list | None]:
        from cad.sketch import LineEntity

        entries = history.entries
        if self.history_idx >= len(entries):
            raise RuntimeError(
                f"SketchEdgeSource: history_idx {self.history_idx} out of range")
        entry = entries[self.history_idx]
        se = entry.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"SketchEdgeSource: entry {self.history_idx} has no sketch_entry")
        if self.entity_idx >= len(se.entities):
            raise RuntimeError(
                f"SketchEdgeSource: entity_idx {self.entity_idx} out of range")
        ent = se.entities[self.entity_idx]

        def _uv_to_world(uv):
            return (se.plane_origin
                    + float(uv[0]) * se.plane_x_axis
                    + float(uv[1]) * se.plane_y_axis)

        from cad.sketch import ArcEntity
        if isinstance(ent, LineEntity):
            p0 = _uv_to_world(ent.p0)
            p1 = _uv_to_world(ent.p1)
            return [p0.tolist(), p1.tolist()], None
        elif isinstance(ent, ArcEntity):
            import math
            from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
            from OCP.Geom import Geom_Circle
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            cx, cy = float(ent.center[0]), float(ent.center[1])
            c3d = _uv_to_world(ent.center)
            ax2 = gp_Ax2(
                gp_Pnt(*c3d.tolist()),
                gp_Dir(float(se.plane_normal[0]),
                       float(se.plane_normal[1]),
                       float(se.plane_normal[2])),
                gp_Dir(float(se.plane_x_axis[0]),
                       float(se.plane_x_axis[1]),
                       float(se.plane_x_axis[2])),
            )
            geom_circ = Geom_Circle(ax2, ent.radius)
            span = ent.end_angle - ent.start_angle
            if abs(span - 2 * math.pi) < 1e-9 or abs(span) < 1e-9:
                occ_edge = BRepBuilderAPI_MakeEdge(geom_circ).Edge()
            else:
                occ_edge = BRepBuilderAPI_MakeEdge(
                    geom_circ, ent.start_angle, ent.end_angle).Edge()
            p0 = _uv_to_world(ent.p0)
            p1 = _uv_to_world(ent.p1)
            return [p0.tolist(), p1.tolist()], [occ_edge]
        else:
            raise RuntimeError(
                f"SketchEdgeSource: entity {self.entity_idx} has unsupported type "
                f"{type(ent).__name__}")

    def to_dict(self) -> dict:
        return {
            "type":        "sketch_edge",
            "history_idx": self.history_idx,
            "entity_idx":  self.entity_idx,
        }
