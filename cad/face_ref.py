"""
cad/face_ref.py

Stable face references that survive boolean operations.

A FaceRef fingerprints a face by:
  - normal vector        (orientation, invariant through most ops)
  - area                 (size, invariant unless face is split)
  - centroid_perp        (centroid projected onto the plane perpendicular
                          to the normal — invariant through extrude/cut)
  - centroid_along       (centroid projected onto the normal — shifts
                          predictably: += distance for extrude/cut)

To find a face after an operation, we match on normal + area +
centroid_perp. That triple uniquely identifies a face in all cases
we've verified, and is robust to topology renumbering.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _occ_face_props(occ_face):
    """Return (centroid_xyz, area) for a raw OCC face."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(occ_face, props)
    c = props.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()]), props.Mass()


def _occ_face_normal(occ_face):
    """
    Return unit normal for a planar OCC face, or None if not planar.
    Normal direction follows the face orientation in the shape.
    """
    try:
        surf = BRepAdaptor_Surface(occ_face)
        if surf.GetType() != GeomAbs_Plane:
            return None
        d = surf.Plane().Axis().Direction()
        n = np.array([d.X(), d.Y(), d.Z()])
        return n / np.linalg.norm(n)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FaceRef
# ---------------------------------------------------------------------------

@dataclass
class FaceRef:
    """
    Geometry-based face identifier — survives boolean topology changes.

    All values are in world (un-normalised) coordinates, matching
    build123d / OCCT space.
    """
    normal:          tuple   # (nx, ny, nz) unit vector
    area:            float
    centroid_perp:   tuple   # centroid projected perpendicular to normal
    centroid_along:  float   # centroid projected along normal

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    @classmethod
    def from_occ_face(cls, occ_face) -> "FaceRef | None":
        """Build a FaceRef from a raw OCC TopoDS_Face. Returns None if not planar."""
        normal = _occ_face_normal(occ_face)
        if normal is None:
            return None
        centroid, area = _occ_face_props(occ_face)
        along = float(np.dot(centroid, normal))
        perp  = centroid - along * normal
        return cls(
            normal         = tuple(np.round(normal, 8)),
            area           = round(float(area), 6),
            centroid_perp  = tuple(np.round(perp, 6)),
            centroid_along = round(along, 6),
        )

    @classmethod
    def from_b3d_face(cls, face) -> "FaceRef | None":
        """Build from a build123d Face object."""
        return cls.from_occ_face(face.wrapped)

    # ----------------------------------------------------------------
    # Matching
    # ----------------------------------------------------------------

    def find_in(self, shape,
                normal_tol:  float = 0.001,
                area_tol:    float = 0.5,
                perp_tol:    float = 0.1,
                ) -> tuple[int, object] | tuple[None, None]:
        """
        Find the best matching face in a build123d shape.

        Matches on: normal direction, area, centroid_perp.
        The centroid_along component is intentionally ignored —
        it shifts predictably with extrude/cut distance.

        Returns (face_index, b3d_face) or (None, None) if no match.
        """
        ref_normal = np.array(self.normal)
        ref_perp   = np.array(self.centroid_perp)

        best_idx        = None
        best_face       = None
        best_perp_dist  = float("inf")
        best_same_dir   = False      # prefer same-direction over opposite
        best_along_dist = float("inf")

        for idx, face in enumerate(shape.faces()):
            occ = face.wrapped
            n = _occ_face_normal(occ)
            if n is None:
                continue

            # Normal must be parallel (same or opposite direction)
            dot = float(np.dot(n, ref_normal))
            if abs(abs(dot) - 1.0) > normal_tol:
                continue
            same_dir = dot > 0.0

            centroid, area = _occ_face_props(occ)

            # Area must match
            if abs(area - self.area) > area_tol:
                continue

            # Perpendicular centroid component must match
            along = float(np.dot(centroid, n))
            perp  = centroid - along * n
            perp_dist = float(np.linalg.norm(perp - ref_perp))
            if perp_dist > perp_tol:
                continue

            along_dist = abs(along - self.centroid_along)

            # Rank: same-direction normal wins over opposite; then closest
            # centroid_along (face didn't move far); then closest perp dist.
            better = False
            if best_idx is None:
                better = True
            elif same_dir and not best_same_dir:
                better = True
            elif same_dir == best_same_dir:
                if along_dist < best_along_dist - 1e-6:
                    better = True
                elif abs(along_dist - best_along_dist) < 1e-6 and perp_dist < best_perp_dist:
                    better = True

            if better:
                best_idx        = idx
                best_face       = face
                best_perp_dist  = perp_dist
                best_same_dir   = same_dir
                best_along_dist = along_dist

        return best_idx, best_face

    # ----------------------------------------------------------------
    # Prediction helpers
    # ----------------------------------------------------------------

    def predict_after_extrude(self, distance: float) -> "FaceRef":
        """
        Return the FaceRef we expect for the new top/bottom face
        after extruding this face by *distance*.

        The new face has the same normal and area, centroid_along
        shifts by distance, centroid_perp is unchanged.
        """
        return FaceRef(
            normal         = self.normal,
            area           = self.area,
            centroid_perp  = self.centroid_perp,
            centroid_along = round(self.centroid_along + distance, 6),
        )


# ---------------------------------------------------------------------------
# AnyFaceRef  —  centroid+area fingerprint that works for non-planar faces
# ---------------------------------------------------------------------------

@dataclass
class AnyFaceRef:
    """
    Geometry-based face identifier for any face type (planar or curved).

    Matches by centroid position and area — sufficient to re-locate a
    face after operations that preserve face identity (thicken, offset).
    """
    centroid: tuple   # (x, y, z) world coords
    area:     float

    @classmethod
    def from_occ_face(cls, occ_face) -> "AnyFaceRef":
        centroid, area = _occ_face_props(occ_face)
        return cls(
            centroid = tuple(np.round(centroid, 6)),
            area     = round(float(area), 6),
        )

    def find_in(self, shape,
                area_tol:     float = 0.5,
                centroid_tol: float = 0.5,
                ) -> tuple[int, object] | tuple[None, None]:
        """
        Find best matching face by centroid proximity and area.
        Returns (face_index, b3d_face) or (None, None).
        """
        ref_c = np.array(self.centroid)
        best_idx  = None
        best_face = None
        best_dist = float("inf")

        for idx, face in enumerate(shape.faces()):
            centroid, area = _occ_face_props(face.wrapped)
            if abs(area - self.area) > area_tol:
                continue
            dist = float(np.linalg.norm(centroid - ref_c))
            if dist > centroid_tol:
                continue
            if dist < best_dist:
                best_dist  = dist
                best_idx   = idx
                best_face  = face

        return best_idx, best_face
