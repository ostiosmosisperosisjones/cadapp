"""
cad/operations/fillet.py

3D edge fillet operations.
Uses OCC BRepFilletAPI_MakeFillet for multi-edge filleting.

Public API
----------
fillet_edges(shape, face_indices, edge_occs, radius)
    Commit-quality fillet.  Accepts either face indices (all edges of those
    faces are filleted) or direct TopoDS_Edge objects, or both.

fillet_preview(shape, face_indices, edge_occs, radius)
    Same but used for live preview — identical result, separate name for clarity.
"""

from build123d import Compound
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.ChFi3d import ChFi3d_FilletShape


def _collect_edges(shape, face_indices, edge_occs):
    """
    Return a deduplicated list of TopoDS_Edge objects from:
      - all edges belonging to each face in face_indices
      - the directly supplied edge_occs list
    """
    seen = set()
    unique = []

    def _add(e_occ):
        eid = id(e_occ)
        if eid not in seen:
            seen.add(eid)
            unique.append(e_occ)

    all_faces = list(shape.faces())
    for fi in face_indices:
        if fi >= len(all_faces):
            raise RuntimeError(f"Fillet: face_idx {fi} out of range")
        for edge in all_faces[fi].edges():
            _add(edge.wrapped)

    for e_occ in (edge_occs or []):
        _add(e_occ)

    return unique


def _run_fillet(source_occ, unique_edges, radius: float):
    if not unique_edges:
        raise RuntimeError("Fillet: no edges selected")
    if radius <= 0:
        raise ValueError(f"Fillet: radius must be positive, got {radius}")

    nf = BRepFilletAPI_MakeFillet(source_occ, ChFi3d_FilletShape.ChFi3d_Rational)
    for e_occ in unique_edges:
        try:
            nf.Add(radius, e_occ)
        except Exception:
            continue   # skip seam / degenerate edges

    if not nf.IsDone():
        try:
            nf.Build()
        except Exception as ex:
            raise RuntimeError(f"Fillet: build failed — {ex}")
    if not nf.IsDone():
        raise RuntimeError(f"Fillet failed — kernel returned: {nf.Error()}")

    return Compound(nf.Shape())


def fillet_edges(shape, face_indices: list, edge_occs: list, radius: float):
    """Commit-quality fillet from face indices and/or direct edge objects."""
    source_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape
    unique = _collect_edges(shape, face_indices, edge_occs)
    return _run_fillet(source_occ, unique, radius)


def fillet_preview(shape, face_indices: list, edge_occs: list, radius: float):
    """Live-preview fillet — same as fillet_edges, separate name for clarity."""
    return fillet_edges(shape, face_indices, edge_occs, radius)


# ---------------------------------------------------------------------------
# Back-compat aliases used by existing callers
# ---------------------------------------------------------------------------

def fillet_face(shape, face_indices, radius: float):
    return fillet_edges(shape, face_indices, [], radius)


def fillet_face_preview(shape, face_indices, all_faces, radius: float):
    return fillet_edges(shape, face_indices, [], radius)
