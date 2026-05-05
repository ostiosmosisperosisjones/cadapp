"""
cad/operations/fillet.py

3D edge fillet operations:
  fillet_face() — fillet selected edges of a solid to a given radius
  fillet_face_preview() — fast preview (no boolean, just the filleted edges)

Uses OCC BRepFilletAPI_MakeFillet for multi-edge filleting.
"""

from build123d import Compound
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.ChFi3d import ChFi3d_FilletShape


def _fillet_edges(shape, face_indices, radius: float):
    """
    Apply edge fillet to the edges belonging to the given faces.

    Uses BRepFilletAPI_MakeFillet.Add(radius, edge) for each edge to fillet.
    Returns the filleted shape or raises on failure.
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    source_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape

    # Collect unique edges from selected faces using build123d face.edges()
    # which returns TopoDS_Edge instances (not TopoDS_Shape from TopExp_Explorer)
    seen = set()
    unique_edges = []
    for fi in face_indices:
        face = shape.faces()[fi]
        for edge in face.edges():
            edge_occ = edge.wrapped  # TopoDS_Edge
            if id(edge_occ) not in seen:
                seen.add(id(edge_occ))
                unique_edges.append(edge_occ)

    if not unique_edges:
        raise RuntimeError("Fillet: no edges found on selected faces")

    # Create fillet algorithm with rational shape
    nf = BRepFilletAPI_MakeFillet(source_occ, ChFi3d_FilletShape.ChFi3d_Rational)

    # Add each edge with the specified radius
    for edge_occ in unique_edges:
        try:
            nf.Add(radius, edge_occ)
        except Exception:
            # Some edges may already be sharp (sharp corners) — skip silently
            continue

    if not nf.IsDone():
        try:
            nf.Build()
        except Exception as ex:
            raise RuntimeError(f"Fillet: build failed — {ex}")

    if not nf.IsDone():
        raise RuntimeError(f"Fillet failed — kernel returned: {nf.Error()}")

    result = nf.Shape()
    return Compound(result)


def fillet_face_preview(shape, face_indices, all_faces, radius: float):
    """
    Fast preview: fillet the edges of selected faces.
    Returns the filleted shape or raises on failure.
    This is a preview only — no boolean with the body.
    """
    return _fillet_edges(shape, face_indices, radius)


def fillet_face(shape, face_indices, radius: float):
    """
    Commit-quality fillet: fillet edges of selected faces.

    Parameters:
        shape: build123d Compound/Solid
        face_indices: list of face indices at commit time
        radius: mm — fillet radius

    Returns the filleted shape or raises on failure.
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    all_face_occs = list(shape.faces())
    source_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape

    try:
        # Collect unique edges from selected faces using build123d face.edges()
        seen = set()
        unique_edges = []
        for fi in face_indices:
            if fi >= len(all_face_occs):
                raise RuntimeError(f"face_idx {fi} out of range")
            for edge in all_face_occs[fi].edges():
                edge_occ = edge.wrapped  # TopoDS_Edge
                if id(edge_occ) not in seen:
                    seen.add(id(edge_occ))
                    unique_edges.append(edge_occ)

        if not unique_edges:
            raise RuntimeError("Fillet: no edges found on selected faces")

        nf = BRepFilletAPI_MakeFillet(source_occ, ChFi3d_FilletShape.ChFi3d_Rational)

        # Add each edge with the specified radius
        for edge_occ in unique_edges:
            try:
                nf.Add(radius, edge_occ)
            except Exception:
                # Some edges may already be sharp (sharp corners) — skip silently
                continue

        if not nf.IsDone():
            try:
                nf.Build()
            except Exception as ex:
                raise RuntimeError(f"Fillet: build failed — {ex}")

        if not nf.IsDone():
            raise RuntimeError(f"Fillet failed — kernel returned: {nf.Error()}")

        result = nf.Shape()
        return Compound(result)
    except Exception as ex:
        raise RuntimeError(f"Fillet operation failed: {ex}")
