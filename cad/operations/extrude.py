"""
cad/operations/extrude.py
"""

import numpy as np
from build123d import extrude, Plane, Face, Compound
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.BOPAlgo import BOPAlgo_GlueEnum


def extrude_face(shape, face_idx: int, distance: float,
                 direction: np.ndarray | None = None):
    """
    Extrude a planar face of *shape* by *distance* along its normal
    (or along *direction* if given).

    Positive distance  → extrude outward, fuse  (adds material).
    Negative distance  → extrude inward,  cut   (removes material).

    Returns the resulting build123d Compound/Solid, or raises on failure.
    """
    faces = list(shape.faces())
    if face_idx < 0 or face_idx >= len(faces):
        raise IndexError(
            f"face_idx {face_idx} out of range (shape has {len(faces)} faces)"
        )

    face: Face = faces[face_idx]
    Plane(face)  # raises if not planar

    if distance == 0:
        return shape

    return _do_extrude_boolean(shape, face, distance, direction=direction)


def extrude_face_direct(body_shape, face: Face, distance: float,
                        direction: np.ndarray | None = None):
    """
    Extrude a face object directly (not looked up by index).

    Used for sketch-profile extrusion where the face is built from a
    SketchEntry wire rather than being an existing face of body_shape.

    Positive distance  → fuse (adds material).
    Negative distance  → cut  (removes material).

    If body_shape is None (first operation on a new body, e.g. from a
    world-plane sketch), the extrusion is returned as-is with no boolean.

    Returns the resulting build123d Compound/Solid, or raises on failure.
    """
    if distance == 0:
        return body_shape

    Plane(face)  # raises if not planar

    if body_shape is None:
        extruded = _do_extrude_solid(face, distance, direction)
        return Compound(extruded.wrapped)

    return _do_extrude_boolean(body_shape, face, distance, glue=False,
                               direction=direction)


def _do_extrude_solid(face: Face, distance: float,
                      direction: np.ndarray | None = None):
    """Produce the raw extruded solid (no boolean)."""
    from build123d import extrude as b3d_extrude, Vector
    if direction is not None:
        # direction is a unit vector; scale by signed distance so the solid
        # extends in the correct direction and by the correct amount.
        d = direction / np.linalg.norm(direction)
        vec = Vector(float(d[0] * distance),
                     float(d[1] * distance),
                     float(d[2] * distance))
        return b3d_extrude(face, amount=abs(distance), dir=vec)
    # No custom direction — negative amount extrudes inward (into the body).
    return b3d_extrude(face, amount=distance)


def _do_extrude_boolean(shape, face: Face, distance: float,
                        glue: bool = True,
                        direction: np.ndarray | None = None):
    """
    Shared boolean logic for both extrude entry points.

    glue=True  — safe only when the tool's base face is the exact same OCCT
                 face object as an existing face on shape (i.e. extrude_face).
                 Tells OCCT shapes share a face → skips full intersection scan.
    glue=False — required for sketch-profile extrudes where the face is
                 reconstructed geometry, not a face already on shape.
    """
    extruded = _do_extrude_solid(face, distance, direction)

    if distance > 0:
        op = BRepAlgoAPI_Fuse()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(extruded.wrapped))
        if glue:
            op.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
    else:
        op = BRepAlgoAPI_Cut()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(extruded.wrapped))

    op.SetRunParallel(True)
    op.Build()

    if not op.IsDone():
        raise RuntimeError(
            f"Boolean {'fuse' if distance > 0 else 'cut'} failed"
        )

    return Compound(op.Shape())


def _to_list(occ_shape):
    from OCP.TopTools import TopTools_ListOfShape
    lst = TopTools_ListOfShape()
    lst.Append(occ_shape)
    return lst
