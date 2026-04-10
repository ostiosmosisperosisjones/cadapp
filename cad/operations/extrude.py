"""
cad/operations/extrude.py
"""

from build123d import extrude, Plane, Face, Compound
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.BOPAlgo import BOPAlgo_GlueEnum


def extrude_face(shape, face_idx: int, distance: float):
    """
    Extrude a planar face of *shape* by *distance* along its normal.

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

    return _do_extrude_boolean(shape, face, distance)


def extrude_face_direct(body_shape, face: Face, distance: float):
    """
    Extrude a face object directly (not looked up by index).

    Used for sketch-profile extrusion where the face is built from a
    SketchEntry wire rather than being an existing face of body_shape.

    Positive distance  → fuse (adds material).
    Negative distance  → cut  (removes material).

    Returns the resulting build123d Compound/Solid, or raises on failure.
    """
    if distance == 0:
        return body_shape

    Plane(face)  # raises if not planar

    return _do_extrude_boolean(body_shape, face, distance)


def _do_extrude_boolean(shape, face: Face, distance: float):
    """Shared boolean logic for both extrude entry points."""
    extrude_amount = abs(distance)
    if distance < 0:
        extrude_amount = -extrude_amount

    extruded = extrude(face, amount=extrude_amount)

    if distance > 0:
        op = BRepAlgoAPI_Fuse()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(extruded.wrapped))
        op.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
    else:
        op = BRepAlgoAPI_Cut()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(extruded.wrapped))

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
