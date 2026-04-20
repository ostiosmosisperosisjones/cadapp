from build123d import Compound
from OCP.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Arc
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.TopTools import TopTools_ListOfShape


def _offset_solid(face_occ, thickness: float, tol: float) -> object:
    om = BRepOffset_MakeOffset()
    om.Initialize(face_occ, abs(float(thickness)), tol,
                  BRepOffset_Skin, False, False, GeomAbs_Arc, True)
    om.MakeOffsetShape()
    if not om.IsDone():
        raise RuntimeError(f"Thicken failed — offset kernel error: {om.Error()}")
    return om.Shape()


def thicken_face_preview(face_occ, thickness: float) -> Compound:
    """
    Fast preview: offset only, coarse tolerance, no fuse with the body.
    The result is just the thickened slab — good enough for visual feedback.
    """
    if thickness == 0:
        raise ValueError("thickness must be non-zero")
    return Compound(_offset_solid(face_occ, thickness, 1e-2))


def thicken_face(body_occ, face_occ, thickness: float) -> Compound:
    """
    Commit-quality thicken: fine tolerance offset fused into the body.
    """
    if thickness == 0:
        raise ValueError("thickness must be non-zero")

    thick_occ = _offset_solid(face_occ, thickness, 1e-4)

    lst_a = TopTools_ListOfShape(); lst_a.Append(body_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(thick_occ)
    fuse = BRepAlgoAPI_Fuse()
    fuse.SetArguments(lst_a)
    fuse.SetTools(lst_b)
    fuse.SetRunParallel(True)
    fuse.Build()
    if not fuse.IsDone():
        raise RuntimeError("Thicken fuse failed — could not boolean-union thickened face with body")

    return Compound(fuse.Shape())
