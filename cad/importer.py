from build123d import import_step, Plane, Compound
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.BRepLib import BRepLib


def _heal(occ_shape):
    """Run the standard healing pipeline on a raw TopoDS_Shape."""
    # 1. General shape fix (bad topology, missing seams, etc.)
    fixer = ShapeFix_Shape(occ_shape)
    fixer.Perform()
    fixed = fixer.Shape()

    # 2. Unify same-domain faces/edges and concatenate B-splines to remove
    #    artificial C0 seams introduced by STEP tessellation or export quirks.
    unifier = ShapeUpgrade_UnifySameDomain(fixed, True, True, True)
    unifier.Build()
    unified = unifier.Shape()

    # 3. Encode geometric continuity so offset/thicken ops can exploit it.
    BRepLib.EncodeRegularity_s(unified)

    return unified


def load_step(path):
    """Load a STEP file, apply geometry healing, and return the shape."""
    raw = import_step(path)
    healed_occ = _heal(raw.wrapped)
    return Compound(healed_occ)

def get_planar_faces(shape):
    """Return list of (index, face) tuples for all planar faces."""
    result = []
    for i, face in enumerate(shape.faces()):
        try:
            Plane(face)
            result.append((i, face))
        except Exception:
            pass
    return result

def get_face(shape, index):
    """Return the build123d Face object at the given index."""
    return list(shape.faces())[index]
