"""
cad/operations/__init__.py

Operation registry — the single source of truth for:
  1. How to execute each operation (REGISTRY)
  2. What parameters are editable and how to present them (EDIT_SCHEMA)

Adding a new operation:
  - Add its execute function to REGISTRY
  - Add its param schema to EDIT_SCHEMA
  - That's it. History replay, edit dialogs, and the panel all
    pick it up automatically.
"""

from cad.operations.extrude import extrude_face, extrude_face_direct

# ---------------------------------------------------------------------------
# Execution registry
#
# Each entry: operation_name -> callable(shape, face_idx, params) -> new_shape
# face_idx may be None for ops that don't target a specific face.
# ---------------------------------------------------------------------------

REGISTRY: dict[str, callable] = {
    "extrude": lambda shape, face_idx, params: extrude_face(
                   shape, face_idx,  params["distance"]),
    "cut":     lambda shape, face_idx, params: extrude_face(
                   shape, face_idx, -abs(params["distance"])),
    # "sketch" is intentionally absent — it is a no-op on geometry,
    # replay just carries shape_before forward unchanged.
    # Future:
    # "fillet":  lambda shape, face_idx, params: fillet_face(
    #                shape, face_idx, params["radius"]),
    # "revolve": lambda shape, face_idx, params: revolve_face(
    #                shape, face_idx, params["angle"]),
}

# ---------------------------------------------------------------------------
# Edit schema
#
# Each entry: operation_name -> list of param descriptors:
#   (param_key, label, type, min, max, decimals)
#
# The edit dialog iterates this list to build its input fields generically.
# ---------------------------------------------------------------------------

EDIT_SCHEMA: dict[str, list[tuple]] = {
    "extrude": [
        ("distance", "Distance", float, 0.001, 1000.0, 3, "length"),
    ],
    "cut": [
        ("distance", "Depth", float, 0.001, 1000.0, 3, "length"),
    ],
}
# Schema tuple: (param_key, label, type, min, max, decimals, kind)
# kind="length" -> ExprSpinBox with unit conversion
# kind="angle"  -> ExprSpinBox (future, degrees)
# kind=None     -> plain QDoubleSpinBox

__all__ = ["extrude_face", "extrude_face_direct", "REGISTRY", "EDIT_SCHEMA"]
