"""
cad/op_types.py

Re-exports all operation types and the dispatch table.
Implementation lives in:
  cad/op_base.py             — Op base class, _push_result()
  cad/op_extrude.py          — FaceExtrudeOp, CrossBodyCutOp, SketchExtrudeOp, SketchOp, ImportOp
  cad/op_revolve_thicken.py  — ThickenOp, FaceRevolveOp, SketchRevolveOp
"""

from cad.op_base import Op, _push_result
from cad.op_extrude import (
    FaceExtrudeOp,
    CrossBodyCutOp,
    SketchExtrudeOp,
    SketchOp,
    ImportOp,
)
from cad.op_revolve_thicken import (
    ThickenOp,
    FaceRevolveOp,
    SketchRevolveOp,
)

from typing import Any


def _extrude_or_cut_from_params(operation: str, params: dict) -> Op:
    """Route extrude/cut params to the right op type, applying sign from operation."""
    sign = -1 if operation == "cut" else 1
    if "cut_body_id" in params:
        return CrossBodyCutOp._from_params(params, sign)
    if "from_sketch_id" in params:
        return SketchExtrudeOp._from_params(params, sign)
    return FaceExtrudeOp._from_params(params, sign)


_FROM_PARAMS: dict[str, Any] = {
    "extrude": _extrude_or_cut_from_params,
    "cut":     _extrude_or_cut_from_params,
    "sketch":  lambda op, p: SketchOp._from_params(p),
    "import":  lambda op, p: ImportOp._from_params(p),
    "thicken": lambda op, p: ThickenOp._from_params(p),
    "revolve": lambda op, p: (FaceRevolveOp._from_params(p)
                               if "source_body_id" in p
                               else SketchRevolveOp._from_params(p)),
}

__all__ = [
    "Op", "_push_result",
    "FaceExtrudeOp", "CrossBodyCutOp", "SketchExtrudeOp", "SketchOp", "ImportOp",
    "ThickenOp", "FaceRevolveOp", "SketchRevolveOp",
    "_FROM_PARAMS", "_extrude_or_cut_from_params",
]
