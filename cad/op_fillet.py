"""
cad/op_fillet.py

FaceFilletOp — fillet selected edges of a body.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class FaceFilletOp(Op):
    """
    Fillet edges on faces of *source_body_id* by *radius* mm.

    source_body_id : body whose faces are selected
    face_indices   : face indices at commit time (fallback for old entries)
    radius         : mm — fillet radius
    """
    source_body_id: str
    face_indices:   list
    radius:         float

    def _resolve_faces(self, shape):
        """Return list of (idx, b3d_face) using face_refs when available."""
        from cad.face_ref import AnyFaceRef
        all_faces = list(shape.faces())
        resolved = []
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"FaceFilletOp: face_idx {idx} out of range")
            resolved.append((idx, all_faces[idx]))
        return resolved

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.fillet import fillet_face

        if shape is None:
            src = history._shape_for_body_at(self.source_body_id, entry_index)
            if src is None:
                raise RuntimeError(
                    f"FaceFilletOp: no shape for source body '{self.source_body_id}'")
            shape = src

        resolved = self._resolve_faces(shape)
        indices = [idx for idx, _ in resolved]

        result = fillet_face(shape, indices, self.radius)
        return result

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        compute, finalize = self._split_commit(viewport, extra_params)
        try:
            shape_after = compute()
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            shape_after = None
            viewport._pending_op_error = str(ex)
        else:
            viewport._pending_op_error = None
        try:
            finalize(shape_after)
        finally:
            viewport._pending_op_error = None
        return shape_after

    def _split_commit(self, viewport: Any, extra_params: dict | None = None):
        from cad.operations.fillet import fillet_face
        from cad.units import format_op_label as _lbl

        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Fillet] No shape for body {self.source_body_id}")
        all_faces = list(shape_before.faces())
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"[Fillet] face_idx {idx} out of range")

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))
        radius = self.radius

        def compute():
            return fillet_face(shape_before, self.face_indices, radius)

        def finalize(shape_after):
            op_str = "fillet"
            _push_result(viewport, op_str, op_params, self.source_body_id,
                         None, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        viewport.reopen_fillet(history_idx)

    def to_params(self) -> dict:
        return {
            "source_body_id": self.source_body_id,
            "face_indices":   self.face_indices,
            "radius":          self.radius,
        }

    @classmethod
    def _from_params(cls, params: dict) -> "FaceFilletOp":
        if "face_indices" in params:
            indices = list(params["face_indices"])
        else:
            indices = [int(params.get("face_idx", 0))]
        return cls(
            source_body_id = params.get("source_body_id", ""),
            face_indices   = indices,
            radius         = float(params.get("radius", 1.0)),
        )
