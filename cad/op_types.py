"""
cad/op_types.py

Typed operation objects for history entries.

Each op is a pure-data container (dataclass) that also owns:
  execute(shape, history, entry_index) -> new_shape
      Replay this operation.  Raises on failure.
  reopen(viewport, history_idx)
      Restore panel UI state for editing.
  commit(viewport, extra_params)
      Run the operation fresh (first time or after edit), push history
      entries, and trigger mesh rebuild.  Returns the resulting shape or None.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cad.history import History


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Op:
    """Marker base class for all operation types."""

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        raise NotImplementedError

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        """
        Run the operation for the first time (or after an edit), push history
        entries, and trigger mesh rebuild.  Returns result shape or None.
        """
        raise NotImplementedError

    def commit_async(self, viewport: Any, extra_params: dict | None = None) -> None:
        """
        Async variant of commit: run the heavy OCC compute in a background
        thread, then push history and rebuild on the main thread.

        Subclasses override _compute_shape() and _finalize() to opt in.
        Falls back to synchronous commit() if not overridden.
        """
        try:
            compute_fn, finalize_fn = self._split_commit(viewport, extra_params)
        except NotImplementedError:
            self.commit(viewport, extra_params)
            return
        except Exception as ex:
            # _split_commit failed before producing a finalize — push a failed
            # entry directly so the op appears red in history instead of vanishing.
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return
        viewport.run_op_async(type(self).__name__.replace("Op", ""), compute_fn, finalize_fn)

    def _push_failed_entry(self, viewport: Any, error_msg: str,
                           extra_params: dict | None = None) -> None:
        """Push a red history entry when the op fails before compute starts."""
        try:
            # FaceExtrudeOp uses "cut" or "extrude" based on distance sign
            dist = getattr(self, 'distance', None)
            if dist is not None:
                op_str = "cut" if dist < 0 else "extrude"
            else:
                op_str = getattr(self, '_op_str', None) or type(self).__name__.replace("Op", "").lower()
            body_id   = getattr(self, 'source_body_id', None) or getattr(self, 'body_id', None)
            op_params = self.to_params()
            if extra_params:
                op_params.update(extra_params)
            shape_before = viewport.workspace.current_shape(body_id) if body_id else None
            from cad.units import format_op_label as _lbl
            label = _lbl(op_str, op_params)
            entry = viewport.history.push(
                label=label, operation=op_str, params=op_params,
                body_id=body_id, face_ref=None,
                shape_before=shape_before, shape_after=None)
            entry.error     = True
            entry.error_msg = error_msg
            viewport._post_push_cascade(body_id)
            viewport.history_changed.emit()
        except Exception as ex2:
            print(f"[Op] Could not push failed entry: {ex2}")

    def _split_commit(self, viewport: Any, extra_params: dict | None):
        """
        Return (compute_fn, finalize_fn) where:
          compute_fn()           — pure OCC, runs in thread, returns shape_after
          finalize_fn(shape_after) — Qt/history, runs on main thread
        Raise NotImplementedError to fall back to synchronous commit().
        """
        raise NotImplementedError

    def reopen(self, viewport: Any, history_idx: int) -> None:
        """
        Restore panel UI state for editing this entry.
        Called by reopen_extrude() after seek/rebuild; op-specific branching
        lives here rather than in the viewport.
        """
        raise NotImplementedError

    def to_params(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_params(cls, operation: str, params: dict) -> "Op | None":
        """
        Reconstruct the correct Op subclass from an operation string + params
        dict.  Returns None if the operation type is not recognised.
        """
        factory = _FROM_PARAMS.get(operation)
        if factory is None:
            return None
        return factory(operation, params)


# ---------------------------------------------------------------------------
# FaceExtrudeOp  —  extrude or self-cut from a body face
# ---------------------------------------------------------------------------

@dataclass
class FaceExtrudeOp(Op):
    """
    Extrude (distance > 0) or self-cut (distance < 0) from a face on a body.

    source_body_id : body whose face is selected
    face_idx       : index of the face at op-creation time (FaceRef handles
                     parametric re-resolution during replay)
    distance       : signed mm — positive = extrude, negative = cut
    direction      : unit vector or None (None → use face normal)
    target_vertex  : world-space point the extrusion depth is measured to, or None
    start_offset   : mm offset applied to the near end of the tool solid
    end_offset     : mm offset applied to the far end
    force_new_body : True when the result should be a new body instead of
                     modifying the source body
    """
    source_body_id: str
    face_idx:       int
    distance:       float
    direction:      list[float] | None = None
    target_vertex:  list[float] | None = None
    start_offset:   float = 0.0
    end_offset:     float = 0.0
    force_new_body: bool  = False
    face_indices:   list  = None   # face indices (single body, compat)
    face_refs:      list  = None   # FaceRef per face, built at commit
    face_pairs:     list  = None   # list of (body_id, face_idx) — multi-body support

    def __post_init__(self):
        if self.face_pairs is None:
            # Build from face_indices + source_body_id for backward compat
            if self.face_indices:
                self.face_pairs = [(self.source_body_id, fi) for fi in self.face_indices]
            else:
                self.face_pairs = [(self.source_body_id, self.face_idx)]
        if self.face_indices is None:
            self.face_indices = [fi for _, fi in self.face_pairs
                                 if _ == self.source_body_id]
        if self.face_refs is None:
            self.face_refs = []

    def _resolve_faces_on(self, shape, refs_subset, indices_subset):
        """Resolve a subset of faces on a given shape. Returns list of (idx, b3d_face)."""
        resolved = []
        if refs_subset:
            for ref in refs_subset:
                idx, face = ref.find_in(shape)
                if idx is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp: could not re-locate face "
                        f"(normal={ref.normal}, area={ref.area:.3f})")
                resolved.append((idx, face))
        else:
            all_faces = list(shape.faces())
            for fi in indices_subset:
                if fi >= len(all_faces):
                    raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
                resolved.append((fi, all_faces[fi]))
        return resolved

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """Replay via face_ref resolution + extrude."""
        import numpy as np
        from cad.operations.extrude import extrude_face, _do_extrude_solid
        from build123d import Compound
        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)

        if self.force_new_body:
            # Faces may be on multiple bodies — look up each body's shape.
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCP.TopTools import TopTools_ListOfShape

            # Group face_refs by their position in face_pairs.
            result_solid = None
            refs = self.face_refs if self.face_refs else []
            for i, (bid, fi) in enumerate(self.face_pairs):
                src_shape = history._shape_for_body_at(bid, entry_index)
                if src_shape is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp (force_new_body): no shape for body '{bid}'")
                if i < len(refs):
                    _, face_obj = refs[i].find_in(src_shape)
                    if face_obj is None:
                        raise RuntimeError(
                            f"FaceExtrudeOp: could not re-locate face on body '{bid}'")
                else:
                    all_f = list(src_shape.faces())
                    if fi >= len(all_f):
                        raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
                    face_obj = all_f[fi]
                extruded = _do_extrude_solid(face_obj, self.distance, direction,
                                             start_offset=self.start_offset,
                                             end_offset=self.end_offset)
                if result_solid is None:
                    result_solid = extruded.wrapped
                else:
                    lst_a = TopTools_ListOfShape(); lst_a.Append(result_solid)
                    lst_b = TopTools_ListOfShape(); lst_b.Append(extruded.wrapped)
                    fuse = BRepAlgoAPI_Fuse()
                    fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                    fuse.SetRunParallel(True); fuse.Build()
                    if fuse.IsDone():
                        result_solid = fuse.Shape()

            result = Compound(result_solid)
            entry = history._entries[entry_index]
            child_body_ids = entry.params.get("child_body_ids", [])
            if child_body_ids and history._workspace is not None:
                solids = list(result.solids())
                for i, bid in enumerate(child_body_ids):
                    body = history._workspace.bodies.get(bid)
                    if body is not None:
                        body.source_shape = Compound(solids[i].wrapped) if i < len(solids) else None
            return shape  # source body (shape) is unchanged

        # In-place extrude/self-cut (single body only).
        # Apply each face sequentially; re-resolve refs on updated shape each time.
        current = shape
        refs = list(self.face_refs) if self.face_refs else []
        # For in-place, all pairs must be on the same body (source_body_id).
        same_body_indices = [fi for _, fi in self.face_pairs]
        n = len(refs) if refs else len(same_body_indices)
        for i in range(n):
            if refs:
                fi, _ = refs[i].find_in(current)
                if fi is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp: could not re-locate face "
                        f"(normal={refs[i].normal}, area={refs[i].area:.3f})")
                # Re-resolve remaining refs after this op shifts topology
                if i + 1 < len(refs):
                    refs = refs  # will be re-resolved on next iteration via find_in
            else:
                all_f = list(current.faces())
                fi = same_body_indices[i]
                if fi >= len(all_f):
                    raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
            current = extrude_face(current, fi, self.distance,
                                   direction=direction,
                                   start_offset=self.start_offset,
                                   end_offset=self.end_offset)
        return current

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
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
        import numpy as np
        from cad.operations.extrude import extrude_face, _do_extrude_solid
        from cad.face_ref import FaceRef
        from build123d import Compound

        # Validate source body (used for in-place ops and as primary body).
        mesh = viewport._meshes.get(self.source_body_id)
        if mesh is None:
            raise RuntimeError(f"[Extrude] No mesh for body {self.source_body_id}")
        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Extrude] No shape for body {self.source_body_id}")

        # Build face_refs and face_objs for all (body_id, face_idx) pairs.
        self.face_refs = []
        face_objs = []
        for bid, fi in self.face_pairs:
            bshape = (shape_before if bid == self.source_body_id
                      else viewport.workspace.current_shape(bid))
            if bshape is None:
                raise RuntimeError(f"[Extrude] No shape for body '{bid}'")
            all_f = list(bshape.faces())
            if fi >= len(all_f):
                raise RuntimeError(f"[Extrude] face_idx {fi} out of range on body '{bid}'")
            ref = FaceRef.from_b3d_face(all_f[fi])
            if ref is None:
                raise RuntimeError(f"[Extrude] Face {fi} on body '{bid}' is not planar.")
            self.face_refs.append(ref)
            face_objs.append(all_f[fi])
        # first ref stored on entry for backward compat
        face_ref = self.face_refs[0]

        direction    = (np.array(self.direction, dtype=float)
                        if self.direction is not None else None)
        op_str       = "cut" if self.distance < 0 else "extrude"
        op_params    = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))

        if self.force_new_body:
            distance     = self.distance
            body_id      = self.source_body_id
            start_offset = self.start_offset
            end_offset   = self.end_offset

            def compute():
                from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
                from OCP.TopTools import TopTools_ListOfShape
                result_occ = None
                for fo in face_objs:
                    extruded = _do_extrude_solid(fo, distance, direction,
                                                 start_offset=start_offset,
                                                 end_offset=end_offset)
                    if result_occ is None:
                        result_occ = extruded.wrapped
                    else:
                        lst_a = TopTools_ListOfShape(); lst_a.Append(result_occ)
                        lst_b = TopTools_ListOfShape(); lst_b.Append(extruded.wrapped)
                        fuse = BRepAlgoAPI_Fuse()
                        fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                        fuse.SetRunParallel(True); fuse.Build()
                        if fuse.IsDone():
                            result_occ = fuse.Shape()
                return Compound(result_occ)

            def finalize(shape_after):
                from viewer.vp_extrude import _strip_split_suffix, _next_split_name
                from cad.units import format_op_label as _lbl
                label     = _lbl(op_str, op_params)
                root_name = _strip_split_suffix(viewport.workspace.bodies[body_id].name)
                solids    = list(shape_after.solids())
                op_params["child_body_ids"] = []
                new_bodies = []
                for solid in solids:
                    new_name = _next_split_name(root_name, viewport.workspace)
                    new_body = viewport.workspace.add_body(new_name, Compound(solid.wrapped))
                    new_bodies.append(new_body)
                    op_params["child_body_ids"].append(new_body.id)
                    print(f"[Extrude] New body '{new_name}' ({len(list(solid.faces()))} faces)")
                primary_body = new_bodies[0] if new_bodies else None
                tag_body_id  = primary_body.id if primary_body else body_id
                parent_entry = viewport.history.push(
                    label=label, operation=op_str, params=op_params,
                    body_id=tag_body_id, face_ref=face_ref,
                    shape_before=None, shape_after=shape_after)
                for new_body in new_bodies:
                    new_body.created_at_entry_id = parent_entry.entry_id
                viewport._rebuild_bodies({b.id for b in new_bodies})
                viewport.history_changed.emit()

            return compute, finalize

        # In-place: apply each face sequentially on a snapshot of shape_before.
        distance     = self.distance
        start_offset = self.start_offset
        end_offset   = self.end_offset
        face_refs_snap = list(self.face_refs)

        def compute():
            current = shape_before
            for ref in face_refs_snap:
                fi, _ = ref.find_in(current)
                if fi is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp compute: could not locate face "
                        f"(normal={ref.normal}, area={ref.area:.3f})")
                current = extrude_face(current, fi, distance, direction=direction,
                                       start_offset=start_offset, end_offset=end_offset)
            if not list(current.solids()):
                raise RuntimeError("Boolean result has no solids.")
            return current

        def finalize(shape_after):
            _push_result(viewport, op_str, op_params, self.source_body_id,
                         face_ref, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry = viewport.history.entries[history_idx]
        vp = viewport

        # Seek and rebuild BEFORE showing panel.
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        vp._show_extrude_panel(
            sketch_idx    = None,
            body_id       = self.source_body_id,
            face_idx      = self.face_indices[0] if self.face_indices else self.face_idx,
            editing_entry = entry)

        # Sync viewport face pair list and populate all face entries in panel.
        vp._extrude_face_pairs   = list(self.face_pairs)
        vp._extrude_face_indices = list(self.face_indices)
        vp._extrude_face_idx     = self.face_pairs[0][1] if self.face_pairs else self.face_idx
        vp._extrude_body_id      = self.face_pairs[0][0] if self.face_pairs else self.source_body_id

        panel = vp._extrude_panel
        if panel is None:
            return

        # Re-populate face list for multi-face ops.
        if len(self.face_pairs) > 1:
            panel.clear_face_entries()
            for bid, fi in self.face_pairs:
                body = vp.workspace.bodies.get(bid)
                name = body.name if body else bid
                panel.add_face_entry(bid, fi, f"{name}  ·  face {fi}")
            vp._update_panel_mode_lock(panel, self.face_pairs)

        if self.distance < 0:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)

        if self.direction is not None:
            panel.set_direction(np.array(self.direction, dtype=float))

        if self.start_offset:
            panel._start_offset.set_mm(self.start_offset)
        if self.end_offset:
            panel._end_offset.set_mm(self.end_offset)

        if self.target_vertex is not None:
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(np.array(self.target_vertex, dtype=float))
        else:
            panel._spinbox.set_mm(abs(self.distance))

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "distance":       abs(self.distance),
            "face_idx":       self.face_idx,
            "face_indices":   list(self.face_indices),
            "face_pairs":     [[bid, fi] for bid, fi in self.face_pairs],
            "source_body_id": self.source_body_id,
        }
        if self.face_refs:
            p["face_refs"] = [
                {"normal": list(r.normal), "area": r.area,
                 "centroid_perp": list(r.centroid_perp),
                 "centroid_along": r.centroid_along}
                for r in self.face_refs
            ]
        if self.direction is not None:
            p["direction"] = self.direction
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        if self.force_new_body:
            p["force_new_body"] = True
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "FaceExtrudeOp":
        from cad.face_ref import FaceRef
        dist     = float(params.get("distance", 0)) * sign
        fi       = int(params.get("face_idx", 0))
        src_body = params.get("source_body_id", "")
        raw_pairs   = params.get("face_pairs")
        raw_indices = params.get("face_indices")
        if raw_pairs:
            face_pairs = [(str(p[0]), int(p[1])) for p in raw_pairs]
        elif raw_indices:
            face_pairs = [(src_body, int(i)) for i in raw_indices]
        else:
            face_pairs = [(src_body, fi)]
        face_indices = [i for _, i in face_pairs if _ == src_body] or [fi]
        raw_refs = params.get("face_refs", [])
        face_refs = [
            FaceRef(normal=tuple(r["normal"]), area=r["area"],
                    centroid_perp=tuple(r["centroid_perp"]),
                    centroid_along=r["centroid_along"])
            for r in raw_refs
        ]
        return cls(
            source_body_id = src_body,
            face_idx       = fi,
            face_indices   = face_indices,
            face_pairs     = face_pairs,
            face_refs      = face_refs if face_refs else None,
            distance       = dist,
            direction      = params.get("direction"),
            target_vertex  = params.get("target_vertex"),
            start_offset   = float(params.get("start_offset", 0.0)),
            end_offset     = float(params.get("end_offset", 0.0)),
            force_new_body = bool(params.get("force_new_body", False)),
        )


# ---------------------------------------------------------------------------
# CrossBodyCutOp  —  cut one body using a face from another body (or sketch)
# ---------------------------------------------------------------------------

@dataclass
class CrossBodyCutOp(Op):
    """
    Use a face (or sketch profile) extruded as a tool solid to cut a target body.

    cut_body_id      : body being cut
    source_body_id   : body whose face defines the tool profile
    source_face_idx  : face index on source_body (None if driven by sketch)
    source_sketch_id: entry_id of the driving sketch entry (None if face-driven)
    distance         : depth of the tool solid (always positive; direction encodes side)
    direction        : unit vector the tool extends along, or None
    target_vertex    : world-space point the tool depth is measured to, or None
    start_offset     : mm offset on near end
    end_offset       : mm offset on far end
    """
    cut_body_id:      str
    source_body_id:   str
    source_face_idx:  int  | None = None
    source_sketch_id: str  | None = None
    distance:          float = 0.0
    direction:         list[float] | None = None
    target_vertex:     list[float] | None = None
    start_offset:      float = 0.0
    end_offset:        float = 0.0

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """
        Rebuild the tool solid from the source face/sketch and cut it from shape.
        shape here is the current shape of cut_body_id.
        """
        import numpy as np
        from build123d import Compound, Plane
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.operations.extrude import _do_extrude_solid

        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)
        # Tool direction: negate panel direction (panel gives extrude direction,
        # tool needs to go inward).
        tool_dir = -direction if direction is not None else None

        # Collect faces for the tool solid.
        if self.source_sketch_id is not None:
            sketch_idx = history.id_to_index(self.source_sketch_id)
            if sketch_idx is None:
                raise RuntimeError(
                    f"CrossBodyCutOp: sketch entry '{self.source_sketch_id}' not found")
            sketch_entry = history._entries[sketch_idx].params.get("sketch_entry")
            if sketch_entry is None:
                raise RuntimeError(
                    f"CrossBodyCutOp: no sketch_entry at id '{self.source_sketch_id}'")
            all_faces, _ = sketch_entry.build_faces()
            if not all_faces:
                raise RuntimeError("CrossBodyCutOp: sketch has no closed loops")
            faces = all_faces
        else:
            src_shape = history._shape_for_body_at(
                self.source_body_id, entry_index)
            if src_shape is None:
                raise RuntimeError(
                    f"CrossBodyCutOp: no shape for source body '{self.source_body_id}'")
            entry = history._entries[entry_index]
            face_ref = getattr(entry, "face_ref", None)
            if face_ref is not None:
                _, face_obj = face_ref.find_in(src_shape)
                if face_obj is None:
                    raise RuntimeError(
                        f"CrossBodyCutOp: could not locate source face "
                        f"(normal={face_ref.normal}, area={face_ref.area:.3f})")
                faces = [face_obj]
            else:
                if self.source_face_idx is None:
                    raise RuntimeError("CrossBodyCutOp: no source_face_idx")
                faces = [list(src_shape.faces())[self.source_face_idx]]

        tool_dist = self.distance + 0.01  # epsilon to avoid coincident-face issues

        # Build the tool solid (fuse multiple faces if sketch-driven).
        tool_solid = None
        for face in faces:
            if tool_dir is None:
                z = Plane(face).z_dir
                fd = -np.array([z.X, z.Y, z.Z], dtype=float)
            else:
                fd = tool_dir
            s = _do_extrude_solid(face, tool_dist, fd,
                                  start_offset=self.start_offset, end_offset=self.end_offset)
            if tool_solid is None:
                tool_solid = s
            else:
                lst_a = TopTools_ListOfShape(); lst_a.Append(tool_solid.wrapped)
                lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                fuse = BRepAlgoAPI_Fuse()
                fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                fuse.SetRunParallel(True); fuse.Build()
                if fuse.IsDone():
                    tool_solid = Compound(fuse.Shape())

        lst_a = TopTools_ListOfShape(); lst_a.Append(shape.wrapped)
        lst_b = TopTools_ListOfShape(); lst_b.Append(tool_solid.wrapped)
        op = BRepAlgoAPI_Cut()
        op.SetArguments(lst_a); op.SetTools(lst_b)
        op.SetRunParallel(True); op.Build()
        if not op.IsDone():
            raise RuntimeError("CrossBodyCutOp: boolean cut failed")
        return Compound(op.Shape())

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
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
        import numpy as np
        from build123d import Compound, Plane
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.operations.extrude import _do_extrude_solid

        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)
        tool_dir = -direction if direction is not None else None

        source_face_ref = None
        if self.source_sketch_id is not None:
            sketch_idx = viewport.history.id_to_index(self.source_sketch_id)
            if sketch_idx is None:
                raise RuntimeError(f"[Cut] Sketch entry '{self.source_sketch_id}' not found.")
            entries = viewport.history.entries
            se = entries[sketch_idx].params.get("sketch_entry")
            if se is None:
                raise RuntimeError("[Cut] No sketch_entry found.")
            all_faces = viewport._sketch_faces.get(sketch_idx, [])
            if not all_faces:
                raise RuntimeError("[Cut] Sketch has no faces.")
            fidx  = viewport._selected_sketch_face
            faces = ([all_faces[fidx][0]]
                     if fidx is not None and 0 <= fidx < len(all_faces)
                     else [f[0] for f in all_faces])
        else:
            from cad.face_ref import FaceRef, AnyFaceRef
            src_shape = viewport.workspace.current_shape(self.source_body_id)
            if src_shape is None:
                raise RuntimeError("[Cut] Source body has no shape.")
            if self.source_face_idx is None:
                raise RuntimeError("[Cut] No source_face_idx.")
            src_faces_list = list(src_shape.faces())
            if self.source_face_idx >= len(src_faces_list):
                raise RuntimeError("[Cut] source_face_idx out of range.")
            src_face = src_faces_list[self.source_face_idx]
            source_face_ref = (FaceRef.from_occ_face(src_face.wrapped)
                               or AnyFaceRef.from_occ_face(src_face.wrapped))
            faces = [src_face]

        target_shape = viewport.workspace.current_shape(self.cut_body_id)
        if target_shape is None:
            raise RuntimeError("[Cut] Target body has no shape.")
        original_solid_count = len(list(target_shape.solids()))
        target_occ   = target_shape.wrapped
        tool_dist    = self.distance + 0.01
        start_offset = self.start_offset
        end_offset   = self.end_offset

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        def compute():
            tool_solid = None
            for face in faces:
                if tool_dir is not None:
                    fd = tool_dir
                else:
                    z  = Plane(face).z_dir
                    fd = -np.array([z.X, z.Y, z.Z], dtype=float)
                s = _do_extrude_solid(face, tool_dist, fd,
                                      start_offset=start_offset, end_offset=end_offset)
                if tool_solid is None:
                    tool_solid = s
                else:
                    lst_a = TopTools_ListOfShape(); lst_a.Append(tool_solid.wrapped)
                    lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                    fuse = BRepAlgoAPI_Fuse()
                    fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                    fuse.SetRunParallel(True); fuse.Build()
                    if fuse.IsDone():
                        tool_solid = Compound(fuse.Shape())
            lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
            lst_b = TopTools_ListOfShape(); lst_b.Append(tool_solid.wrapped)
            op = BRepAlgoAPI_Cut()
            op.SetArguments(lst_a); op.SetTools(lst_b)
            op.SetRunParallel(True); op.Build()
            if not op.IsDone():
                raise RuntimeError("Boolean cut failed")
            return Compound(op.Shape())

        def finalize(result):
            _push_result(viewport, "cut", op_params, self.cut_body_id,
                         source_face_ref, target_shape, result, original_solid_count,
                         split_key="split_from")
            viewport._selected_sketch_entry = None
            viewport._selected_sketch_face  = None

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry      = viewport.history.entries[history_idx]
        vp         = viewport
        sketch_idx = (vp.history.id_to_index(self.source_sketch_id)
                      if self.source_sketch_id is not None else None)

        # Seek and rebuild BEFORE showing panel.
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            rebuild_id = (self.source_body_id if sketch_idx is None
                          else entry.body_id)
            vp._rebuild_body_mesh(rebuild_id)
        vp.history_changed.emit()

        if sketch_idx is not None:
            vp._selected_sketch_entry = sketch_idx
            vp._selected_sketch_face  = None

        vp._show_extrude_panel(
            sketch_idx    = sketch_idx,
            body_id       = self.source_body_id if sketch_idx is None else None,
            face_idx      = self.source_face_idx if sketch_idx is None else None,
            editing_entry = entry)

        panel = vp._extrude_panel
        if panel is None:
            return

        panel._radio_cut.setChecked(True)
        panel._on_mode_changed(1)

        if self.cut_body_id in vp.workspace.bodies:
            panel.set_merge_body(
                self.cut_body_id,
                vp.workspace.bodies[self.cut_body_id].name)

        if self.direction is not None:
            panel.set_direction(np.array(self.direction, dtype=float))

        if self.start_offset:
            panel._start_offset.set_mm(self.start_offset)
        if self.end_offset:
            panel._end_offset.set_mm(self.end_offset)

        if self.target_vertex is not None:
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(np.array(self.target_vertex, dtype=float))
        else:
            panel._spinbox.set_mm(self.distance)

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "distance":       self.distance,
            "cut_body_id":    self.cut_body_id,
            "source_body_id": self.source_body_id,
        }
        if self.source_face_idx is not None:
            p["source_face_idx"] = self.source_face_idx
        if self.source_sketch_id is not None:
            p["source_sketch_id"] = self.source_sketch_id
        if self.direction is not None:
            p["direction"] = self.direction
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "CrossBodyCutOp":
        return cls(
            cut_body_id      = params.get("cut_body_id", ""),
            source_body_id   = params.get("source_body_id", ""),
            source_face_idx  = params.get("source_face_idx"),
            source_sketch_id = params.get("source_sketch_id"),
            distance         = float(params.get("distance", 0)),
            direction        = params.get("direction"),
            target_vertex    = params.get("target_vertex"),
            start_offset     = float(params.get("start_offset", 0.0)),
            end_offset       = float(params.get("end_offset", 0.0)),
        )


# ---------------------------------------------------------------------------
# SketchExtrudeOp  —  extrude or cut a sketch profile
# ---------------------------------------------------------------------------

@dataclass
class SketchExtrudeOp(Op):
    """
    Extrude (or cut) a closed-loop profile built from a SketchEntry.

    from_sketch_id  : entry_id of the sketch entry
    distance        : signed mm — positive = extrude, negative = cut
    merge_body_id   : body to fuse/cut into, or None for a new body
    face_indices    : which faces of the sketch to extrude (None = all)
    direction       : unit vector or None
    target_vertex   : world-space depth target, or None
    start_offset    : mm
    end_offset      : mm
    force_new_body  : True → always create a new body
    merged_from     : source body id when fusing into another body (legacy key)
    """
    from_sketch_id: str
    distance:        float
    merge_body_id:   str  | None       = None
    face_indices:    list[int] | None  = None
    face_centroids:  list[list[float]] | None = None  # UV centroids for stable face matching
    direction:       list[float] | None = None
    target_vertex:   list[float] | None = None
    start_offset:    float = 0.0
    end_offset:      float = 0.0
    force_new_body:  bool  = False
    merged_from:     str   | None = None

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """Rebuild profile from sketch and extrude/cut."""
        from cad.operations.extrude import extrude_face_direct
        from build123d import Compound

        sketch_idx = history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"SketchExtrudeOp: sketch entry '{self.from_sketch_id}' not found")
        if sketch_idx >= entry_index:
            raise RuntimeError(
                f"SketchExtrudeOp: sketch entry is after this entry — invalid reorder")

        sketch_entry_rec = history._entries[sketch_idx]
        if sketch_entry_rec.error:
            raise RuntimeError(
                f"SketchExtrudeOp: sketch entry is in an error state")

        se = sketch_entry_rec.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"SketchExtrudeOp: no sketch_entry at id '{self.from_sketch_id}'")

        # Re-project the sketch plane onto the current geometry before building
        # faces — the sketch op may be on a different body_id so replay_from
        # won't have updated it yet.
        if se.plane_source is not None:
            from cad.history import _replay_sketch_entry
            _replay_sketch_entry(se, history, before_index=sketch_idx)

        all_faces, all_regions = se.build_faces()
        if not all_faces:
            raise RuntimeError("SketchExtrudeOp: sketch has no closed loops")

        if self.face_centroids:
            # Match faces by closest UV centroid — stable across sketch edits
            # that reorder faces by area.
            import numpy as np
            centroids = np.array(self.face_centroids, dtype=float)
            faces = []
            for target in centroids:
                best_i, best_d = 0, float('inf')
                for i, region in enumerate(all_regions):
                    outer_uvs = region[0]
                    c = np.array(outer_uvs).mean(axis=0)
                    d = float(np.linalg.norm(c - target))
                    if d < best_d:
                        best_d, best_i = d, i
                faces.append(all_faces[best_i])
        elif self.face_indices is not None:
            faces = [all_faces[i] for i in self.face_indices
                     if i < len(all_faces)]
            if not faces:
                raise RuntimeError("SketchExtrudeOp: stored face indices out of range")
        else:
            faces = all_faces

        signed_dist = -abs(self.distance) if self.distance < 0 else abs(self.distance)

        result = shape
        for face in faces:
            result = extrude_face_direct(result, face, signed_dist,
                                         start_offset=self.start_offset,
                                         end_offset=self.end_offset)

        solids = list(result.solids())
        solids.sort(key=lambda s: s.volume, reverse=True)
        if not solids:
            raise RuntimeError("SketchExtrudeOp: result contains no solids")

        # Update split-body import entries with their new solids.
        this_id = history._entries[entry_index].entry_id
        for j in range(entry_index + 1, len(history._entries)):
            e = history._entries[j]
            if (e.operation == "import" and
                    e.params.get("source_entry_id") == this_id):
                solid_idx = e.params.get("solid_index", 1)
                e.shape_after = solids[solid_idx] if solid_idx < len(solids) else None

        return solids[0]

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
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
        import numpy as np
        from cad.operations.extrude import extrude_face_direct, _do_extrude_solid
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.face_ref import FaceRef

        entries    = viewport.history.entries
        sketch_idx = viewport.history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(f"[Extrude] Sketch entry '{self.from_sketch_id}' not found.")
        se = entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            raise RuntimeError("[Extrude] No sketch entry.")
        all_faces = viewport._sketch_faces.get(sketch_idx, [])
        if not all_faces:
            raise RuntimeError("[Extrude] Sketch has no closed loops.")

        fidx = viewport._selected_sketch_face
        face_indices = ([fidx] if fidx is not None and 0 <= fidx < len(all_faces)
                        else list(range(len(all_faces))))
        faces = [all_faces[i][0] for i in face_indices]

        # Compute UV centroids of selected faces for stable replay matching
        def _uv_centroid(outer_uvs):
            arr = np.array(outer_uvs)
            return arr.mean(axis=0).tolist()
        face_centroids = [_uv_centroid(all_faces[i][1]) for i in face_indices]

        body_id      = se.body_id
        shape_before = viewport.workspace.current_shape(body_id)
        direction    = (np.array(self.direction, dtype=float)
                        if self.direction is not None else None)
        op_str    = "cut" if self.distance < 0 else "extrude"
        op_params = {"distance": abs(self.distance),
                     "from_sketch_id": self.from_sketch_id,
                     "face_indices": face_indices,
                     "face_centroids": face_centroids}
        if self.force_new_body:
            op_params["force_new_body"] = True
        if direction is not None:
            op_params["direction"] = direction.tolist()
        if extra_params:
            op_params.update(extra_params)
        mesh     = viewport._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx]) if mesh else None)
        distance     = self.distance
        start_offset = self.start_offset
        end_offset   = self.end_offset

        # --- merge branch ---
        if self.merge_body_id is not None and not self.force_new_body:
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Extrude] Merge target has no shape.")
            target_occ = target_shape.wrapped
            original_solid_count = len(list(target_shape.solids()))
            merge_params = {"distance": abs(distance), "merged_from": body_id,
                            "from_sketch_id": self.from_sketch_id,
                            "face_indices": face_indices}
            if direction is not None:
                merge_params["direction"] = direction.tolist()
            if extra_params:
                merge_params.update(extra_params)
            merge_body_id = self.merge_body_id

            def compute():
                tool_solid = None
                for face in faces:
                    s = _do_extrude_solid(face, distance, direction,
                                          start_offset=start_offset, end_offset=end_offset)
                    if tool_solid is None:
                        tool_solid = s
                    else:
                        lst_a = TopTools_ListOfShape(); lst_a.Append(tool_solid.wrapped)
                        lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                        fu = BRepAlgoAPI_Fuse()
                        fu.SetArguments(lst_a); fu.SetTools(lst_b)
                        fu.SetRunParallel(True); fu.Build()
                        if fu.IsDone():
                            tool_solid = Compound(fu.Shape())
                lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
                lst_b = TopTools_ListOfShape(); lst_b.Append(tool_solid.wrapped)
                fu = BRepAlgoAPI_Fuse()
                fu.SetArguments(lst_a); fu.SetTools(lst_b)
                fu.SetRunParallel(True); fu.Build()
                if not fu.IsDone():
                    raise RuntimeError("Fuse failed.")
                return Compound(fu.Shape())

            def finalize(merged):
                _push_result(viewport, "extrude", merge_params, merge_body_id,
                             None, target_shape, merged, original_solid_count)
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        # --- force_new_body / normal branch ---
        original_solid_count = len(list(shape_before.solids())) if shape_before else 0
        force_new = self.force_new_body

        def compute():
            result = None if force_new else shape_before
            for face in faces:
                result = extrude_face_direct(result, face, distance, direction=direction,
                                             start_offset=start_offset, end_offset=end_offset)
            if not force_new and not list(result.solids()):
                raise RuntimeError("Sketch extrude produced no solids.")
            return result

        def finalize(shape_after):
            if force_new:
                from viewer.vp_extrude import _strip_split_suffix, _next_split_name
                from cad.units import format_op_label as _lbl
                root_name  = _strip_split_suffix(viewport.workspace.bodies[body_id].name)
                solids     = list(shape_after.solids())
                op_params["child_body_ids"] = []
                new_bodies = []
                for i, solid in enumerate(solids):
                    new_name = _next_split_name(root_name, viewport.workspace)
                    new_body = viewport.workspace.add_body(new_name, Compound(solid.wrapped))
                    new_bodies.append(new_body)
                    op_params["child_body_ids"].append(new_body.id)
                    print(f"[Extrude] New body '{new_name}' ({len(list(solid.faces()))} faces)")
                primary_body = new_bodies[0] if new_bodies else None
                tag_body_id  = primary_body.id if primary_body else body_id
                parent_entry = viewport.history.push(
                    label=_lbl(op_str, op_params), operation=op_str,
                    params=op_params, body_id=tag_body_id, face_ref=face_ref,
                    shape_before=None, shape_after=shape_after)
                for new_body in new_bodies:
                    new_body.created_at_entry_id = parent_entry.entry_id
                viewport._rebuild_bodies({b.id for b in new_bodies})
                viewport.history_changed.emit()
            else:
                _push_result(viewport, op_str, op_params, body_id,
                             face_ref, shape_before, shape_after, original_solid_count)
            viewport._selected_sketch_entry = None
            viewport._selected_sketch_face  = None

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry      = viewport.history.entries[history_idx]
        vp         = viewport
        sketch_idx = vp.history.id_to_index(self.from_sketch_id)

        # Seek and rebuild BEFORE showing panel.
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        vp._selected_sketch_entry = sketch_idx
        face_indices = self.face_indices
        vp._selected_sketch_face  = (face_indices[0]
            if face_indices and len(face_indices) == 1 else None)

        vp._show_extrude_panel(
            sketch_idx    = sketch_idx,
            body_id       = None,
            face_idx      = None,
            editing_entry = entry)

        panel = vp._extrude_panel
        if panel is None:
            return

        if self.distance < 0:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)

        # Restore merge body if applicable.
        if self.merged_from and self.merged_from in vp.workspace.bodies:
            panel.set_merge_body(
                self.merged_from,
                vp.workspace.bodies[self.merged_from].name)
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)

        if self.direction is not None:
            panel.set_direction(np.array(self.direction, dtype=float))

        if self.start_offset:
            panel._start_offset.set_mm(self.start_offset)
        if self.end_offset:
            panel._end_offset.set_mm(self.end_offset)

        if self.target_vertex is not None:
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(np.array(self.target_vertex, dtype=float))
        else:
            panel._spinbox.set_mm(abs(self.distance))

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "distance":       abs(self.distance),
            "from_sketch_id": self.from_sketch_id,
        }
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        if self.merged_from is not None:
            p["merged_from"] = self.merged_from
        if self.face_indices is not None:
            p["face_indices"] = self.face_indices
        if self.face_centroids is not None:
            p["face_centroids"] = self.face_centroids
        if self.direction is not None:
            p["direction"] = self.direction
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        if self.force_new_body:
            p["force_new_body"] = True
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "SketchExtrudeOp":
        dist = float(params.get("distance", 0)) * sign
        return cls(
            from_sketch_id = params["from_sketch_id"],
            distance       = dist,
            merge_body_id  = params.get("merge_body_id"),
            face_indices   = params.get("face_indices"),
            face_centroids = params.get("face_centroids"),
            direction      = params.get("direction"),
            target_vertex  = params.get("target_vertex"),
            start_offset   = float(params.get("start_offset", 0.0)),
            end_offset     = float(params.get("end_offset", 0.0)),
            force_new_body = bool(params.get("force_new_body", False)),
            merged_from    = params.get("merged_from"),
        )


# ---------------------------------------------------------------------------
# SketchOp  —  sketch creation (geometry no-op, stores SketchEntry)
# ---------------------------------------------------------------------------

@dataclass
class SketchOp(Op):
    """Stores the SketchEntry; replay re-projects parametric references."""
    sketch_entry: Any  # cad.sketch.SketchEntry — avoid circular import

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """Sketch is a geometry no-op — just re-project parametric references."""
        from cad.history import _replay_sketch_entry
        se = self.sketch_entry
        if se is not None and se.plane_source is not None:
            ok, err = _replay_sketch_entry(se, history, before_index=entry_index)
            if not ok:
                raise RuntimeError(err)
        return shape  # shape unchanged

    def to_params(self) -> dict:
        return {"sketch_entry": self.sketch_entry}

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "SketchOp":
        return cls(sketch_entry=params.get("sketch_entry"))


# ---------------------------------------------------------------------------
# ImportOp  —  records a split solid or imported body
# ---------------------------------------------------------------------------

@dataclass
class ImportOp(Op):
    """
    Records a solid that was split off from a parent boolean operation,
    or an externally imported body.

    source_entry_id : entry_id of the parent op that produced the split
    split_from      : body_id of the parent body
    solid_index     : which solid in the parent result this body corresponds to
    """
    source_entry_id: str  | None = None
    split_from:      str  | None = None
    solid_index:     int  = 1

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """Import ops carry their shape_after directly; nothing to recompute."""
        entry = history._entries[entry_index]
        return entry.shape_after  # set by parent op's split propagation

    def to_params(self) -> dict:
        p: dict[str, Any] = {}
        if self.source_entry_id is not None:
            p["source_entry_id"] = self.source_entry_id
        if self.split_from is not None:
            p["split_from"] = self.split_from
        if self.solid_index != 1:
            p["solid_index"] = self.solid_index
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "ImportOp":
        return cls(
            source_entry_id = params.get("source_entry_id"),
            split_from      = params.get("split_from"),
            solid_index     = int(params.get("solid_index", 1)),
        )


# ---------------------------------------------------------------------------
# Shared commit helper
# ---------------------------------------------------------------------------

def _push_result(viewport, op_str: str, op_params: dict, body_id: str,
                 face_ref: Any, shape_before: Any, shape_after: Any,
                 original_solid_count: int, split_key: str = "split_from"):
    """
    Push history entry/entries for a boolean result and trigger mesh rebuild.

    If shape_after is None the op failed — push a single diverged entry so it
    appears red in the history panel instead of silently disappearing.

    If the result has more solids than the original (a split), each extra solid
    gets its own 'import' entry and workspace body.  Otherwise a single entry
    is pushed and _post_push_cascade handles mid-history replay.

    Returns the set of affected body IDs.
    """
    from cad.units import format_op_label as _lbl
    from viewer.vp_extrude import _strip_split_suffix, _next_split_name

    label = _lbl(op_str, op_params)

    if shape_after is None:
        err_msg = getattr(viewport, '_pending_op_error', None) or "Operation failed"
        entry = viewport.history.push(
            label=label, operation=op_str, params=op_params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=None)
        entry.error     = True
        entry.error_msg = err_msg
        viewport._post_push_cascade(body_id)
        viewport.history_changed.emit()
        return {body_id}

    solids = list(shape_after.solids())
    solids.sort(key=lambda s: s.volume, reverse=True)
    did_split = len(solids) > original_solid_count

    if did_split:
        parent_entry = viewport.history.push(
            label=label, operation=op_str, params=op_params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=solids[0])
        root_name = _strip_split_suffix(viewport.workspace.bodies[body_id].name)
        for i, solid in enumerate(solids[1:], start=1):
            new_name = _next_split_name(root_name, viewport.workspace)
            new_body = viewport.workspace.add_body(new_name, None)
            viewport.history.push(
                label=f"Split  {new_name}", operation="import",
                params={split_key: body_id,
                        "source_entry_id": parent_entry.entry_id,
                        "solid_index": i},
                body_id=new_body.id, face_ref=None,
                shape_before=None, shape_after=solid)
            print(f"[{op_str.capitalize()}] Split body '{new_name}' "
                  f"({len(list(solid.faces()))} faces)")
        split_ids = {body_id} | {
            e.body_id for e in viewport.history.entries
            if e.operation == "import" and e.params.get(split_key) == body_id
        }
        viewport._rebuild_bodies(split_ids)
    else:
        viewport.history.push(
            label=label, operation=op_str, params=op_params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=shape_after)
        viewport._post_push_cascade(body_id)

    print(f"[{op_str.capitalize()}] body='{viewport.workspace.bodies[body_id].name}' "
          f"({len(solids)} solid(s))")
    viewport.history_changed.emit()
    return {body_id}


# ---------------------------------------------------------------------------
# ThickenOp  —  uniform outward offset of a body
# ---------------------------------------------------------------------------

@dataclass
class ThickenOp(Op):
    """
    Grow/shrink one or more faces on *source_body_id* by *thickness* mm.

    source_body_id : body whose faces are selected
    face_indices   : face indices at commit time (fallback for old entries)
    face_refs      : AnyFaceRef fingerprints — used for stable replay
    thickness      : mm — positive = grow outward, negative = cut inward
    """
    source_body_id: str
    face_indices:   list
    thickness:      float
    face_refs:      list = None   # list of AnyFaceRef, populated at commit

    def __post_init__(self):
        if self.face_refs is None:
            self.face_refs = []

    # Legacy single-face convenience
    @property
    def face_idx(self) -> int:
        return self.face_indices[0] if self.face_indices else 0

    def _resolve_faces(self, shape):
        """Return list of (idx, b3d_face) using face_refs when available."""
        from cad.face_ref import AnyFaceRef
        all_faces = list(shape.faces())
        resolved = []
        if self.face_refs:
            for ref in self.face_refs:
                idx, face = ref.find_in(shape)
                if idx is None:
                    raise RuntimeError(
                        f"ThickenOp: could not re-locate face "
                        f"(centroid={ref.centroid}, area={ref.area:.3f})")
                resolved.append((idx, face))
        else:
            # Fallback for old entries without face_refs
            for idx in self.face_indices:
                if idx >= len(all_faces):
                    raise RuntimeError(f"ThickenOp: face_idx {idx} out of range")
                resolved.append((idx, all_faces[idx]))
        return resolved

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.thicken import thicken_face
        if shape is None:
            src = history._shape_for_body_at(self.source_body_id, entry_index)
            if src is None:
                raise RuntimeError(
                    f"ThickenOp: no shape for source body '{self.source_body_id}'")
            shape = src
        resolved = self._resolve_faces(shape)
        result = shape
        for _idx, face in resolved:
            result = thicken_face(result.wrapped, face.wrapped, self.thickness)
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
        from cad.operations.thicken import thicken_face
        from cad.face_ref import AnyFaceRef

        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Thicken] No shape for body {self.source_body_id}")
        all_faces = list(shape_before.faces())
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"[Thicken] face_idx {idx} out of range")

        # Build face_refs now so execute() can re-locate faces after history edits.
        self.face_refs = [
            AnyFaceRef.from_occ_face(all_faces[idx].wrapped)
            for idx in self.face_indices
        ]

        # Snapshot OCC pointers — safe to capture for the worker thread.
        body_occ   = shape_before.wrapped
        face_occs  = [all_faces[idx].wrapped for idx in self.face_indices]
        thickness  = self.thickness
        op_params  = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))

        def compute():
            current_body = body_occ
            for face_occ in face_occs:
                result = thicken_face(current_body, face_occ, thickness)
                current_body = result.wrapped
            return result

        def finalize(shape_after):
            _push_result(viewport, "thicken", op_params, self.source_body_id,
                         None, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        entry = viewport.history.entries[history_idx]
        vp = viewport
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_body_mesh(self.source_body_id)
        vp.history_changed.emit()
        vp._show_thicken_panel(self.source_body_id, self.face_indices, editing_entry=entry)
        panel = getattr(vp, '_thicken_panel', None)
        if panel is not None:
            panel.set_thickness(self.thickness)
            panel._emit_preview()

    def to_params(self) -> dict:
        p: dict = {
            "source_body_id": self.source_body_id,
            "face_indices":   self.face_indices,
            "thickness":      self.thickness,
        }
        if self.face_refs:
            p["face_refs"] = [
                {"centroid": list(r.centroid), "area": r.area}
                for r in self.face_refs
            ]
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "ThickenOp":
        from cad.face_ref import AnyFaceRef
        # Support old single-face entries (face_idx key) transparently.
        if "face_indices" in params:
            indices = list(params["face_indices"])
        else:
            indices = [int(params.get("face_idx", 0))]
        raw_refs = params.get("face_refs", [])
        face_refs = [
            AnyFaceRef(centroid=tuple(r["centroid"]), area=r["area"])
            for r in raw_refs
        ]
        return cls(
            source_body_id = params.get("source_body_id", ""),
            face_indices   = indices,
            thickness      = float(params.get("thickness", 0)),
            face_refs      = face_refs,
        )


# ---------------------------------------------------------------------------
# FaceRevolveOp  —  revolve a body face around an axis (new body result)
# ---------------------------------------------------------------------------

@dataclass
class FaceRevolveOp(Op):
    """
    Revolve a planar face on a body around a world-space axis.

    source_body_id : body whose face is revolved
    face_idx       : face index at commit time (face_ref used for replay)
    angle_deg      : degrees of revolution
    axis_point     : world-space point on the rotation axis
    axis_dir       : unit direction of the rotation axis
    face_ref       : FaceRef fingerprint for stable re-resolution during replay
    force_new_body : always True — revolve produces a new standalone solid
    """
    source_body_id : str
    face_idx       : int
    angle_deg      : float
    axis_point     : list[float]
    axis_dir       : list[float]
    face_ref       : Any = None
    force_new_body : bool = True

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from cad.operations.revolve import revolve_face_direct
        from build123d import Compound

        src_shape = history._shape_for_body_at(self.source_body_id, entry_index)
        if src_shape is None:
            raise RuntimeError(
                f"FaceRevolveOp: no shape for body '{self.source_body_id}'")

        if self.face_ref is not None:
            _, face_obj = self.face_ref.find_in(src_shape)
            if face_obj is None:
                raise RuntimeError("FaceRevolveOp: could not re-locate face")
        else:
            all_f = list(src_shape.faces())
            if self.face_idx >= len(all_f):
                raise RuntimeError(f"FaceRevolveOp: face_idx {self.face_idx} out of range")
            face_obj = all_f[self.face_idx]

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)
        result = revolve_face_direct(None, face_obj, axis_pt, axis_dir, self.angle_deg)

        entry = history._entries[entry_index]
        child_body_ids = entry.params.get("child_body_ids", [])
        if child_body_ids and history._workspace is not None:
            solids = list(result.solids())
            for i, bid in enumerate(child_body_ids):
                body = history._workspace.bodies.get(bid)
                if body is not None:
                    body.source_shape = (Compound(solids[i].wrapped)
                                         if i < len(solids) else None)
        return shape   # source body shape unchanged

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
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
        import numpy as np
        from cad.operations.revolve import revolve_face_direct
        from cad.face_ref import FaceRef
        from build123d import Compound
        from viewer.vp_extrude import _strip_split_suffix, _next_split_name
        from cad.units import format_op_label as _lbl

        body_id     = self.source_body_id
        shape       = viewport.workspace.current_shape(body_id)
        if shape is None:
            raise RuntimeError(f"[Revolve] Body '{body_id}' has no shape.")

        all_faces = list(shape.faces())
        if self.face_idx >= len(all_faces):
            raise RuntimeError(f"[Revolve] face_idx {self.face_idx} out of range.")
        face_obj = all_faces[self.face_idx]

        mesh     = viewport._meshes.get(body_id)
        face_ref = FaceRef.from_b3d_face(mesh.occt_faces[self.face_idx]) if mesh else None
        self.face_ref = face_ref

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)
        angle    = self.angle_deg

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        root_name = _strip_split_suffix(viewport.workspace.bodies[body_id].name)

        def compute():
            return revolve_face_direct(None, face_obj, axis_pt, axis_dir, angle)

        def finalize(shape_after):
            if shape_after is None:
                err = getattr(viewport, '_pending_op_error', None) or "Revolve failed"
                entry = viewport.history.push(
                    label=_lbl("revolve", op_params), operation="revolve",
                    params=op_params, body_id=body_id, face_ref=face_ref,
                    shape_before=shape, shape_after=None)
                entry.error = True; entry.error_msg = err
                viewport._post_push_cascade(body_id)
                viewport.history_changed.emit()
                return

            solids = list(shape_after.solids())
            new_bodies = []
            for solid in solids:
                new_name = _next_split_name(root_name, viewport.workspace)
                new_body = viewport.workspace.add_body(
                    new_name, Compound(solid.wrapped))
                new_bodies.append(new_body)
            op_params["child_body_ids"] = [b.id for b in new_bodies]

            tag_body = new_bodies[0] if new_bodies else None
            tag_id   = tag_body.id if tag_body else body_id
            parent_entry = viewport.history.push(
                label=_lbl("revolve", op_params), operation="revolve",
                params=op_params, body_id=tag_id, face_ref=face_ref,
                shape_before=shape, shape_after=shape_after)
            for nb in new_bodies:
                nb.created_at_entry_id = parent_entry.entry_id
            viewport._rebuild_bodies({b.id for b in new_bodies})
            viewport.history_changed.emit()

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry = viewport.history.entries[history_idx]
        if history_idx > 0:
            viewport.history.seek(history_idx - 1)
            viewport._rebuild_body_mesh(self.source_body_id)
        viewport.history_changed.emit()
        viewport._show_revolve_panel(
            body_id=self.source_body_id, face_idx=self.face_idx,
            editing_entry=entry)
        panel = getattr(viewport, '_revolve_panel', None)
        if panel is None:
            return
        panel.set_axis(
            np.array(self.axis_point, dtype=float),
            np.array(self.axis_dir,   dtype=float))
        panel._angle_spinbox.set_mm(self.angle_deg)
        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "angle_deg":      self.angle_deg,
            "source_body_id": self.source_body_id,
            "face_idx":       self.face_idx,
            "axis_point":     self.axis_point,
            "axis_dir":       self.axis_dir,
        }
        if self.face_ref is not None:
            r = self.face_ref
            p["face_ref"] = {"normal": list(r.normal), "area": r.area,
                             "centroid_perp": list(r.centroid_perp),
                             "centroid_along": r.centroid_along}
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "FaceRevolveOp":
        face_ref = None
        if "face_ref" in params:
            from cad.face_ref import FaceRef
            r = params["face_ref"]
            face_ref = FaceRef(normal=tuple(r["normal"]), area=r["area"],
                               centroid_perp=tuple(r["centroid_perp"]),
                               centroid_along=r["centroid_along"])
        return cls(
            source_body_id = params["source_body_id"],
            face_idx       = int(params.get("face_idx", 0)),
            angle_deg      = float(params.get("angle_deg", 360.0)),
            axis_point     = params["axis_point"],
            axis_dir       = params["axis_dir"],
            face_ref       = face_ref,
        )


# ---------------------------------------------------------------------------
# SketchRevolveOp  —  revolve a sketch profile around an axis
# ---------------------------------------------------------------------------

@dataclass
class SketchRevolveOp(Op):
    """
    Revolve a closed-loop profile built from a SketchEntry around an axis.

    from_sketch_id : entry_id of the sketch entry
    angle_deg      : revolution arc in degrees (positive = CCW from axis_dir tip)
    axis_point     : world-space point on the rotation axis
    axis_dir       : unit direction of the rotation axis
    merge_body_id  : body to fuse into, or None for a new body
    face_indices   : which faces of the sketch to revolve (None = all)
    face_centroids : UV centroids for stable face matching across sketch edits
    """
    from_sketch_id : str
    angle_deg      : float
    axis_point     : list[float]
    axis_dir       : list[float]
    merge_body_id  : str | None       = None
    face_indices   : list[int] | None = None
    face_centroids : list[list[float]] | None = None
    force_new_body : bool = True   # revolves always produce a standalone solid

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from cad.operations.revolve import revolve_face_direct
        from build123d import Compound

        sketch_idx = history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"SketchRevolveOp: sketch entry '{self.from_sketch_id}' not found")
        if sketch_idx >= entry_index:
            raise RuntimeError(
                "SketchRevolveOp: sketch entry is after this entry — invalid reorder")

        sketch_entry_rec = history._entries[sketch_idx]
        if sketch_entry_rec.error:
            raise RuntimeError("SketchRevolveOp: sketch entry is in an error state")

        se = sketch_entry_rec.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"SketchRevolveOp: no sketch_entry at id '{self.from_sketch_id}'")

        if se.plane_source is not None:
            from cad.history import _replay_sketch_entry
            _replay_sketch_entry(se, history, before_index=sketch_idx)

        all_faces, all_regions = se.build_faces()
        if not all_faces:
            raise RuntimeError("SketchRevolveOp: sketch has no closed loops")

        _MAX_CENTROID_DIST = 1e4   # UV units — any farther means the face is gone

        if self.face_centroids:
            import numpy as np
            centroids = np.array(self.face_centroids, dtype=float)
            faces = []
            for target in centroids:
                best_i, best_d = 0, float('inf')
                for i, region in enumerate(all_regions):
                    outer_uvs = region[0]
                    c = np.array(outer_uvs).mean(axis=0)
                    d = float(np.linalg.norm(c - target))
                    if d < best_d:
                        best_d, best_i = d, i
                if best_d > _MAX_CENTROID_DIST:
                    raise RuntimeError(
                        "SketchRevolveOp: profile face no longer exists in sketch")
                faces.append(all_faces[best_i])
        elif self.face_indices is not None:
            faces = [all_faces[i] for i in self.face_indices if i < len(all_faces)]
            if not faces:
                raise RuntimeError("SketchRevolveOp: stored face indices out of range")
        else:
            faces = all_faces

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)

        result = None   # always build a fresh solid; merge handled in _split_commit
        for face in faces:
            result = revolve_face_direct(result, face, axis_pt, axis_dir, self.angle_deg)

        solids = list(result.solids())
        solids.sort(key=lambda s: s.volume, reverse=True)
        if not solids:
            raise RuntimeError("SketchRevolveOp: result contains no solids")

        this_id = history._entries[entry_index].entry_id
        for j in range(entry_index + 1, len(history._entries)):
            e = history._entries[j]
            if (e.operation == "import" and
                    e.params.get("source_entry_id") == this_id):
                solid_idx = e.params.get("solid_index", 1)
                e.shape_after = solids[solid_idx] if solid_idx < len(solids) else None

        return solids[0]

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
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
        import numpy as np
        from cad.operations.revolve import revolve_face_direct
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.face_ref import FaceRef

        entries    = viewport.history.entries
        sketch_idx = viewport.history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(f"[Revolve] Sketch entry '{self.from_sketch_id}' not found.")
        se = entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            raise RuntimeError("[Revolve] No sketch entry.")
        all_faces = viewport._sketch_faces.get(sketch_idx, [])
        if not all_faces:
            raise RuntimeError("[Revolve] Sketch has no closed loops.")

        fidx = viewport._selected_sketch_face
        face_indices = ([fidx] if fidx is not None and 0 <= fidx < len(all_faces)
                        else list(range(len(all_faces))))
        faces = [all_faces[i][0] for i in face_indices]

        def _uv_centroid(outer_uvs):
            arr = np.array(outer_uvs)
            return arr.mean(axis=0).tolist()
        face_centroids = [_uv_centroid(all_faces[i][1]) for i in face_indices]

        body_id      = se.body_id
        shape_before = viewport.workspace.current_shape(body_id)
        axis_pt      = np.array(self.axis_point, dtype=float)
        axis_dir     = np.array(self.axis_dir,   dtype=float)
        angle_deg    = self.angle_deg

        op_params = {
            "angle_deg":      angle_deg,
            "from_sketch_id": self.from_sketch_id,
            "face_indices":   face_indices,
            "face_centroids": face_centroids,
            "axis_point":     self.axis_point,
            "axis_dir":       self.axis_dir,
        }
        if extra_params:
            op_params.update(extra_params)

        mesh     = viewport._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx]) if mesh else None)

        original_solid_count = len(list(shape_before.solids())) if shape_before else 0
        force_new = self.merge_body_id is None or self.merge_body_id == "__new_body__"

        if not force_new:
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Revolve] Merge target has no shape.")
            target_occ      = target_shape.wrapped
            merge_body_id   = self.merge_body_id
            orig_merge_count = len(list(target_shape.solids()))
            merge_params = dict(op_params)
            merge_params["merge_body_id"] = merge_body_id

            def compute():
                tool = None
                for face in faces:
                    s = revolve_face_direct(None, face, axis_pt, axis_dir, angle_deg)
                    if tool is None:
                        tool = s
                    else:
                        lst_a = TopTools_ListOfShape(); lst_a.Append(tool.wrapped)
                        lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                        fu = BRepAlgoAPI_Fuse()
                        fu.SetArguments(lst_a); fu.SetTools(lst_b)
                        fu.SetRunParallel(True); fu.Build()
                        if fu.IsDone():
                            tool = Compound(fu.Shape())
                lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
                lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
                fu = BRepAlgoAPI_Fuse()
                fu.SetArguments(lst_a); fu.SetTools(lst_b)
                fu.SetRunParallel(True); fu.Build()
                if not fu.IsDone():
                    raise RuntimeError("Revolve fuse failed.")
                return Compound(fu.Shape())

            def finalize(merged):
                _push_result(viewport, "revolve", merge_params, merge_body_id,
                             None, target_shape, merged, orig_merge_count)
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        def compute():
            result = None if force_new else shape_before
            for face in faces:
                result = revolve_face_direct(result, face, axis_pt, axis_dir, angle_deg)
            return result

        def finalize(shape_after):
            if force_new:
                from viewer.vp_extrude import _strip_split_suffix, _next_split_name
                from cad.units import format_op_label as _lbl
                root_name = _strip_split_suffix(viewport.workspace.bodies[body_id].name)
                solids    = list(shape_after.solids())
                op_params["child_body_ids"] = []
                new_bodies = []
                for solid in solids:
                    new_name = _next_split_name(root_name, viewport.workspace)
                    new_body = viewport.workspace.add_body(new_name, Compound(solid.wrapped))
                    new_bodies.append(new_body)
                    op_params["child_body_ids"].append(new_body.id)
                primary_body = new_bodies[0] if new_bodies else None
                tag_body_id  = primary_body.id if primary_body else body_id
                parent_entry = viewport.history.push(
                    label=_lbl("revolve", op_params), operation="revolve",
                    params=op_params, body_id=tag_body_id, face_ref=face_ref,
                    shape_before=None, shape_after=shape_after)
                for nb in new_bodies:
                    nb.created_at_entry_id = parent_entry.entry_id
                viewport._rebuild_bodies({b.id for b in new_bodies})
                viewport.history_changed.emit()
            else:
                _push_result(viewport, "revolve", op_params, body_id,
                             face_ref, shape_before, shape_after, original_solid_count)
            viewport._selected_sketch_entry = None
            viewport._selected_sketch_face  = None

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry      = viewport.history.entries[history_idx]
        vp         = viewport
        sketch_idx = vp.history.id_to_index(self.from_sketch_id)

        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        vp._selected_sketch_entry = sketch_idx
        face_indices = self.face_indices
        vp._selected_sketch_face  = (face_indices[0]
            if face_indices and len(face_indices) == 1 else None)

        vp._show_revolve_panel(sketch_idx=sketch_idx, editing_entry=entry)

        panel = vp._revolve_panel
        if panel is None:
            return

        panel.set_axis(
            np.array(self.axis_point, dtype=float),
            np.array(self.axis_dir,   dtype=float))
        panel._angle_spinbox.set_mm(abs(self.angle_deg))

        if self.merge_body_id and self.merge_body_id in vp.workspace.bodies:
            panel.set_merge_body(
                self.merge_body_id,
                vp.workspace.bodies[self.merge_body_id].name)
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "angle_deg":      self.angle_deg,
            "from_sketch_id": self.from_sketch_id,
            "axis_point":     self.axis_point,
            "axis_dir":       self.axis_dir,
        }
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        if self.face_indices is not None:
            p["face_indices"] = self.face_indices
        if self.face_centroids is not None:
            p["face_centroids"] = self.face_centroids
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "SketchRevolveOp":
        return cls(
            from_sketch_id = params["from_sketch_id"],
            angle_deg      = float(params.get("angle_deg", 360.0)),
            axis_point     = params["axis_point"],
            axis_dir       = params["axis_dir"],
            merge_body_id  = params.get("merge_body_id"),
            face_indices   = params.get("face_indices"),
            face_centroids = params.get("face_centroids"),
        )


# ---------------------------------------------------------------------------
# Dispatch table  (operation string → from_params factory)
# ---------------------------------------------------------------------------

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
