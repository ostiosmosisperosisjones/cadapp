"""
cad/history.py

Single flat workspace history with cursor-based undo/redo and parametric replay.
Each entry is tagged with a body_id so operations are scoped to their body.

Parametric replay
-----------------
replay_from(index) re-executes all entries for the target body from *index*
onward.  Two special cases beyond the REGISTRY executors:

  "sketch" entries
      Re-resolve the sketch plane via SketchEntry.plane_source and
      re-project every ReferenceEntity that carries a parametric EdgeSource.

  "extrude" / "cut" entries with params["from_sketch_idx"]
      These are sketch-profile extrudes.  The REGISTRY executor (which works
      by face index) cannot reproduce them correctly.  Instead we rebuild the
      sketch profile faces from the SketchEntry and call extrude_face_direct.
      After the extrude, any split-body import entries whose
      params["source_entry_idx"] matches this entry are updated to receive
      the correct solid from the new result.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any
from cad.face_ref import FaceRef


@dataclass
class HistoryEntry:
    label:        str
    operation:    str
    params:       dict
    body_id:      str        # which body this operation targets
    face_ref:     FaceRef | None
    shape_before: Any        # shape of the body before this op
    shape_after:  Any        # shape of the body after this op
    # Non-destructive history state
    diverged:     bool = False   # True when a branch was inserted before this entry
    error:        bool = False   # True when this entry failed to replay
    error_msg:    str  = ""


class History:
    def __init__(self):
        self._entries: list[HistoryEntry] = []
        self._cursor:  int = -1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, label, operation, params, body_id,
             face_ref, shape_before, shape_after) -> HistoryEntry:
        insert_at = self._cursor + 1
        entry = HistoryEntry(
            label=label, operation=operation, params=params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=shape_after,
        )
        if insert_at < len(self._entries):
            # Inserting into the middle: mark displaced entries as diverged
            # and fix up any absolute-index params that now refer to wrong slots.
            self._entries.insert(insert_at, entry)
            self._fix_index_refs(insert_at)
            for i in range(insert_at + 1, len(self._entries)):
                self._entries[i].diverged = True
            self._cursor = insert_at
            return entry
        # Normal append path
        self._entries.append(entry)
        self._cursor = insert_at
        return entry

    @property
    def is_mid_history(self) -> bool:
        """True when cursor is not at the last entry (diverged entries exist)."""
        return self._cursor < len(self._entries) - 1

    def _fix_index_refs(self, insert_at: int):
        """
        After inserting an entry at *insert_at*, bump any params that store
        absolute history indices (from_sketch_idx, source_entry_idx) when
        those stored indices are >= insert_at.
        """
        for i, e in enumerate(self._entries):
            if i <= insert_at:
                continue
            for key in ("from_sketch_idx", "source_entry_idx"):
                val = e.params.get(key)
                if isinstance(val, int) and val >= insert_at:
                    e.params[key] = val + 1

    def undo(self) -> HistoryEntry | None:
        if self._cursor <= 0:
            return None
        entry = self._entries[self._cursor]
        self._cursor -= 1
        return entry

    def redo(self) -> HistoryEntry | None:
        if self._cursor >= len(self._entries) - 1:
            return None
        self._cursor += 1
        return self._entries[self._cursor]

    def seek(self, index: int) -> bool:
        """Move cursor to index. Returns True if index is valid."""
        if index < 0 or index >= len(self._entries):
            return False
        self._cursor = index
        return True

    # ------------------------------------------------------------------
    # Shape lookup helpers (used by plane/edge sources during replay)
    # ------------------------------------------------------------------

    def _shape_for_body_at(self, body_id: str, before_index: int) -> Any | None:
        """
        Return the most recent shape_after for *body_id* among entries with
        index < before_index.  During replay entries are updated in-order,
        so already-replayed entries carry up-to-date values; entries for
        unreplayed bodies retain their last cached values — both correct.
        """
        result = None
        for i, entry in enumerate(self._entries):
            if i >= before_index:
                break
            if entry.body_id == body_id and entry.shape_after is not None:
                result = entry.shape_after
        return result

    # ------------------------------------------------------------------
    # Parametric replay
    # ------------------------------------------------------------------

    def replay_from(self, index: int) -> tuple[bool, str]:
        """
        Re-execute all entries from *index* onward for the body targeted by
        entry[index].  Entries for other bodies are skipped.

        On failure, the failing entry is marked with error=True/error_msg
        instead of aborting immediately, so the rest of the chain can still
        be attempted.  Returns (all_ok, first_error_message).
        """
        from cad.operations import REGISTRY

        if index < 0 or index >= len(self._entries):
            return False, f"Index {index} out of range"

        target_body_id = self._entries[index].body_id

        # Always derive starting shape from the live shape_after chain rather
        # than entry.shape_before, which can be stale after a mid-history insert.
        current_shape = self._shape_for_body_at(target_body_id, index)
        if current_shape is None and self._entries[index].operation != "import":
            # Fall back to stored shape_before only if the chain lookup fails
            current_shape = self._entries[index].shape_before
        if current_shape is None and self._entries[index].operation != "import":
            return False, f"Entry {index} has no shape_before"

        first_error = ""

        for i in range(index, len(self._entries)):
            entry = self._entries[i]

            if entry.body_id != target_body_id:
                continue

            # Clear previous error state on re-play
            entry.error     = False
            entry.error_msg = ""

            # ---- import ---------------------------------------------------
            if entry.operation == "import":
                current_shape = entry.shape_after
                continue

            # ---- sketch (geometry no-op, but re-project references) -------
            if entry.operation == "sketch":
                se = entry.params.get("sketch_entry")
                if se is not None and se.plane_source is not None:
                    ok, err = _replay_sketch_entry(se, self, before_index=i)
                    if not ok:
                        entry.error     = True
                        entry.error_msg = err
                        if not first_error:
                            first_error = err
                        # Shape unchanged — downstream can still use current_shape
                entry.shape_before = current_shape
                entry.shape_after  = current_shape
                continue

            # ---- sketch-profile extrude -----------------------------------
            if "from_sketch_idx" in entry.params:
                ok, new_shape, err = _replay_sketch_extrude(
                    entry, i, current_shape, self._entries)
                if not ok:
                    entry.error     = True
                    entry.error_msg = err
                    if not first_error:
                        first_error = err
                    # Keep current_shape so downstream sees the last good shape
                    entry.shape_before = current_shape
                    entry.shape_after  = current_shape
                else:
                    entry.shape_before = current_shape
                    entry.shape_after  = new_shape
                    current_shape      = new_shape
                    entry.label        = _make_label(entry.operation, entry.params)
                continue

            # ---- direct face extrude / cut --------------------------------
            executor = REGISTRY.get(entry.operation)
            if executor is None:
                err = f"No executor for operation '{entry.operation}'"
                entry.error = True; entry.error_msg = err
                entry.shape_before = current_shape; entry.shape_after = current_shape
                if not first_error: first_error = err
                continue

            if entry.face_ref is None:
                err = f"Entry {i} '{entry.label}' has no face_ref"
                entry.error = True; entry.error_msg = err
                entry.shape_before = current_shape; entry.shape_after = current_shape
                if not first_error: first_error = err
                continue

            face_idx, _ = entry.face_ref.find_in(current_shape)
            if face_idx is None:
                err = (f"Could not find face for entry {i} '{entry.label}'. "
                       f"FaceRef: normal={entry.face_ref.normal} "
                       f"area={entry.face_ref.area:.3f}")
                entry.error = True; entry.error_msg = err
                entry.shape_before = current_shape; entry.shape_after = current_shape
                if not first_error: first_error = err
                continue

            shape_before = current_shape
            try:
                new_shape = executor(current_shape, face_idx, entry.params)
            except Exception as ex:
                err = f"Entry {i} '{entry.label}' failed: {ex}"
                entry.error = True; entry.error_msg = err
                entry.shape_before = current_shape; entry.shape_after = current_shape
                if not first_error: first_error = err
                continue

            entry.shape_before = shape_before
            entry.shape_after  = new_shape
            current_shape      = new_shape
            entry.label        = _make_label(entry.operation, entry.params)

            # Propagate / nullify split-body shapes.
            # Always scan — if the op no longer produces a split, orphaned
            # split-body entries are nullified so they stop appearing.
            new_solids = list(new_shape.solids())
            for j in range(i + 1, len(self._entries)):
                e = self._entries[j]
                if (e.operation == "import" and
                        e.params.get("source_entry_idx") == i):
                    solid_idx = e.params.get("solid_index", 1)
                    if solid_idx < len(new_solids):
                        e.shape_after = new_solids[solid_idx]
                    else:
                        # Split no longer produces this piece — nullify it
                        e.shape_after = None

        return (not first_error), first_error

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def entries_for_body(self, body_id: str) -> list[tuple[int, HistoryEntry]]:
        """Return [(global_index, entry)] for all entries belonging to body_id."""
        return [(i, e) for i, e in enumerate(self._entries)
                if e.body_id == body_id]

    @property
    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._entries) - 1

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replay_sketch_entry(se, history: History, before_index: int
                         ) -> tuple[bool, str]:
    """
    Update the plane cache and re-project all parametric ReferenceEntity
    instances on a SketchEntry.  Called during replay_from() for sketch ops.
    """
    from cad.sketch import ReferenceEntity

    try:
        b3d_plane = se.plane_source.resolve(history, before_index)
    except Exception as ex:
        return False, f"Sketch plane resolve failed: {ex}"

    o = b3d_plane.origin
    x = b3d_plane.x_dir
    y = b3d_plane.y_dir
    z = b3d_plane.z_dir
    se.plane_origin = np.array([o.X, o.Y, o.Z], dtype=np.float64)
    se.plane_x_axis = np.array([x.X, x.Y, x.Z], dtype=np.float64)
    se.plane_y_axis = np.array([y.X, y.Y, y.Z], dtype=np.float64)
    se.plane_normal = np.array([z.X, z.Y, z.Z], dtype=np.float64)

    for ent in se.entities:
        if not isinstance(ent, ReferenceEntity) or ent.source is None:
            continue
        try:
            world_pts, occ_edges = ent.source.resolve(history, before_index)
        except Exception as ex:
            return False, f"Reference entity re-projection failed: {ex}"

        uv_pts = []
        for wp in world_pts:
            wp_arr = np.array(wp, dtype=np.float64)
            delta  = wp_arr - se.plane_origin
            u = float(np.dot(delta, se.plane_x_axis))
            v = float(np.dot(delta, se.plane_y_axis))
            uv_pts.append(np.array([u, v], dtype=np.float64))
        ent.points = uv_pts

        if occ_edges is not None:
            ent.occ_edges = occ_edges

    return True, ""


def _replay_sketch_extrude(entry, entry_index: int, current_shape,
                            all_entries: list) -> tuple[bool, Any, str]:
    """
    Replay a sketch-profile extrude/cut by rebuilding the profile faces from
    the referenced SketchEntry and calling extrude_face_direct.

    Also updates split-body import entries (identified by
    params["source_entry_idx"] == entry_index) with their new solids.

    Returns (success, new_shape, error_message).
    """
    from cad.operations.extrude import extrude_face_direct

    sketch_idx = entry.params.get("from_sketch_idx")
    if sketch_idx is None or sketch_idx >= len(all_entries):
        return False, None, (
            f"sketch extrude: from_sketch_idx {sketch_idx!r} out of range")

    se = all_entries[sketch_idx].params.get("sketch_entry")
    if se is None:
        return False, None, (
            f"sketch extrude: no sketch_entry at history index {sketch_idx}")

    faces = se.build_faces()
    if not faces:
        return False, None, "sketch extrude: sketch has no closed loops"

    distance = entry.params.get("distance", 0)
    if entry.operation == "cut":
        distance = -abs(distance)

    shape_result = current_shape
    for face in faces:
        try:
            shape_result = extrude_face_direct(shape_result, face, distance)
        except Exception as ex:
            return False, None, f"sketch extrude failed: {ex}"

    solids = list(shape_result.solids())
    if not solids:
        return False, None, "sketch extrude: result contains no solids"

    # Update split-body entries that were created from this extrude
    for j in range(entry_index + 1, len(all_entries)):
        e = all_entries[j]
        if (e.operation == "import" and
                e.params.get("source_entry_idx") == entry_index):
            solid_idx = e.params.get("solid_index", 1)
            if solid_idx < len(solids):
                e.shape_after = solids[solid_idx]

    return True, solids[0], ""


def _make_label(operation: str, params: dict) -> str:
    if operation == "extrude":
        return f"Extrude  +{params.get('distance', 0):.3f}mm"
    if operation == "cut":
        return f"Cut  -{abs(params.get('distance', 0)):.3f}mm"
    return operation.capitalize()
