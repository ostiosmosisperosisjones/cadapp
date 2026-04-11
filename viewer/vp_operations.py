"""
viewer/vp_operations.py

OperationsMixin — extrude, undo/redo, seek, and replay logic.

Keeps geometry-mutation code out of viewport.py.
Expects self to have: history, workspace, _meshes, _rebuild_body_mesh(),
_rebuild_all_meshes(), _sketch_faces, _selected_sketch_entry,
history_changed signal.
"""

from __future__ import annotations


class OperationsMixin:

    # ------------------------------------------------------------------
    # Post-push cascade
    # ------------------------------------------------------------------

    def _post_push_cascade(self, primary_body_id: str | None = None):
        """
        Called after every history.push().

        If the push inserted into mid-history (diverged entries exist after
        cursor), replay the affected body forward so diverged entries get
        correct shapes or error flags, then rebuild all meshes.

        If we're at the tip of history (normal append), just rebuild the
        primary body mesh — cheaper and no replay needed.
        """
        if self.history.is_mid_history:
            cursor = self.history.cursor
            ok, err = self.history.replay_from(cursor)
            if not ok:
                print(f"[Cascade] Replay after insert: {err}")
            self._rebuild_all_meshes()
        else:
            if primary_body_id is not None:
                self._rebuild_body_mesh(primary_body_id)
            else:
                self._rebuild_all_meshes()

    # ------------------------------------------------------------------
    # Undo / Redo / Seek / Replay
    # ------------------------------------------------------------------

    def _do_undo(self):
        entry = self.history.undo()
        if entry is None:
            print("[Undo] Nothing to undo."); return
        print(f"[Undo] Reverting: {entry.label}")
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def _do_redo(self):
        entry = self.history.redo()
        if entry is None:
            print("[Redo] Nothing to redo."); return
        print(f"[Redo] Restoring: {entry.label}")
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def handle_undo(self):
        """
        Context-aware undo: per-sketch entity undo while in sketch mode,
        global history undo otherwise.  Route QAction shortcuts here so the
        menu-level Ctrl+Z doesn't bypass sketch-mode handling.
        """
        if self._sketch is not None:
            if self._sketch.undo_entity():
                print("[Sketch] Undo last entity")
                self.update()
            else:
                print("[Sketch] Nothing to undo in sketch")
        else:
            self._do_undo()

    def handle_redo(self):
        """Context-aware redo — no-op inside a sketch session."""
        if self._sketch is not None:
            return
        self._do_redo()

    def seek_history(self, index: int):
        if self.history.seek(index):
            self._rebuild_all_meshes()
            self.history_changed.emit()

    def do_replay(self, from_index: int):
        print(f"[Replay] Replaying from entry {from_index}…")
        ok, err = self.history.replay_from(from_index)
        if not ok:
            print(f"[Replay] FAILED: {err}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Replay failed", err)
            return
        self._rebuild_all_meshes()
        self.history_changed.emit()
        print("[Replay] Done.")

    # ------------------------------------------------------------------
    # Extrude dispatch
    # ------------------------------------------------------------------

    def _try_extrude(self):
        if self._selected_sketch_entry is not None:
            self._do_sketch_extrude_dialog(self._selected_sketch_entry)
            return
        if self.selection.face_count == 0:
            print("[Extrude] No face or sketch selected."); return
        sf = self.selection.single_face or self.selection.faces[0]
        cb = self.request_extrude_distance
        if cb:
            cb(sf.body_id, sf.face_idx)
        else:
            self._do_extrude_dialog(sf.body_id, sf.face_idx)

    def _do_extrude_dialog(self, body_id, face_idx):
        from PyQt6.QtWidgets import QInputDialog
        dist, ok = QInputDialog.getDouble(
            self, "Extrude",
            "Distance (positive = add material, negative = cut):",
            value=5.0, min=-1000.0, max=1000.0, decimals=3)
        if ok:
            self.do_extrude(body_id, face_idx, dist)

    def _do_sketch_extrude_dialog(self, history_idx: int):
        from PyQt6.QtWidgets import QInputDialog
        dist, ok = QInputDialog.getDouble(
            self, "Extrude Sketch",
            "Distance (positive = add material, negative = cut):",
            value=5.0, min=-1000.0, max=1000.0, decimals=3)
        if ok:
            self.do_extrude_sketch(history_idx, dist)

    # ------------------------------------------------------------------
    # Face extrude
    # ------------------------------------------------------------------

    def do_extrude(self, body_id: str, face_idx: int, distance: float):
        from cad.operations import extrude_face
        from cad.face_ref import FaceRef
        mesh = self._meshes.get(body_id)
        if mesh is None:
            print(f"[Extrude] No mesh for body {body_id}"); return
        shape_before = self.workspace.current_shape(body_id)
        face_ref     = FaceRef.from_b3d_face(mesh.occt_faces[face_idx])
        if face_ref is None:
            print(f"[Extrude] Face {face_idx} is not planar."); return
        try:
            shape_after = extrude_face(shape_before, face_idx, distance)
        except Exception as ex:
            print(f"[Extrude] FAILED: {ex}"); return
        op    = "cut" if distance < 0 else "extrude"
        label = (f"Cut  -{abs(distance):.3f}mm" if distance < 0
                 else f"Extrude  +{distance:.3f}mm")

        solids = list(shape_after.solids())
        original_solid_count = len(list(shape_before.solids())) if shape_before is not None else 0
        did_split = len(solids) > 1 and len(solids) > original_solid_count

        if did_split:
            # Operation produced disconnected pieces — each becomes its own body
            parent_entry_idx = self.history.cursor + 1
            self.history.push(
                label=label, operation=op,
                params={"distance": abs(distance)},
                body_id=body_id, face_ref=face_ref,
                shape_before=shape_before, shape_after=solids[0])
            root_name = _strip_split_suffix(self.workspace.bodies[body_id].name)
            for i, solid in enumerate(solids[1:], start=1):
                new_name = _next_split_name(root_name, self.workspace)
                new_body = self.workspace.add_body(new_name, None)
                self.history.push(
                    label=f"Split  {new_name}",
                    operation="import",
                    params={"split_from":       body_id,
                            "source_entry_idx": parent_entry_idx,
                            "solid_index":      i},
                    body_id=new_body.id,
                    face_ref=None,
                    shape_before=None,
                    shape_after=solid,
                )
                print(f"[Extrude] Split body '{new_name}' "
                      f"({len(list(solid.faces()))} faces)")
        else:
            self.history.push(
                label=label, operation=op,
                params={"distance": abs(distance)},
                body_id=body_id, face_ref=face_ref,
                shape_before=shape_before, shape_after=shape_after)

        print(f"[Extrude] body={self.workspace.bodies[body_id].name} "
              f"face={face_idx}  distance={distance:+.3f}  OK  "
              f"({len(solids)} solid(s))")

        if did_split:
            # Rebuild all bodies so new split bodies get meshes immediately
            if not self.history.is_mid_history:
                self._rebuild_all_meshes()
            else:
                ok, err = self.history.replay_from(self.history.cursor - len(solids) + 1)
                if not ok:
                    print(f"[Cascade] {err}")
                self._rebuild_all_meshes()
        else:
            self._post_push_cascade(body_id)

        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Sketch extrude
    # ------------------------------------------------------------------

    def do_extrude_sketch(self, history_idx: int, distance: float):
        """
        Extrude all closed-loop faces from a committed SketchEntry.

        If the boolean produces multiple disconnected solids (i.e. the sketch
        profile didn't touch the original body), each extra solid is registered
        as a new body in the workspace and gets its own history entry, so it
        appears in the parts panel and is independently selectable.
        """
        from cad.operations import extrude_face_direct
        from cad.face_ref import FaceRef

        entries = self.history.entries
        if history_idx >= len(entries):
            print("[Extrude] Invalid sketch history index."); return

        entry = entries[history_idx]
        se    = entry.params.get("sketch_entry")
        if se is None:
            print("[Extrude] No sketch entry found."); return

        faces = self._sketch_faces.get(history_idx, [])
        if not faces:
            print("[Extrude] Sketch has no closed loops to extrude."); return

        body_id      = se.body_id
        shape_before = self.workspace.current_shape(body_id)
        shape_after  = shape_before

        for face in faces:
            try:
                shape_after = extrude_face_direct(shape_after, face, distance)
            except Exception as ex:
                print(f"[Extrude] Sketch extrude FAILED: {ex}"); return

        op    = "cut" if distance < 0 else "extrude"
        label = (f"Cut  -{abs(distance):.3f}mm" if distance < 0
                 else f"Extrude  +{distance:.3f}mm")

        mesh     = self._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx])
                    if mesh else None)

        # ------------------------------------------------------------------
        # Detect disconnected solids — split into separate bodies if needed
        # ------------------------------------------------------------------
        solids = list(shape_after.solids())
        original_solid_count = len(list(shape_before.solids())) if shape_before is not None else 0

        if len(solids) > 1 and len(solids) > original_solid_count:
            # First solid stays on the original body.
            # Record the index this push will occupy so split entries can
            # reference it by source_entry_idx for parametric replay.
            # With non-destructive history, push inserts at cursor+1.
            parent_entry_idx = self.history.cursor + 1
            primary_solid = solids[0]
            self.history.push(
                label        = label,
                operation    = op,
                params       = {"distance": abs(distance),
                                "from_sketch_idx": history_idx},
                body_id      = body_id,
                face_ref     = face_ref,
                shape_before = shape_before,
                shape_after  = primary_solid,
            )
            # Remaining solids become new bodies.
            # source_shape=None so current_shape() only returns data once the
            # import entry is at or before the history cursor (prevents the
            # body appearing prematurely when seeking backward).
            root_name = _strip_split_suffix(self.workspace.bodies[body_id].name)
            for i, solid in enumerate(solids[1:], start=1):
                new_name = _next_split_name(root_name, self.workspace)
                new_body = self.workspace.add_body(new_name, None)
                self.history.push(
                    label        = f"Extrude (new body)  {new_name}",
                    operation    = "import",
                    params       = {"from_sketch_idx": history_idx,
                                    "split_from":      body_id,
                                    "source_entry_idx": parent_entry_idx,
                                    "solid_index":      i},
                    body_id      = new_body.id,
                    face_ref     = None,
                    shape_before = None,
                    shape_after  = solid,
                )
                print(f"[Extrude] New body '{new_name}' "
                      f"({len(list(solid.faces()))} faces)")
        else:
            # Single solid or fused result — normal path
            self.history.push(
                label        = label,
                operation    = op,
                params       = {"distance": abs(distance),
                                "from_sketch_idx": history_idx},
                body_id      = body_id,
                face_ref     = face_ref,
                shape_before = shape_before,
                shape_after  = shape_after,
            )

        self._post_push_cascade(body_id)
        self._selected_sketch_entry = None
        print(f"[Extrude] Sketch → body={self.workspace.bodies[body_id].name} "
              f"distance={distance:+.3f}  OK  ({len(solids)} solid(s))")
        self.history_changed.emit()


# ---------------------------------------------------------------------------
# Naming helpers for split bodies
# ---------------------------------------------------------------------------

import re as _re

def _strip_split_suffix(name: str) -> str:
    """Remove trailing '  [N]' split suffix to get the root body name."""
    return _re.sub(r'\s*\[\d+\]\s*$', '', name).strip()


def _next_split_name(root_name: str, workspace) -> str:
    """
    Return the next available '  [N]' name for a split of *root_name*,
    based on names already present in the workspace.
    """
    existing = {body.name for body in workspace.bodies.values()}
    n = 2
    while True:
        candidate = f"{root_name}  [{n}]"
        if candidate not in existing:
            return candidate
        n += 1
