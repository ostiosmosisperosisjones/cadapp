"""
viewer/vp_operations.py

OperationsMixin — extrude, undo/redo, seek, and replay logic.

Keeps geometry-mutation code out of viewport.py.
Expects self to have: history, workspace, _meshes, _rebuild_body_mesh(),
_rebuild_all_meshes(), _sketch_faces, _selected_sketch_entry,
history_changed signal.
"""

from __future__ import annotations
from cad.units import format_op_label as _make_label


class OperationsMixin:

    # ------------------------------------------------------------------
    # Post-push cascade
    # ------------------------------------------------------------------

    def _post_push_cascade(self, primary_body_id: str | None = None):
        """
        Called after every history.push().

        If the push inserted into mid-history (diverged entries exist after
        cursor), replay the affected body forward so diverged entries get
        correct shapes or error flags, then rebuild only affected meshes.

        If we're at the tip of history (normal append), just rebuild the
        primary body mesh — cheaper and no replay needed.
        """
        if self.history.is_mid_history:
            cursor = self.history.cursor
            ok, err, mutated = self.history.replay_from(cursor)
            if not ok:
                print(f"[Cascade] Replay after insert: {err}")
            self._rebuild_bodies(mutated or {primary_body_id} if primary_body_id else mutated)
        else:
            if primary_body_id is not None:
                self._rebuild_body_mesh(primary_body_id)
            else:
                self._rebuild_all_meshes()

    def _rebuild_bodies(self, body_ids: set):
        """
        Rebuild meshes for a specific set of body IDs.
        Falls back to _rebuild_all_meshes if the set is empty.
        Tessellates all bodies in parallel, then uploads to GL in one pass.
        """
        if not body_ids:
            self._rebuild_all_meshes()
            return
        if len(body_ids) == 1:
            self._rebuild_body_mesh(next(iter(body_ids)))
            return
        # Multiple bodies — tessellate in parallel, upload in one GL context block
        from viewer.viewport import _tessellate_parallel
        from viewer.mesh import Mesh
        from OpenGL.GL import glDeleteBuffers
        bodies = [
            (bid, self.workspace.current_shape(bid))
            for bid in body_ids
            if self.workspace.current_shape(bid) is not None
        ]
        for bid in body_ids:
            self._body_visible.setdefault(bid, True)
        new_meshes = _tessellate_parallel(bodies)

        self.makeCurrent()
        for bid in body_ids:
            old = self._meshes.get(bid)
            if old:
                for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
                    if buf is not None:
                        glDeleteBuffers(1, [buf])
            if bid in new_meshes:
                new_meshes[bid].upload()
                self._meshes[bid] = new_meshes[bid]
            else:
                self._meshes.pop(bid, None)
            self.selection.clear_for_body(bid)
        self._hovered_vertex = (None, None)
        self._hovered_edge   = (None, None)
        self.doneCurrent()
        self._rebuild_sketch_faces()
        self.selection_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Undo / Redo / Seek / Replay
    # ------------------------------------------------------------------

    def _do_undo(self):
        entry = self.history.undo()
        if entry is None:
            print("[Undo] Nothing to undo."); return
        print(f"[Undo] Reverting: {entry.label}")
        self._rebuild_body_mesh(entry.body_id)
        self.history_changed.emit()

    def _do_redo(self):
        entry = self.history.redo()
        if entry is None:
            print("[Redo] Nothing to redo."); return
        print(f"[Redo] Restoring: {entry.label}")
        self._rebuild_body_mesh(entry.body_id)
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
        ok, err, mutated = self.history.replay_from(from_index)
        if not ok:
            print(f"[Replay] FAILED: {err}")
        self._rebuild_bodies(mutated)
        self.history_changed.emit()
        print("[Replay] Done.")

    def do_delete(self, index: int):
        entries = self.history.entries
        if index < 0 or index >= len(entries):
            return
        body_id = entries[index].body_id
        print(f"[History] Deleting entry {index}: '{entries[index].label}'")

        # Collect split-body import entries that were produced by this entry.
        # Must snapshot before any deletions renumber indices.
        split_body_ids = [
            e.body_id for e in entries
            if e.operation == "import" and e.params.get("source_entry_idx") == index
        ]
        split_indices = sorted(
            [i for i, e in enumerate(entries)
             if e.operation == "import" and e.params.get("source_entry_idx") == index],
            reverse=True  # delete back-to-front so earlier indices stay valid
        )

        # Delete split-body import entries first (all come after the parent)
        for i in split_indices:
            self.history.delete(i)

        # Remove split bodies from workspace
        for bid in split_body_ids:
            self.workspace.remove_body(bid)

        # Splits are always pushed after their parent, so deleting them
        # (reverse order, all indices > index) never shifts `index` itself.
        self.history.delete(index)

        # Replay from the first remaining entry for this body, if any
        remaining = [(i, e) for i, e in enumerate(self.history.entries)
                     if e.body_id == body_id]
        mutated = {body_id} | set(split_body_ids)
        if remaining:
            _, _, replay_mutated = self.history.replay_from(remaining[0][0])
            mutated |= replay_mutated
        self._rebuild_bodies(mutated)
        self.history_changed.emit()

    def do_reorder(self, src: int, dst: int):
        entries = self.history.entries
        if src == dst or src < 0 or src >= len(entries):
            return
        print(f"[History] Reordering entry {src} → {dst}")
        self.history.reorder(src, dst)
        # Replay from the earliest affected position for every body touched
        lo = min(src, dst)
        replayed_bodies: set[str] = set()
        all_mutated: set[str] = set()
        for i, e in enumerate(self.history.entries):
            if i < lo:
                continue
            if e.body_id not in replayed_bodies:
                _, _, mutated = self.history.replay_from(i)
                all_mutated |= mutated
                replayed_bodies.add(e.body_id)
        self._rebuild_bodies(all_mutated)
        self.history_changed.emit()

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
        dist = _extrude_distance_dialog(self)
        if dist is not None:
            self.do_extrude(body_id, face_idx, dist)

    def _do_sketch_extrude_dialog(self, history_idx: int):
        dist = _extrude_distance_dialog(self)
        if dist is not None and dist != 0:
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
        params_for_label = {"distance": abs(distance)}
        label = _make_label(op, params_for_label)

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
            if not self.history.is_mid_history:
                # All affected bodies are known — no need for full rebuild
                split_body_ids = {body_id} | {
                    e.body_id for e in self.history.entries
                    if e.operation == "import" and e.params.get("split_from") == body_id
                }
                self._rebuild_bodies(split_body_ids)
            else:
                ok, err, mutated = self.history.replay_from(
                    self.history.cursor - len(solids) + 1)
                if not ok:
                    print(f"[Cascade] {err}")
                self._rebuild_bodies(mutated)
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

        all_faces = self._sketch_faces.get(history_idx, [])
        if not all_faces:
            print("[Extrude] Sketch has no closed loops to extrude."); return

        # If a specific face was clicked, extrude only that one.
        # Otherwise extrude all faces in the sketch.
        fidx = self._selected_sketch_face
        if fidx is not None and 0 <= fidx < len(all_faces):
            faces = [all_faces[fidx][0]]
        else:
            faces = [entry[0] for entry in all_faces]

        body_id      = se.body_id
        shape_before = self.workspace.current_shape(body_id)
        shape_after  = shape_before

        for face in faces:
            try:
                shape_after = extrude_face_direct(shape_after, face, distance)
            except Exception as ex:
                print(f"[Extrude] Sketch extrude FAILED: {ex}"); return

        op    = "cut" if distance < 0 else "extrude"
        params_for_label = {"distance": abs(distance)}
        label = _make_label(op, params_for_label)

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

        if len(solids) > 1:
            # New bodies were created — rebuild all affected bodies at once
            affected = {body_id} | {
                e.body_id for e in self.history.entries
                if e.operation == "import" and e.params.get("split_from") == body_id
            }
            self._rebuild_bodies(affected)
        else:
            self._post_push_cascade(body_id)
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        print(f"[Extrude] Sketch → body={self.workspace.bodies[body_id].name} "
              f"distance={distance:+.3f}  OK  ({len(solids)} solid(s))")
        self.history_changed.emit()


# ---------------------------------------------------------------------------
# Extrude distance dialog
# ---------------------------------------------------------------------------

def _extrude_distance_dialog(parent) -> float | None:
    """
    Modal dialog with an ExprSpinBox for entering an extrude/cut distance.

    Returns the distance in mm (positive = extrude, negative = cut),
    or None if the user cancelled.  Accepts math expressions and unit
    suffixes (e.g. "1in", "2.5cm + 3mm").
    """
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel,
        QDialogButtonBox, QPushButton,
    )
    from PyQt6.QtCore import Qt
    from gui.expr_spinbox import ExprSpinBox
    from cad.prefs import prefs

    dlg = QDialog(parent)
    dlg.setWindowTitle("Extrude / Cut")
    dlg.setMinimumWidth(280)
    dlg.setStyleSheet("background: #1e1e1e; color: #d4d4d4;")

    layout = QVBoxLayout(dlg)
    layout.setSpacing(10)
    layout.setContentsMargins(16, 14, 16, 12)

    lbl = QLabel("Distance  <span style='color:#666; font-size:11px;'>"
                 "(positive = extrude, negative = cut)</span>")
    lbl.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(lbl)

    spinbox = ExprSpinBox(unit=prefs.default_unit)
    layout.addWidget(spinbox)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok |
        QDialogButtonBox.StandardButton.Cancel)
    buttons.setStyleSheet("""
        QPushButton {
            background: #2a2a2a; color: #d4d4d4;
            border: 1px solid #444; border-radius: 3px;
            padding: 4px 14px;
        }
        QPushButton:hover { background: #333; }
        QPushButton:default { border-color: #4a90d9; }
    """)
    def _accept():
        spinbox._on_commit()   # flush any un-committed text before closing
        dlg.accept()

    buttons.accepted.connect(_accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    spinbox.setFocus()

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None

    return spinbox.mm_value()  # None if expression was invalid


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
