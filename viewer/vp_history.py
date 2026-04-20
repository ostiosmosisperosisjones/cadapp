"""
viewer/vp_history.py

HistoryMixin — undo/redo/seek/delete/reorder/replay and mesh-rebuild helpers.

Expects self to have:
  history, workspace, _meshes, _body_visible, _hovered_vertex, _hovered_edge,
  _rebuild_body_mesh(), _rebuild_all_meshes(), _rebuild_sketch_faces(),
  selection, history_changed signal.
"""

from __future__ import annotations


class HistoryMixin:

    # ------------------------------------------------------------------
    # Mesh rebuild helpers
    # ------------------------------------------------------------------

    def _post_push_cascade(self, primary_body_id: str | None = None):
        """
        Called after every history.push().

        Mid-history insert: replay the affected body forward so diverged
        entries get correct shapes or error flags, then rebuild affected meshes.
        Tip-of-history append: just rebuild the primary body — cheaper.
        """
        if self.history.is_mid_history:
            cursor = self.history.cursor
            ok, err, mutated = self.history.replay_from(cursor)
            if not ok:
                print(f"[Cascade] Replay after insert: {err}")
            self._rebuild_bodies(mutated or ({primary_body_id} if primary_body_id else set()))
        else:
            if primary_body_id is not None:
                self._rebuild_body_mesh(primary_body_id)
            else:
                self._rebuild_all_meshes()

    def _rebuild_bodies(self, body_ids: set):
        """
        Rebuild meshes for a specific set of body IDs.
        Falls back to _rebuild_all_meshes if the set is empty.
        Tessellates bodies in parallel, then uploads to GL in one pass.
        """
        if not body_ids:
            self._rebuild_all_meshes()
            return
        if len(body_ids) == 1:
            self._rebuild_body_mesh(next(iter(body_ids)))
            return
        from viewer.viewport import _tessellate_parallel
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
        if entry.operation == "sketch":
            self._rebuild_sketch_faces()
        # Rebuild all workspace bodies — undo may hide split children that were
        # created by the reverted entry, and _rebuild_all_meshes respects cursor.
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def _do_redo(self):
        entry = self.history.redo()
        if entry is None:
            print("[Redo] Nothing to redo."); return
        print(f"[Redo] Restoring: {entry.label}")
        if entry.operation == "sketch":
            self._rebuild_sketch_faces()
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def handle_undo(self):
        """
        Context-aware undo: per-sketch entity undo while in sketch mode,
        global history undo otherwise.
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

    # ------------------------------------------------------------------
    # Delete / Reorder
    # ------------------------------------------------------------------

    def do_delete(self, index: int):
        entries = self.history.entries
        if index < 0 or index >= len(entries):
            return
        body_id = entries[index].body_id
        print(f"[History] Deleting entry {index}: '{entries[index].label}'")

        # Collect child bodies (force_new_body) and split-import children
        # before mutating the list — both need GL cleanup.
        child_body_ids = list(entries[index].params.get("child_body_ids", []))
        entry_id = entries[index].entry_id
        split_body_ids = [
            e.body_id for e in entries
            if e.params.get("source_entry_id") == entry_id
        ]
        split_indices = sorted(
            [i for i, e in enumerate(entries)
             if e.params.get("source_entry_id") == entry_id],
            reverse=True
        )

        # Remove entries and bodies.
        for i in split_indices:
            self.history.delete(i)
        for bid in split_body_ids + child_body_ids:
            self.workspace.remove_body(bid)
        self.history.delete(index)

        # Replay the surviving chain for the primary body, then rebuild meshes.
        # Include removed bodies in mutated so their GL buffers get purged.
        removed_body_ids = set(split_body_ids) | set(child_body_ids)
        remaining = [(i, e) for i, e in enumerate(self.history.entries)
                     if e.body_id == body_id]
        mutated = {body_id} | removed_body_ids
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
