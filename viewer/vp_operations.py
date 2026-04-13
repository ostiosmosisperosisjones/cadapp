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
        if entry.operation == "sketch":
            self._rebuild_sketch_faces()
        self.history_changed.emit()

    def _do_redo(self):
        entry = self.history.redo()
        if entry is None:
            print("[Redo] Nothing to redo."); return
        print(f"[Redo] Restoring: {entry.label}")
        self._rebuild_body_mesh(entry.body_id)
        if entry.operation == "sketch":
            self._rebuild_sketch_faces()
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
            self._show_extrude_panel(sketch_idx=self._selected_sketch_entry)
            return
        if self.selection.face_count == 0:
            print("[Extrude] No face or sketch selected."); return
        sf = self.selection.single_face or self.selection.faces[0]
        cb = self.request_extrude_distance
        if cb:
            cb(sf.body_id, sf.face_idx)
        else:
            self._show_extrude_panel(body_id=sf.body_id, face_idx=sf.face_idx)

    def _show_extrude_panel(self, sketch_idx: int | None = None,
                             body_id: str | None = None,
                             face_idx: int | None = None):
        """Show the floating ExtrudePanel over the viewport."""
        import numpy as np
        from gui.extrude_panel import ExtrudePanel
        if hasattr(self, '_extrude_panel') and self._extrude_panel is not None:
            self._extrude_panel.close()

        panel = ExtrudePanel(self.workspace, parent=self)
        # Exclude the sketch's own body from the merge target list
        if sketch_idx is not None:
            entries = self.history.entries
            se = entries[sketch_idx].params.get("sketch_entry") if sketch_idx < len(entries) else None
            panel.refresh_bodies(exclude_body_id=se.body_id if se else None)
        elif body_id is not None:
            panel.refresh_bodies(exclude_body_id=body_id)

        # Supply face origin + normal so vertex projection works
        origin, normal = self._face_origin_and_normal(sketch_idx, body_id, face_idx)
        if origin is not None:
            panel.set_face_origin(origin)
        if normal is not None:
            panel.set_face_normal(normal)

        panel.extrude_requested.connect(
            lambda dist, direction, merge_id: self._on_extrude_panel_ok(
                dist, direction, merge_id, sketch_idx, body_id, face_idx))
        panel.cancelled.connect(self._close_extrude_panel)
        panel.picking_edge_changed.connect(self._on_extrude_pick_edge)
        panel.picking_vertex_changed.connect(self._on_extrude_pick_vertex)
        panel.preview_changed.connect(
            lambda dist, direction: self._update_extrude_preview(
                dist, direction, sketch_idx, body_id, face_idx))

        self._extrude_panel         = panel
        self._extrude_pick_active   = False
        self._extrude_vtx_active    = False
        self._extrude_preview_mesh  = None
        self._position_extrude_panel()
        panel.show()
        panel.setFocus()
        # Trigger initial preview at default distance
        panel._emit_preview()

    def _face_origin_and_normal(self, sketch_idx, body_id, face_idx):
        """Return (origin_np, normal_np) for the face being extruded, or (None, None)."""
        import numpy as np
        try:
            if sketch_idx is not None:
                entries = self.history.entries
                se = entries[sketch_idx].params.get("sketch_entry") if sketch_idx < len(entries) else None
                if se is not None:
                    origin = np.array(se.plane_origin, dtype=float)
                    # normal = cross of plane axes
                    x = np.array(se.plane_x_axis, dtype=float)
                    y = np.array(se.plane_y_axis, dtype=float)
                    normal = np.cross(x, y)
                    n = np.linalg.norm(normal)
                    if n > 1e-10:
                        return origin, normal / n
            elif body_id is not None and face_idx is not None:
                shape = self.workspace.current_shape(body_id)
                if shape is not None:
                    faces = list(shape.faces())
                    if face_idx < len(faces):
                        from build123d import Plane
                        pl = Plane(faces[face_idx])
                        o  = pl.origin
                        z  = pl.z_dir
                        return (np.array([o.X, o.Y, o.Z], dtype=float),
                                np.array([z.X, z.Y, z.Z], dtype=float))
        except Exception:
            pass
        return None, None

    def _position_extrude_panel(self):
        if not hasattr(self, '_extrude_panel') or self._extrude_panel is None:
            return
        p = self._extrude_panel
        margin = 16
        p.move(self.width() - p.width() - margin, margin)

    def _close_extrude_panel(self):
        if hasattr(self, '_extrude_panel') and self._extrude_panel is not None:
            self._extrude_panel.close()
            self._extrude_panel = None
        self._extrude_pick_active  = False
        self._extrude_vtx_active   = False
        self._extrude_preview_mesh = None
        self.update()

    def _update_extrude_preview(self, dist: float, direction,
                                 sketch_idx, body_id, face_idx):
        """Compute the preview solid and trigger a repaint."""
        from cad.operations.extrude import _do_extrude_solid
        from build123d import Face as B3DFace
        try:
            if sketch_idx is not None:
                all_faces = self._sketch_faces.get(sketch_idx, [])
                if not all_faces:
                    self._extrude_preview_mesh = None
                    self.update(); return
                # Use currently-selected face, or first
                fidx_sel = self._selected_sketch_face
                if fidx_sel is not None and 0 <= fidx_sel < len(all_faces):
                    preview_faces = [all_faces[fidx_sel][0]]
                else:
                    preview_faces = [f[0] for f in all_faces]
            elif body_id is not None and face_idx is not None:
                mesh = self._meshes.get(body_id)
                if mesh is None:
                    self._extrude_preview_mesh = None
                    self.update(); return
                shape = self.workspace.current_shape(body_id)
                faces = list(shape.faces())
                if face_idx >= len(faces):
                    self._extrude_preview_mesh = None
                    self.update(); return
                preview_faces = [faces[face_idx]]
            else:
                self._extrude_preview_mesh = None
                self.update(); return

            solids = []
            for face in preview_faces:
                solid = _do_extrude_solid(face, dist, direction)
                solids.append(solid)
            self._extrude_preview_mesh = solids
            self._extrude_preview_dist = dist
        except Exception as ex:
            print(f"[Preview] {ex}")
            self._extrude_preview_mesh = None
        self.update()

    def _on_extrude_pick_edge(self, active: bool):
        self._extrude_pick_active = active

    def _on_extrude_pick_vertex(self, active: bool):
        self._extrude_vtx_active = active

    def route_vertex_pick_for_extrude(self, body_id: str, vertex_idx: int) -> bool:
        """
        Called when the user clicks a vertex while the extrude panel is in
        pick-vertex mode.  Passes the world-space position to the panel.
        """
        import numpy as np
        if not getattr(self, '_extrude_vtx_active', False):
            return False
        mesh = self._meshes.get(body_id)
        if mesh is None or vertex_idx >= len(mesh.topo_verts):
            return False
        pos = np.array(mesh.topo_verts[vertex_idx], dtype=float)
        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            panel.set_vertex_target(pos)
        return True

    def reopen_extrude(self, history_idx: int):
        """
        Called when the user double-clicks an extrude/cut entry.
        Seeks back to just before the entry, then reopens the panel with
        the original parameters pre-filled.
        """
        import numpy as np
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("extrude", "cut"):
            return

        params     = entry.params
        sketch_idx = params.get("from_sketch_idx")
        face_idx   = params.get("face_idx")
        body_id    = entry.body_id
        dist_abs   = float(params.get("distance", 10.0))
        dir_list   = params.get("direction")
        direction  = np.array(dir_list, dtype=float) if dir_list is not None else None

        # Seek to just before this entry, then delete it so OK replaces it
        self.seek_history(history_idx - 1)
        self.do_delete(history_idx)

        # Restore the sketch face selection so the panel knows what to extrude
        if sketch_idx is not None:
            self._selected_sketch_entry = sketch_idx
            face_indices = params.get("face_indices")
            self._selected_sketch_face = (face_indices[0]
                if face_indices and len(face_indices) == 1 else None)

        # Open panel with existing values
        self._show_extrude_panel(sketch_idx=sketch_idx,
                                 body_id=body_id if sketch_idx is None else None,
                                 face_idx=face_idx if sketch_idx is None else None)

        panel = self._extrude_panel
        if panel is None:
            return
        # Restore cut vs extrude mode
        if entry.operation == "cut":
            panel._radio_cut.setChecked(True)

        # Restore vertex target or manual distance
        vtx_list = params.get("target_vertex")
        if vtx_list is not None:
            import numpy as np
            vtx = np.array(vtx_list, dtype=float)
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(vtx)
        else:
            panel._spinbox.set_mm(dist_abs)

        # Restore offsets
        start_off = float(params.get("start_offset", 0.0))
        end_off   = float(params.get("end_offset",   0.0))
        if start_off != 0.0:
            panel._start_offset.set_mm(start_off)
        if end_off != 0.0:
            panel._end_offset.set_mm(end_off)

        # Restore operation (new body vs merge) — new body is the default so only
        # need to set merge when it was explicitly a merge
        if params.get("force_new_body"):
            panel._radio_new.setChecked(True)

        if direction is not None:
            panel.set_direction(direction)
        panel._emit_preview()

    def _on_extrude_panel_ok(self, dist: float, direction, merge_body_id,
                              sketch_idx, body_id, face_idx):
        # Capture vertex/offset state before closing the panel
        panel = self._extrude_panel
        target_vertex = getattr(panel, '_target_vertex', None)
        start_off     = (panel._start_offset.mm_value() or 0.0) if panel else 0.0
        end_off       = (panel._end_offset.mm_value()   or 0.0) if panel else 0.0

        self._close_extrude_panel()
        if dist == 0:
            return

        extra = {}
        if target_vertex is not None:
            extra['target_vertex'] = target_vertex.tolist()
        if start_off != 0.0:
            extra['start_offset'] = start_off
        if end_off != 0.0:
            extra['end_offset'] = end_off

        # Unpack sentinel
        force_new_body = (merge_body_id == "__new_body__")
        real_merge_id  = None if force_new_body else merge_body_id

        if sketch_idx is not None:
            self.do_extrude_sketch(sketch_idx, dist,
                                   direction=direction,
                                   merge_body_id=real_merge_id,
                                   force_new_body=force_new_body,
                                   extra_params=extra)
        elif body_id is not None and face_idx is not None:
            self.do_extrude(body_id, face_idx, dist,
                            direction=direction,
                            merge_body_id=real_merge_id,
                            force_new_body=force_new_body,
                            extra_params=extra)

    # ------------------------------------------------------------------
    # Face extrude
    # ------------------------------------------------------------------

    def do_extrude(self, body_id: str, face_idx: int, distance: float,
                   direction=None, merge_body_id: str | None = None,
                   force_new_body: bool = False,
                   extra_params: dict | None = None):
        from cad.operations import extrude_face, extrude_face_direct
        from cad.face_ref import FaceRef
        from build123d import Compound
        mesh = self._meshes.get(body_id)
        if mesh is None:
            print(f"[Extrude] No mesh for body {body_id}"); return
        shape_before = self.workspace.current_shape(body_id)
        face_ref     = FaceRef.from_b3d_face(mesh.occt_faces[face_idx])
        if face_ref is None:
            print(f"[Extrude] Face {face_idx} is not planar."); return

        op    = "cut" if distance < 0 else "extrude"
        label = _make_label(op, {"distance": abs(distance)})
        op_params = {"distance": abs(distance), "face_idx": face_idx}
        if force_new_body:
            op_params["force_new_body"] = True
        if direction is not None:
            op_params["direction"] = direction.tolist()
        if extra_params:
            op_params.update(extra_params)

        if force_new_body:
            # Extrude the face standalone — no boolean with existing body
            try:
                face_obj = list(shape_before.faces())[face_idx]
                shape_after = extrude_face_direct(None, face_obj, distance,
                                                  direction=direction)
            except Exception as ex:
                print(f"[Extrude] FAILED: {ex}"); return
            root_name  = _strip_split_suffix(self.workspace.bodies[body_id].name)
            new_bodies = []
            for solid in shape_after.solids():
                new_name = _next_split_name(root_name, self.workspace)
                new_body = self.workspace.add_body(new_name, None)
                new_bodies.append(new_body)
                self.history.push(
                    label=label, operation=op, params=op_params,
                    body_id=new_body.id, face_ref=None,
                    shape_before=None, shape_after=Compound(solid.wrapped))
                print(f"[Extrude] New body '{new_name}' "
                      f"({len(list(solid.faces()))} faces)")
            self._rebuild_bodies({b.id for b in new_bodies})
            self.history_changed.emit()
            return

        try:
            shape_after = extrude_face(shape_before, face_idx, distance,
                                       direction=direction)
        except Exception as ex:
            print(f"[Extrude] FAILED: {ex}"); return
        # Merge into another body if requested
        if merge_body_id is not None:
            shape_after = self._merge_into_body(shape_after, merge_body_id,
                                                 distance, body_id)
            if shape_after is None:
                return

        solids = list(shape_after.solids())
        original_solid_count = len(list(shape_before.solids())) if shape_before is not None else 0
        did_split = len(solids) > 1 and len(solids) > original_solid_count

        if did_split:
            # Operation produced disconnected pieces — each becomes its own body
            parent_entry_idx = self.history.cursor + 1
            self.history.push(
                label=label, operation=op,
                params=op_params,
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
                params=op_params,
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

    def do_extrude_sketch(self, history_idx: int, distance: float,
                          direction=None, merge_body_id: str | None = None,
                          force_new_body: bool = False,
                          extra_params: dict | None = None):
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
            face_indices = [fidx]
        else:
            face_indices = list(range(len(all_faces)))

        faces = [all_faces[i][0] for i in face_indices]

        body_id      = se.body_id
        shape_before = self.workspace.current_shape(body_id)

        if force_new_body:
            # Extrude into empty space — no boolean with existing body
            shape_after = None
            for face in faces:
                try:
                    shape_after = extrude_face_direct(shape_after, face, distance,
                                                      direction=direction)
                except Exception as ex:
                    print(f"[Extrude] Sketch extrude FAILED: {ex}"); return
        else:
            shape_after = shape_before
            for face in faces:
                try:
                    shape_after = extrude_face_direct(shape_after, face, distance,
                                                      direction=direction)
                except Exception as ex:
                    print(f"[Extrude] Sketch extrude FAILED: {ex}"); return

            if merge_body_id is not None:
                shape_after = self._merge_into_body(shape_after, merge_body_id,
                                                     distance, body_id)
                if shape_after is None:
                    return

        op    = "cut" if distance < 0 else "extrude"
        label = _make_label(op, {"distance": abs(distance)})
        op_params = {"distance": abs(distance),
                     "from_sketch_idx": history_idx,
                     "face_indices": face_indices}
        if force_new_body:
            op_params["force_new_body"] = True
        if direction is not None:
            op_params["direction"] = direction.tolist()
        if extra_params:
            op_params.update(extra_params)

        mesh     = self._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx])
                    if mesh else None)

        # ------------------------------------------------------------------
        # force_new_body: register result as brand-new body, don't touch body_id
        # ------------------------------------------------------------------
        if force_new_body:
            root_name  = _strip_split_suffix(self.workspace.bodies[body_id].name)
            solids     = list(shape_after.solids())
            new_bodies = []
            for solid in solids:
                new_name = _next_split_name(root_name, self.workspace)
                new_body = self.workspace.add_body(new_name, None)
                new_bodies.append(new_body)
                self.history.push(
                    label        = label,
                    operation    = op,
                    params       = op_params,
                    body_id      = new_body.id,
                    face_ref     = None,
                    shape_before = None,
                    shape_after  = solid,
                )
                print(f"[Extrude] New body '{new_name}' "
                      f"({len(list(solid.faces()))} faces)")
            self._rebuild_bodies({b.id for b in new_bodies})
            self._selected_sketch_entry = None
            self._selected_sketch_face  = None
            self.history_changed.emit()
            return

        # ------------------------------------------------------------------
        # Detect disconnected solids — split into separate bodies if needed
        # ------------------------------------------------------------------
        solids = list(shape_after.solids())
        original_solid_count = len(list(shape_before.solids())) if shape_before is not None else 0

        if len(solids) > 1 and len(solids) > original_solid_count:
            parent_entry_idx = self.history.cursor + 1
            primary_solid = solids[0]
            self.history.push(
                label        = label,
                operation    = op,
                params       = op_params,
                body_id      = body_id,
                face_ref     = face_ref,
                shape_before = shape_before,
                shape_after  = primary_solid,
            )
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
            self.history.push(
                label        = label,
                operation    = op,
                params       = op_params,
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

    # ------------------------------------------------------------------
    # Merge helper
    # ------------------------------------------------------------------

    def _merge_into_body(self, extruded_shape, merge_body_id: str,
                          distance: float, source_body_id: str):
        """
        Fuse *extruded_shape* into the target body and update the target's
        history.  Returns the merged shape, or None on failure.
        """
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape

        target_shape = self.workspace.current_shape(merge_body_id)
        if target_shape is None:
            print(f"[Extrude] Merge target body has no shape."); return None

        try:
            lst_a = TopTools_ListOfShape(); lst_a.Append(target_shape.wrapped)
            lst_b = TopTools_ListOfShape(); lst_b.Append(extruded_shape.wrapped)
            op = BRepAlgoAPI_Fuse()
            op.SetArguments(lst_a)
            op.SetTools(lst_b)
            op.SetRunParallel(True)
            op.Build()
            if not op.IsDone():
                raise RuntimeError("Fuse failed")
            merged = Compound(op.Shape())
        except Exception as ex:
            print(f"[Extrude] Merge failed: {ex}"); return None

        # Push result onto the target body's history
        from cad.units import format_op_label as _lbl
        self.history.push(
            label        = _lbl("extrude", {"distance": abs(distance)}),
            operation    = "extrude",
            params       = {"distance": abs(distance),
                            "merged_from": source_body_id},
            body_id      = merge_body_id,
            face_ref     = None,
            shape_before = target_shape,
            shape_after  = merged,
        )
        self._post_push_cascade(merge_body_id)
        print(f"[Extrude] Merged into body '{self.workspace.bodies[merge_body_id].name}'")
        return merged

    # ------------------------------------------------------------------
    # Edge pick routing (called by viewport mouse handler)
    # ------------------------------------------------------------------

    def route_edge_pick_for_extrude(self, edge_idx: int, body_id: str):
        """
        Called when the user clicks an edge while the extrude panel is in
        pick-edge mode.  Extracts the edge tangent and passes it to the panel.
        """
        import numpy as np
        if not getattr(self, '_extrude_pick_active', False):
            return False
        mesh = self._meshes.get(body_id)
        if mesh is None or edge_idx >= len(mesh.topo_edges_occ):
            return False
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        try:
            adp = BRepAdaptor_Curve(mesh.topo_edges_occ[edge_idx])
            mid = (adp.FirstParameter() + adp.LastParameter()) * 0.5
            d   = adp.DN(mid, 1)  # first derivative = tangent
            vec = np.array([d.X(), d.Y(), d.Z()], dtype=float)
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                return False
            vec /= norm
        except Exception as ex:
            print(f"[Extrude] Edge tangent failed: {ex}"); return False

        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            panel.set_direction(vec)
        return True


# ---------------------------------------------------------------------------
# Extrude distance dialog (kept for external callers / request_extrude_distance)
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
