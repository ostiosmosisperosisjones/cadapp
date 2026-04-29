"""
viewer/vp_thicken.py

ThickenMixin — uniform body offset panel, preview, and commit.
"""

from __future__ import annotations
from PyQt6.QtCore import pyqtSlot


class ThickenMixin:

    def _try_thicken(self):
        faces = self.selection.faces
        if not faces:
            print("[Thicken] Select a face first."); return
        body_id = faces[0].body_id
        if any(f.body_id != body_id for f in faces):
            print("[Thicken] All selected faces must be on the same body."); return
        if self.workspace.current_shape(body_id) is None:
            print("[Thicken] Active body has no shape."); return
        face_indices = [f.face_idx for f in faces]
        self._show_thicken_panel(body_id, face_indices)

    def _show_thicken_panel(self, body_id: str, face_indices: list, editing_entry=None):
        from gui.thicken_panel import ThickenPanel
        if getattr(self, '_thicken_panel', None) is not None:
            old = self._thicken_panel
            old._preview_timer.stop()
            try:
                old.preview_changed.disconnect(self._update_thicken_preview)
            except Exception:
                pass
            old.close()
            self._thicken_panel = None

        self._thicken_body_id       = body_id
        self._thicken_face_indices  = list(face_indices)
        self._thicken_preview_token = None
        self._thicken_preview_tris  = None
        self._thicken_preview_edges = None
        self._thicken_picking_face  = False

        panel = ThickenPanel(parent=self)
        panel.thicken_requested.connect(self._on_thicken_ok)
        panel.cancelled.connect(self._close_thicken_panel)
        panel.preview_changed.connect(self._update_thicken_preview)
        panel.face_entry_removed.connect(self._on_thicken_face_removed)

        # Populate initial face entries
        shape = self.workspace.current_shape(body_id)
        all_faces = list(shape.faces()) if shape is not None else []
        for fi in face_indices:
            label = self._thicken_face_label(body_id, fi, all_faces)
            panel.add_face_entry(body_id, fi, label)

        self._thicken_panel = panel
        self._position_thicken_panel()
        panel.show()
        panel.setFocus()
        panel._emit_preview()

    def _thicken_face_label(self, body_id: str, face_idx: int, all_faces: list) -> str:
        body = self.workspace.bodies.get(body_id)
        body_name = body.name if body else body_id
        return f"{body_name} · face {face_idx}"

    def _position_thicken_panel(self):
        p = getattr(self, '_thicken_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_thicken_panel(self):
        panel = getattr(self, '_thicken_panel', None)
        if panel is not None:
            panel._preview_timer.stop()
            panel.end_pick_face()
            try:
                panel.preview_changed.disconnect(self._update_thicken_preview)
            except Exception:
                pass
            try:
                panel.face_entry_removed.disconnect(self._on_thicken_face_removed)
            except Exception:
                pass
            panel.close()
            self._thicken_panel = None
        if getattr(self, '_editing_thicken_idx', None) is not None:
            self._cancel_thicken_edit()
        self._thicken_preview_token = None
        self._thicken_preview_tris  = None
        self._thicken_preview_edges = None
        self._thicken_pending_errors = None
        self.update()

    # ------------------------------------------------------------------
    # Face picking for thicken panel
    # ------------------------------------------------------------------

    def _on_thicken_face_removed(self, index: int):
        indices = getattr(self, '_thicken_face_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
            self._thicken_face_indices = indices
        self._thicken_preview_tris  = None
        self._thicken_preview_edges = None
        panel = getattr(self, '_thicken_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def route_face_pick_for_thicken(self, body_id: str, face_idx: int) -> bool:
        """
        Called by the viewport face-pick path when a thicken panel is open
        and the pick_face button is active.  Returns True if consumed.
        """
        panel = getattr(self, '_thicken_panel', None)
        if panel is None or not panel._picking_face:
            return False

        # Thicken only supports a single body — reject different body.
        thicken_body = getattr(self, '_thicken_body_id', None)
        if thicken_body is not None and body_id != thicken_body:
            print("[Thicken] All faces must be on the same body.")
            return True

        indices = getattr(self, '_thicken_face_indices', [])

        if face_idx in indices:
            # Toggle off — remove
            idx = indices.index(face_idx)
            panel.remove_face_entry(idx)   # emits face_entry_removed → _on_thicken_face_removed
        else:
            indices.append(face_idx)
            self._thicken_face_indices = indices
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            label = self._thicken_face_label(body_id, face_idx, all_faces)
            panel.add_face_entry(body_id, face_idx, label)
            panel._emit_preview()

        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_thicken_preview(self, thickness: float):
        from cad.operations.thicken import thicken_face_preview
        import threading

        body_id      = getattr(self, '_thicken_body_id', None)
        face_indices = getattr(self, '_thicken_face_indices', None)
        if body_id is None or not face_indices:
            self._thicken_preview_tris = None; self.update(); return
        shape = self.workspace.current_shape(body_id)
        if shape is None:
            self._thicken_preview_tris = None; self.update(); return
        all_faces = list(shape.faces())

        token = object()
        self._thicken_preview_token = token

        face_occs = [all_faces[idx].wrapped for idx in face_indices
                     if idx < len(all_faces)]
        if not face_occs:
            self._thicken_preview_tris = None; self.update(); return

        # face_indices snapshot for error reporting back to panel
        face_indices_snap = list(face_indices)

        def _compute():
            # Compute per-face: (result_or_None, error_str_or_None)
            per_face = []
            for fo in face_occs:
                try:
                    per_face.append((thicken_face_preview(fo, thickness), None))
                except Exception as ex:
                    per_face.append((None, str(ex)))

            tris = []
            edges = []
            face_errors: dict[int, str] = {}   # face_indices_snap index → error msg
            from OCP.BRep import BRep_Tool
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GCPnts import GCPnts_UniformAbscissa
            for i, (result, err) in enumerate(per_face):
                if err is not None:
                    face_errors[i] = err
                    continue
                wrapped = result.wrapped
                BRepMesh_IncrementalMesh(wrapped, 0.15)
                exp = TopExp_Explorer(wrapped, TopAbs_FACE)
                while exp.More():
                    face = TopoDS.Face_s(exp.Current())
                    tri = BRep_Tool.Triangulation_s(face, face.Location())
                    if tri is not None:
                        for j in range(1, tri.NbTriangles() + 1):
                            n1, n2, n3 = tri.Triangle(j).Get()
                            for ni in (n1, n2, n3):
                                p = tri.Node(ni)
                                tris.extend((p.X(), p.Y(), p.Z()))
                    exp.Next()
                exp2 = TopExp_Explorer(wrapped, TopAbs_EDGE)
                while exp2.More():
                    try:
                        adaptor = BRepAdaptor_Curve(exp2.Current())
                        disc = GCPnts_UniformAbscissa()
                        disc.Initialize(adaptor, 24)
                        if disc.IsDone() and disc.NbPoints() >= 2:
                            strip = []
                            for pi in range(1, disc.NbPoints() + 1):
                                p = adaptor.Value(disc.Parameter(pi))
                                strip.extend((p.X(), p.Y(), p.Z()))
                            edges.append(strip)
                    except Exception:
                        pass
                    exp2.Next()

            if getattr(self, '_thicken_preview_token', None) is token:
                self._thicken_preview_tris  = tris
                self._thicken_preview_edges = edges
                self._thicken_preview_dist  = thickness
                # Store errors for the main-thread slot to pick up, then invoke it.
                self._thicken_pending_errors = (dict(face_errors), list(face_indices_snap))
                from PyQt6.QtCore import QMetaObject, Qt as _Qt
                QMetaObject.invokeMethod(
                    self, "_apply_thicken_face_errors",
                    _Qt.ConnectionType.QueuedConnection)
                self.update()

        threading.Thread(target=_compute, daemon=True).start()

    def _draw_thicken_preview(self):
        tris  = getattr(self, '_thicken_preview_tris',  None)
        edges = getattr(self, '_thicken_preview_edges', None)
        if not tris and not edges:
            return

        from OpenGL.GL import (
            glDisable, glEnable, glColor4f, glBegin, glEnd, glVertex3f,
            glLineWidth, GL_LIGHTING, GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_TRIANGLES, GL_LINE_STRIP,
            glBlendFunc,
        )

        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        is_cut = getattr(self, '_thicken_preview_dist', 0.0) < 0
        if tris:
            if is_cut:
                glColor4f(0.70, 0.22, 0.22, 0.22)
            else:
                glColor4f(0.22, 0.70, 0.45, 0.22)
            glBegin(GL_TRIANGLES)
            for i in range(0, len(tris), 3):
                glVertex3f(tris[i], tris[i+1], tris[i+2])
            glEnd()

        if edges:
            if is_cut:
                glColor4f(0.90, 0.40, 0.40, 0.80)
            else:
                glColor4f(0.40, 0.90, 0.60, 0.80)
            glLineWidth(1.4)
            for strip in edges:
                glBegin(GL_LINE_STRIP)
                for i in range(0, len(strip), 3):
                    glVertex3f(strip[i], strip[i+1], strip[i+2])
                glEnd()
            glLineWidth(1.0)

        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

    @pyqtSlot()
    def _apply_thicken_face_errors(self):
        pending = getattr(self, '_thicken_pending_errors', None)
        if pending is None:
            return
        face_errors, face_indices_snap = pending
        self._thicken_pending_errors = None
        p = getattr(self, '_thicken_panel', None)
        if p is None:
            return
        p.clear_face_errors()
        for i, msg in face_errors.items():
            if i < len(face_indices_snap):
                p.set_face_entry_error(i, msg)

    def _on_thicken_ok(self, thickness: float):
        body_id      = getattr(self, '_thicken_body_id', None)
        face_indices = getattr(self, '_thicken_face_indices', None)
        editing_idx  = getattr(self, '_editing_thicken_idx', None)
        if editing_idx is not None:
            self._editing_thicken_idx = None  # prevent cancel on close
        self._close_thicken_panel()
        if body_id is None or not face_indices:
            return
        from cad.op_types import ThickenOp
        if editing_idx is not None:
            self._editing_thicken_idx = editing_idx  # restore for _commit
            self._commit_thicken_edit(body_id, face_indices, thickness)
            return
        ThickenOp(source_body_id=body_id, face_indices=face_indices, thickness=thickness).commit_async(self)

    def _commit_thicken_edit(self, body_id: str, face_indices: list, thickness: float):
        from cad.op_types import ThickenOp

        idx = getattr(self, '_editing_thicken_idx', None)
        if idx is None:
            return
        self._editing_thicken_idx = None

        entries = self.history.entries
        if idx >= len(entries):
            return

        entry = entries[idx]
        entry.editing = False

        entry_id      = entry.entry_id
        group_indices = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            if entries[j].params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(entries[j].body_id)

        self.history.seek(max(idx - 1, 0))

        if self.workspace.current_shape(body_id) is None:
            print(f"[Edit] Cannot commit thicken: source body '{body_id}' has no "
                  f"valid shape at this point. Fix upstream errors first.")
            self.history.seek(idx)
            entry.editing = False
            self._rebuild_body_mesh(body_id)
            self.history_changed.emit()
            return

        removable_bodies = group_body_ids - {body_id}
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable_bodies:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        ThickenOp(source_body_id=body_id, face_indices=face_indices,
                  thickness=thickness).commit_async(self)

    def reopen_thicken(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "thicken" or entry.op is None:
            return
        entry.editing = True
        self._editing_thicken_idx     = history_idx
        self._thicken_face_indices    = list(entry.op.face_indices)
        entry.op.reopen(self, history_idx)

    def _cancel_thicken_edit(self):
        idx = getattr(self, '_editing_thicken_idx', None)
        self._editing_thicken_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        body_id = getattr(self, '_thicken_body_id', None)
        if body_id is not None:
            self._rebuild_body_mesh(body_id)
        self.history_changed.emit()
