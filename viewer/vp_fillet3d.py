"""
viewer/vp_fillet3d.py

Fillet3DMixin — 3D edge fillet panel, preview, and commit.

Expects self to have:
  history, workspace, _meshes, selection,
  _rebuild_body_mesh(), _post_push_cascade(),
  history_changed signal.
"""

from __future__ import annotations
from PyQt6.QtCore import pyqtSlot, QMetaObject, Qt, Q_ARG


class Fillet3DMixin:

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _try_fillet(self):
        """Open the 3D fillet panel, pre-populated with the current selection."""
        faces = self.selection.faces
        if faces:
            body_id = faces[0].body_id
            if any(f.body_id != body_id for f in faces):
                print("[Fillet] All selected faces must be on the same body.")
                return
            face_indices = [f.face_idx for f in faces]
        else:
            body_id      = None
            face_indices = []
        self._show_fillet3d_panel(body_id, face_indices)

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _show_fillet3d_panel(self, body_id: str | None, face_indices: list,
                              editing_entry=None, radius: float = 1.0):
        from gui.fillet3d_panel import Fillet3DPanel

        if getattr(self, '_fillet3d_panel', None) is not None:
            old = self._fillet3d_panel
            old._preview_timer.stop()
            try:
                old.preview_changed.disconnect(self._update_fillet3d_preview)
            except Exception:
                pass
            old.close()
            self._fillet3d_panel = None

        self._fillet3d_body_id       = body_id
        self._fillet3d_face_indices  = list(face_indices)
        self._fillet3d_preview_token = None
        self._fillet3d_preview_tris  = None
        self._fillet3d_preview_edges = None
        self._fillet3d_pick_active   = False
        self._editing_fillet3d_idx   = (editing_entry and
                                         self.history.entries.index(editing_entry)
                                         if editing_entry is not None else None)

        panel = Fillet3DPanel(self.workspace, parent=self)
        panel.set_radius(radius)

        # Pre-populate face list from initial selection
        if body_id is not None:
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            for fi in face_indices:
                label = self._fillet3d_face_label(body_id, fi, all_faces)
                panel.add_face_entry(body_id, fi, label)

        panel.confirmed.connect(self._on_fillet3d_ok)
        panel.cancelled.connect(self._close_fillet3d_panel)
        panel.preview_changed.connect(self._update_fillet3d_preview)
        panel.face_entry_removed.connect(self._on_fillet3d_face_removed)
        panel.picking_face_changed.connect(self._on_fillet3d_pick_face)

        self._fillet3d_panel = panel
        self._position_fillet3d_panel()
        panel.show()
        panel.setFocus()
        panel._emit_preview()

    def _fillet3d_face_label(self, body_id: str, face_idx: int,
                              all_faces: list) -> str:
        body = self.workspace.bodies.get(body_id)
        name = body.name if body else body_id
        return f"{name}  ·  face {face_idx}"

    def _position_fillet3d_panel(self):
        p = getattr(self, '_fillet3d_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_fillet3d_panel(self):
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is not None:
            panel._preview_timer.stop()
            panel.end_pick_face()
            try:
                panel.preview_changed.disconnect(self._update_fillet3d_preview)
            except Exception:
                pass
            try:
                panel.face_entry_removed.disconnect(self._on_fillet3d_face_removed)
            except Exception:
                pass
            panel.close()
            self._fillet3d_panel = None

        if getattr(self, '_editing_fillet3d_idx', None) is not None:
            self._cancel_fillet3d_edit()

        self._fillet3d_preview_token = None
        self._fillet3d_preview_tris  = None
        self._fillet3d_preview_edges = None
        self._fillet3d_pick_active   = False
        self.update()

    # ------------------------------------------------------------------
    # Face picking
    # ------------------------------------------------------------------

    def _on_fillet3d_pick_face(self, active: bool):
        self._fillet3d_pick_active = active

    def _on_fillet3d_face_removed(self, index: int):
        indices = getattr(self, '_fillet3d_face_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
        self._fillet3d_face_indices = indices
        # If last face removed, clear the body lock too
        if not indices:
            self._fillet3d_body_id = None
        self._fillet3d_preview_tris  = None
        self._fillet3d_preview_edges = None
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def route_face_pick_for_fillet3d(self, body_id: str, face_idx: int) -> bool:
        """Called by mousePressEvent when pick mode is active. Returns True if consumed."""
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is None or not getattr(self, '_fillet3d_pick_active', False):
            return False

        # Lock to the first picked body
        fillet_body = getattr(self, '_fillet3d_body_id', None)
        if fillet_body is None:
            self._fillet3d_body_id = body_id
            fillet_body = body_id
        elif body_id != fillet_body:
            print("[Fillet] All faces must be on the same body.")
            return True

        indices = getattr(self, '_fillet3d_face_indices', [])
        if face_idx in indices:
            # Toggle off — remove
            idx = indices.index(face_idx)
            panel.remove_face_entry(idx)   # emits face_entry_removed
        else:
            indices.append(face_idx)
            self._fillet3d_face_indices = indices
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            label = self._fillet3d_face_label(body_id, face_idx, all_faces)
            panel.add_face_entry(body_id, face_idx, label)
            panel._emit_preview()

        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_fillet3d_preview(self, radius: float):
        from cad.operations.fillet import fillet_face_preview
        import threading

        body_id      = getattr(self, '_fillet3d_body_id', None)
        face_indices = getattr(self, '_fillet3d_face_indices', None)
        if body_id is None or not face_indices:
            self._fillet3d_preview_tris  = None
            self._fillet3d_preview_edges = None
            self.update()
            return

        shape = self.workspace.current_shape(body_id)
        if shape is None:
            self._fillet3d_preview_tris  = None
            self._fillet3d_preview_edges = None
            self.update()
            return

        all_faces = list(shape.faces())

        if radius <= 0:
            self._fillet3d_preview_tris  = None
            self._fillet3d_preview_edges = None
            self.update()
            return

        token = object()
        self._fillet3d_preview_token = token

        def _compute():
            try:
                result = fillet_face_preview(shape, face_indices, all_faces, radius)
            except Exception:
                QMetaObject.invokeMethod(
                    self, "_fillet3d_preview_done",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(object, token),
                    Q_ARG(object, None),
                    Q_ARG(object, None),
                )
                return

            from OCP.BRep import BRep_Tool
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GCPnts import GCPnts_UniformAbscissa

            tris  = []
            edges = []
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

            QMetaObject.invokeMethod(
                self, "_fillet3d_preview_done",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, token),
                Q_ARG(object, tris),
                Q_ARG(object, edges),
            )

        threading.Thread(target=_compute, daemon=True).start()

    @pyqtSlot(object, object, object)
    def _fillet3d_preview_done(self, token, tris, edges):
        if getattr(self, '_fillet3d_preview_token', None) is not token:
            return
        self._fillet3d_preview_tris  = tris
        self._fillet3d_preview_edges = edges
        self.update()

    def _draw_fillet3d_preview(self):
        tris  = getattr(self, '_fillet3d_preview_tris',  None)
        edges = getattr(self, '_fillet3d_preview_edges', None)
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

        if tris:
            glColor4f(0.22, 0.70, 0.45, 0.22)
            glBegin(GL_TRIANGLES)
            for i in range(0, len(tris), 3):
                glVertex3f(tris[i], tris[i+1], tris[i+2])
            glEnd()

        if edges:
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

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    @pyqtSlot(float)
    def _on_fillet3d_ok(self, radius: float):
        body_id      = getattr(self, '_fillet3d_body_id', None)
        face_indices = getattr(self, '_fillet3d_face_indices', None)
        editing_idx  = getattr(self, '_editing_fillet3d_idx', None)

        if editing_idx is not None:
            self._editing_fillet3d_idx = None
        self._close_fillet3d_panel()

        if body_id is None or not face_indices:
            return

        from cad.op_types import FaceFilletOp
        if editing_idx is not None:
            self._editing_fillet3d_idx = editing_idx
            self._commit_fillet3d_edit(body_id, face_indices, radius)
            return
        FaceFilletOp(source_body_id=body_id,
                     face_indices=face_indices,
                     radius=radius).commit_async(self)

    # ------------------------------------------------------------------
    # Edit / reopen
    # ------------------------------------------------------------------

    def reopen_fillet(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "fillet" or entry.op is None:
            return
        op = entry.op
        entry.editing = True
        self._editing_fillet3d_idx = history_idx

        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_body_mesh(op.source_body_id)
        self.history_changed.emit()

        self._show_fillet3d_panel(
            op.source_body_id, list(op.face_indices),
            editing_entry=entry, radius=op.radius)

    def _commit_fillet3d_edit(self, body_id: str, face_indices: list,
                               radius: float):
        from cad.op_types import FaceFilletOp

        idx = getattr(self, '_editing_fillet3d_idx', None)
        if idx is None:
            return
        self._editing_fillet3d_idx = None

        entries = self.history.entries
        if idx >= len(entries):
            return

        entry = entries[idx]
        entry.editing = False
        entry_id = entry.entry_id

        # Collect group (the fillet entry + any split children)
        group_indices  = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            if entries[j].params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(entries[j].body_id)

        self.history.seek(max(idx - 1, 0))

        if self.workspace.current_shape(body_id) is None:
            print(f"[Fillet edit] Source body '{body_id}' has no shape at this point.")
            self.history.seek(idx)
            entry.editing = False
            self._rebuild_body_mesh(body_id)
            self.history_changed.emit()
            return

        removable = group_body_ids - {body_id}
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        FaceFilletOp(source_body_id=body_id,
                     face_indices=face_indices,
                     radius=radius).commit_async(self)

    def _cancel_fillet3d_edit(self):
        idx = getattr(self, '_editing_fillet3d_idx', None)
        self._editing_fillet3d_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        body_id = getattr(self, '_fillet3d_body_id', None)
        if body_id is not None:
            self._rebuild_body_mesh(body_id)
        self.history_changed.emit()
