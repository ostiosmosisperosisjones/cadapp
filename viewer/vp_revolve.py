"""
viewer/vp_revolve.py

RevolveMixin — panel management, pick routing, and geometry dispatch for
the revolve operation.

Expects self to have:
  history, workspace, _meshes, _sketch_faces, _selected_sketch_entry,
  _selected_sketch_face, selection,
  _rebuild_body_mesh(), _rebuild_bodies(), _post_push_cascade(),
  history_changed signal.
"""

from __future__ import annotations


class RevolveMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _try_revolve(self):
        if self._selected_sketch_entry is not None:
            self._show_revolve_panel(sketch_idx=self._selected_sketch_entry)
            return
        if self.selection.face_count > 0:
            sf = self.selection.single_face or self.selection.faces[0]
            self._show_revolve_panel(body_id=sf.body_id, face_idx=sf.face_idx)
        else:
            self._show_revolve_panel()

    def _show_revolve_panel(self, sketch_idx: int | None = None,
                             body_id: str | None = None,
                             face_idx: int | None = None,
                             editing_entry=None):
        from gui.revolve_panel import RevolvePanel
        if hasattr(self, '_revolve_panel') and self._revolve_panel is not None:
            self._revolve_panel.close()

        self._revolve_sketch_idx = sketch_idx
        self._revolve_body_id    = body_id
        self._revolve_face_pairs = ([(body_id, face_idx)]
                                    if body_id is not None and face_idx is not None
                                    else [])

        panel = RevolvePanel(self.workspace, parent=self)

        if body_id is not None and face_idx is not None:
            body = self.workspace.bodies.get(body_id)
            label = f"{body.name}  ·  face {face_idx}" if body else "⚠  face lost"
            panel.add_face_entry(body_id, face_idx, label,
                                 valid=body is not None)
        elif sketch_idx is not None:
            panel.add_face_entry(None, None, f"Sketch {sketch_idx}")

        panel.revolve_requested.connect(self._on_revolve_panel_ok)
        panel.cancelled.connect(self._close_revolve_panel)
        panel.picking_axis_changed.connect(self._on_revolve_pick_axis)
        panel.picking_face_changed.connect(self._on_revolve_pick_face)
        panel.picking_body_changed.connect(self._on_revolve_pick_body)
        panel.preview_changed.connect(self._on_revolve_preview)

        self._revolve_panel        = panel
        self._revolve_axis_active  = False
        self._revolve_face_active  = False
        self._revolve_body_active  = False
        self._revolve_preview_mesh = None

        self._position_revolve_panel()
        panel.show()
        panel.setFocus()

    def _position_revolve_panel(self):
        p = getattr(self, '_revolve_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_revolve_panel(self):
        if hasattr(self, '_revolve_panel') and self._revolve_panel is not None:
            self._revolve_panel.close()
            self._revolve_panel = None
        self._revolve_axis_active  = False
        self._revolve_face_active  = False
        self._revolve_body_active  = False
        self._revolve_preview_mesh = None
        if getattr(self, '_editing_history_idx', None) is not None:
            self._cancel_revolve_edit()
        self.update()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_revolve_preview(self, angle: float, axis_point, axis_dir):
        if axis_point is None or axis_dir is None or angle == 0:
            self._revolve_preview_mesh = None
            self.update()
            return
        import numpy as np
        from cad.operations.revolve import _do_revolve_solid

        sketch_idx = getattr(self, '_revolve_sketch_idx', None)
        face_pairs = getattr(self, '_revolve_face_pairs', [])
        axis_pt  = np.array(axis_point, dtype=float)
        axis_d   = np.array(axis_dir,   dtype=float)

        try:
            faces = []
            if sketch_idx is not None:
                all_sketch = self._sketch_faces.get(sketch_idx, [])
                if not all_sketch:
                    self._revolve_preview_mesh = None
                    self.update(); return
                fidx_sel = self._selected_sketch_face
                faces = ([all_sketch[fidx_sel][0]]
                         if fidx_sel is not None and 0 <= fidx_sel < len(all_sketch)
                         else [f[0] for f in all_sketch])
            elif face_pairs:
                for bid, fi in face_pairs:
                    shape = self.workspace.current_shape(bid)
                    if shape is None:
                        continue
                    all_f = list(shape.faces())
                    if fi < len(all_f):
                        faces.append(all_f[fi])

            if not faces:
                self._revolve_preview_mesh = None
                self.update(); return

            self._revolve_preview_mesh = [
                _do_revolve_solid(f, axis_pt, axis_d, angle)
                for f in faces
            ]
        except Exception as ex:
            print(f"[Revolve Preview] {ex}")
            self._revolve_preview_mesh = None
        self.update()

    def _draw_revolve_preview(self):
        solids = getattr(self, '_revolve_preview_mesh', None)
        if not solids:
            return

        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCP.TopoDS import TopoDS
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_UniformAbscissa
        from OpenGL.GL import (glDisable, glEnable, glColor4f, glBegin, glEnd,
                               glVertex3f, glLineWidth, glBlendFunc,
                               GL_LIGHTING, GL_DEPTH_TEST, GL_CULL_FACE,
                               GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                               GL_TRIANGLES, GL_LINE_STRIP)

        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        from cad.prefs import prefs as _prefs
        r, g, b = _prefs.op_preview_color
        fill_color = (r, g, b, 0.22)
        edge_color = (min(r + 0.23, 1.0), min(g + 0.30, 1.0), min(b + 0.15, 1.0), 0.80)

        for solid in solids:
            try:
                wrapped = solid.wrapped
                BRepMesh_IncrementalMesh(wrapped, 0.15)

                glColor4f(*fill_color)
                exp = TopExp_Explorer(wrapped, TopAbs_FACE)
                while exp.More():
                    face = TopoDS.Face_s(exp.Current())
                    loc  = face.Location()
                    tri  = BRep_Tool.Triangulation_s(face, loc)
                    if tri is not None:
                        glBegin(GL_TRIANGLES)
                        for i in range(1, tri.NbTriangles() + 1):
                            n1, n2, n3 = tri.Triangle(i).Get()
                            for ni in (n1, n2, n3):
                                p = tri.Node(ni)
                                glVertex3f(p.X(), p.Y(), p.Z())
                        glEnd()
                    exp.Next()

                glColor4f(*edge_color)
                glLineWidth(1.4)
                exp2 = TopExp_Explorer(wrapped, TopAbs_EDGE)
                while exp2.More():
                    edge = exp2.Current()
                    try:
                        adaptor = BRepAdaptor_Curve(edge)
                        disc    = GCPnts_UniformAbscissa()
                        disc.Initialize(adaptor, 32)
                        if disc.IsDone() and disc.NbPoints() >= 2:
                            glBegin(GL_LINE_STRIP)
                            for pi in range(1, disc.NbPoints() + 1):
                                p = adaptor.Value(disc.Parameter(pi))
                                glVertex3f(p.X(), p.Y(), p.Z())
                            glEnd()
                    except Exception:
                        pass
                    exp2.Next()
                glLineWidth(1.0)
            except Exception as ex:
                print(f"[Revolve Preview] draw error: {ex}")

        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Pick routing signals
    # ------------------------------------------------------------------

    def _on_revolve_pick_axis(self, active: bool):
        self._revolve_axis_active = active

    def _on_revolve_pick_face(self, active: bool):
        self._revolve_face_active = active

    def _on_revolve_pick_body(self, active: bool):
        self._revolve_body_active = active

    # ------------------------------------------------------------------
    # Incoming picks from mousePressEvent
    # ------------------------------------------------------------------

    def route_sketch_edge_pick_for_revolve(self, history_idx: int,
                                            entity_idx: int) -> bool:
        """
        Called when the user clicks a committed sketch edge while axis-pick
        mode is active.  Extracts the line's world-space endpoints and passes
        the axis to the panel.
        """
        if not getattr(self, '_revolve_axis_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False

        import numpy as np
        try:
            entries = self.history.entries
            if history_idx >= len(entries):
                return False
            se = entries[history_idx].params.get("sketch_entry")
            if se is None:
                return False
            from cad.sketch import LineEntity
            ent = se.entities[entity_idx]
            if not isinstance(ent, LineEntity):
                print("[Revolve] Axis pick requires a line entity.")
                return False

            origin = np.array(se.plane_origin,  dtype=float)
            x_axis = np.array(se.plane_x_axis,  dtype=float)
            y_axis = np.array(se.plane_y_axis,  dtype=float)

            p0 = origin + float(ent.p0[0]) * x_axis + float(ent.p0[1]) * y_axis
            p1 = origin + float(ent.p1[0]) * x_axis + float(ent.p1[1]) * y_axis
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                print("[Revolve] Degenerate sketch line — cannot use as axis.")
                return False
            direction /= norm

            panel.set_axis(p0, direction)
            return True
        except Exception as ex:
            print(f"[Revolve] Axis pick failed: {ex}")
            return False

    def route_edge_pick_for_revolve(self, edge_idx: int, body_id: str) -> bool:
        """
        Called when the user clicks a body edge while axis-pick mode is active.
        Extracts the edge tangent as the axis direction, midpoint as axis point.
        """
        if not getattr(self, '_revolve_axis_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False

        import numpy as np
        mesh = self._meshes.get(body_id)
        if mesh is None or edge_idx >= len(mesh.topo_edges_occ):
            return False
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            adp = BRepAdaptor_Curve(mesh.topo_edges_occ[edge_idx])
            mid = (adp.FirstParameter() + adp.LastParameter()) * 0.5
            pt  = adp.Value(mid)
            d   = adp.DN(mid, 1)
            axis_pt  = np.array([pt.X(), pt.Y(), pt.Z()], dtype=float)
            axis_dir = np.array([d.X(),  d.Y(),  d.Z()],  dtype=float)
            norm = np.linalg.norm(axis_dir)
            if norm < 1e-10:
                return False
            axis_dir /= norm
        except Exception as ex:
            print(f"[Revolve] Edge axis failed: {ex}")
            return False

        panel.set_axis(axis_pt, axis_dir)
        return True

    def route_face_pick_for_revolve(self, body_id: str, face_idx: int) -> bool:
        if not getattr(self, '_revolve_face_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        label = f"{body.name}  ·  face {face_idx}"
        panel.add_face_entry(body_id, face_idx, label)
        self._revolve_face_pairs.append((body_id, face_idx))
        return True

    def route_body_pick_for_revolve(self, body_id: str) -> bool:
        if not getattr(self, '_revolve_body_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        panel.set_merge_body(body_id, body.name)
        return True

    # ------------------------------------------------------------------
    # Reopen (double-click history entry)
    # ------------------------------------------------------------------

    def reopen_revolve(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "revolve" or entry.op is None:
            return
        entry.editing = True
        self._editing_history_idx = history_idx
        self._editing_body_id     = entry.body_id
        entry.op.reopen(self, history_idx)

    def _cancel_revolve_edit(self):
        idx = getattr(self, '_editing_history_idx', None)
        if idx is None:
            return
        self._editing_history_idx = None
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # OK handler
    # ------------------------------------------------------------------

    def _on_revolve_panel_ok(self, angle_deg: float, axis_point, axis_dir,
                              merge_body_id, _unused):
        from cad.op_types import SketchRevolveOp, FaceRevolveOp

        sketch_idx  = getattr(self, '_revolve_sketch_idx', None)
        face_pairs  = getattr(self, '_revolve_face_pairs', [])
        editing_idx = getattr(self, '_editing_history_idx', None)

        axis_pt  = list(map(float, axis_point))
        axis_d   = list(map(float, axis_dir))
        angle    = float(angle_deg)

        # Prevent _close_revolve_panel from cancelling the edit
        if editing_idx is not None:
            self._editing_history_idx = None

        self._close_revolve_panel()

        if sketch_idx is not None:
            entries = self.history.entries
            if sketch_idx >= len(entries):
                print("[Revolve] Invalid sketch index.")
                return
            force_new = (merge_body_id is None or merge_body_id == "__new_body__")
            new_op = SketchRevolveOp(
                from_sketch_id = entries[sketch_idx].entry_id,
                angle_deg      = angle,
                axis_point     = axis_pt,
                axis_dir       = axis_d,
                merge_body_id  = None if force_new else merge_body_id,
            )
        elif face_pairs:
            body_id, face_idx = face_pairs[0]
            new_op = FaceRevolveOp(
                source_body_id = body_id,
                face_idx       = face_idx,
                angle_deg      = angle,
                axis_point     = axis_pt,
                axis_dir       = axis_d,
            )
        else:
            print("[Revolve] No profile selected.")
            return

        if editing_idx is not None:
            self._commit_revolve_edit(editing_idx, new_op)
        else:
            new_op.commit_async(self)

    def _commit_revolve_edit(self, idx: int, new_op):
        entries = self.history.entries
        if idx >= len(entries):
            return
        entry = entries[idx]
        entry.editing = False
        self._editing_body_id = None

        # Collect group: main entry + any split imports it produced
        entry_id = entry.entry_id
        group_indices  = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            e = entries[j]
            if e.params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(e.body_id)

        # Seek to just before the entry
        self.history.seek(max(idx - 1, 0))

        # Delete group entries + remove any created bodies
        # src_bodies are kept (they pre-exist this op)
        from cad.op_types import SketchRevolveOp
        if isinstance(new_op, SketchRevolveOp):
            sketch_idx = self.history.id_to_index(new_op.from_sketch_id)
            se_entry = (entries[sketch_idx]
                        if sketch_idx is not None and sketch_idx < len(entries) else None)
            se = se_entry.params.get("sketch_entry") if se_entry else None
            src_bodies = {se.body_id} if se else set()
        else:
            src_bodies = {new_op.source_body_id}

        child_body_ids = entry.params.get("child_body_ids", [])
        removable = (group_body_ids - src_bodies) | set(child_body_ids)
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        new_op.commit(self)

        new_idx = self.history.cursor
        ok, err, _ = self.history.replay_all_from(new_idx + 1)
        if not ok:
            print(f"[Revolve Edit] Downstream replay failed: {err}")

        self._rebuild_all_meshes()
        self.history_changed.emit()
