"""
viewer/sketch_modal.py

SketchModalMixin — enter, re-enter, exit, and commit sketch mode.

Keeps sketch lifecycle out of viewport.py.
Expects self to have: camera, workspace, history, _meshes, _sketch,
_selected_sketch_entry, _editing_sketch_history_idx, _sketch_faces,
selection, hover, _rebuild_sketch_faces(), _rebuild_all_meshes(),
_post_push_cascade(), history_changed/selection_changed/sketch_mode_changed
signals.
"""

from __future__ import annotations
from build123d import Plane


def _plane_from_sketch_entry(se):
    """Reconstruct a build123d Plane from a SketchEntry's baked plane cache."""
    return Plane(
        origin = tuple(se.plane_origin.tolist()),
        x_dir  = tuple(se.plane_x_axis.tolist()),
        z_dir  = tuple(se.plane_normal.tolist()),
    )


def _next_body_name(workspace) -> str:
    """Return 'Body N' where N is the lowest unused integer."""
    existing = {body.name for body in workspace.bodies.values()}
    n = 1
    while True:
        name = f"Body {n}"
        if name not in existing:
            return name
        n += 1


class SketchModalMixin:

    def _enter_sketch(self, body_id: str, face_idx: int):
        from cad.sketch import SketchMode
        from cad.face_ref import FaceRef
        from cad.plane_ref import FacePlaneSource
        mesh = self._meshes.get(body_id)
        if mesh is None:
            return
        try:
            b3d_plane = Plane(mesh.occt_faces[face_idx])
        except Exception:
            print("[Sketch] Face is not planar — cannot enter sketch mode.")
            return
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z))
        face_ref     = FaceRef.from_b3d_face(mesh.occt_faces[face_idx])
        plane_source = FacePlaneSource(body_id, face_ref) if face_ref else None
        self._sketch = SketchMode(b3d_plane, body_id, face_idx,
                                  plane_source=plane_source)
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Entered sketch on face {face_idx} of "
              f"{self.workspace.bodies[body_id].name}")

    def _reenter_sketch(self, history_idx: int):
        """Re-open a committed SketchEntry for editing."""
        import copy
        from cad.sketch import SketchMode
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        se = entry.params.get("sketch_entry")
        if se is None:
            return
        b3d_plane = _plane_from_sketch_entry(se)
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z))
        mode = SketchMode(b3d_plane, se.body_id, se.face_idx,
                          plane_source=se.plane_source)
        existing = copy.deepcopy(se.entities)
        mode._entity_snapshots = [existing[:i] for i in range(len(existing))]
        mode.entities = existing
        self._sketch = mode
        self._editing_sketch_history_idx = history_idx
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Re-entered sketch entry {history_idx} "
              f"({len(mode.entities)} entities)")

    def _exit_sketch(self):
        if self._sketch is None:
            return
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        self.update()
        print("[Sketch] Exited sketch mode.")

    def _complete_sketch_delete(self, editing_idx: int):
        """Delete a sketch entry (and cascade) when re-edited to empty."""
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        self.do_delete(editing_idx)
        self._rebuild_sketch_faces()
        self.update()
        print(f"[Sketch] Deleted empty sketch entry {editing_idx}.")

    def _complete_sketch(self):
        if self._sketch is None:
            return
        sketch = self._sketch
        from cad.sketch import LineEntity, ReferenceEntity, SketchEntry
        lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
        refs  = [e for e in sketch.entities if isinstance(e, ReferenceEntity)]
        editing_idx = self._editing_sketch_history_idx
        if not lines and not refs:
            if editing_idx is not None:
                # Empty sketch on re-edit — delete the entry and cascade
                self._complete_sketch_delete(editing_idx)
            else:
                print("[Sketch] Nothing to commit — draw lines or include geometry first.")
            return
        entity_count = len(lines) + len(refs)
        new_se = SketchEntry.from_sketch_mode(sketch)

        if editing_idx is not None:
            entries = self.history.entries
            if editing_idx < len(entries):
                entries[editing_idx].params["sketch_entry"] = new_se
                self._sketch = None
                self._editing_sketch_history_idx = None
                self.sketch_mode_changed.emit(False)
                self._rebuild_sketch_faces()
                ok, err, _ = self.history.replay_from(editing_idx)
                if not ok:
                    print(f"[Sketch] Replay after re-entry failed: {err}")
                self._rebuild_all_meshes()
                self.history_changed.emit()
                self.update()
                print(f"[Sketch] Updated sketch entry {editing_idx}, "
                      f"replayed downstream.")
                return

        body_id = sketch.body_id
        if body_id is None:
            body = self.workspace.add_body(
                _next_body_name(self.workspace), None)
            body_id = body.id
            sketch.body_id = body_id
            new_se.body_id  = body_id

        from cad.face_ref import FaceRef
        mesh = self._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[sketch.face_idx])
                    if mesh and sketch.face_idx >= 0 else None)
        current_shape = self.workspace.current_shape(body_id)
        self.history.push(
            label        = f"Sketch  ({entity_count} entities)",
            operation    = "sketch",
            params       = {"sketch_entry": new_se},
            body_id      = body_id,
            face_ref     = face_ref,
            shape_before = current_shape,
            shape_after  = current_shape,
        )
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        self._rebuild_sketch_faces()
        self._post_push_cascade(body_id)
        self.history_changed.emit()
        self.update()
        print(f"[Sketch] Committed {entity_count} entities to history.")

    def _enter_sketch_on_plane(self, axis: str):
        """Enter sketch mode on one of the three world planes."""
        from cad.sketch import SketchMode
        from cad.plane_ref import WorldPlaneSource
        if self._sketch is not None:
            return
        plane_map = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}
        b3d_plane = plane_map.get(axis)
        if b3d_plane is None:
            return
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z))
        plane_source = WorldPlaneSource(axis)
        self._sketch = SketchMode(b3d_plane, None, -1,
                                  plane_source=plane_source)
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Entered sketch on world plane {axis}")
