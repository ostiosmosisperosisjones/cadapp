"""
viewer/viewport.py

Viewport — the main GL widget.  Thin coordinator; logic lives in:

  viewer/renderer.py      — draw_opaque, draw_overlays
  viewer/hover.py         — HoverState
  viewer/sketch_overlay.py— SketchOverlay
  viewer/sketch_pick.py   — SketchPickMixin  (sketch face rebuild/pick/draw)
  viewer/vp_operations.py — OperationsMixin  (extrude, undo/redo, replay)
"""

from __future__ import annotations
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *

from viewer.camera import Camera
from viewer.mesh import Mesh
from viewer.renderer import draw_opaque, draw_overlays
from viewer.hover import HoverState
from viewer.view_cube import ViewCube
from viewer.sketch_pick import SketchPickMixin
from viewer.vp_operations import OperationsMixin
from cad.workspace import Workspace
from cad.history import History
from cad.selection import SelectionSet


def _plane_from_sketch_entry(se):
    """Reconstruct a build123d Plane from a SketchEntry's baked plane cache."""
    from build123d import Plane
    return Plane(
        origin = tuple(se.plane_origin.tolist()),
        x_dir  = tuple(se.plane_x_axis.tolist()),
        z_dir  = tuple(se.plane_normal.tolist()),
    )


class Viewport(SketchPickMixin, OperationsMixin, QOpenGLWidget):
    history_changed     = pyqtSignal()
    selection_changed   = pyqtSignal()
    sketch_mode_changed = pyqtSignal(bool)
    body_selected       = pyqtSignal(object)   # str | None

    def __init__(self, workspace: Workspace, history: History):
        super().__init__()
        self.workspace = workspace
        self.history   = history
        self.camera    = Camera()
        self._view_cube = ViewCube()

        self._meshes: dict[str, Mesh] = {}
        self._body_visible: dict[str, bool] = {}

        self._modelview  = None
        self._projection = None
        self._viewport   = None

        self.selection = SelectionSet()
        self.hover     = HoverState()

        self._hovered_vertex: tuple[str | None, int | None] = (None, None)
        self._hovered_edge:   tuple[str | None, int | None] = (None, None)

        self._sketch = None                              # SketchMode | None
        self._sketch_faces: dict[int, list] = {}        # history_idx → [Face]
        self._selected_sketch_entry: int | None = None
        self._editing_sketch_history_idx: int | None = None  # set during re-entry

        self.camera_projection_changed = None
        self.request_extrude_distance  = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Backward-compat shims
    # ------------------------------------------------------------------

    @property
    def picked_body_id(self) -> str | None:
        sf = self.selection.single_face
        return sf.body_id if sf else None

    @property
    def picked_face(self) -> int | None:
        sf = self.selection.single_face
        return sf.face_idx if sf else None

    @property
    def picked_face_obj(self):
        sf = self.selection.single_face
        if sf is None:
            return None
        mesh = self._meshes.get(sf.body_id)
        return mesh.occt_faces[sf.face_idx] if mesh else None

    # ------------------------------------------------------------------
    # Mesh management
    # ------------------------------------------------------------------

    def build_meshes(self):
        self.makeCurrent()
        for body_id, shape in self.workspace.all_current_shapes():
            self._meshes[body_id] = Mesh(shape)
            self._body_visible.setdefault(body_id, True)
        self.doneCurrent()

    def set_body_visible(self, body_id: str, visible: bool):
        self._body_visible[body_id] = visible
        body = self.workspace.bodies.get(body_id)
        if body is not None:
            body.visible = visible
        self.update()

    def set_active_body(self, body_id: str | None):
        self.selection.clear()
        if body_id is not None:
            self.workspace.set_active_body(body_id)
        self.selection_changed.emit()
        self.update()

    def _visible_meshes(self) -> dict:
        return {bid: m for bid, m in self._meshes.items()
                if self._body_visible.get(bid, True)}

    def fit_camera_to_scene(self):
        if not self._meshes:
            return
        all_min = np.array([ np.inf,  np.inf,  np.inf])
        all_max = np.array([-np.inf, -np.inf, -np.inf])
        for mesh in self._meshes.values():
            all_min = np.minimum(all_min, mesh.bbox_min)
            all_max = np.maximum(all_max, mesh.bbox_max)
        self.camera.fit_scene(all_min, all_max)

    def _rebuild_body_mesh(self, body_id: str):
        self.makeCurrent()
        old = self._meshes.get(body_id)
        if old:
            for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
                if buf is not None:
                    glDeleteBuffers(1, [buf])
        self._body_visible.setdefault(body_id, True)   # new bodies are visible
        shape = self.workspace.current_shape(body_id)
        if shape is not None:
            try:
                mesh = Mesh(shape)
                mesh.upload()
                self._meshes[body_id] = mesh
            except Exception as ex:
                print(f"[Mesh] Could not tessellate body {body_id}: {ex}")
        self.selection.clear_for_body(body_id)
        if self._hovered_vertex[0] == body_id:
            self._hovered_vertex = (None, None)
        if self._hovered_edge[0] == body_id:
            self._hovered_edge = (None, None)
        self.doneCurrent()
        self._rebuild_sketch_faces()
        self.selection_changed.emit()
        self.update()

    def _rebuild_all_meshes(self):
        self.makeCurrent()
        for body_id, old in list(self._meshes.items()):
            for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
                if buf is not None:
                    glDeleteBuffers(1, [buf])
        self._meshes.clear()
        for body_id, shape in self.workspace.all_current_shapes():
            try:
                mesh = Mesh(shape)
                mesh.upload()
                self._meshes[body_id] = mesh
            except Exception as ex:
                print(f"[Mesh] Could not tessellate body {body_id}: {ex}")
        self.selection.clear()
        self._hovered_vertex = (None, None)
        self._hovered_edge   = (None, None)
        self.hover.clear()
        self.doneCurrent()
        self._rebuild_sketch_faces()
        self.selection_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        from cad.prefs import prefs
        r, g, b = prefs.background_color
        glClearColor(r, g, b, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 2, 3, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        for mesh in self._meshes.values():
            mesh.upload()

    def _set_projection(self, w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h
        if self.camera.ortho:
            s = self.camera.ortho_scale
            glOrtho(-s * aspect, s * aspect, -s, s, -50000, 50000)
        else:
            near = max(0.1, self.camera.distance * 0.001)
            gluPerspective(45, aspect, near,
                           max(50000, self.camera.distance * 100))
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self._set_projection(w, h)

    def paintGL(self):
        self._set_projection(self.width(), self.height())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        c = self.camera
        eye, up = c.get_eye(), c.get_up()
        gluLookAt(eye[0], eye[1], eye[2],
                  c.target[0], c.target[1], c.target[2],
                  up[0], up[1], up[2])

        visible = self._visible_meshes()
        draw_opaque(visible, self.workspace, self.selection)

        self._modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._projection = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport   = glGetIntegerv(GL_VIEWPORT)
        self.hover.rebuild(visible, self.workspace,
                           self._modelview, self._projection,
                           self._viewport, self.devicePixelRatio(),
                           camera_eye=self.camera.get_eye(),
                           history=self.history,
                           active_sketch=self._sketch)

        self._draw_sketch_faces()

        draw_overlays(visible, self.selection,
                      self._hovered_vertex, self._hovered_edge,
                      sketch=self._sketch,
                      camera_distance=self.camera.distance,
                      history=self.history,
                      editing_sketch_idx=self._editing_sketch_history_idx)

        from viewer.camera import _quat_to_matrix
        R = _quat_to_matrix(self.camera.rotation)
        self._view_cube.draw(R, self.width(), self.height(),
                             self.devicePixelRatio())

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def get_ray(self, x, y):
        if self._modelview is None:
            return None, None
        dpr  = self.devicePixelRatio()
        x_px = x * dpr
        y_px = self._viewport[3] - y * dpr
        near = np.array(gluUnProject(x_px, y_px, 0.0,
                        self._modelview, self._projection, self._viewport))
        far  = np.array(gluUnProject(x_px, y_px, 1.0,
                        self._modelview, self._projection, self._viewport))
        d    = far - near
        n    = np.linalg.norm(d)
        return (near, d / n) if n > 1e-10 else (None, None)

    def _pick_at(self, pos):
        from cad.picker import pick_face
        origin, direction = self.get_ray(pos.x(), pos.y())
        if origin is None:
            return None, None
        best_t, best_body, best_face = np.inf, None, None
        for body_id, mesh in self._meshes.items():
            body = self.workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            result = pick_face(mesh, origin, direction, return_t=True)
            if result and result[1] < best_t:
                best_t, best_body, best_face = result[1], body_id, result[0]
        return best_body, best_face

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, e):
        from cad.prefs import prefs
        from cad.sketch import SketchTool

        if e.key() == Qt.Key.Key_Escape:
            if self._sketch is not None:
                if self._sketch.tool != SketchTool.NONE:
                    self._sketch.cancel_tool()
                    self.sketch_mode_changed.emit(True)
                else:
                    self._exit_sketch()
            else:
                self._selected_sketch_entry = None
                self.selection.clear()
                self.body_selected.emit(None)
                self.selection_changed.emit()
            self.update()
            return

        if self._sketch is not None:
            if prefs.matches("sketch_line", e):
                self._sketch.set_tool(SketchTool.LINE)
                self.sketch_mode_changed.emit(True)
                self.update()
            elif prefs.matches("sketch_include", e):
                from cad.sketch_tools.include import IncludeTool
                self._sketch.push_undo_snapshot()
                n = IncludeTool.apply_with_history(
                    self._sketch, self.selection, self._meshes, self.history)
                if n:
                    print(f"[Sketch] Included {n} reference "
                          f"{'entity' if n == 1 else 'entities'}")
                    self.selection.clear()
                    self.selection_changed.emit()
                    self.update()
                else:
                    # Nothing was included — discard the snapshot we just pushed
                    self._sketch._entity_snapshots.pop()
                    print("[Sketch] Nothing selected to include — "
                          "select edges, vertices, or sketch lines first.")
            elif prefs.matches("sketch_commit", e):
                self._complete_sketch()
            elif prefs.matches("sketch_projection_toggle", e):
                self.toggle_projection()
            else:
                super().keyPressEvent(e)
            return

        if prefs.matches("undo", e):
            self._do_undo()
        elif prefs.matches("redo", e):
            self._do_redo()
        elif prefs.matches("projection_toggle", e):
            self.toggle_projection()
        elif prefs.matches("extrude", e):
            self._try_extrude()
        else:
            super().keyPressEvent(e)

    def toggle_projection(self):
        self.camera.toggle_ortho()
        if self.camera_projection_changed:
            self.camera_projection_changed(self.camera.ortho)
        self.update()

    # ------------------------------------------------------------------
    # Sketch modal
    # ------------------------------------------------------------------

    def _enter_sketch(self, body_id: str, face_idx: int):
        from cad.sketch import SketchMode
        from cad.face_ref import FaceRef
        from cad.plane_ref import FacePlaneSource
        from build123d import Plane
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
        # Pre-populate the undo stack so every existing entity is individually
        # undoable: snapshots[0]=[], snapshots[1]=[e0], ..., snapshots[n-1]=[e0..e_{n-2}]
        mode._entity_snapshots = [existing[:i] for i in range(len(existing))]
        mode.entities = existing
        self._sketch = mode
        self._editing_sketch_history_idx = history_idx
        self._selected_sketch_entry = None
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

    def _complete_sketch(self):
        if self._sketch is None:
            return
        sketch = self._sketch
        from cad.sketch import LineEntity, ReferenceEntity, SketchEntry
        lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
        refs  = [e for e in sketch.entities if isinstance(e, ReferenceEntity)]
        if not lines and not refs:
            print("[Sketch] Nothing to commit — draw lines or include geometry first.")
            return
        entity_count = len(lines) + len(refs)
        new_se = SketchEntry.from_sketch_mode(sketch)

        editing_idx = self._editing_sketch_history_idx
        if editing_idx is not None:
            # Re-entry: update the existing history entry in place, then replay
            # all downstream ops so extrudes and split bodies stay consistent.
            entries = self.history.entries
            if editing_idx < len(entries):
                entries[editing_idx].params["sketch_entry"] = new_se
                self._sketch = None
                self._editing_sketch_history_idx = None
                self.sketch_mode_changed.emit(False)
                self._rebuild_sketch_faces()
                ok, err = self.history.replay_from(editing_idx)
                if not ok:
                    print(f"[Sketch] Replay after re-entry failed: {err}")
                self._rebuild_all_meshes()
                self.history_changed.emit()
                self.update()
                print(f"[Sketch] Updated sketch entry {editing_idx}, "
                      f"replayed downstream.")
                return

        # Normal first-time commit
        from cad.face_ref import FaceRef
        mesh = self._meshes.get(sketch.body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[sketch.face_idx])
                    if mesh else None)
        current_shape = self.workspace.current_shape(sketch.body_id)
        self.history.push(
            label        = f"Sketch  ({entity_count} entities)",
            operation    = "sketch",
            params       = {"sketch_entry": new_se},
            body_id      = sketch.body_id,
            face_ref     = face_ref,
            shape_before = current_shape,
            shape_after  = current_shape,
        )
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        self._rebuild_sketch_faces()
        self._post_push_cascade(sketch.body_id)
        self.history_changed.emit()
        self.update()
        print(f"[Sketch] Committed {entity_count} entities to history.")

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._sketch is not None:
            return
        # Try solid face first — opens a fresh sketch
        body_id, idx = self._pick_at(e.position())
        if idx is not None:
            self._enter_sketch(body_id, idx)
            return
        # Try sketch face — re-opens the committed sketch for editing
        origin, direction = self.get_ray(e.position().x(), e.position().y())
        if origin is not None and self._sketch_faces:
            hidx, _ = self._pick_sketch_face(origin, direction)
            if hidx is not None:
                self._reenter_sketch(hidx)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            from viewer.camera import _quat_to_matrix
            R = _quat_to_matrix(self.camera.rotation)
            mx, my = int(e.position().x()), int(e.position().y())
            cube_hit = self._view_cube.handle_mouse_press(
                mx, my, self.width(), self.height(), R, self.devicePixelRatio())
            if cube_hit is not None:
                normal, is_corner = cube_hit
                self.camera.snap_to_normal(*normal)
                self.update()
                return

            if self._sketch is not None:
                from cad.sketch import SketchTool
                if self._sketch.tool != SketchTool.NONE:
                    origin, direction = self.get_ray(e.position().x(),
                                                     e.position().y())
                    if origin is not None:
                        shift = bool(e.modifiers() &
                                     Qt.KeyboardModifier.ShiftModifier)
                        self._sketch.snap.set_grid_snap(shift)
                        self._sketch.snap.snap_radius_mm = self._snap_radius_mm()
                        self._sketch.handle_click(
                            origin, direction, self.camera.distance)
                        self.update()
                    return

            additive = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            hov_body,  hov_idx  = self._hovered_vertex
            hov_ebody, hov_eidx = self._hovered_edge

            if hov_body is not None:
                self.selection.select_vertex(hov_body, hov_idx,
                                             additive=additive)
                self.workspace.set_active_body(hov_body)
                self.body_selected.emit(hov_body)
                p = self._meshes[hov_body].topo_verts[hov_idx]
                print(f"Picked vertex {hov_idx} | "
                      f"{self.workspace.bodies[hov_body].name} | "
                      f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

            elif hov_ebody is not None:
                from viewer.hover import parse_sketch_key
                sk = parse_sketch_key(hov_ebody)
                if sk is not None:
                    history_idx, entity_idx = sk
                    self.selection.select_sketch_edge(
                        history_idx, entity_idx, additive=additive)
                    if history_idx == -1:
                        print(f"Picked active sketch line entity {entity_idx}")
                    else:
                        print(f"Picked sketch edge | entry {history_idx} "
                              f"entity {entity_idx}")
                else:
                    self.selection.select_edge(hov_ebody, hov_eidx,
                                               additive=additive)
                    self.workspace.set_active_body(hov_ebody)
                    self.body_selected.emit(hov_ebody)
                    print(f"Picked edge {hov_eidx} | "
                          f"{self.workspace.bodies[hov_ebody].name}")

            else:
                origin, direction = self.get_ray(e.position().x(),
                                                 e.position().y())
                if origin is not None and self._sketch_faces:
                    hidx, _ = self._pick_sketch_face(origin, direction)
                    if hidx is not None:
                        self._selected_sketch_entry = hidx
                        self.selection.clear()
                        self.body_selected.emit(None)
                        self.selection_changed.emit()
                        self.update()
                        print(f"[Sketch] Selected sketch entry {hidx}"
                              f" — press E to extrude")
                        return

                body_id, idx = self._pick_at(e.position())
                if idx is not None:
                    self._selected_sketch_entry = None
                    self.selection.select_face(body_id, idx, additive=additive)
                    self.workspace.set_active_body(body_id)
                    self.body_selected.emit(body_id)
                    mesh = self._meshes[body_id]
                    try:
                        from build123d import Plane
                        plane = Plane(mesh.occt_faces[idx])
                        print(f"Picked face {idx} | "
                              f"{self.workspace.bodies[body_id].name} | "
                              f"planar | normal={plane.z_dir}")
                    except Exception:
                        print(f"Picked face {idx} | "
                              f"{self.workspace.bodies[body_id].name} | "
                              f"non-planar")
                elif not additive:
                    self._selected_sketch_entry = None
                    self.selection.clear()
                    self.body_selected.emit(None)

            self.selection_changed.emit()
            self.update()

        elif e.button() == Qt.MouseButton.RightButton:
            mx, my = int(e.position().x()), int(e.position().y())
            if self._view_cube.is_over_cube(mx, my, self.width(), self.height(),
                                            self.devicePixelRatio()):
                return
            body_id, idx = self._pick_at(e.position())
            pivot = None
            if idx is not None and body_id in self._meshes:
                mesh  = self._meshes[body_id]
                start, count = mesh.get_face_triangle_range(idx)
                pivot = mesh.verts[mesh.tris[start:start+count]].mean(
                    axis=(0, 1)).tolist()
            self.camera.begin_orbit(e.position(), self.width(), self.height(),
                                    pivot=pivot)

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.camera.begin_pan(e.position())

    def mouseMoveEvent(self, e):
        buttons = e.buttons()
        if buttons & Qt.MouseButton.RightButton:
            self.camera.orbit(e.position()); self.update(); return
        if buttons & Qt.MouseButton.MiddleButton:
            self.camera.pan(e.position());   self.update(); return

        x, y = e.position().x(), e.position().y()

        from viewer.camera import _quat_to_matrix
        R = _quat_to_matrix(self.camera.rotation)
        cube_changed = self._view_cube.handle_mouse_move(
            int(x), int(y), self.width(), self.height(), R,
            self.devicePixelRatio())
        if cube_changed:
            self.update()

        new_v = self.hover.vertex_at(x, y)
        new_e = (None, None) if new_v[0] is not None \
                else self.hover.edge_at(x, y)
        hover_changed = (new_v != self._hovered_vertex or
                         new_e != self._hovered_edge)
        self._hovered_vertex = new_v
        self._hovered_edge   = new_e

        sketch_changed = False
        if self._sketch is not None:
            from cad.sketch import SketchTool
            if self._sketch.tool != SketchTool.NONE:
                origin, direction = self.get_ray(x, y)
                if origin is not None:
                    shift = bool(e.modifiers() &
                                 Qt.KeyboardModifier.ShiftModifier)
                    self._sketch.snap.set_grid_snap(shift)
                    self._sketch.snap.snap_radius_mm = self._snap_radius_mm()
                    self._sketch.handle_mouse_move(
                        origin, direction, self.camera.distance)
                    sketch_changed = True

        if hover_changed or sketch_changed:
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.camera.end_orbit()
        elif e.button() == Qt.MouseButton.MiddleButton:
            self.camera.end_pan()

    def wheelEvent(self, e):
        self.camera.scroll(e.angleDelta().y())
        self.update()
