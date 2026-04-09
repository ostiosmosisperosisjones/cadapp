"""
viewer/viewport.py

Thin Qt widget: GL lifecycle, input events, mesh management, operations.
Drawing logic lives in viewer/renderer.py.
Hover/pick logic lives in viewer/hover.py.
"""

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
from cad.workspace import Workspace
from cad.history import History
from cad.selection import SelectionSet


class Viewport(QOpenGLWidget):
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
        self._body_visible: dict[str, bool] = {}  # True = visible (default)

        self._modelview  = None
        self._projection = None
        self._viewport   = None

        self.selection = SelectionSet()
        self.hover     = HoverState()

        self._hovered_vertex: tuple[str | None, int | None] = (None, None)
        self._hovered_edge:   tuple[str | None, int | None] = (None, None)

        # Sketch modal state — None when not in sketch mode
        self._sketch = None   # SketchMode | None

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
        self.update()

    def set_active_body(self, body_id: str | None):
        """Driven externally (e.g. parts panel click) — sets active body
        and clears selection without re-emitting body_selected."""
        self.selection.clear()
        if body_id is not None:
            self.workspace.set_active_body(body_id)
        self.selection_changed.emit()
        self.update()

    def _visible_meshes(self) -> dict:
        """Return only meshes whose body is currently visible."""
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
        shape = self.workspace.current_shape(body_id)
        if shape is not None:
            mesh = Mesh(shape)
            mesh.upload()
            self._meshes[body_id] = mesh
        self.selection.clear_for_body(body_id)
        if self._hovered_vertex[0] == body_id:
            self._hovered_vertex = (None, None)
        if self._hovered_edge[0] == body_id:
            self._hovered_edge = (None, None)
        self.doneCurrent()
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
            mesh = Mesh(shape)
            mesh.upload()
            self._meshes[body_id] = mesh
        self.selection.clear()
        self._hovered_vertex = (None, None)
        self._hovered_edge   = (None, None)
        self.hover.clear()
        self.doneCurrent()
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

        # Capture matrices and rebuild hover cache while depth buffer
        # contains only opaque geometry — gives correct occlusion results.
        self._modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._projection = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport   = glGetIntegerv(GL_VIEWPORT)
        self.hover.rebuild(visible, self.workspace,
                           self._modelview, self._projection,
                           self._viewport, self.devicePixelRatio(),
                           camera_eye=self.camera.get_eye())

        draw_overlays(visible, self.selection,
                      self._hovered_vertex, self._hovered_edge,
                      sketch=self._sketch,
                      camera_distance=self.camera.distance)

        from viewer.camera import _quat_to_matrix
        R = _quat_to_matrix(self.camera.rotation)
        self._view_cube.draw(R, self.width(), self.height(), self.devicePixelRatio())

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
        key  = e.key()
        mods = e.modifiers()

        if key == Qt.Key.Key_Escape:
            if self._sketch is not None:
                from cad.sketch import SketchTool
                if self._sketch.tool != SketchTool.NONE:
                    # First ESC: cancel active tool, stay in sketch
                    self._sketch.set_tool(SketchTool.NONE)
                    self.sketch_mode_changed.emit(True)  # refresh status bar
                else:
                    # Second ESC (or ESC with no tool): exit sketch
                    self._exit_sketch()
            else:
                self.selection.clear()
                self.body_selected.emit(None)
                self.selection_changed.emit()
            self.update()
            return

        # Keys active inside sketch mode
        if self._sketch is not None:
            from cad.sketch import SketchTool
            if key == Qt.Key.Key_L:
                self._sketch.set_tool(SketchTool.LINE)
                self.sketch_mode_changed.emit(True)
                self.update()
            elif key == Qt.Key.Key_I:
                n = self._sketch.include_selection(self.selection, self._meshes)
                if n:
                    print(f"[Sketch] Included {n} reference "
                          f"{'entity' if n == 1 else 'entities'}")
                    self.selection.clear()
                    self.selection_changed.emit()
                    self.update()
                else:
                    print("[Sketch] Nothing selected to include — "
                          "select edges or vertices first.")
            elif key == Qt.Key.Key_5:
                self.toggle_projection()
            else:
                super().keyPressEvent(e)
            return

        # Normal 3D mode keys
        if mods & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:  self._do_undo(); return
            if key == Qt.Key.Key_Y:  self._do_redo(); return
        if key == Qt.Key.Key_5:
            self.toggle_projection()
        elif key == Qt.Key.Key_E:
            self._try_extrude()
        else:
            super().keyPressEvent(e)

    def toggle_projection(self):
        self.camera.toggle_ortho()
        if self.camera_projection_changed:
            self.camera_projection_changed(self.camera.ortho)
        self.update()

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
    # Extrude
    # ------------------------------------------------------------------

    def _try_extrude(self):
        if self.selection.face_count == 0:
            print("[Extrude] No face selected."); return
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
        self.history.push(
            label=label, operation=op, params={"distance": abs(distance)},
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=shape_after)
        print(f"[Extrude] body={self.workspace.bodies[body_id].name} "
              f"face={face_idx}  distance={distance:+.3f}  OK")
        self._rebuild_body_mesh(body_id)
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sketch modal
    # ------------------------------------------------------------------

    def _enter_sketch(self, body_id: str, face_idx: int):
        """Enter sketch mode on a planar face."""
        from cad.sketch import SketchMode
        from build123d import Plane
        mesh = self._meshes.get(body_id)
        if mesh is None:
            return
        try:
            b3d_plane = Plane(mesh.occt_faces[face_idx])
        except Exception:
            print("[Sketch] Face is not planar — cannot enter sketch mode.")
            return
        # Snap camera to face normal so the sketch looks flat
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z))
        self._sketch = SketchMode(b3d_plane, body_id, face_idx)
        # No tool active by default — user picks a tool explicitly
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Entered sketch on face {face_idx} of "
              f"{self.workspace.bodies[body_id].name}")

    def _exit_sketch(self):
        """Exit sketch mode, discarding the current sketch."""
        if self._sketch is None:
            return
        self._sketch = None
        self.sketch_mode_changed.emit(False)
        self.update()
        print("[Sketch] Exited sketch mode.")

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        # In sketch mode, double-click does nothing special
        if self._sketch is not None:
            return
        body_id, idx = self._pick_at(e.position())
        if idx is None:
            return
        self._enter_sketch(body_id, idx)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            # Check view cube first — it lives in a corner and should
            # consume clicks before normal selection logic
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
                    # A tool is active — consume the click for drawing
                    origin, direction = self.get_ray(e.position().x(),
                                                     e.position().y())
                    if origin is not None:
                        self._sketch.handle_click(origin, direction)
                        self.update()
                    return
                # No tool active — fall through to normal vertex/edge/face selection

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
                self.selection.select_edge(hov_ebody, hov_eidx,
                                           additive=additive)
                self.workspace.set_active_body(hov_ebody)
                self.body_selected.emit(hov_ebody)
                print(f"Picked edge {hov_eidx} | "
                      f"{self.workspace.bodies[hov_ebody].name}")

            else:
                body_id, idx = self._pick_at(e.position())
                if idx is not None:
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
                    self.selection.clear()
                    self.body_selected.emit(None)

            self.selection_changed.emit()
            self.update()

        elif e.button() == Qt.MouseButton.RightButton:
            mx, my = int(e.position().x()), int(e.position().y())
            if self._view_cube.is_over_cube(mx, my, self.width(), self.height(), self.devicePixelRatio()):
                return  # don't start orbit when clicking on the cube
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

        # View cube hover — check before scene hover
        from viewer.camera import _quat_to_matrix
        R = _quat_to_matrix(self.camera.rotation)
        cube_changed = self._view_cube.handle_mouse_move(
            int(x), int(y), self.width(), self.height(), R, self.devicePixelRatio())
        if cube_changed:
            self.update()

        # Always update vertex/edge hover (visible in both 3D and sketch mode)
        new_v = self.hover.vertex_at(x, y)
        new_e = (None, None) if new_v[0] is not None \
                else self.hover.edge_at(x, y)
        hover_changed = (new_v != self._hovered_vertex or
                         new_e != self._hovered_edge)
        self._hovered_vertex = new_v
        self._hovered_edge   = new_e

        # If a sketch tool is active, also update the sketch cursor
        sketch_changed = False
        if self._sketch is not None:
            from cad.sketch import SketchTool
            if self._sketch.tool != SketchTool.NONE:
                origin, direction = self.get_ray(x, y)
                if origin is not None:
                    self._sketch.handle_mouse_move(origin, direction)
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
