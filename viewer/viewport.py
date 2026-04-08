"""
viewer/viewport.py

Renders all visible bodies in the workspace. Each body has its own Mesh.
Picking identifies both the body and the face within that body.
"""

import ctypes
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *

from viewer.camera import Camera
from viewer.mesh import Mesh
from cad.workspace import Workspace
from cad.history import History


class Viewport(QOpenGLWidget):
    history_changed = pyqtSignal()

    def __init__(self, workspace: Workspace, history: History):
        super().__init__()
        self.workspace = workspace
        self.history   = history
        self.camera    = Camera()

        # body_id → Mesh, rebuilt whenever that body's shape changes
        self._meshes: dict[str, Mesh] = {}

        self._modelview  = None
        self._projection = None
        self._viewport   = None

        # Current pick — (body_id, face_idx) or (None, None)
        self.picked_body_id:  str | None = None
        self.picked_face:     int | None = None
        self.picked_face_obj             = None

        self.camera_projection_changed = None
        self.request_extrude_distance  = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Mesh management
    # ------------------------------------------------------------------

    def build_meshes(self):
        """Build a Mesh for every visible body. Call after load."""
        self.makeCurrent()
        for body_id, shape in self.workspace.all_current_shapes():
            self._meshes[body_id] = Mesh(shape)
        self.doneCurrent()

    def fit_camera_to_scene(self):
        """Fit camera to the combined bounding box of all loaded meshes."""
        if not self._meshes:
            return
        import numpy as np
        all_min = np.array([ np.inf,  np.inf,  np.inf])
        all_max = np.array([-np.inf, -np.inf, -np.inf])
        for mesh in self._meshes.values():
            all_min = np.minimum(all_min, mesh.bbox_min)
            all_max = np.maximum(all_max, mesh.bbox_max)
        self.camera.fit_scene(all_min, all_max)

    def _rebuild_body_mesh(self, body_id: str):
        """Re-tessellate one body after an operation."""
        self.makeCurrent()
        old = self._meshes.get(body_id)
        if old:
            for buf in (old.vbo_verts, old.vbo_normals,
                        old.vbo_edges, old.ebo):
                if buf is not None:
                    glDeleteBuffers(1, [buf])

        shape = self.workspace.current_shape(body_id)
        if shape is not None:
            mesh = Mesh(shape)
            mesh.upload()
            self._meshes[body_id] = mesh

        # Clear pick if it was on this body
        if self.picked_body_id == body_id:
            self.picked_body_id  = None
            self.picked_face     = None
            self.picked_face_obj = None

        self.doneCurrent()
        self.update()

    def _rebuild_all_meshes(self):
        """Re-tessellate all bodies — used after seek/undo/redo."""
        self.makeCurrent()
        for body_id, old in list(self._meshes.items()):
            for buf in (old.vbo_verts, old.vbo_normals,
                        old.vbo_edges, old.ebo):
                if buf is not None:
                    glDeleteBuffers(1, [buf])

        self._meshes.clear()
        for body_id, shape in self.workspace.all_current_shapes():
            mesh = Mesh(shape)
            mesh.upload()
            self._meshes[body_id] = mesh

        self.picked_body_id  = None
        self.picked_face     = None
        self.picked_face_obj = None
        self.doneCurrent()
        self.update()

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        glClearColor(0.15, 0.15, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 2, 3, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        # Upload all meshes now that GL context exists
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
            gluPerspective(45, aspect, near, max(50000, self.camera.distance * 100))
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self._set_projection(w, h)

    def paintGL(self):
        w, h = self.width(), self.height()
        self._set_projection(w, h)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        c   = self.camera
        eye = c.get_eye()
        up  = c.get_up()
        gluLookAt(eye[0], eye[1], eye[2],
                  c.target[0], c.target[1], c.target[2],
                  up[0], up[1], up[2])

        # Draw all bodies
        for body_id, mesh in self._meshes.items():
            body = self.workspace.bodies.get(body_id)
            if body and not body.visible:
                continue

            # Active body slightly brighter
            if body_id == self.workspace.active_body_id:
                glColor3f(0.6, 0.82, 1.0)
            else:
                glColor3f(0.45, 0.60, 0.78)
            mesh.draw()

        glDisable(GL_LIGHTING)

        # Picked face highlight
        if self.picked_face is not None and self.picked_body_id in self._meshes:
            mesh = self._meshes[self.picked_body_id]
            start, count = mesh.get_face_triangle_range(self.picked_face)
            glColor3f(1.0, 0.4, 0.0)
            glDisable(GL_DEPTH_TEST)
            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo_verts)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            glDrawElements(GL_TRIANGLES, count * 3, GL_UNSIGNED_INT,
                           ctypes.c_void_p(start * 3 * 4))
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glEnable(GL_DEPTH_TEST)

        # Edges for all bodies
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(3.0)
        for body_id, mesh in self._meshes.items():
            body = self.workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            mesh.draw_edges()

        glEnable(GL_LIGHTING)

        self._modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._projection = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport   = glGetIntegerv(GL_VIEWPORT)

    # ------------------------------------------------------------------
    # Ray casting — picks across all visible bodies
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
        direction = far - near
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None, None
        return near, direction / norm

    def _pick_at(self, pos) -> tuple[str | None, int | None]:
        """
        Cast a ray against all visible body meshes.
        Returns (body_id, face_idx) of the closest hit, or (None, None).
        """
        from cad.picker import pick_face
        origin, direction = self.get_ray(pos.x(), pos.y())
        if origin is None:
            return None, None

        best_body  = None
        best_face  = None
        best_t     = float("inf")

        for body_id, mesh in self._meshes.items():
            body = self.workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            result = pick_face(mesh, origin, direction, return_t=True)
            if result is None:
                continue
            face_idx, t = result
            if t < best_t:
                best_t    = t
                best_body = body_id
                best_face = face_idx

        return best_body, best_face

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, e):
        key  = e.key()
        mods = e.modifiers()

        if mods & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                self._do_undo(); return
            if key == Qt.Key.Key_Y:
                self._do_redo(); return

        if key == Qt.Key.Key_5:
            self.toggle_projection()
        elif key == Qt.Key.Key_E:
            self._try_extrude()
        else:
            super().keyPressEvent(e)

    def toggle_projection(self):
        self.camera.toggle_ortho()
        cb = self.camera_projection_changed
        if cb:
            cb(self.camera.ortho)
        self.update()

    # ------------------------------------------------------------------
    # Undo / Redo / Seek
    # ------------------------------------------------------------------

    def _do_undo(self):
        entry = self.history.undo()
        if entry is None:
            print("[Undo] Nothing to undo.")
            return
        print(f"[Undo] Reverting: {entry.label}")
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def _do_redo(self):
        entry = self.history.redo()
        if entry is None:
            print("[Redo] Nothing to redo.")
            return
        print(f"[Redo] Restoring: {entry.label}")
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def seek_history(self, index: int):
        if self.history.seek(index):
            self._rebuild_all_meshes()
            self.history_changed.emit()

    # ------------------------------------------------------------------
    # Parametric replay
    # ------------------------------------------------------------------

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
        if self.picked_face is None or self.picked_body_id is None:
            print("[Extrude] No face selected — click a face first.")
            return
        cb = self.request_extrude_distance
        if cb:
            cb(self.picked_body_id, self.picked_face)
        else:
            self._do_extrude_dialog(self.picked_body_id, self.picked_face)

    def _do_extrude_dialog(self, body_id, face_idx):
        from PyQt6.QtWidgets import QInputDialog
        dist, ok = QInputDialog.getDouble(
            self, "Extrude",
            "Distance (positive = add material, negative = cut):",
            value=5.0, min=-1000.0, max=1000.0, decimals=3,
        )
        if not ok:
            return
        self.do_extrude(body_id, face_idx, dist)

    def do_extrude(self, body_id: str, face_idx: int, distance: float):
        from cad.operations import extrude_face
        from cad.face_ref import FaceRef

        mesh = self._meshes.get(body_id)
        if mesh is None:
            print(f"[Extrude] No mesh for body {body_id}")
            return

        shape_before = self.workspace.current_shape(body_id)
        face_ref     = FaceRef.from_b3d_face(mesh.occt_faces[face_idx])
        if face_ref is None:
            print(f"[Extrude] Face {face_idx} is not planar.")
            return

        try:
            shape_after = extrude_face(shape_before, face_idx, distance)
        except Exception as ex:
            print(f"[Extrude] FAILED: {ex}")
            return

        op    = "cut" if distance < 0 else "extrude"
        label = (f"Cut  -{abs(distance):.3f}mm" if distance < 0
                 else f"Extrude  +{distance:.3f}mm")

        self.history.push(
            label        = label,
            operation    = op,
            params       = {"distance": abs(distance)},
            body_id      = body_id,
            face_ref     = face_ref,
            shape_before = shape_before,
            shape_after  = shape_after,
        )

        print(f"[Extrude] body={self.workspace.bodies[body_id].name} "
              f"face={face_idx}  distance={distance:+.3f}  OK")
        self._rebuild_body_mesh(body_id)
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            body_id, idx = self._pick_at(e.position())
            if idx is None:
                return
            mesh = self._meshes.get(body_id)
            if mesh is None:
                return
            face = mesh.occt_faces[idx]
            try:
                from build123d import Plane
                plane = Plane(face)
                n  = plane.z_dir
                wo = plane.origin
                self.camera.snap_to_normal(
                    n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z))
                self.update()
            except Exception:
                pass

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            body_id, idx = self._pick_at(e.position())
            if idx is not None:
                self.picked_body_id  = body_id
                self.picked_face     = idx
                mesh = self._meshes[body_id]
                self.picked_face_obj = mesh.occt_faces[idx]
                # Make clicked body active
                self.workspace.set_active_body(body_id)
                try:
                    from build123d import Plane
                    plane = Plane(self.picked_face_obj)
                    body_name = self.workspace.bodies[body_id].name
                    print(f"Picked face {idx} | {body_name} | planar | "
                          f"normal={plane.z_dir} origin={plane.origin}")
                except Exception:
                    body_name = self.workspace.bodies[body_id].name
                    print(f"Picked face {idx} | {body_name} | non-planar")
            else:
                self.picked_body_id  = None
                self.picked_face     = None
                self.picked_face_obj = None
            self.update()

        elif e.button() == Qt.MouseButton.RightButton:
            body_id, idx = self._pick_at(e.position())
            pivot = None
            if idx is not None and body_id in self._meshes:
                mesh = self._meshes[body_id]
                start, count = mesh.get_face_triangle_range(idx)
                face_tris = mesh.tris[start:start + count]
                pivot = mesh.verts[face_tris].mean(axis=(0, 1)).tolist()
            self.camera.begin_orbit(e.position(), self.width(), self.height(),
                                    pivot=pivot)

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.camera.begin_pan(e.position())

    def mouseMoveEvent(self, e):
        buttons = e.buttons()
        if buttons & Qt.MouseButton.RightButton:
            self.camera.orbit(e.position())
            self.update()
        if buttons & Qt.MouseButton.MiddleButton:
            self.camera.pan(e.position())
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.camera.end_orbit()
        elif e.button() == Qt.MouseButton.MiddleButton:
            self.camera.end_pan()

    def wheelEvent(self, e):
        self.camera.scroll(e.angleDelta().y())
        self.update()
