"""
viewer/sketch_pick.py

SketchPickMixin — manages pickable/renderable sketch faces.

Owns:
  _rebuild_sketch_faces()  — called after any history mutation
  _pick_sketch_face()      — ray-tests committed sketch faces
  _draw_sketch_faces()     — renders semi-transparent filled polygons
  _snap_radius_mm()        — converts screen pixels → world mm for snap

Expects self to have: history, _sketch_faces, _sketch,
_selected_sketch_entry, camera, _modelview, _projection, _viewport,
devicePixelRatio(), width(), height()
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *


class SketchPickMixin:

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _rebuild_sketch_faces(self):
        """
        Rebuild pickable/renderable faces from all committed sketch entries
        up to the current history cursor.  Called after any history mutation.
        """
        self._sketch_faces.clear()
        cursor = self.history.cursor
        for i, entry in enumerate(self.history.entries):
            if i > cursor:
                break
            if entry.operation != "sketch":
                continue
            se = entry.params.get("sketch_entry")
            if se is None:
                continue
            try:
                segs  = se.line_segments()
                loops = se.closed_loops()
                faces = se.build_faces()
                print(f"[Sketch] Entry {i}: {len(segs)} segments, "
                      f"{len(loops)} closed loops, {len(faces)} faces built")
                if faces:
                    self._sketch_faces[i] = faces
            except Exception as ex:
                print(f"[Sketch] Could not build faces for entry {i}: {ex}")

    # ------------------------------------------------------------------
    # Pick
    # ------------------------------------------------------------------

    def _pick_sketch_face(self, ray_origin: np.ndarray, ray_dir: np.ndarray
                          ) -> tuple[int | None, int | None]:
        """
        Ray-test all committed sketch faces.
        Returns (history_index, face_index_within_entry) or (None, None).
        """
        from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
        from OCP.gp import gp_Lin, gp_Pnt, gp_Dir

        best_t    = np.inf
        best_hidx = None
        best_fidx = None

        d = ray_dir
        o = ray_origin
        try:
            line = gp_Lin(
                gp_Pnt(float(o[0]), float(o[1]), float(o[2])),
                gp_Dir(float(d[0]), float(d[1]), float(d[2])),
            )
        except Exception:
            return None, None

        for hidx, faces in self._sketch_faces.items():
            for fidx, face in enumerate(faces):
                try:
                    inter = BRepIntCurveSurface_Inter()
                    inter.Init(face.wrapped, line, 1e-6)
                    while inter.More():
                        t = inter.W()
                        if 0 < t < best_t:
                            best_t    = t
                            best_hidx = hidx
                            best_fidx = fidx
                        inter.Next()
                except Exception:
                    pass

        return best_hidx, best_fidx

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw_sketch_faces(self):
        """
        Draw committed sketch faces as semi-transparent filled polygons
        with boundary outlines.  Selected entry is drawn more opaque.
        """
        if not self._sketch_faces:
            return

        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_UniformAbscissa

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for hidx, faces in self._sketch_faces.items():
            is_selected = (hidx == self._selected_sketch_entry)
            for face in faces:
                try:
                    BRepMesh_IncrementalMesh(face.wrapped, 0.1)
                    location = face.wrapped.Location()
                    facing   = BRep_Tool.Triangulation_s(face.wrapped, location)
                    if facing is None:
                        continue

                    glColor4f(0.29, 0.43, 0.54, 0.45 if is_selected else 0.20)
                    glBegin(GL_TRIANGLES)
                    for tri_idx in range(1, facing.NbTriangles() + 1):
                        tri = facing.Triangle(tri_idx)
                        n1, n2, n3 = tri.Get()
                        for ni in (n1, n2, n3):
                            node = facing.Node(ni)
                            glVertex3f(node.X(), node.Y(), node.Z())
                    glEnd()

                    if is_selected:
                        glColor4f(0.48, 0.70, 0.84, 0.90)
                        glLineWidth(1.8)
                    else:
                        glColor4f(0.29, 0.43, 0.54, 0.50)
                        glLineWidth(1.2)

                    exp = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
                    while exp.More():
                        edge = exp.Current()
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
                        exp.Next()
                    glLineWidth(1.0)

                except Exception as ex:
                    print(f"[Sketch] draw_sketch_faces error: {ex}")

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Screen-space snap radius
    # ------------------------------------------------------------------

    def _snap_radius_mm(self, snap_pixels: float = 14.0) -> float:
        """
        Convert a fixed pixel radius to world-mm at the sketch plane depth.
        Gives consistent snap feel regardless of zoom level.
        """
        from OpenGL.GLU import gluUnProject

        if self._modelview is None or self._sketch is None:
            return self.camera.distance * 0.015
        try:
            dpr = self.devicePixelRatio()
            cx  = self.width()  * 0.5 * dpr
            cy  = self.height() * 0.5 * dpr

            near0 = np.array(gluUnProject(cx,       cy, 0.0,
                             self._modelview, self._projection, self._viewport))
            near1 = np.array(gluUnProject(cx + 1.0, cy, 0.0,
                             self._modelview, self._projection, self._viewport))
            far0  = np.array(gluUnProject(cx,       cy, 1.0,
                             self._modelview, self._projection, self._viewport))
            far1  = np.array(gluUnProject(cx + 1.0, cy, 1.0,
                             self._modelview, self._projection, self._viewport))

            plane = self._sketch.plane
            _, pt0 = plane.ray_intersect(near0, far0 - near0)
            _, pt1 = plane.ray_intersect(near1, far1 - near1)

            if pt0 is not None and pt1 is not None:
                mm_per_px = float(np.linalg.norm(pt1 - pt0))
                return max(0.01, mm_per_px * snap_pixels)
        except Exception:
            pass
        return self.camera.distance * 0.015
