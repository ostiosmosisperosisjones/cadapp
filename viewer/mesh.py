"""
viewer/mesh.py

Mesh in true world coordinates — no centering or normalization.
Vertices are exactly as OCCT/build123d gives them (millimeters).
The camera fits itself to the scene bounding box instead.
"""

import uuid
import numpy as np
from ocp_tessellate.tessellator import tessellate
from OpenGL.GL import *


class Mesh:
    def __init__(self, shape):
        tess = tessellate(
            shape.wrapped,
            cache_key=f"mesh_{uuid.uuid4().hex}",
            deviation=0.01,
            quality=0.01,
            angular_tolerance=0.1,
        )

        self.triangles_per_face = np.array(tess['triangles_per_face'], dtype=np.int32)
        self.shape      = shape
        self.occt_faces = list(shape.faces())

        if len(self.occt_faces) != len(self.triangles_per_face):
            raise RuntimeError(
                f"Face count mismatch: tessellator gave "
                f"{len(self.triangles_per_face)} faces but "
                f"shape.faces() gave {len(self.occt_faces)}."
            )

        self.verts   = np.array(tess['vertices'], dtype=np.float32).reshape(-1, 3)
        self.tris    = np.array(tess['triangles'], dtype=np.uint32).reshape(-1, 3)
        self.normals = np.array(tess['normals'],   dtype=np.float32).reshape(-1, 3)
        self.edges   = np.array(tess['edges'],     dtype=np.float32).reshape(-1, 3)
        self.tri_count = len(self.tris) * 3

        # World-space bounding box
        self.bbox_min = self.verts.min(axis=0)
        self.bbox_max = self.verts.max(axis=0)

        # Legacy compatibility — code that still references these gets no-op values
        self.center = np.array([0.0, 0.0, 0.0])
        self.scale  = 1.0

        self.vbo_verts   = None
        self.vbo_normals = None
        self.vbo_edges   = None
        self.ebo         = None

    def upload(self):
        self.vbo_verts = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_verts)
        glBufferData(GL_ARRAY_BUFFER, self.verts.nbytes, self.verts, GL_STATIC_DRAW)

        self.vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        indices = self.tris.flatten().astype(np.uint32)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        self.vbo_edges = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edges)
        glBufferData(GL_ARRAY_BUFFER, self.edges.nbytes,
                     self.edges.astype(np.float32), GL_STATIC_DRAW)
        self.edge_count = len(self.edges)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_verts)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glNormalPointer(GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_TRIANGLES, self.tri_count, GL_UNSIGNED_INT, None)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw_edges(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edges)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINES, 0, self.edge_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def get_face_triangle_range(self, face_idx):
        start = int(self.triangles_per_face[:face_idx].sum())
        count = int(self.triangles_per_face[face_idx])
        return start, count
