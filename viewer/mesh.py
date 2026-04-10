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

        # True topological vertices — actual CAD corners where edges meet.
        self.topo_verts = _extract_topo_verts(shape)

        # True topological edges — each is a (N,3) polyline in world mm.
        # topo_edges_occ is the parallel list of raw TopoDS_Edge objects,
        # used when projecting reference geometry to preserve true curve type.
        self.topo_edges, self.topo_edges_occ = _extract_topo_edges(shape)

        # Legacy compatibility
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_topo_verts(shape) -> np.ndarray:
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    seen = set()
    pts  = []

    explorer = TopExp_Explorer(shape.wrapped, TopAbs_VERTEX)
    while explorer.More():
        vert = TopoDS.Vertex_s(explorer.Current())
        pnt  = BRep_Tool.Pnt_s(vert)
        key  = (round(pnt.X(), 4), round(pnt.Y(), 4), round(pnt.Z(), 4))
        if key not in seen:
            seen.add(key)
            pts.append([pnt.X(), pnt.Y(), pnt.Z()])
        explorer.Next()

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def _extract_topo_edges(shape) -> tuple[list[np.ndarray], list]:
    """
    Extract topological edges from a build123d shape via OCCT.

    Returns
    -------
    topo_edges     : list of (N, 3) float32 arrays — discretised polylines
    topo_edges_occ : list of TopoDS_Edge — the raw OCCT edge, parallel to above
    """
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.GCPnts import GCPnts_UniformAbscissa
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    edges     = []
    edges_occ = []
    seen      = set()

    explorer = TopExp_Explorer(shape.wrapped, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())

        try:
            adaptor = BRepAdaptor_Curve(edge)
            first   = adaptor.FirstParameter()
            last    = adaptor.LastParameter()

            discretiser = GCPnts_UniformAbscissa()
            discretiser.Initialize(adaptor, 64, first, last, 0.05)

            pts = []
            if discretiser.IsDone() and discretiser.NbPoints() >= 2:
                for i in range(1, discretiser.NbPoints() + 1):
                    p = adaptor.Value(discretiser.Parameter(i))
                    pts.append([p.X(), p.Y(), p.Z()])
            else:
                p0 = adaptor.Value(first)
                p1 = adaptor.Value(last)
                pts = [[p0.X(), p0.Y(), p0.Z()],
                       [p1.X(), p1.Y(), p1.Z()]]

            if len(pts) < 2:
                explorer.Next()
                continue

            arr = np.array(pts, dtype=np.float32)

            # Deduplicate by rounded midpoint
            mid = arr[len(arr) // 2]
            key = (round(float(mid[0]), 3),
                   round(float(mid[1]), 3),
                   round(float(mid[2]), 3))
            if key not in seen:
                seen.add(key)
                edges.append(arr)
                edges_occ.append(edge)

        except Exception:
            pass

        explorer.Next()

    return edges, edges_occ
