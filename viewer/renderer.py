"""
viewer/renderer.py

OpenGL draw calls split into two phases so the caller can read the depth
buffer between them (for hover occlusion testing):

    draw_opaque(...)   — solid geometry + wireframe edges
    draw_overlays(...) — selected/hovered highlights on top
"""

import ctypes
from OpenGL.GL import *


def draw_opaque(meshes, workspace, selection):
    """
    Phase 1: draw solid geometry and wireframe edges.
    Depth buffer is valid and complete after this returns.
    """
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    for body_id, mesh in meshes.items():
        body = workspace.bodies.get(body_id)
        if body and not body.visible:
            continue
        glColor3f(0.6, 0.82, 1.0) if body_id == workspace.active_body_id \
            else glColor3f(0.45, 0.60, 0.78)
        mesh.draw()

    glDisable(GL_LIGHTING)

    # Selected face highlights
    if selection.face_count > 0:
        glColor3f(1.0, 0.4, 0.0)
        glDisable(GL_DEPTH_TEST)
        glEnableClientState(GL_VERTEX_ARRAY)
        for face_sel in selection.faces:
            mesh = meshes.get(face_sel.body_id)
            if mesh is None:
                continue
            start, count = mesh.get_face_triangle_range(face_sel.face_idx)
            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo_verts)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            glDrawElements(GL_TRIANGLES, count * 3, GL_UNSIGNED_INT,
                           ctypes.c_void_p(start * 3 * 4))
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glEnable(GL_DEPTH_TEST)

    # Wireframe edges
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(3.0)
    for body_id, mesh in meshes.items():
        body = workspace.bodies.get(body_id)
        if body and not body.visible:
            continue
        mesh.draw_edges()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def draw_overlays(meshes, selection, hovered_vertex, hovered_edge):
    """
    Phase 2: edge/vertex highlights drawn on top, depth test off.
    Call this AFTER hover.rebuild() so occlusion is already baked in.
    """
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)

    # Selected edges — cyan
    if selection.edge_count > 0:
        glLineWidth(3.5)
        glColor3f(0.0, 0.85, 1.0)
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue
            _draw_polyline(mesh.topo_edges[es.edge_idx])
        glLineWidth(1.0)

    # Hovered edge — bright cyan
    hov_ebody, hov_eidx = hovered_edge
    if hov_ebody is not None:
        mesh = meshes.get(hov_ebody)
        if mesh is not None and hov_eidx < len(mesh.topo_edges):
            glLineWidth(4.5)
            glColor3f(0.4, 1.0, 1.0)
            _draw_polyline(mesh.topo_edges[hov_eidx])
            glLineWidth(1.0)

    # Selected vertices — yellow
    if selection.vertex_count > 0:
        glPointSize(10.0)
        glColor3f(1.0, 0.85, 0.0)
        glBegin(GL_POINTS)
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            p = mesh.topo_verts[vs.vertex_idx]
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()

    # Hovered vertex — white
    hov_body, hov_idx = hovered_vertex
    if hov_body is not None:
        mesh = meshes.get(hov_body)
        if mesh is not None and hov_idx < len(mesh.topo_verts):
            p = mesh.topo_verts[hov_idx]
            glPointSize(12.0)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_POINTS)
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

    glPointSize(1.0)
    glDisable(GL_POINT_SMOOTH)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def _draw_polyline(pts):
    glBegin(GL_LINE_STRIP)
    for p in pts:
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()
