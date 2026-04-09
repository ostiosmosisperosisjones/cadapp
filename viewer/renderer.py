"""
viewer/renderer.py

OpenGL draw calls split into two phases:

    draw_opaque(...)   — solid geometry + optional wireframe edges
    draw_overlays(...) — selection/hover highlights + sketch overlays

All colors read from cad.prefs so they can be changed without touching
this file.
"""

import ctypes
from OpenGL.GL import *
from cad.prefs import prefs


def draw_opaque(meshes, workspace, selection):
    """
    Phase 1: solid geometry, optional wireframe, selected face highlights.
    Depth buffer is fully populated after this returns.
    """
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    for body_id, mesh in meshes.items():
        body = workspace.bodies.get(body_id)
        if body and not body.visible:
            continue
        c = prefs.body_color_active if body_id == workspace.active_body_id \
            else prefs.body_color
        glColor3f(*c)
        mesh.draw()

    glDisable(GL_LIGHTING)

    # Selected face highlights
    if selection.face_count > 0:
        glColor3f(*prefs.face_selected_color)
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

    # Wireframe edges — optional, respects prefs
    if prefs.show_edges:
        glColor3f(*prefs.edge_color)
        glLineWidth(prefs.edge_width)
        for body_id, mesh in meshes.items():
            body = workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            mesh.draw_edges()
        glLineWidth(1.0)

    glEnable(GL_LIGHTING)


def draw_overlays(meshes, selection, hovered_vertex, hovered_edge,
                  sketch=None, camera_distance: float = 1.0,
                  history=None):
    """
    Phase 2: hover/selection overlays + sketch overlays.

    Draws committed sketches from history first (persistent, dimmed),
    then the active sketch session on top (if any).

    Parameters
    ----------
    history : History | None
        If provided, all visible committed sketch entries are rendered.
    """
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)

    # Selected edges
    if selection.edge_count > 0:
        glLineWidth(3.0)
        glColor3f(*prefs.edge_selected_color)
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue
            _draw_polyline(mesh.topo_edges[es.edge_idx])
        glLineWidth(1.0)

    # Hovered edge
    hov_ebody, hov_eidx = hovered_edge
    if hov_ebody is not None:
        mesh = meshes.get(hov_ebody)
        if mesh is not None and hov_eidx < len(mesh.topo_edges):
            glLineWidth(4.0)
            glColor3f(*prefs.edge_hovered_color)
            _draw_polyline(mesh.topo_edges[hov_eidx])
            glLineWidth(1.0)

    # Selected vertices
    if selection.vertex_count > 0:
        glPointSize(10.0)
        glColor3f(*prefs.vertex_selected_color)
        glBegin(GL_POINTS)
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            p = mesh.topo_verts[vs.vertex_idx]
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()

    # Hovered vertex
    hov_body, hov_idx = hovered_vertex
    if hov_body is not None:
        mesh = meshes.get(hov_body)
        if mesh is not None and hov_idx < len(mesh.topo_verts):
            p = mesh.topo_verts[hov_idx]
            glPointSize(12.0)
            glColor3f(*prefs.vertex_hovered_color)
            glBegin(GL_POINTS)
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

    glPointSize(1.0)
    glDisable(GL_POINT_SMOOTH)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Sketch overlays — drawn last, on top of everything
    # ------------------------------------------------------------------
    from viewer.sketch_overlay import SketchOverlay
    overlay = SketchOverlay()

    # Committed sketches from history (persistent, dimmed)
    if history is not None:
        for entry in history.entries:
            if entry.operation != "sketch":
                continue
            se = entry.params.get("sketch_entry")
            if se is not None and se.visible:
                overlay.draw_committed(se, camera_distance)

    # Active sketch session (grid + axes + live entities + cursor)
    if sketch is not None:
        overlay.draw(sketch, camera_distance)


def _draw_polyline(pts):
    glBegin(GL_LINE_STRIP)
    for p in pts:
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()
