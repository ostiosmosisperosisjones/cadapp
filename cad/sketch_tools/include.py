"""
cad/sketch_tools/include.py

IncludeTool — project selected edges / vertices onto the sketch plane
as ReferenceEntity instances.

Unlike drawing tools, Include is a one-shot action (triggered by pressing I),
not a persistent mode.  It is not in TOOLS and does not subclass BaseTool.
Call IncludeTool.apply(sketch, selection, meshes) directly.
"""

from __future__ import annotations
import numpy as np


class IncludeTool:

    @staticmethod
    def apply(sketch, selection, meshes: dict) -> int:
        """
        Project all selected edges and vertices onto sketch.plane as
        ReferenceEntity instances appended to sketch.entities.

        Parameters
        ----------
        sketch    : SketchMode
        selection : SelectionSet
        meshes    : dict[body_id, Mesh]

        Returns the number of reference entities added.
        """
        from cad.sketch import ReferenceEntity

        added = 0

        # Selected edges → project each polyline point, preserve OCC edge
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue
            world_pts = mesh.topo_edges[es.edge_idx]   # (N, 3) float32
            uv_pts    = [sketch.plane.project_point(p) for p in world_pts]
            if len(uv_pts) >= 2:
                occ_edge = None
                if hasattr(mesh, 'topo_edges_occ') and \
                        es.edge_idx < len(mesh.topo_edges_occ):
                    occ_edge = mesh.topo_edges_occ[es.edge_idx]
                sketch.entities.append(
                    ReferenceEntity(
                        uv_pts,
                        source_type = 'edge',
                        occ_edges   = [occ_edge] if occ_edge is not None else None,
                    )
                )
                added += 1

        # Selected vertices → single-point reference
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            world_pt = mesh.topo_verts[vs.vertex_idx]
            uv_pt    = sketch.plane.project_point(world_pt)
            sketch.entities.append(
                ReferenceEntity([uv_pt], source_type='vertex'))
            added += 1

        return added
