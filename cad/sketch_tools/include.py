"""
cad/sketch_tools/include.py

IncludeTool — project selected edges / vertices onto the sketch plane
as ReferenceEntity instances.

Every included edge now carries a parametric EdgeSource so that the
replay system can re-project the geometry when upstream operations change.

Unlike drawing tools, Include is a one-shot action (triggered by pressing I),
not a persistent mode.  It is not in TOOLS and does not subclass BaseTool.
Call IncludeTool.apply_with_history(sketch, selection, meshes, history) from
the viewport.
"""

from __future__ import annotations
import numpy as np


class IncludeTool:

    @staticmethod
    def apply_with_history(sketch, selection, meshes: dict, history) -> int:
        """
        Project all selected edges, vertices, and committed sketch line
        entities onto sketch.plane as ReferenceEntity instances appended
        to sketch.entities.

        Each included edge is stored with a stable EdgeSource so parametric
        replay can re-derive its world geometry and UV projection.

        Parameters
        ----------
        sketch    : SketchMode
        selection : SelectionSet
        meshes    : dict[body_id, Mesh]
        history   : History | None

        Returns the number of reference entities added.
        """
        from cad.sketch import ReferenceEntity
        from cad.edge_ref import EdgeRef, BodyEdgeSource, SketchEdgeSource

        added = 0

        # ------------------------------------------------------------------
        # Selected mesh edges
        # ------------------------------------------------------------------
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue

            world_pts = mesh.topo_edges[es.edge_idx]          # (N,3) float32
            uv_pts    = [sketch.plane.project_point(p) for p in world_pts]
            if len(uv_pts) < 2:
                continue

            occ_edge = None
            source   = None
            if es.edge_idx < len(mesh.topo_edges_occ):
                occ_edge = mesh.topo_edges_occ[es.edge_idx]
                edge_ref = EdgeRef.from_occ_edge(occ_edge)
                if edge_ref is not None:
                    source = BodyEdgeSource(es.body_id, edge_ref)

            sketch.entities.append(
                ReferenceEntity(
                    uv_pts,
                    source_type = 'edge',
                    occ_edges   = [occ_edge] if occ_edge is not None else None,
                    source      = source,
                )
            )
            added += 1

        # ------------------------------------------------------------------
        # Selected sketch edges (committed LineEntity from a history entry)
        # ------------------------------------------------------------------
        for se in selection.sketch_edges:
            if history is None:
                continue
            entries = history.entries
            if se.history_idx < 0 or se.history_idx >= len(entries):
                continue
            entry = entries[se.history_idx]
            sketch_entry = entry.params.get("sketch_entry")
            if sketch_entry is None:
                continue
            if se.entity_idx >= len(sketch_entry.entities):
                continue

            from cad.sketch import LineEntity
            ent = sketch_entry.entities[se.entity_idx]
            if not isinstance(ent, LineEntity):
                continue

            # World coords from source sketch's current plane cache
            p0_world = (sketch_entry.plane_origin
                        + float(ent.p0[0]) * sketch_entry.plane_x_axis
                        + float(ent.p0[1]) * sketch_entry.plane_y_axis)
            p1_world = (sketch_entry.plane_origin
                        + float(ent.p1[0]) * sketch_entry.plane_x_axis
                        + float(ent.p1[1]) * sketch_entry.plane_y_axis)

            uv0 = sketch.plane.project_point(p0_world)
            uv1 = sketch.plane.project_point(p1_world)

            source = SketchEdgeSource(se.history_idx, se.entity_idx)
            sketch.entities.append(
                ReferenceEntity([uv0, uv1], source_type='sketch_edge',
                                source=source))
            added += 1

        # ------------------------------------------------------------------
        # Selected vertices — point references, no parametric source yet
        # ------------------------------------------------------------------
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
