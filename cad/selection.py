"""
cad/selection.py

SelectionSet — tracks which faces, edges, vertices, and sketch edges
are currently selected.

Designed to be the single source of truth for selection state.
Viewport holds one instance and mutates it on click; anything that
needs to read selection (extrude, status bar, sketch include) reads
from here.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True, eq=True)
class FaceSel:
    """A selected face — identified by body + face index."""
    body_id:  str
    face_idx: int


@dataclass(frozen=True, eq=True)
class VertexSel:
    """A selected vertex — identified by body + vertex index in the mesh."""
    body_id:    str
    vertex_idx: int


@dataclass(frozen=True, eq=True)
class EdgeSel:
    """A selected edge — identified by body + edge index in topo_edges."""
    body_id:  str
    edge_idx: int


@dataclass(frozen=True, eq=True)
class SketchEdgeSel:
    """
    A selected edge from a committed sketch entity.

    history_idx  : index of the sketch HistoryEntry
    entity_idx   : index within SketchEntry.entities
    """
    history_idx: int
    entity_idx:  int


class SelectionSet:
    """
    Mutable set of selected faces, edges, vertices, and sketch edges.

    Supports:
      - Single-click replace (the default)
      - Shift-click toggle (additive / deselect)
      - Clear all
    """

    def __init__(self):
        self._faces:         set[FaceSel]        = set()
        self._edges:         set[EdgeSel]         = set()
        self._vertices:      set[VertexSel]       = set()
        self._sketch_edges:  set[SketchEdgeSel]   = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def select_face(self, body_id: str, face_idx: int, additive: bool = False):
        item = FaceSel(body_id, face_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._sketch_edges.clear()
            self._faces.add(item)
        else:
            if item in self._faces:
                self._faces.discard(item)
            else:
                self._faces.add(item)

    def select_edge(self, body_id: str, edge_idx: int, additive: bool = False):
        item = EdgeSel(body_id, edge_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._sketch_edges.clear()
            self._edges.add(item)
        else:
            if item in self._edges:
                self._edges.discard(item)
            else:
                self._edges.add(item)

    def select_vertex(self, body_id: str, vertex_idx: int,
                      additive: bool = False):
        item = VertexSel(body_id, vertex_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._sketch_edges.clear()
            self._vertices.add(item)
        else:
            if item in self._vertices:
                self._vertices.discard(item)
            else:
                self._vertices.add(item)

    def select_sketch_edge(self, history_idx: int, entity_idx: int,
                           additive: bool = False):
        """Select an edge from a committed sketch entity."""
        item = SketchEdgeSel(history_idx, entity_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._sketch_edges.clear()
            self._sketch_edges.add(item)
        else:
            if item in self._sketch_edges:
                self._sketch_edges.discard(item)
            else:
                self._sketch_edges.add(item)

    def clear(self):
        self._faces.clear()
        self._edges.clear()
        self._vertices.clear()
        self._sketch_edges.clear()

    def clear_for_body(self, body_id: str):
        self._faces    = {f for f in self._faces    if f.body_id != body_id}
        self._edges    = {e for e in self._edges    if e.body_id != body_id}
        self._vertices = {v for v in self._vertices if v.body_id != body_id}
        # sketch edges are not keyed by body_id so no change needed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def faces(self) -> list[FaceSel]:
        return list(self._faces)

    @property
    def edges(self) -> list[EdgeSel]:
        return list(self._edges)

    @property
    def vertices(self) -> list[VertexSel]:
        return list(self._vertices)

    @property
    def sketch_edges(self) -> list[SketchEdgeSel]:
        return list(self._sketch_edges)

    @property
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def sketch_edge_count(self) -> int:
        return len(self._sketch_edges)

    @property
    def is_empty(self) -> bool:
        return not (self._faces or self._edges or
                    self._vertices or self._sketch_edges)

    def has_face(self, body_id: str, face_idx: int) -> bool:
        return FaceSel(body_id, face_idx) in self._faces

    def has_edge(self, body_id: str, edge_idx: int) -> bool:
        return EdgeSel(body_id, edge_idx) in self._edges

    def has_vertex(self, body_id: str, vertex_idx: int) -> bool:
        return VertexSel(body_id, vertex_idx) in self._vertices

    @property
    def single_face(self) -> FaceSel | None:
        if len(self._faces) == 1:
            return next(iter(self._faces))
        return None

    def status_text(self) -> str:
        parts = []
        if self._faces:
            n = len(self._faces)
            parts.append(f"{n} face{'s' if n != 1 else ''}")
        if self._edges:
            n = len(self._edges)
            parts.append(f"{n} edge{'s' if n != 1 else ''}")
        if self._vertices:
            n = len(self._vertices)
            parts.append(f"{n} vertex" if n == 1 else f"{n} vertices")
        if self._sketch_edges:
            n = len(self._sketch_edges)
            parts.append(f"{n} sketch edge{'s' if n != 1 else ''}")
        return ",  ".join(parts) + "  selected" if parts else ""
