"""
cad/selection.py

SelectionSet — tracks which faces and vertices are currently selected.

Designed to be the single source of truth for selection state.
Viewport holds one instance and mutates it on click; anything that
needs to read selection (extrude, status bar, future sketch) reads
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


class SelectionSet:
    """
    Mutable set of selected faces, edges, and vertices.

    Supports:
      - Single-click replace (the default)
      - Shift-click toggle (additive / deselect)
      - Clear all
    """

    def __init__(self):
        self._faces:    set[FaceSel]   = set()
        self._edges:    set[EdgeSel]   = set()
        self._vertices: set[VertexSel] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def select_face(self, body_id: str, face_idx: int, additive: bool = False):
        """
        Select a face.
        additive=False  → clear everything else first (normal click)
        additive=True   → toggle this face, keep others (Shift-click)
        """
        item = FaceSel(body_id, face_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._faces.add(item)
        else:
            if item in self._faces:
                self._faces.discard(item)
            else:
                self._faces.add(item)

    def select_edge(self, body_id: str, edge_idx: int, additive: bool = False):
        """Select an edge, same additive logic as select_face."""
        item = EdgeSel(body_id, edge_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._edges.add(item)
        else:
            if item in self._edges:
                self._edges.discard(item)
            else:
                self._edges.add(item)

    def select_vertex(self, body_id: str, vertex_idx: int, additive: bool = False):
        """Select a vertex, same additive logic as select_face."""
        item = VertexSel(body_id, vertex_idx)
        if not additive:
            self._faces.clear()
            self._edges.clear()
            self._vertices.clear()
            self._vertices.add(item)
        else:
            if item in self._vertices:
                self._vertices.discard(item)
            else:
                self._vertices.add(item)

    def clear(self):
        self._faces.clear()
        self._edges.clear()
        self._vertices.clear()

    def clear_for_body(self, body_id: str):
        """Remove all selections belonging to a specific body (called after rebuild)."""
        self._faces    = {f for f in self._faces    if f.body_id != body_id}
        self._edges    = {e for e in self._edges    if e.body_id != body_id}
        self._vertices = {v for v in self._vertices if v.body_id != body_id}

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
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def is_empty(self) -> bool:
        return not self._faces and not self._edges and not self._vertices

    def has_face(self, body_id: str, face_idx: int) -> bool:
        return FaceSel(body_id, face_idx) in self._faces

    def has_edge(self, body_id: str, edge_idx: int) -> bool:
        return EdgeSel(body_id, edge_idx) in self._edges

    def has_vertex(self, body_id: str, vertex_idx: int) -> bool:
        return VertexSel(body_id, vertex_idx) in self._vertices

    # ------------------------------------------------------------------
    # Convenience — for code that only cares about "the" selected face
    # (single-select path, e.g. current extrude logic)
    # ------------------------------------------------------------------

    @property
    def single_face(self) -> FaceSel | None:
        """Return the sole selected face, or None if 0 or 2+ are selected."""
        if len(self._faces) == 1:
            return next(iter(self._faces))
        return None

    # ------------------------------------------------------------------
    # Status string
    # ------------------------------------------------------------------

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
        return ",  ".join(parts) + "  selected" if parts else ""
