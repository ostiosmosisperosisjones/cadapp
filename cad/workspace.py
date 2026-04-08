"""
cad/workspace.py

Workspace — the top-level container for a CAD session.

Structure
---------
Workspace
  ├── bodies: dict[str, Body]   — all bodies, keyed by stable uuid
  └── history: History          — single flat timeline, entries tagged body_id

Body
  ├── id: str                   — stable uuid (survives rename)
  ├── name: str                 — user-visible, renameable
  ├── visible: bool
  └── source_shape              — the original imported solid (never mutated)

The current shape of a body is derived by replaying all history entries
tagged with that body_id up to the cursor. Workspace.current_shape(body_id)
does this efficiently by returning the shape_after of the last entry for
that body at or before the cursor.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Body:
    id:           str
    name:         str
    visible:      bool  = True
    source_shape: Any   = None   # original solid, set on import


class Workspace:
    def __init__(self):
        self.bodies:  dict[str, Body] = {}
        self.history: "History | None" = None   # set after History is constructed
        # The active body — operations target this one
        self._active_body_id: str | None = None

    # ------------------------------------------------------------------
    # Body management
    # ------------------------------------------------------------------

    def add_body(self, name: str, source_shape) -> Body:
        body = Body(
            id           = uuid.uuid4().hex,
            name         = name,
            source_shape = source_shape,
        )
        self.bodies[body.id] = body
        if self._active_body_id is None:
            self._active_body_id = body.id
        return body

    def remove_body(self, body_id: str):
        self.bodies.pop(body_id, None)
        if self._active_body_id == body_id:
            self._active_body_id = next(iter(self.bodies), None)

    @property
    def active_body(self) -> Body | None:
        return self.bodies.get(self._active_body_id)

    @property
    def active_body_id(self) -> str | None:
        return self._active_body_id

    def set_active_body(self, body_id: str):
        if body_id in self.bodies:
            self._active_body_id = body_id

    # ------------------------------------------------------------------
    # Shape queries
    # ------------------------------------------------------------------

    def current_shape(self, body_id: str) -> Any | None:
        """
        Return the current shape of a body — the shape_after of the last
        history entry for this body at or before the cursor.
        Falls back to source_shape if no operations have been applied.
        """
        if self.history is None:
            return self.bodies[body_id].source_shape if body_id in self.bodies else None

        entries  = self.history.entries
        cursor   = self.history.cursor
        result   = None

        for i, entry in enumerate(entries):
            if i > cursor:
                break
            if entry.body_id == body_id and entry.shape_after is not None:
                result = entry.shape_after

        return result or (self.bodies[body_id].source_shape
                          if body_id in self.bodies else None)

    def all_current_shapes(self) -> list[tuple[str, Any]]:
        """Return [(body_id, current_shape)] for all visible bodies."""
        result = []
        for body_id, body in self.bodies.items():
            if not body.visible:
                continue
            shape = self.current_shape(body_id)
            if shape is not None:
                result.append((body_id, shape))
        return result
