"""
cad/history.py

Single flat workspace history with cursor-based undo/redo and parametric replay.
Each entry is tagged with a body_id so operations are scoped to their body.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from cad.face_ref import FaceRef


@dataclass
class HistoryEntry:
    label:        str
    operation:    str
    params:       dict
    body_id:      str        # which body this operation targets
    face_ref:     FaceRef | None
    shape_before: Any        # shape of the body before this op
    shape_after:  Any        # shape of the body after this op


class History:
    def __init__(self):
        self._entries: list[HistoryEntry] = []
        self._cursor:  int = -1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, label, operation, params, body_id,
             face_ref, shape_before, shape_after) -> HistoryEntry:
        # Wipe redo stack
        if self._cursor < len(self._entries) - 1:
            self._entries = self._entries[:self._cursor + 1]
        entry = HistoryEntry(
            label=label, operation=operation, params=params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=shape_after,
        )
        self._entries.append(entry)
        self._cursor = len(self._entries) - 1
        return entry

    def undo(self) -> HistoryEntry | None:
        if self._cursor <= 0:
            return None
        entry = self._entries[self._cursor]
        self._cursor -= 1
        return entry

    def redo(self) -> HistoryEntry | None:
        if self._cursor >= len(self._entries) - 1:
            return None
        self._cursor += 1
        return self._entries[self._cursor]

    def seek(self, index: int) -> bool:
        """Move cursor to index. Returns True if index is valid."""
        if index < 0 or index >= len(self._entries):
            return False
        self._cursor = index
        return True

    # ------------------------------------------------------------------
    # Parametric replay
    # ------------------------------------------------------------------

    def replay_from(self, index: int) -> tuple[bool, str]:
        """
        Re-execute all entries from *index* onward for their respective bodies.
        Only entries whose body_id matches the edited entry's body_id are
        re-executed — other bodies are unaffected.

        Returns (success, error_message).
        """
        from cad.operations import REGISTRY

        if index < 0 or index >= len(self._entries):
            return False, f"Index {index} out of range"

        # Which body are we replaying?
        target_body_id = self._entries[index].body_id

        # Starting shape for this body is the shape_before of entry[index]
        current_shape = self._entries[index].shape_before
        if current_shape is None and self._entries[index].operation != "import":
            return False, f"Entry {index} has no shape_before"

        for i in range(index, len(self._entries)):
            entry = self._entries[i]

            # Skip entries for other bodies
            if entry.body_id != target_body_id:
                continue

            if entry.operation == "import":
                current_shape = entry.shape_after
                continue

            executor = REGISTRY.get(entry.operation)
            if executor is None:
                return False, f"No executor for operation '{entry.operation}'"

            if entry.face_ref is None:
                return False, f"Entry {i} '{entry.label}' has no face_ref"

            face_idx, _ = entry.face_ref.find_in(current_shape)
            if face_idx is None:
                return False, (
                    f"Could not find face for entry {i} '{entry.label}'. "
                    f"FaceRef: normal={entry.face_ref.normal} "
                    f"area={entry.face_ref.area:.3f}"
                )

            shape_before = current_shape
            try:
                new_shape = executor(current_shape, face_idx, entry.params)
            except Exception as ex:
                return False, f"Entry {i} '{entry.label}' failed: {ex}"

            entry.shape_before = shape_before
            entry.shape_after  = new_shape
            current_shape      = new_shape
            entry.label        = _make_label(entry.operation, entry.params)

        return True, ""

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def entries_for_body(self, body_id: str) -> list[tuple[int, HistoryEntry]]:
        """Return [(global_index, entry)] for all entries belonging to body_id."""
        return [(i, e) for i, e in enumerate(self._entries)
                if e.body_id == body_id]

    @property
    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._entries) - 1

    def __len__(self):
        return len(self._entries)


def _make_label(operation: str, params: dict) -> str:
    if operation == "extrude":
        return f"Extrude  +{params.get('distance', 0):.3f}mm"
    if operation == "cut":
        return f"Cut  -{abs(params.get('distance', 0)):.3f}mm"
    return operation.capitalize()
