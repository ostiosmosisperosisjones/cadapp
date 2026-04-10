"""
cad/sketch_tools/base.py

BaseTool — interface every sketch drawing tool must implement.

A tool instance is created when the user activates that tool and discarded
when they switch tools or exit sketch mode.  All in-progress state (e.g.
the first point of a line, the center of an arc) lives on the instance.

Methods
-------
handle_click(pt2d, sketch)
    Called on every left-click while this tool is active.
    pt2d  : np.ndarray shape (2,) — cursor position in sketch (u,v) coords
    sketch: SketchMode            — the active sketch (append entities here)
    Returns True if the click was consumed.

handle_mouse_move(pt2d, sketch)
    Called on every mouse-move while this tool is active.
    Update any preview state here (e.g. _cursor_2d on the sketch).

cancel()
    Called when the user presses ESC to cancel the current tool action
    without switching tools (e.g. abandon a half-drawn line).

cursor_2d
    Property — current cursor position for the overlay to read, or None.
"""

from __future__ import annotations
import numpy as np


class BaseTool:
    """Abstract base — subclass this for every sketch tool."""

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return None

    def handle_click(self, pt2d: np.ndarray, sketch) -> bool:
        raise NotImplementedError

    def handle_mouse_move(self, pt2d: np.ndarray, sketch) -> None:
        pass

    def cancel(self) -> None:
        """Reset any in-progress state (called on ESC within the tool)."""
        pass
