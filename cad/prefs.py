"""
cad/prefs.py

User preferences — colors, display settings, and anything else the user
might want to persist across sessions.

Stored as JSON in the platform config directory:
  Linux/Mac : ~/.config/cadapp/prefs.json
  Windows   : %APPDATA%/cadapp/prefs.json

Usage
-----
    from cad.prefs import prefs          # the singleton
    prefs.body_color                     # read
    prefs.body_color = (0.5, 0.6, 0.7)  # write (in-memory only)
    prefs.save()                         # persist to disk

All color values are (R, G, B) tuples with components in [0.0, 1.0].
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple


Color3 = Tuple[float, float, float]


def _p(r, g, b) -> Color3:
    return (float(r), float(g), float(b))


@dataclass
class Prefs:
    # ------------------------------------------------------------------
    # Viewport background
    # ------------------------------------------------------------------
    background_color:       Color3 = field(default_factory=lambda: _p(0.15, 0.15, 0.15))

    # ------------------------------------------------------------------
    # Body colors
    # ------------------------------------------------------------------
    body_color:             Color3 = field(default_factory=lambda: _p(0.55, 0.65, 0.75))
    body_color_active:      Color3 = field(default_factory=lambda: _p(0.65, 0.75, 0.85))

    # ------------------------------------------------------------------
    # Edge wireframe
    # ------------------------------------------------------------------
    edge_color:             Color3 = field(default_factory=lambda: _p(0.20, 0.20, 0.22))
    edge_width:             float  = 1.2
    show_edges:             bool   = True

    # ------------------------------------------------------------------
    # Selection highlights
    # ------------------------------------------------------------------
    face_selected_color:    Color3 = field(default_factory=lambda: _p(0.95, 0.60, 0.10))
    edge_hovered_color:     Color3 = field(default_factory=lambda: _p(0.90, 0.90, 0.30))
    edge_selected_color:    Color3 = field(default_factory=lambda: _p(1.00, 1.00, 0.40))
    vertex_hovered_color:   Color3 = field(default_factory=lambda: _p(1.00, 1.00, 1.00))
    vertex_selected_color:  Color3 = field(default_factory=lambda: _p(1.00, 0.85, 0.20))

    # ------------------------------------------------------------------
    # Sketch overlay
    # ------------------------------------------------------------------
    sketch_line_color:      Color3 = field(default_factory=lambda: _p(0.05, 0.05, 0.05))
    sketch_reference_color: Color3 = field(default_factory=lambda: _p(0.45, 0.55, 0.75))
    sketch_preview_color:   Color3 = field(default_factory=lambda: _p(0.95, 0.75, 0.10))
    sketch_cursor_color:    Color3 = field(default_factory=lambda: _p(0.10, 0.10, 0.10))
    sketch_axis_x_color:    Color3 = field(default_factory=lambda: _p(0.85, 0.20, 0.20))
    sketch_axis_y_color:    Color3 = field(default_factory=lambda: _p(0.20, 0.75, 0.30))
    sketch_grid_major_color:Color3 = field(default_factory=lambda: _p(0.35, 0.45, 0.60))
    sketch_grid_minor_color:Color3 = field(default_factory=lambda: _p(0.22, 0.30, 0.42))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _config_path() -> str:
        if os.name == 'nt':
            base = os.environ.get('APPDATA', os.path.expanduser('~'))
        else:
            base = os.environ.get('XDG_CONFIG_HOME',
                                  os.path.join(os.path.expanduser('~'), '.config'))
        return os.path.join(base, 'cadapp', 'prefs.json')

    def save(self):
        path = self._config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def load(self):
        path = self._config_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for key, val in data.items():
                if hasattr(self, key):
                    # Colors are stored as lists in JSON, convert back to tuple
                    if isinstance(val, list):
                        val = tuple(val)
                    setattr(self, key, val)
        except Exception as e:
            print(f"[Prefs] Could not load preferences: {e}")

    def reset(self):
        """Reset all preferences to defaults."""
        defaults = Prefs()
        for key in asdict(self):
            setattr(self, key, getattr(defaults, key))


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------

prefs = Prefs()
prefs.load()
