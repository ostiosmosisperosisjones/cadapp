"""
cad/sketch_tools/__init__.py

Sketch tool registry.

Adding a new tool:
  1. Create cad/sketch_tools/mytool.py with a class that subclasses BaseTool
  2. Import it here and add it to TOOLS
  3. Add its enum value to SketchTool in cad/sketch.py
  4. Add a key binding in viewer/viewport.py keyPressEvent
  That's it — the overlay, history, and commit flow pick it up automatically.
"""

from cad.sketch_tools.base    import BaseTool
from cad.sketch_tools.line    import LineTool
from cad.sketch_tools.include import IncludeTool

from cad.sketch import SketchTool

# Maps SketchTool enum value → tool class (not instance).
# Viewport calls TOOLS[enum]() to create a fresh instance on activation.
TOOLS: dict[SketchTool, type[BaseTool]] = {
    SketchTool.LINE: LineTool,
}

# IncludeTool is invoked directly (not a persistent drawing tool),
# so it's not in TOOLS but is importable from here for convenience.

__all__ = ["BaseTool", "LineTool", "IncludeTool", "TOOLS"]
