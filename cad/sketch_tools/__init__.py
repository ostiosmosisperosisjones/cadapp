"""
cad/sketch_tools/__init__.py

Sketch tool registry.

Adding a new draw tool:
  1. Create cad/sketch_tools/mytool.py with a class subclassing BaseTool
  2. Import it here and add it to TOOLS
  3. Add its enum value to SketchTool in cad/sketch.py
  4. Add a keybind in cad/prefs.py KEYBIND_DEFAULTS + KEYBIND_LABELS
  5. Add a prefs.matches() call in viewer/viewport.py keyPressEvent
  That's it — snap, overlay, history, and commit all work automatically.
"""

from cad.sketch_tools.base    import BaseTool
from cad.sketch_tools.line    import LineTool
from cad.sketch_tools.include import IncludeTool
from cad.sketch_tools.snap    import SnapEngine, SnapResult, SnapType

from cad.sketch import SketchTool

# Maps SketchTool enum value → tool class (instantiated fresh on activation)
TOOLS: dict[SketchTool, type[BaseTool]] = {
    SketchTool.LINE: LineTool,
}

# IncludeTool is a one-shot action, not a persistent drawing tool.
# Call IncludeTool.apply(sketch, selection, meshes) directly.

__all__ = [
    "BaseTool", "LineTool", "IncludeTool",
    "SnapEngine", "SnapResult", "SnapType",
    "TOOLS",
]
