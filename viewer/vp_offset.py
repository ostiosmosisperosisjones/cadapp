"""
viewer/vp_offset.py

Offset tool panel wiring — mixed into Viewport.
"""

from __future__ import annotations


class VpOffsetMixin:

    def _show_offset_panel(self):
        from gui.offset_panel import OffsetPanel
        if getattr(self, '_offset_panel', None) is not None:
            return
        panel = OffsetPanel(parent=self)
        panel.confirmed.connect(self._on_offset_confirmed)
        panel.cancelled.connect(self._close_offset_panel)
        self._offset_panel = panel
        self._position_offset_panel()
        panel.show()
        panel.setFocus()

    def _position_offset_panel(self):
        p = getattr(self, '_offset_panel', None)
        if p is None:
            return
        margin = 16
        p.move(margin, margin)

    def _close_offset_panel(self):
        p = getattr(self, '_offset_panel', None)
        if p is None:
            return
        p.close()
        self._offset_panel = None
        if self._sketch is not None:
            self._sketch.cancel_tool()
        self.setFocus()
        self.update()

    def _on_offset_confirmed(self, dist_mm: float):
        from cad.sketch_tools.offset import OffsetTool
        if self._sketch is None:
            return
        tool = self._sketch._active_tool
        if not isinstance(tool, OffsetTool):
            return
        tool.apply_offset(dist_mm, self._sketch)
        self._close_offset_panel()
        self.update()
