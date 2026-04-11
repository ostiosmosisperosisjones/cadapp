"""
gui/mainwindow.py
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QInputDialog, QSplitter
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt

from viewer.viewport import Viewport
from viewer.mesh import Mesh
from cad.importer import load_step
from cad.history import History
from cad.workspace import Workspace
from gui.sidebar import Sidebar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("cadapp")
        self.resize(1280, 768)
        self._viewport: Viewport | None = None
        self._sidebar:  Sidebar  | None = None
        self._build_menu()
        self._build_statusbar()

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setStyleSheet("""
            QStatusBar {
                background: #161616;
                color: #555;
                font-size: 11px;
                border-top: 1px solid #2a2a2a;
            }
            QStatusBar::item { border: none; }
        """)
        from PyQt6.QtWidgets import QLabel
        self._sketch_label = QLabel("")
        self._sketch_label.setStyleSheet(
            "color: #4fc3f7; font-weight: bold; padding-left: 8px;")
        sb.addWidget(self._sketch_label)

        self._sel_label = QLabel("")
        self._sel_label.setStyleSheet("color: #888; padding-right: 8px;")
        sb.addPermanentWidget(self._sel_label)

    def _update_sketch_label(self, in_sketch: bool):
        if not in_sketch:
            self._sketch_label.setText("")
            return
        sketch = self._viewport._sketch if self._viewport else None
        if sketch is None:
            self._sketch_label.setText("")
            return

        from cad.sketch import SketchTool
        from cad.prefs import prefs

        # Build hint strings from current keybinds so they stay in sync
        k_line    = prefs.key("sketch_line")
        k_include = prefs.key("sketch_include")
        k_commit  = prefs.key("sketch_commit")

        tool_hints = {
            SketchTool.NONE: (
                f"✏  SKETCH MODE  —  "
                f"{k_line} = line  |  "
                f"{k_include} = include geometry  |  "
                f"{k_commit} = commit  |  ESC = exit"
            ),
            SketchTool.LINE: (
                f"✏  LINE TOOL  —  click to draw  |  "
                f"{k_commit} = commit  |  ESC = cancel"
            ),
            SketchTool.CIRCLE: (
                f"✏  CIRCLE TOOL  —  click to draw  |  ESC = cancel"
            ),
            SketchTool.ARC: (
                f"✏  ARC TOOL  —  click to draw  |  ESC = cancel"
            ),
        }
        self._sketch_label.setText(
            tool_hints.get(sketch.tool, "✏  SKETCH MODE"))

    def _update_selection_label(self):
        if self._viewport is None:
            self._sel_label.setText("")
            return
        text = self._viewport.selection.status_text()
        self._sel_label.setText(text)

    def _build_menu(self):
        from cad.prefs import prefs
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_act  = QAction("Open STEP…", self)
        open_act.triggered.connect(self.open_step)
        file_menu.addAction(open_act)

        view_menu = menubar.addMenu("View")
        self._proj_action = QAction("Orthographic", self)
        self._proj_action.setCheckable(True)
        self._proj_action.setShortcut(prefs.key("projection_toggle"))
        self._proj_action.triggered.connect(self._toggle_projection)
        view_menu.addAction(self._proj_action)

        view_menu.addSeparator()
        prefs_act = QAction("Preferences…", self)
        prefs_act.triggered.connect(self._open_prefs)
        view_menu.addAction(prefs_act)

        ops_menu = menubar.addMenu("Operations")

        extrude_act = QAction(
            f"Extrude face…  ({prefs.key('extrude')})", self)
        extrude_act.triggered.connect(self._menu_extrude)
        ops_menu.addAction(extrude_act)

        ops_menu.addSeparator()

        undo_act = QAction("Undo", self)
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(
            lambda: self._viewport and self._viewport.handle_undo())
        ops_menu.addAction(undo_act)

        redo_act = QAction("Redo", self)
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(
            lambda: self._viewport and self._viewport.handle_redo())
        ops_menu.addAction(redo_act)

    def _open_prefs(self):
        from gui.prefs_dialog import PrefsDialog
        dlg = PrefsDialog(self)
        if dlg.exec() and self._viewport:
            from OpenGL.GL import glClearColor
            from cad.prefs import prefs
            r, g, b = prefs.background_color
            self._viewport.makeCurrent()
            glClearColor(r, g, b, 1.0)
            self._viewport.doneCurrent()
            self._viewport.update()

    def _toggle_projection(self):
        if self._viewport:
            self._viewport.toggle_projection()
            is_ortho = self._viewport.camera.ortho
            self._proj_action.setChecked(is_ortho)
            self._proj_action.setText(
                "Orthographic" if is_ortho else "Perspective")

    def _sync_proj_menu(self, is_ortho: bool):
        self._proj_action.setChecked(is_ortho)
        self._proj_action.setText(
            "Orthographic" if is_ortho else "Perspective")

    def _menu_extrude(self):
        if not self._viewport:
            return
        if self._viewport.selection.face_count == 0:
            self.statusBar().showMessage("Select a face first.", 3000)
            return
        sf = self._viewport.selection.single_face or \
             self._viewport.selection.faces[0]
        self._show_extrude_dialog(sf.body_id, sf.face_idx)

    def _show_extrude_dialog(self, body_id: str, face_idx: int):
        dist, ok = QInputDialog.getDouble(
            self, "Extrude",
            "Distance (positive = add material, negative = cut):",
            value=5.0, min=-1000.0, max=1000.0, decimals=3,
        )
        if not ok:
            return
        self._viewport.do_extrude(body_id, face_idx, dist)

    def open_step(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open STEP File", "", "STEP Files (*.step *.stp)"
        )
        if path:
            self.load(path)

    def load(self, path: str):
        print(f"Loading {path}…")
        compound = load_step(path)

        workspace = Workspace()
        history   = History()
        workspace.history = history

        solids = list(compound.solids())
        if not solids:
            solids = [compound]

        basename = os.path.splitext(os.path.basename(path))[0]

        for i, solid in enumerate(solids):
            name = (f"{basename}" if len(solids) == 1
                    else f"{basename}  [{i+1}]")
            body = workspace.add_body(name, solid)
            history.push(
                label        = f"Import  {name}",
                operation    = "import",
                params       = {"path": path, "solid_index": i},
                body_id      = body.id,
                face_ref     = None,
                shape_before = None,
                shape_after  = solid,
            )
            print(f"  Body '{name}' — {len(list(solid.faces()))} faces")

        vp = Viewport(workspace, history)
        vp.build_meshes()
        vp.fit_camera_to_scene()
        vp.camera_projection_changed = self._sync_proj_menu
        vp.request_extrude_distance  = self._show_extrude_dialog

        sidebar = Sidebar(workspace, history)
        sidebar.seek_requested.connect(vp.seek_history)
        sidebar.replay_requested.connect(vp.do_replay)
        sidebar.body_visibility_changed.connect(vp.set_body_visible)
        vp.history_changed.connect(sidebar.refresh)
        vp.body_selected.connect(sidebar.parts_panel.set_selected_body)
        vp.body_selected.connect(sidebar.history_panel.set_selected_body)
        sidebar.parts_panel.body_selected.connect(vp.set_active_body)
        sidebar.parts_panel.body_selected.connect(
            sidebar.history_panel.set_selected_body)
        sidebar.history_panel.sketch_vis_changed.connect(vp.update)
        sidebar.history_panel.reenter_sketch_requested.connect(vp._reenter_sketch)

        vp.selection_changed.connect(self._update_selection_label)
        vp.sketch_mode_changed.connect(self._update_sketch_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(vp)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 1070])
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        self._viewport = vp
        self._sidebar  = sidebar
        self.setCentralWidget(splitter)
        print("Done.")
