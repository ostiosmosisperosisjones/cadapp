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
from gui.history_panel import HistoryPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("cadapp")
        self.resize(1280, 768)
        self._viewport:      Viewport     | None = None
        self._history_panel: HistoryPanel | None = None
        self._build_menu()

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_act  = QAction("Open STEP…", self)
        open_act.triggered.connect(self.open_step)
        file_menu.addAction(open_act)

        view_menu = menubar.addMenu("View")
        self._proj_action = QAction("Orthographic", self)
        self._proj_action.setCheckable(True)
        self._proj_action.setShortcut("5")
        self._proj_action.triggered.connect(self._toggle_projection)
        view_menu.addAction(self._proj_action)

        ops_menu = menubar.addMenu("Operations")

        extrude_act = QAction("Extrude face…  (E)", self)
        extrude_act.triggered.connect(self._menu_extrude)
        ops_menu.addAction(extrude_act)

        ops_menu.addSeparator()

        undo_act = QAction("Undo", self)
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(lambda: self._viewport and self._viewport._do_undo())
        ops_menu.addAction(undo_act)

        redo_act = QAction("Redo", self)
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(lambda: self._viewport and self._viewport._do_redo())
        ops_menu.addAction(redo_act)

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
        if self._viewport.picked_face is None:
            self.statusBar().showMessage("Select a face first.", 3000)
            return
        self._show_extrude_dialog(
            self._viewport.picked_body_id,
            self._viewport.picked_face)

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

        # Split compound into individual solids → one Body each
        solids = list(compound.solids())
        if not solids:
            # Fallback: treat the whole thing as one body
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

        # Build meshes (no GL context yet — upload happens in initializeGL)
        vp = Viewport(workspace, history)
        vp.build_meshes()
        vp.fit_camera_to_scene()
        vp.camera_projection_changed = self._sync_proj_menu
        vp.request_extrude_distance  = self._show_extrude_dialog

        panel = HistoryPanel(workspace, history)
        panel.seek_requested.connect(vp.seek_history)
        panel.replay_requested.connect(vp.do_replay)
        vp.history_changed.connect(panel.refresh)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(panel)
        splitter.addWidget(vp)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 1070])
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        self._viewport      = vp
        self._history_panel = panel
        self.setCentralWidget(splitter)
        print("Done.")
