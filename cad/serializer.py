"""
cad/serializer.py

Save / load a workspace + history to a .vc project file.

Format: zip archive containing a single project.json.

On save  — serialize everything except OCCT shape objects.
On load  — deserialize, then rebuild all shapes by replaying history
           from scratch (same code path as undo/redo).

Import entries re-load geometry from the original STEP path.
If that file is missing the load still succeeds but the body has no shape
(the same error state the history system shows for any failed op).
"""

from __future__ import annotations
import json
import zipfile
import io
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers — numpy arrays ↔ plain lists
# ---------------------------------------------------------------------------

def _arr(v) -> list:
    return np.array(v, dtype=np.float64).tolist()


def _tup(v) -> list:
    return list(v)


# ---------------------------------------------------------------------------
# FaceRef
# ---------------------------------------------------------------------------

def _face_ref_to_dict(fr) -> dict | None:
    if fr is None:
        return None
    return {
        "normal":         _tup(fr.normal),
        "area":           fr.area,
        "centroid_perp":  _tup(fr.centroid_perp),
        "centroid_along": fr.centroid_along,
    }


def _face_ref_from_dict(d: dict | None):
    if d is None:
        return None
    from cad.face_ref import FaceRef
    return FaceRef(
        normal         = tuple(d["normal"]),
        area           = float(d["area"]),
        centroid_perp  = tuple(d["centroid_perp"]),
        centroid_along = float(d["centroid_along"]),
    )


# ---------------------------------------------------------------------------
# SketchPlaneSource
# ---------------------------------------------------------------------------

def _plane_source_to_dict(ps) -> dict | None:
    if ps is None:
        return None
    from cad.plane_ref import FacePlaneSource
    d = ps.to_dict()
    # FacePlaneSource.to_dict() was missing face_ref — add it here
    if isinstance(ps, FacePlaneSource):
        d["face_ref"] = _face_ref_to_dict(ps.face_ref)
    return d


def _plane_source_from_dict(d: dict | None):
    if d is None:
        return None
    t = d.get("type")
    if t == "world":
        from cad.plane_ref import WorldPlaneSource
        return WorldPlaneSource(d["axis"])
    if t == "face":
        from cad.plane_ref import FacePlaneSource
        return FacePlaneSource(
            body_id  = d["body_id"],
            face_ref = _face_ref_from_dict(d.get("face_ref")),
        )
    if t == "offset":
        from cad.plane_ref import OffsetPlaneSource
        return OffsetPlaneSource(
            parent   = _plane_source_from_dict(d["parent"]),
            distance = float(d["distance"]),
        )
    return None


# ---------------------------------------------------------------------------
# EdgeSource
# ---------------------------------------------------------------------------

def _edge_source_to_dict(es) -> dict | None:
    if es is None:
        return None
    from cad.edge_ref import BodyEdgeSource, SketchEdgeSource
    d = es.to_dict()
    if isinstance(es, BodyEdgeSource):
        er = es.edge_ref
        d["edge_ref"] = {
            "midpoint": _tup(er.midpoint),
            "length":   er.length,
            "tangent":  _tup(er.tangent),
        }
    return d


def _edge_source_from_dict(d: dict | None):
    if d is None:
        return None
    t = d.get("type")
    if t == "body_edge":
        from cad.edge_ref import BodyEdgeSource, EdgeRef
        er_d = d.get("edge_ref", {})
        er = EdgeRef(
            midpoint = tuple(er_d["midpoint"]),
            length   = float(er_d["length"]),
            tangent  = tuple(er_d["tangent"]),
        )
        return BodyEdgeSource(body_id=d["body_id"], edge_ref=er)
    if t == "sketch_edge":
        from cad.edge_ref import SketchEdgeSource
        return SketchEdgeSource(
            history_idx = int(d["history_idx"]),
            entity_idx  = int(d["entity_idx"]),
        )
    return None


# ---------------------------------------------------------------------------
# Sketch entities
# ---------------------------------------------------------------------------

def _entity_to_dict(ent) -> dict:
    from cad.sketch import LineEntity, ArcEntity, PointEntity, ReferenceEntity
    if isinstance(ent, LineEntity):
        d: dict[str, Any] = {
            "type": "line",
            "p0":   _arr(ent.p0),
            "p1":   _arr(ent.p1),
        }
        if ent.p0_snap is not None:
            d["p0_snap"] = [ent.p0_snap[0], ent.p0_snap[1].name]
        if ent.p1_snap is not None:
            d["p1_snap"] = [ent.p1_snap[0], ent.p1_snap[1].name]
        return d
    if isinstance(ent, ArcEntity):
        return {
            "type":        "arc",
            "center":      _arr(ent.center),
            "radius":      ent.radius,
            "start_angle": ent.start_angle,
            "end_angle":   ent.end_angle,
        }
    if isinstance(ent, PointEntity):
        return {"type": "point", "pos": _arr(ent.pos)}
    if isinstance(ent, ReferenceEntity):
        return {
            "type":        "reference",
            "points":      [_arr(p) for p in ent.points],
            "source_type": ent.source_type,
            "source":      _edge_source_to_dict(ent.source),
            # occ_edges intentionally omitted — re-resolved on replay
        }
    raise ValueError(f"Unknown entity type: {type(ent)}")


def _entity_from_dict(d: dict):
    from cad.sketch import LineEntity, ArcEntity, PointEntity, ReferenceEntity
    t = d["type"]
    if t == "line":
        ent = LineEntity(d["p0"], d["p1"])
        if "p0_snap" in d:
            from cad.sketch_tools.snap import SnapType
            ent.p0_snap = (int(d["p0_snap"][0]), SnapType[d["p0_snap"][1]])
        if "p1_snap" in d:
            from cad.sketch_tools.snap import SnapType
            ent.p1_snap = (int(d["p1_snap"][0]), SnapType[d["p1_snap"][1]])
        return ent
    if t == "arc":
        return ArcEntity(d["center"], float(d["radius"]),
                         float(d["start_angle"]), float(d["end_angle"]))
    if t == "point":
        return PointEntity(d["pos"])
    if t == "reference":
        points = [np.array(p, dtype=np.float64) for p in d["points"]]
        return ReferenceEntity(
            points      = points,
            source_type = d.get("source_type", "edge"),
            occ_edges   = None,   # will be populated by first replay
            source      = _edge_source_from_dict(d.get("source")),
        )
    raise ValueError(f"Unknown entity type: {t!r}")


# ---------------------------------------------------------------------------
# SketchConstraint
# ---------------------------------------------------------------------------

def _constraint_to_dict(c) -> dict:
    d: dict[str, Any] = {
        "type":    c.type,
        "indices": list(c.indices),
        "value":   c.value,
    }
    if c.label_offset is not None:
        d["label_offset"] = c.label_offset
    return d


def _constraint_from_dict(d: dict):
    from cad.sketch import SketchConstraint
    return SketchConstraint(
        type         = d["type"],
        indices      = tuple(d["indices"]),
        value        = float(d["value"]),
        label_offset = d.get("label_offset"),
    )


# ---------------------------------------------------------------------------
# SketchEntry
# ---------------------------------------------------------------------------

def _snapshot_to_list(snapshot) -> list:
    """Serialize one undo snapshot: (entities, constraints) tuple or bare entity list."""
    if isinstance(snapshot, tuple):
        ents, cons = snapshot
        return {"entities": [_entity_to_dict(e) for e in ents],
                "constraints": [_constraint_to_dict(c) for c in cons]}
    return {"entities": [_entity_to_dict(e) for e in snapshot], "constraints": []}


def _snapshot_from_dict(d) -> tuple:
    if isinstance(d, list):
        # legacy bare list format
        return ([_entity_from_dict(e) for e in d], [])
    ents = [_entity_from_dict(e) for e in d.get("entities", [])]
    cons = [_constraint_from_dict(c) for c in d.get("constraints", [])]
    return (ents, cons)


def _sketch_entry_to_dict(se) -> dict:
    return {
        "plane_origin":   _arr(se.plane_origin),
        "plane_x_axis":   _arr(se.plane_x_axis),
        "plane_y_axis":   _arr(se.plane_y_axis),
        "plane_normal":   _arr(se.plane_normal),
        "body_id":        se.body_id,
        "face_idx":       se.face_idx,
        "visible":        se.visible,
        "plane_source":   _plane_source_to_dict(se.plane_source),
        "entities":       [_entity_to_dict(e) for e in se.entities],
        "constraints":    [_constraint_to_dict(c) for c in se.constraints],
        "undo_snapshots": [_snapshot_to_list(s) for s in getattr(se, 'undo_snapshots', [])],
    }


def _sketch_entry_from_dict(d: dict):
    from cad.sketch import SketchEntry
    return SketchEntry(
        plane_origin   = d["plane_origin"],
        plane_x_axis   = d["plane_x_axis"],
        plane_y_axis   = d["plane_y_axis"],
        plane_normal   = d["plane_normal"],
        entities       = [_entity_from_dict(e) for e in d.get("entities", [])],
        body_id        = d["body_id"],
        face_idx       = int(d["face_idx"]),
        visible        = bool(d.get("visible", True)),
        plane_source   = _plane_source_from_dict(d.get("plane_source")),
        constraints    = [_constraint_from_dict(c) for c in d.get("constraints", [])],
        undo_snapshots = [_snapshot_from_dict(s) for s in d.get("undo_snapshots", [])],
    )


# ---------------------------------------------------------------------------
# params dict  (op-specific, may contain a SketchEntry)
# ---------------------------------------------------------------------------

def _params_to_dict(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k == "sketch_entry" and v is not None:
            out[k] = _sketch_entry_to_dict(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _params_from_dict(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k == "sketch_entry" and isinstance(v, dict):
            out[k] = _sketch_entry_from_dict(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Full project  save / load
# ---------------------------------------------------------------------------

def save(workspace, history, camera=None) -> bytes:
    """
    Serialize workspace + history (+ optional camera) to bytes (zip/json).
    Returns the raw bytes suitable for writing to a .vc file.
    """
    bodies = {}
    for bid, body in workspace.bodies.items():
        bodies[bid] = {
            "id":                   body.id,
            "name":                 body.name,
            "visible":              body.visible,
            "created_at_entry_id":  body.created_at_entry_id,
        }

    entries = []
    for e in history._entries:
        entries.append({
            "label":     e.label,
            "operation": e.operation,
            "params":    _params_to_dict(e.params),
            "body_id":   e.body_id,
            "face_ref":  _face_ref_to_dict(e.face_ref),
            "entry_id":  e.entry_id,
            "op_type":   e.op_type,
            "diverged":  e.diverged,
        })

    doc: dict[str, Any] = {
        "version":             1,
        "bodies":              bodies,
        "active_body_id":      workspace._active_body_id,
        "world_plane_visible": workspace.world_plane_visible,
        "history_cursor":      history._cursor,
        "entries":             entries,
    }

    if camera is not None:
        doc["camera"] = {
            "target":      camera.target.tolist(),
            "distance":    camera.distance,
            "ortho_scale": camera.ortho_scale,
            "ortho":       camera.ortho,
            "rotation":    camera.rotation.tolist(),
        }

    json_bytes = json.dumps(doc, indent=2).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json_bytes)
    return buf.getvalue()


def load(data: bytes):
    """
    Deserialize a .vc file (bytes) and return (workspace, history, camera_dict | None).

    Shapes are NOT present after this call — caller must replay history to
    rebuild them.  Import entries re-load from the original STEP file path
    stored in their params.
    """
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf, "r") as zf:
        doc = json.loads(zf.read("project.json").decode("utf-8"))

    from cad.workspace import Workspace, Body
    from cad.history import History
    from cad.op_types import Op

    workspace = Workspace()
    history   = History()
    workspace.history  = history
    history._workspace = workspace

    # Restore bodies (no source_shape yet — set during replay)
    for bid, bd in doc["bodies"].items():
        body = Body(
            id                  = bd["id"],
            name                = bd["name"],
            visible             = bd.get("visible", True),
            created_at_entry_id = bd.get("created_at_entry_id"),
        )
        workspace.bodies[bid] = body

    workspace._active_body_id      = doc.get("active_body_id")
    workspace.world_plane_visible  = doc.get("world_plane_visible",
                                             {"XY": True, "XZ": False, "YZ": False})

    # Restore history entries (shapes left as None — replay fills them in)
    for ed in doc["entries"]:
        params  = _params_from_dict(ed["params"])
        op      = Op.from_params(ed["operation"], params)
        from cad.history import HistoryEntry
        entry = HistoryEntry(
            label        = ed["label"],
            operation    = ed["operation"],
            params       = params,
            body_id      = ed["body_id"],
            face_ref     = _face_ref_from_dict(ed.get("face_ref")),
            shape_before = None,
            shape_after  = None,
            entry_id     = ed["entry_id"],
            op_type      = ed.get("op_type", ed["operation"]),
            diverged     = ed.get("diverged", False),
            op           = op,
        )
        history._entries.append(entry)

    history._cursor = int(doc.get("history_cursor", len(history._entries) - 1))

    camera_dict = doc.get("camera")
    return workspace, history, camera_dict


def replay_all(workspace, history) -> list[str]:
    """
    Rebuild all shapes by:
      1. Loading STEP files for import entries.
      2. Replaying each body's history chain from its first entry.

    Returns a list of warning strings (non-fatal issues).
    """
    from cad.importer import load_step
    from build123d import Compound

    warnings = []

    # Step 1: resolve import entries — load STEP files and set shape_after
    for entry in history._entries:
        if entry.operation != "import":
            continue
        params = entry.params
        path   = params.get("path")
        if path is None:
            # Split-body import — shape_after is set by parent op replay; skip
            continue
        solid_index = int(params.get("solid_index", 0))
        try:
            compound = load_step(path)
            solids = list(compound.solids())
            if not solids:
                solids = [compound]
            solid = solids[solid_index] if solid_index < len(solids) else solids[0]
            entry.shape_after = solid
            # Also set source_shape on the body
            body = workspace.bodies.get(entry.body_id)
            if body is not None:
                body.source_shape = solid
        except Exception as ex:
            warnings.append(f"Could not load '{path}': {ex}")

    # Step 2: replay each body chain from its first entry
    replayed: set[str] = set()
    for i, entry in enumerate(history._entries):
        body_id = entry.body_id
        if body_id in replayed:
            continue
        replayed.add(body_id)
        ok, err, _ = history.replay_from(i)
        if not ok:
            warnings.append(f"Replay error for body '{body_id}': {err}")

    return warnings
