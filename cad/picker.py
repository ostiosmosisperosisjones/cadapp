"""
cad/picker.py

Vectorized Möller–Trumbore ray-triangle intersection.
Returns face index, or (face_index, t) when return_t=True.
"""

import numpy as np


def pick_face(mesh, ray_origin, ray_dir, return_t: bool = False):
    origin    = np.array(ray_origin, dtype=np.float64)
    direction = np.array(ray_dir,    dtype=np.float64)

    verts = mesh.verts.astype(np.float64)
    tris  = mesh.tris

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    h     = np.cross(direction[np.newaxis, :], e2)
    a     = np.einsum('ij,ij->i', e1, h)
    eps   = 1e-9
    valid = np.abs(a) > eps
    f     = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

    s = origin[np.newaxis, :] - v0
    u = f * np.einsum('ij,ij->i', s, h)
    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, e1)
    v = f * np.einsum('ij,ij->i',
                      direction[np.newaxis, :] * np.ones((len(tris), 1)), q)
    valid &= (v >= 0.0) & (u + v <= 1.0)

    t = f * np.einsum('ij,ij->i', e2, q)
    valid &= (t > eps)

    if not np.any(valid):
        return None

    t_vals      = np.where(valid, t, np.inf)
    closest_tri = int(np.argmin(t_vals))
    min_t       = float(t_vals[closest_tri])

    tpf      = mesh.triangles_per_face
    cumsum   = np.cumsum(tpf)
    face_idx = int(np.searchsorted(cumsum, closest_tri, side='right'))

    if return_t:
        return face_idx, min_t
    return face_idx
