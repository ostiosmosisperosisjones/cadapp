import numpy as np
from math import radians, degrees, asin, atan2, sqrt, cos, sin

# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention)
# ---------------------------------------------------------------------------

def _quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])

def _quat_mul(a, b):
    """Hamilton product of two quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])

def _quat_from_axis_angle(axis, angle_rad):
    """Build a quaternion from a unit axis and angle in radians."""
    s = sin(angle_rad / 2.0)
    c = cos(angle_rad / 2.0)
    return np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])

def _quat_to_matrix(q):
    """Return a 3x3 rotation matrix from a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def _quat_from_matrix(m):
    """Extract a unit quaternion from a 3x3 rotation matrix (Shepperd)."""
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / sqrt(trace + 1.0)
        return np.array([0.25/s,
                         (m[2,1]-m[1,2])*s,
                         (m[0,2]-m[2,0])*s,
                         (m[1,0]-m[0,1])*s])
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        return np.array([(m[2,1]-m[1,2])/s, 0.25*s,
                         (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s])
    elif m[1,1] > m[2,2]:
        s = 2.0 * sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        return np.array([(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s,
                         0.25*s, (m[1,2]+m[2,1])/s])
    else:
        s = 2.0 * sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        return np.array([(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s,
                         (m[1,2]+m[2,1])/s, 0.25*s])

def _screen_to_sphere(px, py, vw, vh):
    """
    Map a pixel coordinate to a point on the unit arcball sphere.
    Points outside the sphere are projected onto the rim (z=0 ring).
    Returns a normalised 3-vector.
    """
    # Normalise to [-1, 1] with y flipped
    x = (2.0 * px - vw) / min(vw, vh)
    y = (vh - 2.0 * py) / min(vw, vh)
    r2 = x*x + y*y
    if r2 <= 1.0:
        z = sqrt(1.0 - r2)
    else:
        # Outside sphere — project onto equator ring
        r = sqrt(r2)
        x, y, z = x/r, y/r, 0.0
    v = np.array([x, y, z])
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self):
        self.target      = np.array([0.0, 0.0, 0.0])
        self._scene_extent = 1.0   # updated by fit_scene()
        self.distance    = 3.0
        self.ortho_scale = 1.5
        self.ortho       = False
        # Rotation stored as a quaternion (w, x, y, z)
        # Default: slight elevation and azimuth so the initial view is isometric-ish
        q_el  = _quat_from_axis_angle(np.array([1,0,0], dtype=float), radians(20))
        q_az  = _quat_from_axis_angle(np.array([0,1,0], dtype=float), radians(-30))
        self.rotation = _quat_mul(q_el, q_az)
        self.rotation /= np.linalg.norm(self.rotation)
        # Orbit drag state
        self._orbit_start_q  = None   # rotation snapshot at drag start
        self._orbit_start_v  = None   # sphere vector at drag start
        self._vw = 1                  # viewport width  (updated each drag)
        self._vh = 1                  # viewport height
        # Pan drag state
        self._pan_last = None

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    def get_eye(self):
        """Camera position — target + distance along the rotation's +Z axis."""
        R = _quat_to_matrix(self.rotation)
        # Local +Z (forward-out) in world space
        forward = R[:, 2]
        return self.target + forward * self.distance

    def get_up(self):
        """Camera up — rotation's +Y axis in world space."""
        R = _quat_to_matrix(self.rotation)
        return R[:, 1]

    # ------------------------------------------------------------------
    # Orbit
    # ------------------------------------------------------------------

    def begin_orbit(self, pos, vw, vh, pivot=None):
        """
        Called on right-mouse-button press.
        vw, vh: current viewport pixel dimensions.
        pivot:  optional world-space 3-vector; if given, re-centre target there.
        """
        if pivot is not None:
            pivot = np.array(pivot, dtype=float)
            old_eye = self.get_eye()
            self.target   = pivot
            self.distance = float(np.linalg.norm(old_eye - pivot))
            self.distance = max(0.01, self.distance)
        self._vw = vw
        self._vh = vh
        self._orbit_start_q = self.rotation.copy()
        self._orbit_start_v = _screen_to_sphere(pos.x(), pos.y(), vw, vh)

    def orbit(self, pos):
        """Called on right-drag mouse-move."""
        if self._orbit_start_v is None:
            return
        v2 = _screen_to_sphere(pos.x(), pos.y(), self._vw, self._vh)
        v1 = self._orbit_start_v

        # Rotation axis is cross product of the two sphere vectors
        axis = np.cross(v1, v2)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-10:
            return
        axis /= axis_len

        # Angle between the two vectors
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angle = 2.0 * asin(sqrt(max(0.0, 1.0 - dot*dot) / 2.0 + 0.0))
        # Simpler and more stable: angle = arccos(dot)
        import math
        angle = math.acos(dot)

        # Delta quaternion in *world* space, applied on the left so the
        # rotation accumulates naturally (orbiting the object, not the camera)
        dq = _quat_from_axis_angle(axis, angle)
        self.rotation = _quat_mul(self._orbit_start_q, dq)
        self.rotation /= np.linalg.norm(self.rotation)

    def end_orbit(self):
        self._orbit_start_q = None
        self._orbit_start_v = None

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    def begin_pan(self, pos):
        self._pan_last = (pos.x(), pos.y())

    def pan(self, pos):
        if self._pan_last is None:
            return
        dx = pos.x() - self._pan_last[0]
        dy = pos.y() - self._pan_last[1]
        R = _quat_to_matrix(self.rotation)
        right = R[:, 0]   # camera local X
        up    = R[:, 1]   # camera local Y
        scale = self.ortho_scale if self.ortho else self.distance
        self.target -= right * dx * 0.002 * scale
        self.target += up    * dy * 0.002 * scale
        self._pan_last = (pos.x(), pos.y())

    def end_pan(self):
        self._pan_last = None

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll(self, delta):
        factor = 0.9 if delta > 0 else 1.1
        if self.ortho:
            self.ortho_scale *= factor
            self.ortho_scale = max(0.001 * self._scene_extent,
                                   min(10.0 * self._scene_extent, self.ortho_scale))
        else:
            self.distance *= factor
            self.distance = max(0.001 * self._scene_extent,
                                min(10.0 * self._scene_extent, self.distance))

    # ------------------------------------------------------------------
    # Snap to face normal
    # ------------------------------------------------------------------

    def snap_to_normal(self, nx, ny, nz, origin=(0.0, 0.0, 0.0)):
        """
        Orient camera to look straight down the face normal,
        centred on the face origin.
        """
        length = sqrt(nx*nx + ny*ny + nz*nz)
        if length < 1e-9:
            return
        nx, ny, nz = nx/length, ny/length, nz/length

        self.target = np.array(origin, dtype=float)

        # We want the camera's local +Z to point along +normal (eye behind face).
        # Build a rotation matrix whose third column is (nx, ny, nz).
        forward = np.array([nx, ny, nz])

        # Choose a stable 'up' reference that isn't parallel to forward
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        # Assemble rotation matrix: columns are right, up, forward
        m = np.array([
            [right[0],   up[0],   forward[0]],
            [right[1],   up[1],   forward[1]],
            [right[2],   up[2],   forward[2]],
        ])
        self.rotation = _quat_from_matrix(m)
        self.rotation /= np.linalg.norm(self.rotation)

    # ------------------------------------------------------------------
    # Projection toggle
    # ------------------------------------------------------------------


    def fit_scene(self, bbox_min, bbox_max):
        """
        Position camera to frame the scene bounding box.
        Sets target to bbox centre, distance to fit everything in view.
        Called once after loading geometry.
        """
        import numpy as np
        mn = np.array(bbox_min, dtype=float)
        mx = np.array(bbox_max, dtype=float)
        centre = (mn + mx) * 0.5
        extent = float(np.linalg.norm(mx - mn))   # diagonal length

        self.target        = centre
        self.distance      = extent * 1.2
        self.ortho_scale   = extent * 0.6
        self._scene_extent = extent   # used by scroll() for adaptive limits

    def toggle_ortho(self):
        self.ortho = not self.ortho
