from build123d import import_step, Plane

def load_step(path):
    """Load a STEP file and return the build123d shape."""
    shape = import_step(path)
    return shape

def get_planar_faces(shape):
    """Return list of (index, face) tuples for all planar faces."""
    result = []
    for i, face in enumerate(shape.faces()):
        try:
            Plane(face)
            result.append((i, face))
        except Exception:
            pass
    return result

def get_face(shape, index):
    """Return the build123d Face object at the given index."""
    return list(shape.faces())[index]
