"""
Load a uterus mesh from .glb (or .obj) for the 3D Digital Twin viewer.
Uses trimesh; PyMesh is optional and not required for the app.
"""
import os
import numpy as np

def _find_uterus_glb():
    """Resolve path to uterus .glb: env UTERUS_GLB_PATH, or project root, or parent (DT)."""
    path = os.environ.get("UTERUS_GLB_PATH", "").strip()
    if path and os.path.isfile(path):
        return path
    # Project root = directory containing this file's package (digital_twin)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("uterus .glb", "uterus.glb", "uterus.glTF"):
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    parent = os.path.dirname(root)
    for name in ("uterus .glb", "uterus.glb"):
        candidate = os.path.join(parent, name)
        if os.path.isfile(candidate):
            return candidate
    return None

def get_uterus_mesh_path():
    """Return resolved path to uterus .glb for cache keys. Exported for Streamlit caching."""
    return _find_uterus_glb()

def load_uterus_mesh_for_plotly(path=None, target_size=12.0, analyze_anatomy=True):
    """
    Load a 3D mesh from .glb (or .obj) and return data for Plotly go.Mesh3d.
    Centers the mesh at origin and scales so the bounding box fits in target_size.
    If analyze_anatomy=True, runs shape analysis to label regions (Cervix / Body / Fundus)
    and returns vertex colors so the system knows which mesh part is what.

    Returns:
        tuple (x, y, z, i, j, k) or (x, y, z, i, j, k, vertex_colors) for go.Mesh3d.
        vertex_colors is a list of 'rgb(r,g,b)' per vertex when anatomy analysis succeeded.
        Returns None if load fails.
    """
    try:
        import trimesh
    except ImportError:
        return None
    path = path or _find_uterus_glb()
    if not path or not os.path.isfile(path):
        return None
    try:
        mesh = trimesh.load(path, force="mesh")
        if mesh is None:
            return None
        # Handle Scene (multiple geometries): take first
        if hasattr(mesh, "geometry"):
            geoms = list(mesh.geometry.values())
            if not geoms:
                return None
            mesh = geoms[0]
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            return None

        # Analyze anatomy (shape → regions) before transforming vertices
        vertex_colors = None
        if analyze_anatomy:
            try:
                from .uterus_anatomy import analyze_uterus_anatomy, vertex_colors_for_plotly
                anatomy = analyze_uterus_anatomy(mesh)
                vertex_colors = vertex_colors_for_plotly(anatomy["vertex_region_id"])
            except Exception:
                pass

        # Center and scale to fit scene (same scale as parametric twin)
        mn = vertices.min(axis=0)
        mx = vertices.max(axis=0)
        center = (mn + mx) / 2.0
        span = (mx - mn)
        scale = target_size / (np.max(span) + 1e-9)
        vertices = (vertices - center) * scale
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        if vertex_colors is not None:
            return (x, y, z, i, j, k, vertex_colors)
        return (x, y, z, i, j, k)
    except Exception:
        return None
