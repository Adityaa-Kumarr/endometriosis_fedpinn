"""
Analyze the reference uterus .glb to extract shape profile (radius vs height)
and return parameters for the Python parametric uterus so it matches the reference.
Used to generate a proper 3D uterus model with correct shapes and structures.
"""
import os
import numpy as np

def _find_reference_glb():
    """Path to reference uterus .glb (same as mesh_loader)."""
    path = os.environ.get("UTERUS_GLB_PATH", "").strip()
    if path and os.path.isfile(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("uterus .glb", "uterus.glb", "uterus.glTF"):
        for base in (root, os.path.dirname(root)):
            c = os.path.join(base, name)
            if os.path.isfile(c):
                return c
    return None

def _load_mesh(path):
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        if hasattr(mesh, "geometry"):
            mesh = list(mesh.geometry.values())[0]
        return mesh
    except Exception:
        return None

def _get_long_axis_and_t(mesh):
    """Long axis and normalized position t in [0,1] (0=cervix, 1=fundus)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    try:
        vecs = mesh.principal_inertia_vectors
        if vecs.shape[0] == 3 and vecs.shape[1] == 3:
            long_axis = np.asarray(vecs[:, 0], dtype=np.float64)
        else:
            long_axis = np.asarray(vecs[0], dtype=np.float64)
    except Exception:
        cov = np.cov(centered.T)
        from numpy.linalg import eigh
        _, evecs = eigh(cov)
        long_axis = np.asarray(evecs[:, 0], dtype=np.float64)
    long_axis = long_axis / (np.linalg.norm(long_axis) + 1e-9)
    t_raw = centered @ long_axis
    t_min, t_max = t_raw.min(), t_raw.max()
    span = t_max - t_min
    t = (t_raw - t_min) / (span + 1e-9)
    # Orient: wider end = fundus = t=1
    dist_from_axis = np.linalg.norm(centered - np.outer(t_raw, long_axis), axis=1)
    if np.mean(dist_from_axis[t < 0.5]) > np.mean(dist_from_axis[t >= 0.5]):
        t = 1.0 - t
        t_raw = t_max + t_min - t_raw
    return long_axis, t, centroid, span

def analyze_reference_uterus_shape(path=None):
    """
    Analyze reference uterus .glb mesh and return shape parameters for the
    parametric generator so the Python-generated uterus matches the reference.
    
    Returns:
        dict with:
          - a, b: coefficients for r_base = a + b*cos(u) (pear profile)
          - z_scale: scale for z (height) to match reference extent
          - xy_scale: scale for x,y (radius) to match reference
          - extent_along_axis: reference length
          - profile_t: t values (0=cervix, 1=fundus)
          - profile_radius: mean radius at each t (for optional fine-tuning)
        or None if load/analysis fails.
    """
    path = path or _find_reference_glb()
    if not path or not os.path.isfile(path):
        return None
    mesh = _load_mesh(path)
    if mesh is None:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        return None
    long_axis, t, centroid, extent = _get_long_axis_and_t(mesh)
    # Distance of each vertex from the long axis
    along = (vertices - centroid) @ long_axis
    radial = np.linalg.norm((vertices - centroid) - np.outer(along, long_axis), axis=1)
    # Bin by t and compute mean radius in each bin
    n_bins = 12
    bins = np.linspace(0, 1, n_bins + 1)
    profile_t = []
    profile_radius = []
    for i in range(n_bins):
        mask = (t >= bins[i]) & (t < bins[i + 1])
        if np.sum(mask) > 5:
            profile_t.append((bins[i] + bins[i + 1]) / 2)
            profile_radius.append(np.mean(radial[mask]))
    if len(profile_t) < 3:
        return None
    profile_t = np.array(profile_t)
    profile_radius = np.array(profile_radius)
    # t=0 cervix, t=1 fundus. Our param: u=0 fundus, u=pi cervix → t_param = 1 - u/pi, u = (1-t)*pi
    # Fit r(u) = a + b*cos(u). At u=0 (fundus): r = a+b. At u=pi (cervix): r = a-b.
    t_fundus = profile_t[-1] if profile_t[-1] > 0.7 else 1.0
    t_cervix = profile_t[0] if profile_t[0] < 0.3 else 0.0
    R_fundus = float(np.interp(1.0, profile_t, profile_radius)) if len(profile_t) else profile_radius[-1]
    R_cervix = float(np.interp(0.0, profile_t, profile_radius)) if len(profile_t) else profile_radius[0]
    # Scale so our parametric extent 13 (z from -6.5 to 6.5, half_height=6.5) matches reference extent
    scale = extent / 13.0
    # r_base = a + b*cos(u) in our "unit" scale; we want (a+b)*scale ≈ R_fundus, (a-b)*scale ≈ R_cervix
    a = (R_fundus + R_cervix) / (2.0 * scale)
    b = (R_fundus - R_cervix) / (2.0 * scale)
    if b < 0.1:
        b = 0.5 * a  # ensure pear shape
    return {
        "a": float(a),
        "b": float(b),
        "z_scale": float(scale),
        "xy_scale": float(scale),
        "extent_along_axis": float(extent),
        "profile_t": profile_t.tolist(),
        "profile_radius": profile_radius.tolist(),
        "R_fundus": R_fundus,
        "R_cervix": R_cervix,
    }

def get_reference_shape_params(path=None):
    """
    Return shape params for simulator (a, b, z_scale, xy_scale).
    Cached by caller if needed. Returns None if no reference .glb or analysis fails.
    """
    out = analyze_reference_uterus_shape(path)
    if out is None:
        return None
    return {
        "a": out["a"],
        "b": out["b"],
        "z_scale": out["z_scale"],
        "xy_scale": out["xy_scale"],
    }
