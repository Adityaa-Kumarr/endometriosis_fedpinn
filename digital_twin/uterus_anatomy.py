"""
Analyze a uterus 3D mesh to label anatomical regions (cervix, body, fundus)
so the system knows which mesh part is what. Uses principal inertia axis and
position along the long axis; optional vertex defect (curvature) for refinement.
"""
import numpy as np

# Region IDs and names
REGION_CERVIX = 0
REGION_BODY = 1
REGION_FUNDUS = 2
REGION_NAMES = ["Cervix", "Body", "Fundus"]

# Fraction of length along long axis for region boundaries (t from 0 = one end to 1 = other)
# Cervix: bottom ~25%; Body: middle ~50%; Fundus: top ~25%
T_CERVIX_END = 0.25
T_FUNDUS_START = 0.75


def _get_mesh_vertices_faces(mesh):
    """Return (vertices, faces) from trimesh or Scene."""
    if hasattr(mesh, "geometry"):
        mesh = list(mesh.geometry.values())[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return vertices, faces


def get_long_axis_and_parameter(mesh):
    """
    Get the principal long axis of the mesh (direction of elongation) and
    per-vertex parameter t in [0, 1] along that axis (0 = one end, 1 = other).
    Uses principal inertia: the axis with smallest moment = long axis.
    Orients so that the end with larger mean "radius" (distance from axis) is t=1 (fundus).
    """
    vertices, _ = _get_mesh_vertices_faces(mesh)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # Principal inertia: columns of principal_inertia_vectors are axes;
    # principal_inertia_components ascending, so [0] = smallest = long axis
    try:
        comp = mesh.principal_inertia_components
        vecs = mesh.principal_inertia_vectors  # (3,3), columns = axes
    except Exception:
        # Fallback: use PCA on vertices (first principal component = long axis)
        cov = np.cov(centered.T)
        from numpy.linalg import eigh
        evals, evecs = eigh(cov)
        comp = evals
        vecs = evecs  # columns = eigenvectors
    # Long axis = direction of smallest inertia (first column or first eigenvector)
    if vecs.shape[0] == 3 and vecs.shape[1] == 3:
        long_axis = np.asarray(vecs[:, 0], dtype=np.float64)
    else:
        long_axis = np.asarray(vecs[0], dtype=np.float64)
    long_axis = long_axis / (np.linalg.norm(long_axis) + 1e-9)
    # Project vertices onto long axis
    t_raw = centered @ long_axis
    t_min, t_max = t_raw.min(), t_raw.max()
    span = t_max - t_min
    t = (t_raw - t_min) / (span + 1e-9)  # t in [0, 1]
    # Orient so fundus (wider end) = t=1: compare mean radius at t<0.5 vs t>0.5
    dist_from_axis = np.linalg.norm(centered - np.outer(t_raw, long_axis), axis=1)
    mean_radius_low = np.mean(dist_from_axis[t < 0.5])
    mean_radius_high = np.mean(dist_from_axis[t >= 0.5])
    if mean_radius_low > mean_radius_high:
        t = 1.0 - t  # flip so wider end is t=1
    return long_axis, t, centroid, span


def analyze_uterus_anatomy(mesh):
    """
    Analyze mesh and assign each vertex to a region: Cervix (0), Body (1), Fundus (2).
    
    Returns:
        dict with:
          - vertex_region_id: (n,) int in {0,1,2}
          - region_names: list of 3 names
          - t_parameter: (n,) float in [0,1] along long axis
          - long_axis: (3,) unit vector
          - centroid: (3,) 
          - extent_along_axis: float (length span)
    """
    vertices, faces = _get_mesh_vertices_faces(mesh)
    long_axis, t, centroid, extent = get_long_axis_and_parameter(mesh)
    n = len(vertices)
    region_id = np.zeros(n, dtype=np.int32)
    region_id[t < T_CERVIX_END] = REGION_CERVIX
    region_id[(t >= T_CERVIX_END) & (t < T_FUNDUS_START)] = REGION_BODY
    region_id[t >= T_FUNDUS_START] = REGION_FUNDUS
    return {
        "vertex_region_id": region_id,
        "region_names": REGION_NAMES,
        "t_parameter": t,
        "long_axis": long_axis,
        "centroid": centroid,
        "extent_along_axis": extent,
        "n_vertices": n,
        "n_faces": len(faces),
    }


def vertex_colors_for_plotly(vertex_region_id, cervix_rgb=(200, 230, 255), body_rgb=(255, 210, 215), fundus_rgb=(255, 180, 190)):
    """
    Return a list of Plotly vertex colors (rgb string per vertex) for Mesh3d.
    cervix = bluish, body = light pink, fundus = slightly darker pink.
    """
    colors = []
    for r in vertex_region_id:
        if r == REGION_CERVIX:
            colors.append(f"rgb({cervix_rgb[0]},{cervix_rgb[1]},{cervix_rgb[2]})")
        elif r == REGION_BODY:
            colors.append(f"rgb({body_rgb[0]},{body_rgb[1]},{body_rgb[2]})")
        else:
            colors.append(f"rgb({fundus_rgb[0]},{fundus_rgb[1]},{fundus_rgb[2]})")
    return colors


def get_region_at_vertex(anatomy, vertex_index):
    """Return region name for a vertex index."""
    rid = anatomy["vertex_region_id"][vertex_index]
    return anatomy["region_names"][rid]


def get_region_for_point(vertices, anatomy, point_xyz):
    """
    Return which region a 3D point belongs to by nearest vertex.
    point_xyz: (3,) or (x,y,z). Uses anatomy['vertex_region_id'] and mesh vertices.
    """
    p = np.asarray(point_xyz, dtype=np.float64).ravel()[:3]
    verts = vertices if hasattr(vertices, "shape") and len(vertices.shape) == 2 else np.column_stack(vertices)
    dists = np.linalg.norm(verts - p, axis=1)
    idx = np.argmin(dists)
    rid = anatomy["vertex_region_id"][idx]
    return anatomy["region_names"][rid], rid
