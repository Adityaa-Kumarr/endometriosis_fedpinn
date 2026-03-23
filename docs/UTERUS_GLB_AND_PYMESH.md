# Using a Real Uterus Mesh (.glb) and PyMesh

## Your `uterus .glb` file

- **Format:** GLB (binary glTF 2.0) — common for 3D models from Blender, Sketchfab, or medical segmentation tools.
- **PyMesh:** [PyMesh](https://pymesh.readthedocs.io/en/latest/) does **not** support .glb. It supports: `.obj`, `.off`, `.ply`, `.stl`, `.mesh` (MEDIT), `.msh` (Gmsh), `.node/.face/.ele` (Tetgen).

## How we use them

| Goal | Tool | Usage |
|------|------|--------|
| **Load and use `uterus .glb` in the app** | **trimesh** | Load .glb → get `vertices` (N×3) and `faces` (M×3) → scale/center → display with Plotly `go.Mesh3d`. |
| **Convert .glb → .obj for PyMesh** | trimesh | `trimesh.load('uterus .glb').export('uterus.obj')` then use PyMesh on the .obj. |
| **Advanced mesh processing** | **PyMesh** | After conversion to .obj: remesh, boolean ops, curvature, subdivision, etc. |

## Pipeline in this project

1. **Optional real uterus:** Place `uterus .glb` (e.g. at project root or in `digital_twin/`) and set `UTERUS_GLB_PATH` or use the default path. The app loads it with **trimesh** and shows it in the 3D Digital Twin tab with `go.Mesh3d`.
2. **Fallback:** If no .glb is found or loading fails, the app uses the existing **parametric** uterus from `digital_twin/simulator.py` (pear-shaped + noise).
3. **PyMesh (optional):** For geometry processing (e.g. remeshing, smoothing), convert .glb → .obj with trimesh, then use PyMesh in a script (e.g. `pymesh.load_mesh("uterus.obj")`). PyMesh is not required to run the Streamlit app.

## Dependencies

- **trimesh** — added to `requirements.txt` for loading .glb and exporting .obj.
- **PyMesh** — optional; install separately if you need mesh processing (see [PyMesh installation](https://pymesh.readthedocs.io/en/latest/installation.html)). Not in requirements by default.

## Reference-guided parametric model

The **Python-generated** uterus model (when no .glb is used) is tuned to match an annotated reference:

- **Part annotations:** The simulator returns `part_labels`: (name, x, y, z) for **Uterus**, **Left Ovary**, **Right Ovary**, **Left Fallopian Tube**, **Right Fallopian Tube**. Label positions are computed from mesh centroids so the system knows which part is what. The 3D viewer draws these as white text with markers.
- **Shapes:** Uterus = stronger pear (narrower cervix, wider fundus); ovaries = slightly rounder; fallopian tubes = more convoluted/meandering to match the reference. So the generated model is optimized and corrected for anatomy while keeping full part annotations.

When you use a **single uterus .glb**, the app still draws parametric ovaries, tubes, and the same part labels so the viewer always shows full anatomy with annotations.

### Python-generated uterus from reference .glb (research)

The app can **analyze the attached uterus .glb** and use it purely as a **reference for shape**:

- **Reference shape analyzer** (`digital_twin/reference_shape.py`): Loads the .glb, computes the long axis (principal inertia), and samples mean radius along the axis (cervix → fundus). It fits a pear profile **r = a + b·cos(u)** and scale factors so the **Python parametric uterus** matches the reference proportions.
- **3D viewer option:** **"Show only Python-generated anatomy (reference-shaped from uterus .glb)"** (default: on). When this is on, the viewer shows **only** the Python-generated model (uterus + ovaries + tubes + lesions + annotations). The uterus shape is driven by the reference .glb when the file is present; otherwise default pear parameters are used. No .glb mesh is drawn—only the parametric model with all parts and labels.
- **Structures:** The Python model includes uterus (reference-shaped), left/right ovaries, left/right fallopian tubes, current/future lesions, adhesions, and anatomical labels (Uterus, Left/Right Ovary, Left/Right Fallopian Tube). All shapes and annotations are generated in Python for full control and consistency.

## File location

- Default path used by the app: `uterus.glb` or `uterus .glb` under the project root (`h:\Akash\DT\`). You can override via environment variable `UTERUS_GLB_PATH` or by placing the file in `endometriosis_fedpinn/` and setting the path in the app.

---

## Understanding mesh shape: which part is what (anatomy)

The app analyzes the uterus mesh so the **system knows which mesh part is what**:

- **`digital_twin/uterus_anatomy.py`** uses the mesh’s **principal inertia axis** (long axis of the shape) and projects each vertex along that axis to assign regions:
  - **Cervix** — bottom ~25% along the long axis
  - **Body** — middle ~50%
  - **Fundus** — top ~25%
- The long axis is oriented so the **wider** end is treated as the fundus (top).
- In the 3D viewer, when a real mesh is loaded, vertices are **colored by region** (e.g. cervix = bluish, body = light pink, fundus = darker pink) so you can see the segmentation.
- Programmatic use: `analyze_uterus_anatomy(mesh)` returns `vertex_region_id` (0/1/2) and metadata; `get_region_for_point(vertices, anatomy, xyz)` returns the region name for any 3D point (by nearest vertex). This can be used to place lesions or report “lesion in fundus” etc.
