# Full Application Analysis Report

Analysis date: 2025-03-09

---

## Executive Summary

The endometriosis_fedpinn application was analyzed end-to-end for conflicts, bugs, and integration issues. **5 fixes were applied**. Remaining items are documented below for future work.

---

## Fixes Applied

| Fix | File | Description |
|-----|------|-------------|
| 1 | `reference_shape.py` | Changed `scale = extent/11.0` to `extent/13.0` to match `half_height=6.5` in uterus_mesh_fixed |
| 2 | `app.py` | Replaced bare `except:` with `except OSError:` when removing model file |
| 3 | `app.py` | Added `st.cache_data.clear()` on "Clear results / New patient" (per FRONTEND_MEMORY doc) |
| 4 | `validation/evaluator.py` | Added `.ravel()` to labels/probs/preds for sklearn compatibility |
| 5 | `docs/UTERUS_3D_GENERATION.md` | Updated half_height from 5.5 to 6.5 |

---

## Architecture Overview

```
app.py (Streamlit)
  ├── models/ (FFNN, image encoder)
  ├── digital_twin/
  │   ├── simulator.py → uterus_mesh_fixed.generate_uterus_mesh()
  │   ├── uterus_mesh_fixed.py (parametric uterus)
  │   ├── mesh_loader.py (.glb uterus)
  │   ├── reference_shape.py (calibrate parametric from .glb)
  │   └── omniverse_export.py (OBJ/USD export)
  ├── federated/ (Flower FedProx)
  ├── data/ (EndometriosisDataset, load_client_data)
  └── validation/ (evaluator)
```

---

## Uterus Data Flow (No Conflicts)

| Source | When Used | Format |
|--------|-----------|--------|
| **uterus_mesh_fixed** | Always (via simulator) | `(x, y, z, i, j, k)` |
| **.glb mesh** | When `use_python_only=False` and `.glb` exists | `(x, y, z, i, j, k)` or 7-tuple with vertex_colors |
| **Reference params** | When `.glb` exists, shapes parametric via `get_reference_shape_params()` | `{a, b, z_scale, xy_scale}` |

Fallback: If `.glb` missing, `uterus_mesh=None` → app uses parametric from `twin_data`.

---

## Remaining Items (Not Fixed)

### Warnings (Lower Priority)

| Item | Location | Notes |
|------|----------|-------|
| sys.path manipulation | app.py:77-78, federated/client.py:9-13 | Fragile if run from different cwd |
| DATA_DIR in K8s | k8s_deployment.yaml | Clients default to `dataset/clients`; ensure image has data or use PVCs |
| min_fit_clients=2 | federated/server.py | Rounds block if fewer than 2 clients connect |
| load_client_data FileNotFoundError | data/data_loader.py:80 | No retry; run synthetic_generator if missing |
| App vs data_loader normalization | app.py vs data_loader.py | Different scaling for inference vs training |
| load_uterus_mesh_for_plotly | mesh_loader.py | Broad except; no user feedback on failure |

### Minor (Optional Cleanup)

| Item | Location |
|------|----------|
| Duplicate noise helpers | simulator.py vs uterus_mesh_fixed.py (both needed for ovaries/tubes vs uterus) |
| build_plotly_traces unused | uterus_mesh_fixed.py (available for standalone use) |
| get_region_at_vertex unused | uterus_anatomy.py |
| future_lesion_sizes unused | simulator returns it; create_3d_plot doesn't use for sizing |
| FedProx config keys | Server sends both proximal_mu and proximal-mu (client handles both) |
| Filename "uterus .glb" | Space may cause issues on some systems |

### Dependencies (Verified)

- **trimesh** — Present in requirements.txt
- **PyPDF2, pytesseract, Pillow** — Present in requirements.txt
- **openpyxl/xlrd** — Excel support; install on demand if needed

---

## K8s / Federated Notes

- Server expects `NUM_ROUNDS` env var
- Clients retry connection (12 attempts) for startup
- Without `dataset/clients` in image or PVC, clients raise FileNotFoundError
- synthetic_generator.py can create data if dataset missing (Dockerfile runs it)

---

## Testing Recommendations

1. Run `py digital_twin/uterus_mesh_fixed.py` — self-test
2. Run `py digital_twin/simulator.py` — generates twin data
3. Run `streamlit run app.py` — full app
4. With .glb: enable "Show only Python-generated anatomy" off to compare mesh vs parametric
