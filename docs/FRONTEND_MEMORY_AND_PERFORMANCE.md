# Frontend Memory & Performance Cross-Check

**Scope:** Streamlit app (`app.py`) — memory leaks and performance.

---

## 1. Memory leak assessment

### Session state

| Key | Purpose | Leak risk |
|-----|---------|-----------|
| `pred_prob`, `pred_prob_std`, `pred_stage` | Scalars | None — overwritten per run. |
| `future_risk`, `gate_probs` | Small arrays | None — overwritten. |
| `clinical_data` | One-row input for model | Low — single row; not appended. |
| `us_embedding_from_image` | 128-d vector from image | Low — overwritten or cleared when non-image uploaded. |

**Mitigation:** Fixed set of keys; no unbounded lists. **"Clear results / New patient"** in the sidebar clears these keys and calls `st.cache_data.clear()` so state and data caches are reset and memory can be reclaimed.

### Cached resources

- **`@st.cache_resource`** used for:
  - `load_models()` — one model in memory.
  - `_get_image_encoder()` — one encoder.
- **`st.cache_resource.clear()`** is called after **Federated fine-tuning** so the app reloads the updated model. No unbounded growth.

### Large objects

- **Plotly figures** — created per render, not stored in `session_state`; Streamlit holds the last widget state only.
- **3D twin data / uterus mesh** — previously recreated on every Tab2 visit; now **cached** with `@st.cache_data` (see below), so no repeated allocation for the same inputs.

**Verdict:** No identified memory leaks from session state or caches. Clear button added to reset state and caches when starting a new patient.

---

## 2. Performance issues and fixes

### Before

| Operation | When | Issue |
|-----------|------|--------|
| `load_uterus_mesh_for_plotly()` | Every Tab2 render | Reload and re-analyze .glb on every visit. |
| `twin.generate_3d_scatter_data()` | Every Tab2 render | Regenerate 120×120 grids + noise every time. |
| SHAP in Tab1 | Every results view | Expensive; no cache (acceptable for accuracy). |

### After

1. **Uterus mesh**
   - **`_cached_uterus_mesh(path_key, target_size)`** with `@st.cache_data`.
   - Key: resolved path from `get_uterus_mesh_path()` (or `""`).
   - First Tab2 visit loads and analyzes the mesh; later visits reuse cache until path changes or cache is cleared.

2. **3D twin geometry**
   - **`_cached_twin_data(pred_prob, pred_stage, future_risk_val)`** with `@st.cache_data`.
   - Key: current prediction state.
   - Same prediction → same geometry; no recomputation when toggling layers/opacity or revisiting Tab2.

3. **Clear results**
   - Sidebar button **"Clear results / New patient"** clears prediction-related `session_state` keys and runs `st.cache_data.clear()`.
   - Use between patients to avoid carrying over data and to allow cache entries to be reclaimed.

### Not cached (by design)

- **SHAP** — depends on model and exact clinical input; caching would require stable hashing of inputs and more invalidation logic. Left uncached for correctness; cost is acceptable for single prediction.
- **Model / image encoder** — `@st.cache_resource`; loaded once and reused.

---

## 3. Checklist

- [x] Session state uses a fixed set of keys; no unbounded append.
- [x] `us_embedding_from_image` cleared when a non-image file is uploaded.
- [x] `st.cache_resource.clear()` after federated training to reload model.
- [x] Uterus mesh loading cached by path.
- [x] 3D twin data cached by prediction state.
- [x] "Clear results / New patient" to reset state and data caches.
- [x] No large uploads stored in session (only parsed one-row `clinical_data`).

---

## 4. Optional future improvements

- **SHAP:** Optional `@st.cache_data` keyed by hash of (clinical_data, prob) with small TTL or max size.
- **Torch:** If using CUDA, call `torch.cuda.empty_cache()` after heavy inference (e.g. after training tab).
- **Plotly:** For very large 3D meshes, consider downsampling or level-of-detail for the viewer.
