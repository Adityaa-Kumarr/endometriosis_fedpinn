# Full Project Cross-Check — Conflicts & Issues

**Purpose:** Single pass over the entire project to catch conflicts, broken references, and inconsistencies. No scope skipped.

---

## 1. Summary

| Area | Status | Notes |
|------|--------|------|
| **Imports & modules** | OK | All Python imports resolve (models, data, xai, digital_twin, federated, validation). No references to deleted .md files in code. |
| **Data contracts** | OK | CLINICAL_FEATURE_COLUMNS (9), label/stage handling, optional .npy row-count check; app Tab4 and backend use same schema. |
| **Model signature** | OK | FullFedPINNModel(clinical, us, genomic, path, sensor) used consistently in app, client, evaluator. |
| **Physics loss indices** | OK | clinical[:, 6]=ca125, clinical[:, 7]=estradiol; column order matches data_loader. |
| **Deployment** | OK | K8s manifests, PDBs, deploy scripts; image name substitution in script. |
| **Unused dependency** | Minor | `lime` in requirements.txt and README; only SHAP is used in app. Optional to remove or add LIME later. |

---

## 2. Module & Import Check

- **app.py:** Imports models, data_loader (normalize, columns, EndometriosisDataset), digital_twin, xai.explainer. All exist.
- **federated/client.py:** Imports models, data_loader.load_client_data; sys.path set to parent. OK.
- **federated/server.py:** Only flwr, sys. OK.
- **xai/explainer.py:** shap, torch, numpy; optional local import of models in `if __name__`. OK.
- **validation/evaluator.py:** Expects batch dict with 'clinical', 'ultrasound', 'label', optional 'genomic', 'pathology', 'sensor'. Matches data_loader.EndometriosisDataset.__getitem__. OK.
- **data_loader:** No references to deleted docs. analyze_full_datasets_folder.py mentions `--out docs/DATASETS_FULL_ANALYSIS.json` only as an example output path (writes .json, not .md). OK.

**Deleted files:** All removed .md cross-check/analysis docs; no code imports or reads them. OK.

---

## 3. Data & Model Contract

| Contract | data_loader | app (Tab4) | federated client | evaluator |
|----------|-------------|------------|------------------|-----------|
| Clinical columns | CLINICAL_FEATURE_COLUMNS (9) | Imports same, fills missing | From batch | From batch |
| Label column | label / endometriosis_present / stage | Adds label/stage if missing | batch['label'] | batch['label'] |
| Stage | Required or derived, clipped 0–4 | Adds if missing | batch['stage'] | Not used |
| Batch keys | clinical, ultrasound, genomic, pathology, sensor, label, stage | TensorDataset → (c,u,g,p,s,y_pres,y_stage) | Same keys | Same keys |
| Model forward | — | (c,u,g,p,s) | (clinical, us_data, genomic, pathology, sensor) | (clinical, us_data, genomic, pathology, sensor) |

- **PINN physics loss:** client.py uses `clinical[:, 7]` (estradiol), `clinical[:, 6]` (ca125). Order in CLINICAL_FEATURE_COLUMNS: 0=age, 1=bmi, 2=pelvic_pain_score, 3=dysmenorrhea_score, 4=dyspareunia, 5=family_history, 6=ca125, 7=estradiol, 8=progesterone. Correct.

---

## 4. App-Specific

- **Prediction path:** mock_means / mock_stds normalize clinical; model receives (1,9) + (1,128) + zeros for g,p,s. OK.
- **SHAP path:** _XAI_MOCK_MEANS / _XAI_MOCK_STDS match prediction normalization; background and instance in normalized space. OK.
- **Tab4 training:** TensorDataset of 7 tensors; loop unpacks (c_data, u_data, g_data, p_data, s_data, target_pres, target_stage); model(c_data, u_data, g_data, p_data, s_data). OK.
- **global_model.pth:** Loaded in load_models(); saved in Tab4 after training. Path is relative to CWD (app run directory). OK.

---

## 5. Data Loader & Synthetic

- **load_client_data:** Expects client_{id}/ with clinical.csv, us_embeddings.npy; optional genomic_data.npy, pathology_data.npy, sensor_data.npy with row count = len(df). Enforced. Stage derived if missing; clipped 0–4. OK.
- **synthetic_generator:** Writes client_{i+1}/clinical.csv, us_embeddings.npy, genomic_data.npy, pathology_data.npy, sensor_data.npy. Naming matches data_loader. OK.

---

## 6. Deployment

- **k8s_deployment.yaml:** Server command `["python", "federated/server.py", "5"]`; clients `["python", "federated/client.py", "<id>"]`; FLOWER_SERVER_URL=fedpinn-service:8080. Server binds 0.0.0.0:8080. OK.
- **deploy-aws.sh:** sed substitutes `endo-fedpinn:latest` with ECR URI; applies k8s_deployment.yaml and k8s_pdb.yaml. On Windows use WSL/Git Bash. OK.
- **k8s_pdb.yaml:** Selectors match deployment labels (app: fedpinn-server, app: fedpinn-client, app: fedpinn-dashboard). OK.

---

## 7. Optional / Non-Blocking

- **LIME:** In requirements.txt and README; not imported or used. Safe to leave for future use or remove from requirements.
- **analyze scripts:** Default roots reference paths like `Trnning dataset raw datset`; only matter when running those scripts with default args. No conflict with main app/federated flow.
- **Normalization mismatch (inference vs training):** App uses fixed mock_means/mock_stds at inference; backend and Tab4 use StandardScaler or upload mean/std. Documented in past; optional improvement to save/load scaler with model.

---

## 8. Checklist

- [x] No broken imports or references to deleted files.
- [x] Clinical column order and count (9) consistent; physics indices 6 and 7 correct.
- [x] Model forward (5 args) and batch keys aligned across app, client, evaluator.
- [x] data_loader stage handling (derive/clip) and optional .npy row check in place.
- [x] K8s and deploy scripts consistent; PDBs and selectors correct.
- [x] SHAP XAI uses same normalization as prediction; heuristic fallback on failure.
- [x] LIME unused but harmless (optional to remove from requirements).

**Conclusion:** No conflicts or blocking issues found. One optional dependency (LIME) is unused; rest of the project is consistent and cross-checked.
