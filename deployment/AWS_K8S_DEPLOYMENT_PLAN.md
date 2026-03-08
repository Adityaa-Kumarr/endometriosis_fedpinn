# AWS EC2 + Kubernetes Deployment Plan

**Project:** Federated Digital Twin for Endometriosis (FedPINN)  
**Target:** Deploy on AWS EC2 with Kubernetes; **each dataset in a different pod** (one pod per FL client).

---

## 1. Current State vs Ready-for-Deploy

| Area | Status | Notes |
|------|--------|------|
| **Docker** | ✅ Ready | `deployment/Dockerfile` builds single image; exposes 8080 (Flower), 8501 (Streamlit). |
| **Kubernetes manifests** | ✅ Updated | One server, **5 client deployments** (one pod per client_id, each uses its own dataset slice), one dashboard. |
| **Client server address** | ✅ Fixed | Client reads `FLOWER_SERVER_URL` (e.g. `fedpinn-service:8080`); falls back to `127.0.0.1:8080` locally. |
| **Per-pod data** | ✅ Supported | Each client pod runs with `client.py <id>`; `load_client_data(client_id)` loads only `dataset/clients/client_<id>/`. Data can be in image or mounted via `DATA_DIR`. |
| **.dockerignore** | ✅ Added | Reduces image size (excludes .git, cache, .pth, etc.). |
| **Resource limits** | ✅ Added | Requests/limits set for server, clients, dashboard. |
| **Probes** | ✅ Added | Liveness/readiness for server and dashboard; dashboard also has readinessProbe. |
| **PodDisruptionBudgets (PDBs)** | ✅ Added | `deployment/k8s_pdb.yaml`: server minAvailable 1, clients minAvailable 2, dashboard minAvailable 1. |
| **Persistent model storage** | Optional | Server does not save model by default; dashboard saves `global_model.pth` in its filesystem. For server-side persistence, add a PVC and mount (see `k8s_volumes_optional.yaml`). |
| **Per-client data on PVC** | Optional | If you do **not** bake dataset into image, use one PVC per client and `DATA_DIR` (see below). |

---

## 2. Architecture: One Dataset per Pod

- **Server:** 1 deployment, 1 replica. Flower FedProx on port 8080. Clients connect to Service `fedpinn-service:8080`.
- **Clients:** 5 deployments (`fedpinn-client-1` … `fedpinn-client-5`), 1 replica each. Each pod runs `federated/client.py <id>` and loads only `dataset/clients/client_<id>/` (either from image or from a mounted volume).
- **Dashboard:** 1 deployment, 1 replica. Streamlit on 8501; exposed via LoadBalancer (on AWS → ELB/ALB).

So **each dataset (client_1 … client_5) runs in a different pod**; no pod sees another client’s data.

---

## 3. Deployment Options on AWS

### Option A: Kubernetes on EC2 (kubeadm, K3s, or EKS)

| Step | Action |
|------|--------|
| 1 | Provision EC2 (e.g. 1 node or a small cluster). Recommended: Amazon Linux 2 or Ubuntu 22.04, min 4 GB RAM per node. |
| 2 | Install Docker and Kubernetes (e.g. `kubeadm`, or K3s, or create an EKS cluster). |
| 3 | Build image: `docker build -t endo-fedpinn:latest -f deployment/Dockerfile .` (on a machine with Docker). |
| 4 | Either push image to Amazon ECR and point K8s to it, or load the same image on each node (e.g. `docker save` / `docker load` for single-node). |
| 5 | Apply manifests: `kubectl apply -f deployment/k8s_deployment.yaml`. |
| 6 | Expose dashboard: Service is already `LoadBalancer`; on AWS this creates an ELB. Get URL: `kubectl get svc dashboard-service`. |

### Option B: EKS (managed Kubernetes)

- Create EKS cluster (e.g. `eksctl` or Terraform).
- Build image and push to ECR in the same account/region.
- Update `k8s_deployment.yaml` image to ECR URL, e.g. `123456789.dkr.ecr.region.amazonaws.com/endo-fedpinn:latest`.
- `kubectl apply -f deployment/k8s_deployment.yaml`.
- Use Ingress + ALB for dashboard if you prefer (optional).

---

## 4. Data in Pods: Two Approaches

### 4.1 Data baked in image (current default)

- Dockerfile: `COPY . .` includes `dataset/` (ensure `.dockerignore` does **not** exclude `dataset/` if you want it in the image).
- Each client pod runs with the same image; `client_id` (1–5) selects which folder under `dataset/clients/` to use. So **each pod uses a different logical dataset** (client_1 … client_5) from the same image.

### 4.2 Data on persistent volumes (one dataset per pod from storage)

- Do **not** include `dataset/` in the image (add `dataset/` to `.dockerignore`).
- Create one PVC (or EFS volume) per client; each volume contains only that client’s folder, e.g.:
  - `fedpinn-data-client-1` → contents of `dataset/clients/client_1/` (clinical.csv + *.npy).
- In each client deployment, mount the corresponding PVC at e.g. `/data/clients` and set env `DATA_DIR=/data/clients`. Ensure directory layout is `.../client_<id>/` so `load_client_data(client_id, data_dir="/data/clients")` finds `client_1`, `client_2`, etc.
- See `deployment/k8s_volumes_optional.yaml` for PVC examples and volume mount comments.

---

## 5. Pre-Deploy Checklist

- [ ] **Python deps:** `requirements.txt` is used in Dockerfile; no system libs missing (e.g. for PyTorch). If you use GPU nodes, use a CUDA base image and install GPU PyTorch.
- [ ] **Flower compatibility:** Current Flower server uses `0.0.0.0:8080`; clients use `FLOWER_SERVER_URL`. No code change needed for K8s DNS.
- [ ] **Startup order:** Clients will retry connecting to the server; start server first, then clients. Optionally use an init container or delay client deployment until server is ready.
- [ ] **Min clients:** Server strategy has `min_available_clients=2`. With 5 client pods, ensure at least 2 are Running before expecting training rounds to complete.
- [ ] **Network:** Pods must resolve `fedpinn-service` and reach it on 8080 (default in-cluster DNS and ClusterIP are sufficient).
- [ ] **Secrets:** No credentials in repo; if you add DB or API keys, use K8s Secrets and env or volume mounts.

---

## 6. Commands Summary

### 6.1 Automatic deploy with scripts (recommended)

From repo root, one-command full deploy (ECR + optional EKS + apply):

```bash
chmod +x deployment/scripts/deploy-aws.sh
./deployment/scripts/deploy-aws.sh
```

Use an existing cluster (no EKS create):

```bash
CREATE_EKS_CLUSTER=0 ./deployment/scripts/deploy-aws.sh
```

See **deployment/scripts/README.md** for prerequisites (AWS CLI, Docker, kubectl, eksctl) and all options.

### 6.2 Manual commands

```bash
# Build image (from repo root)
docker build -t endo-fedpinn:latest -f deployment/Dockerfile .

# Run locally (no K8s)
docker run -p 8080:8080 endo-fedpinn:latest python federated/server.py 5
docker run -p 8501:8501 endo-fedpinn:latest streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Deploy to Kubernetes
kubectl apply -f deployment/k8s_deployment.yaml
kubectl apply -f deployment/k8s_pdb.yaml

# Optional: per-client data on PVCs
# kubectl apply -f deployment/k8s_volumes_optional.yaml
# Then edit client deployments to add volumeMounts and DATA_DIR.

# Check
kubectl get pods -l app=fedpinn-server
kubectl get pods -l app=fedpinn-client
kubectl get pods -l app=fedpinn-dashboard
kubectl get svc
```

---

## 7. Gaps Addressed in This Plan

| Gap | Resolution |
|-----|------------|
| Client hardcoded `127.0.0.1:8080` | Client now uses `FLOWER_SERVER_URL` (set in K8s to `fedpinn-service:8080`). |
| All client pods used same client_id | Replaced single “fedpinn-client” deployment with 5 deployments (client-1 … client-5), each with its own `client_id` and dataset. |
| No resource limits | Added requests/limits and probes in `k8s_deployment.yaml`. |
| No .dockerignore | Added `.dockerignore` to keep image smaller. |
| Per-pod dataset not explicit | Documented: same image + different `client_id` = different dataset per pod; optional PVC + `DATA_DIR` for external data. |

---

## 8. Conclusion

Apply PDBs with: `kubectl apply -f deployment/k8s_pdb.yaml`.

The project **is ready to deploy on AWS EC2 with Kubernetes** with **each dataset in a different pod** (one pod per FL client). Use `deployment/k8s_deployment.yaml` as-is for the default “data in image” setup; use `deployment/k8s_volumes_optional.yaml` and `DATA_DIR` when each pod should load its dataset from a dedicated volume (e.g. EBS/EFS on AWS).
