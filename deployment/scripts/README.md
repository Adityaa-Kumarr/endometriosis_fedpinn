# Deployment Scripts — Automatic AWS Setup and Deploy

These scripts automate building the Docker image, pushing to **Amazon ECR**, optionally creating an **EKS cluster**, and deploying the FedPINN stack (server, 5 clients, dashboard + PDBs) so everything can be set up and deployed on AWS from a single command.

---

## Prerequisites

Install and configure once:

| Tool | Purpose | Install |
|------|--------|--------|
| **AWS CLI v2** | ECR, EKS, auth | [Install AWS CLI](https://aws.amazon.com/cli/) |
| **Docker** | Build and push image | [Install Docker](https://docs.docker.com/get-docker/) |
| **kubectl** | Apply Kubernetes manifests | [Install kubectl](https://kubernetes.io/docs/tasks/tools/) |
| **eksctl** | Create EKS cluster (only for full auto) | [Install eksctl](https://eksctl.io/installation/) |

Configure AWS (if not already):

```bash
aws configure
# Enter AWS Access Key ID, Secret, region (e.g. us-east-1)
```

---

## Scripts

### 1. `deploy-aws.sh` — Full automatic deploy (ECR + optional EKS + deploy)

- Creates ECR repository if it doesn’t exist.
- Builds the image and pushes it to ECR.
- If `CREATE_EKS_CLUSTER=1` (default): creates an EKS cluster with eksctl, then updates kubeconfig.
- Substitutes the image in `k8s_deployment.yaml` with the ECR image and applies all manifests (deployments, services, PDBs).
- Waits for the dashboard LoadBalancer and prints the dashboard URL.

**Run from repository root** (parent of `deployment/`):

```bash
cd /path/to/endometriosis_fedpinn
chmod +x deployment/scripts/deploy-aws.sh
./deployment/scripts/deploy-aws.sh
```

**Environment variables (optional):**

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for ECR and EKS. |
| `ECR_REPO_NAME` | `endo-fedpinn` | ECR repository name. |
| `EKS_CLUSTER_NAME` | `fedpinn-cluster` | EKS cluster name. |
| `CREATE_EKS_CLUSTER` | `1` | Set to `0` to skip creating the cluster (use existing; kubeconfig must already point to it). |
| `IMAGE_TAG` | `latest` | Docker image tag to push and deploy. |

**Examples:**

```bash
# Default: create EKS cluster and deploy
./deployment/scripts/deploy-aws.sh

# Use existing cluster (you already ran create once, or use your own cluster)
CREATE_EKS_CLUSTER=0 ./deployment/scripts/deploy-aws.sh

# Custom region and cluster name
AWS_REGION=eu-west-1 EKS_CLUSTER_NAME=my-fedpinn ./deployment/scripts/deploy-aws.sh
```

**First run with `CREATE_EKS_CLUSTER=1`:** EKS cluster creation can take about 15–20 minutes. Later runs reuse the cluster.

---

### 2. `deploy-to-existing-cluster.sh` — Build, push, deploy (no EKS create)

Use when you already have a cluster (EKS or other) and `kubectl` is configured.

- Builds the image, pushes to ECR (ECR repo must already exist).
- Applies `k8s_deployment.yaml` (with ECR image) and `k8s_pdb.yaml`.

**Run from repository root:**

```bash
# Ensure kubeconfig points to your cluster, e.g.:
# aws eks update-kubeconfig --name my-cluster --region us-east-1

./deployment/scripts/deploy-to-existing-cluster.sh
```

**Environment variables:** `AWS_REGION`, `ECR_REPO_NAME`, `IMAGE_TAG` (same as above).

---

## Run on Windows

- **Option A:** Use **WSL2** (Ubuntu), install the prerequisites there, and run the same bash commands.
- **Option B:** Use **Git Bash** and run the scripts; ensure `aws`, `docker`, `kubectl`, and (for full deploy) `eksctl` are on PATH.
- **Option C:** Run each step manually using PowerShell or Command Prompt (see `AWS_K8S_DEPLOYMENT_PLAN.md`).

---

## What gets deployed

- **Server:** 1 pod (Flower FedProx on 8080).
- **Clients:** 5 pods (client-1 … client-5), each with its own dataset.
- **Dashboard:** 1 pod (Streamlit on 8501), exposed via LoadBalancer (ELB/ALB on AWS).
- **PDBs:** Server minAvailable 1, clients minAvailable 2, dashboard minAvailable 1.

After deploy, the script prints the dashboard URL when the LoadBalancer is ready (e.g. `http://<hostname>:8501`). You can also run:

```bash
kubectl get svc dashboard-service
kubectl get pods -l app=fedpinn-server
kubectl get pods -l app=fedpinn-client
```

---

## Cleanup (optional)

To remove the app from the cluster (keep EKS):

```bash
kubectl delete -f deployment/k8s_deployment.yaml
kubectl delete -f deployment/k8s_pdb.yaml
```

To delete the EKS cluster (only if you created it with eksctl):

```bash
eksctl delete cluster --name fedpinn-cluster --region us-east-1
```

Replace `fedpinn-cluster` and `us-east-1` if you used different values.
