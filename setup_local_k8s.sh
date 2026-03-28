#!/usr/bin/env bash
# setup_local_k8s.sh — Deploy EndoTwin to a local kind/minikube cluster.
#
# Usage:
#   chmod +x setup_local_k8s.sh
#   ./setup_local_k8s.sh                  # Full deploy (build + load + apply)
#   SKIP_BUILD=1 ./setup_local_k8s.sh     # Skip image build (image already loaded)
#
# Requirements: Docker, kubectl, kind (or minikube)

set -euo pipefail

IMAGE_NAME="endo-fedpinn:latest"
DEPLOY_YAML="deployment/k8s_deployment.yaml"
PDB_YAML="deployment/k8s_pdb.yaml"
NAMESPACE="default"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $*${NC}"; }
err()  { echo -e "${RED}❌  $*${NC}"; exit 1; }

echo ""
echo "=================================="
echo " EndoTwin Local K8s Deploy Script "
echo "=================================="
echo ""

# ── 1. Pre-flight checks ──────────────────────────────────────────────────────
echo "→ Checking prerequisites..."

command -v docker  >/dev/null 2>&1 || err "Docker not found. Install Docker Desktop."
command -v kubectl >/dev/null 2>&1 || err "kubectl not found. Install kubectl."

# Check Docker daemon is running
docker info >/dev/null 2>&1 || err "Docker daemon is not running. Start Docker Desktop."
ok "Docker is running"

# Detect cluster type from the ACTIVE kubectl context (not just installed CLIs)
KUBE_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "unknown")
echo "  Active kubectl context: ${KUBE_CONTEXT}"

if [[ "${KUBE_CONTEXT}" == "docker-desktop" ]]; then
    CLUSTER_TOOL="docker-desktop"
elif [[ "${KUBE_CONTEXT}" == kind-* ]] || command -v kind >/dev/null 2>&1 && kind get clusters 2>/dev/null | grep -q .; then
    CLUSTER_TOOL="kind"
elif [[ "${KUBE_CONTEXT}" == minikube* ]]; then
    CLUSTER_TOOL="minikube"
else
    CLUSTER_TOOL="unknown"
    warn "Unknown cluster context '${KUBE_CONTEXT}'. Will skip image loading (assuming image is pre-loaded)."
fi

# Verify kubectl can reach a cluster
kubectl cluster-info >/dev/null 2>&1 || err "kubectl cannot reach any cluster.\nFor Docker Desktop: enable Kubernetes in Docker Desktop Preferences.\nFor kind: kind create cluster --name endotwin\nFor minikube: minikube start"
ok "Kubernetes cluster is reachable (context: ${KUBE_CONTEXT})"

echo ""

# ── 2. Build Docker Image ─────────────────────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    warn "SKIP_BUILD=1 set — skipping Docker image build."
else
    echo "→ Building Docker image: ${IMAGE_NAME} ..."
    echo "  (This may take 5-10 minutes on first run due to PyTorch download)"
    docker build -t "${IMAGE_NAME}" -f deployment/Dockerfile . 2>&1 | tail -5
    ok "Docker image built: ${IMAGE_NAME}"
fi

echo ""

# ── 3. Load Image into Cluster ────────────────────────────────────────────────
echo "→ Loading image into local cluster..."

if [[ "${CLUSTER_TOOL}" == "docker-desktop" ]]; then
    ok "Docker Desktop K8s shares the host Docker daemon — image is already accessible. No load needed."

elif [[ "${CLUSTER_TOOL}" == "kind" ]]; then
    # Get the kind cluster name (first listed)
    KIND_CLUSTER=$(kind get clusters 2>/dev/null | head -1)
    if [[ -z "${KIND_CLUSTER}" ]]; then
        warn "No kind cluster found. Creating one named 'endotwin'..."
        kind create cluster --name endotwin
        KIND_CLUSTER="endotwin"
    fi
    echo "  Loading into kind cluster: ${KIND_CLUSTER}"
    kind load docker-image "${IMAGE_NAME}" --name "${KIND_CLUSTER}"
    ok "Image loaded into kind cluster '${KIND_CLUSTER}'"

elif [[ "${CLUSTER_TOOL}" == "minikube" ]]; then
    echo "  Building image inside minikube's Docker daemon..."
    eval "$(minikube docker-env)"
    docker build -t "${IMAGE_NAME}" -f deployment/Dockerfile . 2>&1 | tail -5
    ok "Image built inside minikube Docker daemon"

else
    warn "Unknown cluster tool — assuming image is already accessible to the cluster."
fi

echo ""

# ── 4. Apply Kubernetes Manifests ─────────────────────────────────────────────
echo "→ Applying Kubernetes manifests..."

# Clean up any old resources that may be stuck
echo "  Cleaning up old deployments (if any)..."
kubectl delete -f "${DEPLOY_YAML}" --ignore-not-found=true -n "${NAMESPACE}" 2>/dev/null || true
sleep 2

kubectl apply -f "${DEPLOY_YAML}" -n "${NAMESPACE}"
ok "Deployment manifest applied: ${DEPLOY_YAML}"

if [[ -f "${PDB_YAML}" ]]; then
    kubectl apply -f "${PDB_YAML}" -n "${NAMESPACE}"
    ok "PodDisruptionBudgets applied"
fi

echo ""

# ── 5. Wait for Pods ──────────────────────────────────────────────────────────
echo "→ Waiting for pods to start (up to 3 minutes)..."
echo "  (FL clients will enter CrashLoopBackOff until the server is Ready — this is expected)"

WAIT_SECS=180
POLL_INTERVAL=10
elapsed=0

while [[ $elapsed -lt $WAIT_SECS ]]; do
    # Count pods that are Running
    RUNNING=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null \
              | grep -c "Running" || true)
    TOTAL=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null \
            | wc -l | tr -d ' ')

    echo "  [${elapsed}s] Running: ${RUNNING}/${TOTAL} pods"
    
    # Consider success once server + dashboard are Running (clients join later)
    SVR_READY=$(kubectl get pods -n "${NAMESPACE}" -l app=fedpinn-server \
                --no-headers 2>/dev/null | grep -c "Running" || true)
    DASH_READY=$(kubectl get pods -n "${NAMESPACE}" -l app=fedpinn-dashboard \
                 --no-headers 2>/dev/null | grep -c "Running" || true)
    
    if [[ "$SVR_READY" -ge 1 && "$DASH_READY" -ge 1 ]]; then
        ok "Server and Dashboard are Running!"
        break
    fi

    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo ""

# ── 6. Final status ───────────────────────────────────────────────────────────
echo "──────────────────────────────────────"
echo " Final Pod Status"
echo "──────────────────────────────────────"
kubectl get pods -n "${NAMESPACE}" -o wide 2>/dev/null || true
echo ""
echo "──────────────────────────────────────"
echo " Services"
echo "──────────────────────────────────────"
kubectl get svc -n "${NAMESPACE}" 2>/dev/null || true
echo ""

# Access instructions
echo "──────────────────────────────────────"
echo " Access Instructions"
echo "──────────────────────────────────────"
if [[ "${CLUSTER_TOOL}" == "kind" ]]; then
    echo "  For kind: port-forward the dashboard with:"
    echo "  kubectl port-forward svc/dashboard-service 8501:8501"
    echo "  Then open: http://localhost:8501"
elif [[ "${CLUSTER_TOOL}" == "minikube" ]]; then
    echo "  For minikube: run:"
    echo "  minikube service dashboard-service"
fi
echo ""
ok "Deploy script complete."
