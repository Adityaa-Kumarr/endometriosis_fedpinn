#!/usr/bin/env bash
#
# Automatically build, push to ECR, optionally create EKS cluster, and deploy
# FedPINN to AWS. Run from repository root (endometriosis_fedpinn).
#
# Prerequisites: AWS CLI, Docker, kubectl, eksctl (only if CREATE_EKS_CLUSTER=1)
# Optional: set AWS_PROFILE or AWS_DEFAULT_REGION
#
# Usage:
#   ./deployment/scripts/deploy-aws.sh
#   CREATE_EKS_CLUSTER=0 ./deployment/scripts/deploy-aws.sh   # use existing cluster
#   AWS_REGION=eu-west-1 EKS_CLUSTER_NAME=my-fedpinn ./deployment/scripts/deploy-aws.sh
#
set -e

# --- Config (override with env) ---
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="${ECR_REPO_NAME:-endo-fedpinn}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-fedpinn-cluster}"
CREATE_EKS_CLUSTER="${CREATE_EKS_CLUSTER:-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deployment"

echo "[deploy-aws] Repository root: $REPO_ROOT"
echo "[deploy-aws] Region: $AWS_REGION, ECR repo: $ECR_REPO_NAME, EKS cluster: $EKS_CLUSTER_NAME"
echo "[deploy-aws] Create EKS cluster: $CREATE_EKS_CLUSTER"

# --- Checks ---
command -v aws >/dev/null 2>&1 || { echo "Need: aws CLI. Install https://aws.amazon.com/cli/"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Need: Docker. Install https://docs.docker.com/get-docker/"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Need: kubectl. Install https://kubernetes.io/docs/tasks/tools/"; exit 1; }
aws sts get-caller-identity >/dev/null 2>&1 || { echo "AWS CLI not configured or no credentials. Run aws configure."; exit 1; }

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

# --- 1. ECR repo + build + push ---
echo "[deploy-aws] Creating ECR repository if not exists..."
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" --image-scanning-configuration scanOnPush=true

echo "[deploy-aws] Logging Docker into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "[deploy-aws] Building image..."
docker build -t "$ECR_REPO_NAME:$IMAGE_TAG" -f "$DEPLOY_DIR/Dockerfile" "$REPO_ROOT"

echo "[deploy-aws] Tagging and pushing to ECR..."
docker tag "$ECR_REPO_NAME:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"
docker push "$ECR_URI:$IMAGE_TAG"

# --- 2. EKS cluster (optional) ---
if [ "$CREATE_EKS_CLUSTER" = "1" ]; then
  command -v eksctl >/dev/null 2>&1 || { echo "CREATE_EKS_CLUSTER=1 but eksctl not found. Install https://eksctl.io/ or set CREATE_EKS_CLUSTER=0."; exit 1; }
  if aws eks describe-cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "[deploy-aws] EKS cluster '$EKS_CLUSTER_NAME' already exists, skipping create."
  else
    echo "[deploy-aws] Creating EKS cluster (this may take 15–20 min)..."
    eksctl create cluster \
      --name "$EKS_CLUSTER_NAME" \
      --region "$AWS_REGION" \
      --nodegroup-name ng1 \
      --node-type t3.medium \
      --nodes 2 \
      --nodes-min 1 \
      --nodes-max 3 \
      --managed \
      --asg-access
  fi
  echo "[deploy-aws] Updating kubeconfig..."
  aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION"
fi

# --- 3. Deploy manifests (substitute image to ECR URI) ---
echo "[deploy-aws] Deploying to Kubernetes..."
export ECR_URI
export IMAGE_TAG
# Substitute image in deployment file and apply
sed -e "s|image: endo-fedpinn:latest|image: ${ECR_URI}:${IMAGE_TAG}|g" \
    -e "s|imagePullPolicy: IfNotPresent|imagePullPolicy: Always|g" \
    "$DEPLOY_DIR/k8s_deployment.yaml" | kubectl apply -f -

kubectl apply -f "$DEPLOY_DIR/k8s_pdb.yaml"

echo "[deploy-aws] Waiting for deployments to roll out..."
kubectl rollout status deployment/fedpinn-server --timeout=300s || true
kubectl rollout status deployment/fedpinn-dashboard --timeout=300s || true

# --- 4. Dashboard URL (LoadBalancer can take 2–5 min on AWS) ---
echo "[deploy-aws] Waiting for dashboard LoadBalancer (may take 2–5 min)..."
for i in {1..30}; do
  HOST="$(kubectl get svc dashboard-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)"
  if [ -n "$HOST" ]; then
    echo ""
    echo "=== Dashboard URL (HTTP): http://$HOST:8501 ==="
    echo " (If using HTTPS, configure Ingress/ALB and use your own URL.)"
    break
  fi
  echo -n "."
  sleep 10
done
if [ -z "$HOST" ]; then
  echo ""
  echo "Run: kubectl get svc dashboard-service"
  echo "When EXTERNAL-IP or hostname appears, open http://<that>:8501"
fi

echo "[deploy-aws] Done."
