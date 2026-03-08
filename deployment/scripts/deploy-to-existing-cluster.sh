#!/usr/bin/env bash
#
# Build image, push to ECR, and deploy to an *existing* Kubernetes cluster
# (EKS or any other). Does not create EKS or ECR repo.
# Run from repository root (endometriosis_fedpinn).
#
# Prerequisites: AWS CLI, Docker, kubectl; cluster already exists and
# kubeconfig points to it (e.g. aws eks update-kubeconfig --name <cluster>).
#
# Usage:
#   ./deployment/scripts/deploy-to-existing-cluster.sh
#   AWS_REGION=eu-west-1 ECR_REPO_NAME=my-endo ./deployment/scripts/deploy-to-existing-cluster.sh
#
set -e

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="${ECR_REPO_NAME:-endo-fedpinn}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deployment"

echo "[deploy-existing] Repository root: $REPO_ROOT"
echo "[deploy-existing] Region: $AWS_REGION, ECR repo: $ECR_REPO_NAME"

command -v aws >/dev/null 2>&1 || { echo "Need: aws CLI"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Need: Docker"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Need: kubectl"; exit 1; }
aws sts get-caller-identity >/dev/null 2>&1 || { echo "AWS CLI not configured"; exit 1; }
kubectl cluster-info >/dev/null 2>&1 || { echo "kubectl not connected to a cluster. Run e.g. aws eks update-kubeconfig --name <cluster>"; exit 1; }

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

echo "[deploy-existing] ECR login..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "[deploy-existing] Build..."
docker build -t "$ECR_REPO_NAME:$IMAGE_TAG" -f "$DEPLOY_DIR/Dockerfile" "$REPO_ROOT"
docker tag "$ECR_REPO_NAME:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"
docker push "$ECR_URI:$IMAGE_TAG"

echo "[deploy-existing] Deploy manifests..."
sed -e "s|image: endo-fedpinn:latest|image: ${ECR_URI}:${IMAGE_TAG}|g" \
    -e "s|imagePullPolicy: IfNotPresent|imagePullPolicy: Always|g" \
    "$DEPLOY_DIR/k8s_deployment.yaml" | kubectl apply -f -
kubectl apply -f "$DEPLOY_DIR/k8s_pdb.yaml"

echo "[deploy-existing] Rollout status..."
kubectl rollout status deployment/fedpinn-server --timeout=300s || true
kubectl rollout status deployment/fedpinn-dashboard --timeout=300s || true

echo "[deploy-existing] Done. Dashboard URL: kubectl get svc dashboard-service"
