#!/usr/bin/env bash
set -e
echo "Starting K3s + Docker Setup on EC2..."

# 1. Update and install prerequisites
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y unzip jq curl

# Ensure conflicting packages are removed
sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y docker docker-engine docker.io containerd runc || true

# Install Docker reliably
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu

# 2. Extract project
echo "Extracting project archive..."
mkdir -p ~/fedpinn
tar -xzf ~/fedpinn.tar.gz -C ~/fedpinn
cd ~/fedpinn

# 3. Build docker image
echo "Building docker image endo-fedpinn:latest..."
sudo docker build -t endo-fedpinn:latest -f deployment/Dockerfile .

# 4. Install K3s configured for minimal setup
echo "Installing K3s Kubernetes..."
curl -sfL https://get.k3s.io | sh -

# 5. Export and import docker image into k3s containerd
echo "Importing image into K3s containerd..."
sudo docker save endo-fedpinn:latest -o endo-fedpinn.tar
sudo k3s ctr images import endo-fedpinn.tar

# 6. Apply manifests
echo "Applying Kubernetes manifests..."
sudo k3s kubectl apply -f deployment/k8s_deployment.yaml
sudo k3s kubectl apply -f deployment/k8s_pdb.yaml

echo "Setup Complete! Streamlit dashboard should be available in a couple minutes on Port 80."
sudo k3s kubectl get all -A
