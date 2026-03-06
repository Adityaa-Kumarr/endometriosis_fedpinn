# Federated Digital Twin for Early Prediction of Endometriosis

This project implements an intelligent healthcare prediction system using:
- **Federated Learning (Flower)** to maintain patient privacy across hospital nodes.
- **Physics-Informed Neural Networks (PINNs)** to integrate biological progression constraints (Estradiol, CA-125 dynamics).
- **Feed Forward Neural Network (FFNN)** for multi-modal feature weighting (Clinical + Ultrasound).
- **Explainable AI (SHAP/LIME)** for clinical transparency.
- **Digital Twin Simulation** for visual mapping of uterus structure and lesion progression.

## Directory Structure
- `data/`: Synthetic data generator and FL client data loaders.
- `models/`: PyTorch definitions for `FeatureWeightingFFNN` and `EndometriosisPINN`.
- `federated/`: Flower `server.py` and `client.py` for Adaptive FedProx.
- `xai/`: SHAP integration script.
- `digital_twin/`: Mock 3D simulator logic.
- `validation/`: Evaluation metrics.
- `deployment/`: Dockerfile and Kubernetes manifests.

## Installation & Setup
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Components

### 1. Data Generation
Generate synthetic multi-modal data for clients:
```bash
python data/synthetic_generator.py
```

### 2. Streamlit Dashboard (Digital Twin & Prediction UI)
Launch the interactive medical dashboard:
```bash
streamlit run app.py
```

### 3. Federated Learning Simulation
Run the Flower Server:
```bash
python federated/server.py 3
```
In a separate terminal, start Client 1:
```bash
python federated/client.py 1
```

### 4. Kubernetes Deployment
Build the docker image:
```bash
docker build -t endo-fedpinn:latest -f deployment/Dockerfile .
```
Apply definitions:
```bash
kubectl apply -f deployment/k8s_deployment.yaml
```
