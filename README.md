# Federated Digital Twin for Early Prediction of Endometriosis

Welcome to the **Federated Digital Twin** project! This repository contains a cutting-edge, privacy-preserving intelligent healthcare prediction system designed for the early detection and progression monitoring of Endometriosis. 

By combining Federated Learning (to preserve patient privacy), Physics-Informed Neural Networks (to model biological constraints), and a Digital Twin interface, this application provides an interactive, detailed visualization of disease states.

---

## 🌟 Key Features

- **Federated Learning (Flower):** Train models across multiple hospital nodes without sharing raw patient data.
- **Physics-Informed Neural Networks (PINNs):** Integrates biological progression constraints such as Estradiol and CA-125 dynamics.
- **Multi-Modal Feature Weighting (FFNN):** Fuses clinical symptoms and ultrasound features.
- **Explainable AI (XAI):** Uses SHAP/LIME to provide transparent, interpretable clinical insights.
- **Dynamic Digital Twin Simulation:** Interactive 3D visual mapping of the uterus structure and lesion progression over time via a Streamlit dashboard.

---

## 📁 Repository Structure

```text
endometriosis_fedpinn/
├── app.py                  # Main Streamlit Dashboard application
├── requirements.txt        # Python package dependencies
├── data/                   # Data generation and dataloaders for FL clients
├── models/                 # PyTorch model architectures (FFNN, FedPINN)
├── federated/              # Flower Server and Client scripts for FedProx
├── xai/                    # Explainable AI scripts (SHAP/LIME integration)
├── digital_twin/           # 3D Simulation logic and visualization components
├── validation/             # Evaluation metrics and validation scripts
└── deployment/             # Dockerfiles and Kubernetes deployment manifests
```

---

## 🚀 Getting Started for Teammates

Follow these step-by-step instructions to set up the environment and run the project locally.

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.9+**
- **Git**
- *(Optional)* Docker and Kubernetes (if you want to test deployment)

### 2. Clone the Repository
```bash
git clone <your-repo-link>
cd endometriosis_fedpinn
```

### 3. Create and Activate a Virtual Environment
It is highly recommended to isolate the project dependencies using a virtual environment. This prevents conflicts with other Python projects on your system.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
Install all required Python packages strictly defined for the environment.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎮 How to Run the Project

This project consists of several components that can be run independently or together depending on what you want to test.

### A. Run the Interactive Streamlit Dashboard (Main UI)
This is the primary user interface showcasing the Digital Twin, patient entry forms, and prediction visualizations. You'll spend most of your time testing features here!

```bash
streamlit run app.py
```
*Once running, open your browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).*

### B. Generate Synthetic Data
If you need to generate new multi-modal data for training and testing the federated learning clients:
```bash
python data/synthetic_generator.py
```

### C. Run the Federated Learning Simulation
To simulate the federated learning setup natively across multiple hospital nodes, you need to run the server and clients in separate terminal windows.

**Terminal 1 (Start the Global Server):**
```bash
# Make sure your virtual environment is activated!
python federated/server.py 3
```

**Terminal 2 (Start Client Node 1):**
```bash
# Make sure your virtual environment is activated!
python federated/client.py 1
```

*(You can open additional terminals to run more clients if needed, e.g., `python federated/client.py 2`)*

---

## 🐳 Docker & Kubernetes Deployment

If you are working on DevOps or want to package the application, you can containerize it natively.

**1. Build the Docker Image:**
```bash
docker build -t endo-fedpinn:latest -f deployment/Dockerfile .
```

**2. Deploy via Kubernetes:**
```bash
kubectl apply -f deployment/k8s_deployment.yaml
```

---

## 💡 Troubleshooting & Tips

- **Missing Modules (`ModuleNotFoundError`):** Make sure your virtual environment is activated (`source venv/bin/activate`) and you have run `pip install -r requirements.txt`.
- **Streamlit Port In Use:** If port 8501 is occupied by another app, you can specify a different port: `streamlit run app.py --server.port 8502`.
- **Hardware Acceleration:** PyTorch is configured to leverage your hardware (e.g., Apple Metal (`mps`) or NVIDIA GPU (`cuda`)) if available.

---