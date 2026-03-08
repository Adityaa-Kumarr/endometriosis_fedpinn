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

## 📊 Endometriosis Digital Twin Dataset Collection

**Prepared by:** chengalva yamini  
**Objective:** This collection of medical datasets was carefully chosen to support the development of the Digital Twin system. The goal is to help detect endometriosis at an early stage and provide personalized insights for better understanding and treatment of the condition in women.

### Primary Datasets Utilized

1. **Uterus Computer Vision Dataset by Endometriosis (Roboflow)** `[Medical Imaging]`
   - **Source:** [Roboflow Universe](https://universe.roboflow.com/endometriosis/uterus-hosuh/dataset/3/download)
   - **Description:** Small set of uterus images with labels for early-stage model trials and learning experiments.

2. **Psychological Wellbeing and Endometriosis** `[Patient Records]`
   - **Source:** [Zenodo](https://zenodo.org/record/7878824)
   - **Description:** Responses from women with endometriosis on stress, mental health, and daily life impact, adding holistic insight to the Digital Twin.

3. **GLENDA v1.0 (Gynecologic Datasets)** `[Medical Imaging]`
   - **Source:** [Zenodo](https://zenodo.org/record/8205020)
   - **Description:** MRI scans with expert labels, surgery details, and pathology data. Enables full-cycle training from diagnosis to simulation.

4. **Endotect 2020 Challenge** `[Medical Imaging]`
   - **Source:** [CodaLab](https://competitions.codalab.org/competitions/25266)
   - **Description:** Contains labeled laparoscopic images, offering rich training material to improve AI object detection and surgical visuals segmentation.

5. **Structured Endometriosis Dataset** `[Patient Records]`
   - **Source:** [Zenodo](https://zenodo.org/record/8220727)
   - **Description:** Well-organized records from over 10,000 patients covering symptoms, diagnoses, and health details. Essential for statistical simulations.

6. **Endometriosis Diagnosis Statistics (ONS)** `[Patient Records]`
   - **Source:** [ONS](https://www.ons.gov.uk)
   - **Description:** Public demographic dataset including diagnosis details by age, ethnicity, and region for fair and inclusive modeling.

7. **Gene Expression: Eutopic vs. Ectopic Tissues** `[Genomic Data]`
   - **Source:** [Mendeley Data](https://data.mendeley.com/datasets/nb737txvr5/1)
   - **Description:** ΔCT Gene Expression Dataset comparing healthy and diseased tissues to track gene behavior and support pattern recognition.

8. **WESAD: Wearable Stress & Affect Dataset** `[Sensor Wearables]`
   - **Source:** [University of Siegen](https://ubicomp.eti.uni-siegen.de/home/datasets)
   - **Description:** Physiological signals (ECG, EDA, Respiration) for simulating real-time stress responses via live sensor feedback.

9. **Weighted Gene Co-Expression Network Data (WGCNA)** `[Genomic Data]`
   - **Source:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/30369943)
   - **Description:** Maps gene interactions linked to endometriosis, supporting molecular modeling inside the Twin.

10. **Gut vs. Cervical Microbiota Profiling** `[Pathology Report]`
    - **Source:** [Frontiers](https://www.frontiersin.org/articles/10.3389/fcimb.2021.788836/full)
    - **Description:** Studies gut and cervical bacteria to help find non-invasive biomarkers using microbiome patterns.

> 📁 **Access All Raw Datasets Here:** [Google Drive Repository](https://drive.google.com/drive/folders/15g4R9VRuKy12Zw5WDNRKpM0ZHEP6vFUp?usp=sharing)

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

**Automatic deploy on AWS (ECR + EKS + apply):**
```bash
./deployment/scripts/deploy-aws.sh
```
See `deployment/scripts/README.md` for prerequisites (AWS CLI, Docker, kubectl, eksctl) and options.

**1. Build the Docker Image (manual):**
```bash
docker build -t endo-fedpinn:latest -f deployment/Dockerfile .
```

**2. Deploy via Kubernetes:**
```bash
kubectl apply -f deployment/k8s_deployment.yaml
kubectl apply -f deployment/k8s_pdb.yaml
```

---

## 💡 Troubleshooting & Tips

- **Missing Modules (`ModuleNotFoundError`):** Make sure your virtual environment is activated (`source venv/bin/activate`) and you have run `pip install -r requirements.txt`.
- **Streamlit Port In Use:** If port 8501 is occupied by another app, you can specify a different port: `streamlit run app.py --server.port 8502`.
- **Hardware Acceleration:** PyTorch is configured to leverage your hardware (e.g., Apple Metal (`mps`) or NVIDIA GPU (`cuda`)) if available.

---