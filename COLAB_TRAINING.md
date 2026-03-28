# Google Colab Training Guide

To train the Federated Endometriosis Model directly in a single Google Colab Notebook using Colab's GPUs and Flower's Simulation Mode, create a new notebook and paste the following codes into separate cells.

### Cell 1: Environment Setup
First, clone your repository and install the dependencies. Since we are simulating multiple clients on one machine, we need the `flwr[simulation]` extra.
```python
# Replace with your actual GitHub repository URL
!git clone https://github.com/<your-username>/endometriosis_fedpinn.git
%cd endometriosis_fedpinn

!pip install -r requirements.txt
!pip install flwr[simulation]
```

### Cell 2: Data Preparation (If needed)
If you haven't pre-generated the client data in your repo, generate the synthetic data now.
```python
!python data/synthetic_generator.py
```

### Cell 3: Flower Simulation Code
Instead of running separate terminal commands for the server and 5 clients, you can use Flower's `start_simulation` to run everything inside Colab's single GPU constraints.

```python
import flwr as fl
import torch
import os

# Import modules from your repository
from federated.client import EndometriosisClient
from models.ffnn_weighting import FeatureWeightingFFNN
from models.pinn import EndometriosisPINN, FullFedPINNModel
from data.data_loader import load_client_data

# Client generator function required by Flower Simulation
def client_fn(cid: str) -> fl.client.Client:
    # `cid` comes in as "0", "1", "2"... Map to your client IDs 1, 2, 3...
    client_id = int(cid) + 1 
    
    # Initialize the model architecture
    ffnn = FeatureWeightingFFNN()
    pinn = EndometriosisPINN()
    net = FullFedPINNModel(ffnn, pinn)
    
    # Load client specific data
    # Assuming dataset was generated in endometriosis_fedpinn/dataset/clients
    data_dir = os.path.join(os.getcwd(), "dataset", "clients")
    train_loader, test_loader, _ = load_client_data(client_id, batch_size=32, data_dir=data_dir)
    
    # Return the client instance
    return EndometriosisClient(net, train_loader, test_loader).to_client()

# Define the FedProx Strategy (Matching your server.py logic)
strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=5, # We simulate 5 clients
    proximal_mu=0.1,
)

# Set resources per client based on Colab GPU availability
# Colab typically provides 1 GPU. We can allow clients to share it.
client_resources = {
    "num_cpus": 2, 
    "num_gpus": 0.2 if torch.cuda.is_available() else 0.0 # Split 1 GPU among 5 clients
}

print("🚀 Starting Flower Federated Simulation on Google Colab...")

# Start the simulation (simulating 50 rounds across 5 clients)
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=5,
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
    client_resources=client_resources,
)
```

### Cell 4: Save the Global Model
Once the training completes, you can run the mock generator to save a `global_model.pth` (or plug in your actual server checkpointing logic code here if configured) and download it to your local machine.
```python
# Save the model
!python generate_model.py

# Download the model to your local machine
from google.colab import files
files.download('global_model.pth')
```
