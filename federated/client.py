import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import sys
import os

# Adjust path to import models and data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ffnn_weighting import FeatureWeightingFFNN
from models.pinn import EndometriosisPINN, FullFedPINNModel
from data.data_loader import load_client_data

# Client training hyperparams
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
LR = 0.001

class EndometriosisClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, test_loader):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        criterion_prob = nn.BCELoss()
        criterion_stage = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        
        # Pre-allocate zero tensors for missing modalities
        max_batch_size = max([len(b['clinical']) for b in self.train_loader] + [BATCH_SIZE])
        self.genomic_zeros = torch.zeros((max_batch_size, 256), device=self.device)
        self.pathology_zeros = torch.zeros((max_batch_size, 64), device=self.device)
        self.sensor_zeros = torch.zeros((max_batch_size, 32), device=self.device)
        
        self.net.train()
        for epoch in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                clinical = batch['clinical'].to(self.device)
                us_data = batch['ultrasound'].to(self.device)
                labels = batch['label'].to(self.device)
                stages = batch['stage'].to(self.device).squeeze().long()
                
                # Use pre-allocated zeros dynamically resized via slices
                bs = len(clinical)
                prob, stage_logits, _, _ = self.net(clinical, us_data, 
                    self.genomic_zeros[:bs], 
                    self.pathology_zeros[:bs], 
                    self.sensor_zeros[:bs]
                )
                
                # Losses
                loss_prob = criterion_prob(prob, labels)
                loss_stage = criterion_stage(stage_logits, stages)
                
                # Physics-Informed Monotonicity Penalty
                # Indices mapping: [6]=CA125, [7]=Estradiol, [9]=IL-6, [11]=CRP 
                loss_phy = self.net.pinn.biomarker_monotonicity_loss(prob, clinical[:, 7], clinical[:, 6], il6=clinical[:, 9], crp=clinical[:, 11])
                
                # Total loss
                loss = loss_prob + loss_stage + loss_phy
                loss.backward()
                optimizer.step()
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion_prob = nn.BCELoss()
        
        self.net.eval()
        loss = 0.0
        correct = 0
        
        with torch.no_grad():
            
            # Pre-allocate zero tensors for missing modalities
            max_batch_size = max([len(b['clinical']) for b in self.test_loader] + [BATCH_SIZE])
            genomic_zeros = torch.zeros((max_batch_size, 256), device=self.device)
            pathology_zeros = torch.zeros((max_batch_size, 64), device=self.device)
            sensor_zeros = torch.zeros((max_batch_size, 32), device=self.device)
            
            for batch in self.test_loader:
                clinical = batch['clinical'].to(self.device)
                us_data = batch['ultrasound'].to(self.device)
                labels = batch['label'].to(self.device)
                
                bs = len(clinical)
                prob, _, _, _ = self.net(clinical, us_data,
                    genomic_zeros[:bs],
                    pathology_zeros[:bs],
                    sensor_zeros[:bs]
                )
                loss += criterion_prob(prob, labels).item()
                
                preds = (prob > 0.5).float()
                correct += (preds == labels).sum().item()
                
        accuracy = correct / len(self.test_loader.dataset)
        avg_loss = loss / len(self.test_loader)
        
        return avg_loss, len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def start_client(client_id):
    # Load Model
    ffnn = FeatureWeightingFFNN()
    pinn = EndometriosisPINN()
    net = FullFedPINNModel(ffnn, pinn)
    
    # Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "clients")
    train_loader, test_loader, _ = load_client_data(client_id, batch_size=BATCH_SIZE, data_dir=data_dir)
    
    # Server address: use FLOWER_SERVER_URL in K8s (e.g. fedpinn-service:8080), else localhost
    server_address = os.environ.get("FLOWER_SERVER_URL", "127.0.0.1:8080")
    print(f"Connecting to server: {server_address}")
    
    # Retry connection when server may not be ready yet (e.g. K8s pod startup order)
    max_retries = int(os.environ.get("FL_CLIENT_RETRIES", "12"))
    retry_delay = int(os.environ.get("FL_CLIENT_RETRY_DELAY", "10"))
    for attempt in range(max_retries):
        try:
            fl.client.start_client(
                server_address=server_address,
                client=EndometriosisClient(net, train_loader, test_loader).to_client(),
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                import time
                time.sleep(retry_delay)
            else:
                raise

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting FL Client {client_id}...")
    start_client(client_id)
