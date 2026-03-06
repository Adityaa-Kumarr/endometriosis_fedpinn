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
        
        self.net.train()
        for epoch in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                clinical = batch['clinical'].to(self.device)
                us_data = batch['ultrasound'].to(self.device)
                labels = batch['label'].to(self.device)
                stages = batch['stage'].to(self.device).squeeze().long()
                
                optimizer.zero_grad()
                prob, stage_logits, _ = self.net(clinical, us_data)
                
                # Losses
                loss_prob = criterion_prob(prob, labels)
                loss_stage = criterion_stage(stage_logits, stages)
                
                # Physics Loss Penalty
                # Assuming index 7 is estradiol and 6 is ca125 in the normalized clinical array 
                # (You would typically denormalize to get true physical constraints but this is demonstrative)
                loss_phy = self.net.pinn.physics_informed_loss(prob, None, clinical[:, 7], clinical[:, 6])
                
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
            for batch in self.test_loader:
                clinical = batch['clinical'].to(self.device)
                us_data = batch['ultrasound'].to(self.device)
                labels = batch['label'].to(self.device)
                
                prob, _, _ = self.net(clinical, us_data)
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
    
    # Start Client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=EndometriosisClient(net, train_loader, test_loader).to_client(),
    )

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting FL Client {client_id}...")
    start_client(client_id)
