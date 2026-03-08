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
        # FedProx: keep reference to global params for proximal term (mu/2) * ||w - w_global||^2
        global_params = [p.detach().clone() for p in self.net.parameters()]

        criterion_prob = nn.BCELoss()
        criterion_stage = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        proximal_mu = float(config.get("proximal_mu", config.get("proximal-mu", 0.0)))

        self.net.train()
        for epoch in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                clinical = batch['clinical'].to(self.device)
                us_data = batch['ultrasound'].to(self.device)
                labels = batch['label'].to(self.device)
                stages = batch['stage'].to(self.device).squeeze().long()
                batch_size = clinical.shape[0]
                genomic = batch.get('genomic', torch.zeros((batch_size, 256))).to(self.device)
                pathology = batch.get('pathology', torch.zeros((batch_size, 64))).to(self.device)
                sensor = batch.get('sensor', torch.zeros((batch_size, 32))).to(self.device)
                optimizer.zero_grad()
                prob, stage_logits, _, _ = self.net(clinical, us_data, genomic, pathology, sensor)

                loss_prob = criterion_prob(prob, labels)
                loss_stage = criterion_stage(stage_logits, stages)
                loss_phy = self.net.pinn.physics_informed_loss(prob, None, clinical[:, 7], clinical[:, 6])
                loss = loss_prob + loss_stage + loss_phy

                # FedProx proximal term: (mu/2) * ||w - w_global||^2
                if proximal_mu > 0:
                    prox_term = torch.tensor(0.0, device=self.device)
                    for p, p_glob in zip(self.net.parameters(), global_params):
                        prox_term = prox_term + (p - p_glob.to(self.device)).pow(2).sum()
                    loss = loss + (proximal_mu / 2.0) * prox_term

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
                batch_size = clinical.shape[0]
                genomic = batch.get('genomic', torch.zeros((batch_size, 256))).to(self.device)
                pathology = batch.get('pathology', torch.zeros((batch_size, 64))).to(self.device)
                sensor = batch.get('sensor', torch.zeros((batch_size, 32))).to(self.device)
                prob, _, _, _ = self.net(clinical, us_data, genomic, pathology, sensor)
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
    
    # Load Data: DATA_DIR can be set in K8s to mount per-pod dataset (e.g. /data/clients)
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "clients")
    train_loader, test_loader, _ = load_client_data(client_id, batch_size=BATCH_SIZE, data_dir=data_dir)
    
    # Server address: use FLOWER_SERVER_URL in K8s (e.g. fedpinn-service:8080), else localhost
    server_address = os.environ.get("FLOWER_SERVER_URL", "127.0.0.1:8080")
    print(f"Connecting to server: {server_address}")
    
    fl.client.start_client(
        server_address=server_address,
        client=EndometriosisClient(net, train_loader, test_loader).to_client(),
    )

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting FL Client {client_id}...")
    start_client(client_id)
