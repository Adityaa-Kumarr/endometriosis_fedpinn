import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class EndometriosisDataset(Dataset):
    def __init__(self, clinical_data, us_embeddings, labels, stages):
        self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        self.us_embeddings = torch.tensor(us_embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.stages = torch.tensor(stages, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'clinical': self.clinical_data[idx],
            'ultrasound': self.us_embeddings[idx],
            'label': self.labels[idx],
            'stage': self.stages[idx]
        }

def load_client_data(client_id, batch_size=32, data_dir="dataset/clients", test_split=0.2):
    """
    Loads and preprocesses data for a specific FL client.
    """
    client_path = os.path.join(data_dir, f"client_{client_id}")
    
    # Load data
    df = pd.read_csv(os.path.join(client_path, "clinical.csv"))
    us_embeddings = np.load(os.path.join(client_path, "us_embeddings.npy"))
    
    # Features & Labels
    feature_cols = ['age', 'bmi', 'pelvic_pain_score', 'dysmenorrhea_score', 'dyspareunia', 'family_history', 'ca125', 'estradiol', 'progesterone']
    
    X_clinical = df[feature_cols].values
    y_labels = df['label'].values
    y_stages = df['stage'].values
    
    # Train/Test Split
    np.random.seed(42 + client_id)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_split))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    # Normalize Clinical Data
    scaler = StandardScaler()
    X_clinical_train = scaler.fit_transform(X_clinical[train_idx])
    X_clinical_test = scaler.transform(X_clinical[test_idx])
    
    train_dataset = EndometriosisDataset(X_clinical_train, us_embeddings[train_idx], y_labels[train_idx], y_stages[train_idx])
    test_dataset = EndometriosisDataset(X_clinical_test, us_embeddings[test_idx], y_labels[test_idx], y_stages[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

def get_input_dims():
    """Returns the dimensions of clinical and US features."""
    return {'clinical': 9, 'ultrasound': 128}

if __name__ == "__main__":
    train_loader, test_loader, scaler = load_client_data(client_id=1, data_dir="../dataset/clients")
    print(f"Client 1 Train batches: {len(train_loader)}")
    for batch in test_loader:
        print("Clinical shape:", batch['clinical'].shape)
        print("US shape:", batch['ultrasound'].shape)
        break
