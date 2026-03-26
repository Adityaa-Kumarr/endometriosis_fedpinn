import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Canonical clinical feature columns (single source of truth for training/eval)
# Order must match model input: age, bmi, pelvic_pain_score, dysmenorrhea_score, dyspareunia, family_history, ca125, estradiol, progesterone
CLINICAL_FEATURE_COLUMNS = [
    'age', 'bmi', 'pelvic_pain_score', 'dysmenorrhea_score', 'dyspareunia',
    'family_history', 'ca125', 'estradiol', 'progesterone'
]
LABEL_COLUMN = 'label'
STAGE_COLUMN = 'stage'
ALTERNATIVE_LABEL_COLUMN = 'endometriosis_present'


def normalize_clinical_dataframe(df):
    """
    Normalize a DataFrame to canonical clinical columns.
    Accepts alternate names: pelvic_pain -> pelvic_pain_score, dysmenorrhea -> dysmenorrhea_score.
    Fills missing columns with NaN (caller should drop or fill).
    """
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    if 'pelvic_pain' in df.columns and 'pelvic_pain_score' not in df.columns:
        df['pelvic_pain_score'] = df['pelvic_pain']
    if 'dysmenorrhea' in df.columns and 'dysmenorrhea_score' not in df.columns:
        df['dysmenorrhea_score'] = df['dysmenorrhea']
    if 'ca-125' in df.columns and 'ca125' not in df.columns:
        df['ca125'] = df['ca-125']
    return df

class EndometriosisDataset(Dataset):
    def __init__(self, clinical_data, us_embeddings, labels, stages,
                 genomic_embeddings=None, pathology_embeddings=None, sensor_embeddings=None):
        self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        self.us_embeddings = torch.tensor(us_embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.stages = torch.tensor(stages, dtype=torch.float32).unsqueeze(1)
        n = len(self.labels)
        self.genomic = torch.tensor(genomic_embeddings, dtype=torch.float32) if genomic_embeddings is not None else torch.zeros((n, 256), dtype=torch.float32)
        self.pathology = torch.tensor(pathology_embeddings, dtype=torch.float32) if pathology_embeddings is not None else torch.zeros((n, 64), dtype=torch.float32)
        self.sensor = torch.tensor(sensor_embeddings, dtype=torch.float32) if sensor_embeddings is not None else torch.zeros((n, 32), dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'clinical': self.clinical_data[idx],
            'ultrasound': self.us_embeddings[idx],
            'genomic': self.genomic[idx],
            'pathology': self.pathology[idx],
            'sensor': self.sensor[idx],
            'label': self.labels[idx],
            'stage': self.stages[idx]
        }

def _load_optional_embeddings(client_path, name, expected_len):
    path = os.path.join(client_path, name)
    if os.path.isfile(path):
        arr = np.load(path)
        if len(arr) != expected_len:
            raise ValueError(f"{name} has {len(arr)} rows but clinical.csv has {expected_len}. Row counts must match.")
        return arr
    return None

def load_client_data(client_id, batch_size=32, data_dir="dataset/clients", test_split=0.2):
    """
    Loads and preprocesses data for a specific FL client.
    Optionally loads genomic, pathology, sensor embeddings if present.
    """
    client_path = os.path.join(data_dir, f"client_{client_id}")
    
    # Load data
    df = pd.read_csv(os.path.join(client_path, "clinical.csv"))
    us_embeddings = np.load(os.path.join(client_path, "us_embeddings.npy"))
    n_rows = len(df)
    genomic = _load_optional_embeddings(client_path, "genomic_data.npy", n_rows)
    pathology = _load_optional_embeddings(client_path, "pathology_data.npy", n_rows)
    sensor = _load_optional_embeddings(client_path, "sensor_data.npy", n_rows)
    
    # Features & Labels (canonical columns)
    df = normalize_clinical_dataframe(df)
    for col in CLINICAL_FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column for training: {col}. Expected: {CLINICAL_FEATURE_COLUMNS}")
    X_clinical = df[CLINICAL_FEATURE_COLUMNS].values
    if LABEL_COLUMN in df.columns:
        y_labels = df[LABEL_COLUMN].values
    elif ALTERNATIVE_LABEL_COLUMN in df.columns:
        y_labels = df[ALTERNATIVE_LABEL_COLUMN].values
    elif STAGE_COLUMN in df.columns:
        y_labels = (df[STAGE_COLUMN].values > 0).astype(int)
    else:
        raise ValueError("Need one of: label, endometriosis_present, or stage (to derive label).")

    if STAGE_COLUMN in df.columns:
        y_stages = np.clip(np.asarray(df[STAGE_COLUMN].values, dtype=np.int64), 0, 4)
    else:
        # Derive stage from label: 0 = no disease, 1+ = default to stage 2 (mild) for positive
        y_stages = np.where(y_labels > 0, 2, 0).astype(np.int64)
    
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
    
    def _slice(arr, idx):
        if arr is not None:
            return arr[idx]
        return None
    train_dataset = EndometriosisDataset(
        X_clinical_train, us_embeddings[train_idx], y_labels[train_idx], y_stages[train_idx],
        _slice(genomic, train_idx), _slice(pathology, train_idx), _slice(sensor, train_idx)
    )
    test_dataset = EndometriosisDataset(
        X_clinical_test, us_embeddings[test_idx], y_labels[test_idx], y_stages[test_idx],
        _slice(genomic, test_idx), _slice(pathology, test_idx), _slice(sensor, test_idx)
    )
    
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
