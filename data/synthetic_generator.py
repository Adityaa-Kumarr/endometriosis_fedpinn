import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_samples=1000, num_nodes=5, output_dir="dataset"):
    """
    Generates synthetic multi-modal healthcare data for Endometriosis prediction.
    Features include clinical records, simulated ultrasound embeddings, and hormones.
    """
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate labels (Endometriosis stages: 0=None, 1=Minimal, 2=Mild, 3=Moderate, 4=Severe)
    # Binary classification for simplicity: 0 (No endo or mild) and 1 (Moderate/Severe)
    labels = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
    stages = np.where(labels == 1, np.random.choice([3, 4], size=num_samples), np.random.choice([0, 1, 2], size=num_samples))
    
    # 1. Clinical Data
    age = np.random.normal(loc=32, scale=7, size=num_samples).clip(18, 55).astype(int)
    bmi = np.random.normal(loc=25, scale=4, size=num_samples).clip(18, 40)
    pelvic_pain = np.where(labels == 1, np.random.randint(5, 11, num_samples), np.random.randint(0, 6, num_samples))
    dysmenorrhea = np.where(labels == 1, np.random.randint(5, 11, num_samples), np.random.randint(0, 5, num_samples))
    dyspareunia = np.where(labels == 1, np.random.choice([0, 1], p=[0.2, 0.8], size=num_samples), np.random.choice([0, 1], p=[0.8, 0.2], size=num_samples))
    family_history = np.where(labels == 1, np.random.choice([0, 1], p=[0.4, 0.6], size=num_samples), np.random.choice([0, 1], p=[0.8, 0.2], size=num_samples))
    
    # 2. Hormonal Data
    # CA-125 is often elevated in endometriosis
    ca125 = np.where(labels == 1, np.random.normal(45, 15, num_samples), np.random.normal(20, 10, num_samples)).clip(0, 100)
    estradiol = np.random.normal(150, 50, num_samples).clip(20, 400)
    progesterone = np.random.normal(10, 5, num_samples).clip(0, 30)
    
    # Combine tabular data
    tabular_data = pd.DataFrame({
        'patient_id': range(num_samples),
        'age': age,
        'bmi': bmi,
        'pelvic_pain_score': pelvic_pain,
        'dysmenorrhea_score': dysmenorrhea,
        'dyspareunia': dyspareunia,
        'family_history': family_history,
        'ca125': ca125,
        'estradiol': estradiol,
        'progesterone': progesterone,
        'stage': stages,
        'label': labels
    })
    
    # 3. Simulate Modality Embeddings (from ResNet/Transformers running on raw data)
    # Ultrasound/MRI (128-dim)
    us_embeddings = np.random.randn(num_samples, 128)
    us_embeddings[labels == 1] += 0.5 
    
    # Genomic Expression/SNPs (256-dim)
    genomic_embeddings = np.random.randn(num_samples, 256)
    genomic_embeddings[labels == 1] += 0.3
    
    # Pathology/Histology (64-dim)
    path_embeddings = np.random.randn(num_samples, 64)
    path_embeddings[labels == 1] += 0.7
    
    # Wearable/Sensor Time-Series (32-dim)
    sensor_embeddings = np.random.randn(num_samples, 32)
    sensor_embeddings[labels == 1] -= 0.4
    
    tabular_data.to_csv(os.path.join(output_dir, "clinical_hormonal_data.csv"), index=False)
    np.save(os.path.join(output_dir, "ultrasound_embeddings.npy"), us_embeddings)
    np.save(os.path.join(output_dir, "genomic_data.npy"), genomic_embeddings)
    np.save(os.path.join(output_dir, "pathology_data.npy"), path_embeddings)
    np.save(os.path.join(output_dir, "sensor_data.npy"), sensor_embeddings)
    
    # Split into 'num_nodes' for federated learning
    print(f"Generated multi-modal data for {num_samples} patients.")
    clients_data_dir = os.path.join(output_dir, "clients")
    os.makedirs(clients_data_dir, exist_ok=True)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_nodes)
    
    for i, idx in enumerate(split_indices):
        client_dir = os.path.join(clients_data_dir, f"client_{i+1}")
        os.makedirs(client_dir, exist_ok=True)
        tabular_data.iloc[idx].to_csv(os.path.join(client_dir, "clinical.csv"), index=False)
        np.save(os.path.join(client_dir, "us_embeddings.npy"), us_embeddings[idx])
        np.save(os.path.join(client_dir, "genomic_data.npy"), genomic_embeddings[idx])
        np.save(os.path.join(client_dir, "pathology_data.npy"), path_embeddings[idx])
        np.save(os.path.join(client_dir, "sensor_data.npy"), sensor_embeddings[idx])
        print(f"Client {i+1} got {len(idx)} multi-modal samples.")

if __name__ == "__main__":
    generate_synthetic_data(num_samples=2500, num_nodes=5, output_dir="../endometriosis_fedpinn/dataset")
