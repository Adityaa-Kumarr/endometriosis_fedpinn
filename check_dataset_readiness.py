import os
import glob
import pandas as pd

DATASET_PATH = r"H:\Downloads\datasets"

EXPECTED_FOLDERS = [
    "genomic data",
    "medical imaging",
    "pathology reports",
    "patient Records",
    "sensor data"
]

CLINICAL_COLS = [
    'age', 'bmi', 'pelvic_pain_score', 'dysmenorrhea_score', 'dyspareunia',
    'family_history', 'ca125', 'estradiol', 'progesterone'
]

def check_dataset_readiness():
    print(f"===== Endometriosis FedPINN Dataset Validator =====")
    print(f"Target Path: {DATASET_PATH}\n")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ CRITICAL: The dataset path '{DATASET_PATH}' does NOT exist!")
        return

    print("--- 1. Modality Folders Check ---")
    for folder in EXPECTED_FOLDERS:
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            num_files = len(os.listdir(folder_path))
            print(f"✅ Found '{folder}' (Contains {num_files} items)")
        else:
            print(f"❌ Missing Expected Modality Folder: '{folder}'")

    print("\n--- 2. Clinical Data (Patient Records) Validation ---")
    patient_dir = os.path.join(DATASET_PATH, "patient Records")
    if os.path.isdir(patient_dir):
        csv_files = glob.glob(os.path.join(patient_dir, "*.csv"))
        if len(csv_files) > 0:
            first_csv = csv_files[0]
            print(f"Analyzing: {os.path.basename(first_csv)}")
            df = pd.read_csv(first_csv, low_memory=False)
            
            # Normalize column names for check
            raw_cols = df.columns.tolist()
            normalized_cols = [str(c).lower().strip().replace(' ', '_').replace('-', '') for c in df.columns]
            
            # Aliasing mapping based on data_loader.py
            if 'pelvic_pain' in normalized_cols and 'pelvic_pain_score' not in normalized_cols:
                normalized_cols.append('pelvic_pain_score')
            if 'dysmenorrhea' in normalized_cols and 'dysmenorrhea_score' not in normalized_cols:
                normalized_cols.append('dysmenorrhea_score')

            missing_cols = [col for col in CLINICAL_COLS if col not in normalized_cols]
            
            if missing_cols:
                print(f"⚠️ Missing canonical clinical columns: {missing_cols}")
            else:
                print(f"✅ Contains all {len(CLINICAL_COLS)} required canonical clinical columns.")

            # Target variables check
            target_vars = ['label', 'endometriosis_present', 'stage']
            found_targets = [t for t in target_vars if t in normalized_cols]
            if found_targets:
                print(f"✅ Found target variables: {found_targets}")
            else:
                print(f"❌ Missing required target variable (needs one of: {target_vars})")
                
            print(f"ℹ️ Total rows in CSV: {len(df)}")
        else:
            print("❌ No CSV files found inside 'patient Records'!")
    else:
         print("❌ 'patient Records' directory does not exist.")

    print("\n--- 3. Federation Compatibility Status ---")
    print("⚠️  Status: RAW / UNPROCESSED")
    print("The training architecture natively expects data to be partitioned across FL clients and mapped into deep embeddings (.npy).")
    print("Required FedPINN format: dataset/clients/client_{id}/ containing:")
    print("   -> clinical.csv, us_embeddings.npy, genomic_data.npy, pathology_data.npy, sensor_data.npy\n")
    print("Action Required: You will need to run an embedding pipeline or a data aggregator to preprocess this raw dataset before Model Training.")

if __name__ == "__main__":
    check_dataset_readiness()
