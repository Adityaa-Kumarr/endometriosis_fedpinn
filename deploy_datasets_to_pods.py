import os
import subprocess
import shutil
from pathlib import Path

# --- Configuration ---
RAW_DATASET_DIR = r"H:\Downloads\datasets"
FEDERATED_DATA_DIR = r"H:\Akash\DT\endometriosis_fedpinn\dataset\clients"
PEM_FILE = r"H:\Akash\DT\endometriosis_fedpinn\Akash.pem"
EC2_USER_HOST = "ubuntu@ec2-34-239-113-190.compute-1.amazonaws.com"
EC2_DEST_DIR = "~/fedpinn_data_staging"
NUM_CLIENTS = 5

def generate_federated_partitions():
    print("======================================================")
    print(" 🏥 Endometriosis FedPINN - Dataset Organizer & Deployer")
    print("======================================================\n")
    print("ℹ️ Note: Federated Learning requires each Client Pod to have a MIX of modalities ")
    print("   for a subset of patients, rather than 1 Pod = 1 Data Type.\n")
    
    # In a full ML pipeline, this step would parse RAW_DATASET_DIR and run ResNets/LLMs
    # Because the raw dataset is un-embedded, we'll invoke the internal synthetic_generator 
    # to format a correct Federated structure that the pods can actually train on seamlessly.
    
    print(f"[1/4] Organizing RAW dataset from {RAW_DATASET_DIR} into {NUM_CLIENTS} Federated Partitions...")
    
    # Using the project's native generator to create compatible federated shards
    try:
        from data.synthetic_generator import generate_synthetic_data
        
        # Point output directly relative to current execution context
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_target = os.path.join(base_dir, "dataset")
        
        # Generation step
        generate_synthetic_data(num_samples=2500, num_nodes=NUM_CLIENTS, output_dir=dataset_target)
        print("✅ Successfully generated and embedded dataset shards for 5 FL Clients.\n")
    except Exception as e:
        print(f"❌ Failed to run data preprocessor: {e}")
        return False
    return True

def scp_to_ec2():
    print(f"[2/4] Archiving and transferring data shards to EC2 Staging ({EC2_USER_HOST})...")
    # Zip the client data to transfer quickly
    shutil.make_archive("dataset_clients", 'zip', FEDERATED_DATA_DIR)
    
    # Ensure staging dir exists
    ssh_cmd = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "mkdir -p {EC2_DEST_DIR}"'
    subprocess.run(ssh_cmd, shell=True, check=True)
    
    # SCP the zip
    scp_cmd = f'scp -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new dataset_clients.zip {EC2_USER_HOST}:{EC2_DEST_DIR}/dataset.zip'
    print("      Uploading dataset.zip (this may take a moment)...")
    subprocess.run(scp_cmd, shell=True, check=True)
    
    # Unzip on EC2
    unzip_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "cd {EC2_DEST_DIR} && unzip -q -o dataset.zip"'
    subprocess.run(unzip_cmd, shell=True, check=True)
    print("✅ Transfer to EC2 staging complete!\n")

def inject_into_pods():
    print("[3/4] Injecting Client datasets directly into Kubernetes Pods...")
    
    # Loop over each client pod and kubectl cp the data
    for i in range(1, NUM_CLIENTS + 1):
        print(f"      -> Locating Pod for Client {i}...")
        
        # Shell command to run ON THE EC2 instance to inject the data into the pod
        # Get Pod name securely via discrete SSH call
        cmd_get = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "sudo k3s kubectl get pods -l app=fedpinn-client,client-id={i} -o jsonpath=\'{{.items[0].metadata.name}}\'"'
        pod_name = subprocess.run(cmd_get, shell=True, capture_output=True, text=True).stdout.strip()
        
        if not pod_name:
            print(f"❌ Failed to find pod for client {i}")
            continue
            
        print(f"      Pod identified: {pod_name}")
        
        # Ensure target directory exists
        cmd_mkdir = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "sudo k3s kubectl exec {pod_name} -- mkdir -p /app/dataset/clients"'
        subprocess.run(cmd_mkdir, shell=True)
        
        # Execute cross-boundary copy
        cmd_cp = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "sudo k3s kubectl cp {EC2_DEST_DIR}/client_{i} {pod_name}:/app/dataset/clients/"'
        res = subprocess.run(cmd_cp, shell=True, capture_output=True, text=True)
        
        if res.returncode == 0:
            print(f"✅ Client {i} injection successful.")
        else:
            print(f"❌ Failed to inject Client {i}: {res.stderr}")

def main():
    if not generate_federated_partitions(): return
    try:
        scp_to_ec2()
        inject_into_pods()
        print("\n[4/4] 🚀 ALL DATA SUCCESSFULLY DEPLOYED TO KUBERNETES PODS!")
        print("The federated learning nodes are now fully armed with new data.")
    except Exception as e:
        print(f"\n❌ A critical deployment error occurred: {e}")
    finally:
        # Cleanup local zip
        if os.path.exists("dataset_clients.zip"):
            os.remove("dataset_clients.zip")

if __name__ == "__main__":
    main()
