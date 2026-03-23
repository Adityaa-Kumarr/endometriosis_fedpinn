import os
import subprocess
import shutil

# --- Configuration ---
RAW_DATASET_DIR = r"H:\Downloads\datasets"
PEM_FILE = r"H:\Akash\DT\endometriosis_fedpinn\Akash.pem"
EC2_USER_HOST = "ubuntu@ec2-34-239-113-190.compute-1.amazonaws.com"
EC2_DEST_DIR = "~/fedpinn_raw_staging"
NUM_CLIENTS = 5

def scp_raw_to_ec2():
    print("======================================================")
    print(" 🏥 Endometriosis FedPINN - Raw Dataset Uploader")
    print("======================================================\n")
    
    print(f"[1/3] Archiving raw dataset completely from {RAW_DATASET_DIR}...")
    if not os.path.exists(RAW_DATASET_DIR):
        print(f"❌ Error: {RAW_DATASET_DIR} not found.")
        return False
        
    shutil.make_archive("raw_dataset", 'zip', RAW_DATASET_DIR)
    
    print(f"[2/3] Transferring archive to EC2 Staging ({EC2_USER_HOST})...")
    # Ensure staging dir exists
    ssh_cmd = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "mkdir -p {EC2_DEST_DIR}"'
    subprocess.run(ssh_cmd, shell=True, check=True)
    
    # SCP the zip
    scp_cmd = f'scp -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new raw_dataset.zip {EC2_USER_HOST}:{EC2_DEST_DIR}/raw_data.zip'
    print("      Uploading raw_dataset.zip (this may take a moment depending on the file sizes)...")
    subprocess.run(scp_cmd, shell=True, check=True)
    
    # Unzip on EC2 into a specific 'raw_extracted' folder
    unzip_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "cd {EC2_DEST_DIR} && unzip -q -o raw_data.zip -d raw_extracted"'
    subprocess.run(unzip_cmd, shell=True, check=True)
    print("✅ Transfer to EC2 staging complete!\n")
    return True

def inject_raw_into_pods():
    print("[3/3] Injecting EXACT Raw Data into Kubernetes Pods...")
    
    # Loop over each client pod and kubectl cp the data
    for i in range(1, NUM_CLIENTS + 1):
        print(f"      -> Locating Pod for Client {i}...")
        
        # Get Pod name
        cmd_get = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "sudo k3s kubectl get pods -l app=fedpinn-client,client-id={i} -o jsonpath=\'{{.items[0].metadata.name}}\'"'
        res_get = subprocess.run(cmd_get, shell=True, capture_output=True, text=True)
        pod_name = res_get.stdout.strip()
        
        if not pod_name:
            print(f"❌ Failed to find pod for client {i}")
            continue
            
        print(f"      Pod identified: {pod_name}. Injecting...")
        
        # We will inject the raw data into `/app/dataset/raw`
        # 1. Create the directory
        cmd_mkdir = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "sudo k3s kubectl exec {pod_name} -- mkdir -p /app/dataset/raw"'
        subprocess.run(cmd_mkdir, shell=True)
        
        # 2. Copy the extracted folders from staging directly into the pod's raw directory
        cmd_cp = f'ssh -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new {EC2_USER_HOST} "sudo k3s kubectl cp {EC2_DEST_DIR}/raw_extracted {pod_name}:/app/dataset/raw"'
        res = subprocess.run(cmd_cp, shell=True, capture_output=True, text=True)
        
        if res.returncode == 0:
            print(f"✅ Client {i} received the raw dataset successfully.")
        else:
            print(f"❌ Failed to inject into Client {i}: {res.stderr}")

def main():
    try:
        if scp_raw_to_ec2():
            inject_raw_into_pods()
            print("\n🚀 ALL RAW DATA SUCCESSFULLY DEPLOYED TO KUBERNETES PODS (/app/dataset/raw)!")
    except Exception as e:
        print(f"\n❌ A critical deployment error occurred: {e}")
    finally:
        if os.path.exists("raw_dataset.zip"):
            os.remove("raw_dataset.zip")

if __name__ == "__main__":
    main()
