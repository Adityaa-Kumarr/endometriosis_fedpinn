import subprocess
import time

PEM_FILE = r"H:\Akash\DT\endometriosis_fedpinn\Akash.pem"
EC2_USER_HOST = "ubuntu@ec2-34-239-113-190.compute-1.amazonaws.com"
EC2_DEST_DIR = "~/fedpinn"

def run_cmd(cmd, desc):
    print(f"\n[⏳] {desc}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[✅] Success: {desc}")
    except subprocess.CalledProcessError as e:
        print(f"[❌] Failed: {desc}")
        print(f"Error details: {e}")
        exit(1)

def main():
    print("======================================================")
    print(" 🏥 Endometriosis FedPINN - EC2 Live Update Trigger")
    print("======================================================\n")

    # 1. SCP the updated app.py & README.md
    scp_cmd = f'scp -i "{PEM_FILE}" -o StrictHostKeyChecking=accept-new app.py README.md {EC2_USER_HOST}:{EC2_DEST_DIR}/'
    run_cmd(scp_cmd, "Transferring app.py and README.md to EC2")

    # 2. SSH to rebuild Docker image
    build_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "cd {EC2_DEST_DIR} && sudo docker build -t endo-fedpinn:latest -f deployment/Dockerfile ."'
    run_cmd(build_cmd, "Building new Docker image on EC2")

    # 3. SSH to save Docker image to tar
    save_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "cd {EC2_DEST_DIR} && sudo docker save endo-fedpinn:latest -o endo-fedpinn.tar"'
    run_cmd(save_cmd, "Saving Docker image to local tarball")

    # 4. SSH to import into K3s container registry
    import_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "cd {EC2_DEST_DIR} && sudo k3s ctr images import endo-fedpinn.tar"'
    run_cmd(import_cmd, "Importing image into K3s container runtime")

    # 5. SSH to rollout restart the dashboard deployment
    restart_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "sudo k3s kubectl rollout restart deployment fedpinn-dashboard"'
    run_cmd(restart_cmd, "Initiating Kubernetes rolling restart for fedpinn-dashboard")
    
    # 6. Delete old pod aggressively to force rapid start (Optional but helpful for fast dev loop)
    # Get old pod name and delete
    delete_cmd = f'ssh -i "{PEM_FILE}" {EC2_USER_HOST} "sudo k3s kubectl delete pods -l app=fedpinn-dashboard"'
    run_cmd(delete_cmd, "Terminating old dashboard pods to force immediate update")

    print("\n🚀 UPDATE COMPLETE! The Live EC2 Dashboard is restarting with the new UI.")

if __name__ == "__main__":
    main()
