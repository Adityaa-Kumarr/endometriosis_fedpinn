"""
generate_model.py — Train the FedPINN model locally on all 5 client datasets
and save the result to global_model.pth.

This replaces the old placeholder that only saved random (untrained) weights.
Run:  python3 generate_model.py [--epochs N] [--lr LR] [--clients 1,2,3,4,5]
"""

import torch
import torch.nn as nn
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.ffnn_weighting import FeatureWeightingFFNN
from models.pinn import EndometriosisPINN, FullFedPINNModel
from data.data_loader import load_client_data


def train(epochs: int = 5, lr: float = 1e-3, client_ids: list = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if client_ids is None:
        client_ids = [1, 2, 3, 4, 5]

    # ── Build model ──────────────────────────────────────────────────────────
    ffnn  = FeatureWeightingFFNN()
    pinn  = EndometriosisPINN()
    model = FullFedPINNModel(ffnn, pinn).to(device)

    # If a previous checkpoint exists, start from it so we improve rather than reset
    model_path = "global_model.pth"
    if os.path.exists(model_path):
        try:
            ck = torch.load(model_path, map_location=device)
            model.load_state_dict(ck.get("full_model", ck))
            print("Loaded existing checkpoint — continuing training from these weights.")
        except Exception as e:
            print(f"Could not load existing checkpoint ({e}), training from scratch.")

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    bce_loss   = nn.BCELoss()
    ce_loss    = nn.CrossEntropyLoss()

    # ── Load data from every client ──────────────────────────────────────────
    print(f"\nLoading data for clients: {client_ids}")
    all_train_loaders, all_test_loaders = [], []
    for cid in client_ids:
        try:
            tr, te, _ = load_client_data(cid, batch_size=32, data_dir="dataset/clients")
            all_train_loaders.append(tr)
            all_test_loaders.append(te)
            print(f"  Client {cid}: {len(tr.dataset)} train, {len(te.dataset)} test samples")
        except Exception as e:
            print(f"  Client {cid}: SKIPPED — {e}")

    if not all_train_loaders:
        print("ERROR: No client data could be loaded. Aborting.")
        sys.exit(1)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training for {epochs} epoch(s) across {len(all_train_loaders)} client(s)...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches    = 0

        for loader in all_train_loaders:
            for batch in loader:
                clinical  = batch["clinical"].to(device)
                us        = batch["ultrasound"].to(device)
                genomic   = batch["genomic"].to(device)
                pathology = batch["pathology"].to(device)
                sensor    = batch["sensor"].to(device)
                labels    = batch["label"].to(device)
                stages    = batch["stage"].squeeze().long().to(device)

                optimizer.zero_grad()
                prob, stage_logits, _, _ = model(clinical, us, genomic, pathology, sensor)

                l_bce   = bce_loss(prob, labels)
                l_stage = ce_loss(stage_logits, stages)
                l_phy   = model.pinn.biomarker_monotonicity_loss(
                    prob, clinical[:, 7], clinical[:, 6]  # estradiol, ca125
                )
                loss = l_bce + 0.5 * l_stage + 0.1 * l_phy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batches    += 1

        scheduler.step()
        avg_loss = total_loss / max(batches, 1)
        print(f"  Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\nEvaluating on held-out test sets...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for loader in all_test_loaders:
            for batch in loader:
                clinical  = batch["clinical"].to(device)
                us        = batch["ultrasound"].to(device)
                genomic   = batch["genomic"].to(device)
                pathology = batch["pathology"].to(device)
                sensor    = batch["sensor"].to(device)
                labels    = batch["label"].to(device)

                prob, _, _, _ = model(clinical, us, genomic, pathology, sensor)
                preds   = (prob > 0.5).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

    accuracy = 100.0 * correct / max(total, 1)
    print(f"\n✅ Final Accuracy on held-out test sets: {accuracy:.2f}%")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save({"full_model": model.state_dict()}, model_path)
    print(f"✅ Trained weights saved to '{model_path}'")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EndoTwin FedPINN model locally.")
    parser.add_argument("--epochs",  type=int, default=5,         help="Number of training epochs (default: 5)")
    parser.add_argument("--lr",      type=float, default=1e-3,    help="Learning rate (default: 0.001)")
    parser.add_argument("--clients", type=str, default="1,2,3,4,5", help="Comma-separated client IDs (default: 1,2,3,4,5)")
    args = parser.parse_args()

    client_ids = [int(c.strip()) for c in args.clients.split(",")]
    train(epochs=args.epochs, lr=args.lr, client_ids=client_ids)

