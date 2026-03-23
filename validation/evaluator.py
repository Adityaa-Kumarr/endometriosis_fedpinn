import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader):
    """
    Evaluates the FedPINN model on a given dataloader 
    and returns comprehensive clinical metrics.
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            clinical = batch['clinical']
            us_data = batch['ultrasound']
            labels = batch['label'].numpy()
            genomic = batch.get('genomic', torch.zeros((len(clinical), 256)))
            pathology = batch.get('pathology', torch.zeros((len(clinical), 64)))
            sensor = batch.get('sensor', torch.zeros((len(clinical), 32)))
            
            prob, _, _, _ = model(clinical, us_data, genomic, pathology, sensor)
            prob_np = prob.numpy()
            
            preds = (prob_np > 0.5).astype(int)
            
            all_labels.extend(labels)
            all_probs.extend(prob_np)
            all_preds.extend(preds)
            
    all_labels = np.array(all_labels).ravel()
    all_probs = np.array(all_probs).ravel()
    all_preds = np.array(all_preds).ravel()
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    # ROC-AUC requires both classes to be present in true labels
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics['roc_auc'] = 0.5 # Default if impossible to calculate
        
    return metrics

if __name__ == "__main__":
    print("Testing evaluator...")
    # Mock labels and probs
    labels = np.array([0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.8, 0.4, 0.7])
    preds = (probs > 0.5).astype(int)
    
    print(f"Accuracy: {accuracy_score(labels, preds)}")
    print(f"ROC-AUC: {roc_auc_score(labels, probs)}")
