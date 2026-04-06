import argparse
import torch
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# Import your modules (assuming they are in the same directory or PYTHONPATH)
from dataset_loader import DAICDataset
from STBNetTrio import SBTNetTrio
from mail import send_email_alert, get_IST

torch.serialization.add_safe_globals([np._core.multiarray.scalar])
def validate_patient_level(model, loader, device):
    model.eval()
    
    # Storage for aggregation
    # Key: Patient_ID, Value: List of probabilities from their turns
    patient_probs = {} 
    patient_labels = {} 
    
    with torch.no_grad():
        for batch in loader:
            # 1. Standard Device Move
            # Ensure your Dataset's __getitem__ includes 'patient_id'
            p_ids = batch['patient_id'] 
            labels = batch['label'].to(device)
            
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
            
            # 2. Get Logits -> Sigmoid for Probabilities
            logits = model(batch_gpu)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # 3. Group turns by Patient ID
            for i, p_id in enumerate(p_ids):
                if p_id not in patient_probs:
                    patient_probs[p_id] = []
                    patient_labels[p_id] = batch['label'][i].item()
                
                patient_probs[p_id].append(probs[i])

    # 4. Final Patient-Level Decision
    y_true = []
    y_pred = []
    
    for p_id in patient_probs:
        # Strategy: Mean Probability
        # If the average probability across all turns > 0.5, classify as Depressed (1)
        avg_prob = np.mean(patient_probs[p_id])
        final_diagnosis = 1 if avg_prob > 0.5 else 0
        
        y_pred.append(final_diagnosis)
        y_true.append(patient_labels[p_id])

    # 5. Calculate Clinical Metrics
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics

def train_full_kfold(args, device):
    # --- STARTING EMAIL ---
    start_subject = f"🚀 Training Started: {args.job_name}"
    start_body = f"""
    <h3>Execution Initiated</h3>
    <ul>
        <li><b>Time:</b> {get_IST()}</li>
        <li><b>Device:</b> {device}</li>
        <li><b>Data Dir:</b> {args.data_dir}</li>
        <li><b>Learning Rate:</b> {args.lr}</li>
        <li><b>Batch Size:</b> {args.batch_size}</li>
    </ul>
    <p>Monitoring logs at: <code>{args.log_file}</code></p>
    """
    send_email_alert(start_subject, start_body)

    # Get all IDs
    all_ids = sorted(list(set([f.split('_')[0] for f in os.listdir(args.data_dir) if f.endswith('.pt')])))
    
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []
    
    # Initialize Log
    if not os.path.exists(args.log_file):
        pd.DataFrame(columns=['fold', 'epoch', 'acc', 'f1', 'precision', 'recall', 'time']).to_csv(args.log_file, index=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        print(f"\n# ====== FOLD {fold+1}/{args.folds} ====== #")
        
        train_ids = [all_ids[i] for i in train_idx]
        val_ids = [all_ids[i] for i in val_idx]

        train_ds = DAICDataset(args.data_dir, participant_ids=train_ids)
        val_ds = DAICDataset(args.data_dir, participant_ids=val_ids)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = SBTNetTrio(dim=768).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) # introduce cosine annealing- will dynamically adjust learning rate
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))

        best_fold_f1 = 0
        
        for epoch in range(args.epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch['label'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            metrics = validate_patient_level(model, val_loader, device)
            
            # Log to CSV
            log_data = [fold+1, epoch+1, metrics['acc'], metrics['f1'], metrics['precision'], metrics['recall'], get_IST()]
            pd.DataFrame([log_data]).to_csv(args.log_file, mode='a', header=False, index=False)
            
            if metrics['f1'] > best_fold_f1:
                best_fold_f1 = metrics['f1']
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")

        fold_results.append(best_fold_f1)
        send_email_alert(
            f"📊 Fold {fold+1} Complete", 
            f"Best F1: {best_fold_f1:.4f}<br>Mean F1: {np.mean(fold_results):.4f}"
        )

    send_email_alert("✅ FINAL SUCCESS", f"Mean F1: {np.mean(fold_results):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--job_name", type=str, default="SBT-Net-Run")
    parser.add_argument("--log_file", type=str, default="training_results.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_full_kfold(args, device)