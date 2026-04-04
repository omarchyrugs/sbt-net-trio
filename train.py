import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# Assuming your files are named exactly as before
from dataset_loader import DepressionDataset
from model import DepressionPredictor

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            wav = batch['audio_values'].to(device)
            label = batch['label'].to(device).float()

            output = model(ids, mask, wav)
            loss = criterion(output, label)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(output) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    
    return total_loss / len(loader), acc, macro_f1, auc

def train():
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = '/root/autodl-tmp/data/daic_woz'
    label_csv = os.path.join(base_dir, 'labels.csv')
    
    # Paper Hyperparameters (Table 6)
    batch_size = 4 # Adjust based on VRAM (Paper uses 16, use grad_accum to match)
    grad_accum_steps = 4 
    initial_lr = 2e-5
    epochs = 85
    
    # 1. Dataset & Balanced Sampling
    full_dataset = DepressionDataset(label_csv, base_dir)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))

    # WeightedRandomSampler for Class Imbalance (Paper Page 18)
    labels = full_dataset.df['label'].values
    train_indices = train_set.indices
    train_labels = labels[train_indices]
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[int(t)] for t in train_labels]))
    sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight))

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. Model, Optimizer, Scheduler
    model = DepressionPredictor().to(device)
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = BCEWithLogitsLoss()

    best_f1 = 0
    os.makedirs('saved_models', exist_ok=True)

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        
        # --- Two-Stage Strategy (Paper Page 18) ---
        if epoch < 25:
            # Stage 1: Freeze Encoders
            for param in model.text_encoder.parameters(): param.requires_grad = False
            for param in model.audio_encoder.parameters(): param.requires_grad = False
        elif epoch == 25:
            # Stage 2: Unfreeze top layers (10 & 11) and lower LR
            print("\n🔓 Entering Stage 2: Unfreezing top encoder layers...")
            for name, param in model.text_encoder.named_parameters():
                if "layer.10" in name or "layer.11" in name: param.requires_grad = True
            for name, param in model.audio_encoder.named_parameters():
                if "layers.10" in name or "layers.11" in name: param.requires_grad = True
            
            for g in optimizer.param_groups:
                g['lr'] = 1e-5 # Fine-tuning LR

        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            wav = batch['audio_values'].to(device)
            label = batch['label'].to(device).float()

            output = model(ids, mask, wav)
            loss = criterion(output, label) / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps
        
        scheduler.step()

        # Evaluation
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        # Save based on Macro-F1 (Paper Page 18)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'saved_models/sbt_net_best.pt')
            print("⭐ New Best Model Saved (Macro-F1 improved)")

if __name__ == "__main__":
    train()