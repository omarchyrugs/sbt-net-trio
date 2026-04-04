import torch
from torch.utils.data import DataLoader
from dataset_loader import DepressionDataset
from model import DepressionPredictor
import torch.nn as nn

def dry_run():
    # 1. Setup paths
    base_dir = r'/Users/gurusai/Desktop/DAIC_Raw'
    label_csv = 'labels.csv'
    device = torch.device('cpu') # Force CPU for laptop testing
    
    print("🚀 Starting Dry Run...")

    # 2. Init Dataset with 2 patients
    # We use a very small median_samples just to speed up the test
    dataset = DepressionDataset(
        label_file=label_csv, 
        base_dir=base_dir, 
        median_samples=16000 * 2 # 2 seconds of audio
    )
    
    # Batch size of 2 (your whole dataset)
    loader = DataLoader(dataset, batch_size=2)

    # 3. Init Model
    print("📦 Loading Model (this may take a minute to download ALBERT/Wav2Vec2)...")
    model = DepressionPredictor().to(device)
    
    # 4. Single Forward Pass
    print("🏃 Running Forward Pass...")
    batch = next(iter(loader))
    
    ids = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    wav = batch['audio_values'].to(device)
    labels = batch['label'].to(device).float()

    try:
        logits = model(ids, mask, wav)
        print(f"✅ Forward Pass Success! Output shape: {logits.shape}")
        
        # 5. Single Backward Pass (The real test)
        print("📉 Running Backward Pass...")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"✅ Backward Pass Success! Loss: {loss.item():.4f}")
        
        print("\n🎉 Dry run complete. The model is ready for the full dataset!")

    except Exception as e:
        print(f"\n❌ Dry Run Failed!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dry_run()