import torch
import os
from tqdm import tqdm

data_dir = "/scratch/dipanjan/rugraj/DIAC-WOZ/processed_data/packets"
files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

print(f"Checking {len(files)} files...")
for f in tqdm(files):
    try:
        _ = torch.load(os.path.join(data_dir, f), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"\n❌ CORRUPTED: {f}")
        print(f"Reason: {e}")