import torch
from torch.utils.data import DataLoader
from dataset_loader import DepressionDataset
from model import DepressionPredictor
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader):
        # Extract from dictionary keys
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        wav = batch['audio_values'].to(device)
        label = batch['label'].to(device)
        
        output = model(ids, mask, wav)
        # ... rest of your code
        preds = torch.sigmoid(output).cpu().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(label.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"[Test] Accuracy: {acc:.4f}, AUC: {auc:.4f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = DepressionDataset('data/daic_woz/labels.csv', 'data/daic_woz')
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
    model = DepressionPredictor().to(device)
    model.load_state_dict(torch.load("saved_models/best_model.pt"))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()