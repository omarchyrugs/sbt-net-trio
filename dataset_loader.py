import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os

class DAICDataset(Dataset):
    def __init__(self, data_dir, participant_ids=None):
        """
        Args:
            data_dir: Path to your 'packets' folder containing .pt files.
            participant_ids: Optional list of IDs to include (for K-Fold/Train-Test split).
        """
        self.data_dir = data_dir
        
        # 1. Get all .pt files in the directory
        all_files = glob.glob(os.path.join(data_dir, "*.pt"))
        
        # 2. Filter by IDs if provided (Strict Participant Splitting)
        if participant_ids is not None:
            # Convert IDs to strings just in case
            p_id_strs = [str(pid) for pid in participant_ids]
            self.file_list = [f for f in all_files if os.path.basename(f).split('_')[0] in p_id_strs]
            print(self.file_list)
        else:
            self.file_list = all_files

        # 3. Flatten the turns
        # We want to know which file + which index inside that file corresponds to a 'global' index
        self.turn_map = []
        for f_path in self.file_list:
            # These are trusted local packet files with custom metadata,
            # so we load them with weights_only=False.
            data = torch.load(f_path, map_location="cpu", weights_only=False)
            num_turns = data['answer_embed'].shape[0]
            label = data['label'][0].item() # All turns in one file have the same label
            patient_id = os.path.basename(f_path).split('_')[0]
            for i in range(num_turns):
                self.turn_map.append({
                    'file': f_path,
                    'index': i,
                    'label': label,
                    'patient_id': patient_id
                })
        
        print(f"📊 Dataset initialized: {len(self.file_list)} participants, {len(self.turn_map)} total turns.")

    def __len__(self):
        return len(self.turn_map)

    def __getitem__(self, idx):
        target = self.turn_map[idx]
        
        # Load the specific file (PyTorch mmap=True makes this very fast)
        data = torch.load(target['file'], map_location="cpu", weights_only=False, mmap=True)
        i = target['index']
        
        # Extract the specific turn
        # answer_embed: [128, 768], audio: [128, 768], visual: [128, 768], mask: [128]
        print(target['patient_id'], i, target['label'])
        return {
            'text': data['answer_embed'][i],
            'audio': data['audio_embed'][i],
            'visual': data['visual_embed'][i],
            'context': data['context_embed'][i],
            'mask': data['mask'][i],
            'label': torch.tensor(target['label'], dtype=torch.float32)
        }

# Example usage:
dataset = DAICDataset(data_dir="./processed_data/packets", participant_ids=[300, 301, 302])