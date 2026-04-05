import os
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer

class DepressionDataset(Dataset):
    def __init__(self, label_file, base_dir, tokenizer_name="albert-large-v2", max_len=128, sr=16000, median_samples=None):
        self.df = pd.read_csv(label_file)
        self.base_dir = base_dir
        self.tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.sr = sr
        
        # Calculate median if not provided
        if median_samples is None:
            print("Calculating dataset median duration...")
            self.median_samples = self.calculate_dataset_median()
        else:
            self.median_samples = median_samples

    def calculate_dataset_median(self):
        durations = []
        # Use itertuples() for efficient row-by-row iteration
        for row in self.df.itertuples():
            p_id = str(row.participant_id)
            # Match DAIC-WOZ structure: BASE/300_P/300_AUDIO.wav
            audio_path = os.path.join(self.base_dir, f"{p_id}_P", f"{p_id}_AUDIO.wav")
            
            if os.path.exists(audio_path):
                duration = librosa.get_duration(path=audio_path)
                durations.append(duration)
        
        if not durations:
            print("⚠️ Warning: No audio files found! Defaulting to 20s.")
            return self.sr * 20
            
        median_sec = np.median(durations)
        print(f"Dataset Median: {median_sec:.2f} seconds")
        return int(median_sec * self.sr)

    def _prepare_audio(self, wav):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
            
        if len(wav) > self.median_samples:
            wav = wav[:self.median_samples]
        else:
            pad_len = self.median_samples - len(wav)
            wav = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0)
        
        # Standardize: (x - mu) / std
        wav = (wav - wav.mean()) / (wav.std() + 1e-5)
        return wav

    def _load_transcript(self, path):
        try:
            df_t = pd.read_csv(path, sep='\t')
            # Extract only Participant speech
            participant_speech = " ".join(df_t[df_t['speaker'] == 'Participant']['value'].astype(str).tolist())
            return participant_speech if participant_speech.strip() else "empty transcript"
        except Exception:
            return "empty transcript"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p_id = str(row['participant_id'])
        
        # 1. Text Processing
        transcript_path = os.path.join(self.base_dir, f"{p_id}_P", f"{p_id}_TRANSCRIPT.csv")
        text = self._load_transcript(transcript_path)
        
        encoded = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt'
        )

        # 2. Audio Processing
        audio_path = os.path.join(self.base_dir, f"{p_id}_P", f"{p_id}_AUDIO.wav")
        wav, _ = librosa.load(audio_path, sr=self.sr)
        wav_tensor = self._prepare_audio(wav)

        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'audio_values': wav_tensor,
            'label': label
        }

if __name__ == "__main__":
    # Ensure these paths are correct for your local machine
    label_path = 'labels.csv' 
    base_dir = r'/Users/gurusai/Desktop/DAIC_Raw'
    
    print("--- Initializing Dataset ---")
    try:
        # Note: Testing with a fixed 5s to save time, 
        # but the class will calculate real median if you pass None
        dataset = DepressionDataset(
            label_file=label_path,
            base_dir=base_dir,
            median_samples=16000 * 5 
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        print("\n--- Success! Sample Extracted ---")
        print(f"Input IDs shape:      {batch['input_ids'].shape}")
        print(f"Audio Values shape:   {batch['audio_values'].shape}")
        print(f"Label:                {batch['label']}")

    except Exception as e:
        print(f"\n[Error]: {e}")
        import traceback
        traceback.print_exc()