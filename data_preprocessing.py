import torch.nn.functional as F
import torch
import librosa
from transformers import AlbertModel, AlbertTokenizer, Wav2Vec2Model, Wav2Vec2Processor, BertModel, BertTokenizer
from torch import  nn
import pandas as pd
import os
import soundfile as sf
import glob
import numpy as np
import argparse
TARGET_LENGTH = 128  # Target audio sequence length after pooling
import mail

class MultimodalDataProcessor:
    """
    End-to-end pipeline for processing DAIC-WOZ multimodal data.
    Handles transcript stitching, audio feature extraction, and multimodal packet generation.
    """
    
    def __init__(self, device=None, target_sr=16000, chunk_duration=30, target_packet_length=128,labels_path ='./labels.csv'):
        """
        Initialize the processor with pre-loaded models and device.
        
        Args:
            device: torch device (auto-detected if None)
            target_sr: sample rate for audio (default 16000)
            chunk_duration: process audio in chunks of N seconds
            target_packet_length: target sequence length for pooled embeddings (default 128)
        """
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self.target_packet_length = target_packet_length

        self.labels = pd.read_csv(labels_path)
        print(f"[Processor] Initializing on device: {self.device}")
        
        # Text encoders
        self.text_tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")
        self.text_model = AlbertModel.from_pretrained("albert-large-v2").to(self.device).eval()
        self.text_proj = nn.Linear(1024, 768).to(self.device)  # Project ALBERT's 1024-dim to 768 for fusion
        self.context_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.context_model = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()
        
        # Audio encoder
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device).eval()
        
        # Visual projection
        self.visual_proj = nn.Linear(242, 768).to(self.device)
        
        print("[Processor] ✓ All models loaded")
    
    def _normalize_visual_df(self, df, prefix):
        """Normalize OpenFace dataframe columns."""
        df = df.rename(columns={c: c.strip() for c in df.columns}).copy()
        required_keys = ["frame", "timestamp"]
        missing = [c for c in required_keys if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {prefix} file")
        
        drop_cols = [c for c in ["confidence", "success"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        
        feature_cols = [c for c in df.columns if c not in required_keys]
        rename_map = {c: f"{prefix}_{c}" for c in feature_cols}
        return df.rename(columns=rename_map)
    
    def align_and_stitch(self, p_id, p_folder, output_dir="data"):
        """
        Step 1: Align transcript turns with audio, OpenFace, and stitch into participant-only stream.
        
        Returns:
            aligned_data: list of turn dicts with timestamps and features
        """
        print(f"\n[{p_id}] Step 1: Aligning and stitching...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load transcript
        df_t = pd.read_csv(os.path.join(p_folder, f"{p_id}_TRANSCRIPT.csv"), sep='\t')
        
        # Load OpenFace streams
        df_clnf = pd.read_csv(os.path.join(p_folder, f"{p_id}_CLNF_features3D.txt"))
        df_gaze = pd.read_csv(os.path.join(p_folder, f"{p_id}_CLNF_gaze.txt"))
        df_pose = pd.read_csv(os.path.join(p_folder, f"{p_id}_CLNF_pose.txt"))
        df_au = pd.read_csv(os.path.join(p_folder, f"{p_id}_CLNF_AUs.txt"))
        
        # Normalize and merge visual data
        df_clnf = self._normalize_visual_df(df_clnf, "clnf")
        df_gaze = self._normalize_visual_df(df_gaze, "gaze")
        df_pose = self._normalize_visual_df(df_pose, "pose")
        df_au = self._normalize_visual_df(df_au, "au")
        
        df_visual = (
            df_clnf
            .merge(df_gaze, on=["frame", "timestamp"], how="inner")
            .merge(df_pose, on=["frame", "timestamp"], how="inner")
            .merge(df_au, on=["frame", "timestamp"], how="inner")
        )
        
        # Load full audio once
        raw_audio_path = os.path.join(p_folder, f"{p_id}_AUDIO.wav")
        y, sr = librosa.load(raw_audio_path, sr=self.target_sr)
        
        processed_turns = []
        stitched_audio = []
        cumulative_time = 0.0
        last_question = "Initial Greeting"
        
        stop_col = "stop_time" if "stop_time" in df_t.columns else "end_time"
        
        for _, row in df_t.iterrows():
            text = str(row["value"]).lower()
            
            # Update context on interviewer turns
            if row["speaker"] == "Ellie":
                if "?" in text or any(q in text for q in ["how", "why", "tell me", "describe", "do you", "did you", "can you", "when", "have you"]):
                    if len(text.split()) > 3:
                        last_question = text
                continue
            
            # Process participant turns
            if row["speaker"] == "Participant":
                start, stop = row["start_time"], row[stop_col]
                
                visual_mask = (df_visual["timestamp"] >= start) & (df_visual["timestamp"] <= stop)
                turn_visual_data = df_visual[visual_mask]
                
                duration = stop - start
                new_start = cumulative_time
                new_stop = cumulative_time + duration
                
                start_sample = int(start * self.target_sr)
                stop_sample = int(stop * self.target_sr)
                segment = y[start_sample:stop_sample]
                stitched_audio.extend(segment)
                
                if not turn_visual_data.empty:
                    processed_turns.append({
                        "participant_id": p_id,
                        "question_context": last_question,
                        "answer_text": text,
                        "start_time": start,
                        "stop_time": stop,
                        "visual_features": turn_visual_data.drop(columns=["frame", "timestamp"]).values,
                        "stitched_audio_start": new_start,
                        "stitched_audio_stop": new_stop,
                    })
                
                cumulative_time += duration
        
        # Save stitched audio
        stitched_path = os.path.join(output_dir, f"{p_id}_CLEAN_AUDIO.wav")
        sf.write(stitched_path, stitched_audio, self.target_sr)
        
        print(f"  ✓ Stitched {len(processed_turns)} turns from {len(stitched_audio)} audio samples")
        print(f"  ✓ Saved: {stitched_path}")
        
        return processed_turns, stitched_path
    
    def extract_audio_features(self, stitched_audio_path):
        """
        Step 2: Extract Wav2Vec2 features from stitched audio in chunks.
        
        Returns:
            audio_features: tensor [1, Total_Frames, 768]
        """
        print(f"[Audio] Extracting features (chunk_duration={self.chunk_duration}s)...")
        
        y, sr = librosa.load(stitched_audio_path, sr=self.target_sr)
        chunk_samples = int(self.chunk_duration * self.target_sr)
        
        all_features = []
        chunk_count = 0
        
        for start_idx in range(0, len(y), chunk_samples):
            end_idx = min(start_idx + chunk_samples, len(y))
            chunk = y[start_idx:end_idx]
            
            input_values = self.audio_processor(chunk, return_tensors="pt", sampling_rate=self.target_sr).input_values.to(self.device)
            
            with torch.no_grad():
                chunk_features = self.audio_model(input_values).last_hidden_state
            
            all_features.append(chunk_features)
            chunk_count += 1
        
        audio_features = torch.cat(all_features, dim=1)
        print(f"  ✓ Processed {chunk_count} chunks → shape {audio_features.shape}")
        
        return audio_features
    
    def build_multimodal_packets(self, aligned_data, audio_features):
        """
        Step 3: Build multimodal packets with all 4 modalities aligned.
        
        Returns:
            packets: list of dicts with answer, context, audio, visual embeddings
        """
        print(f"[Packets] Building {len(aligned_data)} multimodal packets...")
        
        packets = []
        total_feature_frames = audio_features.shape[1]
        total_audio_samples = int(aligned_data[-1]["stitched_audio_stop"] * self.target_sr)
        samples_per_frame = total_audio_samples / total_feature_frames
        
        for i, turn in enumerate(aligned_data):
            # A. TEXT & CONTEXT ENCODING
            answer_text = turn["answer_text"]
            answer_tokens = self.text_tokenizer(
                answer_text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            answer_ids = answer_tokens['input_ids'].to(self.device)
            answer_mask = answer_tokens['attention_mask'].to(self.device)
            
            with torch.no_grad():
                answer_embed = self.text_model(answer_ids, answer_mask).last_hidden_state
            answer_embed = self.text_proj(answer_embed)  # Project to 768-dim
            # Context: BERT [CLS] token
            context_text = turn["question_context"]
            context_tokens = self.context_tokenizer(
                context_text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            context_ids = context_tokens['input_ids'].to(self.device)
            context_mask = context_tokens['attention_mask'].to(self.device)
            
            with torch.no_grad():
                context_output = self.context_model(context_ids, context_mask).last_hidden_state
                context_embed = context_output[:, 0, :]
            
            # B. AUDIO SLICING & POOLING
            start_frame = int(round((turn["stitched_audio_start"] * self.target_sr) / samples_per_frame))
            stop_frame = int(round((turn["stitched_audio_stop"] * self.target_sr) / samples_per_frame))
            
            if stop_frame <= start_frame:
                stop_frame = start_frame + 1
            stop_frame = min(stop_frame, total_feature_frames)
            
            turn_audio = audio_features[:, start_frame:stop_frame, :]
            turn_audio_cpu = turn_audio.cpu()
            turn_audio_t = turn_audio_cpu.transpose(1, 2)
            audio_pooled = F.adaptive_avg_pool1d(turn_audio_t, self.target_packet_length)
            audio_embed = audio_pooled.transpose(1, 2)
            
            # C. VISUAL FEATURES
            if len(turn['visual_features']) == 0:
                vis_embed = torch.zeros((1, self.target_packet_length, 768))
            else:
                raw_visual = torch.from_numpy(turn['visual_features']).float().to(self.device)
                with torch.no_grad():
                    vis_projected = self.visual_proj(raw_visual).unsqueeze(0)
                    vis_projected_cpu = vis_projected.cpu()
                    vis_pooled = F.adaptive_avg_pool1d(vis_projected_cpu.transpose(1, 2), self.target_packet_length)
                    vis_embed = vis_pooled.transpose(1, 2)
            
            packet = {
                'participant_id': turn['participant_id'],
                'turn_id': i,
                'answer_text': answer_text,
                'context_text': context_text,
                'answer_embed': answer_embed.squeeze(0).cpu(),
                'context_embed': context_embed.squeeze(0).cpu(),
                'audio_embed': audio_embed.squeeze(0),
                'visual_embed': vis_embed.squeeze(0),
                'mask': answer_mask.squeeze(0).cpu(),
                'label': self.labels[self.labels['Participant_ID'] == int(turn['participant_id'])]['depressed'].values[0]
            }
            packets.append(packet)
        
        print(f"  ✓ Built {len(packets)} packets")
        return packets

    def pack_packets(self, packets):
        """
        Convert list-of-dicts packets into tensor-major storage for fast save/load.

        Returns:
            dict with stacked tensors and lightweight metadata lists.
        """
        if not packets:
            raise ValueError("Cannot pack empty packet list")

        packed = {
            "answer_embed": torch.stack([p["answer_embed"] for p in packets]),
            "context_embed": torch.stack([p["context_embed"] for p in packets]),
            "audio_embed": torch.stack([p["audio_embed"] for p in packets]),
            "visual_embed": torch.stack([p["visual_embed"] for p in packets]),
            "mask": torch.stack([p["mask"] for p in packets]),
            "label": torch.tensor([int(p["label"]) for p in packets], dtype=torch.long),
            "participant_id": [p["participant_id"] for p in packets],
            "turn_id": torch.tensor([int(p["turn_id"]) for p in packets], dtype=torch.long),
            "answer_text": [p["answer_text"] for p in packets],
            "context_text": [p["context_text"] for p in packets],
        }
        return packed

    def save_packets(self, packets, out_path):
        """Save packets to a single .pt file for fast retrieval."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        packed = self.pack_packets(packets)
        torch.save(packed, out_path)
        return out_path

    def load_packets(self, path, map_location="cpu"):
        """
        Load packed packets from .pt file.

        Uses mmap when available for faster startup on large files.
        """
        try:
            return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
        except TypeError:
            # Fallback for older torch versions without mmap/weights_only.
            return torch.load(path, map_location=map_location)
    
    def process_participant(self, p_id, p_folder, output_dir="data", save_packets_to_disk=True):
        """
        Full pipeline: align → extract audio → build packets.
        
        Returns:
            packets: multimodal data ready for training
        """
        print(f"\n{'='*60}")
        print(f"Processing Participant: {p_id}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Align and stitch
            aligned_data, stitched_path = self.align_and_stitch(p_id, p_folder, output_dir)
            
            # Step 2: Extract audio features
            audio_features = self.extract_audio_features(stitched_path)
            
            # Step 3: Build packets
            packets = self.build_multimodal_packets(aligned_data, audio_features)

            if save_packets_to_disk:
                packet_path = os.path.join(output_dir, "packets", f"{p_id}_packets.pt")
                self.save_packets(packets, packet_path)
                print(f"  ✓ Saved packets: {packet_path}")
            
            print(f"{'='*60}")
            print(f"✓ {p_id} complete: {len(packets)} packets ready\n")
            
            return packets
        
        except Exception as e:
            print(f"✗ Error processing {p_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_batch(self, participant_list, base_dir, output_dir="data"):
        """
        Process multiple participants in sequence.
        Args:
            participant_list: list of participant IDs
            base_dir: base directory containing {P_ID}_P folders
            output_dir: where to save outputs
        
        Returns:
            results: dict mapping p_id → packets
        """
        results = {}
        
        for p_id in participant_list:
            p_folder = os.path.join(base_dir, f"{p_id}_P")
            packets = self.process_participant(p_id, p_folder, output_dir)
            
            if packets is not None:
                results[p_id] = packets
        
        return results


import traceback

# ... [Your existing imports and MultimodalDataProcessor class] ...

def validate_outputs_internal(output_dir):
    """
    Internal helper to check if the generated .pt files are healthy.
    Returns (is_valid, message)
    """
    packet_path = os.path.join(output_dir, "packets", "*.pt")
    pt_files = glob.glob(packet_path)
    
    if not pt_files:
        return False, "No .pt files were found in the output directory."

    errors = []
    for f_path in pt_files[:10]: # Sample check the first 10 for speed
        try:
            data = torch.load(f_path, map_location="cpu")
            # Verify the core 768-dim symmetry we worked for
            if data['answer_embed'].shape[-1] != 768:
                errors.append(f"{os.path.basename(f_path)} has wrong dimensions.")
            if torch.isnan(data['answer_embed']).any():
                errors.append(f"{os.path.basename(f_path)} contains NaNs.")
        except Exception as e:
            errors.append(f"Could not load {os.path.basename(f_path)}: {e}")

    if errors:
        return False, "<br>".join(errors[:5])
    return True, f"Verified {len(pt_files)} files. Shapes and NaNs look clean."

def run_monitored_job(processor, participant_ids, base_dir, output_dir):
    start_time = mail.get_IST()
    job_subject = f"🚀 DAIC-WOZ Job Started: {start_time}"
    job_body = (
        f"<b>Status:</b> Extraction and Encoding Started<br>"
        f"<b>Participants:</b> {len(participant_ids)}<br>"
        f"<b>Target:</b> {output_dir}/packets"
    )
    
    # 1. Alert: Job Start
    mail.send_email_alert(job_subject, job_body)

    try:
        # 2. The Actual Processing
        results = processor.process_batch(participant_ids, base_dir, output_dir)
        
        # 3. Internal Validation Check
        is_valid, validation_msg = validate_outputs_internal(output_dir)
        
        # 4. Alert: Success / Partial Success
        end_time = mail.get_IST()
        if is_valid:
            success_subject = f"✅ DAIC-WOZ Job Complete: {end_time}"
            status_icon = "🟢"
        else:
            success_subject = f"⚠️ DAIC-WOZ Job Finished with VALIDATION ERRORS: {end_time}"
            status_icon = "🟠"

        success_body = (
            f"{status_icon} <b>Processing Summary:</b><br>"
            f"Participants Processed: {len(results)}<br>"
            f"Validation Results: {validation_msg}<br><br>"
            f"Ready for Phase 2: Model Training."
        )
        mail.send_email_alert(success_subject, success_body)

    except Exception as e:
        # 5. Alert: Critical Failure (Script Crash)
        error_time = mail.get_IST()
        # Formatting stack trace for HTML email
        error_stack = traceback.format_exc().replace("\n", "<br>").replace(" ", "&nbsp;")
        
        fail_subject = f"❌ DAIC-WOZ Job CRASHED: {error_time}"
        fail_body = (
            f"<b>Error Type:</b> {type(e).__name__}<br>"
            f"<b>Message:</b> {str(e)}<br><br>"
            f"<b>Full Traceback:</b><br><font face='monospace' size='2'>{error_stack}</font>"
        )
        mail.send_email_alert(fail_subject, fail_body)
        raise e  # Re-raise after alerting
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal DAIC-WOZ preprocessing pipeline")
    parser.add_argument("--base-dir", type=str, required=True, help="Path to DAIC root directory containing <PID>_P folders")
    parser.add_argument("--labels-csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save stitched audio and packet files")
    parser.add_argument("--participant-id", type=int, default=None, help="Single participant id to process, e.g. 301")
    parser.add_argument("--chunk-duration", type=int, default=30, help="Audio chunk duration (seconds)")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target audio sample rate")
    parser.add_argument("--target-packet-length", type=int, default=128, help="Target pooled sequence length")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    processor = MultimodalDataProcessor(
        device=device,
        target_sr=args.target_sr,
        chunk_duration=args.chunk_duration,
        target_packet_length=args.target_packet_length,
        labels_path=args.labels_csv,
    )
    run_monitored_job(processor,processor.labels['Participant_ID'].values if args.participant_id is None else [args.participant_id], args.base_dir, args.output_dir)

        
    """
    output:
            packet = {
                'participant_id': turn['participant_id'],
                'turn_id': i,
                'answer_text': answer_text,
                'context_text': context_text,
                'answer_embed': answer_embed.squeeze(0).cpu(),
                'context_embed': context_embed.squeeze(0).cpu(),
                'audio_embed': audio_embed.squeeze(0),
                'visual_embed': vis_embed.squeeze(0),
                'mask': answer_mask.squeeze(0).cpu()
            }

        n such packets for each participant, where:
        - answer_embed: [128, 1024] (ALBERT large hidden size)
        - context_embed: [768] (BERT base CLS token)
        - audio_embed: [128, 768] (Wav2Vec2 pooled)
        - visual_embed: [128, 768] (projected OpenFace features)

    """