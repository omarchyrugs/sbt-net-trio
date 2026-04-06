
from torch import nn
import torch


class ClinicalFusionHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # Cross-Attention to let Audio and Video "talk" to each other
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        
        # Final Classifier
        self.classifier = nn.Sequential(
                    nn.Linear(dim * 4, 1024), # Widened to capture Max+Avg relationships
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),          # High dropout to fight that memorization
                    nn.Linear(1024, 512),     # Bottleneck layer
                    nn.LayerNorm(512),
                    nn.ReLU(),
                    nn.Linear(512, 1)         # Final logit
                )

    def forward(self, audio_branch_out, video_branch_out, mask):
            # 1. Cross-Attention (Symmetry looks good)
            # Q = Audio, K/V = Video
            audio_refined, _ = self.cross_attn(audio_branch_out, video_branch_out, video_branch_out, 
                                            key_padding_mask=(mask == 0))
            
            # 2. Let Video look at Audio
            video_refined, _ = self.cross_attn(video_branch_out, audio_branch_out, audio_branch_out, 
                                            key_padding_mask=(mask == 0))
            
            # --- 3. Hybrid Pooling Logic (The "Guru" Upgrade) ---
            mask_unsq = mask.unsqueeze(-1) # [Batch, Segments, 1]
            
            # Prepare for Max-Pooling: Set masked areas to a very small number
            # so they don't get picked as the "Maximum" feature
            a_for_max = audio_refined.masked_fill(mask_unsq == 0, -1e9)
            v_for_max = video_refined.masked_fill(mask_unsq == 0, -1e9)

            # A. Max Pooling: Captures the most intense clinical spikes
            a_max = torch.max(a_for_max, dim=1)[0]
            v_max = torch.max(v_for_max, dim=1)[0]

            # B. Average Pooling: Captures the overall patient "vibe"
            a_avg = torch.sum(audio_refined * mask_unsq, dim=1) / torch.sum(mask_unsq, dim=1).clamp(min=1)
            v_avg = torch.sum(video_refined * mask_unsq, dim=1) / torch.sum(mask_unsq, dim=1).clamp(min=1)
            
            # 4. Final Cat + Classify
            # We now have [Max_Audio, Avg_Audio, Max_Video, Avg_Video]
            # This gives the classifier much more granular evidence to work with.
            combined = torch.cat([a_max, a_avg, v_max, v_avg], dim=-1) # [Batch, 1536 * 2]
            
            return self.classifier(combined)