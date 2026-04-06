
from torch import nn
import torch


class ClinicalFusionHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # Cross-Attention to let Audio and Video "talk" to each other
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 512), # Concatenating the two specialized branches
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, audio_branch_out, video_branch_out, mask):
        # 1. Let Audio look at Video
        # Q = Audio, K/V = Video
        audio_refined, _ = self.cross_attn(audio_branch_out, video_branch_out, video_branch_out, 
                                           key_padding_mask=(mask == 0))
        
        # 2. Let Video look at Audio
        video_refined, _ = self.cross_attn(video_branch_out, audio_branch_out, audio_branch_out, 
                                           key_padding_mask=(mask == 0))
        
        # 3. Global Pooling (Ignoring padding)
        mask_unsq = mask.unsqueeze(-1)
        a_pooled = torch.sum(audio_refined * mask_unsq, dim=1) / torch.sum(mask_unsq, dim=1).clamp(min=1)
        v_pooled = torch.sum(video_refined * mask_unsq, dim=1) / torch.sum(mask_unsq, dim=1).clamp(min=1)
        
        # 4. Final Cat + Classify
        combined = torch.cat([a_pooled, v_pooled], dim=-1) # [Batch, 1536]
        return self.classifier(combined)