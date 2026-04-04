import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasGuidedTensorAttention(nn.Module):
    def __init__(self, d_model=768, rank=64):
        super().__init__()
        self.rank = rank
        
        # Low-rank decomposition (U and V) to represent the Bias Matrix
        self.U = nn.Linear(d_model, rank, bias=False)
        self.V = nn.Linear(d_model, rank, bias=False)
        
        # Final projection for the attended values
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, text_feat, audio_feat):
        """
        text_feat: [Batch, Lt, 768]
        audio_feat: [Batch, La, 768]
        """
        # 1. Project to low-rank space
        q_low = self.U(text_feat) # [B, Lt, r]
        k_low = self.V(audio_feat) # [B, La, r]
        
        # 2. Compute Bilinear Scores (Tensor Product)
        # Resulting shape: [Batch, Lt, La]
        scores = torch.matmul(q_low, k_low.transpose(-1, -2))
        
        # 3. Attention Weights
        weights = F.softmax(scores, dim=-1)
        
        # 4. Apply weights to Audio Values
        v = self.v_proj(audio_feat)
        output = torch.matmul(weights, v) # [Batch, Lt, 768]
        
        return output