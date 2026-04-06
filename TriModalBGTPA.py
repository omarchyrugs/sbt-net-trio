import torch
import torch.nn as nn
import torch.nn.functional as F

class TriModalBGTPA(nn.Module):
    def __init__(self, dim=768, heads=8, dropout=0.1):
        super(TriModalBGTPA, self).__init__()
        self.heads = heads
        self.d_k = dim // heads
        
        # 1. Text Projections (The Bias/Query)
        self.q_text = nn.Linear(dim, dim)
        
        # 2. Physical Signal Projections (Key and Value)
        # This will be used for Audio or Video depending on the branch
        self.k_phys = nn.Linear(dim, dim)
        self.v_phys = nn.Linear(dim, dim)
        
        # 3. Output Projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, text, physical_signal, mask):
        """
        Args:
            text: [Batch, 128, 768] (The Anchor)
            physical_signal: [Batch, 128, 768] (Audio or Visual)
            mask: [Batch, 128] (1 for real data, 0 for padding)
        """
        B, L, D = text.shape
        
        # --- STEP 1: Multi-Head Projection ---
        # Text becomes the Query
        query = self.q_text(text).view(B, L, self.heads, self.d_k).transpose(1, 2)
        # Physical signal becomes Key and Value
        key = self.k_phys(physical_signal).view(B, L, self.heads, self.d_k).transpose(1, 2)
        value = self.v_phys(physical_signal).view(B, L, self.heads, self.d_k).transpose(1, 2)

        # --- STEP 2: Scaled Dot-Product Attention ---
        # (Query @ Key^T) / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # --- STEP 3: Masking (Crucial for short turns) ---
        # mask: [B, 128] -> [B, 1, 1, 128]
        mask_expanded = mask.view(B, 1, 1, L)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # --- STEP 4: Context Vector calculation ---
        # (Weights @ Value)
        context = torch.matmul(attn_weights, value) # [B, heads, 128, d_k]
        
        # --- STEP 5: Final Projection & Residual ---
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(context)
        
        # Residual connection with the original text (The "Bias-Guided" part)
        return self.ln(out + text)