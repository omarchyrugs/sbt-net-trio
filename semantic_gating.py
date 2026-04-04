import torch
import torch.nn as nn

class SemanticGating(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # Linear layer to transform pooled text features into a gating vector
        self.text_to_gate = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_feat, text_hidden):
        """
        audio_feat: [Batch, La, 768]
        text_hidden: [Batch, Lt, 768]
        """
        # 1. Pool text to get a global semantic context (v_t)
        v_t = torch.mean(text_hidden, dim=1) # [Batch, 768]
        
        # 2. Generate the gate
        gate = self.sigmoid(self.text_to_gate(v_t)) # [Batch, 768]
        
        # 3. Apply gate to audio channels
        # Unsqueezing to [Batch, 1, 768] to broadcast across all audio frames (La)
        gated_audio = audio_feat * gate.unsqueeze(1)
        
        return gated_audio