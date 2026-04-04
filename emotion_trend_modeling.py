import torch
import torch.nn as nn

class EmotionTrendModule(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        # Table 5 specifies a GRU-based ETM
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True # Bidirectional to capture full context of the "trend"
        )
        # Map 512 (256*2) back to the fusion dimension 768
        self.trend_proj = nn.Linear(hidden_dim * 2, 768)

    def forward(self, audio_hidden):
        """
        audio_hidden: [Batch, La, 768]
        """
        # 1. Sequence modeling
        # h: [Batch, La, 512]
        h, _ = self.gru(audio_hidden)
        
        # 2. Mean pooling to get the "Global Trend" (Eq. 7 in paper)
        v_etm = torch.mean(h, dim=1) # [Batch, 512]
        
        # 3. Project to match modality dimensions
        return self.trend_proj(v_etm) # [Batch, 768]