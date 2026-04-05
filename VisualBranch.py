import torch
from torch import nn

class VisualBranch(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, num_layers=2):
        super(VisualBranch, self).__init__()
        
        # We use hidden_dim // 2 because it's bidirectional (2 directions = 768 total)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # LayerNorm helps stabilize the high-dimensional hidden states
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [Batch, 128, 768] (Projected OpenFace features)
        Returns:
            out: [Batch, 128, 768] (Dynamic facial features)
        """
        # gru_out shape: [Batch, 128, 768]
        # we ignore the hidden state 'h' for now
        gru_out, _ = self.gru(x)
        
        # Residual Connection: Add the original frames back to the motion vectors
        # This ensures the model doesn't "forget" the static pose while learning the motion.
        combined = gru_out + x 
        
        return self.ln(combined)