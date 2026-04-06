import torch
from torch import nn

class SemanticGatingModule(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.gate_generator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, text, context, physical_signal, **kwargs):
        # Expand context [B, 768] to match sequence [B, 128, 768]
        c_expanded = context.unsqueeze(1).expand(-1, 128, -1)
        gate_input = torch.cat([text, c_expanded], dim=-1)
        gate = self.gate_generator(gate_input)
        
        gated = physical_signal * gate
        return self.ln(self.proj(gated) + physical_signal)