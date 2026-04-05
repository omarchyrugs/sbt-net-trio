
from torch import nn


class VisualETM(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # Kernel size 3 looks at 3-frame "windows" to find micro-trends
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [Batch, 128, 768]
        # Conv1d expects [Batch, Channels, Length]
        x_t = x.transpose(1, 2) 
        
        trend = self.conv(x_t)
        trend = self.activation(trend).transpose(1, 2)
        
        return self.ln(trend + x)