import torch
import torch.nn as nn

class VisualETMModule(nn.Module):
    def __init__(self, visual_input_dim=388, hidden_dim=256, fusion_dim=768):
        super().__init__()
        # 1. Project raw OpenFace/COFA features to the model's hidden dimension
        self.visual_projection = nn.Linear(visual_input_dim, fusion_dim)
        
        # 2. GRU to track the temporal "slope" of facial expressions
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True 
        )
        
        # 3. Final projection to map back to the fusion space (768)
        self.trend_proj = nn.Linear(hidden_dim * 2, fusion_dim)

    def forward(self, video_features):
        """
        video_features: [Batch, Frames, 388] 
        (e.g., Action Units, Gaze, Pose from DAIC-WOZ)
        """
        # Project to 768
        v_feat = self.visual_projection(video_features) # [B, T, 768]
        
        # Track temporal dependencies (Visual fatigue/Flattening)
        h, _ = self.gru(v_feat) # [B, T, 512]
        
        # Global Mean Pooling to get the "Evolution Trend" (Eq. 7)
        v_etm_visual = torch.mean(h, dim=1) # [B, 512]
        
        return self.trend_proj(v_etm_visual) # [B, 768]