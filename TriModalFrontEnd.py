
from torch import nn
from VisualBranch import VisualBranch
from VisualETM import VisualETM
from semantic_gating import SemanticGating

class TriModalFrontEnd(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # 2.1 & 2.2: The Visual Powerhouse
        self.visual_dynamics = VisualBranch(dim, dim)
        self.visual_trend = VisualETM(dim)
        
        # 2.2: The Semantic Gaters
        self.audio_gater = SemanticGating(dim)
        self.visual_gater = SemanticGating(dim)

    def forward(self, batch):
        text = batch['text']       # [B, 128, 768]
        audio = batch['audio']     # [B, 128, 768]
        visual = batch['visual']   # [B, 128, 768]
        context = batch['context'] # [B, 768]
        
        # 1. Process Visual branch (Dynamics -> Trend)
        v_feat = self.visual_dynamics(visual)
        v_feat = self.visual_trend(v_feat)
        
        # 2. Apply Semantic Gating (Text + Context filters Physicals)
        gated_audio = self.audio_gater(text, context, audio)
        gated_visual = self.visual_gater(text, context, v_feat)
        
        return text, gated_audio, gated_visual