
from torch import nn

from VisualBranch import VisualBranch
from semantic_gating import SemanticGatingModule
from TriModalBGTPA import TriModalBGTPA
from VisualETM import VisualETM
from FusionHead import ClinicalFusionHead


class SBTNetTrio(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.vis_dyn = VisualBranch(dim, dim)
        self.a_gater = SemanticGatingModule(dim)
        self.v_gater = SemanticGatingModule(dim)
        
        self.a_tpa = TriModalBGTPA(dim)
        self.v_tpa = TriModalBGTPA(dim)
        
        self.a_etm = VisualETM(dim)
        self.v_etm = VisualETM(dim)
        
        self.fusion_head = ClinicalFusionHead(dim)

    def forward(self, batch):
        t, a, v = batch['text'], batch['audio'], batch['visual']
        c, m = batch['context'], batch['mask']
        
        # Video Stream
        v_feat = self.vis_dyn(v)
        v_gated = self.v_gater(t, c, v_feat)
        v_stream = self.v_tpa(t, v_gated, m)
        v_stream = self.v_etm(v_stream)
        
        # Audio Stream
        a_gated = self.a_gater(t, c, a)
        a_stream = self.a_tpa(t, a_gated, m)
        a_stream = self.a_etm(a_stream)
        
        return self.fusion_head(a_stream, v_stream, m).squeeze(-1)