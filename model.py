import torch
import torch.nn as nn
from transformers import AlbertModel, Wav2Vec2Model
from semantic_gating import SemanticGating
from bias_tensor_attention import BiasGuidedTensorAttention
from emotion_trend_modeling import EmotionTrendModule

class DepressionPredictor(nn.Module):
    def __init__(self, hidden_dim=768): # Table 5 says 768
        super().__init__()
        # Use ALBERT-large-v2 as per the paper
        self.text_encoder = AlbertModel.from_pretrained("albert-large-v2") 
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Projection layer: ALBERT-large (1024) -> Hidden (768)
        self.text_proj = nn.Linear(1024, hidden_dim)
        
        self.sgcmg = SemanticGating(hidden_dim)
        self.bg_tpa = BiasGuidedTensorAttention(hidden_dim)
        self.etm = EmotionTrendModule(hidden_dim, hidden_dim) # Hidden size 256 for GRU? Table 5 says 256
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, wav):
        # 1. Text Encoding
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_out = self.text_proj(text_out) # [B, 128, 768]
        
        # 2. Audio Encoding 
        audio_out = self.audio_encoder(wav).last_hidden_state # [B, T, 768]

        # 3. SGCMG (Semantic Gating)
        # Paper says SGCMG uses text to gate audio channels
        audio_gated = self.sgcmg(audio_out, text_out) 

        # 4. BG-TPA (Bias Attention) - Requires BOTH modalities
        audio_aligned = self.bg_tpa(text_out, audio_gated) # [B, 128, 768]

        # 5. ETM (Emotion Trend)
        audio_trend = self.etm(audio_gated) # [B, 768]

        # 6. Fusion
        # Incorporate trend into the sequence
        fused_feat = audio_aligned + audio_trend.unsqueeze(1)
        
        # Cross Attention
        attn_output, _ = self.cross_attn(text_out, fused_feat, fused_feat)

        # 7. Output
        pooled = attn_output.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits