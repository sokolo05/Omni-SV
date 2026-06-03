import torch
import torch.nn as nn
import torch.nn.functional as F

class EAANR(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super(EAANR, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.gate_net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_syn, h_evo, l_evo):
        # 1. Gating Factor Calculation from Base Prediction Logits
        prob = F.softmax(l_evo, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=-1, keepdim=True)
        
        # 2. Rectification Scaling (High entropy / uncertainty suppresses the gain factor)
        gain_factor = torch.sigmoid(self.gate_net(-entropy))
        
        # 3. Asymmetric Cross-Attention (Query from Evo, Key/Value from Syn)
        h_attended, _ = self.cross_attn(query=h_evo, key=h_syn, value=h_syn)
        
        # 4. Feature Modulation (Hadamard product)
        f_modulated = h_attended * gain_factor
        
        # 5. Residual Aggregation with Evolutionary Base
        f_local = self.norm(f_modulated + h_evo)
        return f_local
