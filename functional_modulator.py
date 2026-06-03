import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalModulator(nn.Module):
    def __init__(self, d_model=256):
        super(FunctionalModulator, self).__init__()
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.ago = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, f_local, h_omics):
        # Encode multi-omics landscape features
        c_envs = self.context_encoder(h_omics)
        
        # Strict Eq. (12) Adaptive Gain Operator mapping to (1, +inf) boost domain
        gain_mask = 1.0 + F.softplus(self.ago(c_envs))
        
        # Functional element-wise modulation
        f_final = f_local * gain_mask
        return f_final
