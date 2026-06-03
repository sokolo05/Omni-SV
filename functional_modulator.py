import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalModulator(nn.Module):
    def __init__(self, d_model=256):
        super(FunctionalModulator, self).__init__()
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True
        )
        
        # Converted from nn.Sigmoid to match the (1, +inf) boost domain in Eq. (12)
        self.ago = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, f_local, h_omics):
        c_envs = self.context_encoder(h_omics)
        
        # Strict Eq. (12) implementation: Multiplier = 1.0 + Softplus(MLP(.))
        gain_mask = 1.0 + F.softplus(self.ago(c_envs))
        
        f_final = f_local * gain_mask
        return f_final
