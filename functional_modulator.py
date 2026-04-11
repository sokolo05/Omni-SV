import torch
import torch.nn as nn

class FunctionalModulator(nn.Module):
    """
    Section 4.5: Multi-omics Functional Landscape Modulation.
    Core: Applies Adaptive Gain Operator (AGO) based on 
    genomic context (e.g., ATAC, ChIP-seq tracks).
    """
    def __init__(self, d_model=256):
        super(FunctionalModulator, self).__init__()
        # Context Awareness: Perceive global functional environment
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True
        )
        
        # Adaptive Gain Operator (AGO)
        self.ago = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid()
        )

    def forward(self, f_local, h_omics):
        # Encode functional tracks (AlphaGenome features)
        c_envs = self.context_encoder(h_omics)
        
        # Generate Gain Mask based on biological environment
        gain_mask = self.ago(c_envs)
        
        # Element-wise modulation (Eq. 9)
        f_final = f_local * gain_mask
        return f_final