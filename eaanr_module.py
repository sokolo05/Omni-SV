import torch
import torch.nn as nn
import torch.nn.functional as F

class EAANR(nn.Module):
    """
    Section 4.4: Evolutionary-Anchored Asymmetric Neural Rectification.
    Core: Uses Evo2 semantics (H_evo) and entropy (L_evo) to rectify 
    noisy syntactic features (H_syn) from CIGAR images.
    """
    def __init__(self, d_model=256, nhead=8):
        super(EAANR, self).__init__()
        # 4.4.1 Asymmetric Cross-Attention (ECA)
        # Query: Syntactic (noisy), Key/Value: Evolutionary (prior)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 4.4.2 Entropy Gated Rectification (EGR)
        # Calculates gain factor based on biological uncertainty (Entropy)
        self.gate_net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_syn, h_evo, l_evo):
        # Calculate Shannon Entropy from Logits (L_evo)
        # Eq. 8: H_i = -sum(p * log p)
        prob = F.softmax(l_evo, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=-1, keepdim=True)
        
        # Generate Adaptive Gain Factor
        gain_factor = self.gate_net(entropy)
        
        # Asymmetric Attention Mapping
        # Resynchronizes physical signals with evolutionary logic
        attn_out, _ = self.cross_attn(query=h_syn, key=h_evo, value=h_evo)
        
        # Rectification (Residual connection weighted by EGR)
        f_local = self.norm(h_syn + gain_factor * attn_out)
        return f_local