import torch
import torch.nn as nn
import torch.nn.functional as F

from eaanr_module import EAANR
from functional_modulator import FunctionalModulator

class OmniSVModel(nn.Module):
    def __init__(self, d_model=256, evo_dim=2048, omics_dim=14):
        super(OmniSVModel, self).__init__()
        
        # Dimensionality Alignment Projectors
        self.syn_proj = nn.Linear(512, d_model)  
        self.evo_proj = nn.Linear(evo_dim, d_model) 
        self.omics_proj = nn.Linear(omics_dim, d_model) 
        
        # Core Modality Synthesizers
        self.eaanr = EAANR(d_model=d_model, nhead=8)
        self.f_modulator = FunctionalModulator(d_model=d_model)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3) 
        )

    def forward(self, x_syn, x_evo_bundle, x_omics):
        # 1. Feature Decomposition
        l_evo = x_evo_bundle[:, :, :4]     
        h_evo_raw = x_evo_bundle[:, :, 4:] 
        
        # 2. Joint Space Projection (Map to d_model=256)
        h_syn = self.syn_proj(x_syn)
        h_evo = self.evo_proj(h_evo_raw)
        h_omics = self.omics_proj(x_omics) 
        
        # 3. Asymmetric Rectification
        f_local = self.eaanr(h_syn, h_evo, l_evo)
        
        # 4. Multi-omics Functional Modulation
        f_final = self.f_modulator(f_local, h_omics)
        
        # 5. Global Pooling
        feat_pool = torch.mean(f_final, dim=1)
        
        # 6. Task Prediction
        logits = self.classifier(feat_pool)
        
        return logits, f_local, h_evo, f_final, h_omics

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniSVModel(d_model=256).to(device)
    
    # 1024 sequence length test setup
    batch_size = 16
    syn_feat = torch.randn(batch_size, 1024, 512).to(device)
    evo_feat = torch.randn(batch_size, 1024, 2052).to(device)
    omics_feat = torch.randn(batch_size, 1024, 14).to(device)
    
    logits, f_local, h_evo, f_final, h_omics = model(syn_feat, evo_feat, omics_feat)
    print(f"Output Dimensions verified successfully: {logits.shape}")
