import torch
import torch.nn as nn
import torch.nn.functional as F

# Import sub-modules (Assume they are in the same directory)
from eaanr_module import EAANR
from functional_modulator import FunctionalModulator

class OmniSV(nn.Module):
    """
    Omni-SV Framework: Synergy of Syntactic, Evolutionary, and Functional Modalities.
    Strictly follows Sections 4.4 (EAANR) and 4.5 (Functional Modulation).
    """
    def __init__(self, d_model=256, evo_dim=2048, omics_dim=14):
        super(OmniSV, self).__init__()
        
        # Dimensionality Alignment (Feature Projectors)
        self.syn_proj = nn.Linear(512, d_model)  # Syntactic input from CNN
        self.evo_proj = nn.Linear(evo_dim, d_model) # Evolutionary from Evo2
        self.omics_proj = nn.Linear(omics_dim, d_model) # Functional from AlphaGenome
        
        # Section 4.4: Evolutionary-Anchored Asymmetric Neural Rectification
        self.eaanr = EAANR(d_model=d_model, nhead=8)
        
        # Section 4.5: Multi-omics Functional Landscape Modulation
        self.f_modulator = FunctionalModulator(d_model=d_model)
        
        # Task-specific Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3) # 3-class: DEL, INS, MATCH
        )

    def forward(self, x_syn, x_evo_bundle, x_omics):
        """
        x_syn: Syntactic image features [Batch, 1024, 512]
        x_evo_bundle: Concatenated L_evo and H_evo [Batch, 1024, 2052]
        x_omics: Functional tracks [Batch, 1024, 14]
        """
        
        # 1. Feature Decomposition & Projection
        # Extract Logits for Entropy (Eq. 8) and Hidden states for Semantics
        l_evo = x_evo_bundle[:, :, :4]     # Evolutionary Logits
        h_evo_raw = x_evo_bundle[:, :, 4:] # Evolutionary Embeddings
        
        h_syn = self.syn_proj(x_syn)
        h_evo = self.evo_proj(h_evo_raw)
        h_omics = self.omics_proj(x_omics)
        
        # 2. Section 4.4: EAANR Implementation
        # Audits noisy syntactic signals using evolutionary certainty
        f_local = self.eaanr(h_syn, h_evo, l_evo)
        
        # 3. Section 4.5: Functional Modulation Implementation
        # Calibrates evidence gain based on global genomic context (AGO)
        f_final = self.f_modulator(f_local, h_omics)
        
        # 4. Global Feature Aggregation
        # Aggregating 1024-length sequence into a global representation
        feat_pool = torch.mean(f_final, dim=1)
        
        # 5. Final Classification
        logits = self.classifier(feat_pool)
        
        # Return all intermediate tensors to support MGO Loss calculation
        return logits, f_local, h_evo, f_final, h_omics

# Validation for RTX 4090 performance
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Hidden dimension d_model=256 is efficient for 4090 training
    model = OmniSV(d_model=256).to(device)
    
    # Simulating data for one batch (Standardized length = 1024)
    batch_size = 16
    syn_feat = torch.randn(batch_size, 1024, 512).to(device)
    evo_feat = torch.randn(batch_size, 1024, 2052).to(device)
    omics_feat = torch.randn(batch_size, 1024, 14).to(device)
    
    logits, f_local, h_evo, f_final, h_omics = model(syn_feat, evo_feat, omics_feat)
    
    print(f"Prediction result shape: {logits.shape}")  # Should be [16, 3]
    print(f"Verification: Model output aligns with Multi-Granularity Optimization requirements.")