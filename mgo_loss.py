import torch
import torch.nn as nn
import torch.nn.functional as F

class MGOLoss(nn.Module):
    """
    Section 4.6: Multi-Granularity Optimization (MGO) Strategy.
    Equation: L_total = L_CE + lambda1 * L_cons + lambda2 * L_mine
    """
    def __init__(self, lambda1=0.5, lambda2=0.1, d_model=256):
        super(MGOLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda1 = lambda1  # Weight for Representation Consistency
        self.lambda2 = lambda2  # Weight for Mutual Information Maximization
        
        # Mutual Information Neural Estimator (MINE)
        # Used to measure and maximize the dependency between f_final and h_omics
        self.mi_estimator = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, pred, target, f_local, h_evo, f_final, h_omics):
        """
        pred: Model output [B, 3]
        target: Ground truth labels [B]
        f_local: Rectified syntactic features [B, L, D]
        h_evo: Evolutionary manifold from Evo2 [B, L, D]
        f_final: Globally modulated features [B, L, D]
        h_omics: Functional tracks from AlphaGenome [B, L, D]
        """
        
        # 1. Classification Loss (L_CE)
        # Supervises the basic categorization of SV types (DEL, INS, MATCH)
        l_ce = self.ce_loss(pred, target)
        
        # 2. Representation Consistency Loss (L_cons)
        # Equation 11: Forces rectified physical signals to align with evolutionary priors
        # Normalization is applied to ensure stability on the hypersphere
        f_norm = F.normalize(f_local, p=2, dim=-1)
        h_norm = F.normalize(h_evo, p=2, dim=-1)
        l_cons = F.mse_loss(f_norm, h_norm)
        
        # 3. Mutual Information Maximization (L_mine)
        # Ensures that f_final adequately absorbs the functional environmental context
        # Implemented via a simplified MINE objective
        joint_samples = torch.cat([f_final, h_omics], dim=-1) # Joint distribution
        mi_score = torch.mean(self.mi_estimator(joint_samples))
        
        # Negate MI score because we want to maximize it
        l_mine = -mi_score
        
        # Total Joint Optimization
        total_loss = l_ce + self.lambda1 * l_cons + self.lambda2 * l_mine
        
        return total_loss, l_ce, l_cons, l_mine