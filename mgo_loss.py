import torch
import torch.nn as nn
import torch.nn.functional as F

class MGOLoss(nn.Module):
    def __init__(self, lambda1=0.5, lambda2=0.1, d_model=256):
        super(MGOLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Learnable scaling constraints optimized via backpropagation
        self.lambda1 = nn.Parameter(torch.tensor(lambda1, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(lambda2, dtype=torch.float32))
        
        self.mi_estimator = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, pred, target, f_local, h_evo, f_final, h_omics):
        # 1. Classification Target
        l_ce = self.ce_loss(pred, target)
        
        # 2. Representation Consistency Target
        f_norm = F.normalize(f_local, p=2, dim=-1)
        h_norm = F.normalize(h_evo, p=2, dim=-1)
        l_cons = F.text = F.mse_loss(f_norm, h_norm)
        
        # 3. Complete Mutual Information Estimation (Donsker-Varadhan Bound)
        # Joint Distribution Processing
        joint_samples = torch.cat([f_final, h_omics], dim=-1)
        t_joint = self.mi_estimator(joint_samples)
        
        # Marginal Distribution Processing via Batch Shuffling
        batch_size = h_omics.size(0)
        shuffled_idx = torch.randperm(batch_size, device=h_omics.device)
        h_omics_marginal = h_omics[shuffled_idx]
        
        marginal_samples = torch.cat([f_final, h_omics_marginal], dim=-1)
        t_marginal = self.mi_estimator(marginal_samples)
        
        # Donsker-Varadhan implementation using stable logsumexp
        mi_estimate = torch.mean(t_joint) - torch.logsumexp(t_marginal, dim=0) + torch.log(torch.tensor(batch_size, dtype=torch.float32, device=h_omics.device))
        l_mine = -mi_estimate
        
        # Multi-Granularity Joint Target Optimization
        total_loss = l_ce + self.lambda1 * l_cons + self.lambda2 * l_mine
        
        return total_loss, l_ce, l_cons, l_mine
