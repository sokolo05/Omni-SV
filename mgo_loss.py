import torch
import torch.nn as nn
import torch.nn.functional as F

class MGOLoss(nn.Module):
    def __init__(self, lambda1=0.5, lambda2=0.1, d_model=256):
        super(MGOLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Fixed (c): Registered as nn.Parameter to ensure they are learnable via backpropagation
        self.lambda1 = nn.Parameter(torch.tensor(lambda1, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(lambda2, dtype=torch.float32))
        
        self.mi_estimator = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, pred, target, f_local, h_evo, f_final, h_omics):
        # 1. Classification Loss (L_CE)
        l_ce = self.ce_loss(pred, target)
        
        # 2. Representation Consistency Loss (L_cons)
        f_norm = F.normalize(f_local, p=2, dim=-1)
        h_norm = F.normalize(h_evo, p=2, dim=-1)
        l_cons = F.mse_loss(f_norm, h_norm)
        
        # 3. Fixed (d): Mutual Information Maximization (L_mine) via Donsker-Varadhan representation
        # Joint samples (paired combinations)
        joint_samples = torch.cat([f_final, h_omics], dim=-1)
        t_joint = self.mi_estimator(joint_samples)
        
        # Marginal samples: shuffle the batch dimension of h_omics to break the alignment
        batch_size = h_omics.size(0)
        shuffled_idx = torch.randperm(batch_size, device=h_omics.device)
        h_omics_marginal = h_omics[shuffled_idx]
        
        marginal_samples = torch.cat([f_final, h_omics_marginal], dim=-1)
        t_marginal = self.mi_estimator(marginal_samples)
        
        # Eq. (16) Donsker-Varadhan formula: MI = E[T_joint] - log(E[exp(T_marginal)])
        # We use logsumexp for mathematical stability against overflow
        mi_estimate = torch.mean(t_joint) - torch.logsumexp(t_marginal, dim=0) + torch.log(torch.tensor(batch_size, dtype=torch.float32, device=h_omics.device))
        
        # Negate to maximize mutual information
        l_mine = -mi_estimate
        
        # Total Joint Optimization with learnable scaling factors
        total_loss = l_ce + self.lambda1 * l_cons + self.lambda2 * l_mine
        
        return total_loss, l_ce, l_cons, l_mine
