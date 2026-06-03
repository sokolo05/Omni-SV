import torch
import torch.nn as nn
import torch.nn.functional as F

def param_groups_l2_dual(model, loss_fn=None, backbone_wd=1e-5, head_wd=1e-3, no_decay_bn_bias=True):
    """
    Perform layer-wise L2 regularization parameter grouping optimized for the OmniSV framework.
    Guarantees learnable loss scaling factors and normalization layers are handled safely.
    """
    backbone, head, no_decay = [], [], []
    
    # 1. Collect trainable parameters from the structural backbone and fusion heads
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        *prefix, leaf = name.split('.')
        prefix = '.'.join(prefix)

        # Route core fusion modules and classification head to the head group
        if any(k in prefix for k in ('classifier', 'cross_attn', 'eaanr', 'f_modulator')):
            target = head
        else:
            target = backbone

        # Isolate Normalization (BatchNorm/LayerNorm) and all biases from weight decay
        if no_decay_bn_bias and (leaf.endswith('bias') or 'bn' in prefix or 'BatchNorm' in prefix or 'norm' in leaf or 'LayerNorm' in prefix):
            no_decay.append(param)
        else:
            target.append(param)

    # 2. Collect learnable optimization weights (lambda1/lambda2) from loss function if provided
    if loss_fn is not None:
        for name, param in loss_fn.named_parameters():
            if param.requires_grad:
                # Task scaling factors must never undergo weight decay
                no_decay.append(param)

    groups = [
        {'params': backbone, 'weight_decay': backbone_wd},
        {'params': head,     'weight_decay': head_wd},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    return groups

class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        log_probs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
