"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Target labels of shape (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassWeightedBCE(nn.Module):
    """Binary cross-entropy with class weights."""
    
    def __init__(self, pos_weight: Optional[float] = None):
        super(ClassWeightedBCE, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Logits of shape (batch_size, num_classes) or (batch_size,)
            targets: Target labels of shape (batch_size,)
        """
        if inputs.dim() > 1:
            # Multi-class: use cross-entropy
            return F.cross_entropy(inputs, targets, weight=None)
        else:
            # Binary: use BCE with logits
            return F.binary_cross_entropy_with_logits(
                inputs, targets.float(), pos_weight=self.pos_weight
            )


class MultitaskLoss(nn.Module):
    """
    Combined loss for multitask learning (classification + regression).
    """
    
    def __init__(self,
                 classification_loss: nn.Module,
                 regression_loss: nn.Module = nn.MSELoss(),
                 classification_weight: float = 1.0,
                 regression_weight: float = 0.1):
        super(MultitaskLoss, self).__init__()
        self.classification_loss = classification_loss
        self.regression_loss = regression_loss
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
    
    def forward(self, pred_class, pred_reg, target_class, target_reg):
        """
        Compute combined multitask loss.
        
        Args:
            pred_class: Classification predictions (logits)
            pred_reg: Regression predictions
            target_class: Classification targets
            target_reg: Regression targets
        """
        cls_loss = self.classification_loss(pred_class, target_class)
        reg_loss = self.regression_loss(pred_reg.squeeze(), target_reg.squeeze())
        
        total_loss = (self.classification_weight * cls_loss + 
                     self.regression_weight * reg_loss)
        
        return total_loss, cls_loss, reg_loss

