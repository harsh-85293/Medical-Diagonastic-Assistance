"""
Loss functions for Medical AI Assistant.
Implements FocalLoss with class balancing for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical image classification.
    
    Reference: "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor (num_classes,)
            gamma: Focusing parameter (higher values focus more on hard examples)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            logger.info(f"FocalLoss initialized with alpha weights and gamma={gamma}")
        else:
            logger.info(f"FocalLoss initialized without alpha weights and gamma={gamma}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Logits tensor of shape (batch_size, num_classes)
            targets: Target tensor of shape (batch_size,) with class indices
            
        Returns:
            Loss tensor
        """
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Gather the predicted probabilities for the target classes
        batch_size = inputs.size(0)
        probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        pt = probs
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            alpha = self.alpha.gather(0, targets)
            focal_weight = alpha * focal_weight
        
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weighting
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance."""
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Initialize Weighted Cross Entropy Loss.
        
        Args:
            weights: Class weights tensor (num_classes,)
        """
        super().__init__()
        self.weights = weights
        
        if weights is not None:
            logger.info(f"WeightedCrossEntropyLoss initialized with class weights")
        else:
            logger.info("WeightedCrossEntropyLoss initialized without weights")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Weighted Cross Entropy Loss.
        
        Args:
            inputs: Logits tensor of shape (batch_size, num_classes)
            targets: Target tensor of shape (batch_size,) with class indices
            
        Returns:
            Loss tensor
        """
        return F.cross_entropy(inputs, targets, weight=self.weights)


class CombinedLoss(nn.Module):
    """Combination of Focal Loss and other losses."""
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        ce_weight: float = 0.0,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0
    ):
        """
        Initialize Combined Loss.
        
        Args:
            focal_weight: Weight for focal loss component
            ce_weight: Weight for cross entropy component
            alpha: Class weights for focal loss
            gamma: Focusing parameter for focal loss
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_loss = nn.CrossEntropyLoss()
        
        logger.info(f"CombinedLoss initialized: focal_weight={focal_weight}, ce_weight={ce_weight}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Combined Loss.
        
        Args:
            inputs: Logits tensor
            targets: Target tensor
            
        Returns:
            Combined loss tensor
        """
        focal_loss = self.focal_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        
        return self.focal_weight * focal_loss + self.ce_weight * ce_loss


def create_loss_function(config: Dict[str, Any], class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration dictionary
        class_weights: Class weights tensor
        
    Returns:
        Loss function
    """
    loss_config = config['training']['focal_loss']
    
    if loss_config['alpha_from_weights'] and class_weights is not None:
        alpha = class_weights
        logger.info("Using class weights for focal loss alpha")
    else:
        alpha = None
        logger.info("Not using class weights for focal loss alpha")
    
    focal_loss = FocalLoss(
        alpha=alpha,
        gamma=loss_config['gamma']
    )
    
    return focal_loss


def calculate_class_weights_from_counts(class_counts: torch.Tensor, method: str = "inverse") -> torch.Tensor:
    """
    Calculate class weights from class counts.
    
    Args:
        class_counts: Tensor of class counts
        method: Weighting method ('inverse', 'sqrt_inverse', 'log_inverse')
        
    Returns:
        Class weights tensor
    """
    if method == "inverse":
        weights = 1.0 / class_counts
    elif method == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(class_counts)
    elif method == "log_inverse":
        weights = 1.0 / torch.log(class_counts + 1)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return weights


def get_loss_function_name(loss_fn: nn.Module) -> str:
    """Get the name of the loss function."""
    if isinstance(loss_fn, FocalLoss):
        return "FocalLoss"
    elif isinstance(loss_fn, WeightedCrossEntropyLoss):
        return "WeightedCrossEntropyLoss"
    elif isinstance(loss_fn, CombinedLoss):
        return "CombinedLoss"
    else:
        return loss_fn.__class__.__name__


if __name__ == "__main__":
    # Test loss functions
    batch_size = 4
    num_classes = 8
    
    # Create dummy data
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test FocalLoss
    focal_loss = FocalLoss(gamma=2.0)
    focal_output = focal_loss(inputs, targets)
    print(f"FocalLoss output: {focal_output.item():.4f}")
    
    # Test with class weights
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2.0  # Give more weight to first class
    
    focal_loss_weighted = FocalLoss(alpha=class_weights, gamma=2.0)
    focal_weighted_output = focal_loss_weighted(inputs, targets)
    print(f"FocalLoss with weights output: {focal_weighted_output.item():.4f}")
    
    # Test WeightedCrossEntropyLoss
    weighted_ce = WeightedCrossEntropyLoss(weights=class_weights)
    weighted_ce_output = weighted_ce(inputs, targets)
    print(f"WeightedCrossEntropyLoss output: {weighted_ce_output.item():.4f}")
    
    # Test CombinedLoss
    combined_loss = CombinedLoss(focal_weight=0.7, ce_weight=0.3)
    combined_output = combined_loss(inputs, targets)
    print(f"CombinedLoss output: {combined_output.item():.4f}")
