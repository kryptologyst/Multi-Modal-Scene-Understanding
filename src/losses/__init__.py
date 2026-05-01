"""Loss functions for multi-modal scene understanding."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneUnderstandingLoss(nn.Module):
    """Combined loss function for scene understanding tasks.
    
    Args:
        scene_weight: Weight for scene classification loss.
        object_weight: Weight for object detection loss.
        contrastive_weight: Weight for contrastive learning loss.
        temperature: Temperature for contrastive loss.
    """
    
    def __init__(
        self,
        scene_weight: float = 1.0,
        object_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.scene_weight = scene_weight
        self.object_weight = object_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.scene_loss = nn.CrossEntropyLoss()
        self.object_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss.
        
        Args:
            outputs: Model outputs dictionary.
            targets: Target labels dictionary.
            
        Returns:
            Dictionary containing individual and total losses.
        """
        losses = {}
        
        # Scene classification loss
        if "scene_logits" in outputs and "scene_labels" in targets:
            scene_loss = self.scene_loss(outputs["scene_logits"], targets["scene_labels"])
            losses["scene_loss"] = scene_loss
        
        # Object detection loss (multi-label)
        if "object_logits" in outputs and "object_labels" in targets:
            object_loss = self.object_loss(outputs["object_logits"], targets["object_labels"].float())
            losses["object_loss"] = object_loss
        
        # Contrastive loss between vision and text features
        if "vision_features" in outputs and "text_features" in outputs:
            contrastive_loss = self._compute_contrastive_loss(
                outputs["vision_features"],
                outputs["text_features"]
            )
            losses["contrastive_loss"] = contrastive_loss
        
        # Compute total loss
        total_loss = (
            self.scene_weight * losses.get("scene_loss", 0) +
            self.object_weight * losses.get("object_loss", 0) +
            self.contrastive_weight * losses.get("contrastive_loss", 0)
        )
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_contrastive_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between vision and text features.
        
        Args:
            vision_features: Vision features tensor.
            text_features: Text features tensor.
            
        Returns:
            Contrastive loss value.
        """
        # Global average pooling
        vision_pooled = vision_features.mean(dim=1)  # (batch_size, hidden_size)
        text_pooled = text_features.mean(dim=1)  # (batch_size, hidden_size)
        
        # Normalize features
        vision_pooled = F.normalize(vision_pooled, p=2, dim=1)
        text_pooled = F.normalize(text_pooled, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(vision_pooled, text_pooled.T) / self.temperature
        
        batch_size = similarity_matrix.size(0)
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Compute loss for both directions
        loss_v2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2v = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_v2t + loss_t2v) / 2


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor for rare class.
        gamma: Focusing parameter.
        reduction: Reduction method.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions.
            targets: Ground truth labels.
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            ce_loss = ce_loss * at
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization.
    
    Args:
        smoothing: Label smoothing factor.
        reduction: Reduction method.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.
        
        Args:
            inputs: Model predictions.
            targets: Ground truth labels.
            
        Returns:
            Label smoothing loss value.
        """
        log_preds = F.log_softmax(inputs, dim=1)
        nll_loss = -log_preds.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        smooth_loss = -log_preds.mean(dim=1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
