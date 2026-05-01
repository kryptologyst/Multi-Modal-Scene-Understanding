"""Modern multi-modal scene understanding models."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class VisionEncoder(nn.Module):
    """Vision encoder using Vision Transformer or ResNet backbone.
    
    Args:
        model_name: Name of the vision model to use.
        pretrained: Whether to use pretrained weights.
        freeze_backbone: Whether to freeze the backbone weights.
    """
    
    def __init__(
        self,
        model_name: str = "vit-base-patch16-224",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        if "vit" in model_name.lower():
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(
                model_name if pretrained else None,
                add_pooling_layer=False
            )
            self.hidden_size = self.backbone.config.hidden_size
        else:
            # Fallback to ResNet
            import torchvision.models as models
            if model_name == "resnet50":
                self.backbone = models.resnet50(pretrained=pretrained)
                self.hidden_size = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.
        
        Args:
            images: Input images tensor of shape (batch_size, 3, H, W).
            
        Returns:
            Visual features tensor.
        """
        if hasattr(self.backbone, 'config'):  # ViT
            outputs = self.backbone(images)
            return outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        else:  # ResNet
            features = self.backbone(images)  # (batch_size, hidden_size)
            return features.unsqueeze(1)  # Add sequence dimension


class TextEncoder(nn.Module):
    """Text encoder using transformer models.
    
    Args:
        model_name: Name of the text model to use.
        pretrained: Whether to use pretrained weights.
        freeze_backbone: Whether to freeze the backbone weights.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(
            model_name if pretrained else None
        )
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the text encoder.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for the input.
            
        Returns:
            Text features tensor.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing vision and text features.
    
    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through cross-modal attention.
        
        Args:
            query: Query tensor (e.g., text features).
            key: Key tensor (e.g., visual features).
            value: Value tensor (e.g., visual features).
            mask: Optional attention mask.
            
        Returns:
            Attended features tensor.
        """
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, key_seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, key_seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attended)


class SceneUnderstandingModel(nn.Module):
    """Main multi-modal scene understanding model.
    
    Args:
        vision_model: Name of the vision model.
        text_model: Name of the text model.
        num_scene_classes: Number of scene type classes.
        num_object_classes: Number of object classes.
        hidden_size: Hidden dimension size.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        vision_model: str = "vit-base-patch16-224",
        text_model: str = "bert-base-uncased",
        num_scene_classes: int = 2,  # indoor/outdoor
        num_object_classes: int = 10,
        hidden_size: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionEncoder(vision_model)
        self.text_encoder = TextEncoder(text_model)
        
        # Projection layers to align dimensions
        self.vision_proj = nn.Linear(self.vision_encoder.hidden_size, hidden_size)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, hidden_size)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(hidden_size, dropout=dropout)
        
        # Classification heads
        self.scene_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_scene_classes)
        )
        
        self.object_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_object_classes)
        )
        
        # Description generation head
        self.description_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            images: Input images tensor.
            input_ids: Input token IDs.
            attention_mask: Attention mask for text.
            
        Returns:
            Dictionary containing model outputs.
        """
        # Encode vision and text
        vision_features = self.vision_encoder(images)  # (batch_size, seq_len, hidden_size)
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch_size, seq_len, hidden_size)
        
        # Project to common dimension
        vision_features = self.vision_proj(vision_features)
        text_features = self.text_proj(text_features)
        
        # Cross-modal attention (text attends to vision)
        attended_features = self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
            mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )
        
        # Global average pooling for classification
        pooled_features = attended_features.mean(dim=1)  # (batch_size, hidden_size)
        pooled_features = self.dropout(pooled_features)
        
        # Classification outputs
        scene_logits = self.scene_classifier(pooled_features)
        object_logits = self.object_classifier(pooled_features)
        
        # Description features
        description_features = self.description_head(pooled_features)
        
        return {
            "scene_logits": scene_logits,
            "object_logits": object_logits,
            "description_features": description_features,
            "vision_features": vision_features,
            "text_features": text_features,
            "attended_features": attended_features
        }
    
    def get_attention_weights(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get attention weights for visualization.
        
        Args:
            images: Input images tensor.
            input_ids: Input token IDs.
            attention_mask: Attention mask for text.
            
        Returns:
            Attention weights tensor.
        """
        # Encode vision and text
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Project to common dimension
        vision_features = self.vision_proj(vision_features)
        text_features = self.text_proj(text_features)
        
        # Compute attention weights
        batch_size, seq_len, _ = text_features.size()
        
        q = self.cross_attention.q_proj(text_features)
        k = self.cross_attention.k_proj(vision_features)
        
        q = q.view(batch_size, seq_len, self.cross_attention.num_heads, self.cross_attention.head_dim)
        k = k.view(batch_size, -1, self.cross_attention.num_heads, self.cross_attention.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.cross_attention.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights.mean(dim=1)  # Average over heads
