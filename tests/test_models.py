"""Unit tests for multi-modal scene understanding."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, count_parameters
from models import SceneUnderstandingModel, VisionEncoder, TextEncoder, CrossModalAttention
from losses import SceneUnderstandingLoss
from eval import SceneUnderstandingMetrics
from data import SceneUnderstandingDataset


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seeds are set (no exception should be raised)
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5 (bias)


class TestModels:
    """Test model components."""
    
    def test_vision_encoder(self):
        """Test vision encoder."""
        encoder = VisionEncoder(model_name="resnet50", pretrained=False)
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        output = encoder(images)
        
        assert output.shape[0] == batch_size
        assert len(output.shape) == 3  # (batch_size, seq_len, hidden_size)
    
    def test_text_encoder(self):
        """Test text encoder."""
        encoder = TextEncoder(model_name="bert-base-uncased", pretrained=False)
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = encoder(input_ids, attention_mask)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
    
    def test_cross_modal_attention(self):
        """Test cross-modal attention."""
        hidden_size = 768
        num_heads = 8
        batch_size = 2
        seq_len = 10
        
        attention = CrossModalAttention(hidden_size, num_heads)
        
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        output = attention(query, key, value)
        
        assert output.shape == query.shape
    
    def test_scene_understanding_model(self):
        """Test main model."""
        model = SceneUnderstandingModel(
            vision_model="resnet50",
            text_model="bert-base-uncased",
            num_scene_classes=2,
            num_object_classes=10,
            hidden_size=768,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        
        outputs = model(images, input_ids, attention_mask)
        
        assert "scene_logits" in outputs
        assert "object_logits" in outputs
        assert "description_features" in outputs
        
        assert outputs["scene_logits"].shape[0] == batch_size
        assert outputs["object_logits"].shape[0] == batch_size


class TestLosses:
    """Test loss functions."""
    
    def test_scene_understanding_loss(self):
        """Test scene understanding loss."""
        loss_fn = SceneUnderstandingLoss()
        
        batch_size = 2
        num_scene_classes = 2
        num_object_classes = 10
        
        # Mock outputs
        outputs = {
            "scene_logits": torch.randn(batch_size, num_scene_classes),
            "object_logits": torch.randn(batch_size, num_object_classes),
            "vision_features": torch.randn(batch_size, 10, 768),
            "text_features": torch.randn(batch_size, 10, 768)
        }
        
        # Mock targets
        targets = {
            "scene_labels": torch.randint(0, num_scene_classes, (batch_size,)),
            "object_labels": torch.randint(0, 2, (batch_size, num_object_classes)).float()
        }
        
        losses = loss_fn(outputs, targets)
        
        assert "total_loss" in losses
        assert "scene_loss" in losses
        assert "object_loss" in losses
        assert "contrastive_loss" in losses
        
        assert losses["total_loss"] > 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_scene_understanding_metrics(self):
        """Test scene understanding metrics."""
        metrics = SceneUnderstandingMetrics(num_scene_classes=2, num_object_classes=10)
        
        batch_size = 4
        
        # Mock predictions and targets
        scene_logits = torch.randn(batch_size, 2)
        object_logits = torch.randn(batch_size, 10)
        scene_targets = torch.randint(0, 2, (batch_size,))
        object_targets = torch.randint(0, 2, (batch_size, 10)).float()
        
        metrics.update(scene_logits, object_logits, scene_targets, object_targets)
        
        computed_metrics = metrics.compute()
        
        assert "scene_accuracy" in computed_metrics
        assert "scene_f1_macro" in computed_metrics
        assert "object_f1_macro" in computed_metrics
        
        # Reset metrics
        metrics.reset()
        assert len(metrics.scene_predictions) == 0


class TestData:
    """Test data loading."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        # Create a temporary data directory
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dataset
            dataset = SceneUnderstandingDataset(
                data_dir=temp_dir,
                split="train",
                image_size=(224, 224),
                max_text_length=128,
                tokenizer_name="bert-base-uncased"
            )
            
            # Test dataset length
            assert len(dataset) > 0
            
            # Test getting a sample
            sample = dataset[0]
            
            assert "image" in sample
            assert "text" in sample
            assert "description" in sample
            assert "objects" in sample
            assert "scene_type" in sample
            
            # Test image tensor shape
            assert sample["image"].shape == (3, 224, 224)
            
            # Test text tensor shapes
            assert "input_ids" in sample["text"]
            assert "attention_mask" in sample["text"]


if __name__ == "__main__":
    pytest.main([__file__])
