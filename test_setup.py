#!/usr/bin/env python3
"""Test script to verify the multi-modal scene understanding setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils import set_seed, get_device
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        from models import SceneUnderstandingModel
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        from data import SceneUnderstandingDataset
        print("✓ Data imported successfully")
    except ImportError as e:
        print(f"✗ Data import failed: {e}")
        return False
    
    try:
        from losses import SceneUnderstandingLoss
        print("✓ Losses imported successfully")
    except ImportError as e:
        print(f"✗ Losses import failed: {e}")
        return False
    
    try:
        from eval import SceneUnderstandingMetrics
        print("✓ Evaluation imported successfully")
    except ImportError as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    try:
        from viz import visualize_attention
        print("✓ Visualization imported successfully")
    except ImportError as e:
        print(f"✗ Visualization import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that the model can be created."""
    print("\nTesting model creation...")
    
    try:
        from models import SceneUnderstandingModel
        from utils import get_device
        
        device = get_device()
        model = SceneUnderstandingModel(
            vision_model="resnet50",  # Use ResNet for faster testing
            text_model="bert-base-uncased",
            num_scene_classes=2,
            num_object_classes=10,
            hidden_size=768,
            dropout=0.1
        ).to(device)
        
        print(f"✓ Model created successfully on {device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test that data can be loaded."""
    print("\nTesting data loading...")
    
    try:
        from data import SceneUnderstandingDataset
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SceneUnderstandingDataset(
                data_dir=temp_dir,
                split="train",
                image_size=(224, 224),
                max_text_length=128,
                tokenizer_name="bert-base-uncased"
            )
            
            print(f"✓ Dataset created successfully with {len(dataset)} samples")
            
            # Test getting a sample
            sample = dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  - Image shape: {sample['image'].shape}")
            print(f"  - Text keys: {list(sample['text'].keys())}")
            print(f"  - Description: {sample['description'][:50]}...")
            
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        from omegaconf import OmegaConf
        
        config_path = "configs/model/default.yaml"
        if Path(config_path).exists():
            config = OmegaConf.load(config_path)
            print(f"✓ Configuration loaded from {config_path}")
            print(f"  - Vision model: {config.model.vision_model}")
            print(f"  - Text model: {config.model.text_model}")
            print(f"  - Scene classes: {config.model.num_scene_classes}")
        else:
            print(f"✗ Configuration file not found: {config_path}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Multi-Modal Scene Understanding - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_loading,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train.py")
        print("2. Run evaluation: python scripts/evaluate.py --checkpoint checkpoints/best_model.pt")
        print("3. Launch demo: python demo/app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
