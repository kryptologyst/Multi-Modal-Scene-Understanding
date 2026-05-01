"""Core utilities for multi-modal scene understanding."""

import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For Apple Silicon MPS
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.deterministic = True


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        DictConfig: Loaded configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        save_path: Path to save the configuration.
    """
    OmegaConf.save(config, save_path)


def create_dir_if_not_exists(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format.
        
    Returns:
        str: Formatted number string.
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}K"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best_weights: Whether to restore best weights when stopping.
    """
    
    def __init__(
        self, 
        patience: int = 7, 
        min_delta: float = 0.0, 
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop early.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially restore weights for.
            
        Returns:
            bool: True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        self.best_weights = model.state_dict().copy()


def suppress_warnings() -> None:
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
