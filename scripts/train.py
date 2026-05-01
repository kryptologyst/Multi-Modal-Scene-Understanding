#!/usr/bin/env python3
"""Training script for multi-modal scene understanding."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import set_seed, get_device, EarlyStopping, suppress_warnings
from data import SceneUnderstandingDataset, collate_fn
from models import SceneUnderstandingModel
from losses import SceneUnderstandingLoss
from eval import SceneUnderstandingMetrics


class Trainer:
    """Main trainer class for scene understanding model."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Suppress warnings
        suppress_warnings()
        
        # Create output directories
        self._create_directories()
        
        # Initialize model, loss, and optimizer
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        
        # Initialize metrics and logging
        self._setup_metrics()
        self._setup_logging()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            restore_best_weights=config.training.early_stopping.restore_best_weights
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        
    def _create_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.config.paths.output_dir, exist_ok=True)
        os.makedirs(self.config.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.paths.log_dir, exist_ok=True)
    
    def _setup_model(self) -> None:
        """Setup the model."""
        self.model = SceneUnderstandingModel(
            vision_model=self.config.model.vision_model,
            text_model=self.config.model.text_model,
            num_scene_classes=self.config.model.num_scene_classes,
            num_object_classes=self.config.model.num_object_classes,
            hidden_size=self.config.model.hidden_size,
            dropout=self.config.model.dropout
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_loss(self) -> None:
        """Setup loss function."""
        self.criterion = SceneUnderstandingLoss(
            scene_weight=self.config.loss.scene_weight,
            object_weight=self.config.loss.object_weight,
            contrastive_weight=self.config.loss.contrastive_weight,
            temperature=self.config.loss.temperature
        )
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Setup scheduler
        if self.config.training.scheduler.name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.scheduler.min_lr
            )
        else:
            self.scheduler = None
    
    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.metrics = SceneUnderstandingMetrics(
            num_scene_classes=self.config.model.num_scene_classes,
            num_object_classes=self.config.model.num_object_classes
        )
    
    def _setup_logging(self) -> None:
        """Setup logging."""
        if self.config.logging.tensorboard:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.config.paths.log_dir, "tensorboard")
            )
        else:
            self.writer = None
    
    def _setup_data_loaders(self) -> None:
        """Setup data loaders."""
        # Create datasets
        self.train_dataset = SceneUnderstandingDataset(
            data_dir=self.config.paths.data_dir,
            split="train",
            image_size=tuple(self.config.data.image_size),
            max_text_length=self.config.data.max_text_length,
            tokenizer_name=self.config.data.tokenizer_name
        )
        
        self.val_dataset = SceneUnderstandingDataset(
            data_dir=self.config.paths.data_dir,
            split="val",
            image_size=tuple(self.config.data.image_size),
            max_text_length=self.config.data.max_text_length,
            tokenizer_name=self.config.data.tokenizer_name
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.data_loader.batch_size,
            shuffle=self.config.training.data_loader.shuffle,
            num_workers=self.config.training.data_loader.num_workers,
            pin_memory=self.config.training.data_loader.pin_memory,
            drop_last=self.config.training.data_loader.drop_last,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.data_loader.batch_size,
            shuffle=False,
            num_workers=self.config.training.data_loader.num_workers,
            pin_memory=self.config.training.data_loader.pin_memory,
            collate_fn=collate_fn
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_masks = batch["attention_masks"].to(self.device)
            
            # Create targets (simplified for demo)
            scene_labels = torch.randint(0, self.config.model.num_scene_classes, (images.size(0),)).to(self.device)
            object_labels = torch.randint(0, 2, (images.size(0), self.config.model.num_object_classes)).to(self.device)
            
            targets = {
                "scene_labels": scene_labels,
                "object_labels": object_labels
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, input_ids, attention_masks)
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            loss = losses["total_loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.metrics.update(
                outputs["scene_logits"],
                outputs["object_logits"],
                scene_labels,
                object_labels
            )
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config.logging.log_interval == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar("train/loss", loss.item(), global_step)
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics["loss"] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_masks"].to(self.device)
                
                # Create targets (simplified for demo)
                scene_labels = torch.randint(0, self.config.model.num_scene_classes, (images.size(0),)).to(self.device)
                object_labels = torch.randint(0, 2, (images.size(0), self.config.model.num_object_classes)).to(self.device)
                
                targets = {
                    "scene_labels": scene_labels,
                    "object_labels": object_labels
                }
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_masks)
                
                # Compute loss
                losses = self.criterion(outputs, targets)
                loss = losses["total_loss"]
                
                # Update metrics
                self.metrics.update(
                    outputs["scene_logits"],
                    outputs["object_logits"],
                    scene_labels,
                    object_labels
                )
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics["loss"] = total_loss / num_batches
        
        return epoch_metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_score": self.best_val_score,
            "config": self.config
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.paths.checkpoint_dir,
            f"checkpoint_epoch_{self.current_epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.paths.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def train(self) -> None:
        """Main training loop."""
        print("Setting up data loaders...")
        self._setup_data_loaders()
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        print("Starting training...")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics.get('scene_accuracy', 0):.4f}")
            
            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("epoch/val_accuracy", val_metrics.get("scene_accuracy", 0), epoch)
            
            # Check if best model
            val_score = val_metrics.get("scene_accuracy", 0)
            is_best = val_score > self.best_val_score
            
            if is_best:
                self.best_val_score = val_score
            
            # Save checkpoint
            if epoch % self.config.logging.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics["loss"], self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print("Training completed!")
        print(f"Best validation score: {self.best_val_score:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-modal scene understanding model")
    parser.add_argument("--config", type=str, default="configs/model/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Path to output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config.paths.data_dir = args.data_dir
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
