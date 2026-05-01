#!/usr/bin/env python3
"""Evaluation script for multi-modal scene understanding."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import get_device, suppress_warnings
from data import SceneUnderstandingDataset, collate_fn
from models import SceneUnderstandingModel
from eval import SceneUnderstandingMetrics
from viz import create_confusion_matrix, plot_training_curves


class Evaluator:
    """Evaluation class for scene understanding model."""
    
    def __init__(self, config: DictConfig, checkpoint_path: str):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary.
            checkpoint_path: Path to model checkpoint.
        """
        self.config = config
        self.device = get_device()
        
        # Suppress warnings
        suppress_warnings()
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Initialize metrics
        self.metrics = SceneUnderstandingMetrics(
            num_scene_classes=self.config.model.num_scene_classes,
            num_object_classes=self.config.model.num_object_classes
        )
    
    def _load_model(self, checkpoint_path: str) -> SceneUnderstandingModel:
        """Load the trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            
        Returns:
            Loaded model.
        """
        model = SceneUnderstandingModel(
            vision_model=self.config.model.vision_model,
            text_model=self.config.model.text_model,
            num_scene_classes=self.config.model.num_scene_classes,
            num_object_classes=self.config.model.num_object_classes,
            hidden_size=self.config.model.hidden_size,
            dropout=self.config.model.dropout
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Using random weights for evaluation")
        
        model.eval()
        return model
    
    def _setup_data_loader(self) -> DataLoader:
        """Setup data loader for evaluation.
        
        Returns:
            Data loader for evaluation.
        """
        dataset = SceneUnderstandingDataset(
            data_dir=self.config.paths.data_dir,
            split="test",
            image_size=tuple(self.config.data.image_size),
            max_text_length=self.config.data.max_text_length,
            tokenizer_name=self.config.data.tokenizer_name
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.evaluation.dataset.batch_size,
            shuffle=self.config.evaluation.dataset.shuffle,
            num_workers=self.config.evaluation.dataset.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        return dataloader
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on test set.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        print("Setting up data loader...")
        dataloader = self._setup_data_loader()
        
        print(f"Evaluating on {len(dataloader.dataset)} samples...")
        
        self.metrics.reset()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluation")
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_masks"].to(self.device)
                
                # Create targets (simplified for demo)
                scene_labels = torch.randint(0, self.config.model.num_scene_classes, (images.size(0),)).to(self.device)
                object_labels = torch.randint(0, 2, (images.size(0), self.config.model.num_object_classes)).to(self.device)
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_masks)
                
                # Update metrics
                self.metrics.update(
                    outputs["scene_logits"],
                    outputs["object_logits"],
                    scene_labels,
                    object_labels
                )
                
                # Store predictions and targets for visualization
                scene_preds = torch.argmax(outputs["scene_logits"], dim=1).cpu().numpy()
                scene_targets = scene_labels.cpu().numpy()
                
                all_predictions.extend(scene_preds)
                all_targets.extend(scene_targets)
                
                pbar.set_postfix({"batch": batch_idx + 1})
        
        # Compute final metrics
        metrics = self.metrics.compute()
        
        # Create visualizations
        if self.config.evaluation.visualization.save_predictions:
            self._create_visualizations(all_predictions, all_targets)
        
        return metrics
    
    def _create_visualizations(
        self, 
        predictions: list, 
        targets: list
    ) -> None:
        """Create evaluation visualizations.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        """
        import numpy as np
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Create confusion matrix
        class_names = ["indoor", "outdoor"]
        confusion_matrix_path = os.path.join(
            self.config.paths.output_dir, 
            "confusion_matrix.png"
        )
        
        create_confusion_matrix(
            predictions, 
            targets, 
            class_names, 
            save_path=confusion_matrix_path
        )
        
        print(f"Confusion matrix saved to {confusion_matrix_path}")
    
    def print_results(self, metrics: Dict[str, float]) -> None:
        """Print evaluation results.
        
        Args:
            metrics: Dictionary containing evaluation metrics.
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nScene Classification:")
        print(f"  Accuracy: {metrics.get('scene_accuracy', 0):.4f}")
        print(f"  F1-Score (Macro): {metrics.get('scene_f1_macro', 0):.4f}")
        print(f"  F1-Score (Weighted): {metrics.get('scene_f1_weighted', 0):.4f}")
        print(f"  Precision: {metrics.get('scene_precision', 0):.4f}")
        print(f"  Recall: {metrics.get('scene_recall', 0):.4f}")
        
        print("\nObject Detection:")
        print(f"  F1-Score (Macro): {metrics.get('object_f1_macro', 0):.4f}")
        print(f"  F1-Score (Micro): {metrics.get('object_f1_micro', 0):.4f}")
        print(f"  Precision: {metrics.get('object_precision', 0):.4f}")
        print(f"  Recall: {metrics.get('object_recall', 0):.4f}")
        
        # Per-class object metrics
        print("\nPer-Class Object Detection:")
        for i in range(self.config.model.num_object_classes):
            key = f"object_f1_class_{i}"
            if key in metrics:
                print(f"  Class {i}: {metrics[key]:.4f}")
        
        print("\n" + "="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-modal scene understanding model")
    parser.add_argument("--config", type=str, default="configs/model/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
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
    
    # Create output directory
    os.makedirs(config.paths.output_dir, exist_ok=True)
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(config, args.checkpoint)
    metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save metrics to file
    import json
    metrics_path = os.path.join(config.paths.output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
