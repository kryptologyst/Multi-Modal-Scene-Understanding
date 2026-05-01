"""Visualization tools for multi-modal scene understanding."""

import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle


def visualize_attention(
    image: np.ndarray,
    attention_weights: torch.Tensor,
    text_tokens: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Visualize attention weights overlaid on image.
    
    Args:
        image: Input image as numpy array.
        attention_weights: Attention weights tensor.
        text_tokens: List of text tokens.
        save_path: Path to save the visualization.
        figsize: Figure size for the plot.
    """
    # Convert attention weights to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Average over sequence dimension if needed
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Resize attention weights to match image size
    h, w = image.shape[:2]
    attention_map = cv2.resize(attention_weights, (w, h))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Attention map
    im1 = axes[0, 1].imshow(attention_map, cmap='hot', interpolation='bilinear')
    axes[0, 1].set_title("Attention Map")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Overlay attention on image
    attention_colored = cv2.applyColorMap(
        (attention_map * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image, 0.7, attention_colored, 0.3, 0)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Attention Overlay")
    axes[1, 0].axis('off')
    
    # Text tokens with attention weights
    if text_tokens:
        token_weights = attention_weights.mean(axis=1) if len(attention_weights.shape) > 1 else attention_weights
        axes[1, 1].bar(range(len(text_tokens)), token_weights)
        axes[1, 1].set_xticks(range(len(text_tokens)))
        axes[1, 1].set_xticklabels(text_tokens, rotation=45, ha='right')
        axes[1, 1].set_title("Token Attention Weights")
        axes[1, 1].set_ylabel("Attention Weight")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    descriptions: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 4
) -> None:
    """Visualize model predictions alongside ground truth.
    
    Args:
        images: Batch of input images.
        predictions: Model predictions dictionary.
        targets: Ground truth targets dictionary.
        descriptions: Text descriptions.
        save_path: Path to save the visualization.
        num_samples: Number of samples to visualize.
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Denormalize image
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        
        # Display image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis('off')
        
        # Display predictions and targets
        pred_text = f"Pred: {predictions.get('scene_pred', 'N/A')}\n"
        pred_text += f"Target: {targets.get('scene_target', 'N/A')}\n"
        pred_text += f"Description: {descriptions[i][:50]}..."
        
        axes[1, i].text(0.1, 0.5, pred_text, transform=axes[1, i].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """Create and display confusion matrix.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth labels.
        class_names: List of class names.
        save_path: Path to save the confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot training curves for losses and metrics.
    
    Args:
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        train_metrics: Training metrics per epoch.
        val_metrics: Validation metrics per epoch.
        save_path: Path to save the plots.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
        axes[0, 1].plot(epochs, train_metrics['accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # F1 Score curves
    if 'f1_score' in train_metrics and 'f1_score' in val_metrics:
        axes[1, 0].plot(epochs, train_metrics['f1_score'], 'b-', label='Training F1')
        axes[1, 0].plot(epochs, val_metrics['f1_score'], 'r-', label='Validation F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate curve (if available)
    if 'learning_rate' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['learning_rate'], 'g-', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """Create a leaderboard visualization of model results.
    
    Args:
        results: Dictionary containing model results.
        save_path: Path to save the leaderboard.
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values.round(4),
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code the cells
    for i in range(len(df.columns)):
        for j in range(len(df.index)):
            cell = table[(j+1, i)]
            if i == 0:  # First column (model names)
                cell.set_facecolor('#f0f0f0')
            else:
                # Color based on performance (higher is better)
                value = df.iloc[j, i]
                if pd.notna(value):
                    normalized = (value - df.iloc[:, i].min()) / (df.iloc[:, i].max() - df.iloc[:, i].min())
                    cell.set_facecolor(plt.cm.Greens(normalized))
    
    plt.title('Model Performance Leaderboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_object_detections(
    image: np.ndarray,
    detections: List[Dict[str, Union[str, float, Tuple[int, int, int, int]]]],
    save_path: Optional[str] = None
) -> None:
    """Visualize object detections on image.
    
    Args:
        image: Input image as numpy array.
        detections: List of detection dictionaries with 'label', 'confidence', 'bbox'.
        save_path: Path to save the visualization.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']  # (x, y, w, h)
        label = detection['label']
        confidence = detection['confidence']
        
        # Create rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(bbox[0], bbox[1] - 5, f"{label}: {confidence:.2f}",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                fontsize=10, fontweight='bold')
    
    ax.set_title('Object Detections')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
