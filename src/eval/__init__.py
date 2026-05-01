"""Evaluation metrics for multi-modal scene understanding."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SceneUnderstandingMetrics:
    """Metrics calculator for scene understanding tasks."""
    
    def __init__(self, num_scene_classes: int = 2, num_object_classes: int = 10):
        self.num_scene_classes = num_scene_classes
        self.num_object_classes = num_object_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.scene_predictions = []
        self.scene_targets = []
        self.object_predictions = []
        self.object_targets = []
        self.scene_confidences = []
        self.object_confidences = []
    
    def update(
        self,
        scene_logits: torch.Tensor,
        object_logits: torch.Tensor,
        scene_targets: torch.Tensor,
        object_targets: torch.Tensor
    ) -> None:
        """Update metrics with new predictions and targets.
        
        Args:
            scene_logits: Scene classification logits.
            object_logits: Object detection logits.
            scene_targets: Scene ground truth labels.
            object_targets: Object ground truth labels.
        """
        # Convert to numpy for sklearn metrics
        scene_preds = torch.argmax(scene_logits, dim=1).cpu().numpy()
        scene_targets_np = scene_targets.cpu().numpy()
        
        # Object predictions (multi-label)
        object_preds = torch.sigmoid(object_logits).cpu().numpy()
        object_targets_np = object_targets.cpu().numpy()
        
        # Store predictions and targets
        self.scene_predictions.extend(scene_preds)
        self.scene_targets.extend(scene_targets_np)
        self.object_predictions.extend(object_preds)
        self.object_targets.extend(object_targets_np)
        
        # Store confidences
        scene_conf = torch.softmax(scene_logits, dim=1).cpu().numpy()
        self.scene_confidences.extend(scene_conf)
        self.object_confidences.extend(object_preds)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing computed metrics.
        """
        metrics = {}
        
        # Scene classification metrics
        scene_preds = np.array(self.scene_predictions)
        scene_targets = np.array(self.scene_targets)
        
        metrics["scene_accuracy"] = accuracy_score(scene_targets, scene_preds)
        metrics["scene_f1_macro"] = f1_score(scene_targets, scene_preds, average="macro")
        metrics["scene_f1_weighted"] = f1_score(scene_targets, scene_preds, average="weighted")
        metrics["scene_precision"] = precision_score(scene_targets, scene_preds, average="macro")
        metrics["scene_recall"] = recall_score(scene_targets, scene_preds, average="macro")
        
        # Object detection metrics (multi-label)
        object_preds = np.array(self.object_predictions)
        object_targets = np.array(self.object_targets)
        
        # Convert to binary predictions using threshold
        object_preds_binary = (object_preds > 0.5).astype(int)
        
        metrics["object_f1_macro"] = f1_score(
            object_targets.flatten(), 
            object_preds_binary.flatten(), 
            average="macro"
        )
        metrics["object_f1_micro"] = f1_score(
            object_targets.flatten(), 
            object_preds_binary.flatten(), 
            average="micro"
        )
        metrics["object_precision"] = precision_score(
            object_targets.flatten(), 
            object_preds_binary.flatten(), 
            average="macro"
        )
        metrics["object_recall"] = recall_score(
            object_targets.flatten(), 
            object_preds_binary.flatten(), 
            average="macro"
        )
        
        # Per-class metrics for objects
        for i in range(self.num_object_classes):
            if np.sum(object_targets[:, i]) > 0:  # Only if class exists in targets
                class_f1 = f1_score(object_targets[:, i], object_preds_binary[:, i])
                metrics[f"object_f1_class_{i}"] = class_f1
        
        return metrics


class RetrievalMetrics:
    """Metrics for cross-modal retrieval tasks."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.image_to_text_similarities = []
        self.text_to_image_similarities = []
    
    def update(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> None:
        """Update with new feature similarities.
        
        Args:
            image_features: Image feature embeddings.
            text_features: Text feature embeddings.
        """
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrices
        i2t_sim = torch.matmul(image_features, text_features.T)
        t2i_sim = torch.matmul(text_features, image_features.T)
        
        self.image_to_text_similarities.append(i2t_sim.cpu().numpy())
        self.text_to_image_similarities.append(t2i_sim.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute retrieval metrics.
        
        Returns:
            Dictionary containing retrieval metrics.
        """
        if not self.image_to_text_similarities:
            return {}
        
        # Concatenate all similarities
        i2t_sim = np.concatenate(self.image_to_text_similarities, axis=0)
        t2i_sim = np.concatenate(self.text_to_image_similarities, axis=0)
        
        metrics = {}
        
        # Image-to-text retrieval
        metrics["i2t_recall@1"] = self._compute_recall_at_k(i2t_sim, k=1)
        metrics["i2t_recall@5"] = self._compute_recall_at_k(i2t_sim, k=5)
        metrics["i2t_recall@10"] = self._compute_recall_at_k(i2t_sim, k=10)
        metrics["i2t_median_rank"] = self._compute_median_rank(i2t_sim)
        
        # Text-to-image retrieval
        metrics["t2i_recall@1"] = self._compute_recall_at_k(t2i_sim, k=1)
        metrics["t2i_recall@5"] = self._compute_recall_at_k(t2i_sim, k=5)
        metrics["t2i_recall@10"] = self._compute_recall_at_k(t2i_sim, k=10)
        metrics["t2i_median_rank"] = self._compute_median_rank(t2i_sim)
        
        return metrics
    
    def _compute_recall_at_k(self, similarities: np.ndarray, k: int) -> float:
        """Compute recall@k metric.
        
        Args:
            similarities: Similarity matrix.
            k: Number of top results to consider.
            
        Returns:
            Recall@k value.
        """
        batch_size = similarities.shape[0]
        ranks = np.argsort(-similarities, axis=1)
        
        recall_at_k = 0
        for i in range(batch_size):
            if i in ranks[i, :k]:
                recall_at_k += 1
        
        return recall_at_k / batch_size
    
    def _compute_median_rank(self, similarities: np.ndarray) -> float:
        """Compute median rank metric.
        
        Args:
            similarities: Similarity matrix.
            
        Returns:
            Median rank value.
        """
        batch_size = similarities.shape[0]
        ranks = np.argsort(-similarities, axis=1)
        
        ranks_list = []
        for i in range(batch_size):
            rank = np.where(ranks[i] == i)[0][0] + 1  # 1-indexed
            ranks_list.append(rank)
        
        return np.median(ranks_list)


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score for text generation.
    
    Args:
        predictions: Generated text predictions.
        references: Ground truth text references.
        
    Returns:
        BLEU score.
    """
    try:
        from sacrebleu import BLEU
        bleu = BLEU()
        score = bleu.corpus_score(predictions, [references])
        return score.score / 100.0  # Normalize to [0, 1]
    except ImportError:
        # Fallback implementation
        return 0.0


def compute_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores for text generation.
    
    Args:
        predictions: Generated text predictions.
        references: Ground truth text references.
        
    Returns:
        Dictionary containing ROUGE scores.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores['rouge1'] += score['rouge1'].fmeasure
            scores['rouge2'] += score['rouge2'].fmeasure
            scores['rougeL'] += score['rougeL'].fmeasure
        
        # Average over all samples
        num_samples = len(predictions)
        for key in scores:
            scores[key] /= num_samples
        
        return scores
    except ImportError:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
