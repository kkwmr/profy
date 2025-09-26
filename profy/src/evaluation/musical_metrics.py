"""
Musical evaluation metrics for comprehensive performance assessment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class MusicalMetrics:
    """Compute musically meaningful evaluation metrics."""
    
    def __init__(self):
        """Initialize musical metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators."""
        self.predictions = []
        self.targets = []
        self.timing_errors = []
        self.dynamic_variations = []
        self.legato_qualities = []
        self.note_accuracies = []
        
    def update(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        note_features: Optional[torch.Tensor] = None,
        sensor_data: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            note_features: Note-level features if available
            sensor_data: Raw sensor data for detailed analysis
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        # Extract musical features if data available
        if sensor_data is not None:
            self._extract_timing_metrics(sensor_data)
            self._extract_dynamic_metrics(sensor_data)
            
        if note_features is not None:
            self._extract_note_metrics(note_features)
    
    def _extract_timing_metrics(self, sensor_data: torch.Tensor):
        """Extract timing-related metrics from sensor data."""
        # Analyze inter-onset intervals
        for sample in sensor_data:
            # Find note onsets (when keys are pressed)
            key_presses = (sample > 0.1).float()  # Threshold for key press
            onsets = torch.diff(key_presses, dim=0)
            onset_times = torch.where(onsets > 0)[0]
            
            if len(onset_times) > 1:
                # Calculate inter-onset interval variation
                ioi = torch.diff(onset_times.float())
                if len(ioi) > 0:
                    cv = torch.std(ioi) / (torch.mean(ioi) + 1e-7)
                    self.timing_errors.append(cv.item())
    
    def _extract_dynamic_metrics(self, sensor_data: torch.Tensor):
        """Extract dynamics-related metrics from sensor data."""
        for sample in sensor_data:
            # Analyze key press depths (velocity proxy)
            max_depths = torch.max(sample, dim=0)[0]
            pressed_keys = max_depths > 0.1
            
            if pressed_keys.sum() > 0:
                depths = max_depths[pressed_keys]
                # Calculate dynamic consistency
                cv = torch.std(depths) / (torch.mean(depths) + 1e-7)
                self.dynamic_variations.append(cv.item())
    
    def _extract_note_metrics(self, note_features: torch.Tensor):
        """Extract metrics from note features."""
        # Note features contain pre-computed musical metrics
        # Extract legato quality (assuming it's in the feature vector)
        for features in note_features:
            # Legato score is typically in position 15 of our feature vector
            if len(features) > 15:
                self.legato_qualities.append(features[15].item())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        metrics = {}
        
        # Basic classification metrics
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Handle both binary and multi-class
        if len(np.unique(targets)) == 2:
            # Binary classification
            metrics['accuracy'] = accuracy_score(targets, preds)
            metrics['f1_score'] = f1_score(targets, preds, average='binary')
            
            # Add AUC if predictions are probabilities
            if preds.min() >= 0 and preds.max() <= 1:
                try:
                    metrics['auc'] = roc_auc_score(targets, preds)
                except:
                    pass
        else:
            # Multi-class
            metrics['accuracy'] = accuracy_score(targets, preds)
            metrics['f1_macro'] = f1_score(targets, preds, average='macro')
            metrics['f1_weighted'] = f1_score(targets, preds, average='weighted')
        
        # Musical metrics
        if self.timing_errors:
            metrics['timing_precision'] = 1.0 / (1.0 + np.mean(self.timing_errors))
            metrics['timing_consistency'] = 1.0 / (1.0 + np.std(self.timing_errors))
        
        if self.dynamic_variations:
            metrics['dynamic_consistency'] = 1.0 / (1.0 + np.mean(self.dynamic_variations))
            metrics['dynamic_range'] = np.std(self.dynamic_variations)
        
        if self.legato_qualities:
            metrics['legato_quality'] = np.mean(self.legato_qualities)
            metrics['legato_consistency'] = 1.0 / (1.0 + np.std(self.legato_qualities))
        
        # Overall musical score (weighted combination)
        musical_scores = []
        if 'timing_precision' in metrics:
            musical_scores.append(metrics['timing_precision'])
        if 'dynamic_consistency' in metrics:
            musical_scores.append(metrics['dynamic_consistency'])
        if 'legato_quality' in metrics:
            musical_scores.append(metrics['legato_quality'])
        
        if musical_scores:
            metrics['overall_musicality'] = np.mean(musical_scores)
        
        # Add confusion matrix for detailed analysis
        cm = confusion_matrix(targets, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class accuracy
        for i in range(len(cm)):
            if cm[i].sum() > 0:
                metrics[f'class_{i}_accuracy'] = cm[i, i] / cm[i].sum()
        
        return metrics
    
    def get_detailed_report(self) -> str:
        """
        Generate a detailed evaluation report.
        
        Returns:
            Formatted string report
        """
        metrics = self.compute()
        
        if not metrics:
            return "No metrics computed yet."
        
        report = []
        report.append("=" * 60)
        report.append("MUSICAL PERFORMANCE EVALUATION REPORT")
        report.append("=" * 60)
        
        # Classification metrics
        report.append("\n### Classification Performance ###")
        if 'accuracy' in metrics:
            report.append(f"Accuracy: {metrics['accuracy']:.3f}")
        if 'f1_score' in metrics:
            report.append(f"F1 Score: {metrics['f1_score']:.3f}")
        if 'auc' in metrics:
            report.append(f"AUC-ROC: {metrics['auc']:.3f}")
        
        # Musical metrics
        report.append("\n### Musical Quality Metrics ###")
        if 'timing_precision' in metrics:
            report.append(f"Timing Precision: {metrics['timing_precision']:.3f}")
        if 'dynamic_consistency' in metrics:
            report.append(f"Dynamic Consistency: {metrics['dynamic_consistency']:.3f}")
        if 'legato_quality' in metrics:
            report.append(f"Legato Quality: {metrics['legato_quality']:.3f}")
        if 'overall_musicality' in metrics:
            report.append(f"Overall Musicality: {metrics['overall_musicality']:.3f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            report.append("\n### Confusion Matrix ###")
            cm = metrics['confusion_matrix']
            for i, row in enumerate(cm):
                report.append(f"Class {i}: {row}")
        
        # Per-class performance
        report.append("\n### Per-Class Performance ###")
        for key, value in metrics.items():
            if key.startswith('class_'):
                report.append(f"{key}: {value:.3f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class WeightedMusicalLoss(nn.Module):
    """Loss function that incorporates musical quality metrics."""
    
    def __init__(
        self, 
        classification_weight: float = 0.7,
        musical_weight: float = 0.3,
        num_classes: int = 2
    ):
        """
        Initialize weighted musical loss.
        
        Args:
            classification_weight: Weight for classification loss
            musical_weight: Weight for musical quality loss
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.musical_weight = musical_weight
        self.num_classes = num_classes
        
        # Base classification loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        note_features: Optional[torch.Tensor] = None,
        attention_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss combining classification and musical quality.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            note_features: Note-level features for musical quality assessment
            attention_weights: Attention weights for regularization
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        losses = {}
        
        # Classification loss
        cls_loss = self.ce_loss(predictions, targets)
        losses['classification'] = cls_loss.item()
        
        total_loss = self.classification_weight * cls_loss
        
        # Musical quality losses (if note features available)
        if note_features is not None and self.musical_weight > 0:
            # Timing consistency loss (encourage consistent timing)
            timing_features = note_features[:, 8:12]  # IOI statistics
            timing_var = torch.var(timing_features, dim=1).mean()
            timing_loss = timing_var
            losses['timing'] = timing_loss.item()
            
            # Dynamic consistency loss
            dynamic_features = note_features[:, 12:16]  # Depth statistics
            dynamic_var = torch.var(dynamic_features, dim=1).mean()
            dynamic_loss = dynamic_var
            losses['dynamics'] = dynamic_loss.item()
            
            # Combine musical losses
            musical_loss = (timing_loss + dynamic_loss) / 2
            total_loss += self.musical_weight * musical_loss
            losses['musical'] = musical_loss.item()
        
        # Attention regularization (encourage focused attention)
        if attention_weights is not None:
            if 'audio' in attention_weights:
                audio_entropy = -torch.sum(
                    attention_weights['audio'] * torch.log(attention_weights['audio'] + 1e-10),
                    dim=-1
                ).mean()
                attention_reg = audio_entropy * 0.01  # Small regularization
                total_loss += attention_reg
                losses['attention_reg'] = attention_reg.item()
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses