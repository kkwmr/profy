"""
Evaluation metrics for piano performance assessment.
"""

import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score, 
                           f1_score, precision_recall_fscore_support, confusion_matrix,
                           roc_auc_score, classification_report)
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple, Optional, Union


class PerformanceMetrics:
    """Performance metrics calculator for piano evaluation."""
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Calculate all available metrics."""
        metrics = {}
        metrics.update(compute_regression_metrics(y_true, y_pred))
        return metrics


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """
    Compute regression metrics for continuous scores.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'pearson_r': pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0,
        'spearman_r': spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
    }
    
    # Handle NaN values
    for key, value in metrics.items():
        if np.isnan(value):
            metrics[key] = 0.0
    
    return metrics


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute classification metrics for categorical predictions.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of comprehensive metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Precision and recall
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f
    
    # Macro averages
    metrics['precision_macro'] = np.mean(precision)
    metrics['recall_macro'] = np.mean(recall)
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC AUC (if probabilities available)
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception:
            metrics['roc_auc'] = 0.0
    
    # Handle NaN values
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and np.isnan(value):
            metrics[key] = 0.0
    
    return metrics