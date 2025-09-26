"""
Evaluation module for Profy Piano Performance Evaluator.

This module provides utilities for evaluating model performance,
computing metrics, and generating reports.
"""

from .metrics import compute_regression_metrics, compute_classification_metrics
from .performance_result import PerformanceResult

__all__ = [
    'compute_regression_metrics',
    'compute_classification_metrics', 
    'PerformanceResult'
]