"""
Visualization module for Profy

Contains visualization tools for:
- Attention weight visualization
- Performance analysis plots
- Score and feedback displays
- Interactive web visualizations
"""

from .attention_viz import AttentionVisualizer
from .performance_viz import PerformanceVisualizer
from .modality_weight_plots import (
    plot_modality_weight_distribution,
    plot_directory_distributions,
)
# from .interactive_plots import InteractivePlotter

__all__ = [
    "AttentionVisualizer",
    "PerformanceVisualizer", 
    "InteractivePlotter",
    "plot_modality_weight_distribution",
    "plot_directory_distributions",
]
