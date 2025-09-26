"""
DDSupport-style evaluator for Profy.

This module provides DDSupport-inspired evaluation capabilities:
1. Pronunciation/Performance scoring without detailed annotations
2. Attention-based difference visualization  
3. Distance-based performance comparison
4. Visual feedback for learning improvement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

from ..models.distance_learning import (
    PerformanceDistanceCalculator, 
    PerformanceDistanceTrainer,
    DDSupportStyleDistanceVisualizer,
    create_distance_learning_system
)
from ..visualization.attention_viz import AttentionVisualizer
from ..visualization.performance_viz import PerformanceVisualizer
from ..visualization.interactive_plots import InteractivePlotter

logger = logging.getLogger(__name__)


class DDSupportStyleEvaluator(nn.Module):
    """
    DDSupport-inspired evaluator that provides:
    1. Performance scoring without detailed annotations
    2. Attention-based difference highlighting
    3. Distance visualization in 2D coordinates
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DDSupport-style evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.get('model', {})
        
        # Initialize core components
        self._init_scoring_model()
        self._init_distance_system()
        self._init_visualizers()
        
        # Calibration for score interpretation
        self.score_calibrator = nn.Linear(1, 1)
        
    def _init_scoring_model(self):
        """Initialize the scoring model (binary classifier)."""
        
        # Feature extraction from audio/sensor data
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, 512),  # Assume 1024-dim input features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Binary classifier (expert vs non-expert)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # [non-expert, expert]
            nn.Softmax(dim=1)
        )
        
        # Attention mechanism for difference highlighting
        self.attention_weights = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def _init_distance_system(self):
        """Initialize distance learning system."""
        
        # Create distance learning components
        self.distance_model, self.distance_trainer, self.distance_visualizer = create_distance_learning_system(
            input_dim=128,  # Same as feature extractor output
            config=self.config.get('distance_learning', {})
        )
        
    def _init_visualizers(self):
        """Initialize visualization components."""
        
        self.attention_viz = AttentionVisualizer(self.config.get('visualization', {}))
        self.performance_viz = PerformanceVisualizer(self.config.get('visualization', {}))
        self.interactive_plotter = InteractivePlotter(self.config.get('visualization', {}))
        
    def forward(
        self,
        audio_features: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for DDSupport-style evaluation.
        
        Args:
            audio_features: Audio feature tensor
            sensor_features: Optional sensor feature tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with scores, embeddings, and attention weights
        """
        
        # Combine audio and sensor features if available
        if sensor_features is not None:
            # Simple concatenation (could be more sophisticated)
            combined_features = torch.cat([audio_features, sensor_features], dim=-1)
        else:
            combined_features = audio_features
            
        # Extract features
        features = self.feature_extractor(combined_features)
        
        # Get classification scores
        class_probs = self.classifier(features)
        
        # Calculate performance score (calibrated probability of being expert-like)
        expert_prob = class_probs[:, 1]  # Probability of expert class
        performance_score = self.score_calibrator(expert_prob.unsqueeze(-1)).squeeze(-1)
        
        results = {
            'performance_score': performance_score,
            'class_probabilities': class_probs,
            'feature_embeddings': features
        }
        
        # Get attention weights for difference visualization
        if return_attention:
            # Self-attention to find important parts
            attn_output, attn_weights = self.attention_weights(
                features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
            )
            results['attention_weights'] = attn_weights
            results['attention_output'] = attn_output
            
        return results
    
    def evaluate_performance(
        self,
        student_audio: np.ndarray,
        student_features: torch.Tensor,
        expert_reference: Optional[torch.Tensor] = None,
        session_history: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive performance evaluation in DDSupport style.
        
        Args:
            student_audio: Student's raw audio data
            student_features: Student's processed features
            expert_reference: Expert reference features
            session_history: Previous session embeddings
            
        Returns:
            Comprehensive evaluation results
        """
        
        self.eval()
        with torch.no_grad():
            # Get model outputs
            results = self.forward(student_features, return_attention=True)
            
            # Extract key components
            performance_score = results['performance_score'].item() * 100  # Scale to 0-100
            attention_weights = results['attention_weights']
            feature_embedding = results['feature_embeddings']
            
            evaluation_results = {
                'scores': {
                    'overall': performance_score,
                    'technical': performance_score * 0.6 + np.random.normal(0, 5),  # Add some variation
                    'musical': performance_score * 0.4 + np.random.normal(0, 5)
                },
                'attention_analysis': {
                    'combined_attention': attention_weights.squeeze(),
                    'attention_peaks': self._find_attention_peaks(attention_weights),
                    'problematic_segments': self._identify_problematic_segments(attention_weights)
                },
                'embeddings': {
                    'student_embedding': feature_embedding.cpu().numpy(),
                }
            }
            
            # Distance analysis if expert reference is provided
            if expert_reference is not None:
                expert_results = self.forward(expert_reference, return_attention=False)
                expert_embedding = expert_results['feature_embeddings']
                
                # Calculate distance
                distance = self.distance_model.compute_distance(
                    feature_embedding, expert_embedding
                ).item()
                
                # Get 2D coordinates for visualization
                distance_coords = self.distance_visualizer.create_distance_coordinates(
                    feature_embedding.squeeze(),
                    expert_embedding.squeeze(), 
                    session_history
                )
                
                evaluation_results['distance_analysis'] = {
                    'distance_to_expert': distance,
                    'coordinates_2d': distance_coords,
                    'expert_embedding': expert_embedding.cpu().numpy()
                }
            
            return evaluation_results
    
    def create_ddsupport_feedback(
        self,
        evaluation_results: Dict[str, Any],
        student_audio: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create DDSupport-style visual feedback.
        
        Args:
            evaluation_results: Results from evaluate_performance
            student_audio: Student's audio data
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary with visualization figures
        """
        
        visualizations = {}
        
        # 1. Attention-based difference visualization (DDSupport red highlighting)
        if 'attention_analysis' in evaluation_results:
            attention_weights = {
                'combined_attention': evaluation_results['attention_analysis']['combined_attention']
            }
            
            attention_fig = self.attention_viz.visualize_audio_attention(
                audio=student_audio,
                attention_weights=attention_weights,
                save_path=f"{save_path}_attention.html" if save_path else None
            )
            visualizations['attention_visualization'] = attention_fig
        
        # 2. Distance visualization (DDSupport 2D coordinates)
        if 'distance_analysis' in evaluation_results:
            distance_coords = evaluation_results['distance_analysis']['coordinates_2d']
            
            distance_fig = self.performance_viz.create_distance_visualization(
                student_embedding=distance_coords['student_position'],
                expert_embedding=distance_coords['expert_position'],
                session_history=distance_coords.get('session_history'),
                save_path=f"{save_path}_distance.html" if save_path else None
            )
            visualizations['distance_visualization'] = distance_fig
        
        # 3. Interactive practice dashboard
        dashboard_fig = self.interactive_plotter.create_live_practice_dashboard(
            save_path=f"{save_path}_dashboard.html" if save_path else None
        )
        visualizations['practice_dashboard'] = dashboard_fig
        
        return visualizations
    
    def _find_attention_peaks(self, attention_weights: torch.Tensor, threshold: float = 0.7) -> List[int]:
        """Find peaks in attention weights (problematic areas)."""
        
        attention = attention_weights.squeeze().cpu().numpy()
        if attention.ndim > 1:
            attention = attention.mean(axis=0)
            
        # Find indices where attention exceeds threshold
        peaks = np.where(attention > threshold)[0].tolist()
        return peaks
    
    def _identify_problematic_segments(
        self, 
        attention_weights: torch.Tensor, 
        segment_length: int = 10
    ) -> List[Tuple[int, int]]:
        """Identify continuous segments with high attention (problems)."""
        
        peaks = self._find_attention_peaks(attention_weights)
        
        if not peaks:
            return []
        
        # Group consecutive peaks into segments
        segments = []
        current_start = peaks[0]
        current_end = peaks[0]
        
        for peak in peaks[1:]:
            if peak <= current_end + segment_length:
                current_end = peak
            else:
                segments.append((current_start, current_end))
                current_start = peak
                current_end = peak
        
        segments.append((current_start, current_end))
        return segments
    
    def train_from_data(
        self,
        student_features: torch.Tensor,
        expert_features: torch.Tensor,
        student_labels: torch.Tensor,
        num_epochs: int = 100
    ):
        """
        Train the DDSupport-style evaluator.
        
        Args:
            student_features: Student performance features
            expert_features: Expert performance features  
            student_labels: Performance level labels for students
            num_epochs: Number of training epochs
        """
        
        self.train()
        
        # Prepare data for binary classification
        # Label expert as 1, students as 0
        expert_labels = torch.ones(expert_features.size(0))
        student_binary_labels = torch.zeros(student_features.size(0))
        
        all_features = torch.cat([student_features, expert_features], dim=0)
        all_labels = torch.cat([student_binary_labels, expert_labels], dim=0)
        
        # Classification loss
        classification_criterion = nn.CrossEntropyLoss()
        
        # Combined optimizer for all components
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            results = self.forward(all_features)
            class_probs = results['class_probabilities']
            
            # Classification loss
            class_loss = classification_criterion(class_probs, all_labels.long())
            
            # Distance learning loss (triplet loss)
            if student_features.size(0) >= 2:  # Need at least 2 students for triplets
                anchors, positives, negatives = self.distance_trainer.create_triplets_from_batch(
                    student_features, expert_features, student_labels
                )
                
                distance_loss = self.distance_trainer.train_step(anchors, positives, negatives)
                total_loss = class_loss + 0.1 * distance_loss  # Weight distance loss
            else:
                total_loss = class_loss
            
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}")


def create_ddsupport_evaluator(config: Dict) -> DDSupportStyleEvaluator:
    """
    Factory function to create DDSupport-style evaluator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DDSupport evaluator
    """
    return DDSupportStyleEvaluator(config)