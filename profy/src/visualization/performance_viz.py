"""
Performance visualization module for Profy - DDSupport inspired implementation.

This module provides visualization tools for:
1. Distance visualization between student and expert performance (2D coordinates)
2. Score progression tracking
3. Performance improvement metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime


class PerformanceVisualizer:
    """
    Visualizes performance metrics and distances between student and expert performances,
    inspired by DDSupport's distance visualization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize performance visualizer.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or {}
        self.expert_center = (0, 0)  # Expert is at center like DDSupport
        
    def create_distance_visualization(
        self,
        student_embedding: np.ndarray,
        expert_embedding: np.ndarray,
        session_history: Optional[List[np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create DDSupport-style 2D distance visualization showing how far student is from expert.
        
        Args:
            student_embedding: Current student performance embedding
            expert_embedding: Expert performance embedding (reference)
            session_history: List of previous student embeddings to show progress
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure with distance visualization
        """
        
        # Reduce to 2D if needed (using PCA or t-SNE-like approach)
        if student_embedding.shape[0] > 2:
            student_2d = self._reduce_to_2d([student_embedding, expert_embedding])[0]
            expert_2d = self._reduce_to_2d([student_embedding, expert_embedding])[1]
        else:
            student_2d = student_embedding[:2]
            expert_2d = expert_embedding[:2]
        
        # Center expert at origin (DDSupport style)
        offset = expert_2d
        student_2d = student_2d - offset
        expert_2d = expert_2d - offset  # Now at (0, 0)
        
        fig = go.Figure()
        
        # Add expert point at center (red dot like DDSupport)
        fig.add_trace(
            go.Scatter(
                x=[expert_2d[0]],
                y=[expert_2d[1]],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='circle',
                    line=dict(width=2, color='darkred')
                ),
                name='Expert Reference',
                hovertemplate='Expert Reference<br>Position: (0, 0)<extra></extra>'
            )
        )
        
        # Add student point (blue dot like DDSupport)
        distance = np.linalg.norm(student_2d - expert_2d)
        fig.add_trace(
            go.Scatter(
                x=[student_2d[0]],
                y=[student_2d[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='blue',
                    symbol='circle',
                    line=dict(width=2, color='darkblue')
                ),
                name='Student Performance',
                hovertemplate=f'Student Performance<br>Distance from Expert: {distance:.3f}<br>Position: ({student_2d[0]:.3f}, {student_2d[1]:.3f})<extra></extra>'
            )
        )
        
        # Add trajectory if session history is provided
        if session_history:
            history_2d = []
            for embedding in session_history:
                if embedding.shape[0] > 2:
                    emb_2d = self._reduce_to_2d([embedding, expert_embedding + offset])[0]
                else:
                    emb_2d = embedding[:2]
                history_2d.append(emb_2d - offset)  # Center relative to expert
            
            history_2d = np.array(history_2d)
            
            # Add trajectory line
            fig.add_trace(
                go.Scatter(
                    x=history_2d[:, 0],
                    y=history_2d[:, 1],
                    mode='lines+markers',
                    line=dict(color='lightblue', width=2, dash='dash'),
                    marker=dict(size=6, color='lightblue', opacity=0.7),
                    name='Progress Trajectory',
                    hovertemplate='Attempt %{pointNumber}<br>Position: (%{x:.3f}, %{y:.3f})<extra></extra>'
                )
            )
        
        # Add distance circle to show current distance
        fig.add_shape(
            type="circle",
            x0=-distance, y0=-distance,
            x1=distance, y1=distance,
            line=dict(color="gray", width=1, dash="dot"),
            opacity=0.3
        )
        
        # Add distance line from expert to student
        fig.add_shape(
            type="line",
            x0=expert_2d[0], y0=expert_2d[1],
            x1=student_2d[0], y1=student_2d[1],
            line=dict(color="orange", width=2, dash="solid"),
            opacity=0.7
        )
        
        # Style the plot (DDSupport-like)
        fig.update_layout(
            title=f"Performance Distance Visualization<br>Current Distance: {distance:.3f}",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=600,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            ),
            yaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_score_progression(
        self,
        session_scores: List[Dict[str, float]],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create score progression visualization showing improvement over time.
        
        Args:
            session_scores: List of score dictionaries for each attempt
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure with score progression
        """
        
        if not session_scores:
            return go.Figure()
        
        attempts = list(range(1, len(session_scores) + 1))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Overall Score Progression',
                'Technical vs Musical Scores',
                'Detailed Technical Breakdown',
                'Detailed Musical Breakdown'
            ]
        )
        
        # Overall score progression
        overall_scores = [scores.get('overall', 0) for scores in session_scores]
        fig.add_trace(
            go.Scatter(
                x=attempts,
                y=overall_scores,
                mode='lines+markers',
                name='Overall Score',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Technical vs Musical
        technical_scores = [scores.get('technical', 0) for scores in session_scores]
        musical_scores = [scores.get('musical', 0) for scores in session_scores]
        
        fig.add_trace(
            go.Scatter(
                x=attempts,
                y=technical_scores,
                mode='lines+markers',
                name='Technical Score',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=attempts,
                y=musical_scores,
                mode='lines+markers',
                name='Musical Score',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Technical breakdown
        technical_metrics = ['technical_pitch', 'technical_rhythm', 'technical_tempo', 'technical_finger', 'technical_touch']
        colors_tech = ['navy', 'royalblue', 'skyblue', 'steelblue', 'lightsteelblue']
        
        for i, metric in enumerate(technical_metrics):
            values = [scores.get(metric, 0) for scores in session_scores]
            if any(v > 0 for v in values):  # Only plot if we have data
                fig.add_trace(
                    go.Scatter(
                        x=attempts,
                        y=values,
                        mode='lines+markers',
                        name=metric.replace('technical_', '').title(),
                        line=dict(color=colors_tech[i % len(colors_tech)], width=1.5),
                        marker=dict(size=4)
                    ),
                    row=2, col=1
                )
        
        # Musical breakdown
        musical_metrics = ['musical_dynamics', 'musical_articulation', 'musical_phrasing', 'musical_interpretation']
        colors_mus = ['darkred', 'crimson', 'lightcoral', 'pink']
        
        for i, metric in enumerate(musical_metrics):
            values = [scores.get(metric, 0) for scores in session_scores]
            if any(v > 0 for v in values):  # Only plot if we have data
                fig.add_trace(
                    go.Scatter(
                        x=attempts,
                        y=values,
                        mode='lines+markers',
                        name=metric.replace('musical_', '').title(),
                        line=dict(color=colors_mus[i % len(colors_mus)], width=1.5),
                        marker=dict(size=4)
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Performance Score Progression",
            height=800,
            showlegend=True
        )
        
        # Update axes
        for row in [1, 2]:
            for col in [1, 2]:
                fig.update_xaxes(title_text="Attempt Number", row=row, col=col)
                fig.update_yaxes(title_text="Score", range=[0, 100], row=row, col=col)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_performance_radar(
        self,
        current_scores: Dict[str, float],
        target_scores: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create radar chart showing performance across different dimensions.
        
        Args:
            current_scores: Current performance scores
            target_scores: Optional target scores to compare against
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure with radar chart
        """
        
        # Define performance dimensions
        technical_dims = ['technical_pitch', 'technical_rhythm', 'technical_tempo', 'technical_finger', 'technical_touch']
        musical_dims = ['musical_dynamics', 'musical_articulation', 'musical_phrasing', 'musical_interpretation']
        
        all_dims = technical_dims + musical_dims
        dim_labels = [dim.replace('technical_', '').replace('musical_', '').title() for dim in all_dims]
        
        # Get current values
        current_values = [current_scores.get(dim, 0) for dim in all_dims]
        current_values += [current_values[0]]  # Close the radar chart
        
        fig = go.Figure()
        
        # Add current performance
        fig.add_trace(go.Scatterpolar(
            r=current_values,
            theta=dim_labels + [dim_labels[0]],
            fill='toself',
            name='Current Performance',
            line=dict(color='blue'),
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))
        
        # Add target performance if provided
        if target_scores:
            target_values = [target_scores.get(dim, 100) for dim in all_dims]
            target_values += [target_values[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=target_values,
                theta=dim_labels + [dim_labels[0]],
                fill='toself',
                name='Target Performance',
                line=dict(color='red', dash='dash'),
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Performance Radar Chart"
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def _reduce_to_2d(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Reduce high-dimensional embeddings to 2D using PCA.
        
        Args:
            embeddings: List of embeddings to reduce
            
        Returns:
            List of 2D embeddings
        """
        from sklearn.decomposition import PCA
        
        # Stack embeddings
        stacked = np.vstack(embeddings)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(stacked)
        
        # Split back into individual embeddings
        result = []
        start_idx = 0
        for emb in embeddings:
            end_idx = start_idx + 1
            result.append(reduced[start_idx:end_idx].flatten())
            start_idx = end_idx
            
        return result
    
    def create_improvement_summary(
        self,
        session_data: List[Dict],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive improvement summary visualization.
        
        Args:
            session_data: List of session data dictionaries
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure with improvement summary
        """
        
        if len(session_data) < 2:
            return go.Figure()
        
        # Calculate improvements
        first_session = session_data[0]
        last_session = session_data[-1]
        
        improvements = {}
        for key in first_session.get('scores', {}):
            if key in last_session.get('scores', {}):
                improvement = last_session['scores'][key] - first_session['scores'][key]
                improvements[key] = improvement
        
        # Create bar chart of improvements
        fig = go.Figure()
        
        metrics = list(improvements.keys())
        values = list(improvements.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                name='Score Improvement'
            )
        )
        
        fig.update_layout(
            title=f"Performance Improvement Summary ({len(session_data)} sessions)",
            xaxis_title="Performance Metrics",
            yaxis_title="Score Change",
            xaxis_tickangle=-45
        )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig