"""
Attention visualization module for Profy - DDSupport inspired implementation.

This module provides visualization tools to show:
1. Where in the audio the model focuses attention (difference visualization)
2. Attention heatmaps overlaid on spectrograms
3. Hierarchical attention across note/measure/phrase levels
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json


class AttentionVisualizer:
    """
    Visualizes attention weights to show where the model focuses 
    when evaluating piano performance, inspired by DDSupport.
    
    NOTE: This is the legacy visualization. For new MERT-based models,
    use MERTAttentionVisualizer from mert_attention_viz.py instead.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize attention visualizer.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or {}
        self.attention_threshold = self.config.get('attention_threshold', 0.5)
        self.sample_rate = self.config.get('sample_rate', 24000)  # Updated for MERT
        
    def visualize_audio_attention(
        self,
        audio: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        timestamps: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create DDSupport-style visualization showing where attention focuses on audio.
        
        Args:
            audio: Audio waveform data
            attention_weights: Dictionary of attention weights by level
            timestamps: Time stamps for attention weights
            save_path: Optional path to save the visualization
            
        Returns:
            Plotly figure with attention visualization
        """
        if timestamps is None:
            timestamps = np.linspace(0, len(audio) / self.sample_rate, len(audio))
            
        # Create subplot figure
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Audio Waveform with Attention Overlay',
                'Note-level Attention',
                'Measure-level Attention', 
                'Phrase-level Attention'
            ],
            vertical_spacing=0.08
        )
        
        # 1. Audio waveform with attention overlay (DDSupport style)
        self._add_waveform_with_attention(fig, audio, attention_weights, timestamps, row=1)
        
        # 2. Hierarchical attention levels
        attention_levels = ['note_attention', 'measure_attention', 'phrase_attention']
        for i, level in enumerate(attention_levels, start=2):
            if level in attention_weights:
                self._add_attention_heatmap(fig, attention_weights[level], level, row=i)
        
        # Style the figure
        fig.update_layout(
            title="Performance Attention Analysis (DDSupport Style)",
            height=800,
            showlegend=True,
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def _add_waveform_with_attention(
        self,
        fig: go.Figure,
        audio: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        timestamps: np.ndarray,
        row: int
    ):
        """Add waveform with attention overlay similar to DDSupport's red highlighting."""
        
        # Normalize audio for display
        audio_norm = audio / np.max(np.abs(audio))
        time_axis = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        
        # Add base waveform
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=audio_norm,
                mode='lines',
                name='Audio Waveform',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=row, col=1
        )
        
        # Add attention overlay (DDSupport-style red highlighting)
        if 'combined_attention' in attention_weights:
            attention = attention_weights['combined_attention'].cpu().numpy()
            
            # Interpolate attention to match audio length
            attention_interp = np.interp(time_axis, timestamps, attention)
            
            # Create attention mask (above threshold gets highlighted)
            attention_mask = attention_interp > self.attention_threshold
            
            # Add red overlay for high attention areas
            for i in range(len(attention_mask) - 1):
                if attention_mask[i]:
                    # Add red background for problematic areas
                    fig.add_shape(
                        type="rect",
                        x0=time_axis[i],
                        x1=time_axis[i+1],
                        y0=-1,
                        y1=1,
                        fillcolor="red",
                        opacity=min(0.3 + attention_interp[i] * 0.4, 0.7),  # Darker for higher attention
                        line=dict(width=0),
                        row=row, col=1
                    )
        
        # Style the subplot
        fig.update_xaxes(title_text="Time (s)", row=row, col=1)
        fig.update_yaxes(title_text="Amplitude", row=row, col=1)
    
    def _add_attention_heatmap(
        self,
        fig: go.Figure,
        attention: torch.Tensor,
        level_name: str,
        row: int
    ):
        """Add attention heatmap for a specific hierarchical level."""
        
        attention_np = attention.cpu().numpy()
        if attention_np.ndim == 3:  # [batch, heads, seq_len]
            attention_np = attention_np.mean(axis=(0, 1))  # Average over batch and heads
        elif attention_np.ndim == 2:  # [heads, seq_len] 
            attention_np = attention_np.mean(axis=0)  # Average over heads
            
        fig.add_trace(
            go.Heatmap(
                z=attention_np.reshape(1, -1),
                colorscale='Reds',
                showscale=True,
                name=level_name.replace('_', ' ').title()
            ),
            row=row, col=1
        )
        
        fig.update_xaxes(title_text="Time Steps", row=row, col=1)
        fig.update_yaxes(title_text=level_name.replace('_', ' ').title(), row=row, col=1)
    
    def create_performance_difference_map(
        self,
        student_audio: np.ndarray,
        expert_audio: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a DDSupport-style difference visualization between student and expert performance.
        
        Args:
            student_audio: Student's audio performance
            expert_audio: Expert/reference audio performance  
            attention_weights: Attention weights indicating problematic areas
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure showing performance differences
        """
        
        # Ensure same length
        min_len = min(len(student_audio), len(expert_audio))
        student_audio = student_audio[:min_len]
        expert_audio = expert_audio[:min_len]
        
        time_axis = np.linspace(0, min_len / self.sample_rate, min_len)
        
        # Calculate difference
        audio_diff = np.abs(student_audio - expert_audio)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Expert Reference Performance',
                'Student Performance with Problem Areas',
                'Difference Analysis'
            ],
            vertical_spacing=0.1
        )
        
        # Expert reference
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=expert_audio / np.max(np.abs(expert_audio)),
                mode='lines',
                name='Expert Reference',
                line=dict(color='green', width=1.5)
            ),
            row=1, col=1
        )
        
        # Student with attention overlay (DDSupport style)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=student_audio / np.max(np.abs(student_audio)),
                mode='lines',
                name='Student Performance',
                line=dict(color='blue', width=1.5)
            ),
            row=2, col=1
        )
        
        # Add attention-based highlighting
        if 'combined_attention' in attention_weights:
            attention = attention_weights['combined_attention'].cpu().numpy()
            
            # Interpolate attention to audio length
            attention_interp = np.interp(time_axis, np.linspace(0, time_axis[-1], len(attention)), attention)
            
            # Highlight problem areas in red
            for i in range(len(attention_interp) - 1):
                if attention_interp[i] > self.attention_threshold:
                    fig.add_shape(
                        type="rect",
                        x0=time_axis[i],
                        x1=time_axis[i+1], 
                        y0=-1,
                        y1=1,
                        fillcolor="red",
                        opacity=0.3 + attention_interp[i] * 0.4,
                        line=dict(width=0),
                        row=2, col=1
                    )
        
        # Difference analysis
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=audio_diff / np.max(audio_diff),
                mode='lines',
                name='Absolute Difference',
                line=dict(color='red', width=1),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Performance Difference Analysis (DDSupport Style)",
            height=700,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_spectrogram_attention(
        self,
        audio: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create spectrogram with attention overlay for detailed frequency analysis.
        
        Args:
            audio: Audio waveform
            attention_weights: Attention weights to overlay
            save_path: Optional path to save visualization
            
        Returns:
            Plotly figure with spectrogram and attention
        """
        
        # Compute spectrogram
        D = librosa.stft(audio, hop_length=512, n_fft=2048)
        magnitude = np.abs(D)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Time and frequency axes
        times = librosa.frames_to_time(np.arange(magnitude_db.shape[1]), sr=self.sample_rate, hop_length=512)
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        
        fig = go.Figure()
        
        # Add spectrogram
        fig.add_trace(
            go.Heatmap(
                x=times,
                y=freqs[:magnitude_db.shape[0]],
                z=magnitude_db,
                colorscale='Viridis',
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.1f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>'
            )
        )
        
        # Overlay attention if available
        if 'combined_attention' in attention_weights:
            attention = attention_weights['combined_attention'].cpu().numpy().mean(axis=0) if attention_weights['combined_attention'].ndim > 1 else attention_weights['combined_attention'].cpu().numpy()
            
            # Interpolate attention to match spectrogram time axis
            attention_interp = np.interp(times, np.linspace(0, times[-1], len(attention)), attention)
            
            # Add attention contour
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=attention_interp * freqs[-1] * 0.1,  # Scale to frequency range
                    mode='lines',
                    name='Attention Weight',
                    line=dict(color='red', width=3),
                    yaxis='y2'
                )
            )
        
        fig.update_layout(
            title="Spectrogram with Attention Overlay",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            yaxis2=dict(
                title="Attention Weight",
                overlaying='y',
                side='right'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def save_attention_report(
        self,
        attention_weights: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save attention analysis report for further analysis.
        
        Args:
            attention_weights: Dictionary of attention weights
            output_path: Path to save the report
            metadata: Optional metadata to include
        """
        
        report = {
            'metadata': metadata or {},
            'attention_analysis': {},
            'summary_statistics': {}
        }
        
        # Process each attention level
        for level, weights in attention_weights.items():
            weights_np = weights.cpu().numpy()
            
            report['attention_analysis'][level] = {
                'shape': list(weights_np.shape),
                'mean': float(np.mean(weights_np)),
                'std': float(np.std(weights_np)),
                'max': float(np.max(weights_np)),
                'min': float(np.min(weights_np)),
                'above_threshold_ratio': float(np.mean(weights_np > self.attention_threshold))
            }
        
        # Calculate summary statistics
        if 'combined_attention' in attention_weights:
            combined = attention_weights['combined_attention'].cpu().numpy()
            report['summary_statistics'] = {
                'total_attention_peaks': int(np.sum(combined > self.attention_threshold)),
                'attention_concentration': float(np.std(combined)),
                'problematic_ratio': float(np.mean(combined > self.attention_threshold))
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)