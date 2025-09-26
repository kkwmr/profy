"""
Interactive plotting module for Profy - DDSupport inspired real-time visualization.

This module provides interactive plotting capabilities for:
1. Real-time feedback during practice sessions
2. Interactive attention exploration
3. Live performance monitoring
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pc
import asyncio
import json
from datetime import datetime

# Import MERT attention visualizer for the new architecture
from .mert_attention_viz import MERTAttentionVisualizer


class InteractivePlotter:
    """
    Creates interactive visualizations for real-time feedback and exploration,
    inspired by DDSupport's interactive learning approach.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize interactive plotter.
        
        Args:
            config: Configuration dictionary for plotting settings
        """
        self.config = config or {}
        self.update_callbacks = []
        
    def create_live_practice_dashboard(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a live practice dashboard that can be updated in real-time.
        
        Args:
            save_path: Optional path to save dashboard
            
        Returns:
            Plotly figure with live dashboard layout
        """
        
        # Create subplots for different live metrics
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Live Audio with Attention',
                'Distance from Expert (DDSupport Style)',
                'Real-time Score Meter',
                'Attention Heatmap',
                'Progress Tracking',
                'Performance Feedback'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "indicator"}, {"type": "heatmap"}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Initialize empty traces that will be updated
        
        # 1. Live audio waveform (will be updated with real audio)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Live Audio',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # 2. Distance visualization (DDSupport style)
        fig.add_trace(
            go.Scatter(
                x=[0],  # Expert at center
                y=[0],
                mode='markers',
                marker=dict(size=20, color='red', symbol='circle'),
                name='Expert Reference'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0],  # Student position (will be updated)
                y=[0],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='circle'),
                name='Your Performance'
            ),
            row=1, col=2
        )
        
        # 3. Real-time score meter
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=1
        )
        
        # 4. Attention heatmap (will be updated with real attention)
        fig.add_trace(
            go.Heatmap(
                z=[[0]],
                colorscale='Reds',
                showscale=False,
                name='Attention'
            ),
            row=2, col=2
        )
        
        # 5. Progress tracking
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode='lines+markers',
                name='Score Progress',
                line=dict(color='green', width=3)
            ),
            row=3, col=1
        )
        
        # Style the dashboard
        fig.update_layout(
            title="Live Practice Dashboard (DDSupport Style)",
            height=900,
            showlegend=True,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * 6}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True, True, False, False, False, False]}],
                            label="Audio & Distance Only",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        
        fig.update_xaxes(title_text="Dimension 1", row=1, col=2)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=2)
        
        fig.update_xaxes(title_text="Attempt", row=3, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def update_live_dashboard(
        self,
        fig: go.Figure,
        audio_data: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        current_score: Dict[str, float],
        student_position: Tuple[float, float],
        session_history: List[float]
    ) -> go.Figure:
        """
        Update the live dashboard with new data.
        
        Args:
            fig: Existing dashboard figure
            audio_data: Current audio waveform
            attention_weights: Current attention weights
            current_score: Current performance scores
            student_position: Current student position in 2D space
            session_history: Historical scores
            
        Returns:
            Updated figure
        """
        
        with fig.batch_update():
            # Update audio waveform
            time_axis = np.linspace(0, len(audio_data) / 16000, len(audio_data))
            fig.data[0].x = time_axis
            fig.data[0].y = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
            
            # Update student position (DDSupport style)
            fig.data[2].x = [student_position[0]]
            fig.data[2].y = [student_position[1]]
            
            # Update score meter
            overall_score = current_score.get('overall', 0)
            fig.data[3].value = overall_score
            
            # Update attention heatmap
            if 'combined_attention' in attention_weights:
                attention = attention_weights['combined_attention'].cpu().numpy()
                if attention.ndim > 1:
                    attention = attention.mean(axis=0)
                fig.data[4].z = [attention.reshape(1, -1)]
            
            # Update progress tracking
            attempts = list(range(1, len(session_history) + 1))
            fig.data[5].x = attempts
            fig.data[5].y = session_history
        
        return fig
    
    def create_interactive_attention_explorer(
        self,
        audio: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        timestamps: np.ndarray,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive attention explorer with clickable regions.
        
        Args:
            audio: Audio waveform data
            attention_weights: Multi-level attention weights
            timestamps: Time stamps for attention
            save_path: Optional path to save visualization
            
        Returns:
            Interactive Plotly figure
        """
        
        time_axis = np.linspace(0, len(audio) / 16000, len(audio))
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Audio Waveform (Click to hear segment)',
                'Note-level Attention',
                'Measure-level Attention', 
                'Phrase-level Attention'
            ],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Audio waveform with click interaction
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio,
                mode='lines',
                name='Audio',
                line=dict(color='blue', width=1),
                hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>',
                customdata=audio  # Store audio data for interaction
            ),
            row=1, col=1
        )
        
        # Add attention overlays with different colors
        attention_levels = ['note_attention', 'measure_attention', 'phrase_attention']
        colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)']
        
        for i, (level, color) in enumerate(zip(attention_levels, colors)):
            if level in attention_weights:
                attention = attention_weights[level].cpu().numpy()
                if attention.ndim > 1:
                    attention = attention.mean(axis=0)
                
                # Interpolate attention to audio length
                attention_interp = np.interp(time_axis, timestamps, attention)
                
                # Add attention trace
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=attention_interp,
                        mode='lines',
                        name=level.replace('_', ' ').title(),
                        line=dict(width=2),
                        fill='tonexty' if i == 0 else None,
                        fillcolor=color,
                        hovertemplate=f'{level}: %{{y:.3f}}<extra></extra>'
                    ),
                    row=i+2, col=1
                )
        
        # Add interactive features
        fig.update_layout(
            title="Interactive Attention Explorer<br><sub>Click on waveform to explore different time segments</sub>",
            height=800,
            hovermode='x unified',
            showlegend=True
        )
        
        # Add custom JavaScript for audio playback (if needed)
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': [
                {
                    'name': 'Play Audio Segment',
                    'icon': {'width': 857.1, 'height': 1000, 'path': 'm857 350v300l-257-150z'},
                    'click': 'function(gd) { alert("Audio playback feature would be implemented here"); }'
                }
            ]
        }
        
        if save_path:
            fig.write_html(save_path, config=config)
            
        return fig
    
    def create_mert_attention_dashboard(
        self,
        model,
        audio: np.ndarray,
        sensor_data: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive dashboard for MERT attention analysis.
        
        Args:
            model: MERT-based model
            audio: Audio waveform
            sensor_data: Optional sensor data
            save_path: Optional path to save dashboard
            
        Returns:
            Interactive MERT attention dashboard
        """
        # Initialize MERT attention visualizer
        visualizer = MERTAttentionVisualizer()
        
        # Convert audio to tensor
        input_values = torch.FloatTensor(audio).unsqueeze(0)
        sensor_tensor = torch.FloatTensor(sensor_data).unsqueeze(0) if sensor_data is not None else None
        
        # Get model predictions and attention
        with torch.no_grad():
            outputs = model(input_values, sensor_tensor, return_attention=True)
            predictions = model.predict_performance(input_values, sensor_tensor)
        
        # Create interactive visualization
        if 'attentions' in outputs:
            attention_weights = outputs['attentions'][-1]  # Last layer attention
            fig = visualizer.create_interactive_attention_plot(
                audio=audio,
                attention_weights=attention_weights,
                predictions=predictions
            )
        else:
            # Fallback if no attention available
            fig = go.Figure()
            fig.add_annotation(
                text="No attention weights available from model",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Add model information
        fig.update_layout(
            title=f"MERT Attention Analysis - Overall Score: {predictions['overall_score']}",
            annotations=[
                dict(
                    text=f"Skill Level: {predictions['skill_level']} (Confidence: {predictions['skill_confidence']:.2f})",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    font=dict(size=12, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white", borderwidth=1
                )
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_comparison_slider(
        self,
        student_audio: np.ndarray,
        expert_audio: np.ndarray,
        attention_weights: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create slider-based comparison between student and expert performance.
        
        Args:
            student_audio: Student's audio performance
            expert_audio: Expert reference audio
            attention_weights: Attention weights for comparison
            save_path: Optional path to save visualization
            
        Returns:
            Interactive figure with slider
        """
        
        # Ensure same length
        min_len = min(len(student_audio), len(expert_audio))
        student_audio = student_audio[:min_len]
        expert_audio = expert_audio[:min_len]
        time_axis = np.linspace(0, min_len / 16000, min_len)
        
        # Create figure with slider
        fig = go.Figure()
        
        # Add expert trace (always visible)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=expert_audio / np.max(np.abs(expert_audio)),
                mode='lines',
                name='Expert Reference',
                line=dict(color='red', width=2),
                visible=True
            )
        )
        
        # Add student trace (controllable with slider)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=student_audio / np.max(np.abs(student_audio)),
                mode='lines',
                name='Student Performance',
                line=dict(color='blue', width=2),
                visible=True
            )
        )
        
        # Add difference trace
        difference = np.abs(student_audio - expert_audio)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=difference / np.max(difference),
                mode='lines',
                name='Difference',
                line=dict(color='orange', width=1),
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.3)',
                visible=False  # Hidden by default
            )
        )
        
        # Create slider for comparison modes
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "View: "},
                pad={"t": 50},
                steps=[
                    dict(
                        label="Both",
                        method="update",
                        args=[{"visible": [True, True, False]}]
                    ),
                    dict(
                        label="Expert Only",
                        method="update", 
                        args=[{"visible": [True, False, False]}]
                    ),
                    dict(
                        label="Student Only",
                        method="update",
                        args=[{"visible": [False, True, False]}]
                    ),
                    dict(
                        label="Difference",
                        method="update",
                        args=[{"visible": [True, True, True]}]
                    )
                ]
            )
        ]
        
        fig.update_layout(
            title="Performance Comparison with Interactive Controls",
            sliders=sliders,
            xaxis_title="Time (s)",
            yaxis_title="Normalized Amplitude",
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_gamified_progress_tracker(
        self,
        session_data: List[Dict],
        achievements: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create gamified progress tracker inspired by DDSupport's interactive learning.
        
        Args:
            session_data: List of session performance data
            achievements: Optional list of achievements/milestones
            save_path: Optional path to save visualization
            
        Returns:
            Gamified progress visualization
        """
        
        if not session_data:
            return go.Figure()
        
        # Extract data
        attempts = list(range(1, len(session_data) + 1))
        overall_scores = [s.get('scores', {}).get('overall', 0) for s in session_data]
        
        # Create main progress line
        fig = go.Figure()
        
        # Add progress line with varying colors based on improvement
        colors = []
        for i, score in enumerate(overall_scores):
            if i == 0:
                colors.append('blue')
            else:
                if score > overall_scores[i-1]:
                    colors.append('green')  # Improvement
                elif score == overall_scores[i-1]:
                    colors.append('orange')  # No change
                else:
                    colors.append('red')  # Decline
        
        # Add main progress trace
        fig.add_trace(
            go.Scatter(
                x=attempts,
                y=overall_scores,
                mode='lines+markers',
                name='Progress',
                line=dict(color='blue', width=3),
                marker=dict(
                    size=12,
                    color=colors,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                hovertemplate='Attempt: %{x}<br>Score: %{y:.1f}<br>Change: %{customdata}<extra></extra>',
                customdata=[f"+{overall_scores[i] - overall_scores[i-1]:.1f}" if i > 0 and overall_scores[i] > overall_scores[i-1] 
                           else f"{overall_scores[i] - overall_scores[i-1]:.1f}" if i > 0 
                           else "First attempt" for i in range(len(overall_scores))]
            )
        )
        
        # Add achievement markers if provided
        if achievements:
            for achievement in achievements:
                attempt = achievement.get('attempt', 0)
                if attempt <= len(attempts):
                    fig.add_annotation(
                        x=attempt,
                        y=overall_scores[attempt-1] if attempt > 0 else 0,
                        text=f"🏆 {achievement.get('name', 'Achievement')}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='gold',
                        bgcolor='gold',
                        bordercolor='orange',
                        font=dict(color='black', size=10)
                    )
        
        # Add progress zones
        fig.add_hrect(y0=0, y1=40, fillcolor="rgba(255,0,0,0.1)", annotation_text="Beginner", annotation_position="left")
        fig.add_hrect(y0=40, y1=70, fillcolor="rgba(255,255,0,0.1)", annotation_text="Intermediate", annotation_position="left")
        fig.add_hrect(y0=70, y1=85, fillcolor="rgba(0,255,0,0.1)", annotation_text="Advanced", annotation_position="left")
        fig.add_hrect(y0=85, y1=100, fillcolor="rgba(0,0,255,0.1)", annotation_text="Expert", annotation_position="left")
        
        # Style the chart
        fig.update_layout(
            title="🎹 Your Piano Performance Journey 🎹",
            xaxis_title="Practice Session",
            yaxis_title="Performance Score",
            yaxis=dict(range=[0, 100]),
            template='plotly_white',
            showlegend=True
        )
        
        # Add trend line
        if len(overall_scores) > 1:
            z = np.polyfit(attempts, overall_scores, 1)
            trend_line = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=attempts,
                    y=trend_line(attempts),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', color='gray'),
                    opacity=0.7
                )
            )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def register_update_callback(self, callback: Callable):
        """Register a callback function for real-time updates."""
        self.update_callbacks.append(callback)
    
    def trigger_update(self, data: Dict):
        """Trigger all registered update callbacks with new data."""
        for callback in self.update_callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in update callback: {e}")