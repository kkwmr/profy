"""
Performance result container for storing and analyzing evaluation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch

logger = logging.getLogger(__name__)


class PerformanceResult:
    """
    Container for comprehensive performance evaluation results.
    
    Stores model outputs, metrics, visualizations, and analysis reports.
    """
    
    def __init__(
        self,
        results: Dict[str, torch.Tensor],
        config: Dict,
        audio_path: Optional[str] = None,
        sensor_data_path: Optional[str] = None,
        note_data_path: Optional[str] = None
    ):
        """
        Initialize performance result container.
        
        Args:
            results: Raw model outputs
            config: Model configuration
            audio_path: Path to audio file
            sensor_data_path: Path to sensor data
            note_data_path: Path to note data
        """
        self.results = results
        self.config = config
        self.audio_path = audio_path
        self.sensor_data_path = sensor_data_path
        self.note_data_path = note_data_path
        
        # Process results
        self.scores = self._extract_scores()
        self.skill_info = self._extract_skill_info()
        self.detailed_analysis = self._extract_detailed_analysis()
        
    def _extract_scores(self) -> Dict[str, float]:
        """Extract numerical scores from results."""
        scores = {}
        
        # Main scores
        if 'overall_score' in self.results:
            scores['overall'] = self.results['overall_score'].item()
        if 'technical_score' in self.results:
            scores['technical'] = self.results['technical_score'].item()
        if 'musical_score' in self.results:
            scores['musical'] = self.results['musical_score'].item()
            
        # Detailed technical scores
        if 'technical_analysis' in self.results:
            tech_scores = self.results['technical_analysis'].get('detailed_scores', {})
            for key, value in tech_scores.items():
                if isinstance(value, torch.Tensor):
                    scores[f'technical_{key}'] = value.item()
                    
        # Detailed musical scores  
        if 'musical_analysis' in self.results:
            mus_scores = self.results['musical_analysis'].get('detailed_scores', {})
            for key, value in mus_scores.items():
                if isinstance(value, torch.Tensor):
                    scores[f'musical_{key}'] = value.item()
        
        return scores
    
    def _extract_skill_info(self) -> Dict[str, Any]:
        """Extract skill level information."""
        skill_info = {}
        
        if 'skill_level' in self.results:
            skill_info['predicted_level'] = self.results['skill_level'].item()
            
        if 'skill_probabilities' in self.results:
            probs = self.results['skill_probabilities'].cpu().numpy()
            skill_info['probabilities'] = probs.tolist()
            skill_info['confidence'] = float(np.max(probs))
            
            # Map to skill names
            skill_names = ['beginner', 'intermediate', 'advanced', 'expert']
            skill_info['level_name'] = skill_names[skill_info['predicted_level']]
            
        return skill_info
    
    def _extract_detailed_analysis(self) -> Dict[str, Any]:
        """Extract detailed analysis information."""
        analysis = {}
        
        # Technical analysis
        if 'technical_analysis' in self.results:
            analysis['technical'] = self._process_analysis_dict(
                self.results['technical_analysis']
            )
            
        # Musical analysis
        if 'musical_analysis' in self.results:
            analysis['musical'] = self._process_analysis_dict(
                self.results['musical_analysis']
            )
            
        # Attention weights
        if 'attention_weights' in self.results:
            analysis['attention'] = self._process_attention_weights(
                self.results['attention_weights']
            )
        
        return analysis
    
    def _process_analysis_dict(self, analysis_dict: Dict) -> Dict:
        """Convert tensor values to serializable format."""
        processed = {}
        
        for key, value in analysis_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    processed[key] = value.item()
                else:
                    processed[key] = value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                processed[key] = self._process_analysis_dict(value)
            else:
                processed[key] = value
                
        return processed
    
    def _process_attention_weights(self, attention_dict: Dict) -> Dict:
        """Process attention weight tensors."""
        processed = {}
        
        for level, weights in attention_dict.items():
            if isinstance(weights, torch.Tensor):
                # Sample only key attention weights to avoid huge files
                processed[level] = {
                    'shape': list(weights.shape),
                    'mean': weights.mean().item(),
                    'std': weights.std().item(),
                    'max': weights.max().item(),
                    'min': weights.min().item()
                }
            else:
                processed[level] = weights
                
        return processed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the performance evaluation."""
        summary = {
            'overall_score': self.scores.get('overall', 0),
            'technical_score': self.scores.get('technical', 0),
            'musical_score': self.scores.get('musical', 0),
            'skill_level': self.skill_info.get('level_name', 'unknown'),
            'skill_confidence': self.skill_info.get('confidence', 0),
            'file_info': {
                'audio_path': self.audio_path,
                'sensor_data_path': self.sensor_data_path,
                'note_data_path': self.note_data_path
            }
        }
        
        return summary
    
    def save_to_file(self, output_path: str):
        """Save results to JSON file."""
        output_data = {
            'summary': self.get_summary(),
            'detailed_scores': self.scores,
            'skill_info': self.skill_info,
            'detailed_analysis': self.detailed_analysis,
            'config': self.config
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Performance results saved to {output_path}")
    
    def __str__(self) -> str:
        """String representation of results."""
        summary = self.get_summary()
        
        return f"""
Profy Performance Evaluation Results
===================================
Overall Score: {summary['overall_score']:.1f}/100
Technical Score: {summary['technical_score']:.1f}/100  
Musical Score: {summary['musical_score']:.1f}/100
Skill Level: {summary['skill_level']} (confidence: {summary['skill_confidence']:.2f})

Technical Breakdown:
- Pitch: {self.scores.get('technical_pitch', 0):.2f}
- Rhythm: {self.scores.get('technical_rhythm', 0):.2f}
- Tempo: {self.scores.get('technical_tempo', 0):.2f}
- Finger Technique: {self.scores.get('technical_finger', 0):.2f}
- Touch Quality: {self.scores.get('technical_touch', 0):.2f}

Musical Breakdown:
- Dynamics: {self.scores.get('musical_dynamics', 0):.2f}
- Articulation: {self.scores.get('musical_articulation', 0):.2f}
- Phrasing: {self.scores.get('musical_phrasing', 0):.2f}
- Interpretation: {self.scores.get('musical_interpretation', 0):.2f}
"""