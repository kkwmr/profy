#!/usr/bin/env python3
"""Unified Attention Model with enhanced audio fusion."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Lightweight TCN block operating on [B, T, C] tensors."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.transpose(1, 2)
        return x + residual


class ParametricResampler(nn.Module):
    """Learnable resampler combining adaptive pooling with depthwise convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.adaptive_avg_pool1d(x, target_length)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class UnifiedAttentionModel(nn.Module):
    """Unified model supporting sensor, audio, and multimodal inputs."""

    def __init__(
        self,
        sensor_dim: int = 88,
        audio_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_modality_gating: bool = True,
        modality_dropout_p: float = 0.0,
        audio_quality_dim: int = 3,
        min_sensor_weight: float = 0.35,
        gating_temperature_scale: float = 2.0,
    ) -> None:
        super().__init__()

        self.use_modality_gating = use_modality_gating
        self.modality_dropout_p = modality_dropout_p
        self.audio_quality_dim = audio_quality_dim
        self.min_sensor_weight = min_sensor_weight
        self.gating_temperature_scale = gating_temperature_scale

        # Sensor encoder
        self.sensor_conv3 = nn.Sequential(
            nn.Conv1d(sensor_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.sensor_conv5 = nn.Sequential(
            nn.Conv1d(sensor_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.sensor_conv9 = nn.Sequential(
            nn.Conv1d(sensor_dim, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.sensor_fuse = nn.Sequential(
            nn.Conv1d(128 * 3, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.sensor_proj = nn.Linear(256, hidden_dim)

        # Audio encoder with temporal conv enhancement
        self.audio_cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.audio_proj = nn.Linear(128, hidden_dim)
        self.audio_tcn = nn.Sequential(
            TemporalConvBlock(hidden_dim, dilation=1, dropout=dropout),
            TemporalConvBlock(hidden_dim, dilation=2, dropout=dropout),
        )

        # Audio quality projection
        self.quality_proj = nn.Linear(audio_quality_dim, hidden_dim) if audio_quality_dim > 0 else None

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Parametric resamplers
        self.audio_resampler = ParametricResampler(hidden_dim)
        self.sensor_resampler = ParametricResampler(hidden_dim)

        # Fusion gate (logits only; softmax applied in forward)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
        )
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        # Shared BiLSTM and downstream heads
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.evidence_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def encode_sensor(self, sensor_data: torch.Tensor) -> torch.Tensor:
        if sensor_data.dim() == 2:
            sensor_data = sensor_data.unsqueeze(0)
        if sensor_data.dim() != 3:
            raise ValueError(f'sensor_data must be [B, T, 88], got {tuple(sensor_data.shape)}')
        x = sensor_data.transpose(1, 2)
        x = torch.cat([
            self.sensor_conv3(x),
            self.sensor_conv5(x),
            self.sensor_conv9(x),
        ], dim=1)
        x = self.sensor_fuse(x)
        x = x.transpose(1, 2)
        return self.sensor_proj(x)

    def encode_audio(
        self,
        audio_data: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if audio_data.dim() == 2:
            audio_data = audio_data.unsqueeze(0)
        if audio_data.dim() != 3:
            raise ValueError(f'audio_data must be [B, T, F], got {tuple(audio_data.shape)}')
        if audio_data.size(-1) != 128:
            raise ValueError('audio_data last dimension must be 128 features')

        spec = audio_data.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 128, T]
        x2d = self.audio_cnn2d(spec)
        x = torch.mean(x2d, dim=2)  # [B, 128, T/4]
        x = x.transpose(1, 2)
        x = self.audio_proj(x)
        x = self.audio_tcn(x)

        mask_out: Optional[torch.Tensor] = None
        if audio_mask is not None:
            if audio_mask.dim() == 1:
                audio_mask = audio_mask.unsqueeze(0)
            audio_mask = audio_mask.unsqueeze(1).float()
            mask_out = F.adaptive_avg_pool1d(audio_mask, x.size(1)).squeeze(1)
            mask_out = mask_out.clamp(0.0, 1.0)
            x = x * mask_out.unsqueeze(-1)

        return x, mask_out

    def _apply_modality_dropout(self, sensor_feat: torch.Tensor, audio_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.modality_dropout_p > 0.0:
            drop_mask = torch.rand(sensor_feat.size(0), device=sensor_feat.device)
            drop_audio = drop_mask < (self.modality_dropout_p / 2)
            drop_sensor = (drop_mask >= (self.modality_dropout_p / 2)) & (drop_mask < self.modality_dropout_p)
            if drop_audio.any():
                audio_feat[drop_audio] = 0.0
            if drop_sensor.any():
                sensor_feat[drop_sensor] = 0.0
        return sensor_feat, audio_feat

    def _temperature_softmax(self, logits: torch.Tensor, quality: Optional[torch.Tensor]) -> torch.Tensor:
        if quality is not None and quality.size(1) > 0:
            nsr = quality[:, 0:1].clamp(0.0, 1.0)
            temperature = 1.0 + self.gating_temperature_scale * (1.0 - nsr)
        else:
            temperature = 1.0
        return torch.softmax(logits / temperature, dim=-1)

    def forward(
        self,
        sensor_data: Optional[torch.Tensor] = None,
        audio_data: Optional[torch.Tensor] = None,
        audio_quality: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | dict[str, bool] | None]:
        use_sensor = sensor_data is not None
        use_audio = audio_data is not None
        if not use_sensor and not use_audio:
            raise ValueError('At least one modality must be provided')

        modality_used = {'sensor': use_sensor, 'audio': use_audio}
        modality_weights = None

        if use_sensor:
            sensor_feat = self.encode_sensor(sensor_data)
        else:
            sensor_feat = None

        if use_audio:
            audio_feat, audio_mask_feat = self.encode_audio(audio_data, audio_mask)
        else:
            audio_feat, audio_mask_feat = None, None

        if use_sensor and use_audio:
            sensor_feat, audio_feat = self._apply_modality_dropout(sensor_feat, audio_feat)

            audio_padding = None
            if audio_mask_feat is not None:
                audio_padding = (audio_mask_feat < 0.5)

            attended_audio_on_sensor, _ = self.cross_attention(
                query=sensor_feat,
                key=audio_feat,
                value=audio_feat,
                key_padding_mask=audio_padding,
            )
            attended_sensor_on_audio, _ = self.cross_attention(
                query=audio_feat,
                key=sensor_feat,
                value=sensor_feat,
                key_padding_mask=None,
            )

            audio_resampled = self.audio_resampler(audio_feat, sensor_feat.size(1))
            attended_sensor_resampled = self.sensor_resampler(attended_sensor_on_audio, sensor_feat.size(1))

            if audio_mask_feat is not None:
                mask_sensor = F.adaptive_avg_pool1d(audio_mask_feat.unsqueeze(1), sensor_feat.size(1)).squeeze(1)
                mask_sensor = mask_sensor.clamp(0.0, 1.0)
                attended_audio_on_sensor = attended_audio_on_sensor * mask_sensor.unsqueeze(-1)
                audio_resampled = audio_resampled * mask_sensor.unsqueeze(-1)

            quality_proj = self.quality_proj(audio_quality) if (self.quality_proj is not None and audio_quality is not None) else None
            s_sum = sensor_feat.mean(dim=1)
            a_sum = attended_audio_on_sensor.mean(dim=1)
            if quality_proj is None:
                gate_in = torch.cat([s_sum, a_sum, torch.zeros_like(s_sum)], dim=-1)
            else:
                gate_in = torch.cat([s_sum, a_sum, quality_proj], dim=-1)

            logits = self.fusion_gate(gate_in)
            weights4 = self._temperature_softmax(logits, audio_quality if audio_quality is not None else None)

            sensor_total = torch.clamp(weights4[:, 0] + weights4[:, 3], min=1e-6)
            audio_total = torch.clamp(weights4[:, 1] + weights4[:, 2], min=1e-6)
            target_sensor = torch.clamp(sensor_total, min=self.min_sensor_weight, max=1.0 - 1e-6)
            target_audio = 1.0 - target_sensor

            new_w0 = target_sensor * (weights4[:, 0] / sensor_total)
            new_w3 = target_sensor * (weights4[:, 3] / sensor_total)
            new_w1 = target_audio * (weights4[:, 1] / audio_total)
            new_w2 = target_audio * (weights4[:, 2] / audio_total)

            weights = torch.stack([new_w0, new_w1, new_w2, new_w3], dim=1)
            weights = weights.clamp(min=0.0)
            weights = weights / weights.sum(dim=1, keepdim=True)

            w0 = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
            w1 = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
            w2 = weights[:, 2].unsqueeze(-1).unsqueeze(-1)
            w3 = weights[:, 3].unsqueeze(-1).unsqueeze(-1)

            fused = (
                w0 * sensor_feat +
                w1 * attended_audio_on_sensor +
                w2 * audio_resampled +
                w3 * attended_sensor_resampled
            )
            features = self.fusion_ln(fused)
            modality_weights = torch.stack([
                weights[:, 0] + weights[:, 3],
                weights[:, 1] + weights[:, 2],
            ], dim=-1)
        elif use_sensor:
            features = sensor_feat
            modality_weights = torch.stack([
                torch.ones(sensor_feat.size(0), device=sensor_feat.device),
                torch.zeros(sensor_feat.size(0), device=sensor_feat.device),
            ], dim=-1)
        else:
            features = audio_feat
            modality_weights = torch.stack([
                torch.zeros(audio_feat.size(0), device=audio_feat.device),
                torch.ones(audio_feat.size(0), device=audio_feat.device),
            ], dim=-1)

        lstm_out, _ = self.lstm(features)
        att_scores = self.temporal_attention(lstm_out).squeeze(-1)
        att_weights = torch.softmax(att_scores, dim=-1)

        evidence_scores = self.evidence_layer(lstm_out).squeeze(-1)
        context = torch.sum(lstm_out * att_weights.unsqueeze(-1), dim=1)

        logits = self.classifier(context)

        return {
            'logits': logits,
            'attention_weights': att_weights,
            'evidence_scores': evidence_scores,
            'modality_weights': modality_weights if (use_sensor and use_audio) else None,
            'modality_used': modality_used,
        }

    def get_attention_visualization(self, sensor_data: Optional[torch.Tensor] = None, audio_data: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.forward(sensor_data, audio_data)
            attention = outputs['attention_weights'].cpu().numpy()
            evidence = outputs['evidence_scores'].cpu().numpy()
            problem_threshold = 0.5
            problem_mask = evidence > problem_threshold
            problem_regions = []
            for mask in problem_mask:
                regions = []
                start = None
                for idx, val in enumerate(mask):
                    if val and start is None:
                        start = idx
                    elif not val and start is not None:
                        regions.append((start, idx))
                        start = None
                if start is not None:
                    regions.append((start, len(mask)))
                problem_regions.append(regions)
            return {
                'attention_weights': attention,
                'evidence_scores': evidence,
                'problem_regions': problem_regions,
                'problem_severity': evidence * attention,
            }
