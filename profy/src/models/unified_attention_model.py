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
        max_sensor_weight: float = 0.6,
        gating_temperature_scale: float = 2.0,
        pretrained_audio_dim: Optional[int] = None,
        use_temporal_attention: bool = True,
        use_residual_heads: bool = True,
        use_audio_tcn: bool = True,
    ) -> None:
        super().__init__()

        self.use_modality_gating = use_modality_gating
        self.modality_dropout_p = modality_dropout_p
        self.audio_quality_dim = audio_quality_dim
        self.min_sensor_weight = min_sensor_weight
        max_sensor_weight = max(max_sensor_weight, min_sensor_weight)
        if max_sensor_weight >= 1.0:
            max_sensor_weight = 0.999
        self.max_sensor_weight = max_sensor_weight
        self.gating_temperature_scale = gating_temperature_scale
        self.pretrained_audio_dim = pretrained_audio_dim
        self.use_temporal_attention = use_temporal_attention
        self.use_residual_heads = use_residual_heads
        self.use_audio_tcn = use_audio_tcn
        self.audio_dim = audio_dim

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
        # Small frequency-attention head to avoid collapsing informative bands
        # Produces per-frequency weights (softmax over frequency) for weighted pooling
        self.audio_freq_attn = nn.Conv2d(128, 1, kernel_size=(3, 1), padding=(1, 0))
        self.audio_proj = nn.Linear(128, hidden_dim)
        if self.use_audio_tcn:
            self.audio_tcn = nn.Sequential(
                TemporalConvBlock(hidden_dim, dilation=1, dropout=dropout),
                TemporalConvBlock(hidden_dim, dilation=2, dropout=dropout),
            )
        else:
            self.audio_tcn = nn.Identity()

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
        if self.use_temporal_attention:
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.temporal_attention = None
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
        if audio_data.size(-1) != self.audio_dim:
            raise ValueError(f'audio_data last dimension must be {self.audio_dim} features')

        spec = audio_data.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 128, T]
        x2d = self.audio_cnn2d(spec)  # [B, 128, F, T/4]
        # Frequency attention: softmax over frequency axis
        att_f = self.audio_freq_attn(x2d)  # [B, 1, F, T/4]
        att_f = torch.softmax(att_f.squeeze(1), dim=1).unsqueeze(1)  # [B,1,F,T/4]
        x = (x2d * att_f).sum(dim=2)  # [B, 128, T/4]
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

        seq_mask: Optional[torch.Tensor] = None  # [B, L] mask of valid timesteps where available

        if use_sensor and use_audio:
            sensor_feat, audio_feat = self._apply_modality_dropout(sensor_feat, audio_feat)

            audio_padding = None
            sensor_padding = None
            if audio_mask_feat is not None:
                audio_padding = (audio_mask_feat < 0.5)
            # Build a lightweight sensor padding mask from raw sensor energy
            if sensor_data is not None:
                with torch.no_grad():
                    se = sensor_data.abs().mean(dim=-1)  # [B, T_sensor_raw]
                    se_ds = F.adaptive_avg_pool1d(se.unsqueeze(1), sensor_feat.size(1)).squeeze(1)
                    # Dynamic threshold relative to per-sample mean to handle scale
                    thr = (se_ds.mean(dim=1, keepdim=True) * 0.1)
                    valid = (se_ds > thr).float()
                    # If all invalid for a sample, disable padding for that sample later
                    sensor_padding = (valid < 0.5)

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
                key_padding_mask=None if (sensor_padding is None or sensor_padding.all(dim=1).any()) else sensor_padding,
            )

            audio_resampled = self.audio_resampler(audio_feat, sensor_feat.size(1))
            attended_sensor_resampled = self.sensor_resampler(attended_sensor_on_audio, sensor_feat.size(1))

            if audio_mask_feat is not None:
                mask_sensor = F.adaptive_avg_pool1d(audio_mask_feat.unsqueeze(1), sensor_feat.size(1)).squeeze(1)
                mask_sensor = mask_sensor.clamp(0.0, 1.0)
                attended_audio_on_sensor = attended_audio_on_sensor * mask_sensor.unsqueeze(-1)
                audio_resampled = audio_resampled * mask_sensor.unsqueeze(-1)
                # propagate sequence mask for downstream temporal attention/evidence
                seq_mask = mask_sensor
            # Merge sensor validity into sequence mask if available
            if sensor_padding is not None:
                sensor_valid = (~sensor_padding).float()
                if seq_mask is None:
                    seq_mask = sensor_valid
                else:
                    seq_mask = (seq_mask * sensor_valid).clamp(0.0, 1.0)

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
            if audio_quality is not None and audio_quality.size(1) > 0:
                nsr = audio_quality[:, 0:1].clamp(0.0, 1.0)
                max_floor = torch.full_like(sensor_total.unsqueeze(1), self.max_sensor_weight)
                min_floor = torch.full_like(sensor_total.unsqueeze(1), self.min_sensor_weight)
                adaptive_floor = torch.lerp(max_floor, min_floor, nsr).squeeze(1)
            else:
                adaptive_floor = torch.full_like(sensor_total, self.min_sensor_weight)
            target_sensor = torch.maximum(sensor_total, adaptive_floor)
            target_sensor = torch.clamp(target_sensor, max=1.0 - 1e-6)
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
            # When audio-only, carry forward the downsampled audio mask as sequence mask
            if audio_mask_feat is not None:
                seq_mask = audio_mask_feat

        lstm_out, _ = self.lstm(features)
        if self.use_temporal_attention and self.temporal_attention is not None:
            att_scores = self.temporal_attention(lstm_out).squeeze(-1)  # [B, L]
            if seq_mask is not None:
                # Exclude invalid/silent timesteps from attention (softmax) by setting -inf
                invalid = (seq_mask < 0.5)
                # guard: if all invalid, fall back to uniform
                if invalid.all(dim=1).any():
                    # For any batch item with all invalid, ignore masking for that item
                    # (avoid NaNs); handle per-row by zeroing invalid where not all invalid
                    row_all_invalid = invalid.all(dim=1)
                    if (~row_all_invalid).any():
                        att_scores[~row_all_invalid] = att_scores[~row_all_invalid].masked_fill(
                            invalid[~row_all_invalid], float('-inf')
                        )
                else:
                    att_scores = att_scores.masked_fill(invalid, float('-inf'))
            att_weights = torch.softmax(att_scores, dim=-1)
            if seq_mask is not None:
                # Extra safety: zero-out and renormalize attention on invalid steps
                att_weights = att_weights * (seq_mask >= 0.5).float()
                denom = att_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                att_weights = att_weights / denom
        else:
            seq_len = lstm_out.size(1)
            att_weights = torch.full(
                (lstm_out.size(0), seq_len),
                1.0 / max(seq_len, 1),
                device=lstm_out.device,
            )

        evidence_scores = self.evidence_layer(lstm_out).squeeze(-1)
        if seq_mask is not None:
            # Suppress evidence on invalid/silent timesteps so downstream alignment isn't polluted
            evidence_scores = evidence_scores * seq_mask
        context = torch.sum(lstm_out * att_weights.unsqueeze(-1), dim=1)

        logits = self.classifier(context)

        modality_entropy = None
        if modality_weights is not None and use_sensor and use_audio:
            probs = modality_weights.clamp(min=1e-8)
            modality_entropy = -torch.sum(probs * torch.log(probs), dim=-1)

        return {
            'logits': logits,
            'attention_weights': att_weights,
            'evidence_scores': evidence_scores,
            'modality_weights': modality_weights if (use_sensor and use_audio) else None,
            'modality_used': modality_used,
            'modality_entropy': modality_entropy if modality_entropy is not None else None,
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
