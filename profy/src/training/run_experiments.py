#!/usr/bin/env python3
"""Run three-modality experiments (multimodal/sensor/audio) with UnifiedAttentionModel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..data.real_data_loader import load_real_piano_data
from ..models.unified_attention_model import UnifiedAttentionModel

LOGGER = logging.getLogger(__name__)


def ensure_results_dir() -> Path:
    results_dir = Path(os.environ.get('RESULTS_DIR', f'results/experiment_{time.strftime("%Y%m%d_%H%M%S")}'))
    results_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('logs', 'figures', 'models', 'diagnostics'):
        (results_dir / sub).mkdir(exist_ok=True)
    return results_dir


def train_one_epoch(
    model: UnifiedAttentionModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    scaler: torch.cuda.amp.GradScaler | None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    entropy_regularization: float = 0.0,
    lambda_mil: float = 0.0,
    lambda_evidence_l1: float = 0.0,
    mil_topk_frac: float = 0.0,
    mil_blend_alpha: float = 0.5,
    lambda_attention_energy: float = 0.0,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_tgts = [], []
    mixup_enabled = mixup_alpha > 0.0 and mixup_prob > 0.0

    for batch in loader:
        sensor = batch.get('sensor')
        audio = batch.get('audio')
        quality = batch.get('quality')
        audio_mask = batch.get('audio_mask')
        label = batch['label']

        sensor = sensor.to(device) if sensor is not None else None
        audio = audio.to(device) if audio is not None else None
        quality = quality.to(device) if quality is not None else None
        audio_mask = audio_mask.to(device) if audio_mask is not None else None
        label = label.to(device)

        mixup_state: tuple[torch.Tensor, torch.Tensor, float] | None = None
        if (
            mixup_enabled
            and label.size(0) > 1
            and torch.rand(1, device=device).item() < mixup_prob
        ):
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam = max(lam, 1.0 - lam)
            index = torch.randperm(label.size(0), device=device)
            if sensor is not None:
                sensor = lam * sensor + (1.0 - lam) * sensor[index]
            if audio is not None:
                audio = lam * audio + (1.0 - lam) * audio[index]
            if quality is not None:
                quality = lam * quality + (1.0 - lam) * quality[index]
            if audio_mask is not None:
                audio_mask = lam * audio_mask + (1.0 - lam) * audio_mask[index]
                audio_mask = audio_mask.clamp(0.0, 1.0)
            mixup_state = (label, label[index], lam)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == 'cuda'
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            if mode == 'sensor':
                outputs = model(sensor_data=sensor, audio_data=None)
            elif mode == 'audio':
                outputs = model(
                    sensor_data=None,
                    audio_data=audio,
                    audio_quality=quality,
                    audio_mask=audio_mask,
                )
            else:
                outputs = model(
                    sensor_data=sensor,
                    audio_data=audio,
                    audio_quality=quality,
                    audio_mask=audio_mask,
                )
            loss = criterion(outputs['logits'], label)
            if mixup_state is not None:
                la, lb, lam = mixup_state
                loss = lam * criterion(outputs['logits'], la) + (1.0 - lam) * criterion(outputs['logits'], lb)

            # Auxiliary losses to directly train evidence heads
            evid = outputs.get('evidence_scores')
            if isinstance(evid, torch.Tensor) and evid.ndim == 2:
                # MIL aggregation: probability that any timestep indicates positive (expert-like problem evidence)
                # Stable complement-of-product: P(any) = 1 - Π(1 - p_i)
                one_minus = (1.0 - evid.clamp(0.0, 1.0)).clamp(1e-6, 1.0)
                prod_compl = torch.prod(one_minus, dim=1)
                p_noisy_or = (1.0 - prod_compl).clamp(0.0, 1.0)
                if mil_topk_frac > 0.0:
                    k = max(1, int(evid.size(1) * float(mil_topk_frac)))
                    topk_vals, _ = torch.topk(evid, k=k, dim=1)
                    p_topk = topk_vals.mean(dim=1)
                    clip_pos_prob = (float(mil_blend_alpha) * p_topk + (1.0 - float(mil_blend_alpha)) * p_noisy_or).clamp(0.0, 1.0)
                else:
                    clip_pos_prob = p_noisy_or
                target_pos = (label == 1).float()
                # Compute BCE outside autocast for numerical safety
                with torch.amp.autocast('cuda', enabled=False):
                    if mixup_state is None:
                        mil_loss = F.binary_cross_entropy(clip_pos_prob.float(), target_pos.float())
                    else:
                        la, lb, lam = mixup_state
                        mil_loss = lam * F.binary_cross_entropy(clip_pos_prob.float(), (la == 1).float()) \
                                   + (1.0 - lam) * F.binary_cross_entropy(clip_pos_prob.float(), (lb == 1).float())
                if lambda_mil > 0.0:
                    loss = loss + float(lambda_mil) * mil_loss

                if lambda_evidence_l1 > 0.0:
                    evi_l1 = evid.abs().mean()
                    loss = loss + float(lambda_evidence_l1) * evi_l1

            # Attention energy regularization (discourage pure loudness-driven attention)
            if lambda_attention_energy > 0.0 and outputs.get('attention_weights') is not None and audio is not None:
                att = outputs['attention_weights']  # [B, L]
                energy = audio.abs().mean(dim=-1)  # [B, T]
                energy_ds = torch.nn.functional.adaptive_avg_pool1d(energy.unsqueeze(1), att.size(1)).squeeze(1)
                # Normalize to a distribution
                energy_pos = (energy_ds - energy_ds.min(dim=1, keepdim=True).values).clamp_min(0.0)
                denom = energy_pos.sum(dim=1, keepdim=True).clamp_min(1e-8)
                energy_dist = energy_pos / denom
                align = (att * energy_dist).sum(dim=1).mean()
                loss = loss + float(lambda_attention_energy) * align

        entropy_term = outputs.get('modality_entropy') if isinstance(outputs, dict) else None
        if entropy_term is not None and entropy_regularization > 0.0:
            loss = loss + entropy_regularization * torch.exp(-entropy_term).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs['logits'].argmax(1).detach().cpu().numpy())
        all_tgts.extend(label.detach().cpu().numpy())

        gate = outputs.get('modality_weights')
        if (
            mode == 'multimodal'
            and gate is not None
            and torch.rand(1, device=device).item() < 0.2
        ):
            gate_mean = gate.detach().mean(dim=0).cpu().numpy()
            quality_snapshot = (
                quality.detach().mean(dim=0).cpu().numpy().tolist()
                if quality is not None
                else None
            )
            LOGGER.info(
                'Gate snapshot sensor=%.3f audio=%.3f quality=%s',
                float(gate_mean[0]),
                float(gate_mean[1]),
                quality_snapshot,
            )

    f1 = f1_score(all_tgts, all_preds, average='weighted')
    acc = accuracy_score(all_tgts, all_preds)
    mean_loss = total_loss / max(1, len(loader))
    return mean_loss, f1, acc


@torch.no_grad()
def evaluate(model: UnifiedAttentionModel, loader: DataLoader, device: torch.device, mode: str) -> Dict[str, Any]:
    model.eval()
    all_preds, all_tgts = [], []
    modality_weights: list[np.ndarray] = []
    pieces: list[str] = []
    all_probs: list[float] = []
    entropy_values: list[float] = []

    for batch in loader:
        sensor = batch.get('sensor')
        audio = batch.get('audio')
        quality = batch.get('quality')
        audio_mask = batch.get('audio_mask')
        label = batch['label']

        sensor = sensor.to(device) if sensor is not None else None
        audio = audio.to(device) if audio is not None else None
        quality = quality.to(device) if quality is not None else None
        audio_mask = audio_mask.to(device) if audio_mask is not None else None
        label = label.to(device)

        if mode == 'sensor':
            outputs = model(sensor_data=sensor, audio_data=None)
        elif mode == 'audio':
            outputs = model(
                sensor_data=None,
                audio_data=audio,
                audio_quality=quality,
                audio_mask=audio_mask,
            )
        else:
            outputs = model(
                sensor_data=sensor,
                audio_data=audio,
                audio_quality=quality,
                audio_mask=audio_mask,
            )

        preds = outputs['logits'].argmax(1)
        probs = torch.softmax(outputs['logits'], dim=-1)[:, 1]
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_tgts.extend(label.detach().cpu().numpy())

        gate = outputs.get('modality_weights')
        if gate is not None:
            modality_weights.append(gate.detach().cpu().numpy())
        entropy = outputs.get('modality_entropy')
        if entropy is not None:
            entropy_values.extend(entropy.detach().cpu().numpy().tolist())

        batch_pieces = batch.get('piece')
        if batch_pieces is not None:
            pieces.extend(batch_pieces)

    f1 = f1_score(all_tgts, all_preds, average='weighted')
    acc = accuracy_score(all_tgts, all_preds)
    cm = confusion_matrix(all_tgts, all_preds)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        all_tgts,
        all_preds,
        labels=[0, 1],
        zero_division=0,
    )

    gate_matrix = np.concatenate(modality_weights, axis=0) if modality_weights else None
    return {
        'f1': f1,
        'accuracy': acc,
        'confusion_matrix': cm,
        'predictions': np.asarray(all_preds, dtype=int),
        'targets': np.asarray(all_tgts, dtype=int),
        'modality_weights': gate_matrix,
        'modality_entropy': np.array(entropy_values, dtype=float) if entropy_values else None,
        'precision': precision,
        'recall': recall,
        'per_class_f1': per_class_f1,
        'support': support,
        'pieces': pieces,
        'probabilities': np.asarray(all_probs, dtype=float),
    }


def _make_dataloader(
    mode: str,
    tensors: Dict[str, Optional[torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    has_piece = tensors.get('piece') is not None

    if mode == 'sensor':
        if has_piece:
            dataset = list(zip(tensors['sensor'], tensors['label'], tensors['piece']))

            def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor, str]]) -> Dict[str, torch.Tensor | list[str]]:
                sensor, label, piece = list(zip(*batch))
                return {
                    'sensor': torch.stack(sensor),
                    'label': torch.stack(label),
                    'piece': list(piece),
                }
        else:
            dataset = TensorDataset(tensors['sensor'], tensors['label'])

            def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
                sensors, labels = list(zip(*batch))
                return {
                    'sensor': torch.stack(sensors),
                    'label': torch.stack(labels),
                }

    elif mode == 'audio':
        if has_piece:
            dataset = list(zip(tensors['audio'], tensors['quality'], tensors['audio_mask'], tensors['label'], tensors['piece']))

            def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> Dict[str, torch.Tensor | list[str]]:
                audio, quality, mask, labels, piece = list(zip(*batch))
                return {
                    'audio': torch.stack(audio),
                    'quality': torch.stack(quality),
                    'audio_mask': torch.stack(mask),
                    'label': torch.stack(labels),
                    'piece': list(piece),
                }
        else:
            dataset = TensorDataset(tensors['audio'], tensors['quality'], tensors['audio_mask'], tensors['label'])

            def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
                audio, quality, mask, labels = list(zip(*batch))
                return {
                    'audio': torch.stack(audio),
                    'quality': torch.stack(quality),
                    'audio_mask': torch.stack(mask),
                    'label': torch.stack(labels),
                }

    else:  # multimodal
        if has_piece:
            dataset = list(zip(
                tensors['sensor'],
                tensors['audio'],
                tensors['quality'],
                tensors['audio_mask'],
                tensors['label'],
                tensors['piece'],
            ))

            def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> Dict[str, torch.Tensor | list[str]]:
                sensor, audio, quality, mask, labels, piece = list(zip(*batch))
                return {
                    'sensor': torch.stack(sensor),
                    'audio': torch.stack(audio),
                    'quality': torch.stack(quality),
                    'audio_mask': torch.stack(mask),
                    'label': torch.stack(labels),
                    'piece': list(piece),
                }
        else:
            dataset = TensorDataset(
                tensors['sensor'],
                tensors['audio'],
                tensors['quality'],
                tensors['audio_mask'],
                tensors['label'],
            )

            def collate(batch: Iterable[tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
                sensor, audio, quality, mask, labels = list(zip(*batch))
                return {
                    'sensor': torch.stack(sensor),
                    'audio': torch.stack(audio),
                    'quality': torch.stack(quality),
                    'audio_mask': torch.stack(mask),
                    'label': torch.stack(labels),
                }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate,
    )


def run_cv(
    sensor_data: np.ndarray,
    audio_data: Optional[np.ndarray],
    audio_quality: Optional[np.ndarray],
    audio_masks: Optional[np.ndarray],
    labels: np.ndarray,
    session_ids: np.ndarray,
    metadata: list[Dict[str, Any]],
    mode: str,
    results_dir: Path,
    n_folds: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    fusion_warmup_epochs: int = 0,
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_checkpoints: bool = False,
    checkpoint_prefix: Optional[str] = None,
    entropy_regularization: float = 0.0,
    use_balanced_sampler: bool = False,
    noise_settings: Optional[Dict[str, Any]] = None,
    lambda_mil: float = 0.0,
    lambda_evidence_l1: float = 0.0,
    mil_topk_frac: float = 0.0,
    mil_blend_alpha: float = 0.5,
    lambda_attention_energy: float = 0.0,
) -> Dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gkf = GroupKFold(n_splits=n_folds)
    fold_results: list[Dict[str, Any]] = []
    model_kwargs = model_kwargs or {}
    checkpoint_prefix = checkpoint_prefix or mode
    noise_settings = noise_settings or {}

    pieces_array = np.array([m.get('piece', 'unknown') for m in metadata])
    session_array = np.array([m.get('session_id', f'idx_{idx}') for idx, m in enumerate(metadata)])

    piece_accumulator: defaultdict[str, list[Dict[str, float]]] = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(sensor_data, labels, groups=session_ids), start=1):
        sensor_test_np = sensor_data[test_idx].copy() if sensor_data is not None else None
        audio_test_np = audio_data[test_idx].copy() if audio_data is not None else None
        quality_test_np = audio_quality[test_idx].copy() if audio_quality is not None else None
        mask_test_np = audio_masks[test_idx].copy() if audio_masks is not None else None

        if noise_settings:
            rng = np.random.default_rng(seed=fold * 9973)
            if audio_test_np is not None and noise_settings.get('audio_snr') is not None:
                audio_test_np = _apply_gaussian_noise(audio_test_np, noise_settings['audio_snr'], rng)
            if sensor_test_np is not None and noise_settings.get('sensor_dropout') is not None:
                sensor_test_np = _apply_sensor_dropout(sensor_test_np, noise_settings['sensor_dropout'], rng)

        tensors_test: Dict[str, Optional[torch.Tensor]] = {
            'sensor': torch.FloatTensor(sensor_test_np) if sensor_test_np is not None else None,
            'audio': torch.FloatTensor(audio_test_np) if audio_test_np is not None else None,
            'quality': torch.FloatTensor(quality_test_np) if quality_test_np is not None else None,
            'audio_mask': torch.FloatTensor(mask_test_np) if mask_test_np is not None else None,
            'label': torch.LongTensor(labels[test_idx]),
            'piece': [pieces_array[idx] for idx in test_idx],
        }
        test_loader = _make_dataloader(mode, tensors_test, batch_size, shuffle=False)

        # Build training / validation tensors
        sensor_train = sensor_data[train_idx]
        audio_train = audio_data[train_idx] if audio_data is not None else None
        quality_train = audio_quality[train_idx] if audio_quality is not None else None
        mask_train = audio_masks[train_idx] if audio_masks is not None else None
        labels_train = labels[train_idx]

        n_train = len(train_idx)
        val_mask = np.zeros(n_train, dtype=bool)
        val_mask[: max(1, int(0.15 * n_train))] = True
        rng = np.random.default_rng(seed=fold)
        rng.shuffle(val_mask)

        Xs_val = torch.FloatTensor(sensor_train[val_mask])
        Xs_sub = torch.FloatTensor(sensor_train[~val_mask])
        Xa_val = torch.FloatTensor(audio_train[val_mask]) if audio_train is not None else None
        Xa_sub = torch.FloatTensor(audio_train[~val_mask]) if audio_train is not None else None
        Q_val = torch.FloatTensor(quality_train[val_mask]) if quality_train is not None else None
        Q_sub = torch.FloatTensor(quality_train[~val_mask]) if quality_train is not None else None
        M_val = torch.FloatTensor(mask_train[val_mask]) if mask_train is not None else None
        M_sub = torch.FloatTensor(mask_train[~val_mask]) if mask_train is not None else None
        y_val = torch.LongTensor(labels_train[val_mask])
        y_sub = torch.LongTensor(labels_train[~val_mask])

        tensors_val: Dict[str, Optional[torch.Tensor]] = {
            'sensor': Xs_val,
            'audio': Xa_val,
            'quality': Q_val,
            'audio_mask': M_val,
            'label': y_val,
        }
        tensors_sub: Dict[str, Optional[torch.Tensor]] = {
            'sensor': Xs_sub,
            'audio': Xa_sub,
            'quality': Q_sub,
            'audio_mask': M_sub,
            'label': y_sub,
        }

        train_sampler = None
        if use_balanced_sampler:
            class_counts = np.bincount(labels_train[~val_mask], minlength=2)
            inv_counts = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
            sample_weights = inv_counts[labels_train[~val_mask]]
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )

        train_loader = _make_dataloader(mode, tensors_sub, batch_size, shuffle=True, sampler=train_sampler)
        val_loader = _make_dataloader(mode, tensors_val, batch_size, shuffle=False)

        model = UnifiedAttentionModel(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        audio_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'audio' in name or 'pretrained' in name:
                audio_params.append(param)
            else:
                other_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if other_params:
            param_groups.append({'params': other_params, 'lr': 1e-3})
        if audio_params:
            param_groups.append({'params': audio_params, 'lr': 5e-4})
        if not param_groups:
            param_groups.append({'params': model.parameters(), 'lr': 1e-3})
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        best_state = model.state_dict()
        best_val = -1.0
        bad_epochs = 0
        mode_patience = patience + 2 if mode == 'multimodal' else patience
        warmup_epochs = fusion_warmup_epochs if mode == 'multimodal' else 0
        warmup_epochs = max(0, warmup_epochs)
        original_gating = getattr(model, 'use_modality_gating', True)
        original_dropout = getattr(model, 'modality_dropout_p', 0.0)

        for epoch_idx in range(max_epochs):
            if mode == 'multimodal':
                model.use_modality_gating = epoch_idx >= warmup_epochs
                model.modality_dropout_p = 0.0 if epoch_idx < warmup_epochs else original_dropout
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                mode,
                scaler,
                mixup_alpha=mixup_alpha,
                mixup_prob=mixup_prob,
                entropy_regularization=entropy_regularization,
                lambda_mil=lambda_mil,
                lambda_evidence_l1=lambda_evidence_l1,
                mil_topk_frac=mil_topk_frac,
                mil_blend_alpha=mil_blend_alpha,
                lambda_attention_energy=lambda_attention_energy,
            )
            val_metrics = evaluate(model, val_loader, device, mode)
            if val_metrics['f1'] > best_val:
                best_val = val_metrics['f1']
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= mode_patience:
                break

        model.use_modality_gating = original_gating
        model.modality_dropout_p = original_dropout

        model.load_state_dict(best_state)
        if mode == 'multimodal':
            model.use_modality_gating = True
            model.modality_dropout_p = original_dropout
        test_metrics = evaluate(model, test_loader, device, mode)

        fold_record: Dict[str, Any] = {
            'fold': fold,
            'f1_score': float(test_metrics['f1']),
            'accuracy': float(test_metrics['accuracy']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'precision': np.asarray(test_metrics['precision'], dtype=float).tolist(),
            'recall': np.asarray(test_metrics['recall'], dtype=float).tolist(),
            'per_class_f1': np.asarray(test_metrics['per_class_f1'], dtype=float).tolist(),
            'support': np.asarray(test_metrics['support'], dtype=int).tolist(),
        }

        if test_metrics['pieces']:
            piece_metrics_fold: Dict[str, Dict[str, float]] = {}
            piece_targets: defaultdict[str, list[int]] = defaultdict(list)
            piece_preds: defaultdict[str, list[int]] = defaultdict(list)
            for piece_name, pred, true in zip(test_metrics['pieces'], test_metrics['predictions'], test_metrics['targets']):
                piece_preds[piece_name].append(int(pred))
                piece_targets[piece_name].append(int(true))

            for piece_name, preds_list in piece_preds.items():
                targets_list = piece_targets[piece_name]
                piece_f1 = f1_score(targets_list, preds_list, average='macro', zero_division=0)
                piece_acc = float(np.mean(np.array(preds_list) == np.array(targets_list)))
                support = len(preds_list)
                metrics_entry = {
                    'f1': float(piece_f1),
                    'accuracy': piece_acc,
                    'support': support,
                }
                piece_metrics_fold[piece_name] = metrics_entry
                piece_accumulator[piece_name].append(metrics_entry)

            fold_record['piece_metrics'] = piece_metrics_fold

        fold_results.append(fold_record)

        diag_dir = results_dir / 'diagnostics' / mode
        diag_dir.mkdir(parents=True, exist_ok=True)

        with (diag_dir / f'fold_{fold}_confusion_matrix.json').open('w') as fh:
            json.dump(
                {
                    'fold': fold,
                    'mode': mode,
                    'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
                },
                fh,
                indent=2,
            )

        pred_df = pd.DataFrame({
            'piece': test_metrics.get('pieces', []),
            'probability': test_metrics.get('probabilities', []),
            'prediction': test_metrics['predictions'],
            'label': test_metrics['targets'],
        })

        if test_metrics['modality_weights'] is not None:
            weights = np.asarray(test_metrics['modality_weights'], dtype=float)
            pred_df['sensor_weight'] = weights[:, 0]
            pred_df['audio_weight'] = weights[:, 1]
            if test_metrics.get('modality_entropy') is not None:
                pred_df['modality_entropy'] = np.asarray(test_metrics['modality_entropy'], dtype=float)
            pred_df['correct'] = (pred_df['prediction'] == pred_df['label']).astype(int)
            summary = {
                'fold': fold,
                'samples': int(len(pred_df)),
                'sensor_mean': float(pred_df['sensor_weight'].mean()),
                'audio_mean': float(pred_df['audio_weight'].mean()),
                'sensor_correct_mean': float(pred_df.loc[pred_df['correct'] == 1, 'sensor_weight'].mean()),
                'audio_correct_mean': float(pred_df.loc[pred_df['correct'] == 1, 'audio_weight'].mean()),
                'sensor_incorrect_mean': float(pred_df.loc[pred_df['correct'] == 0, 'sensor_weight'].mean()),
                'audio_incorrect_mean': float(pred_df.loc[pred_df['correct'] == 0, 'audio_weight'].mean()),
            }
            if 'modality_entropy' in pred_df.columns:
                summary['entropy_mean'] = float(pred_df['modality_entropy'].mean())
            with (diag_dir / f'fold_{fold}_modality_summary.json').open('w') as fh:
                json.dump(summary, fh, indent=2)
            pred_df.to_csv(diag_dir / f'fold_{fold}_modality_weights.csv', index=False)

        pred_df.to_csv(diag_dir / f'fold_{fold}_predictions.csv', index=False)

        if save_checkpoints:
            checkpoint_path = results_dir / 'models' / f'{checkpoint_prefix}_fold{fold}.pth'
            torch.save(model.state_dict(), checkpoint_path)

        if mode == 'multimodal':
            attention_dir = results_dir / 'figures' / 'attention'
            attention_dir.mkdir(parents=True, exist_ok=True)
            attention_data_dir = diag_dir / 'attention_arrays'
            attention_data_dir.mkdir(parents=True, exist_ok=True)
            preds_fold = test_metrics['predictions']
            targets_fold = test_metrics['targets']
            attention_records: list[Dict[str, Any]] = []
            max_attention_samples = 3
            for local_idx, global_idx in enumerate(test_idx):
                if len(attention_records) >= max_attention_samples:
                    break
                if preds_fold[local_idx] == targets_fold[local_idx]:
                    continue
                sensor_sample = sensor_data[global_idx]
                audio_sample = audio_data[global_idx] if audio_data is not None else None
                if audio_sample is None:
                    continue
                quality_sample = audio_quality[global_idx] if audio_quality is not None else None
                mask_sample = audio_masks[global_idx] if audio_masks is not None else None
                with torch.no_grad():
                    sensor_tensor = torch.FloatTensor(sensor_sample).unsqueeze(0).to(device)
                    audio_tensor = torch.FloatTensor(audio_sample).unsqueeze(0).to(device)
                    quality_tensor = torch.FloatTensor(quality_sample).unsqueeze(0).to(device) if quality_sample is not None else None
                    mask_tensor = torch.FloatTensor(mask_sample).unsqueeze(0).to(device) if mask_sample is not None else None
                    outputs = model(
                        sensor_data=sensor_tensor,
                        audio_data=audio_tensor,
                        audio_quality=quality_tensor,
                        audio_mask=mask_tensor,
                    )
                attention = outputs['attention_weights'][0].detach().cpu().numpy()
                evidence = outputs['evidence_scores'][0].detach().cpu().numpy()
                severity = attention * evidence
                time_axis = np.arange(attention.shape[0], dtype=float)
                time_axis = time_axis / max(time_axis[-1], 1.0)

                data_path = attention_data_dir / f'fold_{fold}_{session_array[global_idx]}_{local_idx}.npz'
                np.savez(
                    data_path,
                    attention=attention,
                    evidence=evidence,
                    severity=severity,
                    time=time_axis,
                )

                fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
                time_axis = np.arange(attention.shape[0])
                axes[0].plot(time_axis, attention, color='#1f77b4')
                axes[0].set_ylabel('Attention')
                axes[0].grid(alpha=0.2)
                axes[1].plot(time_axis, evidence, color='#ff7f0e')
                axes[1].set_ylabel('Evidence')
                axes[1].grid(alpha=0.2)
                axes[2].plot(time_axis, severity, color='#2ca02c')
                axes[2].set_ylabel('Severity')
                axes[2].set_xlabel('Time Index')
                axes[2].grid(alpha=0.2)
                fig.suptitle(f"Fold {fold} - {pieces_array[global_idx]} ({session_array[global_idx]})")
                attention_path = attention_dir / f'fold_{fold}_{session_array[global_idx]}_{local_idx}_attention.png'
                fig.tight_layout()
                fig.savefig(attention_path, dpi=300)
                plt.close(fig)
                attention_records.append(
                    {
                        'fold': fold,
                        'session_id': session_array[global_idx],
                        'piece': pieces_array[global_idx],
                        'prediction': int(preds_fold[local_idx]),
                        'label': int(targets_fold[local_idx]),
                        'figure': str(attention_path.relative_to(results_dir)),
                        'data': str(data_path.relative_to(results_dir)),
                    }
                )
            if attention_records:
                (diag_dir / f'fold_{fold}_attention_samples.json').write_text(json.dumps(attention_records, indent=2))

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    mean_f1 = float(np.mean([r['f1_score'] for r in fold_results]))
    std_f1 = float(np.std([r['f1_score'] for r in fold_results]))
    mean_acc = float(np.mean([r['accuracy'] for r in fold_results]))
    std_acc = float(np.std([r['accuracy'] for r in fold_results]))
    mean_precision = float(np.mean([np.mean(r['precision']) for r in fold_results]))
    mean_recall = float(np.mean([np.mean(r['recall']) for r in fold_results]))

    if std_f1 > 0.05:
        LOGGER.warning('%s fold variance high: std(F1)=%.4f', mode, std_f1)

    piece_summary: Dict[str, Dict[str, float]] = {}
    for piece_name, entries in piece_accumulator.items():
        piece_summary[piece_name] = {
            'f1_mean': float(np.mean([e['f1'] for e in entries])),
            'f1_std': float(np.std([e['f1'] for e in entries])),
            'accuracy_mean': float(np.mean([e['accuracy'] for e in entries])),
            'accuracy_std': float(np.std([e['accuracy'] for e in entries])),
            'total_support': int(sum(e['support'] for e in entries)),
        }

    return {
        'mode': mode,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'fold_results': fold_results,
        'piece_metrics': piece_summary,
    }


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open('r') as fh:
        return yaml.safe_load(fh) or {}


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in {'1', 'true', 'yes', 'y'}:
            return True
        if lower in {'0', 'false', 'no', 'n'}:
            return False
    return bool(value)


def _maybe_cache_audio(
    cache_path: Path,
    sensor: np.ndarray,
    audio: np.ndarray,
    quality: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    metadata: list[Dict[str, Any]],
) -> None:
    payload = {
        'sensor': sensor.astype(np.float32),
        'audio': audio.astype(np.float32),
        'quality': quality.astype(np.float32),
        'mask': mask.astype(np.float32),
        'labels': labels.astype(np.int64),
        'metadata': np.array(metadata, dtype=object),
    }
    np.savez_compressed(cache_path, **payload)


def _load_cached_audio(cache_path: Path) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]]:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    return (
        data['sensor'],
        data['audio'],
        data['quality'],
        data['mask'],
        data['labels'],
        data['metadata'].tolist(),
    )


def _apply_gaussian_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if audio.size == 0:
        return audio
    signal_power = float(np.mean(audio ** 2))
    if signal_power <= 0.0:
        return audio
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-6)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def _apply_sensor_dropout(sensor: np.ndarray, dropout_rate: float, rng: np.random.Generator) -> np.ndarray:
    if dropout_rate <= 0.0 or sensor.size == 0:
        return sensor
    dropout_rate = min(max(dropout_rate, 0.0), 1.0)
    mask = rng.random(sensor.shape, dtype=np.float32) >= dropout_rate
    return (sensor * mask).astype(np.float32)


def run_smoke_test(
    target_fold: int,
    sensor_data: np.ndarray,
    audio_data: np.ndarray,
    audio_quality: np.ndarray,
    audio_masks: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray,
    results_dir: Path,
    model_kwargs: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gkf = GroupKFold(n_splits=3)
    for fold, (_, test_idx) in enumerate(gkf.split(sensor_data, labels, groups=session_ids), start=1):
        if fold != target_fold:
            continue
        tensors = {
            'sensor': torch.FloatTensor(sensor_data[test_idx]),
            'audio': torch.FloatTensor(audio_data[test_idx]),
            'quality': torch.FloatTensor(audio_quality[test_idx]),
            'audio_mask': torch.FloatTensor(audio_masks[test_idx]),
            'label': torch.LongTensor(labels[test_idx]),
        }
        loader = _make_dataloader('multimodal', tensors, batch_size, shuffle=False)
        model = UnifiedAttentionModel(**model_kwargs).to(device)
        checkpoint = results_dir / 'models' / f'mm-audiofix_fold{fold}.pth'
        if not checkpoint.exists():
            LOGGER.warning('Smoke test skipped: checkpoint %s not found', checkpoint)
            return {}
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        metrics = evaluate(model, loader, device, 'multimodal')
        output_path = results_dir / 'diagnostics' / 'smoke_test_fold.json'
        with output_path.open('w') as fh:
            json.dump(
                {
                    'fold': fold,
                    'f1': float(metrics['f1']),
                    'accuracy': float(metrics['accuracy']),
                    'precision': np.asarray(metrics['precision'], dtype=float).tolist(),
                    'recall': np.asarray(metrics['recall'], dtype=float).tolist(),
                },
                fh,
                indent=2,
            )
        LOGGER.info('Smoke test fold %s: F1=%.3f Acc=%.3f', fold, metrics['f1'], metrics['accuracy'])
        return metrics
    return {}


def save_piece_heatmap(results: Dict[str, Any], results_dir: Path) -> None:
    # Prefer decision-level fusion as the displayed "Multimodal" if available
    multi_source = 'multimodal'
    if results.get('decision_poe'):
        multi_source = 'decision_poe'
    elif results.get('decision_stacking'):
        multi_source = 'decision_stacking'

    # Build metrics map for rows: Sensor / Audio / Multimodal(aliased)
    row_modes = ['sensor', 'audio', 'multimodal']
    metrics_map: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in ['sensor', 'audio']:
        metrics_map[key] = results.get(key, {}).get('piece_metrics', {})
    metrics_map['multimodal'] = results.get(multi_source, {}).get('piece_metrics', {})

    pieces = sorted({piece for m in row_modes for piece in metrics_map.get(m, {}).keys()})
    if not pieces:
        LOGGER.warning('No piece metrics available; heatmap generation skipped')
        return

    matrix = np.full((len(row_modes), len(pieces)), np.nan)
    for i, mode in enumerate(row_modes):
        metrics = metrics_map.get(mode, {})
        for j, piece in enumerate(pieces):
            if piece in metrics:
                cell = metrics[piece].get('f1_mean', metrics[piece].get('f1'))
                if cell is not None:
                    try:
                        matrix[i, j] = float(cell)
                    except Exception:
                        pass

    fig_width = max(10, len(pieces) * 0.5)
    import seaborn as sns  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    sns.set_theme(style='white', context='paper')
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    hm = sns.heatmap(
        matrix,
        annot=False,
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        mask=np.isnan(matrix),
        linewidths=0.6,
        linecolor='white',
        xticklabels=pieces,
        yticklabels=['Sensor', 'Audio', 'Multimodal'],
        cbar_kws={'label': 'F1 Score'},
        ax=ax,
    )
    # Manual annotations with contrast-aware colors
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val >= 0.65 else 'black'
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_ylabel('Modality')
    ax.set_title('Piece-wise F1 Heatmap (values annotated)')
    fig.tight_layout()

    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    heatmap_path = figures_dir / 'piece_f1_heatmap.png'
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)

def save_piece_correlation_plot(results: Dict[str, Any], results_dir: Path, data_csv: Optional[Path] = None) -> None:
    """Correlate piece-wise F1 with aggregate piece descriptors and plot.

    - Loads piece-level F1 (preferred decision fusion -> 'multimodal').
    - Aggregates numeric descriptors per piece from a CSV if provided.
    - Computes Pearson correlations and plots F1 vs top positively correlated metric,
      with color indicating the second most positively correlated metric.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        LOGGER.warning('pandas not available; skipping piece correlation plot')
        return

    # Choose preferred source for multimodal piece metrics
    preferred = 'decision_poe' if results.get('decision_poe') else (
        'decision_stacking' if results.get('decision_stacking') else 'multimodal'
    )
    piece_metrics = results.get(preferred, {}).get('piece_metrics', {})
    if not piece_metrics:
        LOGGER.info('No piece metrics for correlation plot')
        return

    # Build piece-level F1 dataframe
    rows = []
    for piece, m in piece_metrics.items():
        f1 = m.get('f1_mean', m.get('f1'))
        if f1 is None:
            continue
        rows.append({'piecetask_name': piece, 'f1': float(f1), 'support': float(m.get('total_support', m.get('support', 0)))})
    if not rows:
        return
    df_f1 = pd.DataFrame(rows)

    # Helper: derive basic descriptors from playdata (duration, note_count)
    def _basic_piece_descriptors(playdata_root: Path, max_per_piece: int = 80) -> Optional[pd.DataFrame]:
        try:
            import json  # noqa: F401
        except Exception:
            return None
        if not playdata_root.exists():
            return None
        counts: Dict[str, int] = {}
        acc = {}
        for meta_path in playdata_root.glob('*/meta.json'):
            try:
                import json
                meta = json.loads(meta_path.read_text())
                name = str(meta.get('name', ''))
                parts = name.split('_')
                if len(parts) < 2:
                    continue
                piece = parts[1]
                # Limit per piece to cap runtime
                c = counts.get(piece, 0)
                if c >= max_per_piece:
                    continue
                # duration
                dur = meta.get('length', None)
                # note count
                note_path = meta_path.parent / 'files' / 'hackkey' / 'note.json'
                note_count = None
                if note_path.exists():
                    try:
                        with note_path.open('r') as fh:
                            first = fh.read(1024)
                        # Reload fully only if necessary
                        import json
                        obj = json.loads((meta_path.parent / 'files' / 'hackkey' / 'note.json').read_text())
                        if isinstance(obj, dict) and 'note' in obj and isinstance(obj['note'], list):
                            note_count = int(len(obj['note']))
                    except Exception:
                        pass
                rec = acc.get(piece, {'n': 0, 'dur_sum': 0.0, 'note_sum': 0.0, 'note_n': 0})
                if isinstance(dur, (int, float)):
                    rec['dur_sum'] += float(dur)
                if isinstance(note_count, int):
                    rec['note_sum'] += float(note_count)
                    rec['note_n'] += 1
                rec['n'] += 1
                acc[piece] = rec
                counts[piece] = c + 1
            except Exception:
                continue
        if not acc:
            return None
        rows = []
        for piece, r in acc.items():
            dur_mean = (r['dur_sum'] / r['n']) if r['n'] > 0 else None
            note_mean = (r['note_sum'] / r['note_n']) if r['note_n'] > 0 else None
            rows.append({'piecetask_name': piece, 'duration_sec': dur_mean, 'note_count': note_mean, 'sessions_sampled': r['n']})
        return pd.DataFrame(rows)

    # Load external descriptors CSV if present
    if data_csv is None:
        # Default dataset CSV path relative to repo root
        data_csv = Path('profy/data/2024skillcheck_scale_arpeggio_unrefined_raw.csv')
    if not data_csv.exists():
        LOGGER.info('Descriptor CSV not found: %s; correlation plot skipped', data_csv)
        return
    try:
        df_raw = pd.read_csv(data_csv)
    except Exception as exc:
        LOGGER.warning('Failed to read %s: %s', data_csv, exc)
        return

    # Keep rows present in F1 df and coerce numeric columns
    keep = df_raw['piecetask_name'].isin(df_f1['piecetask_name'])
    df_raw = df_raw.loc[keep].copy()
    # Identify numeric columns by coercion
    for col in df_raw.columns:
        if col in ('piecetask_name',):
            continue
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    # Aggregate per piece (mean)
    df_desc = df_raw.groupby('piecetask_name').mean(numeric_only=True).reset_index()
    df = pd.merge(df_f1, df_desc, on='piecetask_name', how='left')
    # Merge basic descriptors (duration, note_count)
    basic_df = _basic_piece_descriptors(Path('profy/data/playdata'))
    if basic_df is not None and not basic_df.empty:
        df = pd.merge(df, basic_df, on='piecetask_name', how='left')
        # Derived rate: notes per second
        if 'note_count' in df.columns and 'duration_sec' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                rate = df['note_count'] / df['duration_sec']
            df['notes_per_sec'] = rate.replace([np.inf, -np.inf], np.nan)

    # Extra note-derived descriptors (IOI CV, duration CV, pitch span)
    def _extra_note_descriptors(playdata_root: Path, limits_per_piece: int = 20, max_total: int = 400) -> Optional[pd.DataFrame]:
        import json
        if not playdata_root.exists():
            return None
        target_pieces = set(df['piecetask_name'].tolist())
        acc: Dict[str, Dict[str, Any]] = {}
        counts: Dict[str, int] = {}
        scanned = 0
        for meta_path in playdata_root.glob('*/meta.json'):
            if scanned >= max_total:
                break
            try:
                meta = json.loads(meta_path.read_text())
                name = str(meta.get('name', ''))
                parts = name.split('_')
                if len(parts) < 2:
                    continue
                piece = parts[1]
                if piece not in target_pieces:
                    continue
                c = counts.get(piece, 0)
                if c >= limits_per_piece:
                    continue
                note_path = meta_path.parent / 'files' / 'hackkey' / 'note.json'
                if not note_path.exists():
                    continue
                obj = json.loads(note_path.read_text())
                if not (isinstance(obj, dict) and isinstance(obj.get('note'), list) and obj['note']):
                    continue
                notes = obj['note']
                onsets = [n.get('onset') for n in notes if isinstance(n.get('onset'), (int, float))]
                offsets = [n.get('offset') for n in notes if isinstance(n.get('offset'), (int, float))]
                keys = [n.get('key') for n in notes if isinstance(n.get('key'), (int, float))]
                ioi_cv = None
                if len(onsets) >= 2:
                    isorted = sorted(onsets)
                    ioi = np.diff(isorted)
                    mu = float(np.mean(ioi))
                    sd = float(np.std(ioi))
                    if mu > 0:
                        ioi_cv = sd / mu
                dur_cv = None
                if len(onsets) >= 1 and len(offsets) >= 1 and len(onsets) == len(offsets):
                    durs = np.asarray(offsets) - np.asarray(onsets)
                    durs = durs[durs > 0]
                    if durs.size >= 1:
                        mu = float(np.mean(durs))
                        sd = float(np.std(durs))
                        if mu > 0:
                            dur_cv = sd / mu
                span = None
                if len(keys) >= 1:
                    span = float(max(keys) - min(keys))
                rec = acc.get(piece, {'n': 0, 'ioi_cv': [], 'dur_cv': [], 'span': []})
                if ioi_cv is not None:
                    rec['ioi_cv'].append(ioi_cv)
                if dur_cv is not None:
                    rec['dur_cv'].append(dur_cv)
                if span is not None:
                    rec['span'].append(span)
                rec['n'] += 1
                acc[piece] = rec
                counts[piece] = c + 1
                scanned += 1
            except Exception:
                continue
        if not acc:
            return None
        rows: list[dict] = []
        for piece, r in acc.items():
            rows.append({
                'piecetask_name': piece,
                'ioi_cv_mean': float(np.mean(r['ioi_cv'])) if r['ioi_cv'] else np.nan,
                'dur_cv_mean': float(np.mean(r['dur_cv'])) if r['dur_cv'] else np.nan,
                'pitch_span_mean': float(np.mean(r['span'])) if r['span'] else np.nan,
                'sessions': int(r['n']),
            })
        return pd.DataFrame(rows)

    extra_df = _extra_note_descriptors(Path('profy/data/playdata'))
    if extra_df is not None and not extra_df.empty:
        df = pd.merge(df, extra_df, on='piecetask_name', how='left')

    # Heuristic key-difficulty features from piece name (accidentals count)
    def _accidentals_from_name(name: str) -> Optional[int]:
        mapping = {
            # Majors (dur)
            'C-dur': 0,
            'F-dur': -1,
            'B-dur': -2,   # Bb major
            'Es-dur': -3,  # Eb
            'As-dur': -4,  # Ab
            'Des-dur': -5, # Db
            'Ges-dur': -6, # Gb
            'D-dur': +2,
            # Minors (moll)
            'b-moll': +2,   # B minor
            'f-moll': -4,   # F minor
            'es-moll': -6,  # Eb minor
            'fis-moll': +3, # F# minor
            'cis-moll': +4, # C# minor
            'gis-moll': +5, # G# minor
        }
        # Extract key token at end of name (e.g., 'Scale Es-dur' -> 'Es-dur')
        parts = name.split()
        if not parts:
            return None
        key = parts[-1]
        return mapping.get(key)

    acc_vals: list[Optional[int]] = []
    for nm in df['piecetask_name'].tolist():
        acc_vals.append(_accidentals_from_name(str(nm)))
    df['accidentals'] = acc_vals
    df['accidentals_abs'] = df['accidentals'].abs()
    num_cols = [c for c in df.columns if c not in ('piecetask_name',) and pd.api.types.is_numeric_dtype(df[c])]
    if 'f1' not in num_cols:
        LOGGER.info('No numeric descriptors found for correlation plot')
        return
    # Filter to valid numeric columns (enough non-NaNs and non-zero variance)
    valid_cols: list[str] = []
    for c in num_cols:
        series = pd.to_numeric(df[c], errors='coerce')
        if series.notna().sum() < 3:
            continue
        if float(series.std(ddof=0)) <= 1e-12:
            continue
        valid_cols.append(c)
    if 'f1' not in valid_cols:
        LOGGER.info('F1 has zero variance or insufficient samples; correlation plot skipped')
        return
    # Pairwise Pearson correlations with F1 (avoid global corr() divide-by-zero warnings)
    pearson_map: dict[str, float] = {}
    for c in valid_cols:
        if c == 'f1':
            continue
        sub = df[[c, 'f1']].dropna()
        if len(sub) < 3:
            continue
        if float(sub[c].std(ddof=0)) <= 1e-12 or float(sub['f1'].std(ddof=0)) <= 1e-12:
            continue
        try:
            pearson_map[c] = float(sub[c].corr(sub['f1']))
        except Exception:
            continue
    pearson = pd.Series(pearson_map).dropna()
    if pearson.empty:
        LOGGER.info('No valid correlations found for piece correlation plot')
        return
    # Partial correlations controlling for support (residual method)
    def _partial_corr(col: str, control: str = 'support') -> float:
        if col not in df.columns or control not in df.columns:
            return float('nan')
        sub = df[[col, 'f1', control]].dropna()
        if len(sub) < 4:
            return float('nan')
        import numpy as _np
        try:
            zx = _np.polyfit(sub[control].values.astype(float), sub[col].values.astype(float), 1)
            zy = _np.polyfit(sub[control].values.astype(float), sub['f1'].values.astype(float), 1)
            x_res = sub[col].values.astype(float) - (zx[0] * sub[control].values.astype(float) + zx[1])
            y_res = sub['f1'].values.astype(float) - (zy[0] * sub[control].values.astype(float) + zy[1])
            r = _np.corrcoef(x_res, y_res)[0, 1]
            return float(r)
        except Exception:
            return float('nan')

    partial = {col: _partial_corr(col, 'support') for col in pearson.index}
    # Persist correlation summary (pearson + partial)
    diag_dir = results_dir / 'diagnostics'
    diag_dir.mkdir(exist_ok=True)
    import numpy as _np
    corr_df = pd.DataFrame({
        'descriptor': [c for c in num_cols if c != 'f1'],
        'pearson_with_f1': [pearson.get(c, _np.nan) for c in num_cols if c != 'f1'],
        'partial_with_f1|support': [partial.get(c, _np.nan) for c in num_cols if c != 'f1'],
    }).sort_values(by='partial_with_f1|support', ascending=False)
    corr_df.to_csv(diag_dir / 'piece_correlation_summary.csv', index=False)

    # Select plotting axes by partial correlation (exclude support-like)
    exclude = {'support', 'sessions'}
    ranked = [row for row in corr_df.itertuples(index=False) if row.descriptor not in exclude]
    if not ranked:
        ranked = list(corr_df.itertuples(index=False))
    top1 = ranked[0].descriptor
    top2 = ranked[1].descriptor if len(ranked) > 1 else None

    # Plot scatter: F1 vs top1, color by top2
    import seaborn as sns  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as _np  # type: ignore
    sns.set_theme(style='whitegrid', context='paper')
    plt.figure(figsize=(7.6, 4.6))
    x = df[top1].values
    y = df['f1'].values
    if top2 is not None:
        c = df[top2].values
        sc = plt.scatter(x, y, c=c, cmap='viridis', s=60, edgecolor='white', linewidth=0.6)
        cb = plt.colorbar(sc)
        cb.set_label(top2, rotation=270, labelpad=12)
    else:
        plt.scatter(x, y, color='#1f77b4', s=60, edgecolor='white', linewidth=0.6)
    # Annotate points with piece name (shortened)
    import matplotlib.patheffects as pe  # type: ignore
    for _, row in df.iterrows():
        label = row['piecetask_name'].replace('Scale ', 'S. ').replace('Arpeggio ', 'A. ')
        txt = plt.text(row[top1] + 1e-3, row['f1'] + 1e-3, label, fontsize=8.5, alpha=0.95, color='black')
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])
    r_partial = partial.get(top1, _np.nan)
    title = f'Piece-wise F1 vs {top1}' + (f' (partial r|support={r_partial:.2f})' if r_partial==r_partial else '')
    plt.xlabel(top1)
    plt.ylabel('F1 (piece-wise)')
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_dir = results_dir / 'figures'
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'piece_f1_correlation.png', dpi=300)
    plt.close()

def save_correlation_barplot(results: Dict[str, Any], results_dir: Path, data_csv: Optional[Path] = None, top_k: int = 10) -> None:
    """Create a horizontal bar plot of partial correlations (| support controlled) for descriptors.

    Excludes support-like descriptors from the bars. Saves figures/piece_f1_correlation_bar.png.
    """
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        LOGGER.warning('Dependencies missing for correlation barplot')
        return
    # Reuse summary created by save_piece_correlation_plot; if not present, try to compute quickly
    diag_dir = results_dir / 'diagnostics'
    corr_path = diag_dir / 'piece_correlation_summary.csv'
    if not corr_path.exists():
        save_piece_correlation_plot(results, results_dir, data_csv=data_csv)
    if not corr_path.exists():
        LOGGER.info('Correlation summary not found; barplot skipped')
        return
    try:
        corr_df = pd.read_csv(corr_path)
    except Exception as exc:
        LOGGER.warning('Failed to read correlation summary: %s', exc)
        return
    # Rank by partial correlation and exclude support-like
    exclude = {'support', 'sessions'}
    df = corr_df.loc[~corr_df['descriptor'].isin(exclude)].copy()
    df = df.sort_values(by='partial_with_f1|support', ascending=False).head(top_k)
    # Nice labels
    df['label'] = df['descriptor'].str.replace('_', ' ').str.replace('/', ' / ')
    sns.set_theme(style='whitegrid', context='paper')
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = sns.color_palette('viridis', n_colors=len(df))
    ax.barh(df['label'], df['partial_with_f1|support'], color=colors, edgecolor='white')
    ax.invert_yaxis()
    ax.set_xlabel('Partial correlation with F1 (controlling for support)')
    ax.set_ylabel('Descriptor')
    ax.axvline(0.0, color='black', linewidth=1.0, alpha=0.6)
    # Annotate values
    for i, v in enumerate(df['partial_with_f1|support']):
        offset = 0.008 if v >= 0 else -0.058
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, f"{v:.2f}", va='center', ha=ha, fontsize=9)
    fig.tight_layout()
    out_dir = results_dir / 'figures'
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / 'piece_f1_correlation_bar.png', dpi=300)
    plt.close(fig)


def save_modality_summary_and_barplot(results: Dict[str, Any], results_dir: Path) -> Optional[pd.DataFrame]:
    """Persist a compact CSV summary and a bar plot with error bars for F1 by modality.

    Files:
      - diagnostics/modality_summary.csv
      - figures/modality_f1_bar.png
    """
    # Build summary rows, aliasing decision fusion as 'multimodal' if available
    preferred = 'decision_poe' if results.get('decision_poe') else (
        'decision_stacking' if results.get('decision_stacking') else 'multimodal'
    )
    rows = []
    for mode in ("sensor", "audio"):
        if mode in results and results.get(mode) and results[mode].get("mean_f1") is not None:
            rows.append(
                {
                    "mode": mode,
                    "mean_f1": results[mode].get("mean_f1"),
                    "std_f1": results[mode].get("std_f1"),
                    "mean_accuracy": results[mode].get("mean_accuracy"),
                    "std_accuracy": results[mode].get("std_accuracy"),
                }
            )
    if preferred in results and results.get(preferred) and results[preferred].get("mean_f1") is not None:
        rows.append(
            {
                "mode": "multimodal",
                "mean_f1": results[preferred].get("mean_f1"),
                "std_f1": results[preferred].get("std_f1"),
                "mean_accuracy": results[preferred].get("mean_accuracy"),
                "std_accuracy": results[preferred].get("std_accuracy"),
            }
        )

    if not rows:
        LOGGER.warning("No results available for modality summary")
        return None

    diag_dir = results_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(diag_dir / "modality_summary.csv", index=False)

    # Bar plot with error bars (alias decision as multimodal if available)
    # Prefer PoE over stacking for display when both exist
    preferred = 'decision_poe' if results.get('decision_poe') else ('decision_stacking' if results.get('decision_stacking') else None)
    # Rebuild df_plot with only sensor/audio/multimodal
    records = []
    for key in ['sensor', 'audio']:
        if key in results and results.get(key):
            records.append({
                'mode': key,
                'mean_f1': results[key].get('mean_f1'),
                'std_f1': results[key].get('std_f1'),
            })
    # Multimodal row from preferred decision fusion if present
    source = preferred if preferred else 'multimodal'
    if source in results and results.get(source):
        records.append({
            'mode': 'multimodal',
            'mean_f1': results[source].get('mean_f1'),
            'std_f1': results[source].get('std_f1'),
        })
    df_plot = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    color_map = {"sensor": "#1b9e77", "audio": "#7570b3", "multimodal": "#d95f02"}
    colors = [color_map[m] for m in df_plot["mode"].tolist()]
    ax.bar(df_plot["mode"], df_plot["mean_f1"], yerr=df_plot["std_f1"], capsize=4, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 (mean ± std)")
    ax.set_title("Modality comparison (GroupKFold)")
    for i, v in enumerate(df_plot["mean_f1"].tolist()):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    (results_dir / "figures").mkdir(exist_ok=True)
    plt.savefig(results_dir / "figures" / "modality_f1_bar.png", dpi=300)
    plt.close()

    return df


def save_audio_quality_histograms(audio_quality: np.ndarray, results_dir: Path) -> None:
    if audio_quality.size == 0:
        LOGGER.warning('Audio quality array empty; histogram skipped')
        return
    diag_dir = results_dir / 'diagnostics' / 'audio'
    diag_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ('Non-silence rate', 0, '#2b8cbe'),
        ('Spectral flatness', 1, '#74c476'),
        ('Loudness (dB)', 2, '#fd8d3c'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (title, idx, color) in zip(axes, metrics):
        sns.histplot(audio_quality[:, idx], bins=40, kde=True, ax=ax, color=color)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].set_xlabel('Ratio')
    axes[1].set_xlabel('Flatness')
    axes[2].set_xlabel('dB')
    for ax in axes:
        ax.set_ylabel('Count')
    fig.tight_layout()
    plt.savefig(diag_dir / 'audio_quality_hist.png', dpi=300)
    plt.close(fig)

    summary: Dict[str, Dict[str, float]] = {}
    for name, idx, _ in metrics:
        column = audio_quality[:, idx]
        summary[name] = {
            'mean': float(np.mean(column)),
            'std': float(np.std(column)),
            'p05': float(np.percentile(column, 5)),
            'p50': float(np.median(column)),
            'p95': float(np.percentile(column, 95)),
        }
    (diag_dir / 'audio_quality_summary.json').write_text(json.dumps(summary, indent=2))


# Deprecated gating diagnostics removed (decision-level fusion is the canonical multimodal)


def save_audio_piece_roc(results_dir: Path, top_k: int = 6) -> None:
    """Generate per-piece ROC curves for audio-only predictions."""

    diag_dir = results_dir / 'diagnostics' / 'audio'
    if not diag_dir.exists():
        LOGGER.info('Audio diagnostics directory missing; skipping ROC plots')
        return
    csv_paths = sorted(diag_dir.glob('fold_*_predictions.csv'))
    if not csv_paths:
        LOGGER.info('No audio prediction CSVs found for ROC plotting')
        return

    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning('Failed to read %s: %s', path, exc)
            continue
        required = {'piece', 'probability', 'label'}
        if not required.issubset(df.columns):
            continue
        frames.append(df[list(required)])
    if not frames:
        LOGGER.info('Audio prediction CSVs missing required columns; skipping ROC plots')
        return

    df = pd.concat(frames, ignore_index=True)
    roc_entries: list[tuple[str, np.ndarray, np.ndarray, float, int]] = []
    for piece, group in df.groupby('piece'):
        labels = group['label'].values
        if len(np.unique(labels)) < 2:
            continue
        probs = group['probability'].values
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = float(auc(fpr, tpr))
        roc_entries.append((piece, fpr, tpr, roc_auc, len(group)))

    if not roc_entries:
        LOGGER.info('Insufficient class diversity per piece for ROC plots')
        return

    roc_entries.sort(key=lambda item: item[3], reverse=True)
    top_k = min(top_k, len(roc_entries))
    cols = 3
    rows = int(np.ceil(top_k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    for idx, (piece, fpr, tpr, roc_auc, support) in enumerate(roc_entries[:top_k]):
        ax = axes[idx]
        ax.plot(fpr, tpr, color='#1f77b4', linewidth=1.5, label=f'AUC={roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1)
        ax.set_title(f'{piece}\nN={support}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)

    for ax in axes[top_k:]:
        ax.axis('off')

    fig.tight_layout()
    out_dir = results_dir / 'figures'
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / 'audio_piece_roc.png', dpi=300)
    plt.close(fig)


def _load_prediction_frames(results_dir: Path, mode: str) -> pd.DataFrame:
    diag_dir = results_dir / 'diagnostics' / mode
    if not diag_dir.exists():
        LOGGER.info('Prediction diagnostics missing for mode %s', mode)
        return pd.DataFrame()
    frames = []
    for csv_path in sorted(diag_dir.glob('fold_*_predictions.csv')):
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning('Failed to read %s: %s', csv_path, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# -------------------------
# Decision-level fusion (PoE / Stacking)
# -------------------------

def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _collect_fold_pred_paths(results_dir: Path, mode: str) -> list[Path]:
    diag_dir = results_dir / 'diagnostics' / mode
    if not diag_dir.exists():
        return []
    return sorted(diag_dir.glob('fold_*_predictions.csv'))


def _parse_fold_num(path: Path) -> int:
    name = path.stem  # e.g., fold_1_predictions
    try:
        return int(name.split('_')[1])
    except Exception:
        return -1


def _evaluate_decision_for_fold(
    train_s: pd.DataFrame,
    train_a: pd.DataFrame,
    test_s: pd.DataFrame,
    test_a: pd.DataFrame,
    method: str = 'poe',
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, predictions) for the test fold using the chosen method.

    train_* frames correspond to concatenated folds excluding the current test fold.
    test_* frames correspond to the current test fold.
    """
    # Align columns
    for df in (train_s, train_a, test_s, test_a):
        if 'probability' not in df.columns or 'label' not in df.columns:
            raise ValueError('Prediction CSV missing required columns: probability/label')

    y_train = train_s['label'].to_numpy(dtype=int)
    ps_train = train_s['probability'].to_numpy(dtype=float)
    pa_train = train_a['probability'].to_numpy(dtype=float)
    ps_log_train = _safe_logit(ps_train)
    pa_log_train = _safe_logit(pa_train)

    ps_test = test_s['probability'].to_numpy(dtype=float)
    pa_test = test_a['probability'].to_numpy(dtype=float)
    ps_log_test = _safe_logit(ps_test)
    pa_log_test = _safe_logit(pa_test)

    if method == 'stacking':
        X_tr = np.stack([ps_log_train, pa_log_train], axis=1)
        clf = LogisticRegression(max_iter=200, solver='liblinear')
        clf.fit(X_tr, y_train)
        X_te = np.stack([ps_log_test, pa_log_test], axis=1)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return prob, pred

    # Default: PoE with simple weight grid search on training folds
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.5, 1.0, 1.5, 2.0]
    best_alpha = 1.0
    best_beta = 1.0
    best_f1 = -1.0
    for a in alphas:
        for b in betas:
            z = a * ps_log_train + b * pa_log_train
            p = _sigmoid(z)
            y_hat = (p >= 0.5).astype(int)
            f1 = f1_score(y_train, y_hat, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = a
                best_beta = b
    z_te = best_alpha * ps_log_test + best_beta * pa_log_test
    prob = _sigmoid(z_te)
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def run_decision_fusion(results_dir: Path, method: str = 'poe') -> Dict[str, Any]:
    """Evaluate decision-level fusion using existing sensor/audio fold predictions.

    method: 'poe' or 'stacking'
    Returns a results dict matching run_cv() structure, including per-piece metrics.
    """
    sensor_paths = _collect_fold_pred_paths(results_dir, 'sensor')
    audio_paths = _collect_fold_pred_paths(results_dir, 'audio')
    if not sensor_paths or not audio_paths:
        LOGGER.warning('Decision fusion skipped: missing sensor/audio diagnostics')
        return {}

    # Map fold -> DataFrame
    fold_to_s: Dict[int, pd.DataFrame] = {}
    fold_to_a: Dict[int, pd.DataFrame] = {}
    folds: list[int] = []
    for sp in sensor_paths:
        f = _parse_fold_num(sp)
        if f < 0:
            continue
        fold_to_s[f] = pd.read_csv(sp)
        folds.append(f)
    for ap in audio_paths:
        f = _parse_fold_num(ap)
        if f < 0:
            continue
        fold_to_a[f] = pd.read_csv(ap)
    folds = sorted(set(folds) & set(fold_to_a.keys()))
    if not folds:
        LOGGER.warning('Decision fusion skipped: no aligned folds')
        return {}

    fold_results: list[Dict[str, Any]] = []
    piece_accumulator: defaultdict[str, list[Dict[str, float]]] = defaultdict(list)

    # Save under diagnostics/multimodal to unify downstream tooling
    diag_dir = results_dir / 'diagnostics' / 'multimodal'
    diag_dir.mkdir(parents=True, exist_ok=True)

    for f in folds:
        test_s = fold_to_s[f]
        test_a = fold_to_a[f]
        # Build train (OOF) from other folds
        train_s = pd.concat([fold_to_s[k] for k in folds if k != f], ignore_index=True)
        train_a = pd.concat([fold_to_a[k] for k in folds if k != f], ignore_index=True)

        # Defensive alignment check (length equality)
        if len(test_s) != len(test_a):
            min_len = min(len(test_s), len(test_a))
            test_s = test_s.iloc[:min_len].reset_index(drop=True)
            test_a = test_a.iloc[:min_len].reset_index(drop=True)

        prob, pred = _evaluate_decision_for_fold(train_s, train_a, test_s, test_a, method=method)
        y_true = test_s['label'].to_numpy(dtype=int)

        # Metrics
        f1 = f1_score(y_true, pred, average='weighted')
        acc = accuracy_score(y_true, pred)
        cm = confusion_matrix(y_true, pred)
        precision, recall, per_class_f1, support = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0
        )

        fold_record: Dict[str, Any] = {
            'fold': f,
            'f1_score': float(f1),
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist(),
            'precision': np.asarray(precision, dtype=float).tolist(),
            'recall': np.asarray(recall, dtype=float).tolist(),
            'per_class_f1': np.asarray(per_class_f1, dtype=float).tolist(),
            'support': np.asarray(support, dtype=int).tolist(),
        }

        # Piece-wise metrics
        if 'piece' in test_s.columns:
            piece_metrics_fold: Dict[str, Dict[str, float]] = {}
            pieces = test_s['piece'].tolist()
            for piece_name in set(pieces):
                mask = (test_s['piece'] == piece_name).to_numpy()
                if not np.any(mask):
                    continue
                pf1 = f1_score(y_true[mask], pred[mask], average='macro', zero_division=0)
                pacc = float(np.mean(y_true[mask] == pred[mask]))
                supp = int(mask.sum())
                entry = {'f1': float(pf1), 'accuracy': pacc, 'support': supp}
                piece_metrics_fold[piece_name] = entry
                piece_accumulator[piece_name].append(entry)
            fold_record['piece_metrics'] = piece_metrics_fold

        fold_results.append(fold_record)

        # Save diagnostics
        with (diag_dir / f'fold_{f}_confusion_matrix.json').open('w') as fh:
            json.dump({'fold': f, 'mode': 'multimodal', 'confusion_matrix': cm.tolist()}, fh, indent=2)
        out_df = pd.DataFrame(
            {
                'piece': test_s.get('piece', pd.Series(['unknown'] * len(y_true))).tolist(),
                'probability': prob,
                'prediction': pred,
                'label': y_true,
            }
        )
        out_df.to_csv(diag_dir / f'fold_{f}_predictions.csv', index=False)

    # Aggregate
    mean_f1 = float(np.mean([r['f1_score'] for r in fold_results]))
    std_f1 = float(np.std([r['f1_score'] for r in fold_results]))
    mean_acc = float(np.mean([r['accuracy'] for r in fold_results]))
    std_acc = float(np.std([r['accuracy'] for r in fold_results]))
    mean_precision = float(np.mean([np.mean(r['precision']) for r in fold_results]))
    mean_recall = float(np.mean([np.mean(r['recall']) for r in fold_results]))

    piece_summary: Dict[str, Dict[str, float]] = {}
    for piece_name, entries in piece_accumulator.items():
        piece_summary[piece_name] = {
            'f1_mean': float(np.mean([e['f1'] for e in entries])),
            'f1_std': float(np.std([e['f1'] for e in entries])),
            'accuracy_mean': float(np.mean([e['accuracy'] for e in entries])),
            'accuracy_std': float(np.std([e['accuracy'] for e in entries])),
            'total_support': int(sum(e['support'] for e in entries)),
        }

    return {
        'mode': 'multimodal',
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'fold_results': fold_results,
        'piece_metrics': piece_summary,
    }


def save_confusion_matrix_figure(results: Dict[str, Any], results_dir: Path) -> None:
    modes = [
        ('sensor', 'Sensor-only'),
        ('audio', 'Audio-only'),
        ('multimodal', 'Multimodal'),
    ]
    fig_dir = results_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(modes), figsize=(14, 4))
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    for idx, ((mode_key, title), ax) in enumerate(zip(modes, axes)):
        fold_results = results.get(mode_key, {}).get('fold_results', [])
        if not fold_results:
            ax.axis('off')
            continue
        matrix = None
        for fold in fold_results:
            cm = np.asarray(fold['confusion_matrix'], dtype=float)
            matrix = cm if matrix is None else matrix + cm
        if matrix is None:
            ax.axis('off')
            continue
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
        sns.heatmap(
            normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar=idx == 0,
            cbar_ax=cbar_ax if idx == 0 else None,
            annot_kws={'fontsize': 10},
        )
        ax.set_title(f"{title}\nF1 {results[mode_key]['mean_f1']:.3f} ± {results[mode_key]['std_f1']:.3f}")
        ax.set_xlabel('Predicted')
        if idx == 0:
            ax.set_ylabel('Actual')
        ax.set_xticklabels(['Amateur', 'Professional'], rotation=45, ha='right')
        ax.set_yticklabels(['Amateur', 'Professional'], rotation=0)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    pdf_path = fig_dir / 'confusion_matrix.pdf'
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix('.png'), dpi=300)
    plt.close(fig)


def save_baseline_comparison_figure(results: Dict[str, Any], results_dir: Path) -> None:
    # Prefer decision-level fusion as Multimodal for display
    preferred = 'decision_poe' if results.get('decision_poe') else (
        'decision_stacking' if results.get('decision_stacking') else 'multimodal'
    )

    rows = []
    def add_row(key: str, label: str) -> None:
        if key in results and results.get(key):
            rows.append({
                'Modality': label,
                'F1': results[key].get('mean_f1'),
                'F1_std': results[key].get('std_f1'),
                'Accuracy': results[key].get('mean_accuracy'),
                'Accuracy_std': results[key].get('std_accuracy'),
            })

    add_row('sensor', 'Sensor')
    add_row('audio', 'Audio')
    add_row(preferred, 'Multimodal')
    rows = [r for r in rows if r.get('F1') is not None]
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    palette = ['#1b9e77', '#7570b3', '#d95f02']
    for ax, metric in zip(axes, ['F1', 'Accuracy']):
        ax.bar(df['Modality'], df[metric], yerr=df[f'{metric}_std'], capsize=5, color=palette[: len(df)])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} (mean ± std)')
        for tick, value in enumerate(df[metric]):
            ax.text(tick, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    out_path = results_dir / 'figures' / 'baseline_comparison.png'
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix('.pdf'))
    plt.close(fig)


def save_synthetic_interventions_figure(results: Dict[str, Any], results_dir: Path) -> None:
    multi = results.get('multimodal', {}).get('piece_metrics', {})
    sensor = results.get('sensor', {}).get('piece_metrics', {})
    audio = results.get('audio', {}).get('piece_metrics', {})
    if not multi:
        LOGGER.info('No piece metrics available for synthetic intervention plot')
        return

    sensor_delta = []
    audio_delta = []
    for piece, metrics in multi.items():
        if piece in sensor:
            sensor_delta.append((piece, metrics['f1_mean'] - sensor[piece]['f1_mean']))
        if piece in audio:
            audio_delta.append((piece, metrics['f1_mean'] - audio[piece]['f1_mean']))

    sensor_delta.sort(key=lambda item: item[1], reverse=True)
    audio_delta.sort(key=lambda item: item[1], reverse=True)
    sensor_delta = sensor_delta[:8]
    audio_delta = audio_delta[:8]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    for ax, deltas, title in zip(
        axes,
        [sensor_delta, audio_delta],
        ['Benefit of adding audio features', 'Benefit of adding sensor features'],
    ):
        if not deltas:
            ax.axis('off')
            continue
        pieces, values = zip(*deltas)
        ax.barh(pieces[::-1], values[::-1], color='#1f77b4')
        ax.set_xlabel('ΔF1 (multimodal - baseline)')
        ax.set_title(title)
        ax.set_xlim(left=min(0.0, min(values) - 0.01), right=max(values) + 0.02)
        for idx, val in enumerate(values[::-1]):
            ax.text(val + 0.002, pieces[::-1][idx], f'{val:.3f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / 'figures' / 'synthetic_interventions.png', dpi=300)
    plt.close(fig)


def save_statistical_analysis_figure(results: Dict[str, Any], results_dir: Path) -> None:
    multimodal = results.get('multimodal', {}).get('piece_metrics', {})
    if not multimodal:
        LOGGER.info('No multimodal piece metrics for statistical analysis figure')
        return
    sensor = results.get('sensor', {}).get('piece_metrics', {})
    audio = results.get('audio', {}).get('piece_metrics', {})

    rows = []
    for piece, metrics in multimodal.items():
        rows.append(
            {
                'piece': piece,
                'support': metrics.get('total_support'),
                'f1': metrics.get('f1_mean'),
                'acc': metrics.get('accuracy_mean'),
                'delta_sensor': metrics.get('f1_mean') - sensor.get(piece, {}).get('f1_mean', np.nan),
                'delta_audio': metrics.get('f1_mean') - audio.get(piece, {}).get('f1_mean', np.nan),
            }
        )

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(
        df['support'],
        df['f1'],
        c=df['delta_sensor'] - df['delta_audio'],
        cmap='coolwarm',
        s=np.clip(df['support'] / df['support'].max(), 0.05, 1.0) * 400,
        alpha=0.75,
    )
    ax.set_xlabel('Total support (samples)')
    ax.set_ylabel('F1 (multimodal)')
    ax.set_title('Piece-level performance and modality gains')
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label('Δ Sensor gain - Δ Audio gain')
    for _, row in df.nlargest(5, 'support').iterrows():
        ax.annotate(row['piece'], (row['support'], row['f1']), fontsize=8, xytext=(5, 5), textcoords='offset points')
    fig.tight_layout()
    out_path = results_dir / 'figures' / 'experiment_1_statistical_analysis.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'), dpi=300)
    plt.close(fig)


def save_efficiency_analysis_figure(gating_df: Optional[pd.DataFrame], results_dir: Path) -> None:
    """Deprecated (gating removed). No-op."""
    return


def save_attention_faithfulness_figure(results_dir: Path, gating_df: Optional[pd.DataFrame]) -> None:
    """Deprecated (gating removed). No-op."""
    return


def save_prescription_calibration_figure(results_dir: Path, predictions_df: pd.DataFrame) -> None:
    if predictions_df.empty or 'probability' not in predictions_df.columns:
        LOGGER.info('Skipping calibration plot: probability column missing')
        return
    bins = np.linspace(0.0, 1.0, 11)
    predictions_df = predictions_df.copy()
    predictions_df['bin'] = pd.cut(predictions_df['probability'], bins, include_lowest=True)
    grouped = predictions_df.groupby('bin')
    calibration = grouped['correct'].mean()
    confidence = grouped['probability'].mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect calibration')
    ax.plot(confidence, calibration, marker='o', color='#1f78b4', label='Profy')
    ax.set_xlabel('Predicted probability (Pro)')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title('Prediction calibration (multimodal)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = results_dir / 'figures' / 'prescription_calibration.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_domain_robustness_figure(results: Dict[str, Any], results_dir: Path) -> None:
    multimodal = results.get('multimodal', {}).get('piece_metrics', {})
    if not multimodal:
        return
    df = (
        pd.DataFrame(
            [
                {
                    'piece': piece,
                    'f1': metrics['f1_mean'],
                    'support': metrics['total_support'],
                }
                for piece, metrics in multimodal.items()
            ]
        )
        .sort_values('f1', ascending=False)
    )
    top_n = min(15, len(df))
    fig, ax = plt.subplots(figsize=(8, 6))
    subset = df.head(top_n)
    ax.barh(subset['piece'][::-1], subset['f1'][::-1], color='#4c78a8')
    for idx, (piece, f1, support) in enumerate(zip(subset['piece'][::-1], subset['f1'][::-1], subset['support'][::-1])):
        ax.text(f1 + 0.01, idx, f'{f1:.3f}\nN={support}', va='center', fontsize=8)
    ax.set_xlabel('F1 score')
    ax.set_title('Top pieces by multimodal performance')
    ax.set_xlim(0.0, 1.05)
    ax.grid(alpha=0.2, axis='x')
    fig.tight_layout()
    fig.savefig(results_dir / 'figures' / 'domain_robustness.png', dpi=300)
    plt.close(fig)


def save_window_size_analysis_figure(results_dir: Path) -> None:
    attention_dir = results_dir / 'diagnostics' / 'multimodal' / 'attention_arrays'
    out_path = results_dir / 'figures' / 'window_size_analysis.png'
    if not attention_dir.exists():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No attention samples available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return
    window_lengths: list[int] = []
    for npz_path in attention_dir.glob('*.npz'):
        with np.load(npz_path) as payload:
            severity = payload['severity']
        threshold = np.quantile(severity, 0.9)
        mask = severity > threshold
        if not mask.any():
            continue
        current = 0
        for value in mask:
            if value:
                current += 1
            elif current:
                window_lengths.append(current)
                current = 0
        if current:
            window_lengths.append(current)
    if not window_lengths:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'Attention samples present but no high-severity spans', ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(window_lengths, bins=20, ax=ax, color='#f28e2b')
    ax.set_xlabel('Consecutive high-attention timesteps')
    ax.set_ylabel('Count')
    ax.set_title('Local attention span distribution (90th percentile)')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_cross_validation_results_figure(results: Dict[str, Any], results_dir: Path) -> None:
    rows = []
    label_map = {'sensor': 'Sensor', 'audio': 'Audio', 'multimodal': 'Multimodal'}
    for key, label in label_map.items():
        for fold in results.get(key, {}).get('fold_results', []):
            rows.append(
                {'Modality': label, 'Fold': fold['fold'], 'F1': fold['f1_score'], 'Accuracy': fold['accuracy']}
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=df, x='Fold', y='F1', hue='Modality', ax=axes[0])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title('Fold-wise F1')
    axes[0].grid(alpha=0.2, axis='y')
    sns.barplot(data=df, x='Fold', y='Accuracy', hue='Modality', ax=axes[1])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title('Fold-wise Accuracy')
    axes[1].grid(alpha=0.2, axis='y')
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='lower right')
    fig.tight_layout()
    out_path = results_dir / 'figures' / 'cross_validation_results.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'), dpi=300)
    plt.close(fig)


def save_feature_importance_figures(
    results: Dict[str, Any],
    results_dir: Path,
    gating_df: Optional[pd.DataFrame],
) -> None:
    """Deprecated (gating removed). No-op."""
    return


def generate_attention_case_studies(results_dir: Path) -> None:
    """Deprecated (attention case studies for gating model removed). No-op."""
    return

def save_ablation_summary(results: Dict[str, Any], results_dir: Path, config_payload: Dict[str, Any]) -> None:
    """Persist a simple summary highlighting modality deltas for quick ablations."""

    metrics_summary: list[Dict[str, Any]] = []
    for mode in ('sensor', 'audio', 'multimodal'):
        if mode not in results:
            continue
        entry = {
            'mode': mode,
            'mean_f1': results[mode].get('mean_f1'),
            'std_f1': results[mode].get('std_f1'),
            'mean_accuracy': results[mode].get('mean_accuracy'),
            'std_accuracy': results[mode].get('std_accuracy'),
        }
        metrics_summary.append(entry)

    if not metrics_summary:
        return

    ablation_dir = results_dir / 'diagnostics'
    ablation_dir.mkdir(exist_ok=True)
    payload = {
        'config': config_payload,
        'metrics': metrics_summary,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }
    (ablation_dir / 'ablation_summary.json').write_text(json.dumps(payload, indent=2))


def _validate_sensor_vs_audio(
    results: Dict[str, Any],
    allow_audio_dominant: bool = False,
    tolerance: float = 0.0,
) -> None:
    """Ensure the sensor-only baseline is not worse than the audio-only branch."""

    sensor = results.get('sensor', {}).get('mean_f1') if isinstance(results.get('sensor'), dict) else None
    audio = results.get('audio', {}).get('mean_f1') if isinstance(results.get('audio'), dict) else None

    if sensor is None or audio is None:
        LOGGER.warning('Skipping sensor/audio validation (missing metrics): sensor=%s audio=%s', sensor, audio)
        return

    if allow_audio_dominant:
        LOGGER.info('Sensor/audio validation skipped (allow_audio_dominant=True); sensor=%.3f audio=%.3f', sensor, audio)
        return

    if sensor + tolerance < audio:
        raise ValueError(
            f'Sensor Macro-F1 ({sensor:.3f}) is lower than audio Macro-F1 ({audio:.3f}); investigate fusion gating or data integrity.'
        )


def summarize_expert_annotations(base_dir: Path, results_dir: Path) -> None:
    """Summarize Piano-Performance-Marker annotations if available.

    Scans JSON files under `base_dir` (excluding users.json and _index.json) and
    writes summary JSON and simple figures for inclusion in papers.

    Outputs:
      - diagnostics/expert_eval/summary.json
      - figures/expert_segments_hist.png
      - figures/expert_evaluator_bars.png
    """
    if not base_dir.exists():
        LOGGER.info("Expert annotation base_dir %s not found; skipping", base_dir)
        return

    json_paths = [
        p
        for p in base_dir.rglob("*.json")
        if p.name != "users.json" and not p.name.startswith("_index")
    ]
    if not json_paths:
        LOGGER.info("No expert annotation JSON files under %s", base_dir)
        return
    annotations = []
    evaluator_counts: dict[str, int] = {}
    seg_lengths: list[float] = []
    for path in json_paths:
        try:
            data = json.loads(path.read_text())
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Failed to parse %s: %s", path, e)
            continue
        evaluator = data.get("evaluator", "unknown")
        evaluator_counts[evaluator] = evaluator_counts.get(evaluator, 0) + 1
        for seg in data.get("problem_sections", []) or []:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end > start:
                seg_lengths.append(end - start)
        annotations.append({
            "evaluator": evaluator,
            "audio": data.get("audio_filename") or data.get("audio_file"),
            "n_segments": len(data.get("problem_sections", []) or []),
            "total_score": data.get("total_score"),
        })

    diag_dir = results_dir / "diagnostics" / "expert_eval"
    diag_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_annotations": len(annotations),
        "n_problem_segments": int(len(seg_lengths)),
        "segment_duration_sec": {
            "mean": float(np.mean(seg_lengths)) if seg_lengths else 0.0,
            "median": float(np.median(seg_lengths)) if seg_lengths else 0.0,
        },
        "by_evaluator": evaluator_counts,
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figures
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    if seg_lengths:
        plt.figure(figsize=(6, 3.2))
        sns.histplot(seg_lengths, bins=20, kde=True, color="#3182bd")
        plt.xlabel("Problem segment duration (sec)")
        plt.ylabel("Count")
        plt.title("Expert-marked segment durations")
        plt.tight_layout()
        plt.savefig(fig_dir / "expert_segments_hist.png", dpi=300)
        plt.close()

    if evaluator_counts:
        names = list(evaluator_counts.keys())
        counts = [evaluator_counts[k] for k in names]
        plt.figure(figsize=(6, 3.2))
        sns.barplot(x=names, y=counts, color="#a1d99b")
        plt.ylabel("# annotations")
        plt.title("Expert annotations by evaluator")
        plt.tight_layout()
        plt.savefig(fig_dir / "expert_evaluator_bars.png", dpi=300)
        plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description='Run Profy tri-modality experiments')
    parser.add_argument('--debug', action='store_true', help='Run a quick pipeline check on a tiny subset')
    parser.add_argument('--config', type=str, help='Path to YAML config overriding defaults')
    parser.add_argument('--data-dir', type=str, default='/home/kazuki/Projects/Profy/data', help='Dataset root directory')
    parser.add_argument('--modality-dropout', type=float, default=0.0)
    parser.add_argument('--min-sensor-weight', type=float, default=0.35)
    parser.add_argument(
        '--max-sensor-weight',
        type=float,
        default=0.6,
        help='Upper clamp applied when audio quality is low (>= min-sensor-weight)',
    )
    parser.add_argument('--gating-temp-scale', type=float, default=2.0)
    parser.add_argument('--spec-augment-prob', type=float, default=0.0)
    parser.add_argument('--spec-augment-ratio', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='Beta distribution alpha for mixup (0 to disable)')
    parser.add_argument('--mixup-prob', type=float, default=0.0, help='Probability of applying mixup per batch')
    parser.add_argument('--fusion-warmup-epochs', type=int, default=2, help='Number of warmup epochs (per fold) before enabling gating')
    parser.add_argument('--cache-audio', action='store_true', help='Use cached audio features if available')
    parser.add_argument('--cache-path', type=str, help='Override cache path for audio features')
    parser.add_argument('--save-checkpoints', action='store_true', help='Persist model checkpoints per fold')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test with best sensor fold')
    parser.add_argument('--entropy-regularization', type=float, default=0.1, help='Entropy regularization weight (lambda) for gating diversity (set 0 to disable)')
    parser.add_argument('--lambda-mil', type=float, default=0.5, help='Weight for MIL loss on evidence aggregation (0 to disable)')
    parser.add_argument('--lambda-evidence-l1', type=float, default=0.001, help='L1 sparsity regularization weight on evidence scores')
    parser.add_argument('--mil-topk-frac', type=float, default=0.1, help='Soft top-k fraction for MIL (0 to disable)')
    parser.add_argument('--mil-blend-alpha', type=float, default=0.5, help='Blend between top-k and Noisy-OR (0..1)')
    parser.add_argument('--lambda-attention-energy', type=float, default=0.0, help='Penalty weight to discourage attention aligning with frame energy')
    parser.add_argument('--use-balanced-sampler', action='store_true', help='Enable class-balanced sampling for training splits')
    parser.add_argument('--disable-temporal-attention', action='store_true', help='Disable local attention pooling (uniform averaging instead)')
    parser.add_argument('--disable-residual-heads', action='store_true', help='Disable residual statistical heads for sensor/audio summaries')
    parser.add_argument('--disable-audio-tcn', action='store_true', help='Disable dilated temporal convolutions in the audio encoder')
    parser.add_argument('--noise-audio-snr', type=float, nargs='*', default=[], help='Evaluate robustness with Gaussian audio noise at specified SNR levels (dB)')
    parser.add_argument('--noise-sensor-dropout', type=float, nargs='*', default=[], help='Evaluate robustness with sensor dropout rates (0-1)')
    parser.add_argument('--export-modality-summaries', action='store_true', help='Write modality summary CSV/plots to shared locations')
    parser.add_argument('--latest-modality-dir', type=str, help='Directory to store the latest modality summary (defaults to results/latest_modality)')
    parser.add_argument('--paper-table-dir', type=str, help='Directory for paper-ready tables (defaults to paper/tables)')
    parser.add_argument('--override-config', action='append', default=[], help='Additional YAML config(s) applied after --config')
    args = parser.parse_args()

    config = {
        'n_folds': 3,
        'max_epochs': 15,
        'patience': 5,
        'batch_size': 32,
        'min_sensor_weight': args.min_sensor_weight,
        'max_sensor_weight': args.max_sensor_weight,
    }
    config.update(_load_config(args.config))
    for override_path in args.override_config or []:
        override_payload = _load_config(override_path)
        if override_payload:
            LOGGER.info('Applying override config: %s', override_path)
            config.update(override_payload)

    if args.debug:
        config.update({'n_folds': 2, 'max_epochs': 3, 'patience': 1, 'batch_size': 16})

    mixup_alpha = float(config.get('mixup_alpha', args.mixup_alpha))
    mixup_prob = float(config.get('mixup_prob', args.mixup_prob))
    fusion_warmup = int(config.get('fusion_warmup_epochs', args.fusion_warmup_epochs))
    entropy_regularization = float(config.get('entropy_regularization', args.entropy_regularization))
    lambda_mil = float(config.get('lambda_mil', args.lambda_mil))
    lambda_evidence_l1 = float(config.get('lambda_evidence_l1', args.lambda_evidence_l1))
    mil_topk_frac = float(config.get('mil_topk_frac', args.mil_topk_frac))
    mil_blend_alpha = float(config.get('mil_blend_alpha', args.mil_blend_alpha))
    lambda_attention_energy = float(config.get('lambda_attention_energy', args.lambda_attention_energy))
    use_balanced_sampler = _resolve_bool(config.get('use_balanced_sampler', args.use_balanced_sampler), False)

    noise_audio_snr_cfg = config.get('noise_audio_snr', args.noise_audio_snr)
    if isinstance(noise_audio_snr_cfg, (int, float)):
        noise_audio_snr = [float(noise_audio_snr_cfg)]
    else:
        noise_audio_snr = [float(x) for x in (noise_audio_snr_cfg or [])]

    noise_sensor_dropout_cfg = config.get('noise_sensor_dropout', args.noise_sensor_dropout)
    if isinstance(noise_sensor_dropout_cfg, (int, float)):
        noise_sensor_dropout = [float(noise_sensor_dropout_cfg)]
    else:
        noise_sensor_dropout = [float(x) for x in (noise_sensor_dropout_cfg or [])]

    results_dir = ensure_results_dir()
    log_file = results_dir / 'logs' / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    LOGGER.info('Initialized logging. Results dir: %s', results_dir)

    data_dir = Path(config.get('data_dir', args.data_dir)).resolve()
    max_samples = 60 if args.debug else None
    cache_path = Path(args.cache_path) if args.cache_path else data_dir / 'audio_feature_cache.npz'

    dataset = None
    if args.cache_audio and max_samples is None:
        cached = _load_cached_audio(cache_path)
        if cached is not None:
            dataset = cached
            LOGGER.info('Loaded audio cache from %s', cache_path)

    if dataset is None:
        X, y, metadata, audio, quality, mask = load_real_piano_data(
            data_dir=str(data_dir),
            max_samples=max_samples,
            load_audio=True,
            augment_audio=args.spec_augment_prob > 0.0,
            spec_augment_prob=args.spec_augment_prob,
            spec_augment_max_ratio=args.spec_augment_ratio,
        )
        dataset = (X, audio, quality, mask, y, metadata)
        if args.cache_audio and max_samples is None:
            _maybe_cache_audio(cache_path, X, audio, quality, mask, y, metadata)
            LOGGER.info('Cached audio features to %s', cache_path)

    sensor_data, audio_data, audio_quality, audio_masks, labels, metadata = dataset
    session_ids = np.array([m['session_id'] for m in metadata])

    repo_root = Path(__file__).resolve().parents[2]
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo_root), text=True).strip()
    except Exception:  # pragma: no cover
        git_head = 'unknown'

    min_sensor_weight = float(config.get('min_sensor_weight', args.min_sensor_weight))
    max_sensor_weight = float(config.get('max_sensor_weight', args.max_sensor_weight))
    model_kwargs = {
        'modality_dropout_p': args.modality_dropout,
        'min_sensor_weight': min_sensor_weight,
        'max_sensor_weight': max_sensor_weight,
        'gating_temperature_scale': args.gating_temp_scale,
        'use_temporal_attention': not args.disable_temporal_attention,
        'use_residual_heads': not args.disable_residual_heads,
        'use_audio_tcn': not args.disable_audio_tcn,
    }
    config_model_kwargs = config.get('model_kwargs') or {}
    if config_model_kwargs:
        model_kwargs.update(config_model_kwargs)

    min_sensor_weight = float(model_kwargs.get('min_sensor_weight', min_sensor_weight))
    max_sensor_weight = float(model_kwargs.get('max_sensor_weight', max_sensor_weight))
    model_kwargs['min_sensor_weight'] = min_sensor_weight
    model_kwargs['max_sensor_weight'] = max(max_sensor_weight, min_sensor_weight)
    max_sensor_weight = float(model_kwargs['max_sensor_weight'])
    gating_temperature = float(model_kwargs.get('gating_temperature_scale', args.gating_temp_scale))

    config_payload = {
        'n_folds': config['n_folds'],
        'max_epochs': config['max_epochs'],
        'patience': config['patience'],
        'batch_size': config['batch_size'],
        'debug_mode': args.debug,
        'modality_dropout': args.modality_dropout,
        'min_sensor_weight': min_sensor_weight,
        'max_sensor_weight': max_sensor_weight,
        'gating_temp_scale': gating_temperature,
        'spec_augment_prob': args.spec_augment_prob,
        'spec_augment_ratio': args.spec_augment_ratio,
        'mixup_alpha': mixup_alpha,
        'mixup_prob': mixup_prob,
        'fusion_warmup_epochs': fusion_warmup,
        'entropy_regularization': entropy_regularization,
        'lambda_mil': lambda_mil,
        'lambda_evidence_l1': lambda_evidence_l1,
        'mil_topk_frac': mil_topk_frac,
        'mil_blend_alpha': mil_blend_alpha,
        'lambda_attention_energy': lambda_attention_energy,
        'use_balanced_sampler': use_balanced_sampler,
        'noise_audio_snr': noise_audio_snr,
        'noise_sensor_dropout': noise_sensor_dropout,
        'override_configs': args.override_config,
    }
    config_hash = hashlib.sha1(json.dumps(config_payload, sort_keys=True).encode('utf-8')).hexdigest()

    LOGGER.info('Run metadata: git_head=%s config_hash=%s debug=%s', git_head, config_hash, args.debug)
    LOGGER.info(
        'Dataset shapes: sensor=%s audio=%s labels=%s quality=%s mask=%s',
        sensor_data.shape,
        audio_data.shape,
        labels.shape,
        audio_quality.shape,
        audio_masks.shape,
    )
    LOGGER.info(
        'Audio quality mean=%s std=%s',
        np.round(audio_quality.mean(axis=0), 4).tolist(),
        np.round(audio_quality.std(axis=0), 4).tolist(),
    )

    save_audio_quality_histograms(audio_quality, results_dir)

    manifest_payload = {
        'config': config_payload,
        'model_kwargs': model_kwargs,
    }
    (results_dir / 'config_manifest.json').write_text(json.dumps(manifest_payload, indent=2))

    save_models = args.save_checkpoints or not args.debug

    results: Dict[str, Any] = {}
    results['sensor'] = run_cv(
        sensor_data,
        None,
        None,
        None,
        labels,
        session_ids,
        metadata,
        mode='sensor',
        results_dir=results_dir,
        n_folds=config['n_folds'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        model_kwargs=model_kwargs,
        save_checkpoints=save_models,
        checkpoint_prefix='sensor-',
        entropy_regularization=entropy_regularization,
        use_balanced_sampler=use_balanced_sampler,
        lambda_mil=lambda_mil,
        lambda_evidence_l1=lambda_evidence_l1,
        mil_topk_frac=mil_topk_frac,
        mil_blend_alpha=mil_blend_alpha,
        lambda_attention_energy=lambda_attention_energy,
    )
    results['audio'] = run_cv(
        sensor_data,
        audio_data,
        audio_quality,
        audio_masks,
        labels,
        session_ids,
        metadata,
        mode='audio',
        results_dir=results_dir,
        n_folds=config['n_folds'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
        fusion_warmup_epochs=0,
        model_kwargs=model_kwargs,
        save_checkpoints=save_models,
        checkpoint_prefix='audio-',
        entropy_regularization=entropy_regularization,
        use_balanced_sampler=use_balanced_sampler,
        lambda_mil=lambda_mil,
        lambda_evidence_l1=lambda_evidence_l1,
        mil_topk_frac=mil_topk_frac,
        mil_blend_alpha=mil_blend_alpha,
        lambda_attention_energy=lambda_attention_energy,
    )

    # Decision-level fusion as the canonical Multimodal result (post-hoc; negligible training cost)
    try:
        results['multimodal'] = run_decision_fusion(results_dir, method='poe')
    except Exception as exc:  # pragma: no cover
        LOGGER.warning('Decision fusion failed: %s', exc)
        results['multimodal'] = {}

    noise_conditions: list[Dict[str, Any]] = []
    for snr in noise_audio_snr:
        noise_conditions.append({'tag': f'audio_snr_{int(snr)}', 'audio_snr': snr})
    for rate in noise_sensor_dropout:
        percentage = int(rate * 100)
        noise_conditions.append({'tag': f'sensor_dropout_{percentage}', 'sensor_dropout': rate})

    noise_summary_rows: list[Dict[str, Any]] = []
    for cond in noise_conditions:
        cond_results: Dict[str, Any] = {}
        for mode_name, prefix in [('multimodal', 'noise-mm'), ('sensor', 'noise-sensor'), ('audio', 'noise-audio')]:
            cond_results[mode_name] = run_cv(
                sensor_data,
                audio_data,
                audio_quality,
                audio_masks,
                labels,
                session_ids,
                metadata,
                mode=mode_name,
                results_dir=results_dir,
                n_folds=config['n_folds'],
                max_epochs=config['max_epochs'],
                patience=config['patience'],
                batch_size=config['batch_size'],
                mixup_alpha=mixup_alpha if mode_name != 'sensor' else 0.0,
                mixup_prob=mixup_prob if mode_name != 'sensor' else 0.0,
                fusion_warmup_epochs=fusion_warmup if mode_name == 'multimodal' else 0,
                model_kwargs=model_kwargs,
                save_checkpoints=False,
                checkpoint_prefix=f"{prefix}-{cond['tag']}",
                entropy_regularization=entropy_regularization,
                use_balanced_sampler=use_balanced_sampler,
                noise_settings=cond,
                lambda_mil=lambda_mil,
                lambda_evidence_l1=lambda_evidence_l1,
                mil_topk_frac=mil_topk_frac,
                mil_blend_alpha=mil_blend_alpha,
                lambda_attention_energy=lambda_attention_energy,
            )
            metrics = cond_results[mode_name]
            noise_summary_rows.append({
                'condition': cond['tag'],
                'mode': mode_name,
                'mean_f1': metrics.get('mean_f1'),
                'std_f1': metrics.get('std_f1'),
                'mean_accuracy': metrics.get('mean_accuracy'),
                'std_accuracy': metrics.get('std_accuracy'),
            })
        results[f"noise_{cond['tag']}"] = cond_results

    if noise_summary_rows:
        noise_df = pd.DataFrame(noise_summary_rows)
        diag_dir = results_dir / 'diagnostics'
        diag_dir.mkdir(exist_ok=True)
        noise_df.to_csv(diag_dir / 'noise_summary.csv', index=False)

    if args.smoke_test:
        best_sensor_fold = int(
            max(results['sensor']['fold_results'], key=lambda item: item['f1_score'])['fold']
        )
        run_smoke_test(
            best_sensor_fold,
            sensor_data,
            audio_data,
            audio_quality,
            audio_masks,
            labels,
            session_ids,
            results_dir,
            model_kwargs,
            config['batch_size'],
        )

    with (results_dir / 'complete_results.json').open('w') as fh:
        json.dump(results, fh, indent=2)

    _validate_sensor_vs_audio(results, allow_audio_dominant=args.debug)

    save_piece_heatmap(results, results_dir)
    modality_df = save_modality_summary_and_barplot(results, results_dir)
    save_audio_piece_roc(results_dir)
    # Piece-wise correlation analysis (uses dataset CSV if available)
    try:
        data_csv_path = Path('profy/data/2024skillcheck_scale_arpeggio_unrefined_raw.csv')
    except Exception:
        data_csv_path = None
    save_piece_correlation_plot(results, results_dir, data_csv=data_csv_path)
    save_ablation_summary(results, results_dir, config_payload)

    save_confusion_matrix_figure(results, results_dir)
    save_baseline_comparison_figure(results, results_dir)
    save_synthetic_interventions_figure(results, results_dir)
    save_statistical_analysis_figure(results, results_dir)

    multimodal_predictions = _load_prediction_frames(results_dir, 'multimodal')
    if not multimodal_predictions.empty:
        if 'correct' not in multimodal_predictions.columns:
            multimodal_predictions['correct'] = (
                multimodal_predictions['prediction'] == multimodal_predictions['label']
            ).astype(int)
        save_prescription_calibration_figure(results_dir, multimodal_predictions)

    save_domain_robustness_figure(results, results_dir)
    save_window_size_analysis_figure(results_dir)
    save_cross_validation_results_figure(results, results_dir)

    figure_artifacts = [
        'baseline_comparison.png',
        'confusion_matrix.pdf',
        'confusion_matrix.png',
        'synthetic_interventions.png',
        'experiment_1_statistical_analysis.pdf',
        'experiment_1_statistical_analysis.png',
        'prescription_calibration.png',
        'domain_robustness.png',
        'window_size_analysis.png',
        'cross_validation_results.pdf',
        'cross_validation_results.png',
        'piece_f1_heatmap.png',
        'modality_f1_bar.png',
        'audio_piece_roc.png',
        'expert_segments_hist.png',
        'expert_evaluator_bars.png',
    ]
    # Optional: summarize external expert annotations if present
    summarize_expert_annotations(
        Path("/home/kazuki/Projects/Profy/piano-performance-marker/web_evaluations"),
        results_dir,
    )

    if args.export_modality_summaries and modality_df is not None:
        latest_dir = (
            Path(args.latest_modality_dir).resolve()
            if args.latest_modality_dir
            else (results_dir.parent / 'latest_modality').resolve()
        )
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_csv = latest_dir / 'modality_summary.csv'
        modality_df.to_csv(latest_csv, index=False)

        bar_plot = results_dir / 'figures' / 'modality_f1_bar.png'
        if bar_plot.exists():
            shutil.copy(bar_plot, latest_dir / 'modality_f1_bar.png')

        for fig_name in figure_artifacts:
            fig_path = results_dir / 'figures' / fig_name
            if fig_path.exists():
                shutil.copy(fig_path, latest_dir / fig_name)

        paper_tables_dir = (
            Path(args.paper_table_dir).resolve()
            if args.paper_table_dir
            else (repo_root / 'paper' / 'tables')
        )
        paper_tables_dir.mkdir(parents=True, exist_ok=True)
        paper_csv = paper_tables_dir / 'modality_summary.csv'
        modality_df.to_csv(paper_csv, index=False)

        heatmap_src = results_dir / 'figures' / 'piece_f1_heatmap.png'
        if heatmap_src.exists():
            shutil.copy(heatmap_src, latest_dir / 'piece_f1_heatmap.png')

    paper_fig_dir = repo_root / 'paper' / 'figures'
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_name in figure_artifacts:
        fig_path = results_dir / 'figures' / fig_name
        if fig_path.exists():
            shutil.copy(fig_path, paper_fig_dir / fig_name)

    print('Experiment complete. Results saved to:', results_dir)


if __name__ == '__main__':
    main()
