#!/usr/bin/env python3
"""Run three-modality experiments (multimodal/sensor/audio) with UnifiedAttentionModel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

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
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_tgts = [], []

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
        all_preds.extend(preds.detach().cpu().numpy())
        all_tgts.extend(label.detach().cpu().numpy())

        gate = outputs.get('modality_weights')
        if gate is not None:
            modality_weights.append(gate.detach().cpu().numpy())

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
        'precision': precision,
        'recall': recall,
        'per_class_f1': per_class_f1,
        'support': support,
    }


def _make_dataloader(
    mode: str,
    tensors: Dict[str, Optional[torch.Tensor]],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if mode == 'sensor':
        dataset = TensorDataset(tensors['sensor'], tensors['label'])

        def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            sensors, labels = list(zip(*batch))
            return {
                'sensor': torch.stack(sensors),
                'label': torch.stack(labels),
            }

    elif mode == 'audio':
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

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def run_cv(
    sensor_data: np.ndarray,
    audio_data: Optional[np.ndarray],
    audio_quality: Optional[np.ndarray],
    audio_masks: Optional[np.ndarray],
    labels: np.ndarray,
    session_ids: np.ndarray,
    mode: str,
    results_dir: Path,
    n_folds: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_checkpoints: bool = False,
    checkpoint_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gkf = GroupKFold(n_splits=n_folds)
    fold_results: list[Dict[str, Any]] = []
    model_kwargs = model_kwargs or {}
    checkpoint_prefix = checkpoint_prefix or mode

    for fold, (train_idx, test_idx) in enumerate(gkf.split(sensor_data, labels, groups=session_ids), start=1):
        tensors_test: Dict[str, Optional[torch.Tensor]] = {
            'sensor': torch.FloatTensor(sensor_data[test_idx]),
            'audio': torch.FloatTensor(audio_data[test_idx]) if audio_data is not None else None,
            'quality': torch.FloatTensor(audio_quality[test_idx]) if audio_quality is not None else None,
            'audio_mask': torch.FloatTensor(audio_masks[test_idx]) if audio_masks is not None else None,
            'label': torch.LongTensor(labels[test_idx]),
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

        train_loader = _make_dataloader(mode, tensors_sub, batch_size, shuffle=True)
        val_loader = _make_dataloader(mode, tensors_val, batch_size, shuffle=False)

        model = UnifiedAttentionModel(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        best_state = model.state_dict()
        best_val = -1.0
        bad_epochs = 0

        for _ in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device, mode, scaler)
            val_metrics = evaluate(model, val_loader, device, mode)
            if val_metrics['f1'] > best_val:
                best_val = val_metrics['f1']
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break

        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader, device, mode)

        fold_results.append({
            'fold': fold,
            'f1_score': float(test_metrics['f1']),
            'accuracy': float(test_metrics['accuracy']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'precision': np.asarray(test_metrics['precision'], dtype=float).tolist(),
            'recall': np.asarray(test_metrics['recall'], dtype=float).tolist(),
            'per_class_f1': np.asarray(test_metrics['per_class_f1'], dtype=float).tolist(),
            'support': np.asarray(test_metrics['support'], dtype=int).tolist(),
        })

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

        if test_metrics['modality_weights'] is not None:
            df = pd.DataFrame(test_metrics['modality_weights'], columns=['sensor_weight', 'audio_weight'])
            df['prediction'] = test_metrics['predictions']
            df['label'] = test_metrics['targets']
            df.to_csv(diag_dir / f'fold_{fold}_modality_weights.csv', index=False)

        if save_checkpoints:
            checkpoint_path = results_dir / 'models' / f'{checkpoint_prefix}_fold{fold}.pth'
            torch.save(model.state_dict(), checkpoint_path)

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

    return {
        'mode': mode,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'fold_results': fold_results,
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Profy tri-modality experiments')
    parser.add_argument('--debug', action='store_true', help='Run a quick pipeline check on a tiny subset')
    parser.add_argument('--config', type=str, help='Path to YAML config overriding defaults')
    parser.add_argument('--data-dir', type=str, default='/home/kazuki/Projects/Profy/data', help='Dataset root directory')
    parser.add_argument('--modality-dropout', type=float, default=0.0)
    parser.add_argument('--min-sensor-weight', type=float, default=0.35)
    parser.add_argument('--gating-temp-scale', type=float, default=2.0)
    parser.add_argument('--spec-augment-prob', type=float, default=0.0)
    parser.add_argument('--spec-augment-ratio', type=float, default=0.1)
    parser.add_argument('--cache-audio', action='store_true', help='Use cached audio features if available')
    parser.add_argument('--cache-path', type=str, help='Override cache path for audio features')
    parser.add_argument('--save-checkpoints', action='store_true', help='Persist model checkpoints per fold')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test with best sensor fold')
    args = parser.parse_args()

    config = {
        'n_folds': 3,
        'max_epochs': 15,
        'patience': 5,
        'batch_size': 32,
    }
    config.update(_load_config(args.config))

    if args.debug:
        config.update({'n_folds': 2, 'max_epochs': 3, 'patience': 1, 'batch_size': 16})

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

    config_payload = {
        'n_folds': config['n_folds'],
        'max_epochs': config['max_epochs'],
        'patience': config['patience'],
        'batch_size': config['batch_size'],
        'debug_mode': args.debug,
        'modality_dropout': args.modality_dropout,
        'min_sensor_weight': args.min_sensor_weight,
        'gating_temp_scale': args.gating_temp_scale,
        'spec_augment_prob': args.spec_augment_prob,
        'spec_augment_ratio': args.spec_augment_ratio,
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

    model_kwargs = {
        'modality_dropout_p': args.modality_dropout,
        'min_sensor_weight': args.min_sensor_weight,
        'gating_temperature_scale': args.gating_temp_scale,
    }

    save_models = args.save_checkpoints or not args.debug

    results: Dict[str, Any] = {}
    results['multimodal'] = run_cv(
        sensor_data,
        audio_data,
        audio_quality,
        audio_masks,
        labels,
        session_ids,
        mode='multimodal',
        results_dir=results_dir,
        n_folds=config['n_folds'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        model_kwargs=model_kwargs,
        save_checkpoints=save_models,
        checkpoint_prefix='mm-audiofix',
    )
    results['sensor'] = run_cv(
        sensor_data,
        None,
        None,
        None,
        labels,
        session_ids,
        mode='sensor',
        results_dir=results_dir,
        n_folds=config['n_folds'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        model_kwargs=model_kwargs,
        save_checkpoints=save_models,
        checkpoint_prefix='sensor-v1',
    )
    results['audio'] = run_cv(
        sensor_data,
        audio_data,
        audio_quality,
        audio_masks,
        labels,
        session_ids,
        mode='audio',
        results_dir=results_dir,
        n_folds=config['n_folds'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        model_kwargs=model_kwargs,
        save_checkpoints=save_models,
        checkpoint_prefix='audio-v1',
    )

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

    print('Experiment complete. Results saved to:', results_dir)


if __name__ == '__main__':
    main()
