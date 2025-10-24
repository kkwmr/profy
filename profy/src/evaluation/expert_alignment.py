#!/usr/bin/env python3
"""
Expert annotation alignment analysis.

This module aligns human web annotations (Piano Performance Marker) with
model-indicated problematic segments over time and reports agreement metrics.

It supports the following modalities, if checkpoints are available:
- audio-only
- sensor-only
- multimodal (skipped if no checkpoint)

Inputs
- Web annotations: piano-performance-marker/web_evaluations/<user>/*.json
- Audio files: piano-performance-marker/public/audio/<filename>
- Manifest: piano-performance-marker/public/audio/manifest_top20.json
- Trained model checkpoints: results/<experiment>/models

Outputs (saved under results/<out_dir>)
- summary.json: aggregate metrics per modality and per evaluator
- per_audio_metrics.csv: row per (user, audio)
- figures/overlay_<mode>_<user>_<audio>.png: severity vs annotation overlay

Filtering
- Excludes usernames starting with 'test' (case-insensitive)
- Excludes evaluators who do not have all 20 top-audio evaluations

Usage
  python -m src.evaluation.expert_alignment \
    --experiment results/experiment_20251009_224938 \
    --web-evals ../piano-performance-marker/web_evaluations \
    --audio-dir ../piano-performance-marker/public/audio \
    --manifest ../piano-performance-marker/public/audio/manifest_top20.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from ..models.unified_attention_model import UnifiedAttentionModel


LOGGER = logging.getLogger(__name__)


# -------------------------
# Data structures
# -------------------------


@dataclass
class EvalItem:
    username: str
    audio_filename: str
    problem_sections: List[Tuple[float, float]]  # seconds
    json_path: Path


@dataclass
class AudioFeatures:
    features: np.ndarray  # [T, 128]
    mask: np.ndarray  # [T]
    quality: np.ndarray  # [3]
    duration_sec: float


@dataclass
class SensorSequence:
    sequence: np.ndarray  # [T, 88]


# -------------------------
# Utilities and parameters
# -------------------------


@dataclass
class EvalParams:
    severity_type: str = "product"  # one of: product, evidence, attention
    severity_power: float = 1.0      # gamma to sharpen peaks
    smooth_win: int = 1              # moving average window (odd, >=1)
    norm_minmax: bool = False        # scale to [0,1]
    lag_sec: float = 0.0             # global shift (seconds, +right)
    ann_margin_sec: float = 0.0      # expand annotation at both ends


def _load_manifest(manifest_path: Path) -> List[str]:
    payload = json.loads(manifest_path.read_text())
    filenames = [str(entry.get("filename")) for entry in payload if entry.get("filename")]
    return filenames


def _iter_web_evals(base_dir: Path) -> Iterable[Tuple[str, Path]]:
    for user_dir in sorted(base_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        username = user_dir.name
        if username.lower().startswith("test"):
            continue
        for json_path in sorted(user_dir.glob("*.json")):
            yield username, json_path


def _parse_eval_json(username: str, path: Path) -> Optional[EvalItem]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to parse %s: %s", path, exc)
        return None
    audio = data.get("audio_filename") or data.get("audio_file") or data.get("audio")
    if not audio:
        return None
    sections = []
    for seg in (data.get("problem_sections") or []):
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end > start:
                sections.append((start, end))
        except Exception:
            continue
    return EvalItem(username=username, audio_filename=str(audio), problem_sections=sections, json_path=path)


def _ensure_out_dir(out_dir: Path) -> None:
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def _load_audio_feature_stats(data_root: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    stats_path = data_root / "audio_feature_stats.json"
    if not stats_path.exists():
        return None
    try:
        payload = json.loads(stats_path.read_text())
        mean = np.array(payload.get("mean"), dtype=np.float32)
        std = np.array(payload.get("std"), dtype=np.float32)
        if mean.shape != (128,) or std.shape != (128,):
            return None
        return mean, std
    except Exception:
        return None


def _extract_audio_features(audio_path: Path, sequence_length: int = 1000) -> AudioFeatures:
    """Replicate the training audio feature pipeline for a single audio file.

    Produces 128-dim features, an RMS-derived mask, and a simple quality vector.
    """
    import librosa
    import librosa.util as librosa_util

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Loudness normalization similar to training loader
    rms_ref = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms_ref > 1e-6:
        y = y * float(0.1 / rms_ref)

    target_frames = sequence_length
    hop_length = max(1, len(y) // (target_frames + 1))
    n_fft = min(2048, len(y))

    # Core features
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=n_fft, hop_length=hop_length, fmin=30.0, fmax=8000.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=4)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    # Spectral flux from mel diffs
    mel_mag = np.sqrt(np.maximum(S, 1e-10))
    mel_diff = np.diff(mel_mag, axis=1)
    if mel_diff.size == 0:
        spectral_flux = np.zeros((1, mel_mag.shape[1]), dtype=np.float64)
    else:
        flux = np.sqrt(np.square(mel_diff).sum(axis=0, dtype=np.float64))
        spectral_flux = np.concatenate([[flux[0]], flux]).reshape(1, -1)
    # Temporal features
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    rms_delta = librosa.feature.delta(rms)

    frame_count = S_db.shape[1]
    align_list = [
        mfccs,
        chroma,
        tonnetz,
        spec_contrast,
        spec_centroid,
        spec_bandwidth,
        spec_rolloff,
        spectral_flatness,
        spectral_flux,
        zcr,
        rms,
        rms_delta,
    ]
    # Ensure all have same frame count
    for idx, arr in enumerate(align_list):
        if arr.shape[1] != frame_count:
            align_list[idx] = librosa_util.fix_length(arr, frame_count, axis=1)
    (
        mfccs,
        chroma,
        tonnetz,
        spec_contrast,
        spec_centroid,
        spec_bandwidth,
        spec_rolloff,
        spectral_flatness,
        spectral_flux,
        zcr,
        rms_feat,
        rms_delta_feat,
    ) = align_list

    # Reduce chroma to 9 dims to keep total at 128
    chroma_var = chroma.var(axis=1)
    drop_idx = np.argsort(chroma_var)[:3]
    chroma_reduced = np.delete(chroma, drop_idx, axis=0)

    feature_stack = [
        S_db,
        mfccs,
        chroma_reduced,
        tonnetz,
        spec_contrast,
        spec_centroid,
        spec_bandwidth,
        spec_rolloff,
        spectral_flatness,
        spectral_flux,
        zcr,
        rms_feat,
        rms_delta_feat,
    ]
    feats = np.vstack(feature_stack).T.astype(np.float32)
    if feats.shape[1] != 128:
        feats = librosa_util.fix_length(feats.T, 128, axis=0).T.astype(np.float32)

    # Per-session clipping (same heuristic)
    clip_vals = np.percentile(np.abs(feats), 99.0, axis=0)
    clip_vals = np.where(clip_vals < 1e-5, 1.0, clip_vals)
    feats = np.clip(feats, -clip_vals, clip_vals, out=feats)

    raw_length = feats.shape[0]
    pad_len = 0
    if raw_length > sequence_length:
        feats = feats[:sequence_length]
    elif raw_length < sequence_length:
        pad_len = sequence_length - raw_length
        feats = np.vstack([feats, np.zeros((pad_len, 128), dtype=np.float32)])

    frame_rms = rms_feat.flatten()
    nsr = float((frame_rms > (frame_rms.mean() * 0.5)).mean())
    flat = float(np.mean(spectral_flatness))
    loud = float(20 * np.log10(max(1e-8, frame_rms.mean())))
    quality = np.array([nsr, flat, loud], dtype=np.float32)

    # Build mask for non-silent frames; drop padded tail
    mask = (frame_rms > (frame_rms.mean() * 0.3)).astype(np.float32)
    if mask.size > sequence_length:
        mask = mask[:sequence_length]
    elif mask.size < sequence_length:
        mask = np.pad(mask, (0, sequence_length - mask.size), mode="constant", constant_values=0.0)
    if pad_len > 0:
        mask[-pad_len:] = 0.0

    return AudioFeatures(features=feats, mask=mask, quality=quality, duration_sec=duration)


def _load_sensor_sequence(hackkey_csv: Path, sequence_length: int = 1000) -> Optional[SensorSequence]:
    import pandas as pd  # lazy import to keep module import light

    if not hackkey_csv.exists():
        return None
    try:
        df = pd.read_csv(hackkey_csv, header=None)
    except Exception:  # pragma: no cover
        return None
    data = df.iloc[:, 1:89].values.astype(np.float32)
    if data.shape[1] != 88:
        return None
    if data.shape[0] > sequence_length:
        idx = np.linspace(0, data.shape[0] - 1, sequence_length, dtype=int)
        data = data[idx]
    elif data.shape[0] < sequence_length:
        pad = np.zeros((sequence_length - data.shape[0], 88), dtype=np.float32)
        data = np.vstack([data, pad])
    return SensorSequence(sequence=data)


def _mask_from_segments(segments: List[Tuple[float, float]], duration: float, out_len: int, margin_sec: float = 0.0) -> np.ndarray:
    mask = np.zeros((out_len,), dtype=np.float32)
    if duration <= 0.0:
        return mask
    for (start, end) in segments:
        s_sec = max(0.0, start - margin_sec)
        e_sec = max(0.0, end + margin_sec)
        s = int(max(0, min(out_len, np.floor(s_sec / duration * out_len))))
        e = int(max(0, min(out_len, np.ceil(e_sec / duration * out_len))))
        if e > s:
            mask[s:e] = 1.0
    return mask

def _series_from_model_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Return time-series components needed for severity building."""
    att_t = outputs.get("attention_weights")  # [B, L]
    evid_t = outputs.get("evidence_scores")   # [B, L]
    if att_t is None or evid_t is None:
        return {"attention": np.array([], dtype=np.float32),
                "evidence": np.array([], dtype=np.float32),
                "product": np.array([], dtype=np.float32)}
    att = att_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    evid = evid_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    prod = (att * evid).astype(np.float32)
    return {"attention": att, "evidence": evid, "product": prod}


def _apply_params_to_series(series: np.ndarray, duration_sec: float, params: EvalParams) -> np.ndarray:
    x = series.astype(np.float32)
    # Power transform to emphasize peaks
    if params.severity_power != 1.0:
        # Ensure non-negativity and numeric stability
        x = np.clip(x, a_min=0.0, a_max=None) ** float(params.severity_power)
    # Smoothing with moving average
    win = max(1, int(params.smooth_win))
    if win > 1:
        if win % 2 == 0:
            win += 1
        k = np.ones((win,), dtype=np.float32) / float(win)
        x = np.convolve(x, k, mode="same")
    # Normalize to [0,1]
    if params.norm_minmax:
        mn = float(x.min())
        mx = float(x.max())
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x)
    # Apply global lag (seconds)
    if abs(params.lag_sec) > 1e-9 and duration_sec > 0.0:
        shift_frames = int(round(params.lag_sec / duration_sec * len(x)))
        if shift_frames != 0:
            if shift_frames > 0:
                x = np.concatenate([np.zeros((shift_frames,), dtype=np.float32), x])[: len(series)]
            else:
                x = np.concatenate([x, np.zeros((-shift_frames,), dtype=np.float32)])[-len(series):]
    return x


def _load_model(model_kwargs: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> UnifiedAttentionModel:
    model = UnifiedAttentionModel(**model_kwargs)
    sd = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


def _collect_checkpoints(models_dir: Path) -> Dict[str, List[Path]]:
    """Find available checkpoints by modality key."""
    mapping: Dict[str, List[Path]] = {"audio": [], "sensor": [], "multimodal": []}
    if not models_dir.exists():
        return mapping
    for p in models_dir.iterdir():
        name = p.name.lower()
        if name.endswith(".pth"):
            if name.startswith("audio-"):
                mapping["audio"].append(p)
            elif name.startswith("sensor-"):
                mapping["sensor"].append(p)
            elif name.startswith("mm-") or name.startswith("multimodal-"):
                mapping["multimodal"].append(p)
    for k in mapping:
        mapping[k] = sorted(mapping[k])
    return mapping


def _aggregate_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if y_true.size == 0 or scores.size == 0 or y_true.sum() == 0:
        return {"roc_auc": float("nan"), "ap": float("nan"), "best_f1": float("nan"), "best_iou": float("nan"), "pearson": float("nan")}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["ap"] = float(average_precision_score(y_true, scores))
    except Exception:
        metrics["ap"] = float("nan")
    # Pearson correlation (point-biserial)
    try:
        x = (scores - scores.mean()) / (scores.std() + 1e-8)
        y = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
        metrics["pearson"] = float(np.clip((x * y).mean(), -1.0, 1.0))
    except Exception:
        metrics["pearson"] = float("nan")
    # Threshold sweep for best F1 and IoU
    best_f1 = 0.0
    best_iou = 0.0
    for q in np.linspace(0.1, 0.9, 17):
        thr = float(np.quantile(scores, q))
        pred = (scores >= thr).astype(np.uint8)
        f1 = f1_score(y_true, pred)
        inter = float((pred & (y_true > 0)).sum())
        union = float(((pred > 0) | (y_true > 0)).sum())
        iou = inter / union if union > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
        if iou > best_iou:
            best_iou = iou
    metrics["best_f1"] = float(best_f1)
    metrics["best_iou"] = float(best_iou)
    return metrics


def _plot_overlay(out_path: Path, scores: np.ndarray, labels: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 3))
    t = np.arange(len(scores))
    plt.plot(t, scores, color="#2ca02c", label="severity")
    plt.fill_between(t, 0, labels, color="#d62728", alpha=0.3, step="pre", label="annotation")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Score / Mask")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Main analysis
# -------------------------


def run_alignment(
    experiment_dir: Path,
    web_eval_dir: Path,
    audio_dir: Path,
    manifest_path: Path,
    data_root: Path,
    out_dir: Path,
    device: Optional[str] = None,
    tune: bool = False,
    primary_metric: str = "ap",
    params_grid: Optional[Dict[str, List[Any]]] = None,
    fixed_params: Optional[EvalParams] = None,
    consensus: str = "none",
    consensus_thr: float = 0.5,
    min_evaluators: int = 1,
) -> None:
    _ensure_out_dir(out_dir)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model kwargs
    config_manifest = experiment_dir / "config_manifest.json"
    model_kwargs: Dict[str, Any] = {}
    if config_manifest.exists():
        try:
            payload = json.loads(config_manifest.read_text())
            model_kwargs = payload.get("model_kwargs", {}) or {}
        except Exception:
            model_kwargs = {}

    # Collect checkpoints
    ckpts = _collect_checkpoints(experiment_dir / "models")
    if not any(ckpts.values()):
        raise FileNotFoundError(f"No checkpoints found under {experiment_dir / 'models'}")

    # Manifest list (Top-20 set)
    top20_filenames = set(_load_manifest(manifest_path))

    # Gather and filter evaluations
    items: List[EvalItem] = []
    for username, json_path in _iter_web_evals(web_eval_dir):
        item = _parse_eval_json(username, json_path)
        if item is None:
            continue
        if item.audio_filename not in top20_filenames:
            # Ignore files outside Top-20
            continue
        items.append(item)

    # Keep evaluators with exactly all 20 top-20 evaluations
    by_user: Dict[str, List[EvalItem]] = {}
    for it in items:
        by_user.setdefault(it.username, []).append(it)
    eligible_users = {u for u, lst in by_user.items() if len({x.audio_filename for x in lst}) == 20}
    items = [it for it in items if it.username in eligible_users]

    if not items:
        LOGGER.warning("No eligible evaluations after filtering; nothing to do")
        return

    # Preload audio feature stats if available
    stats = _load_audio_feature_stats(data_root)
    global_mean, global_std = (None, None)
    if stats is not None:
        global_mean, global_std = stats

    # Cache models per modality/fold
    models_cache: Dict[Tuple[str, Path], UnifiedAttentionModel] = {}

    # Accumulators
    rows: List[Dict[str, Any]] = []
    per_user_summary: Dict[str, Dict[str, List[float]]] = {}

    # Cache severity components for reuse across parameter sweeps
    base_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for item in sorted(items, key=lambda x: (x.username.lower(), x.audio_filename)):
        audio_path = audio_dir / item.audio_filename

        # Extract audio features
        af = _extract_audio_features(audio_path)
        feats = af.features
        if global_mean is not None and global_std is not None:
            feats = (feats - global_mean.reshape(1, -1)) / (global_std.reshape(1, -1) + 1e-6)
        audio_t = torch.from_numpy(feats).unsqueeze(0).float().to(device_t)  # [1, T, 128]
        mask_t = torch.from_numpy(af.mask).unsqueeze(0).float().to(device_t)

        # Derive sensor path from manifest's source_path if present
        # The manifest stores it as sibling JSON; we parse from web evaluation JSON too if present
        # but to keep simple, use known path convention replacing audio filename with source mapping not needed now
        # We'll infer session dir from a mapping stored in manifest entries
        # Load manifest map for this filename to get source_path
        session_sensor_seq: Optional[SensorSequence] = None
        try:
            manifest_data = json.loads(manifest_path.read_text())
            m_lookup = {e.get("filename"): e for e in manifest_data}
            src = m_lookup.get(item.audio_filename, {}).get("source_path")
            if src:
                src_path = Path(src)
                hackkey_csv = src_path.parent.parent / "hackkey" / "hackkey.csv"
                session_sensor_seq = _load_sensor_sequence(hackkey_csv)
        except Exception:
            session_sensor_seq = None

        # Build annotation mask at model output frame rate per modality (computed per forward)
        # We'll compute per modality separately because output lengths differ.

        # Iterate modalities available
        for mode in ("audio", "sensor", "multimodal"):
            if not ckpts.get(mode):
                # Skip if no checkpoints found
                continue
            if mode == "sensor" and session_sensor_seq is None:
                # Cannot run sensor-only without sensor data
                continue
            # Average series over folds if multiple checkpoints
            series_list: List[Dict[str, np.ndarray]] = []
            for ckpt_path in ckpts[mode]:
                key = (mode, ckpt_path)
                if key not in models_cache:
                    models_cache[key] = _load_model(model_kwargs, ckpt_path, device_t)
                model = models_cache[key]
                with torch.no_grad():
                    if mode == "audio":
                        out = model(sensor_data=None, audio_data=audio_t, audio_mask=mask_t, audio_quality=None)
                    elif mode == "sensor":
                        sensor_np = session_sensor_seq.sequence
                        sensor_t = torch.from_numpy(sensor_np).unsqueeze(0).float().to(device_t)
                        out = model(sensor_data=sensor_t, audio_data=None)
                    else:  # multimodal
                        if session_sensor_seq is None:
                            continue
                        sensor_np = session_sensor_seq.sequence
                        sensor_t = torch.from_numpy(sensor_np).unsqueeze(0).float().to(device_t)
                        out = model(sensor_data=sensor_t, audio_data=audio_t, audio_mask=mask_t, audio_quality=None)
                series = _series_from_model_outputs(out)
                if series.get("product", np.array([])).size > 0:
                    series_list.append(series)
            if not series_list:
                continue
            # Average across folds for each series type
            keys = ["product", "evidence", "attention"]
            avg_series = {k: np.mean(np.stack([s[k] for s in series_list], axis=0), axis=0) for k in keys}
            base_cache[(item.username, item.audio_filename, mode)] = {
                "duration": af.duration_sec,
                "series": avg_series,
            }

    if not base_cache:
        LOGGER.warning("No base series computed; cannot proceed")
        return

    # Build parameter grid
    if params_grid is None:
        params_grid = {
            "severity_type": ["product"],
            "severity_power": [1.0],
            "smooth_win": [1],
            "norm_minmax": [False],
            "lag_sec": [0.0],
            "ann_margin_sec": [0.0],
        }

    def _iter_param_sets(grid: Dict[str, List[Any]]) -> List[EvalParams]:
        from itertools import product
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        params_list: List[EvalParams] = []
        for combo in product(*values):
            kw = dict(zip(keys, combo))
            params_list.append(EvalParams(**kw))
        return params_list

    # If not tuning, just use fixed or first in grid
    if not tune:
        p = fixed_params or _iter_param_sets(params_grid)[0]
        best_by_mode = {"audio": p, "sensor": p, "multimodal": p}
    else:
        # Grid search per mode
        best_by_mode: Dict[str, EvalParams] = {}
        for mode in ("audio", "sensor", "multimodal"):
            # Skip modes with no data
            if not any(k[2] == mode for k in base_cache.keys()):
                continue
            best_score = -1.0
            best_params_mode: Optional[EvalParams] = None
            for p in _iter_param_sets(params_grid):
                scores: List[float] = []
                for (user, audio_fn, m), data in base_cache.items():
                    if m != mode:
                        continue
                    duration = float(data["duration"]) if data.get("duration") is not None else 0.0
                    series = data["series"].get(p.severity_type, np.array([], dtype=np.float32))
                    if series.size == 0:
                        continue
                    sev = _apply_params_to_series(series, duration, p)
                    ann = _mask_from_segments(by_user[user][0].problem_sections if False else [], duration, len(sev), margin_sec=p.ann_margin_sec)
                    # Note: annotations are per (user,audio); fetch proper item
                # To build correct annotation quickly, pre-index items
                break
            # We break to implement proper annotation lookup below

        # Build index for annotation lookup once
        index_ann: Dict[Tuple[str, str], Tuple[List[Tuple[float, float]], float]] = {}
        for (user, audio_fn, m), data in base_cache.items():
            index_ann[(user, audio_fn)] = (by_user[user][0].problem_sections if False else [], data["duration"])  # placeholder
        # Rebuild index correctly from items
        index_ann.clear()
        for it in items:
            index_ann[(it.username, it.audio_filename)] = (it.problem_sections, None)
        # Now fill durations from cache
        for key in list(index_ann.keys()):
            # find any mode for duration
            dur = None
            for m in ("audio", "sensor", "multimodal"):
                d = base_cache.get((key[0], key[1], m))
                if d is not None:
                    dur = float(d["duration"])
                    break
            if dur is None:
                index_ann[key] = (index_ann[key][0], 0.0)
            else:
                index_ann[key] = (index_ann[key][0], dur)

        # Actual grid search
        best_by_mode = {}
        for mode in ("audio", "sensor", "multimodal"):
            if not any(k[2] == mode for k in base_cache.keys()):
                continue
            best_score = -1.0
            best_params_mode = None
            for p in _iter_param_sets(params_grid):
                vals: List[float] = []
                for (user, audio_fn, m), data in base_cache.items():
                    if m != mode:
                        continue
                    duration = float(data["duration"]) if data.get("duration") is not None else 0.0
                    series = data["series"].get(p.severity_type, np.array([], dtype=np.float32))
                    if series.size == 0:
                        continue
                    sev = _apply_params_to_series(series, duration, p)
                    segs, dur2 = index_ann.get((user, audio_fn), ([], duration))
                    duration_use = float(dur2) if dur2 is not None else duration
                    ann = _mask_from_segments(segs, duration_use, len(sev), margin_sec=p.ann_margin_sec)
                    met = _aggregate_metrics(ann.astype(int), sev.astype(float))
                    vals.append(float(met.get(primary_metric, float("nan"))))
                if vals:
                    score = float(np.nanmean(vals))
                    if score > best_score:
                        best_score = score
                        best_params_mode = p
            if best_params_mode is not None:
                best_by_mode[mode] = best_params_mode

    # With best params decided, compute final rows, overlays, and summary
    rows = []
    per_user_summary = {}
    for (user, audio_fn, mode), data in base_cache.items():
        if mode not in best_by_mode:
            continue
        p = best_by_mode[mode]
        duration = float(data["duration"]) if data.get("duration") is not None else 0.0
        series = data["series"][p.severity_type]
        sev = _apply_params_to_series(series, duration, p)
        # annotation
        segs, dur2 = index_ann.get((user, audio_fn), ([], duration))
        duration_use = float(dur2) if dur2 is not None else duration
        ann = _mask_from_segments(segs, duration_use, len(sev), margin_sec=p.ann_margin_sec)
        metrics = _aggregate_metrics(ann.astype(int), sev.astype(float))
        rows.append({"user": user, "audio": audio_fn, "mode": mode, **metrics})
        us = per_user_summary.setdefault(user, {})
        for k, v in metrics.items():
            us.setdefault(f"{mode}.{k}", []).append(v)
        # figure
        fig_name = f"overlay_{mode}_{user}_{Path(audio_fn).stem}.png"
        _plot_overlay(out_dir / "figures" / fig_name, sev, ann, title=f"{mode} | {user} | {audio_fn}")

    if not rows:
        LOGGER.warning("No metrics computed; please check inputs and checkpoints")
        return

    # Save per-audio metrics
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_audio_metrics.csv", index=False)

    # Aggregate per-user and overall summary
    user_summary: Dict[str, Dict[str, float]] = {}
    for user, md in per_user_summary.items():
        user_summary[user] = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in md.items()}

    overall: Dict[str, float] = {}
    for mode in ("audio", "sensor", "multimodal"):
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        for k in ("roc_auc", "ap", "best_f1", "best_iou", "pearson"):
            overall[f"{mode}.{k}"] = float(np.nanmean(sub[k])) if k in sub.columns else float("nan")

    # Save per-audio metrics
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_audio_metrics.csv", index=False)

    # Aggregate per-user and overall summary
    user_summary: Dict[str, Dict[str, float]] = {}
    for user, md in per_user_summary.items():
        user_summary[user] = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in md.items()}

    overall: Dict[str, float] = {}
    for mode in ("audio", "sensor", "multimodal"):
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        for k in ("roc_auc", "ap", "best_f1", "best_iou", "pearson"):
            overall[f"{mode}.{k}"] = float(np.nanmean(sub[k])) if k in sub.columns else float("nan")

    # Summarize chosen params
    best_params_dump = {m: vars(p) for m, p in best_by_mode.items()}
    summary_payload = {
        "experiment": str(experiment_dir),
        "n_rows": int(len(df)),
        "users": sorted(list(user_summary.keys())),
        "overall": overall,
        "per_user": user_summary,
        "primary_metric": primary_metric,
        "best_params": best_params_dump,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))

    # Optional: consensus across evaluators (mean/vote) per audio
    if consensus and consensus.lower() != "none":
        # Build items by audio
        by_audio: Dict[str, List[EvalItem]] = {}
        for it in items:
            by_audio.setdefault(it.audio_filename, []).append(it)

        # Helper to get duration and a representative severity length per (audio, mode)
        def _get_audio_mode_cache(audio_fn: str, mode: str) -> Optional[Dict[str, Any]]:
            for (user, afn, m), data in base_cache.items():
                if afn == audio_fn and m == mode:
                    return data
            return None

        thr = float(consensus_thr)
        rows_c: List[Dict[str, Any]] = []
        for mode, p in best_by_mode.items():
            for audio_fn, its in by_audio.items():
                data = _get_audio_mode_cache(audio_fn, mode)
                if data is None:
                    continue
                duration = float(data.get("duration") or 0.0)
                series = data["series"][p.severity_type]
                sev = _apply_params_to_series(series, duration, p)
                L = len(sev)
                # Build per-evaluator masks aligned to sev
                masks: List[np.ndarray] = []
                for it in its:
                    segs = it.problem_sections
                    m = _mask_from_segments(segs, duration, L, margin_sec=p.ann_margin_sec)
                    masks.append(m.astype(np.float32))
                if len(masks) < int(min_evaluators):
                    continue
                M = np.stack(masks, axis=0)  # [K, L]
                prop = M.mean(axis=0)  # proportion of evaluators who marked positive
                if consensus.lower() in ("mean", "vote"):
                    ann_bin = (prop >= thr).astype(np.uint8)
                else:
                    ann_bin = (prop >= thr).astype(np.uint8)
                metrics = _aggregate_metrics(ann_bin.astype(int), sev.astype(float))
                rows_c.append({"user": "CONSENSUS", "audio": audio_fn, "mode": mode, **metrics})

        if rows_c:
            dfc = pd.DataFrame(rows_c)
            dfc.to_csv(out_dir / "per_audio_metrics_consensus.csv", index=False)
            overall_c: Dict[str, float] = {}
            for mode in ("audio", "sensor", "multimodal"):
                sub = dfc[dfc["mode"] == mode]
                if len(sub) == 0:
                    continue
                for k in ("roc_auc", "ap", "best_f1", "best_iou", "pearson"):
                    overall_c[f"{mode}.{k}"] = float(np.nanmean(sub[k])) if k in sub.columns else float("nan")
            payload_c = {
                "consensus": consensus,
                "threshold": thr,
                "min_evaluators": int(min_evaluators),
                "overall": overall_c,
            }
            (out_dir / "summary_consensus.json").write_text(json.dumps(payload_c, indent=2, ensure_ascii=False))

            # Save overlay figures for consensus masks per (audio, mode)
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for mode, p in best_by_mode.items():
                for audio_fn, its in by_audio.items():
                    data = _get_audio_mode_cache(audio_fn, mode)
                    if data is None:
                        continue
                    duration = float(data.get("duration") or 0.0)
                    series = data["series"][p.severity_type]
                    sev = _apply_params_to_series(series, duration, p)
                    L = len(sev)
                    masks: List[np.ndarray] = []
                    for it in its:
                        segs = it.problem_sections
                        m = _mask_from_segments(segs, duration, L, margin_sec=p.ann_margin_sec)
                        masks.append(m.astype(np.float32))
                    if len(masks) < int(min_evaluators):
                        continue
                    M = np.stack(masks, axis=0)
                    prop = M.mean(axis=0)
                    ann_bin = (prop >= thr).astype(np.uint8)
                    fig_name = f"overlay_consensus_{mode}_{Path(audio_fn).stem}.png"
                    _plot_overlay(fig_dir / fig_name, sev, ann_bin, title=f"CONSENSUS | {mode} | {audio_fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Align expert annotations with model-indicated problematic segments")
    ap.add_argument("--experiment", type=str, required=True, help="Path to experiment dir (with models/)")
    ap.add_argument("--web-evals", type=str, default="../piano-performance-marker/web_evaluations", help="Path to web_evaluations dir")
    ap.add_argument("--audio-dir", type=str, default="../piano-performance-marker/public/audio", help="Path to public audio dir")
    ap.add_argument("--manifest", type=str, default="../piano-performance-marker/public/audio/manifest_top20.json", help="Path to manifest_top20.json")
    ap.add_argument("--data-root", type=str, default="/home/kazuki/Projects/Profy/data", help="Profy data root for normalization stats and sensors")
    ap.add_argument("--out", type=str, default="", help="Output directory under results/. If empty, create timestamped dir.")
    ap.add_argument("--device", type=str, default=None, help="Torch device override (e.g., cuda, cpu)")
    # Tuning and parameters
    ap.add_argument("--tune", action="store_true", help="Grid search parameters to maximize primary metric per mode")
    ap.add_argument("--primary-metric", type=str, default="ap", choices=["ap", "roc_auc", "best_f1", "best_iou", "pearson"], help="Objective for tuning")
    ap.add_argument("--severity-type", type=str, default="product", help="Comma-separated list or single: product,evidence,attention")
    ap.add_argument("--severity-power", type=str, default="1.0", help="Comma-separated list of gamma values (e.g., 1.0,1.5,2.0)")
    ap.add_argument("--smooth-win", type=str, default="1", help="Comma-separated list of odd window sizes (e.g., 1,5,9)")
    ap.add_argument("--norm-minmax", type=str, default="0", help="Comma-separated list of 0/1")
    ap.add_argument("--lag-sec", type=str, default="0.0", help="Comma-separated list of global lags (sec), e.g., -0.2,0.0,0.2")
    ap.add_argument("--ann-margin-sec", type=str, default="0.0", help="Comma-separated list of margins (sec) to expand annotations at both ends")
    ap.add_argument("--consensus", type=str, default="none", choices=["none", "mean", "vote"], help="Consensus method across evaluators per audio")
    ap.add_argument("--consensus-thr", type=float, default=0.5, help="Threshold for consensus (proportion >= thr is positive)")
    ap.add_argument("--min-evaluators", type=int, default=1, help="Minimum evaluators required for consensus per audio")
    args = ap.parse_args()

    exp_dir = Path(args.experiment).resolve()
    web_dir = Path(args.web_evals).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()

    results_root = Path("results").resolve()
    results_root.mkdir(exist_ok=True)
    if args.out:
        out_dir = results_root / args.out
    else:
        from datetime import datetime
        out_dir = results_root / f"expert_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Experiment: %s", exp_dir)
    LOGGER.info("Web evals: %s", web_dir)
    LOGGER.info("Audio dir: %s", audio_dir)
    LOGGER.info("Manifest: %s", manifest_path)
    LOGGER.info("Out dir: %s", out_dir)

    # Build parameter grid from CLI
    def _parse_csv_floats(s: str) -> List[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_csv_ints(s: str) -> List[int]:
        return [int(float(x.strip())) for x in s.split(",") if x.strip()]

    def _parse_csv_bools(s: str) -> List[bool]:
        out: List[bool] = []
        for x in s.split(","):
            x = x.strip().lower()
            if not x:
                continue
            out.append(x in ("1", "true", "yes", "y"))
        return out or [False]

    params_grid = {
        "severity_type": [t.strip() for t in args.severity_type.split(",") if t.strip()],
        "severity_power": _parse_csv_floats(args.severity_power),
        "smooth_win": _parse_csv_ints(args.smooth_win),
        "norm_minmax": _parse_csv_bools(args.norm_minmax),
        "lag_sec": _parse_csv_floats(args.lag_sec),
        "ann_margin_sec": _parse_csv_floats(args.ann_margin_sec),
    }

    run_alignment(
        experiment_dir=exp_dir,
        web_eval_dir=web_dir,
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        data_root=data_root,
        out_dir=out_dir,
        device=args.device,
        tune=args.tune,
        primary_metric=args.primary_metric,
        params_grid=params_grid,
        consensus=args.consensus,
        consensus_thr=args.consensus_thr,
        min_evaluators=args.min_evaluators,
    )


if __name__ == "__main__":
    main()
