#!/usr/bin/env python3
"""
Generate sensor time-series overlays with model attention and human annotations.

Outputs one image per (user, audio, mode) under:
  results/<out_dir>/figures/sensor_overlays/

Design goals
- Row 1: Sensor heatmap (88 keys x time [sec]) with annotation spans
- Row 2: Attention / Evidence / Severity lines with annotation spans
- Clear legends and consistent colors

Usage example:
  python -m src.evaluation.sensor_attention_overlays \
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
import torch
import matplotlib.pyplot as plt

from ..models.unified_attention_model import UnifiedAttentionModel

LOGGER = logging.getLogger(__name__)


@dataclass
class EvalItem:
    username: str
    audio_filename: str
    problem_sections: List[Tuple[float, float]]  # seconds


@dataclass
class EvalParams:
    severity_type: str = "product"  # product, evidence, attention
    severity_power: float = 1.5
    smooth_win: int = 9
    norm_minmax: bool = True
    lag_sec: float = 0.0


def _load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    return {str(e.get("filename")): e for e in payload if e.get("filename")}


def _iter_web_evals(base_dir: Path) -> Iterable[Tuple[str, Path]]:
    for user_dir in sorted(base_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        username = user_dir.name
        if username.lower().startswith("test"):
            continue
        for json_path in sorted(user_dir.glob("*.json")):
            yield username, json_path


essential_keys = ["attention", "evidence", "product"]


def _series_from_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    att_t = outputs.get("attention_weights")
    evid_t = outputs.get("evidence_scores")
    if att_t is None or evid_t is None:
        return {k: np.array([], dtype=np.float32) for k in essential_keys}
    att = att_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    evid = evid_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    prod = (att * evid).astype(np.float32)
    return {"attention": att, "evidence": evid, "product": prod}


def _apply_params(series: np.ndarray, duration_sec: float, p: EvalParams) -> np.ndarray:
    x = series.astype(np.float32)
    x = np.clip(x, 0.0, None) ** float(p.severity_power)
    win = max(1, int(p.smooth_win))
    if win % 2 == 0:
        win += 1
    if win > 1:
        k = np.ones((win,), dtype=np.float32) / float(win)
        x = np.convolve(x, k, mode="same")
    if p.norm_minmax:
        mn, mx = float(x.min()), float(x.max())
        x = (x - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(x)
    # lag not applied here; we align via x-axis
    return x


def _plot_sensor_overlay(
    out_path: Path,
    sensor_seq: Optional[np.ndarray],
    duration_sec: float,
    series_raw: Dict[str, np.ndarray],
    p: EvalParams,
    segments: List[Tuple[float, float]],
    title: str,
) -> None:
    L = len(next(iter(series_raw.values()))) if series_raw else 0
    if duration_sec <= 0.0:
        duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, num=max(1, L))

    sf = float(max(1, L))
    att = _apply_params(series_raw.get("attention", np.array([])) * sf, duration_sec, p)
    evid = _apply_params(series_raw.get("evidence", np.array([])), duration_sec, p)
    base_att = series_raw.get("attention", np.array([]))
    base_evid = series_raw.get("evidence", np.array([]))
    sev = _apply_params((base_att * sf) * base_evid if base_att.size and base_evid.size else np.array([]), duration_sec, p)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})

    # Row 1: Sensor heatmap
    if sensor_seq is not None and sensor_seq.size:
        S = sensor_seq.astype(np.float32)
        vmax = float(np.percentile(S, 99.0)) if np.isfinite(S).all() else None
        extent = [0.0, duration_sec, 0, S.shape[1]]
        axes[0].imshow(S.T, aspect='auto', origin='lower', extent=extent, cmap='magma', vmin=0.0, vmax=vmax)
        axes[0].set_ylabel('Keys (88)')
    else:
        axes[0].text(0.5, 0.5, 'Sensor data unavailable', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_ylabel('Keys')

    for (s, e) in segments:
        axes[0].axvspan(max(0.0, s), max(0.0, e), color='#ffcc00', alpha=0.25, lw=0)

    # Row 2: Lines
    if att.size:
        axes[1].plot(t, att, color='#1f77b4', lw=1.8, label='attention')
    if evid.size:
        axes[1].plot(t, evid, color='#ff7f0e', lw=1.2, alpha=0.85, label='evidence')
    if sev.size:
        axes[1].plot(t, sev, color='#2ca02c', lw=1.8, label='severity (att*evid)')
    for (s, e) in segments:
        axes[1].axvspan(max(0.0, s), max(0.0, e), color='#ffcc00', alpha=0.25, lw=0)
    axes[1].set_xlabel('Time (sec)')
    axes[1].set_ylabel('Score')
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc='upper right')

    axes[0].set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _extract_audio_features(audio_path: Path, sequence_length: int = 1000) -> Tuple[np.ndarray, np.ndarray, float]:
    import librosa
    import librosa.util as librosa_util

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    rms_ref = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms_ref > 1e-6:
        y = y * float(0.1 / rms_ref)
    hop_length = max(1, len(y) // (sequence_length + 1))
    n_fft = min(2048, len(y))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=n_fft, hop_length=hop_length, fmin=30.0, fmax=8000.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=4)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    mel_mag = np.sqrt(np.maximum(S, 1e-10))
    mel_diff = np.diff(mel_mag, axis=1)
    if mel_diff.size == 0:
        spectral_flux = np.zeros((1, mel_mag.shape[1]), dtype=np.float64)
    else:
        flux = np.sqrt(np.square(mel_diff).sum(axis=0, dtype=np.float64))
        spectral_flux = np.concatenate([[flux[0]], flux]).reshape(1, -1)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    rms_delta = librosa.feature.delta(rms)

    frame_count = S_db.shape[1]
    align_list = [mfccs, chroma, tonnetz, spec_contrast, spec_centroid, spec_bandwidth, spec_rolloff, spectral_flatness, spectral_flux, zcr, rms, rms_delta]
    for i, arr in enumerate(align_list):
        if arr.shape[1] != frame_count:
            align_list[i] = librosa_util.fix_length(arr, frame_count, axis=1)
    mfccs, chroma, tonnetz, spec_contrast, spec_centroid, spec_bandwidth, spec_rolloff, spectral_flatness, spectral_flux, zcr, rms_feat, rms_delta_feat = align_list

    chroma_var = chroma.var(axis=1)
    drop_idx = np.argsort(chroma_var)[:3]
    chroma_reduced = np.delete(chroma, drop_idx, axis=0)

    feature_stack = [S_db, mfccs, chroma_reduced, tonnetz, spec_contrast, spec_centroid, spec_bandwidth, spec_rolloff, spectral_flatness, spectral_flux, zcr, rms_feat, rms_delta_feat]
    feats = np.vstack(feature_stack).T.astype(np.float32)
    if feats.shape[1] != 128:
        feats = librosa_util.fix_length(feats.T, 128, axis=0).T.astype(np.float32)

    raw_length = feats.shape[0]
    if raw_length > sequence_length:
        feats = feats[:sequence_length]
    elif raw_length < sequence_length:
        feats = np.vstack([feats, np.zeros((sequence_length - raw_length, 128), dtype=np.float32)])

    mask = (rms_feat.flatten() > (float(rms_feat.mean()) * 0.3)).astype(np.float32)
    if mask.size > sequence_length:
        mask = mask[:sequence_length]
    elif mask.size < sequence_length:
        mask = np.pad(mask, (0, sequence_length - mask.size), constant_values=0.0)

    return feats, mask, duration


def _load_sensor_sequence(hackkey_csv: Path, sequence_length: int = 1000) -> Optional[np.ndarray]:
    import pandas as pd
    if not hackkey_csv.exists():
        return None
    try:
        df = pd.read_csv(hackkey_csv, header=None)
    except Exception:
        return None
    data = df.iloc[:, 1:89].values.astype(np.float32)
    if data.shape[1] != 88:
        return None
    if data.shape[0] > sequence_length:
        idx = np.linspace(0, data.shape[0] - 1, sequence_length, dtype=int)
        data = data[idx]
    elif data.shape[0] < sequence_length:
        data = np.vstack([data, np.zeros((sequence_length - data.shape[0], 88), dtype=np.float32)])
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--web-evals", default="../piano-performance-marker/web_evaluations")
    ap.add_argument("--audio-dir", default="../piano-performance-marker/public/audio")
    ap.add_argument("--manifest", default="../piano-performance-marker/public/audio/manifest_top20.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    exp_dir = Path(args.experiment).resolve()
    web_dir = Path(args.web_evals).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    manifest_path = Path(args.manifest).resolve()

    # Output dir
    results_root = Path("results").resolve()
    if not args.out:
        from datetime import datetime
        out_dir = results_root / f"sensor_overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        out_dir = results_root / args.out
    fig_dir = out_dir / "figures" / "sensor_overlays"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Model kwargs
    model_kwargs = {}
    cfg = exp_dir / "config_manifest.json"
    if cfg.exists():
        try:
            payload = json.loads(cfg.read_text())
            model_kwargs = payload.get("model_kwargs", {}) or {}
        except Exception:
            pass

    # Checkpoints
    models_dir = exp_dir / "models"
    ckpts: Dict[str, List[Path]] = {"audio": [], "sensor": [], "multimodal": []}
    if models_dir.exists():
        for p in models_dir.glob("*.pth"):
            n = p.name.lower()
            if n.startswith("audio-"):
                ckpts["audio"].append(p)
            elif n.startswith("sensor-"):
                ckpts["sensor"].append(p)
            elif n.startswith("mm-") or n.startswith("multimodal-"):
                ckpts["multimodal"].append(p)
        for k in ckpts:
            ckpts[k] = sorted(ckpts[k])

    if not (ckpts["audio"] or ckpts["sensor"] or ckpts["multimodal"]):
        raise SystemExit(f"No checkpoints found under {models_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Manifest and evals
    manifest = _load_manifest(manifest_path)

    # Collect items and filter users with exactly 20 top-20 files
    items: List[EvalItem] = []
    items_by_user: Dict[str, List[EvalItem]] = {}
    for username, jp in _iter_web_evals(web_dir):
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        audio_fn = data.get("audio_filename") or data.get("audio_file")
        if not audio_fn or audio_fn not in manifest:
            continue
        segs = []
        for seg in (data.get("problem_sections") or []):
            try:
                s = float(seg.get("start", 0.0)); e = float(seg.get("end", 0.0))
                if e > s:
                    segs.append((s, e))
            except Exception:
                pass
        it = EvalItem(username=username, audio_filename=str(audio_fn), problem_sections=segs)
        items.append(it)
        items_by_user.setdefault(username, []).append(it)
    eligible_users = {u for u, lst in items_by_user.items() if len({x.audio_filename for x in lst}) == 20}
    items = [it for it in items if it.username in eligible_users]

    # Cache models
    model_cache: Dict[Tuple[str, Path], UnifiedAttentionModel] = {}

    # Iterate
    for it in items:
        entry = manifest.get(it.audio_filename, {})
        src_path = entry.get("source_path")
        sensor_seq: Optional[np.ndarray] = None
        duration_sec: float = 0.0

        # Prepare audio tensors
        feats, mask, duration_sec = _extract_audio_features(audio_dir / it.audio_filename)
        audio_t = torch.from_numpy(feats).unsqueeze(0).float().to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)

        # Load sensor
        if src_path:
            sp = Path(src_path)
            hackkey = sp.parent.parent / "hackkey" / "hackkey.csv"
            sensor_seq = _load_sensor_sequence(hackkey)

        for mode in ("audio", "sensor", "multimodal"):
            if not ckpts.get(mode):
                continue
            if mode != "audio" and sensor_seq is None:
                continue
            series_accum: List[Dict[str, np.ndarray]] = []
            for ck in ckpts[mode]:
                key = (mode, ck)
                if key not in model_cache:
                    m = UnifiedAttentionModel(**model_kwargs)
                    sd = torch.load(str(ck), map_location=device)
                    m.load_state_dict(sd)
                    m.eval(); m.to(device)
                    model_cache[key] = m
                model = model_cache[key]
                with torch.no_grad():
                    if mode == "audio":
                        out = model(sensor_data=None, audio_data=audio_t, audio_quality=None, audio_mask=mask_t)
                    elif mode == "sensor":
                        s_t = torch.from_numpy(sensor_seq).unsqueeze(0).float().to(device)
                        out = model(sensor_data=s_t, audio_data=None)
                    else:
                        s_t = torch.from_numpy(sensor_seq).unsqueeze(0).float().to(device)
                        out = model(sensor_data=s_t, audio_data=audio_t, audio_quality=None, audio_mask=mask_t)
                series_accum.append(_series_from_outputs(out))
            if not series_accum:
                continue
            # average series
            keys = essential_keys
            series_avg = {k: np.mean(np.stack([s[k] for s in series_accum], axis=0), axis=0) for k in keys}
            # plot
            title = f"{mode} | {it.username} | {it.audio_filename}"
            out_name = f"overlay_sensor_{mode}_{it.username}_{Path(it.audio_filename).stem}.png"
            _plot_sensor_overlay(fig_dir / out_name, sensor_seq, duration_sec, series_avg, EvalParams(), it.problem_sections, title)

    print(f"Saved overlays to: {fig_dir}")


if __name__ == "__main__":
    main()
