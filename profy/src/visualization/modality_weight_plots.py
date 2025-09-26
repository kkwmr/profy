#!/usr/bin/env python3
"""Utility plots for modality-weight diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

__all__ = [
    'plot_modality_weight_distribution',
    'plot_directory_distributions',
]


def plot_modality_weight_distribution(
    csv_path: str | Path,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Path:
    """Plot sensor/audio modality weight histograms for a single fold CSV."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if not {'sensor_weight', 'audio_weight'}.issubset(df.columns):
        raise ValueError('CSV must contain sensor_weight and audio_weight columns')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle(title or csv_path.stem)

    axes[0].hist(df['sensor_weight'], bins=20, color='#4E79A7', alpha=0.8)
    axes[0].set_title('Sensor weight distribution')
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Count')
    axes[0].grid(alpha=0.2)

    axes[1].hist(df['audio_weight'], bins=20, color='#F28E2B', alpha=0.8)
    axes[1].set_title('Audio weight distribution')
    axes[1].set_xlabel('Weight')
    axes[1].grid(alpha=0.2)

    for ax in axes:
        ax.set_xlim(0.0, 1.0)

    save_path = Path(save_path) if save_path else csv_path.with_suffix('.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    if show:
        plt.show()

    return save_path


def plot_directory_distributions(
    directory: str | Path,
    output_dir: Optional[str | Path] = None,
    glob_pattern: str = '*_modality_weights.csv',
) -> Iterable[Path]:
    """Plot all modality-weight CSVs under a directory."""
    directory = Path(directory)
    output_dir = Path(output_dir) if output_dir else directory
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for csv_file in sorted(directory.glob(glob_pattern)):
        figure_path = output_dir / f'{csv_file.stem}.png'
        fig_title = figure_path.stem.replace('_', ' ').title()
        generated.append(plot_modality_weight_distribution(csv_file, figure_path, title=fig_title))
    return generated
