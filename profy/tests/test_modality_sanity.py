from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg', force=True)

from src.training.run_experiments import (
    _validate_sensor_vs_audio,
    save_modality_summary_and_barplot,
)


def _dummy_results(sensor_f1: float, audio_f1: float, mm_f1: float = 0.75) -> dict:
    return {
        'sensor': {'mean_f1': sensor_f1, 'std_f1': 0.01, 'mean_accuracy': 0.7, 'std_accuracy': 0.02},
        'audio': {'mean_f1': audio_f1, 'std_f1': 0.02, 'mean_accuracy': 0.65, 'std_accuracy': 0.03},
        'multimodal': {'mean_f1': mm_f1, 'std_f1': 0.03, 'mean_accuracy': 0.76, 'std_accuracy': 0.04},
    }


def test_save_modality_summary_returns_dataframe(tmp_path: Path) -> None:
    results = _dummy_results(0.70, 0.60)
    df = save_modality_summary_and_barplot(results, tmp_path)
    assert df is not None
    csv_path = tmp_path / 'diagnostics' / 'modality_summary.csv'
    assert csv_path.exists()
    stored = pd.read_csv(csv_path)
    assert pytest.approx(stored.loc[stored['mode'] == 'sensor', 'mean_f1'].item(), rel=1e-6) == 0.70


def test_validate_sensor_vs_audio_passes() -> None:
    results = _dummy_results(0.70, 0.60)
    _validate_sensor_vs_audio(results)


def test_validate_sensor_vs_audio_raises_when_audio_better() -> None:
    results = _dummy_results(0.58, 0.61)
    with pytest.raises(ValueError):
        _validate_sensor_vs_audio(results)


def test_validate_sensor_vs_audio_allows_when_flag_set() -> None:
    results = _dummy_results(0.58, 0.61)
    _validate_sensor_vs_audio(results, allow_audio_dominant=True)
