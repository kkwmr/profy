# Release Notes — Audio/Multimodal Overhaul

## Highlights
- Global audio normalization with cached statistics to stabilise training.
- 128-dim audio encoder with temporal convolutional refinement and silence-aware masking.
- Learnable fusion gate enforcing minimum sensor dominance while tracking audio quality.
- Parametric resamplers replaced linear interpolation for cross-modal alignment.

## Expected Metric Improvements
- Audio-only F1 target: ≥0.62 with reduced fold variance.
- Multimodal F1 target: ≥0.78 with std ≤0.02.
- Additional precision/recall reporting in `complete_results.json`.

## Diagnostics
- Per-fold confusion matrices in `diagnostics/<mode>/fold_*_confusion_matrix.json`.
- Modality-weight CSVs plus histogram helper (`src/visualization/modality_weight_plots.py`).
- Smoke test replay of best sensor fold using multimodal checkpoint (`--smoke-test`).

## Migration Notes
- Run experiments with `python -m profy.src.training.run_experiments --save-checkpoints --cache-audio`.
- Visualise modality weights via `plot_directory_distributions` in the new visualization module.
- Dashboard notebook: `profy/docs/diagnostics_dashboard.ipynb`.
