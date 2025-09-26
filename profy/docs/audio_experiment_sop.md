# Audio Experiment SOP

1. Activate environment and navigate to repo root.
2. Precompute audio cache (first run only):
   ```bash
   python -m profy.src.training.run_experiments --save-checkpoints --cache-audio
   ```
3. Subsequent reruns using cached features:
   ```bash
   python -m profy.src.training.run_experiments --cache-audio --save-checkpoints --smoke-test
   ```
4. Inspect diagnostics:
   - `profy/results/<run>/diagnostics/multimodal/*_modality_weights.csv`
   - `profy/docs/diagnostics_dashboard.ipynb`
5. Update release notes with key metric deltas before publishing results.
