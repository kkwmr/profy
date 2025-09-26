# Audio & Multimodal Performance Improvement Plan

> **Last updated:** 2025-09-26 23:41  


## Strategic Goals
- [ ] Recover audio-only F1 to ≥0.60 across all folds.
- [ ] Restore multimodal F1 to surpass sensor-only baseline by ≥0.05 absolute.
- [ ] Establish repeatable diagnostics visualizing modality weights, attention, and audio quality.

## Phase 0 — Immediate Visibility
- [x] Instrument training loop to persist `modality_weights` and fold-wise confusion matrices (CSV/JSON).
- [x] Add plotting script for modality-weight distributions per fold in `profy/src/visualization`.
- [x] Append run metadata (config hash, git HEAD, audio stats) to `training.log`.

## Phase 1 — Audio Feature Integrity
- [x] Expand `audio_cnn2d` input to consume all 128 feature channels (update reshape logic in `UnifiedAttentionModel`).
- [x] Replace per-sample z-score with global normalization derived from full training set statistics (persist stats artifact).
- [x] Introduce silence-aware masking instead of zero-padding for short clips to avoid attention dilution.
- [x] Add regression test ensuring audio feature tensor retains ≥120 non-zero frames post-preprocessing.

## Phase 2 — Fusion Reliability
- [x] Reduce or disable `modality_dropout_p` until audio branch improves; expose value via config.
- [x] Enforce minimum sensor weighting in fusion gate (e.g., clamp sensor-related weights ≥0.35).
- [x] Incorporate audio-quality vector into gating decision with temperature scaling tied to SNR proxy.
- [x] Log gate outputs alongside quality metrics for 20% of batches to validate trust calibration.

## Phase 3 — Audio Encoder Enhancements
- [x] Swap linear interpolation with parametric up/downsampling (e.g., Conv1d with learnable kernels).
- [x] Experiment with lightweight Temporal Convolutional Network stacked after CNN for better rhythmic modeling.
- [x] Benchmark SpecAugment-style augmentation focusing on time masking for pros/cons balance.
- [x] Add ablation script comparing current encoder vs. TCN variant on single fold.

## Phase 4 — Evaluation & Guardrails
- [x] Extend `complete_results.json` generation with precision/recall and per-class F1.
- [x] Track fold-level variance and trigger warning when std(F1) > 0.05.
- [x] Create dashboard notebook in `profy/docs` to summarize latest run metrics and gate stats.
- [x] Document SOP for rerunning experiment with cached audio features (speeds up iteration).

## Phase 5 — Deployment Readiness
- [x] Update experiment script to accept external config profiles (yaml) for reproducibility.
- [x] Version saved checkpoints with semantic tags (`sensor-v1`, `mm-audiofix`).
- [x] Draft release notes capturing deltas vs. experiment_20250925_105205 outcomes.
- [x] Schedule smoke test that replays best-performing sensor-only fold with multimodal model.

## Dependencies & Sequencing Notes
- [x] Phase 0 tasks unblock accurate tracking for later phases.
- [x] Phase 1 must complete before enabling aggressive fusion changes in Phase 2.
- [x] Phase 3 experiments depend on finalized normalization pipeline (Phase 1).
- [x] Phase 4 reporting expects telemetry from Phases 0–2.

## Acceptance Criteria Checklist
- [ ] Multimodal run (3-fold CV) reports ≥0.78 F1 with std ≤0.02.
- [ ] Audio-only run reports ≥0.62 F1 with std ≤0.04.
- [ ] Latest `training.log` includes modality weight summary and normalization config reference.
- [ ] Visualization artifacts show sensor weight dominance ≥50% for ≥80% of validation samples.
