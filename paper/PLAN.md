Profy paper sync plan — Align “Profy: Methodology and Architecture” with latest code

Source of truth for implementation inspected
- Runner: /home/kazuki/Projects/Profy/profy/scripts/run.sh (invokes profy.src.training.run_experiments)
- Model: /home/kazuki/Projects/Profy/profy/src/models/unified_attention_model.py
- Data/Features: /home/kazuki/Projects/Profy/profy/src/data/real_data_loader.py
- Overlays (severity rendering): /home/kazuki/Projects/Profy/profy/src/evaluation/sensor_attention_overlays.py

Only list of required edits (no text changes yet)

1) Terminology consistency
- main.tex:294 — Replace “(amateur/professional)” with “(expert/amateur)”.

2) Preprocessing and leakage controls (sensor/audio)
- main.tex:321 — Sensor pipeline description is incorrect for current code.
  - Paper claims per‑key winsorization (1–99 percentile), z‑normalization on train‑only stats, and [0,1] scaling. Implementation: reads raw key positions, uniformly resamples/pads to T=1000; no winsorization or per‑fold normalization is applied.
  - Action: Rewrite sensor preprocessing to: “uniform resampling/padding to T=1000; no per‑fold normalization in current pipeline.” Optionally note future intent if needed.
- main.tex:321–333 — Audio normalization/leakage control is overstated.
  - Paper: “train‑only non‑silent frames per fold; cached; applied to val/test.”
  - Code: computes global mean/std over non‑silent frames across the entire loaded set and persists to profy/data/audio_feature_stats.json (not per‑fold; potential leakage). Hop is chosen to yield exactly T frames (no fixed 46 ms window).
  - Action: Replace with: “global non‑silent frame normalization persisted to audio_feature_stats.json; hop chosen to produce T frames; mask threshold m_t = 1[RMS_t > 0.3·mean], NSR = share with RMS_t > 0.5·mean.” Remove the fixed 46 ms mention.

3) Model architecture — temporal heads and parameter count
- main.tex:354–366 (Temporal Modeling and Dual Heads) — Mismatch.
  - Paper: per‑frame logits r_t + MIL aggregator drives final decision; attention aligned to σ(r_t); ~0.44M params; <50 ms on RTX 3090.
  - Code: BiLSTM produces per‑frame evidence e_t; clip‑level logits are computed from attention‑pooled context (no per‑frame r_t path). MIL is used as an auxiliary loss on e_t (complement‑of‑product) vs label, not as the final aggregator. Parameter count ≈ 4.16M (measured). Latency claims are not measured in code.
  - Actions:
    - Remove per‑frame logits r_t narrative; state clip‑level classifier on attention‑pooled context + per‑frame evidence head.
    - Remove “align attention to σ(r_t)” claim.
    - Update parameter count to ~4.16M (or soften to “~4.2M”).
    - Soften/remove specific latency numbers unless verified.

4) Gating temperature and floors
- main.tex:348–356 and 339–349 — Temperature formula and floors differ.
  - Paper: T0 − β·NSR, with explicit (T0, β, Tmin, Tmax) defaults; constant sensor floor α_min.
  - Code: temperature = 1 + γ·(1 − NSR) with γ=2.0; adaptive sensor‑share floor interpolates from max_sensor_weight=0.6 (low NSR) to min_sensor_weight=0.35 (high NSR).
  - Action: Replace with: “temperature 1 + γ(1−NSR), γ=2.0; adaptive floor on sensor share ∈ [0.35, 0.6] based on NSR.” Remove T0/β/Tmin/Tmax.

5) Weak supervision and objective
- main.tex:346–366 — Loss terms not matching implementation.
  - Paper lists BCE + (−entropy on e_t) + total‑variation on e_t + alignment σ(r_t)↔e_t + KL(att ↔ normalized σ(r_t)), with LSE pooling controlled by τ.
  - Code uses: CrossEntropyLoss on clip‑level logits + MIL BCE on aggregated evidence p(any)=1−∏(1−e_t) (λ_mil, default 0.5) + L1 sparsity on e_t (λ_evidence_l1, default 0.001) + entropy regularization on modality weights via exp(−H) (λ≈0.1). No TV, no σ(r_t)↔e_t alignment, no KL(att↔logits), no LSE pooling τ.
  - Action: Replace loss description with the above; remove LSE/τ and the three unused regularizers.

6) Severity and UI mapping
- main.tex:368–370 (From Inference to UI) — Severity formula and calibration differ.
  - Paper: s_t = σ(α_cal·r_t) · e_t with Platt scaling; note/beat overlays with top‑k selection, Otsu+hysteresis; beat fallback.
  - Code (overlays): severity = attention · evidence; optional min–max norm, exponent (default 1.5), smoothing window (default 9). No Platt scaling; current tooling renders timeline overlays; score‑synchronization via DTW/HMM is not implemented in the provided script.
  - Actions:
    - Replace severity with product(attention, evidence); mention optional smoothing and normalization as used in overlays.
    - Soften “score‑synchronized overlays” to “timeline overlays by default; score alignment is optional/future when transcripts are available.” Remove Otsu/hysteresis specifics unless we implement them.

7) Fusion policy (decision‑level)
- main.tex:442–451 — Keep, but clarify implementation.
  - Code: decision‑level fusion as product‑of‑experts on log‑odds with per‑fold grid search over weights (α, β); preferred multimodal result is this decision fusion.
  - Action: Add brief note: “PoE over log‑odds with small grid search on training folds; reported as multimodal.”

8) Inputs and outputs table (notation)
- main.tex:309–314 — Remove r_t entry and any symbols tied to per‑frame logits; keep e_t, α_t, s_t; define s_t as attention·evidence.

9) Optional regularizations/augments not mentioned
- Add short note (Architecture or Training setup): modality dropout (default 0.0), fusion warm‑up epochs (default 2 before enabling gating), SpecAugment time‑mask (default 0), Mixup (default 0). These exist in code but are disabled by default.

10) Minor consistency fixes
- main.tex:292–295 — Where “score following φ” is mentioned, qualify it as “optional; not used in current experiments unless transcripts are provided.”
- main.tex:482 — Remove remaining \emph usages to respect Related Work constraint (already enforced elsewhere, but re‑check this section after edits).

Validation after edits
- Rebuild paper with ./compile.sh and verify:
  - No claims remain that contradict: feature set (128‑dim), mask thresholds (0.3/0.5), gating (γ=2.0, floor [0.35,0.6]), loss terms (CE + MIL + L1 + entropy reg), severity (att·evid), parameter count (~4.2M), decision‑fusion (PoE).
  - Terminology is “expert/amateur” throughout.
