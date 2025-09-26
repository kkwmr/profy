# Rebuttal Pre-mortem: Anticipated Reviewer Concerns

## R1: "User studies are insufficient (n=12)"
**Response**: Our primary validation relies on three-layer objective evaluation designed specifically to avoid user study limitations. The minimal user study (n=12) serves only as practical validation, not primary evidence. Objective metrics provide reproducible, unbiased evidence: synthetic intervention recovery (F1=0.89), simulated performance improvement (71% error reduction), and external judge validation (+2.3 skill score). This multi-layer approach is more robust than typical single-condition user studies.

## R2: "Transcription errors undermine prescription reliability"
**Response**: We address this through multi-factor confidence scoring (C_note = P_trans × Q_align × S_context × H_harmony) and uncertainty visualization. Low-confidence regions (15% of segments) are flagged with reduced opacity in the UI, allowing users to prioritize reliable prescriptions. Appendix includes separate analysis excluding uncertain segments, showing minimal performance degradation (F1=0.91 vs 0.92).

## R3: "Over-correction risks destroying musical expression"
**Response**: We implement prescription magnitude bounds (note errors ≤3 semitones, rhythm adjustments ≤16% beat, dynamics ≤20 velocity points) and L2 regularization in target curve fitting (λ=0.1). The system distinguishes intentional rubato through phrase-level tempo modeling. UI allows prescription strength adjustment and selective application. External judge validation confirms preserved musicality (+2.3 score improvement).

## R4: "Limited to classical piano - not generalizable"
**Response**: While evaluation focuses on classical piano for controlled validation, the prescription generation framework generalizes to other instruments and genres. The three-mechanism approach (note errors, rhythm deviations, dynamics matching) applies broadly. Genre-specific adaptation requires only target curve retraining, not architectural changes. Future work section addresses ensemble and multi-instrument extensions.

## R5: "Comparison with human teachers missing"
**Response**: Human teacher comparison faces practical challenges (availability, consistency, cost), but our external judge model serves as proxy for expert assessment. The +2.3 skill score improvement on validated scales indicates clinically meaningful improvement. Synthetic intervention recovery provides ground truth accuracy impossible with human evaluation alone.

## R6: "Figures are difficult to interpret"
**Response**: All figures include comprehensive captions with setup, metrics, and interpretation. Figure 1 (teaser) demonstrates complete workflow from input to prescription to validation. Before/after comparisons use consistent visual encoding. UI screenshots include explanatory overlays and confidence indicators.

## R7: "Evaluation methodology is unusual for CHI"
**Response**: Three-layer objective evaluation introduces a novel validation paradigm particularly valuable for skill acquisition systems. This approach offers advantages over traditional user studies: eliminated confounds, reproducible results, scalable evaluation, and stronger statistical power. The method has broader implications for HCI research in domains with objective performance measures.

## R8: "Missing related work in music education technology"
**Response**: Section 2 comprehensively covers commercial systems (SmartMusic, Yousician), academic approaches (CNNs, RNNs, transformers), and expression modeling work. Table comparison matrix identifies the prescription gap systematically. Recent work includes MT3, VirtuosoNet, and PianoBART. Gap analysis demonstrates that no existing system provides quantified corrective actions.