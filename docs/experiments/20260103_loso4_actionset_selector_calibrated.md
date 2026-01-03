# 2026-01-03 — LOSO 4-class — Cross-action-set calibrated selector (CSP+LDA family)

Goal: build a **cross-action-set** selector (contextual bandit style) that can choose, per target subject (unlabeled),
among multiple alignment/model “actions” while keeping a **safe fallback to EA**.

This is motivated by the large **oracle union headroom** observed in `docs/experiments/20260102_loso4_actionset_expand_v1.md`.

## Quick note on the provided RL paper (AlphaDev)
We reviewed: `papers/Faster sorting algorithms discovered using deep reinforcement learning_20260103092244_2007261308147834880.md`.

Takeaway for our setting:
- AlphaDev is RL for **algorithm discovery** where reward (latency + correctness) can be evaluated in a controlled environment.
- In our LOSO target-subject setting, **true reward (target accuracy) is unavailable** at test time; therefore RL must reduce to
  **offline policy learning / contextual bandits** trained on source folds.
- Practically, this means our next step is **not “online RL on target”**, but improving the **certificate/selector** that predicts
  improvement from label-free signals.

## Protocol (strictly comparable)
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO (cross-subject)
- Classes: `left_hand,right_hand,feet,tongue` (4-class)
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5 s, resample 250 Hz
- Model backbone: CSP(`n_components=6`) + LDA

## What we implemented (action-set selector)
We upgraded `ea-stack-multi-safe-csp-lda` to support a broader action set:
- add **EA-FBCSP** as a candidate family inside the stacked selector
- fix a feature-mismatch issue by ensuring candidate meta (`cand_family`, and for chan: `rank/λ`) is present during calibration

This is intended to be an “offline contextual bandit” baseline: train ridge/guard on pseudo-target subjects (source-only),
then select per target subject using only label-free features, with fallback to EA if predicted improvement ≤ 0.

## Experiments and results

### A) Cross-family stacked selector (calibrated_ridge_guard, calAll)
Run:
- `outputs/20260103/4class/loso4_actionset_selector_stack_v2_calAll/20260103_results.txt`

Key numbers:
- EA mean acc: `0.5320`
- EA-STACK-MULTI-SAFE mean acc: `0.5220` (**-1.00% abs**), neg-transfer rate `0.111` (1 subject)

Failure mode (clear):
- One catastrophic selection: Subject 9 drops `0.6840 → 0.5347` (Δ=-0.1493), dominating the mean.

Figure:
- `docs/experiments/figures/20260103_loso4_actionset_selector_stack_v2_delta.png`

### B) Per-family calibrated selector attempt (v3, negative result)
We attempted per-family ridge/guard inside the stacked selector (commit-reproducible), but it selected risky families
(RPA/TSA/FBCSP) too often and produced larger negative transfer overall.

Run:
- `outputs/20260103/4class/loso4_actionset_selector_stack_v3_perfamily/20260103_results.txt`

Key numbers:
- EA mean acc: `0.5320`
- EA-SI-CHAN-MULTI-SAFE mean acc: `0.5463` (**+1.43% abs**, neg-transfer `0.0`) — still the best stable method here
- EA-STACK-MULTI-SAFE mean acc: `0.5197` (**-1.23% abs**), neg-transfer `0.444`

Figure (Δ vs EA):
- `docs/experiments/figures/20260103_loso4_actionset_selector_stack_v3_delta_compare.png`

## Conclusion (decision about RL)
- The action set has **large oracle headroom**, so the problem is *worth solving*.
- However, with the current label-free feature set (entropy/drift/marginal statistics), the cross-family selector is still
  vulnerable to **certificate failure / model-selection errors**. Offline RL would inherit the same issue unless we design
  **stronger probe/certificate features** (e.g., MixVal-style probes) or impose **family-specific risk constraints**.

## Next iteration (one lever)
Focus: **certificate effectiveness** for cross-family selection.
Two concrete options (pick one):
1) Add MixVal-style probe signals to candidate records (and train a stacked certificate on them).
2) Treat high-variance families (e.g., FBCSP) as “high-risk arms” with explicit conservative gates (family-specific drift/thresholds),
   while keeping the proven-safe `EA-SI-CHAN-MULTI-SAFE` as the anchor selector.
