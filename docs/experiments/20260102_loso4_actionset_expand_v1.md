# 2026-01-02 — LOSO 4-class — Expanded Action Set (decide whether RL is worth it)

Goal: expand the **per-subject action set** (candidate alignment/model variants) under **strictly comparable** protocol, then measure the **oracle union headroom** (analysis-only) to decide whether it is worth investing in RL/bandit-style selection.

Key idea: RL/selection can only help if the candidate action set has **large achievable headroom**. If the best possible per-subject choice (oracle) barely beats the best single method, RL is unlikely to yield large gains.

## Setup (strictly comparable)
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO (cross-subject)
- Classes: `left_hand,right_hand,feet,tongue` (4-class)
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5 s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA (for all actions in this run)

## Action set (methods)
We include several **diverse** candidates that are known to behave differently per subject:
- `ea-csp-lda` (EA anchor)
- `ea-fbcsp-lda` (EA + filterbank CSP+LDA)
- `rpa-csp-lda` (RPA/LEA family)
- `tsa-csp-lda` (TSA)
- `ea-si-chan-multi-safe-csp-lda` (EA-SI-CHAN multi-candidate + calibrated ridge+guard + fallback)
- `ea-stack-multi-safe-csp-lda` (EA/RPA/TSA/EA-SI-CHAN stacked candidate selection + calibrated ridge+guard + fallback)

## Run
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-fbcsp-lda,rpa-csp-lda,tsa-csp-lda,ea-si-chan-multi-safe-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_actionset_expand_v1
```

Outputs:
- `outputs/20260102/4class/loso4_actionset_expand_v1/20260102_method_comparison.csv`
- `outputs/20260102/4class/loso4_actionset_expand_v1/20260102_predictions_all_methods.csv`
- `outputs/20260102/4class/loso4_actionset_expand_v1/20260102_results.txt`

## Results (per-method)
From `20260102_method_comparison.csv` (mean acc):
- EA anchor: `ea-csp-lda` = **0.5320**
- `ea-si-chan-multi-safe-csp-lda` = **0.5463** (**+1.43% abs**, neg-transfer=0.0)
- `ea-stack-multi-safe-csp-lda` = **0.5390** (+0.69% abs, neg-transfer=0.111)
- `ea-fbcsp-lda` = **0.5058** (worse on average)
- `rpa-csp-lda` = **0.5193** (worse on average)
- `tsa-csp-lda` = **0.5123** (worse on average)

## Oracle union headroom (analysis-only)
Using `20260102_predictions_all_methods.csv`, we computed the **oracle union**:
for each test subject, pick the method with the highest *true* accuracy among the action set.

Key numbers:
- Mean EA: **0.5320**
- Mean oracle union: **0.5579** (**+2.58% abs** vs EA)
- Worst-subject EA: **0.2569**
- Worst-subject oracle union: **0.2813**

Oracle winners by subject (what the oracle would pick):
- `ea-csp-lda`: 3 subjects
- `rpa-csp-lda`: 2 subjects
- `ea-si-chan-multi-safe-csp-lda`: 2 subjects
- `ea-stack-multi-safe-csp-lda`: 1 subject
- `ea-fbcsp-lda`: 1 subject (but with a large win on that subject)

Figures:
- Per-subject Δacc (oracle union − EA): `docs/experiments/figures/20260102_loso4_actionset_expand_v1_oracle_union_delta.png`
- Oracle winner counts: `docs/experiments/figures/20260102_loso4_actionset_expand_v1_oracle_union_winners.png`

## Conclusion: should we consider RL?
Yes **in principle**, because the expanded action set has substantial **oracle headroom**:
`0.5579 - 0.5320 = +2.58% abs`.

However, the current best **unlabeled safe** method in this set is `ea-si-chan-multi-safe-csp-lda` at `0.5463`,
so the remaining gap to the oracle union is:
`0.5579 - 0.5463 ≈ +1.16% abs`.

This suggests the next most cost-effective step is to implement a **learned selector over this action set**
(a contextual bandit / calibrated certificate over actions, with safe fallback), and see how much of the ~1.16% gap can be recovered.
If a simple calibrated selector cannot close the gap, then moving to RL-style policy learning is better justified.
