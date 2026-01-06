# 20260106 — BNCI2014_001 4-class LOSO: add `pred_disagree` feature to calibrated selector (FAILED)

## Post-mortem (previous iteration)
Baseline we want to improve from:
- `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4/`
- EA mean acc: **0.5320**
- Stack mean acc: **0.5320**
- `accept_rate`: **0.7778** (7/9)
- `neg_transfer_rate_vs_ea`: **0.2222** (S8, S9)

Failure pattern: there exist subjects (esp. S9) where a candidate is selected even though the candidate-set oracle equals EA (no headroom),
meaning the current calibrated selector still produces **false-positive** acceptance.

---

## Hypothesis (this iteration)
Some negative-transfer cases might be caused by candidates that change the decision rule too much relative to the EA anchor.
Add a cheap, fully unlabeled diagnostic:

\[
\texttt{pred\_disagree}(c) \;=\; \frac{1}{n}\sum_{i=1}^n \mathbb{1}\big[\arg\max p_{\text{EA}}(x_i)\ne \arg\max p_c(x_i)\big]
\]
(top-1 disagreement rate between anchor and candidate).

Use it as an additional feature for the calibrated ridge/guard models (no new hyper-parameters).

---

## What changed (single lever)
- Add `pred_disagree` to candidate records and to the calibrated feature vector used by `calibrated_stack_ridge_guard`.

Commit (code): `f7a05b24c3e1a047a09a9a47fe4d97140fcbb03e`

---

## Run (same protocol + same candidate set)
Output dir:
- `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_pred_disagree_feat_v1/`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
  --oea-zo-selector calibrated_stack_ridge_guard \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-anchor-guard-delta 0.05 \
  --stack-safe-anchor-probe-hard-worsen 0.03 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_pred_disagree_feat_v1
```

---

## Results (FAILED)
From `20260106_method_comparison.csv`:
- EA mean acc: **0.5320**
- Stack mean acc: **0.5274** (**−0.46% abs**)
- `accept_rate`: **0.7778**
- `neg_transfer_rate_vs_ea`: **0.3333** (3/9)

Regressed subjects (EA → stack):
- S9: **0.6840 → 0.5868** (−9.72%)
- S4: **0.4514 → 0.4062** (−4.51%)
- S8: **0.7188 → 0.6910** (−2.78%)

Candidate-set oracle (unchanged headroom):
- `oracle_mean`: **0.5633**
- `gap_sel_mean`: **0.0359** (worse than the previous run)

Figures:
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_pred_disagree_feat_v1_delta.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_pred_disagree_feat_v1_oracle_gap.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_pred_disagree_feat_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_pred_disagree_feat_v1_guard_vs_true.png`

---

## Failure analysis (why this lever hurts)
This feature perturbs the calibrated guard/ridge models enough that in some subjects the EA anchor is assigned a very low `guard_p_pos`,
making the EA-anchor-relative δ gate effectively collapse to the loose absolute threshold (0.5).

Example S4 (from `diagnostics/.../subject_04/candidates.csv`):
- EA(anchor): `guard_p_pos=0.388` (very low)
- RPA: `guard_p_pos=0.566`, `ridge_pred_improve=+0.001` → passes the gates and is selected
- But accuracy drops: **0.4514 → 0.4062**

So the core issue is not “missing a feature”, but **guard/cert instability**: a small feature tweak can change selection qualitatively,
which is dangerous under large candidate sets.

---

## Decision / next step
Reject this lever as a main improvement.
Keep `pred_disagree` only as a diagnostic (optional), but do **not** include it in the calibrated feature set unless we also redesign the guard
to be anchor-stable (e.g., model selection that does not use `p_pos(EA)` as a stability signal).

