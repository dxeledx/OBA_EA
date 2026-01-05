# 20260105 — BNCI2014_001 4-class LOSO: offline contextual bandit selector (bandit_guard) on the same action set

## Post-mortem (previous iteration)
Previous best safe run (same protocol and gates):
- `outputs/20260105/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_v1/`
- Mean: **0.5432** vs EA **0.5320** (**+1.12% abs**), `neg_transfer_rate_vs_ea=0.0`, `accept_rate=0.4444`
- But candidate-level ranking signal was still weak: `rho_ridge_mean≈0.016`, `gap_sel_mean≈0.020`

Hypothesis: the *linear ridge certificate* cannot learn a reliable per-candidate ranking from the current feature set, so the selector often fails to pick the true best candidate.

---

## Goal (this iteration)
Test a single lever:
- Replace ridge-based ranking with an **offline contextual bandit policy** trained on pseudo-target candidate sets (full-information Δacc within training folds),
  while keeping the action set, safety gates, and per-family shrinkage blending fixed.

This directly probes whether *policy learning* can improve certificate validity (correlation / oracle gap) without sacrificing safety.

---

## Protocol (fixed, strictly comparable)
- Dataset: `BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO, `sessions=0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR order=50), epoch 0.5–3.5s, resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- High-risk gates (unchanged): FBCSP/TSA stricter acceptance (`guard>=0.95`, `min_pred_improve>=0.05`, `drift_delta<=0.15`)
- Diagnostics: `--diagnose-subjects 1..9` writes per-subject `candidates.csv` (analysis-only uses true labels)

---

## What changed (single lever)
- Selector: `calibrated_stack_bandit_guard`
  - Train a linear softmax bandit policy on calibration candidate-sets to score candidates.
  - Still uses the existing guard thresholding + fallback logic.

Note: this run was executed before committing the bandit changes; reproducibility is by re-running the exact command on the follow-up commit that introduces `calibrated_stack_bandit_guard`.

---

## Run
Output dir: `outputs/20260105/4class/loso4_actionset_stack_bandit_guard_familyblend_k20_v1/`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_stack_bandit_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_bandit_guard_familyblend_k20_v1
```

---

## Main results (method-level)
From `20260105_method_comparison.csv`:
- EA mean acc: **0.5320**
- Stack+bandit mean acc: **0.5382** (**+0.62% abs**)
- Worst-subject acc: **0.2569** (same as EA)
- `accept_rate`: **0.5556** (5/9)
- `neg_transfer_rate_vs_ea`: **0.1111** (1/9)  ← **safety regression**

Key negative-transfer subject (from `*_results.txt`):
- S4: `0.4514 → 0.4410` (−1.04%)

Interpretation:
- Bandit increases acceptance, but does not maintain the previous run’s “near-zero negative transfer” property.

---

## Candidate-level diagnostics (certificate validity)
Using `scripts/analyze_candidate_certificates.py` on `diagnostics/.../candidates.csv`:
- `oracle_mean`: **0.5633**
- `gap_sel_mean` (oracle − selected): **0.0251** (worse than previous **≈0.020**)
- Candidate ranking correlation (mean across subjects):
  - `rho_bandit_mean`: **0.0952** (still weak)
  - `rho_ridge_mean`: **0.0079**, `rho_guard_mean`: **0.0556**

Conclusion:
- The offline bandit policy **did not materially improve** per-candidate ranking validity, and it increased oracle gap + introduced a negative-transfer case.

Figures (this run):
- `docs/experiments/figures/20260105_stack_bandit_guard_familyblend_k20_v1_delta.png`
- `docs/experiments/figures/20260105_stack_bandit_guard_familyblend_k20_v1_oracle_gap.png`
- `docs/experiments/figures/20260105_stack_bandit_guard_familyblend_k20_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260105_stack_bandit_guard_familyblend_k20_v1_guard_vs_true.png`
- `docs/experiments/figures/20260105_stack_bandit_guard_familyblend_k20_v1_bandit_vs_true.png`

---

## Decision / next step
Keep the action set and gates fixed; revert to the ridge-guard selector as the safe baseline.
Next iteration should focus on *reducing selection extrapolation* and tightening safety:
- densify `--si-chan-lambdas` grid (more candidate coverage; less reliance on ranking generalization),
- and/or strengthen guard for the CHAN family (S4-style failures).

