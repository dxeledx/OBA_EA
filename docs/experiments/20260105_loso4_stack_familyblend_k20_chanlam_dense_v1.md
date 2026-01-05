# 20260105 — BNCI2014_001 4-class LOSO: dense SI-CHAN λ grid (candidate explosion test)

## Post-mortem (previous iteration)
Safe baseline (same protocol / selector / gates):
- `outputs/20260105/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_v1/`
- Mean: **0.5432** vs EA **0.5320** (**+1.12% abs**), `neg_transfer_rate_vs_ea=0.0`, `accept_rate=0.4444`

Observed limitation:
- Per-candidate ranking signal is weak, so selection quality depends heavily on the *candidate set*.

---

## Goal (this iteration)
Test a single lever:
- **Densify the SI-CHAN λ candidate grid** to reduce “certificate extrapolation” and increase the chance that a near-optimal λ exists in the discrete action set.

Intended outcome:
- Maintain `neg_transfer≈0` while (potentially) increasing mean by better coverage.

---

## Protocol (fixed, strictly comparable)
- Dataset: `BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO, `sessions=0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR order=50), epoch 0.5–3.5s, resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Selector: `calibrated_stack_ridge_guard` (unchanged)
- High-risk gates (unchanged): FBCSP/TSA stricter acceptance (`guard>=0.95`, `min_pred_improve>=0.05`, `drift_delta<=0.15`)
- Diagnostics: `--diagnose-subjects 1..9` writes per-subject `candidates.csv` (analysis-only uses true labels)

---

## What changed (single lever)
- SI-CHAN λ grid:
  - from: `0.5,1,2`
  - to: `0.25,0.35,0.5,0.7,1,1.4,2`

---

## Run
Output dir: `outputs/20260105/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_chanlam_dense_v1/`

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
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_chanlam_dense_v1
```

---

## Main results (method-level)
From `20260105_method_comparison.csv`:
- EA mean acc: **0.5320**
- Dense-λ stack mean acc: **0.5220** (**−1.00% abs**)
- Worst-subject acc: **0.2569** (same as EA)
- `accept_rate`: **1.0** (9/9)  ← **over-accept**
- `neg_transfer_rate_vs_ea`: **0.3333** (3/9)  ← **unsafe**

Largest regressions (from `*_results.txt`):
- S9: `0.6840 → 0.5625` (−12.15%)
- S8: `0.7188 → 0.6701` (−4.86%)
- S4: `0.4514 → 0.4063` (−4.51%)

Interpretation:
- Increasing candidate count triggers a **multiple-testing / false-positive** effect: even with the same guard threshold,
  the probability that *some* candidate looks “safe” rises, causing over-accept and negative transfer.

---

## Candidate-level diagnostics (certificate validity)
Using `scripts/analyze_candidate_certificates.py`:
- `oracle_mean`: **0.5633**
- `gap_sel_mean` (oracle − selected): **0.0413** (much worse than baseline **≈0.020**)
- Candidate ranking correlation (mean across subjects):
  - `rho_ridge_mean`: **0.0515**
  - `rho_guard_mean`: **0.1758**

Conclusion:
- Dense λ grid **does not** help under the current certificate/guard; it makes selection harder and safety worse.

Figures (this run):
- `docs/experiments/figures/20260105_stack_familyblend_k20_chanlam_dense_v1_delta.png`
- `docs/experiments/figures/20260105_stack_familyblend_k20_chanlam_dense_v1_oracle_gap.png`
- `docs/experiments/figures/20260105_stack_familyblend_k20_chanlam_dense_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260105_stack_familyblend_k20_chanlam_dense_v1_guard_vs_true.png`

---

## Decision / next step
Revert dense λ grid; keep the smaller grid (`0.5,1,2`) until safety is strengthened.
If we want larger action sets, we need a stronger acceptance rule that accounts for “candidate explosion”, e.g.:
- stricter CHAN-family gate (higher guard threshold / minimum predicted improvement),
- or a calibrated acceptance target / multiple-testing-aware thresholding.

