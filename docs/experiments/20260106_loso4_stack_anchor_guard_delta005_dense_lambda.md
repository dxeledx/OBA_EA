# 20260106 — BNCI2014_001 4-class LOSO: EA-anchor-relative guard margin δ (anti multiple-testing) on dense λ grid

## Post-mortem (previous iteration: candidate explosion failure)
Dense SI-CHAN λ grid without anchor-relative control:
- `outputs/20260105/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_chanlam_dense_v1/`
- EA mean: **0.5320**
- Stack mean: **0.5220** (−1.00% abs)
- `accept_rate`: **1.0** (9/9) → **over-accept**
- `neg_transfer_rate_vs_ea`: **0.3333** (3/9)
- Worst regressions: S9 (−12.15%), S8 (−4.86%), S4 (−4.51%)

Interpretation: increasing the candidate set (dense λ) caused a **multiple-testing / false-positive** effect:
some candidates look “safe” under the unlabeled guard/certificate, but still harm accuracy.

---

## Goal (this iteration)
Test a single lever to reduce false-positive acceptance under large candidate sets:

**EA-anchor-relative guard margin** (δ rule)
\[
\text{accept cand only if } p_{\text{pos}}(\text{cand}) \ge \tau \ \text{and}\ p_{\text{pos}}(\text{cand}) \ge p_{\text{pos}}(\text{EA}) + \delta.
\]
- EA anchor is the identity candidate (`kind=identity`, `cand_family=ea`) in the candidate set.
- δ is applied uniformly to all non-EA candidates (family-agnostic).

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
- Per-family calibration: shrinkage blend `K=20` (unchanged)
- Candidate set: dense SI-CHAN λ grid `0.25,0.35,0.5,0.7,1,1.4,2` (same as failure case)

---

## What changed (single lever)
- New option: `--stack-safe-anchor-guard-delta 0.05`
- Implementation: enforce the δ rule inside guarded selection, and also inside the “re-select after high-risk family block” logic
  (so blocked FBCSP/TSA cannot bypass the anchor-relative acceptance rule).

Commit (code): `5cc4406e63916d91d71732c6257a75a4b50560b7`

---

## Run
Output dir: `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1/`

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
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1
```

---

## Main results (method-level)
From `20260105_method_comparison.csv`:
- EA mean acc: **0.5320**
- Dense-λ + δ stack mean acc: **0.5316** (−0.04% abs vs EA; +0.96% abs vs dense-λ baseline)
- Worst-subject acc: **0.2569** (same as EA)
- `accept_rate`: **0.7778** (7/9)  ↓ from **1.0**
- `neg_transfer_rate_vs_ea`: **0.2222** (2/9) ↓ from **0.3333**

Remaining negative-transfer subjects:
- S8: `0.7188 → 0.6701` (−4.86%)
- S9: `0.6840 → 0.6042` (−7.99%)

Interpretation:
- δ margin reduces over-accept and recovers mean relative to the “dense grid collapse” run,
  but is **not sufficient** to guarantee near-zero negative transfer under a large candidate set.

---

## Candidate-level diagnostics (certificate validity)
Using `scripts/analyze_candidate_certificates.py`:
- `oracle_mean`: **0.5633**
- `gap_sel_mean` (oracle − selected): **0.0316** (improved vs **0.0413**, still worse than safe baseline **≈0.020**)
- Candidate ranking correlation (mean across subjects):
  - `rho_ridge_mean`: **0.0515**
  - `rho_guard_mean`: **0.1758**

Figures (this run):
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1_delta.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1_oracle_gap.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1_guard_vs_true.png`

---

## Decision / next step (per $eeg-q1-loso-loop)
The δ rule is directionally correct (reduces over-accept), but we still get large negative-transfer cases when the anchor itself has low guard confidence.
Next iteration should keep protocol + candidate set fixed and change **only** the acceptance rule again, e.g.:
- make δ adaptive to candidate-set size (multiple-testing correction), or
- enforce a stricter absolute threshold when `guard_p_pos(EA)` is low (avoid “anchor uncertain → accept anything” failure mode).
