# 20260105 — BNCI2014_001 4-class LOSO: shrinkage/partial-pooling per-family calibration (blend, K=20)

## Goal (this iteration)
Previous finding (`20260105_loso4_stack_cert_correlation_diagall_v1.md`):
- Global calibrated selector is **safe** but rarely “selects the best candidate” (large oracle gap, low accept rate).
- Naive per-family calibration (`--stack-calib-per-family`, hard switch) **over-accepts** and causes negative transfer.

This iteration tests a single lever:
- **Partial pooling / shrinkage** when using per-family calibrated models: instead of hard switching to family-specific models, blend
  \[
  \hat y = (1-w)\hat y_{\text{global}} + w\hat y_{\text{family}},\quad w=\frac{n}{n+K}.
  \]

---

## Protocol (fixed, strictly comparable)
- Dataset: `BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO, `sessions=0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR order=50), epoch 0.5–3.5s, resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Selector: `calibrated_stack_ridge_guard`
- High-risk gates (unchanged): FBCSP/TSA stricter acceptance (`guard>=0.95`, `min_pred_improve>=0.05`, `drift_delta<=0.15`)
- Diagnostics: `--diagnose-subjects 1..9` writes per-subject `candidates.csv` (analysis-only uses true labels)

---

## What changed (single lever)
**New option:** `--stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20`

Commit: `e75b9524360c69ab1fe4377c4188f6ab349066f3`

---

## Run
Output dir: `outputs/20260105/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_v1/`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_stack_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_fbcsp_tsa_gate_diagall_familyblend_k20_v1
```

---

## Main results (method-level)
From `20260105_method_comparison.csv`:
- EA mean acc: **0.5320**
- Stack mean acc: **0.5432** (**+1.12% abs**)
- Worst-subject acc: **0.2604** (EA worst: **0.2569**)
- `neg_transfer_rate_vs_ea`: **0.0**
- `accept_rate`: **0.4444** (4/9)

Accepted & improved subjects (from `*_results.txt`):
- S2: `0.2569 → 0.2604` (+0.0035)
- S3: `0.7500 → 0.7951` (+0.0451)
- S5: `0.3090 → 0.3403` (+0.0313)
- S7: `0.5729 → 0.5938` (+0.0208)

Comparison to previous runs (same protocol):
- Global (safe baseline): mean **0.5405**, `neg_transfer=0.0`, `accept_rate=0.3333`  
  (`outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_global_v1/`)
- Per-family hard (failed): mean **0.5301**, `neg_transfer=0.2222`, `accept_rate=0.7778`  
  (`outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_perfamily_v1/`)

---

## Candidate-level diagnostics (certificate validity)
Using `scripts/analyze_candidate_certificates.py` on `diagnostics/.../candidates.csv`:
- `oracle_mean`: **0.5633**
- `gap_sel_mean` (oracle − selected): **0.0201** (global baseline was **0.0228**)
- Candidate ranking correlation (mean across subjects):
  - `rho_ridge_mean`: **0.0159** (still ~0)
  - `rho_guard_mean`: **0.0159** (still ~0)

Interpretation:
- Shrinkage blending helps **avoid the per-family over-accept failure mode**, and slightly reduces the oracle gap,
  but the learned per-candidate ranking signal is still weak.

Figures (this run):
- `docs/experiments/figures/20260105_stack_diagall_familyblend_k20_v1_delta.png`
- `docs/experiments/figures/20260105_stack_diagall_familyblend_k20_v1_oracle_gap.png`
- `docs/experiments/figures/20260105_stack_diagall_familyblend_k20_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260105_stack_diagall_familyblend_k20_v1_guard_vs_true.png`

---

## Next step (if we continue improving certificate validity)
Keep the action set + high-risk gates fixed, and change **only** the certificate feature/model (not the fallback rules):
- add family-conditional interactions (global model with per-family slopes under strong regularization), or
- enrich features with classifier-consistent signals (e.g., LDA margin statistics / stability under perturbations),
then re-evaluate via `gap_sel_mean` and candidate-level Spearman.

