# 2025-12-28 — LOSO 4-class — EA-STACK-MULTI-SAFE (EA vs RPA vs TSA vs EA-SI-CHAN candidates)

Goal: test whether adding **RPA/TSA-style candidates** (as additional “lower-level” alignment families) improves
the **calibrated certificate + safe selection** framework on **4-class LOSO**.

## Method (EA-STACK-MULTI-SAFE)
Candidate families (per fold):
- Anchor: `EA` (`A=I` on EA-whitened data)
- `RPA` baseline: per-subject **Log-Euclidean whitening** (`rpa-csp-lda`)
- `TSA` baseline: **TSA closed-form target rotation** on top of Log-Euclidean whitening (`tsa-csp-lda`)
- `EA-SI-CHAN` candidates: rank-deficient channel projectors `A=QQᵀ` learned on the source fold (grid over λ at fixed rank)

Selector (no target labels):
- `selector=calibrated_ridge_guard`: fold-local ridge regressor for expected improvement + logistic guard to reject negative transfer,
  trained on pseudo-target subjects from the source fold; fallback to EA anchor when predicted improvement ≤ 0 or marginal entropy safety triggers.

## Setup
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Classes: `left_hand,right_hand,feet,tongue`
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Run (comparison)
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,rpa-csp-lda,tsa-csp-lda,ea-si-chan-csp-lda,ea-si-chan-multi-safe-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_stack_vs_chan_multi_safe
```

Outputs:
- `outputs/20251228/4class/loso4_stack_vs_chan_multi_safe/20251228_method_comparison.csv`
- `outputs/20251228/4class/loso4_stack_vs_chan_multi_safe/20251228_results.txt`

## Key numbers (mean accuracy; neg-transfer vs EA)
From `20251228_method_comparison.csv`:
- EA: **0.5320**
- RPA (LEA whitening): **0.5193** (worse than EA)
- TSA (LEA+TSA): **0.5123** (worse than EA)
- EA-SI-CHAN: **0.5394** (+0.73% abs; neg-transfer rate 0.1111)
- EA-SI-CHAN-MULTI-SAFE: **0.5463** (**+1.43% abs**; neg-transfer rate **0.0**)
- EA-STACK-MULTI-SAFE: **0.5390** (+0.69% abs; neg-transfer rate 0.1111)

## Observation
In this setting, **RPA/TSA baselines are not strong** (both underperform EA), and including them as additional candidates
in the stacked selector did **not** improve over the best current approach (**EA-SI-CHAN-MULTI-SAFE**).
The likely reason is that adding weaker candidate families increases mis-selection risk for the certificate/guard.

