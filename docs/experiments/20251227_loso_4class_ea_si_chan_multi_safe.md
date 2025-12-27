# 2025-12-27 — LOSO 4-class — EA-SI-CHAN-MULTI-SAFE (multi-candidate + calibrated ridge+guard + fallback)

Goal: move beyond binary `{EA anchor vs one EA-SI-CHAN candidate}` and test **multi-candidate selection** while keeping a **safe fallback to EA**.

## Method (EA-SI-CHAN-MULTI-SAFE)
Candidate set (per fold):
- Anchor: `A = I` (EA only)
- Candidates: `A = QQᵀ` (rank-deficient channel projector), with a small grid of hyper-params

Selector (no target labels):
- `selector=calibrated_ridge_guard`: learn a fold-local ridge regressor for *expected improvement* and a logistic guard for *rejecting negative transfer* using pseudo-target subjects from the source fold; then select the best candidate; fallback to `A=I` if not confident / predicted non-positive.

## Setup
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Classes: `left_hand,right_hand,feet,tongue`
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Run (recommended config)
We found that **varying λ at a fixed rank** worked better than mixing ranks (mixing ranks could mis-select and hurt).

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_easichan_multi_safe_r21_l05-2_calAll_thr05
```

Outputs:
- `outputs/20251227/4class/loso4_easichan_multi_safe_r21_l05-2_calAll_thr05/20251227_method_comparison.csv`
- `outputs/20251227/4class/loso4_easichan_multi_safe_r21_l05-2_calAll_thr05/20251227_results.txt`

## Key numbers
From `20251227_method_comparison.csv`:
- EA mean acc: **0.5320**
- EA-SI-CHAN mean acc: **0.5394**
- EA-SI-CHAN-MULTI-SAFE mean acc: **0.5463** (**+1.43% abs vs EA**, **+0.69% abs vs EA-SI-CHAN**)
- Negative transfer rate vs EA: **0.0**
- Worst-subject acc: **0.2604** (slightly above EA’s 0.2569)

Certificate effectiveness (across subjects, for the selected transform):
- `cert_improve_spearman`: **0.3667**

