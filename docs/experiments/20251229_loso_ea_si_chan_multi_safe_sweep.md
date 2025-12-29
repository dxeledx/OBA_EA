# 2025-12-29 — LOSO (4-class + 2-class) — EA-SI-CHAN-MULTI-SAFE sweep (candidate design & selector hyperparams)

Goal: push Route B (**EA-SI-CHAN-MULTI-SAFE**) performance by exploring candidate design (λ grid / rank) and selector hyper-params,
and sanity-check on **2-class LOSO**.

Dataset/protocol shared across runs unless noted:
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA
- Selector: `calibrated_ridge_guard` (fold-local), fallback to EA anchor

## 4-class (left/right/feet/tongue), sessions=0train

### Baseline (best so far)
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 21 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_chan_ms_r21_l05-2_thr05
```
Result: mean acc **0.5463**, neg-transfer-rate vs EA **0.0**.  
This matches the previous 2025-12-27 best config (multi-candidate λ at fixed rank).

### Candidate λ grid too wide → hurts (mis-selection / generalization)
- λ = `0.25,0.5,0.75,1,1.5,2,3` (rank=21): mean **0.5340**, neg-transfer-rate **0.1111**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l025-3_thr05/20251229_method_comparison.csv`

- λ = `0.25,0.5,1,2` (rank=21): mean **0.5363**, neg-transfer-rate **0.0**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l025-2_thr05/20251229_method_comparison.csv`

- λ = `0.5,1,2,3` (rank=21): mean **0.5359**, neg-transfer-rate **0.1111**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-3_thr05/20251229_method_comparison.csv`

### Selector hyperparams (baseline λ=0.5,1,2, rank=21)
- Guard threshold `0.6` (more conservative): mean **0.5401**, neg-transfer-rate **0.0**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_thr06/20251229_method_comparison.csv`

- Guard margin `0.01` (positives = improve≥1% abs): mean **0.5363**, neg-transfer-rate **0.0**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_thr05_margin001/20251229_method_comparison.csv`

- Ridge alpha `0.1`: mean **0.5413**, neg-transfer-rate **0.1111**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_thr05_ra01/20251229_method_comparison.csv`

- Ridge alpha `10`: mean **0.5390**, neg-transfer-rate **0.0**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_thr05_ra10/20251229_method_comparison.csv`

### Rank sensitivity
- rank=20 (λ=0.5,1,2): accept-rate became **0.0** (always fallback), mean = EA  
  Output: `outputs/20251229/4class/loso4_chan_ms_r20_l05-2_thr05/20251229_method_comparison.csv`

### “Stability” under calibration subsampling (max_subjects=4)
With `--oea-zo-calib-max-subjects 4`, seeds 0–2 all kept neg-transfer-rate **0.0**,
but mean accuracy dropped compared to using all pseudo-target subjects.

- seed0: mean **0.5363**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_cal4_seed0/20251229_method_comparison.csv`
- seed1: mean **0.5397**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_cal4_seed1/20251229_method_comparison.csv`
- seed2: mean **0.5367**  
  Output: `outputs/20251229/4class/loso4_chan_ms_r21_l05-2_cal4_seed2/20251229_method_comparison.csv`

## 2-class (left/right), sessions=0train
Sanity check: in this setting, RPA/TSA baselines are again below EA, and stacked selection does not help.

Run:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand --sessions 0train \
  --methods ea-csp-lda,rpa-csp-lda,tsa-csp-lda,ea-si-chan-csp-lda,ea-si-chan-multi-safe-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso2_stack_vs_chan_multi_safe
```
Output: `outputs/20251228/2class/loso2_stack_vs_chan_multi_safe/20251228_method_comparison.csv`

Key means:
- EA: **0.7323**
- RPA: **0.7284**
- TSA: **0.7269**
- EA-SI-CHAN-MULTI-SAFE: **0.7323** (no gain over EA in this setting)
- EA-STACK-MULTI-SAFE: **0.7299**

## Takeaway
- For Route B, the best current setting is still: **rank=21**, **λ ∈ {0.5,1,2}**, guard threshold **0.5**, ridge alpha **1.0**, `si_ridge=1e-6`, `calib_max_subjects=0`.
- Widening the candidate grid (more λ) **hurts**, suggesting the current certificate/guard generalizes poorly over a larger hypothesis set (more candidates ⇒ higher mis-selection risk).

