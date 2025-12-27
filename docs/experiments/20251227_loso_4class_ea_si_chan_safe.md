# 2025-12-27 — LOSO 4-class — EA-SI-CHAN + calibrated guard (binary) + fallback

Goal: take the best-performing **EA-SI-CHAN** setting (rank=21) and wrap it in a **binary calibrated guard + fallback**:
- Candidate set: `{EA anchor (A=I), EA-SI-CHAN candidate (A=QQᵀ)}`
- Selector: `calibrated_guard` trained on pseudo-target subjects (fold-local)
- Fallback: if not accepted, use EA (anchor)

## Setup
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Classes: `left_hand,right_hand,feet,tongue`
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA
- EA-SI-CHAN: `rank=21`, `si_lambda=1`, `si_ridge=1e-6`

## Runs (two seeds to sanity-check stability)

### Seed=0
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-csp-lda,ea-si-chan-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --oea-zo-calib-max-subjects 4 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_easichan_safe_r21_thr05_cal4
```
Outputs:
- `outputs/20251227/4class/loso4_easichan_safe_r21_thr05_cal4/20251227_method_comparison.csv`

Key numbers (accuracy):
| method | mean | worst-subject | neg-transfer vs EA |
|---|---:|---:|---:|
| EA | 0.5320 | 0.2569 | — |
| EA-SI-CHAN | 0.5394 | 0.2708 | 0.1111 |
| EA-SI-CHAN-SAFE | 0.5409 | 0.2708 | 0.0000 |

### Seed=1
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-csp-lda,ea-si-chan-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --oea-zo-calib-max-subjects 4 --oea-zo-calib-seed 1 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_easichan_safe_r21_thr05_cal4_seed1
```
Outputs:
- `outputs/20251227/4class/loso4_easichan_safe_r21_thr05_cal4_seed1/20251227_method_comparison.csv`

Key numbers (accuracy):
| method | mean | worst-subject | neg-transfer vs EA |
|---|---:|---:|---:|
| EA | 0.5320 | 0.2569 | — |
| EA-SI-CHAN | 0.5394 | 0.2708 | 0.1111 |
| EA-SI-CHAN-SAFE | 0.5436 | 0.2708 | 0.0000 |

## Observation
With `rank=21`, **EA-SI-CHAN** has small mean gains but still shows negative transfer on some subjects.
The **EA-SI-CHAN-SAFE** wrapper (binary calibrated guard + fallback to EA) removes negative transfer in these two runs and slightly improves mean accuracy.
