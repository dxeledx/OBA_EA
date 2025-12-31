# 2025-12-30 — Multi-dataset LOSO sanity (EA vs EA-SI-CHAN-MULTI-SAFE)

Goal: check whether our current best **EA-SI-CHAN-MULTI-SAFE** setting transfers beyond BNCI2014_001,
and register results for reproducible multi-dataset comparison.

## Run 1 — BNCI2014_001 (BCI IV-2a), 4-class, sessions=0train

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 21 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_bnci2014_001_chan_ms_r21_l05-2_thr05
```

Output:
- `outputs/20251230/4class/loso4_bnci2014_001_chan_ms_r21_l05-2_thr05/20251230_results.txt`

Key result (mean acc):
- EA: **0.5320**
- EA-SI-CHAN-MULTI-SAFE: **0.5463**  (+1.43% abs)

## Run 2 — BNCI2014_002, 2-class, sessions=ALL

Note: BNCI datasets require MOABB downloads; sandboxed env needs a writable MNE config dir.
We use `_MNE_FAKE_HOME_DIR` and `MNE_DATA` inside the command.

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_002 \
  --preprocess moabb --n-components 6 \
  --events right_hand,feet --sessions ALL \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 14 --si-ridge 1e-6 \
  --si-chan-ranks 14 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 4 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso2_bnci2014_002_chan_ms_r14_l05-2_thr05_cal4
```

Output:
- `outputs/20251230/2class/loso2_bnci2014_002_chan_ms_r14_l05-2_thr05_cal4/20251230_results.txt`

Key result:
- EA: **0.7357**
- EA-SI-CHAN-MULTI-SAFE: **0.7357** (no change; `chan_multi_accept=0` for all subjects → always fallback to EA)

## Takeaway

- The method reproduces the BNCI2014_001 4-class gain under the current codebase.
- On BNCI2014_002 (2-class, 15 channels), the calibrated guard/selector becomes too conservative (no acceptance),
  so the method degenerates to the EA anchor.

## Run 3 — BNCI2014_004, 2-class, sessions=ALL (note: only 3 channels)

We had to add a CSP regularization fallback in `fit_csp_lda` because the rank-2 channel projector can make the
generalized eigen-problem ill-posed when the covariance becomes non-SPD.

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_004 \
  --preprocess moabb --n-components 4 \
  --events left_hand,right_hand --sessions ALL \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 2 --si-ridge 1e-6 \
  --si-chan-ranks 2 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso2_bnci2014_004_chan_ms_r2_l05-2_thr05
```

Output:
- `outputs/20251230/2class/loso2_bnci2014_004_chan_ms_r2_l05-2_thr05/20251230_results.txt`

Key result:
- EA: **0.7285**
- EA-SI-CHAN-MULTI-SAFE: **0.7290** (tiny gain; mostly fallback)

## Run 4 — BNCI2015_001, 2-class, sessions=ALL (13 channels)

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2015_001 \
  --preprocess moabb --n-components 6 \
  --events right_hand,feet --sessions ALL \
  --methods ea-csp-lda,ea-si-chan-multi-safe-csp-lda \
  --si-proj-dim 12 --si-ridge 1e-6 \
  --si-chan-ranks 12 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso2_bnci2015_001_chan_ms_r12_l05-2_thr05
```

Output:
- `outputs/20251230/2class/loso2_bnci2015_001_chan_ms_r12_l05-2_thr05/20251230_results.txt`

Key result:
- EA: **0.7220**
- EA-SI-CHAN-MULTI-SAFE: **0.7196** (slightly worse; some accepted candidates were negative-transfer)
