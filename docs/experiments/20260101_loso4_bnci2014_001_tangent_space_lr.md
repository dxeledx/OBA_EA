# 2026-01-01 — LOSO 4-class (BNCI2014_001): TangentSpace + LogisticRegression (TS-LR) baselines

Goal: try a stronger *classifier family* (beyond CSP+LDA) before further tuning certificates/guards.

We evaluate TangentSpace features on per-trial SPD covariances with a multinomial LogisticRegression.

## Setup (same protocol as our main table)

- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz

## Methods added

- `ts-lr`: TangentSpace(metric=`riemann`) + StandardScaler + LogisticRegression on raw per-trial covariances
- `rpa-ts-lr`: TLCenter+TLStretch (unlabeled target) then TS-LR on covariances
- `ea-ts-lr`: EA whitening on time series first, then TS-LR on per-trial covariances

## Runs + results

### Run A: TS-LR vs EA (CSP+LDA anchor)

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ts-lr \
  --no-plots \
  --run-name loso4_bnci2014_001_ts_lr
```

Output:
- `outputs/20260101/4class/loso4_bnci2014_001_ts_lr/20260101_method_comparison.csv`

Key numbers (mean acc):
- EA: `0.5320`
- TS-LR: `0.3272` (**worse**)

### Run B: TS-LR vs RPA-TS-LR vs EA

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ts-lr,rpa-ts-lr \
  --no-plots \
  --run-name loso4_bnci2014_001_ts_lr_rpa_ts_lr
```

Output:
- `outputs/20260101/4class/loso4_bnci2014_001_ts_lr_rpa_ts_lr/20260101_method_comparison.csv`

Key numbers (mean acc):
- EA: `0.5320`
- RPA-TS-LR: `0.5015` (still worse than EA)
- TS-LR: `0.3275` (worst)

### Run C: EA-TS-LR vs EA

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-ts-lr \
  --no-plots \
  --run-name loso4_bnci2014_001_ea_ts_lr
```

Output:
- `outputs/20260101/4class/loso4_bnci2014_001_ea_ts_lr/20260101_method_comparison.csv`

Key numbers (mean acc):
- EA: `0.5320`
- EA-TS-LR: `0.4907` (worse than EA)

## Takeaway

On `BNCI2014_001` 4-class LOSO (`paper_fir`, `0train`), TS-LR variants are **not competitive** with the EA+CSP+LDA anchor.
This suggests that simply expanding the classifier family to (covariance → tangent space → linear classifier) does **not**
provide additional headroom under this protocol; to get a *large* improvement we likely need to expand the feature family
(e.g., filterbank / multi-band / time-frequency) or move to a stronger model family, then re-apply our certificate+safe selection.

