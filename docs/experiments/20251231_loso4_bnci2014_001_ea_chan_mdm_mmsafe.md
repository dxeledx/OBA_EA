# 2025-12-31 — LOSO 4-class (BNCI2014_001): EA vs CHAN-safe vs RPA-MDM vs EA-MM-SAFE

Goal: a **same-protocol** comparison table on `BNCI2014_001` 4-class LOSO (session `0train`), including:
- `ea-csp-lda` (anchor)
- `ea-si-chan-safe-csp-lda` (CHAN-only safe selection)
- `rpa-mdm` (MDM baseline)
- `ea-mm-safe` (cross-model candidates + calibrated guard/ridge + fallback)

## Command

```bash
_MNE_FAKE_HOME_DIR=$PWD/.mne_home MNE_DATA=$PWD/.mne_data \
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-safe-csp-lda,rpa-mdm,ea-mm-safe \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,2 \
  --oea-zo-selector calibrated_ridge_guard \
  --oea-zo-calib-guard-c 4 \
  --oea-zo-calib-guard-threshold 0.85 \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --mm-safe-mdm-guard-threshold 0.9 \
  --mm-safe-mdm-drift-delta 0.15 \
  --no-plots \
  --run-name loso4_bnci2014_001_ea_chanSafe_rpaMdm_mmSafe_thr85_mdm90_d15
```

Outputs:
- `outputs/20251231/4class/loso4_bnci2014_001_ea_chanSafe_rpaMdm_mmSafe_thr85_mdm90_d15/20251231_results.txt`
- `outputs/20251231/4class/loso4_bnci2014_001_ea_chanSafe_rpaMdm_mmSafe_thr85_mdm90_d15/20251231_method_comparison.csv`

## Main table (mean / worst / neg-transfer)

| method | mean acc | worst-subject | Δ vs EA | neg-transfer rate vs EA | accept_rate |
|---|---:|---:|---:|---:|---:|
| EA (`ea-csp-lda`) | 0.5320 | 0.2569 | 0.0000 | — | — |
| CHAN-safe (`ea-si-chan-safe-csp-lda`) | 0.5320 | 0.2569 | 0.0000 | 0.0000 | 0.0000 |
| RPA-MDM (`rpa-mdm`) | 0.5181 | 0.2569 | -0.0139 | 0.5556 | — |
| **EA-MM-SAFE** (`ea-mm-safe`) | **0.5475** | **0.2604** | **+0.0154** | **0.0000** | 0.4444 |

## Notes / Interpretation

- `ea-si-chan-safe-csp-lda` accepted **no candidates** (`accept_rate=0`), so it degenerates to the EA anchor in this setting.
- `rpa-mdm` is unstable here: worse mean and high negative transfer rate.
- `ea-mm-safe` achieves the best mean (+1.54% abs) while keeping `neg_transfer_rate_vs_ea=0.0`.

