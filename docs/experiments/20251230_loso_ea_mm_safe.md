# 2025-12-30 — LOSO: EA-MM-SAFE (cross-model safe selection)

Goal: test the **certificate effectiveness + safe selection/fallback** line with a *cross-model* candidate set.

Implemented method: `ea-mm-safe`
- **Anchor**: EA + CSP+LDA (`ea-csp-lda`)
- **Candidates**:
  - EA-SI-CHAN projectors (rank-deficient channel projectors + CSP+LDA)
  - MDM(RPA): `TLCenter + TLStretch` on SPD covariances + `MDM(metric="riemann")`
- **Selector**: `calibrated_ridge_guard` trained on pseudo-target subjects (source-only), with fallback to anchor.

### Update: per-family calibration (chan vs mdm)
To reduce “cross-model probability mismatch” (CSP+LDA vs MDM), `ea-mm-safe` now trains **separate** ridge/guard models for:
- `chan` candidates (EA-SI-CHAN)
- `mdm` candidates (MDM(RPA))

Selection compares candidates by the **predicted accuracy improvement** (same unit), but uses family-specific mappings
`features -> improvement` and `features -> P(pos_improve)`.

## Code changes
- `csp_lda/evaluation.py`: add alignment branch `ea_mm_safe` (multi-model candidates + calibrated guard/ridge + fallback).
  - Per-family cert/guard: `csp_lda/evaluation.py` (`ea_mm_safe` block).

## 4-class (BNCI2014_001, 0train, paper_fir, CSP=6)

Command:
`run_csp_lda_loso.py --dataset BNCI2014_001 --preprocess paper_fir --events left_hand,right_hand,feet,tongue --sessions 0train --n-components 6 --methods ea-csp-lda,ea-mm-safe --si-chan-ranks 21 --si-chan-lambdas 0.5,2 --oea-zo-selector calibrated_ridge_guard --oea-zo-calib-guard-c 4 --oea-zo-calib-guard-threshold 0.5 --oea-zo-calib-max-subjects 4 --run-name loso4_bnci2014_001_mm_safe_q --no-plots`

Outputs:
- `outputs/20251230/4class/loso4_bnci2014_001_mm_safe_q/20251230_results.txt`
- `outputs/20251230/4class/loso4_bnci2014_001_mm_safe_q/20251230_method_comparison.csv`

Key outcome (mean acc, LOSO):
- EA: `0.5320`
- EA-MM-SAFE: `0.5432` (**Δ +0.0112**)
- accept_rate: `0.2222` (2/9 accepted), neg_transfer_rate_vs_ea: `0.0`

Notes:
- Selected families observed: mostly fallback to EA, with a few accepted `chan`/`mdm` picks.

### Per-family calibration run
Run: `outputs/20251230/4class/loso4_bnci2014_001_mm_safe_pf2/`
- EA mean acc: `0.5320`
- EA-MM-SAFE mean acc: `0.5440` (**Δ +0.0120**)
- accept_rate: `0.5556`, neg_transfer_rate_vs_ea: `0.1111`
- cert/guard vs true-improve Spearman (method-level): `0.70` / `0.80`

### Treating MDM as “high-risk” (stricter accept rules)
Run: `outputs/20251230/4class/loso4_bnci2014_001_mm_safe_mdmhard1_thr85/`
- Settings: `--oea-zo-calib-guard-threshold 0.85 --mm-safe-mdm-guard-threshold 0.9 --mm-safe-mdm-drift-delta 0.1`
- EA-MM-SAFE mean acc: `0.5413` (**Δ +0.0093**)
- accept_rate: `0.3333`, neg_transfer_rate_vs_ea: `0.0`

## 2-class (BNCI2015_001, ALL, moabb, CSP=6)

Command:
`run_csp_lda_loso.py --dataset BNCI2015_001 --preprocess moabb --events right_hand,feet --sessions ALL --n-components 6 --methods ea-csp-lda,ea-mm-safe --si-chan-ranks 12 --si-chan-lambdas 0.5,2 --oea-zo-selector calibrated_ridge_guard --oea-zo-calib-guard-c 4 --oea-zo-calib-guard-threshold 0.5 --oea-zo-calib-max-subjects 4 --run-name loso2_bnci2015_001_mm_safe_q --no-plots`

Outputs:
- `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_q/20251230_results.txt`
- `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_q/20251230_method_comparison.csv`

Key outcome (mean acc, LOSO):
- EA: `0.7220`
- EA‑MM‑SAFE: `0.7381` (**Δ +0.0161**)
- accept_rate: `0.5833`, neg_transfer_rate_vs_ea: `0.25`

Notes:
- Large mean gain comes with non-zero negative transfer on some subjects → the current guard/ridge is not yet a “hard safety certificate” on this dataset.

### Per-family calibration run
Run: `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_pf2/`
- EA mean acc: `0.7220`
- EA-MM-SAFE mean acc: `0.7253` (**Δ +0.0033**)
- accept_rate: `0.75`, neg_transfer_rate_vs_ea: `0.3333`
- cert/guard vs true-improve Spearman (method-level): `0.52` / `0.40`

### Treating MDM as “high-risk” (stricter accept rules)
1) MDM-only strictness (keeps global threshold at 0.5)
- Run: `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_mdmhard1/`
- Settings: `--mm-safe-mdm-guard-threshold 0.9 --mm-safe-mdm-drift-delta 0.1`
- EA-MM-SAFE mean acc: `0.7244` (**Δ +0.0024**)
- accept_rate: `0.5`, neg_transfer_rate_vs_ea: `0.25` (remaining negatives came from `chan`)

2) Strong “safety-first” gating (raise global threshold too)
- Run: `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_mdmhard1_thr85/`
- Settings: `--oea-zo-calib-guard-threshold 0.85 --mm-safe-mdm-guard-threshold 0.9 --mm-safe-mdm-drift-delta 0.1`
- EA-MM-SAFE mean acc: `0.7278` (**Δ +0.0058**)
- accept_rate: `0.25`, neg_transfer_rate_vs_ea: `0.0`

### Quick ablations (same dataset; safety/acceptance tradeoff)
1) Higher guard threshold (`thr=0.8`)
- Run: `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_thr08/`
- Mean Δ vs EA: `-0.0017`, accept_rate: `0.25`, neg_transfer_rate_vs_ea: `0.1667`

2) Positive-margin guard labels (`margin=0.02`, `thr=0.5`)
- Run: `outputs/20251230/2class/loso2_bnci2015_001_mm_safe_m02/`
- Mean Δ vs EA: `+0.0132`, accept_rate: `0.6667`, neg_transfer_rate_vs_ea: `0.3333`

## Environment note (MOABB/MNE cache)
To avoid MNE writing config under `~/.mne`, run with:
- `_MNE_FAKE_HOME_DIR=$PWD/.mne_home`
- `MNE_DATA=$PWD/.mne_data`
