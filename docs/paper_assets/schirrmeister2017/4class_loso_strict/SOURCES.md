# Schirrmeister2017 (HGD) — 4-class — LOSO (strict)

All assets in this folder are derived from the following **merged** run folder (same protocol / same trials):

- `outputs/20260117/4class/loso4_schirr2017_0train_rs50_strong_baselines_plus_ours_v1_merged/`

Protocol:
- Dataset: MOABB `Schirrmeister2017` (HGD)
- Task: 4-class (`left_hand,right_hand,feet,rest`)
- Split: cross-subject LOSO, `--sessions 0train`
- Preprocessing: `--preprocess moabb`, bandpass 8–30 Hz, `--resample 50`, epoch `tmin=0.5s`, `tmax=3.5s`

Methods included (CSP+LDA family):
- `ea-csp-lda` (EA anchor)
- `rpa-csp-lda` (LEA-CSP; closed-form alignment baseline)
- `tsa-csp-lda` (LEA-ROT-CSP; closed-form rotation-on-LEA baseline)
- `fbcsp-lda` (no EA)
- `ea-fbcsp-lda` (EA + FBCSP candidate; typically high-reward but high-risk)
- `ea-fbcsp-pred-disagree-safe` (ours): EA↔EA-FBCSP **pred_disagree** gate with **train-only LOSO-style calibration** of `tau` and safe fallback to EA

Source runs used to build the merged run:
- `outputs/20260114/4class/loso4_schirr2017_0train_rs50_csp_family_table_v3_merged/`
- `outputs/20260114/4class/loso4_schirr2017_0train_rs50_fbcsp_lda_v1/`
- `outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged/`

Build script:
- `scripts/merge_loso_runs.py`
