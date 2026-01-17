# Schirrmeister2017 (HGD) — 4-class strict LOSO — pred_disagree gate (EA ↔ EA-FBCSP)

This folder contains **paper-ready** tables/figures exported from reproducible artifacts.

## Protocol
- Dataset: MOABB `Schirrmeister2017` (HGD)
- Task: 4-class (`left_hand,right_hand,feet,rest`)
- Split: cross-subject LOSO, `sessions=0train`
- Preprocess: `moabb` (8–30 Hz), `resample=50`, `tmin=0.5`, `tmax=3.5`
- Strictness: **no target labels used for selection** (labels only used for evaluation).

## Method (ours)
We build a safe selector between:
- `ea-csp-lda` (EA anchor)
- `ea-fbcsp-lda` (EA + FilterBank-CSP + LDA)

Gate certificate (unlabeled, per subject):
\[
\mathrm{pred\_disagree}=\frac{1}{n}\sum_i \mathbf{1}[\arg\max p^{EA}(x_i)\ne \arg\max p^{FBCSP}(x_i)].
\]

Threshold calibration: **LOSO-style train-only calibration**, i.e. for each test subject we choose `tau`
using only the remaining training subjects, maximizing train mean accuracy subject to **0 neg-transfer on train subjects**.
Implementation: `scripts/build_pred_disagree_calib_run.py --mode calib_loso_safe0`.

## Source runs (raw predictions)
- EA anchor: `outputs/20260112/4class/loso4_schirr2017_0train_rs50_ea_only_v2/`
- EA-FBCSP candidate: `outputs/20260115/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_lda_v1/`

## Merged run directory (used to generate paper assets)
- `outputs/20260117/4class/loso4_schirr2017_0train_rs50_pred_disagree_calib_tau_v1_merged/`

## Export scripts
- Build merged run: `scripts/build_pred_disagree_calib_run.py`
- Main table + paired stats: `scripts/make_main_table_and_stats.py`
- Main figures: `scripts/plot_strong_baselines_table_figures.py`
- Confusion matrices: `scripts/plot_confusion_from_predictions.py`
- Sensitivity (analysis-only): `scripts/eval_pred_disagree_gate_from_predictions.py`

