# 2025-12-23 — Safe/Calibrated certificate attempts for EA‑ZO (4-class)

This note follows the discussion that **unlabeled surrogate model selection can fail** (S4 is a clear example).
We implement two “certificate upgrade” mechanisms that do **not** use target labels and do **not** update the classifier:

1) **Drift-guard (safe gate)**: reject/penalize candidates whose predictions drift too much from the EA anchor.
2) **Calibrated selector (offline)**: train a lightweight regressor on *source subjects* (within each LOSO fold)
   to predict which candidate is likely to improve vs EA.

## A) Prediction drift guard
We define drift using the EA (Q=I) predicted probabilities as an anchor:

`D_drift(Q) = mean_i KL(p_i^EA || p_i^Q)`

Implementation:
- `csp_lda/evaluation.py` (inside `_optimize_qt_oea_zo`)
- CLI:
  - `--oea-zo-drift-mode {none,penalty,hard}`
  - `--oea-zo-drift-gamma` (penalty weight γ)
  - `--oea-zo-drift-delta` (hard threshold δ)

### Observation: drift alone cannot separate S2 vs S4
From `outputs/20251223/diag24_kl_uniform_nodrift/diagnostics/ea-zo-im-csp-lda/subject_02/candidates.csv`:
- S2 improvement comes from `q_delta` with `drift_best ≈ 0.196` and accuracy `0.2569 → 0.3299`.

From `outputs/20251223/diag4_kl_uniform_drift_pen1/diagnostics/ea-zo-im-csp-lda/subject_04/candidates.csv`:
- S4 drop also comes from `q_delta` with `drift_best ≈ 0.214` and accuracy `0.4514 → 0.3681`.

So a single global γ/δ cannot “keep S2 but reject S4” using drift magnitude alone.

### Drift penalty demonstration (safe but conservative)
Run:
`outputs/20251223/diag4_kl_uniform_drift_pen1/`

- config: `kl_uniform*0.5`, `drift=penalty(gamma=1.0)`, holdout=0.3
- effect: EA‑ZO collapses to **EA** (all subjects choose identity), i.e. safe but removes gains.

## B) Calibrated selector (Ridge regression on source subjects)
We treat candidate selection as a **model selection** problem.
Within each LOSO fold, we train a regressor on pseudo-target source subjects:
- for each pseudo-target subject `s` in the training set:
  - train a model on training subjects excluding `s`
  - generate candidate `Q` set via EA‑ZO (unlabeled)
  - compute true accuracy on subject `s` (labels are available because it is source)
  - train Ridge to predict **improvement over identity** from unlabeled features

Features (label-free, per candidate):
- `objective_base`, `pen_marginal`, `drift_best`, `mean_entropy`, `entropy_bar`, `keep_ratio`, `p̄` entropy and `p̄` entries

CLI:
- `--oea-zo-selector calibrated_ridge`
- `--oea-zo-calib-ridge-alpha`
- `--oea-zo-calib-max-subjects` (speed control)

### Result (max_subjects=3): safe, but no gains yet
Run:
`outputs/20251223/4c_calib_ridge3_improv/`

Outcome:
- EA‑ZO == EA across all subjects (the regressor predicts no positive improvement, so it falls back to identity).

## What this tells us
- The diagnostics confirm the core issue: **unlabeled objective is not a reliable certificate** (S4).
- Drift guard is effective as a *safety gate*, but drift magnitude alone cannot keep “good drift” (S2) while rejecting “bad drift” (S4).
- The first calibrated selector is safe but currently too conservative under small calibration data (`max_subjects=3`).

## Next options (if we continue)
1) Increase calibration data (`--oea-zo-calib-max-subjects 0`) and/or try different regressors/feature sets.
2) Use a 2-stage certificate: (i) drift gate, (ii) calibrated selection inside the safe set.
3) Add additional unlabeled features that better predict negative transfer (beyond drift magnitude).

