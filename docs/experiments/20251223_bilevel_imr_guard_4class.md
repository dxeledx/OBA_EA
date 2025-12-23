# 2025-12-23 — Continuous bilevel (w,q) + calibrated guard (4-class)

Goal: implement the “continuous bilevel” main line for **4-class** LOSO:
- **Inner/lower**: solve continuous reliability weights `w_i ∈ [0,1]` + soft labels `q_i ∈ Δ^{K-1}`.
- **Outer/upper**: optimize `Q_t ∈ O(C)` at test-time with frozen classifier via ZO (SPSA).
- Add a **negative-transfer binary calibrated guard** for candidate selection.

## Implementation notes

### A) New bilevel objective
Implemented `infomax_bilevel` / `entropy_bilevel`:
- file: `csp_lda/evaluation.py` (`_optimize_qt_oea_zo`)
- inner solver `_solve_inner_wq(p, prior)`:
  - `q_i ∝ (p_i)^{1/T} ⊙ a` with iterative scaling of `a` to match a prior π (weighted),
  - `w_i` updated continuously by sigmoid gating (confidence/entropy),
  - prior-matching step strength is **coverage-gated** by `coverage = mean(w)`.

CLI knobs:
- `--oea-zo-objective infomax_bilevel` (or `ea-zo-imr-csp-lda` alias)
- `--oea-zo-bilevel-iters`, `--oea-zo-bilevel-temp`, `--oea-zo-bilevel-step`,
  `--oea-zo-bilevel-coverage-target`, `--oea-zo-bilevel-coverage-power`

### B) Drift distribution + coverage features for “certificate”
Candidate records now log additional **distributional drift** and **coverage** stats:
- per-sample drift `KL(p_EA || p_Q)` quantiles (`q50/q90/q95/max`), std, tail fraction,
- `coverage`, `eff_n`, and (if bilevel) `q_bar`.

This updates the feature extractor:
- file: `csp_lda/certificate.py` (`candidate_features_from_record`)

### C) Negative-transfer calibrated guard (binary)
New selector mode:
- `--oea-zo-selector calibrated_guard`
- trains a `LogisticRegression` guard on pseudo-target subjects inside each LOSO fold:
  - label: `improvement >= margin` is positive (identity is positive when `margin=0`)
  - select: keep candidates passing `P(pos) >= threshold`, then choose by recorded unlabeled score.

Files:
- `csp_lda/certificate.py`: `LogisticGuard`, `train_logistic_guard`, `select_by_guarded_objective`
- `csp_lda/evaluation.py`: guard training/selection in `alignment == "ea_zo"` branch

## 4-class runs (paper_fir, CSP=6)

### 1) Objective selection (no calibration)
Output:
- `outputs/20251223/4class/4c_imr_obj/20251223_results.txt`

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --preprocess paper_fir --n-components 6 \
  --oea-q-blend 0.3 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --oea-zo-holdout-fraction 0.3 --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 identity \
  --oea-zo-reliable-metric entropy --oea-zo-reliable-threshold 1.0 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-prior anchor_pred --oea-zo-marginal-prior-mix 0.2 --oea-zo-marginal-beta 0.2 \
  --oea-zo-bilevel-iters 5 --oea-zo-bilevel-temp 1.0 --oea-zo-bilevel-step 1.0 --oea-zo-bilevel-coverage-target 0.5 \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --run-name 4c_imr_obj
```

Result (mean acc):
- `ea-csp-lda`: 0.5320
- `ea-zo-imr-csp-lda`: ~0.5316 (nearly identical; one subject slightly down)

### 2) Calibrated guard selection
Output:
- `outputs/20251223/4class/4c_imr_guard3/20251223_results.txt`

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --preprocess paper_fir --n-components 6 \
  --oea-q-blend 0.3 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --oea-zo-holdout-fraction 0.3 --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 identity \
  --oea-zo-reliable-metric entropy --oea-zo-reliable-threshold 1.0 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-prior anchor_pred --oea-zo-marginal-prior-mix 0.2 --oea-zo-marginal-beta 0.2 \
  --oea-zo-bilevel-iters 5 --oea-zo-bilevel-temp 1.0 --oea-zo-bilevel-step 1.0 --oea-zo-bilevel-coverage-target 0.5 \
  --oea-zo-selector calibrated_guard --oea-zo-calib-max-subjects 3 \
  --oea-zo-calib-guard-threshold 0.6 --oea-zo-calib-guard-margin 0.0 \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --run-name 4c_imr_guard3
```

Result:
- `ea-zo-imr-csp-lda` ~= EA baseline (guard is conservative; tends to keep identity-like candidates in this setting).

## Observation
- With the current “safe” configuration (holdout + trust + prior constraint), bilevel IM-R behaves stably but tends to be conservative on 4-class.
- The infrastructure is now in place to run the next iteration the user proposed:
  - richer certificate features (already added),
  - explicit rejection statistics / negative-transfer rate reporting,
  - sweeping guard threshold / coverage gates, and
  - (optionally) two-stage selection (guard + regression) once guard is predictive.

