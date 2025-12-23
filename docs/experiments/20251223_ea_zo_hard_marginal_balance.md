# 20251223 — EA‑ZO with “harder” class‑marginal balance (4‑class)

## Goal
Improve **4‑class** LOSO robustness by explicitly discouraging **class‑marginal collapse** in unlabeled test‑time optimization (EA‑ZO), while keeping the safety rails:

- candidate‑set selection including `Q=I` (EA anchor)
- unlabeled holdout best‑iterate selection
- trust‑region penalty
- reliability weighting

This corresponds to the “B” direction: a structural constraint targeting the common 4‑class failure mode (marginal collapse / missing classes).

## Implementation
Added a class‑marginal regularizer on the predicted marginal distribution:

- `p̄ = mean_i p(y|x_i; Q)` (unweighted marginal; avoids bias from reliability weights)
- options via `--oea-zo-marginal-mode`:
  - `kl_uniform`: `-mean(log p̄_k)` (KL(u||p̄) up to a constant; barrier‑like)
  - `hinge_uniform`: `mean(max(0, τ - p̄_k)^2)`
  - `l2_uniform`: `mean((p̄_k - 1/K)^2)`
  - `hard_min`: reject candidates if `min(p̄_k) < τ`

Code:
- `csp_lda/evaluation.py` (`_objective_from_proba` inside `_optimize_qt_oea_zo`)
- `run_csp_lda_loso.py` (CLI wiring)

## 4‑class experiment (paper_fir, CSP=6)

Baseline + EA‑ZO (InfoMax) with KL‑uniform marginal penalty:

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --events left_hand,right_hand,feet,tongue \
  --preprocess paper_fir --n-components 6 \
  --methods ea-csp-lda,ea-zo-im-csp-lda \
  --oea-zo-objective infomax \
  --oea-zo-marginal-mode kl_uniform --oea-zo-marginal-beta 0.5 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 identity \
  --oea-zo-holdout-fraction 0.3 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 1 \
  --oea-zo-fallback-min-marginal-entropy 0.5 \
  --oea-zo-min-improvement 0.0 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --run-name ea_zo_4c_klb05_unw
```

Results:
- `outputs/20251223/ea_zo_4c_klb05_unw/20251223_results.txt`

Mean accuracy:
- `ea-csp-lda`: `0.532022`
- `ea-zo-im-csp-lda` (+ marginal `kl_uniform`): `0.535880` (**+0.39% abs**)

Per‑subject behavior (key deltas):
- S2: `0.256944 → 0.329861` (**+7.29% abs**)
- S4: `0.451389 → 0.368056` (**−8.33% abs**)
- S5: `0.309028 → 0.343750` (**+3.47% abs**)
- S6: `0.357639 → 0.375000` (**+1.74% abs**)

Notes:
- Increasing `--oea-zo-marginal-beta` from `0.5 → 1.0` produced the same outcome in this setup (`outputs/20251223/ea_zo_4c_klb1_unw/20251223_results.txt`).
- More aggressive “hard” constraints (`hard_min` with large `τ`) can harm accuracy by forcing class‑marginal uniformity even when it conflicts with the classifier’s current decision boundary.

