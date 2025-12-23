# 2025-12-23 — Step A (minimal sweep) for **4-class** EA → EA‑ZO (InfoMax + marginal balance)

This note is the requested **Step A** after the Step B structural change (class‑marginal balance penalty).  
Goal: find a small “sweet spot” (or at least avoid regressions) for **4‑class** LOSO on BCI IV 2a.

## Common setup
- Dataset: MOABB `BNCI2014_001` (BCI Competition IV 2a), sessions `0train`
- Events (4-class): `left_hand`, `right_hand`, `feet`, `tongue`
- Preprocess: `paper_fir` (causal 50-order FIR Hamming), bandpass 8–30 Hz, resample 250 Hz
- Epoch window: `tmin=0.5s`, `tmax=3.5s`
- Model: CSP `n_components=6` + LDA(default)
- Evaluation: LOSO (9 subjects), metrics reported in each run folder

Baseline EA run is identical across runs:
- `ea-csp-lda` overall accuracy = **0.532022**

## Sweep summary (overall accuracy)
All runs compare `ea-csp-lda` vs `ea-zo-im-csp-lda` (EA training, test-time ZO only optimizes `Q_t`).

| Run folder (outputs/20251223/...) | Key change | EA‑ZO acc | Δ vs EA |
|---|---:|---:|---:|
| `4c_A_base_q1_beta05_trust01/` | `q_blend=1.0`, `kl_uniform*0.5`, holdout=0.3 | **0.535880** | **+0.003858** |
| `4c_A_q03_beta05_trust01/` | `q_blend=0.3` | 0.532793 | +0.000771 |
| `4c_A_q05_beta05_trust01/` | `q_blend=0.5` | 0.522377 | −0.009645 |
| `4c_A_q1_margNone/` | **no marginal penalty** | 0.513117 | −0.018905 |
| `4c_A_holdout05/` | holdout=0.5 | 0.513117 | −0.018905 |

Extra checks (no observable change vs the base run):
- `4c_A_q1_beta05_trust0/`, `4c_A_q1_beta05_trust02/`: trust `ρ ∈ {0, 0.2}` → same result as base
- `4c_A_reliable06/`, `4c_A_reliable08/`: reliability threshold `τ ∈ {0.6, 0.8}` → same result as base
- `4c_A_lam2/`: InfoMax `λ=2` → same result as base

## Per-subject accuracy deltas (base run)
Run: `outputs/20251223/4c_A_base_q1_beta05_trust01/20251223_results.txt`

| Subject | EA | EA‑ZO | Δ |
|---:|---:|---:|---:|
| 1 | 0.687500 | 0.687500 | +0.000000 |
| 2 | 0.256944 | 0.329861 | **+0.072917** |
| 3 | 0.750000 | 0.750000 | +0.000000 |
| 4 | 0.451389 | 0.368056 | **−0.083333** |
| 5 | 0.309028 | 0.343750 | +0.034722 |
| 6 | 0.357639 | 0.375000 | +0.017361 |
| 7 | 0.572917 | 0.565972 | −0.006945 |
| 8 | 0.718750 | 0.718750 | +0.000000 |
| 9 | 0.684028 | 0.684028 | +0.000000 |

Interpretation:
- The **overall gain** is mainly driven by **S2** (+7.29% abs), while **S4 drops** (−8.33% abs).
- `q_blend=0.3` becomes too conservative (little change); `q_blend=0.5` becomes harmful (S6/S7 drop).
- **Removing the marginal-balance term** is dangerous (S1 collapses badly).

## Commands (repro)
Base (best in this sweep):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --events left_hand,right_hand,feet,tongue \
  --preprocess paper_fir --n-components 6 \
  --methods ea-csp-lda,ea-zo-im-csp-lda \
  --oea-q-blend 1.0 \
  --oea-zo-marginal-mode kl_uniform --oea-zo-marginal-beta 0.5 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 identity \
  --oea-zo-holdout-fraction 0.3 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 1 \
  --oea-zo-fallback-min-marginal-entropy 0.5 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --run-name 4c_A_base_q1_beta05_trust01
```

Worst ablation (remove marginal penalty):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --events left_hand,right_hand,feet,tongue \
  --preprocess paper_fir --n-components 6 \
  --methods ea-csp-lda,ea-zo-im-csp-lda \
  --oea-q-blend 1.0 \
  --oea-zo-marginal-mode none --oea-zo-marginal-beta 0.0 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 identity \
  --oea-zo-holdout-fraction 0.3 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 1 \
  --oea-zo-fallback-min-marginal-entropy 0.5 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --run-name 4c_A_q1_margNone
```

