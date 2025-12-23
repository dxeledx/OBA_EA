# 20251223 — Reliable InfoMax + Trust-Region (bilevel-style) sanity check

## Motivation
In cross-subject LOSO MI-EEG, unlabeled test-time objectives (entropy/InfoMax/pseudo-CE) can cause **negative transfer** (over-confidence collapse on some subjects).  
This experiment adds two stabilizers to the existing ZO(Q\_t) pipeline:

1. **Reliable weighting (continuous)**: per-trial weight `w_i∈(0,1)` computed from prediction confidence/entropy via a sigmoid (proxy for “lower-level sample selection”).
2. **Trust-region**: penalize `||Q - Q0||_F^2` to keep the learned orthogonal rotation close to a safe anchor (`Q0=I` or `Q0=Q_Δ`).

We also add an **EA-ZO** variant: train on EA-whitened data (no source-side Q\_s selection), then adapt only Q\_t at test time.

Code changes landed in:
- `csp_lda/evaluation.py` (reliable weighting + trust + `alignment=ea_zo`)
- `run_csp_lda_loso.py` (CLI wiring + EA-ZO methods)

---

## Binary (2-class): `left_hand,right_hand` (MOABB preprocessing, CSP=4)

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess moabb --n-components 4 --oea-q-blend 0.3 \
  --oea-pseudo-iters 0 \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-im-csp-lda,ea-zo-im-csp-lda \
  --oea-zo-objective infomax \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 delta \
  --oea-zo-warm-start delta --oea-zo-warm-iters 1 \
  --oea-zo-holdout-fraction 0.3 \
  --oea-zo-fallback-min-marginal-entropy 0.1 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --run-name bilevel_bin_imr
```

Results file:
- `outputs/20251223/bilevel_bin_imr/20251223_results.txt`

Mean accuracy (across 9 subjects):
- `ea-csp-lda`: **0.698302**
- `oea-csp-lda` (strict LOSO, `--oea-pseudo-iters 0`): **0.709877**
- `oea-zo-im-csp-lda` (InfoMax + reliable weights + trust): **0.711420**
- `ea-zo-im-csp-lda` (EA training + test-time ZO): **0.692901**

Takeaway (2-class):
- In this run, **OEA / OEA-ZO** improve over **EA** by ~**+1.15% / +1.31%** absolute accuracy.
- **EA-ZO** does not help here (no source-side optimistic selection).

---

## 4-class: `left_hand,right_hand,feet,tongue` (paper\_fir, CSP=6)

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --events left_hand,right_hand,feet,tongue \
  --preprocess paper_fir --n-components 6 --oea-q-blend 0.3 \
  --oea-pseudo-iters 0 \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-im-csp-lda,ea-zo-im-csp-lda \
  --oea-zo-objective infomax \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 delta \
  --oea-zo-warm-start delta --oea-zo-warm-iters 1 \
  --oea-zo-holdout-fraction 0.3 \
  --oea-zo-fallback-min-marginal-entropy 0.5 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --run-name bilevel_4c_imr
```

Results file:
- `outputs/20251223/bilevel_4c_imr/20251223_results.txt`

Mean accuracy (across 9 subjects):
- `ea-csp-lda`: **0.532022**
- `ea-zo-im-csp-lda`: **0.533565** (very small +0.15% abs)
- `oea-csp-lda`: **0.516975**
- `oea-zo-im-csp-lda`: (close to OEA, slightly higher on S7; still below EA overall)

Takeaway (4-class):
- **Source-side signature selection (OEA)** still hurts overall (consistent with earlier 4-class ablations).
- **EA-ZO** is **safer** than OEA-ZO for 4-class in this setting, but improvement is marginal.

---

## Notes for next iteration
If we want 4-class gains, the main suspect is still the **multi-class signature choice for OEA** (source-side Q\_s selection).  
A practical next ablation is: keep training as EA (no Δ/scatter selection), and focus on stronger/safer test-time Q\_t objectives (e.g., InfoMax with better weighting/anti-collapse, or class-balanced marginal constraints).

