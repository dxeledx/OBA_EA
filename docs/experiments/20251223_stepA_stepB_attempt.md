# 20251223 — Step A/Step B attempt (safety selection + multiclass signature)

This note documents implementing the next actions you proposed:

## Step A (safety): explicit candidate-set selection with EA anchor
Changes (code):
- `csp_lda/evaluation.py`: `_optimize_qt_oea_zo` now **explicitly evaluates candidates on the holdout set**:
  - `Q=I` (EA anchor)
  - `Q=Q_Δ` (if available from warm-start)
  - all ZO iterates `{Q^(t)}`
- Adds `--oea-zo-min-improvement`: require a minimum holdout-objective improvement over identity; otherwise keep `Q=I`.
- If `Q_Δ` is computed, it is now evaluated **as an exact candidate** (not only the Givens approximation).

CLI:
- `run_csp_lda_loso.py`: added `--oea-zo-min-improvement` and wires it into evaluation.

## Step B (4-class): replace multiclass signature with an LDA-inspired operator
Changes (code):
- `csp_lda/alignment.py`: multiclass signature now computes:
  - `S_w := Σ̄ = Σ_k π_k Σ_k`
  - `S_b := Σ_k π_k (Σ_k-Σ̄)(Σ_k-Σ̄)`
  - `M := S_w^{-1/2} S_b S_w^{-1/2}`
- `csp_lda/evaluation.py`: `_soft_class_cov_diff` multiclass branch updated consistently.

Important observation:
- In the current pipeline, `class_cov_diff()` is computed on **EA-whitened data** (`z = C_s^{-1/2} X`), where the overall covariance is (approximately) `I`.
- Therefore, the multiclass `S_w` above is also (approximately) `I`, so `M ≈ S_b`, i.e., **Step B becomes (nearly) equivalent to the previous scatter signature** under EA whitening.
- This explains why Step B did not produce a meaningful change in the 4-class results below.

---

## 4-class run (paper_fir, CSP=6)

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
  --oea-zo-min-improvement 0.001 \
  --oea-zo-iters 50 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --run-name stepA_stepB_4c
```

Results file:
- `outputs/20251223/stepA_stepB_4c/20251223_results.txt`

Qualitative outcome:
- `ea-csp-lda` remains the strongest overall in this configuration.
- `oea-csp-lda` is still below EA (source-side multiclass selection still hurts).
- `oea-zo-im-csp-lda` does not recover the loss (slightly improves some subjects, hurts others).
- `ea-zo-im-csp-lda` stays close to EA (safe, but small effect).

---

## Next adjustment to make Step B “non-trivial”
If we want an LDA-style multiclass signature to actually change behavior under the EA-solution-set framework, `S_w` must **not collapse to I** after EA whitening. Two options:

1) Define `S_w` in a different statistic space (not the overall covariance), e.g. within-class **trial-level covariance dispersion** (4th-order), or
2) Compute the signature in raw space and map it into the EA-whitened coordinates in a way that keeps a non-identity `S_w`.

