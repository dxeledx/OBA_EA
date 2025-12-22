# 2025-12-22 — OEA-ZO Stability Probe (q_blend=0.5) (BCIIV2a / BNCI2014_001, 2-class)

## Purpose
Record an experiment where OEA-ZO performs **worse than EA** under a more aggressive/higher-variance ZO configuration, to highlight a *stability/negative-transfer* regime for later analysis.

## Fixed setup
- Dataset: MOABB `BNCI2014_001` (BCI Competition IV 2a)
- Classes: `left_hand` vs `right_hand` (2-class)
- Session(s): `0train`
- Preprocess: `paper_fir` (causal FIR, Hamming, order=50), bandpass 8–30 Hz, resample 250 Hz
- Epoch window: `tmin=0.5s`, `tmax=3.5s`
- Model: `CSP(n_components=6) + LDA(default)`
- Evaluation: LOSO (9 subjects)

## Command
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --methods ea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --oea-q-blend 0.5 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance
```

## Environment versions (from outputs)
- moabb 1.2.0
- braindecode 0.8.1
- mne 1.8.0
- scikit-learn 1.5.2

## Results (mean accuracy across subjects)
From `outputs/20251222/194423/20251222_results.txt` (not tracked by git):

- EA (`ea-csp-lda`): **0.732253**
- OEA-ZO-entropy (`oea-zo-ent-csp-lda`): **0.726080**
- OEA-ZO-infomax (`oea-zo-im-csp-lda`, λ=1): **0.726080**
- OEA-ZO-pCE (`oea-zo-pce-csp-lda`): **0.726080**

Absolute gap vs EA: **-0.006173** (≈ **-0.62%**).

### Notable per-subject drops (EA → OEA-ZO-entropy)
- Subject 5: `0.562500 → 0.486111` (**-0.076389**)
- Subject 9: `0.791667 → 0.743056` (**-0.048611**)

## Observation
This run shows **mixed per-subject behavior**: some subjects improve, while a few subjects degrade substantially, and the mean ends up below the EA baseline. This is a concrete example of the “has potential but unstable” regime for test-time Q selection under this configuration.

