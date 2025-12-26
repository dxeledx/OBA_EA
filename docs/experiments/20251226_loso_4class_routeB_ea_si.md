# 2025-12-26 — LOSO 4-class (BCI IV 2a) — Route B: EA-SI (+ optional ZO)

## Goal
Validate **Route B** on **4-class LOSO**:

- **EA** baseline: per-subject Euclidean Alignment (EA) whitening → CSP → LDA
- **EA-SI**: EA whitening → CSP → **subject-invariant linear projector** (HSIC-style) → LDA
- **EA-ZO-IMR**: EA whitening → CSP → LDA, then **test-time** ZO optimizes `Q_t` (objective = `infomax_bilevel`)
- **EA-SI-ZO-IMR**: EA-SI training, then test-time ZO optimizes `Q_t`

> Note: to support EA-SI-ZO, ZO now applies an optional `proj` step (if present) between CSP features and LDA.

## Dataset / Protocol
- Dataset: MOABB `BNCI2014_001` (BCI Competition IV 2a)
- Protocol: **LOSO** (each subject held out once)
- Task: **4-class** `left_hand,right_hand,feet,tongue`
- Sessions: `0train`

## Preprocess / Model
- Preprocess: `paper_fir` (causal FIR order=50, bandpass 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Route B params (EA-SI)
- `--si-subject-lambda 1`
- `--si-ridge 1e-6`
- `--si-proj-dim 2`

## ZO params (IMR)
- `--oea-zo-iters 30 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05`
- `--oea-q-blend 0.3`
- Objective: `infomax_bilevel` (via `*-zo-imr-*` method name override)

---

## Run 1 (seed=0)

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --oea-q-blend 0.3 \
  --oea-zo-iters 30 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --methods ea-csp-lda,ea-zo-imr-csp-lda,ea-si-csp-lda,ea-si-zo-imr-csp-lda \
  --si-subject-lambda 1 --si-ridge 1e-6 --si-proj-dim 2 \
  --run-name loso4_easi_zo_imr
```

Outputs:
- `outputs/20251226/4class/loso4_easi_zo_imr/20251226_results.txt`
- Per-trial predictions:
  - `outputs/20251226/4class/loso4_easi_zo_imr/20251226_ea-csp-lda_predictions.csv`
  - `outputs/20251226/4class/loso4_easi_zo_imr/20251226_ea-zo-imr-csp-lda_predictions.csv`
  - `outputs/20251226/4class/loso4_easi_zo_imr/20251226_ea-si-csp-lda_predictions.csv`
  - `outputs/20251226/4class/loso4_easi_zo_imr/20251226_ea-si-zo-imr-csp-lda_predictions.csv`

Mean accuracy (across subjects):
- EA: **0.532022**
- EA-SI: **0.532022** (same as EA)
- EA-ZO-IMR: **0.532407**
- EA-SI-ZO-IMR: **0.532407**

## Run 2 (seed=1)

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --oea-q-blend 0.3 \
  --oea-zo-iters 30 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 \
  --oea-zo-seed 1 \
  --methods ea-csp-lda,ea-zo-imr-csp-lda,ea-si-zo-imr-csp-lda \
  --si-subject-lambda 1 --si-ridge 1e-6 --si-proj-dim 2 \
  --run-name loso4_easi_zo_imr_s1
```

Outputs:
- `outputs/20251226/4class/loso4_easi_zo_imr_s1/20251226_results.txt`

Mean accuracy (across subjects):
- EA: **0.532022**
- EA-ZO-IMR: **0.532022**
- EA-SI-ZO-IMR: **0.531636**

---

## Notes / Observations (current)
- **EA-SI == EA** in these runs (per-subject metrics identical). With full-rank linear transforms, LDA can be invariant up to reparameterization; with low-rank projection, the classifier can still end up unchanged if the discarded subspace is not used. This suggests EA-SI (as currently instantiated) is not providing measurable benefit on this configuration.
- **ZO-IMR gain is tiny and seed-sensitive** here (≈ ±0.04% abs at mean level).

## Where are 2-class / 4-class outputs?
- `outputs/YYYYMMDD/2class/...`
- `outputs/YYYYMMDD/4class/...`

