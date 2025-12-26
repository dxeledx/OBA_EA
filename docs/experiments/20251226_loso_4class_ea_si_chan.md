# 2025-12-26 — LOSO 4-class — EA-SI-CHAN (channel-space subject-invariant projection)

## Motivation (why the previous EA-SI had no effect)
The earlier **EA-SI** variant learned a **linear projector in CSP feature space** and then trained **LDA** on the projected features.  
For CSP→LDA pipelines, re-training LDA makes the classifier largely **invariant to invertible linear feature transforms**, so EA-SI could become a pure reparameterization and produce **identical predictions**.

This experiment switches to **channel-space (signal-space) projection before CSP**, so the signal is constrained to a (rank-deficient) subspace and cannot be cancelled by the downstream classifier.

## Method (EA-SI-CHAN)
Pipeline:
- EA whitening (per subject) → **A·X** (rank-r channel projector) → CSP → LDA

Projector learning (on training subjects within each LOSO fold):
- Build class between-scatter `B` from class-conditional mean covariances
- Build subject scatter `S` from deviations of subject-wise class covariances from the global class covariances
- Solve generalized eigenproblem to get a discriminative-but-stable subspace, then form a rank-r projector `A = QQᵀ`

## Setup
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Classes: `left_hand,right_hand,feet,tongue`
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Commands / Results

### Baseline vs EA-SI-CHAN (rank=12) — too aggressive (worse)
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-csp-lda \
  --si-proj-dim 12 --si-subject-lambda 1 --si-ridge 1e-6 \
  --run-name loso4_easichan_r12
```
Outputs:
- `outputs/20251226/4class/loso4_easichan_r12/20251226_results.txt`

Observation:
- Mean accuracy dropped (projection rank too small).

### Baseline vs EA-SI-CHAN (rank=21) — best so far
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-si-chan-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --run-name loso4_easichan_r21
```
Outputs:
- `outputs/20251226/4class/loso4_easichan_r21/20251226_results.txt`

Key numbers (mean accuracy across subjects):
- EA: **0.532022**
- EA-SI-CHAN (rank=21): **0.539352**  (**+0.733% abs**)

Notes:
- Gains are not uniform across subjects (some negative transfer remains).

