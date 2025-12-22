# 2025-12-22 — OEA-ZO-IM (InfoMax λ) Sweep (BCIIV2a / BNCI2014_001, 2-class)

## Goal
Test whether changing `--oea-zo-infomax-lambda` affects **OEA-ZO-IM** performance under a fixed, paper-aligned configuration, and compare against the **EA CSP+LDA** baseline.

## Fixed configuration
- Dataset: MOABB `BNCI2014_001` (BCI Competition IV 2a)
- Classes: `left_hand` vs `right_hand` (2-class)
- Session(s): `0train`
- Preprocess: `paper_fir` (causal FIR, Hamming, order=50), bandpass 8–30 Hz, resample 250 Hz
- Epoch window: `tmin=0.5s`, `tmax=3.5s` (relative to cue)
- Model: `CSP(n_components=6) + LDA(default)`
- Evaluation: LOSO (9 subjects)

### Alignment methods
- EA: per-subject whitening (`ea-csp-lda`)
- OEA-ZO-IM: source uses Δ-alignment for `Q_s`; target optimizes `Q_t` by SPSA (frozen classifier, unlabeled target data)

### OEA-ZO settings (fixed)
- `--oea-q-blend 0.3`
- `--oea-zo-iters 30`
- `--oea-zo-lr 0.5`
- `--oea-zo-mu 0.1`
- `--oea-zo-k 50`
- `--oea-zo-seed 0`
- `--oea-zo-l2 0.0`
- pseudo settings (unused for infomax): `--oea-pseudo-confidence 0.0 --oea-pseudo-topk-per-class 0` (no filtering)

## Environment versions (from outputs)
- moabb 1.2.0
- braindecode 0.8.1
- mne 1.8.0
- scikit-learn 1.5.2

## Runs

### Run A (λ = 2)
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 --oea-q-blend 0.3 \
  --methods ea-csp-lda,oea-zo-im-csp-lda \
  --oea-zo-infomax-lambda 2 \
  --run-name im_l2
```

Summary (mean across subjects):
- EA accuracy: **0.732253**
- OEA-ZO-IM accuracy: **0.734568**

Raw output (not tracked by git): `outputs/20251222/im_l2/20251222_results.txt`

### Run B (λ = 5)
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 --oea-q-blend 0.3 \
  --methods ea-csp-lda,oea-zo-im-csp-lda \
  --oea-zo-infomax-lambda 5 \
  --run-name im_l5
```

Summary (mean across subjects):
- EA accuracy: **0.732253**
- OEA-ZO-IM accuracy: **0.734568**

Raw output (not tracked by git): `outputs/20251222/im_l5/20251222_results.txt`

## Observation
With this setup, changing `--oea-zo-infomax-lambda` from **2** to **5** produced **no observable change** in per-subject/mean accuracy (only negligible floating-point differences in AUC in the saved text outputs).

