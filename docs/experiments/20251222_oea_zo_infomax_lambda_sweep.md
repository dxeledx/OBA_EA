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

## Per-subject metrics (tables)
The per-subject tables for **Run A (λ=2)** are included below. Run B (λ=5) was identical under this setup, so it is omitted.

<details>
<summary>Per-subject metrics (LOSO)</summary>


**ea-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.847222   0.847490 0.847222 0.847193 0.911651 0.694444
       2     1152     144  0.520833   0.520870 0.520833 0.520625 0.563079 0.041667
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.983410 0.833333
       4     1152     144  0.701389   0.701739 0.701389 0.701259 0.813657 0.402778
       5     1152     144  0.562500   0.568311 0.562500 0.552993 0.573688 0.125000
       6     1152     144  0.680556   0.685714 0.680556 0.678322 0.775463 0.361111
       7     1152     144  0.659722   0.660496 0.659722 0.659311 0.746335 0.319444
       8     1152     144  0.909722   0.910435 0.909722 0.909683 0.957176 0.819444
       9     1152     144  0.791667   0.793706 0.791667 0.791304 0.892361 0.583333
```

**oea-zo-im-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.861111   0.868214 0.861111 0.860438 0.921682 0.722222
       2     1152     144  0.534722   0.534729 0.534722 0.534700 0.559221 0.069444
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.987654 0.833333
       4     1152     144  0.722222   0.722222 0.722222 0.722222 0.809799 0.444444
       5     1152     144  0.527778   0.530100 0.527778 0.518489 0.562307 0.055556
       6     1152     144  0.687500   0.689289 0.687500 0.686760 0.774691 0.375000
       7     1152     144  0.659722   0.660496 0.659722 0.659311 0.740355 0.319444
       8     1152     144  0.902778   0.905594 0.902778 0.902609 0.961613 0.805556
       9     1152     144  0.798611   0.799130 0.798611 0.798524 0.880208 0.597222
```

</details>
