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

## Per-subject metrics (tables)
From `outputs/20251222/194423/20251222_results.txt`:

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

**oea-zo-ent-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.833333   0.837500 0.833333 0.832817 0.916474  0.666667
       2     1152     144  0.555556   0.555598 0.555556 0.555470 0.594329  0.111111
       3     1152     144  0.930556   0.930556 0.930556 0.930556 0.988619  0.861111
       4     1152     144  0.743056   0.743102 0.743056 0.743043 0.818094  0.486111
       5     1152     144  0.486111   0.484375 0.486111 0.471429 0.546489 -0.027778
       6     1152     144  0.687500   0.687826 0.687500 0.687364 0.766975  0.375000
       7     1152     144  0.659722   0.661246 0.659722 0.658916 0.737076  0.319444
       8     1152     144  0.895833   0.897752 0.895833 0.895708 0.958526  0.791667
       9     1152     144  0.743056   0.743478 0.743056 0.742944 0.850309  0.486111
```

**oea-zo-im-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.833333   0.837500 0.833333 0.832817 0.915895  0.666667
       2     1152     144  0.555556   0.555598 0.555556 0.555470 0.594329  0.111111
       3     1152     144  0.930556   0.930556 0.930556 0.930556 0.988619  0.861111
       4     1152     144  0.743056   0.743102 0.743056 0.743043 0.818094  0.486111
       5     1152     144  0.486111   0.484375 0.486111 0.471429 0.546682 -0.027778
       6     1152     144  0.687500   0.687826 0.687500 0.687364 0.766590  0.375000
       7     1152     144  0.659722   0.661246 0.659722 0.658916 0.737076  0.319444
       8     1152     144  0.895833   0.897752 0.895833 0.895708 0.958526  0.791667
       9     1152     144  0.743056   0.743478 0.743056 0.742944 0.850309  0.486111
```

**oea-zo-pce-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.826389   0.829503 0.826389 0.825978 0.912230  0.652778
       2     1152     144  0.555556   0.555598 0.555556 0.555470 0.592785  0.111111
       3     1152     144  0.930556   0.930556 0.930556 0.930556 0.988233  0.861111
       4     1152     144  0.750000   0.750193 0.750000 0.749952 0.817901  0.500000
       5     1152     144  0.486111   0.484375 0.486111 0.471429 0.546489 -0.027778
       6     1152     144  0.687500   0.687826 0.687500 0.687364 0.766975  0.375000
       7     1152     144  0.659722   0.661246 0.659722 0.658916 0.739198  0.319444
       8     1152     144  0.895833   0.897752 0.895833 0.895708 0.955440  0.791667
       9     1152     144  0.743056   0.743478 0.743056 0.742944 0.850502  0.486111
```

</details>
