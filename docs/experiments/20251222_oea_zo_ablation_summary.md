# 2025-12-22 — OEA / OEA-ZO Ablations (LOSO, BNCI2014_001, 2-class)

This note summarizes the ablations requested in the discussion:

- **Priority 0 (no code change):** sweep `--oea-q-blend` back to `0.3/0.2`, and add `oea-csp-lda` with `--oea-pseudo-iters 0` as an intermediate point to separate “source Δ-alignment” vs “target Q_t selection”.
- **Priority 1/2 (code change):** add (a) “reliable trial keep” for entropy/infomax/confidence objectives, (b) unlabeled holdout for best-iterate selection, (c) warm-start from pseudo-Δ alignment, (d) an unlabeled safety fallback (class-marginal entropy).

## Environment / common setup
- Dataset: MOABB `BNCI2014_001` (BCI Competition IV 2a)
- Classes: `left_hand` vs `right_hand` (2-class)
- Session(s): `0train`
- Evaluation: LOSO (9 subjects)
- Metrics: mean accuracy reported below (full metrics saved in each run folder)

## Key takeaway (what the numbers show)
1. **`--oea-q-blend` is a critical knob** because it affects *both* (a) source-subject alignment used to train CSP+LDA and (b) target-subject alignment at test time.  
   In these runs, `q_blend=0.5` consistently hurts compared to `0.2–0.3`.
2. Adding `oea-csp-lda` with `--oea-pseudo-iters 0` makes it clear that **a lot of the gain comes from source-side Δ-alignment alone**, while **target-side ZO adds only small extra gains** (sometimes none).
3. The “stability options” (keep/holdout/warm-start/fallback) were implemented and tested, but under these tested settings they were **often conservative** and did **not reliably improve the mean** over the simpler `q_blend=0.2–0.3` baseline.

---

## A) “CSP=4 + non-causal preprocessing (MOABB)” run (the 69% vs 72% comparison)
This is the setup you mentioned: **EA ≈ 69%**, **ours ≈ 72%**.

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess moabb --n-components 4 \
  --oea-pseudo-iters 0 \
  --methods csp-lda,ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-pce-csp-lda \
  --run-name moabb_csp4_ablation
```

Mean accuracy (from `outputs/20251222/moabb_csp4_ablation/20251222_results.txt`):
- `csp-lda`: **0.690586**
- `ea-csp-lda`: **0.698302**
- `oea-csp-lda` (`pseudo-iters=0`, source Δ only): **0.723765**
- `oea-zo-ent-csp-lda`: **0.722222**
- `oea-zo-pce-csp-lda`: **0.723765**

---

## B) Priority 0 ablation on paper-aligned preprocessing (paper_fir, CSP=6)

### B1) `q_blend=0.3` + include `oea-csp-lda --oea-pseudo-iters 0`
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-q-blend 0.3 --oea-pseudo-iters 0 \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name p0_blend03
```

Mean accuracy (from `outputs/20251222/p0_blend03/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.734568**
- `oea-zo-ent-csp-lda`: **0.733796**
- `oea-zo-im-csp-lda`: **0.733796**
- `oea-zo-pce-csp-lda`: **0.735340**

### B2) `q_blend=0.2` + include `oea-csp-lda --oea-pseudo-iters 0`
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-q-blend 0.2 --oea-pseudo-iters 0 \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name p0_blend02
```

Mean accuracy (from `outputs/20251222/p0_blend02/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.733025**
- `oea-zo-ent-csp-lda`: **0.734568**
- `oea-zo-im-csp-lda`: **0.734568**
- `oea-zo-pce-csp-lda`: **0.733796**

---

## C) “Unstable regime” example (`q_blend=0.5`)

### C1) Without holdout/warm-start/fallback (keep selection is active via pseudo args)
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.5 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name stab_qb05_keep
```

Mean accuracy (from `outputs/20251222/stab_qb05_keep/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.724537**
- `oea-zo-ent-csp-lda`: **0.726080**
- `oea-zo-im-csp-lda`: **0.726080**
- `oea-zo-pce-csp-lda`: **0.726080**

### C2) With holdout + warm-start(Δ) + fallback enabled
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.5 \
  --oea-zo-holdout-fraction 0.5 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 2 \
  --oea-zo-fallback-min-marginal-entropy 0.2 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name improved_qb05
```

Mean accuracy (from `outputs/20251222/improved_qb05/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.724537**
- `oea-zo-ent-csp-lda`: **0.716049**
- `oea-zo-im-csp-lda`: **0.716821**
- `oea-zo-pce-csp-lda`: **0.719136**

---

## D) Priority 1/2 options tested in a “good” blend regime (`q_blend=0.3`)

### D1) Aggressive ZO hyperparameters + stability options
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.3 \
  --oea-zo-holdout-fraction 0.5 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 2 \
  --oea-zo-fallback-min-marginal-entropy 0.2 \
  --oea-zo-iters 100 --oea-zo-k 20 --oea-zo-lr 0.3 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name improved_qb03
```

Mean accuracy (from `outputs/20251222/improved_qb03/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.734568**
- `oea-zo-ent-csp-lda`: **0.734568**
- `oea-zo-im-csp-lda`: **0.734568**
- `oea-zo-pce-csp-lda`: **0.732253**

### D2) Default ZO hyperparameters + stability options
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --oea-pseudo-iters 0 --oea-q-blend 0.3 \
  --oea-zo-holdout-fraction 0.5 \
  --oea-zo-warm-start delta --oea-zo-warm-iters 2 \
  --oea-zo-fallback-min-marginal-entropy 0.2 \
  --oea-pseudo-confidence 0.7 --oea-pseudo-topk-per-class 40 --oea-pseudo-balance \
  --methods ea-csp-lda,oea-csp-lda,oea-zo-ent-csp-lda,oea-zo-im-csp-lda,oea-zo-pce-csp-lda \
  --run-name improved_defaultzo
```

Mean accuracy (from `outputs/20251222/improved_defaultzo/20251222_results.txt`):
- `ea-csp-lda`: **0.732253**
- `oea-csp-lda` (`pseudo-iters=0`): **0.734568**
- `oea-zo-ent-csp-lda`: **0.732253**
- `oea-zo-im-csp-lda`: **0.732253**
- `oea-zo-pce-csp-lda`: **0.733025**

---

## Appendix: Per-subject metrics (tables)
To keep the main note readable, only the most representative runs are expanded here:
- The “**69% vs 72%**” comparison (MOABB preprocess, CSP=4).
- A typical “good” `q_blend=0.3` run (paper_fir, CSP=6).
- The “unstable regime” run (`q_blend=0.5`) where some subjects collapse.

### Run A — `moabb_csp4_ablation`
<details>
<summary>Per-subject metrics (LOSO)</summary>


**csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.743056   0.782828 0.743056 0.733693 0.883102  0.486111
       2     1152     144  0.541667   0.566502 0.541667 0.494468 0.613040  0.083333
       3     1152     144  0.916667   0.921875 0.916667 0.916409 0.982832  0.833333
       4     1152     144  0.638889   0.642857 0.638889 0.636364 0.732446  0.277778
       5     1152     144  0.479167   0.465812 0.479167 0.422799 0.473187 -0.041667
       6     1152     144  0.562500   0.721463 0.562500 0.466823 0.698495  0.125000
       7     1152     144  0.625000   0.647273 0.625000 0.610265 0.663387  0.250000
       8     1152     144  0.916667   0.921875 0.916667 0.916409 0.982832  0.833333
       9     1152     144  0.791667   0.811111 0.791667 0.788360 0.841628  0.583333
```

**ea-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.604167   0.605820 0.604167 0.602614 0.678434 0.208333
       2     1152     144  0.562500   0.562512 0.562500 0.562479 0.606867 0.125000
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.975694 0.833333
       4     1152     144  0.715278   0.715319 0.715278 0.715264 0.797840 0.430556
       5     1152     144  0.500000   0.500000 0.500000 0.485714 0.570216 0.000000
       6     1152     144  0.687500   0.689289 0.687500 0.686760 0.772762 0.375000
       7     1152     144  0.652778   0.652896 0.652778 0.652711 0.733218 0.305556
       8     1152     144  0.826389   0.831570 0.826389 0.825708 0.893711 0.652778
       9     1152     144  0.819444   0.821678 0.819444 0.819130 0.880980 0.638889
```

**oea-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.833333   0.839890 0.833333 0.832526 0.924961 0.666667
       2     1152     144  0.569444   0.569444 0.569444 0.569444 0.596451 0.138889
       3     1152     144  0.944444   0.944788 0.944444 0.944434 0.987461 0.888889
       4     1152     144  0.729167   0.729211 0.729167 0.729154 0.808256 0.458333
       5     1152     144  0.527778   0.533613 0.527778 0.506352 0.562500 0.055556
       6     1152     144  0.673611   0.676367 0.673611 0.672331 0.770255 0.347222
       7     1152     144  0.638889   0.639319 0.638889 0.638610 0.733218 0.277778
       8     1152     144  0.888889   0.890093 0.888889 0.888803 0.942515 0.777778
       9     1152     144  0.708333   0.709790 0.708333 0.707826 0.787230 0.416667
```

**oea-zo-ent-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.840278   0.845679 0.840278 0.839651 0.927469 0.680556
       2     1152     144  0.562500   0.562512 0.562500 0.562479 0.604745 0.125000
       3     1152     144  0.944444   0.944788 0.944444 0.944434 0.988619 0.888889
       4     1152     144  0.729167   0.729211 0.729167 0.729154 0.814043 0.458333
       5     1152     144  0.500000   0.500000 0.500000 0.470480 0.572531 0.000000
       6     1152     144  0.659722   0.663539 0.659722 0.657725 0.775463 0.319444
       7     1152     144  0.652778   0.652896 0.652778 0.652711 0.728781 0.305556
       8     1152     144  0.902778   0.903089 0.902778 0.902759 0.963349 0.805556
       9     1152     144  0.708333   0.709790 0.708333 0.707826 0.795332 0.416667
```

**oea-zo-pce-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.833333   0.839890 0.833333 0.832526 0.927855 0.666667
       2     1152     144  0.569444   0.569444 0.569444 0.569444 0.600116 0.138889
       3     1152     144  0.937500   0.937584 0.937500 0.937497 0.987654 0.875000
       4     1152     144  0.729167   0.729211 0.729167 0.729154 0.810957 0.458333
       5     1152     144  0.520833   0.525574 0.520833 0.497547 0.569444 0.041667
       6     1152     144  0.666667   0.669945 0.666667 0.665051 0.775270 0.333333
       7     1152     144  0.652778   0.652896 0.652778 0.652711 0.730903 0.305556
       8     1152     144  0.888889   0.890093 0.888889 0.888803 0.951775 0.777778
       9     1152     144  0.715278   0.716321 0.715278 0.714934 0.790123 0.430556
```

</details>

### Run B1 — `p0_blend03`
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

**oea-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.861111   0.868214 0.861111 0.860438 0.921489 0.722222
       2     1152     144  0.534722   0.534729 0.534722 0.534700 0.558063 0.069444
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.987269 0.833333
       4     1152     144  0.722222   0.722222 0.722222 0.722222 0.809028 0.444444
       5     1152     144  0.527778   0.530100 0.527778 0.518489 0.562500 0.055556
       6     1152     144  0.687500   0.689289 0.687500 0.686760 0.774498 0.375000
       7     1152     144  0.659722   0.660496 0.659722 0.659311 0.741319 0.319444
       8     1152     144  0.902778   0.905594 0.902778 0.902609 0.961227 0.805556
       9     1152     144  0.798611   0.799130 0.798611 0.798524 0.880015 0.597222
```

**oea-zo-ent-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.861111   0.868214 0.861111 0.860438 0.922261 0.722222
       2     1152     144  0.534722   0.534729 0.534722 0.534700 0.559221 0.069444
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.987654 0.833333
       4     1152     144  0.722222   0.722222 0.722222 0.722222 0.809799 0.444444
       5     1152     144  0.527778   0.530100 0.527778 0.518489 0.562693 0.055556
       6     1152     144  0.687500   0.689289 0.687500 0.686760 0.774884 0.375000
       7     1152     144  0.659722   0.660496 0.659722 0.659311 0.740355 0.319444
       8     1152     144  0.902778   0.905594 0.902778 0.902609 0.961613 0.805556
       9     1152     144  0.791667   0.792570 0.791667 0.791506 0.880594 0.583333
```

**oea-zo-im-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.861111   0.868214 0.861111 0.860438 0.921875 0.722222
       2     1152     144  0.534722   0.534729 0.534722 0.534700 0.559221 0.069444
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.987654 0.833333
       4     1152     144  0.722222   0.722222 0.722222 0.722222 0.809799 0.444444
       5     1152     144  0.527778   0.530100 0.527778 0.518489 0.562307 0.055556
       6     1152     144  0.687500   0.689289 0.687500 0.686760 0.774691 0.375000
       7     1152     144  0.659722   0.660496 0.659722 0.659311 0.740355 0.319444
       8     1152     144  0.902778   0.905594 0.902778 0.902609 0.961613 0.805556
       9     1152     144  0.791667   0.792570 0.791667 0.791506 0.880594 0.583333
```

**oea-zo-pce-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc    kappa
       1     1152     144  0.861111   0.868214 0.861111 0.860438 0.921682 0.722222
       2     1152     144  0.534722   0.534729 0.534722 0.534700 0.558642 0.069444
       3     1152     144  0.916667   0.916988 0.916667 0.916651 0.987461 0.833333
       4     1152     144  0.722222   0.722222 0.722222 0.722222 0.809992 0.444444
       5     1152     144  0.527778   0.530100 0.527778 0.518489 0.562307 0.055556
       6     1152     144  0.687500   0.689289 0.687500 0.686760 0.775077 0.375000
       7     1152     144  0.666667   0.667183 0.666667 0.666409 0.740741 0.333333
       8     1152     144  0.902778   0.905594 0.902778 0.902609 0.961806 0.805556
       9     1152     144  0.798611   0.799130 0.798611 0.798524 0.880015 0.597222
```

</details>

### Run C1 — `stab_qb05_keep`
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

**oea-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.826389   0.829503 0.826389 0.825978 0.913002  0.652778
       2     1152     144  0.555556   0.555598 0.555556 0.555470 0.593557  0.111111
       3     1152     144  0.930556   0.930556 0.930556 0.930556 0.988040  0.861111
       4     1152     144  0.743056   0.743102 0.743056 0.743043 0.816937  0.486111
       5     1152     144  0.479167   0.476311 0.479167 0.462980 0.546682 -0.041667
       6     1152     144  0.687500   0.687826 0.687500 0.687364 0.767168  0.375000
       7     1152     144  0.659722   0.661246 0.659722 0.658916 0.739776  0.319444
       8     1152     144  0.895833   0.897752 0.895833 0.895708 0.952739  0.791667
       9     1152     144  0.743056   0.743478 0.743056 0.742944 0.850309  0.486111
```

**oea-zo-ent-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.826389   0.829503 0.826389 0.825978 0.911265  0.652778
       2     1152     144  0.548611   0.548696 0.548611 0.548415 0.592400  0.097222
       3     1152     144  0.930556   0.930556 0.930556 0.930556 0.988426  0.861111
       4     1152     144  0.750000   0.750193 0.750000 0.749952 0.817323  0.500000
       5     1152     144  0.486111   0.484375 0.486111 0.471429 0.547068 -0.027778
       6     1152     144  0.687500   0.687826 0.687500 0.687364 0.766782  0.375000
       7     1152     144  0.666667   0.667832 0.666667 0.666087 0.740934  0.333333
       8     1152     144  0.895833   0.897752 0.895833 0.895708 0.958526  0.791667
       9     1152     144  0.743056   0.743478 0.743056 0.742944 0.850694  0.486111
```

**oea-zo-im-csp-lda**
```text
 subject  n_train  n_test  accuracy  precision   recall       f1      auc     kappa
       1     1152     144  0.826389   0.829503 0.826389 0.825978 0.911458  0.652778
       2     1152     144  0.548611   0.548696 0.548611 0.548415 0.593750  0.097222
       3     1152     144  0.930556   0.930556 0.930556 0.930556 0.988426  0.861111
       4     1152     144  0.750000   0.750193 0.750000 0.749952 0.817323  0.500000
       5     1152     144  0.486111   0.484375 0.486111 0.471429 0.546875 -0.027778
       6     1152     144  0.687500   0.687826 0.687500 0.687364 0.766782  0.375000
       7     1152     144  0.666667   0.667832 0.666667 0.666087 0.740162  0.333333
       8     1152     144  0.895833   0.897752 0.895833 0.895708 0.958526  0.791667
       9     1152     144  0.743056   0.743478 0.743056 0.742944 0.850694  0.486111
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
