# 2025-12-29 — LOSO 4-class — LocalMix-IMR (no electrode coordinates)

Goal: test a physiology-motivated **local channel mixing** transform (to model electrode shift) on a public dataset **without subject-specific electrode coordinates**, and evaluate whether a **safe calibrated guard** can make the unlabeled selection reliable.

## Motivation (why local mixing is physiologically plausible)
EEG scalp potentials are spatially smooth due to volume conduction. Small electrode placement shifts can be approximated as a *local spatial interpolation* of nearby channels, rather than an arbitrary global mixing. Public datasets like BCI IV-2a do not provide per-subject electrode coordinates, so we impose locality using only channel names + a template montage.

## Method: `ea-zo-imr-csp-lda` with `transform=local_mix`
Pipeline (4-class LOSO):
1) **EA whitening** per subject (unsupervised).
2) Train frozen **CSP+LDA** on source subjects (EA space).
3) On target subject (unlabeled), optimize a **row-stochastic local mixing** matrix `A`:
   - Each row mixes **self + k neighbors** (kNN on `standard_1020` template montage).
   - Non-negative row weights with sum=1 (softmax), initialized near identity via self-bias.
4) Optimize `A` using **ZO/SPSA** on an **InfoMax-R (bilevel) objective** (`infomax_bilevel`).
5) Select the final candidate using a **calibrated guard** (`selector=calibrated_guard`) to reject likely negative transfer (fallback to EA anchor).

## Setup
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Classes: `left_hand,right_hand,feet,tongue`
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Experiments

### A) Oracle headroom (analysis-only, uses labels)
Purpose: verify the candidate set contains any improvement.

`k=8`, `iters=50`:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform local_mix \
  --oea-zo-localmix-neighbors 8 --oea-zo-localmix-self-bias 3.0 \
  --oea-zo-iters 50 --oea-zo-lr 0.2 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-beta 0 --oea-zo-marginal-prior anchor_pred \
  --oea-zo-selector oracle \
  --no-plots \
  --run-name loso4_localmix_imr_oracle_k8
```
Outputs:
- `outputs/20251229/4class/loso4_localmix_imr_oracle_k8/20251229_method_comparison.csv`

Key numbers:
- EA mean acc: `0.5320`
- Oracle (best-by-acc among candidates) mean acc: `0.5367` (**+0.46% abs**)

### B) Objective selection (unlabeled, but can be unsafe)
Purpose: check if unlabeled IMR objective alone is reliable.

`k=8`, `selector=objective`:
Outputs:
- `outputs/20251229/4class/loso4_localmix_imr_objective_k8/20251229_method_comparison.csv`

Observed:
- Mean acc drops slightly to `0.5313` and **negative transfer rate ~0.44** vs EA → unlabeled objective alone is not safe at this freedom level.

### C) Safe selection via calibrated guard (unlabeled + fallback)
Purpose: make selection reliable (avoid negative transfer) while capturing oracle headroom.

`k=8`, `selector=calibrated_guard`:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform local_mix \
  --oea-zo-localmix-neighbors 8 --oea-zo-localmix-self-bias 3.0 \
  --oea-zo-iters 30 --oea-zo-lr 0.2 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-beta 0 --oea-zo-marginal-prior anchor_pred \
  --oea-zo-selector calibrated_guard --oea-zo-calib-guard-threshold 0.5 \
  --no-plots \
  --run-name loso4_localmix_imr_calguard_k8
```
Outputs:
- `outputs/20251229/4class/loso4_localmix_imr_calguard_k8/20251229_method_comparison.csv`

Key numbers:
- EA mean acc: `0.5320`, worst-subject: `0.2569`
- LocalMix-IMR (calibrated_guard) mean acc: `0.5359` (**+0.39% abs**)
- Worst-subject: `0.2604`
- Negative transfer rate vs EA: `0.0`

### D) Removing EA whitening (RAW-ZO) — fails badly
Purpose: test the physiology-motivated idea “local electrode remapping should replace EA whitening”.

Same transform/objective as (C), but train the classifier on **raw** signals (no EA whitening), and adapt the target with LocalMix:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,raw-zo-imr-csp-lda \
  --oea-zo-transform local_mix \
  --oea-zo-localmix-neighbors 8 --oea-zo-localmix-self-bias 3.0 \
  --oea-zo-iters 15 --oea-zo-lr 0.2 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-beta 0 --oea-zo-marginal-prior anchor_pred \
  --oea-zo-selector calibrated_guard --oea-zo-calib-guard-threshold 0.5 --oea-zo-calib-max-subjects 3 \
  --no-plots \
  --run-name loso4_rawzo_localmix_imr_calguard_k8_cal3_i15
```
Outputs:
- `outputs/20251229/4class/loso4_rawzo_localmix_imr_calguard_k8_cal3_i15/20251229_method_comparison.csv`

Observed:
- `raw-zo-imr-csp-lda` mean acc: `0.4101` (**-12.19% abs vs EA**), negative transfer rate: `1.0`

Interpretation: on this dataset/protocol, EA whitening is not just a “coordinate change”, but a crucial per-subject normalization/preconditioning step. A local remapping alone (even if physiologically motivated) does **not** replace it.

### E) Moving EA whitening after LocalMix (LocalMix→EA) — no gain (falls back to EA)
Purpose: test whether “electrode remapping” (LocalMix) can be treated as a *pre*-whitening physical correction, with EA applied afterwards.

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform local_mix_then_ea \
  --oea-zo-localmix-neighbors 8 --oea-zo-localmix-self-bias 3.0 \
  --oea-zo-iters 15 --oea-zo-lr 0.2 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-beta 0 --oea-zo-marginal-prior anchor_pred \
  --oea-zo-selector calibrated_guard --oea-zo-calib-guard-threshold 0.5 --oea-zo-calib-max-subjects 3 \
  --no-plots \
  --run-name loso4_eazo_localmix_then_ea_imr_calguard_k8_cal3_i15
```

Outputs:
- `outputs/20251230/4class/loso4_eazo_localmix_then_ea_imr_calguard_k8_cal3_i15_01/20251230_method_comparison.csv`

Observed:
- Mean acc is **exactly** the EA baseline (`0.5320`, Δ=0.0), and neg-transfer rate `0.0`.

Interpretation: in this setting, LocalMix→EA does not provide a selectable improvement over EA (the safe selector effectively keeps the EA anchor).

## Reference comparison (best existing method in this repo)
- `EA-SI-CHAN-MULTI-SAFE` (rank=21, λ={0.5,1,2}, selector=calibrated_ridge_guard) achieves mean acc `0.5463` (+1.43% abs vs EA) on the same setup; see:
  - `docs/experiments/20251227_loso_4class_ea_si_chan_multi_safe.md`

## Takeaway
- The **local-mixing family** has measurable oracle headroom on 4-class LOSO without needing electrode coordinates.
- **Unlabeled objective selection can be unreliable** when the transform family becomes more expressive (k larger).
- A **calibrated guard + fallback** recovers a stable gain (**+0.39% abs, 0 negative transfer**) but still trails the current strongest baseline (`EA-SI-CHAN-MULTI-SAFE`).
