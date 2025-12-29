# 2025-12-29 — LOSO 4-class — pyRiemann MDM + RPA-style transfer baselines

Goal: test a **stronger RPA/TSA-family lower-level alignment** (on SPD covariances) as an alternative to EA→CSP→LDA.

## Setup
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO
- Classes: `left_hand,right_hand,feet,tongue`
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5s, resample 250 Hz

## Methods
- `riemann-mdm`: `MDM(metric="riemann")` on per-trial SPD covariances.
- `rpa-mdm`: `TLCenter(target)+TLStretch(centered_data=True)` then `MDM(metric="riemann")`.
- `rpa-rot-mdm`: center+stretch, then one pseudo-label step + `TLRotate(metric="euclid")`, then `MDM(metric="riemann")`.

Note: this is **not** EA→CSP→LDA; it’s a separate Riemannian pipeline on trial covariances.

## Run
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,riemann-mdm,rpa-mdm,rpa-rot-mdm \
  --no-plots \
  --run-name loso4_riemann_rpa_mdm_baselines
```

Outputs:
- `outputs/20251229/4class/loso4_riemann_rpa_mdm_baselines/20251229_method_comparison.csv`
- `outputs/20251229/4class/loso4_riemann_rpa_mdm_baselines/20251229_results.txt`

## Key numbers
From `20251229_method_comparison.csv`:
- EA mean acc: **0.5320**
- `riemann-mdm` mean acc: **0.3353** (Δ vs EA: **-0.1968**, neg-transfer-rate: **1.0**)
- `rpa-mdm` mean acc: **0.5181** (Δ vs EA: **-0.0139**, neg-transfer-rate: **0.556**)
- `rpa-rot-mdm` mean acc: **0.5305** (Δ vs EA: **-0.0015**, neg-transfer-rate: **0.667**)

Takeaway: on this 4-class LOSO setup, these pyRiemann MDM/RPA-style baselines **do not beat EA→CSP→LDA**; the best (`rpa-rot-mdm`) is close but still slightly worse on average and shows frequent negative transfer vs EA.

