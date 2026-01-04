# 20260105 — BNCI2014_001 4-class LOSO: candidate-level certificate correlation (diagall, v1)

## Goal (this iteration)
After stabilizing **EA-STACK-MULTI-SAFE** with high-risk gates (FBCSP/TSA) to keep `neg_transfer_rate≈0`, the limiting factor is now **certificate validity**: the selector rarely picks the best candidate within the action set.

This iteration adds **candidate-level diagnostics** (`candidates.csv`) and tests a single lever:
- **Per-family calibration** for `calibrated_stack_ridge_guard` (vs global calibration).

Protocol is **strict LOSO**, same preprocessing/model as baseline.

---

## Prior failure analysis (why we changed something)
Observed in the previous stable run (`20260104_loso4_actionset_stack_fbcsp_tsa_gate_v1.md`):
- Mean improves slightly, but **accept rate is low** (most subjects fall back to EA).
- Evidence suggests the gain is mostly from **safety gate + fallback**, not from the learned selector.

Hypothesis: the selector is trained on heterogeneous candidates; splitting calibration by **candidate family** might improve ranking/correlation.

---

## Experiment setup (fixed)
- Dataset: `BNCI2014_001`
- Task: **4-class** (`left_hand,right_hand,feet,tongue`)
- Protocol: **LOSO**
- Preprocess: `paper_fir` (8–30 Hz, 250 Hz, t=0.5–3.5s, causal FIR)
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Selector: `calibrated_stack_ridge_guard`
- High-risk gates (kept identical in both runs):
  - FBCSP: `guard>=0.95`, `min_pred_improve>=0.05`, `drift_delta<=0.15`
  - TSA: `guard>=0.95`, `min_pred_improve>=0.05`, `drift_delta<=0.15`
- Diagnostics: `--diagnose-subjects 1,2,3,4,5,6,7,8,9` (writes per-subject `candidates.csv`).

Action set inside `ea-stack-multi-safe-csp-lda`:
- EA anchor (`Q=I`)
- EA-FBCSP (high-risk)
- RPA (LEA whitening)
- TSA (LEA+TSA rotation, high-risk)
- EA-SI-CHAN (ranks=21, lambdas=0.5,1,2)

---

## Run A — Global calibration (baseline)
- Code commit for both runs: `a4358cd024687fcead3187772bb76b4a4b8d01d5`
- Output dir: `outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_global_v1/`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_stack_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_fbcsp_tsa_gate_diagall_global_v1
```

Main results (`20260104_method_comparison.csv`):
- EA mean acc: **0.5320**
- Stack mean acc: **0.5405** (**+0.00849** abs)
- Worst-subject acc: **0.2569** (same as EA)
- `neg_transfer_rate_vs_ea`: **0.0**
- `accept_rate`: **0.3333**

Candidate-level diagnostics (from `scripts/analyze_candidate_certificates.py`):
- `oracle_mean`: **0.5633**
- `gap_sel_mean` (oracle − selected): **0.0228**
- Candidate Spearman (mean across subjects):
  - `rho_ridge_mean`: **0.0079**
  - `rho_guard_mean`: **0.0556**

Figures:
- `docs/experiments/figures/20260104_stack_diagall_global_v1_delta.png`
- `docs/experiments/figures/20260104_stack_diagall_global_v1_oracle_gap.png`
- `docs/experiments/figures/20260104_stack_diagall_global_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260104_stack_diagall_global_v1_guard_vs_true.png`

---

## Run B — Per-family calibration (the only lever)
- Output dir: `outputs/20260104/4class/loso4_actionset_stack_fbcsp_tsa_gate_diagall_perfamily_v1/`
- Change: added `--stack-calib-per-family`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.5,1,2 \
  --oea-zo-selector calibrated_stack_ridge_guard \
  --oea-zo-calib-max-subjects 0 --oea-zo-calib-seed 0 \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_fbcsp_tsa_gate_diagall_perfamily_v1
```

Main results (`20260104_method_comparison.csv`):
- Stack mean acc: **0.5301** (**−0.00193** vs EA)
- `neg_transfer_rate_vs_ea`: **0.2222** (2/9 subjects)
- `accept_rate`: **0.7778**

Candidate-level diagnostics:
- `oracle_mean`: **0.5633**
- `gap_sel_mean`: **0.0332** (worse than Run A)
- Candidate Spearman:
  - `rho_ridge_mean`: **0.0992** (slightly higher, but not enough)
  - `rho_guard_mean`: **0.0278**

Negative-transfer subjects (selected worse than EA):
- Subject 4: `0.4514 → 0.4063` (−0.0451)
- Subject 9: `0.6840 → 0.5868` (−0.0972)

Figures:
- `docs/experiments/figures/20260104_stack_diagall_perfamily_v1_delta.png`
- `docs/experiments/figures/20260104_stack_diagall_perfamily_v1_oracle_gap.png`
- `docs/experiments/figures/20260104_stack_diagall_perfamily_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260104_stack_diagall_perfamily_v1_guard_vs_true.png`

---

## Takeaways
1) We are now in the **certificate correlation stage**: the action set has headroom (`oracle_mean≈0.563`), but the selector does not reliably pick it (`gap_sel_mean≈0.023` even for the safe global setting).
2) Naive **per-family calibration** increases acceptance, but **over-accepts** and causes negative transfer (Run B), likely due to limited calibration samples per family (overfitting / miscalibration).

## Next step (planned)
Keep the safety gate fixed, and improve certificate validity without overfitting:
- Use **partial pooling / shrinkage** instead of fully separate per-family models (e.g., global ridge + family-specific bias; enable per-family only if sample count ≥ threshold; or strict family-specific gate only for high-risk families).

