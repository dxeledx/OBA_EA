# 20260107 — BNCI2014_001 4-class LOSO: fix **Borda** selector (exclude EA identity from ranking) → +1.77% abs, 0 neg-transfer

## Post-mortem (previous iteration)
The first Borda attempt did **not** change selection behavior vs ridge:
- `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_v1/`
- EA mean acc: **0.5320**
- Stack mean acc: **0.5413** (**+0.93% abs**)
- `neg_transfer_rate_vs_ea`: **0.1111** (S8 regressed)

Root cause: in `csp_lda/certificate.py::select_by_guarded_predicted_improvement`, the **identity (EA)** record was included in
the ranked candidate set for the Borda aggregation. With tie-breaking (`np.argmin`) this could cause identity to “win” even when
a non-identity candidate had better combined rank, effectively collapsing Borda back to ridge+fallback behavior.

---

## Goal (this iteration)
Change **one lever**: make Borda ranking operate only over **non-identity** candidates; identity is kept only as the anchor/fallback.

---

## Protocol (fixed, strictly comparable)
- Dataset: `BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO, `sessions=0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR order=50), epoch 0.5–3.5s, resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Candidate set: multi-family (EA anchor, RPA, TSA, EA-FBCSP, EA-SI-CHAN(λ-grid))
- Safety: anchor-δ=0.05, probe_hard **min-improve** gate (ε=0.01), high-risk family gates for FBCSP/TSA, per-family calibration blend K=20

---

## What changed (single lever)
In the guarded selector, do **not** rank identity as an eligible candidate:
- identity (EA) is **only** the anchor for relative thresholds and the safe fallback.

Commit: `47fcd295faf271b1e2fc05f7b0b3b28277e845e5`

---

## Run
Output dir:
- `outputs/20260107/4class/loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1/`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
  --oea-zo-selector calibrated_stack_ridge_guard_borda \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-anchor-guard-delta 0.05 \
  --stack-safe-anchor-probe-hard-worsen -0.01 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_familyblend_k20_probe_minimp001_borda_fix_v1
```

---

## Main results
From `20260107_method_comparison.csv`:
- EA mean acc: **0.5320**
- Stack mean acc: **0.5498** (**+1.77% abs**)
- Worst-subject: **0.2569** (unchanged)
- `accept_rate`: **0.6667** (6/9)
- `neg_transfer_rate_vs_ea`: **0.0000** (0/9)

Per-subject Δacc (stack − EA):
- Improved: **5/9** subjects (S1 +1.74%, S3 +4.51%, S5 +3.13%, S6 +3.47%, S8 +3.13%)
- Tied: 4/9 (S2/S4/S7/S9)

Candidate diagnostics (`scripts/analyze_candidate_certificates.py`):
- `oracle_mean`: **0.5633**
- `gap_sel_mean`: **0.0135** (smaller gap vs the earlier runs)

Selected family distribution (from per-subject `candidates.csv`):
- `chan`: 4 subjects, `rpa`: 2 subjects, fallback `ea`: 3 subjects

---

## Figures
Generated via:
- `python3 scripts/plot_stack_multi_safe_summary.py ...`
- `python3 scripts/plot_candidate_diagnostics.py ...`

Saved to:
- `docs/experiments/figures/20260107_loso4_borda_fix/loso4_borda_fix_delta.png`
- `docs/experiments/figures/20260107_loso4_borda_fix/loso4_borda_fix_oracle_gap.png`
- `docs/experiments/figures/20260107_loso4_borda_fix/loso4_borda_fix_family.png`
- `docs/experiments/figures/20260107_loso4_borda_fix/loso4_borda_fix_ridge_vs_true.png`
- `docs/experiments/figures/20260107_loso4_borda_fix/loso4_borda_fix_guard_vs_true.png`

---

## Interpretation
This iteration demonstrates that the “small gain / occasional drop” issue was (partly) **selection logic**, not only certificate quality:
fixing the Borda selector to correctly rank **only non-identity** candidates turns the same candidate set + same gates into:
- higher mean,
- higher accept-rate,
- and **0 negative transfer** under strict LOSO.

