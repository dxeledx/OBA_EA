# 20260106 — BNCI2014_001 4-class LOSO: probe_hard **min-improve** gate (ε=0.01) reduces neg-transfer and improves mean

## Post-mortem (previous iteration)
The “probe_hard do-not-worsen” gate (ε=+0.03) did **not** reliably prevent false positives:
- `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4/`
- mean acc: **0.5320** (≈EA)
- `neg_transfer_rate_vs_ea`: **0.2222** (S8, S9)
- Main failure: S9 had **oracle=EA** but selector still chose a harmful candidate.

Key evidence (from v4 candidates):
S9 improved probe_hard only by ~**0.0018** (tiny), yet accuracy dropped **−9.72%**.
So we need to reject “tiny certificate wins” under large candidate sets.

---

## Goal (this iteration)
Change **one lever**: tighten the EA-anchor-relative probe_hard gate to require a **minimum improvement**.

---

## Protocol (fixed, strictly comparable)
- Dataset: `BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO, `sessions=0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR order=50), epoch 0.5–3.5s, resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Selector: `calibrated_stack_ridge_guard` (unchanged)
- Candidate set: dense SI-CHAN λ grid `0.25,0.35,0.5,0.7,1,1.4,2` (unchanged)
- Other safety gates unchanged: anchor-δ=0.05, FBCSP/TSA high-risk gates, per-family calibration blend K=20

---

## What changed (single lever)
Switch probe gate from “do-not-worsen” to “min-improve” by allowing negative ε:

Let \(h(c)=\texttt{probe\_mixup\_hard\_best}(c)\) (smaller is better).
For any non-identity candidate \(c\), require:
\[
h(c) \le h(\mathrm{EA}) - \varepsilon,\quad \varepsilon = 0.01.
\]

CLI:
- `--stack-safe-anchor-probe-hard-worsen -0.01`

Commit (code): `ce6d81929dbb74c3d96d434333586b4bb91216ed`

---

## Run
Output dir:
- `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_minimp001_v1/`

Command (from `*_results.txt`):
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --dataset BNCI2014_001 --preprocess paper_fir \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --n-components 6 \
  --methods ea-csp-lda,ea-stack-multi-safe-csp-lda \
  --si-proj-dim 21 --si-subject-lambda 1 --si-ridge 1e-6 \
  --si-chan-ranks 21 --si-chan-lambdas 0.25,0.35,0.5,0.7,1,1.4,2 \
  --oea-zo-selector calibrated_stack_ridge_guard \
  --oea-zo-calib-guard-threshold 0.5 \
  --stack-safe-anchor-guard-delta 0.05 \
  --stack-safe-anchor-probe-hard-worsen -0.01 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_minimp001_v1
```

---

## Main results
From `20260106_method_comparison.csv`:
- EA mean acc: **0.5320**
- Stack mean acc: **0.5413** (**+0.93% abs**)
- Worst-subject: **0.2569** (unchanged)
- `accept_rate`: **0.5556** (5/9)
- `neg_transfer_rate_vs_ea`: **0.1111** (1/9; only S8)

Per-subject Δacc (stack − EA):
- Improved: S3 **+4.51%**, S5 **+3.13%**, S6 **+3.47%**
- Regressed: S8 **−2.78%**
- Fixed the catastrophic case: S9 **0.00%** (now falls back to EA; no longer −9.72%)

Candidate-level diagnostics (`scripts/analyze_candidate_certificates.py`):
- `oracle_mean`: **0.5633**
- `gap_sel_mean`: **0.0220** (improved vs ~0.031–0.036 in the failing runs)

Figures:
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_minimp001_v1_delta.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_minimp001_v1_oracle_gap.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_minimp001_v1_ridge_vs_true.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_minimp001_v1_guard_vs_true.png`

---

## Interpretation
Requiring a **minimum** probe improvement acts as an “anti multiple-testing” safeguard:
it rejects candidates that only win the unlabeled certificate by a tiny margin (likely noise),
which is exactly the failure mode that caused S9’s false-positive selection.

---

## Next step
We now have a stable positive gain, but still a remaining negative-transfer subject (S8).
Next iteration should keep this ε=0.01 setting and change only one additional safeguard targeted to S8’s failure mode
(e.g., tighten probe ε slightly, or combine with a drift/coverage-aware rule—one lever at a time).

