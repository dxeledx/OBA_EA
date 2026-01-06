# 20260106 — BNCI2014_001 4-class LOSO: EA-anchor-relative `probe_mixup_hard_best` gate (ε=0.03) on dense λ grid

## Post-mortem (previous iteration: anchor-δ still leaves false positives)
Previous run (dense SI-CHAN λ grid + EA-anchor-relative guard margin δ):
- `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_clean_v1/`
- EA mean acc: **0.5320**
- Stack mean acc: **0.5316** (≈EA)
- `accept_rate`: **0.7778** (7/9)
- `neg_transfer_rate_vs_ea`: **0.2222** (2/9)
- Remaining negative-transfer: **S8**, **S9** (large drops)

Interpretation: anchor-relative δ reduces “over-accept” under large candidate sets, but does not prevent **certificate false positives**.

---

## Goal (this iteration)
Keep protocol + candidate set fixed and change **one lever only**:
add a second EA-anchor-relative gate based on a MixVal-style probe score to further reject false positives.

---

## Protocol (fixed, strictly comparable)
- Dataset: `BNCI2014_001`
- Task: 4-class (`left_hand,right_hand,feet,tongue`)
- Protocol: LOSO, `sessions=0train`
- Preprocess: `paper_fir` (8–30 Hz, causal FIR order=50), epoch 0.5–3.5s, resample 250 Hz
- Model: `CSP(n_components=6) + LDA`
- Methods: `ea-csp-lda` vs `ea-stack-multi-safe-csp-lda`
- Selector: `calibrated_stack_ridge_guard` (unchanged)
- Per-family calibration: shrinkage blend `K=20` (unchanged)
- Candidate set: dense SI-CHAN λ grid `0.25,0.35,0.5,0.7,1,1.4,2` (unchanged)
- High-risk family gates (unchanged): FBCSP/TSA stricter acceptance (`guard>=0.95`, `min_pred_improve>=0.05`, `drift_delta<=0.15`)

---

## What changed (single lever)
Add **EA-anchor-relative probe_hard gate** (family-agnostic):

Let \(h(c)=\texttt{probe\_mixup\_hard\_best}(c)\) (smaller is better), and let EA anchor be the identity candidate.

For any non-identity candidate \(c\), require:
\[
h(c) \le h(\mathrm{EA}) + \varepsilon_{\text{worsen}}.
\]

Settings:
- `--stack-safe-anchor-probe-hard-worsen 0.03` (ε\_worsen = 0.03)

Implementation commit: `83920025bf5b00bb606b823d38a48df16f9edac7`

---

## Run
Output dir:
- `outputs/20260106/4class/loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4/`

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
  --stack-safe-anchor-probe-hard-worsen 0.03 \
  --stack-safe-fbcsp-guard-threshold 0.95 --stack-safe-fbcsp-min-pred-improve 0.05 --stack-safe-fbcsp-drift-delta 0.15 \
  --stack-safe-tsa-guard-threshold 0.95 --stack-safe-tsa-min-pred-improve 0.05 --stack-safe-tsa-drift-delta 0.15 \
  --stack-calib-per-family --stack-calib-per-family-mode blend --stack-calib-per-family-shrinkage 20 \
  --diagnose-subjects 1,2,3,4,5,6,7,8,9 \
  --no-plots \
  --run-name loso4_actionset_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4
```

---

## Main results (method-level)
From `20260106_method_comparison.csv`:
- EA mean acc: **0.5320**
- Stack mean acc: **0.5320** (Δ≈0.00% abs vs EA)
- Worst-subject acc: **0.2569** (same as EA)
- `accept_rate`: **0.7778** (7/9)
- `neg_transfer_rate_vs_ea`: **0.2222** (2/9)

Negative-transfer subjects (EA → stack):
- S8: **0.7188 → 0.6910** (−2.78% abs)
- S9: **0.6840 → 0.5868** (−9.72% abs)

Observation: the probe_hard gate does **not** eliminate negative transfer under dense candidates; failures can still happen when a harmful candidate has \(h(c)\le h(\mathrm{EA})+\varepsilon\).

---

## Candidate-level diagnostics (certificate validity)
Using `scripts/analyze_candidate_certificates.py`:
- `oracle_mean`: **0.5633**
- `gap_sel_mean` (oracle − selected): **0.0313**
- Candidate ranking correlation (mean Spearman across subjects):
  - `rho_probe_hard_mean`: **0.1970**
  - `rho_guard_mean`: **0.1758**
  - `rho_ridge_mean`: **0.0515**

Key failure mode (example S9):
- Selected candidate (RPA) has **slightly better** `probe_mixup_hard_best` than EA anchor, yet much worse accuracy.
- Candidate-set oracle equals EA (no headroom), but the selector still picks a harmful non-identity candidate.

Figures (this run):
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4_delta.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4_oracle_gap.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4_ridge_vs_true.png`
- `docs/experiments/figures/20260106_stack_familyblend_k20_chanlam_dense_anchor_delta005_probe_worsen003_v4_guard_vs_true.png`

---

## Decision / next step (per $eeg-q1-loso-loop)
This lever is **not sufficient**: anchor-relative “do-not-worsen” probe gating can still accept harmful candidates that *also* improve the probe score.

Next iteration should keep protocol + candidate set fixed and change **only one** of:
- make the acceptance rule require a **minimum probe improvement** (not just “not worse”), or
- add an explicit “EA-oracle-gap aware” guard: if the candidate-set oracle ≈ EA on pseudo-target subjects, tighten acceptance.

