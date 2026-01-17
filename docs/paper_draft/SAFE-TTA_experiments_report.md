# SAFE‑TTA — Experiment report (CSP+LDA line) — BNCI2014_001 + HGD

This document is written for **advisor reporting** and can be adapted into the **Experiments/Results** section of the SAFE‑TTA paper draft.

Paper name (working): *SAFE‑TTA: Certificate‑Guided Risk‑Controlled Test‑Time Adaptation for Cross‑Subject EEG Decoding*

---

## 1) What we are trying to prove (scientific claims)

**Claim A — Safety**: Under strict LOSO (no target labels used for selection), SAFE‑TTA can reduce **negative transfer** to ~0 by combining calibrated certificates with explicit gates and fallback to an anchor.

**Claim B — Utility**: Under the same strict protocol, SAFE‑TTA achieves **higher mean accuracy** than strong anchors/baselines on multiple public datasets.

**Claim C — Certificate effectiveness**: Certificate signals and calibrated predictors correlate positively with true improvement; SAFE‑TTA “eats headroom” while rejecting false positives.

---

## 2) Protocol consistency (how we avoid reviewer complaints)

Across datasets, we keep the following **consistent**:
- Evaluation: **cross‑subject LOSO**, strict (selection/adaptation uses only target unlabeled data).
- Metrics: subject‑mean accuracy & Cohen’s κ; worst‑subject; negative transfer rate vs EA anchor; paired statistical tests.
- Comparability: within each dataset, all methods use the **same preprocessing** and **same label space**; only the method differs.
- SAFE‑TTA rule: action‑set selection with explicit gates + fallback to anchor.

Dataset‑specific settings (e.g., sampling rate/resample) are fixed **per dataset** and stated explicitly, which is standard for multi‑dataset EEG papers.

---

## 3) Dataset 1 — BNCI2014_001 (BCI‑IV 2a), 4‑class strict LOSO (main task)

### 3.1 Main result table (paper‑ready)
- `docs/paper_assets/bnci2014_001/4class_loso_strict/tables/main_table.md`

### 3.2 SAFE‑TTA summary (what to say in the report)
Our strict LOSO SAFE‑TTA variant (stacked‑delta + global minPred gate) achieves:
- mean accuracy improvement over EA of **~+1.5pp**,
- **0 negative transfer** across 9 subjects,
- positive 95% bootstrap CI on Δacc (see `main_table.csv`).

### 3.3 Key figures to show (paper‑ready paths)
1) Mean accuracy/kappa + negative transfer (main figure):
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_main_bar_mean_accuracy.png`
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_main_bar_mean_kappa.png`
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_main_bar_neg_transfer_rate.png`
2) Per‑subject deltas (stability / worst‑case):
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_main_bar_subject_delta_acc_ea-stack-multi-safe-csp-lda_strict_delta_minpred002_minus_ea-csp-lda.png`
3) Confusion matrices (EA vs SAFE‑TTA):
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_cm_ea_norm.png`
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_cm_ours_norm.png`
4) Certificate effectiveness + oracle headroom (the “science evidence”):
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/tables/certificate_reliability.md`
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_ridge_vs_true.png`
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_oracle_gap.png`
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/figures/bnci4_strict_candidate_v1_headroom.png`

### 3.4 What remains weak on BNCI (honest post‑mortem)
The paired tests on BNCI are **borderline** (small n=9). This is exactly why adding a 3rd dataset with many subjects (PhysioNetMI) is expected to strengthen statistical evidence.

---

## 4) Dataset 2 — Schirrmeister2017 (HGD), 4‑class strict LOSO

### 4.1 Strong baselines main table (paper‑ready)
- `docs/paper_assets/schirrmeister2017/4class_loso_strict/tables/main_table.md`

This table includes EA anchor + closed‑form alignment baselines + (no‑EA) FBCSP + EA‑FBCSP (high risk) + SAFE‑TTA (risk‑controlled).

### 4.2 SAFE‑TTA summary (what to say)
On HGD, the EA‑FBCSP candidate has positive mean gain but causes notable negative transfer. SAFE‑TTA treats it as a **high‑risk arm** and adds a family‑specific certificate gate (pred_disagree) with **train‑only LOSO‑style calibration** of the gate threshold. The resulting SAFE‑TTA:
- achieves **~+2.7pp** mean accuracy vs EA,
- reaches **0/14 negative transfer**,
- is statistically stronger due to larger n=14.

### 4.3 Key figures to show (paper‑ready)
1) Main comparison:
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/figures/hgd4_strict_main_bar_mean_accuracy.png`
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/figures/hgd4_strict_main_bar_neg_transfer_rate.png`
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/figures/hgd4_strict_main_bar_subject_delta_acc_ea-fbcsp-pred-disagree-safe_minus_ea-csp-lda.png`
2) Confusion matrices:
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/figures/hgd4_strict_cm_ea_norm.png`
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/figures/hgd4_strict_cm_ea_fbcsp_norm.png`
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/figures/hgd4_strict_cm_ours_norm.png`
3) Risk‑reward sweep (supplementary, shows the “sweet spot” and risk control):
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/figures/hgd4_strict_pred_disagree_tau037/tau_sweep.png`

---

## 5) Takeaway: what the current evidence supports

With two public datasets:
- We already have a coherent story: **unlabeled certificates can be unreliable → SAFE‑TTA calibrates certificates on source, adds explicit gates, and falls back to EA to control risk.**
- Empirically, SAFE‑TTA improves mean accuracy while driving negative transfer to 0 on both datasets.

However, for JBHI/TBME submissions, a third dataset with many subjects is strongly recommended to:
- strengthen statistical evidence (larger n),
- demonstrate cross‑dataset generality of the same SAFE‑TTA framework.

---

## 6) Planned Dataset 3 — PhysioNetMI (MOABB PhysionetMI)

**Why**: many subjects (MOABB reports 109 subjects), which directly strengthens statistical tests and stability claims.

From local MOABB inspection:
- subject count: 109 (`subject_list` 1..109)
- events: `{left_hand, right_hand, feet, hands, rest}`
- sessions: 1
- interval: `[0, 3]`

**To keep protocols comparable**:
- We will run **4‑class** using the shared label set with HGD: `left_hand,right_hand,feet,rest` (exclude `hands`), under strict LOSO.
- Use the same SAFE‑TTA framework (action set + calibrated selector + gates + fallback), the same metrics, and the same reporting package (main table + per‑subject + confusion matrices + certificate effectiveness diagnostics).

---

## 7) What the advisor should look at first (2‑minute checklist)

1) BNCI main table + per‑subject Δacc:
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/tables/main_table.md`
2) HGD main table + neg‑transfer bar:
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict/tables/main_table.md`
3) Certificate reliability evidence on BNCI:
   - `docs/paper_assets/bnci2014_001/4class_loso_strict/tables/certificate_reliability.md`
4) HGD risk‑reward sweep (shows why the gate is needed):
   - `docs/paper_assets/schirrmeister2017/4class_loso_strict_calib_pred_disagree/figures/hgd4_strict_pred_disagree_tau037/tau_sweep.png`

