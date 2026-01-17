# 20260113 — BNCI2014_001 4-class LOSO: ATCNet baseline sanity check (paper_fir)

## 0) Post-mortem (why we tried a deep baseline)

Recent strict-LOSO runs on `BNCI2014_001` (4-class, `paper_fir`) show:

- `ea-csp-lda` is a strong anchor around **0.53 mean accuracy**.
- Several added action families (e.g., TS-SVC / TSA-TS-SVC / FgMDM) did **not** provide headroom under the same protocol.

**Hypothesis:** if the bottleneck is the *classifier family*, a stronger deep MI architecture might raise absolute accuracy beyond EA.

So we test **ATCNet** as a deep baseline under the **same** protocol and metrics.

---

## 1) This iteration (one lever)

**Primary lever:** run a Braindecode `ATCNet` trialwise deep baseline inside our LOSO script.

Notes:
- Strict LOSO: train on pooled source subjects only; **no** target adaptation.
- Per-channel z-score standardization is fit on training fold only.
- CPU-only here (CUDA/NVML not available in this environment), so long runs are slow.

---

## 2) Commands (reproducible)

### Attempted “longer” run (timed out)

We first attempted a larger epoch budget, but it exceeded the practical runtime (no artifacts were written).

### Completed run (ep=20 quick direction check)

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir \
  --n-components 4 \
  --events left_hand,right_hand,feet,tongue \
  --methods ea-csp-lda,atcnet \
  --deep-max-epochs 20 --deep-patience 5 \
  --run-name loso4_bnci_atcnet_ep20_v1
```

---

## 3) Outputs

Folder:
- `outputs/20260113/4class/loso4_bnci_atcnet_ep20_v1/`

Key artifacts:
- `outputs/20260113/4class/loso4_bnci_atcnet_ep20_v1/20260113_method_comparison.csv`
- `outputs/20260113/4class/loso4_bnci_atcnet_ep20_v1/20260113_predictions_all_methods.csv`
- `outputs/20260113/4class/loso4_bnci_atcnet_ep20_v1/20260113_model_compare_accuracy.png`
- `outputs/20260113/4class/loso4_bnci_atcnet_ep20_v1/20260113_atcnet_confusion_matrix.png`
- `outputs/20260113/4class/loso4_bnci_atcnet_ep20_v1/20260113_ea-csp-lda_confusion_matrix.png`

---

## 4) Results (strict LOSO, 9 subjects)

From `20260113_method_comparison.csv`:

- `ea-csp-lda`: mean **0.5297**, worst-subject **0.2778**
- `atcnet` (ep=20): mean **0.4090**, worst-subject **0.2361**
  - mean Δ vs EA: **−12.08 pp**
  - neg-transfer rate vs EA: **8/9**

**Conclusion:** with this quick-but-controlled training recipe, **ATCNet is not competitive** under our strict `paper_fir` LOSO protocol.

---

## 5) Diagnosis (why it likely underperforms here)

1) **Recipe mismatch under strict protocol**: deep MI models usually rely on a different preprocessing/normalization/augmentation pipeline (often wider band, different time window, cropped training, etc.). We intentionally held protocol fixed for comparability.
2) **CPU-only constraint**: pushing to the long schedules typically used by deep baselines is slow here; a fully tuned deep baseline may require GPU and careful training recipes.
3) **Cross-subject generalization is hard**: without subject-wise adaptation or domain generalization tricks, deep nets can underperform strong classical alignment baselines.

---

## 6) Next step

Given both `Deep4Net` and `ATCNet` are far below EA under this strict protocol, the most promising path remains:

- keep the **CSP-family main table** as the comparable anchor,
- improve “safe selection / certificate reliability” over candidate families that already show headroom under this protocol,
- treat deep baselines (if we later include them) as a separate branch with a clearly stated non-comparable preprocessing/training recipe.

