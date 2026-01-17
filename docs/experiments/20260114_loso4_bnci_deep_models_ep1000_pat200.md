# 20260114 — BNCI2014_001 4-class LOSO: Deep baselines “fully trained” (ep=1000, patience=200)

## 0) Post-mortem (why we did this)

Deep baselines (`Deep4Net`, `ATCNet`) previously looked much worse than `ea-csp-lda` under our strict protocol.  
Concern: **undertraining** (too few epochs / too-small patience) might explain the gap.

This iteration therefore fixes the deep training budget to a very large setting to remove that doubt:

- `--deep-max-epochs 1000`
- `--deep-patience 200`

All else kept identical (dataset, preprocessing, LOSO protocol, metrics).

---

## 1) Commands (reproducible)

### Deep4Net

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir \
  --n-components 4 \
  --events left_hand,right_hand,feet,tongue \
  --methods ea-csp-lda,deep4net \
  --deep-max-epochs 1000 --deep-patience 200 \
  --run-name loso4_bnci_deep4net_ep1000_pat200_v1
```

### ATCNet

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir \
  --n-components 4 \
  --events left_hand,right_hand,feet,tongue \
  --methods ea-csp-lda,atcnet \
  --deep-max-epochs 1000 --deep-patience 200 \
  --run-name loso4_bnci_atcnet_ep1000_pat200_v1
```

---

## 2) Outputs

### Deep4Net

- `outputs/20260114/4class/loso4_bnci_deep4net_ep1000_pat200_v1/20260114_method_comparison.csv`
- `outputs/20260114/4class/loso4_bnci_deep4net_ep1000_pat200_v1/20260114_predictions_all_methods.csv`

### ATCNet

- `outputs/20260114/4class/loso4_bnci_atcnet_ep1000_pat200_v1/20260114_method_comparison.csv`
- `outputs/20260114/4class/loso4_bnci_atcnet_ep1000_pat200_v1/20260114_predictions_all_methods.csv`

---

## 3) Results (strict LOSO, 9 subjects)

### Anchor (EA)

`ea-csp-lda` mean accuracy **0.5297**, worst-subject **0.2778**

### Deep4Net (ep=1000, patience=200)

From `.../loso4_bnci_deep4net_ep1000_pat200_v1/20260114_method_comparison.csv`:

- mean accuracy **0.3465**
- worst-subject **0.2465**
- mean Δ vs EA **−18.33 pp**
- neg-transfer rate vs EA **8/9**

### ATCNet (ep=1000, patience=200)

From `.../loso4_bnci_atcnet_ep1000_pat200_v1/20260114_method_comparison.csv`:

- mean accuracy **0.4117**
- worst-subject **0.2778**
- mean Δ vs EA **−11.81 pp**
- neg-transfer rate vs EA **8/9**

---

## 4) Conclusion

Even after a “very large” training budget, **both deep baselines remain far below EA** under our strict `paper_fir` LOSO protocol.

So the earlier poor results are **not** explained by insufficient epochs/patience; the likely cause is a **pipeline/recipe mismatch** (deep models typically need different preprocessing/augmentation/cropping and/or different evaluation choices to be competitive).

Implication for the paper:
- deep models can still be reported as baselines, but to keep “strict comparability” we should **not** change the protocol just to make deep nets look better in the main table.
- If we later run deep baselines with their recommended recipes, those results should be clearly marked as **non-comparable** (supplementary/appendix).

