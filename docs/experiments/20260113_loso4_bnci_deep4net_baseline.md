# 20260113 — BNCI2014_001 4-class LOSO: Deep4Net baseline sanity check (paper_fir)

## 0) Post-mortem (why we changed anything)

**Last stable anchor (strict LOSO, BNCI2014_001 4-class, `paper_fir`):**

- `ea-csp-lda` mean accuracy ≈ **0.532** (see recent runs in `docs/experiments/results_registry.csv`).
- “Action-set expansion” attempts with TS-SVC / TSA-TS-SVC / FgMDM did **not** show headroom under our protocol (either much worse than EA, or safe selector abstained).

**Hypothesis:** we may be bottlenecked by the *classifier family* (CSP+LDA / shallow Riemannian arms) rather than the “certificate”.  
So we first test a **stronger baseline family** under the **same split + same preprocessing band/window**: Braindecode `Deep4Net`.

This is **not** our proposed method; it’s a “does a stronger family even help here?” probe.

---

## 1) This iteration (one lever)

**Primary lever:** add a new baseline method `deep4net` (Braindecode Deep4Net, trialwise) into `run_csp_lda_loso.py`.

Implementation notes:
- Strict LOSO: train on pooled *source subjects* only; no target adaptation.
- Per-channel z-score standardization computed on training fold only.
- Optimizer: AdamW, early stopping on validation loss (random split inside training fold).

---

## 2) Command (reproducible)

```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir \
  --n-components 4 \
  --events left_hand,right_hand,feet,tongue \
  --methods ea-csp-lda,deep4net \
  --run-name loso4_bnci_deep4net_v1
```

---

## 3) Outputs

Folder:
- `outputs/20260113/4class/loso4_bnci_deep4net_v1/`

Key artifacts:
- `outputs/20260113/4class/loso4_bnci_deep4net_v1/20260113_method_comparison.csv`
- `outputs/20260113/4class/loso4_bnci_deep4net_v1/20260113_predictions_all_methods.csv`
- `outputs/20260113/4class/loso4_bnci_deep4net_v1/20260113_model_compare_accuracy.png`
- `outputs/20260113/4class/loso4_bnci_deep4net_v1/20260113_deep4net_confusion_matrix.png`
- `outputs/20260113/4class/loso4_bnci_deep4net_v1/20260113_ea-csp-lda_confusion_matrix.png`

---

## 4) Results (strict LOSO, 9 subjects)

From `20260113_method_comparison.csv`:

- `ea-csp-lda`: mean **0.5297**, worst-subject **0.2778**
- `deep4net`: mean **0.3835**, worst-subject **0.2431**
  - mean Δ vs EA: **−14.62 pp**
  - neg-transfer rate vs EA: **8/9**

**Conclusion (this run):** Under *our strict protocol* (0train-only, 8–30 Hz, 0.5–3.5 s, 250 Hz), this Deep4Net recipe is **not a competitive baseline**; it is far below EA.

---

## 5) Diagnosis (why it likely failed)

This failure is consistent with common MI-EEG deep-model pitfalls under strict cross-subject protocols:

1) **Training recipe mismatch**: deep baselines often rely on stronger augmentation, longer schedules, cropped training, and/or different band/window choices. Our current setup is intentionally fixed to match the CSP paper-like protocol.
2) **Small effective sample size per fold**: per LOSO fold we have ~2304 trials total; deep nets can underperform without careful regularization/augmentation.
3) **Protocol comparability constraints**: using both sessions / different preprocessing might rescue Deep4Net, but would no longer be directly comparable to the EA-CSP-LDA protocol we’re using as the main table.

This is a negative result worth keeping (it supports the claim that “just switching to a deep model is not automatically better under strict LOSO+paper_fir”).

---

## 6) Next step (what we do next)

Given Deep4Net is far below EA here, we should avoid spending time “minor tuning” unless we decide to make a **deep-family** its own controlled branch.

**Next baseline family to try (still via Braindecode/MOABB, more MI-focused):**
- `EEGConformer` or `ATCNet` (expected stronger but slower).

Decision rule:
- If a deep baseline can reach **≥ EA** under the same protocol, then it’s worth adding into our safe-selector action set as a new “family”.
- If not, we should keep focusing on CSP-family improvements + certificate reliability (our main direction).

