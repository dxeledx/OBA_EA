# 2026-01-02 — LOSO 4-class — `local_mix_then_ea` Oracle Headroom (post-fix)

Goal: re-run the **oracle headroom** experiment for `local_mix_then_ea` after fixing the `*_then_ea` transform dispatch in `csp_lda/zo.py` (commit `94bc866`).

## Setup (strictly comparable)
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO (cross-subject)
- Classes: `left_hand,right_hand,feet,tongue` (4-class)
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5 s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Experiment — Oracle headroom (analysis-only; uses labels)
Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform local_mix_then_ea \
  --oea-zo-localmix-neighbors 8 --oea-zo-localmix-self-bias 3.0 \
  --oea-zo-iters 50 --oea-zo-lr 0.2 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-beta 0 --oea-zo-marginal-prior anchor_pred \
  --oea-zo-selector oracle \
  --no-plots \
  --run-name loso4_local_mix_then_ea_imr_oracle_k8_fix
```

Outputs:
- `outputs/20260102/4class/loso4_local_mix_then_ea_imr_oracle_k8_fix/20260102_method_comparison.csv`
- `outputs/20260102/4class/loso4_local_mix_then_ea_imr_oracle_k8_fix/20260102_results.txt`

Key numbers (mean acc, 9 subjects):
- EA mean acc: `0.5320`, worst-subject: `0.2569`
- Oracle (best-by-acc among candidates) mean acc: `0.5328` (**+0.08% abs**), worst-subject: `0.2569`
- Per-subject: 2 improved, 7 unchanged, 0 worse (oracle selection).

## Interpretation
Even after the bugfix, `local_mix_then_ea` shows **near-zero oracle headroom** under this protocol/budget. This suggests that applying EA whitening *after* local mixing does not create a more beneficial candidate family (at least for EA→CSP→LDA on BNCI2014_001 4-class LOSO).
