# 2026-01-02 — LOSO 4-class — `local_affine_then_ea` Oracle Headroom (BNCI2014_001)

Goal: test a **stronger physiology-motivated channel transform family** for cross-subject MI-EEG *without electrode coordinates*, and measure whether it increases the **oracle headroom** over EA on BNCI2014_001 4-class LOSO.

## Motivation
In public MI datasets we typically **lack per-subject electrode coordinates**, yet electrode placement / head-shape mismatch is a plausible contributor to cross-subject performance variance. A reasonable inductive bias is that such mismatch can be approximated by a **local spatial mixing** of nearby channels.

Previously we tested `local_mix` (row-stochastic, nonnegative). Here we test a more expressive variant:

- `local_affine_then_ea`: **signed**, neighbor-sparse local linear mixing `A` (row-wise, L2-normalized) **followed by EA whitening** in the mixed space.

This expands the action space (can model sign changes / referencing-like effects) while keeping the EA preconditioning.

## Setup (strictly comparable)
- Dataset: MOABB `BNCI2014_001` (BCI IV-2a)
- Protocol: LOSO (cross-subject)
- Classes: `left_hand,right_hand,feet,tongue` (4-class)
- Sessions: `0train`
- Preprocess: `paper_fir` (causal FIR order=50, 8–30 Hz), epoch 0.5–3.5 s, resample 250 Hz
- Model: CSP(`n_components=6`) + LDA

## Experiment A — Oracle headroom (analysis-only; uses labels)
Purpose: verify whether the candidate set for `local_affine_then_ea` contains improvements over the EA anchor.

Command:
```bash
conda run -n eeg python run_csp_lda_loso.py \
  --preprocess paper_fir --n-components 6 \
  --events left_hand,right_hand,feet,tongue --sessions 0train \
  --methods ea-csp-lda,ea-zo-imr-csp-lda \
  --oea-zo-transform local_affine_then_ea \
  --oea-zo-localmix-neighbors 8 --oea-zo-localmix-self-bias 3.0 \
  --oea-zo-iters 50 --oea-zo-lr 0.2 --oea-zo-mu 0.05 --oea-zo-l2 0.001 \
  --oea-zo-reliable-metric confidence --oea-zo-reliable-threshold 0.7 --oea-zo-reliable-alpha 10 \
  --oea-zo-marginal-mode kl_prior --oea-zo-marginal-beta 0 --oea-zo-marginal-prior anchor_pred \
  --oea-zo-selector oracle \
  --no-plots \
  --run-name loso4_local_affine_then_ea_imr_oracle_k8_v3
```

Outputs:
- `outputs/20260102/4class/loso4_local_affine_then_ea_imr_oracle_k8_v3/20260102_method_comparison.csv`
- `outputs/20260102/4class/loso4_local_affine_then_ea_imr_oracle_k8_v3/20260102_results.txt`

Key numbers (mean acc, 9 subjects):
- EA mean acc: `0.5320`, worst-subject: `0.2569`
- Oracle (best-by-acc among candidates) mean acc: `0.5347` (**+0.27% abs**), worst-subject: `0.2569`
- Per-subject: 5 improved, 4 unchanged, 0 worse (oracle selection).

Figures:
- Per-subject Δacc: `docs/experiments/figures/20260102_loso4_local_affine_then_ea_oracle_delta.png`
- Baseline-vs-oracle scatter: `docs/experiments/figures/20260102_loso4_local_affine_then_ea_oracle_scatter.png`

## Interpretation
- `local_affine_then_ea` **does not increase oracle headroom** relative to the previously tested `local_mix` oracle (+0.46% abs; see `docs/experiments/20251229_loso_4class_localmix_imr.md`).
- This suggests that, under the current EA→CSP→LDA pipeline and candidate-generation budget, simply enlarging the action space with signed local mixing is **not enough** to create a larger attainable gain.

## Implementation note (bugfix)
While running this experiment we found that `*_then_ea` transforms were previously being overridden by the orthogonal/rot_scale branch (effectively becoming identity). This was fixed in commit `94bc866`. Any earlier runs using `local_mix_then_ea` before this commit should be treated as **not comparable**.

## Next step
Given the small oracle headroom, the next iteration should focus on **increasing headroom** (stronger features/baselines) before considering RL/policy learning for selection.
