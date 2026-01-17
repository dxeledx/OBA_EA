# BNCI2014_001 — 4-class — LOSO (strict)

All assets in this folder are derived from the following run folders (same protocol / same trials):

## Strict baselines + (ours, stacked_delta without min_pred)

- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_v1/`

## Strict (ours, stacked_delta + global min_pred gate)

- `outputs/20260116/4class/loso4_bnci_strict_stackdelt_minpred002_v1/`

Notes:
- “strict” means `--oea-pseudo-iters 0` (no target pseudo-iteration / no TSA rotation).
- Method naming uses paper-faithful naming (`lea-*`); historical aliases remain supported.

