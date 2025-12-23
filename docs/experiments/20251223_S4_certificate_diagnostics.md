# 2025-12-23 — 4-class S4 “unlabeled certificate failure” diagnostics (EA‑ZO)

This note implements the 3 diagnostics you proposed to explain why **subject 4** drops under EA‑ZO:

1) track the test‑time class‑marginal `p̄` trajectory,
2) check correlation between the unlabeled objective and true accuracy over candidate `Q`,
3) replace `KL(uniform || p̄)` with `KL(π || p̄)` where `π` is an estimated prior.

## What was added
- New ZO diagnostics output (`--diagnose-subjects 4`):
  - `diagnostics/<method>/subject_XX/pbar_trajectory.png`
  - `diagnostics/<method>/subject_XX/objective_vs_accuracy.png`
  - `diagnostics/<method>/subject_XX/candidates.csv`
  - `diagnostics/<method>/subject_XX/summary.txt`
- New marginal mode `kl_prior` and prior selector:
  - `--oea-zo-marginal-mode kl_prior`
  - `--oea-zo-marginal-prior {uniform,source,anchor_pred}`
    - `source`: empirical class prior from **training labels**
    - `anchor_pred`: target marginal predicted at **Q=I (EA)** (fixed during optimization)

## Diagnostic runs (4-class, paper_fir, CSP=6)
Common config:
- `--events left_hand,right_hand,feet,tongue`
- `--preprocess paper_fir --n-components 6`
- `--methods ea-csp-lda,ea-zo-im-csp-lda`
- `--oea-q-blend 1.0`
- `--oea-zo-objective infomax`
- `--oea-zo-marginal-beta 0.5`
- `--oea-zo-holdout-fraction 0.3`
- `--oea-zo-warm-start delta --oea-zo-warm-iters 1`
- `--oea-zo-trust-lambda 0.1 --oea-zo-trust-q0 identity`
- `--diagnose-subjects 4`

### Run A — KL-uniform
Run folder:
- `outputs/20251223/diag4_kl_uniform/`

Diagnostics:
- `outputs/20251223/diag4_kl_uniform/diagnostics/ea-zo-im-csp-lda/subject_04/summary.txt`
- `outputs/20251223/diag4_kl_uniform/diagnostics/ea-zo-im-csp-lda/subject_04/candidates.csv`
- `outputs/20251223/diag4_kl_uniform/diagnostics/ea-zo-im-csp-lda/subject_04/pbar_trajectory.png`
- `outputs/20251223/diag4_kl_uniform/diagnostics/ea-zo-im-csp-lda/subject_04/objective_vs_accuracy.png`

Key numbers (S4):
- best by **true acc**: `identity` (EA), acc `0.451389`
- best by **unlabeled objective**: `q_delta`, acc `0.368056`
- correlation:
  - Pearson(objective, acc) = `+0.200358`
  - Spearman(objective, acc) = `−0.046041`

Interpretation:
- The “certificate” (unlabeled objective) is **not aligned** with true accuracy on S4.
- `q_delta` is also the **closest to uniform** `p̄`, which suggests the marginal penalty is driving selection.

### Run B — KL-prior with π = anchor_pred (target marginal at Q=I)
Run folder:
- `outputs/20251223/diag4_kl_prior_anchor/`

Diagnostics:
- `outputs/20251223/diag4_kl_prior_anchor/diagnostics/ea-zo-im-csp-lda/subject_04/summary.txt`

Key numbers (S4):
- still best by objective: `q_delta` (worse acc), best by acc: `identity`
- correlation stays near 0 / wrong‑sign:
  - Pearson(objective, acc) = `+0.218939`
  - Spearman(objective, acc) = `−0.007338`

Note: `π` here equals the EA anchor predicted marginal:
`[0.2109, 0.2059, 0.3080, 0.2752]` (see `summary.txt`).

### Run C — KL-prior with π = source prior (training labels)
Run folder:
- `outputs/20251223/diag4_kl_prior_source/`

Outcome:
- identical per-subject results to Run A/B under this configuration (π is close to uniform for this dataset).

## Conclusion (what the diagnostics show)
For S4 in 4-class:
- The unlabeled objective currently **selects a Q with worse accuracy than EA**.
- The objective is dominated by the marginal penalty term (and/or the base InfoMax surrogate is not predictive of accuracy for this subject), so “unlabeled holdout selection” becomes a weak certificate.
- Switching from uniform prior to an estimated prior **does not fix** the core misalignment in this configuration.

Next actionable direction (if we continue later):
- Redesign the unlabeled objective/certificate to be more predictive of accuracy (or add a safety acceptance criterion that can reject “objective wins but accuracy drops” cases using only unlabeled statistics).

