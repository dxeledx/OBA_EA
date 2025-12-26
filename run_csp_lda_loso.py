from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import mne

from csp_lda.config import ExperimentConfig, ModelConfig, PreprocessingConfig
from csp_lda.data import BCIIV2aMoabbLoader, split_by_subject
from csp_lda.evaluation import compute_metrics, loso_cross_subject_evaluation
from csp_lda.plots import (
    plot_confusion_matrix,
    plot_csp_patterns,
    plot_method_comparison_bar,
)
from csp_lda.reporting import today_yyyymmdd, write_results_txt_multi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSP+LDA LOSO on MOABB BNCI2014_001 (BCI IV 2a).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root directory. Run outputs go to OUT_DIR/YYYYMMDD/<N>class/HHMMSS_* (no overwrite).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subfolder name under OUT_DIR/YYYYMMDD/. Default: current time HHMMSS.",
    )
    p.add_argument("--fmin", type=float, default=8.0)
    p.add_argument("--fmax", type=float, default=30.0)
    # Align with He & Wu (EA) paper for BCI IV 2a:
    # use 0.5–3.5s after cue appearance.
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    p.add_argument("--resample", type=float, default=250.0)
    p.add_argument("--n-components", type=int, default=4, help="CSP components (n_components).")
    p.add_argument(
        "--preprocess",
        choices=["moabb", "paper_fir"],
        default="moabb",
        help="Preprocessing pipeline: 'moabb' (default) or 'paper_fir' (causal 50-order FIR Hamming).",
    )
    p.add_argument("--fir-order", type=int, default=50, help="FIR order for paper_fir mode.")
    p.add_argument(
        "--events",
        type=str,
        default="left_hand,right_hand",
        help="Comma-separated events/classes (e.g., left_hand,right_hand).",
    )
    p.add_argument(
        "--sessions",
        type=str,
        default="0train",
        help="Comma-separated MOABB session names to include (e.g., 0train). Use 'ALL' to include all.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="csp-lda,ea-csp-lda",
        help=(
            "Comma-separated methods to run: csp-lda, ea-csp-lda, oea-cov-csp-lda, oea-csp-lda, "
            "oea-zo-csp-lda, oea-zo-ent-csp-lda, oea-zo-im-csp-lda, oea-zo-pce-csp-lda, oea-zo-conf-csp-lda, "
            "oea-zo-imr-csp-lda, "
            "ea-si-csp-lda, ea-si-zo-csp-lda, ea-si-zo-ent-csp-lda, ea-si-zo-im-csp-lda, ea-si-zo-imr-csp-lda, "
            "ea-si-zo-pce-csp-lda, ea-si-zo-conf-csp-lda, "
            "ea-zo-csp-lda, ea-zo-ent-csp-lda, ea-zo-im-csp-lda, ea-zo-pce-csp-lda, ea-zo-conf-csp-lda, "
            "ea-zo-imr-csp-lda"
        ),
    )
    p.add_argument(
        "--oea-eps",
        type=float,
        default=1e-10,
        help="Numeric stability epsilon used by EA/OEA (eigenvalue floor via eps*max_eig).",
    )
    p.add_argument(
        "--oea-shrinkage",
        type=float,
        default=0.0,
        help="Optional covariance shrinkage in EA/OEA (0 means disabled).",
    )
    p.add_argument(
        "--si-subject-lambda",
        type=float,
        default=1.0,
        help=(
            "For ea-si-* methods: subject invariance strength λ (>=0) in the HSIC-style projector "
            "(larger enforces more subject invariance)."
        ),
    )
    p.add_argument(
        "--si-ridge",
        type=float,
        default=1e-6,
        help="For ea-si-* methods: ridge (>0) used in the generalized eigen-problem for the projector.",
    )
    p.add_argument(
        "--si-proj-dim",
        type=int,
        default=0,
        help="For ea-si-* methods: projector output dimension r (0 keeps full CSP feature dimension).",
    )
    p.add_argument(
        "--oea-pseudo-iters",
        type=int,
        default=2,
        help="For oea-csp-lda only: number of pseudo-label iterations on the target subject.",
    )
    p.add_argument(
        "--oea-pseudo-mode",
        choices=["hard", "soft"],
        default="hard",
        help="For oea-csp-lda only: pseudo-label mode for target Q_t selection ('hard' or 'soft').",
    )
    p.add_argument(
        "--oea-pseudo-confidence",
        type=float,
        default=0.0,
        help=(
            "For oea-csp-lda only (hard mode) and oea-zo-* with objective=pseudo_ce: "
            "minimum confidence to keep a pseudo-labeled trial (0 disables)."
        ),
    )
    p.add_argument(
        "--oea-pseudo-topk-per-class",
        type=int,
        default=0,
        help=(
            "For oea-csp-lda only (hard mode) and oea-zo-* with objective=pseudo_ce: "
            "keep top-k confident trials per class (0 disables)."
        ),
    )
    p.add_argument(
        "--oea-pseudo-balance",
        action="store_true",
        help=(
            "For oea-csp-lda only (hard mode) and oea-zo-* with objective=pseudo_ce: "
            "balance pseudo-labeled trials per class (uses min count)."
        ),
    )
    p.add_argument(
        "--oea-zo-objective",
        choices=["entropy", "infomax", "pseudo_ce", "confidence", "lda_nll", "entropy_bilevel", "infomax_bilevel"],
        default="entropy",
        help="For oea-zo-* methods: zero-order objective on target unlabeled data.",
    )
    p.add_argument(
        "--oea-zo-transform",
        choices=["orthogonal", "rot_scale"],
        default="orthogonal",
        help=(
            "For oea-zo-* methods: channel-space transform family. "
            "'orthogonal' uses Q∈O(C) (pure rotation); "
            "'rot_scale' uses A=diag(exp(s))·Q (rotation + per-channel scaling)."
        ),
    )
    p.add_argument(
        "--oea-zo-infomax-lambda",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=infomax: weight λ for H(mean p) term (must be > 0).",
    )
    p.add_argument(
        "--oea-zo-marginal-mode",
        choices=["none", "l2_uniform", "kl_uniform", "hinge_uniform", "hard_min", "kl_prior"],
        default="none",
        help=(
            "For oea-zo-* methods: add a class-marginal balance penalty on the predicted marginal p_bar "
            "(none disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-marginal-beta",
        type=float,
        default=0.0,
        help="For oea-zo-* methods: class-marginal penalty weight β (>=0).",
    )
    p.add_argument(
        "--oea-zo-marginal-tau",
        type=float,
        default=0.05,
        help="For oea-zo-* methods with marginal_mode=hinge_uniform: lower-bound threshold τ in [0,1].",
    )
    p.add_argument(
        "--oea-zo-marginal-prior",
        choices=["uniform", "source", "anchor_pred"],
        default="uniform",
        help=(
            "For oea-zo-* methods with marginal_mode=kl_prior: choose π used in KL(π||p_bar). "
            "uniform uses π=1/K; source uses training-label empirical prior; anchor_pred uses "
            "the target marginal predicted at Q=I (EA)."
        ),
    )
    p.add_argument(
        "--oea-zo-marginal-prior-mix",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with marginal_mode=kl_prior: mix π with uniform (0=no mix, 1=all uniform).",
    )
    p.add_argument(
        "--oea-zo-bilevel-iters",
        type=int,
        default=5,
        help="For oea-zo-* methods with objective=*_bilevel: lower-level iterations for (w,q) solver.",
    )
    p.add_argument(
        "--oea-zo-bilevel-temp",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=*_bilevel: temperature for soft labels q (T>0).",
    )
    p.add_argument(
        "--oea-zo-bilevel-step",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=*_bilevel: prior-matching step size (>=0).",
    )
    p.add_argument(
        "--oea-zo-bilevel-coverage-target",
        type=float,
        default=0.5,
        help="For oea-zo-* methods with objective=*_bilevel: coverage target in (0,1] for gating prior strength.",
    )
    p.add_argument(
        "--oea-zo-bilevel-coverage-power",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=*_bilevel: power (>=0) applied to coverage gating.",
    )
    p.add_argument(
        "--oea-zo-reliable-metric",
        choices=["none", "confidence", "entropy"],
        default="none",
        help=(
            "For oea-zo-* methods: optional reliability weighting metric used inside entropy/infomax/confidence "
            "objectives (none disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-reliable-threshold",
        type=float,
        default=0.7,
        help=(
            "For oea-zo-* methods with reliable_metric != none: threshold for reliability weighting. "
            "If metric=confidence, must be in [0,1]. If metric=entropy, must be >=0."
        ),
    )
    p.add_argument(
        "--oea-zo-reliable-alpha",
        type=float,
        default=10.0,
        help="For oea-zo-* methods with reliable_metric != none: sigmoid sharpness (alpha > 0).",
    )
    p.add_argument(
        "--oea-zo-trust-lambda",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: trust-region penalty weight ρ for ||Q - Q0||_F^2 "
            "(0 disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-trust-q0",
        choices=["identity", "delta"],
        default="identity",
        help="For oea-zo-* methods: trust-region anchor Q0 (identity|delta).",
    )
    p.add_argument(
        "--oea-zo-drift-mode",
        choices=["none", "penalty", "hard"],
        default="none",
        help=(
            "For oea-zo-* methods: prediction-drift guard relative to the EA anchor (Q=I). "
            "none disables; penalty adds γ * mean KL(p0||pQ); hard enforces drift<=δ at selection time."
        ),
    )
    p.add_argument(
        "--oea-zo-drift-gamma",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with drift_mode=penalty: penalty weight γ (>=0).",
    )
    p.add_argument(
        "--oea-zo-drift-delta",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with drift_mode=hard: drift threshold δ (>=0).",
    )
    p.add_argument(
        "--oea-zo-selector",
        choices=[
            "objective",
            "dev",
            "evidence",
            "probe_mixup",
            "probe_mixup_hard",
            "iwcv",
            "iwcv_ucb",
            "calibrated_ridge",
            "calibrated_guard",
            "calibrated_ridge_guard",
            "calibrated_stack_ridge",
            "oracle",
        ],
        default="objective",
        help=(
            "For oea-zo-* methods: how to select Q_t from the candidate set. "
            "objective selects by the unlabeled objective (plus optional drift guard); "
            "evidence selects by LDA evidence (-log p(z)) under the frozen CSP+LDA model; "
            "probe_mixup selects by a MixUp-style probe score in CSP feature space; "
            "probe_mixup_hard selects by a MixUp-style probe with hard-major pseudo labels (MixVal-style λ>0.5); "
            "dev selects by a DEV-style control-variate IW certificate on labeled source; "
            "iwcv selects by importance-weighted NLL on labeled source (covariate-shift certificate); "
            "iwcv_ucb selects by IWCV-UCB (IWCV-NLL plus kappa*SE via n_eff); "
            "calibrated_ridge learns a regressor on source subjects to predict improvement; "
            "calibrated_guard learns a binary guard to reject likely negative transfer; "
            "calibrated_ridge_guard uses the learned guard to filter candidates, then selects by ridge-predicted improvement; "
            "calibrated_stack_ridge learns a ridge regressor on stacked certificate features (objective+evidence+probe+drift) to predict improvement; "
            "oracle selects by true accuracy (analysis-only upper bound; uses labels)."
        ),
    )
    p.add_argument(
        "--oea-zo-iwcv-kappa",
        type=float,
        default=1.0,
        help="For selector=iwcv_ucb: UCB penalty strength kappa (>=0).",
    )
    p.add_argument(
        "--oea-zo-calib-ridge-alpha",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with selector=calibrated_ridge: Ridge alpha (>0).",
    )
    p.add_argument(
        "--oea-zo-calib-max-subjects",
        type=int,
        default=0,
        help="For oea-zo-* methods with selector=calibrated_ridge: limit pseudo-target subjects per fold (0=all).",
    )
    p.add_argument(
        "--oea-zo-calib-seed",
        type=int,
        default=0,
        help="For oea-zo-* methods with selector=calibrated_ridge: random seed for sampling pseudo-targets.",
    )
    p.add_argument(
        "--oea-zo-calib-guard-c",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with selector=calibrated_guard: LogisticRegression C (>0).",
    )
    p.add_argument(
        "--oea-zo-calib-guard-threshold",
        type=float,
        default=0.5,
        help="For oea-zo-* methods with selector=calibrated_guard: keep candidates with P(pos) >= threshold.",
    )
    p.add_argument(
        "--oea-zo-calib-guard-margin",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with selector=calibrated_guard: label as positive if improvement > margin.",
    )
    p.add_argument(
        "--oea-zo-min-improvement",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: require at least this much holdout-objective improvement over identity "
            "before accepting an adapted Q_t (0 disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: holdout fraction in [0,1) used for best-iterate selection "
            "(updates use the remaining trials). 0 disables."
        ),
    )
    p.add_argument(
        "--oea-zo-warm-start",
        choices=["none", "delta"],
        default="none",
        help="For oea-zo-* methods: initialization strategy for SPSA (none|delta).",
    )
    p.add_argument(
        "--oea-zo-warm-iters",
        type=int,
        default=1,
        help="For oea-zo-* methods with warm_start=delta: number of pseudo-Δ refinement iterations.",
    )
    p.add_argument(
        "--oea-zo-fallback-min-marginal-entropy",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: if >0, enable an unlabeled safety fallback when the predicted "
            "class-marginal entropy H(mean p) falls below this threshold (nats)."
        ),
    )
    p.add_argument(
        "--oea-zo-iters",
        type=int,
        default=30,
        help="For oea-zo-* methods: SPSA iterations for optimizing Q_t.",
    )
    p.add_argument(
        "--oea-zo-lr",
        type=float,
        default=0.5,
        help="For oea-zo-* methods: SPSA learning rate (base).",
    )
    p.add_argument(
        "--oea-zo-mu",
        type=float,
        default=0.1,
        help="For oea-zo-* methods: SPSA perturbation size.",
    )
    p.add_argument(
        "--oea-zo-k",
        type=int,
        default=50,
        help="For oea-zo-* methods: number of Givens rotations (low-dim Q parameterization).",
    )
    p.add_argument(
        "--oea-zo-seed",
        type=int,
        default=0,
        help="For oea-zo-* methods: random seed for Givens planes and SPSA directions.",
    )
    p.add_argument(
        "--oea-zo-l2",
        type=float,
        default=0.0,
        help="For oea-zo-* methods: L2 regularization on Givens angles.",
    )
    p.add_argument(
        "--oea-q-blend",
        type=float,
        default=1.0,
        help="Blend factor in [0,1] to control how aggressive the selected Q is (0=I, 1=full Q).",
    )
    p.add_argument(
        "--diagnose-subjects",
        type=str,
        default="",
        help="Optional comma-separated subject ids to write ZO diagnostics (e.g. '4').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", message=r"warnEpochs.*")
    warnings.filterwarnings(
        "ignore", message=r"Concatenation of Annotations within Epochs is not supported yet.*"
    )
    date_prefix = today_yyyymmdd()
    out_root = Path(args.out_dir)

    events = tuple([e.strip() for e in str(args.events).split(",") if e.strip()])
    if len(events) < 2:
        raise ValueError("--events must contain at least two classes.")
    task_dir = f"{len(events)}class"
    base_dir = out_root / date_prefix / task_dir
    run_name = args.run_name or datetime.now().strftime("%H%M%S")
    out_dir = base_dir / run_name
    i = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_name}_{i:02d}"
        i += 1

    sessions_raw = str(args.sessions).strip()
    sessions = None if sessions_raw.upper() == "ALL" else tuple(
        [s.strip() for s in sessions_raw.split(",") if s.strip()]
    )
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    diagnose_subjects: tuple[int, ...] = ()
    if str(args.diagnose_subjects).strip():
        diagnose_subjects = tuple(
            sorted({int(x) for x in str(args.diagnose_subjects).split(",") if str(x).strip()})
        )

    preprocessing = PreprocessingConfig(
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        resample=float(args.resample),
        events=events,
        sessions=tuple(sessions) if sessions is not None else (),
        preprocess=str(args.preprocess),
        paper_fir_order=int(args.fir_order),
    )
    model_cfg = ModelConfig(csp_n_components=int(args.n_components))
    config = ExperimentConfig(out_dir=out_dir, preprocessing=preprocessing, model=model_cfg)

    loader = BCIIV2aMoabbLoader(
        fmin=config.preprocessing.fmin,
        fmax=config.preprocessing.fmax,
        tmin=config.preprocessing.tmin,
        tmax=config.preprocessing.tmax,
        resample=config.preprocessing.resample,
        events=config.preprocessing.events,
        sessions=sessions,
        preprocess=config.preprocessing.preprocess,
        paper_fir_order=config.preprocessing.paper_fir_order,
        paper_fir_window=config.preprocessing.paper_fir_window,
    )
    X, y, meta = loader.load_arrays(dtype="float32")
    subject_data = split_by_subject(X, y, meta)
    info = loader.load_epochs_info()

    metric_columns = ["accuracy", "precision", "recall", "f1", "auc", "kappa"]
    class_order = list(config.preprocessing.events)

    results_by_method: dict[str, pd.DataFrame] = {}
    overall_by_method: dict[str, dict[str, float]] = {}
    predictions_by_method: dict[str, tuple] = {}
    trial_predictions_by_method: dict[str, pd.DataFrame] = {}
    method_details: dict[str, str] = {}

    for method in methods:
        # Per-method override for OEA-ZO objectives (so we can compare multiple ZO objectives in one run).
        zo_objective_override: str | None = None

        if method == "csp-lda":
            alignment = "none"
            method_details[method] = "No alignment."
        elif method == "ea-csp-lda":
            alignment = "ea"
            method_details[method] = f"EA: per-subject whitening (eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
        elif method == "oea-cov-csp-lda":
            alignment = "oea_cov"
            method_details[method] = (
                "OEA (cov-eig selection): choose Q_s from EA solution set using covariance eigen-basis "
                f"(eps={args.oea_eps}, shrinkage={args.oea_shrinkage}, q_blend={args.oea_q_blend})."
            )
        elif method == "oea-csp-lda":
            alignment = "oea"
            method_details[method] = (
                "OEA (discriminative optimistic selection): choose Q_s by aligning a covariance signature to a reference "
                "(binary: Δ=Cov(c1)-Cov(c0); multiclass: between-class covariance scatter); "
                f"target uses {args.oea_pseudo_iters} pseudo-label iters "
                f"(eps={args.oea_eps}, shrinkage={args.oea_shrinkage}, q_blend={args.oea_q_blend}, "
                f"pseudo_mode={args.oea_pseudo_mode}, pseudo_conf={args.oea_pseudo_confidence}, "
                f"topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method in {
            "oea-zo-csp-lda",
            "oea-zo-ent-csp-lda",
            "oea-zo-im-csp-lda",
            "oea-zo-imr-csp-lda",
            "oea-zo-pce-csp-lda",
            "oea-zo-conf-csp-lda",
        }:
            alignment = "oea_zo"
            if method == "oea-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "oea-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "oea-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "oea-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "oea-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            elif str(args.oea_zo_selector) == "calibrated_guard":
                selector_str += (
                    f"(C={args.oea_zo_calib_guard_c}, "
                    f"thr={args.oea_zo_calib_guard_threshold}, margin={args.oea_zo_calib_guard_margin}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "OEA-ZO (target optimistic selection): source uses covariance-signature alignment for Q_s "
                "(binary Δ; multiclass scatter); "
                "target optimizes Q_t by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method == "ea-si-csp-lda":
            alignment = "ea_si"
            method_details[method] = (
                "EA-SI: source trains on EA-whitened data with a subject-invariant linear projector (HSIC-style), "
                f"(si_lambda={args.si_subject_lambda}, si_ridge={args.si_ridge}, si_proj_dim={args.si_proj_dim})."
            )
        elif method in {
            "ea-si-zo-csp-lda",
            "ea-si-zo-ent-csp-lda",
            "ea-si-zo-im-csp-lda",
            "ea-si-zo-imr-csp-lda",
            "ea-si-zo-pce-csp-lda",
            "ea-si-zo-conf-csp-lda",
        }:
            alignment = "ea_si_zo"
            if method == "ea-si-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "ea-si-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "ea-si-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "ea-si-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "ea-si-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            elif str(args.oea_zo_selector) == "calibrated_guard":
                selector_str += (
                    f"(C={args.oea_zo_calib_guard_c}, "
                    f"thr={args.oea_zo_calib_guard_threshold}, margin={args.oea_zo_calib_guard_margin}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "EA-SI-ZO: source trains on EA-whitened data with a subject-invariant linear projector (HSIC-style), "
                f"(si_lambda={args.si_subject_lambda}, si_ridge={args.si_ridge}, si_proj_dim={args.si_proj_dim}); "
                "target optimizes Q_t by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method in {
            "ea-zo-ent-csp-lda",
            "ea-zo-im-csp-lda",
            "ea-zo-imr-csp-lda",
            "ea-zo-pce-csp-lda",
            "ea-zo-conf-csp-lda",
            "ea-zo-csp-lda",
        }:
            alignment = "ea_zo"
            if method == "ea-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "ea-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "ea-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "ea-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "ea-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "EA-ZO (target optimistic selection): source trains on EA-whitened data (no Q_s selection); "
                "target optimizes Q_t by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        else:
            raise ValueError(
                "Unknown method "
                f"'{method}'. Supported: csp-lda, ea-csp-lda, oea-cov-csp-lda, oea-csp-lda, "
                "oea-zo-csp-lda, oea-zo-ent-csp-lda, oea-zo-im-csp-lda, oea-zo-imr-csp-lda, "
                "oea-zo-pce-csp-lda, oea-zo-conf-csp-lda, "
                "ea-si-csp-lda, ea-si-zo-csp-lda, ea-si-zo-ent-csp-lda, ea-si-zo-im-csp-lda, ea-si-zo-imr-csp-lda, "
                "ea-si-zo-pce-csp-lda, ea-si-zo-conf-csp-lda, "
                "ea-zo-csp-lda, ea-zo-ent-csp-lda, ea-zo-im-csp-lda, ea-zo-imr-csp-lda, "
                "ea-zo-pce-csp-lda, ea-zo-conf-csp-lda"
            )

        (
            results_df,
            pred_df,
            y_true_all,
            y_pred_all,
            y_proba_all,
            _class_order,
            _models_by_subject,
        ) = (
            loso_cross_subject_evaluation(
                subject_data,
                class_order=class_order,
                n_components=config.model.csp_n_components,
                average=config.metrics_average,
                alignment=alignment,
                oea_eps=float(args.oea_eps),
                oea_shrinkage=float(args.oea_shrinkage),
                oea_pseudo_iters=int(args.oea_pseudo_iters),
                oea_q_blend=float(args.oea_q_blend),
                oea_pseudo_mode=str(args.oea_pseudo_mode),
                oea_pseudo_confidence=float(args.oea_pseudo_confidence),
                oea_pseudo_topk_per_class=int(args.oea_pseudo_topk_per_class),
                oea_pseudo_balance=bool(args.oea_pseudo_balance),
                oea_zo_objective=str(zo_objective_override or args.oea_zo_objective),
                oea_zo_transform=str(args.oea_zo_transform),
                oea_zo_infomax_lambda=float(args.oea_zo_infomax_lambda),
                oea_zo_marginal_mode=str(args.oea_zo_marginal_mode),
                oea_zo_marginal_beta=float(args.oea_zo_marginal_beta),
                oea_zo_marginal_tau=float(args.oea_zo_marginal_tau),
                oea_zo_marginal_prior=str(args.oea_zo_marginal_prior),
                oea_zo_marginal_prior_mix=float(args.oea_zo_marginal_prior_mix),
                oea_zo_bilevel_iters=int(args.oea_zo_bilevel_iters),
                oea_zo_bilevel_temp=float(args.oea_zo_bilevel_temp),
                oea_zo_bilevel_step=float(args.oea_zo_bilevel_step),
                oea_zo_bilevel_coverage_target=float(args.oea_zo_bilevel_coverage_target),
                oea_zo_bilevel_coverage_power=float(args.oea_zo_bilevel_coverage_power),
                oea_zo_reliable_metric=str(args.oea_zo_reliable_metric),
                oea_zo_reliable_threshold=float(args.oea_zo_reliable_threshold),
                oea_zo_reliable_alpha=float(args.oea_zo_reliable_alpha),
                oea_zo_trust_lambda=float(args.oea_zo_trust_lambda),
                oea_zo_trust_q0=str(args.oea_zo_trust_q0),
                oea_zo_drift_mode=str(args.oea_zo_drift_mode),
                oea_zo_drift_gamma=float(args.oea_zo_drift_gamma),
                oea_zo_drift_delta=float(args.oea_zo_drift_delta),
                oea_zo_selector=str(args.oea_zo_selector),
                oea_zo_iwcv_kappa=float(args.oea_zo_iwcv_kappa),
                oea_zo_calib_ridge_alpha=float(args.oea_zo_calib_ridge_alpha),
                oea_zo_calib_max_subjects=int(args.oea_zo_calib_max_subjects),
                oea_zo_calib_seed=int(args.oea_zo_calib_seed),
                oea_zo_calib_guard_c=float(args.oea_zo_calib_guard_c),
                oea_zo_calib_guard_threshold=float(args.oea_zo_calib_guard_threshold),
                oea_zo_calib_guard_margin=float(args.oea_zo_calib_guard_margin),
                oea_zo_min_improvement=float(args.oea_zo_min_improvement),
                oea_zo_holdout_fraction=float(args.oea_zo_holdout_fraction),
                oea_zo_warm_start=str(args.oea_zo_warm_start),
                oea_zo_warm_iters=int(args.oea_zo_warm_iters),
                oea_zo_fallback_min_marginal_entropy=float(args.oea_zo_fallback_min_marginal_entropy),
                oea_zo_iters=int(args.oea_zo_iters),
                oea_zo_lr=float(args.oea_zo_lr),
                oea_zo_mu=float(args.oea_zo_mu),
                oea_zo_k=int(args.oea_zo_k),
                oea_zo_seed=int(args.oea_zo_seed),
                oea_zo_l2=float(args.oea_zo_l2),
                si_subject_lambda=float(args.si_subject_lambda),
                si_ridge=float(args.si_ridge),
                si_proj_dim=int(args.si_proj_dim),
                diagnostics_dir=out_dir if diagnose_subjects else None,
                diagnostics_subjects=diagnose_subjects,
                diagnostics_tag=method,
            )
        )
        results_by_method[method] = results_df
        trial_predictions_by_method[method] = pred_df
        overall_by_method[method] = compute_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            y_proba=y_proba_all,
            class_order=class_order,
            average=config.metrics_average,
        )
        predictions_by_method[method] = (y_true_all, y_pred_all)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{date_prefix}_results.txt"
    write_results_txt_multi(
        results_by_method,
        config=config,
        output_path=results_path,
        metric_columns=metric_columns,
        overall_metrics_by_method=overall_by_method,
        method_details_by_method=method_details,
        protocol_name="LOSO",
    )

    # Detailed per-trial predictions (for per-subject analysis).
    all_pred_parts: list[pd.DataFrame] = []
    for method, pred_df in trial_predictions_by_method.items():
        out_path = out_dir / f"{date_prefix}_{method}_predictions.csv"
        pred_df.to_csv(out_path, index=False)
        df2 = pred_df.copy()
        df2.insert(0, "method", method)
        all_pred_parts.append(df2)
    if all_pred_parts:
        pd.concat(all_pred_parts, axis=0, ignore_index=True).to_csv(
            out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False
        )

    # Plots
    # 1) CSP patterns (fit on full data) + confusion matrices for each method
    from csp_lda.model import fit_csp_lda
    from csp_lda.alignment import (
        EuclideanAligner,
        apply_spatial_transform,
        blend_with_identity,
        class_cov_diff,
        orthogonal_align_symmetric,
        sorted_eigh,
    )

    for method in results_by_method.keys():
        if method == "ea-csp-lda" or method.startswith("ea-zo"):
            # Align each subject independently, then concatenate for a representative visualization.
            X_parts = []
            y_parts = []
            for s, sd in subject_data.items():
                X_parts.append(
                    EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit_transform(
                        sd.X
                    )
                )
                y_parts.append(sd.y)
            X_fit = np.concatenate(X_parts, axis=0)
            y_fit = np.concatenate(y_parts, axis=0)
        elif method == "oea-cov-csp-lda":
            # Visualization-only: build U_ref from all subjects.
            covs = []
            ea_by_subject = {}
            for s, sd in subject_data.items():
                ea = EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit(sd.X)
                ea_by_subject[int(s)] = ea
                covs.append(ea.cov_)
            c_ref = np.mean(np.stack(covs, axis=0), axis=0)
            _evals_ref, u_ref = sorted_eigh(c_ref)

            X_parts = []
            y_parts = []
            for s, sd in subject_data.items():
                ea = ea_by_subject[int(s)]
                z = ea.transform(sd.X)
                q = u_ref @ ea.eigvecs_.T
                q = blend_with_identity(q, float(args.oea_q_blend))
                X_parts.append(apply_spatial_transform(q, z))
                y_parts.append(sd.y)
            X_fit = np.concatenate(X_parts, axis=0)
            y_fit = np.concatenate(y_parts, axis=0)
        elif method == "oea-csp-lda" or method.startswith("oea-zo"):
            # Visualization-only: use true labels to compute a covariance-signature reference and Q_s for all subjects.
            class_labels = tuple([str(c) for c in class_order])

            ea_by_subject = {}
            z_by_subject = {}
            diffs = []
            for s, sd in subject_data.items():
                ea = EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit(sd.X)
                ea_by_subject[int(s)] = ea
                z = ea.transform(sd.X)
                z_by_subject[int(s)] = z
                diffs.append(
                    class_cov_diff(
                        z,
                        sd.y,
                        class_order=class_labels,
                        eps=float(args.oea_eps),
                        shrinkage=float(args.oea_shrinkage),
                    )
                )
            d_ref = np.mean(np.stack(diffs, axis=0), axis=0)

            X_parts = []
            y_parts = []
            for s, sd in subject_data.items():
                d_s = class_cov_diff(
                    z_by_subject[int(s)],
                    sd.y,
                    class_order=class_labels,
                    eps=float(args.oea_eps),
                    shrinkage=float(args.oea_shrinkage),
                )
                q_s = orthogonal_align_symmetric(d_s, d_ref)
                q_s = blend_with_identity(q_s, float(args.oea_q_blend))
                X_parts.append(apply_spatial_transform(q_s, z_by_subject[int(s)]))
                y_parts.append(sd.y)
            X_fit = np.concatenate(X_parts, axis=0)
            y_fit = np.concatenate(y_parts, axis=0)
        else:
            X_fit, y_fit = X, y

        final_model = fit_csp_lda(X_fit, y_fit, n_components=config.model.csp_n_components)
        plot_csp_patterns(
            final_model.csp,
            info,
            output_path=out_dir / f"{date_prefix}_{method}_csp_patterns.png",
            title=f"{method} CSP patterns (n_components={config.model.csp_n_components})",
        )

        y_true_all, y_pred_all = predictions_by_method[method]
        plot_confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=class_order,
            output_path=out_dir / f"{date_prefix}_{method}_confusion_matrix.png",
            title=f"{method} confusion matrix (LOSO, all subjects)",
        )

    # 2) Model performance comparison bar (per-subject accuracy)
    plot_method_comparison_bar(
        results_by_method,
        metric="accuracy",
        output_path=out_dir / f"{date_prefix}_model_compare_accuracy.png",
        title="LOSO accuracy by subject (model comparison)",
    )

    # Console short summary
    pd.set_option("display.width", 120)
    for method in results_by_method.keys():
        print(f"\n=== {method} ===")
        print(results_by_method[method])
    print("\nSaved:", results_path)


if __name__ == "__main__":
    main()
