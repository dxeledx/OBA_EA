from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import SubjectData
from .alignment import (
    EuclideanAligner,
    LogEuclideanAligner,
    apply_spatial_transform,
    blend_with_identity,
    class_cov_diff,
    orthogonal_align_tsa_procrustes,
    orthogonal_align_symmetric,
    sorted_eigh,
)
from .model import TrainedModel, fit_csp_lda
from .model import fit_csp_projected_lda
from .subject_invariant import (
    HSICProjectorParams,
    CenteredLinearProjector,
    learn_hsic_subject_invariant_projector,
    ChannelProjectorParams,
    learn_subject_invariant_channel_projector,
)
from .certificate import (
    candidate_features_from_record,
    stacked_candidate_features_from_record,
    select_by_dev_nll,
    select_by_evidence_nll,
    select_by_guarded_predicted_improvement,
    select_by_guarded_objective,
    select_by_iwcv_nll,
    select_by_iwcv_ucb,
    select_by_predicted_improvement,
    select_by_probe_mixup,
    select_by_probe_mixup_hard,
    train_logistic_guard,
    train_ridge_certificate,
)
from .metrics import compute_metrics, summarize_results
from .proba import reorder_proba_columns as _reorder_proba_columns
from .zo import (
    _optimize_qt_oea_zo,
    _select_pseudo_indices,
    _soft_class_cov_diff,
    _write_zo_diagnostics,
)
from .riemann import covariances_from_epochs


def _compute_tsa_target_rotation(
    *,
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_target: np.ndarray,
    model: TrainedModel,
    class_order: Sequence[str],
    pseudo_mode: str,
    pseudo_iters: int,
    q_blend: float,
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
    eps: float,
    shrinkage: float,
) -> np.ndarray:
    """Compute a TSA-style closed-form target rotation using (pseudo-)class anchors."""

    class_order = [str(c) for c in class_order]
    if pseudo_mode not in {"hard", "soft"}:
        raise ValueError("pseudo_mode must be one of: 'hard', 'soft'")
    if int(pseudo_iters) <= 0:
        return np.eye(int(z_target.shape[1]), dtype=np.float64)

    q_t = np.eye(int(z_target.shape[1]), dtype=np.float64)
    for _ in range(int(pseudo_iters)):
        X_cur = apply_spatial_transform(q_t, z_target)
        proba = model.predict_proba(X_cur)
        proba = _reorder_proba_columns(proba, model.classes_, list(class_order))

        try:
            if pseudo_mode == "soft":
                q_new = orthogonal_align_tsa_procrustes(
                    z_train,
                    y_train,
                    z_target,
                    pseudo_mode="soft",
                    proba_target=proba,
                    y_pseudo_target=None,
                    class_order=class_order,
                    eps=float(eps),
                    shrinkage=float(shrinkage),
                )
            else:
                y_pseudo = np.asarray(model.predict(X_cur))
                keep = _select_pseudo_indices(
                    y_pseudo=y_pseudo,
                    proba=proba,
                    class_order=class_order,
                    confidence=float(pseudo_confidence),
                    topk_per_class=int(pseudo_topk_per_class),
                    balance=bool(pseudo_balance),
                )
                if keep.size == 0:
                    break
                q_new = orthogonal_align_tsa_procrustes(
                    z_train,
                    y_train,
                    z_target[keep],
                    pseudo_mode="hard",
                    proba_target=None,
                    y_pseudo_target=y_pseudo[keep],
                    class_order=class_order,
                    eps=float(eps),
                    shrinkage=float(shrinkage),
                )
        except ValueError:
            break

        q_t = blend_with_identity(q_new, float(q_blend))

    return q_t

def _csp_features_from_filters(*, model: TrainedModel, X: np.ndarray) -> np.ndarray:
    """Compute CSP log-power features (same convention as ZO evaluation)."""

    X = np.asarray(X, dtype=np.float64)
    csp = model.csp
    F = np.asarray(csp.filters_[: int(csp.n_components)], dtype=np.float64)

    use_log = True if (getattr(csp, "log", None) is None) else bool(getattr(csp, "log"))
    Y = np.einsum("kc,nct->nkt", F, X, optimize=True)
    power = np.mean(Y * Y, axis=2)
    power = np.maximum(power, 1e-20)
    return np.log(power) if use_log else power


def _compute_lda_evidence_params(
    *,
    model: TrainedModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_order: Sequence[str],
    ridge: float = 1e-6,
) -> dict:
    """Build Gaussian-mixture evidence params from CSP+LDA training data.

    Returns a dict with:
    - mu: (K,d) class means in feature space
    - priors: (K,) class priors
    - cov: (d,d) pooled within-class covariance (ridge-stabilized)
    - cov_inv: (d,d) inverse covariance
    - logdet: log|cov|
    """

    class_order = [str(c) for c in class_order]
    feats = _csp_features_from_filters(model=model, X=X_train)
    y_train = np.asarray(y_train)
    n = int(feats.shape[0])
    if n != int(y_train.shape[0]):
        raise ValueError("X_train/y_train length mismatch for evidence params.")
    k = int(len(class_order))
    if k < 2:
        raise ValueError("Need at least 2 classes for evidence params.")

    mu = np.zeros((k, feats.shape[1]), dtype=np.float64)
    priors = np.zeros(k, dtype=np.float64)
    present = 0
    for i, c in enumerate(class_order):
        mask = y_train == c
        if not np.any(mask):
            continue
        present += 1
        priors[i] = float(np.sum(mask)) / float(n)
        mu[i] = np.mean(feats[mask], axis=0)
    if present < 2:
        raise ValueError("At least two classes must be present to compute evidence params.")

    priors = np.clip(priors, 1e-12, 1.0)
    priors = priors / float(np.sum(priors))

    # Pooled within-class covariance.
    d = int(feats.shape[1])
    scatter = np.zeros((d, d), dtype=np.float64)
    for i, c in enumerate(class_order):
        mask = y_train == c
        if not np.any(mask):
            continue
        fc = feats[mask] - mu[i]
        scatter += fc.T @ fc

    denom = max(1, int(n - present))
    cov = scatter / float(denom)
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov)) / float(d) if float(np.trace(cov)) > 0.0 else 1.0
    cov = cov + float(ridge) * float(scale) * np.eye(d, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0 or not np.isfinite(logdet):
        cov = cov + 1e-3 * np.eye(d, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(cov)
    cov_inv = np.linalg.pinv(cov) if (sign <= 0.0 or not np.isfinite(logdet)) else np.linalg.inv(cov)

    return {
        "mu": mu,
        "priors": priors,
        "cov": cov,
        "cov_inv": cov_inv,
        "logdet": float(logdet) if np.isfinite(logdet) else float("nan"),
    }


@dataclass(frozen=True)
class FoldResult:
    subject: int
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    kappa: float


def loso_cross_subject_evaluation(
    subject_data: Dict[int, SubjectData],
    *,
    class_order: Sequence[str],
    channel_names: Sequence[str] | None = None,
    n_components: int = 4,
    average: str = "macro",
    alignment: str = "none",
    oea_eps: float = 1e-10,
    oea_shrinkage: float = 0.0,
    oea_pseudo_iters: int = 2,
    oea_q_blend: float = 1.0,
    oea_pseudo_mode: str = "hard",
    oea_pseudo_confidence: float = 0.0,
    oea_pseudo_topk_per_class: int = 0,
    oea_pseudo_balance: bool = False,
    oea_zo_objective: str = "entropy",
    oea_zo_transform: str = "orthogonal",
    oea_zo_localmix_neighbors: int = 4,
    oea_zo_localmix_self_bias: float = 3.0,
    oea_zo_infomax_lambda: float = 1.0,
    oea_zo_reliable_metric: str = "none",
    oea_zo_reliable_threshold: float = 0.0,
    oea_zo_reliable_alpha: float = 10.0,
    oea_zo_trust_lambda: float = 0.0,
    oea_zo_trust_q0: str = "identity",
    oea_zo_marginal_mode: str = "none",
    oea_zo_marginal_beta: float = 0.0,
    oea_zo_marginal_tau: float = 0.05,
    oea_zo_marginal_prior: str = "uniform",
    oea_zo_marginal_prior_mix: float = 0.0,
    oea_zo_bilevel_iters: int = 5,
    oea_zo_bilevel_temp: float = 1.0,
    oea_zo_bilevel_step: float = 1.0,
    oea_zo_bilevel_coverage_target: float = 0.5,
    oea_zo_bilevel_coverage_power: float = 1.0,
    oea_zo_drift_mode: str = "none",
    oea_zo_drift_gamma: float = 0.0,
    oea_zo_drift_delta: float = 0.0,
    oea_zo_selector: str = "objective",
    oea_zo_iwcv_kappa: float = 1.0,
    oea_zo_calib_ridge_alpha: float = 1.0,
    oea_zo_calib_max_subjects: int = 0,
    oea_zo_calib_seed: int = 0,
    oea_zo_calib_guard_c: float = 1.0,
    oea_zo_calib_guard_threshold: float = 0.5,
    oea_zo_calib_guard_margin: float = 0.0,
    oea_zo_min_improvement: float = 0.0,
    oea_zo_holdout_fraction: float = 0.0,
    oea_zo_warm_start: str = "none",
    oea_zo_warm_iters: int = 1,
    oea_zo_fallback_min_marginal_entropy: float = 0.0,
    oea_zo_iters: int = 30,
    oea_zo_lr: float = 0.5,
    oea_zo_mu: float = 0.1,
    oea_zo_k: int = 50,
    oea_zo_seed: int = 0,
    oea_zo_l2: float = 0.0,
    mm_safe_mdm_guard_threshold: float = -1.0,
    mm_safe_mdm_min_pred_improve: float = 0.0,
    mm_safe_mdm_drift_delta: float = 0.0,
    si_subject_lambda: float = 1.0,
    si_ridge: float = 1e-6,
    si_proj_dim: int = 0,
    si_chan_candidate_ranks: Sequence[int] = (),
    si_chan_candidate_lambdas: Sequence[float] = (),
    diagnostics_dir: Path | None = None,
    diagnostics_subjects: Sequence[int] = (),
    diagnostics_tag: str = "",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[int, TrainedModel],
]:
    """LOSO evaluation: each subject is test once; others are training.

    Returns
    -------
    results_df:
        Per-subject metrics.
    y_true_all, y_pred_all, y_proba_all:
        Aggregated predictions across all folds (aligned to `class_order` columns).
    class_order:
        Class names order used.
    models_by_subject:
        Trained model for each test subject (useful for inspection/plotting).
    """

    subject_data_raw = subject_data
    subject_data_rpa: Dict[int, SubjectData] | None = None

    subjects = sorted(subject_data.keys())
    fold_rows: List[FoldResult] = []
    models_by_subject: Dict[int, TrainedModel] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    if alignment not in {
        "none",
        "ea",
        "rpa",
        "ea_si",
        "ea_si_chan",
        "ea_si_chan_safe",
        "ea_si_chan_multi_safe",
        "ea_mm_safe",
        "ea_stack_multi_safe",
        "riemann_mdm",
        "rpa_mdm",
        "rpa_rot_mdm",
        "ea_si_zo",
        "ea_zo",
        "raw_zo",
        "rpa_zo",
        "tsa",
        "tsa_zo",
        "oea_cov",
        "oea",
        "oea_zo",
    }:
        raise ValueError(
            "alignment must be one of: "
            "'none', 'ea', 'rpa', 'ea_si', 'ea_si_chan', 'ea_si_chan_safe', 'ea_si_chan_multi_safe', 'ea_mm_safe', 'ea_stack_multi_safe', "
            "'riemann_mdm', 'rpa_mdm', 'rpa_rot_mdm', "
            "'ea_si_zo', 'ea_zo', 'raw_zo', 'rpa_zo', 'tsa', 'tsa_zo', 'oea_cov', 'oea', 'oea_zo'"
        )

    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")
    if not (0.0 <= float(oea_pseudo_confidence) <= 1.0):
        raise ValueError("oea_pseudo_confidence must be in [0,1].")
    if int(oea_pseudo_topk_per_class) < 0:
        raise ValueError("oea_pseudo_topk_per_class must be >= 0.")

    if oea_zo_objective not in {
        "entropy",
        "pseudo_ce",
        "confidence",
        "infomax",
        "lda_nll",
        "entropy_bilevel",
        "infomax_bilevel",
    }:
        raise ValueError(
            "oea_zo_objective must be one of: "
            "'entropy', 'pseudo_ce', 'confidence', 'infomax', 'lda_nll', 'entropy_bilevel', 'infomax_bilevel'"
        )
    if float(oea_zo_infomax_lambda) <= 0.0:
        raise ValueError("oea_zo_infomax_lambda must be > 0.")
    if oea_zo_reliable_metric not in {"none", "confidence", "entropy"}:
        raise ValueError("oea_zo_reliable_metric must be one of: 'none', 'confidence', 'entropy'")
    if float(oea_zo_reliable_alpha) <= 0.0:
        raise ValueError("oea_zo_reliable_alpha must be > 0.")
    if oea_zo_reliable_metric == "confidence" and not (
        0.0 <= float(oea_zo_reliable_threshold) <= 1.0
    ):
        raise ValueError("oea_zo_reliable_threshold must be in [0,1] when metric='confidence'.")
    if oea_zo_reliable_metric == "entropy" and float(oea_zo_reliable_threshold) < 0.0:
        raise ValueError("oea_zo_reliable_threshold must be >= 0 when metric='entropy'.")
    if float(oea_zo_trust_lambda) < 0.0:
        raise ValueError("oea_zo_trust_lambda must be >= 0.")
    if oea_zo_trust_q0 not in {"identity", "delta"}:
        raise ValueError("oea_zo_trust_q0 must be one of: 'identity', 'delta'.")
    if oea_zo_drift_mode not in {"none", "penalty", "hard"}:
        raise ValueError("oea_zo_drift_mode must be one of: 'none', 'penalty', 'hard'.")
    if float(oea_zo_drift_gamma) < 0.0:
        raise ValueError("oea_zo_drift_gamma must be >= 0.")
    if float(oea_zo_drift_delta) < 0.0:
        raise ValueError("oea_zo_drift_delta must be >= 0.")
    if oea_zo_selector not in {
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
    }:
        raise ValueError(
            "oea_zo_selector must be one of: "
            "'objective', 'dev', 'evidence', 'probe_mixup', 'probe_mixup_hard', 'iwcv', 'iwcv_ucb', 'calibrated_ridge', 'calibrated_guard', 'calibrated_ridge_guard', 'calibrated_stack_ridge', 'oracle'."
        )
    if float(oea_zo_iwcv_kappa) < 0.0:
        raise ValueError("oea_zo_iwcv_kappa must be >= 0.")
    if float(oea_zo_calib_ridge_alpha) <= 0.0:
        raise ValueError("oea_zo_calib_ridge_alpha must be > 0.")
    if int(oea_zo_calib_max_subjects) < 0:
        raise ValueError("oea_zo_calib_max_subjects must be >= 0.")
    if float(oea_zo_calib_guard_c) <= 0.0:
        raise ValueError("oea_zo_calib_guard_c must be > 0.")
    if not (0.0 <= float(oea_zo_calib_guard_threshold) <= 1.0):
        raise ValueError("oea_zo_calib_guard_threshold must be in [0,1].")
    if float(oea_zo_calib_guard_margin) < 0.0:
        raise ValueError("oea_zo_calib_guard_margin must be >= 0.")
    if oea_zo_marginal_mode not in {
        "none",
        "l2_uniform",
        "kl_uniform",
        "hinge_uniform",
        "hard_min",
        "kl_prior",
    }:
        raise ValueError(
            "oea_zo_marginal_mode must be one of: "
            "'none', 'l2_uniform', 'kl_uniform', 'hinge_uniform', 'hard_min', 'kl_prior'."
        )
    if float(oea_zo_marginal_beta) < 0.0:
        raise ValueError("oea_zo_marginal_beta must be >= 0.")
    if not (0.0 <= float(oea_zo_marginal_tau) <= 1.0):
        raise ValueError("oea_zo_marginal_tau must be in [0,1].")
    if oea_zo_marginal_prior not in {"uniform", "source", "anchor_pred"}:
        raise ValueError("oea_zo_marginal_prior must be one of: 'uniform', 'source', 'anchor_pred'.")
    if not (0.0 <= float(oea_zo_marginal_prior_mix) <= 1.0):
        raise ValueError("oea_zo_marginal_prior_mix must be in [0,1].")
    if int(oea_zo_bilevel_iters) < 0:
        raise ValueError("oea_zo_bilevel_iters must be >= 0.")
    if float(oea_zo_bilevel_temp) <= 0.0:
        raise ValueError("oea_zo_bilevel_temp must be > 0.")
    if float(oea_zo_bilevel_step) < 0.0:
        raise ValueError("oea_zo_bilevel_step must be >= 0.")
    if not (0.0 < float(oea_zo_bilevel_coverage_target) <= 1.0):
        raise ValueError("oea_zo_bilevel_coverage_target must be in (0,1].")
    if float(oea_zo_bilevel_coverage_power) < 0.0:
        raise ValueError("oea_zo_bilevel_coverage_power must be >= 0.")
    if float(oea_zo_min_improvement) < 0.0:
        raise ValueError("oea_zo_min_improvement must be >= 0.")
    if not (0.0 <= float(oea_zo_holdout_fraction) < 1.0):
        raise ValueError("oea_zo_holdout_fraction must be in [0,1).")
    if oea_zo_warm_start not in {"none", "delta"}:
        raise ValueError("oea_zo_warm_start must be one of: 'none', 'delta'")
    if int(oea_zo_warm_iters) < 0:
        raise ValueError("oea_zo_warm_iters must be >= 0.")
    if float(oea_zo_fallback_min_marginal_entropy) < 0.0:
        raise ValueError("oea_zo_fallback_min_marginal_entropy must be >= 0.")
    if int(oea_zo_iters) < 0:
        raise ValueError("oea_zo_iters must be >= 0.")
    if float(oea_zo_lr) <= 0.0:
        raise ValueError("oea_zo_lr must be > 0.")
    if float(oea_zo_mu) <= 0.0:
        raise ValueError("oea_zo_mu must be > 0.")
    if int(oea_zo_k) < 1:
        raise ValueError("oea_zo_k must be >= 1.")
    if int(oea_zo_localmix_neighbors) < 0:
        raise ValueError("oea_zo_localmix_neighbors must be >= 0.")
    if float(oea_zo_localmix_self_bias) < 0.0:
        raise ValueError("oea_zo_localmix_self_bias must be >= 0.")
    if float(oea_zo_l2) < 0.0:
        raise ValueError("oea_zo_l2 must be >= 0.")
    if float(mm_safe_mdm_guard_threshold) >= 0.0 and not (0.0 <= float(mm_safe_mdm_guard_threshold) <= 1.0):
        raise ValueError("mm_safe_mdm_guard_threshold must be in [0,1] (or <0 to disable).")
    if float(mm_safe_mdm_min_pred_improve) < 0.0:
        raise ValueError("mm_safe_mdm_min_pred_improve must be >= 0.")
    if float(mm_safe_mdm_drift_delta) < 0.0:
        raise ValueError("mm_safe_mdm_drift_delta must be >= 0.")
    if float(si_subject_lambda) < 0.0:
        raise ValueError("si_subject_lambda must be >= 0.")
    if float(si_ridge) <= 0.0:
        raise ValueError("si_ridge must be > 0.")
    if int(si_proj_dim) < 0:
        raise ValueError("si_proj_dim must be >= 0 (0 means keep full dim).")
    if any(int(r) < 0 for r in si_chan_candidate_ranks):
        raise ValueError("si_chan_candidate_ranks must be all >= 0.")
    if any(float(lam) < 0.0 for lam in si_chan_candidate_lambdas):
        raise ValueError("si_chan_candidate_lambdas must be all >= 0.")

    diag_subjects_set = {int(s) for s in diagnostics_subjects} if diagnostics_subjects else set()

    # Optional: extra per-subject diagnostics for specific alignments.
    extra_rows: list[dict] | None = (
        [] if alignment in {"ea_si_chan_safe", "ea_si_chan_multi_safe", "ea_mm_safe", "ea_stack_multi_safe"} else None
    )

    def _rankdata(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(x.size, dtype=np.float64)
        return ranks

    # Fast path: subject-wise EA can be precomputed once.
    if alignment in {
        "ea",
        "ea_si",
        "ea_si_chan",
        "ea_si_chan_safe",
        "ea_si_chan_multi_safe",
        "ea_mm_safe",
        "ea_zo",
        "ea_si_zo",
    }:
        aligned: Dict[int, SubjectData] = {}
        for s, sd in subject_data.items():
            X_aligned = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
        subject_data = aligned
    elif alignment in {"rpa", "tsa"}:
        aligned = {}
        for s, sd in subject_data.items():
            X_aligned = LogEuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
        subject_data = aligned
    elif alignment == "ea_stack_multi_safe":
        # For stacked candidate selection we need both EA (anchor) and LEA/RPA aligned views.
        aligned_ea: Dict[int, SubjectData] = {}
        aligned_rpa: Dict[int, SubjectData] = {}
        for s, sd in subject_data.items():
            X_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            X_rpa = LogEuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned_ea[int(s)] = SubjectData(subject=int(s), X=X_ea, y=sd.y)
            aligned_rpa[int(s)] = SubjectData(subject=int(s), X=X_rpa, y=sd.y)
        subject_data = aligned_ea
        subject_data_rpa = aligned_rpa

    # Cache for expensive per-train-set computations (used by ea_si_chan_multi_safe).
    chan_bundle_cache: dict[tuple[int, ...], dict] = {}
    chan_candidate_grid: list[tuple[int, float]] = []
    if alignment in {"ea_si_chan_multi_safe", "ea_mm_safe"}:
        ranks = [int(r) for r in (list(si_chan_candidate_ranks) or [int(si_proj_dim)])]
        lambdas = [float(l) for l in (list(si_chan_candidate_lambdas) or [float(si_subject_lambda)])]
        seen: set[tuple[int, float]] = set()
        for r in ranks:
            for lam in lambdas:
                key = (int(r), float(lam))
                if key in seen:
                    continue
                seen.add(key)
                chan_candidate_grid.append(key)

    def _get_chan_bundle(train_subjects_subset: Sequence[int]) -> dict:
        """Return cached models for a given train-subject subset (used by ea_si_chan_multi_safe)."""

        key = tuple(sorted(int(s) for s in train_subjects_subset))
        if key in chan_bundle_cache:
            return chan_bundle_cache[key]

        X_train_parts = [subject_data[int(s)].X for s in key]
        y_train_parts = [subject_data[int(s)].y for s in key]
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        subj_train = np.concatenate(
            [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in key],
            axis=0,
        )

        bundle: dict = {"model_id": fit_csp_lda(X_train, y_train, n_components=n_components), "candidates": {}}
        n_channels = int(X_train.shape[1])

        # Candidate channel projectors + per-candidate CSP+LDA models.
        for r, lam in chan_candidate_grid:
            r = int(r)
            lam = float(lam)
            if r <= 0 or r >= n_channels:
                continue
            chan_params = ChannelProjectorParams(subject_lambda=float(lam), ridge=float(si_ridge), n_components=int(r))
            A = learn_subject_invariant_channel_projector(
                X=X_train,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )
            if np.allclose(A, np.eye(n_channels, dtype=np.float64), atol=1e-10):
                continue
            X_train_A = apply_spatial_transform(A, X_train)
            model_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)
            bundle["candidates"][(int(r), float(lam))] = {"A": A, "model": model_A, "rank": int(r), "lambda": float(lam)}

        chan_bundle_cache[key] = bundle
        return bundle

    # Cache for expensive per-train-set computations (used by ea_stack_multi_safe).
    stack_bundle_cache: dict[tuple[int, ...], dict] = {}
    stack_chan_candidate_grid: list[tuple[int, float]] = []
    if alignment == "ea_stack_multi_safe":
        ranks = [int(r) for r in (list(si_chan_candidate_ranks) or [int(si_proj_dim)])]
        lambdas = [float(l) for l in (list(si_chan_candidate_lambdas) or [float(si_subject_lambda)])]
        seen: set[tuple[int, float]] = set()
        for r in ranks:
            for lam in lambdas:
                key = (int(r), float(lam))
                if key in seen:
                    continue
                seen.add(key)
                stack_chan_candidate_grid.append(key)

    def _get_stack_bundle(train_subjects_subset: Sequence[int]) -> dict:
        """Return cached models for a given train-subject subset (used by ea_stack_multi_safe)."""

        if subject_data_rpa is None:
            raise RuntimeError("ea_stack_multi_safe requires precomputed subject_data_rpa.")

        key = tuple(sorted(int(s) for s in train_subjects_subset))
        if key in stack_bundle_cache:
            return stack_bundle_cache[key]

        # Anchor (EA) view.
        X_train_ea = np.concatenate([subject_data[int(s)].X for s in key], axis=0)
        y_train = np.concatenate([subject_data[int(s)].y for s in key], axis=0)
        subj_train = np.concatenate(
            [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in key],
            axis=0,
        )
        model_ea = fit_csp_lda(X_train_ea, y_train, n_components=n_components)

        # RPA/LEA view.
        X_train_rpa = np.concatenate([subject_data_rpa[int(s)].X for s in key], axis=0)
        model_rpa = fit_csp_lda(X_train_rpa, y_train, n_components=n_components)

        bundle: dict = {
            "subjects": key,
            "ea": {"model": model_ea},
            "rpa": {"model": model_rpa},
            "chan": {"candidates": {}},
        }

        # Channel projector candidates (learned on EA view).
        n_channels = int(X_train_ea.shape[1])
        for r, lam in stack_chan_candidate_grid:
            r = int(r)
            lam = float(lam)
            if r <= 0 or r >= n_channels:
                continue
            chan_params = ChannelProjectorParams(subject_lambda=float(lam), ridge=float(si_ridge), n_components=int(r))
            A = learn_subject_invariant_channel_projector(
                X=X_train_ea,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )
            if np.allclose(A, np.eye(n_channels, dtype=np.float64), atol=1e-10):
                continue
            X_train_A = apply_spatial_transform(A, X_train_ea)
            model_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)
            bundle["chan"]["candidates"][(int(r), float(lam))] = {
                "A": A,
                "model": model_A,
                "rank": int(r),
                "lambda": float(lam),
            }

        stack_bundle_cache[key] = bundle
        return bundle

    for test_subject in subjects:
        model: TrainedModel | None = None
        train_subjects = [s for s in subjects if s != test_subject]
        do_diag = diagnostics_dir is not None and int(test_subject) in diag_subjects_set
        zo_diag: dict | None = None
        z_test_base: np.ndarray | None = None

        # Build per-fold aligned train/test data if needed.
        if alignment in {"none", "ea", "rpa"}:
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
        elif alignment in {"riemann_mdm", "rpa_mdm", "rpa_rot_mdm"}:
            # pyRiemann baselines on SPD covariances (Riemannian TL / Procrustes family).
            #
            # Note: this branch operates on covariance matrices directly (not on time series),
            # so `X_test` is replaced by (n_trials,C,C) SPD matrices.
            from pyriemann.classification import MDM
            from pyriemann.transfer import TLCenter, TLRotate, TLStretch, encode_domains

            X_test_raw = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = []
            y_train_parts = []
            dom_train_parts = []
            for s in train_subjects:
                sd = subject_data[int(s)]
                X_train_parts.append(sd.X)
                y_train_parts.append(sd.y)
                dom_train_parts.append(np.full(sd.y.shape[0], f"src_{int(s)}", dtype=object))
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            dom_train = np.concatenate(dom_train_parts, axis=0)

            dom_test = np.full(y_test.shape[0], "target", dtype=object)

            cov_train = covariances_from_epochs(X_train, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
            cov_test = covariances_from_epochs(X_test_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))

            if alignment == "riemann_mdm":
                model = MDM(metric="riemann")
                model.fit(cov_train, y_train)
                X_test = cov_test
            else:
                # Center + stretch (RPA without rotation) using both source and target (unlabeled) covariances.
                y_dummy = np.full(y_test.shape[0], str(class_order[0]), dtype=object)
                cov_all = np.concatenate([cov_train, cov_test], axis=0)
                y_all = np.concatenate([y_train, y_dummy], axis=0)
                dom_all = np.concatenate([dom_train, dom_test], axis=0)
                _, y_enc = encode_domains(cov_all, y_all, dom_all)

                center = TLCenter(target_domain="target", metric="riemann")
                cov_centered = center.fit_transform(cov_all, y_enc)

                stretch = TLStretch(target_domain="target", centered_data=True, metric="riemann")
                cov_stretched = stretch.fit_transform(cov_centered, y_enc)

                cov_src = cov_stretched[: cov_train.shape[0]]
                cov_tgt = cov_stretched[cov_train.shape[0] :]

                if alignment == "rpa_rot_mdm":
                    # One-step pseudo-label rotation (RPA full): predict pseudo labels on target then rotate sources.
                    base = MDM(metric="riemann")
                    base.fit(cov_src, y_train)
                    y_pseudo = np.asarray(base.predict(cov_tgt))
                    y_all2 = np.concatenate([y_train, y_pseudo], axis=0)
                    _, y_enc2 = encode_domains(cov_stretched, y_all2, dom_all)

                    rotate = TLRotate(target_domain="target", metric="euclid", n_jobs=1)
                    cov_rot = rotate.fit_transform(cov_stretched, y_enc2)
                    cov_src = cov_rot[: cov_train.shape[0]]
                    cov_tgt = cov_rot[cov_train.shape[0] :]

                model = MDM(metric="riemann")
                model.fit(cov_src, y_train)
                X_test = cov_tgt
        elif alignment == "tsa":
            # Tangent-space alignment (TSA) using pseudo-label anchors in the LEA/RPA-whitened space.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
            q_tsa = _compute_tsa_target_rotation(
                z_train=X_train,
                y_train=y_train,
                z_target=X_test,
                model=model,
                class_order=tuple([str(c) for c in class_order]),
                pseudo_mode=str(oea_pseudo_mode),
                pseudo_iters=int(max(0, oea_pseudo_iters)),
                q_blend=float(oea_q_blend),
                pseudo_confidence=float(oea_pseudo_confidence),
                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                pseudo_balance=bool(oea_pseudo_balance),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
            )
            X_test = apply_spatial_transform(q_tsa, X_test)
        elif alignment == "ea_stack_multi_safe":
            # Stacked multi-candidate selection with safe fallback to the EA anchor.
            #
            # Candidate families (per fold):
            # - EA anchor (A=I, using EA-whitened data)
            # - RPA (LEA-whitened data)
            # - TSA (LEA-whitened + TSA target rotation)
            # - EA-SI-CHAN (rank-deficient channel projectors on EA-whitened data)
            if subject_data_rpa is None:
                raise RuntimeError("ea_stack_multi_safe requires subject_data_rpa.")

            class_labels = tuple([str(c) for c in class_order])
            selector = str(oea_zo_selector)
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}

            outer_bundle = _get_stack_bundle(train_subjects)
            model_ea = outer_bundle["ea"]["model"]
            model_rpa = outer_bundle["rpa"]["model"]
            chan_outer: dict = dict(outer_bundle.get("chan", {}).get("candidates", {}))
            # For reporting only (n_train) and consistency with other branches.
            y_train = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
            X_train = np.empty((0,) + tuple(subject_data[int(train_subjects[0])].X.shape[1:]), dtype=np.float64)

            # Per-fold calibration on pseudo-target subjects (source-only).
            cert = None
            guard = None
            ridge_train_spearman = float("nan")
            ridge_train_pearson = float("nan")
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])
                return rec

            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_ridge_rows: List[np.ndarray] = []
                y_ridge_rows: List[float] = []
                X_guard_rows: List[np.ndarray] = []
                y_guard_rows: List[int] = []
                improve_guard_rows: List[float] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    inner_bundle = _get_stack_bundle(inner_train)

                    # Anchor (EA) predictions on the pseudo-target.
                    z_p_ea = subject_data[int(pseudo_t)].X
                    y_p = subject_data[int(pseudo_t)].y
                    m_ea = inner_bundle["ea"]["model"]
                    p_id = _reorder_proba_columns(m_ea.predict_proba(z_p_ea), m_ea.classes_, list(class_labels))
                    acc_id = float(accuracy_score(y_p, np.asarray(m_ea.predict(z_p_ea))))

                    # RPA candidate.
                    z_p_rpa = subject_data_rpa[int(pseudo_t)].X
                    m_rpa = inner_bundle["rpa"]["model"]
                    p_rpa = _reorder_proba_columns(m_rpa.predict_proba(z_p_rpa), m_rpa.classes_, list(class_labels))
                    acc_rpa = float(accuracy_score(y_p, np.asarray(m_rpa.predict(z_p_rpa))))
                    improve_rpa = float(acc_rpa - acc_id)
                    rec_rpa = _record_for_candidate(p_id=p_id, p_c=p_rpa)
                    feats_rpa, names = candidate_features_from_record(rec_rpa, n_classes=len(class_labels), include_pbar=True)
                    if feat_names is None:
                        feat_names = names
                    if use_ridge:
                        X_ridge_rows.append(feats_rpa)
                        y_ridge_rows.append(improve_rpa)
                    if use_guard:
                        X_guard_rows.append(feats_rpa)
                        y_guard_rows.append(1 if improve_rpa >= float(oea_zo_calib_guard_margin) else 0)
                        improve_guard_rows.append(improve_rpa)

                    # TSA candidate (built on the RPA/LEA view).
                    try:
                        z_tr_rpa = np.concatenate([subject_data_rpa[int(s)].X for s in inner_train], axis=0)
                        y_tr = np.concatenate([subject_data[int(s)].y for s in inner_train], axis=0)
                        q_tsa = _compute_tsa_target_rotation(
                            z_train=z_tr_rpa,
                            y_train=y_tr,
                            z_target=z_p_rpa,
                            model=m_rpa,
                            class_order=class_labels,
                            pseudo_mode=str(oea_pseudo_mode),
                            pseudo_iters=int(max(0, oea_pseudo_iters)),
                            q_blend=float(oea_q_blend),
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                            eps=float(oea_eps),
                            shrinkage=float(oea_shrinkage),
                        )
                        z_p_tsa = apply_spatial_transform(q_tsa, z_p_rpa)
                        p_tsa = _reorder_proba_columns(
                            m_rpa.predict_proba(z_p_tsa), m_rpa.classes_, list(class_labels)
                        )
                        acc_tsa = float(accuracy_score(y_p, np.asarray(m_rpa.predict(z_p_tsa))))
                        improve_tsa = float(acc_tsa - acc_id)
                        rec_tsa = _record_for_candidate(p_id=p_id, p_c=p_tsa)
                        feats_tsa, _ = candidate_features_from_record(
                            rec_tsa, n_classes=len(class_labels), include_pbar=True
                        )
                        if use_ridge:
                            X_ridge_rows.append(feats_tsa)
                            y_ridge_rows.append(improve_tsa)
                        if use_guard:
                            X_guard_rows.append(feats_tsa)
                            y_guard_rows.append(1 if improve_tsa >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows.append(improve_tsa)
                    except Exception:
                        pass

                    # Channel projector candidates (EA view).
                    cand_inner_chan: dict = dict(inner_bundle.get("chan", {}).get("candidates", {}))
                    for cand_key, info in cand_inner_chan.items():
                        A = info["A"]
                        m_A = info["model"]
                        z_p_A = apply_spatial_transform(A, z_p_ea)
                        p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, list(class_labels))
                        acc_A = float(accuracy_score(y_p, np.asarray(m_A.predict(z_p_A))))
                        improve_A = float(acc_A - acc_id)
                        rec_A = _record_for_candidate(p_id=p_id, p_c=p_A)
                        feats_A, _ = candidate_features_from_record(
                            rec_A, n_classes=len(class_labels), include_pbar=True
                        )
                        if use_ridge:
                            X_ridge_rows.append(feats_A)
                            y_ridge_rows.append(improve_A)
                        if use_guard:
                            X_guard_rows.append(feats_A)
                            y_guard_rows.append(1 if improve_A >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows.append(improve_A)

                if use_ridge and X_ridge_rows and feat_names is not None:
                    X_ridge = np.vstack(X_ridge_rows)
                    y_ridge = np.asarray(y_ridge_rows, dtype=np.float64)
                    cert = train_ridge_certificate(
                        X_ridge,
                        y_ridge,
                        feature_names=feat_names,
                        alpha=float(oea_zo_calib_ridge_alpha),
                    )
                    try:
                        pred = np.asarray(cert.predict_accuracy(X_ridge), dtype=np.float64).reshape(-1)
                        if y_ridge.size >= 2:
                            ridge_train_pearson = float(np.corrcoef(pred, y_ridge)[0, 1])
                            ridge_train_spearman = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                    except Exception:
                        pass

                if use_guard and X_guard_rows and feat_names is not None:
                    X_guard = np.vstack(X_guard_rows)
                    y_guard = np.asarray(y_guard_rows, dtype=int)
                    if len(np.unique(y_guard)) >= 2:
                        guard = train_logistic_guard(
                            X_guard,
                            y_guard,
                            feature_names=feat_names,
                            c=float(oea_zo_calib_guard_c),
                        )
                        try:
                            p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                            improve_train = np.asarray(improve_guard_rows, dtype=np.float64).reshape(-1)
                            guard_train_auc = float(roc_auc_score(y_guard, p_train))
                            if improve_train.size == p_train.size and improve_train.size >= 2:
                                guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                                guard_train_spearman = float(
                                    np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1]
                                )
                        except Exception:
                            pass

            # Build target-subject candidate records (unlabeled).
            X_test_ea = subject_data[int(test_subject)].X
            y_test = subject_data[int(test_subject)].y
            X_test_rpa = subject_data_rpa[int(test_subject)].X

            p_id_t = _reorder_proba_columns(model_ea.predict_proba(X_test_ea), model_ea.classes_, list(class_labels))
            rec_id = _record_for_candidate(p_id=p_id_t, p_c=p_id_t)
            rec_id["kind"] = "identity"
            rec_id["cand_family"] = "ea"
            records: list[dict] = [rec_id]

            # RPA candidate.
            p_rpa_t = _reorder_proba_columns(model_rpa.predict_proba(X_test_rpa), model_rpa.classes_, list(class_labels))
            rec_rpa_t = _record_for_candidate(p_id=p_id_t, p_c=p_rpa_t)
            rec_rpa_t["cand_family"] = "rpa"
            records.append(rec_rpa_t)

            # TSA candidate.
            try:
                z_tr_rpa = np.concatenate([subject_data_rpa[int(s)].X for s in train_subjects], axis=0)
                y_tr = np.concatenate([subject_data[int(s)].y for s in train_subjects], axis=0)
                q_tsa = _compute_tsa_target_rotation(
                    z_train=z_tr_rpa,
                    y_train=y_tr,
                    z_target=X_test_rpa,
                    model=model_rpa,
                    class_order=class_labels,
                    pseudo_mode=str(oea_pseudo_mode),
                    pseudo_iters=int(max(0, oea_pseudo_iters)),
                    q_blend=float(oea_q_blend),
                    pseudo_confidence=float(oea_pseudo_confidence),
                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                    pseudo_balance=bool(oea_pseudo_balance),
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                )
                X_test_tsa = apply_spatial_transform(q_tsa, X_test_rpa)
                p_tsa_t = _reorder_proba_columns(
                    model_rpa.predict_proba(X_test_tsa), model_rpa.classes_, list(class_labels)
                )
                rec_tsa_t = _record_for_candidate(p_id=p_id_t, p_c=p_tsa_t)
                rec_tsa_t["cand_family"] = "tsa"
                rec_tsa_t["tsa_q_blend"] = float(oea_q_blend)
                records.append(rec_tsa_t)
            except Exception:
                X_test_tsa = None

            # Channel projector candidates.
            for cand_key, info in chan_outer.items():
                A = info["A"]
                m_A = info["model"]
                X_test_A = apply_spatial_transform(A, X_test_ea)
                p_A_t = _reorder_proba_columns(m_A.predict_proba(X_test_A), m_A.classes_, list(class_labels))
                rec = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                rec["cand_family"] = "chan"
                rec["cand_key"] = cand_key
                rec["cand_rank"] = float(info.get("rank", float("nan")))
                rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                records.append(rec)

            selected = rec_id
            if selector == "calibrated_ridge_guard" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "calibrated_ridge" and cert is not None:
                selected = select_by_predicted_improvement(
                    records,
                    cert=cert,
                    n_classes=len(class_labels),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set="base",
                )
            elif selector == "calibrated_guard" and guard is not None:
                selected = select_by_guarded_objective(
                    records,
                    guard=guard,
                    n_classes=len(class_labels),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "objective":
                best = min(records, key=lambda r: float(r.get("score", r.get("objective_base", 0.0))))
                selected = best

            if (
                float(oea_zo_fallback_min_marginal_entropy) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and float(selected.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
            ):
                selected = rec_id

            accept = str(selected.get("kind", "")) != "identity"
            sel_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            sel_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            sel_family = str(selected.get("cand_family", "ea"))
            sel_rank = float(selected.get("cand_rank", float("nan")))
            sel_lam = float(selected.get("cand_lambda", float("nan")))

            # Apply the selected candidate.
            if not accept or sel_family == "ea":
                model = model_ea
                X_test = X_test_ea
            elif sel_family == "rpa":
                model = model_rpa
                X_test = X_test_rpa
            elif sel_family == "tsa" and X_test_tsa is not None:
                model = model_rpa
                X_test = X_test_tsa
            elif sel_family == "chan":
                sel_key = selected.get("cand_key", None)
                if sel_key in chan_outer:
                    A_sel = chan_outer[sel_key]["A"]
                    model = chan_outer[sel_key]["model"]
                    X_test = apply_spatial_transform(A_sel, X_test_ea)
                else:
                    model = model_ea
                    X_test = X_test_ea
            else:
                model = model_ea
                X_test = X_test_ea

            # Analysis-only: compute true improvement for the selected candidate (not used in selection).
            try:
                acc_id_t = float(accuracy_score(y_test, np.asarray(model_ea.predict(X_test_ea))))
            except Exception:
                acc_id_t = float("nan")
            try:
                acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
            except Exception:
                acc_sel_t = float("nan")
            improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

            if extra_rows is not None:
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "stack_multi_accept": int(bool(accept)),
                        "stack_multi_family": str(sel_family),
                        "stack_multi_guard_pos": float(sel_guard_pos),
                        "stack_multi_ridge_pred_improve": float(sel_ridge_pred),
                        "stack_multi_acc_anchor": float(acc_id_t),
                        "stack_multi_acc_selected": float(acc_sel_t),
                        "stack_multi_improve": float(improve_t),
                        "stack_multi_sel_rank": float(sel_rank),
                        "stack_multi_sel_lambda": float(sel_lam),
                        "stack_multi_ridge_train_spearman": float(ridge_train_spearman),
                        "stack_multi_ridge_train_pearson": float(ridge_train_pearson),
                        "stack_multi_guard_train_auc": float(guard_train_auc),
                        "stack_multi_guard_train_spearman": float(guard_train_spearman),
                        "stack_multi_guard_train_pearson": float(guard_train_pearson),
                    }
                )
        elif alignment == "ea_si":
            # Train on EA-whitened data with a subject-invariant feature projector (Route B),
            # then evaluate directly (no test-time Q_t optimization).
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )
            # Fit CSP once to define feature space, learn projection, then train LDA on projected features.
            from mne.decoding import CSP  # local import to keep module import cost low

            csp = CSP(n_components=int(n_components))
            csp.fit(X_train, y_train)
            feats = np.asarray(csp.transform(X_train), dtype=np.float64)
            proj_params = HSICProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            mean_f, W = learn_hsic_subject_invariant_projector(
                X=feats,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                params=proj_params,
            )
            projector = CenteredLinearProjector(mean=mean_f, W=W)
            model = fit_csp_projected_lda(
                X_train=X_train,
                y_train=y_train,
                projector=projector,
                csp=csp,
                n_components=n_components,
            )
        elif alignment == "ea_si_chan":
            # Channel-space subject-invariant projection (pre-CSP): learn a low-rank projector A (C×C),
            # apply it to both train/test, then train a standard CSP+LDA model.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            chan_params = ChannelProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            A = learn_subject_invariant_channel_projector(
                X=X_train,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )
            X_train = apply_spatial_transform(A, X_train)
            X_test = apply_spatial_transform(A, X_test)
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
        elif alignment == "ea_si_chan_safe":
            # EA anchor vs. channel projector candidate (binary choice). Use a fold-local calibrated guard
            # trained on pseudo-target subjects; otherwise fallback to EA (identity).
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            chan_params = ChannelProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            A = learn_subject_invariant_channel_projector(
                X=X_train,
                y=y_train,
                subjects=subj_train,
                class_order=tuple([str(c) for c in class_order]),
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                params=chan_params,
            )

            # Train both models on the same source fold (EA anchor vs projected).
            model_id = fit_csp_lda(X_train, y_train, n_components=n_components)
            X_train_A = apply_spatial_transform(A, X_train)
            model_A = fit_csp_lda(X_train_A, y_train, n_components=n_components)

            # Calibrate a per-fold guard using pseudo-targets from the source subjects.
            rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
            calib_subjects = list(train_subjects)
            if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                rng.shuffle(calib_subjects)
                calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

            X_guard_rows: List[np.ndarray] = []
            y_guard_rows: List[int] = []
            improve_rows: List[float] = []
            feat_names: tuple[str, ...] | None = None

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                return rec

            for pseudo_t in calib_subjects:
                inner_train = [s for s in train_subjects if s != pseudo_t]
                if len(inner_train) < 2:
                    continue

                X_inner = np.concatenate([subject_data[s].X for s in inner_train], axis=0)
                y_inner = np.concatenate([subject_data[s].y for s in inner_train], axis=0)
                subj_inner = np.concatenate(
                    [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in inner_train],
                    axis=0,
                )

                A_inner = learn_subject_invariant_channel_projector(
                    X=X_inner,
                    y=y_inner,
                    subjects=subj_inner,
                    class_order=tuple([str(c) for c in class_order]),
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    params=chan_params,
                )
                m_id = fit_csp_lda(X_inner, y_inner, n_components=n_components)
                X_inner_A = apply_spatial_transform(A_inner, X_inner)
                m_A = fit_csp_lda(X_inner_A, y_inner, n_components=n_components)

                z_p = subject_data[int(pseudo_t)].X
                y_p = subject_data[int(pseudo_t)].y
                p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, list(class_order))
                p_A = _reorder_proba_columns(
                    m_A.predict_proba(apply_spatial_transform(A_inner, z_p)),
                    m_A.classes_,
                    list(class_order),
                )

                yp_id = np.asarray(m_id.predict(z_p))
                yp_A = np.asarray(m_A.predict(apply_spatial_transform(A_inner, z_p)))
                acc_id = float(accuracy_score(y_p, yp_id))
                acc_A = float(accuracy_score(y_p, yp_A))
                improve = float(acc_A - acc_id)

                rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                feats_vec, names = candidate_features_from_record(rec, n_classes=len(class_order), include_pbar=True)
                if feat_names is None:
                    feat_names = names
                X_guard_rows.append(feats_vec)
                y_guard_rows.append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                improve_rows.append(float(improve))

            guard = None
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")
            if X_guard_rows and feat_names is not None:
                X_guard = np.vstack(X_guard_rows)
                y_guard = np.asarray(y_guard_rows, dtype=int)
                # Need both classes to train.
                if len(np.unique(y_guard)) >= 2:
                    guard = train_logistic_guard(
                        X_guard,
                        y_guard,
                        feature_names=feat_names,
                        c=float(oea_zo_calib_guard_c),
                    )
                    try:
                        p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                        improve_train = np.asarray(improve_rows, dtype=np.float64).reshape(-1)
                        guard_train_auc = float(roc_auc_score(y_guard, p_train))
                        if improve_train.size >= 2:
                            guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                            guard_train_spearman = float(
                                np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1]
                            )
                    except Exception:
                        pass

            # Decide whether to accept the projected candidate on the target subject.
            p_id_t = _reorder_proba_columns(model_id.predict_proba(X_test), model_id.classes_, list(class_order))
            X_test_A = apply_spatial_transform(A, X_test)
            p_A_t = _reorder_proba_columns(model_A.predict_proba(X_test_A), model_A.classes_, list(class_order))

            accept = False
            pos = float("nan")
            acc_id_t = float("nan")
            acc_A_t = float("nan")
            improve_t = float("nan")
            if guard is not None:
                rec_t = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                feats_t, _names = candidate_features_from_record(rec_t, n_classes=len(class_order), include_pbar=True)
                pos = float(guard.predict_pos_proba(feats_t)[0])
                accept = pos >= float(oea_zo_calib_guard_threshold)

                # Optional hard drift guard.
                if str(oea_zo_drift_mode) == "hard" and float(oea_zo_drift_delta) > 0.0:
                    drift_mean = float(np.mean(_drift_vec(p_id_t, p_A_t)))
                    if drift_mean > float(oea_zo_drift_delta):
                        accept = False

                # Optional marginal-entropy fallback.
                if float(oea_zo_fallback_min_marginal_entropy) > 0.0 and accept:
                    p_bar = np.mean(np.clip(p_A_t, 1e-12, 1.0), axis=0)
                    p_bar = p_bar / float(np.sum(p_bar))
                    ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))
                    if ent_bar < float(oea_zo_fallback_min_marginal_entropy):
                        accept = False

            # Analysis-only: compute true improvement on the target subject (not used in selection).
            try:
                yp_id_t = np.asarray(model_id.predict(X_test))
                yp_A_t = np.asarray(model_A.predict(X_test_A))
                acc_id_t = float(accuracy_score(y_test, yp_id_t))
                acc_A_t = float(accuracy_score(y_test, yp_A_t))
                improve_t = float(acc_A_t - acc_id_t)
            except Exception:
                pass

            if extra_rows is not None:
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "chan_safe_accept": int(bool(accept)),
                        "chan_safe_guard_pos": float(pos),
                        "chan_safe_acc_anchor": float(acc_id_t),
                        "chan_safe_acc_candidate": float(acc_A_t),
                        "chan_safe_improve": float(improve_t),
                        "chan_safe_guard_train_auc": float(guard_train_auc),
                        "chan_safe_guard_train_spearman": float(guard_train_spearman),
                        "chan_safe_guard_train_pearson": float(guard_train_pearson),
                    }
                )

            # Fallback to EA unless confidently accepted.
            if accept:
                model = model_A
                X_test = X_test_A
            else:
                model = model_id
                X_test = X_test
        elif alignment == "ea_si_chan_multi_safe":
            # Multi-candidate EA-SI-CHAN with calibrated selection (ridge/guard) and safe fallback to EA anchor.
            #
            # Candidate set includes:
            # - identity anchor (A=I)
            # - multiple channel projectors A=QQᵀ learned with different (rank, λ) on the source subjects
            #
            # Selection is performed on the target subject without using target labels.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y
            X_test_raw = X_test

            # For reporting only (n_train) and consistency with other branches.
            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            selector = str(oea_zo_selector)
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}

            outer_bundle = _get_chan_bundle(train_subjects)
            model_id = outer_bundle["model_id"]
            candidates_outer: dict = dict(outer_bundle.get("candidates", {}))

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                # Convenience aliases for selectors that expect `score`/`objective`.
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])
                return rec

            # Calibrate ridge/guard on pseudo-target subjects (source-only; per outer fold).
            cert = None
            guard = None
            ridge_train_spearman = float("nan")
            ridge_train_pearson = float("nan")
            guard_train_auc = float("nan")
            guard_train_spearman = float("nan")
            guard_train_pearson = float("nan")

            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_ridge_rows: List[np.ndarray] = []
                y_ridge_rows: List[float] = []
                X_guard_rows: List[np.ndarray] = []
                y_guard_rows: List[int] = []
                improve_guard_rows: List[float] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    inner_bundle = _get_chan_bundle(inner_train)
                    m_id = inner_bundle["model_id"]
                    cand_inner: dict = dict(inner_bundle.get("candidates", {}))
                    if not cand_inner:
                        continue

                    z_p = subject_data[int(pseudo_t)].X
                    y_p = subject_data[int(pseudo_t)].y
                    p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, list(class_order))
                    yp_id = np.asarray(m_id.predict(z_p))
                    acc_id = float(accuracy_score(y_p, yp_id))

                    for cand_key, info in cand_inner.items():
                        A = info["A"]
                        m_A = info["model"]
                        z_p_A = apply_spatial_transform(A, z_p)
                        p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, list(class_order))
                        yp_A = np.asarray(m_A.predict(z_p_A))
                        acc_A = float(accuracy_score(y_p, yp_A))
                        improve = float(acc_A - acc_id)

                        rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                        feats_vec, names = candidate_features_from_record(
                            rec, n_classes=len(class_order), include_pbar=True
                        )
                        if feat_names is None:
                            feat_names = names
                        if use_ridge:
                            X_ridge_rows.append(feats_vec)
                            y_ridge_rows.append(float(improve))
                        if use_guard:
                            X_guard_rows.append(feats_vec)
                            y_guard_rows.append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows.append(float(improve))

                if use_ridge and X_ridge_rows and feat_names is not None:
                    X_ridge = np.vstack(X_ridge_rows)
                    y_ridge = np.asarray(y_ridge_rows, dtype=np.float64)
                    cert = train_ridge_certificate(
                        X_ridge,
                        y_ridge,
                        feature_names=feat_names,
                        alpha=float(oea_zo_calib_ridge_alpha),
                    )
                    try:
                        pred = np.asarray(cert.predict_accuracy(X_ridge), dtype=np.float64).reshape(-1)
                        if y_ridge.size >= 2:
                            ridge_train_pearson = float(np.corrcoef(pred, y_ridge)[0, 1])
                            ridge_train_spearman = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                    except Exception:
                        pass

                if use_guard and X_guard_rows and feat_names is not None:
                    X_guard = np.vstack(X_guard_rows)
                    y_guard = np.asarray(y_guard_rows, dtype=int)
                    if len(np.unique(y_guard)) >= 2:
                        guard = train_logistic_guard(
                            X_guard,
                            y_guard,
                            feature_names=feat_names,
                            c=float(oea_zo_calib_guard_c),
                        )
                        try:
                            p_train = np.asarray(guard.predict_pos_proba(X_guard), dtype=np.float64).reshape(-1)
                            improve_train = np.asarray(improve_guard_rows, dtype=np.float64).reshape(-1)
                            guard_train_auc = float(roc_auc_score(y_guard, p_train))
                            if improve_train.size == p_train.size and improve_train.size >= 2:
                                guard_train_pearson = float(np.corrcoef(p_train, improve_train)[0, 1])
                                guard_train_spearman = float(
                                    np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1]
                                )
                        except Exception:
                            pass

            # Build candidate records on the target subject (unlabeled).
            p_id_t = _reorder_proba_columns(model_id.predict_proba(X_test_raw), model_id.classes_, list(class_order))
            rec_id = _record_for_candidate(p_id=p_id_t, p_c=p_id_t)
            rec_id["kind"] = "identity"
            rec_id["cand_key"] = None
            records: list[dict] = [rec_id]

            for cand_key, info in candidates_outer.items():
                A = info["A"]
                m_A = info["model"]
                X_test_A = apply_spatial_transform(A, X_test_raw)
                p_A_t = _reorder_proba_columns(m_A.predict_proba(X_test_A), m_A.classes_, list(class_order))
                rec = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                rec["kind"] = "candidate"
                rec["cand_key"] = cand_key
                rec["cand_rank"] = float(info.get("rank", float("nan")))
                rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                records.append(rec)

            selected = rec_id
            if selector == "calibrated_ridge_guard" and cert is not None and guard is not None:
                selected = select_by_guarded_predicted_improvement(
                    records,
                    cert=cert,
                    guard=guard,
                    n_classes=len(class_order),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "calibrated_ridge" and cert is not None:
                selected = select_by_predicted_improvement(
                    records,
                    cert=cert,
                    n_classes=len(class_order),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    feature_set="base",
                )
            elif selector == "calibrated_guard" and guard is not None:
                selected = select_by_guarded_objective(
                    records,
                    guard=guard,
                    n_classes=len(class_order),
                    threshold=float(oea_zo_calib_guard_threshold),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                )
            elif selector == "objective":
                best = min(records, key=lambda r: float(r.get("score", r.get("objective_base", 0.0))))
                selected = best

            # Optional marginal-entropy fallback (unlabeled safety valve).
            if (
                float(oea_zo_fallback_min_marginal_entropy) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and float(selected.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
            ):
                selected = rec_id

            # Apply selection.
            accept = str(selected.get("kind", "")) != "identity"
            sel_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            sel_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            sel_key = selected.get("cand_key", None)
            sel_rank = float(selected.get("cand_rank", float("nan")))
            sel_lam = float(selected.get("cand_lambda", float("nan")))

            if accept and sel_key in candidates_outer:
                A_sel = candidates_outer[sel_key]["A"]
                model = candidates_outer[sel_key]["model"]
                X_test = apply_spatial_transform(A_sel, X_test_raw)
            else:
                model = model_id
                X_test = X_test_raw

            # Analysis-only: compute true improvement for the selected transform (not used in selection).
            try:
                acc_id_t = float(accuracy_score(y_test, np.asarray(model_id.predict(X_test_raw))))
            except Exception:
                acc_id_t = float("nan")
            try:
                acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
            except Exception:
                acc_sel_t = float("nan")
            improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

            if extra_rows is not None:
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "chan_multi_accept": int(bool(accept)),
                        "chan_multi_guard_pos": float(sel_guard_pos),
                        "chan_multi_ridge_pred_improve": float(sel_ridge_pred),
                        "chan_multi_acc_anchor": float(acc_id_t),
                        "chan_multi_acc_selected": float(acc_sel_t),
                        "chan_multi_improve": float(improve_t),
                        "chan_multi_sel_rank": float(sel_rank),
                        "chan_multi_sel_lambda": float(sel_lam),
                        "chan_multi_ridge_train_spearman": float(ridge_train_spearman),
                        "chan_multi_ridge_train_pearson": float(ridge_train_pearson),
                        "chan_multi_guard_train_auc": float(guard_train_auc),
                        "chan_multi_guard_train_spearman": float(guard_train_spearman),
                        "chan_multi_guard_train_pearson": float(guard_train_pearson),
                    }
                )
        elif alignment == "ea_mm_safe":
            # Cross-model multi-candidate selection with calibrated selection (ridge/guard)
            # and safe fallback to the EA anchor.
            #
            # Candidate set includes:
            # - EA anchor (CSP+LDA on EA-whitened time series)
            # - EA-SI-CHAN candidates (rank-deficient channel projectors + CSP+LDA)
            # - MDM(RPA) candidate (TLCenter+TLStretch on SPD covariances + MDM classifier)
            #
            # Selection is performed on the target subject without using target labels.
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y
            X_test_raw = X_test

            # For reporting only (n_train) and consistency with other branches.
            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            selector = str(oea_zo_selector)
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}

            outer_bundle = _get_chan_bundle(train_subjects)
            model_id = outer_bundle["model_id"]
            candidates_outer: dict = dict(outer_bundle.get("candidates", {}))

            class_labels = list([str(c) for c in class_order])

            def _row_entropy(p: np.ndarray) -> np.ndarray:
                p = np.asarray(p, dtype=np.float64)
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p, axis=1, keepdims=True)
                return -np.sum(p * np.log(p), axis=1)

            def _drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
                p0 = np.asarray(p0, dtype=np.float64)
                p1 = np.asarray(p1, dtype=np.float64)
                p0 = np.clip(p0, 1e-12, 1.0)
                p1 = np.clip(p1, 1e-12, 1.0)
                p0 = p0 / np.sum(p0, axis=1, keepdims=True)
                p1 = p1 / np.sum(p1, axis=1, keepdims=True)
                return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

            def _safe_float_local(x, default: float = 0.0) -> float:
                try:
                    v = float(x)
                except Exception:
                    return float(default)
                if not np.isfinite(v):
                    return float(default)
                return float(v)

            def _record_for_candidate(*, p_id: np.ndarray, p_c: np.ndarray) -> dict:
                p_c = np.asarray(p_c, dtype=np.float64)
                p_bar = np.mean(np.clip(p_c, 1e-12, 1.0), axis=0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent = _row_entropy(p_c)
                ent_bar = float(-np.sum(p_bar * np.log(np.clip(p_bar, 1e-12, 1.0))))

                d = _drift_vec(p_id, p_c)
                rec = {
                    "kind": "candidate",
                    "objective_base": float(np.mean(ent)),
                    "pen_marginal": 0.0,
                    "mean_entropy": float(np.mean(ent)),
                    "entropy_bar": float(ent_bar),
                    "drift_best": float(np.mean(d)),
                    "drift_best_std": float(np.std(d)),
                    "drift_best_q90": float(np.quantile(d, 0.90)),
                    "drift_best_q95": float(np.quantile(d, 0.95)),
                    "drift_best_max": float(np.max(d)),
                    "drift_best_tail_frac": float(np.mean(d > float(oea_zo_drift_delta)))
                    if float(oea_zo_drift_delta) > 0.0
                    else 0.0,
                    "p_bar_full": p_bar.astype(np.float64),
                    "q_bar": np.zeros_like(p_bar),
                }
                # Convenience aliases for selectors that expect `score`/`objective`.
                rec["objective"] = float(rec["objective_base"])
                rec["score"] = float(rec["objective_base"])
                return rec

            def _fit_rpa_mdm(
                *,
                inner_train_subjects: Sequence[int],
                target_subject: int,
            ):
                from pyriemann.classification import MDM
                from pyriemann.transfer import TLCenter, TLStretch, encode_domains

                X_train_parts_raw: list[np.ndarray] = []
                y_train_parts_raw: list[np.ndarray] = []
                dom_train_parts: list[np.ndarray] = []
                for s in inner_train_subjects:
                    sd = subject_data_raw[int(s)]
                    X_train_parts_raw.append(sd.X)
                    y_train_parts_raw.append(sd.y)
                    dom_train_parts.append(np.full(sd.y.shape[0], f"src_{int(s)}", dtype=object))
                X_tr_raw = np.concatenate(X_train_parts_raw, axis=0)
                y_tr = np.concatenate(y_train_parts_raw, axis=0)
                dom_train = np.concatenate(dom_train_parts, axis=0)

                z_t_raw = subject_data_raw[int(target_subject)].X

                cov_train = covariances_from_epochs(X_tr_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))
                cov_test = covariances_from_epochs(z_t_raw, eps=float(oea_eps), shrinkage=float(oea_shrinkage))

                dom_test = np.full(cov_test.shape[0], "target", dtype=object)
                y_dummy = np.full(cov_test.shape[0], str(class_order[0]), dtype=object)
                cov_all = np.concatenate([cov_train, cov_test], axis=0)
                y_all = np.concatenate([y_tr, y_dummy], axis=0)
                dom_all = np.concatenate([dom_train, dom_test], axis=0)
                _, y_enc = encode_domains(cov_all, y_all, dom_all)

                center = TLCenter(target_domain="target", metric="riemann")
                cov_centered = center.fit_transform(cov_all, y_enc)

                stretch = TLStretch(target_domain="target", centered_data=True, metric="riemann")
                cov_stretched = stretch.fit_transform(cov_centered, y_enc)

                cov_src = cov_stretched[: cov_train.shape[0]]
                cov_tgt = cov_stretched[cov_train.shape[0] :]

                model_mdm = MDM(metric="riemann")
                model_mdm.fit(cov_src, y_tr)
                return model_mdm, cov_tgt

            # Calibrate ridge/guard per candidate family (chan vs mdm) on pseudo-target subjects.
            cert_by_family: dict[str, RidgeCertificate | None] = {"chan": None, "mdm": None}
            guard_by_family: dict[str, LogisticGuard | None] = {"chan": None, "mdm": None}

            ridge_train_spearman_chan = float("nan")
            ridge_train_pearson_chan = float("nan")
            ridge_train_spearman_mdm = float("nan")
            ridge_train_pearson_mdm = float("nan")
            guard_train_auc_chan = float("nan")
            guard_train_spearman_chan = float("nan")
            guard_train_pearson_chan = float("nan")
            guard_train_auc_mdm = float("nan")
            guard_train_spearman_mdm = float("nan")
            guard_train_pearson_mdm = float("nan")

            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_ridge_rows: dict[str, List[np.ndarray]] = {"chan": [], "mdm": []}
                y_ridge_rows: dict[str, List[float]] = {"chan": [], "mdm": []}
                X_guard_rows: dict[str, List[np.ndarray]] = {"chan": [], "mdm": []}
                y_guard_rows: dict[str, List[int]] = {"chan": [], "mdm": []}
                improve_guard_rows: dict[str, List[float]] = {"chan": [], "mdm": []}
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    inner_bundle = _get_chan_bundle(inner_train)
                    m_id = inner_bundle["model_id"]
                    cand_inner: dict = dict(inner_bundle.get("candidates", {}))

                    z_p = subject_data[int(pseudo_t)].X
                    y_p = subject_data[int(pseudo_t)].y
                    p_id = _reorder_proba_columns(m_id.predict_proba(z_p), m_id.classes_, class_labels)
                    yp_id = np.asarray(m_id.predict(z_p))
                    acc_id = float(accuracy_score(y_p, yp_id))

                    # Channel candidates.
                    for cand_key, info in cand_inner.items():
                        A = info["A"]
                        m_A = info["model"]
                        z_p_A = apply_spatial_transform(A, z_p)
                        p_A = _reorder_proba_columns(m_A.predict_proba(z_p_A), m_A.classes_, class_labels)
                        yp_A = np.asarray(m_A.predict(z_p_A))
                        acc_A = float(accuracy_score(y_p, yp_A))
                        improve = float(acc_A - acc_id)

                        rec = _record_for_candidate(p_id=p_id, p_c=p_A)
                        rec["cand_family"] = "chan"
                        rec["cand_key"] = cand_key
                        rec["cand_rank"] = float(info.get("rank", float("nan")))
                        rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                        feats_vec, names = candidate_features_from_record(
                            rec, n_classes=len(class_order), include_pbar=True
                        )
                        if feat_names is None:
                            feat_names = names
                        if use_ridge:
                            X_ridge_rows["chan"].append(feats_vec)
                            y_ridge_rows["chan"].append(float(improve))
                        if use_guard:
                            X_guard_rows["chan"].append(feats_vec)
                            y_guard_rows["chan"].append(1 if improve >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows["chan"].append(float(improve))

                    # MDM(RPA) candidate.
                    try:
                        mdm_inner, cov_tgt_inner = _fit_rpa_mdm(
                            inner_train_subjects=inner_train,
                            target_subject=int(pseudo_t),
                        )
                        p_mdm = _reorder_proba_columns(
                            mdm_inner.predict_proba(cov_tgt_inner), mdm_inner.classes_, class_labels
                        )
                        yp_mdm = np.asarray(mdm_inner.predict(cov_tgt_inner))
                        acc_mdm = float(accuracy_score(y_p, yp_mdm))
                        improve_mdm = float(acc_mdm - acc_id)

                        rec_mdm = _record_for_candidate(p_id=p_id, p_c=p_mdm)
                        rec_mdm["cand_family"] = "mdm"
                        rec_mdm["cand_key"] = "mdm_rpa"
                        rec_mdm["cand_rank"] = float("nan")
                        rec_mdm["cand_lambda"] = float("nan")
                        feats_mdm, names_mdm = candidate_features_from_record(
                            rec_mdm, n_classes=len(class_order), include_pbar=True
                        )
                        if feat_names is None:
                            feat_names = names_mdm
                        if use_ridge:
                            X_ridge_rows["mdm"].append(feats_mdm)
                            y_ridge_rows["mdm"].append(float(improve_mdm))
                        if use_guard:
                            X_guard_rows["mdm"].append(feats_mdm)
                            y_guard_rows["mdm"].append(1 if improve_mdm >= float(oea_zo_calib_guard_margin) else 0)
                            improve_guard_rows["mdm"].append(float(improve_mdm))
                    except Exception:
                        pass

                # Train per-family models.
                for fam in ("chan", "mdm"):
                    if use_ridge and X_ridge_rows[fam] and feat_names is not None:
                        X_ridge = np.vstack(X_ridge_rows[fam])
                        y_ridge = np.asarray(y_ridge_rows[fam], dtype=np.float64)
                        cert_by_family[fam] = train_ridge_certificate(
                            X_ridge,
                            y_ridge,
                            feature_names=feat_names,
                            alpha=float(oea_zo_calib_ridge_alpha),
                        )
                        try:
                            pred = np.asarray(
                                cert_by_family[fam].predict_accuracy(X_ridge), dtype=np.float64
                            ).reshape(-1)
                            if y_ridge.size >= 2:
                                pear = float(np.corrcoef(pred, y_ridge)[0, 1])
                                spear = float(np.corrcoef(_rankdata(pred), _rankdata(y_ridge))[0, 1])
                                if fam == "chan":
                                    ridge_train_pearson_chan = pear
                                    ridge_train_spearman_chan = spear
                                else:
                                    ridge_train_pearson_mdm = pear
                                    ridge_train_spearman_mdm = spear
                        except Exception:
                            pass

                    if use_guard and X_guard_rows[fam] and feat_names is not None:
                        X_guard = np.vstack(X_guard_rows[fam])
                        y_guard = np.asarray(y_guard_rows[fam], dtype=int)
                        if len(np.unique(y_guard)) >= 2:
                            guard_by_family[fam] = train_logistic_guard(
                                X_guard,
                                y_guard,
                                feature_names=feat_names,
                                c=float(oea_zo_calib_guard_c),
                            )
                            try:
                                p_train = np.asarray(
                                    guard_by_family[fam].predict_pos_proba(X_guard), dtype=np.float64
                                ).reshape(-1)
                                improve_train = np.asarray(improve_guard_rows[fam], dtype=np.float64).reshape(-1)
                                auc = float(roc_auc_score(y_guard, p_train))
                                pear = float("nan")
                                spear = float("nan")
                                if improve_train.size == p_train.size and improve_train.size >= 2:
                                    pear = float(np.corrcoef(p_train, improve_train)[0, 1])
                                    spear = float(np.corrcoef(_rankdata(p_train), _rankdata(improve_train))[0, 1])
                                if fam == "chan":
                                    guard_train_auc_chan = auc
                                    guard_train_pearson_chan = pear
                                    guard_train_spearman_chan = spear
                                else:
                                    guard_train_auc_mdm = auc
                                    guard_train_pearson_mdm = pear
                                    guard_train_spearman_mdm = spear
                            except Exception:
                                pass

            # Build candidate records on the target subject (unlabeled).
            p_id_t = _reorder_proba_columns(model_id.predict_proba(X_test_raw), model_id.classes_, class_labels)
            rec_id = _record_for_candidate(p_id=p_id_t, p_c=p_id_t)
            rec_id["kind"] = "identity"
            rec_id["cand_key"] = None
            rec_id["cand_family"] = "ea"
            rec_id["cand_rank"] = float("nan")
            rec_id["cand_lambda"] = float("nan")
            records: list[dict] = [rec_id]

            # Channel candidates.
            for cand_key, info in candidates_outer.items():
                A = info["A"]
                m_A = info["model"]
                X_test_A = apply_spatial_transform(A, X_test_raw)
                p_A_t = _reorder_proba_columns(m_A.predict_proba(X_test_A), m_A.classes_, class_labels)
                rec = _record_for_candidate(p_id=p_id_t, p_c=p_A_t)
                rec["kind"] = "candidate"
                rec["cand_key"] = cand_key
                rec["cand_family"] = "chan"
                rec["cand_rank"] = float(info.get("rank", float("nan")))
                rec["cand_lambda"] = float(info.get("lambda", float("nan")))
                records.append(rec)

            # MDM(RPA) candidate.
            mdm_outer = None
            cov_tgt_outer = None
            try:
                mdm_outer, cov_tgt_outer = _fit_rpa_mdm(inner_train_subjects=train_subjects, target_subject=int(test_subject))
                p_mdm_t = _reorder_proba_columns(
                    mdm_outer.predict_proba(cov_tgt_outer), mdm_outer.classes_, class_labels
                )
                rec_mdm = _record_for_candidate(p_id=p_id_t, p_c=p_mdm_t)
                rec_mdm["kind"] = "candidate"
                rec_mdm["cand_key"] = "mdm_rpa"
                rec_mdm["cand_family"] = "mdm"
                rec_mdm["cand_rank"] = float("nan")
                rec_mdm["cand_lambda"] = float("nan")
                records.append(rec_mdm)
            except Exception:
                mdm_outer = None
                cov_tgt_outer = None

            # Per-family selection (avoid mixing CSP+LDA vs MDM probability statistics).
            selected = rec_id
            drift_mode = str(oea_zo_drift_mode)
            drift_gamma = float(oea_zo_drift_gamma)
            drift_delta = float(oea_zo_drift_delta)

            if selector in {"calibrated_ridge_guard", "calibrated_ridge", "calibrated_guard"}:
                best_pred = -float("inf")
                best_score = float("inf")
                best_rec: dict | None = None

                for rec in records:
                    if str(rec.get("kind", "")) == "identity":
                        continue
                    fam = str(rec.get("cand_family", "")).strip().lower()
                    if fam not in {"chan", "mdm"}:
                        continue

                    feats, _names = candidate_features_from_record(
                        rec, n_classes=len(class_order), include_pbar=True
                    )

                    # Guard probability (if required).
                    if selector in {"calibrated_ridge_guard", "calibrated_guard"}:
                        g = guard_by_family.get(fam, None)
                        if g is None:
                            continue
                        p_pos = float(g.predict_pos_proba(feats)[0])
                        rec["guard_p_pos"] = float(p_pos)
                        thr = float(oea_zo_calib_guard_threshold)
                        if fam == "mdm" and float(mm_safe_mdm_guard_threshold) >= 0.0:
                            thr = max(float(thr), float(mm_safe_mdm_guard_threshold))
                        if p_pos < float(thr):
                            continue

                    drift = _safe_float_local(rec.get("drift_best", 0.0))
                    if fam == "mdm" and float(mm_safe_mdm_drift_delta) > 0.0 and drift > float(mm_safe_mdm_drift_delta):
                        continue
                    if drift_mode == "hard" and drift_delta > 0.0 and drift > drift_delta:
                        continue

                    if selector in {"calibrated_ridge_guard", "calibrated_ridge"}:
                        c = cert_by_family.get(fam, None)
                        if c is None:
                            continue
                        pred_improve = float(c.predict_accuracy(feats)[0])
                        rec["ridge_pred_improve"] = float(pred_improve)
                        if fam == "mdm" and float(mm_safe_mdm_min_pred_improve) > 0.0 and float(pred_improve) < float(
                            mm_safe_mdm_min_pred_improve
                        ):
                            continue
                        if drift_mode == "penalty" and drift_gamma > 0.0:
                            pred_improve = float(pred_improve) - drift_gamma * float(drift)
                        if pred_improve > best_pred:
                            best_pred = float(pred_improve)
                            best_rec = rec
                    else:
                        # Guard-only selector: pick by objective among accepted candidates.
                        score = _safe_float_local(rec.get("score", rec.get("objective", 0.0)))
                        if drift_mode == "penalty" and drift_gamma > 0.0:
                            score = float(score) + drift_gamma * float(drift)
                        if score < best_score:
                            best_score = float(score)
                            best_rec = rec

                if selector == "calibrated_guard":
                    selected = best_rec if best_rec is not None else rec_id
                else:
                    if best_rec is None or not np.isfinite(best_pred) or float(best_pred) <= 0.0:
                        selected = rec_id
                    else:
                        selected = best_rec
            elif selector == "objective":
                selected = min(records, key=lambda r: float(r.get("score", r.get("objective_base", 0.0))))

            # Optional marginal-entropy fallback (unlabeled safety valve).
            if (
                float(oea_zo_fallback_min_marginal_entropy) > 0.0
                and str(selected.get("kind", "")) != "identity"
                and float(selected.get("entropy_bar", float("inf"))) < float(oea_zo_fallback_min_marginal_entropy)
            ):
                selected = rec_id

            # Apply selection.
            accept = str(selected.get("kind", "")) != "identity"
            sel_family = str(selected.get("cand_family", "ea"))
            sel_guard_pos = float(selected.get("guard_p_pos", float("nan")))
            sel_ridge_pred = float(selected.get("ridge_pred_improve", float("nan")))
            sel_key = selected.get("cand_key", None)

            if accept and sel_family == "chan" and sel_key in candidates_outer:
                A_sel = candidates_outer[sel_key]["A"]
                model = candidates_outer[sel_key]["model"]
                X_test = apply_spatial_transform(A_sel, X_test_raw)
            elif accept and sel_family == "mdm" and mdm_outer is not None and cov_tgt_outer is not None:
                model = mdm_outer
                X_test = cov_tgt_outer
            else:
                model = model_id
                X_test = X_test_raw
                accept = False
                sel_family = "ea"

            # Analysis-only: compute true improvement for the selected candidate (not used in selection).
            try:
                acc_id_t = float(accuracy_score(y_test, np.asarray(model_id.predict(X_test_raw))))
            except Exception:
                acc_id_t = float("nan")
            try:
                acc_sel_t = float(accuracy_score(y_test, np.asarray(model.predict(X_test))))
            except Exception:
                acc_sel_t = float("nan")
            improve_t = float(acc_sel_t - acc_id_t) if np.isfinite(acc_sel_t) and np.isfinite(acc_id_t) else float("nan")

            if extra_rows is not None:
                # Aggregate training stats (compat with existing columns).
                mm_ridge_train_spearman = float(
                    np.nanmean(np.asarray([ridge_train_spearman_chan, ridge_train_spearman_mdm], dtype=np.float64))
                )
                mm_ridge_train_pearson = float(
                    np.nanmean(np.asarray([ridge_train_pearson_chan, ridge_train_pearson_mdm], dtype=np.float64))
                )
                mm_guard_train_auc = float(
                    np.nanmean(np.asarray([guard_train_auc_chan, guard_train_auc_mdm], dtype=np.float64))
                )
                mm_guard_train_spearman = float(
                    np.nanmean(np.asarray([guard_train_spearman_chan, guard_train_spearman_mdm], dtype=np.float64))
                )
                mm_guard_train_pearson = float(
                    np.nanmean(np.asarray([guard_train_pearson_chan, guard_train_pearson_mdm], dtype=np.float64))
                )
                extra_rows.append(
                    {
                        "subject": int(test_subject),
                        "mm_safe_accept": int(bool(accept)),
                        "mm_safe_family": str(sel_family),
                        "mm_safe_guard_pos": float(sel_guard_pos),
                        "mm_safe_ridge_pred_improve": float(sel_ridge_pred),
                        "mm_safe_acc_anchor": float(acc_id_t),
                        "mm_safe_acc_selected": float(acc_sel_t),
                        "mm_safe_improve": float(improve_t),
                        "mm_safe_ridge_train_spearman": float(mm_ridge_train_spearman),
                        "mm_safe_ridge_train_pearson": float(mm_ridge_train_pearson),
                        "mm_safe_guard_train_auc": float(mm_guard_train_auc),
                        "mm_safe_guard_train_spearman": float(mm_guard_train_spearman),
                        "mm_safe_guard_train_pearson": float(mm_guard_train_pearson),
                        "mm_safe_chan_ridge_train_spearman": float(ridge_train_spearman_chan),
                        "mm_safe_chan_ridge_train_pearson": float(ridge_train_pearson_chan),
                        "mm_safe_mdm_ridge_train_spearman": float(ridge_train_spearman_mdm),
                        "mm_safe_mdm_ridge_train_pearson": float(ridge_train_pearson_mdm),
                        "mm_safe_chan_guard_train_auc": float(guard_train_auc_chan),
                        "mm_safe_chan_guard_train_spearman": float(guard_train_spearman_chan),
                        "mm_safe_chan_guard_train_pearson": float(guard_train_pearson_chan),
                        "mm_safe_mdm_guard_train_auc": float(guard_train_auc_mdm),
                        "mm_safe_mdm_guard_train_spearman": float(guard_train_spearman_mdm),
                        "mm_safe_mdm_guard_train_pearson": float(guard_train_pearson_mdm),
                    }
                )
        elif alignment == "ea_si_zo":
            # Train on EA-whitened source data with subject-invariant projection, then
            # adapt only Q_t at test time via ZO (upper-level).
            class_labels = tuple([str(c) for c in class_order])

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            subj_train = np.concatenate(
                [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in train_subjects],
                axis=0,
            )

            from mne.decoding import CSP  # local import to keep module import cost low

            csp = CSP(n_components=int(n_components))
            csp.fit(X_train, y_train)
            feats = np.asarray(csp.transform(X_train), dtype=np.float64)
            proj_params = HSICProjectorParams(
                subject_lambda=float(si_subject_lambda),
                ridge=float(si_ridge),
                n_components=(int(si_proj_dim) if int(si_proj_dim) > 0 else None),
            )
            mean_f, W = learn_hsic_subject_invariant_projector(
                X=feats,
                y=y_train,
                subjects=subj_train,
                class_order=class_labels,
                params=proj_params,
            )
            projector = CenteredLinearProjector(mean=mean_f, W=W)
            model = fit_csp_projected_lda(
                X_train=X_train,
                y_train=y_train,
                projector=projector,
                csp=csp,
                n_components=n_components,
            )

            y_test = subject_data[int(test_subject)].y

            # Optional: offline calibrated certificate / guard (trained only on source subjects in this fold).
            selector = str(oea_zo_selector)
            use_stack = selector == "calibrated_stack_ridge"
            use_ridge_guard = selector == "calibrated_ridge_guard"
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}
            use_evidence = selector == "evidence"
            use_probe_mixup = selector == "probe_mixup"
            use_probe_mixup_hard = selector == "probe_mixup_hard"
            use_iwcv = selector == "iwcv"
            use_iwcv_ucb = selector == "iwcv_ucb"
            use_dev = selector == "dev"
            use_oracle = selector == "oracle"
            cert = None
            guard = None
            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_calib_rows: List[np.ndarray] = []
                y_calib_rows: List[float] = []
                y_guard_rows: List[int] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    X_inner = np.concatenate([subject_data[s].X for s in inner_train], axis=0)
                    y_inner = np.concatenate([subject_data[s].y for s in inner_train], axis=0)

                    subj_inner = np.concatenate(
                        [np.full(subject_data[int(s)].y.shape[0], int(s), dtype=int) for s in inner_train],
                        axis=0,
                    )
                    csp_inner = CSP(n_components=int(n_components))
                    csp_inner.fit(X_inner, y_inner)
                    feats_inner = np.asarray(csp_inner.transform(X_inner), dtype=np.float64)
                    mean_i, W_i = learn_hsic_subject_invariant_projector(
                        X=feats_inner,
                        y=y_inner,
                        subjects=subj_inner,
                        class_order=class_labels,
                        params=proj_params,
                    )
                    projector_i = CenteredLinearProjector(mean=mean_i, W=W_i)
                    model_inner = fit_csp_projected_lda(
                        X_train=X_inner,
                        y_train=y_inner,
                        projector=projector_i,
                        csp=csp_inner,
                        n_components=n_components,
                    )

                    diffs_inner = []
                    for s in inner_train:
                        diffs_inner.append(
                            class_cov_diff(
                                subject_data[int(s)].X,
                                subject_data[int(s)].y,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        )
                    d_ref_inner = np.mean(np.stack(diffs_inner, axis=0), axis=0)

                    z_pseudo = subject_data[int(pseudo_t)].X
                    y_pseudo = subject_data[int(pseudo_t)].y

                    marginal_prior_inner: np.ndarray | None = None
                    if oea_zo_marginal_mode == "kl_prior":
                        if oea_zo_marginal_prior == "uniform":
                            marginal_prior_inner = np.ones(len(class_labels), dtype=np.float64) / float(
                                len(class_labels)
                            )
                        elif oea_zo_marginal_prior == "source":
                            counts = np.array([(y_inner == c).sum() for c in class_labels], dtype=np.float64)
                            marginal_prior_inner = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                        else:
                            proba_id = model_inner.predict_proba(z_pseudo)
                            proba_id = _reorder_proba_columns(proba_id, model_inner.classes_, list(class_order))
                            marginal_prior_inner = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))
                        mix = float(oea_zo_marginal_prior_mix)
                        if mix > 0.0 and marginal_prior_inner is not None:
                            u = np.ones_like(marginal_prior_inner) / float(marginal_prior_inner.shape[0])
                            marginal_prior_inner = (1.0 - mix) * marginal_prior_inner + mix * u
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))

                    lda_ev_inner = None
                    if str(oea_zo_objective) == "lda_nll" or use_stack:
                        lda_ev_inner = _compute_lda_evidence_params(
                            model=model_inner,
                            X_train=X_inner,
                            y_train=y_inner,
                            class_order=class_labels,
                        )

                    _q_sel, diag_inner = _optimize_qt_oea_zo(
                        z_t=z_pseudo,
                        model=model_inner,
                        class_order=class_labels,
                        d_ref=d_ref_inner,
                        lda_evidence=lda_ev_inner,
                        channel_names=channel_names,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
                        localmix_neighbors=int(oea_zo_localmix_neighbors),
                        localmix_self_bias=float(oea_zo_localmix_self_bias),
                        infomax_lambda=float(oea_zo_infomax_lambda),
                        reliable_metric=str(oea_zo_reliable_metric),
                        reliable_threshold=float(oea_zo_reliable_threshold),
                        reliable_alpha=float(oea_zo_reliable_alpha),
                        trust_lambda=float(oea_zo_trust_lambda),
                        trust_q0=str(oea_zo_trust_q0),
                        marginal_mode=str(oea_zo_marginal_mode),
                        marginal_beta=float(oea_zo_marginal_beta),
                        marginal_tau=float(oea_zo_marginal_tau),
                        marginal_prior=marginal_prior_inner,
                        bilevel_iters=int(oea_zo_bilevel_iters),
                        bilevel_temp=float(oea_zo_bilevel_temp),
                        bilevel_step=float(oea_zo_bilevel_step),
                        bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                        bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        holdout_fraction=float(oea_zo_holdout_fraction),
                        fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                        iters=int(oea_zo_iters),
                        lr=float(oea_zo_lr),
                        mu=float(oea_zo_mu),
                        n_rotations=int(oea_zo_k),
                        seed=int(oea_zo_seed) + int(pseudo_t) * 997,
                        l2=float(oea_zo_l2),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        return_diagnostics=True,
                    )

                    recs = list(diag_inner.get("records", []))
                    if not recs:
                        continue
                    feats_list: List[np.ndarray] = []
                    acc_list: List[float] = []
                    acc_id: float | None = None
                    for rec in recs:
                        if use_stack:
                            feats_vec, names = stacked_candidate_features_from_record(
                                rec, n_classes=len(class_labels)
                            )
                        else:
                            feats_vec, names = candidate_features_from_record(rec, n_classes=len(class_labels))
                        if feat_names is None:
                            feat_names = names
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        yp = model_inner.predict(apply_spatial_transform(Q, z_pseudo))
                        acc = float(accuracy_score(y_pseudo, yp))
                        if str(rec.get("kind", "")) == "identity":
                            acc_id = acc
                        feats_list.append(feats_vec)
                        acc_list.append(acc)
                    if acc_id is None:
                        continue
                    for feats_vec, acc in zip(feats_list, acc_list):
                        improve = float(acc - float(acc_id))
                        y_calib_rows.append(float(improve))
                        y_guard_rows.append(1 if float(improve) >= float(oea_zo_calib_guard_margin) else 0)
                        X_calib_rows.append(feats_vec)

                if X_calib_rows and feat_names is not None:
                    X_calib = np.vstack(X_calib_rows)
                    y_calib = np.asarray(y_calib_rows, dtype=np.float64)
                    y_guard = np.asarray(y_guard_rows, dtype=int)
                    if use_ridge:
                        cert = train_ridge_certificate(
                            X=X_calib,
                            y=y_calib,
                            feature_names=feat_names,
                            alpha=float(oea_zo_calib_ridge_alpha),
                        )
                    if use_guard:
                        guard = train_logistic_guard(
                            X=X_calib,
                            y=y_guard,
                            feature_names=feat_names,
                            C=float(oea_zo_calib_guard_c),
                        )

            z_t = subject_data[int(test_subject)].X
            d_ref = np.mean(
                np.stack(
                    [
                        class_cov_diff(
                            subject_data[int(s)].X,
                            subject_data[int(s)].y,
                            class_order=class_labels,
                            eps=oea_eps,
                            shrinkage=oea_shrinkage,
                        )
                        for s in train_subjects
                    ],
                    axis=0,
                ),
                axis=0,
            )

            want_diag = (
                bool(do_diag)
                or (use_ridge and cert is not None)
                or (use_guard and guard is not None)
                or use_evidence
                or use_probe_mixup
                or use_probe_mixup_hard
                or use_iwcv
                or use_iwcv_ucb
                or use_dev
                or use_oracle
            )
            if use_oracle:
                want_diag = True

            lda_ev = None
            if str(oea_zo_objective) == "lda_nll" or use_evidence or use_stack or bool(do_diag):
                lda_ev = _compute_lda_evidence_params(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    class_order=class_labels,
                )

            opt_res = _optimize_qt_oea_zo(
                z_t=z_t,
                model=model,
                class_order=class_labels,
                d_ref=d_ref,
                lda_evidence=lda_ev,
                channel_names=channel_names,
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                pseudo_mode=str(oea_pseudo_mode),
                warm_start=str(oea_zo_warm_start),
                warm_iters=int(oea_zo_warm_iters),
                q_blend=float(oea_q_blend),
                objective=str(oea_zo_objective),
                transform=str(oea_zo_transform),
                localmix_neighbors=int(oea_zo_localmix_neighbors),
                localmix_self_bias=float(oea_zo_localmix_self_bias),
                infomax_lambda=float(oea_zo_infomax_lambda),
                reliable_metric=str(oea_zo_reliable_metric),
                reliable_threshold=float(oea_zo_reliable_threshold),
                reliable_alpha=float(oea_zo_reliable_alpha),
                trust_lambda=float(oea_zo_trust_lambda),
                trust_q0=str(oea_zo_trust_q0),
                marginal_mode=str(oea_zo_marginal_mode),
                marginal_beta=float(oea_zo_marginal_beta),
                marginal_tau=float(oea_zo_marginal_tau),
                marginal_prior=None,
                bilevel_iters=int(oea_zo_bilevel_iters),
                bilevel_temp=float(oea_zo_bilevel_temp),
                bilevel_step=float(oea_zo_bilevel_step),
                bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                drift_mode=str(oea_zo_drift_mode),
                drift_gamma=float(oea_zo_drift_gamma),
                drift_delta=float(oea_zo_drift_delta),
                min_improvement=float(oea_zo_min_improvement),
                holdout_fraction=float(oea_zo_holdout_fraction),
                fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                iters=int(oea_zo_iters),
                lr=float(oea_zo_lr),
                mu=float(oea_zo_mu),
                n_rotations=int(oea_zo_k),
                seed=int(oea_zo_seed) + int(test_subject) * 997,
                l2=float(oea_zo_l2),
                pseudo_confidence=float(oea_pseudo_confidence),
                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                pseudo_balance=bool(oea_pseudo_balance),
                return_diagnostics=bool(want_diag),
            )
            if want_diag:
                q_t, zo_diag = opt_res
            else:
                q_t = opt_res

            if zo_diag is not None:
                selected: dict | None = None
                if use_oracle:
                    best_rec = None
                    best_acc = -1.0
                    for rec in zo_diag.get("records", []):
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        yp = model.predict(apply_spatial_transform(Q, z_t))
                        acc = float(accuracy_score(y_test, yp))
                        if acc > best_acc:
                            best_acc = acc
                            best_rec = rec
                    selected = best_rec
                elif use_evidence:
                    selected = select_by_evidence_nll(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup:
                    selected = select_by_probe_mixup(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup_hard:
                    selected = select_by_probe_mixup_hard(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_iwcv:
                    selected = select_by_iwcv_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_iwcv_ucb:
                    selected = select_by_iwcv_ucb(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        kappa=float(oea_zo_iwcv_kappa),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_dev:
                    selected = select_by_dev_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_ridge_guard and cert is not None and guard is not None:
                    selected = select_by_guarded_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        feature_set="stacked" if use_stack else "base",
                    )
                elif use_ridge and cert is not None:
                    selected = select_by_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        n_classes=len(class_labels),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        feature_set="stacked" if use_stack else "base",
                    )
                elif use_guard and guard is not None:
                    selected = select_by_guarded_objective(
                        zo_diag.get("records", []),
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                    )
                if selected is not None:
                    q_t = np.asarray(selected.get("Q"), dtype=np.float64)

            X_test = apply_spatial_transform(q_t, z_t)
        elif alignment in {"ea_zo", "raw_zo"}:
            # Train on the current channel space, then adapt only Q_t at test time.
            # - ea_zo: the current space is EA-whitened (per-subject).
            # - raw_zo: the current space is the raw (preprocessed) channel space (no whitening).
            class_labels = tuple([str(c) for c in class_order])
            use_post_ea = str(oea_zo_transform) == "local_mix_then_ea"
            if use_post_ea and alignment != "ea_zo":
                raise ValueError("oea_zo_transform='local_mix_then_ea' is only supported with alignment='ea_zo'.")

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
            y_test = subject_data[int(test_subject)].y

            # Optional: offline calibrated certificate / guard (trained only on source subjects in this fold).
            selector = str(oea_zo_selector)
            use_stack = selector == "calibrated_stack_ridge"
            use_ridge_guard = selector == "calibrated_ridge_guard"
            use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
            use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}
            use_evidence = selector == "evidence"
            use_probe_mixup = selector == "probe_mixup"
            use_probe_mixup_hard = selector == "probe_mixup_hard"
            use_iwcv = selector == "iwcv"
            use_iwcv_ucb = selector == "iwcv_ucb"
            use_dev = selector == "dev"
            use_oracle = selector == "oracle"
            cert = None
            guard = None
            if use_ridge or use_guard:
                rng = np.random.RandomState(int(oea_zo_calib_seed) + int(test_subject) * 997)
                calib_subjects = list(train_subjects)
                if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(calib_subjects):
                    rng.shuffle(calib_subjects)
                    calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                X_calib_rows: List[np.ndarray] = []
                y_calib_rows: List[float] = []
                y_guard_rows: List[int] = []
                feat_names: tuple[str, ...] | None = None

                for pseudo_t in calib_subjects:
                    inner_train = [s for s in train_subjects if s != pseudo_t]
                    if len(inner_train) < 2:
                        continue
                    X_inner = np.concatenate([subject_data[s].X for s in inner_train], axis=0)
                    y_inner = np.concatenate([subject_data[s].y for s in inner_train], axis=0)
                    model_inner = fit_csp_lda(X_inner, y_inner, n_components=n_components)

                    diffs_inner = []
                    for s in inner_train:
                        diffs_inner.append(
                            class_cov_diff(
                                subject_data[int(s)].X,
                                subject_data[int(s)].y,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        )
                    d_ref_inner = np.mean(np.stack(diffs_inner, axis=0), axis=0)

                    z_pseudo = (
                        subject_data_raw[int(pseudo_t)].X if use_post_ea else subject_data[int(pseudo_t)].X
                    )
                    y_pseudo = subject_data[int(pseudo_t)].y

                    marginal_prior_inner: np.ndarray | None = None
                    if oea_zo_marginal_mode == "kl_prior":
                        if oea_zo_marginal_prior == "uniform":
                            marginal_prior_inner = np.ones(len(class_labels), dtype=np.float64) / float(
                                len(class_labels)
                            )
                        elif oea_zo_marginal_prior == "source":
                            counts = np.array([(y_inner == c).sum() for c in class_labels], dtype=np.float64)
                            marginal_prior_inner = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                        else:
                            if use_post_ea:
                                z_pseudo_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(
                                    z_pseudo
                                )
                                proba_id = model_inner.predict_proba(z_pseudo_ea)
                            else:
                                proba_id = model_inner.predict_proba(z_pseudo)
                            proba_id = _reorder_proba_columns(
                                proba_id, model_inner.classes_, list(class_order)
                            )
                            marginal_prior_inner = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))
                        mix = float(oea_zo_marginal_prior_mix)
                        if mix > 0.0 and marginal_prior_inner is not None:
                            u = np.ones_like(marginal_prior_inner) / float(marginal_prior_inner.shape[0])
                            marginal_prior_inner = (1.0 - mix) * marginal_prior_inner + mix * u
                            marginal_prior_inner = marginal_prior_inner / float(np.sum(marginal_prior_inner))

                    lda_ev_inner = None
                    if str(oea_zo_objective) == "lda_nll" or use_stack:
                        lda_ev_inner = _compute_lda_evidence_params(
                            model=model_inner,
                            X_train=X_inner,
                            y_train=y_inner,
                            class_order=class_labels,
                        )

                    _qt_inner, diag_inner = _optimize_qt_oea_zo(
                        z_t=z_pseudo,
                        model=model_inner,
                        class_order=class_labels,
                        d_ref=d_ref_inner,
                        lda_evidence=lda_ev_inner,
                        channel_names=channel_names,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
                        localmix_neighbors=int(oea_zo_localmix_neighbors),
                        localmix_self_bias=float(oea_zo_localmix_self_bias),
                        infomax_lambda=float(oea_zo_infomax_lambda),
                        reliable_metric=str(oea_zo_reliable_metric),
                        reliable_threshold=float(oea_zo_reliable_threshold),
                        reliable_alpha=float(oea_zo_reliable_alpha),
                        trust_lambda=float(oea_zo_trust_lambda),
                        trust_q0=str(oea_zo_trust_q0),
                        marginal_mode=str(oea_zo_marginal_mode),
                        marginal_beta=float(oea_zo_marginal_beta),
                        marginal_tau=float(oea_zo_marginal_tau),
                        marginal_prior=marginal_prior_inner,
                        bilevel_iters=int(oea_zo_bilevel_iters),
                        bilevel_temp=float(oea_zo_bilevel_temp),
                        bilevel_step=float(oea_zo_bilevel_step),
                        bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                        bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        holdout_fraction=float(oea_zo_holdout_fraction),
                        fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                        iters=int(oea_zo_iters),
                        lr=float(oea_zo_lr),
                        mu=float(oea_zo_mu),
                        n_rotations=int(oea_zo_k),
                        seed=int(oea_zo_seed) + int(pseudo_t) * 997,
                        l2=float(oea_zo_l2),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        return_diagnostics=True,
                    )
                    recs = list(diag_inner.get("records", []))
                    if not recs:
                        continue
                    feats_list: List[np.ndarray] = []
                    acc_list: List[float] = []
                    acc_id: float | None = None
                    for rec in recs:
                        if use_stack:
                            feats, names = stacked_candidate_features_from_record(rec, n_classes=len(class_labels))
                        else:
                            feats, names = candidate_features_from_record(rec, n_classes=len(class_labels))
                        if feat_names is None:
                            feat_names = names
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        Xp = apply_spatial_transform(Q, z_pseudo)
                        yp = model_inner.predict(Xp)
                        acc = float(accuracy_score(y_pseudo, yp))
                        if str(rec.get("kind", "")) == "identity":
                            acc_id = acc
                        feats_list.append(feats)
                        acc_list.append(acc)
                    if acc_id is None:
                        continue
                    for feats, acc in zip(feats_list, acc_list):
                        improve = float(acc - float(acc_id))
                        y_calib_rows.append(float(improve))
                        y_guard_rows.append(1 if float(improve) >= float(oea_zo_calib_guard_margin) else 0)
                        X_calib_rows.append(feats)

                if X_calib_rows and feat_names is not None:
                    X_cal = np.stack(X_calib_rows, axis=0)
                    if use_ridge:
                        cert = train_ridge_certificate(
                            X_cal,
                            np.asarray(y_calib_rows, dtype=np.float64),
                            feature_names=feat_names,
                            alpha=float(oea_zo_calib_ridge_alpha),
                        )
                    if use_guard:
                        y_guard = np.asarray(y_guard_rows, dtype=int).reshape(-1)
                        if np.unique(y_guard).size >= 2:
                            guard = train_logistic_guard(
                                X_cal,
                                y_guard,
                                feature_names=feat_names,
                                c=float(oea_zo_calib_guard_c),
                            )
                        else:
                            guard = None
                else:
                    cert = None
                    guard = None

            diffs_train = []
            for s in train_subjects:
                diffs_train.append(
                    class_cov_diff(
                        subject_data[int(s)].X,
                        subject_data[int(s)].y,
                        class_order=class_labels,
                        eps=oea_eps,
                        shrinkage=oea_shrinkage,
                    )
                )
            d_ref = np.mean(np.stack(diffs_train, axis=0), axis=0)

            z_t = subject_data_raw[int(test_subject)].X if use_post_ea else subject_data[int(test_subject)].X
            z_test_base = z_t
            marginal_prior_vec: np.ndarray | None = None
            if oea_zo_marginal_mode == "kl_prior":
                if oea_zo_marginal_prior == "uniform":
                    marginal_prior_vec = np.ones(len(class_labels), dtype=np.float64) / float(len(class_labels))
                elif oea_zo_marginal_prior == "source":
                    counts = np.array([(y_train == c).sum() for c in class_labels], dtype=np.float64)
                    marginal_prior_vec = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                else:
                    # anchor_pred: use target predicted marginal at Q=I (EA), fixed during optimization.
                    if use_post_ea:
                        z_t_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(z_t)
                        proba_id = model.predict_proba(z_t_ea)
                    else:
                        proba_id = model.predict_proba(z_t)
                    proba_id = _reorder_proba_columns(proba_id, model.classes_, list(class_order))
                    marginal_prior_vec = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                    marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))
                mix = float(oea_zo_marginal_prior_mix)
                if mix > 0.0 and marginal_prior_vec is not None:
                    u = np.ones_like(marginal_prior_vec) / float(marginal_prior_vec.shape[0])
                    marginal_prior_vec = (1.0 - mix) * marginal_prior_vec + mix * u
                    marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

            want_diag = (
                bool(do_diag)
                or (use_ridge and cert is not None)
                or (use_guard and guard is not None)
                or use_evidence
                or use_probe_mixup
                or use_probe_mixup_hard
                or use_iwcv
                or use_iwcv_ucb
                or use_oracle
            )
            lda_ev = None
            if str(oea_zo_objective) == "lda_nll" or use_evidence or use_stack or bool(do_diag):
                lda_ev = _compute_lda_evidence_params(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    class_order=class_labels,
                )
            opt_res = _optimize_qt_oea_zo(
                z_t=z_t,
                model=model,
                class_order=class_labels,
                d_ref=d_ref,
                lda_evidence=lda_ev,
                channel_names=channel_names,
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                pseudo_mode=str(oea_pseudo_mode),
                warm_start=str(oea_zo_warm_start),
                warm_iters=int(oea_zo_warm_iters),
                q_blend=float(oea_q_blend),
                objective=str(oea_zo_objective),
                transform=str(oea_zo_transform),
                localmix_neighbors=int(oea_zo_localmix_neighbors),
                localmix_self_bias=float(oea_zo_localmix_self_bias),
                infomax_lambda=float(oea_zo_infomax_lambda),
                reliable_metric=str(oea_zo_reliable_metric),
                reliable_threshold=float(oea_zo_reliable_threshold),
                reliable_alpha=float(oea_zo_reliable_alpha),
                trust_lambda=float(oea_zo_trust_lambda),
                trust_q0=str(oea_zo_trust_q0),
                marginal_mode=str(oea_zo_marginal_mode),
                marginal_beta=float(oea_zo_marginal_beta),
                marginal_tau=float(oea_zo_marginal_tau),
                marginal_prior=marginal_prior_vec,
                bilevel_iters=int(oea_zo_bilevel_iters),
                bilevel_temp=float(oea_zo_bilevel_temp),
                bilevel_step=float(oea_zo_bilevel_step),
                bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                drift_mode=str(oea_zo_drift_mode),
                drift_gamma=float(oea_zo_drift_gamma),
                drift_delta=float(oea_zo_drift_delta),
                min_improvement=float(oea_zo_min_improvement),
                holdout_fraction=float(oea_zo_holdout_fraction),
                fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                iters=int(oea_zo_iters),
                lr=float(oea_zo_lr),
                mu=float(oea_zo_mu),
                n_rotations=int(oea_zo_k),
                seed=int(oea_zo_seed) + int(test_subject) * 997,
                l2=float(oea_zo_l2),
                pseudo_confidence=float(oea_pseudo_confidence),
                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                pseudo_balance=bool(oea_pseudo_balance),
                return_diagnostics=bool(want_diag),
            )
            if want_diag:
                q_t, zo_diag = opt_res
            else:
                q_t = opt_res

            if zo_diag is not None:
                selected: dict | None = None
                if use_oracle:
                    best_rec = None
                    best_acc = -1.0
                    for rec in zo_diag.get("records", []):
                        Q = np.asarray(rec.get("Q"), dtype=np.float64)
                        yp = model.predict(apply_spatial_transform(Q, z_t))
                        acc = float(accuracy_score(y_test, yp))
                        if acc > best_acc:
                            best_acc = acc
                            best_rec = rec
                    selected = best_rec
                elif use_evidence:
                    selected = select_by_evidence_nll(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup:
                    selected = select_by_probe_mixup(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_probe_mixup_hard:
                    selected = select_by_probe_mixup_hard(
                        zo_diag.get("records", []),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                    )
                elif use_iwcv:
                    selected = select_by_iwcv_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_iwcv_ucb:
                    selected = select_by_iwcv_ucb(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        kappa=float(oea_zo_iwcv_kappa),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_dev:
                    selected = select_by_dev_nll(
                        zo_diag.get("records", []),
                        model=model,
                        z_source=X_train,
                        y_source=y_train,
                        z_target=z_t,
                        class_order=class_labels,
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        seed=int(oea_zo_seed) + int(test_subject) * 997,
                    )
                elif use_ridge_guard and cert is not None and guard is not None:
                    selected = select_by_guarded_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                    )
                elif use_ridge and cert is not None:
                    selected = select_by_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        n_classes=len(class_labels),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        feature_set="stacked" if use_stack else "base",
                    )
                elif use_guard and guard is not None:
                    selected = select_by_guarded_objective(
                        zo_diag.get("records", []),
                        guard=guard,
                        n_classes=len(class_labels),
                        threshold=float(oea_zo_calib_guard_threshold),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                    )
                if selected is not None:
                    q_t = np.asarray(selected.get("Q"), dtype=np.float64)
            X_test = apply_spatial_transform(q_t, z_t)
        elif alignment == "oea_cov":
            # OEA (cov-eig) selection: pick Q_s = U_ref U_sᵀ, where U_s is eigenbasis of C_s
            # and U_ref from the average covariance of the training subjects.
            ea_by_subject: Dict[int, EuclideanAligner] = {}
            covs_train: List[np.ndarray] = []
            for s in train_subjects:
                ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(subject_data[s].X)
                ea_by_subject[int(s)] = ea
                covs_train.append(ea.cov_)
            c_ref = np.mean(np.stack(covs_train, axis=0), axis=0)
            _evals_ref, u_ref = sorted_eigh(c_ref)

            def _align_one(subj: int) -> np.ndarray:
                ea = ea_by_subject.get(int(subj))
                if ea is None:
                    ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(
                        subject_data[int(subj)].X
                    )
                    ea_by_subject[int(subj)] = ea
                z = ea.transform(subject_data[int(subj)].X)
                q = u_ref @ ea.eigvecs_.T
                q = blend_with_identity(q, oea_q_blend)
                return apply_spatial_transform(q, z)

            X_train = np.concatenate([_align_one(s) for s in train_subjects], axis=0)
            y_train = np.concatenate([subject_data[s].y for s in train_subjects], axis=0)
            X_test = _align_one(test_subject)
            y_test = subject_data[test_subject].y
        else:
            # alignment in {"oea","oea_zo"}: optimistic selection based on a discriminative
            # covariance signature (binary: Δ=Cov(c1)-Cov(c0); multiclass: between-class scatter).
            class_labels = tuple([str(c) for c in class_order])

            # 1) EA whitening for each subject (no Q yet).
            ea_by_subject: Dict[int, EuclideanAligner] = {}
            z_by_subject: Dict[int, np.ndarray] = {}
            for s in subjects:
                ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(subject_data[s].X)
                ea_by_subject[int(s)] = ea
                z_by_subject[int(s)] = ea.transform(subject_data[s].X)

            # 2) Build training reference Δ_ref from labeled source subjects.
            diffs_train = []
            for s in train_subjects:
                diffs_train.append(
                    class_cov_diff(
                        z_by_subject[int(s)],
                        subject_data[int(s)].y,
                        class_order=class_labels,
                        eps=oea_eps,
                        shrinkage=oea_shrinkage,
                    )
                )
            d_ref = np.mean(np.stack(diffs_train, axis=0), axis=0)

            # 3) Align each training subject by choosing Q_s that best matches Δ_ref.
            def _align_train_subject(s: int) -> np.ndarray:
                d_s = class_cov_diff(
                    z_by_subject[int(s)],
                    subject_data[int(s)].y,
                    class_order=class_labels,
                    eps=oea_eps,
                    shrinkage=oea_shrinkage,
                )
                q_s = orthogonal_align_symmetric(d_s, d_ref)
                q_s = blend_with_identity(q_s, oea_q_blend)
                return apply_spatial_transform(q_s, z_by_subject[int(s)])

            X_train = np.concatenate([_align_train_subject(s) for s in train_subjects], axis=0)
            y_train = np.concatenate([subject_data[s].y for s in train_subjects], axis=0)

            # 4) Train the classifier once (frozen after this).
            model = fit_csp_lda(X_train, y_train, n_components=n_components)

            # 5) Target subject: select Q_t using *unlabeled* target data (no classifier update).
            z_t = z_by_subject[int(test_subject)]
            if alignment == "oea":
                q_t = np.eye(z_t.shape[1], dtype=np.float64)
                for _ in range(int(max(0, oea_pseudo_iters))):
                    X_t_cur = apply_spatial_transform(q_t, z_t)
                    proba = model.predict_proba(X_t_cur)
                    proba = _reorder_proba_columns(proba, model.classes_, list(class_labels))

                    if oea_pseudo_mode == "soft":
                        d_t = _soft_class_cov_diff(
                            z_t,
                            proba=proba,
                            class_order=class_labels,
                            eps=oea_eps,
                            shrinkage=oea_shrinkage,
                        )
                    else:
                        y_pseudo = np.asarray(model.predict(X_t_cur))
                        keep = _select_pseudo_indices(
                            y_pseudo=y_pseudo,
                            proba=proba,
                            class_order=class_labels,
                            confidence=float(oea_pseudo_confidence),
                            topk_per_class=int(oea_pseudo_topk_per_class),
                            balance=bool(oea_pseudo_balance),
                        )
                        if keep.size == 0:
                            break
                        d_t = class_cov_diff(
                            z_t[keep],
                            y_pseudo[keep],
                            class_order=class_labels,
                            eps=oea_eps,
                            shrinkage=oea_shrinkage,
                        )
                    q_t = orthogonal_align_symmetric(d_t, d_ref)
                    q_t = blend_with_identity(q_t, oea_q_blend)
            else:
                # If using KL(π||p̄) in the ZO objective, build π per fold.
                marginal_prior_vec: np.ndarray | None = None
                if oea_zo_marginal_mode == "kl_prior":
                    if oea_zo_marginal_prior == "uniform":
                        marginal_prior_vec = np.ones(len(class_labels), dtype=np.float64) / float(
                            len(class_labels)
                        )
                    elif oea_zo_marginal_prior == "source":
                        counts = np.array([(y_train == c).sum() for c in class_labels], dtype=np.float64)
                        marginal_prior_vec = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                else:
                    if use_post_ea:
                        z_t_ea = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(z_t)
                        proba_id = model.predict_proba(z_t_ea)
                    else:
                        proba_id = model.predict_proba(z_t)
                    proba_id = _reorder_proba_columns(proba_id, model.classes_, list(class_order))
                    marginal_prior_vec = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                    marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))
                    mix = float(oea_zo_marginal_prior_mix)
                    if mix > 0.0 and marginal_prior_vec is not None:
                        u = np.ones_like(marginal_prior_vec) / float(marginal_prior_vec.shape[0])
                        marginal_prior_vec = (1.0 - mix) * marginal_prior_vec + mix * u
                        marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

                selector = str(oea_zo_selector)
                use_evidence = selector == "evidence"
                use_probe_mixup = selector == "probe_mixup"
                use_probe_mixup_hard = selector == "probe_mixup_hard"
                use_iwcv = selector == "iwcv"
                use_iwcv_ucb = selector == "iwcv_ucb"
                use_dev = selector == "dev"
                use_oracle = selector == "oracle"
                want_diag = (
                    bool(do_diag)
                    or use_evidence
                    or use_probe_mixup
                    or use_probe_mixup_hard
                    or use_iwcv
                    or use_iwcv_ucb
                    or use_dev
                    or use_oracle
                )
                lda_ev = None
                if str(oea_zo_objective) == "lda_nll" or use_evidence or bool(do_diag):
                    lda_ev = _compute_lda_evidence_params(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        class_order=class_labels,
                    )
                opt_res = _optimize_qt_oea_zo(
                    z_t=z_t,
                    model=model,
                    class_order=class_labels,
                    d_ref=d_ref,
                    lda_evidence=lda_ev,
                    channel_names=channel_names,
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    pseudo_mode=str(oea_pseudo_mode),
                    warm_start=str(oea_zo_warm_start),
                    warm_iters=int(oea_zo_warm_iters),
                    q_blend=float(oea_q_blend),
                    objective=str(oea_zo_objective),
                    transform=str(oea_zo_transform),
                    localmix_neighbors=int(oea_zo_localmix_neighbors),
                    localmix_self_bias=float(oea_zo_localmix_self_bias),
                    infomax_lambda=float(oea_zo_infomax_lambda),
                    reliable_metric=str(oea_zo_reliable_metric),
                    reliable_threshold=float(oea_zo_reliable_threshold),
                    reliable_alpha=float(oea_zo_reliable_alpha),
                    trust_lambda=float(oea_zo_trust_lambda),
                    trust_q0=str(oea_zo_trust_q0),
                    marginal_mode=str(oea_zo_marginal_mode),
                    marginal_beta=float(oea_zo_marginal_beta),
                    marginal_tau=float(oea_zo_marginal_tau),
                    marginal_prior=marginal_prior_vec,
                    bilevel_iters=int(oea_zo_bilevel_iters),
                    bilevel_temp=float(oea_zo_bilevel_temp),
                    bilevel_step=float(oea_zo_bilevel_step),
                    bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                    bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                    drift_mode=str(oea_zo_drift_mode),
                    drift_gamma=float(oea_zo_drift_gamma),
                    drift_delta=float(oea_zo_drift_delta),
                    min_improvement=float(oea_zo_min_improvement),
                    holdout_fraction=float(oea_zo_holdout_fraction),
                    fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                    iters=int(oea_zo_iters),
                    lr=float(oea_zo_lr),
                    mu=float(oea_zo_mu),
                    n_rotations=int(oea_zo_k),
                    seed=int(oea_zo_seed) + int(test_subject) * 997,
                    l2=float(oea_zo_l2),
                    pseudo_confidence=float(oea_pseudo_confidence),
                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                    pseudo_balance=bool(oea_pseudo_balance),
                    return_diagnostics=bool(want_diag),
                )
                if want_diag:
                    q_t, zo_diag = opt_res
                    if use_oracle:
                        y_test = subject_data[test_subject].y
                        best_rec = None
                        best_acc = -1.0
                        for rec in zo_diag.get("records", []):
                            Q = np.asarray(rec.get("Q"), dtype=np.float64)
                            yp = model.predict(apply_spatial_transform(Q, z_t))
                            acc = float(accuracy_score(y_test, yp))
                            if acc > best_acc:
                                best_acc = acc
                                best_rec = rec
                        if best_rec is not None:
                            q_t = np.asarray(best_rec.get("Q"), dtype=np.float64)
                    elif use_evidence:
                        sel = select_by_evidence_nll(
                            zo_diag.get("records", []),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_probe_mixup:
                        sel = select_by_probe_mixup(
                            zo_diag.get("records", []),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_probe_mixup_hard:
                        sel = select_by_probe_mixup_hard(
                            zo_diag.get("records", []),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_iwcv:
                        sel = select_by_iwcv_nll(
                            zo_diag.get("records", []),
                            model=model,
                            z_source=X_train,
                            y_source=y_train,
                            z_target=z_t,
                            class_order=class_labels,
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                            seed=int(oea_zo_seed) + int(test_subject) * 997,
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_iwcv_ucb:
                        sel = select_by_iwcv_ucb(
                            zo_diag.get("records", []),
                            model=model,
                            z_source=X_train,
                            y_source=y_train,
                            z_target=z_t,
                            class_order=class_labels,
                            kappa=float(oea_zo_iwcv_kappa),
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                            seed=int(oea_zo_seed) + int(test_subject) * 997,
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                    elif use_dev:
                        sel = select_by_dev_nll(
                            zo_diag.get("records", []),
                            model=model,
                            z_source=X_train,
                            y_source=y_train,
                            z_target=z_t,
                            class_order=class_labels,
                            drift_mode=str(oea_zo_drift_mode),
                            drift_gamma=float(oea_zo_drift_gamma),
                            drift_delta=float(oea_zo_drift_delta),
                            min_improvement=float(oea_zo_min_improvement),
                            seed=int(oea_zo_seed) + int(test_subject) * 997,
                        )
                        if sel is not None:
                            q_t = np.asarray(sel.get("Q"), dtype=np.float64)
                else:
                    q_t = opt_res
                z_test_base = z_t

            X_test = apply_spatial_transform(q_t, z_t)
            y_test = subject_data[test_subject].y

        if model is None:
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_proba = _reorder_proba_columns(y_proba, model.classes_, class_order)

        if do_diag and zo_diag is not None:
            _write_zo_diagnostics(
                zo_diag,
                out_dir=Path(diagnostics_dir),
                tag=str(diagnostics_tag),
                subject=int(test_subject),
                model=model,
                z_t=z_test_base if z_test_base is not None else subject_data[int(test_subject)].X,
                y_true=y_test,
                class_order=class_order,
            )

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )

        fold_rows.append(
            FoldResult(
                subject=int(test_subject),
                n_train=int(len(y_train)),
                n_test=int(len(y_test)),
                **metrics,
            )
        )
        models_by_subject[int(test_subject)] = model
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(test_subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))

    results_df = pd.DataFrame([asdict(r) for r in fold_rows]).sort_values("subject")
    if extra_rows is not None and extra_rows:
        extra_df = pd.DataFrame(extra_rows).sort_values("subject")
        results_df = results_df.merge(extra_df, on="subject", how="left")
    y_true_cat = np.concatenate(y_true_all, axis=0)
    y_pred_cat = np.concatenate(y_pred_all, axis=0)
    y_proba_cat = np.concatenate(y_proba_all, axis=0)
    subj_cat = np.concatenate(subj_all, axis=0)
    trial_cat = np.concatenate(trial_all, axis=0)

    pred_df = pd.DataFrame(
        {
            "subject": subj_cat,
            "trial": trial_cat,
            "y_true": y_true_cat,
            "y_pred": y_pred_cat,
        }
    )
    for i, c in enumerate(list(class_order)):
        pred_df[f"proba_{c}"] = y_proba_cat[:, int(i)]

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )


def cross_session_within_subject_evaluation(
    subject_session_data: Dict[int, Dict[str, SubjectData]],
    *,
    train_sessions: Sequence[str],
    test_sessions: Sequence[str],
    class_order: Sequence[str],
    channel_names: Sequence[str] | None = None,
    n_components: int = 4,
    average: str = "macro",
    alignment: str = "ea",
    oea_eps: float = 1e-10,
    oea_shrinkage: float = 0.0,
    oea_pseudo_iters: int = 2,
    oea_q_blend: float = 1.0,
    oea_pseudo_mode: str = "hard",
    oea_pseudo_confidence: float = 0.0,
    oea_pseudo_topk_per_class: int = 0,
    oea_pseudo_balance: bool = False,
    oea_zo_objective: str = "entropy",
    oea_zo_transform: str = "orthogonal",
    oea_zo_localmix_neighbors: int = 4,
    oea_zo_localmix_self_bias: float = 3.0,
    oea_zo_infomax_lambda: float = 1.0,
    oea_zo_reliable_metric: str = "none",
    oea_zo_reliable_threshold: float = 0.0,
    oea_zo_reliable_alpha: float = 10.0,
    oea_zo_trust_lambda: float = 0.0,
    oea_zo_trust_q0: str = "identity",
    oea_zo_marginal_mode: str = "none",
    oea_zo_marginal_beta: float = 0.0,
    oea_zo_marginal_tau: float = 0.05,
    oea_zo_marginal_prior: str = "uniform",
    oea_zo_marginal_prior_mix: float = 0.0,
    oea_zo_bilevel_iters: int = 5,
    oea_zo_bilevel_temp: float = 1.0,
    oea_zo_bilevel_step: float = 1.0,
    oea_zo_bilevel_coverage_target: float = 0.5,
    oea_zo_bilevel_coverage_power: float = 1.0,
    oea_zo_drift_mode: str = "none",
    oea_zo_drift_gamma: float = 0.0,
    oea_zo_drift_delta: float = 0.0,
    oea_zo_selector: str = "objective",
    oea_zo_iwcv_kappa: float = 1.0,
    oea_zo_calib_ridge_alpha: float = 1.0,
    oea_zo_calib_max_subjects: int = 0,
    oea_zo_calib_seed: int = 0,
    oea_zo_calib_guard_c: float = 1.0,
    oea_zo_calib_guard_threshold: float = 0.5,
    oea_zo_calib_guard_margin: float = 0.0,
    oea_zo_min_improvement: float = 0.0,
    oea_zo_holdout_fraction: float = 0.0,
    oea_zo_warm_start: str = "none",
    oea_zo_warm_iters: int = 1,
    oea_zo_fallback_min_marginal_entropy: float = 0.0,
    oea_zo_iters: int = 30,
    oea_zo_lr: float = 0.5,
    oea_zo_mu: float = 0.1,
    oea_zo_k: int = 50,
    oea_zo_seed: int = 0,
    oea_zo_l2: float = 0.0,
    diagnostics_dir: Path | None = None,
    diagnostics_subjects: Sequence[int] = (),
    diagnostics_tag: str = "",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[int, TrainedModel],
]:
    """Within-subject cross-session evaluation.

    For each subject, train on `train_sessions` and test on `test_sessions`.
    This is useful for single-subject cross-session domain shift (often smaller than cross-subject).
    """

    train_sessions = [str(s) for s in train_sessions]
    test_sessions = [str(s) for s in test_sessions]
    class_order = [str(c) for c in class_order]
    if alignment not in {"none", "ea", "rpa", "ea_zo", "rpa_zo", "tsa", "tsa_zo", "oea_cov", "oea", "oea_zo"}:
        raise ValueError(
            "alignment must be one of: "
            "'none', 'ea', 'rpa', 'ea_zo', 'rpa_zo', 'tsa', 'tsa_zo', 'oea_cov', 'oea', 'oea_zo'"
        )
    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")
    if int(oea_zo_localmix_neighbors) < 0:
        raise ValueError("oea_zo_localmix_neighbors must be >= 0.")
    if float(oea_zo_localmix_self_bias) < 0.0:
        raise ValueError("oea_zo_localmix_self_bias must be >= 0.")

    use_post_ea = str(oea_zo_transform) == "local_mix_then_ea"
    if use_post_ea and alignment != "ea_zo":
        raise ValueError("oea_zo_transform='local_mix_then_ea' is only supported with alignment='ea_zo'.")

    subjects = sorted(subject_session_data.keys())
    if not subjects:
        raise ValueError("Empty subject_session_data.")

    diag_subjects_set = {int(s) for s in diagnostics_subjects} if diagnostics_subjects else set()

    fold_rows: List[FoldResult] = []
    models_by_subject: Dict[int, TrainedModel] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []
    train_sess_all: List[np.ndarray] = []
    test_sess_all: List[np.ndarray] = []

    class_labels = tuple(class_order)

    for subject in subjects:
        sess_map = subject_session_data[int(subject)]
        available = sorted(sess_map.keys())
        missing_train = [s for s in train_sessions if s not in sess_map]
        missing_test = [s for s in test_sessions if s not in sess_map]
        if missing_train or missing_test:
            raise ValueError(
                f"Subject {int(subject)} missing requested sessions. "
                f"train_missing={missing_train}, test_missing={missing_test}, available={available}"
            )

        X_train = np.concatenate([sess_map[s].X for s in train_sessions], axis=0)
        y_train = np.concatenate([sess_map[s].y for s in train_sessions], axis=0)
        X_test_raw = np.concatenate([sess_map[s].X for s in test_sessions], axis=0)
        y_test = np.concatenate([sess_map[s].y for s in test_sessions], axis=0)

        do_diag = diagnostics_dir is not None and int(subject) in diag_subjects_set
        zo_diag: dict | None = None
        z_test_base: np.ndarray | None = None

        if alignment == "none":
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
            X_test = X_test_raw
        else:
            base_aligner_cls = (
                LogEuclideanAligner
                if alignment in {"rpa", "rpa_zo", "tsa", "tsa_zo"}
                else EuclideanAligner
            )
            base_train = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_train)
            base_test = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_test_raw)
            z_train = base_train.transform(X_train)
            z_test = base_test.transform(X_test_raw)
            z_test_base = X_test_raw if use_post_ea else z_test

            if alignment in {"ea", "rpa"}:
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                X_test = z_test
            elif alignment == "oea_cov":
                # Session-wise cov-eig alignment: align test eigen-basis to train eigen-basis.
                _evals_ref, u_ref = sorted_eigh(base_train.cov_)
                q_t = u_ref @ base_test.eigvecs_.T
                q_t = blend_with_identity(q_t, oea_q_blend)
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                X_test = apply_spatial_transform(q_t, z_test)
            elif alignment == "tsa":
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                q_tsa = _compute_tsa_target_rotation(
                    z_train=z_train,
                    y_train=y_train,
                    z_target=z_test,
                    model=model,
                    class_order=class_labels,
                    pseudo_mode=str(oea_pseudo_mode),
                    pseudo_iters=int(max(0, oea_pseudo_iters)),
                    q_blend=float(oea_q_blend),
                    pseudo_confidence=float(oea_pseudo_confidence),
                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                    pseudo_balance=bool(oea_pseudo_balance),
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                )
                X_test = apply_spatial_transform(q_tsa, z_test)
            else:
                # Discriminative signature reference from labeled train session(s).
                d_ref = class_cov_diff(
                    z_train,
                    y_train,
                    class_order=class_labels,
                    eps=oea_eps,
                    shrinkage=oea_shrinkage,
                )

                if alignment == "oea":
                    model = fit_csp_lda(z_train, y_train, n_components=n_components)
                    q_t = np.eye(z_test.shape[1], dtype=np.float64)
                    for _ in range(int(max(0, oea_pseudo_iters))):
                        X_t_cur = apply_spatial_transform(q_t, z_test)
                        proba = model.predict_proba(X_t_cur)
                        proba = _reorder_proba_columns(proba, model.classes_, list(class_labels))

                        if oea_pseudo_mode == "soft":
                            d_t = _soft_class_cov_diff(
                                z_test,
                                proba=proba,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        else:
                            y_pseudo = np.asarray(model.predict(X_t_cur))
                            keep = _select_pseudo_indices(
                                y_pseudo=y_pseudo,
                                proba=proba,
                                class_order=class_labels,
                                confidence=float(oea_pseudo_confidence),
                                topk_per_class=int(oea_pseudo_topk_per_class),
                                balance=bool(oea_pseudo_balance),
                            )
                            if keep.size == 0:
                                break
                            d_t = class_cov_diff(
                                z_test[keep],
                                y_pseudo[keep],
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )
                        q_t = orthogonal_align_symmetric(d_t, d_ref)
                        q_t = blend_with_identity(q_t, oea_q_blend)
                    X_test = apply_spatial_transform(q_t, z_test)
                else:
                    # alignment == "ea_zo" or "oea_zo": freeze classifier, optimize Q_t on unlabeled target.
                    model = fit_csp_lda(z_train, y_train, n_components=n_components)
                    if alignment == "tsa_zo":
                        q_base = _compute_tsa_target_rotation(
                            z_train=z_train,
                            y_train=y_train,
                            z_target=z_test,
                            model=model,
                            class_order=class_labels,
                            pseudo_mode=str(oea_pseudo_mode),
                            pseudo_iters=int(max(0, oea_pseudo_iters)),
                            q_blend=float(oea_q_blend),
                            pseudo_confidence=float(oea_pseudo_confidence),
                            pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                            pseudo_balance=bool(oea_pseudo_balance),
                            eps=float(oea_eps),
                            shrinkage=float(oea_shrinkage),
                        )
                        z_test_base = apply_spatial_transform(q_base, z_test)
                    else:
                        z_test_base = z_test

                    selector = str(oea_zo_selector)
                    use_stack = selector == "calibrated_stack_ridge"
                    use_ridge_guard = selector == "calibrated_ridge_guard"
                    use_ridge = selector in {"calibrated_ridge", "calibrated_ridge_guard", "calibrated_stack_ridge"}
                    use_guard = selector in {"calibrated_guard", "calibrated_ridge_guard"}
                    use_evidence = selector == "evidence"
                    use_probe_mixup = selector == "probe_mixup"
                    use_probe_mixup_hard = selector == "probe_mixup_hard"
                    use_iwcv = selector == "iwcv"
                    use_iwcv_ucb = selector == "iwcv_ucb"
                    use_dev = selector == "dev"
                    use_oracle = selector == "oracle"
                    cert = None
                    guard = None

                    if use_ridge or use_guard:
                        rng = np.random.RandomState(int(oea_zo_calib_seed) + int(subject) * 997)
                        calib_subjects = [s for s in subjects if s != int(subject)]
                        if int(oea_zo_calib_max_subjects) > 0 and int(oea_zo_calib_max_subjects) < len(
                            calib_subjects
                        ):
                            rng.shuffle(calib_subjects)
                            calib_subjects = calib_subjects[: int(oea_zo_calib_max_subjects)]

                        X_calib_rows: List[np.ndarray] = []
                        y_calib_rows: List[float] = []
                        y_guard_rows: List[int] = []
                        feat_names: tuple[str, ...] | None = None

                        for pseudo_t in calib_subjects:
                            pseudo_map = subject_session_data[int(pseudo_t)]
                            X_tr_p = np.concatenate([pseudo_map[s].X for s in train_sessions], axis=0)
                            y_tr_p = np.concatenate([pseudo_map[s].y for s in train_sessions], axis=0)
                            X_te_p = np.concatenate([pseudo_map[s].X for s in test_sessions], axis=0)
                            y_te_p = np.concatenate([pseudo_map[s].y for s in test_sessions], axis=0)

                            base_tr_p = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_tr_p)
                            base_te_p = base_aligner_cls(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_te_p)
                            z_tr_p = base_tr_p.transform(X_tr_p)
                            z_te_p = base_te_p.transform(X_te_p)

                            model_p = fit_csp_lda(z_tr_p, y_tr_p, n_components=n_components)
                            d_ref_p = class_cov_diff(
                                z_tr_p,
                                y_tr_p,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )

                            z_te_p_base = X_te_p if use_post_ea else z_te_p
                            if alignment == "tsa_zo":
                                q_base_p = _compute_tsa_target_rotation(
                                    z_train=z_tr_p,
                                    y_train=y_tr_p,
                                    z_target=z_te_p,
                                    model=model_p,
                                    class_order=class_labels,
                                    pseudo_mode=str(oea_pseudo_mode),
                                    pseudo_iters=int(max(0, oea_pseudo_iters)),
                                    q_blend=float(oea_q_blend),
                                    pseudo_confidence=float(oea_pseudo_confidence),
                                    pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                                    pseudo_balance=bool(oea_pseudo_balance),
                                    eps=float(oea_eps),
                                    shrinkage=float(oea_shrinkage),
                                )
                                z_te_p_base = apply_spatial_transform(q_base_p, z_te_p)

                            marginal_prior_p: np.ndarray | None = None
                            if oea_zo_marginal_mode == "kl_prior":
                                if oea_zo_marginal_prior == "uniform":
                                    marginal_prior_p = np.ones(len(class_labels), dtype=np.float64) / float(
                                        len(class_labels)
                                    )
                                elif oea_zo_marginal_prior == "source":
                                    counts = np.array([(y_tr_p == c).sum() for c in class_labels], dtype=np.float64)
                                    marginal_prior_p = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                                else:
                                    proba_id = model_p.predict_proba(z_te_p if use_post_ea else z_te_p_base)
                                    proba_id = _reorder_proba_columns(
                                        proba_id, model_p.classes_, list(class_labels)
                                    )
                                    marginal_prior_p = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                                    marginal_prior_p = marginal_prior_p / float(np.sum(marginal_prior_p))
                                mix = float(oea_zo_marginal_prior_mix)
                                if mix > 0.0 and marginal_prior_p is not None:
                                    u = np.ones_like(marginal_prior_p) / float(marginal_prior_p.shape[0])
                                    marginal_prior_p = (1.0 - mix) * marginal_prior_p + mix * u
                                    marginal_prior_p = marginal_prior_p / float(np.sum(marginal_prior_p))

                            lda_ev_p = None
                            if str(oea_zo_objective) == "lda_nll" or use_stack:
                                lda_ev_p = _compute_lda_evidence_params(
                                    model=model_p,
                                    X_train=z_tr_p,
                                    y_train=y_tr_p,
                                    class_order=class_labels,
                                )

                            _q_sel, diag_p = _optimize_qt_oea_zo(
                                z_t=z_te_p_base,
                                model=model_p,
                                class_order=class_labels,
                                d_ref=d_ref_p,
                                lda_evidence=lda_ev_p,
                                channel_names=channel_names,
                                eps=float(oea_eps),
                                shrinkage=float(oea_shrinkage),
                                pseudo_mode=str(oea_pseudo_mode),
                                warm_start=str(oea_zo_warm_start),
                                warm_iters=int(oea_zo_warm_iters),
                                q_blend=float(oea_q_blend),
                                objective=str(oea_zo_objective),
                                transform=str(oea_zo_transform),
                                localmix_neighbors=int(oea_zo_localmix_neighbors),
                                localmix_self_bias=float(oea_zo_localmix_self_bias),
                                infomax_lambda=float(oea_zo_infomax_lambda),
                                reliable_metric=str(oea_zo_reliable_metric),
                                reliable_threshold=float(oea_zo_reliable_threshold),
                                reliable_alpha=float(oea_zo_reliable_alpha),
                                trust_lambda=float(oea_zo_trust_lambda),
                                trust_q0=str(oea_zo_trust_q0),
                                marginal_mode=str(oea_zo_marginal_mode),
                                marginal_beta=float(oea_zo_marginal_beta),
                                marginal_tau=float(oea_zo_marginal_tau),
                                marginal_prior=marginal_prior_p,
                                bilevel_iters=int(oea_zo_bilevel_iters),
                                bilevel_temp=float(oea_zo_bilevel_temp),
                                bilevel_step=float(oea_zo_bilevel_step),
                                bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                                bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                holdout_fraction=float(oea_zo_holdout_fraction),
                                fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                                iters=int(oea_zo_iters),
                                lr=float(oea_zo_lr),
                                mu=float(oea_zo_mu),
                                n_rotations=int(oea_zo_k),
                                seed=int(oea_zo_seed) + int(pseudo_t) * 997,
                                l2=float(oea_zo_l2),
                                pseudo_confidence=float(oea_pseudo_confidence),
                                pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                                pseudo_balance=bool(oea_pseudo_balance),
                                return_diagnostics=True,
                            )

                            recs = list(diag_p.get("records", []))
                            if not recs:
                                continue
                            feats_list: List[np.ndarray] = []
                            acc_list: List[float] = []
                            acc_id: float | None = None
                            for rec in recs:
                                if use_stack:
                                    feats, names = stacked_candidate_features_from_record(
                                        rec, n_classes=len(class_labels)
                                    )
                                else:
                                    feats, names = candidate_features_from_record(rec, n_classes=len(class_labels))
                                if feat_names is None:
                                    feat_names = names
                                Q = np.asarray(rec.get("Q"), dtype=np.float64)
                                yp = model_p.predict(apply_spatial_transform(Q, z_te_p_base))
                                acc = float(accuracy_score(y_te_p, yp))
                                if str(rec.get("kind", "")) == "identity":
                                    acc_id = acc
                                feats_list.append(feats)
                                acc_list.append(acc)
                            if acc_id is None:
                                continue
                            for feats, acc in zip(feats_list, acc_list):
                                improve = float(acc - float(acc_id))
                                y_calib_rows.append(float(improve))
                                y_guard_rows.append(
                                    1 if float(improve) >= float(oea_zo_calib_guard_margin) else 0
                                )
                                X_calib_rows.append(feats)

                        if X_calib_rows and feat_names is not None:
                            X_cal = np.stack(X_calib_rows, axis=0)
                            if use_ridge:
                                cert = train_ridge_certificate(
                                    X_cal,
                                    np.asarray(y_calib_rows, dtype=np.float64),
                                    feature_names=feat_names,
                                    alpha=float(oea_zo_calib_ridge_alpha),
                                )
                            if use_guard:
                                y_guard = np.asarray(y_guard_rows, dtype=int).reshape(-1)
                                if np.unique(y_guard).size >= 2:
                                    guard = train_logistic_guard(
                                        X_cal,
                                        y_guard,
                                        feature_names=feat_names,
                                        c=float(oea_zo_calib_guard_c),
                                    )
                                else:
                                    guard = None

                    marginal_prior_vec: np.ndarray | None = None
                    if oea_zo_marginal_mode == "kl_prior":
                        if oea_zo_marginal_prior == "uniform":
                            marginal_prior_vec = np.ones(len(class_labels), dtype=np.float64) / float(
                                len(class_labels)
                            )
                        elif oea_zo_marginal_prior == "source":
                            counts = np.array([(y_train == c).sum() for c in class_labels], dtype=np.float64)
                            marginal_prior_vec = (counts + 1e-3) / float(np.sum(counts + 1e-3))
                        else:
                            proba_id = model.predict_proba(z_test)
                            proba_id = _reorder_proba_columns(proba_id, model.classes_, list(class_labels))
                            marginal_prior_vec = np.mean(np.clip(proba_id, 1e-12, 1.0), axis=0)
                            marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))
                        mix = float(oea_zo_marginal_prior_mix)
                        if mix > 0.0 and marginal_prior_vec is not None:
                            u = np.ones_like(marginal_prior_vec) / float(marginal_prior_vec.shape[0])
                            marginal_prior_vec = (1.0 - mix) * marginal_prior_vec + mix * u
                            marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

                    want_diag = (
                        bool(do_diag)
                        or (use_ridge and cert is not None)
                        or (use_guard and guard is not None)
                        or use_evidence
                        or use_probe_mixup
                        or use_probe_mixup_hard
                        or use_iwcv
                        or use_iwcv_ucb
                        or use_dev
                        or use_oracle
                    )
                    if use_oracle:
                        want_diag = True
                    lda_ev = None
                    if str(oea_zo_objective) == "lda_nll" or use_evidence or use_stack or bool(do_diag):
                        lda_ev = _compute_lda_evidence_params(
                            model=model,
                            X_train=z_train,
                            y_train=y_train,
                            class_order=class_labels,
                        )
                    opt_res = _optimize_qt_oea_zo(
                        z_t=z_test_base if z_test_base is not None else z_test,
                        model=model,
                        class_order=class_labels,
                        d_ref=d_ref,
                        lda_evidence=lda_ev,
                        channel_names=channel_names,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
                        localmix_neighbors=int(oea_zo_localmix_neighbors),
                        localmix_self_bias=float(oea_zo_localmix_self_bias),
                        infomax_lambda=float(oea_zo_infomax_lambda),
                        reliable_metric=str(oea_zo_reliable_metric),
                        reliable_threshold=float(oea_zo_reliable_threshold),
                        reliable_alpha=float(oea_zo_reliable_alpha),
                        trust_lambda=float(oea_zo_trust_lambda),
                        trust_q0=str(oea_zo_trust_q0),
                        marginal_mode=str(oea_zo_marginal_mode),
                        marginal_beta=float(oea_zo_marginal_beta),
                        marginal_tau=float(oea_zo_marginal_tau),
                        marginal_prior=marginal_prior_vec,
                        bilevel_iters=int(oea_zo_bilevel_iters),
                        bilevel_temp=float(oea_zo_bilevel_temp),
                        bilevel_step=float(oea_zo_bilevel_step),
                        bilevel_coverage_target=float(oea_zo_bilevel_coverage_target),
                        bilevel_coverage_power=float(oea_zo_bilevel_coverage_power),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
                        min_improvement=float(oea_zo_min_improvement),
                        holdout_fraction=float(oea_zo_holdout_fraction),
                        fallback_min_marginal_entropy=float(oea_zo_fallback_min_marginal_entropy),
                        iters=int(oea_zo_iters),
                        lr=float(oea_zo_lr),
                        mu=float(oea_zo_mu),
                        n_rotations=int(oea_zo_k),
                        seed=int(oea_zo_seed) + int(subject) * 997,
                        l2=float(oea_zo_l2),
                        pseudo_confidence=float(oea_pseudo_confidence),
                        pseudo_topk_per_class=int(oea_pseudo_topk_per_class),
                        pseudo_balance=bool(oea_pseudo_balance),
                        return_diagnostics=bool(want_diag),
                    )
                    if want_diag:
                        q_t, zo_diag = opt_res
                    else:
                        q_t = opt_res

                    if zo_diag is not None:
                        selected: dict | None = None
                        if use_oracle:
                            best_rec = None
                            best_acc = -1.0
                            for rec in zo_diag.get("records", []):
                                Q = np.asarray(rec.get("Q"), dtype=np.float64)
                                yp = model.predict(
                                    apply_spatial_transform(Q, z_test_base if z_test_base is not None else z_test)
                                )
                                acc = float(accuracy_score(y_test, yp))
                                if acc > best_acc:
                                    best_acc = acc
                                    best_rec = rec
                            selected = best_rec
                        elif use_evidence:
                            selected = select_by_evidence_nll(
                                zo_diag.get("records", []),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                            )
                        elif use_probe_mixup:
                            selected = select_by_probe_mixup(
                                zo_diag.get("records", []),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                            )
                        elif use_probe_mixup_hard:
                            selected = select_by_probe_mixup_hard(
                                zo_diag.get("records", []),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                            )
                        elif use_iwcv:
                            selected = select_by_iwcv_nll(
                                zo_diag.get("records", []),
                                model=model,
                                z_source=z_train,
                                y_source=y_train,
                                z_target=z_test_base if z_test_base is not None else z_test,
                                class_order=class_labels,
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                seed=int(oea_zo_seed) + int(subject) * 997,
                            )
                        elif use_iwcv_ucb:
                            selected = select_by_iwcv_ucb(
                                zo_diag.get("records", []),
                                model=model,
                                z_source=z_train,
                                y_source=y_train,
                                z_target=z_test_base if z_test_base is not None else z_test,
                                class_order=class_labels,
                                kappa=float(oea_zo_iwcv_kappa),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                seed=int(oea_zo_seed) + int(subject) * 997,
                            )
                        elif use_dev:
                            selected = select_by_dev_nll(
                                zo_diag.get("records", []),
                                model=model,
                                z_source=z_train,
                                y_source=y_train,
                                z_target=z_test_base if z_test_base is not None else z_test,
                                class_order=class_labels,
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                min_improvement=float(oea_zo_min_improvement),
                                seed=int(oea_zo_seed) + int(subject) * 997,
                            )
                        elif use_ridge_guard and cert is not None and guard is not None:
                            selected = select_by_guarded_predicted_improvement(
                                zo_diag.get("records", []),
                                cert=cert,
                                guard=guard,
                                n_classes=len(class_labels),
                                threshold=float(oea_zo_calib_guard_threshold),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                            )
                        elif use_ridge and cert is not None:
                            selected = select_by_predicted_improvement(
                                zo_diag.get("records", []),
                                cert=cert,
                                n_classes=len(class_labels),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                                feature_set="stacked" if use_stack else "base",
                            )
                        elif use_guard and guard is not None:
                            selected = select_by_guarded_objective(
                                zo_diag.get("records", []),
                                guard=guard,
                                n_classes=len(class_labels),
                                threshold=float(oea_zo_calib_guard_threshold),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
                            )
                        if selected is not None:
                            q_t = np.asarray(selected.get("Q"), dtype=np.float64)

                    X_test = apply_spatial_transform(q_t, z_test_base if z_test_base is not None else z_test)

        y_pred = np.asarray(model.predict(X_test))
        y_proba = np.asarray(model.predict_proba(X_test))
        y_proba = _reorder_proba_columns(y_proba, model.classes_, list(class_order))

        if zo_diag is not None and do_diag and diagnostics_dir is not None:
            _write_zo_diagnostics(
                zo_diag,
                out_dir=Path(diagnostics_dir),
                tag=str(diagnostics_tag),
                subject=int(subject),
                model=model,
                z_t=z_test_base if z_test_base is not None else X_test_raw,
                y_true=y_test,
                class_order=class_order,
            )

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_order=class_order,
            average=average,
        )

        fold_rows.append(
            FoldResult(
                subject=int(subject),
                n_train=int(len(y_train)),
                n_test=int(len(y_test)),
                **metrics,
            )
        )
        models_by_subject[int(subject)] = model
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        subj_all.append(np.full(shape=(int(len(y_test)),), fill_value=int(subject), dtype=int))
        trial_all.append(np.arange(int(len(y_test)), dtype=int))
        train_sess_all.append(np.full(shape=(int(len(y_test)),), fill_value=",".join(train_sessions), dtype=object))
        test_sess_all.append(np.full(shape=(int(len(y_test)),), fill_value=",".join(test_sessions), dtype=object))

    results_df = pd.DataFrame([asdict(r) for r in fold_rows]).sort_values("subject")
    y_true_cat = np.concatenate(y_true_all, axis=0)
    y_pred_cat = np.concatenate(y_pred_all, axis=0)
    y_proba_cat = np.concatenate(y_proba_all, axis=0)
    subj_cat = np.concatenate(subj_all, axis=0)
    trial_cat = np.concatenate(trial_all, axis=0)
    tr_sess_cat = np.concatenate(train_sess_all, axis=0)
    te_sess_cat = np.concatenate(test_sess_all, axis=0)

    pred_df = pd.DataFrame(
        {
            "subject": subj_cat,
            "train_sessions": tr_sess_cat,
            "test_sessions": te_sess_cat,
            "trial": trial_cat,
            "y_true": y_true_cat,
            "y_pred": y_pred_cat,
        }
    )
    for i, c in enumerate(list(class_order)):
        pred_df[f"proba_{c}"] = y_proba_cat[:, int(i)]

    return (
        results_df,
        pred_df,
        y_true_cat,
        y_pred_cat,
        y_proba_cat,
        list(class_order),
        models_by_subject,
    )
