from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .data import SubjectData
from .alignment import (
    EuclideanAligner,
    apply_spatial_transform,
    blend_with_identity,
    class_cov_diff,
    orthogonal_align_symmetric,
    sorted_eigh,
)
from .model import TrainedModel, fit_csp_lda
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

    subjects = sorted(subject_data.keys())
    fold_rows: List[FoldResult] = []
    models_by_subject: Dict[int, TrainedModel] = {}

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    subj_all: List[np.ndarray] = []
    trial_all: List[np.ndarray] = []

    if alignment not in {"none", "ea", "ea_zo", "oea_cov", "oea", "oea_zo"}:
        raise ValueError("alignment must be one of: 'none', 'ea', 'ea_zo', 'oea_cov', 'oea', 'oea_zo'")

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
    if float(oea_zo_l2) < 0.0:
        raise ValueError("oea_zo_l2 must be >= 0.")

    diag_subjects_set = {int(s) for s in diagnostics_subjects} if diagnostics_subjects else set()

    # Fast path: subject-wise EA can be precomputed once.
    if alignment in {"ea", "ea_zo"}:
        aligned: Dict[int, SubjectData] = {}
        for s, sd in subject_data.items():
            X_aligned = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
        subject_data = aligned

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        do_diag = diagnostics_dir is not None and int(test_subject) in diag_subjects_set
        zo_diag: dict | None = None
        z_test_base: np.ndarray | None = None

        # Build per-fold aligned train/test data if needed.
        if alignment in {"none", "ea"}:
            X_test = subject_data[test_subject].X
            y_test = subject_data[test_subject].y

            X_train_parts = [subject_data[s].X for s in train_subjects]
            y_train_parts = [subject_data[s].y for s in train_subjects]
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
        elif alignment == "ea_zo":
            # Train on EA-whitened source data (no Q_s selection), then adapt only Q_t at test time.
            class_labels = tuple([str(c) for c in class_order])

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
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
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

            z_t = subject_data[int(test_subject)].X
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
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                pseudo_mode=str(oea_pseudo_mode),
                warm_start=str(oea_zo_warm_start),
                warm_iters=int(oea_zo_warm_iters),
                q_blend=float(oea_q_blend),
                objective=str(oea_zo_objective),
                transform=str(oea_zo_transform),
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
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    pseudo_mode=str(oea_pseudo_mode),
                    warm_start=str(oea_zo_warm_start),
                    warm_iters=int(oea_zo_warm_iters),
                    q_blend=float(oea_q_blend),
                    objective=str(oea_zo_objective),
                    transform=str(oea_zo_transform),
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

        if alignment not in {"oea", "oea_zo", "ea_zo"}:
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
    if alignment not in {"none", "ea", "ea_zo", "oea_cov", "oea", "oea_zo"}:
        raise ValueError("alignment must be one of: 'none', 'ea', 'ea_zo', 'oea_cov', 'oea', 'oea_zo'")
    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")

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
            ea_train = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_train)
            ea_test = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_test_raw)
            z_train = ea_train.transform(X_train)
            z_test = ea_test.transform(X_test_raw)
            z_test_base = z_test

            if alignment == "ea":
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                X_test = z_test
            elif alignment == "oea_cov":
                # Session-wise cov-eig alignment: align test eigen-basis to train eigen-basis.
                _evals_ref, u_ref = sorted_eigh(ea_train.cov_)
                q_t = u_ref @ ea_test.eigvecs_.T
                q_t = blend_with_identity(q_t, oea_q_blend)
                model = fit_csp_lda(z_train, y_train, n_components=n_components)
                X_test = apply_spatial_transform(q_t, z_test)
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

                            ea_tr_p = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_tr_p)
                            ea_te_p = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit(X_te_p)
                            z_tr_p = ea_tr_p.transform(X_tr_p)
                            z_te_p = ea_te_p.transform(X_te_p)

                            model_p = fit_csp_lda(z_tr_p, y_tr_p, n_components=n_components)
                            d_ref_p = class_cov_diff(
                                z_tr_p,
                                y_tr_p,
                                class_order=class_labels,
                                eps=oea_eps,
                                shrinkage=oea_shrinkage,
                            )

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
                                    proba_id = model_p.predict_proba(z_te_p)
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
                                z_t=z_te_p,
                                model=model_p,
                                class_order=class_labels,
                                d_ref=d_ref_p,
                                lda_evidence=lda_ev_p,
                                eps=float(oea_eps),
                                shrinkage=float(oea_shrinkage),
                                pseudo_mode=str(oea_pseudo_mode),
                                warm_start=str(oea_zo_warm_start),
                                warm_iters=int(oea_zo_warm_iters),
                                q_blend=float(oea_q_blend),
                                objective=str(oea_zo_objective),
                                transform=str(oea_zo_transform),
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
                                yp = model_p.predict(apply_spatial_transform(Q, z_te_p))
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
                        z_t=z_test,
                        model=model,
                        class_order=class_labels,
                        d_ref=d_ref,
                        lda_evidence=lda_ev,
                        eps=float(oea_eps),
                        shrinkage=float(oea_shrinkage),
                        pseudo_mode=str(oea_pseudo_mode),
                        warm_start=str(oea_zo_warm_start),
                        warm_iters=int(oea_zo_warm_iters),
                        q_blend=float(oea_q_blend),
                        objective=str(oea_zo_objective),
                        transform=str(oea_zo_transform),
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
                                yp = model.predict(apply_spatial_transform(Q, z_test))
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
                                z_target=z_test,
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
                                z_target=z_test,
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
                                z_target=z_test,
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

                    X_test = apply_spatial_transform(q_t, z_test)

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
