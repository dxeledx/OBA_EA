from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

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
    select_by_evidence_nll,
    select_by_guarded_objective,
    select_by_predicted_improvement,
    train_logistic_guard,
    train_ridge_certificate,
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


def _reorder_proba_columns(
    proba: np.ndarray, model_classes: Sequence[str], class_order: Sequence[str]
) -> np.ndarray:
    model_classes = list(model_classes)
    indices = []
    for c in class_order:
        if c not in model_classes:
            raise ValueError(f"Class '{c}' not found in model classes {model_classes}.")
        indices.append(model_classes.index(c))
    return proba[:, indices]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    class_order: Sequence[str],
    average: str = "macro",
) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(class_order),
        average=average,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=list(class_order))

    n_classes = len(class_order)
    if n_classes == 2:
        # For binary, sklearn expects either:
        # - y_true as 1D {0,1} and y_score as (n_samples,), or
        # - y_true as (n_samples,1) and y_score as (n_samples,)
        # Use the 2nd class in `class_order` as positive label.
        y_true_bin = (y_true == class_order[1]).astype(int)
        auc = roc_auc_score(y_true_bin, y_proba[:, 1])
    else:
        # Multiclass AUC (macro OVR).
        y_true_bin = label_binarize(y_true, classes=list(class_order))
        auc = roc_auc_score(
            y_true_bin,
            y_proba,
            average=average,
            multi_class="ovr",
        )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "kappa": float(kappa),
    }


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
    if oea_zo_selector not in {"objective", "evidence", "calibrated_ridge", "calibrated_guard", "oracle"}:
        raise ValueError(
            "oea_zo_selector must be one of: "
            "'objective', 'evidence', 'calibrated_ridge', 'calibrated_guard', 'oracle'."
        )
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
            use_ridge = selector == "calibrated_ridge"
            use_guard = selector == "calibrated_guard"
            use_evidence = selector == "evidence"
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
                    if str(oea_zo_objective) == "lda_nll":
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
                or use_oracle
            )
            lda_ev = None
            if str(oea_zo_objective) == "lda_nll" or use_evidence:
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
                elif use_ridge and cert is not None:
                    selected = select_by_predicted_improvement(
                        zo_diag.get("records", []),
                        cert=cert,
                        n_classes=len(class_labels),
                        drift_mode=str(oea_zo_drift_mode),
                        drift_gamma=float(oea_zo_drift_gamma),
                        drift_delta=float(oea_zo_drift_delta),
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
                use_oracle = selector == "oracle"
                want_diag = bool(do_diag) or use_evidence or use_oracle
                lda_ev = None
                if str(oea_zo_objective) == "lda_nll" or use_evidence:
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
                    use_ridge = selector == "calibrated_ridge"
                    use_guard = selector == "calibrated_guard"
                    use_evidence = selector == "evidence"
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
                            if str(oea_zo_objective) == "lda_nll":
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
                        or use_oracle
                    )
                    if use_oracle:
                        want_diag = True
                    lda_ev = None
                    if str(oea_zo_objective) == "lda_nll" or use_evidence:
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
                        elif use_ridge and cert is not None:
                            selected = select_by_predicted_improvement(
                                zo_diag.get("records", []),
                                cert=cert,
                                n_classes=len(class_labels),
                                drift_mode=str(oea_zo_drift_mode),
                                drift_gamma=float(oea_zo_drift_gamma),
                                drift_delta=float(oea_zo_drift_delta),
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


def _write_zo_diagnostics(
    zo_diag: dict,
    *,
    out_dir: Path,
    tag: str,
    subject: int,
    model: TrainedModel,
    z_t: np.ndarray,
    y_true: np.ndarray,
    class_order: Sequence[str],
) -> None:
    """Write per-subject EA-ZO/OEA-ZO diagnostics (analysis-only; uses labels)."""

    out_dir = Path(out_dir)
    tag = tag or "zo"
    diag_dir = out_dir / "diagnostics" / str(tag) / f"subject_{int(subject):02d}"
    diag_dir.mkdir(parents=True, exist_ok=True)

    records = list(zo_diag.get("records", []))
    if not records:
        return

    # Candidate evaluation on the labeled target fold (analysis-only).
    rows = []
    for idx, rec in enumerate(records):
        Q = np.asarray(rec.get("Q"), dtype=np.float64)
        X = apply_spatial_transform(Q, z_t)
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        row = {
            "idx": int(idx),
            "kind": str(rec.get("kind", "")),
            "iter": int(rec.get("iter", -1)),
            "order": int(rec.get("order", idx)),
            "objective": float(rec.get("objective", np.nan)),
            "score": float(rec.get("score", np.nan)),
            "objective_base": float(rec.get("objective_base", np.nan)),
            "evidence_nll_best": float(rec.get("evidence_nll_best", np.nan)),
            "evidence_nll_full": float(rec.get("evidence_nll_full", np.nan)),
            "pen_marginal": float(rec.get("pen_marginal", np.nan)),
            "pen_trust": float(rec.get("pen_trust", np.nan)),
            "pen_l2": float(rec.get("pen_l2", np.nan)),
            "drift_best": float(rec.get("drift_best", np.nan)),
            "drift_full": float(rec.get("drift_full", np.nan)),
            "mean_entropy": float(rec.get("mean_entropy", np.nan)),
            "mean_confidence": float(rec.get("mean_confidence", np.nan)),
            "entropy_bar": float(rec.get("entropy_bar", np.nan)),
            "n_keep": int(rec.get("n_keep", -1)),
            "n_best_total": int(rec.get("n_best_total", -1)),
            "n_full_total": int(rec.get("n_full_total", -1)),
            "accuracy": float(acc),
        }
        p_bar = np.asarray(rec.get("p_bar_full", []), dtype=np.float64).reshape(-1)
        for k, name in enumerate(class_order):
            row[f"pbar_{name}"] = float(p_bar[k]) if k < p_bar.shape[0] else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["order", "idx"])
    df.to_csv(diag_dir / "candidates.csv", index=False)

    obj = df["objective"].to_numpy(dtype=np.float64)
    acc = df["accuracy"].to_numpy(dtype=np.float64)
    pearson = float(np.corrcoef(obj, acc)[0, 1]) if obj.size >= 2 else float("nan")
    # Spearman via ranks (no ties handling needed for a quick diagnostic).
    obj_r = obj.argsort().argsort().astype(np.float64)
    acc_r = acc.argsort().argsort().astype(np.float64)
    spearman = float(np.corrcoef(obj_r, acc_r)[0, 1]) if obj.size >= 2 else float("nan")

    ev = df["evidence_nll_best"].to_numpy(dtype=np.float64)
    pearson_ev = float("nan")
    spearman_ev = float("nan")
    if ev.size >= 2 and np.isfinite(ev).any():
        pearson_ev = float(np.corrcoef(ev, acc)[0, 1])
        ev_r = ev.argsort().argsort().astype(np.float64)
        spearman_ev = float(np.corrcoef(ev_r, acc_r)[0, 1])

    prior = zo_diag.get("marginal_prior")
    prior_arr = None if prior is None else np.asarray(prior, dtype=np.float64).reshape(-1)

    # Plots
    from .plots import plot_class_marginal_trajectory, plot_objective_vs_accuracy_scatter

    p_cols = [c for c in df.columns if c.startswith("pbar_")]
    p_bars = df[p_cols].to_numpy(dtype=np.float64)
    x = df["order"].to_numpy(dtype=int)
    plot_class_marginal_trajectory(
        p_bars,
        class_order=class_order,
        x=x,
        prior=prior_arr,
        output_path=diag_dir / "pbar_trajectory.png",
        title=f"Subject {subject} — p̄ trajectory ({tag})",
    )
    plot_objective_vs_accuracy_scatter(
        obj,
        acc,
        output_path=diag_dir / "objective_vs_accuracy.png",
        title=f"Subject {subject} — objective vs acc (pearson={pearson:.3f}, spearman={spearman:.3f})",
    )
    if np.isfinite(ev).any():
        plot_objective_vs_accuracy_scatter(
            ev,
            acc,
            output_path=diag_dir / "evidence_vs_accuracy.png",
            title=f"Subject {subject} — evidence(-log p) vs acc (pearson={pearson_ev:.3f}, spearman={spearman_ev:.3f})",
        )
    if "score" in df.columns and np.isfinite(df["score"].to_numpy()).any():
        score = df["score"].to_numpy(dtype=np.float64)
        pearson_s = float(np.corrcoef(score, acc)[0, 1]) if score.size >= 2 else float("nan")
        score_r = score.argsort().argsort().astype(np.float64)
        spearman_s = float(np.corrcoef(score_r, acc_r)[0, 1]) if score.size >= 2 else float("nan")
        plot_objective_vs_accuracy_scatter(
            score,
            acc,
            output_path=diag_dir / "score_vs_accuracy.png",
            title=f"Subject {subject} — score vs acc (pearson={pearson_s:.3f}, spearman={spearman_s:.3f})",
        )

    # Small text summary
    best_by_evidence = -1
    if np.isfinite(ev).any():
        try:
            best_by_evidence = int(df.loc[df["evidence_nll_best"].idxmin(), "idx"])
        except Exception:
            best_by_evidence = -1
    lines = [
        f"tag: {tag}",
        f"subject: {subject}",
        f"n_candidates: {len(df)}",
        f"pearson(objective, accuracy): {pearson:.6f}",
        f"spearman(objective, accuracy): {spearman:.6f}",
        f"pearson(evidence, accuracy): {pearson_ev:.6f}",
        f"spearman(evidence, accuracy): {spearman_ev:.6f}",
        f"best_by_objective: idx={int(df.loc[df['objective'].idxmin(), 'idx'])}",
        f"best_by_evidence: idx={best_by_evidence}",
        f"best_by_accuracy: idx={int(df.loc[df['accuracy'].idxmax(), 'idx'])}",
    ]
    if "score" in df.columns and np.isfinite(df["score"].to_numpy()).any():
        lines.append(f"best_by_score: idx={int(df.loc[df['score'].idxmin(), 'idx'])}")
    if prior_arr is not None:
        lines.append("marginal_prior: " + ", ".join([f"{x:.4f}" for x in prior_arr.tolist()]))
    (diag_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_pseudo_indices(
    *,
    y_pseudo: np.ndarray,
    proba: np.ndarray,
    class_order: Sequence[str],
    confidence: float,
    topk_per_class: int,
    balance: bool,
) -> np.ndarray:
    """Select trial indices to use for pseudo-label covariance estimation.

    This is a simple stabilization layer for OEA(TTA-Q) to reduce the impact of noisy pseudo labels.
    """

    y_pseudo = np.asarray(y_pseudo)
    proba = np.asarray(proba, dtype=np.float64)
    class_order = [str(c) for c in class_order]
    n_classes = len(class_order)
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")
    if proba.ndim != 2 or proba.shape[1] != n_classes:
        raise ValueError(f"Expected proba shape (n_samples,{n_classes}); got {proba.shape}.")

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    try:
        pred_idx = np.fromiter((class_to_idx[str(c)] for c in y_pseudo), dtype=int, count=len(y_pseudo))
    except KeyError as e:
        raise ValueError(f"Pseudo label contains unknown class '{e.args[0]}'.") from e

    conf = proba[np.arange(len(y_pseudo)), pred_idx]
    keep = conf >= float(confidence)
    if not np.any(keep):
        return np.array([], dtype=int)

    idx_by_class: Dict[str, np.ndarray] = {}
    for c in class_order:
        idx_by_class[c] = np.where(keep & (y_pseudo == c))[0]
    nonempty = [c for c in class_order if idx_by_class[c].size > 0]
    if len(nonempty) < 2:
        return np.array([], dtype=int)

    if int(topk_per_class) > 0:
        k = int(topk_per_class)
        for c in nonempty:
            idx = idx_by_class[c]
            ci = class_to_idx[c]
            idx = idx[np.argsort(proba[idx, ci])[::-1][:k]]
            idx_by_class[c] = idx
        nonempty = [c for c in class_order if idx_by_class[c].size > 0]
        if len(nonempty) < 2:
            return np.array([], dtype=int)

    if balance:
        k = int(min(idx_by_class[c].size for c in nonempty))
        for c in nonempty:
            idx_by_class[c] = idx_by_class[c][:k]

    out = np.concatenate([idx_by_class[c] for c in nonempty], axis=0)
    return np.asarray(out, dtype=int)


def _soft_class_cov_diff(
    X: np.ndarray,
    *,
    proba: np.ndarray,
    class_order: Sequence[str],
    eps: float,
    shrinkage: float,
) -> np.ndarray:
    """Soft pseudo-label covariance signature using class probabilities as weights.

    For each trial i, compute Ci = Xi Xi^T. Then:
      Σ_c = sum_i w_{i,c} Ci / sum_i w_{i,c}

    - Binary: return Δ = Σ_1 - Σ_0.
    - Multiclass: return between-class scatter D = Σ_k π_k (Σ_k - Σ̄)(Σ_k - Σ̄),
      where π_k is the (soft) class mass and Σ̄ = Σ_k π_k Σ_k.
    """

    X = np.asarray(X, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    class_order = [str(c) for c in class_order]
    n_classes = len(class_order)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n_trials,n_channels,n_times); got {X.shape}.")
    if proba.ndim != 2 or proba.shape[0] != X.shape[0] or proba.shape[1] != n_classes:
        raise ValueError(f"Expected proba shape (n_trials,{n_classes}); got {proba.shape}.")

    w = np.clip(proba, 0.0, 1.0)
    row_sum = np.sum(w, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-12)
    w = w / row_sum
    w_sum = np.sum(w, axis=0)  # (n_classes,)
    if float(np.min(w_sum)) <= 0.0:
        raise ValueError("Soft pseudo-label weights degenerate (some class mass sums to zero).")

    n_trials, n_channels, _ = X.shape
    # Trial covariances Ci = Xi Xi^T.
    cov_trials = np.einsum("nct,ndt->ncd", X, X, optimize=True)
    covs = np.einsum("nk,ncd->kcd", w, cov_trials, optimize=True)
    covs = covs / w_sum[:, None, None]
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

    if shrinkage > 0.0:
        alpha = float(shrinkage)
        eye = np.eye(n_channels, dtype=np.float64)
        traces = np.trace(covs, axis1=1, axis2=2) / float(n_channels)
        covs = (1.0 - alpha) * covs + alpha * traces[:, None, None] * eye[None, :, :]

    if n_classes == 2:
        diff = covs[1] - covs[0]
        diff = 0.5 * (diff + diff.T)
        jitter = float(eps) * float(np.max(np.abs(np.diag(diff))) + 1.0)
        diff = diff + jitter * np.eye(n_channels, dtype=np.float64)
        return diff

    pi = w_sum / float(np.sum(w_sum))
    sw = np.einsum("k,kcd->cd", pi, covs, optimize=True)
    sw = 0.5 * (sw + sw.T)
    delta = covs - sw[None, :, :]
    sb = np.einsum("k,kce,ked->cd", pi, delta, delta, optimize=True)
    sb = 0.5 * (sb + sb.T)

    eigvals, eigvecs = np.linalg.eigh(sw)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Deterministic signs for stability.
    for i in range(eigvecs.shape[1]):
        col = eigvecs[:, i]
        j = int(np.argmax(np.abs(col)))
        if col[j] < 0:
            eigvecs[:, i] = -col
    floor = float(eps) * float(np.max(eigvals)) if np.max(eigvals) > 0 else float(eps)
    eigvals = np.maximum(eigvals, floor)
    sw_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    sig = sw_inv_sqrt @ sb @ sw_inv_sqrt
    sig = 0.5 * (sig + sig.T)
    jitter = float(eps) * float(np.max(np.abs(np.diag(sig))) + 1.0)
    sig = sig + jitter * np.eye(n_channels, dtype=np.float64)
    return sig


def _optimize_qt_oea_zo(
    *,
    z_t: np.ndarray,
    model: TrainedModel,
    class_order: Sequence[str],
    d_ref: np.ndarray,
    lda_evidence: dict | None,
    eps: float,
    shrinkage: float,
    pseudo_mode: str,
    warm_start: str,
    warm_iters: int,
    q_blend: float,
    objective: str,
    transform: str = "orthogonal",
    infomax_lambda: float,
    reliable_metric: str,
    reliable_threshold: float,
    reliable_alpha: float,
    trust_lambda: float,
    trust_q0: str,
    marginal_mode: str,
    marginal_beta: float,
    marginal_tau: float,
    marginal_prior: np.ndarray | None,
    bilevel_iters: int,
    bilevel_temp: float,
    bilevel_step: float,
    bilevel_coverage_target: float,
    bilevel_coverage_power: float,
    drift_mode: str,
    drift_gamma: float,
    drift_delta: float,
    min_improvement: float,
    holdout_fraction: float,
    fallback_min_marginal_entropy: float,
    iters: int,
    lr: float,
    mu: float,
    n_rotations: int,
    seed: int,
    l2: float,
    pseudo_confidence: float,
    pseudo_topk_per_class: int,
    pseudo_balance: bool,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Zero-order optimize a target transform on channel space (Q or A) via a low-dim parameterization.

    This implements a practical "optimistic selection" variant for the target subject:
    freeze the trained classifier and update only Q_t using unlabeled target data.
    """

    z_t = np.asarray(z_t, dtype=np.float64)
    n_trials, n_channels, _n_times = z_t.shape
    rng = np.random.RandomState(int(seed))
    class_order = [str(c) for c in class_order]
    n_classes = len(class_order)
    if n_classes < 2:
        raise ValueError("class_order must contain at least 2 classes.")

    if pseudo_mode not in {"hard", "soft"}:
        raise ValueError("pseudo_mode must be one of: 'hard', 'soft'")
    if warm_start not in {"none", "delta"}:
        raise ValueError("warm_start must be one of: 'none', 'delta'")
    if not (0.0 <= float(holdout_fraction) < 1.0):
        raise ValueError("holdout_fraction must be in [0,1).")
    if float(fallback_min_marginal_entropy) < 0.0:
        raise ValueError("fallback_min_marginal_entropy must be >= 0.")
    if reliable_metric not in {"none", "confidence", "entropy"}:
        raise ValueError("reliable_metric must be one of: 'none', 'confidence', 'entropy'")
    if float(reliable_alpha) <= 0.0:
        raise ValueError("reliable_alpha must be > 0.")
    if reliable_metric == "confidence" and not (0.0 <= float(reliable_threshold) <= 1.0):
        raise ValueError("reliable_threshold must be in [0,1] when metric='confidence'.")
    if reliable_metric == "entropy" and float(reliable_threshold) < 0.0:
        raise ValueError("reliable_threshold must be >= 0 when metric='entropy'.")
    if float(trust_lambda) < 0.0:
        raise ValueError("trust_lambda must be >= 0.")
    if trust_q0 not in {"identity", "delta"}:
        raise ValueError("trust_q0 must be one of: 'identity', 'delta'.")
    if drift_mode not in {"none", "penalty", "hard"}:
        raise ValueError("drift_mode must be one of: 'none', 'penalty', 'hard'.")
    if float(drift_gamma) < 0.0:
        raise ValueError("drift_gamma must be >= 0.")
    if float(drift_delta) < 0.0:
        raise ValueError("drift_delta must be >= 0.")
    if marginal_mode not in {"none", "l2_uniform", "kl_uniform", "hinge_uniform", "hard_min", "kl_prior"}:
        raise ValueError(
            "marginal_mode must be one of: "
            "'none', 'l2_uniform', 'kl_uniform', 'hinge_uniform', 'hard_min', 'kl_prior'."
        )
    if float(marginal_beta) < 0.0:
        raise ValueError("marginal_beta must be >= 0.")
    if not (0.0 <= float(marginal_tau) <= 1.0):
        raise ValueError("marginal_tau must be in [0,1].")
    if float(min_improvement) < 0.0:
        raise ValueError("min_improvement must be >= 0.")
    if int(bilevel_iters) < 0:
        raise ValueError("bilevel_iters must be >= 0.")
    if float(bilevel_temp) <= 0.0:
        raise ValueError("bilevel_temp must be > 0.")
    if float(bilevel_step) < 0.0:
        raise ValueError("bilevel_step must be >= 0.")
    if not (0.0 < float(bilevel_coverage_target) <= 1.0):
        raise ValueError("bilevel_coverage_target must be in (0,1].")
    if float(bilevel_coverage_power) < 0.0:
        raise ValueError("bilevel_coverage_power must be >= 0.")

    transform = str(transform)
    if transform not in {"orthogonal", "rot_scale"}:
        raise ValueError("transform must be one of: 'orthogonal', 'rot_scale'")

    # Optional KL(π || p̄) prior (π fixed during optimization).
    marginal_prior_vec: np.ndarray | None = None
    if marginal_mode == "kl_prior":
        if marginal_prior is None:
            raise ValueError("marginal_prior must be provided when marginal_mode='kl_prior'.")
        marginal_prior_vec = np.asarray(marginal_prior, dtype=np.float64).reshape(-1)
        if marginal_prior_vec.shape[0] != int(n_classes):
            raise ValueError(
                f"marginal_prior length mismatch: expected {n_classes}, got {marginal_prior_vec.shape[0]}."
            )
        marginal_prior_vec = np.clip(marginal_prior_vec, 1e-12, 1.0)
        marginal_prior_vec = marginal_prior_vec / float(np.sum(marginal_prior_vec))

    do_diag = bool(return_diagnostics)
    diag_records: List[dict] = []

    # Unlabeled holdout split: use one subset to update (SPSA gradient estimation) and
    # the other subset to select the best iterate (reduces overfitting to the same trials).
    if float(holdout_fraction) > 0.0 and int(n_trials) > 1:
        perm = rng.permutation(int(n_trials))
        n_hold = int(round(float(holdout_fraction) * float(n_trials)))
        n_hold = max(1, min(int(n_trials) - 1, n_hold))
        idx_best = perm[:n_hold]
        idx_opt = perm[n_hold:]
        z_opt = z_t[idx_opt]
        z_best = z_t[idx_best]
    else:
        z_opt = z_t
        z_best = z_t

    # Random set of (i,j) planes; fixed per fold for reproducibility.
    pairs = _sample_givens_pairs(n_channels=n_channels, n_rotations=int(n_rotations), rng=rng)
    rot_dim = int(len(pairs))
    scale_dim = int(n_channels) if transform == "rot_scale" else 0
    theta = np.zeros(rot_dim + scale_dim, dtype=np.float64)
    best_theta = theta.copy()
    best_obj = float("inf")

    csp = model.csp
    lda = model.pipeline.named_steps["lda"]
    F = np.asarray(csp.filters_[: int(csp.n_components)], dtype=np.float64)

    # Determine whether CSP uses log(power).
    use_log = True if (getattr(csp, "log", None) is None) else bool(getattr(csp, "log"))

    max_abs_log_scale = 2.0  # exp(±2) ~= [0.135, 7.39]

    def _build_transform(theta_vec: np.ndarray) -> np.ndarray:
        theta_vec = np.asarray(theta_vec, dtype=np.float64)
        phi_vec = theta_vec[:rot_dim]
        Q = _build_q_from_givens(n_channels=n_channels, pairs=pairs, angles=phi_vec)
        if float(q_blend) < 1.0:
            Q = blend_with_identity(Q, float(q_blend))
        if transform == "rot_scale":
            log_s = np.clip(theta_vec[rot_dim:], -max_abs_log_scale, max_abs_log_scale)
            scales = np.exp(log_s).reshape(-1, 1)
            return scales * Q
        return Q

    def _proba_from_theta(theta_vec: np.ndarray, z_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = _build_transform(theta_vec)
        FQ = F @ A
        Y = np.einsum("kc,nct->nkt", FQ, z_data, optimize=True)
        power = np.mean(Y * Y, axis=2)
        power = np.maximum(power, 1e-20)
        feats = np.log(power) if use_log else power
        proba = lda.predict_proba(feats)
        return _reorder_proba_columns(proba, lda.classes_, list(class_order)), A, feats

    def _proba_from_Q(Q: np.ndarray, z_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = np.asarray(Q, dtype=np.float64)
        if Q.shape != (int(n_channels), int(n_channels)):
            raise ValueError(f"Expected transform shape ({n_channels},{n_channels}); got {Q.shape}.")
        FQ = F @ Q
        Y = np.einsum("kc,nct->nkt", FQ, z_data, optimize=True)
        power = np.mean(Y * Y, axis=2)
        power = np.maximum(power, 1e-20)
        feats = np.log(power) if use_log else power
        proba = lda.predict_proba(feats)
        return _reorder_proba_columns(proba, lda.classes_, list(class_order)), feats

    # Anchor predictions at EA (Q=I). Used for drift guard/certificate.
    proba_anchor_best, feats_anchor_best = _proba_from_Q(np.eye(int(n_channels), dtype=np.float64), z_best)
    proba_anchor_full, feats_anchor_full = _proba_from_Q(np.eye(int(n_channels), dtype=np.float64), z_t)

    def _kl_drift_vec(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Per-sample KL(p0 || p1)."""

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        p0 = np.clip(p0, 1e-12, 1.0)
        p1 = np.clip(p1, 1e-12, 1.0)
        p0 = p0 / np.sum(p0, axis=1, keepdims=True)
        p1 = p1 / np.sum(p1, axis=1, keepdims=True)
        return np.sum(p0 * (np.log(p0) - np.log(p1)), axis=1)

    def _kl_drift(p0: np.ndarray, p1: np.ndarray) -> float:
        """Mean KL(p0 || p1) across samples."""

        return float(np.mean(_kl_drift_vec(p0, p1)))

    def _score_with_drift(obj: float, drift: float) -> float:
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            return float("inf")
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            return float(obj) + float(drift_gamma) * float(drift)
        return float(obj)

    def _maybe_select_keep(proba: np.ndarray) -> np.ndarray:
        """Optionally select a reliable subset based on confidence/top-k/balance settings.

        When the user provides any of {pseudo_confidence, pseudo_topk_per_class, pseudo_balance},
        we reuse the pseudo selection logic to keep only confident trials. This is used to
        stabilize *all* ZO objectives (entropy/infomax/confidence) and not only pseudo_ce.
        """

        if float(pseudo_confidence) <= 0.0 and int(pseudo_topk_per_class) == 0 and not bool(pseudo_balance):
            return np.arange(proba.shape[0], dtype=int)
        pred_idx = np.argmax(proba, axis=1)
        classes_arr = np.asarray(class_order, dtype=object)
        y_pseudo = classes_arr[pred_idx]
        return _select_pseudo_indices(
            y_pseudo=y_pseudo,
            proba=proba,
            class_order=class_order,
            confidence=float(pseudo_confidence),
            topk_per_class=int(pseudo_topk_per_class),
            balance=bool(pseudo_balance),
        )

    q0 = np.eye(int(n_channels), dtype=np.float64)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    objective_name = str(objective)
    is_bilevel = objective_name.endswith("_bilevel")
    objective_core = objective_name[: -len("_bilevel")] if is_bilevel else objective_name
    if objective_core not in {"entropy", "infomax", "confidence", "pseudo_ce", "lda_nll"}:
        raise ValueError(
            "objective must be one of: "
            "'entropy', 'infomax', 'confidence', 'pseudo_ce', 'lda_nll', or bilevel variants ending with '_bilevel'."
        )

    def _row_entropy(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        return -np.sum(p * np.log(p), axis=1)

    def _solve_inner_wq(*, p: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Lower-level solver: continuous weights w and soft labels q for a fixed Q.

        This is an explicit (iterative) solver that:
        - assigns continuous reliability weights w_i in [0,1] (based on confidence/entropy),
        - computes soft pseudo-labels q_i via class scaling to match a prior π (weighted),
        - returns q̄ = weighted mean(q).
        """

        p = np.asarray(p, dtype=np.float64)
        n = int(p.shape[0])
        if n == 0:
            raise ValueError("Empty p in bilevel solver.")
        prior = np.asarray(prior, dtype=np.float64).reshape(-1)
        prior = np.clip(prior, 1e-12, 1.0)
        prior = prior / float(np.sum(prior))

        temp = float(bilevel_temp)
        it_n = int(bilevel_iters)
        step0 = float(bilevel_step)
        cov_target = float(bilevel_coverage_target)
        cov_pow = float(bilevel_coverage_power)

        a = np.ones(p.shape[1], dtype=np.float64)
        p_adj = np.clip(p, 1e-12, 1.0) ** (1.0 / temp)
        q = p_adj / np.sum(p_adj, axis=1, keepdims=True)
        w = np.ones(n, dtype=np.float64)

        for _ in range(max(1, it_n)):
            q_unn = p_adj * a.reshape(1, -1)
            q_unn = np.clip(q_unn, 1e-12, 1.0)
            q = q_unn / np.sum(q_unn, axis=1, keepdims=True)

            if reliable_metric == "confidence":
                conf = np.max(q, axis=1)
                w = _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
            elif reliable_metric == "entropy":
                ent = _row_entropy(q)
                w = _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
            else:
                w = np.ones(n, dtype=np.float64)

            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                w = np.ones(n, dtype=np.float64)
                w_sum = float(n)

            q_bar = np.sum(w.reshape(-1, 1) * q, axis=0) / w_sum
            q_bar = np.clip(q_bar, 1e-12, 1.0)
            q_bar = q_bar / float(np.sum(q_bar))

            coverage = float(w_sum) / float(n)
            cov_scale = min(1.0, max(0.0, coverage / cov_target))
            if cov_pow != 1.0:
                cov_scale = float(cov_scale) ** float(cov_pow)
            step = float(step0) * float(cov_scale)
            if step > 0.0:
                ratio = prior / q_bar
                ratio = np.clip(ratio, 1e-6, 1e6)
                a = a * (ratio**step)
                a = np.clip(a, 1e-6, 1e6)
                a = a / float(np.exp(np.mean(np.log(a))))

        q_unn = p_adj * a.reshape(1, -1)
        q_unn = np.clip(q_unn, 1e-12, 1.0)
        q = q_unn / np.sum(q_unn, axis=1, keepdims=True)
        w_sum = float(np.sum(w))
        q_bar = np.sum(w.reshape(-1, 1) * q, axis=0) / max(1e-12, w_sum)
        q_bar = np.clip(q_bar, 1e-12, 1.0)
        q_bar = q_bar / float(np.sum(q_bar))

        eff_n = 0.0
        denom = float(np.sum(w * w))
        if denom > 1e-12:
            eff_n = float(w_sum * w_sum) / denom

        stats = {
            "coverage": float(w_sum) / float(n),
            "w_sum": float(w_sum),
            "eff_n": float(eff_n),
            "a": a.copy(),
        }
        return w, q, q_bar, stats

    def _objective_from_proba(
        *, proba: np.ndarray, feats: np.ndarray | None, Q: np.ndarray, theta_vec: np.ndarray | None
    ) -> float:
        proba = np.asarray(proba, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        if objective_core in {"entropy", "infomax", "confidence", "lda_nll"}:
            keep = _maybe_select_keep(proba)
            if keep.size == 0:
                return 1e6
            proba = proba[keep]
            if feats is not None:
                feats = np.asarray(feats, dtype=np.float64)[keep]

        # Normalize to a valid distribution for entropy computations.
        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)

        if objective_core == "lda_nll":
            if feats is None:
                return 1e6
            if lda_evidence is None:
                raise ValueError("lda_evidence must be provided when objective='lda_nll'.")
            mu_e = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
            priors_e = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
            cov_inv_e = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
            logdet_e = float(lda_evidence.get("logdet", 0.0))
            if mu_e.ndim != 2:
                return 1e6
            if cov_inv_e.ndim != 2:
                return 1e6
            if priors_e.shape[0] != int(mu_e.shape[0]):
                return 1e6
            if feats.shape[1] != int(mu_e.shape[1]):
                return 1e6
            if cov_inv_e.shape != (int(mu_e.shape[1]), int(mu_e.shape[1])):
                return 1e6

            f = np.asarray(feats, dtype=np.float64)
            diff = f[:, None, :] - mu_e[None, :, :]
            qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv_e, diff, optimize=True)
            log_norm = float(mu_e.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet_e)
            log_gauss = -0.5 * (log_norm + qf)
            log_pr = np.log(np.clip(priors_e, 1e-12, 1.0)).reshape(1, -1)
            log_joint = log_pr + log_gauss
            m = np.max(log_joint, axis=1, keepdims=True)
            log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

            # Optional reliability weighting w_i based on posterior entropy/confidence.
            w = np.ones(p.shape[0], dtype=np.float64)
            if reliable_metric != "none":
                ent = _row_entropy(p)
                conf = np.max(p, axis=1)
                if reliable_metric == "confidence":
                    w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
                else:
                    w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                return 1e6
            val = float(-np.sum(w * log_p) / w_sum)
        elif objective_core == "pseudo_ce":
            pred_idx = np.argmax(p, axis=1)
            classes_arr = np.asarray(class_order, dtype=object)
            y_pseudo = classes_arr[pred_idx]
            keep = _select_pseudo_indices(
                y_pseudo=y_pseudo,
                proba=p,
                class_order=class_order,
                confidence=float(pseudo_confidence),
                topk_per_class=int(pseudo_topk_per_class),
                balance=bool(pseudo_balance),
            )
            if keep.size == 0:
                return 1e6
            pred_idx_k = np.argmax(p[keep], axis=1)
            conf_k = p[keep, pred_idx_k]
            conf_k = np.clip(conf_k, 1e-12, 1.0)
            nll = -np.log(conf_k)
            val = float(np.mean(conf_k * nll))
        else:
            if is_bilevel:
                if objective_core not in {"entropy", "infomax"}:
                    return 1e6
                prior = (
                    np.ones(int(n_classes), dtype=np.float64) / float(n_classes)
                    if marginal_prior_vec is None
                    else np.asarray(marginal_prior_vec, dtype=np.float64)
                )
                w, q, q_bar, stats = _solve_inner_wq(p=p, prior=prior)
                w_sum = float(stats["w_sum"])
                if w_sum <= 1e-12:
                    return 1e6

                ent_q = _row_entropy(q)
                base = float(np.sum(w * ent_q) / w_sum)
                if objective_core == "infomax":
                    ent_bar = -float(np.sum(q_bar * np.log(q_bar)))
                    base = float(base) - float(infomax_lambda) * float(ent_bar)
                val = float(base)

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    if marginal_mode == "hard_min":
                        if float(np.min(q_bar)) < tau:
                            return 1e6
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((q_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(q_bar)))
                        elif marginal_mode == "kl_prior":
                            if marginal_prior_vec is None:
                                return 1e6
                            pen = float(-np.sum(marginal_prior_vec * np.log(q_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - q_bar) ** 2))
                        val = float(val) + float(marginal_beta) * float(pen)
            else:
                ent = _row_entropy(p)
                conf = np.max(p, axis=1)

                w = np.ones(p.shape[0], dtype=np.float64)
                if reliable_metric != "none":
                    if reliable_metric == "confidence":
                        w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
                    else:
                        w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))

                w_sum = float(np.sum(w))
                if w_sum <= 1e-12:
                    return 1e6

                p_bar = np.mean(p, axis=0)
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))

                if objective_core == "entropy":
                    val = float(np.sum(w * ent) / w_sum)
                elif objective_core == "confidence":
                    val = float(np.sum(w * (1.0 - conf)) / w_sum)
                else:
                    ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
                    val = float(np.sum(w * ent) / w_sum) - float(infomax_lambda) * ent_bar

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    if marginal_mode == "hard_min":
                        if float(np.min(p_bar)) < tau:
                            return 1e6
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((p_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(p_bar)))
                        elif marginal_mode == "kl_prior":
                            if marginal_prior_vec is None:
                                return 1e6
                            pen = float(-np.sum(marginal_prior_vec * np.log(p_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - p_bar) ** 2))
                        val = float(val) + float(marginal_beta) * float(pen)

        if float(trust_lambda) > 0.0:
            val += float(trust_lambda) * float(np.mean((Q - q0) ** 2))
        if l2 > 0.0 and theta_vec is not None:
            val += float(l2) * float(np.mean(theta_vec * theta_vec))
        return float(val)

    def _objective_details_from_proba(
        *, proba: np.ndarray, feats: np.ndarray | None, Q: np.ndarray, theta_vec: np.ndarray | None
    ) -> tuple[float, dict]:
        proba = np.asarray(proba, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        details: dict = {}

        keep = np.arange(proba.shape[0], dtype=int)
        if objective_core in {"entropy", "infomax", "confidence", "lda_nll"}:
            keep = _maybe_select_keep(proba)
            if keep.size == 0:
                return 1e6, {"n_keep": 0}
            proba = proba[keep]
            if feats is not None:
                feats = np.asarray(feats, dtype=np.float64)[keep]
        details["n_keep"] = int(keep.size)

        # Normalize to a valid distribution for entropy computations.
        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)
        details["n_samples"] = int(p.shape[0])

        if objective_core == "lda_nll":
            if feats is None:
                return 1e6, {"n_keep": int(keep.size)}
            if lda_evidence is None:
                raise ValueError("lda_evidence must be provided when objective='lda_nll'.")
            mu_e = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
            priors_e = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
            cov_inv_e = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
            logdet_e = float(lda_evidence.get("logdet", 0.0))

            f = np.asarray(feats, dtype=np.float64)
            diff = f[:, None, :] - mu_e[None, :, :]
            qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv_e, diff, optimize=True)
            log_norm = float(mu_e.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet_e)
            log_gauss = -0.5 * (log_norm + qf)
            log_pr = np.log(np.clip(priors_e, 1e-12, 1.0)).reshape(1, -1)
            log_joint = log_pr + log_gauss
            m = np.max(log_joint, axis=1, keepdims=True)
            log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

            ent_p = _row_entropy(p)
            conf = np.max(p, axis=1)
            w = np.ones(p.shape[0], dtype=np.float64)
            if reliable_metric != "none":
                if reliable_metric == "confidence":
                    w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
                else:
                    w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent_p))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                return 1e6, {"n_keep": int(keep.size)}
            base = float(-np.sum(w * log_p) / w_sum)
            details["mean_entropy"] = float(np.sum(w * ent_p) / w_sum)
            details["mean_confidence"] = float(np.sum(w * conf) / w_sum)
            details["objective_base"] = float(base)
            val = float(base)

            p_bar = np.mean(p, axis=0)
            p_bar = np.clip(p_bar, 1e-12, 1.0)
            p_bar = p_bar / float(np.sum(p_bar))
            ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
            details["entropy_bar"] = float(ent_bar)

            if marginal_mode != "none":
                tau = float(marginal_tau)
                pen = 0.0
                if marginal_mode == "hard_min":
                    if float(np.min(p_bar)) < tau:
                        return 1e6, details
                elif marginal_mode == "kl_prior":
                    if marginal_prior_vec is None:
                        return 1e6, details
                    pen = float(-np.sum(marginal_prior_vec * np.log(p_bar)))
                elif float(marginal_beta) > 0.0:
                    if marginal_mode == "l2_uniform":
                        u = 1.0 / float(n_classes)
                        pen = float(np.mean((p_bar - u) ** 2))
                    elif marginal_mode == "kl_uniform":
                        pen = float(-np.mean(np.log(p_bar)))
                    else:
                        pen = float(np.mean(np.maximum(0.0, tau - p_bar) ** 2))
                if float(marginal_beta) > 0.0 and marginal_mode != "hard_min":
                    val = float(val) + float(marginal_beta) * float(pen)
                details["pen_marginal"] = float(pen)
        elif objective_core == "pseudo_ce":
            # pseudo_ce: hard pseudo labels + optional filtering
            pred_idx = np.argmax(p, axis=1)
            classes_arr = np.asarray(class_order, dtype=object)
            y_pseudo = classes_arr[pred_idx]
            keep = _select_pseudo_indices(
                y_pseudo=y_pseudo,
                proba=p,
                class_order=class_order,
                confidence=float(pseudo_confidence),
                topk_per_class=int(pseudo_topk_per_class),
                balance=bool(pseudo_balance),
            )
            if keep.size == 0:
                return 1e6, {"n_keep": 0}
            pred_idx_k = np.argmax(p[keep], axis=1)
            conf_k = p[keep, pred_idx_k]
            conf_k = np.clip(conf_k, 1e-12, 1.0)
            nll = -np.log(conf_k)
            # Weight by confidence (encourages self-consistent high-confidence predictions).
            base = float(np.mean(conf_k * nll))
            details["objective_base"] = base
            val = float(base)
        else:
            ent_p = _row_entropy(p)
            conf_p = np.max(p, axis=1)
            details["mean_entropy"] = float(np.mean(ent_p))
            details["mean_confidence"] = float(np.mean(conf_p))

            if is_bilevel:
                if objective_core not in {"entropy", "infomax"}:
                    return 1e6, details
                prior = (
                    np.ones(int(n_classes), dtype=np.float64) / float(n_classes)
                    if marginal_prior_vec is None
                    else np.asarray(marginal_prior_vec, dtype=np.float64)
                )
                w, q, q_bar, stats = _solve_inner_wq(p=p, prior=prior)
                w_sum = float(stats["w_sum"])
                if w_sum <= 1e-12:
                    return 1e6, details
                details["coverage"] = float(stats["coverage"])
                details["eff_n"] = float(stats["eff_n"])
                details["q_bar"] = np.asarray(q_bar, dtype=np.float64).copy()

                ent_q = _row_entropy(q)
                details["mean_entropy_q"] = float(np.mean(ent_q))
                base = float(np.sum(w * ent_q) / w_sum)
                if objective_core == "infomax":
                    ent_bar = -float(np.sum(q_bar * np.log(q_bar)))
                    details["entropy_bar"] = float(ent_bar)
                    base = float(base) - float(infomax_lambda) * float(ent_bar)
                details["objective_base"] = float(base)
                val = float(base)

                # For reference: unweighted marginal of p.
                p_bar = np.mean(p, axis=0)
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))
                details["p_bar"] = p_bar.copy()

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    pen = 0.0
                    if marginal_mode == "hard_min":
                        if float(np.min(q_bar)) < tau:
                            return 1e6, details
                    elif marginal_mode == "kl_prior":
                        if marginal_prior_vec is None:
                            return 1e6, details
                        pen = float(-np.sum(marginal_prior_vec * np.log(q_bar)))
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((q_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(q_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - q_bar) ** 2))
                    if float(marginal_beta) > 0.0 and marginal_mode != "hard_min":
                        val = float(val) + float(marginal_beta) * float(pen)
                    details["pen_marginal"] = float(pen)
            else:
                w = np.ones(p.shape[0], dtype=np.float64)
                if reliable_metric != "none":
                    if reliable_metric == "confidence":
                        w = w * _sigmoid(float(reliable_alpha) * (conf_p - float(reliable_threshold)))
                    else:
                        w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent_p))

                w_sum = float(np.sum(w))
                if w_sum <= 1e-12:
                    return 1e6, {"n_keep": int(p.shape[0])}

                p_bar = np.mean(p, axis=0)
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))
                details["p_bar"] = p_bar.copy()

                if objective_core == "entropy":
                    base = float(np.sum(w * ent_p) / w_sum)
                    details["objective_base"] = base
                    val = float(base)
                elif objective_core == "confidence":
                    base = float(np.sum(w * (1.0 - conf_p)) / w_sum)
                    details["objective_base"] = base
                    val = float(base)
                else:
                    ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
                    base = float(np.sum(w * ent_p) / w_sum) - float(infomax_lambda) * ent_bar
                    details["entropy_bar"] = float(ent_bar)
                    details["objective_base"] = base
                    val = float(base)

                if marginal_mode != "none":
                    tau = float(marginal_tau)
                    pen = 0.0
                    if marginal_mode == "hard_min":
                        if float(np.min(p_bar)) < tau:
                            return 1e6, details
                    elif marginal_mode == "kl_prior":
                        if marginal_prior_vec is None:
                            return 1e6, details
                        pen = float(-np.sum(marginal_prior_vec * np.log(p_bar)))
                    elif float(marginal_beta) > 0.0:
                        if marginal_mode == "l2_uniform":
                            u = 1.0 / float(n_classes)
                            pen = float(np.mean((p_bar - u) ** 2))
                        elif marginal_mode == "kl_uniform":
                            pen = float(-np.mean(np.log(p_bar)))
                        else:
                            pen = float(np.mean(np.maximum(0.0, tau - p_bar) ** 2))
                    if float(marginal_beta) > 0.0 and marginal_mode != "hard_min":
                        val = float(val) + float(marginal_beta) * float(pen)
                    details["pen_marginal"] = float(pen)

        pen_trust = 0.0
        if float(trust_lambda) > 0.0:
            pen_trust = float(trust_lambda) * float(np.mean((Q - q0) ** 2))
            val += pen_trust
        details["pen_trust"] = float(pen_trust)
        pen_l2 = 0.0
        if l2 > 0.0 and theta_vec is not None:
            pen_l2 = float(l2) * float(np.mean(theta_vec * theta_vec))
            val += pen_l2
        details["pen_l2"] = float(pen_l2)
        details["objective"] = float(val)
        return float(val), details

    def eval_theta(theta_vec: np.ndarray, z_data: np.ndarray) -> float:
        proba, Q, feats = _proba_from_theta(theta_vec, z_data)
        return _objective_from_proba(proba=proba, feats=feats, Q=Q, theta_vec=theta_vec)

    def _evidence_nll_from_outputs(*, proba: np.ndarray, feats: np.ndarray) -> float:
        """Compute -log p(z) under the frozen CSP+LDA Gaussian mixture (if available)."""

        if lda_evidence is None:
            return float("nan")

        proba = np.asarray(proba, dtype=np.float64)
        feats = np.asarray(feats, dtype=np.float64)

        keep = _maybe_select_keep(proba)
        if keep.size == 0:
            return float("nan")
        proba = proba[keep]
        feats = feats[keep]

        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)

        mu_e = np.asarray(lda_evidence.get("mu"), dtype=np.float64)
        priors_e = np.asarray(lda_evidence.get("priors"), dtype=np.float64).reshape(-1)
        cov_inv_e = np.asarray(lda_evidence.get("cov_inv"), dtype=np.float64)
        logdet_e = float(lda_evidence.get("logdet", 0.0))
        if mu_e.ndim != 2:
            return float("nan")
        if feats.shape[1] != int(mu_e.shape[1]):
            return float("nan")

        diff = feats[:, None, :] - mu_e[None, :, :]
        qf = np.einsum("nkd,dd,nkd->nk", diff, cov_inv_e, diff, optimize=True)
        log_norm = float(mu_e.shape[1]) * float(np.log(2.0 * np.pi)) + float(logdet_e)
        log_gauss = -0.5 * (log_norm + qf)
        log_pr = np.log(np.clip(priors_e, 1e-12, 1.0)).reshape(1, -1)
        log_joint = log_pr + log_gauss
        m = np.max(log_joint, axis=1, keepdims=True)
        log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))

        w = np.ones(p.shape[0], dtype=np.float64)
        if reliable_metric != "none":
            ent = _row_entropy(p)
            conf = np.max(p, axis=1)
            if reliable_metric == "confidence":
                w = w * _sigmoid(float(reliable_alpha) * (conf - float(reliable_threshold)))
            else:
                w = w * _sigmoid(float(reliable_alpha) * (float(reliable_threshold) - ent))
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return float("nan")
        return float(-np.sum(w * log_p) / w_sum)

    def _record_candidate(*, kind: str, iter_idx: int, theta_vec: np.ndarray | None, Q: np.ndarray) -> None:
        if not do_diag:
            return
        proba_best, feats_best = _proba_from_Q(Q, z_best)
        obj, details = _objective_details_from_proba(proba=proba_best, feats=feats_best, Q=Q, theta_vec=theta_vec)
        drift_best_vec = _kl_drift_vec(proba_anchor_best, proba_best)
        drift_best = float(np.mean(drift_best_vec))
        proba_full, feats_full = _proba_from_Q(Q, z_t)
        drift_full_vec = _kl_drift_vec(proba_anchor_full, proba_full)
        drift_full = float(np.mean(drift_full_vec))
        p_bar_full = np.mean(np.clip(proba_full, 1e-12, 1.0), axis=0)
        p_bar_full = p_bar_full / float(np.sum(p_bar_full))
        evidence_nll_best = _evidence_nll_from_outputs(proba=proba_best, feats=feats_best)
        evidence_nll_full = _evidence_nll_from_outputs(proba=proba_full, feats=feats_full)
        rec = {
            "kind": str(kind),
            "iter": int(iter_idx),
            "order": int(len(diag_records)),
            "objective": float(obj),
            "score": float(_score_with_drift(float(obj), float(drift_best))),
            "objective_base": float(details.get("objective_base", np.nan)),
            "pen_marginal": float(details.get("pen_marginal", 0.0)),
            "pen_trust": float(details.get("pen_trust", 0.0)),
            "pen_l2": float(details.get("pen_l2", 0.0)),
            "mean_entropy": float(details.get("mean_entropy", np.nan)),
            "mean_entropy_q": float(details.get("mean_entropy_q", np.nan)),
            "mean_confidence": float(details.get("mean_confidence", np.nan)),
            "entropy_bar": float(details.get("entropy_bar", np.nan)),
            "n_keep": int(details.get("n_keep", -1)),
            "n_best_total": int(z_best.shape[0]),
            "n_full_total": int(z_t.shape[0]),
            "evidence_nll_best": float(evidence_nll_best),
            "evidence_nll_full": float(evidence_nll_full),
            "drift_best": float(drift_best),
            "drift_best_std": float(np.std(drift_best_vec)),
            "drift_best_q50": float(np.quantile(drift_best_vec, 0.50)),
            "drift_best_q90": float(np.quantile(drift_best_vec, 0.90)),
            "drift_best_q95": float(np.quantile(drift_best_vec, 0.95)),
            "drift_best_max": float(np.max(drift_best_vec)),
            "drift_best_tail_frac": float(np.mean(drift_best_vec > float(drift_delta))) if float(drift_delta) > 0.0 else 0.0,
            "drift_full": float(drift_full),
            "drift_full_std": float(np.std(drift_full_vec)),
            "drift_full_q50": float(np.quantile(drift_full_vec, 0.50)),
            "drift_full_q90": float(np.quantile(drift_full_vec, 0.90)),
            "drift_full_q95": float(np.quantile(drift_full_vec, 0.95)),
            "drift_full_max": float(np.max(drift_full_vec)),
            "drift_full_tail_frac": float(np.mean(drift_full_vec > float(drift_delta))) if float(drift_delta) > 0.0 else 0.0,
            "coverage": float(details.get("coverage", np.nan)),
            "eff_n": float(details.get("eff_n", np.nan)),
            "p_bar_full": p_bar_full.astype(np.float64),
            "q_bar": details.get("q_bar", None),
            "transform": str(transform),
            "Q": np.asarray(Q, dtype=np.float64),
        }
        if transform == "rot_scale" and theta_vec is not None and int(theta_vec.shape[0]) > rot_dim:
            log_s = np.asarray(theta_vec[rot_dim:], dtype=np.float64)
            rec["log_s_mean_abs"] = float(np.mean(np.abs(log_s)))
            rec["log_s_max_abs"] = float(np.max(np.abs(log_s)))
        diag_records.append(rec)

    # Optional warm start: build Q_Δ from pseudo-label Δ-alignment and approximate it with our Givens pairs.
    q_delta: np.ndarray | None = None
    if (warm_start == "delta" or trust_q0 == "delta") and int(warm_iters) > 0:
        q_cur = np.eye(int(n_channels), dtype=np.float64)
        for _ in range(int(warm_iters)):
            X_cur = apply_spatial_transform(q_cur, z_t)
            proba = model.predict_proba(X_cur)
            proba = _reorder_proba_columns(proba, model.classes_, list(class_order))
            try:
                if pseudo_mode == "soft":
                    d_t = _soft_class_cov_diff(
                        z_t,
                        proba=proba,
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
                        q_delta = None
                        break
                    d_t = class_cov_diff(
                        z_t[keep],
                        y_pseudo[keep],
                        class_order=class_order,
                        eps=float(eps),
                        shrinkage=float(shrinkage),
                    )
                q_cur = orthogonal_align_symmetric(d_t, d_ref)
                q_delta = q_cur
            except ValueError:
                q_delta = None
                break

        if trust_q0 == "delta" and q_delta is not None:
            q0 = blend_with_identity(q_delta, float(q_blend))

        if q_delta is not None:
            # Greedy 2D Procrustes per plane to get a good rotation initialization (angles on our Givens pairs).
            q_work = np.eye(int(n_channels), dtype=np.float64)
            phi_init = np.zeros(rot_dim, dtype=np.float64)
            for k, (i, j) in enumerate(pairs):
                a = q_work[:, i]
                b = q_work[:, j]
                ti = q_delta[:, i]
                tj = q_delta[:, j]
                m11 = float(np.dot(a, ti))
                m12 = float(np.dot(a, tj))
                m21 = float(np.dot(b, ti))
                m22 = float(np.dot(b, tj))
                angle = float(np.arctan2(m21 - m12, m11 + m22))
                phi_init[k] = angle
                # Apply the rotation to q_work columns i,j (right multiplication).
                c = float(np.cos(angle))
                s = float(np.sin(angle))
                col_i = q_work[:, i].copy()
                col_j = q_work[:, j].copy()
                q_work[:, i] = c * col_i + s * col_j
                q_work[:, j] = -s * col_i + c * col_j
            if warm_start == "delta":
                theta[:rot_dim] = phi_init.copy()

    # Candidate set on holdout (Step A): always include identity (EA) and, if available, Q_delta.
    best_Q_override: np.ndarray | None = None
    theta_id = np.zeros_like(best_theta)
    proba_id = proba_anchor_best
    obj_id = _objective_from_proba(
        proba=proba_id,
        feats=feats_anchor_best,
        Q=np.eye(int(n_channels), dtype=np.float64),
        theta_vec=theta_id,
    )
    score_id = _score_with_drift(float(obj_id), 0.0)
    _record_candidate(kind="identity", iter_idx=-1, theta_vec=theta_id, Q=np.eye(int(n_channels), dtype=np.float64))
    best_theta = theta_id.copy()
    best_obj = float(obj_id)
    best_score = float(score_id)

    if q_delta is not None:
        q_delta_b = blend_with_identity(q_delta, float(q_blend))
        proba_qd, feats_qd = _proba_from_Q(q_delta_b, z_best)
        obj_qd = _objective_from_proba(proba=proba_qd, feats=feats_qd, Q=q_delta_b, theta_vec=None)
        drift_qd = _kl_drift(proba_anchor_best, proba_qd)
        score_qd = _score_with_drift(float(obj_qd), float(drift_qd))
        _record_candidate(kind="q_delta", iter_idx=-2, theta_vec=None, Q=q_delta_b)
        if score_qd < best_score:
            best_obj = float(obj_qd)
            best_score = float(score_qd)
            best_Q_override = q_delta_b

    # If we warm-started (rotation angles), compare that initial point too.
    if np.any(theta != 0.0):
        proba_init, Q_init, feats_init = _proba_from_theta(theta, z_best)
        obj_init = _objective_from_proba(proba=proba_init, feats=feats_init, Q=Q_init, theta_vec=theta)
        drift_init = _kl_drift(proba_anchor_best, proba_init)
        score_init = _score_with_drift(float(obj_init), float(drift_init))
        if do_diag:
            q_init = _build_transform(theta)
            _record_candidate(kind="warm_init", iter_idx=0, theta_vec=theta.copy(), Q=q_init)
        if score_init < best_score:
            best_obj = float(obj_init)
            best_score = float(score_init)
            best_theta = theta.copy()
            best_Q_override = None

    # SPSA / two-point random-direction estimator
    for t in range(int(iters)):
        u = rng.choice([-1.0, 1.0], size=theta.shape[0]).astype(np.float64)
        theta_plus = theta + float(mu) * u
        theta_minus = theta - float(mu) * u
        f_plus = eval_theta(theta_plus, z_opt)
        f_minus = eval_theta(theta_minus, z_opt)
        g = (f_plus - f_minus) / (2.0 * float(mu)) * u
        step = float(lr) / np.sqrt(float(t) + 1.0)
        theta = theta - step * g

        # Track best iterate (not the +/- perturbations used only for gradient estimation).
        proba_tmp, Q_tmp, feats_tmp = _proba_from_theta(theta, z_best)
        f_theta = _objective_from_proba(proba=proba_tmp, feats=feats_tmp, Q=Q_tmp, theta_vec=theta)
        drift_tmp = _kl_drift(proba_anchor_best, proba_tmp)
        score_tmp = _score_with_drift(float(f_theta), float(drift_tmp))
        if do_diag:
            _record_candidate(kind="iter", iter_idx=int(t + 1), theta_vec=theta.copy(), Q=Q_tmp)
        if score_tmp < best_score:
            best_obj = float(f_theta)
            best_score = float(score_tmp)
            best_theta = theta.copy()
            best_Q_override = None

    # Optional safety: require a minimum improvement over identity (otherwise keep identity).
    if float(min_improvement) > 0.0 and (float(score_id) - float(best_score)) < float(min_improvement):
        best_theta = theta_id.copy()
        best_obj = float(obj_id)
        best_score = float(score_id)
        best_Q_override = None

    if best_Q_override is not None:
        Q = best_Q_override
    else:
        Q = _build_transform(best_theta)

    # Safety fallback: if target predictions collapse to a single class, fall back to a safer Q.
    if float(fallback_min_marginal_entropy) > 0.0:
        proba_best, _feats_best = _proba_from_Q(Q, z_t)
        p_bar = np.mean(np.clip(proba_best, 1e-12, 1.0), axis=0)
        p_bar = p_bar / float(np.sum(p_bar))
        ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
        if ent_bar < float(fallback_min_marginal_entropy):
            # Compare against identity (EA) and optional Q_delta.
            candidates: List[np.ndarray] = [np.eye(int(n_channels), dtype=np.float64)]
            if q_delta is not None:
                candidates.append(blend_with_identity(q_delta, float(q_blend)))

            best_ent = -1.0
            best_q = candidates[0]
            for q_cand in candidates:
                FQ = F @ q_cand
                Y = np.einsum("kc,nct->nkt", FQ, z_t, optimize=True)
                power = np.mean(Y * Y, axis=2)
                power = np.maximum(power, 1e-20)
                feats = np.log(power) if use_log else power
                proba_cand = lda.predict_proba(feats)
                proba_cand = _reorder_proba_columns(proba_cand, lda.classes_, list(class_order))
                p_bar_c = np.mean(np.clip(proba_cand, 1e-12, 1.0), axis=0)
                p_bar_c = p_bar_c / float(np.sum(p_bar_c))
                ent_c = -float(np.sum(p_bar_c * np.log(p_bar_c)))
                if ent_c > best_ent:
                    best_ent = ent_c
                    best_q = q_cand
            Q = best_q

    if do_diag:
        diag = {
            "records": diag_records,
            "class_order": list(class_order),
            "transform": str(transform),
            "marginal_mode": str(marginal_mode),
            "marginal_prior": None if marginal_prior_vec is None else marginal_prior_vec.astype(np.float64),
            "drift_mode": str(drift_mode),
            "drift_gamma": float(drift_gamma),
            "drift_delta": float(drift_delta),
        }
        return Q, diag
    return Q


def _sample_givens_pairs(
    *, n_channels: int, n_rotations: int, rng: np.random.RandomState
) -> List[tuple[int, int]]:
    if n_rotations < 1:
        raise ValueError("n_rotations must be >= 1.")
    all_pairs: List[tuple[int, int]] = []
    for i in range(int(n_channels)):
        for j in range(i + 1, int(n_channels)):
            all_pairs.append((i, j))
    rng.shuffle(all_pairs)
    n_rotations = min(int(n_rotations), len(all_pairs))
    return all_pairs[:n_rotations]


def _apply_givens_right(mat: np.ndarray, *, pairs: List[tuple[int, int]], angles: np.ndarray) -> np.ndarray:
    """Return mat @ Q(angles) where Q is a product of Givens rotations (right-multiplication)."""

    out = np.asarray(mat, dtype=np.float64).copy()
    angles = np.asarray(angles, dtype=np.float64)
    if len(pairs) != angles.shape[0]:
        raise ValueError("pairs and angles length mismatch.")

    for (i, j), theta in zip(pairs, angles):
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        col_i = out[:, i].copy()
        col_j = out[:, j].copy()
        out[:, i] = c * col_i + s * col_j
        out[:, j] = -s * col_i + c * col_j
    return out


def _build_q_from_givens(
    *, n_channels: int, pairs: List[tuple[int, int]], angles: np.ndarray
) -> np.ndarray:
    Q = np.eye(int(n_channels), dtype=np.float64)
    Q = _apply_givens_right(Q, pairs=pairs, angles=angles)
    return Q


def summarize_results(results_df: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    """Return mean/std/min/max summary for selected metric columns."""

    summary = {
        "mean": results_df[metric_columns].mean(numeric_only=True),
        "std": results_df[metric_columns].std(numeric_only=True),
        "min": results_df[metric_columns].min(numeric_only=True),
        "max": results_df[metric_columns].max(numeric_only=True),
    }
    return pd.DataFrame(summary)
