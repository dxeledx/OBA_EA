from __future__ import annotations

from dataclasses import asdict, dataclass
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
    oea_zo_infomax_lambda: float = 1.0,
    oea_zo_reliable_metric: str = "none",
    oea_zo_reliable_threshold: float = 0.0,
    oea_zo_reliable_alpha: float = 10.0,
    oea_zo_trust_lambda: float = 0.0,
    oea_zo_trust_q0: str = "identity",
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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, TrainedModel]]:
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

    if alignment not in {"none", "ea", "ea_zo", "oea_cov", "oea", "oea_zo"}:
        raise ValueError("alignment must be one of: 'none', 'ea', 'ea_zo', 'oea_cov', 'oea', 'oea_zo'")

    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")
    if not (0.0 <= float(oea_pseudo_confidence) <= 1.0):
        raise ValueError("oea_pseudo_confidence must be in [0,1].")
    if int(oea_pseudo_topk_per_class) < 0:
        raise ValueError("oea_pseudo_topk_per_class must be >= 0.")

    if oea_zo_objective not in {"entropy", "pseudo_ce", "confidence", "infomax"}:
        raise ValueError("oea_zo_objective must be one of: 'entropy', 'pseudo_ce', 'confidence', 'infomax'")
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

    # Fast path: subject-wise EA can be precomputed once.
    if alignment in {"ea", "ea_zo"}:
        aligned: Dict[int, SubjectData] = {}
        for s, sd in subject_data.items():
            X_aligned = EuclideanAligner(eps=oea_eps, shrinkage=oea_shrinkage).fit_transform(sd.X)
            aligned[int(s)] = SubjectData(subject=int(s), X=X_aligned, y=sd.y)
        subject_data = aligned

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]

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
            q_t = _optimize_qt_oea_zo(
                z_t=z_t,
                model=model,
                class_order=class_labels,
                d_ref=d_ref,
                eps=float(oea_eps),
                shrinkage=float(oea_shrinkage),
                pseudo_mode=str(oea_pseudo_mode),
                warm_start=str(oea_zo_warm_start),
                warm_iters=int(oea_zo_warm_iters),
                q_blend=float(oea_q_blend),
                objective=str(oea_zo_objective),
                infomax_lambda=float(oea_zo_infomax_lambda),
                reliable_metric=str(oea_zo_reliable_metric),
                reliable_threshold=float(oea_zo_reliable_threshold),
                reliable_alpha=float(oea_zo_reliable_alpha),
                trust_lambda=float(oea_zo_trust_lambda),
                trust_q0=str(oea_zo_trust_q0),
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
            )
            X_test = apply_spatial_transform(q_t, z_t)
            y_test = subject_data[int(test_subject)].y
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
                q_t = _optimize_qt_oea_zo(
                    z_t=z_t,
                    model=model,
                    class_order=class_labels,
                    d_ref=d_ref,
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    pseudo_mode=str(oea_pseudo_mode),
                    warm_start=str(oea_zo_warm_start),
                    warm_iters=int(oea_zo_warm_iters),
                    q_blend=float(oea_q_blend),
                    objective=str(oea_zo_objective),
                    infomax_lambda=float(oea_zo_infomax_lambda),
                    reliable_metric=str(oea_zo_reliable_metric),
                    reliable_threshold=float(oea_zo_reliable_threshold),
                    reliable_alpha=float(oea_zo_reliable_alpha),
                    trust_lambda=float(oea_zo_trust_lambda),
                    trust_q0=str(oea_zo_trust_q0),
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
                )

            X_test = apply_spatial_transform(q_t, z_t)
            y_test = subject_data[test_subject].y

        if alignment not in {"oea", "oea_zo", "ea_zo"}:
            model = fit_csp_lda(X_train, y_train, n_components=n_components)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_proba = _reorder_proba_columns(y_proba, model.classes_, class_order)

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

    results_df = pd.DataFrame([asdict(r) for r in fold_rows]).sort_values("subject")
    return (
        results_df,
        np.concatenate(y_true_all, axis=0),
        np.concatenate(y_pred_all, axis=0),
        np.concatenate(y_proba_all, axis=0),
        list(class_order),
        models_by_subject,
    )


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
    cov_mean = np.einsum("k,kcd->cd", pi, covs, optimize=True)
    cov_mean = 0.5 * (cov_mean + cov_mean.T)
    delta = covs - cov_mean[None, :, :]
    sig = np.einsum("k,kce,ked->cd", pi, delta, delta, optimize=True)
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
    eps: float,
    shrinkage: float,
    pseudo_mode: str,
    warm_start: str,
    warm_iters: int,
    q_blend: float,
    objective: str,
    infomax_lambda: float,
    reliable_metric: str,
    reliable_threshold: float,
    reliable_alpha: float,
    trust_lambda: float,
    trust_q0: str,
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
) -> np.ndarray:
    """Zero-order optimize Q_t on the orthogonal group via a low-dim Givens parameterization.

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
    phi = np.zeros(len(pairs), dtype=np.float64)
    best_phi = phi.copy()
    best_obj = float("inf")

    csp = model.csp
    lda = model.pipeline.named_steps["lda"]
    F = np.asarray(csp.filters_[: int(csp.n_components)], dtype=np.float64)

    # Determine whether CSP uses log(power).
    use_log = True if (getattr(csp, "log", None) is None) else bool(getattr(csp, "log"))

    def _proba_from_q(phi_vec: np.ndarray, z_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = _build_q_from_givens(n_channels=n_channels, pairs=pairs, angles=phi_vec)
        if float(q_blend) < 1.0:
            Q = blend_with_identity(Q, float(q_blend))
        FQ = F @ Q
        Y = np.einsum("kc,nct->nkt", FQ, z_data, optimize=True)
        power = np.mean(Y * Y, axis=2)
        power = np.maximum(power, 1e-20)
        feats = np.log(power) if use_log else power
        proba = lda.predict_proba(feats)
        return _reorder_proba_columns(proba, lda.classes_, list(class_order)), Q

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

    phi_anchor = np.zeros_like(phi)
    q0 = np.eye(int(n_channels), dtype=np.float64)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def eval_phi(phi_vec: np.ndarray, z_data: np.ndarray) -> float:
        proba, Q = _proba_from_q(phi_vec, z_data)

        if objective in {"entropy", "infomax", "confidence"}:
            keep = _maybe_select_keep(proba)
            if keep.size == 0:
                return 1e6
            proba = proba[keep]

        # Normalize to a valid distribution for entropy computations.
        p = np.clip(proba, 1e-12, 1.0)
        p = p / np.sum(p, axis=1, keepdims=True)

        if objective == "pseudo_ce":
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
                return 1e6
            pred_idx_k = np.argmax(p[keep], axis=1)
            conf_k = p[keep, pred_idx_k]
            conf_k = np.clip(conf_k, 1e-12, 1.0)
            nll = -np.log(conf_k)
            # Weight by confidence (encourages self-consistent high-confidence predictions).
            val = float(np.mean(conf_k * nll))
        else:
            ent = -np.sum(p * np.log(p), axis=1)
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

            if objective == "entropy":
                val = float(np.sum(w * ent) / w_sum)
            elif objective == "confidence":
                val = float(np.sum(w * (1.0 - conf)) / w_sum)
            else:
                # infomax: maximize mutual information I(Y;X) = H(mean p) - mean H(p).
                # We minimize: mean H(p) - λ * H(mean p).
                p_bar = np.sum(p * w[:, None], axis=0) / w_sum
                p_bar = np.clip(p_bar, 1e-12, 1.0)
                p_bar = p_bar / float(np.sum(p_bar))
                ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
                val = float(np.sum(w * ent) / w_sum) - float(infomax_lambda) * ent_bar

        if float(trust_lambda) > 0.0:
            val += float(trust_lambda) * float(np.mean((Q - q0) ** 2))
        if l2 > 0.0:
            val += float(l2) * float(np.mean((phi_vec - phi_anchor) ** 2))
        return val

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
            # Greedy 2D Procrustes per plane to get a good phi initialization.
            q_work = np.eye(int(n_channels), dtype=np.float64)
            phi_init = np.zeros(len(pairs), dtype=np.float64)
            for k, (i, j) in enumerate(pairs):
                a = q_work[:, i]
                b = q_work[:, j]
                ti = q_delta[:, i]
                tj = q_delta[:, j]
                m11 = float(np.dot(a, ti))
                m12 = float(np.dot(a, tj))
                m21 = float(np.dot(b, ti))
                m22 = float(np.dot(b, tj))
                theta = float(np.arctan2(m21 - m12, m11 + m22))
                phi_init[k] = theta
                # Apply the rotation to q_work columns i,j (right multiplication).
                c = float(np.cos(theta))
                s = float(np.sin(theta))
                col_i = q_work[:, i].copy()
                col_j = q_work[:, j].copy()
                q_work[:, i] = c * col_i + s * col_j
                q_work[:, j] = -s * col_i + c * col_j
            if warm_start == "delta":
                phi = phi_init.copy()
                phi_anchor = phi_init.copy()

    # Baseline: identity alignment is always a valid candidate (best selection uses z_best).
    best_phi = np.zeros_like(best_phi)
    best_obj = eval_phi(best_phi, z_best)
    # If we warm-started, compare that initial point too.
    if np.any(phi != 0.0):
        obj_init = eval_phi(phi, z_best)
        if obj_init < best_obj:
            best_obj = obj_init
            best_phi = phi.copy()

    # SPSA / two-point random-direction estimator
    for t in range(int(iters)):
        u = rng.choice([-1.0, 1.0], size=phi.shape[0]).astype(np.float64)
        phi_plus = phi + float(mu) * u
        phi_minus = phi - float(mu) * u
        f_plus = eval_phi(phi_plus, z_opt)
        f_minus = eval_phi(phi_minus, z_opt)
        g = (f_plus - f_minus) / (2.0 * float(mu)) * u
        step = float(lr) / np.sqrt(float(t) + 1.0)
        phi = phi - step * g

        # Track best iterate (not the +/- perturbations used only for gradient estimation).
        f_phi = eval_phi(phi, z_best)
        if f_phi < best_obj:
            best_obj = f_phi
            best_phi = phi.copy()

    Q = _build_q_from_givens(n_channels=n_channels, pairs=pairs, angles=best_phi)
    Q = blend_with_identity(Q, float(q_blend))

    # Safety fallback: if target predictions collapse to a single class, fall back to a safer Q.
    if float(fallback_min_marginal_entropy) > 0.0:
        proba_best, _Q_best = _proba_from_q(best_phi, z_t)
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
