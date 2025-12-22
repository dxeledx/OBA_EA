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

    if alignment not in {"none", "ea", "oea_cov", "oea"}:
        raise ValueError("alignment must be one of: 'none', 'ea', 'oea_cov', 'oea'")

    if oea_pseudo_mode not in {"hard", "soft"}:
        raise ValueError("oea_pseudo_mode must be one of: 'hard', 'soft'")
    if not (0.0 <= float(oea_pseudo_confidence) <= 1.0):
        raise ValueError("oea_pseudo_confidence must be in [0,1].")
    if int(oea_pseudo_topk_per_class) < 0:
        raise ValueError("oea_pseudo_topk_per_class must be >= 0.")

    # Fast path: subject-wise EA can be precomputed once.
    if alignment == "ea":
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
            # alignment == "oea": optimistic selection based on discriminative covariance difference Δ.
            if len(class_order) != 2:
                raise ValueError("oea currently supports 2-class problems only.")
            class_pair = (str(class_order[0]), str(class_order[1]))

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
                        class_order=class_pair,
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
                    class_order=class_pair,
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

            # 5) Target subject: select Q_t using *unlabeled* target data via pseudo-labeling,
            #    without updating model parameters.
            z_t = z_by_subject[int(test_subject)]
            q_t = np.eye(z_t.shape[1], dtype=np.float64)
            for _ in range(int(max(0, oea_pseudo_iters))):
                X_t_cur = apply_spatial_transform(q_t, z_t)
                proba = model.predict_proba(X_t_cur)
                proba = _reorder_proba_columns(proba, model.classes_, list(class_pair))

                if oea_pseudo_mode == "soft":
                    d_t = _soft_class_cov_diff(
                        z_t,
                        proba=proba,
                        class_order=class_pair,
                        eps=oea_eps,
                        shrinkage=oea_shrinkage,
                    )
                else:
                    y_pseudo = np.asarray(model.predict(X_t_cur))
                    keep = _select_pseudo_indices(
                        y_pseudo=y_pseudo,
                        proba=proba,
                        class_order=class_pair,
                        confidence=float(oea_pseudo_confidence),
                        topk_per_class=int(oea_pseudo_topk_per_class),
                        balance=bool(oea_pseudo_balance),
                    )
                    if keep.size == 0:
                        break
                    d_t = class_cov_diff(
                        z_t[keep],
                        y_pseudo[keep],
                        class_order=class_pair,
                        eps=oea_eps,
                        shrinkage=oea_shrinkage,
                    )
                q_t = orthogonal_align_symmetric(d_t, d_ref)
                q_t = blend_with_identity(q_t, oea_q_blend)

            X_test = apply_spatial_transform(q_t, z_t)
            y_test = subject_data[test_subject].y

        if alignment != "oea":
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
    class_order: tuple[str, str],
    confidence: float,
    topk_per_class: int,
    balance: bool,
) -> np.ndarray:
    """Select trial indices to use for pseudo-label covariance estimation.

    This is a simple stabilization layer for OEA(TTA-Q) to reduce the impact of noisy pseudo labels.
    """

    y_pseudo = np.asarray(y_pseudo)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"Expected proba shape (n_samples,2); got {proba.shape}.")

    pred_idx = np.where(y_pseudo == class_order[0], 0, 1)
    conf = proba[np.arange(len(y_pseudo)), pred_idx]
    keep = conf >= float(confidence)
    if not np.any(keep):
        return np.array([], dtype=int)

    idx0 = np.where(keep & (y_pseudo == class_order[0]))[0]
    idx1 = np.where(keep & (y_pseudo == class_order[1]))[0]
    if idx0.size == 0 or idx1.size == 0:
        return np.array([], dtype=int)

    if int(topk_per_class) > 0:
        k = int(topk_per_class)
        idx0 = idx0[np.argsort(proba[idx0, 0])[::-1][:k]]
        idx1 = idx1[np.argsort(proba[idx1, 1])[::-1][:k]]
        if idx0.size == 0 or idx1.size == 0:
            return np.array([], dtype=int)

    if balance:
        k = int(min(idx0.size, idx1.size))
        idx0 = idx0[:k]
        idx1 = idx1[:k]

    out = np.concatenate([idx0, idx1], axis=0)
    return np.asarray(out, dtype=int)


def _soft_class_cov_diff(
    X: np.ndarray,
    *,
    proba: np.ndarray,
    class_order: tuple[str, str],
    eps: float,
    shrinkage: float,
) -> np.ndarray:
    """Soft pseudo-label covariance difference using class probabilities as weights.

    For each trial i, compute Ci = Xi Xi^T. Then:
        Cov_c = sum_i w_{i,c} Ci / sum_i w_{i,c}
    and Δ = Cov_{class1} - Cov_{class0}.
    """

    X = np.asarray(X, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n_trials,n_channels,n_times); got {X.shape}.")
    if proba.ndim != 2 or proba.shape[0] != X.shape[0] or proba.shape[1] != 2:
        raise ValueError(f"Expected proba shape (n_trials,2); got {proba.shape}.")

    w0 = proba[:, 0].clip(0.0, 1.0)
    w1 = proba[:, 1].clip(0.0, 1.0)
    if float(np.sum(w0)) <= 0.0 or float(np.sum(w1)) <= 0.0:
        raise ValueError("Soft pseudo-label weights degenerate (sum to zero).")

    n_trials, n_channels, _ = X.shape
    cov0 = np.zeros((n_channels, n_channels), dtype=np.float64)
    cov1 = np.zeros((n_channels, n_channels), dtype=np.float64)
    for i in range(n_trials):
        xi = X[i]
        ci = xi @ xi.T
        cov0 += float(w0[i]) * ci
        cov1 += float(w1[i]) * ci
    cov0 /= float(np.sum(w0))
    cov1 /= float(np.sum(w1))
    cov0 = 0.5 * (cov0 + cov0.T)
    cov1 = 0.5 * (cov1 + cov1.T)

    if shrinkage > 0.0:
        alpha = float(shrinkage)
        cov0 = (1.0 - alpha) * cov0 + alpha * (np.trace(cov0) / float(n_channels)) * np.eye(
            n_channels, dtype=np.float64
        )
        cov1 = (1.0 - alpha) * cov1 + alpha * (np.trace(cov1) / float(n_channels)) * np.eye(
            n_channels, dtype=np.float64
        )

    # eps only affects eigenvalue flooring in the EA helper; we mimic that by flooring on the diff stage.
    diff = cov1 - cov0
    diff = 0.5 * (diff + diff.T)
    # Light diagonal jitter for numerical stability (keeps symmetry).
    jitter = float(eps) * float(np.max(np.abs(np.diag(diff))) + 1.0)
    diff = diff + jitter * np.eye(n_channels, dtype=np.float64)
    return diff


def summarize_results(results_df: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    """Return mean/std/min/max summary for selected metric columns."""

    summary = {
        "mean": results_df[metric_columns].mean(numeric_only=True),
        "std": results_df[metric_columns].std(numeric_only=True),
        "min": results_df[metric_columns].min(numeric_only=True),
        "max": results_df[metric_columns].max(numeric_only=True),
    }
    return pd.DataFrame(summary)
