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

    if alignment not in {"none", "ea", "oea_cov", "oea", "oea_zo"}:
        raise ValueError("alignment must be one of: 'none', 'ea', 'oea_cov', 'oea', 'oea_zo'")

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
            # alignment in {"oea","oea_zo"}: optimistic selection based on discriminative covariance difference Δ.
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

            # 5) Target subject: select Q_t using *unlabeled* target data (no classifier update).
            z_t = z_by_subject[int(test_subject)]
            if alignment == "oea":
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
            else:
                q_t = _optimize_qt_oea_zo(
                    z_t=z_t,
                    model=model,
                    class_pair=class_pair,
                    d_ref=d_ref,
                    eps=float(oea_eps),
                    shrinkage=float(oea_shrinkage),
                    pseudo_mode=str(oea_pseudo_mode),
                    warm_start=str(oea_zo_warm_start),
                    warm_iters=int(oea_zo_warm_iters),
                    q_blend=float(oea_q_blend),
                    objective=str(oea_zo_objective),
                    infomax_lambda=float(oea_zo_infomax_lambda),
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

        if alignment not in {"oea", "oea_zo"}:
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


def _optimize_qt_oea_zo(
    *,
    z_t: np.ndarray,
    model: TrainedModel,
    class_pair: tuple[str, str],
    d_ref: np.ndarray,
    eps: float,
    shrinkage: float,
    pseudo_mode: str,
    warm_start: str,
    warm_iters: int,
    q_blend: float,
    objective: str,
    infomax_lambda: float,
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

    if pseudo_mode not in {"hard", "soft"}:
        raise ValueError("pseudo_mode must be one of: 'hard', 'soft'")
    if warm_start not in {"none", "delta"}:
        raise ValueError("warm_start must be one of: 'none', 'delta'")
    if not (0.0 <= float(holdout_fraction) < 1.0):
        raise ValueError("holdout_fraction must be in [0,1).")
    if float(fallback_min_marginal_entropy) < 0.0:
        raise ValueError("fallback_min_marginal_entropy must be >= 0.")

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

    def _proba_from_q(phi_vec: np.ndarray, z_data: np.ndarray) -> np.ndarray:
        Q = _build_q_from_givens(n_channels=n_channels, pairs=pairs, angles=phi_vec)
        if float(q_blend) < 1.0:
            Q = blend_with_identity(Q, float(q_blend))
        FQ = F @ Q
        Y = np.einsum("kc,nct->nkt", FQ, z_data, optimize=True)
        power = np.mean(Y * Y, axis=2)
        power = np.maximum(power, 1e-20)
        feats = np.log(power) if use_log else power
        proba = lda.predict_proba(feats)
        return _reorder_proba_columns(proba, lda.classes_, list(class_pair))

    def _maybe_select_keep(proba: np.ndarray) -> np.ndarray:
        """Optionally select a reliable subset based on confidence/top-k/balance settings.

        When the user provides any of {pseudo_confidence, pseudo_topk_per_class, pseudo_balance},
        we reuse the pseudo selection logic to keep only confident trials. This is used to
        stabilize *all* ZO objectives (entropy/infomax/confidence) and not only pseudo_ce.
        """

        if float(pseudo_confidence) <= 0.0 and int(pseudo_topk_per_class) == 0 and not bool(pseudo_balance):
            return np.arange(proba.shape[0], dtype=int)
        pred_idx = np.argmax(proba, axis=1)
        y_pseudo = np.where(pred_idx == 0, class_pair[0], class_pair[1])
        return _select_pseudo_indices(
            y_pseudo=y_pseudo,
            proba=proba,
            class_order=class_pair,
            confidence=float(pseudo_confidence),
            topk_per_class=int(pseudo_topk_per_class),
            balance=bool(pseudo_balance),
        )

    def eval_phi(phi_vec: np.ndarray, z_data: np.ndarray) -> float:
        proba = _proba_from_q(phi_vec, z_data)

        if objective in {"entropy", "infomax", "confidence"}:
            keep = _maybe_select_keep(proba)
            if keep.size == 0:
                return 1e6
            proba = proba[keep]

        if objective == "entropy":
            p = np.clip(proba, 1e-12, 1.0)
            p = p / np.sum(p, axis=1, keepdims=True)
            ent = -np.sum(p * np.log(p), axis=1)
            val = float(np.mean(ent))
        elif objective == "infomax":
            # Maximize mutual information I(Y;X) = H(mean p) - mean H(p).
            # We minimize: mean H(p) - λ * H(mean p).
            p = np.clip(proba, 1e-12, 1.0)
            p = p / np.sum(p, axis=1, keepdims=True)
            ent = -np.sum(p * np.log(p), axis=1)
            p_bar = np.mean(p, axis=0)
            p_bar = np.clip(p_bar, 1e-12, 1.0)
            p_bar = p_bar / np.sum(p_bar)
            ent_bar = -float(np.sum(p_bar * np.log(p_bar)))
            val = float(np.mean(ent)) - float(infomax_lambda) * ent_bar
        elif objective == "confidence":
            conf = np.max(proba, axis=1)
            val = float(np.mean(1.0 - conf))
        else:
            # pseudo_ce: hard pseudo labels + optional filtering
            pred_idx = np.argmax(proba, axis=1)
            y_pseudo = np.where(pred_idx == 0, class_pair[0], class_pair[1])
            keep = _select_pseudo_indices(
                y_pseudo=y_pseudo,
                proba=proba,
                class_order=class_pair,
                confidence=float(pseudo_confidence),
                topk_per_class=int(pseudo_topk_per_class),
                balance=bool(pseudo_balance),
            )
            if keep.size == 0:
                return 1e6
            pred_idx_k = np.argmax(proba[keep], axis=1)
            conf_k = proba[keep, pred_idx_k]
            conf_k = np.clip(conf_k, 1e-12, 1.0)
            nll = -np.log(conf_k)
            # Weight by confidence (encourages self-consistent high-confidence predictions).
            val = float(np.mean(conf_k * nll))

        if l2 > 0.0:
            val += float(l2) * float(np.mean(phi_vec * phi_vec))
        return val

    # Optional warm start: build Q_Δ from pseudo-label Δ-alignment and approximate it with our Givens pairs.
    q_delta: np.ndarray | None = None
    if warm_start == "delta" and int(warm_iters) > 0:
        q_cur = np.eye(int(n_channels), dtype=np.float64)
        for _ in range(int(warm_iters)):
            X_cur = apply_spatial_transform(q_cur, z_t)
            proba = model.predict_proba(X_cur)
            proba = _reorder_proba_columns(proba, model.classes_, list(class_pair))
            try:
                if pseudo_mode == "soft":
                    d_t = _soft_class_cov_diff(
                        z_t,
                        proba=proba,
                        class_order=class_pair,
                        eps=float(eps),
                        shrinkage=float(shrinkage),
                    )
                else:
                    y_pseudo = np.asarray(model.predict(X_cur))
                    keep = _select_pseudo_indices(
                        y_pseudo=y_pseudo,
                        proba=proba,
                        class_order=class_pair,
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
                        class_order=class_pair,
                        eps=float(eps),
                        shrinkage=float(shrinkage),
                    )
                q_cur = orthogonal_align_symmetric(d_t, d_ref)
                q_delta = q_cur
            except ValueError:
                q_delta = None
                break

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
            phi = phi_init.copy()

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
        proba_best = _proba_from_q(best_phi, z_t)
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
                proba_cand = _reorder_proba_columns(proba_cand, lda.classes_, list(class_pair))
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
