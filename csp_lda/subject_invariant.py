from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

from .alignment import EuclideanAligner


def _one_hot(values: Sequence, *, classes: Sequence) -> np.ndarray:
    classes = [str(c) for c in classes]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(values)
    k = len(classes)
    out = np.zeros((n, k), dtype=np.float64)
    for i, v in enumerate(values):
        out[i, cls_to_idx[str(v)]] = 1.0
    return out


def _center_cols(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return M - np.mean(M, axis=0, keepdims=True)


@dataclass(frozen=True)
class HSICProjectorParams:
    """Hyper-params for the linear HSIC-style subject-invariant projector."""

    subject_lambda: float = 1.0
    ridge: float = 1e-6
    n_components: int | None = None


class CenteredLinearProjector(BaseEstimator, TransformerMixin):
    """Centered linear feature projector: (x-mean) @ W.

    This is intentionally a simple deterministic transformer so it can be inserted
    between CSP and LDA. It is fitted externally (we set `mean_` and `W_`).
    """

    def __init__(self, mean: np.ndarray | None = None, W: np.ndarray | None = None) -> None:
        self.mean = mean
        self.W = W
        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        if self.mean is None or self.W is None:
            raise ValueError("CenteredLinearProjector must be constructed with mean and W (already learned).")
        self.mean_ = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.W_ = np.asarray(self.W, dtype=np.float64)
        return self

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        if self.mean_ is None or self.W_ is None:
            # Allow using the object without a prior `fit()` if mean/W were provided.
            if self.mean is None or self.W is None:
                raise RuntimeError("CenteredLinearProjector is not fitted.")
            self.mean_ = np.asarray(self.mean, dtype=np.float64).reshape(-1)
            self.W_ = np.asarray(self.W, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_.reshape(1, -1)) @ self.W_


def learn_hsic_subject_invariant_projector(
    *,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    class_order: Sequence[str],
    params: HSICProjectorParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn a linear projector that encourages subject invariance while preserving label structure.

    We work in feature space (e.g. CSP log-variance features) and learn W by a
    generalized eigen-problem:

      maximize   tr(Wᵀ (A - λ B) W)
      s.t.       Wᵀ (C + ρI) W = I

    where:
      A = Xᵀ Y Yᵀ X   (linear HSIC with class one-hot)
      B = Xᵀ S Sᵀ X   (linear HSIC with subject one-hot)
      C = Xᵀ X        (whitening constraint / scale control)

    Returns
    -------
    mean:
        Feature mean to center by (shape (d,)).
    W:
        Projection matrix (shape (d,r)), r = params.n_components or d.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D features (n_samples,d); got {X.shape}.")
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least 2 samples to learn a projector.")
    if y.shape[0] != n or subjects.shape[0] != n:
        raise ValueError("X/y/subjects length mismatch.")

    if float(params.subject_lambda) < 0.0:
        raise ValueError("params.subject_lambda must be >= 0.")
    if float(params.ridge) <= 0.0:
        raise ValueError("params.ridge must be > 0.")

    mean = np.mean(X, axis=0)
    Xc = X - mean.reshape(1, -1)

    # One-hot label matrices.
    Y = _one_hot(y, classes=list(class_order))
    subj_classes = [str(s) for s in sorted({int(s) for s in subjects.tolist()})]
    S = _one_hot(subjects, classes=subj_classes)
    Yc = _center_cols(Y)
    Sc = _center_cols(S)

    # Build (A - λB) and the constraint matrix C.
    M_y = Yc @ Yc.T
    M_s = Sc @ Sc.T
    A = Xc.T @ M_y @ Xc
    B = Xc.T @ M_s @ Xc
    M = A - float(params.subject_lambda) * B
    M = 0.5 * (M + M.T)

    C = Xc.T @ Xc
    C = 0.5 * (C + C.T)
    C = C + float(params.ridge) * np.eye(d, dtype=np.float64)

    # Generalized eigen-decomposition: M v = λ C v.
    evals, evecs = linalg.eigh(M, C)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    r = int(params.n_components) if params.n_components is not None else int(d)
    r = max(1, min(int(d), r))
    W = np.asarray(evecs[:, :r], dtype=np.float64)
    return np.asarray(mean, dtype=np.float64), W


@dataclass(frozen=True)
class ChannelProjectorParams:
    """Hyper-params for a channel-space subject-invariant projection.

    We learn a low-rank channel projector (rank r) from labeled multi-subject data,
    then apply it before CSP. This avoids the "LDA cancels linear feature transforms"
    degeneracy of feature-space projectors, because we constrain the *signal* to a
    subspace (rank-deficient transform).
    """

    subject_lambda: float = 1.0
    ridge: float = 1e-6
    n_components: int | None = None


def _fix_vector_signs(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    out = V.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        i = int(np.argmax(np.abs(col)))
        if col[i] < 0:
            out[:, j] = -col
    return out


def _sym_inv_sqrt_spd(A: np.ndarray, *, eps: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = _fix_vector_signs(evecs[:, order])
    floor = float(eps) * float(np.max(evals)) if np.max(evals) > 0 else float(eps)
    evals = np.maximum(evals, floor)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


def learn_subject_invariant_channel_projector(
    *,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    class_order: Sequence[str],
    eps: float = 1e-10,
    shrinkage: float = 0.0,
    params: ChannelProjectorParams,
) -> np.ndarray:
    """Learn a rank-r channel projector A (C×C) to reduce inter-subject variability.

    We build two PSD matrices in channel space:
    - Class between-scatter B: variability of class-conditional mean covariances.
    - Subject scatter S: variability of subject-wise class-conditional covariances around the global ones.

    Then we solve the generalized eigen-problem:
        maximize  vᵀ B v    subject to  vᵀ (λ S + ρ I) v = 1

    and form a rank-r projector A = W Wᵀ (rank r) from the top eigenvectors.

    Notes
    -----
    - If r >= C (or r is None/0), we return identity (no-op).
    - This is intended to be applied *before* CSP, so it changes the signal space
      and cannot be cancelled by a linear classifier reparameterization.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    class_order = [str(c) for c in class_order]

    if X.ndim != 3:
        raise ValueError(f"X must be (n_trials,n_channels,n_times); got {X.shape}.")
    n_trials, n_channels, _n_times = X.shape
    if int(n_trials) < 2:
        raise ValueError("Need at least 2 trials to learn a channel projector.")
    if y.shape[0] != int(n_trials) or subjects.shape[0] != int(n_trials):
        raise ValueError("X/y/subjects length mismatch.")

    if float(params.subject_lambda) < 0.0:
        raise ValueError("params.subject_lambda must be >= 0.")
    if float(params.ridge) <= 0.0:
        raise ValueError("params.ridge must be > 0.")

    r = params.n_components
    if r is None or int(r) <= 0 or int(r) >= int(n_channels):
        return np.eye(int(n_channels), dtype=np.float64)

    subj_ids = sorted({int(s) for s in subjects.tolist()})
    if len(subj_ids) < 2:
        # No inter-subject shift to reduce.
        return np.eye(int(n_channels), dtype=np.float64)

    # Compute subject-wise class covariances Σ_{s,k}.
    cov_sk: dict[tuple[int, str], tuple[np.ndarray, int]] = {}
    n_total = int(n_trials)
    for s in subj_ids:
        mask_s = subjects == int(s)
        for c in class_order:
            mask = mask_s & (y == c)
            n_sc = int(np.sum(mask))
            if n_sc <= 0:
                continue
            cov = EuclideanAligner(eps=float(eps), shrinkage=float(shrinkage)).fit(X[mask]).cov_
            cov_sk[(int(s), str(c))] = (np.asarray(cov, dtype=np.float64), n_sc)

    # Global class covariances Σ_k (weighted by counts across subjects).
    cov_k: dict[str, np.ndarray] = {}
    pi_k: dict[str, float] = {}
    for c in class_order:
        num = np.zeros((n_channels, n_channels), dtype=np.float64)
        denom = 0
        for s in subj_ids:
            key = (int(s), str(c))
            if key not in cov_sk:
                continue
            cov, n_sc = cov_sk[key]
            num += float(n_sc) * cov
            denom += int(n_sc)
        if denom <= 0:
            continue
        cov_k[str(c)] = 0.5 * (num / float(denom) + (num / float(denom)).T)
        pi_k[str(c)] = float(denom) / float(n_total)

    if len(cov_k) < 2:
        return np.eye(int(n_channels), dtype=np.float64)

    # S_w := Σ̄ = Σ_k π_k Σ_k.
    sw = np.zeros((n_channels, n_channels), dtype=np.float64)
    for c, cov in cov_k.items():
        sw += float(pi_k[c]) * cov
    sw = 0.5 * (sw + sw.T)

    # Between-class scatter in channel space: B := Σ_k π_k (Σ_k - Σ̄)^2.
    B = np.zeros((n_channels, n_channels), dtype=np.float64)
    for c, cov in cov_k.items():
        d = cov - sw
        B += float(pi_k[c]) * (d @ d)
    B = 0.5 * (B + B.T)

    # Subject scatter: S := Σ_{s,k} ω_{s,k} (Σ_{s,k} - Σ_k)^2.
    S = np.zeros((n_channels, n_channels), dtype=np.float64)
    for (s, c), (cov, n_sc) in cov_sk.items():
        if c not in cov_k:
            continue
        d = cov - cov_k[c]
        S += (float(n_sc) / float(n_total)) * (d @ d)
    S = 0.5 * (S + S.T)

    C = float(params.subject_lambda) * S + float(params.ridge) * np.eye(n_channels, dtype=np.float64)
    C = 0.5 * (C + C.T)

    C_inv_sqrt = _sym_inv_sqrt_spd(C, eps=float(eps))
    M = C_inv_sqrt @ B @ C_inv_sqrt
    M = 0.5 * (M + M.T)

    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(evals)[::-1]
    evecs = _fix_vector_signs(evecs[:, order])

    U = evecs[:, : int(r)]
    W = C_inv_sqrt @ U
    # Orthonormalize for a stable Euclidean projector.
    Q, _ = np.linalg.qr(W)
    A = Q @ Q.T
    A = 0.5 * (A + A.T)
    return np.asarray(A, dtype=np.float64)
