from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


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

