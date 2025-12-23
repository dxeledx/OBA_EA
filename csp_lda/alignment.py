from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Sequence


class BaseAligner(BaseEstimator, TransformerMixin):
    """Base interface for (future) alignment methods, e.g., Euclidean Alignment (EA).

    This follows the scikit-learn transformer API so it can be inserted into a
    `sklearn.pipeline.Pipeline` before CSP.

    EA (Euclidean Alignment) typically aims to reduce inter-subject covariate shift
    by aligning covariance structure across subjects. We keep this as an extension
    point (see user requirement).
    """

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        return self

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        return X


class NoAligner(BaseAligner):
    """No-op aligner (default)."""


@dataclass
class _EAState:
    cov: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    whitening: np.ndarray


class EuclideanAligner(BaseAligner):
    """Euclidean Alignment (EA) as in He & Wu.

    For a set of trials from *one subject*, compute:

        R_bar = (1/n) * sum_i (X_i X_i^T)
        X_i_tilde = R_bar^{-1/2} X_i

    where X_i has shape (n_channels, n_times). This aligner is unsupervised.

    Notes
    -----
    - To match the paper's intended usage, call `fit_transform` **per subject**.
      (In LOSO, align each subject independently.)
    - We use an eigen-decomposition to compute the inverse square root with
      eigenvalue flooring for numerical stability.
    """

    def __init__(self, eps: float = 1e-10, shrinkage: float = 0.0) -> None:
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
        self._state: _EAState | None = None

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"EA expects X with shape (n_trials,n_channels,n_times), got {X.shape}.")
        n_trials, n_channels, _n_times = X.shape
        if n_trials < 1:
            raise ValueError("EA requires at least 1 trial.")

        # R_bar = mean_i (X_i X_i^T)
        r_bar = np.zeros((n_channels, n_channels), dtype=np.float64)
        for i in range(n_trials):
            xi = X[i]
            r_bar += xi @ xi.T
        r_bar /= float(n_trials)

        # Symmetrize for numerical stability
        r_bar = 0.5 * (r_bar + r_bar.T)

        # Optional shrinkage to enforce a spectral floor (useful for stability arguments).
        # Using trace-scaled identity keeps units consistent.
        if self.shrinkage > 0.0:
            if not (0.0 <= self.shrinkage < 1.0):
                raise ValueError("shrinkage must be in [0, 1).")
            alpha = float(self.shrinkage)
            r_bar = (1.0 - alpha) * r_bar + alpha * (np.trace(r_bar) / float(n_channels)) * np.eye(
                n_channels, dtype=np.float64
            )

        # Compute whitening matrix: r_bar^{-1/2}.
        eigvals, eigvecs = np.linalg.eigh(r_bar)
        # Sort eigenpairs by descending eigenvalue for deterministic orientation.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        eigvecs = _fix_eigvec_signs(eigvecs)

        floor = self.eps * float(np.max(eigvals)) if np.max(eigvals) > 0 else self.eps
        eigvals = np.maximum(eigvals, floor)
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        self._state = _EAState(cov=r_bar, eigvals=eigvals, eigvecs=eigvecs, whitening=whitening)
        return self

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        if self._state is None:
            raise RuntimeError("EuclideanAligner must be fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        whitening = self._state.whitening
        return np.einsum("ij,njt->nit", whitening, X, optimize=True)

    @property
    def whitening_(self) -> np.ndarray:
        if self._state is None:
            raise AttributeError("whitening_ is not available before fit().")
        return self._state.whitening

    @property
    def cov_(self) -> np.ndarray:
        if self._state is None:
            raise AttributeError("cov_ is not available before fit().")
        return self._state.cov

    @property
    def eigvecs_(self) -> np.ndarray:
        if self._state is None:
            raise AttributeError("eigvecs_ is not available before fit().")
        return self._state.eigvecs


def _fix_eigvec_signs(eigvecs: np.ndarray) -> np.ndarray:
    """Make eigenvector signs deterministic (per-column).

    Eigenvectors are defined up to sign; we fix it by enforcing the entry with the
    largest absolute magnitude to be positive.
    """

    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    out = eigvecs.copy()
    for i in range(out.shape[1]):
        col = out[:, i]
        j = int(np.argmax(np.abs(col)))
        if col[j] < 0:
            out[:, i] = -col
    return out


def apply_spatial_transform(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a channel-space linear transform to trials.

    Parameters
    ----------
    W:
        (n_channels, n_channels) matrix.
    X:
        (n_trials, n_channels, n_times) epochs array.
    """

    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    return np.einsum("ij,njt->nit", W, X, optimize=True)


def class_cov_diff(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_order: Sequence[str],
    eps: float = 1e-10,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Compute a symmetric discriminative covariance signature.

    - Binary (2-class): Δ = Cov(class1) - Cov(class0).
    - Multiclass (>=3): an LDA-inspired discriminative operator:

        Let Σ_k = Cov(X | y=k), and Σ̄ = Σ_k π_k Σ_k (π_k empirical class frequency).
        Define within-scatter S_w := Σ̄ and between-scatter
          S_b := Σ_k π_k (Σ_k - Σ̄)(Σ_k - Σ̄).
        Return M := S_w^{-1/2} S_b S_w^{-1/2}.

    This M is symmetric PSD and transforms as M -> Q M Qᵀ under orthogonal Q,
    making it suitable for eigen-basis alignment (OEA) in multiclass settings.
    """

    class_order = [str(c) for c in class_order]
    if len(class_order) < 2:
        raise ValueError("class_order must contain at least 2 classes.")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials,n_channels,n_times); got {X.shape}.")

    if len(class_order) == 2:
        c0, c1 = class_order
        mask0 = y == c0
        mask1 = y == c1
        if not (np.any(mask0) and np.any(mask1)):
            raise ValueError("Both classes must be present to compute class_cov_diff.")

        cov0 = EuclideanAligner(eps=eps, shrinkage=shrinkage).fit(X[mask0]).cov_
        cov1 = EuclideanAligner(eps=eps, shrinkage=shrinkage).fit(X[mask1]).cov_
        diff = cov1 - cov0
        return 0.5 * (diff + diff.T)

    n_trials = int(X.shape[0])
    n_channels = int(X.shape[1])
    if n_trials < 2:
        raise ValueError("Need at least 2 trials to compute class covariance signature.")

    covs: list[np.ndarray] = []
    weights: list[float] = []
    for c in class_order:
        mask = y == c
        if not np.any(mask):
            continue
        cov_c = EuclideanAligner(eps=eps, shrinkage=shrinkage).fit(X[mask]).cov_
        covs.append(cov_c)
        weights.append(float(np.sum(mask)) / float(n_trials))

    if len(covs) < 2:
        raise ValueError("At least two classes must be present to compute a multiclass signature.")

    w = np.asarray(weights, dtype=np.float64)
    w = w / float(np.sum(w))
    sw = np.zeros((n_channels, n_channels), dtype=np.float64)
    for wk, cov_k in zip(w, covs):
        sw += float(wk) * cov_k
    sw = 0.5 * (sw + sw.T)

    sb = np.zeros((n_channels, n_channels), dtype=np.float64)
    for wk, cov_k in zip(w, covs):
        d = cov_k - sw
        sb += float(wk) * (d @ d)
    sb = 0.5 * (sb + sb.T)

    # Whitening by S_w^{-1/2} (stable via eigenvalue flooring).
    eigvals, eigvecs = np.linalg.eigh(sw)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvecs = _fix_eigvec_signs(eigvecs)
    floor = float(eps) * float(np.max(eigvals)) if np.max(eigvals) > 0 else float(eps)
    eigvals = np.maximum(eigvals, floor)
    sw_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    sig = sw_inv_sqrt @ sb @ sw_inv_sqrt
    sig = 0.5 * (sig + sig.T)

    # Light diagonal jitter for numerical stability (keeps symmetry).
    jitter = float(eps) * float(np.max(np.abs(np.diag(sig))) + 1.0)
    sig = sig + jitter * np.eye(n_channels, dtype=np.float64)
    return sig


def orthogonal_align_symmetric(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return Q∈O(C) that (approximately) aligns symmetric A to symmetric B.

    We use eigen-basis alignment: if A=UΛUᵀ and B=VΜVᵀ with eigenvalues sorted in the
    same order, then Q = V Uᵀ minimizes ||Q A Qᵀ - B||_F over orthogonal Q (up to
    eigenvalue permutations/sign ambiguities).
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A and B must be same-shape square matrices; got {A.shape} and {B.shape}.")

    # Symmetrize to guard numerical noise.
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    evals_a, evecs_a = np.linalg.eigh(A)
    evals_b, evecs_b = np.linalg.eigh(B)

    idx_a = np.argsort(evals_a)[::-1]
    idx_b = np.argsort(evals_b)[::-1]
    evecs_a = _fix_eigvec_signs(evecs_a[:, idx_a])
    evecs_b = _fix_eigvec_signs(evecs_b[:, idx_b])

    Q = evecs_b @ evecs_a.T
    # Numerical safeguard: project to nearest orthogonal (polar factor).
    # For well-conditioned eigen-bases this should already be orthogonal.
    U, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return U @ Vt


def sorted_eigh(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition with eigenvalues sorted descending and deterministic signs."""

    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"mat must be square; got {mat.shape}.")
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = _fix_eigvec_signs(evecs[:, order])
    return evals, evecs


def blend_with_identity(Q: np.ndarray, alpha: float) -> np.ndarray:
    """Blend an orthogonal matrix with identity and re-project to O(C).

    This is a simple way to control how aggressive a Q-selection is:
    - alpha=0 => identity
    - alpha=1 => Q
    """

    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square; got {Q.shape}.")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    n = int(Q.shape[0])
    M = (1.0 - float(alpha)) * np.eye(n, dtype=np.float64) + float(alpha) * Q
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt
