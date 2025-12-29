from __future__ import annotations

import numpy as np


def covariances_from_epochs(
    X: np.ndarray,
    *,
    eps: float = 1e-10,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Compute SPD trial covariances from epochs.

    Parameters
    ----------
    X:
        Epochs array with shape (n_trials, n_channels, n_times).
    eps:
        Eigenvalue floor as eps * max_eig (per-trial) to ensure SPD.
    shrinkage:
        Optional shrinkage in [0,1): (1-a)*C + a*(tr(C)/C)I.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials,n_channels,n_times); got {X.shape}.")
    n_trials, n_channels, n_times = X.shape
    if int(n_trials) < 1:
        raise ValueError("Need at least 1 trial to compute covariances.")

    if float(shrinkage) > 0.0 and not (0.0 <= float(shrinkage) < 1.0):
        raise ValueError("shrinkage must be in [0, 1).")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0.")

    covs = np.empty((int(n_trials), int(n_channels), int(n_channels)), dtype=np.float64)
    eye = np.eye(int(n_channels), dtype=np.float64)
    scale = 1.0 / float(max(1, int(n_times)))
    alpha = float(shrinkage)

    for i in range(int(n_trials)):
        xi = X[int(i)]
        cov = scale * (xi @ xi.T)
        cov = 0.5 * (cov + cov.T)

        if alpha > 0.0:
            cov = (1.0 - alpha) * cov + alpha * (float(np.trace(cov)) / float(n_channels)) * eye

        # SPD floor via eigenvalue clipping.
        evals, evecs = np.linalg.eigh(cov)
        floor = float(eps) * float(np.max(evals)) if float(np.max(evals)) > 0.0 else float(eps)
        evals = np.maximum(evals, floor)
        covs[int(i)] = evecs @ np.diag(evals) @ evecs.T

    return covs

