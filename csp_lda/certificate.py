from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RidgeCertificate:
    """Simple calibrated certificate model for candidate selection.

    This regresses (unlabeled features) -> (expected improvement over the EA anchor)
    on pseudo-target subjects.
    """

    model: Pipeline
    feature_names: tuple[str, ...]

    def predict_accuracy(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return np.asarray(self.model.predict(features), dtype=np.float64)


def _safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def candidate_features_from_record(
    rec: dict,
    *,
    n_classes: int,
    include_pbar: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a feature vector from a candidate record (label-free)."""

    p_bar = np.asarray(rec.get("p_bar_full", np.zeros(n_classes)), dtype=np.float64).reshape(-1)
    if p_bar.shape[0] != n_classes:
        p_bar = np.zeros(n_classes, dtype=np.float64)
    p_bar = np.clip(p_bar, 1e-12, 1.0)
    p_bar = p_bar / float(np.sum(p_bar))
    p_bar_ent = float(-np.sum(p_bar * np.log(p_bar)))

    n_keep = int(rec.get("n_keep", -1))
    n_best_total = int(rec.get("n_best_total", -1))
    keep_ratio = 0.0
    if n_keep >= 0 and n_best_total > 0:
        keep_ratio = float(n_keep) / float(n_best_total)

    feats: list[float] = [
        _safe_float(rec.get("objective_base", 0.0)),
        _safe_float(rec.get("pen_marginal", 0.0)),
        _safe_float(rec.get("drift_best", 0.0)),
        _safe_float(rec.get("mean_entropy", 0.0)),
        _safe_float(rec.get("entropy_bar", 0.0)),
        _safe_float(keep_ratio),
        _safe_float(p_bar_ent),
    ]
    names: list[str] = [
        "objective_base",
        "pen_marginal",
        "drift_best",
        "mean_entropy",
        "entropy_bar",
        "keep_ratio",
        "pbar_entropy",
    ]

    if include_pbar:
        feats.extend([_safe_float(x) for x in p_bar.tolist()])
        names.extend([f"pbar_{k}" for k in range(n_classes)])

    return np.asarray(feats, dtype=np.float64), tuple(names)


def train_ridge_certificate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    alpha: float = 1.0,
) -> RidgeCertificate:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch.")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be > 0.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(X, y)
    return RidgeCertificate(model=model, feature_names=tuple(feature_names))


def select_by_predicted_improvement(
    records: Iterable[dict],
    *,
    cert: RidgeCertificate,
    n_classes: int,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
) -> dict:
    """Select the best candidate record using the calibrated certificate.

    Returns the selected record (dict).
    """

    best: dict | None = None
    best_score = -float("inf")
    identity: dict | None = None

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        feats, _names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        pred_improve = float(cert.predict_accuracy(feats)[0])
        drift = _safe_float(rec.get("drift_best", 0.0))

        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            pred_improve = float(pred_improve) - float(drift_gamma) * float(drift)

        if pred_improve > best_score:
            best_score = float(pred_improve)
            best = rec

    # Safety: if the best predicted improvement is non-positive, fall back to identity.
    if best is None or best_score <= 0.0:
        if identity is not None:
            return identity
        # If identity missing, fall back to the first record.
        for rec in records:
            return rec
        raise ValueError("No candidates to select from.")

    return best
