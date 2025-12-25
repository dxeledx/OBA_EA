from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
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


@dataclass(frozen=True)
class LogisticGuard:
    """Binary guard model for rejecting likely negative-transfer candidates.

    Models: P(improvement_over_identity > margin | unlabeled_features).
    """

    model: Pipeline
    feature_names: tuple[str, ...]

    def predict_pos_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = np.asarray(self.model.predict_proba(features), dtype=np.float64)
        if proba.ndim != 2:
            raise ValueError("Unexpected proba shape.")
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]


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

    # Optional bilevel stats (may be missing on older runs).
    coverage = _safe_float(rec.get("coverage", 0.0))
    eff_n = _safe_float(rec.get("eff_n", 0.0))
    mean_entropy_q = _safe_float(rec.get("mean_entropy_q", 0.0))

    drift_best = _safe_float(rec.get("drift_best", 0.0))
    drift_best_std = _safe_float(rec.get("drift_best_std", 0.0))
    drift_best_q90 = _safe_float(rec.get("drift_best_q90", 0.0))
    drift_best_q95 = _safe_float(rec.get("drift_best_q95", 0.0))
    drift_best_max = _safe_float(rec.get("drift_best_max", 0.0))
    drift_best_tail_frac = _safe_float(rec.get("drift_best_tail_frac", 0.0))

    q_bar = np.asarray(rec.get("q_bar", np.zeros(n_classes)), dtype=np.float64).reshape(-1)
    if q_bar.shape[0] != n_classes:
        q_bar = np.zeros(n_classes, dtype=np.float64)
    q_bar = np.clip(q_bar, 1e-12, 1.0)
    q_bar = q_bar / float(np.sum(q_bar))
    q_bar_ent = float(-np.sum(q_bar * np.log(q_bar)))

    feats: list[float] = [
        _safe_float(rec.get("objective_base", 0.0)),
        _safe_float(rec.get("pen_marginal", 0.0)),
        drift_best,
        drift_best_std,
        drift_best_q90,
        drift_best_q95,
        drift_best_max,
        drift_best_tail_frac,
        _safe_float(rec.get("mean_entropy", 0.0)),
        mean_entropy_q,
        _safe_float(rec.get("entropy_bar", 0.0)),
        _safe_float(keep_ratio),
        coverage,
        eff_n,
        _safe_float(p_bar_ent),
        _safe_float(q_bar_ent),
    ]
    names: list[str] = [
        "objective_base",
        "pen_marginal",
        "drift_best",
        "drift_best_std",
        "drift_best_q90",
        "drift_best_q95",
        "drift_best_max",
        "drift_best_tail_frac",
        "mean_entropy",
        "mean_entropy_q",
        "entropy_bar",
        "keep_ratio",
        "coverage",
        "eff_n",
        "pbar_entropy",
        "qbar_entropy",
    ]

    if include_pbar:
        feats.extend([_safe_float(x) for x in p_bar.tolist()])
        names.extend([f"pbar_{k}" for k in range(n_classes)])
        feats.extend([_safe_float(x) for x in q_bar.tolist()])
        names.extend([f"qbar_{k}" for k in range(n_classes)])

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


def train_logistic_guard(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    c: float = 1.0,
) -> LogisticGuard:
    """Train a negative-transfer guard on pseudo-target subjects."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch.")
    if float(c) <= 0.0:
        raise ValueError("c must be > 0.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(c),
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X, y)
    return LogisticGuard(model=model, feature_names=tuple(feature_names))


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


def select_by_guarded_objective(
    records: Iterable[dict],
    *,
    guard: LogisticGuard,
    n_classes: int,
    threshold: float = 0.5,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
) -> dict:
    """Guarded selection: first reject likely negative-transfer candidates, then pick by objective.

    Selection rule:
    - keep candidates with P(pos_improve) >= threshold (identity is always allowed),
    - apply optional drift hard/penalty,
    - choose the minimum recorded `score` (or `objective` if score missing).
    """

    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0,1].")

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        feats, _names = candidate_features_from_record(rec, n_classes=n_classes, include_pbar=True)
        p_pos = float(guard.predict_pos_proba(feats)[0])
        if p_pos < float(threshold) and str(rec.get("kind", "")) != "identity":
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = _safe_float(rec.get("score", rec.get("objective", 0.0)))
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if best is not None:
        return best
    if identity is not None:
        return identity
    for rec in records:
        return rec
    raise ValueError("No candidates to select from.")


def select_by_evidence_nll(
    records: Iterable[dict],
    *,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
) -> dict:
    """Select the best candidate using LDA evidence (-log p(z)).

    Candidates must have `evidence_nll_best` recorded (smaller is better).
    If no candidate improves over the identity anchor, return identity.
    """

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        try:
            ev = float(rec.get("evidence_nll_best", float("nan")))
        except Exception:
            ev = float("nan")
        if not np.isfinite(ev):
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = float(ev)
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        ev_id = float(identity.get("evidence_nll_best", float("nan")))
    except Exception:
        ev_id = float("nan")
    if not np.isfinite(ev_id):
        return best if best is not None else identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(ev_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(ev_id):
        return identity

    return best


def select_by_probe_mixup(
    records: Iterable[dict],
    *,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
) -> dict:
    """Select the best candidate using a MixUp-style probe score.

    Candidates must have `probe_mixup_best` recorded (smaller is better).
    If no candidate improves over the identity anchor, return identity.
    """

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        try:
            s = float(rec.get("probe_mixup_best", float("nan")))
        except Exception:
            s = float("nan")
        if not np.isfinite(s):
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = float(s)
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("probe_mixup_best", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        return best if best is not None else identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best


def select_by_probe_mixup_hard(
    records: Iterable[dict],
    *,
    drift_mode: str = "none",
    drift_gamma: float = 0.0,
    drift_delta: float = 0.0,
    min_improvement: float = 0.0,
) -> dict:
    """Select the best candidate using a hard-major MixUp probe score.

    This corresponds to a MixVal-style heuristic: when λ>0.5, assign the (hard)
    pseudo label of the dominant sample (implemented in the probe score).

    Candidates must have `probe_mixup_hard_best` recorded (smaller is better).
    If no candidate improves over the identity anchor, return identity.
    """

    identity: dict | None = None
    best: dict | None = None
    best_score = float("inf")

    for rec in records:
        if str(rec.get("kind", "")) == "identity":
            identity = rec

        try:
            s = float(rec.get("probe_mixup_hard_best", float("nan")))
        except Exception:
            s = float("nan")
        if not np.isfinite(s):
            continue

        drift = _safe_float(rec.get("drift_best", 0.0))
        if drift_mode == "hard" and float(drift_delta) > 0.0 and float(drift) > float(drift_delta):
            continue

        score = float(s)
        if drift_mode == "penalty" and float(drift_gamma) > 0.0:
            score = float(score) + float(drift_gamma) * float(drift)

        if score < best_score:
            best_score = float(score)
            best = rec

    if identity is None:
        return best if best is not None else next(iter(records))

    try:
        s_id = float(identity.get("probe_mixup_hard_best", float("nan")))
    except Exception:
        s_id = float("nan")
    if not np.isfinite(s_id):
        return best if best is not None else identity

    if best is None:
        return identity

    if float(min_improvement) > 0.0 and (float(s_id) - float(best_score)) < float(min_improvement):
        return identity

    if float(best_score) >= float(s_id):
        return identity

    return best
