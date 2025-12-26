from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mne.decoding import CSP
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from .alignment import BaseAligner, NoAligner
from .subject_invariant import CenteredLinearProjector


class EnsureFloat64(BaseEstimator, TransformerMixin):
    """Cast EEG epochs array to float64 for MNE decoding utilities."""

    def fit(self, X, y=None):  # noqa: N803  (match sklearn signature)
        return self

    def transform(self, X):  # noqa: N803  (match sklearn signature)
        return np.asarray(X, dtype=np.float64)


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Pipeline

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self):
        return self.pipeline.named_steps["lda"].classes_

    @property
    def csp(self) -> CSP:
        return self.pipeline.named_steps["csp"]


def build_csp_lda_pipeline(
    n_components: int = 4,
    aligner: Optional[BaseAligner] = None,
) -> Pipeline:
    """Build CSP+LDA sklearn pipeline.

    Notes
    -----
    - CSP uses `mne.decoding.CSP` (classical CSP implementation used in many
      motor-imagery baselines). We set `n_components=4` per requirement.
    - LDA uses scikit-learn default parameters.
    """

    if aligner is None:
        aligner = NoAligner()

    return Pipeline(
        steps=[
            ("to_float64", EnsureFloat64()),
            ("align", aligner),
            ("csp", CSP(n_components=int(n_components))),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )


def fit_csp_lda(
    X_train,
    y_train,
    n_components: int = 4,
    aligner: Optional[BaseAligner] = None,
) -> TrainedModel:
    pipeline = build_csp_lda_pipeline(n_components=n_components, aligner=aligner)
    pipeline.fit(X_train, y_train)
    return TrainedModel(pipeline=pipeline)


def fit_csp_projected_lda(
    *,
    X_train,
    y_train,
    projector: CenteredLinearProjector,
    csp: CSP | None = None,
    n_components: int = 4,
    aligner: Optional[BaseAligner] = None,
) -> TrainedModel:
    """Fit CSP then train LDA on projected CSP features.

    This is used for subject-invariant feature learning where the projector is
    learned externally (may depend on subject IDs), but we still want a standard
    sklearn Pipeline for inference and for ZO utilities that access pipeline steps.
    """

    if aligner is None:
        aligner = NoAligner()

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    if csp is None:
        csp = CSP(n_components=int(n_components))
        csp.fit(X_train, y_train)

    feats = np.asarray(csp.transform(X_train), dtype=np.float64)
    feats_proj = np.asarray(projector.transform(feats), dtype=np.float64)

    lda = LinearDiscriminantAnalysis()
    lda.fit(feats_proj, y_train)

    pipeline = Pipeline(
        steps=[
            ("to_float64", EnsureFloat64()),
            ("align", aligner),
            ("csp", csp),
            ("proj", projector),
            ("lda", lda),
        ]
    )
    return TrainedModel(pipeline=pipeline)
