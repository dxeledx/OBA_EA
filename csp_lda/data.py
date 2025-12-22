from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from scipy.signal import firwin, lfilter


@dataclass(frozen=True)
class SubjectData:
    subject: int
    X: np.ndarray  # shape: (n_trials, n_channels, n_times)
    y: np.ndarray  # shape: (n_trials,)


class BCIIV2aMoabbLoader:
    """Load BCI Competition IV 2a (MOABB BNCI2014-001) using MOABB Dataset+Paradigm.

    References (MOABB)
    ------------------
    - Paradigms API (`MotorImagery`): https://github.com/NeuroTechX/moabb/blob/develop/docs/source/api.rst
      (see paradigms; `MotorImagery` supports bandpass via `fmin/fmax`, epoching via
      `tmin/tmax`, and resampling via `resample`).
    - MOABB examples also demonstrate `MotorImagery(...).get_data(dataset=BNCI2014_001(), ...)`.

    References (Braindecode)
    ------------------------
    Braindecode lists MOABB as an optional dependency and notes that certain older
    MOABB versions (e.g., 1.0.0) had incorrect epoching; ensure a recent MOABB:
    https://github.com/braindecode/braindecode/blob/master/docs/whats_new.rst
    """

    def __init__(
        self,
        fmin: float,
        fmax: float,
        tmin: float,
        tmax: float,
        resample: float,
        events: Sequence[str],
        sessions: Optional[Sequence[str]] = None,
        preprocess: str = "moabb",
        paper_fir_order: int = 50,
        paper_fir_window: str = "hamming",
    ) -> None:
        self.dataset = BNCI2014_001()
        self.sessions = tuple(sessions) if sessions is not None else None
        self.preprocess = str(preprocess)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.resample = float(resample)
        self.events = tuple(events)
        self.paper_fir_order = int(paper_fir_order)
        self.paper_fir_window = str(paper_fir_window)

        if self.preprocess == "moabb":
            self.paradigm = MotorImagery(
                events=list(events),
                n_classes=len(events),
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                resample=resample,
            )
        elif self.preprocess == "paper_fir":
            # Paper-matched preprocessing is implemented manually on MOABB Raw objects.
            self.paradigm = None
        else:
            raise ValueError("preprocess must be one of: 'moabb', 'paper_fir'")

    @property
    def subject_list(self) -> List[int]:
        return list(self.dataset.subject_list)

    def load_arrays(
        self,
        subjects: Optional[Sequence[int]] = None,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Return (X, y, meta) as numpy arrays.

        - preprocess="moabb": use MOABB paradigm standard pipeline.
        - preprocess="paper_fir": use causal 50-order FIR(Hamming) bandpass, then epoch.
        """

        if subjects is None:
            subjects = self.subject_list
        subjects = list(subjects)

        if self.preprocess == "moabb":
            if self.paradigm is None:
                raise RuntimeError("MOABB paradigm is not initialized.")
            X, y, meta = self.paradigm.get_data(dataset=self.dataset, subjects=subjects)

            if self.sessions is not None:
                if "session" not in meta.columns:
                    raise ValueError("MOABB metadata must contain a 'session' column.")
                session_col = meta["session"].astype(str)
                mask = session_col.isin([str(s) for s in self.sessions]).to_numpy()
                X = X[mask]
                y = y[mask]
                meta = meta.loc[mask].reset_index(drop=True)

            X = np.asarray(X, dtype=dtype, order="C")
            y = np.asarray(y)
            return X, y, meta

        # paper_fir mode
        return self._load_arrays_paper_fir(subjects=subjects, dtype=dtype)

    def load_epochs_info(self, subject: Optional[int] = None):
        """Load one subject as MNE Epochs to obtain `info` for topographic plotting."""

        if subject is None:
            subject = self.subject_list[0]
        subject = int(subject)

        if self.preprocess == "moabb":
            if self.paradigm is None:
                raise RuntimeError("MOABB paradigm is not initialized.")
            epochs, y, meta = self.paradigm.get_data(
                dataset=self.dataset, subjects=[subject], return_epochs=True
            )
            if self.sessions is not None:
                session_col = meta["session"].astype(str)
                mask = session_col.isin([str(s) for s in self.sessions]).to_numpy()
                epochs = epochs[mask]
            return epochs.info

        # paper_fir: load one raw and keep EEG info (channel positions, etc.)
        raws = self.dataset.get_data(subjects=[subject])[subject]
        for session_name, runs in raws.items():
            if self.sessions is not None and str(session_name) not in [str(s) for s in self.sessions]:
                continue
            for _run_name, raw in runs.items():
                raw = raw.copy().pick_types(eeg=True, eog=False, stim=False, misc=False)
                return raw.info
        raise RuntimeError("Could not find matching session/run to extract MNE info.")

    def _load_arrays_paper_fir(
        self,
        *,
        subjects: Sequence[int],
        dtype: np.dtype,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Paper-matched preprocessing on Raw:

        - causal FIR (order=50, Hamming), bandpass [fmin, fmax]
        - resample to target sfreq if needed
        - epoch tmin..tmax relative to cue (annotation onset)
        - select requested events
        """

        event_id = {e: int(self.dataset.event_id[e]) for e in self.events}
        code_to_label = {v: k for k, v in event_id.items()}

        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        meta_rows: List[dict] = []

        for subject in subjects:
            sessions = self.dataset.get_data(subjects=[int(subject)])[int(subject)]
            for session_name, runs in sessions.items():
                if self.sessions is not None and str(session_name) not in [str(s) for s in self.sessions]:
                    continue

                for run_name, raw in runs.items():
                    raw = raw.copy().pick_types(eeg=True, eog=False, stim=False, misc=False)

                    # Resample first (paper uses 250Hz already for Dataset 2a).
                    if raw.info["sfreq"] != self.resample:
                        raw.resample(self.resample, npad="auto")

                    # Apply causal FIR filter to EEG channels.
                    self._apply_causal_fir(raw)

                    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
                    if len(events) == 0:
                        continue

                    epochs = mne.Epochs(
                        raw,
                        events,
                        event_id=event_id,
                        tmin=self.tmin,
                        tmax=self.tmax,
                        baseline=None,
                        preload=True,
                        on_missing="ignore",
                        verbose="ERROR",
                    )
                    if len(epochs) == 0:
                        continue

                    X_e = epochs.get_data()
                    y_e = np.asarray([code_to_label[c] for c in epochs.events[:, 2]])

                    X_parts.append(np.asarray(X_e, dtype=dtype, order="C"))
                    y_parts.append(y_e)
                    meta_rows.extend(
                        {
                            "subject": int(subject),
                            "session": str(session_name),
                            "run": str(run_name),
                        }
                        for _ in range(len(y_e))
                    )

        if not X_parts:
            raise RuntimeError("No epochs found after paper_fir preprocessing.")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        meta = pd.DataFrame(meta_rows)
        return X, y, meta

    def _apply_causal_fir(self, raw: mne.io.BaseRaw) -> None:
        """Apply causal linear-phase FIR bandpass (Hamming window) in-place."""

        sfreq = float(raw.info["sfreq"])
        numtaps = int(self.paper_fir_order) + 1  # order 50 -> 51 taps
        b = firwin(
            numtaps=numtaps,
            cutoff=[float(self.fmin), float(self.fmax)],
            pass_zero=False,
            fs=sfreq,
            window=self.paper_fir_window,
        )
        data = raw.get_data().astype(np.float64, copy=False)
        # Causal filtering (one-pass), matching Matlab `filter` behavior.
        filtered = lfilter(b, [1.0], data, axis=-1)
        raw._data[:] = filtered


def split_by_subject(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> Dict[int, SubjectData]:
    """Split MOABB-returned arrays into a dict keyed by subject id."""

    if "subject" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'subject' column.")

    out: Dict[int, SubjectData] = {}
    for subject in sorted(meta["subject"].unique()):
        mask = meta["subject"].to_numpy() == subject
        out[int(subject)] = SubjectData(subject=int(subject), X=X[mask], y=y[mask])
    return out
