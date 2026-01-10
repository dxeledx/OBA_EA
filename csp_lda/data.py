from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import re
from moabb.paradigms import MotorImagery
from scipy.signal import firwin, lfilter


@dataclass(frozen=True)
class SubjectData:
    subject: int
    X: np.ndarray  # shape: (n_trials, n_channels, n_times)
    y: np.ndarray  # shape: (n_trials,)

def resolve_moabb_dataset(dataset: str):
    """Resolve a MOABB dataset name to a dataset instance.

    Accepts common aliases/casings, e.g.:
    - BNCI2014_001 / bnci2014_001 / BNCI2014001
    - Cho2017 / PhysionetMI / Schirrmeister2017
    """

    dataset = str(dataset).strip()
    if not dataset:
        raise ValueError("dataset must be a non-empty string.")

    import moabb.datasets as moabb_datasets

    # Fast path: exact match.
    if hasattr(moabb_datasets, dataset):
        cls = getattr(moabb_datasets, dataset)
        return cls()

    # Normalize BNCI variants: BNCI2014001 -> BNCI2014_001 when available.
    m = re.match(r"^BNCI(\d{4})_?(\d{3})$", dataset.strip().upper())
    if m:
        with_underscore = f"BNCI{m.group(1)}_{m.group(2)}"
        without_underscore = f"BNCI{m.group(1)}{m.group(2)}"
        if hasattr(moabb_datasets, with_underscore):
            return getattr(moabb_datasets, with_underscore)()
        if hasattr(moabb_datasets, without_underscore):
            return getattr(moabb_datasets, without_underscore)()

    # Case-insensitive match for other datasets.
    lower_to_name = {name.lower(): name for name in dir(moabb_datasets)}
    key = dataset.lower()
    if key in lower_to_name:
        cls = getattr(moabb_datasets, lower_to_name[key])
        return cls()

    raise ValueError(f"Unknown MOABB dataset: {dataset}")


class MoabbMotorImageryLoader:
    """Load a MOABB MotorImagery dataset using MOABB Dataset+Paradigm."""

    def __init__(
        self,
        *,
        dataset: str,
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
        self.dataset = resolve_moabb_dataset(dataset)
        self.dataset_id = str(self.dataset.__class__.__name__)
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
        - preprocess="paper_fir": use causal FIR(Hamming) bandpass, then epoch.
        """

        if subjects is None:
            subjects = self.subject_list
        subjects = list(subjects)

        if self.preprocess == "moabb":
            if self.paradigm is None:
                raise RuntimeError("MOABB paradigm is not initialized.")
            # IMPORTANT (memory): for large datasets (e.g., Schirrmeister2017 / HGD),
            # calling `get_data` for all subjects at once can transiently hold many Raw/Epochs
            # objects in memory (filtering/resampling/copying), leading to OOM kills.
            # We therefore load *per subject* and concatenate at the end.
            X_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            meta_parts: List[pd.DataFrame] = []

            for subject in subjects:
                X_s, y_s, meta_s = self.paradigm.get_data(dataset=self.dataset, subjects=[int(subject)])

                if self.sessions is not None:
                    # Most MOABB MI datasets expose a `session` column. Some (e.g., Schirrmeister2017)
                    # also expose meaningful splits in `run` (e.g., 0train/1test) while `session`
                    # stays constant. We therefore allow filtering by *either* column.
                    allowed = [str(s) for s in self.sessions]
                    mask = None
                    if "session" in meta_s.columns:
                        session_col = meta_s["session"].astype(str)
                        mask = session_col.isin(allowed).to_numpy()
                    if "run" in meta_s.columns:
                        run_col = meta_s["run"].astype(str)
                        mask_run = run_col.isin(allowed).to_numpy()
                        mask = mask_run if mask is None else (mask | mask_run)
                    if mask is None:
                        raise ValueError("MOABB metadata must contain a 'session' or 'run' column for filtering.")
                    if not bool(np.any(mask)):
                        continue
                    X_s = X_s[mask]
                    y_s = y_s[mask]
                    meta_s = meta_s.loc[mask].reset_index(drop=True)

                X_parts.append(np.asarray(X_s, dtype=dtype, order="C"))
                y_parts.append(np.asarray(y_s))
                meta_parts.append(meta_s)

            if not X_parts:
                raise RuntimeError("No trials found after MOABB preprocessing (empty subject list or session filter?).")

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            meta = pd.concat(meta_parts, axis=0, ignore_index=True)
            return np.asarray(X, dtype=dtype, order="C"), np.asarray(y), meta

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
            epochs, _y, meta = self.paradigm.get_data(
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
        """Paper-matched preprocessing on Raw.

        - causal FIR (order=`paper_fir_order`, Hamming by default), bandpass [fmin, fmax]
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

                    if raw.info["sfreq"] != self.resample:
                        raw.resample(self.resample, npad="auto")

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


class BCIIV2aMoabbLoader(MoabbMotorImageryLoader):
    """Backward-compatible loader for MOABB BNCI2014_001 (BCI Competition IV 2a)."""

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
        super().__init__(
            dataset="BNCI2014_001",
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            resample=resample,
            events=events,
            sessions=sessions,
            preprocess=preprocess,
            paper_fir_order=paper_fir_order,
            paper_fir_window=paper_fir_window,
        )


def split_by_subject(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> Dict[int, SubjectData]:
    """Split MOABB-returned arrays into a dict keyed by subject id."""

    if "subject" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'subject' column.")

    out: Dict[int, SubjectData] = {}
    for subject in sorted(meta["subject"].unique()):
        mask = meta["subject"].to_numpy() == subject
        out[int(subject)] = SubjectData(subject=int(subject), X=X[mask], y=y[mask])
    return out


def split_by_subject_session(
    X: np.ndarray, y: np.ndarray, meta: pd.DataFrame
) -> Dict[int, Dict[str, SubjectData]]:
    """Split MOABB-returned arrays into {subject -> {session -> SubjectData}}."""

    if "subject" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'subject' column.")
    if "session" not in meta.columns:
        raise ValueError("MOABB metadata must contain a 'session' column.")

    out: Dict[int, Dict[str, SubjectData]] = {}
    subj_col = meta["subject"].to_numpy()
    sess_col = meta["session"].astype(str).to_numpy()

    for subject in sorted(meta["subject"].unique()):
        subject = int(subject)
        out[subject] = {}
        subj_mask = subj_col == subject
        sessions = sorted(pd.Series(sess_col[subj_mask]).unique().tolist())
        for session in sessions:
            sess_mask = sess_col == str(session)
            mask = subj_mask & sess_mask
            out[subject][str(session)] = SubjectData(subject=subject, X=X[mask], y=y[mask])
    return out
