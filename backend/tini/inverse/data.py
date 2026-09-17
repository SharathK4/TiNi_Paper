"""Transition-event sample preparation (inverse-problem input).

The sample is the set of critical forces {f‡ᵢ} at one force rate ḟ. Both
come from the V1 pipeline outputs (``traces.npz``) re-processed with the
V1 statistics functions, so the V1 prototype assumptions (ḟ fit window,
peak-picker smoothing) are explicit inputs here and can be varied.

Two definitions of f‡ are supported:
    "rate_x_time"     f‡ = ḟ · t‡           the paper's definition (Methods)
    "smoothed_peak"   f‡ = smoothed peak     secondary, for sensitivity only
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import statistics as st
from ..config import Settings

SOURCES = ("rate_x_time", "smoothed_peak")


@dataclass
class EventSample:
    forces: np.ndarray             # f‡ᵢ [pN], one entry per trace
    force_rate: float              # ḟ [pN/ps]
    force_rate_se: float
    source: str
    smoothing_points: int
    window_ps: float
    trace_ids: list[str]
    t_peak_ps: np.ndarray
    n_traces_total: int
    dataset_fingerprint: str       # hash of traces.npz content
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.forces.size)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.dataset_fingerprint.encode())
        h.update(f"{self.source}|{self.smoothing_points}|{self.window_ps}".encode())
        h.update(np.round(self.forces, 9).tobytes())
        return h.hexdigest()[:16]

    def summary(self) -> dict[str, Any]:
        f = self.forces
        return {
            "n": self.n,
            "source": self.source,
            "smoothing_points": self.smoothing_points,
            "window_ps": self.window_ps,
            "force_rate_pN_per_ps": self.force_rate,
            "force_rate_se": self.force_rate_se,
            "mean_pN": float(f.mean()),
            "sd_pN": float(f.std(ddof=1)) if self.n > 1 else 0.0,
            "sem_pN": float(f.std(ddof=1) / np.sqrt(self.n)) if self.n > 1 else 0.0,
            "median_pN": float(np.median(f)),
            "min_pN": float(f.min()),
            "max_pN": float(f.max()),
            "skewness": float(((f - f.mean()) ** 3).mean() / f.std() ** 3) if self.n > 2 else 0.0,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_fingerprint": self.fingerprint(),
            "notes": self.notes,
        }


def traces_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_traces(settings: Settings) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    p = settings.artifacts_path / "traces.npz"
    if not p.exists():
        raise FileNotFoundError("traces.npz not found — run the Version 1 pipeline first")
    z = np.load(p)
    return z["traces"], z["time_ps"], [str(s) for s in z["trace_ids"]], traces_fingerprint(p)


def prepare_sample(
    settings: Settings,
    source: str = "rate_x_time",
    smoothing_points: int | None = None,
    window_ps: float | None = None,
) -> EventSample:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    traces, t, ids, fp = load_traces(settings)
    sm = smoothing_points or settings.peak_smoothing_points
    w = window_ps or settings.force_rate_window_ps

    avg = st.ensemble_average(traces, settings.percentiles)
    fit = st.force_rate(t, avg["mean_f"], w)
    fdot = fit["slope"]
    ev = st.transition_events(t, traces, sm, fdot)
    t_peak = np.array([e["t_peak_ps"] for e in ev["events"]])
    if source == "rate_x_time":
        forces = fdot * t_peak
    else:
        forces = np.array([e["f_peak_smoothed_pN"] for e in ev["events"]])

    notes = [
        "f‡ = ḟ·t‡ with t‡ from the V1 prototype peak picker (centred moving average, argmax)" if source == "rate_x_time"
        else "f‡ = peak of the smoothed trace; not the paper's definition, used for sensitivity only",
        f"ḟ from a linear fit of ⟨f(t)⟩ on t ≤ {w:g} ps (V1 prototype window)",
    ]
    return EventSample(
        forces=forces.astype(float),
        force_rate=float(fdot),
        force_rate_se=float(fit["slope_se"]),
        source=source,
        smoothing_points=int(sm),
        window_ps=float(w),
        trace_ids=ids,
        t_peak_ps=t_peak,
        n_traces_total=int(traces.shape[0]),
        dataset_fingerprint=fp,
        notes=notes,
    )
