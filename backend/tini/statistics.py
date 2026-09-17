"""Force statistics over the ensemble of traces.

Inputs are the stacked array produced by the preprocessing stage,
``traces[n_traces, n_samples, 3]``, plus the shared time grid. Every function
is pure so it can be re-run for a different window or smoothing width from
the API without touching the checkpoints.

Verified (paper, Methods "Critical forces and normalised histograms"):
  * ⟨f(t)⟩ is the mean over traces at fixed t                     (Eq. 2)
  * ḟ is the gradient of ⟨f(t)⟩ "in a small time interval near t = 0"
  * critical force f‡ = ḟ · t‡, with t‡ the time of the peak force
    before the transition

Prototype assumptions (not specified by the paper):
  * the length of the "small time interval" (default 100 ps, see Settings)
  * the peak picker: a centred moving average before argmax
"""

from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------- averaging
def ensemble_average(traces: np.ndarray, percentiles: tuple[float, ...]) -> dict[str, np.ndarray]:
    """Column-wise statistics over traces. Requires a shared time grid."""
    f = traces[:, :, 2]
    x = traces[:, :, 1]
    n = f.shape[0]
    out = {
        "mean_f": f.mean(axis=0),
        "std_f": f.std(axis=0, ddof=1) if n > 1 else np.zeros(f.shape[1]),
        "median_f": np.median(f, axis=0),
        "mean_x": x.mean(axis=0),
        "std_x": x.std(axis=0, ddof=1) if n > 1 else np.zeros(x.shape[1]),
    }
    out["sem_f"] = out["std_f"] / np.sqrt(n)
    for p in percentiles:
        out[f"p{p:g}_f"] = np.percentile(f, p, axis=0)
    return out


def moving_average(y: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average with edge shrinkage (no phantom values)."""
    width = max(1, int(width) | 1)  # force odd
    if width == 1:
        return y.astype(np.float64)
    k = np.ones(width)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y, dtype=np.float64), k, mode="same")
    return num / den


# --------------------------------------------------------------- force rate
def linear_fit(t: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n = len(t)
    # standard error of the slope (ordinary least squares)
    se = float(np.sqrt(ss_res / (n - 2) / np.sum((t - t.mean()) ** 2))) if n > 2 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "slope_se": se,
        "n_points": int(n),
    }


def force_rate(t: np.ndarray, mean_f: np.ndarray, window_ps: float) -> dict[str, Any]:
    """ḟ = ∂⟨f(t)⟩/∂t from a linear fit of the mean trace on [0, window]."""
    m = t <= window_ps
    if m.sum() < 3:
        raise ValueError("window too short for a linear fit")
    fit = linear_fit(t[m], mean_f[m])
    return {"window_ps": float(window_ps), **fit, "t_fit": t[m], "f_fit": fit["slope"] * t[m] + fit["intercept"]}


def force_rate_sensitivity(t: np.ndarray, mean_f: np.ndarray, windows: tuple[float, ...]) -> list[dict[str, float]]:
    rows = []
    for w in windows:
        fit = force_rate(t, mean_f, w)
        rows.append({k: fit[k] for k in ("window_ps", "slope", "slope_se", "r2", "n_points")})
    return rows


def per_trace_rates(t: np.ndarray, traces: np.ndarray, window_ps: float) -> np.ndarray:
    m = t <= window_ps
    return np.array([np.polyfit(t[m], tr[m, 2], 1)[0] for tr in traces])


# --------------------------------------------------------------- peaks (prototype)
def detect_peak(t: np.ndarray, f: np.ndarray, smoothing_points: int) -> dict[str, float]:
    """Peak of the smoothed trace = transition time t‡ (prototype picker)."""
    fs = moving_average(f, smoothing_points)
    i = int(np.argmax(fs))
    half = max(1, smoothing_points // 2)
    lo, hi = max(0, i - half), min(len(f), i + half + 1)
    j = lo + int(np.argmax(f[lo:hi]))
    # post-transition trough: minimum of the smoothed trace after the peak
    # (the twinned lattice is briefly compressed by the box, then reloads)
    k = i + int(np.argmin(fs[i:]))
    return {
        "t_peak_ps": float(t[i]),
        "f_peak_smoothed_pN": float(fs[i]),
        "f_peak_raw_pN": float(f[j]),
        "t_raw_peak_ps": float(t[j]),
        "t_trough_ps": float(t[k]),
        "f_trough_pN": float(fs[k]),
        "drop_pN": float(fs[i] - fs[k]),
    }


def transition_events(t: np.ndarray, traces: np.ndarray, smoothing_points: int, fdot: float) -> dict[str, Any]:
    rows = []
    for k in range(traces.shape[0]):
        r = detect_peak(t, traces[k, :, 2], smoothing_points)
        r["f_crit_from_rate_pN"] = fdot * r["t_peak_ps"]   # paper's f‡ = ḟ · t‡
        r["x_peak_A"] = float(traces[k, int(np.argmin(np.abs(t - r["t_peak_ps"]))), 1])
        rows.append(r)
    t_peak = np.array([r["t_peak_ps"] for r in rows])
    f_rate = np.array([r["f_crit_from_rate_pN"] for r in rows])
    f_sm = np.array([r["f_peak_smoothed_pN"] for r in rows])
    f_raw = np.array([r["f_peak_raw_pN"] for r in rows])

    def summ(a: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "min": float(a.min()),
            "max": float(a.max()),
            "median": float(np.median(a)),
        }

    return {
        "smoothing_points": int(smoothing_points),
        "force_rate_used_pN_per_ps": float(fdot),
        "n": len(rows),
        "t_peak_ps": summ(t_peak),
        "f_crit_from_rate_pN": summ(f_rate),
        "f_peak_smoothed_pN": summ(f_sm),
        "f_peak_raw_pN": summ(f_raw),
        "events": rows,
    }


def histogram(values: np.ndarray, bins: int = 15) -> dict[str, list[float]]:
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}
