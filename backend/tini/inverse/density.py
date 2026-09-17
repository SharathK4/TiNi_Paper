"""Non-parametric estimates of the transition-force distribution.

These describe the data without the model: a histogram with Poisson error
bars, a Gaussian kernel density estimate, and the empirical CDF. Two
estimators are given on purpose so the reader can see how much of the
"shape" is the estimator's choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def freedman_diaconis_bins(x: np.ndarray, min_bins: int = 6, max_bins: int = 40) -> int:
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min_bins
    h = 2.0 * iqr / x.size ** (1.0 / 3.0)
    return int(np.clip(np.ceil((x.max() - x.min()) / h), min_bins, max_bins))


def histogram_density(x: np.ndarray, bins: int | None = None) -> dict[str, Any]:
    """Density histogram (area 1) with Poisson standard errors per bin."""
    nb = bins or freedman_diaconis_bins(x)
    counts, edges = np.histogram(x, bins=nb)
    width = np.diff(edges)
    n = x.size
    dens = counts / (n * width)
    err = np.sqrt(np.maximum(counts, 1)) / (n * width)   # sqrt(k), floor 1 for empty bins
    return {
        "bins": int(nb),
        "rule": "Freedman–Diaconis" if bins is None else "fixed",
        "edges": edges.tolist(),
        "centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "counts": counts.tolist(),
        "density": dens.tolist(),
        "density_err": err.tolist(),
    }


def silverman_bandwidth(x: np.ndarray) -> float:
    n = x.size
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    return float(0.9 * a * n ** (-0.2))


def kde(x: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> dict[str, Any]:
    """Gaussian KDE evaluated on ``grid``; bandwidth by Silverman's rule unless given."""
    h = bandwidth or silverman_bandwidth(x)
    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (x.size * h * np.sqrt(2 * np.pi))
    return {"bandwidth_pN": float(h), "rule": "Silverman" if bandwidth is None else "fixed", "grid": grid.tolist(), "density": dens.tolist()}


def ecdf(x: np.ndarray) -> dict[str, list[float]]:
    xs = np.sort(x)
    return {"x": xs.tolist(), "F": (np.arange(1, xs.size + 1) / xs.size).tolist()}
