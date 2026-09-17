"""Model–data comparison and goodness of fit.

Everything here compares an observed sample with a *given* parameter set —
the fitted one, or the paper's Table 1 values — so the same functions serve
for "how well does the fit describe the data" and "how well do the paper's
parameters describe the data".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .density import histogram_density
from .model import LandscapeKinetics


def compare(land: LandscapeKinetics, forces: np.ndarray, force_rate: float, n_params: int, bins: int | None = None, n_grid: int = 400) -> dict[str, Any]:
    f = np.asarray(forces, dtype=float)
    n = f.size
    lp = land.log_pdf(f, force_rate)
    in_support = bool(np.all(np.isfinite(lp)))
    ll = float(lp.sum()) if in_support else float("-inf")

    # PDF / CDF curves
    grid = np.linspace(0.0, land.f_c, n_grid, endpoint=False)
    pdf = land.pdf(grid, force_rate)
    cdf = land.cdf(grid, force_rate)

    # KS against the model CDF (continuous part; residual mass is negligible for all data-compatible fits)
    ks = stats.kstest(f, lambda x: land.cdf(np.asarray(x), force_rate))

    # Pearson chi-square on the density histogram, bins merged so expected ≥ 5
    hist = histogram_density(f, bins)
    edges = np.array(hist["edges"])
    obs = np.array(hist["counts"], dtype=float)
    exp = n * np.diff(land.cdf(edges, force_rate))
    # tails: add mass outside the histogram range to the end bins
    exp[0] += n * land.cdf(np.array([edges[0]]), force_rate)[0]
    exp[-1] += n * (1.0 - land.cdf(np.array([edges[-1]]), force_rate)[0])
    o_m, e_m = _merge_bins(obs, exp, 5.0)
    dof = max(1, len(o_m) - 1 - n_params)
    chi2 = float(np.sum((o_m - e_m) ** 2 / e_m)) if len(e_m) else float("nan")
    chi2_p = float(stats.chi2.sf(chi2, dof)) if np.isfinite(chi2) else float("nan")

    # per-bin residuals in Poisson units
    resid = (obs - exp) / np.sqrt(np.maximum(exp, 1e-9))

    # QQ: model quantiles at plotting positions
    pp = (np.arange(1, n + 1) - 0.5) / n
    q_model = land.quantile(pp, force_rate)
    q_data = np.sort(f)

    mom = land.moments_exact(force_rate)
    return {
        "in_support": in_support,
        "log_likelihood": ll,
        "nll": -ll,
        "aic": 2 * n_params - 2 * ll if in_support else float("inf"),
        "bic": n_params * np.log(n) - 2 * ll if in_support else float("inf"),
        "n": int(n),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "chi2": chi2,
        "chi2_dof": int(dof),
        "chi2_pvalue": chi2_p,
        "chi2_bins_merged": int(len(o_m)),
        "histogram": hist,
        "expected_counts": exp.tolist(),
        "residuals_poisson": resid.tolist(),
        "pdf_grid": grid.tolist(),
        "pdf": pdf.tolist(),
        "cdf": cdf.tolist(),
        "qq": {"model": q_model.tolist(), "data": q_data.tolist()},
        "model_mean_exact_pN": mom["mean"],
        "model_sd_exact_pN": mom["sd"],
        "model_mean_eq8_pN": land.mean_eq8(force_rate),
        "model_mode_pN": land.mode(force_rate),
        "residual_mass": mom["residual_mass"],
        "sample_mean_pN": float(f.mean()),
        "sample_sd_pN": float(f.std(ddof=1)) if n > 1 else 0.0,
    }


def _merge_bins(obs: np.ndarray, exp: np.ndarray, min_expected: float) -> tuple[np.ndarray, np.ndarray]:
    o_out, e_out = [], []
    o_acc, e_acc = 0.0, 0.0
    for o, e in zip(obs, exp):
        o_acc += o
        e_acc += e
        if e_acc >= min_expected:
            o_out.append(o_acc)
            e_out.append(e_acc)
            o_acc, e_acc = 0.0, 0.0
    if e_acc > 0:
        if e_out:
            o_out[-1] += o_acc
            e_out[-1] += e_acc
        else:
            o_out.append(o_acc)
            e_out.append(e_acc)
    return np.array(o_out), np.array(e_out)


def likelihood_ratio(nll_restricted: float, nll_full: float, df: int) -> dict[str, float]:
    """LR test of a restricted parameter set (e.g. the paper's) against the fit."""
    stat = 2.0 * (nll_restricted - nll_full)
    return {"statistic": float(stat), "df": int(df), "pvalue": float(stats.chi2.sf(max(stat, 0.0), df))}
