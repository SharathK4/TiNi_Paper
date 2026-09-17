"""Uncertainty of the fitted parameters, three ways.

1. Observed information: finite-difference Hessian of NLL(θ) at the optimum,
   θ = ln w. Cov = H⁻¹ (asymptotic). Its condition number and eigenvectors
   are reported because a near-singular H *is* the single-rate degeneracy.
2. Profile likelihood: for each free parameter, scan a grid and re-optimise
   the others; the 95 % interval is where 2·ΔNLL ≤ χ²₁(0.95) = 3.841.
   An interval that reaches the bounds is reported as unbounded on that side.
3. Non-parametric bootstrap: resample the events with replacement, refit,
   report percentile intervals and the parameter scatter (seeded).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from .fit import FitOptions, FitResult
from .likelihood import Bounds, Likelihood

CHI2_1_95 = 3.841458820694124


# ------------------------------------------------------------------ Hessian
def numerical_hessian(fun: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian in θ; h is in log-parameter units."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    f0 = fun(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        H[i, i] = (fun(x + ei) - 2 * f0 + fun(x - ei)) / h**2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h
            H[i, j] = H[j, i] = (fun(x + ei + ej) - fun(x + ei - ej) - fun(x - ei + ej) + fun(x - ei - ej)) / (4 * h**2)
    return H


def observed_information(lik: Likelihood, fit: FitResult, h: float = 1e-3) -> dict[str, Any]:
    H = numerical_hessian(lik.nll, fit.theta, h)
    out: dict[str, Any] = {"hessian_theta": H.tolist(), "step_h": h, "free": fit.free}
    try:
        eig, vec = np.linalg.eigh(H)
        out["eigenvalues"] = eig.tolist()
        out["eigenvectors"] = vec.T.tolist()   # rows = eigenvectors in θ space
        out["positive_definite"] = bool(np.all(eig > 0))
        out["condition_number"] = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
        if np.all(eig > 0):
            cov = np.linalg.inv(H)
            se_log = np.sqrt(np.diag(cov))
            corr = cov / np.outer(se_log, se_log)
            out["cov_theta"] = cov.tolist()
            out["correlation"] = corr.tolist()
            out["se_log"] = {k: float(s) for k, s in zip(fit.free, se_log)}
            # delta method: s.e.(w) = w · s.e.(ln w)
            out["se"] = {k: float(fit.params[k] * s) for k, s in zip(fit.free, se_log)}
            out["ci95_asymptotic"] = {k: [float(fit.params[k] * np.exp(-1.96 * s)), float(fit.params[k] * np.exp(1.96 * s))] for k, s in zip(fit.free, se_log)}
        else:
            out["note"] = "Hessian not positive definite at the optimum: asymptotic errors undefined (flat or saddle direction)."
    except np.linalg.LinAlgError as exc:
        out["error"] = str(exc)
    return out


# ------------------------------------------------------------------ profiles
def profile_likelihood(
    lik: Likelihood,
    fit: FitResult,
    n_points: int = 21,
    span_factor: float | dict[str, float] = 4.0,
    opts: FitOptions | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """For each free parameter k: profile NLL_k(v) = min over the others of NLL,
    on a log grid v ∈ [v̂/span, v̂·span] clipped to the bounds."""
    opts = opts or FitOptions(maxiter=1500, maxfev=3000, polish=False)
    profiles: dict[str, Any] = {}
    for k in fit.free:
        if cancel and cancel():
            break
        lo_b, hi_b = getattr(lik.bounds, k)
        v_hat = fit.params[k]
        span = span_factor[k] if isinstance(span_factor, dict) else span_factor
        lo = max(lo_b, v_hat / span)
        hi = min(hi_b, v_hat * span)
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_points))
        # make sure the optimum itself is on the grid
        grid = np.sort(np.append(grid, v_hat))
        others = [p for p in fit.free if p != k]
        rows = []
        t0 = time.time()
        for gi, v in enumerate(grid):
            fixed = {**lik.fixed, k: float(v)}
            sub = Likelihood(lik.groups, lik.bounds, fixed, lik.kBT)
            if others:
                theta0 = sub.params_to_theta({**fixed, **{p: fit.params[p] for p in others}})
                res = minimize(sub.nll, theta0, method="Nelder-Mead", options={"maxiter": opts.maxiter, "maxfev": opts.maxfev, "xatol": opts.xatol, "fatol": opts.fatol, "adaptive": True})
                nll = float(res.fun)
                p_other = sub.theta_to_params(res.x)
            else:
                nll = sub.nll(np.array([]))
                p_other = {}
            feasible = bool(nll < 1e5)   # penalty values mean f_c ≤ max f‡ or out of bounds
            rows.append({"value": float(v), "nll": nll if feasible else None, "delta_nll": (nll - fit.nll) if feasible else None, "feasible": feasible, **{p: p_other[p] for p in others}})
            if progress:
                progress(k, gi + 1, len(grid))
        # likelihood-ratio interval: contiguous region around v̂ with 2ΔNLL ≤ 3.84
        vals = np.array([r["value"] for r in rows])
        d2 = 2 * np.array([r["delta_nll"] if r["feasible"] else np.inf for r in rows])
        i_hat = int(np.argmin(np.abs(vals - v_hat)))
        lo_i, hi_i = i_hat, i_hat
        while lo_i > 0 and d2[lo_i - 1] <= CHI2_1_95:
            lo_i -= 1
        while hi_i < len(vals) - 1 and d2[hi_i + 1] <= CHI2_1_95:
            hi_i += 1
        lower = _interp_crossing(vals, d2, lo_i, -1) if lo_i > 0 else None
        upper = _interp_crossing(vals, d2, hi_i, +1) if hi_i < len(vals) - 1 else None
        profiles[k] = {
            "grid": rows,
            "mle": v_hat,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "lower_unbounded_in_scan": lower is None,
            "upper_unbounded_in_scan": upper is None,
            "lower_support_limited": bool(lo_i > 0 and not rows[lo_i - 1]["feasible"]),
            "upper_support_limited": bool(hi_i < len(vals) - 1 and not rows[hi_i + 1]["feasible"]),
            "scan_range": [float(grid[0]), float(grid[-1])],
            "duration_s": round(time.time() - t0, 3),
        }
    return {"chi2_1_95": CHI2_1_95, "n_points": n_points, "span_factor": span_factor, "profiles": profiles}


def _interp_crossing(vals: np.ndarray, d2: np.ndarray, i: int, direction: int) -> float:
    """Linear interpolation (in ln v) of where 2ΔNLL crosses the threshold between i and i+direction."""
    j = i + direction
    x0, x1 = np.log(vals[i]), np.log(vals[j])
    y0, y1 = d2[i], d2[j]
    if y1 == y0:
        return float(vals[j])
    t = (CHI2_1_95 - y0) / (y1 - y0)
    return float(np.exp(x0 + t * (x1 - x0)))


# ------------------------------------------------------------------ bootstrap
def bootstrap(
    lik: Likelihood,
    fit: FitResult,
    n_boot: int = 200,
    seed: int = 0,
    opts: FitOptions | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    opts = opts or FitOptions(maxiter=1500, maxfev=3000, polish=False)
    rng = np.random.default_rng(seed)
    theta0 = fit.theta
    draws: list[dict[str, float]] = []
    n_fail = 0
    t0 = time.time()
    for b in range(n_boot):
        if cancel and cancel():
            break
        groups = [(rng.choice(f, size=f.size, replace=True), r) for f, r in lik.groups]
        sub = Likelihood(groups, lik.bounds, lik.fixed, lik.kBT)
        res = minimize(sub.nll, theta0, method="Nelder-Mead", options={"maxiter": opts.maxiter, "maxfev": opts.maxfev, "xatol": opts.xatol, "fatol": opts.fatol, "adaptive": True})
        if np.isfinite(res.fun) and res.fun < 1e5:
            draws.append({**sub.theta_to_params(res.x), "nll": float(res.fun)})
        else:
            n_fail += 1
        if progress:
            progress(b + 1, n_boot)
    out: dict[str, Any] = {"n_boot": n_boot, "n_ok": len(draws), "n_failed": n_fail, "seed": seed, "duration_s": round(time.time() - t0, 3), "draws": draws}
    if draws:
        for k in fit.free:
            v = np.array([d[k] for d in draws])
            out[k] = {
                "median": float(np.median(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "ci68": [float(np.percentile(v, 16)), float(np.percentile(v, 84))],
                "geometric_sd": float(np.exp(np.std(np.log(v)))),
                "hit_lower_bound": float(np.mean(v <= getattr(lik.bounds, k)[0] * 1.001)),
                "hit_upper_bound": float(np.mean(v >= getattr(lik.bounds, k)[1] * 0.999)),
            }
        # derived: f_c per draw
        fc = np.array([3 * d["U_barrier"] / (2 * d["xi_barrier"]) for d in draws])
        out["f_c_pN"] = {"median": float(np.median(fc)), "ci95": [float(np.percentile(fc, 2.5)), float(np.percentile(fc, 97.5))]}
        if len(fit.free) >= 2:
            L = np.array([[np.log(d[k]) for k in fit.free] for d in draws])
            c = np.corrcoef(L, rowvar=False)
            out["log_correlation"] = c.tolist()
    return out


__all__ = ["observed_information", "profile_likelihood", "bootstrap", "numerical_hessian", "CHI2_1_95", "Bounds"]
