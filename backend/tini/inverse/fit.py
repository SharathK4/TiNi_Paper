"""Maximum-likelihood fitting: multi-start Nelder–Mead, then a bounded polish.

Nelder–Mead is the paper's optimiser (Methods, "Estimation of parameterised
landscape"). It is derivative-free, so it tolerates the penalty walls in
``Likelihood.nll``; the L-BFGS-B polish afterwards tightens the optimum
inside the bounds using finite-difference gradients. Every start, every
accepted best point and the final status are recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from .likelihood import PARAM_NAMES, Likelihood


@dataclass
class FitOptions:
    maxiter: int = 4000
    maxfev: int = 8000
    xatol: float = 1e-6        # simplex size tolerance in θ = ln w
    fatol: float = 1e-8        # NLL tolerance
    polish: bool = True
    polish_maxiter: int = 200
    record_every: int = 1      # history thinning

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class StartResult:
    start: dict[str, float]
    params: dict[str, float]
    nll: float
    n_eval: int
    n_iter: int
    converged: bool
    message: str
    duration_s: float


@dataclass
class FitResult:
    params: dict[str, float]
    theta: np.ndarray
    nll: float
    free: list[str]
    fixed: dict[str, float]
    starts: list[StartResult]
    best_start_index: int
    history: list[dict[str, float]]          # of the best start: iteration, nll, params
    polish: dict[str, Any]
    n_eval_total: int
    converged: bool
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "theta": self.theta.tolist(),
            "nll": self.nll,
            "log_likelihood": -self.nll,
            "free": self.free,
            "fixed": self.fixed,
            "starts": [s.__dict__ for s in self.starts],
            "best_start_index": self.best_start_index,
            "history": self.history,
            "polish": self.polish,
            "n_eval_total": self.n_eval_total,
            "converged": self.converged,
            "duration_s": self.duration_s,
        }


def default_starts(reference: dict[str, float], free: list[str], factors: tuple[float, ...] = (1.0, 0.5, 2.0)) -> list[dict[str, float]]:
    """Deterministic multi-start set: the reference point, then each free
    parameter scaled by the factors one at a time (2·n_free + 1 starts)."""
    starts = [dict(reference)]
    for k in free:
        for fac in factors:
            if fac == 1.0:
                continue
            s = dict(reference)
            s[k] = reference[k] * fac
            starts.append(s)
    return starts


def fit_mle(
    lik: Likelihood,
    starts: list[dict[str, float]],
    opts: FitOptions | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> FitResult:
    opts = opts or FitOptions()
    t0 = time.time()
    results: list[StartResult] = []
    histories: list[list[dict[str, float]]] = []
    n0 = lik.n_eval

    for si, start in enumerate(starts):
        if cancel and cancel():
            break
        theta0 = lik.params_to_theta({**lik.fixed, **start})
        hist: list[dict[str, float]] = []
        it = {"i": 0}

        def cb(xk: np.ndarray) -> None:
            it["i"] += 1
            if it["i"] % opts.record_every == 0:
                p = lik.theta_to_params(xk)
                hist.append({"iteration": it["i"], "nll": lik.nll(xk), **p})

        e0 = lik.n_eval
        ts = time.time()
        res = minimize(
            lik.nll, theta0, method="Nelder-Mead", callback=cb,
            options={"maxiter": opts.maxiter, "maxfev": opts.maxfev, "xatol": opts.xatol, "fatol": opts.fatol, "adaptive": True},
        )
        results.append(
            StartResult(
                start=dict(start), params=lik.theta_to_params(res.x), nll=float(res.fun),
                n_eval=lik.n_eval - e0, n_iter=int(res.nit), converged=bool(res.success),
                message=str(res.message), duration_s=round(time.time() - ts, 4),
            )
        )
        histories.append(hist)
        if progress:
            progress(si + 1, len(starts))

    if not results:
        raise RuntimeError("no starts were run")
    best = int(np.argmin([r.nll for r in results]))
    theta = lik.params_to_theta({**lik.fixed, **results[best].params})
    nll = results[best].nll
    polish_info: dict[str, Any] = {"applied": False}

    if opts.polish:
        e0 = lik.n_eval
        try:
            pres = minimize(lik.nll, theta, method="L-BFGS-B", bounds=lik.bounds.log_bounds(lik.free), options={"maxiter": opts.polish_maxiter})
            improved = float(pres.fun) < nll - 1e-12
            polish_info = {
                "applied": True, "improved": improved, "nll_before": nll, "nll_after": float(pres.fun),
                "success": bool(pres.success), "message": str(pres.message), "n_eval": lik.n_eval - e0,
            }
            if improved and np.isfinite(pres.fun):
                theta, nll = pres.x, float(pres.fun)
        except Exception as exc:  # noqa: BLE001  (a failed polish must not lose the NM optimum)
            polish_info = {"applied": True, "error": f"{type(exc).__name__}: {exc}"}

    params = lik.theta_to_params(theta)
    # Convergence: the best NM start reported success, and at least two starts
    # agree on the optimum to the tolerance (guards against a single lucky/unlucky start).
    agree = sum(1 for r in results if abs(r.nll - results[best].nll) < 1e-3 * max(1.0, abs(results[best].nll)))
    converged = results[best].converged and (agree >= min(2, len(results)))
    return FitResult(
        params=params, theta=np.asarray(theta), nll=nll, free=list(lik.free), fixed=dict(lik.fixed),
        starts=results, best_start_index=best, history=histories[best], polish=polish_info,
        n_eval_total=lik.n_eval - n0, converged=converged, duration_s=round(time.time() - t0, 4),
    )


def derived_quantities(params: dict[str, float], force_rate: float, kBT: float) -> dict[str, float]:
    from .model import LandscapeKinetics

    land = LandscapeKinetics(params["U_barrier"], params["xi_barrier"], params["gamma0"], kBT)
    mom = land.moments_exact(force_rate)
    return {
        **land.to_dict(force_rate),
        "mean_eq8_pN": land.mean_eq8(force_rate),
        "mean_exact_pN": mom["mean"],
        "sd_exact_pN": mom["sd"],
        "mode_pN": land.mode(force_rate),
        "residual_mass": mom["residual_mass"],
    }


__all__ = ["FitOptions", "FitResult", "StartResult", "default_starts", "fit_mle", "derived_quantities", "PARAM_NAMES"]
