"""Synthetic parameter-recovery experiments.

Sample the forward model at known parameters, refit, and report how well
each experimental design recovers the truth. This is how the single-rate
identifiability limitation becomes a number instead of an argument:

    "S=100, one rate"        this dataset's design
    "S=400, one rate"        the paper's per-rate sample size
    "4 rates × 400"          the paper's MLE design (rates iii–vi)

Each design is replicated R times with distinct seeds; the recovered
parameters are summarised by their median and 16–84 % range relative to
the truth. Cost: R × (2·n_free + 1) Nelder–Mead fits per design — seconds.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .fit import FitOptions, default_starts, fit_mle
from .likelihood import Bounds, Likelihood
from .model import LandscapeKinetics

PAPER_MLE_RATES = [1.482e-2, 3.097e-2, 9.452e-2, 2.927e-1]   # rates (iii)–(vi), pN/ps


def default_designs(force_rate: float) -> list[dict[str, Any]]:
    return [
        {"name": "this dataset: S = 100, one rate", "rates": [force_rate], "S": 100},
        {"name": "S = 400, one rate", "rates": [force_rate], "S": 400},
        {"name": "paper design: 4 rates × 400", "rates": PAPER_MLE_RATES, "S": 400},
    ]


def recovery_experiment(
    truth: dict[str, float],
    designs: list[dict[str, Any]],
    replicates: int = 10,
    seed: int = 0,
    bounds: Bounds | None = None,
    kBT: float | None = None,
    opts: FitOptions | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    bounds = bounds or Bounds()
    opts = opts or FitOptions(maxiter=2000, maxfev=4000, polish=False)
    land = LandscapeKinetics(truth["U_barrier"], truth["xi_barrier"], truth["gamma0"], **({"kBT": kBT} if kBT else {}))
    total = len(designs) * replicates
    done = 0
    out: list[dict[str, Any]] = []
    t0 = time.time()
    for d in designs:
        fits: list[dict[str, float]] = []
        for r in range(replicates):
            if cancel and cancel():
                break
            rng_seed = seed * 100_003 + r
            groups = []
            for gi, rate in enumerate(d["rates"]):
                f = land.sample(d["S"], rate, seed=rng_seed * 7 + gi)
                # drop residual-mass samples at f_c: they carry no density information
                groups.append((f[f < land.f_c], rate))
            lik = Likelihood(groups, bounds, {}, land.kBT)
            # start from the truth perturbed, as a fitter would from Table 1 values
            start_ref = {k: truth[k] * fac for k, fac in zip(truth, (1.3, 0.8, 3.0))}
            fit = fit_mle(lik, default_starts(start_ref, lik.free, factors=(1.0, 0.5, 2.0)), opts)
            fits.append({**fit.params, "nll": fit.nll, "converged": fit.converged, "n_eval": fit.n_eval_total})
            done += 1
            if progress:
                progress(done, total)
        summary: dict[str, Any] = {}
        for k in truth:
            v = np.array([f[k] for f in fits]) if fits else np.array([])
            if v.size:
                ratio = v / truth[k]
                summary[k] = {
                    "truth": truth[k],
                    "median": float(np.median(v)),
                    "p16": float(np.percentile(v, 16)),
                    "p84": float(np.percentile(v, 84)),
                    "median_ratio": float(np.median(ratio)),
                    "log10_spread_16_84": float(np.log10(np.percentile(ratio, 84) / np.percentile(ratio, 16))),
                    "within_factor_2": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
                    "within_20pct": float(np.mean(np.abs(ratio - 1) < 0.2)),
                }
        if fits:
            fc = np.array([3 * f["U_barrier"] / (2 * f["xi_barrier"]) for f in fits])
            summary["f_c_pN"] = {"truth": land.f_c, "median": float(np.median(fc)), "p16": float(np.percentile(fc, 16)), "p84": float(np.percentile(fc, 84))}
        out.append({**{k: d[k] for k in ("name", "rates", "S")}, "n_events": d["S"] * len(d["rates"]), "replicates": len(fits), "fits": fits, "summary": summary})
    return {"truth": truth, "replicates": replicates, "seed": seed, "designs": out, "duration_s": round(time.time() - t0, 3)}
