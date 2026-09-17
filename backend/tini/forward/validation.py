"""Numerical validation of the forward solver (docs/FORWARD_MODEL.md §6).

Each function returns a dictionary of quantitative metrics; the tests assert
on them and the run pipeline stores them, so the evidence shown in the UI is
the same evidence the test suite checks.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..inverse.model import LandscapeKinetics
from .solver import ForwardParams, SolverSettings, equilibrium, force_density, make_grid, mean_first_passage_exact, solve


# ------------------------------------------------------------------ A, B
def conservation_and_positivity(sol) -> dict[str, float]:
    return {
        "conservation_error_max": sol.conservation_error_max,
        "min_density": sol.min_density,
        "final_survival": float(sol.survival[-1]),
        "final_absorbed": float(sol.absorbed[-1]),
    }


# ------------------------------------------------------------------ C1
def reflecting_equilibrium(params: ForwardParams, n_cells: int = 200, n_steps: int = 2000, t_end: float = 50.0) -> dict[str, float]:
    """Both walls reflecting, zero force: the discrete Boltzmann density must be stationary."""
    st = SolverSettings(n_cells=n_cells, n_steps=n_steps, boundary="barrier_top")
    grid, _ = make_grid(params, st)
    P0 = equilibrium(params, grid, 0.0)
    sol = solve(params, 1.0, st, force_override=lambda t: 0.0, t_end=t_end, initial=P0, absorbing=False)
    drift = float(np.abs(sol.snapshots[-1] - P0).sum() * grid.h)
    return {"l1_drift": drift, "mass_error": abs(float(sol.survival[-1]) - 1.0), "t_end_ps": t_end}


# ------------------------------------------------------------------ C2
def mfpt_benchmark(params: ForwardParams, force_fraction: float = 0.6, n_cells: int = 300, n_steps: int = 6000, boundary: str = "beyond_barrier", horizon_factor: float = 14.0) -> dict[str, float]:
    """Fixed force, start from the Boltzmann density at that force: the mean
    first-passage time ∫Ψ dt against the exact 1-D quadrature formula."""
    f = force_fraction * params.f_c
    st = SolverSettings(n_cells=n_cells, n_steps=n_steps, boundary=boundary)
    grid, _ = make_grid(params, st)
    tau_exact = mean_first_passage_exact(params, grid, f)
    T = horizon_factor * tau_exact
    P0 = equilibrium(params, grid, f)
    sol = solve(params, 1.0, st, force_override=lambda t: f, t_end=T, initial=P0)
    tau_pde = float(np.trapezoid(sol.survival, sol.t))
    # tail beyond the horizon: exponential with the late-time rate
    k = -np.log(sol.survival[-1] / sol.survival[-1000]) / (sol.t[-1] - sol.t[-1000]) if sol.survival[-1] > 0 else np.inf
    tail = float(sol.survival[-1] / k) if np.isfinite(k) and k > 0 else 0.0
    tau_pde += tail
    return {
        "force_pN": f, "barrier_remaining_kBT": params.U_barrier * (1 - force_fraction) ** 1.5 / params.kBT,
        "tau_exact_ps": tau_exact, "tau_pde_ps": tau_pde, "relative_error": abs(tau_pde - tau_exact) / tau_exact,
        "survival_at_horizon": float(sol.survival[-1]), "n_cells": n_cells, "n_steps": n_steps, "dt_ps": float(T / n_steps),
    }


# ------------------------------------------------------------------ C3
def free_diffusion(params: ForwardParams, n_cells: int = 400, n_steps: int = 2000, t_end: float = 2.0) -> dict[str, float]:
    """U ≡ 0 on a wide reflecting box: a narrow initial Gaussian spreads with variance 2Dt."""
    L = 12.0 * np.sqrt(2 * params.D * t_end)
    st = SolverSettings(n_cells=n_cells, n_steps=n_steps, boundary="barrier_top", xi_A_override=L / 2, xi_left_factor=(L / 2) / params.xi_barrier)
    grid, _ = make_grid(params, st)
    s0 = np.sqrt(2 * params.D * t_end) / 6.0
    P0 = np.exp(-0.5 * (grid.centers / s0) ** 2)
    P0 /= P0.sum() * grid.h
    flat = lambda x, f: np.zeros_like(x)
    sol = solve(params, 1.0, st, force_override=lambda t: 0.0, t_end=t_end, initial=P0, absorbing=False, potential=flat)
    P = sol.snapshots[-1]
    var = float((grid.centers**2 * P).sum() * grid.h - ((grid.centers * P).sum() * grid.h) ** 2)
    expected = s0**2 + 2 * params.D * t_end
    return {"variance_pde_A2": var, "variance_exact_A2": expected, "relative_error": abs(var - expected) / expected, "mass_error": abs(float(sol.survival[-1]) - 1.0)}


# ------------------------------------------------------------------ D, E
def refinement(params: ForwardParams, force_rate: float, base: SolverSettings, axis: str = "grid", levels: int = 3, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Halve h (axis='grid') or Δt (axis='time') `levels` times; report the
    change of ⟨f‡⟩, the L1 distance between successive p(f), and the observed
    order from the last three levels."""
    rows = []
    prev = None
    for k in range(levels):
        st = SolverSettings(**{**base.to_dict(), ("n_cells" if axis == "grid" else "n_steps"): (base.n_cells if axis == "grid" else base.n_steps) * 2**k})
        sol = solve(params, force_rate, st)
        fd = force_density(sol)
        row = {"level": k, "n_cells": st.n_cells, "n_steps": st.n_steps, "h_A": sol.grid.h, "dt_ps": float(sol.t[1] - sol.t[0]), "mean_pN": fd["mean_pN"], "sd_pN": fd["sd_pN"], "duration_s": sol.duration_s}
        if prev is not None:
            # interpolate the coarser p(f) onto the finer force grid for the L1 distance
            p_prev = np.interp(fd["f_pN"], prev["f"], prev["pdf"])
            row["l1_vs_previous"] = float(np.trapezoid(np.abs(fd["pdf"] - p_prev), fd["f_pN"]))
            row["mean_change_pN"] = fd["mean_pN"] - prev["mean"]
        rows.append(row)
        prev = {"f": fd["f_pN"], "pdf": fd["pdf"], "mean": fd["mean_pN"]}
        if progress:
            progress(k + 1, levels)
    order = None
    if levels >= 3 and all("mean_change_pN" in r for r in rows[1:]):
        d1, d2 = abs(rows[-2]["mean_change_pN"]), abs(rows[-1]["mean_change_pN"])
        order = float(np.log2(d1 / d2)) if d1 > 0 and d2 > 0 else None
    richardson = None
    if order and order > 0:
        richardson = abs(rows[-1]["mean_change_pN"]) / (2**order - 1)
    return {"axis": axis, "levels": rows, "observed_order_mean": order, "richardson_error_estimate_pN": richardson}


# ------------------------------------------------------------------ F
def kramers_check(params: ForwardParams, force_fraction: float = 0.6, n_cells: int = 300, n_steps: int = 6000) -> dict[str, Any]:
    """Fixed force: late-time escape rate −d ln Ψ/dt against the paper's Γ(f) (Eq. 4)
    for three absorbing placements: on the transition state, at ξ‡, beyond."""
    f = force_fraction * params.f_c
    lk = LandscapeKinetics(params.U_barrier, params.xi_barrier, params.gamma0, params.kBT)
    gamma_eq4 = float(lk.rate(np.array([f]))[0])
    xi_ts = params.xi_barrier / 2 * (1 + np.sqrt(1 - force_fraction))
    out = {"force_pN": f, "barrier_remaining_kBT": params.U_barrier * (1 - force_fraction) ** 1.5 / params.kBT, "gamma_eq4_per_ps": gamma_eq4, "xi_ts_A": float(xi_ts), "placements": []}
    for name, override in (("transition_state", float(xi_ts)), ("barrier_top", None), ("beyond_barrier", None)):
        st = SolverSettings(n_cells=n_cells, n_steps=n_steps, boundary="barrier_top" if name != "beyond_barrier" else "beyond_barrier", xi_A_override=override)
        grid, info = make_grid(params, st)
        T = 8.0 / gamma_eq4
        P0 = equilibrium(params, grid, f)
        sol = solve(params, 1.0, st, force_override=lambda t: f, t_end=T, initial=P0)
        # fit the exponential regime: between Ψ = 0.8 and Ψ = 0.05
        m = (sol.survival < 0.8) & (sol.survival > 0.05)
        if m.sum() > 10:
            slope = -np.polyfit(sol.t[m], np.log(sol.survival[m]), 1)[0]
        else:
            slope = float("nan")
        tau_exact = mean_first_passage_exact(params, grid, f)
        out["placements"].append({"placement": name, "xi_A_A": info["xi_A_A"], "rate_pde_per_ps": float(slope), "ratio_pde_over_eq4": float(slope / gamma_eq4), "mfpt_exact_ps": tau_exact, "ratio_1_over_mfpt_over_eq4": float(1 / (tau_exact * gamma_eq4))})
    return out


# ------------------------------------------------------------------ G
def reproducibility(params: ForwardParams, force_rate: float, settings: SolverSettings) -> dict[str, Any]:
    a = solve(params, force_rate, settings)
    b = solve(params, force_rate, settings)
    return {"identical_survival": bool(np.array_equal(a.survival, b.survival)), "identical_snapshots": bool(np.array_equal(a.snapshots, b.snapshots)), "max_abs_diff": float(np.max(np.abs(a.survival - b.survival)))}


# ------------------------------------------------------------------ H
def physics_checks(params: ForwardParams, force_rate: float, sol, rates: tuple[float, ...] = (3.246e-3, 9.452e-2, 9.242e-1), settings: SolverSettings | None = None) -> dict[str, Any]:
    fd = force_density(sol)
    lk = LandscapeKinetics(params.U_barrier, params.xi_barrier, params.gamma0, params.kBT)
    p6 = lk.pdf(fd["f_pN"], force_rate)
    l1 = float(np.trapezoid(np.abs(fd["pdf"] - p6), fd["f_pN"]))
    cdf6 = lk.cdf(fd["f_pN"], force_rate)
    dmax = float(np.max(np.abs(fd["cdf"] - cdf6)))
    m6 = lk.moments_exact(force_rate)
    means = []
    st = settings or SolverSettings(n_cells=200, n_steps=8000)
    for r in rates:
        s = solve(params, r, st)
        d = force_density(s)
        means.append({"force_rate": r, "mean_pde_pN": d["mean_pN"], "sd_pde_pN": d["sd_pN"], "mean_eq8_pN": lk.mean_eq8(r), "mean_eq6_exact_pN": lk.moments_exact(r, n=4001)["mean"], "residual_mass": d["residual_mass"]})
    return {
        "pdf_nonnegative": bool(np.all(fd["pdf"] >= 0)),
        "norm_plus_residual": fd["norm"],
        "mean_pde_pN": fd["mean_pN"], "sd_pde_pN": fd["sd_pN"],
        "mean_eq6_pN": m6["mean"], "sd_eq6_pN": m6["sd"], "mean_eq8_pN": lk.mean_eq8(force_rate),
        "l1_pde_vs_eq6": l1, "max_cdf_diff_pde_vs_eq6": dmax,
        "mean_increases_with_rate": bool(all(np.diff([m["mean_pde_pN"] for m in means]) > 0)),
        "spectrum": means,
    }
