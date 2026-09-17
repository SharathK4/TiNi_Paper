"""Forward Smoluchowski solver: scheme properties, exact benchmarks,
convergence, limits, and agreement with the paper's analytic model."""

import numpy as np
import pytest

from tini.forward import validation as V
from tini.forward.solver import ForwardParams, SolverSettings, bernoulli, build_operator, equilibrium, force_density, make_grid, solve, transition_state
from tini.inverse.model import LandscapeKinetics

P = ForwardParams(638.14, 29.2, 3.96e-7)   # Table 1, MLE column
FDOT = 0.0943


@pytest.fixture(scope="module")
def ramp():
    return solve(P, FDOT, SolverSettings(n_cells=200, n_steps=6000))


# ------------------------------------------------------------- helpers
def test_bernoulli_limits():
    assert bernoulli(np.array([0.0]))[0] == 1.0
    assert bernoulli(np.array([1e-12]))[0] == pytest.approx(1.0, abs=1e-11)
    x = np.array([-800.0, -5.0, 5.0, 50.0])
    b = bernoulli(x)
    assert b[0] == pytest.approx(800.0)
    assert b[1] == pytest.approx(5 / (np.exp(5) - 1) * np.exp(5), rel=1e-12)  # B(-x) = x/(1-e^-x)
    assert b[2] == pytest.approx(5 / (np.exp(5) - 1), rel=1e-12)
    assert 0 < b[3] < 1e-18


def test_eta_and_D_from_kramers_prefactor():
    # η from the paper's Γ0 formula, in pN·ps/Å; D = k_BT/η
    d = P.to_dict()
    assert d["eta_pN_ps_per_A"] == pytest.approx(0.3676, abs=1e-3)
    assert d["D_A2_per_ps"] == pytest.approx(112.7, abs=0.2)
    # the same Γ0 must come back from the Kramers formula
    kappa = 6 * P.U_barrier / P.xi_barrier**2
    assert kappa * np.exp(-P.U_barrier / P.kBT) / (2 * np.pi * P.eta) == pytest.approx(P.gamma0, rel=1e-12)


def test_operator_is_m_matrix_and_conservative():
    st = SolverSettings(n_cells=120)
    grid, _ = make_grid(P, st)
    lo, di, up, c = build_operator(P, grid, 10.0)
    assert np.all(lo[1:] > 0) and np.all(up[:-1] > 0) and np.all(di < 0)
    # column sums of A vanish except for the absorbing cell, whose loss is the boundary flux
    colsum = di + np.append(lo[1:], 0.0) + np.append(0.0, up[:-1])
    assert np.allclose(colsum[:-1], 0.0, atol=1e-12)
    assert colsum[-1] == pytest.approx(-c / grid.h)


def test_transition_state_matches_v1_formula():
    assert transition_state(P, 0.0) == pytest.approx(P.xi_barrier)
    assert transition_state(P, P.f_c) == pytest.approx(P.xi_barrier / 2)


# ------------------------------------------------------------- A, B
def test_conservation_and_positivity(ramp):
    m = V.conservation_and_positivity(ramp)
    assert m["conservation_error_max"] < 1e-11
    assert m["min_density"] >= 0.0
    assert m["final_survival"] + m["final_absorbed"] == pytest.approx(1.0, abs=1e-11)


# ------------------------------------------------------------- C
def test_reflecting_equilibrium_is_stationary():
    m = V.reflecting_equilibrium(P)
    assert m["l1_drift"] < 1e-12 and m["mass_error"] < 1e-12


@pytest.mark.parametrize("frac,boundary", [(0.6, "beyond_barrier"), (0.6, "barrier_top"), (0.5, "beyond_barrier")])
def test_mean_first_passage_matches_exact_formula(frac, boundary):
    m = V.mfpt_benchmark(P, force_fraction=frac, boundary=boundary)
    assert m["relative_error"] < 5e-3, m


def test_free_diffusion_variance():
    m = V.free_diffusion(P)
    assert m["relative_error"] < 1e-6 and m["mass_error"] < 1e-11


# ------------------------------------------------------------- D, E
def test_grid_convergence_second_order():
    r = V.refinement(P, FDOT, SolverSettings(n_cells=60, n_steps=1500), "grid", 4)
    assert 1.7 < r["observed_order_mean"] < 2.3, r
    assert r["levels"][-1]["l1_vs_previous"] < 2e-3


def test_time_convergence_first_order():
    r = V.refinement(P, FDOT, SolverSettings(n_cells=150, n_steps=1000), "time", 4)
    assert 0.85 < r["observed_order_mean"] < 1.15, r
    assert r["richardson_error_estimate_pN"] < 5e-3


# ------------------------------------------------------------- F
def test_kramers_rate_and_recrossing_factor():
    r = V.kramers_check(P, force_fraction=0.3, n_steps=6000)
    by = {p["placement"]: p for p in r["placements"]}
    # beyond the barrier: Kramers' rate within the O(k_BT/U‡) correction
    assert 0.9 < by["beyond_barrier"]["ratio_pde_over_eq4"] < 1.05
    # on the transition state: faster by the recrossing factor, approaching 2
    assert 1.5 < by["transition_state"]["ratio_pde_over_eq4"] < 2.0
    # the two ways of measuring the rate agree
    for p in r["placements"]:
        assert p["ratio_pde_over_eq4"] == pytest.approx(p["ratio_1_over_mfpt_over_eq4"], rel=0.03)


# ------------------------------------------------------------- G
def test_reproducible():
    m = V.reproducibility(P, FDOT, SolverSettings(n_cells=100, n_steps=1000))
    assert m["identical_survival"] and m["identical_snapshots"]


# ------------------------------------------------------------- H
def test_force_density_is_a_density_and_agrees_with_eq6(ramp):
    fd = force_density(ramp)
    assert np.all(fd["pdf"] >= 0)
    assert fd["norm"] == pytest.approx(1.0, abs=2e-4)
    lk = LandscapeKinetics(P.U_barrier, P.xi_barrier, P.gamma0)
    m6 = lk.moments_exact(FDOT)
    assert abs(fd["mean_pN"] - m6["mean"]) < 0.4          # Kramers/steady-flux approximation error, documented
    assert abs(fd["sd_pN"] - m6["sd"]) < 0.2
    l1 = np.trapezoid(np.abs(fd["pdf"] - lk.pdf(fd["f_pN"], FDOT)), fd["f_pN"])
    assert l1 < 0.12


def test_mean_increases_with_rate():
    means = [force_density(solve(P, r, SolverSettings(n_cells=120, n_steps=2500)))["mean_pN"] for r in (3.2e-3, 9.4e-2, 9.2e-1)]
    assert means[0] < means[1] < means[2] < P.f_c


def test_cancel_raises():
    with pytest.raises(InterruptedError):
        solve(P, FDOT, SolverSettings(n_cells=50, n_steps=500), cancel=lambda: True)


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        SolverSettings(boundary="elsewhere")
    with pytest.raises(ValueError):
        ForwardParams(1.0, 1.0, 0.0)
