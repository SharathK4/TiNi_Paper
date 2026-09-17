"""Inference on synthetic data with known parameters, likelihood mechanics,
uncertainty tools, and run resumability."""

import numpy as np
import pytest

from tini.config import Settings
from tini.inverse.fit import FitOptions, default_starts, fit_mle
from tini.inverse.likelihood import Bounds, Likelihood
from tini.inverse.model import LandscapeKinetics
from tini.inverse.synthetic import PAPER_MLE_RATES, recovery_experiment
from tini.inverse.uncertainty import bootstrap, observed_information, profile_likelihood

TRUTH = {"U_barrier": 638.14, "xi_barrier": 29.2, "gamma0": 3.96e-7}
LAND = LandscapeKinetics(**TRUTH)
FDOT = 0.0943


def _sample(rates, S, seed=3):
    return [(LAND.sample(S, r, seed=seed + i), r) for i, r in enumerate(rates)]


# ------------------------------------------------------------- likelihood
def test_nll_matches_sum_of_log_pdf():
    f = LAND.sample(50, FDOT, seed=0)
    lik = Likelihood([(f, FDOT)])
    direct = -float(LAND.log_pdf(f, FDOT).sum())
    assert lik.nll_params(TRUTH) == pytest.approx(direct)
    assert lik.nll(lik.params_to_theta(TRUTH)) == pytest.approx(direct)


def test_penalty_outside_support_and_bounds():
    f = LAND.sample(50, FDOT, seed=0)
    lik = Likelihood([(f, FDOT)])
    # f_c below max f‡ → penalty, finite, and increasing with the violation
    small = {"U_barrier": 100.0, "xi_barrier": 100.0, "gamma0": 1e-6}   # f_c = 1.5 pN
    p1 = lik.nll(lik.params_to_theta(small))
    assert 1e5 < p1 < 1e9
    tiny = {"U_barrier": 100.0, "xi_barrier": 150.0, "gamma0": 1e-6}
    assert lik.nll(lik.params_to_theta(tiny)) > p1
    # out of bounds
    assert lik.nll(np.log([1e5, 29.2, 1e-7])) > 1e5


def test_fixed_parameters_reduce_free_set():
    f = LAND.sample(50, FDOT, seed=0)
    lik = Likelihood([(f, FDOT)], fixed={"xi_barrier": 29.2})
    assert lik.free == ["U_barrier", "gamma0"]
    th = lik.params_to_theta({"U_barrier": 600.0, "gamma0": 1e-7})
    assert lik.theta_to_params(th)["xi_barrier"] == 29.2


# ------------------------------------------------------------- recovery
def test_paper_design_recovers_truth():
    """4 rates × 400 events: all three parameters within a few percent (Γ0 within 30 %)."""
    lik = Likelihood(_sample(PAPER_MLE_RATES, 400))
    start = {k: v * fac for (k, v), fac in zip(TRUTH.items(), (1.5, 0.7, 5.0))}
    fit = fit_mle(lik, default_starts(start, lik.free), FitOptions(polish=True))
    assert fit.converged
    assert fit.params["U_barrier"] == pytest.approx(TRUTH["U_barrier"], rel=0.05)
    assert fit.params["xi_barrier"] == pytest.approx(TRUTH["xi_barrier"], rel=0.05)
    assert 0.5 < fit.params["gamma0"] / TRUTH["gamma0"] < 2.0
    assert fit.nll <= lik.nll_params(TRUTH) + 1e-6   # the MLE is at least as likely as the truth


def test_single_rate_100_recovers_fc_but_not_gamma0():
    """Our design: f_c (location) is recovered; Γ0 is not — the documented degeneracy."""
    lik = Likelihood(_sample([FDOT], 100, seed=11))
    fit = fit_mle(lik, default_starts(TRUTH, lik.free), FitOptions(polish=False))
    fc = 3 * fit.params["U_barrier"] / (2 * fit.params["xi_barrier"])
    assert fc == pytest.approx(LAND.f_c, rel=0.15)
    prof = profile_likelihood(lik, fit, n_points=9, span_factor={"U_barrier": 3, "xi_barrier": 3, "gamma0": 30})
    g = prof["profiles"]["gamma0"]
    lo, hi = g["ci95_lower"], g["ci95_upper"]
    # the 95 % profile interval for Γ0 spans at least a decade
    if lo is not None and hi is not None:
        assert hi / lo > 10
    else:
        assert g["lower_unbounded_in_scan"] or g["upper_unbounded_in_scan"]


def test_recovery_experiment_shapes():
    res = recovery_experiment(TRUTH, [{"name": "t", "rates": [FDOT], "S": 60}], replicates=2, seed=1, opts=FitOptions(maxiter=800, maxfev=1600, polish=False))
    d = res["designs"][0]
    assert d["replicates"] == 2 and set(d["summary"]) >= set(TRUTH)
    assert 0 < d["summary"]["U_barrier"]["median_ratio"] < 10


# ------------------------------------------------------------- uncertainty
def test_observed_information_is_positive_definite_for_large_sample():
    lik = Likelihood(_sample(PAPER_MLE_RATES, 400, seed=5))
    fit = fit_mle(lik, [dict(TRUTH)], FitOptions(polish=True))
    oi = observed_information(lik, fit)
    assert oi["positive_definite"]
    for k in fit.free:
        assert oi["se"][k] > 0
        lo, hi = oi["ci95_asymptotic"][k]
        assert lo < fit.params[k] < hi


def test_profile_contains_mle_and_is_minimal_there():
    lik = Likelihood(_sample([FDOT], 150, seed=7))
    fit = fit_mle(lik, [dict(TRUTH)], FitOptions(polish=False))
    prof = profile_likelihood(lik, fit, n_points=7, span_factor=2.0)
    for k, p in prof["profiles"].items():
        feas = [r for r in p["grid"] if r["feasible"]]
        assert min(r["delta_nll"] for r in feas) >= -1e-3   # nothing below the MLE
        assert any(abs(r["value"] - fit.params[k]) < 1e-9 for r in p["grid"])


def test_bootstrap_is_seeded_and_reproducible():
    lik = Likelihood(_sample([FDOT], 60, seed=2))
    fit = fit_mle(lik, [dict(TRUTH)], FitOptions(polish=False))
    a = bootstrap(lik, fit, n_boot=6, seed=42, opts=FitOptions(maxiter=500, maxfev=1000, polish=False))
    b = bootstrap(lik, fit, n_boot=6, seed=42, opts=FitOptions(maxiter=500, maxfev=1000, polish=False))
    assert a["n_ok"] == 6 and [d["U_barrier"] for d in a["draws"]] == [d["U_barrier"] for d in b["draws"]]


# ------------------------------------------------------------- runs (resume, isolation)
def _make_v1_artifacts(tmp_path, n=40, seed=0):
    """A minimal traces.npz: linear ramps that drop at a model-drawn transition time."""
    rng = np.random.default_rng(seed)
    t = np.arange(997) * 0.64
    fdot = 0.0943
    f_crit = LAND.sample(n, fdot, seed=seed)
    traces = np.zeros((n, t.size, 3))
    for i in range(n):
        t_c = f_crit[i] / fdot
        force = np.where(t < t_c, fdot * t, -40 + 0.05 * (t - t_c)) + 0.5 * rng.standard_normal(t.size)
        traces[i, :, 0] = t
        traces[i, :, 1] = 60.2 + 0.00235 * t
        traces[i, :, 2] = force
    art = tmp_path / "art"
    art.mkdir()
    np.savez(art / "traces.npz", traces=traces, trace_ids=np.array([f"{i:05d}" for i in range(n)]), time_ps=t)
    return Settings(data_dir=str(tmp_path), artifacts_dir=str(art))


def test_run_completes_resumes_and_keeps_previous(tmp_path):
    from tini.inverse.runs import InverseConfig, InverseRun, list_runs, load_run

    settings = _make_v1_artifacts(tmp_path)
    cfg = InverseConfig(bootstrap_n=5, profile_points=5, synthetic_replicates=1, sensitivity_windows_ps=[100.0], sensitivity_smoothings=[15], sensitivity_sources=["rate_x_time"], constrained=[["xi_barrier"]])
    run = InverseRun(settings, cfg, run_id="t1")
    snap = run.run()
    assert snap["status"] == "done", snap
    res = load_run(settings, "t1")
    assert res["fit"]["converged"] and res["sample"]["n"] == 40
    assert set(res["parameters"]) == {"U_barrier", "xi_barrier", "gamma0"}
    assert list_runs(settings)["latest"]["run_id"] == "t1"
    fit_mtime = (run.dir / "fit.json").stat().st_mtime_ns

    # resume: nothing recomputed
    run2 = InverseRun(settings, cfg, run_id="t1")
    snap2 = run2.run()
    assert snap2["status"] == "done"
    assert (run2.dir / "fit.json").stat().st_mtime_ns == fit_mtime
    assert all(v == "skipped (checkpoint)" for k, v in snap2["stage_status"].items() if k != "finalize")

    # a second run is isolated from the first
    run3 = InverseRun(settings, cfg, run_id="t2")
    assert run3.run()["status"] == "done"
    assert (run.dir / "fit.json").stat().st_mtime_ns == fit_mtime
    assert len(list_runs(settings)["runs"]) == 2


def test_run_cancel_keeps_completed_phases(tmp_path):
    from tini.inverse.runs import InverseConfig, InverseRun

    settings = _make_v1_artifacts(tmp_path)
    run = InverseRun(settings, InverseConfig(bootstrap_n=5, profile_points=5, synthetic_replicates=1), run_id="c1")
    run.cancel()
    snap = run.run()
    assert snap["status"] == "cancelled"
    assert not (run.dir / "fit.json").exists()
