"""Forward-model runs: storage, resume, isolation, cancellation, parameter
resolution, and the read-only API surface."""

import json

import numpy as np
import pytest

from tini.config import Settings
from tini.forward.runs import ForwardConfig, ForwardRun, list_runs, load_run, resolve_parameters


def _settings(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    (art / "force_rate.json").write_text(json.dumps({"force_rate_pN_per_ps": 0.0943, "force_rate_se": 0.001, "window_ps": 100}))
    return Settings(data_dir=str(tmp_path), artifacts_dir=str(art))


SMALL = dict(parameter_source="paper:pdf_mle", n_cells=80, n_steps=1500, validation_enabled=False, convergence_enabled=False, spectrum_enabled=False)


def test_resolve_parameter_sources(tmp_path):
    s = _settings(tmp_path)
    p, prov = resolve_parameters(s, ForwardConfig(parameter_source="paper:mean_lsq"))
    assert p.U_barrier == 659.0 and prov["source"] == "paper:mean_lsq"
    p, prov = resolve_parameters(s, ForwardConfig(parameter_source="custom", custom={"U_barrier": 500, "xi_barrier": 20, "gamma0": 1e-6}))
    assert p.f_c == pytest.approx(37.5)
    with pytest.raises(FileNotFoundError):
        resolve_parameters(s, ForwardConfig(parameter_source="inverse:latest"))
    with pytest.raises(ValueError):
        resolve_parameters(s, ForwardConfig(parameter_source="custom", custom={"U_barrier": 500}))


def test_temperature_keeps_eta_and_rescales_gamma0(tmp_path):
    s = _settings(tmp_path)
    p300, _ = resolve_parameters(s, ForwardConfig(parameter_source="paper:pdf_mle"))
    p350, prov = resolve_parameters(s, ForwardConfig(parameter_source="paper:pdf_mle", temperature_K=350.0))
    assert p350.eta == pytest.approx(p300.eta, rel=1e-9)
    assert p350.gamma0 > p300.gamma0 and "temperature_note" in prov
    assert p350.kBT == pytest.approx(p300.kBT * 350 / 300, rel=1e-9)


def test_run_completes_resumes_and_isolates(tmp_path):
    s = _settings(tmp_path)
    run = ForwardRun(s, ForwardConfig(**SMALL), run_id="f1")
    snap = run.run()
    assert snap["status"] == "done", snap
    res = load_run(s, "f1")
    assert res["observables"]["norm"] == pytest.approx(1.0, abs=2e-3)
    assert res["observables"]["data_comparison"].get("unavailable")   # no V1 traces in this tmp project
    assert (run.dir / "solution.npz").exists()
    assert list_runs(s)["latest"]["run_id"] == "f1"
    mtime = (run.dir / "solution.npz").stat().st_mtime_ns

    run2 = ForwardRun(s, ForwardConfig(**SMALL), run_id="f1")
    snap2 = run2.run()
    assert snap2["status"] == "done"
    assert (run2.dir / "solution.npz").stat().st_mtime_ns == mtime
    assert all(v == "skipped (checkpoint)" for k, v in snap2["stage_status"].items() if k != "finalize")

    run3 = ForwardRun(s, ForwardConfig(**{**SMALL, "parameter_source": "paper:mean_lsq"}), run_id="f2")
    assert run3.run()["status"] == "done"
    assert (run.dir / "solution.npz").stat().st_mtime_ns == mtime
    assert len(list_runs(s)["runs"]) == 2
    assert load_run(s, "f2")["parameters"]["U_barrier_pN_A"] == 659.0


def test_run_with_validation_and_convergence_records_metrics(tmp_path):
    s = _settings(tmp_path)
    cfg = ForwardConfig(parameter_source="paper:pdf_mle", n_cells=80, n_steps=2000, validation_enabled=True, convergence_enabled=True, spectrum_enabled=False)
    run = ForwardRun(s, cfg, run_id="v1")
    assert run.run()["status"] == "done"
    res = load_run(s, "v1")
    v = res["validation"]
    assert v["A_B_conservation_positivity"]["conservation_error_max"] < 1e-11
    assert v["C1_reflecting_equilibrium"]["l1_drift"] < 1e-12
    assert all(m["relative_error"] < 5e-3 for m in v["C2_mfpt_exact"])
    assert v["G_reproducibility"]["identical_survival"]
    c = res["convergence"]
    assert 1.5 < c["grid_order"] < 2.5 and 0.8 < c["time_order"] < 1.2


def test_cancel_keeps_completed_phases(tmp_path):
    s = _settings(tmp_path)
    run = ForwardRun(s, ForwardConfig(**SMALL), run_id="c1")
    run.cancel()
    assert run.run()["status"] == "cancelled"
    assert not (run.dir / "solution.npz").exists()


def test_solution_file_roundtrip(tmp_path):
    s = _settings(tmp_path)
    run = ForwardRun(s, ForwardConfig(**SMALL), run_id="r1")
    run.run()
    z = np.load(run.dir / "solution.npz")
    assert z["survival"].shape == (SMALL["n_steps"] + 1,)
    assert z["snapshots"].shape[1] == SMALL["n_cells"]
    assert abs(z["survival"][-1] + z["absorbed"][-1] - 1) < 1e-11


# ------------------------------------------------------------- API (read-only surface on the real artifacts)
def test_api_forward_read_endpoints():
    from fastapi.testclient import TestClient
    from tini.api import app

    c = TestClient(app)
    spec = c.get("/api/forward/spec").json()
    assert spec["model_version"] and "beyond_barrier" in spec["boundaries"]
    r = c.get("/api/forward/parameters", params={"source": "paper:pdf_mle"})
    assert r.status_code == 200 and r.json()["parameters"]["eta_pN_ps_per_A"] == pytest.approx(0.3676, abs=1e-3)
    assert c.get("/api/forward/progress").status_code == 200
    assert c.get("/api/forward/runs").status_code == 200
    bad = c.post("/api/forward/run", json={"config": {"parameter_source": "nowhere"}})
    assert bad.status_code == 400
    bad2 = c.post("/api/forward/run", json={"config": {"n_cells": 5}})
    assert bad2.status_code == 400
