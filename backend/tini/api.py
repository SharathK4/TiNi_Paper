"""FastAPI application.

Read endpoints serve the checkpointed artifacts; a small mtime-keyed cache
avoids re-reading traces.npz on every request. The only computation done on
request is cheap and pure: re-fitting the force rate for a different window,
re-running the peak picker for a different smoothing width, and evaluating
the cubic landscape for slider values.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import statistics as st
from .config import FIG3_READINGS_BY_EYE, K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS, PAPER_RATES, TEMPERATURE_K, load_settings
from .io_atomic import json_safe, read_json
from .pipeline import STAGES, Pipeline
from .potential import Landscape, potential_curves

settings = load_settings()
pipeline = Pipeline(settings)

app = FastAPI(title="TiNi force-trace analysis", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ART = settings.artifacts_path


# ------------------------------------------------------------------ cache
class _TraceCache:
    def __init__(self) -> None:
        self.mtime = 0.0
        self.traces: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.ids: list[str] = []

    def get(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        p = ART / "traces.npz"
        if not p.exists():
            raise HTTPException(404, "traces not processed yet — run the pipeline")
        m = p.stat().st_mtime
        if self.traces is None or m != self.mtime:
            z = np.load(p)
            self.traces, self.t, self.ids = z["traces"], z["time_ps"], [str(s) for s in z["trace_ids"]]
            self.mtime = m
        return self.traces, self.t, self.ids


_cache = _TraceCache()


def _artifact(name: str) -> Any:
    data = read_json(ART / name, default=None)
    if data is None:
        raise HTTPException(404, f"{name} not available yet — run the pipeline")
    return data


def _round(a: np.ndarray, nd: int = 5) -> list[float]:
    return [round(float(v), nd) for v in a]


# ------------------------------------------------------------------ status
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "time": time.time(), "running": pipeline.running}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "settings": settings.to_dict(),
        "constants": {"kBT_pN_A": K_B_T_PN_ANGSTROM, "temperature_K": TEMPERATURE_K},
        "paper_parameter_sets": PAPER_PARAMETER_SETS,
        "paper_rates": PAPER_RATES,
        "stages": STAGES,
    }


@app.get("/api/progress")
def progress() -> dict[str, Any]:
    snap = pipeline.progress.snapshot()
    snap["running"] = pipeline.running
    snap["checkpoints"] = pipeline.registry.all()
    return snap


class RunRequest(BaseModel):
    force: bool = False
    stages: list[str] | None = None


@app.post("/api/pipeline/run")
def run_pipeline(req: RunRequest) -> dict[str, Any]:
    if pipeline.running:
        raise HTTPException(409, "pipeline already running")
    bad = [s for s in (req.stages or []) if s not in STAGES]
    if bad:
        raise HTTPException(400, f"unknown stages: {bad}")
    run_id = pipeline.run_async(force=req.force, stages=req.stages)
    return {"run_id": run_id}


@app.post("/api/pipeline/cancel")
def cancel_pipeline() -> dict[str, Any]:
    pipeline.cancel()
    return {"cancelling": pipeline.running}


# ------------------------------------------------------------------ dataset
@app.get("/api/dataset/inspection")
def inspection() -> Any:
    return _artifact("inspection.json")


@app.get("/api/dataset/validation")
def validation() -> Any:
    return _artifact("validation.json")


@app.get("/api/dataset/files")
def files() -> Any:
    val = _artifact("validation.json")
    ev = read_json(ART / "events.json", default=None)
    ev_by_id = {}
    if ev:
        ev_by_id = dict(zip(ev.get("trace_ids", []), ev["events"]))
    rows = []
    for r in val["files"]:
        e = ev_by_id.get(r["trace_id"], {})
        rows.append(
            {
                "trace_id": r["trace_id"],
                "name": r["name"],
                "n_rows": r["n_rows"],
                "ok": r["ok"],
                "n_warnings": r["n_warnings"],
                "n_errors": r["n_errors"],
                **{k: r["stats"].get(k) for k in ("x_start_A", "x_end_A", "ext_rate_A_per_ps", "f_max_pN", "f_mean_pN", "f_std_pN")},
                "t_peak_ps": e.get("t_peak_ps"),
                "f_peak_smoothed_pN": e.get("f_peak_smoothed_pN"),
                "f_crit_from_rate_pN": e.get("f_crit_from_rate_pN"),
            }
        )
    return {"files": rows, "collection": val["collection"]}


# ------------------------------------------------------------------ traces
@app.get("/api/traces/{trace_id}")
def trace(trace_id: str, stride: int = Query(1, ge=1, le=50), smoothing: int = Query(0, ge=0, le=201)) -> Any:
    traces, t, ids = _cache.get()
    if trace_id not in ids:
        raise HTTPException(404, f"trace {trace_id} not found")
    k = ids.index(trace_id)
    a = traces[k]
    out = {
        "trace_id": trace_id,
        "stride": stride,
        "time_ps": _round(a[::stride, 0], 3),
        "x_A": _round(a[::stride, 1], 5),
        "force_pN": _round(a[::stride, 2], 4),
    }
    if smoothing:
        out["force_smoothed_pN"] = _round(st.moving_average(a[:, 2], smoothing)[::stride], 4)
    return out


@app.get("/api/traces")
def traces_overlay(ids: str = Query(""), stride: int = Query(2, ge=1, le=50)) -> Any:
    traces, t, all_ids = _cache.get()
    wanted = [s for s in ids.split(",") if s] or all_ids[: settings.overlay_max_traces]
    wanted = wanted[: settings.overlay_max_traces]
    out = []
    for tid in wanted:
        if tid not in all_ids:
            continue
        k = all_ids.index(tid)
        out.append({"trace_id": tid, "force_pN": _round(traces[k, ::stride, 2], 3)})
    return {"time_ps": _round(t[::stride], 3), "traces": out, "available": all_ids, "max_traces": settings.overlay_max_traces}


# ------------------------------------------------------------------ statistics
@app.get("/api/stats/average")
def average(window: float | None = Query(None, gt=1.0, le=600.0), stride: int = Query(1, ge=1, le=50)) -> Any:
    traces, t, ids = _cache.get()
    avg = st.ensemble_average(traces, settings.percentiles)
    w = window or settings.force_rate_window_ps
    fit = st.force_rate(t, avg["mean_f"], w)
    sens = st.force_rate_sensitivity(t, avg["mean_f"], settings.force_rate_windows_ps)
    return {
        "n_traces": int(traces.shape[0]),
        "time_ps": _round(t[::stride], 3),
        "mean_f_pN": _round(avg["mean_f"][::stride], 4),
        "std_f_pN": _round(avg["std_f"][::stride], 4),
        "sem_f_pN": _round(avg["sem_f"][::stride], 4),
        "median_f_pN": _round(avg["median_f"][::stride], 4),
        "p25_f_pN": _round(avg["p25_f"][::stride], 4),
        "p75_f_pN": _round(avg["p75_f"][::stride], 4),
        "mean_x_A": _round(avg["mean_x"][::stride], 5),
        "force_rate": {
            "window_ps": fit["window_ps"],
            "slope_pN_per_ps": fit["slope"],
            "slope_se": fit["slope_se"],
            "intercept_pN": fit["intercept"],
            "r2": fit["r2"],
            "n_points": fit["n_points"],
            "t_fit": _round(fit["t_fit"], 3),
            "f_fit": _round(fit["f_fit"], 4),
        },
        "sensitivity": sens,
        "default_window_ps": settings.force_rate_window_ps,
    }


@app.get("/api/stats/events")
def events(smoothing: int | None = Query(None, ge=1, le=201), window: float | None = Query(None, gt=1.0, le=600.0)) -> Any:
    if smoothing is None and window is None:
        return _artifact("events.json")
    traces, t, ids = _cache.get()
    avg = st.ensemble_average(traces, settings.percentiles)
    fit = st.force_rate(t, avg["mean_f"], window or settings.force_rate_window_ps)
    ev = st.transition_events(t, traces, smoothing or settings.peak_smoothing_points, fit["slope"])
    ev["trace_ids"] = ids
    ev["histograms"] = {
        "t_peak_ps": st.histogram(np.array([e["t_peak_ps"] for e in ev["events"]])),
        "f_crit_from_rate_pN": st.histogram(np.array([e["f_crit_from_rate_pN"] for e in ev["events"]])),
        "f_peak_smoothed_pN": st.histogram(np.array([e["f_peak_smoothed_pN"] for e in ev["events"]])),
    }
    return ev


# ------------------------------------------------------------------ potential
@app.get("/api/potential/params")
def potential_params() -> Any:
    return _artifact("potential.json")


@app.get("/api/potential")
def potential(
    U: float = Query(..., gt=0, description="U‡ in pN·Å"),
    xi: float = Query(..., gt=0, description="ξ‡ in Å"),
    f: float = Query(0.0, ge=0, description="applied force in pN"),
    n: int = Query(241, ge=51, le=1001),
) -> Any:
    land = Landscape(U, xi)
    rate = read_json(ART / "force_rate.json", default={}) or {}
    fdot = rate.get("force_rate_pN_per_ps")
    tilt_forces = sorted({fr * land.f_c for fr in settings.tilt_fractions} | {min(f, land.f_c * 1.5)})
    c = potential_curves(land, n, tilt_forces, fdot)
    return {
        "parameters": c["parameters"],
        "xi_A": _round(c["xi_A"], 4),
        "U0_pN_A": _round(c["U0_pN_A"], 4),
        "F_pN": _round(c["F_pN"], 4),
        "F_numerical_pN": _round(c["F_numerical_pN"], 4),
        "force_check_max_abs_pN": c["force_check_max_abs_pN"],
        "inflection_xi_A": c["inflection_xi_A"],
        "selected": next(({**tl, "U": _round(tl["U"], 4)} for tl in c["tilts"] if abs(tl["f_pN"] - min(f, land.f_c * 1.5)) < 1e-9), None),
        "tilts": [{**tl, "U": _round(tl["U"], 4)} for tl in c["tilts"]],
        "force_rate_pN_per_ps": fdot,
    }


# ------------------------------------------------------------------ summary & charts
@app.get("/api/summary")
def summary() -> Any:
    insp = read_json(ART / "inspection.json", default=None)
    val = read_json(ART / "validation.json", default=None)
    rate = read_json(ART / "force_rate.json", default=None)
    ev = read_json(ART / "events.json", default=None)
    pot = read_json(ART / "potential.json", default=None)
    return {
        "paper": {
            "title": "Extracting rates and activation free energies of martensitic transitions using nanomechanical force statistics: theory, models, and analysis",
            "authors": "Arijit Maitra & M. P. Gururajan",
            "journal": "Scientific Reports 15:3008 (2025)",
            "doi": "10.1038/s41598-025-87324-w",
            "system": "Single-crystalline TiNi (Ti:Ni = 1:1), 20×10×10 cubic cells, 60×30×30 Å box, periodic",
            "ensemble": "NPT, T = 300 K, P_yy = P_zz = 1.013 bar (Nosé–Hoover), 1 fs timestep",
            "loading": "Homogeneous uniaxial extension along [100] at constant rate; max strain 2.5 %",
            "force_definition": "f(t) = (A_xx / n) σ_xx(t), force per atom",
        },
        "inspection": insp and {k: insp[k] for k in ("n_files", "total_size_mb", "rows_per_file_first", "dt_ps", "t_end_ps", "x_start_A", "x_end_A", "strain_end", "extension_rate_A_per_ps", "paper_rate_match")},
        "validation": val and val["collection"],
        "force_rate": rate and {k: rate.get(k) for k in ("force_rate_pN_per_ps", "force_rate_se", "window_ps", "r2", "mean_trace_max_pN", "mean_trace_argmax_ps", "n_traces")},
        "events": ev and {k: ev.get(k) for k in ("t_peak_ps", "f_crit_from_rate_pN", "f_peak_smoothed_pN", "smoothing_points", "n")},
        "potential": pot and {k: {"parameters": v["parameters"], "label": v["label"]} for k, v in pot["parameter_sets"].items()},
        "constants": {"kBT_pN_A": K_B_T_PN_ANGSTROM, "temperature_K": TEMPERATURE_K},
    }


@app.get("/api/charts")
def charts() -> Any:
    cdir = ART / "charts"
    if not cdir.exists():
        return {"charts": []}
    return {"charts": sorted(p.name for p in cdir.glob("*.png"))}


@app.get("/api/charts/{name}")
def chart_file(name: str) -> Any:
    p = (ART / "charts" / Path(name).name)
    if not p.exists() or p.suffix != ".png":
        raise HTTPException(404, "chart not found")
    return FileResponse(p, media_type="image/png")


# ================================================================== Version 2: inverse problem
from .inverse import density as _density  # noqa: E402
from .inverse.data import SOURCES, prepare_sample  # noqa: E402
from .inverse.model import LandscapeKinetics  # noqa: E402
from .inverse.runs import MODEL_VERSION, PHASES, InverseConfig, InverseRun, code_hash, list_runs, load_run  # noqa: E402


class _InverseService:
    """Holds the in-flight run (one at a time) and the progress tracker."""

    def __init__(self) -> None:
        self.current: InverseRun | None = None

    @property
    def running(self) -> bool:
        return self.current is not None and self.current.running

    def progress(self) -> dict[str, Any]:
        if self.current is not None:
            snap = self.current.progress.snapshot()
            snap["run_id"] = self.current.run_id
        else:
            # no run in this process: read the file left by the last one
            from .progress import ProgressTracker
            snap = ProgressTracker(ART / "inverse", PHASES).snapshot()
        snap["running"] = self.running
        return snap


_inverse = _InverseService()


@app.get("/api/inverse/spec")
def inverse_spec() -> Any:
    return {
        "model_version": MODEL_VERSION,
        "code_hash": code_hash(),
        "phases": PHASES,
        "defaults": InverseConfig().to_dict(),
        "sources": list(SOURCES),
        "paper_parameter_sets": PAPER_PARAMETER_SETS,
        "kBT_pN_A": K_B_T_PN_ANGSTROM,
        "temperature_K": TEMPERATURE_K,
        "paper_rates": PAPER_RATES,
        "fig3_readings_by_eye": FIG3_READINGS_BY_EYE,
    }


@app.get("/api/inverse/sample")
def inverse_sample(source: str = Query("rate_x_time"), smoothing: int | None = Query(None, ge=1, le=201), window: float | None = Query(None, gt=1.0, le=600.0)) -> Any:
    if source not in SOURCES:
        raise HTTPException(400, f"source must be one of {SOURCES}")
    try:
        smp = prepare_sample(settings, source, smoothing, window)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    f = smp.forces
    grid = np.linspace(0.0, max(f.max() * 1.25, 1.0), 300)
    return {**smp.summary(), "forces_pN": _round(f, 4), "histogram": _density.histogram_density(f), "kde": _density.kde(f, grid), "ecdf": _density.ecdf(f)}


@app.get("/api/inverse/model")
def inverse_model(
    U: float = Query(..., gt=0), xi: float = Query(..., gt=0), gamma0: float = Query(..., gt=0),
    fdot: float | None = Query(None, gt=0), n: int = Query(300, ge=50, le=1000),
) -> Any:
    rate = fdot
    if rate is None:
        fr = read_json(ART / "force_rate.json", default={}) or {}
        rate = fr.get("force_rate_pN_per_ps")
        if rate is None:
            raise HTTPException(404, "no force rate available — run the V1 pipeline or pass fdot")
    land = LandscapeKinetics(U, xi, gamma0)
    grid = np.linspace(0.0, land.f_c, n, endpoint=False)
    mom = land.moments_exact(rate)
    return {
        "parameters": land.to_dict(rate),
        "force_rate_pN_per_ps": rate,
        "f_pN": _round(grid, 4),
        "pdf": _round(land.pdf(grid, rate), 6),
        "cdf": _round(land.cdf(grid, rate), 6),
        "survival": _round(land.survival(grid, rate), 6),
        "log10_rate": _round(np.log10(np.maximum(land.rate(grid), 1e-300)), 4),
        "mean_eq8_pN": land.mean_eq8(rate),
        "mean_exact_pN": mom["mean"],
        "sd_exact_pN": mom["sd"],
        "mode_pN": land.mode(rate),
        "residual_mass": mom["residual_mass"],
        # the force spectrum ⟨f⟩(ḟ) for this parameter set across the paper's rates (forward, Eq. 8 and exact)
        "spectrum": [
            {"force_rate": r["force_rate_pN_per_ps"], "index": r["index"], "mean_eq8": land.mean_eq8(r["force_rate_pN_per_ps"]), **{f"{k}_exact": v for k, v in land.moments_exact(r["force_rate_pN_per_ps"], n=4001).items() if k in ("mean", "sd")}}
            for r in PAPER_RATES
        ],
    }


class InverseRunRequest(BaseModel):
    config: dict[str, Any] = {}
    run_id: str | None = None      # pass an existing id to resume it


@app.post("/api/inverse/run")
def inverse_run(req: InverseRunRequest) -> dict[str, Any]:
    if _inverse.running:
        raise HTTPException(409, "an inverse run is already in progress")
    if req.run_id is not None and not Path(req.run_id).name == req.run_id:
        raise HTTPException(400, "invalid run_id")
    try:
        cfg = InverseConfig.from_dict(req.config)
        if cfg.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not (1 <= cfg.bootstrap_n <= 5000 and 3 <= cfg.profile_points <= 101 and 0 <= cfg.synthetic_replicates <= 100 and 10 <= cfg.maxiter <= 100000):
            raise ValueError("configuration outside safe limits")
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"bad configuration: {exc}") from exc
    if req.run_id is not None:
        # resuming: use the stored configuration if there is one
        stored = load_run(settings, req.run_id, "config")
        if stored:
            cfg = InverseConfig.from_dict(stored["config"])
    run = InverseRun(settings, cfg, run_id=req.run_id)
    _inverse.current = run
    return {"run_id": run.run_async()}


@app.post("/api/inverse/cancel")
def inverse_cancel() -> dict[str, Any]:
    if _inverse.current is not None:
        _inverse.current.cancel()
    return {"cancelling": _inverse.running}


@app.get("/api/inverse/progress")
def inverse_progress() -> Any:
    return _inverse.progress()


@app.get("/api/inverse/runs")
def inverse_runs() -> Any:
    return list_runs(settings)


@app.get("/api/inverse/latest")
def inverse_latest() -> Any:
    lr = list_runs(settings)["latest"]
    if not lr:
        raise HTTPException(404, "no completed inverse run yet")
    res = load_run(settings, lr["run_id"])
    if res is None:
        raise HTTPException(404, "latest run has no result file")
    return res


@app.get("/api/inverse/runs/{run_id}")
def inverse_run_result(run_id: str) -> Any:
    res = load_run(settings, Path(run_id).name)
    if res is None:
        raise HTTPException(404, "run not found or not finished")
    return res


@app.get("/api/inverse/runs/{run_id}/{phase}")
def inverse_run_phase(run_id: str, phase: str) -> Any:
    allowed = {"config", "sample", "fit", "constrained", "hessian", "profiles", "bootstrap", "diagnostics", "sensitivity", "synthetic", "result"}
    if phase not in allowed:
        raise HTTPException(400, f"phase must be one of {sorted(allowed)}")
    res = load_run(settings, Path(run_id).name, phase)
    if res is None:
        raise HTTPException(404, "phase file not found")
    return res


# ================================================================== Version 3: forward Smoluchowski model
from .forward.runs import MODEL_VERSION as FWD_MODEL_VERSION, PARAMETER_SOURCES, PHASES as FWD_PHASES, ForwardConfig, ForwardRun, code_hash as fwd_code_hash, list_runs as fwd_list_runs, load_run as fwd_load_run, resolve_force_rate, resolve_parameters  # noqa: E402
from .forward.solver import BOUNDARIES, ForwardParams  # noqa: E402


class _ForwardService:
    def __init__(self) -> None:
        self.current: ForwardRun | None = None

    @property
    def running(self) -> bool:
        return self.current is not None and self.current.running

    def progress(self) -> dict[str, Any]:
        if self.current is not None:
            snap = self.current.progress.snapshot()
            snap["run_id"] = self.current.run_id
        else:
            from .progress import ProgressTracker
            snap = ProgressTracker(ART / "forward", FWD_PHASES).snapshot()
        snap["running"] = self.running
        return snap


_forward = _ForwardService()


@app.get("/api/forward/spec")
def forward_spec() -> Any:
    return {
        "model_version": FWD_MODEL_VERSION, "code_hash": fwd_code_hash(), "phases": FWD_PHASES,
        "defaults": ForwardConfig().to_dict(), "parameter_sources": list(PARAMETER_SOURCES), "boundaries": list(BOUNDARIES),
        "paper_parameter_sets": PAPER_PARAMETER_SETS, "paper_rates": PAPER_RATES, "fig3_readings_by_eye": FIG3_READINGS_BY_EYE,
        "kBT_pN_A": K_B_T_PN_ANGSTROM, "temperature_K": TEMPERATURE_K,
    }


@app.get("/api/forward/parameters")
def forward_parameters(source: str = Query("inverse:latest"), inverse_run_id: str | None = None, temperature_K: float = Query(300.0, gt=0, le=2000)) -> Any:
    """Resolve a parameter source to (U‡, ξ‡, Γ0, η, D) without running."""
    try:
        params, prov = resolve_parameters(settings, ForwardConfig(parameter_source=source, inverse_run_id=inverse_run_id, temperature_K=temperature_K))
        fdot, fprov = resolve_force_rate(settings, ForwardConfig())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400, str(exc)) from exc
    return {"parameters": params.to_dict(), "provenance": prov, "force_rate_pN_per_ps": fdot, "force_rate_provenance": fprov, "t_end_ps": params.f_c / fdot}


class ForwardRunRequest(BaseModel):
    config: dict[str, Any] = {}
    run_id: str | None = None


@app.post("/api/forward/run")
def forward_run(req: ForwardRunRequest) -> dict[str, Any]:
    if _forward.running:
        raise HTTPException(409, "a forward run is already in progress")
    if req.run_id is not None and Path(req.run_id).name != req.run_id:
        raise HTTPException(400, "invalid run_id")
    try:
        cfg = ForwardConfig.from_dict(req.config)
        cfg.settings()  # validates solver limits
        if cfg.parameter_source not in PARAMETER_SOURCES:
            raise ValueError(f"parameter_source must be one of {PARAMETER_SOURCES}")
        if not (50 <= cfg.n_cells <= 4000 and 500 <= cfg.n_steps <= 400000):
            raise ValueError("n_cells must be 50-4000 and n_steps 500-400000 for an interactive run")
        if cfg.parameter_source == "custom":
            ForwardParams(float(cfg.custom["U_barrier"]), float(cfg.custom["xi_barrier"]), float(cfg.custom["gamma0"]))
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(400, f"bad configuration: {exc}") from exc
    if req.run_id is not None:
        stored = fwd_load_run(settings, req.run_id, "config")
        if stored:
            cfg = ForwardConfig.from_dict(stored["config"])
    run = ForwardRun(settings, cfg, run_id=req.run_id)
    _forward.current = run
    return {"run_id": run.run_async()}


@app.post("/api/forward/cancel")
def forward_cancel() -> dict[str, Any]:
    if _forward.current is not None:
        _forward.current.cancel()
    return {"cancelling": _forward.running}


@app.get("/api/forward/progress")
def forward_progress() -> Any:
    return _forward.progress()


@app.get("/api/forward/runs")
def forward_runs() -> Any:
    return fwd_list_runs(settings)


@app.get("/api/forward/latest")
def forward_latest() -> Any:
    lr = fwd_list_runs(settings)["latest"]
    if not lr:
        raise HTTPException(404, "no completed forward run yet")
    res = fwd_load_run(settings, lr["run_id"])
    if res is None:
        raise HTTPException(404, "latest run has no result file")
    return res


@app.get("/api/forward/runs/{run_id}")
def forward_run_result(run_id: str) -> Any:
    res = fwd_load_run(settings, Path(run_id).name)
    if res is None:
        raise HTTPException(404, "run not found or not finished")
    return res


@app.get("/api/forward/runs/{run_id}/{phase}")
def forward_run_phase(run_id: str, phase: str) -> Any:
    allowed = {"config", "observables", "validation", "convergence", "spectrum", "result"}
    if phase not in allowed:
        raise HTTPException(400, f"phase must be one of {sorted(allowed)}")
    res = fwd_load_run(settings, Path(run_id).name, phase)
    if res is None:
        raise HTTPException(404, "phase file not found")
    return res


# =============================================================== Version 4 ==
# Inflect: upload a table, have it read, segmented and narrated. Nothing here
# knows about titanium nickel; the sample set is one session among any others.

from .inflect.ingest import read_table as inf_read_table                     # noqa: E402
from .inflect.sessions import (                                              # noqa: E402
    PHASES as INF_PHASES,
    Session,
    SessionConfig,
    list_sessions as inf_list_sessions,
    open_session as inf_open_session,
)

INF_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
INF_MAX_FILES = 2000


class _InflectService:
    """One ensemble comparison at a time, in the background."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.session_id: str | None = None
        self.state: dict[str, Any] = {"status": "idle", "done": 0, "total": 0, "message": ""}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, session: Session, cfg: SessionConfig) -> None:
        if self.running:
            raise RuntimeError("a comparison is already running")
        self._cancel.clear()
        self.session_id = session.id
        total = len(session.sources())
        self.state = {"status": "running", "done": 0, "total": total, "message": "reading files"}
        service = self

        class _Tick:
            """Just enough of a ProgressTracker for the session to report into."""

            def start_stage(self, stage: str, files_total: int = 0, message: str = "") -> None:
                service.state.update(total=files_total, done=0)

            def file_done(self, n: int | None = None) -> None:
                service.state["done"] = n if n is not None else service.state["done"] + 1

        def work() -> None:
            try:
                session.run_ensemble(cfg, progress=_Tick(), cancel=self._cancel)
                self.state.update(status="cancelled" if self._cancel.is_set() else "done",
                                  message="")
            except Exception as exc:                       # surfaced, never swallowed
                self.state.update(status="failed", message=f"{type(exc).__name__}: {exc}")

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def progress(self) -> dict[str, Any]:
        d = dict(self.state)
        d["running"] = self.running
        d["session_id"] = self.session_id
        total, done = d.get("total") or 0, d.get("done") or 0
        d["percent"] = round(100.0 * done / total, 1) if total else 0.0
        return d


_inflect = _InflectService()


def _inflect_session(session_id: str) -> Session:
    try:
        return inf_open_session(settings, Path(session_id).name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _sample_directory() -> Path:
    return Path(settings.data_dir).parent / "640TH_300K_comp_share"


@app.get("/api/inflect/spec")
def inflect_spec() -> Any:
    return {
        "phases": INF_PHASES,
        "defaults": SessionConfig().to_dict(),
        "max_upload_bytes": INF_MAX_UPLOAD_BYTES,
        "max_files": INF_MAX_FILES,
        "sample_available": _sample_directory().exists(),
    }


@app.get("/api/inflect/sessions")
def inflect_sessions() -> Any:
    return {"sessions": inf_list_sessions(settings)}


@app.post("/api/inflect/sessions")
async def inflect_create(files: list[UploadFile] = File(...), title: str = Form("")) -> Any:
    if len(files) > INF_MAX_FILES:
        raise HTTPException(400, f"at most {INF_MAX_FILES} files")
    session = Session.create(settings, title=title or (files[0].filename or "Upload"))
    total = 0
    accepted = 0
    problems: list[dict[str, str]] = []
    for f in files:
        data = await f.read()
        total += len(data)
        if total > INF_MAX_UPLOAD_BYTES:
            raise HTTPException(413, "upload exceeds the size limit")
        try:
            inf_read_table(data, name=f.filename or "file")
        except Exception as exc:
            problems.append({"name": f.filename or "file", "error": str(exc)})
            continue
        session.add_upload(f.filename or "file", data)
        accepted += 1
    if not accepted:
        raise HTTPException(400, "no readable table in the upload")
    return {"id": session.id, "accepted": accepted, "problems": problems,
            "session": session.meta()}


class InflectSampleRequest(BaseModel):
    directory: str | None = None
    pattern: str = "*.dat"
    title: str = "Sample: TiNi compression"


@app.post("/api/inflect/sample")
def inflect_sample(req: InflectSampleRequest) -> Any:
    """Open the bundled sample set without pushing 100 MB through a form."""
    directory = Path(req.directory) if req.directory else _sample_directory()
    if not directory.exists():
        raise HTTPException(404, f"no directory {directory}")
    session = Session.create(settings, title=req.title)
    added = session.add_directory(directory, req.pattern)
    if not added:
        raise HTTPException(404, f"no files matching {req.pattern} in {directory}")
    return {"id": session.id, "accepted": len(added), "session": session.meta()}


@app.get("/api/inflect/sessions/{session_id}")
def inflect_session(session_id: str) -> Any:
    s = _inflect_session(session_id)
    meta = s.meta()
    try:
        table, report = s.column_report()
    except Exception as exc:
        raise HTTPException(400, f"could not read the first file: {exc}") from exc
    return {
        "session": meta,
        "first_file": table.to_dict(),
        "columns": report.to_dict(),
        "files": [src.to_dict() for src in s.sources()],
        "has_ensemble": s.ensemble_path().exists(),
    }


@app.post("/api/inflect/sessions/{session_id}/config")
def inflect_set_config(session_id: str, config: dict[str, Any]) -> Any:
    s = _inflect_session(session_id)
    cfg = SessionConfig.from_dict({**s.config().to_dict(), **config})
    s.set_config(cfg)
    return {"config": cfg.to_dict()}


@app.get("/api/inflect/sessions/{session_id}/curve")
def inflect_curve(
    session_id: str,
    file: str | None = None,
    x_column: int | None = None,
    y_column: int | None = None,
    n_segments: int | None = Query(None, ge=1, le=12),
    penalty_scale: float = Query(1.0, gt=0.05, le=20.0),
    detect_transitions: bool = True,
    x_scale: float | None = None,
) -> Any:
    """Parse, segment and narrate one file. The page's main call."""
    s = _inflect_session(session_id)
    sources = s.sources()
    if not sources:
        raise HTTPException(400, "session has no files")
    name = file or sources[0].name
    base = s.config().to_dict()
    for key, value in (("x_column", x_column), ("y_column", y_column),
                       ("n_segments", n_segments), ("x_scale", x_scale)):
        if value is not None:
            base[key] = value
    base["penalty_scale"] = penalty_scale
    base["detect_transitions"] = detect_transitions
    cfg = SessionConfig.from_dict(base)
    try:
        return s.curve(name, cfg)
    except KeyError as exc:
        raise HTTPException(404, f"no file {name} in this session") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/inflect/sessions/{session_id}/columns")
def inflect_columns(session_id: str, file: str | None = None) -> Any:
    s = _inflect_session(session_id)
    try:
        table, report = s.column_report(file)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"file": table.to_dict(), "columns": report.to_dict()}


@app.post("/api/inflect/sessions/{session_id}/ensemble")
def inflect_run_ensemble(session_id: str) -> Any:
    s = _inflect_session(session_id)
    try:
        _inflect.start(s, s.config())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _inflect.progress()


@app.post("/api/inflect/ensemble/cancel")
def inflect_cancel_ensemble() -> Any:
    _inflect.cancel()
    return _inflect.progress()


@app.get("/api/inflect/ensemble/progress")
def inflect_ensemble_progress() -> Any:
    return _inflect.progress()


@app.get("/api/inflect/sessions/{session_id}/ensemble")
def inflect_ensemble(session_id: str) -> Any:
    s = _inflect_session(session_id)
    data = s.load_ensemble()
    if data is None:
        raise HTTPException(404, "no comparison has been run for this session")
    return data


# --- Version 4, phase 2: how sure is it, and what does the whole set look like

from .inflect.uncertainty import (                                           # noqa: E402
    DEFAULT_RESAMPLES as INF_RESAMPLES,
    bootstrap_curve as inf_bootstrap,
    stability_sweep as inf_stability,
)
from .inflect.segment import segment_curve as inf_segment                    # noqa: E402


def _inflect_xy(s: Session, file: str | None, cfg: SessionConfig):
    """The chosen curve of a session, with its labels."""
    sources = s.sources()
    if not sources:
        raise HTTPException(400, "session has no files")
    name = file or sources[0].name
    try:
        table = s.table(name)
    except KeyError as exc:
        raise HTTPException(404, f"no file {name} in this session") from exc
    xi, yi = s.resolve_columns(table, cfg)
    x = table.data[:, xi] * (cfg.x_scale or 1.0)
    y = table.data[:, yi]
    good = np.isfinite(x) & np.isfinite(y)
    return name, x[good], y[good], table.columns[xi], table.columns[yi]


def _inflect_config(s: Session, n_segments, penalty_scale, detect_transitions) -> SessionConfig:
    base = s.config().to_dict()
    if n_segments is not None:
        base["n_segments"] = n_segments
    base["penalty_scale"] = penalty_scale
    base["detect_transitions"] = detect_transitions
    return SessionConfig.from_dict(base)


@app.get("/api/inflect/sessions/{session_id}/uncertainty")
def inflect_uncertainty(
    session_id: str,
    file: str | None = None,
    n_segments: int | None = Query(None, ge=1, le=12),
    penalty_scale: float = Query(1.0, gt=0.05, le=20.0),
    detect_transitions: bool = True,
    n_resamples: int = Query(INF_RESAMPLES, ge=20, le=400),
    seed: int = 0,
) -> Any:
    """Bootstrap the whole reading and report intervals on what it found."""
    s = _inflect_session(session_id)
    cfg = _inflect_config(s, n_segments, penalty_scale, detect_transitions)
    name, x, y, xcol, ycol = _inflect_xy(s, file, cfg)
    try:
        seg = inf_segment(x, y, **cfg.segment_kwargs())
        result = inf_bootstrap(x, y, seg, n_resamples=n_resamples, seed=seed,
                               **cfg.segment_kwargs())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return json_safe({
        "file": name,
        "x_unit": cfg.x_unit or xcol.unit,
        "y_unit": cfg.y_unit or ycol.unit,
        **result.to_dict(),
    })


@app.get("/api/inflect/sessions/{session_id}/stability")
def inflect_stability(
    session_id: str,
    file: str | None = None,
    n_segments: int | None = Query(None, ge=1, le=12),
    penalty_scale: float = Query(1.0, gt=0.05, le=20.0),
    detect_transitions: bool = True,
) -> Any:
    """Re-derive the transition under every setting worth trying."""
    s = _inflect_session(session_id)
    cfg = _inflect_config(s, n_segments, penalty_scale, detect_transitions)
    name, x, y, xcol, ycol = _inflect_xy(s, file, cfg)
    try:
        seg = inf_segment(x, y, **cfg.segment_kwargs())
        result = inf_stability(x, y, seg, x_unit=cfg.x_unit or xcol.unit,
                               **cfg.segment_kwargs())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return json_safe({"file": name, **result.to_dict()})


@app.get("/api/inflect/sessions/{session_id}/chorus")
def inflect_chorus(session_id: str) -> Any:
    s = _inflect_session(session_id)
    data = s.load_chorus()
    if data is None:
        raise HTTPException(404, "read every file first")
    return data


@app.get("/api/inflect/sessions/{session_id}/results.csv")
def inflect_results_csv(session_id: str) -> Any:
    """Every file's numbers, for a spreadsheet or a supplementary table."""
    s = _inflect_session(session_id)
    report = s.load_ensemble()
    if report is None:
        raise HTTPException(404, "read every file first")
    cfg = s.config()
    xu = f" [{cfg.x_unit}]" if cfg.x_unit else ""
    yu = f" [{cfg.y_unit}]" if cfg.y_unit else ""
    head = ["file", "ok", "shape", "regimes", f"transition{xu}", f"onset{yu}",
            f"step{yu}", f"width{xu}", f"range{yu}", "initial slope", "noise", "error"]
    rows = [",".join(head)]
    for c in report.get("curves", []):
        def cell(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.6g}"
            text = str(v)
            return f'"{text}"' if ("," in text or '"' in text) else text
        rows.append(",".join(cell(c.get(k)) for k in (
            "name", "ok", "shape", "n_regimes", "transition_x", "transition_onset",
            "transition_step", "transition_width", "y_range", "first_slope", "noise", "error")))
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        "\n".join(rows) + "\n",
        media_type="text/csv",
        headers={"content-disposition": f'attachment; filename="inflect-{session_id}.csv"'},
    )


@app.get("/api/inflect/sessions/{session_id}/methods")
def inflect_methods(session_id: str, file: str | None = None) -> Any:
    """A methods paragraph describing exactly what was done to this data."""
    s = _inflect_session(session_id)
    cfg = s.config()
    meta = s.meta()
    report = s.load_ensemble()
    try:
        table, columns = s.column_report(file)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    xi, yi = s.resolve_columns(table, cfg)
    xname = table.columns[xi].label
    yname = table.columns[yi].label
    n_files = len(meta.get("sources", []))
    d = table.dialect

    regimes = ("a fixed " + str(cfg.n_segments) + " regimes"
               if cfg.n_segments else
               "a number of regimes chosen by a Bayesian information criterion")

    text = (
        f"Curve analysis was performed with Inflect (version 4). "
        f"{'A single file' if n_files == 1 else f'{n_files} files'} "
        f"({d.delimiter_name}-separated, {d.n_fields} columns, "
        f"{table.n_rows} rows) "
        f"{'was' if n_files == 1 else 'were'} read and {yname} was analysed "
        f"against {xname}. "
        f"Transitions were located before any regime was fitted, using a locally "
        f"detrended step statistic: straight lines fitted to the samples before "
        f"and after each point were extrapolated to it and subtracted, so that a "
        f"smoothly varying background — however steep or curved — cancels to "
        f"first order. The statistic was evaluated over a ladder of window "
        f"widths and judged against its own median absolute deviation "
        f"(threshold {cfg.event_threshold:g}). Candidates were rejected when the "
        f"apparent step grew in proportion to the window width, or when the "
        f"straight lines either side intersected within the candidate's own "
        f"span, both of which indicate a change of slope rather than a "
        f"discontinuity. "
        f"The smooth stretches between transitions were partitioned into "
        f"piecewise-polynomial regimes of degree at most two by exact dynamic "
        f"programming over candidate breakpoints, with {regimes}. "
        f"The value reported at a transition is that of the fitted regime at "
        f"its onset rather than the extremum of the trace, which on a noisy "
        f"measurement is a property of the noise and of the smoothing applied. "
        f"Uncertainties are 95 % percentile intervals from a moving-block "
        f"bootstrap of the residuals, with the block length taken from the "
        f"residuals' autocorrelation, re-running detection and partitioning in "
        f"full for each resample. "
        f"Robustness was checked by re-deriving each transition across a sweep "
        f"of detector thresholds, regime counts and penalties."
    )
    if report:
        stats = report.get("stats", {})
        tx = stats.get("transition_x", {})
        if tx.get("n"):
            unit = f" {cfg.x_unit}" if cfg.x_unit else ""
            text += (f" Across the set, {tx['n']} of {report.get('n_ok', 0)} curves showed a "
                     f"transition, at {tx['median']:.4g} ± {tx['spread']:.2g}{unit} "
                     f"(median ± robust spread).")
        if report.get("outliers"):
            names = ", ".join(o["name"] for o in report["outliers"][:6])
            text += (f" {len(report['outliers'])} file(s) were flagged as belonging to a "
                     f"different population ({names}).")

    return {"methods": text, "session": session_id}
