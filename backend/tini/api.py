"""FastAPI application.

Read endpoints serve the checkpointed artifacts; a small mtime-keyed cache
avoids re-reading traces.npz on every request. The only computation done on
request is cheap and pure: re-fitting the force rate for a different window,
re-running the peak picker for a different smoothing width, and evaluating
the cubic landscape for slider values.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import statistics as st
from .config import K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS, PAPER_RATES, TEMPERATURE_K, load_settings
from .io_atomic import read_json
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
