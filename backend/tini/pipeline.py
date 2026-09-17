"""The processing pipeline: six resumable stages.

    inspect     → inspection.json
    validate    → validation.json, per-file records
    preprocess  → traces/<id>.npy (per file, resumable), traces.npz, files.csv
    statistics  → average_curves.csv, force_rate.json, events.json, events.csv
    potential   → potential.json, potential_curves.csv
    charts      → charts/*.png

Each stage is keyed by a fingerprint (settings that affect it + dataset
manifest). A stage whose fingerprint matches and whose outputs exist is
skipped. ``preprocess`` additionally checkpoints per file, so a run killed
half-way resumes at the next unprocessed file.

The pipeline is intentionally sequential and single-threaded: the whole
dataset is 4 MB and the work is dominated by file I/O and chart rendering,
so parallelism would only add memory and complexity.
"""

from __future__ import annotations

import csv
import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import charts as ch
from . import statistics as st
from .checkpoints import CheckpointRegistry, StageRecord
from .config import PAPER_PARAMETER_SETS, Settings
from .dataset import TraceFile, dataset_manifest_hash, discover_files, inspect_dataset, load_trace
from .io_atomic import atomic_write_json, atomic_write_npy, atomic_write_npz, atomic_write_text, read_json
from .potential import landscape_from_paper, potential_curves
from .progress import STATUS_CANCELLED, STATUS_DONE, STATUS_FAILED, ProgressTracker
from .validation import validate_collection, validate_trace

STAGES = ["inspect", "validate", "preprocess", "statistics", "potential", "charts"]

# which settings each stage's fingerprint depends on
_STAGE_KEYS: dict[str, tuple[str, ...]] = {
    "inspect": ("file_glob", "max_files", "config_version"),
    "validate": ("file_glob", "max_files", "expected_dt_ps", "dt_rel_tol", "x_plausible_range_A", "config_version"),
    "preprocess": ("file_glob", "max_files", "config_version"),
    "statistics": (
        "file_glob", "max_files", "force_rate_window_ps", "force_rate_windows_ps",
        "peak_smoothing_points", "percentiles", "config_version",
    ),
    "potential": ("default_parameter_set", "potential_grid_points", "tilt_fractions", "force_rate_window_ps", "config_version"),
    "charts": ("chart_dpi", "force_rate_window_ps", "peak_smoothing_points", "config_version"),
}


# which source modules each stage's computation lives in; their content hash
# is part of the fingerprint so editing the physics re-runs the stage.
_STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "inspect": ("dataset",),
    "validate": ("dataset", "validation"),
    "preprocess": ("dataset",),
    "statistics": ("statistics",),
    "potential": ("potential", "config"),
    "charts": ("charts", "statistics", "potential"),
}


def _code_hash(modules: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for m in modules:
        h.update((here / f"{m}.py").read_bytes())
    return h.hexdigest()[:8]


class Cancelled(Exception):
    pass


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.art = settings.artifacts_path
        self.art.mkdir(parents=True, exist_ok=True)
        self.registry = CheckpointRegistry(self.art)
        self.progress = ProgressTracker(self.art, STAGES)
        self.cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._files: list[TraceFile] | None = None
        self._manifest: str | None = None

    # ------------------------------------------------------------ helpers
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def files(self) -> list[TraceFile]:
        if self._files is None:
            self._files = discover_files(self.settings)
            self._manifest = dataset_manifest_hash(self._files)
        return self._files

    def fingerprint(self, stage: str) -> str:
        self.files()
        return f"{self.settings.fingerprint(_STAGE_KEYS[stage])}-{self._manifest}-{_code_hash(_STAGE_MODULES[stage])}"

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def _complete(self, stage: str, t0: float, outputs: list[Path], summary: dict[str, Any] | None = None) -> None:
        rec = StageRecord(
            stage=stage,
            fingerprint=self.fingerprint(stage),
            completed_at=time.time(),
            duration_s=round(time.time() - t0, 3),
            outputs=[self.registry.rel(p) for p in outputs],
            summary=summary or {},
        )
        self.registry.mark_complete(rec)
        self.progress.end_stage(stage, rec.outputs[0] if rec.outputs else None)

    # ------------------------------------------------------------ control
    def run_async(self, force: bool = False, stages: list[str] | None = None) -> str:
        if self.running:
            raise RuntimeError("pipeline already running")
        self.cancel_event.clear()
        run_id = uuid.uuid4().hex[:8]
        self._thread = threading.Thread(target=self.run, kwargs=dict(force=force, stages=stages, run_id=run_id), daemon=True)
        self._thread.start()
        return run_id

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self, force: bool = False, stages: list[str] | None = None, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or uuid.uuid4().hex[:8]
        wanted = stages or STAGES
        self._files = None  # re-discover so a changed dataset is noticed
        self.files()
        if force:
            for s in wanted:
                self.registry.invalidate(s)
        resumed = [s for s in STAGES if s not in wanted or self.registry.is_complete(s, self.fingerprint(s))]
        self.progress.start_run(run_id, resumed)
        atomic_write_json(self.art / "config_used.json", {"run_id": run_id, "settings": self.settings.to_dict()})

        runners: dict[str, Callable[[], None]] = {
            "inspect": self.stage_inspect,
            "validate": self.stage_validate,
            "preprocess": self.stage_preprocess,
            "statistics": self.stage_statistics,
            "potential": self.stage_potential,
            "charts": self.stage_charts,
        }
        current = None
        try:
            for stage in STAGES:
                if stage not in wanted or stage in resumed:
                    continue
                current = stage
                self._check_cancel()
                runners[stage]()
            self.progress.finish(STATUS_DONE, "All stages complete")
        except Cancelled:
            self.progress.finish(STATUS_CANCELLED, f"Cancelled during {current}; checkpoints kept — resume to continue")
        except Exception as exc:  # noqa: BLE001
            if current:
                self.progress.error(current, exc)
            self.progress.finish(STATUS_FAILED, f"{type(exc).__name__} in {current}: {exc}")
        return self.progress.snapshot()

    # ------------------------------------------------------------ stages
    def stage_inspect(self) -> None:
        t0 = time.time()
        self.progress.start_stage("inspect", 0, "Inspecting files and estimating memory")
        info = inspect_dataset(self.settings)
        info["dataset_manifest"] = self._manifest
        out = self.art / "inspection.json"
        atomic_write_json(out, info)
        self._complete("inspect", t0, [out], {"n_files": info["n_files"], "total_size_mb": info["total_size_mb"]})

    def stage_validate(self) -> None:
        t0 = time.time()
        files = self.files()
        self.progress.start_stage("validate", len(files), "Validating traces")
        results, grids = [], []
        for i, f in enumerate(files):
            self._check_cancel()
            arr = load_trace(f.path)
            r = validate_trace(f.trace_id, f.name, arr, self.settings)
            results.append(r)
            if r.ok:
                grids.append(arr[:, 0])
            self.progress.file_done(i + 1)
        coll = validate_collection(results, grids)
        out = self.art / "validation.json"
        atomic_write_json(out, {"collection": coll, "files": [r.to_dict() for r in results]})
        self._complete("validate", t0, [out], {"n_ok": coll["n_ok"], "n_failed": coll["n_failed"], "ok": coll["ok"]})

    def stage_preprocess(self) -> None:
        """Parse every valid file to a compact .npy (per-file checkpoint) then
        stack. Re-running after a crash skips files already converted."""
        t0 = time.time()
        files = self.files()
        val = read_json(self.art / "validation.json", default={}) or {}
        ok_ids = {r["trace_id"] for r in val.get("files", []) if r["ok"]} if val else {f.trace_id for f in files}
        todo = [f for f in files if f.trace_id in ok_ids]
        tdir = self.art / "traces"
        tdir.mkdir(exist_ok=True)
        done_path = self.art / "completed_files.json"
        done: dict[str, Any] = read_json(done_path, default={}) or {}
        fp = self.fingerprint("preprocess")
        if done.get("fingerprint") != fp:
            done = {"fingerprint": fp, "files": {}}

        self.progress.start_stage("preprocess", len(todo), "Converting traces to binary")
        n_done = 0
        for i, f in enumerate(todo):
            self._check_cancel()
            npy = tdir / f"{f.trace_id}.npy"
            if f.trace_id in done["files"] and npy.exists():
                n_done += 1
            else:
                arr = load_trace(f.path)
                atomic_write_npy(npy, arr.astype(np.float64), keep_previous=False)
                done["files"][f.trace_id] = {"name": f.name, "rows": int(arr.shape[0]), "time": time.time()}
                n_done += 1
                if n_done % self.settings.batch_size == 0:
                    atomic_write_json(done_path, done)
            self.progress.file_done(i + 1)
        atomic_write_json(done_path, done)

        self.progress.message("Stacking traces")
        arrs = [np.load(tdir / f"{f.trace_id}.npy") for f in todo]
        stacked = np.stack(arrs)                          # (n, samples, 3)
        ids = np.array([f.trace_id for f in todo])
        out = self.art / "traces.npz"
        atomic_write_npz(out, traces=stacked, trace_ids=ids, time_ps=stacked[0, :, 0])

        csv_out = self.art / "files.csv"
        rows = ["trace_id,name,rows,size_bytes,x_start_A,x_end_A,f_max_pN,t_at_raw_max_ps"]
        for f, a in zip(todo, arrs):
            rows.append(
                f"{f.trace_id},{f.name},{a.shape[0]},{f.size_bytes},{a[0,1]:.6f},{a[-1,1]:.6f},{a[:,2].max():.6f},{a[int(np.argmax(a[:,2])),0]:.3f}"
            )
        atomic_write_text(csv_out, "\n".join(rows) + "\n")
        self._complete("preprocess", t0, [out, csv_out, done_path], {"n_traces": int(stacked.shape[0]), "shape": list(stacked.shape)})

    def _load_traces(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        z = np.load(self.art / "traces.npz")
        return z["traces"], z["time_ps"], [str(s) for s in z["trace_ids"]]

    def stage_statistics(self) -> None:
        t0 = time.time()
        s = self.settings
        self.progress.start_stage("statistics", 0, "Averaging traces and estimating the force rate")
        traces, t, ids = self._load_traces()
        avg = st.ensemble_average(traces, s.percentiles)
        fit = st.force_rate(t, avg["mean_f"], s.force_rate_window_ps)
        sens = st.force_rate_sensitivity(t, avg["mean_f"], s.force_rate_windows_ps)
        per_trace = st.per_trace_rates(t, traces, s.force_rate_window_ps)
        ext = np.array([np.polyfit(t, tr[:, 1], 1)[0] for tr in traces])

        # average curves as CSV (one row per time sample)
        cols = ["time_ps", "mean_x_A", "std_x_A", "mean_f_pN", "std_f_pN", "sem_f_pN", "median_f_pN"] + [f"p{p:g}_f_pN" for p in s.percentiles]
        data = np.column_stack([t, avg["mean_x"], avg["std_x"], avg["mean_f"], avg["std_f"], avg["sem_f"], avg["median_f"]] + [avg[f"p{p:g}_f"] for p in s.percentiles])
        avg_out = self.art / "average_curves.csv"
        atomic_write_text(avg_out, ",".join(cols) + "\n" + "\n".join(",".join(f"{v:.6g}" for v in row) for row in data) + "\n")

        rate_out = self.art / "force_rate.json"
        atomic_write_json(
            rate_out,
            {
                "method": "linear least-squares fit of ⟨f(t)⟩ on t ∈ [0, window]; slope = ḟ (paper Methods, Eq. 2)",
                "assumption": "window length is a prototype assumption; paper says 'small time interval near t = 0'",
                "window_ps": fit["window_ps"],
                "force_rate_pN_per_ps": fit["slope"],
                "force_rate_se": fit["slope_se"],
                "intercept_pN": fit["intercept"],
                "r2": fit["r2"],
                "n_points": fit["n_points"],
                "sensitivity": sens,
                "per_trace_rate": {"mean": float(per_trace.mean()), "std": float(per_trace.std(ddof=1)), "values": per_trace.tolist()},
                "extension_rate_A_per_ps": {"mean": float(ext.mean()), "std": float(ext.std(ddof=1))},
                "mean_trace_max_pN": float(avg["mean_f"].max()),
                "mean_trace_argmax_ps": float(t[int(np.argmax(avg["mean_f"]))]),
                "n_traces": int(traces.shape[0]),
            },
        )

        self.progress.message("Detecting transition events (prototype)")
        ev = st.transition_events(t, traces, s.peak_smoothing_points, fit["slope"])
        ev["trace_ids"] = ids
        ev["histograms"] = {
            "t_peak_ps": st.histogram(np.array([e["t_peak_ps"] for e in ev["events"]])),
            "f_crit_from_rate_pN": st.histogram(np.array([e["f_crit_from_rate_pN"] for e in ev["events"]])),
            "f_peak_smoothed_pN": st.histogram(np.array([e["f_peak_smoothed_pN"] for e in ev["events"]])),
        }
        ev["assumption"] = "Peak picker (centred moving average, then argmax) is a prototype choice; the paper does not specify one."
        ev_out = self.art / "events.json"
        atomic_write_json(ev_out, ev)
        ev_csv = self.art / "events.csv"
        with open(ev_csv.with_suffix(".tmp"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            keys = list(ev["events"][0].keys())
            w.writerow(["trace_id"] + keys)
            for tid, e in zip(ids, ev["events"]):
                w.writerow([tid] + [e[k] for k in keys])
        ev_csv.with_suffix(".tmp").replace(ev_csv)

        self._complete(
            "statistics", t0, [avg_out, rate_out, ev_out, ev_csv],
            {"force_rate_pN_per_ps": fit["slope"], "mean_t_peak_ps": ev["t_peak_ps"]["mean"], "mean_f_crit_pN": ev["f_crit_from_rate_pN"]["mean"]},
        )

    def stage_potential(self) -> None:
        t0 = time.time()
        s = self.settings
        self.progress.start_stage("potential", 0, "Constructing the cubic landscape")
        rate = read_json(self.art / "force_rate.json", default={}) or {}
        fdot = rate.get("force_rate_pN_per_ps")
        result: dict[str, Any] = {"parameter_sets": {}, "default_parameter_set": s.default_parameter_set, "force_rate_pN_per_ps": fdot}
        csv_rows: list[str] = []
        for key in PAPER_PARAMETER_SETS:
            land = landscape_from_paper(key)
            tilt_forces = [fr * land.f_c for fr in s.tilt_fractions]
            curves = potential_curves(land, s.potential_grid_points, tilt_forces, fdot)
            result["parameter_sets"][key] = {
                "label": PAPER_PARAMETER_SETS[key]["label"],
                "source": PAPER_PARAMETER_SETS[key],
                "parameters": curves["parameters"],
                "force_check_max_abs_pN": curves["force_check_max_abs_pN"],
                "inflection_xi_A": curves["inflection_xi_A"],
                "tilts": [{k: v for k, v in tl.items() if k != "U"} for tl in curves["tilts"]],
            }
            for xi, u, F in zip(curves["xi_A"], curves["U0_pN_A"], curves["F_pN"]):
                csv_rows.append(f"{key},{xi:.6g},{u:.6g},{F:.6g}")
        out = self.art / "potential.json"
        atomic_write_json(out, result)
        csv_out = self.art / "potential_curves.csv"
        atomic_write_text(csv_out, "parameter_set,xi_A,U0_pN_A,F_pN\n" + "\n".join(csv_rows) + "\n")
        self._complete("potential", t0, [out, csv_out], {"f_c_pN": result["parameter_sets"][s.default_parameter_set]["parameters"]["f_c_pN"]})

    def stage_charts(self) -> None:
        t0 = time.time()
        s = self.settings
        self.progress.start_stage("charts", 5, "Rendering static charts")
        traces, t, ids = self._load_traces()
        avg = st.ensemble_average(traces, s.percentiles)
        fit = st.force_rate(t, avg["mean_f"], s.force_rate_window_ps)
        cdir = self.art / "charts"
        outs = []
        outs.append(Path(ch.chart_traces_overlay(t, traces, ids, cdir / "traces_overlay.png", s.chart_dpi)))
        self.progress.file_done(1)
        outs.append(Path(ch.chart_mean_trace(t, avg, fit, cdir / "mean_trace.png", s.chart_dpi)))
        self.progress.file_done(2)
        outs.append(Path(ch.chart_force_vs_x(avg, cdir / "force_vs_x.png", s.chart_dpi)))
        self.progress.file_done(3)
        land = landscape_from_paper(s.default_parameter_set)
        curves = potential_curves(land, s.potential_grid_points, [fr * land.f_c for fr in s.tilt_fractions], fit["slope"])
        outs.append(Path(ch.chart_potential(curves, cdir / "potential.png", s.chart_dpi)))
        self.progress.file_done(4)
        ev = read_json(self.art / "events.json", default=None)
        if ev:
            outs.append(Path(ch.chart_events(ev, cdir / "events.png", s.chart_dpi)))
        self.progress.file_done(5)
        self._complete("charts", t0, outs, {"n_charts": len(outs)})
