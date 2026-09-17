"""Forward-model runs: configuration, orchestration, storage.

Mirrors ``tini.inverse.runs``: a run is ``artifacts/forward/runs/<run_id>/``
with one JSON file per phase (plus ``solution.npz`` for the full arrays),
written atomically when the phase completes; re-running a run_id resumes.

    config.json        ForwardConfig, resolved parameters, fingerprints
    solution.npz       t, Ψ, absorbed, ρ, snapshots of P(ξ,t), grid
    observables.json   p(f|ḟ) on a display grid, moments, comparison with Eq. 6 and the V1 sample
    validation.json    tests A–C, F, G on this run's parameters
    convergence.json   grid and time refinement around this run's settings
    spectrum.json      ⟨f‡⟩(ḟ) from the PDE at the paper's rates vs Eq. 8 / exact Eq. 6
    result.json        summary for the UI with provenance
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import FIG3_READINGS_BY_EYE, K_B_J_PER_K, K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS, PAPER_RATES, Settings, TEMPERATURE_K
from ..inverse.data import prepare_sample
from ..inverse.density import histogram_density
from ..inverse.model import LandscapeKinetics
from ..inverse.runs import list_runs as list_inverse_runs, load_run as load_inverse_run
from ..io_atomic import atomic_write_json, atomic_write_npz, read_json
from ..progress import STATUS_CANCELLED, STATUS_DONE, STATUS_FAILED, ProgressTracker
from . import validation as V
from .solver import ForwardParams, SolverSettings, force_density, solve

MODEL_VERSION = "smoluchowski-fv-sg-be-1"
PHASES = ["prepare", "solve", "observables", "validation", "convergence", "spectrum", "finalize"]
PHASE_FILES = {**{p: p for p in PHASES}, "prepare": "config", "solve": "solution", "finalize": "result"}
PARAMETER_SOURCES = ("paper:pdf_mle", "paper:mean_lsq", "inverse:latest", "inverse:run", "custom")


class Cancelled(Exception):
    pass


@dataclass
class ForwardConfig:
    parameter_source: str = "inverse:latest"
    inverse_run_id: str | None = None          # with parameter_source = "inverse:run"
    custom: dict[str, float] = field(default_factory=dict)   # U_barrier, xi_barrier, gamma0
    force_rate: float | None = None            # None → V1 measured ḟ
    temperature_K: float = TEMPERATURE_K       # η held at its 300 K value; k_BT and hence D, Γ0 follow
    # solver
    n_cells: int = 250
    n_steps: int = 20000
    theta: float = 1.0
    boundary: str = "beyond_barrier"
    margin_kBT: float = 4.0
    xi_left_factor: float = 0.6
    n_snapshots: int = 80
    # extras
    validation_enabled: bool = True
    convergence_enabled: bool = True
    spectrum_enabled: bool = True
    compare_source: str = "rate_x_time"        # V1 sample definition for the data comparison
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def settings(self) -> SolverSettings:
        return SolverSettings(n_cells=self.n_cells, n_steps=self.n_steps, theta=self.theta, boundary=self.boundary, margin_kBT=self.margin_kBT, xi_left_factor=self.xi_left_factor, n_snapshots=self.n_snapshots)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForwardConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for p in sorted(here.glob("*.py")):
        h.update(p.read_bytes())
    h.update((here.parent / "potential.py").read_bytes())
    h.update((here.parent / "inverse" / "model.py").read_bytes())
    return h.hexdigest()[:12]


def resolve_parameters(settings: Settings, cfg: ForwardConfig) -> tuple[ForwardParams, dict[str, Any]]:
    """Turn the configured source into (U‡, ξ‡, Γ0) with provenance."""
    src = cfg.parameter_source
    if src not in PARAMETER_SOURCES:
        raise ValueError(f"parameter_source must be one of {PARAMETER_SOURCES}")
    if src.startswith("paper:"):
        key = src.split(":", 1)[1]
        ps = PAPER_PARAMETER_SETS[key]
        vals = {"U_barrier": ps["U_barrier_pN_A"], "xi_barrier": ps["xi_barrier_A"], "gamma0": ps["gamma0_per_ps"]}
        prov = {"source": src, "label": ps["label"]}
    elif src.startswith("inverse:"):
        run_id = cfg.inverse_run_id if src == "inverse:run" else (list_inverse_runs(settings)["latest"] or {}).get("run_id")
        if not run_id:
            raise FileNotFoundError("no inverse run available — run Version 2 first or choose paper parameters")
        res = load_inverse_run(settings, run_id)
        if not res:
            raise FileNotFoundError(f"inverse run {run_id} has no result")
        vals = dict(res["fit"]["params"])
        prov = {"source": src, "inverse_run_id": run_id, "label": f"Version 2 maximum likelihood, run {run_id}", "inverse_sample_n": res["sample"]["n"], "inverse_converged": res["fit"]["converged"]}
    else:
        missing = [k for k in ("U_barrier", "xi_barrier", "gamma0") if k not in cfg.custom]
        if missing:
            raise ValueError(f"custom parameters missing: {missing}")
        vals = {k: float(cfg.custom[k]) for k in ("U_barrier", "xi_barrier", "gamma0")}
        prov = {"source": "custom", "label": "user-specified"}
    # temperature: η is a property of the dynamics, fixed at its 300 K value; k_BT scales with T
    kBT_ref = K_B_T_PN_ANGSTROM
    base = ForwardParams(vals["U_barrier"], vals["xi_barrier"], vals["gamma0"], kBT_ref)
    if abs(cfg.temperature_K - TEMPERATURE_K) > 1e-9:
        kBT = K_B_J_PER_K * cfg.temperature_K * 1e22
        eta = base.eta
        # Γ0(T) from Kramers with the same η: Γ0 = κ e^{−U‡/k_BT} / (2πη)
        gamma_T = base.curvature * np.exp(-vals["U_barrier"] / kBT) / (2 * np.pi * eta)
        params = ForwardParams(vals["U_barrier"], vals["xi_barrier"], float(gamma_T), kBT)
        prov["temperature_note"] = f"η held at its {TEMPERATURE_K:g} K value ({eta:.4g} pN·ps/Å); Γ0 rescaled by Kramers to {gamma_T:.3e} /ps at {cfg.temperature_K:g} K — an assumption, not the paper's"
    else:
        params = base
    return params, prov


def resolve_force_rate(settings: Settings, cfg: ForwardConfig) -> tuple[float, dict[str, Any]]:
    if cfg.force_rate is not None:
        if cfg.force_rate <= 0:
            raise ValueError("force_rate must be positive")
        return float(cfg.force_rate), {"source": "custom"}
    fr = read_json(settings.artifacts_path / "force_rate.json", default=None)
    if not fr:
        raise FileNotFoundError("no V1 force rate — run the Version 1 pipeline or pass force_rate")
    return float(fr["force_rate_pN_per_ps"]), {"source": "v1", "window_ps": fr.get("window_ps"), "se": fr.get("force_rate_se")}


def _display(a: np.ndarray, n: int = 400) -> list[float]:
    a = np.asarray(a, dtype=float)
    if a.size <= n:
        return [float(v) for v in a]
    idx = np.linspace(0, a.size - 1, n).astype(int)
    return [float(v) for v in a[idx]]


class ForwardRun:
    def __init__(self, settings: Settings, config: ForwardConfig, run_id: str | None = None):
        self.settings = settings
        self.config = config
        self.root = settings.artifacts_path / "forward"
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = self.root / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressTracker(self.root, PHASES)
        self.cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.params: ForwardParams | None = None
        self.force_rate: float | None = None
        self.sol = None

    # ------------------------------------------------------------ helpers
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def _check(self) -> None:
        if self._cancelled():
            raise Cancelled()

    def _path(self, phase: str) -> Path:
        name = PHASE_FILES.get(phase, phase)
        return self.dir / (f"{name}.npz" if phase == "solve" else f"{name}.json")

    def _have(self, phase: str) -> Any:
        p = self._path(phase)
        if phase == "solve":
            return p if p.exists() else None
        return read_json(p, default=None)

    def _save(self, phase: str, obj: Any) -> None:
        atomic_write_json(self._path(phase), obj, keep_previous=True)
        self.progress.end_stage(phase, f"forward/runs/{self.run_id}/{self._path(phase).name}")

    def run_async(self) -> str:
        if self.running:
            raise RuntimeError("run already in progress")
        self.cancel_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self.run_id

    def run(self) -> dict[str, Any]:
        resumed = [p for p in PHASES if self._have(p) is not None]
        self.progress.start_run(self.run_id, resumed)
        current = None
        try:
            for phase in PHASES:
                current = phase
                self._check()
                getattr(self, f"phase_{phase}")()
            self.progress.finish(STATUS_DONE, f"Run {self.run_id} complete")
            atomic_write_json(self.root / "latest.json", {"run_id": self.run_id, "completed_at": time.time()})
            self._index_add()
        except (Cancelled, InterruptedError):
            self.progress.finish(STATUS_CANCELLED, f"Cancelled during {current}; completed phases are kept — rerun the same run_id to resume")
        except Exception as exc:  # noqa: BLE001
            self.progress.error(current or "run", exc)
            self.progress.finish(STATUS_FAILED, f"{type(exc).__name__} in {current}: {exc}")
        return self.progress.snapshot()

    def _index_add(self) -> None:
        idx_path = self.root / "runs.json"
        idx = read_json(idx_path, default={"runs": []}) or {"runs": []}
        res = self._have("finalize") or {}
        entry = {
            "run_id": self.run_id, "completed_at": time.time(), "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "parameter_source": self.config.parameter_source, "boundary": self.config.boundary,
            "force_rate": res.get("force_rate_pN_per_ps"), "mean_pN": (res.get("observables") or {}).get("mean_pN"),
            "n_cells": self.config.n_cells, "n_steps": self.config.n_steps,
        }
        idx["runs"] = [r for r in idx["runs"] if r["run_id"] != self.run_id] + [entry]
        atomic_write_json(idx_path, idx)

    # ------------------------------------------------------------ phases
    def _load_params(self) -> None:
        if self.params is None:
            cfgdoc = self._have("prepare")
            if cfgdoc:
                p = cfgdoc["parameters"]
                self.params = ForwardParams(p["U_barrier_pN_A"], p["xi_barrier_A"], p["gamma0_per_ps"], p["kBT_pN_A"])
                self.force_rate = cfgdoc["force_rate_pN_per_ps"]
            else:
                self.params, _ = resolve_parameters(self.settings, self.config)
                self.force_rate, _ = resolve_force_rate(self.settings, self.config)

    def phase_prepare(self) -> None:
        if self._have("prepare") is not None:
            self._load_params()
            return
        self.progress.start_stage("prepare", 0, "Resolving parameters and force rate")
        params, prov = resolve_parameters(self.settings, self.config)
        fdot, fprov = resolve_force_rate(self.settings, self.config)
        self.params, self.force_rate = params, fdot
        st = self.config.settings()
        from .solver import make_grid
        grid, binfo = make_grid(params, st)
        self._save("prepare", {
            "run_id": self.run_id, "config": self.config.to_dict(), "model_version": self.config.model_version, "code_hash": code_hash(),
            "parameters": params.to_dict(), "parameter_provenance": prov, "force_rate_pN_per_ps": fdot, "force_rate_provenance": fprov,
            "grid": grid.to_dict(), "boundary": binfo, "t_end_ps": params.f_c / fdot, "dt_ps": params.f_c / fdot / st.n_steps,
            "created_at": time.time(), "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def _load_solution(self):
        if self.sol is None:
            from .solver import ForwardSolution, Grid
            z = np.load(self._path("solve"))
            self._load_params()
            assert self.params is not None
            st = self.config.settings()
            grid = Grid(float(z["xi_L"]), float(z["xi_A"]), int(z["n_cells"]))
            self.sol = ForwardSolution(
                t=z["t"], survival=z["survival"], absorbed=z["absorbed"], flux=z["flux"], snapshot_t=z["snapshot_t"], snapshots=z["snapshots"],
                grid=grid, params=self.params, settings=st, force_rate=float(z["force_rate"]), min_density=float(z["min_density"]),
                conservation_error_max=float(z["conservation_error_max"]), duration_s=float(z["duration_s"]),
            )
        return self.sol

    def phase_solve(self) -> None:
        if self._have("solve") is not None:
            self._load_solution()
            return
        self._load_params()
        assert self.params is not None and self.force_rate is not None
        st = self.config.settings()
        self.progress.start_stage("solve", st.n_steps, f"Integrating the Smoluchowski equation ({st.n_cells} cells × {st.n_steps} steps)")
        sol = solve(self.params, self.force_rate, st, progress=lambda i, n: self.progress.file_done(i), cancel=self._cancelled)
        self.sol = sol
        atomic_write_npz(
            self._path("solve"), keep_previous=True,
            t=sol.t, survival=sol.survival, absorbed=sol.absorbed, flux=sol.flux, snapshot_t=sol.snapshot_t, snapshots=sol.snapshots,
            centers=sol.grid.centers, xi_L=sol.grid.xi_L, xi_A=sol.grid.xi_A, n_cells=sol.grid.n, force_rate=sol.force_rate,
            min_density=sol.min_density, conservation_error_max=sol.conservation_error_max, duration_s=sol.duration_s,
        )
        self.progress.end_stage("solve", f"forward/runs/{self.run_id}/solution.npz")

    def phase_observables(self) -> None:
        if self._have("observables") is not None:
            return
        sol = self._load_solution()
        p = self.params
        assert p is not None
        self.progress.start_stage("observables", 0, "Critical-force density and comparison")
        fd = force_density(sol)
        fdot = sol.force_rate
        lk = LandscapeKinetics(p.U_barrier, p.xi_barrier, p.gamma0, p.kBT)
        f = fd["f_pN"]
        stride = max(1, sol.snapshots.shape[0] // 40)
        p6 = lk.pdf(f, fdot)
        cdf6 = lk.cdf(f, fdot)
        m6 = lk.moments_exact(fdot)
        doc: dict[str, Any] = {
            "force_rate_pN_per_ps": fdot, "t_end_ps": float(sol.t[-1]),
            "f_pN": _display(f), "pdf": _display(fd["pdf"]), "cdf": _display(fd["cdf"]), "survival_t": _display(sol.survival), "t_ps": _display(sol.t), "flux": _display(sol.flux),
            "pdf_eq6": _display(p6), "cdf_eq6": _display(cdf6),
            "mean_pN": fd["mean_pN"], "sd_pN": fd["sd_pN"], "mode_pN": fd["mode_pN"], "residual_mass": fd["residual_mass"], "norm": fd["norm"],
            "eq6": {"mean_pN": m6["mean"], "sd_pN": m6["sd"], "residual_mass": m6["residual_mass"], "mean_eq8_pN": lk.mean_eq8(fdot)},
            "l1_pde_vs_eq6": float(np.trapezoid(np.abs(fd["pdf"] - p6), f)),
            "max_cdf_diff_pde_vs_eq6": float(np.max(np.abs(fd["cdf"] - cdf6))),
            "snapshots": {"t_ps": [float(v) for v in sol.snapshot_t[::stride]], "f_pN": [float(v * fdot) for v in sol.snapshot_t[::stride]], "xi_A": _display(sol.grid.centers, 250), "P": [_display(row, 250) for row in sol.snapshots[::stride]]},
            "snapshot_stride": stride,
        }
        # comparison with the V1 sample (forward prediction against measured critical forces)
        try:
            smp = prepare_sample(self.settings, self.config.compare_source)
            x = smp.forces
            logp_pde = np.log(np.maximum(np.interp(x, f, fd["pdf"]), 1e-300))
            logp_eq6 = lk.log_pdf(x, fdot)
            cdf_pde_at = np.interp(np.sort(x), f, fd["cdf"])
            emp = np.arange(1, x.size + 1) / x.size
            D = float(max(np.max(np.abs(emp - cdf_pde_at)), np.max(np.abs(emp - 1.0 / x.size - cdf_pde_at))))
            hist = histogram_density(x)
            fitted_here = self.config.parameter_source.startswith("inverse:")
            doc["data_comparison"] = {
                "source": self.config.compare_source, "n": int(x.size), "sample_mean_pN": float(x.mean()), "sample_sd_pN": float(x.std(ddof=1)),
                "sample_force_rate_pN_per_ps": smp.force_rate, "same_force_rate": bool(abs(smp.force_rate - fdot) < 1e-9),
                "log_likelihood_pde": float(logp_pde.sum()), "log_likelihood_eq6": float(logp_eq6.sum()) if np.all(np.isfinite(logp_eq6)) else None,
                "ks_D_vs_pde": D,
                "ks_note": "parameters were fitted to this same sample; D is descriptive, no p-value" if fitted_here else "parameters fixed independently of this sample; D is a valid one-sample statistic",
                "histogram": hist,
            }
        except FileNotFoundError as exc:
            doc["data_comparison"] = {"unavailable": str(exc)}
        self._save("observables", doc)

    def phase_validation(self) -> None:
        if self._have("validation") is not None:
            return
        if not self.config.validation_enabled:
            self._save("validation", {"skipped": True})
            return
        sol = self._load_solution()
        p = self.params
        assert p is not None
        self.progress.start_stage("validation", 6, "Numerical validation on these parameters")
        out: dict[str, Any] = {"A_B_conservation_positivity": V.conservation_and_positivity(sol)}
        self.progress.file_done(1)
        out["C1_reflecting_equilibrium"] = V.reflecting_equilibrium(p)
        self.progress.file_done(2)
        self._check()
        out["C2_mfpt_exact"] = [V.mfpt_benchmark(p, 0.6, boundary="beyond_barrier"), V.mfpt_benchmark(p, 0.6, boundary="barrier_top")]
        self.progress.file_done(3)
        out["C3_free_diffusion"] = V.free_diffusion(p)
        self.progress.file_done(4)
        self._check()
        out["F_kramers"] = V.kramers_check(p, 0.3, n_steps=6000)
        self.progress.file_done(5)
        out["G_reproducibility"] = V.reproducibility(p, sol.force_rate, SolverSettings(n_cells=100, n_steps=1000, boundary=self.config.boundary))
        self.progress.file_done(6)
        self._save("validation", out)

    def phase_convergence(self) -> None:
        if self._have("convergence") is not None:
            return
        if not self.config.convergence_enabled:
            self._save("convergence", {"skipped": True})
            return
        self._load_params()
        assert self.params is not None and self.force_rate is not None
        st = self.config.settings()
        self.progress.start_stage("convergence", 6, "Grid and time-step refinement")
        base_g = SolverSettings(**{**st.to_dict(), "n_cells": max(10, st.n_cells // 2), "n_steps": max(10, st.n_steps // 4)})
        base_t = SolverSettings(**{**st.to_dict(), "n_cells": max(10, st.n_cells // 2), "n_steps": max(10, st.n_steps // 4)})
        g = V.refinement(self.params, self.force_rate, base_g, "grid", 3, progress=lambda i, n: self.progress.file_done(i))
        self._check()
        t = V.refinement(self.params, self.force_rate, base_t, "time", 3, progress=lambda i, n: self.progress.file_done(3 + i))
        self._save("convergence", {"grid": g, "time": t, "note": "levels 0–2 double the resolution each time; the run itself uses the finest grid level and the finest time level"})

    def phase_spectrum(self) -> None:
        if self._have("spectrum") is not None:
            return
        if not self.config.spectrum_enabled:
            self._save("spectrum", {"skipped": True})
            return
        self._load_params()
        p = self.params
        assert p is not None
        rates = [r["force_rate_pN_per_ps"] for r in PAPER_RATES]
        self.progress.start_stage("spectrum", len(rates), "Force spectrum ⟨f‡⟩(ḟ) from the PDE at the paper's rates")
        lk = LandscapeKinetics(p.U_barrier, p.xi_barrier, p.gamma0, p.kBT)
        st = SolverSettings(**{**self.config.settings().to_dict(), "n_cells": 200, "n_steps": 8000, "n_snapshots": 2})
        rows = []
        for i, (r, meta) in enumerate(zip(rates, PAPER_RATES)):
            self._check()
            s = solve(p, r, st, cancel=self._cancelled)
            d = force_density(s)
            rows.append({"index": meta["index"], "force_rate": r, "mean_pde_pN": d["mean_pN"], "sd_pde_pN": d["sd_pN"], "residual_mass_pde": d["residual_mass"],
                         "mean_eq8_pN": lk.mean_eq8(r), "mean_eq6_exact_pN": lk.moments_exact(r, n=4001)["mean"], "sd_eq6_pN": lk.moments_exact(r, n=4001)["sd"]})
            self.progress.file_done(i + 1)
        self._save("spectrum", {"rows": rows, "fig3_readings_by_eye": FIG3_READINGS_BY_EYE, "solver": st.to_dict()})

    def phase_finalize(self) -> None:
        self.progress.start_stage("finalize", 0, "Writing result summary")
        cfg = self._have("prepare") or {}
        obs = self._have("observables") or {}
        val = self._have("validation") or {}
        conv = self._have("convergence") or {}
        spec = self._have("spectrum") or {}
        result = {
            "run_id": self.run_id, "model_version": self.config.model_version, "code_hash": cfg.get("code_hash"),
            "created_at_iso": cfg.get("created_at_iso"), "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": self.config.to_dict(), "parameters": cfg.get("parameters"), "parameter_provenance": cfg.get("parameter_provenance"),
            "force_rate_pN_per_ps": cfg.get("force_rate_pN_per_ps"), "force_rate_provenance": cfg.get("force_rate_provenance"),
            "grid": cfg.get("grid"), "boundary": cfg.get("boundary"), "t_end_ps": cfg.get("t_end_ps"), "dt_ps": cfg.get("dt_ps"),
            "observables": {k: obs.get(k) for k in ("mean_pN", "sd_pN", "mode_pN", "residual_mass", "norm", "eq6", "l1_pde_vs_eq6", "max_cdf_diff_pde_vs_eq6", "data_comparison")},
            "validation": {k: v for k, v in val.items()} if val and not val.get("skipped") else val,
            "convergence": {"grid_order": (conv.get("grid") or {}).get("observed_order_mean"), "time_order": (conv.get("time") or {}).get("observed_order_mean"),
                            "richardson_grid_pN": (conv.get("grid") or {}).get("richardson_error_estimate_pN"), "richardson_time_pN": (conv.get("time") or {}).get("richardson_error_estimate_pN"),
                            "grid_levels": (conv.get("grid") or {}).get("levels"), "time_levels": (conv.get("time") or {}).get("levels")} if conv and not conv.get("skipped") else conv,
            "spectrum": spec.get("rows") if spec and not spec.get("skipped") else spec,
            "solver_duration_s": float(np.load(self._path("solve"))["duration_s"]) if self._path("solve").exists() else None,
            "phase_files": {p: f"forward/runs/{self.run_id}/{self._path(p).name}" for p in PHASES if self._path(p).exists()},
        }
        self._save("finalize", result)


def list_runs(settings: Settings) -> dict[str, Any]:
    root = settings.artifacts_path / "forward"
    idx = read_json(root / "runs.json", default={"runs": []}) or {"runs": []}
    latest = read_json(root / "latest.json", default=None)
    return {"runs": sorted(idx["runs"], key=lambda r: r["completed_at"], reverse=True), "latest": latest}


def load_run(settings: Settings, run_id: str, phase: str = "result") -> Any:
    return read_json(settings.artifacts_path / "forward" / "runs" / run_id / f"{phase}.json", default=None)
