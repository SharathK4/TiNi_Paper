"""Inverse-problem runs: configuration, orchestration, storage.

A run is a directory ``artifacts/inverse/runs/<run_id>/`` holding one JSON
file per phase, written atomically when the phase completes:

    config.json        the full InverseConfig, dataset + sample + code fingerprints
    sample.json        the event sample and its non-parametric density estimates
    fit.json           unconstrained MLE (starts, history, polish, status)
    constrained.json   fits with one parameter fixed at the paper's value
    uncertainty.json   observed information, profile likelihoods, bootstrap
    diagnostics.json   model–data comparison for the fit and for the paper's sets
    sensitivity.json   refits under the V1 prototype assumptions varied
    synthetic.json     parameter-recovery experiments
    result.json        the summary the UI reads, with provenance

Re-running a run_id resumes: phases whose file exists are loaded, not
recomputed. A new run never touches an older run's directory; ``latest.json``
points at the most recent *completed* run. Progress goes to
``artifacts/inverse/progress.json`` through the V1 ProgressTracker.
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

from ..config import K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS, Settings
from ..io_atomic import atomic_write_json, read_json
from ..progress import STATUS_CANCELLED, STATUS_DONE, STATUS_FAILED, ProgressTracker
from . import density
from .data import EventSample, prepare_sample
from .diagnostics import compare, likelihood_ratio
from .fit import FitOptions, FitResult, default_starts, derived_quantities, fit_mle
from .likelihood import PARAM_NAMES, Bounds, Likelihood
from .model import LandscapeKinetics
from .synthetic import default_designs, recovery_experiment
from .uncertainty import bootstrap, observed_information, profile_likelihood

MODEL_VERSION = "eq6-mle-1"     # bump when the likelihood or its numerics change
PHASES = ["prepare", "fit", "constrained", "hessian", "profiles", "bootstrap", "diagnostics", "sensitivity", "synthetic", "finalize"]
PHASE_FILES = {**{p: p for p in PHASES}, "prepare": "sample", "finalize": "result"}


class Cancelled(Exception):
    pass


@dataclass
class InverseConfig:
    # sample
    source: str = "rate_x_time"
    smoothing_points: int | None = None       # None → V1 setting
    window_ps: float | None = None            # None → V1 setting
    # model / fit
    reference_set: str = "pdf_mle"            # Table 1 column used for starts and comparisons
    bounds: dict[str, list[float]] = field(default_factory=lambda: Bounds().as_dict())
    start_factors: list[float] = field(default_factory=lambda: [1.0, 0.5, 2.0])
    maxiter: int = 4000
    maxfev: int = 8000
    xatol: float = 1e-6
    fatol: float = 1e-8
    polish: bool = True
    # constrained fits: each entry fixes the listed parameters at the reference set's values
    constrained: list[list[str]] = field(default_factory=lambda: [["xi_barrier"], ["U_barrier"], ["gamma0"]])
    # uncertainty
    hessian_step: float = 1e-3
    profile_points: int = 21
    profile_span: dict[str, float] = field(default_factory=lambda: {"U_barrier": 3.0, "xi_barrier": 3.0, "gamma0": 30.0})
    bootstrap_n: int = 200
    seed: int = 0
    # sensitivity to V1 prototype assumptions (MLE refit only)
    sensitivity_windows_ps: list[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    sensitivity_smoothings: list[int] = field(default_factory=lambda: [7, 15, 31])
    sensitivity_sources: list[str] = field(default_factory=lambda: ["rate_x_time", "smoothed_peak"])
    # synthetic recovery
    synthetic_enabled: bool = True
    synthetic_replicates: int = 10
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fit_options(self) -> FitOptions:
        return FitOptions(maxiter=self.maxiter, maxfev=self.maxfev, xatol=self.xatol, fatol=self.fatol, polish=self.polish)

    def bounds_obj(self) -> Bounds:
        return Bounds(**{k: tuple(v) for k, v in self.bounds.items()})

    def reference(self) -> dict[str, float]:
        p = PAPER_PARAMETER_SETS[self.reference_set]
        return {"U_barrier": p["U_barrier_pN_A"], "xi_barrier": p["xi_barrier_A"], "gamma0": p["gamma0_per_ps"]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InverseConfig":
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


def code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for p in sorted(here.glob("*.py")):
        h.update(p.read_bytes())
    h.update((here.parent / "statistics.py").read_bytes())
    return h.hexdigest()[:12]


class InverseRun:
    def __init__(self, settings: Settings, config: InverseConfig, run_id: str | None = None):
        self.settings = settings
        self.config = config
        self.root = settings.artifacts_path / "inverse"
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = self.root / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressTracker(self.root, PHASES)
        self.cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        # state carried between phases
        self.sample: EventSample | None = None
        self.lik: Likelihood | None = None
        self.fit: FitResult | None = None

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
        return self.dir / f"{PHASE_FILES.get(phase, phase)}.json"

    def _have(self, phase: str) -> Any:
        return read_json(self._path(phase), default=None)

    def _save(self, phase: str, obj: Any) -> None:
        """Write the phase's file atomically and mark the phase complete."""
        atomic_write_json(self._path(phase), obj, keep_previous=True)
        self.progress.end_stage(phase, f"inverse/runs/{self.run_id}/{PHASE_FILES.get(phase, phase)}.json")

    def run_async(self) -> str:
        if self.running:
            raise RuntimeError("run already in progress")
        self.cancel_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self.run_id

    # ------------------------------------------------------------ main
    def run(self) -> dict[str, Any]:
        cfg = self.config
        resumed = [p for p in PHASES if self._path(p).exists()]
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
        except Cancelled:
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
            "run_id": self.run_id,
            "completed_at": time.time(),
            "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": self.config.source,
            "n": res.get("sample", {}).get("n"),
            "params": res.get("fit", {}).get("params"),
            "nll": res.get("fit", {}).get("nll"),
            "converged": res.get("fit", {}).get("converged"),
        }
        idx["runs"] = [r for r in idx["runs"] if r["run_id"] != self.run_id] + [entry]
        atomic_write_json(idx_path, idx)

    # ------------------------------------------------------------ phases
    def _load_sample(self) -> None:
        if self.sample is None:
            self.sample = prepare_sample(self.settings, self.config.source, self.config.smoothing_points, self.config.window_ps)
        if self.lik is None:
            self.lik = Likelihood([(self.sample.forces, self.sample.force_rate)], self.config.bounds_obj(), {}, K_B_T_PN_ANGSTROM)

    def _load_fit(self) -> None:
        self._load_sample()
        if self.fit is None:
            d = self._have("fit")
            if d is None:
                raise RuntimeError("fit phase missing")
            assert self.lik is not None
            self.fit = FitResult(
                params=d["params"], theta=np.array(d["theta"]), nll=d["nll"], free=d["free"], fixed=d["fixed"], starts=[], best_start_index=d["best_start_index"],
                history=d["history"], polish=d["polish"], n_eval_total=d["n_eval_total"], converged=d["converged"], duration_s=d["duration_s"],
            )

    def phase_prepare(self) -> None:
        if self._have("prepare") is not None and self._have("config") is not None:
            self._load_sample()
            return
        self.progress.start_stage("prepare", 0, "Preparing the transition-event sample")
        self._load_sample()
        assert self.sample is not None
        f = self.sample.forces
        grid = np.linspace(0.0, max(f.max() * 1.25, 1.0), 300)
        sample_doc = {
            **self.sample.summary(),
            "forces_pN": f.tolist(),
            "t_peak_ps": self.sample.t_peak_ps.tolist(),
            "trace_ids": self.sample.trace_ids,
            "histogram": density.histogram_density(f),
            "kde": density.kde(f, grid),
            "ecdf": density.ecdf(f),
        }
        atomic_write_json(self._path("config"), {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "model_version": self.config.model_version,
            "code_hash": code_hash(),
            "dataset_fingerprint": self.sample.dataset_fingerprint,
            "sample_fingerprint": self.sample.fingerprint(),
            "kBT_pN_A": K_B_T_PN_ANGSTROM,
            "reference": self.config.reference(),
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save("prepare", sample_doc)

    def phase_fit(self) -> None:
        if self._have("fit") is not None:
            self._load_fit()
            return
        self._load_sample()
        assert self.lik is not None
        starts = default_starts(self.config.reference(), self.lik.free, tuple(self.config.start_factors))
        self.progress.start_stage("fit", len(starts), "Maximum-likelihood fit (Nelder–Mead, multi-start)")
        self.fit = fit_mle(self.lik, starts, self.config.fit_options(), progress=lambda i, n: self.progress.file_done(i), cancel=self._cancelled)
        self._check()
        doc = self.fit.to_dict()
        doc["derived"] = derived_quantities(self.fit.params, self.sample.force_rate, K_B_T_PN_ANGSTROM)  # type: ignore[union-attr]
        doc["nll_at_reference"] = self.lik.nll_params(self.config.reference())
        doc["likelihood_ratio_vs_reference"] = likelihood_ratio(doc["nll_at_reference"], self.fit.nll, len(self.fit.free))
        self._save("fit", doc)

    def phase_constrained(self) -> None:
        if self._have("constrained") is not None:
            return
        self._load_fit()
        assert self.sample is not None and self.fit is not None
        ref = self.config.reference()
        self.progress.start_stage("constrained", len(self.config.constrained), "Constrained fits (one parameter fixed at Table 1)")
        out = []
        for i, fixed_names in enumerate(self.config.constrained):
            self._check()
            fixed = {k: ref[k] for k in fixed_names if k in PARAM_NAMES}
            lik = Likelihood([(self.sample.forces, self.sample.force_rate)], self.config.bounds_obj(), fixed, K_B_T_PN_ANGSTROM)
            start_ref = {k: self.fit.params[k] for k in lik.free}
            fit = fit_mle(lik, default_starts(start_ref, lik.free, tuple(self.config.start_factors)), self.config.fit_options(), cancel=self._cancelled)
            oi = observed_information(lik, fit, self.config.hessian_step)
            out.append({
                "fixed": fixed,
                "label": " and ".join(fixed_names) + " fixed at Table 1",
                "params": fit.params, "nll": fit.nll, "converged": fit.converged, "n_eval": fit.n_eval_total,
                "delta_nll_vs_unconstrained": fit.nll - self.fit.nll,
                "likelihood_ratio": likelihood_ratio(fit.nll, self.fit.nll, len(fixed)),
                "se": oi.get("se"), "condition_number": oi.get("condition_number"),
                "derived": derived_quantities(fit.params, self.sample.force_rate, K_B_T_PN_ANGSTROM),
            })
            self.progress.file_done(i + 1)
        self._save("constrained", {"reference_set": self.config.reference_set, "fits": out})

    def phase_hessian(self) -> None:
        if self._have("hessian") is not None:
            return
        self._load_fit()
        assert self.lik is not None and self.fit is not None
        self.progress.start_stage("hessian", 0, "Observed information (numerical Hessian)")
        self._save("hessian", observed_information(self.lik, self.fit, self.config.hessian_step))

    def phase_profiles(self) -> None:
        if self._have("profiles") is not None:
            return
        self._load_fit()
        assert self.lik is not None and self.fit is not None
        n_total = len(self.fit.free) * (self.config.profile_points + 1)
        self.progress.start_stage("profiles", n_total, "Profile likelihoods")
        counter = {"done": 0}

        def prog(k: str, i: int, n: int) -> None:
            counter["done"] += 1
            self.progress.file_done(counter["done"])

        prof = profile_likelihood(self.lik, self.fit, self.config.profile_points, self.config.profile_span, progress=prog, cancel=self._cancelled)
        self._check()
        self._save("profiles", prof)

    def phase_bootstrap(self) -> None:
        if self._have("bootstrap") is not None:
            return
        self._load_fit()
        assert self.lik is not None and self.fit is not None
        self.progress.start_stage("bootstrap", self.config.bootstrap_n, f"Bootstrap ({self.config.bootstrap_n} resamples, seed {self.config.seed})")
        bs = bootstrap(self.lik, self.fit, self.config.bootstrap_n, self.config.seed, progress=lambda i, n: self.progress.file_done(i), cancel=self._cancelled)
        self._check()
        self._save("bootstrap", bs)

    def phase_diagnostics(self) -> None:
        if self._have("diagnostics") is not None:
            return
        self._load_fit()
        assert self.sample is not None and self.fit is not None
        self.progress.start_stage("diagnostics", 0, "Model–data comparison")
        f, fdot = self.sample.forces, self.sample.force_rate
        p = self.fit.params
        fitted = compare(LandscapeKinetics(p["U_barrier"], p["xi_barrier"], p["gamma0"]), f, fdot, n_params=len(self.fit.free))
        paper = {}
        for key, ps in PAPER_PARAMETER_SETS.items():
            land = LandscapeKinetics(ps["U_barrier_pN_A"], ps["xi_barrier_A"], ps["gamma0_per_ps"])
            paper[key] = {"label": ps["label"], "params": {"U_barrier": land.U_barrier, "xi_barrier": land.xi_barrier, "gamma0": land.gamma0}, **compare(land, f, fdot, n_params=0)}
            paper[key]["likelihood_ratio_vs_fit"] = likelihood_ratio(paper[key]["nll"], self.fit.nll, len(self.fit.free))
        self._save("diagnostics", {"fitted": fitted, "paper": paper})

    def phase_sensitivity(self) -> None:
        if self._have("sensitivity") is not None:
            return
        self._load_fit()
        assert self.fit is not None
        cfg = self.config
        variants = []
        for w in cfg.sensitivity_windows_ps:
            variants.append({"source": "rate_x_time", "window_ps": w, "smoothing_points": cfg.smoothing_points or self.settings.peak_smoothing_points, "axis": "window"})
        for sm in cfg.sensitivity_smoothings:
            variants.append({"source": "rate_x_time", "window_ps": cfg.window_ps or self.settings.force_rate_window_ps, "smoothing_points": sm, "axis": "smoothing"})
        for src in cfg.sensitivity_sources:
            variants.append({"source": src, "window_ps": cfg.window_ps or self.settings.force_rate_window_ps, "smoothing_points": cfg.smoothing_points or self.settings.peak_smoothing_points, "axis": "source"})
        self.progress.start_stage("sensitivity", len(variants), "Sensitivity to the V1 prototype assumptions")
        rows = []
        for i, v in enumerate(variants):
            self._check()
            smp = prepare_sample(self.settings, v["source"], v["smoothing_points"], v["window_ps"])
            lik = Likelihood([(smp.forces, smp.force_rate)], cfg.bounds_obj(), {}, K_B_T_PN_ANGSTROM)
            fit = fit_mle(lik, default_starts(self.fit.params, lik.free, (1.0, 0.5, 2.0)), FitOptions(maxiter=cfg.maxiter, maxfev=cfg.maxfev, xatol=cfg.xatol, fatol=cfg.fatol, polish=False), cancel=self._cancelled)
            rows.append({**v, "force_rate": smp.force_rate, "sample_mean": float(smp.forces.mean()), "sample_sd": float(smp.forces.std(ddof=1)), "params": fit.params, "nll": fit.nll, "converged": fit.converged,
                         "f_c_pN": 3 * fit.params["U_barrier"] / (2 * fit.params["xi_barrier"])})
            self.progress.file_done(i + 1)
        self._save("sensitivity", {"variants": rows})

    def phase_synthetic(self) -> None:
        if self._have("synthetic") is not None or not self.config.synthetic_enabled:
            if not self.config.synthetic_enabled and self._have("synthetic") is None:
                self._save("synthetic", {"skipped": True})
            return
        self._load_sample()
        assert self.sample is not None
        designs = default_designs(self.sample.force_rate)
        self.progress.start_stage("synthetic", len(designs) * self.config.synthetic_replicates, "Synthetic parameter recovery")
        res = recovery_experiment(self.config.reference(), designs, self.config.synthetic_replicates, self.config.seed, self.config.bounds_obj(), K_B_T_PN_ANGSTROM,
                                  progress=lambda i, n: self.progress.file_done(i), cancel=self._cancelled)
        self._check()
        self._save("synthetic", res)

    def phase_finalize(self) -> None:
        self.progress.start_stage("finalize", 0, "Writing result summary")
        cfg_doc = self._have("config") or {}
        fit = self._have("fit") or {}
        hess = self._have("hessian") or {}
        prof = (self._have("profiles") or {}).get("profiles", {})
        bs = self._have("bootstrap") or {}
        diag = self._have("diagnostics") or {}
        cons = (self._have("constrained") or {}).get("fits", [])
        sens = (self._have("sensitivity") or {}).get("variants", [])
        syn = self._have("synthetic") or {}
        sample = self._have("prepare") or {}

        params_out = {}
        for k in fit.get("free", []):
            pk = prof.get(k, {})
            params_out[k] = {
                "mle": fit["params"][k],
                "se_asymptotic": (hess.get("se") or {}).get(k),
                "ci95_asymptotic": (hess.get("ci95_asymptotic") or {}).get(k),
                "ci95_profile": [pk.get("ci95_lower"), pk.get("ci95_upper")],
                "profile_unbounded": [pk.get("lower_unbounded_in_scan"), pk.get("upper_unbounded_in_scan")],
                "profile_support_limited": [pk.get("lower_support_limited"), pk.get("upper_support_limited")],
                "ci95_bootstrap": (bs.get(k) or {}).get("ci95"),
                "bootstrap_median": (bs.get(k) or {}).get("median"),
                "bootstrap_hit_bounds": [(bs.get(k) or {}).get("hit_lower_bound"), (bs.get(k) or {}).get("hit_upper_bound")],
                "paper": cfg_doc.get("reference", {}).get(k),
            }
        result = {
            "run_id": self.run_id,
            "model_version": self.config.model_version,
            "code_hash": cfg_doc.get("code_hash"),
            "dataset_fingerprint": cfg_doc.get("dataset_fingerprint"),
            "sample_fingerprint": cfg_doc.get("sample_fingerprint"),
            "created_at_iso": cfg_doc.get("created_at_iso"),
            "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": self.config.to_dict(),
            "sample": {k: sample.get(k) for k in ("n", "source", "smoothing_points", "window_ps", "force_rate_pN_per_ps", "force_rate_se", "mean_pN", "sd_pN", "sem_pN", "median_pN", "min_pN", "max_pN", "skewness", "notes")},
            "fit": {k: fit.get(k) for k in ("params", "nll", "log_likelihood", "converged", "n_eval_total", "duration_s", "polish", "best_start_index", "derived", "nll_at_reference", "likelihood_ratio_vs_reference")},
            "parameters": params_out,
            "hessian": {k: hess.get(k) for k in ("eigenvalues", "eigenvectors", "condition_number", "positive_definite", "correlation", "free")},
            "bootstrap": {k: bs.get(k) for k in ("n_boot", "n_ok", "n_failed", "seed", "f_c_pN", "log_correlation")},
            "goodness_of_fit": {k: (diag.get("fitted") or {}).get(k) for k in ("ks_statistic", "ks_pvalue", "chi2", "chi2_dof", "chi2_pvalue", "aic", "bic", "model_mean_exact_pN", "model_sd_exact_pN", "model_mean_eq8_pN", "residual_mass", "sample_mean_pN", "sample_sd_pN")},
            "paper_comparison": {key: {k: v.get(k) for k in ("label", "params", "nll", "ks_statistic", "ks_pvalue", "chi2_pvalue", "model_mean_exact_pN", "model_sd_exact_pN", "model_mean_eq8_pN", "residual_mass", "likelihood_ratio_vs_fit")} for key, v in (diag.get("paper") or {}).items()},
            "constrained": [{k: c.get(k) for k in ("label", "fixed", "params", "nll", "delta_nll_vs_unconstrained", "likelihood_ratio", "se", "converged", "derived")} for c in cons],
            "sensitivity": sens,
            "synthetic": {"designs": [{k: d.get(k) for k in ("name", "rates", "S", "n_events", "replicates", "summary")} for d in syn.get("designs", [])], "truth": syn.get("truth"), "replicates": syn.get("replicates")} if syn and not syn.get("skipped") else syn,
            "phase_files": {p: f"inverse/runs/{self.run_id}/{PHASE_FILES[p]}.json" for p in PHASES if self._path(p).exists()},
        }
        self._save("finalize", result)


# ---------------------------------------------------------------- registry
def list_runs(settings: Settings) -> dict[str, Any]:
    root = settings.artifacts_path / "inverse"
    idx = read_json(root / "runs.json", default={"runs": []}) or {"runs": []}
    latest = read_json(root / "latest.json", default=None)
    return {"runs": sorted(idx["runs"], key=lambda r: r["completed_at"], reverse=True), "latest": latest}


def load_run(settings: Settings, run_id: str, phase: str = "result") -> Any:
    p = settings.artifacts_path / "inverse" / "runs" / run_id / f"{phase}.json"
    return read_json(p, default=None)
