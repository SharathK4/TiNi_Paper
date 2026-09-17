"""Runtime configuration.

Everything a run depends on lives in one dataclass so it can be hashed: a
stage checkpoint is only reused when the settings that produced it are the
settings in force now.  Values come from ``config/settings.json`` (optional)
with environment overrides for the two paths.

Physical constants and the paper's published parameters live here too, with
their provenance, so nothing numeric is buried in a computation module.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- physical constants (SI, then converted) ---------------------------------
K_B_J_PER_K = 1.380649e-23            # CODATA 2018, exact
TEMPERATURE_K = 300.0                  # paper, Methods: "T = 300 K"
# 1 J = 1 N.m = 1e12 pN . 1e10 A  ->  1e22 pN.A
K_B_T_PN_ANGSTROM = K_B_J_PER_K * TEMPERATURE_K * 1e22   # = 41.42 pN.A

# --- paper, Table 1: landscape parameters (verified from the PDF) -----------
# Two independent fits are reported. The first column is from MLE of the PDF
# p(f|fdot) (Eq. 6); the second from least-squares regression of <f>(fdot)
# (Eq. 8). Both are kept so the explorer can switch between them.
PAPER_PARAMETER_SETS: dict[str, dict[str, Any]] = {
    "pdf_mle": {
        "label": "Table 1 · fit of p(f | ḟ), Eq. (6), maximum likelihood",
        "U_barrier_pN_A": 638.14,
        "U_barrier_se": 9.62,
        "xi_barrier_A": 29.2,
        "xi_barrier_se": 0.45,
        "gamma0_per_ps": 3.96e-7,
        "gamma0_se": 6.75e-8,
    },
    "mean_lsq": {
        "label": "Table 1 · fit of ⟨f⟩(ḟ), Eq. (8), least squares",
        "U_barrier_pN_A": 659.0,
        "U_barrier_se": 38.6,
        "xi_barrier_A": 36.1,
        "xi_barrier_se": 3.6,
        "gamma0_per_ps": 6.12e-8,
        "gamma0_se": 6e-8,
    },
}

# --- paper, Methods: the seven deformation rates and their force rates ------
# (dlx/dt in A/ps, fdot in pN/ps). Used only to identify which rate the
# supplied dataset corresponds to, and to cross-check our own fdot estimate.
PAPER_RATES = [
    {"index": "i",   "ext_rate_A_per_ps": 9.03e-5,  "force_rate_pN_per_ps": 3.246e-3, "n_traces": 30},
    {"index": "ii",  "ext_rate_A_per_ps": 1.79e-4,  "force_rate_pN_per_ps": 7.137e-3, "n_traces": 30},
    {"index": "iii", "ext_rate_A_per_ps": 3.679e-4, "force_rate_pN_per_ps": 1.482e-2, "n_traces": 400},
    {"index": "iv",  "ext_rate_A_per_ps": 7.839e-4, "force_rate_pN_per_ps": 3.097e-2, "n_traces": 400},
    {"index": "v",   "ext_rate_A_per_ps": 2.352e-3, "force_rate_pN_per_ps": 9.452e-2, "n_traces": 400},
    {"index": "vi",  "ext_rate_A_per_ps": 7.525e-3, "force_rate_pN_per_ps": 2.927e-1, "n_traces": 400},
    {"index": "vii", "ext_rate_A_per_ps": 2.35e-2,  "force_rate_pN_per_ps": 9.242e-1, "n_traces": 400},
]

# --- paper, Fig. 3: mean critical force vs force rate, READ BY EYE ---------
# Not digitised. Symbols were read from the figure to about ±1 pN; they are
# kept only to place this dataset's single point on the paper's force
# spectrum and are labelled as approximate wherever they are shown.
FIG3_READINGS_BY_EYE = [
    {"index": "i",   "force_rate_pN_per_ps": 3.246e-3, "mean_f_pN": 12.5, "uncertainty_pN": 1.0},
    {"index": "ii",  "force_rate_pN_per_ps": 7.137e-3, "mean_f_pN": 15.3, "uncertainty_pN": 1.0},
    {"index": "iii", "force_rate_pN_per_ps": 1.482e-2, "mean_f_pN": 16.4, "uncertainty_pN": 1.0},
    {"index": "iv",  "force_rate_pN_per_ps": 3.097e-2, "mean_f_pN": 17.4, "uncertainty_pN": 1.0},
    {"index": "v",   "force_rate_pN_per_ps": 9.452e-2, "mean_f_pN": 19.5, "uncertainty_pN": 1.0},
    {"index": "vi",  "force_rate_pN_per_ps": 2.927e-1, "mean_f_pN": 22.0, "uncertainty_pN": 1.0},
    {"index": "vii", "force_rate_pN_per_ps": 9.242e-1, "mean_f_pN": 26.5, "uncertainty_pN": 1.0},
]


@dataclass
class Settings:
    # paths
    data_dir: str = str(PROJECT_ROOT / "data" / "raw" / "Force_traces")
    artifacts_dir: str = str(PROJECT_ROOT / "artifacts")
    file_glob: str = "property.T300.*.dat"

    # processing limits -----------------------------------------------------
    # ``max_files`` caps how many traces are processed (None = all). Start
    # small, confirm timing in progress.json, then raise it.
    max_files: int | None = None
    batch_size: int = 10          # files per progress/checkpoint flush
    # expected grid, used by validation (tolerances are relative)
    expected_dt_ps: float = 0.64
    dt_rel_tol: float = 1e-6
    x_plausible_range_A: tuple[float, float] = (50.0, 70.0)

    # statistics ------------------------------------------------------------
    # PROTOTYPE ASSUMPTION: the paper computes fdot "in a small time interval
    # near t = 0" but does not state the interval. We use 100 ps by default
    # and report the sensitivity over ``force_rate_windows_ps``.
    force_rate_window_ps: float = 100.0
    force_rate_windows_ps: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0)
    # PROTOTYPE ASSUMPTION: peak detection on a noisy trace needs smoothing;
    # the paper does not describe its peak picker. Moving-average width in
    # samples (odd). 15 samples = 9.6 ps.
    peak_smoothing_points: int = 15
    percentiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)

    # potential explorer defaults -------------------------------------------
    default_parameter_set: str = "pdf_mle"
    potential_grid_points: int = 241
    # show U(xi, t) tilted at these fractions of f_c
    tilt_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    # display ---------------------------------------------------------------
    display_stride: int = 1        # 1 = every sample (997 pts) — cheap enough
    overlay_max_traces: int = 12   # traces drawn at once in the overlay chart
    chart_dpi: int = 130

    # bookkeeping -----------------------------------------------------------
    config_version: int = 1

    # ------------------------------------------------------------------ utils
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self, keys: tuple[str, ...] | None = None) -> str:
        """Stable hash of the settings that a stage depends on.

        Paths are excluded on purpose: moving the project must not invalidate
        checkpoints; a changed dataset is caught by the dataset manifest.
        """
        d = self.to_dict()
        d.pop("data_dir", None)
        d.pop("artifacts_dir", None)
        if keys is not None:
            d = {k: d[k] for k in keys if k in d}
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_settings(path: Path | None = None) -> Settings:
    """Defaults, overlaid by ``config/settings.json`` if present, then env."""
    cfg = Settings()
    p = path or (PROJECT_ROOT / "config" / "settings.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            if hasattr(cfg, k):
                current = getattr(cfg, k)
                if isinstance(current, tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(cfg, k, v)
    if os.environ.get("TINI_DATA_DIR"):
        cfg.data_dir = os.environ["TINI_DATA_DIR"]
    if os.environ.get("TINI_ARTIFACTS_DIR"):
        cfg.artifacts_dir = os.environ["TINI_ARTIFACTS_DIR"]
    return cfg
