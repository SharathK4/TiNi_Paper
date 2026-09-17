"""Per-file and cross-file validation of force traces.

Each check yields ``(level, code, detail)`` where level is ``error`` (the file
is excluded from statistics) or ``warn`` (kept, but flagged). The cross-file
pass confirms every trace shares one time grid so averaging over traces is a
plain column-wise mean with no interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings


@dataclass
class FileValidation:
    trace_id: str
    name: str
    n_rows: int
    ok: bool
    issues: list[dict[str, str]] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "n_rows": self.n_rows,
            "ok": self.ok,
            "n_errors": sum(1 for i in self.issues if i["level"] == "error"),
            "n_warnings": sum(1 for i in self.issues if i["level"] == "warn"),
            "issues": self.issues,
            "stats": self.stats,
        }


def validate_trace(trace_id: str, name: str, arr: np.ndarray, settings: Settings) -> FileValidation:
    issues: list[dict[str, str]] = []

    def err(code: str, detail: str) -> None:
        issues.append({"level": "error", "code": code, "detail": detail})

    def warn(code: str, detail: str) -> None:
        issues.append({"level": "warn", "code": code, "detail": detail})

    if arr.ndim != 2 or arr.shape[1] != 3:
        err("shape", f"expected (n, 3), got {arr.shape}")
        return FileValidation(trace_id, name, int(arr.shape[0]) if arr.ndim else 0, False, issues)
    n = int(arr.shape[0])
    if n < 10:
        err("too_short", f"only {n} rows")
        return FileValidation(trace_id, name, n, False, issues)

    t, x, f = arr[:, 0], arr[:, 1], arr[:, 2]

    if not np.isfinite(arr).all():
        err("non_finite", f"{int((~np.isfinite(arr)).sum())} non-finite values")

    if abs(t[0]) > 1e-9:
        warn("t_start", f"time starts at {t[0]:.6g} ps, expected 0")
    dts = np.diff(t)
    if not (dts > 0).all():
        err("time_not_monotonic", "time is not strictly increasing")
    else:
        dt = float(np.median(dts))
        rel = float(np.max(np.abs(dts - dt)) / dt)
        if rel > settings.dt_rel_tol:
            warn("dt_nonuniform", f"max |Δt − median|/median = {rel:.2e}")
        if abs(dt - settings.expected_dt_ps) / settings.expected_dt_ps > 1e-3:
            warn("dt_unexpected", f"Δt = {dt:.4f} ps, expected {settings.expected_dt_ps}")

    dxs = np.diff(x)
    if not (dxs > 0).all():
        err("x_not_monotonic", "box length is not strictly increasing under tensile ramp")
    lo, hi = settings.x_plausible_range_A
    if x.min() < lo or x.max() > hi:
        warn("x_range", f"x in [{x.min():.3f}, {x.max():.3f}] Å outside plausible [{lo}, {hi}]")

    slope, intercept = np.polyfit(t, x, 1)
    resid = x - (slope * t + intercept)
    if np.max(np.abs(resid)) > 1e-3 * (x.max() - x.min()):
        warn("x_not_linear", "box length deviates from a linear ramp")

    if f.max() <= 0:
        err("force_nonpositive", "force never exceeds zero")

    stats = {
        "t_end_ps": float(t[-1]),
        "dt_ps": round(float(np.median(dts)), 9) if len(dts) else float("nan"),
        "x_start_A": float(x[0]),
        "x_end_A": float(x[-1]),
        "ext_rate_A_per_ps": float(slope),
        "f_min_pN": float(f.min()),
        "f_max_pN": float(f.max()),
        "f_mean_pN": float(f.mean()),
        "f_std_pN": float(f.std()),
        "t_at_raw_max_ps": float(t[int(np.argmax(f))]),
    }
    ok = not any(i["level"] == "error" for i in issues)
    return FileValidation(trace_id, name, n, ok, issues, stats)


def validate_collection(results: list[FileValidation], time_grids: list[np.ndarray]) -> dict[str, Any]:
    """Cross-file checks: identical row count and time grid."""
    issues: list[dict[str, str]] = []
    ok_results = [r for r in results if r.ok]
    row_counts = sorted({r.n_rows for r in ok_results})
    same_rows = len(row_counts) == 1
    if not same_rows:
        issues.append({"level": "error", "code": "row_count_mismatch", "detail": f"row counts: {row_counts}"})
    same_grid = True
    if time_grids and same_rows:
        ref = time_grids[0]
        for g in time_grids[1:]:
            if g.shape != ref.shape or not np.allclose(g, ref, rtol=0, atol=1e-9):
                same_grid = False
                break
    if not same_grid:
        issues.append({"level": "error", "code": "time_grid_mismatch", "detail": "traces do not share one time grid"})
    rates = np.array([r.stats.get("ext_rate_A_per_ps", np.nan) for r in ok_results])
    x0 = np.array([r.stats.get("x_start_A", np.nan) for r in ok_results])
    return {
        "n_files": len(results),
        "n_ok": len(ok_results),
        "n_failed": len(results) - len(ok_results),
        "n_with_warnings": sum(1 for r in results if any(i["level"] == "warn" for i in r.issues)),
        "same_row_count": same_rows,
        "same_time_grid": same_grid,
        "row_counts": row_counts,
        "ext_rate_A_per_ps": {
            "mean": float(np.nanmean(rates)) if len(rates) else None,
            "std": float(np.nanstd(rates)) if len(rates) else None,
            "min": float(np.nanmin(rates)) if len(rates) else None,
            "max": float(np.nanmax(rates)) if len(rates) else None,
        },
        "x_start_A": {
            "mean": float(np.nanmean(x0)) if len(x0) else None,
            "std": float(np.nanstd(x0)) if len(x0) else None,
            "note": "Each trace starts from its own NPT-equilibrated box, so L_x(0) varies slightly between traces.",
        },
        "issues": issues,
        "ok": same_rows and same_grid and len(ok_results) > 0,
    }
