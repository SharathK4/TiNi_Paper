"""The same analysis over many files at once.

A researcher rarely has one curve. They have a directory of repeats, and the
questions are: where does the transition sit and how much does it move; do all
the curves behave the same way; and is anything in here not like the others.

The last question is the one worth automating. An outlier is reported with the
measurement that makes it one — "its range is nine times the spread of the
rest" — because a file that differs is either a discovery or a mistake, and
the reader can only tell which if they are told what differs.

Robust statistics throughout: medians and median absolute deviations, so that
the outliers being looked for do not set the scale they are judged against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .segment import Segmentation, segment_curve

MAD_TO_SIGMA = 1.4826
OUTLIER_Z = 5.0
# A set of near-identical repeats has a vanishing spread, and dividing by it
# would make a difference in the fourth decimal place look like a discovery.
# No deviation counts unless it is also this fraction of the typical value,
# so "unusual" has to mean "unusual and worth mentioning".
RELATIVE_FLOOR = 0.02


def robust_scale(v: np.ndarray) -> tuple[float, float]:
    """Median and a standard-deviation-like spread that outliers cannot inflate."""
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    # No fall back to the standard deviation when the median absolute
    # deviation is zero: with most values identical the standard deviation is
    # set almost entirely by the outlier, so dividing by it would hide exactly
    # what is being looked for. Zero is returned, and the caller supplies a
    # floor of its own.
    return med, MAD_TO_SIGMA * mad


@dataclass
class CurveSummary:
    name: str
    ok: bool
    error: str | None = None
    n_rows: int = 0
    shape: str = ""
    n_regimes: int = 0
    noise: float = float("nan")
    y_min: float = float("nan")
    y_max: float = float("nan")
    y_range: float = float("nan")
    first_slope: float = float("nan")
    n_transitions: int = 0
    transition_x: float = float("nan")
    transition_onset: float = float("nan")
    transition_step: float = float("nan")
    transition_width: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# Features compared across files. Each is (key, label, how to read a difference).
FEATURES: list[tuple[str, str]] = [
    ("transition_x", "transition position"),
    ("transition_onset", "value at the transition"),
    ("transition_step", "size of the step"),
    ("y_range", "range covered"),
    ("first_slope", "initial slope"),
    ("noise", "noise level"),
]


@dataclass
class Outlier:
    name: str
    score: float
    feature: str
    feature_label: str
    value: float
    median: float
    z: float
    sentence: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EnsembleReport:
    n_files: int
    n_ok: int
    n_failed: int
    curves: list[CurveSummary]
    stats: dict[str, dict[str, float]]
    outliers: list[Outlier]
    groups: list[dict[str, Any]]
    headline: str
    failures: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_files": self.n_files,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "curves": [c.to_dict() for c in self.curves],
            "stats": self.stats,
            "outliers": [o.to_dict() for o in self.outliers],
            "groups": self.groups,
            "headline": self.headline,
            "failures": self.failures,
        }


def summarise_curve(name: str, x: np.ndarray, y: np.ndarray,
                    seg: Segmentation | None = None, **kw: Any) -> CurveSummary:
    """Reduce one curve to the handful of numbers worth comparing."""
    try:
        s = seg or segment_curve(x, y, **kw)
    except Exception as exc:  # a curve too short or entirely flat
        return CurveSummary(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")

    trans = [b for b in s.boundaries if b.kind == "transition"]
    main = max(trans, key=lambda b: abs(b.gap)) if trans else None

    # Measured from the data, not from the first fitted segment: whether the
    # partitioner happened to open with a plateau is a property of the
    # description, not of the curve, and comparing descriptions across files
    # manufactures outliers that are not there.
    head = max(8, int(0.2 * x.size))
    good = np.isfinite(x[:head]) & np.isfinite(y[:head])
    if good.sum() >= 3:
        first_slope = float(np.polyfit(x[:head][good], y[:head][good], 1)[0])
    else:
        first_slope = float("nan")
    return CurveSummary(
        name=name, ok=True, n_rows=int(x.size), shape=s.shape,
        n_regimes=len(s.segments), noise=float(s.sigma),
        y_min=float(np.nanmin(y)), y_max=float(np.nanmax(y)),
        y_range=float(np.nanmax(y) - np.nanmin(y)),
        first_slope=first_slope,
        n_transitions=len(trans),
        transition_x=float(main.x_lo) if main else float("nan"),
        transition_onset=float(main.y_before) if main else float("nan"),
        transition_step=float(main.gap) if main else float("nan"),
        transition_width=float(main.width) if main else float("nan"),
    )


def _shape_signature(shape: str) -> str:
    """The structure of a curve, ignoring how finely its regimes were split.

    Two curves that both ramp, step once and settle belong together even if
    one of them earned an extra knee on the way up.
    """
    parts = [p.strip() for p in shape.split("->")]
    out: list[str] = []
    for p in parts:
        token = "transition" if p == "transition" else ("bend" if p in ("knee", "break", "step") else "run")
        if out and out[-1] == token and token == "run":
            continue
        if token == "bend":
            continue
        out.append(token)
    collapsed: list[str] = []
    for t in out:
        if collapsed and collapsed[-1] == t == "run":
            continue
        collapsed.append(t)
    return " -> ".join(collapsed) or "run"


def analyse_ensemble(summaries: Iterable[CurveSummary],
                     outlier_z: float = OUTLIER_Z,
                     relative_floor: float = RELATIVE_FLOOR) -> EnsembleReport:
    """Statistics, grouping and outliers across a set of summarised curves."""
    curves = list(summaries)
    ok = [c for c in curves if c.ok]
    failures = [{"name": c.name, "error": c.error or "unknown"} for c in curves if not c.ok]

    stats: dict[str, dict[str, float]] = {}
    scales: dict[str, tuple[float, float]] = {}
    for key, _label in FEATURES:
        v = np.array([getattr(c, key) for c in ok], dtype=float)
        finite = v[np.isfinite(v)]
        med, scale = robust_scale(v)
        if np.isfinite(med) and finite.size:
            typical = max(abs(med), float(np.median(np.abs(finite))))
            scale = max(scale, relative_floor * typical)
        scales[key] = (med, scale)
        stats[key] = {
            "n": int(finite.size),
            "median": med,
            "spread": scale,
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "sd": float(np.std(finite)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }

    outliers: list[Outlier] = []
    for c in ok:
        worst_z, worst_key, worst_label, worst_value = 0.0, "", "", float("nan")
        for key, label in FEATURES:
            med, scale = scales[key]
            value = getattr(c, key)
            if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
                continue
            z = abs(value - med) / scale
            if z > worst_z:
                worst_z, worst_key, worst_label, worst_value = z, key, label, value
        if worst_z >= outlier_z:
            med, _ = scales[worst_key]
            direction = "above" if worst_value > med else "below"
            outliers.append(Outlier(
                name=c.name, score=float(worst_z), feature=worst_key,
                feature_label=worst_label, value=float(worst_value), median=float(med),
                z=float(worst_z),
                sentence=(f"its {worst_label} is {worst_value:.4g} against a typical "
                          f"{med:.4g} — {worst_z:.0f} spreads {direction} the rest"),
            ))
    outliers.sort(key=lambda o: -o.score)

    groups: dict[str, list[str]] = {}
    for c in ok:
        groups.setdefault(_shape_signature(c.shape), []).append(c.name)
    group_list = [
        {"signature": sig, "count": len(names), "members": names[:200]}
        for sig, names in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]

    tx = stats.get("transition_x", {})
    if tx.get("n"):
        headline = (f"{tx['n']} of {len(ok)} curves have a transition, at "
                    f"{tx['median']:.4g} give or take {tx['spread']:.2g}")
    else:
        headline = f"{len(ok)} curves, none with a clear transition"
    if outliers:
        headline += f"; {len(outliers)} file(s) do not belong with the rest"

    return EnsembleReport(
        n_files=len(curves), n_ok=len(ok), n_failed=len(failures),
        curves=curves, stats=stats, outliers=outliers, groups=group_list,
        headline=headline, failures=failures,
    )


def run_ensemble(
    loader: Callable[[str], tuple[np.ndarray, np.ndarray]],
    names: Iterable[str],
    progress: Callable[[int, int, str], None] | None = None,
    outlier_z: float = OUTLIER_Z,
    **segment_kw: Any,
) -> EnsembleReport:
    """Summarise every named curve, then compare them.

    ``loader`` maps a name to its (x, y); keeping the reading outside lets the
    caller stream from disk without this module knowing about files.
    """
    names = list(names)
    summaries: list[CurveSummary] = []
    for i, name in enumerate(names):
        try:
            x, y = loader(name)
            summaries.append(summarise_curve(name, x, y, **segment_kw))
        except Exception as exc:
            summaries.append(CurveSummary(name=name, ok=False,
                                          error=f"{type(exc).__name__}: {exc}"))
        if progress:
            progress(i + 1, len(names), name)
    return analyse_ensemble(summaries, outlier_z=outlier_z)
