"""Decide what each column is, and which pair is worth plotting.

A researcher uploading a 21-column simulation log should not have to read a
drop-down. This module measures every column — is it monotone, is it evenly
spaced, how much of its range is noise — recognises the common physical
quantities by name, and ranks the columns for the two jobs that matter: the
controlled variable on the x-axis, and the response worth looking at on the y.

Every judgement carries a short reason string. Nothing here is irreversible:
the interface offers the ranking, the user can pick anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ingest import Table

# --- physical vocabulary ----------------------------------------------------
# Ordered: the first pattern that matches wins, so the specific ones come
# first. These only inform ranking and axis labels; an unrecognised name is
# handled purely on its statistics.
KIND_PATTERNS: list[tuple[str, str]] = [
    ("time", r"^\s*-?(t|time|timestep|step|frame)\s*$|(^|[\s_])time([\s_]|$)"),
    ("strain", r"strain|epsilon|^\s*eps\s*$|elongation|\bε\b"),
    ("stress", r"stress|sigma|\bσ\b|^-?p[xyz][xyz]$|^-?pressure$|^-?p$|virial"),
    ("force", r"force|load|^\s*-?f\s*$|tension"),
    ("energy", r"energy|enthalpy|^-?(pe|ke|etotal|etot|epot|ekin|u|h)$|potential|hamiltonian"),
    ("temperature", r"^-?(t|temp|temperature)$|kelvin"),
    ("length", r"^-?l[xyz]$|length|displacement|extension|distance|separation|position|^-?[xyz]$|box"),
    ("volume", r"volume|^-?vol$|density"),
    ("count", r"^n$|count|number|atoms|particles"),
]

# Which kinds make a natural controlled variable, and how strongly.
X_KIND_BONUS = {"time": 0.45, "strain": 0.50, "length": 0.30, "count": 0.15}
# Which kinds are the response a reader usually wants to see.
Y_KIND_BONUS = {"stress": 0.50, "force": 0.50, "energy": 0.35, "volume": 0.12}
# Kinds that are usually instrumentation, not the story.
Y_KIND_PENALTY = {"time": 0.60, "temperature": 0.30, "count": 0.35}

DUPLICATE_R = 0.99995

# "-pxx", "sigma_yz", "Sxy": a second-rank tensor component and its two axes.
TENSOR_COMPONENT = re.compile(r"^-?(?:p|s|sig|sigma|tau|stress|strain)[\s_]*(?P<a>[xyz123])(?P<b>[xyz123])$", re.I)

# Primary outputs are conventionally written before quantities derived from
# them, so a column's position is weak evidence — enough to break a tie only.
POSITION_PRIOR = 0.04


def classify_name(name: str, unit: str | None = None) -> tuple[str, str]:
    """Guess the physical kind of a column from its name. Returns (kind, why)."""
    probe = f"{name} {unit or ''}".strip().lower()
    bare = name.strip().lower()
    for kind, pattern in KIND_PATTERNS:
        if re.search(pattern, bare, flags=re.I) or re.search(pattern, probe, flags=re.I):
            return kind, f"name '{name}' reads as {kind}"
    return "other", f"name '{name}' not recognised; judged on its numbers alone"


def refine_kind(kind: str, name: str, monotonic: str, mono_fraction: float) -> tuple[str, str] | None:
    """Correct a name-only guess using the column's behaviour.

    A bare 'T' matches both time and temperature. Time marches forward; a
    thermostat's temperature does not, so monotonicity settles it.
    """
    if kind == "time" and monotonic == "none" and mono_fraction < 0.9:
        if re.search(r"^-?(t|temp|temperature)$", name.strip(), flags=re.I):
            return "temperature", f"name '{name}' could be time or temperature; it is not monotone, so temperature"
        return "other", f"name '{name}' reads as time, but it is not monotone, so it is something else"
    return None


def smooth_trend(y: np.ndarray, window: int | None = None) -> np.ndarray:
    """Heavy moving average: what is left is trend, not noise."""
    v = np.where(np.isfinite(y), y, np.nan)
    n = v.size
    if n < 5:
        return v
    w = window or max(5, n // 50)
    if w % 2 == 0:
        w += 1
    filled = v.copy()
    if np.isnan(filled).any():
        idx = np.arange(n)
        good = ~np.isnan(filled)
        if good.sum() < 2:
            return filled
        filled = np.interp(idx, idx[good], filled[good])
    pad = np.r_[np.full(w, filled[0]), filled, np.full(w, filled[-1])]
    k = np.ones(w) / w
    return np.convolve(pad, k, mode="same")[w:-w]


def estimate_noise(y: np.ndarray) -> float:
    """Robust noise level from second differences.

    For independent noise, Var(y[i-1] - 2y[i] + y[i+1]) = 6 sigma^2. Using the
    median absolute deviation instead of the variance keeps real jumps and
    kinks — which are signal — from inflating the estimate.
    """
    v = y[np.isfinite(y)]
    if v.size < 5:
        return 0.0
    d2 = v[:-2] - 2 * v[1:-1] + v[2:]
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return mad / (0.6744897501960817 * np.sqrt(6.0))


@dataclass
class ColumnStats:
    index: int
    name: str
    unit: str | None
    label: str
    kind: str
    kind_reason: str
    n_finite: int
    n_nan: int
    minimum: float
    maximum: float
    mean: float
    std: float
    span: float
    monotonic: str            # "increasing" | "decreasing" | "none"
    monotone_fraction: float
    spacing_cv: float         # coefficient of variation of successive differences
    uniform_spacing: bool
    noise_sigma: float
    snr: float                # span / noise
    trend_span: float         # span of the heavily smoothed column
    trend_strength: float     # trend span / noise — real structure, not noise excursions
    constant: bool
    x_score: float = 0.0
    y_score: float = 0.0
    x_reasons: list[str] = field(default_factory=list)
    y_reasons: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in (
            "index", "name", "unit", "label", "kind", "kind_reason", "n_finite", "n_nan",
            "minimum", "maximum", "mean", "std", "span", "monotonic", "monotone_fraction",
            "spacing_cv", "uniform_spacing", "noise_sigma", "snr", "trend_span",
            "trend_strength", "constant",
            "x_score", "y_score", "x_reasons", "y_reasons", "duplicate_of")}
        return d


def describe_column(col_index: int, name: str, unit: str | None, label: str, v: np.ndarray) -> ColumnStats:
    finite = np.isfinite(v)
    f = v[finite]
    n_finite = int(finite.sum())
    if n_finite == 0:
        return ColumnStats(
            index=col_index, name=name, unit=unit, label=label, kind="other",
            kind_reason="all values missing", n_finite=0, n_nan=int(v.size),
            minimum=float("nan"), maximum=float("nan"), mean=float("nan"), std=float("nan"),
            span=0.0, monotonic="none", monotone_fraction=0.0, spacing_cv=float("inf"),
            uniform_spacing=False, noise_sigma=0.0, snr=0.0, trend_span=0.0,
            trend_strength=0.0, constant=True)

    kind, kind_reason = classify_name(name, unit)
    lo, hi = float(np.min(f)), float(np.max(f))
    span = hi - lo
    d = np.diff(f)
    n_up = int((d > 0).sum())
    n_dn = int((d < 0).sum())
    total = max(1, d.size)
    mono_frac = max(n_up, n_dn) / total
    if span == 0:
        monotonic = "none"
    elif n_dn == 0 and n_up > 0:
        monotonic = "increasing"
    elif n_up == 0 and n_dn > 0:
        monotonic = "decreasing"
    else:
        monotonic = "none"

    nz = d[d != 0]
    spacing_cv = float(np.std(nz) / abs(np.mean(nz))) if nz.size and np.mean(nz) != 0 else float("inf")
    sigma = estimate_noise(f)
    snr = span / sigma if sigma > 0 else (float("inf") if span > 0 else 0.0)

    refined = refine_kind(kind, name, monotonic, mono_frac)
    if refined:
        kind, kind_reason = refined

    trend = smooth_trend(f)
    trend_span = float(np.nanmax(trend) - np.nanmin(trend)) if trend.size else 0.0
    trend_strength = trend_span / sigma if sigma > 0 else (float("inf") if trend_span > 0 else 0.0)

    return ColumnStats(
        index=col_index, name=name, unit=unit, label=label, kind=kind, kind_reason=kind_reason,
        n_finite=n_finite, n_nan=int(v.size - n_finite), minimum=lo, maximum=hi,
        mean=float(np.mean(f)), std=float(np.std(f)), span=span,
        monotonic=monotonic, monotone_fraction=mono_frac, spacing_cv=spacing_cv,
        uniform_spacing=bool(spacing_cv < 1e-6), noise_sigma=sigma, snr=snr,
        trend_span=trend_span, trend_strength=trend_strength,
        constant=bool(span == 0 or (sigma > 0 and span < 0.5 * sigma)),
    )


def _score_x(c: ColumnStats) -> tuple[float, list[str]]:
    if c.constant:
        return 0.0, ["constant — cannot be an axis"]
    s, why = 0.0, []
    if c.monotonic != "none":
        s += 0.45
        why.append(f"strictly {c.monotonic}")
    elif c.monotone_fraction > 0.98:
        s += 0.28
        why.append(f"{100 * c.monotone_fraction:.0f} % monotone")
    else:
        why.append("not monotone — an unusual controlled variable")
    if c.uniform_spacing:
        s += 0.25
        why.append("evenly spaced")
    elif c.spacing_cv < 0.05:
        s += 0.12
        why.append("nearly evenly spaced")
    bonus = X_KIND_BONUS.get(c.kind, 0.0)
    if bonus:
        s += bonus
        why.append(f"{c.kind} is a natural controlled variable")
    if c.snr > 1e6:
        s += 0.08
        why.append("essentially noiseless")
    return s, why


def _score_y(c: ColumnStats) -> tuple[float, list[str]]:
    """Rank a column as a response worth plotting.

    The measure is the *trend* relative to the noise, not the raw range: a
    thermostat's fluctuations have a wide range and no story, while a stress
    that ramps and then drops has a modest range and all of one.
    """
    if c.constant:
        return 0.0, ["constant — nothing to see"]
    s, why = 0.0, []
    ts = c.trend_strength
    if ts >= 2:
        # Continuous and saturating, so a column is never ranked above another
        # on a hair's breadth of trend strength.
        s += 0.45 * min(1.0, np.log10(1.0 + min(ts, 1e6)) / np.log10(31.0))
        why.append(f"trend stands {ts:.1f}x above the noise"
                   if ts >= 6 else f"weak trend, {ts:.1f}x the noise — smoothing will matter")
    else:
        s -= 0.25
        why.append(f"no trend above the noise ({ts:.1f}x) — looks like fluctuation only")

    # A normal (diagonal) tensor component is conjugate to a normal strain; a
    # shear component responds to it only indirectly.
    comp = TENSOR_COMPONENT.match(c.name.strip())
    if comp:
        a, b = comp.group("a"), comp.group("b")
        if a == b:
            s += 0.10
            why.append(f"{a}{b} is a normal component — conjugate to axial loading")
        else:
            s -= 0.10
            why.append(f"{a}{b} is a shear component — responds indirectly")
    bonus = Y_KIND_BONUS.get(c.kind, 0.0)
    if bonus:
        s += bonus
        why.append(f"{c.kind} is usually the response of interest")
    pen = Y_KIND_PENALTY.get(c.kind, 0.0)
    if pen:
        s -= pen
        why.append(f"{c.kind} is usually a control, not a response")
    if c.duplicate_of:
        s -= 0.5
        why.append(f"carries the same information as {c.duplicate_of}")
    return s, why


@dataclass
class ColumnReport:
    stats: list[ColumnStats]
    x_ranking: list[int]
    y_ranking: list[int]
    recommended_x: int | None
    recommended_y: int | None
    headline: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [s.to_dict() for s in self.stats],
            "x_ranking": self.x_ranking,
            "y_ranking": self.y_ranking,
            "recommended_x": self.recommended_x,
            "recommended_y": self.recommended_y,
            "headline": self.headline,
        }


def analyse_columns(table: Table) -> ColumnReport:
    """Measure every column and rank them for the x and y roles."""
    stats = [
        describe_column(c.index, c.name, c.unit, c.label, table.data[:, c.index])
        for c in table.columns
    ]

    # Mark columns that duplicate an earlier one (exact linear dependence is
    # common in simulation logs: a stress written twice in different units).
    #
    # A column that is perfectly correlated with the *axis* is not redundant —
    # it is a linear response, which is the most ordinary result there is — so
    # evenly spaced monotone columns are never counted as the original.
    for j in range(len(stats)):
        if stats[j].constant:
            continue
        a = table.data[:, j]
        for i in range(j):
            if stats[i].constant or (stats[i].uniform_spacing and stats[i].monotonic != "none"):
                continue
            b = table.data[:, i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 8:
                continue
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            if abs(r) >= DUPLICATE_R:
                stats[j].duplicate_of = stats[i].name
                break

    n_cols = max(1, len(stats) - 1)
    for c in stats:
        c.x_score, c.x_reasons = _score_x(c)
        c.y_score, c.y_reasons = _score_y(c)
        if c.y_score > 0:
            c.y_score += POSITION_PRIOR * (1.0 - c.index / n_cols)

    x_rank = sorted((c for c in stats if c.x_score > 0), key=lambda c: -c.x_score)
    rec_x = x_rank[0].index if x_rank else None

    y_candidates = [c for c in stats if c.index != rec_x and c.y_score > 0]
    y_rank = sorted(y_candidates, key=lambda c: -c.y_score)
    rec_y = y_rank[0].index if y_rank else None

    if rec_x is not None and rec_y is not None:
        cx, cy = stats[rec_x], stats[rec_y]
        headline = f"{cy.label} against {cx.label}"
    elif rec_x is not None:
        headline = f"{stats[rec_x].label} on the x-axis; no clear response column"
    else:
        headline = "no column looks like a controlled variable"

    return ColumnReport(
        stats=stats,
        x_ranking=[c.index for c in x_rank],
        y_ranking=[c.index for c in y_rank],
        recommended_x=rec_x,
        recommended_y=rec_y,
        headline=headline,
    )
