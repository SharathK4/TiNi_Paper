"""Every curve in the set, on one pair of axes.

A histogram of where the transitions fall answers one question. Drawing all
the curves together answers a different and larger one — what the measurement
looks like, how much of what you see is the sample and how much is noise, and
whether the set is one population or several.

Two arrangements:

* **as measured** — each curve on the common axis. Shows the spread.
* **aligned on the transition** — each curve shifted so that its own
  transition sits at zero. The scatter between repeats is removed and the
  shape that every repeat shares emerges from underneath it. Anything that
  survives alignment is real; anything that does not was the jitter between
  runs.

A median curve and an interquartile ribbon are computed across every curve;
only a sample of the individual lines is sent, because two hundred faint
lines look exactly like a hundred and cost twice as much to draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

GRID = 180
MAX_LINES = 120


def _resample(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """y on the common grid, NaN wherever this curve has nothing to say."""
    out = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return out


@dataclass
class Arrangement:
    grid: list[float]
    median: list[float | None]
    low: list[float | None]
    high: list[float | None]
    lines: list[list[float | None]]
    n_curves: int
    n_drawn: int
    origin: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "median": self.median,
            "low": self.low,
            "high": self.high,
            "lines": self.lines,
            "n_curves": self.n_curves,
            "n_drawn": self.n_drawn,
            "origin": self.origin,
        }


@dataclass
class Chorus:
    measured: Arrangement | None = None
    aligned: Arrangement | None = None
    names: list[str] = field(default_factory=list)
    x_label: str = "x"
    y_label: str = "y"
    x_unit: str | None = None
    y_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured.to_dict() if self.measured else None,
            "aligned": self.aligned.to_dict() if self.aligned else None,
            "names": self.names,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
        }


def _clean(v: np.ndarray, digits: int = 5) -> list[float | None]:
    out: list[float | None] = []
    for f in v:
        out.append(None if not np.isfinite(f) else float(f"{f:.{digits}g}"))
    return out


def _arrange(pairs: list[tuple[np.ndarray, np.ndarray]], lo: float, hi: float,
             origin: str, max_lines: int = MAX_LINES) -> Arrangement | None:
    if not pairs or not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    grid = np.linspace(lo, hi, GRID)
    stack = np.vstack([_resample(x, y, grid) for x, y in pairs])

    with np.errstate(all="ignore"):
        counts = np.sum(np.isfinite(stack), axis=0)
        med = np.full(GRID, np.nan)
        q1 = np.full(GRID, np.nan)
        q3 = np.full(GRID, np.nan)
        # A column where only a handful of curves reach would give a median of
        # those few, which is not a median of the set.
        enough = counts >= max(3, int(0.25 * len(pairs)))
        if enough.any():
            sub = stack[:, enough]
            med[enough] = np.nanmedian(sub, axis=0)
            q1[enough] = np.nanpercentile(sub, 25, axis=0)
            q3[enough] = np.nanpercentile(sub, 75, axis=0)

    step = max(1, len(pairs) // max_lines)
    drawn = stack[::step][:max_lines]
    return Arrangement(
        grid=_clean(grid, 6),
        median=_clean(med),
        low=_clean(q1),
        high=_clean(q3),
        lines=[_clean(row, 4) for row in drawn],
        n_curves=len(pairs),
        n_drawn=int(drawn.shape[0]),
        origin=origin,
    )


class ChorusBuilder:
    """Accumulates curves during a pass over the files, then arranges them.

    Built incrementally so that reading every file for the comparison and
    reading every file for this picture are the same pass.
    """

    def __init__(self, max_keep: int = 400):
        self.max_keep = max_keep
        self.items: list[tuple[str, np.ndarray, np.ndarray, float | None]] = []

    def add(self, name: str, x: np.ndarray, y: np.ndarray, transition: float | None) -> None:
        if len(self.items) >= self.max_keep or x.size < 4:
            return
        # Thinned on the way in: the arrangement grid is coarser than the data
        # and keeping full traces for four hundred files is pointless.
        step = max(1, x.size // (GRID * 3))
        self.items.append((name, x[::step].copy(), y[::step].copy(), transition))

    def build(self, x_label: str = "x", y_label: str = "y",
              x_unit: str | None = None, y_unit: str | None = None) -> Chorus:
        if not self.items:
            return Chorus(x_label=x_label, y_label=y_label, x_unit=x_unit, y_unit=y_unit)

        pairs = [(x, y) for _n, x, y, _t in self.items]
        lo = float(np.median([x[0] for x, _ in pairs]))
        hi = float(np.median([x[-1] for x, _ in pairs]))
        measured = _arrange(pairs, lo, hi, origin="as measured")

        aligned = None
        shifted = [(x - t, y) for (_n, x, y, t) in self.items if t is not None and np.isfinite(t)]
        if len(shifted) >= 3:
            before = float(np.median([x[0] for x, _ in shifted]))
            after = float(np.median([x[-1] for x, _ in shifted]))
            # A window that most curves actually cover on both sides.
            half = min(abs(before), abs(after))
            span = half if half > 0 else max(abs(before), abs(after))
            aligned = _arrange(shifted, -span, span, origin="aligned on each curve's transition")

        return Chorus(
            measured=measured, aligned=aligned,
            names=[n for n, _x, _y, _t in self.items],
            x_label=x_label, y_label=y_label, x_unit=x_unit, y_unit=y_unit,
        )


def sparkline(x: np.ndarray, y: np.ndarray, points: int = 40) -> list[float]:
    """A thumbnail of a curve, scaled to 0–1, for browsing a set at a glance."""
    if x.size < 2:
        return []
    edges = np.linspace(0, x.size, points + 1).astype(int)
    vals: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = y[a:b]
        vals.append(float(np.nanmean(chunk)) if b > a and np.isfinite(chunk).any() else np.nan)
    v = np.array(vals, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size < 2:
        return []
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo or 1.0
    return [round(float((f - lo) / span), 3) if np.isfinite(f) else 0.0 for f in v]
