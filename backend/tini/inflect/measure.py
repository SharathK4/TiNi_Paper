"""Every number the reading produces, keyed, so that it can be re-measured.

The bootstrap, the stability sweep and the findings ledger all need the same
thing: given a segmentation, the complete set of quantities it implies —
and, when a *reference* segmentation is supplied, those quantities matched to
the reference's features so that a replicate's second transition cannot be
mistaken for movement of the first, and a stretch's slope is the slope over
the reference's stretch rather than of whichever piece the replicate put
under it.

Keys are stable strings (``transition.0.position``, ``regime.2.slope``) and
are what the ledger, the intervals and the sweep are joined on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .segment import Segmentation, model_curve


def _transitions(seg: Segmentation):
    return [b for b in seg.boundaries if b.kind == "transition"]


def _knees(seg: Segmentation):
    return [b for b in seg.boundaries if b.kind in ("knee", "break", "step")]


def measure(
    x: np.ndarray,
    y: np.ndarray,
    seg: Segmentation,
    reference: Segmentation | None = None,
) -> dict[str, float]:
    """A flat dictionary of every quantity in a reading.

    With ``reference`` given, features are matched to the reference's by
    position and regimes are measured over the reference's stretches; keys
    absent from the result are features the replicate did not find.
    """
    ref = reference or seg
    out: dict[str, float] = {"noise": float(seg.sigma)}

    fit = model_curve(x, seg)

    # --- transitions, matched to the reference's by position -------------
    mine = _transitions(seg)
    for k, rb in enumerate(_transitions(ref)):
        if not mine:
            break
        b = min(mine, key=lambda c: abs(c.x - rb.x))
        # A match further than half the reference's own spacing to its
        # neighbour is a different feature, not this one moved.
        window = max(rb.width * 6, 0.02 * abs(float(x[-1] - x[0])))
        if abs(b.x - rb.x) > window:
            continue
        p = f"transition.{k}"
        out[f"{p}.position"] = float(b.x_lo)
        out[f"{p}.onset"] = float(b.y_before)
        out[f"{p}.after"] = float(b.y_after)
        out[f"{p}.step"] = float(b.gap)
        out[f"{p}.width"] = float(b.width)
        out[f"{p}.significance"] = float(b.gap_sigma)

    # --- knees -----------------------------------------------------------
    mine_k = _knees(seg)
    for k, rb in enumerate(_knees(ref)):
        if not mine_k:
            break
        b = min(mine_k, key=lambda c: abs(c.x - rb.x))
        if abs(b.x - rb.x) > 0.05 * abs(float(x[-1] - x[0])):
            continue
        p = f"knee.{k}"
        out[f"{p}.position"] = float(b.x)
        out[f"{p}.slope_before"] = float(b.slope_before)
        out[f"{p}.slope_after"] = float(b.slope_after)

    # --- regimes, over the reference's stretches ------------------------
    for i, s in enumerate(ref.segments):
        lo, hi = s.i0, min(s.i1, x.size)
        if hi - lo < 3:
            continue
        xs, fs = x[lo:hi], fit[lo:hi]
        p = f"regime.{i}"
        out[f"{p}.level"] = float(np.mean(fs))
        out[f"{p}.slope"] = float(np.polyfit(xs, fs, 1)[0])
        if hi - lo >= 4:
            out[f"{p}.curvature"] = float(2.0 * np.polyfit(xs, fs, 2)[0])

    # --- the extremum, a plain fact about the samples ----------------------
    k = int(np.argmax(np.abs(y)))
    out["extremum.value"] = float(y[k])
    out["extremum.position"] = float(x[k])
    return out


def summarise(values: list[float], point: float, keep: int = 60) -> dict[str, Any]:
    """Point estimate, 95 % percentile interval, and a thinned sample of the
    replicates so the distribution itself can be drawn."""
    v = np.array([f for f in values if np.isfinite(f)], dtype=float)
    if v.size < 8:
        return {"value": float(point), "ci": None, "sd": None, "n": int(v.size), "samples": []}
    lo, hi = np.percentile(v, [2.5, 97.5])
    step = max(1, v.size // keep)
    med = float(np.median(v))
    # The replicates can sit to one side of the estimate. A transition's
    # width, for instance, is at its resolution floor on a clean step and
    # can only grow under added noise. That is information about the
    # estimator, and it is reported, not corrected away: the interval stays
    # what the replicates say, and the shift is stated beside it.
    biased = bool(point < lo or point > hi)
    return {
        "value": float(point),
        "ci": [float(lo), float(hi)],
        "sd": float(np.std(v, ddof=1)),
        "n": int(v.size),
        "median": med,
        "shift": med - float(point),
        "biased": biased,
        "samples": [float(f"{f:.6g}") for f in np.sort(v)[::step]],
    }
