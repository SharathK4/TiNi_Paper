"""How sure is any of this?

Two questions a reader is entitled to ask of a number on a chart, and which
the rest of this package cannot answer on its own:

1. *What is the error bar?* Answered by resampling the curve's own residuals
   and re-running the entire analysis — detection, partitioning and all — so
   the interval covers every step of the reading rather than the last fit.

   The resampling is by **moving blocks**, not point by point. Noise in a
   measured or simulated trace is correlated from one sample to the next, and
   an independent bootstrap would break that correlation, understate the
   scatter, and report intervals several times too narrow. The block length
   is read off the residuals' own autocorrelation.

2. *Did you choose the settings to get the answer you wanted?* Answered by
   re-deriving the transition across the whole plausible range of settings
   and showing what it does. A result that holds across the sweep is a
   property of the data; one that moves with the knobs is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .segment import Segmentation, model_curve, prepare, segment_curve

DEFAULT_RESAMPLES = 80
CI = 95.0


def autocorrelation_length(r: np.ndarray, max_lag: int | None = None) -> int:
    """How many samples it takes the residuals to forget each other.

    The first lag at which the autocorrelation falls below 1/e. Returned as a
    block length, so that a resampled series keeps the correlation the real
    one has.
    """
    r = r[np.isfinite(r)]
    n = r.size
    if n < 16:
        return 1
    r = r - r.mean()
    denom = float(r @ r)
    if denom <= 0:
        return 1
    top = max_lag or min(200, n // 4)
    for lag in range(1, top + 1):
        rho = float(r[:-lag] @ r[lag:]) / denom
        if rho < 1.0 / np.e:
            return max(1, lag)
    return max(1, top)


def moving_block_resample(r: np.ndarray, block: int, rng: np.random.Generator,
                          length: int | None = None) -> np.ndarray:
    """A residual series built from random blocks of r.

    ``length`` defaults to r's own, and is given explicitly when noise is
    needed for more samples than there are residuals — the stretches inside a
    transition have no fitted regime to take a residual from, but still have
    to receive noise like everywhere else.
    """
    n = r.size
    out_len = length or n
    block = int(np.clip(block, 1, max(1, n // 2)))
    n_blocks = int(np.ceil(out_len / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    out = np.concatenate([r[s:s + block] for s in starts])
    return out[:out_len]


def _percentile_summary(values: list[float], point: float) -> dict[str, Any]:
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 8:
        return {"value": point, "ci": None, "sd": None, "n": int(v.size)}
    lo, hi = np.percentile(v, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return {
        "value": float(point),
        "ci": [float(lo), float(hi)],
        "sd": float(np.std(v, ddof=1)),
        "n": int(v.size),
    }


@dataclass
class CurveUncertainty:
    n_resamples: int
    n_used: int
    block_length: int
    detection_rate: float
    transition: dict[str, Any] | None
    segments: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "n_used": self.n_used,
            "block_length": self.block_length,
            "detection_rate": self.detection_rate,
            "transition": self.transition,
            "segments": self.segments,
            "note": self.note,
        }


def bootstrap_curve(
    x: np.ndarray,
    y: np.ndarray,
    seg: Segmentation,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    **segment_kw: Any,
) -> CurveUncertainty:
    """Re-run the whole reading on resampled residuals, and report the spread.

    Every quantity is re-derived from scratch in each replicate, so the
    interval on a transition's position includes the chance that the detector
    would have placed it elsewhere — not merely the scatter of a fit whose
    breakpoint was held fixed.
    """
    x, y = prepare(x, y)
    fitted = model_curve(x, seg)

    # Inside a transition there is no fitted regime, only a straight join
    # between the two sides. Two things have to be true of the curve the
    # replicates are built on, and getting either wrong ruins the interval:
    #
    #   * it must keep the transition as sharp as the one measured. Joining
    #     the sides with a line hands every replicate a gentler transition
    #     and the detector then misses it about half the time.
    #   * noise must still reach it. Pinning the transition to the observed
    #     samples freezes the very stretch that decides where the onset is,
    #     and the interval on the position comes back the same width whether
    #     the data are clean or barely readable.
    #
    # So the transition's *shape* is kept — lightly smoothed, to avoid
    # counting its noise twice — and resampled noise is laid over the whole
    # curve, transition included.
    from .columns import smooth_trend

    in_event = np.zeros(x.size, dtype=bool)
    for e in seg.events:
        in_event[max(0, e.i0):min(x.size, e.i1 + 1)] = True
    if in_event.any():
        fitted[in_event] = smooth_trend(y, window=5)[in_event]

    smooth = ~in_event
    resid = (y - fitted)[smooth]
    if resid.size < 16:
        raise ValueError("too few points outside the transitions to resample")
    block = autocorrelation_length(resid)
    rng = np.random.default_rng(seed)

    main = None
    transitions = [b for b in seg.boundaries if b.kind == "transition"]
    if transitions:
        main = max(transitions, key=lambda b: abs(b.gap))

    mids = [0.5 * (s.x0 + s.x1) for s in seg.segments]
    pos: list[float] = []
    onset: list[float] = []
    step: list[float] = []
    slopes: list[list[float]] = [[] for _ in seg.segments]
    found = 0
    used = 0

    for k in range(n_resamples):
        y_star = fitted + moving_block_resample(resid, block, rng, length=x.size)
        try:
            s_star = segment_curve(x, y_star, **segment_kw)
        except Exception:
            continue
        used += 1

        # The slope of a stretch, not of whichever piece the replicate happened
        # to place under its midpoint: a replicate that split the stretch into
        # a short steep piece would otherwise report that piece's slope and
        # blow the interval open.
        star_fit = model_curve(x, s_star)
        for i, s in enumerate(seg.segments):
            lo, hi = s.i0, min(s.i1, x.size)
            if hi - lo >= 3:
                slopes[i].append(float(np.polyfit(x[lo:hi], star_fit[lo:hi], 1)[0]))

        t_star = [b for b in s_star.boundaries if b.kind == "transition"]
        if not t_star:
            continue
        found += 1
        if main is None:
            continue
        # The one nearest the original, so a second transition elsewhere in
        # the curve cannot masquerade as movement of the one being measured.
        near = min(t_star, key=lambda b: abs(b.x - main.x))
        pos.append(near.x_lo)
        onset.append(near.y_before)
        step.append(near.gap)

    transition = None
    if main is not None:
        transition = {
            "position": _percentile_summary(pos, main.x_lo),
            "onset": _percentile_summary(onset, main.y_before),
            "step": _percentile_summary(step, main.gap),
        }

    note = ""
    if block > 1:
        note = (f"residuals stay correlated for about {block} samples, so they are "
                f"resampled in blocks of that length")

    return CurveUncertainty(
        n_resamples=n_resamples,
        n_used=used,
        block_length=int(block),
        detection_rate=float(found / used) if used else 0.0,
        transition=transition,
        segments=[{"x0": s.x0, "x1": s.x1, "model": s.model,
                   "slope": _percentile_summary(
                       slopes[i],
                       float(np.polyfit(x[s.i0:min(s.i1, x.size)],
                                        fitted[s.i0:min(s.i1, x.size)], 1)[0])
                       if min(s.i1, x.size) - s.i0 >= 3 else s.slope(mids[i]))}
                  for i, s in enumerate(seg.segments)],
        note=note,
    )


# --------------------------------------------------------------- stability --
@dataclass
class StabilityPoint:
    setting: str
    label: str
    value: float
    is_default: bool
    found: bool
    position: float | None
    onset: float | None
    n_segments: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Stability:
    points: list[StabilityPoint]
    reference: float | None
    spread: float | None
    holds: bool
    sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "reference": self.reference,
            "spread": self.spread,
            "holds": self.holds,
            "sentence": self.sentence,
        }


def stability_sweep(
    x: np.ndarray,
    y: np.ndarray,
    reference: Segmentation,
    x_unit: str | None = None,
    thresholds: tuple[float, ...] = (4.0, 5.0, 6.0, 8.0, 10.0, 14.0),
    regime_counts: tuple[int, ...] = (2, 3, 4, 5, 6),
    penalties: tuple[float, ...] = (0.4, 0.7, 1.0, 1.6, 3.0),
    **segment_kw: Any,
) -> Stability:
    """Re-derive the transition under every setting worth trying.

    The question is not whether the analysis can be made to say something
    else, but whether it says the same thing unless it is forced not to.
    """
    x, y = prepare(x, y)
    base = dict(segment_kw)
    default_threshold = float(base.pop("event_threshold", 6.0))
    default_penalty = float(base.pop("penalty_scale", 1.0))
    default_segments = base.pop("n_segments", None)

    trans = [b for b in reference.boundaries if b.kind == "transition"]
    ref = max(trans, key=lambda b: abs(b.gap)).x_lo if trans else None

    points: list[StabilityPoint] = []

    def run(setting: str, label: str, value: float, is_default: bool, **kw: Any) -> None:
        try:
            s = segment_curve(x, y, **{**base, **kw})
        except Exception:
            points.append(StabilityPoint(setting, label, value, is_default, False, None, None, 0))
            return
        t = [b for b in s.boundaries if b.kind == "transition"]
        best = max(t, key=lambda b: abs(b.gap)) if t else None
        points.append(StabilityPoint(
            setting=setting, label=label, value=value, is_default=is_default,
            found=best is not None,
            position=float(best.x_lo) if best else None,
            onset=float(best.y_before) if best else None,
            n_segments=len(s.segments),
        ))

    for t in thresholds:
        run("threshold", f"detector at {t:g}", t, t == default_threshold,
            event_threshold=t, penalty_scale=default_penalty, n_segments=default_segments)
    for p in penalties:
        run("penalty", f"detail x{1 / p:.1f}" if p != 1 else "detail as chosen", p,
            p == default_penalty and default_segments is None,
            event_threshold=default_threshold, penalty_scale=p, n_segments=None)
    for k in regime_counts:
        run("regimes", f"{k} regimes", float(k), default_segments == k,
            event_threshold=default_threshold, penalty_scale=default_penalty, n_segments=k)

    located = [p.position for p in points if p.found and p.position is not None]
    spread = float(np.max(located) - np.min(located)) if len(located) > 1 else 0.0
    n_found = len(located)
    n_total = len(points)
    tolerance = max(1e-12, 0.02 * abs(float(np.ptp(x))))
    # A setting extreme enough to find nothing is information, not a failure;
    # what matters is that the settings which do find it agree.
    holds = bool(located and spread <= tolerance and n_found >= 0.75 * n_total)

    unit = f" {x_unit}" if x_unit else ""
    agree = "all put it in the same place" if spread <= 1e-9 else f"agree to within {spread:.3g}{unit}"
    if not located:
        sentence = "No setting in the sweep finds a transition."
    elif n_found == n_total:
        sentence = f"All {n_total} settings tried find the transition, and they {agree}."
    elif holds:
        sentence = (f"{n_found} of {n_total} settings find the transition, and they {agree}; "
                    f"the rest are too strict to see it at all.")
    else:
        sentence = (f"{n_found} of {n_total} settings find the transition, and they place it "
                    f"across a range of {spread:.3g}{unit} — worth reporting with the value.")

    return Stability(points=points, reference=ref, spread=spread, holds=holds, sentence=sentence)
