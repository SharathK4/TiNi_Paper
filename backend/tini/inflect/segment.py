"""Split a curve into regimes.

This is the core of Version 4. Given a noisy (x, y) curve, find the places
where its *behaviour* changes — not where it happens to peak — and describe
each stretch between them as a constant, a straight line, or a curve.

Two passes, in this order, because the order is what makes the answer
readable:

1. **Events.** A transition is a place where the curve moves far faster than
   it does anywhere else, and it occupies a finite width — it is not another
   polynomial regime. Those stretches are found first against the robust
   spread of the curve's own rate of change, and then held out.
2. **Regimes.** What is left between the events is smooth, and each stretch
   is partitioned by exact dynamic programming: the residual sum of squares
   of a degree 0, 1 or 2 fit on any interval is available in O(1) from prefix
   moments, so the optimal split into k pieces needs no greedy shortcut. A
   budget of pieces is then shared between the stretches optimally.

Doing it the other way round — letting the partitioner discover the drop —
spends five pieces describing the inside of one transition and reports nine
regimes where a reader can see three.

A boundary is therefore a *transition* (an event, with a width and a step),
a *knee* (the slope changes, the value does not) or a *break* (a weak
boundary the criterion wanted but which carries little meaning).

Nothing here knows about titanium nickel, stress or strain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODEL_NAMES = {0: "constant", 1: "linear", 2: "quadratic"}
MAX_DEGREE = 2
MIN_SEGMENT_POINTS = 8
MAX_SEGMENTS = 10
# Candidate breakpoints per smooth stretch. The partition cost is quadratic
# in this, and a knee is not located to better than a few samples anyway.
CANDIDATE_CUTS = 200
# With event detection off, a boundary counts as a step when the two fits
# disagree there by more than this many times the noise.
JUMP_SIGMA = 4.0


@dataclass
class Segment:
    i0: int
    i1: int                     # exclusive
    x0: float
    x1: float
    degree: int
    model: str
    coeffs: list[float]         # in the centred basis u = x - x_mid
    x_mid: float
    rss: float
    sigma: float                # residual scatter within the segment
    r2: float
    n: int
    y_start: float              # fitted value at x0
    y_end: float                # fitted value at x1
    slope_start: float
    slope_end: float
    curvature: float

    @property
    def span(self) -> float:
        return self.x1 - self.x0

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        u = np.asarray(x, dtype=float) - self.x_mid
        out = np.zeros_like(u, dtype=float)
        for p, c in enumerate(self.coeffs):
            out = out + c * u ** p
        return out

    def slope(self, x: float) -> float:
        u = x - self.x_mid
        s = 0.0
        for p, c in enumerate(self.coeffs):
            if p >= 1:
                s += p * c * u ** (p - 1)
        return float(s)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "i0", "i1", "x0", "x1", "degree", "model", "coeffs", "x_mid", "rss",
            "sigma", "r2", "n", "y_start", "y_end", "slope_start", "slope_end",
            "curvature")} | {"span": self.span}


@dataclass
class Event:
    """A fast change occupying a finite stretch of x — a transition.

    Treating this as an event rather than as another polynomial regime is the
    difference between "quadratic, then a transition at 2.30 %, then a
    plateau" and nine meaningless pieces through the middle of the drop.
    """

    i0: int
    i1: int
    x0: float
    x1: float
    width: float
    y_before: float
    y_after: float
    drop: float
    drop_sigma: float
    peak_slope: float
    sharpness: float        # peak |slope| relative to the curve's usual slope

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "i0", "i1", "x0", "x1", "width", "y_before", "y_after", "drop",
            "drop_sigma", "peak_slope", "sharpness")}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask as [start, stop) pairs."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [mask.size]
    return list(zip(starts, stops))


def _rolling_line(x: np.ndarray, y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a straight line to every window of w points, by prefix sums.

    Returns, for each window start, the slope and the value the line takes at
    the window's own right-hand edge. Windows are half-open: window j covers
    samples [j, j + w).
    """
    n = x.size
    c = lambda v: np.concatenate(([0.0], np.cumsum(v)))  # noqa: E731
    S0 = c(np.ones(n))
    S1 = c(x)
    S2 = c(x * x)
    T0 = c(y)
    T1 = c(x * y)
    j = np.arange(0, n - w + 1)
    k = j + w
    s0 = S0[k] - S0[j]
    s1 = S1[k] - S1[j]
    s2 = S2[k] - S2[j]
    t0 = T0[k] - T0[j]
    t1 = T1[k] - T1[j]
    det = s0 * s2 - s1 * s1
    det = np.where(np.abs(det) < 1e-300, np.nan, det)
    slope = (s0 * t1 - s1 * t0) / det
    intercept = (s2 * t0 - s1 * t1) / det
    return slope, intercept


def step_statistic(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """How big a step sits at each sample, after the local trend is removed.

    A straight line is fitted to the w points before the sample and to the w
    points after it; both are extrapolated to the sample and subtracted. On a
    smooth curve — however steeply it rises or curves — the two agree and the
    statistic is zero to first order. Only a genuine discontinuity survives.

    This is what makes the detector work on a quadratic ramp: testing the
    derivative against its own spread cannot separate a transition from the
    ramp's own steepening, and subtracting the trend can.
    """
    n = x.size
    out = np.zeros(n)
    if n < 2 * w + 2:
        return out
    slope, intercept = _rolling_line(x, y, w)
    # window [i-w, i) has start j = i-w ; window [i, i+w) has start j = i
    idx = np.arange(w, n - w)
    left_at = intercept[idx - w] + slope[idx - w] * x[idx]
    right_at = intercept[idx] + slope[idx] * x[idx]
    out[idx] = right_at - left_at
    return out


def detect_events(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    smooth_points: int | None = None,
    threshold: float = 6.0,
    min_drop_sigma: float = 4.0,
    merge_gap: int | None = None,
) -> list[Event]:
    """Find the places where the curve steps.

    The statistic is the locally detrended step (see ``step_statistic``),
    judged against its own robust spread away from the steps themselves. It is
    free of units, of sign, and of however steeply the curve is moving for
    honest reasons.
    """
    n = x.size
    if smooth_points:
        scales = [int(smooth_points)]
    else:
        # A wide window sees a step through more noise; a narrow one can tell
        # two nearby steps apart. Testing a ladder of widths and keeping the
        # best answer at each sample gets both, which one fixed width cannot.
        scales = sorted({max(6, n // 60), max(8, n // 30), max(10, n // 15)})
    scales = [w for w in scales if 2 * w + 2 < n] or [max(3, (n - 3) // 2)]

    ratio = np.zeros(n)
    scale_at = np.full(n, scales[0], dtype=int)
    raw: dict[int, np.ndarray] = {}
    for w in scales:
        step = step_statistic(x, y, w)
        raw[w] = step
        inner = step[w:n - w]
        if inner.size == 0:
            continue
        med = float(np.median(inner))
        mad = float(np.median(np.abs(inner - med)))
        spread = mad / 0.6744897501960817
        if not np.isfinite(spread) or spread <= 0:
            spread = float(np.std(inner)) or 1.0
        r = np.abs(step - med) / spread
        r[:w] = 0.0
        r[n - w:] = 0.0
        better = r > ratio
        ratio[better] = r[better]
        scale_at[better] = w

    astep = ratio
    hi = threshold
    lo = max(2.0, 0.25 * threshold)
    w = int(np.median(scales))

    # Is a candidate a real discontinuity, or a bend seen through a wide
    # window? A step measures the same however wide the window; the apparent
    # step at a knee grows in proportion to the window, because the fits
    # either side straddle the bend. Measured here: real transitions score
    # 1.3-1.4, knees 3-7, for a window ratio of 4.
    jump_like = np.ones(n, dtype=bool)
    if len(scales) > 1 and not smooth_points:
        w_lo, w_hi = min(scales), max(scales)
        lo_abs = np.abs(raw[w_lo])
        hi_abs = np.abs(raw[w_hi])
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(lo_abs > 0, hi_abs / np.where(lo_abs > 0, lo_abs, 1.0), np.inf)
        jump_like = growth <= 0.6 * (w_hi / w_lo)

    # The statistic is a convolution, so one step shows up as a plateau about
    # w wide; runs closer than that belong to the same event.
    gap = merge_gap if merge_gap is not None else w
    spans: list[list[int]] = []
    for a, b in _runs((astep > hi) & jump_like):
        while a > 1 and astep[a - 1] > lo:
            a -= 1
        while b < n - 1 and astep[b] > lo:
            b += 1
        if spans and a - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])

    from .columns import smooth_trend
    ys = smooth_trend(y, window=max(5, w | 1))
    d = np.gradient(ys, x)
    base = float(np.median(np.abs(d)))

    events: list[Event] = []
    for a, b in spans:
        # Take the peak among the jump-like samples, not the span's overall
        # peak: a knee sitting beside a transition would otherwise capture it.
        window = np.where(jump_like[a:b], astep[a:b], -np.inf)
        centre = a + int(np.argmax(window)) if b > a and np.isfinite(window).any() else a

        # Measure at the width that saw this step most clearly.
        w = int(scale_at[centre])
        # Levels either side, from the trend outside the disturbed region.
        la, ra = max(0, centre - 2 * w), max(0, centre - w)
        lb, rb = min(n, centre + w), min(n, centre + 2 * w)
        if ra <= la or rb <= lb:
            continue
        sl_l, ic_l = _rolling_line(x[la:ra], y[la:ra], ra - la)
        sl_r, ic_r = _rolling_line(x[lb:rb], y[lb:rb], rb - lb)
        left_model = lambda t: ic_l[0] + sl_l[0] * t   # noqa: E731
        right_model = lambda t: ic_r[0] + sl_r[0] * t  # noqa: E731

        # The transition occupies the stretch where the curve has left the
        # regime behind it and not yet joined the one ahead.
        tol = 3.0 * sigma if sigma > 0 else 3.0 * spread
        i0 = centre
        while i0 > max(0, centre - 3 * w) and abs(y[i0] - left_model(x[i0])) > tol:
            i0 -= 1
        i1 = centre
        while i1 < min(n - 1, centre + 3 * w) and abs(y[i1] - right_model(x[i1])) > tol:
            i1 += 1
        if i1 <= i0:
            i0, i1 = max(0, centre - 1), min(n - 1, centre + 1)

        # Do the two regimes actually meet? Extended, the line before and the
        # line after a *bend* cross at the bend itself, so the apparent step
        # is only the angle between them. The lines either side of a real
        # discontinuity cross far away, or not at all. This is the geometric
        # statement of the difference, and unlike a threshold on the step it
        # does not care how sharp the bend is.
        db = float(sl_l[0] - sl_r[0])
        if abs(db) > 1e-300:
            cross = (float(ic_r[0]) - float(ic_l[0])) / db
            margin = 0.5 * (x[min(n - 1, centre + w)] - x[max(0, centre - w)])
            if x[i0] - margin <= cross <= x[i1] + margin:
                continue

        y_before = float(left_model(x[i0]))
        y_after = float(right_model(x[i1]))
        drop = y_after - y_before
        drop_sigma = abs(drop) / sigma if sigma > 0 else float("inf")
        if drop_sigma < min_drop_sigma:
            continue
        peak = float(np.max(np.abs(d[i0:i1 + 1]))) if i1 > i0 else 0.0
        events.append(Event(
            i0=int(i0), i1=int(i1), x0=float(x[i0]), x1=float(x[i1]),
            width=float(x[i1] - x[i0]),
            y_before=y_before, y_after=y_after, drop=float(drop),
            drop_sigma=float(drop_sigma), peak_slope=peak,
            sharpness=float(peak / base) if base > 0 else float("inf"),
        ))
    return events


@dataclass
class Boundary:
    index: int                  # sample index of the boundary
    x: float
    kind: str                   # "jump" | "knee" | "break"
    y_before: float             # left fit evaluated at x
    y_after: float              # right fit evaluated at x
    gap: float                  # y_after - y_before
    gap_sigma: float            # gap in units of the local standard error
    slope_before: float
    slope_after: float
    slope_ratio: float
    left_model: str
    right_model: str
    x_lo: float = float("nan")  # start of the transition zone (jumps)
    x_hi: float = float("nan")  # end of it
    width: float = 0.0
    sharpness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "index", "x", "kind", "y_before", "y_after", "gap", "gap_sigma",
            "slope_before", "slope_after", "slope_ratio", "left_model",
            "right_model", "x_lo", "x_hi", "width", "sharpness")}


@dataclass
class Segmentation:
    segments: list[Segment]
    boundaries: list[Boundary]
    events: list[Event]
    sigma: float                # noise level used
    n_segments: int
    cost: float
    bic: float
    chosen_by: str              # "penalty" | "requested"
    shape: str                  # e.g. "linear -> jump -> constant"
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "boundaries": [b.to_dict() for b in self.boundaries],
            "events": [e.to_dict() for e in self.events],
            "sigma": self.sigma,
            "n_segments": self.n_segments,
            "cost": self.cost,
            "bic": self.bic,
            "chosen_by": self.chosen_by,
            "shape": self.shape,
            "candidates": self.candidates,
        }


# ------------------------------------------------------------------- moments
class _Moments:
    """Prefix sums that give any interval's polynomial fit in constant time.

    x is mapped to [0, 1] first: the fourth moment of a raw time axis would
    otherwise be ~1e11 and the differences would lose their leading digits.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_lo = float(x[0])
        self.x_scale = float(x[-1] - x[0]) or 1.0
        u = (x - self.x_lo) / self.x_scale
        # y is centred once, globally: an energy column sits at -20080 and its
        # square at 4e8, so the residual sum of squares would otherwise be a
        # difference of two nearly equal large numbers. Centring shifts only
        # the fitted constant, which leaves every residual untouched.
        self.y_mean = float(np.mean(y))
        y = y - self.y_mean
        self.u = u
        self.y = y
        z = np.ones_like(u)
        self.S = []
        for k in range(2 * MAX_DEGREE + 1):
            self.S.append(np.concatenate(([0.0], np.cumsum(z))))
            z = z * u
        z = np.ones_like(u)
        self.T = []
        for k in range(MAX_DEGREE + 1):
            self.T.append(np.concatenate(([0.0], np.cumsum(z * y))))
            z = z * u
        self.Y2 = np.concatenate(([0.0], np.cumsum(y * y)))

    def rss(self, i: int, j: int, degree: int) -> float:
        """Residual sum of squares of the best degree-d fit on [i, j)."""
        return float(self.rss_batch(np.array([i]), np.array([j]), degree)[0])

    def rss_batch(self, i: np.ndarray, j: np.ndarray, degree: int) -> np.ndarray:
        """The same, for many intervals at once.

        The dynamic programme asks for O(m^2) interval fits. The normal
        equations are 1x1, 2x2 or 3x3 and symmetric, so they are solved here
        in closed form rather than through a general routine — which is the
        difference between a fifth of a second per curve and a hundredth.

        An interval whose normal matrix is degenerate returns infinity: that
        degree cannot be fitted there, and the partitioner should not use it.
        """
        s = [self.S[k][j] - self.S[k][i] for k in range(2 * degree + 1)]
        t = [self.T[k][j] - self.T[k][i] for k in range(degree + 1)]
        tot = self.Y2[j] - self.Y2[i]

        if degree == 0:
            det = s[0]
            ok = det > 0
            fit = np.where(ok, t[0] * t[0] / np.where(ok, det, 1.0), 0.0)
        elif degree == 1:
            det = s[0] * s[2] - s[1] * s[1]
            ok = np.abs(det) > 1e-14 * np.maximum(s[0] * s[2], 1e-300)
            safe = np.where(ok, det, 1.0)
            c0 = (s[2] * t[0] - s[1] * t[1]) / safe
            c1 = (s[0] * t[1] - s[1] * t[0]) / safe
            fit = c0 * t[0] + c1 * t[1]
        else:
            # Central moments about the interval's own mean. Taken raw, the
            # third and fourth moments of a short interval far from the origin
            # lose most of their significant digits to cancellation, and a
            # fit that "explains" more than the data contains would hand the
            # partitioner a free segment of zero cost.
            mu = s[1] / np.maximum(s[0], 1e-300)
            c_0 = s[0]
            c_2 = s[2] - mu * s[1]
            c_3 = s[3] - 3 * mu * s[2] + 2 * mu * mu * s[1]
            c_4 = s[4] - 4 * mu * s[3] + 6 * mu * mu * s[2] - 3 * mu ** 3 * s[1]
            d_0 = t[0]
            d_1 = t[1] - mu * t[0]
            d_2 = t[2] - 2 * mu * t[1] + mu * mu * t[0]

            m00 = c_2 * c_4 - c_3 * c_3
            m01 = c_2 * c_3
            m02 = -c_2 * c_2
            m11 = c_0 * c_4 - c_2 * c_2
            m12 = -c_0 * c_3
            m22 = c_0 * c_2
            det = c_0 * m00 - c_2 * c_2 * c_2
            ok = np.abs(det) > 1e-12 * np.maximum(c_0 * c_2 * c_4, 1e-300)
            safe = np.where(ok, det, 1.0)
            k0 = (m00 * d_0 + m01 * d_1 + m02 * d_2) / safe
            k1 = (m01 * d_0 + m11 * d_1 + m12 * d_2) / safe
            k2 = (m02 * d_0 + m12 * d_1 + m22 * d_2) / safe
            fit = k0 * d_0 + k1 * d_1 + k2 * d_2

        return np.where(ok, np.maximum(tot - fit, 0.0), np.inf)


def fit_segment(x: np.ndarray, y: np.ndarray, i0: int, i1: int, degree: int) -> Segment:
    """Refit an interval directly, in a centred basis, for reporting."""
    xs, ys = x[i0:i1], y[i0:i1]
    x_mid = float(0.5 * (xs[0] + xs[-1]))
    u = xs - x_mid
    deg = min(degree, max(0, len(xs) - 2))
    V = np.vander(u, deg + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, ys, rcond=None)
    pred = V @ coeffs
    resid = ys - pred
    rss = float(resid @ resid)
    n = len(xs)
    dof = max(1, n - (deg + 1))
    var = float(np.var(ys))
    r2 = 1.0 - rss / (n * var) if var > 0 else 1.0

    seg = Segment(
        i0=int(i0), i1=int(i1), x0=float(xs[0]), x1=float(xs[-1]),
        degree=int(deg), model=MODEL_NAMES[int(deg)], coeffs=[float(c) for c in coeffs],
        x_mid=x_mid, rss=rss, sigma=float(np.sqrt(rss / dof)), r2=float(r2), n=int(n),
        y_start=0.0, y_end=0.0, slope_start=0.0, slope_end=0.0,
        curvature=float(2 * coeffs[2]) if deg >= 2 else 0.0,
    )
    seg.y_start = float(seg.evaluate(seg.x0))
    seg.y_end = float(seg.evaluate(seg.x1))
    seg.slope_start = seg.slope(seg.x0)
    seg.slope_end = seg.slope(seg.x1)
    return seg


# ------------------------------------------------------------------------ DP
def _interval_costs(mom: _Moments, cuts: np.ndarray, sigma2: float, min_pts: int,
                    param_penalty: float) -> np.ndarray:
    """Cost of every candidate interval, as an m x m matrix (inf where invalid).

    Cost is the residual sum of squares in units of the noise variance plus a
    penalty per fitted coefficient, minimised over degree 0, 1 and 2 — so a
    straight line beats a parabola unless the parabola earns its coefficient.
    """
    m = cuts.size
    a_idx, b_idx = np.triu_indices(m, k=1)
    i, j = cuts[a_idx], cuts[b_idx]
    valid = (j - i) >= min_pts

    best = np.full(i.size, np.inf)
    if valid.any():
        iv, jv = i[valid], j[valid]
        stack = []
        for d in range(MAX_DEGREE + 1):
            long_enough = (jv - iv) > d + 2
            c = np.full(iv.size, np.inf)
            if long_enough.any():
                r = mom.rss_batch(iv[long_enough], jv[long_enough], d)
                c[long_enough] = r / sigma2 + param_penalty * (d + 1)
            stack.append(c)
        best[valid] = np.min(np.stack(stack), axis=0)

    C = np.full((m, m), np.inf)
    C[a_idx, b_idx] = best
    return C


def _partition(mom: _Moments, n: int, k_max: int, sigma2: float, stride: int,
               min_pts: int, param_penalty: float) -> tuple[list[float], dict[int, list[int]]]:
    """Optimal partition into exactly k segments, for every k up to k_max.

    Exact dynamic programming, vectorised over the left endpoint: for each k,
    ``best[k][b] = min_a best[k-1][a] + C[a][b]`` is one broadcast minimum.
    """
    cut_list = list(range(0, n, stride))
    if cut_list[-1] != n:
        cut_list.append(n)
    cuts = np.asarray(cut_list, dtype=int)
    m = cuts.size

    C = _interval_costs(mom, cuts, sigma2, min_pts, param_penalty)

    best = np.full((k_max + 1, m), np.inf)
    back = np.full((k_max + 1, m), -1, dtype=int)
    best[0, 0] = 0.0
    for k in range(1, k_max + 1):
        trial = best[k - 1][:, None] + C           # (a, b); only ever +inf
        a_star = np.argmin(trial, axis=0)
        vals = trial[a_star, np.arange(m)]
        best[k] = vals
        back[k] = np.where(np.isfinite(vals), a_star, -1)

    costs = [float(best[k, m - 1]) for k in range(k_max + 1)]
    paths: dict[int, list[int]] = {}
    for k in range(1, k_max + 1):
        if not np.isfinite(best[k, m - 1]):
            continue
        edges: list[int] = []
        b = m - 1
        ok = True
        for kk in range(k, 0, -1):
            a = int(back[kk, b])
            if a < 0:
                ok = False
                break
            edges.append(int(cuts[b]))
            b = a
        if ok and b == 0:
            paths[k] = [0] + sorted(edges)
    return costs, paths


def _classify_boundary(left: Segment, right: Segment, x_b: float, sigma: float,
                       allow_step: bool = False) -> tuple[str, float, float]:
    """Knee, step, or a weak break, from the two fits meeting at x_b.

    When transition detection is on, the event finder is the sole authority on
    discontinuities and this is never allowed to call one: two polynomials
    meeting inside a smooth stretch disagree a little by construction, and
    reading that disagreement as a step turns one transition into three.
    """
    y_l = float(left.evaluate(x_b))
    y_r = float(right.evaluate(x_b))
    gap = y_r - y_l
    # Against the raw noise, not the fits' standard error: "the step is ten
    # times the scatter" is a statement a reader can check by eye.
    gap_sigma = abs(gap) / sigma if sigma > 0 else 0.0

    if allow_step and gap_sigma >= JUMP_SIGMA:
        return "step", gap, gap_sigma
    s_l, s_r = left.slope(x_b), right.slope(x_b)
    scale = max(abs(s_l), abs(s_r), 1e-30)
    kind = "knee" if abs(s_r - s_l) / scale > 0.25 else "break"
    return kind, gap, gap_sigma


def describe_shape(segments: list[Segment], boundaries: list[Boundary]) -> str:
    """A one-line grammar of the curve: 'linear -> jump -> constant'."""
    parts: list[str] = []
    for i, s in enumerate(segments):
        if i > 0:
            b = boundaries[i - 1]
            parts.append(b.kind)
        name = s.model
        if s.degree == 0:
            name = "plateau"
        elif s.degree == 1 and abs(s.slope_start) < 1e-12:
            name = "plateau"
        parts.append(name)
    return " -> ".join(parts)


def _best_segments(x: np.ndarray, y: np.ndarray, bounds: list[int], sigma2: float,
                   param_penalty: float) -> list[Segment]:
    out: list[Segment] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        best_seg, best_cost = None, float("inf")
        for d in range(MAX_DEGREE + 1):
            if b - a <= d + 2:
                continue
            s = fit_segment(x, y, a, b, d)
            c = s.rss / sigma2 + param_penalty * (d + 1)
            if c < best_cost:
                best_seg, best_cost = s, c
        out.append(best_seg or fit_segment(x, y, a, b, 0))
    return out


def _allocate(per_stretch: list[list[float]], total: int) -> list[int] | None:
    """Split a budget of `total` regimes across stretches at least total cost.

    Each stretch already knows its cost for every possible piece count, so the
    optimal split is a small dynamic programme over those tables.
    """
    INF = float("inf")
    n_s = len(per_stretch)
    best = [[INF] * (total + 1) for _ in range(n_s + 1)]
    back = [[0] * (total + 1) for _ in range(n_s + 1)]
    best[0][0] = 0.0
    for s in range(1, n_s + 1):
        table = per_stretch[s - 1]
        for used in range(1, total + 1):
            for take in range(1, min(used, len(table) - 1) + 1):
                prev = best[s - 1][used - take]
                if prev == INF or not np.isfinite(table[take]):
                    continue
                v = prev + table[take]
                if v < best[s][used]:
                    best[s][used] = v
                    back[s][used] = take
    if best[n_s][total] == INF:
        return None
    alloc, used = [], total
    for s in range(n_s, 0, -1):
        take = back[s][used]
        alloc.append(take)
        used -= take
    return list(reversed(alloc))


def segment_curve(
    x: np.ndarray,
    y: np.ndarray,
    n_segments: int | None = None,
    sigma: float | None = None,
    max_segments: int = MAX_SEGMENTS,
    min_points: int = MIN_SEGMENT_POINTS,
    penalty_scale: float = 1.0,
    stride: int | None = None,
    detect_transitions: bool = True,
    event_threshold: float = 8.0,
) -> Segmentation:
    """Partition a curve into regimes separated by transitions.

    Transitions are located first, as stretches where the curve moves far
    faster than it does anywhere else, and are then held out of the fitting:
    a drop is an event with a width, not a regime. The smooth stretches left
    over are each partitioned exactly, and a budget of regimes is shared
    between them optimally.

    ``n_segments`` fixes the number of smooth regimes; None lets the penalised
    criterion choose. ``penalty_scale`` above 1 asks for a simpler reading of
    the curve, below 1 for a more detailed one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = x.size
    if n < 2 * min_points:
        raise ValueError(f"need at least {2 * min_points} finite points, got {n}")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]

    from .columns import estimate_noise
    sig = float(sigma) if sigma and sigma > 0 else estimate_noise(y)
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(y)) or 1.0
    sigma2 = sig * sig

    events = detect_events(x, y, sig, threshold=event_threshold) if detect_transitions else []

    # The smooth stretches are what is left between the events. Each stretch
    # records the event that precedes it: an event whose following stretch is
    # too short to fit is dropped, and the indices must not silently shift.
    stretches: list[tuple[int, int, Event | None]] = []
    prev = 0
    pending: Event | None = None
    for e in events:
        if e.i0 - prev >= min_points:
            stretches.append((prev, e.i0, pending))
            pending = e
        elif pending is None:
            pending = e
        prev = max(prev, e.i1)
    if n - prev >= min_points:
        stretches.append((prev, n, pending))
    if not stretches:
        stretches = [(0, n, None)]

    param_penalty = penalty_scale * np.log(max(n, 2))
    change_penalty = penalty_scale * 2.0 * np.log(max(n, 2))

    tables: list[list[float]] = []
    path_sets: list[dict[int, list[int]]] = []
    for a, b, _ev in stretches:
        m = b - a
        # Breakpoint resolution: transitions are located by the event detector
        # at full resolution, so the partitioner only has to place knees, and
        # a knee is not sharp enough to be worth a sample-by-sample search.
        sub_stride = stride or max(1, m // CANDIDATE_CUTS)
        k_max = min(max_segments, max(1, m // max(min_points, 1)))
        mom = _Moments(x[a:b], y[a:b])
        costs, paths = _partition(mom, m, k_max, sigma2, sub_stride, min_points, param_penalty)
        tables.append([c + change_penalty * max(0, k - 1) for k, c in enumerate(costs)])
        path_sets.append({k: [a + v for v in p] for k, p in paths.items()})

    auto_alloc = []
    for table, paths in zip(tables, path_sets):
        valid = [(table[k], k) for k in paths]
        auto_alloc.append(min(valid)[1] if valid else 1)

    chosen_by = "penalty"
    alloc = auto_alloc
    if n_segments is not None:
        want = int(np.clip(n_segments, len(stretches), max_segments * len(stretches)))
        got = _allocate(tables, want)
        if got:
            alloc, chosen_by = got, "requested"

    segments: list[Segment] = []
    boundaries: list[Boundary] = []
    for s_i, ((a, b, ev), k) in enumerate(zip(stretches, alloc)):
        paths = path_sets[s_i]
        if k not in paths:
            k = min(paths, key=lambda kk: abs(kk - k)) if paths else 1
        bounds = paths.get(k, [a, b])
        new = _best_segments(x, y, bounds, sigma2, param_penalty)

        if segments and ev is not None:
            left, right = segments[-1], new[0]
            x_b = float(0.5 * (ev.x0 + ev.x1))
            s_l, s_r = left.slope(left.x1), right.slope(right.x0)
            boundaries.append(Boundary(
                index=int(ev.i0), x=x_b, kind="transition",
                y_before=float(ev.y_before), y_after=float(ev.y_after),
                gap=float(ev.drop), gap_sigma=float(ev.drop_sigma),
                slope_before=float(s_l), slope_after=float(s_r),
                slope_ratio=float(s_r / s_l) if s_l != 0 else float("inf"),
                left_model=left.model, right_model=right.model,
                x_lo=float(ev.x0), x_hi=float(ev.x1),
                width=float(ev.width), sharpness=float(ev.sharpness),
            ))

        for i in range(len(new) - 1):
            left, right = new[i], new[i + 1]
            idx = left.i1
            x_b = float(0.5 * (x[min(idx, n - 1)] + x[max(idx - 1, 0)]))
            kind, gap, gap_sigma = _classify_boundary(
                left, right, x_b, sig, allow_step=not detect_transitions)
            s_l, s_r = left.slope(x_b), right.slope(x_b)
            boundaries.append(Boundary(
                index=int(idx), x=x_b, kind=kind,
                y_before=float(left.evaluate(x_b)), y_after=float(right.evaluate(x_b)),
                gap=float(gap), gap_sigma=float(gap_sigma),
                slope_before=float(s_l), slope_after=float(s_r),
                slope_ratio=float(s_r / s_l) if s_l != 0 else float("inf"),
                left_model=left.model, right_model=right.model,
                x_lo=x_b, x_hi=x_b,
            ))
        segments.extend(new)

    total_rss = sum(s.rss for s in segments)
    n_params = sum(s.degree + 1 for s in segments) + (len(segments) - 1)
    covered = max(1, sum(s.n for s in segments))
    bic = covered * np.log(max(total_rss / covered, 1e-300)) + n_params * np.log(covered)

    candidates = [{"n_segments": k, "cost": float(sum(t[min(k, len(t) - 1)] for t in tables))}
                  for k in range(1, min(max_segments, min(len(t) for t in tables)))]

    return Segmentation(
        segments=segments,
        boundaries=boundaries,
        events=events,
        sigma=sig,
        n_segments=len(segments),
        cost=float(sum(t[a] for t, a in zip(tables, alloc) if a < len(t))),
        bic=float(bic),
        chosen_by=chosen_by,
        shape=describe_shape(segments, boundaries),
        candidates=candidates,
    )
