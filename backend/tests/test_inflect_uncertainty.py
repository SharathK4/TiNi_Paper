"""Version 4 — intervals, robustness, and the whole-set picture."""

from __future__ import annotations

import numpy as np
import pytest

from tini.inflect.chorus import ChorusBuilder, sparkline
from tini.inflect.segment import model_curve, segment_curve
from tini.inflect.uncertainty import (
    autocorrelation_length,
    bootstrap_curve,
    moving_block_resample,
    stability_sweep,
)


def stepped(seed=0, n=800, step_at=6.0, sigma=0.05, rho=0.0):
    """A ramp, a step and a plateau, optionally with correlated noise."""
    x = np.linspace(0, 10, n)
    y = np.where(x < step_at, -2 * x, -2 * step_at + 9.0)
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    if rho:
        for i in range(1, n):
            e[i] = rho * e[i - 1] + np.sqrt(1 - rho * rho) * e[i]
    return x, y + e


# ------------------------------------------------------------ block length
def test_white_noise_needs_no_blocks():
    r = np.random.default_rng(0).normal(0, 1, 2000)
    assert autocorrelation_length(r) == 1


def test_correlated_noise_asks_for_longer_blocks():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 4000)
    for i in range(1, e.size):
        e[i] = 0.9 * e[i - 1] + 0.436 * e[i]
    assert autocorrelation_length(e) >= 5


def test_resampling_keeps_the_length_and_draws_from_the_data():
    r = np.arange(100.0)
    out = moving_block_resample(r, 10, np.random.default_rng(0))
    assert out.size == r.size
    assert set(np.unique(out)).issubset(set(r))


def test_blocks_preserve_correlation_that_points_would_destroy():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 4000)
    for i in range(1, e.size):
        e[i] = 0.9 * e[i - 1] + 0.436 * e[i]

    def lag1(v):
        v = v - v.mean()
        return float(v[:-1] @ v[1:]) / float(v @ v)

    blocked = moving_block_resample(e, 40, rng)
    pointwise = moving_block_resample(e, 1, rng)
    assert lag1(blocked) > 0.6
    assert abs(lag1(pointwise)) < 0.15


# --------------------------------------------------------------- bootstrap
def test_the_fitted_curve_covers_every_sample():
    x, y = stepped()
    seg = segment_curve(x, y)
    m = model_curve(x, seg)
    assert np.isfinite(m).all()
    # Where a regime was fitted it should follow the data closely. Across a
    # transition it deliberately does not: no regime is fitted there, and the
    # straight join is what the bootstrap then replaces with the real shape.
    inside = np.zeros(x.size, bool)
    for s in seg.segments:
        inside[s.i0:s.i1] = True
    assert np.abs(m - y)[inside].max() < 0.5


def test_intervals_bracket_the_estimate_and_are_not_absurd():
    x, y = stepped()
    seg = segment_curve(x, y)
    u = bootstrap_curve(x, y, seg, n_resamples=40, seed=0)
    t = u.transition
    assert t is not None
    for key in ("position", "onset", "step"):
        lo, hi = t[key]["ci"]
        assert lo <= t[key]["value"] <= hi
    # A step this clear should be located to a small fraction of the range.
    lo, hi = t["position"]["ci"]
    assert hi - lo < 0.5


def test_the_transition_is_preserved_so_detection_does_not_collapse():
    """The null must keep the transition the data actually shows.

    Joining the two sides with a straight line instead hands every replicate
    a gentler transition than the measured one, and the detector then misses
    it about half the time — an interval describing a curve nobody recorded.
    """
    x, y = stepped()
    seg = segment_curve(x, y)
    u = bootstrap_curve(x, y, seg, n_resamples=40, seed=1)
    assert u.detection_rate > 0.9


def test_a_slope_interval_stays_near_the_slope():
    x, y = stepped()
    seg = segment_curve(x, y)
    u = bootstrap_curve(x, y, seg, n_resamples=40, seed=2)
    ramp = min(u.segments, key=lambda s: s["x0"])
    lo, hi = ramp["slope"]["ci"]
    assert lo <= ramp["slope"]["value"] <= hi
    assert hi - lo < 0.5 * abs(ramp["slope"]["value"]) + 0.05


def test_noisier_data_gives_wider_intervals():
    """The interval has to answer to the data, not just look plausible.

    Pinning the transition to the observed samples makes this fail: the
    position then comes back with the same interval whether the step stands
    three hundred times above the noise or four.
    """
    widths: dict[str, list[float]] = {"position": [], "onset": []}
    for sigma in (0.03, 0.3, 1.5):
        x, y = stepped(seed=3, sigma=sigma)
        u = bootstrap_curve(x, y, segment_curve(x, y), n_resamples=50, seed=4)
        for key in widths:
            lo, hi = u.transition[key]["ci"]
            widths[key].append(hi - lo)

    # The value at the onset is set by the noise alone, so it must widen
    # every time; the position is also limited by the sample spacing, so it
    # is only required to widen over the full range.
    assert widths["onset"] == sorted(widths["onset"])
    assert widths["onset"][-1] > 4 * widths["onset"][0]
    assert widths["position"][-1] > widths["position"][0]


def test_correlated_noise_is_recognised():
    x, y = stepped(seed=5, rho=0.9, sigma=0.08)
    seg = segment_curve(x, y)
    u = bootstrap_curve(x, y, seg, n_resamples=30, seed=6)
    assert u.block_length > 1
    assert "correlated" in u.note


def test_a_curve_with_no_transition_still_reports_slopes():
    x = np.linspace(0, 10, 600)
    y = 2 * x + np.random.default_rng(7).normal(0, 0.05, 600)
    seg = segment_curve(x, y)
    u = bootstrap_curve(x, y, seg, n_resamples=30, seed=8)
    assert u.transition is None
    assert u.segments and u.segments[0]["slope"]["ci"] is not None


def test_uncertainty_serialises():
    import json
    x, y = stepped()
    seg = segment_curve(x, y)
    json.dumps(bootstrap_curve(x, y, seg, n_resamples=20).to_dict())


# --------------------------------------------------------------- stability
def test_a_clear_transition_holds_across_every_setting():
    x, y = stepped()
    seg = segment_curve(x, y)
    s = stability_sweep(x, y, seg, x_unit="s")
    assert s.holds
    assert s.spread < 0.3
    found = [p for p in s.points if p.found]
    assert len(found) >= 0.75 * len(s.points)
    assert all(abs(p.position - 6.0) < 0.6 for p in found)


def test_the_sweep_covers_more_than_one_kind_of_setting():
    x, y = stepped()
    s = stability_sweep(x, y, segment_curve(x, y))
    assert {p.setting for p in s.points} == {"threshold", "penalty", "regimes"}


def test_a_curve_with_nothing_in_it_says_so():
    x = np.linspace(0, 10, 500)
    y = np.random.default_rng(9).normal(0, 1, 500)
    s = stability_sweep(x, y, segment_curve(x, y))
    assert not s.holds
    assert "No setting" in s.sentence or "settings find" in s.sentence


# ------------------------------------------------------------------ chorus
def test_the_chorus_arranges_curves_both_ways():
    b = ChorusBuilder()
    for i in range(12):
        x, y = stepped(seed=i, step_at=6.0 + 0.1 * i)
        b.add(f"c{i}", x, y, 6.0 + 0.1 * i)
    ch = b.build("strain", "stress", "%", "GPa")
    assert ch.measured and ch.aligned
    assert ch.measured.n_curves == 12
    assert ch.aligned.grid[0] < 0 < ch.aligned.grid[-1]


def test_alignment_sharpens_the_shared_step():
    """The point of aligning: the step stops being smeared by the jitter."""
    b = ChorusBuilder()
    for i in range(24):
        at = 5.0 + 0.4 * (i % 6)
        x, y = stepped(seed=i, step_at=at, sigma=0.02)
        b.add(f"c{i}", x, y, at)
    ch = b.build()

    def steepest(a):
        v = np.array([g if g is not None else np.nan for g in a.median], dtype=float)
        return np.nanmax(np.abs(np.diff(v)))

    assert steepest(ch.aligned) > 1.5 * steepest(ch.measured)


def test_curves_without_a_transition_are_left_out_of_the_alignment():
    b = ChorusBuilder()
    for i in range(6):
        x, y = stepped(seed=i)
        b.add(f"c{i}", x, y, 6.0)
    for i in range(3):
        x, y = stepped(seed=100 + i)
        b.add(f"flat{i}", x, y, None)
    ch = b.build()
    assert ch.measured.n_curves == 9
    assert ch.aligned.n_curves == 6


def test_chorus_of_nothing_is_empty_not_broken():
    ch = ChorusBuilder().build()
    assert ch.measured is None and ch.aligned is None


def test_sparkline_is_short_and_scaled():
    x, y = stepped()
    s = sparkline(x, y, points=40)
    assert len(s) == 40
    assert min(s) == pytest.approx(0, abs=0.01)
    assert max(s) == pytest.approx(1, abs=0.01)


def test_sparkline_of_a_flat_line_does_not_divide_by_zero():
    x = np.linspace(0, 1, 100)
    assert all(np.isfinite(v) for v in sparkline(x, np.ones_like(x)))
