"""Version 4 — regime segmentation and transition detection.

The curves here are built so the right answer is known by construction. The
two failure modes that matter in practice each have a test: reading one broad
transition as several regimes, and reading a bend as a discontinuity.
"""

from __future__ import annotations

import numpy as np
import pytest

from tini.inflect.segment import (
    _Moments,
    detect_events,
    fit_segment,
    segment_curve,
    step_statistic,
)


@pytest.fixture
def x():
    return np.linspace(0.0, 10.0, 600)


def noisy(y, sigma=0.05, seed=0):
    return y + np.random.default_rng(seed).normal(0, sigma, y.size)


def kinds(seg):
    return [b.kind for b in seg.boundaries]


def transitions(seg):
    return [b for b in seg.boundaries if b.kind == "transition"]


# --------------------------------------------------------------- the moments
def test_prefix_moments_match_a_direct_fit():
    rng = np.random.default_rng(0)
    x = np.linspace(2.0, 9.0, 400)
    y = 1.5 - 0.7 * x + 0.25 * x * x + rng.normal(0, 0.1, x.size)
    mom = _Moments(x, y)
    for i, j in [(0, 400), (10, 120), (250, 399), (100, 140)]:
        for degree in (0, 1, 2):
            direct = fit_segment(x, y, i, j, degree).rss
            assert mom.rss(i, j, degree) == pytest.approx(direct, rel=1e-6, abs=1e-9)


def test_short_interval_quadratic_stays_accurate():
    """The closed-form normal equations must not "explain" more than there is.

    Taken on raw moments this fit loses its significant digits on a short
    interval far from the origin and returns a residual of zero, which the
    partitioner would then spend segments collecting.
    """
    rng = np.random.default_rng(1)
    x = np.linspace(1000.0, 1001.0, 800)
    y = 5.0 + rng.normal(0, 1.0, x.size)
    mom = _Moments(x, y)
    for i in range(0, 700, 97):
        j = i + 12
        assert mom.rss(i, j, 2) == pytest.approx(fit_segment(x, y, i, j, 2).rss, rel=1e-4)
        assert mom.rss(i, j, 2) > 0.0


# ---------------------------------------------------------------- the basics
def test_a_straight_line_is_one_segment(x):
    seg = segment_curve(x, noisy(2 * x + 1))
    assert seg.n_segments == 1
    assert seg.segments[0].model == "linear"


def test_pure_noise_is_one_segment(x):
    seg = segment_curve(x, noisy(np.zeros_like(x), sigma=1.0, seed=5))
    assert seg.n_segments == 1
    assert not transitions(seg)


def test_a_parabola_is_one_quadratic_segment(x):
    seg = segment_curve(x, noisy(0.3 * (x - 5) ** 2))
    assert seg.n_segments == 1
    assert seg.segments[0].model == "quadratic"


# ----------------------------------------------------------------- the knees
def test_a_bilinear_knee_is_found_and_is_not_a_transition(x):
    y = np.where(x < 6, 2 * x, 12 + 0.3 * (x - 6))
    seg = segment_curve(x, noisy(y))
    assert seg.n_segments == 2
    assert kinds(seg) == ["knee"]
    assert seg.boundaries[0].x == pytest.approx(6.0, abs=0.15)
    assert seg.boundaries[0].slope_before == pytest.approx(2.0, abs=0.1)
    assert seg.boundaries[0].slope_after == pytest.approx(0.3, abs=0.1)


def test_a_plateau_between_two_ramps_gives_two_knees(x):
    y = np.piecewise(
        x, [x < 3, (x >= 3) & (x < 7), x >= 7],
        [lambda v: v, lambda v: 3.0, lambda v: 3 + 2 * (v - 7)],
    )
    seg = segment_curve(x, noisy(y, sigma=0.04))
    assert seg.n_segments == 3
    assert kinds(seg) == ["knee", "knee"]
    assert [b.x for b in seg.boundaries] == pytest.approx([3.0, 7.0], abs=0.2)


# ----------------------------------------------------------- the transitions
def test_a_step_is_reported_as_a_transition(x):
    y = np.where(x < 6, -2 * x, -3.0)
    seg = segment_curve(x, noisy(y))
    t = transitions(seg)
    assert len(t) == 1
    assert t[0].x == pytest.approx(6.0, abs=0.2)
    assert t[0].gap == pytest.approx(9.0, abs=0.4)
    assert t[0].gap_sigma > 20


def test_one_broad_transition_is_one_event_not_several():
    """A transition that takes a little while to happen is still one event."""
    x = np.linspace(0, 10, 1200)
    ramp = np.clip((x - 5.0) / 0.25, 0.0, 1.0)      # steps over a finite width
    y = -2 * x * (1 - ramp) + (-3.0) * ramp
    seg = segment_curve(x, noisy(y, sigma=0.06, seed=3))
    assert len(transitions(seg)) == 1


def test_a_sharp_knee_is_never_called_a_transition():
    """The regression that matters: a steep bend is not a discontinuity."""
    x = np.linspace(0, 10, 900)
    y = np.where(x < 5, 0.05 * x, 0.05 * 5 + 8.0 * (x - 5))
    seg = segment_curve(x, noisy(y, sigma=0.05, seed=4))
    assert not transitions(seg)
    assert "knee" in kinds(seg)


def test_two_separated_transitions_are_both_found():
    x = np.linspace(0, 20, 1600)
    y = np.zeros_like(x)
    y[x >= 6] += 5.0
    y[x >= 14] -= 8.0
    seg = segment_curve(x, noisy(y, sigma=0.1, seed=6))
    t = transitions(seg)
    assert len(t) == 2
    assert [b.x for b in t] == pytest.approx([6.0, 14.0], abs=0.3)


def test_transition_survives_a_steep_quadratic_background():
    """The case the derivative test cannot do.

    Against a ramp that is itself steepening, thresholding the rate of change
    cannot separate the transition from the background. The detrended step
    can, and this curve is built so the background rate at the step exceeds
    the rate anywhere earlier.
    """
    x = np.linspace(0, 4, 1000)
    y = 3.0 * x ** 2
    y[x >= 3.0] += 9.0
    seg = segment_curve(x, noisy(y, sigma=0.6, seed=7))
    t = transitions(seg)
    assert len(t) == 1
    assert t[0].x == pytest.approx(3.0, abs=0.1)
    assert t[0].gap == pytest.approx(9.0, abs=1.5)


def test_step_statistic_vanishes_on_a_straight_line():
    x = np.linspace(0, 5, 800)
    s = step_statistic(x, 3.0 * x - 2.0, 40)
    assert np.max(np.abs(s[40:-40])) < 1e-9 * np.ptp(3.0 * x)


def test_step_statistic_is_small_on_a_curve_and_large_on_a_step():
    """Curvature leaks a little into the statistic; a step dominates it."""
    x = np.linspace(0, 5, 800)
    curved = step_statistic(x, 2.0 * x ** 2 - 3 * x, 40)
    stepped = step_statistic(x, np.where(x < 2.5, 0.0, 1.0), 40)
    assert np.max(np.abs(curved[40:-40])) < 0.02 * np.max(np.abs(stepped[40:-40]))


def test_events_are_scale_free():
    """The same curve in different units must give the same answer."""
    x = np.linspace(0, 10, 800)
    y = np.where(x < 4, 1.0, 2.0)
    a = detect_events(x, noisy(y, sigma=0.02, seed=8), sigma=0.02)
    b = detect_events(x, noisy(y, sigma=0.02, seed=8) * 1000, sigma=20.0)
    assert len(a) == len(b) == 1
    assert a[0].x0 == pytest.approx(b[0].x0, abs=0.1)


# ------------------------------------------------------------------ controls
def test_asking_for_more_regimes_gives_more_regimes(x):
    y = noisy(np.where(x < 6, 2 * x, 12 + 0.3 * (x - 6)))
    auto = segment_curve(x, y)
    more = segment_curve(x, y, n_segments=4)
    assert more.n_segments > auto.n_segments
    assert more.chosen_by == "requested"


def test_a_higher_penalty_never_gives_more_regimes(x):
    y = noisy(np.piecewise(
        x, [x < 3, (x >= 3) & (x < 7), x >= 7],
        [lambda v: v, lambda v: 3.0, lambda v: 3 + 2 * (v - 7)]), sigma=0.04)
    counts = [segment_curve(x, y, penalty_scale=p).n_segments for p in (0.3, 1.0, 4.0)]
    assert counts == sorted(counts, reverse=True)


def test_segments_tile_the_curve_in_order(x):
    y = noisy(np.where(x < 6, -2 * x, -3.0))
    seg = segment_curve(x, y)
    for a, b in zip(seg.segments[:-1], seg.segments[1:]):
        assert a.x1 <= b.x0 + 1e-9
    assert [b.x for b in seg.boundaries] == sorted(b.x for b in seg.boundaries)


def test_result_is_serialisable(x):
    seg = segment_curve(x, noisy(np.where(x < 6, -2 * x, -3.0)))
    import json
    json.dumps(seg.to_dict())
