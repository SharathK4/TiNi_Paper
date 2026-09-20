"""Version 4 — the reading of a curve, and comparison across many curves."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tini.inflect.ensemble import (
    _shape_signature,
    analyse_ensemble,
    robust_scale,
    run_ensemble,
    summarise_curve,
)
from tini.inflect.segment import segment_curve
from tini.inflect.transitions import read_curve


def render(beat) -> str:
    """Substitute a beat's numbers, the way the interface will."""
    text = beat.sentence
    for s in beat.slots:
        text = text.replace("{" + s.name + "}", f"{s.value:.{s.decimals}f}")
    return text


@pytest.fixture
def stepped():
    x = np.linspace(0, 10, 900)
    y = np.where(x < 6, -2 * x, -3.0) + np.random.default_rng(0).normal(0, 0.05, 900)
    return x, y


# ------------------------------------------------------------------- reading
def test_every_slot_is_filled_and_no_placeholder_survives(stepped):
    x, y = stepped
    r = read_curve(x, y, segment_curve(x, y), "strain", "stress", "%", "GPa")
    for beat in r.beats:
        text = render(beat)
        assert "{" not in text and "}" not in text, f"unfilled placeholder in {beat.id}: {text}"
        assert text[0].isupper() or text[0].isdigit()
        assert text.rstrip().endswith(".")


def test_labels_and_units_reach_the_sentences(stepped):
    x, y = stepped
    r = read_curve(x, y, segment_curve(x, y), "strain", "stress", "%", "GPa")
    joined = " ".join(render(b) for b in r.beats)
    assert "stress" in joined and "strain" in joined and "GPa" in joined


def test_a_missing_unit_leaves_no_double_space(stepped):
    x, y = stepped
    r = read_curve(x, y, segment_curve(x, y), "x", "y")
    for beat in r.beats:
        assert "  " not in render(beat)


def test_the_transition_beat_reports_onset_not_extremum():
    """The distinction the product exists for.

    A noise spike early in the curve is larger than anything at the
    transition; the reported transition value must be the onset regardless.
    """
    x = np.linspace(0, 10, 900)
    y = np.where(x < 6, -2 * x, -3.0) + np.random.default_rng(3).normal(0, 0.05, 900)
    y[150] = -30.0                                    # a single wild sample
    r = read_curve(x, y, segment_curve(x, y), "strain", "stress", "%", "GPa")
    assert r.transition is not None
    assert r.transition["onset_value"] == pytest.approx(-12.0, abs=0.6)
    assert r.transition["peak_value"] == pytest.approx(-30.0, abs=0.1)
    # and the reader is told about the difference
    assert any(b.kind == "caution" for b in r.beats)


def test_the_regime_count_is_offered_as_a_control(stepped):
    x, y = stepped
    r = read_curve(x, y, segment_curve(x, y), "x", "y")
    shape = next(b for b in r.beats if b.id == "shape")
    slot = next(s for s in shape.slots if s.control == "n_segments")
    assert slot.minimum is not None and slot.maximum is not None


def test_annotations_stay_inside_the_data(stepped):
    x, y = stepped
    r = read_curve(x, y, segment_curve(x, y), "x", "y")
    for beat in r.beats:
        for a in beat.annotations:
            for key in ("x0", "x1"):
                v = getattr(a, key)
                if v is not None:
                    assert x[0] - 1e-9 <= v <= x[-1] + 1e-9


def test_reading_serialises(stepped):
    x, y = stepped
    r = read_curve(x, y, segment_curve(x, y), "x", "y", "s", "V")
    json.dumps(r.to_dict())


def test_a_featureless_curve_still_reads(stepped):
    x = np.linspace(0, 1, 400)
    y = np.random.default_rng(9).normal(0, 1, 400)
    r = read_curve(x, y, segment_curve(x, y), "x", "y")
    assert r.transition is None
    assert r.beats


# ------------------------------------------------------------------ ensemble
def test_robust_scale_ignores_a_contaminating_tail():
    v = np.concatenate([np.random.default_rng(0).normal(10, 1, 200), np.full(8, 1000.0)])
    med, scale = robust_scale(v)
    assert med == pytest.approx(10, abs=0.4)
    assert scale == pytest.approx(1.0, rel=0.3)


def make_curve(seed, step_at=6.0, amplitude=2.0, n=700):
    x = np.linspace(0, 10, n)
    y = np.where(x < step_at, -amplitude * x, -amplitude * step_at + 9.0)
    return x, y + np.random.default_rng(seed).normal(0, 0.05, n)


def test_an_ensemble_of_similar_curves_has_no_outliers():
    curves = {f"c{i}": make_curve(i) for i in range(25)}
    rep = run_ensemble(lambda k: curves[k], curves)
    assert rep.n_ok == 25
    assert rep.outliers == []
    assert rep.stats["transition_x"]["median"] == pytest.approx(6.0, abs=0.25)


def test_a_curve_from_another_population_is_singled_out():
    curves = {f"c{i}": make_curve(i) for i in range(25)}
    curves["odd"] = make_curve(99, amplitude=12.0)     # far larger range
    rep = run_ensemble(lambda k: curves[k], curves)
    names = [o.name for o in rep.outliers]
    assert names and names[0] == "odd"
    # Several measurements set it apart; the report names one of them.
    assert rep.outliers[0].feature in {"y_range", "first_slope", "transition_onset"}
    assert "spreads" in rep.outliers[0].sentence


def flat_step(seed, step_at=5.0, n=700):
    """A curve where only the transition's position varies between repeats."""
    x = np.linspace(0, 10, n)
    y = np.where(x < step_at, 0.0, 5.0)
    return x, y + np.random.default_rng(seed).normal(0, 0.05, n)


def test_an_outlier_reports_which_measurement_makes_it_one():
    curves = {f"c{i}": flat_step(i) for i in range(20)}
    curves["late"] = flat_step(99, step_at=8.0)
    rep = run_ensemble(lambda k: curves[k], curves)
    o = next(o for o in rep.outliers if o.name == "late")
    assert o.feature == "transition_x"
    assert o.value == pytest.approx(8.0, abs=0.3)
    assert "transition position" in o.sentence and "typical" in o.sentence


def test_failures_are_collected_not_raised():
    def loader(name):
        if name == "bad":
            raise ValueError("unreadable")
        return make_curve(1)
    rep = run_ensemble(loader, ["a", "bad", "c"])
    assert rep.n_ok == 2 and rep.n_failed == 1
    assert rep.failures[0]["name"] == "bad"
    assert "unreadable" in rep.failures[0]["error"]


def test_curves_group_by_structure_not_by_regime_count():
    assert _shape_signature("quadratic -> knee -> linear -> transition -> quadratic") == \
           _shape_signature("linear -> transition -> linear")
    assert _shape_signature("linear") != _shape_signature("linear -> transition -> linear")


def test_ensemble_report_serialises():
    curves = {f"c{i}": make_curve(i) for i in range(6)}
    rep = run_ensemble(lambda k: curves[k], curves)
    json.dumps(rep.to_dict())


def test_summary_of_an_impossible_curve_is_marked_not_raised():
    s = summarise_curve("tiny", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert not s.ok and s.error


def test_ensemble_of_nothing_does_not_crash():
    rep = analyse_ensemble([])
    assert rep.n_ok == 0 and rep.outliers == []
