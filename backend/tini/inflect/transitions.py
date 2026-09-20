"""Turn a segmentation into something a person can read.

The interface for Version 4 is a narrated sequence of annotations over one
chart, so the reading of a curve is produced here rather than in the browser:
the sentence, the numbers it contains, and where on the chart each one points.
Keeping it in Python means the interpretation is testable, and means two
clients would say the same thing about the same curve.

A *beat* is one step of that narration. It carries a sentence with named slots,
the values for those slots, and the annotations to draw while it is on screen.

On the choice of transition point: the value reported is the response at the
*onset* of the transition — the last state before the curve leaves its regime
— not the largest value reached. On a noisy trace the extremum is a property
of the noise, and it drifts with however much smoothing was applied; the onset
is a property of the curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .segment import Boundary, Segment, Segmentation


@dataclass
class Annotation:
    """One mark on the chart. The frontend knows how to draw each kind."""

    kind: str                 # region | vline | hline | circle | bracket | slope | point | curve
    tone: str = "accent"      # accent (measured) | model (fitted) | muted
    x0: float | None = None
    x1: float | None = None
    y0: float | None = None
    y1: float | None = None
    text: str | None = None
    detail: str | None = None
    points: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Slot:
    """A number inside a sentence that the reader may take hold of."""

    name: str
    value: float
    unit: str | None = None
    decimals: int = 2
    control: str | None = None       # the setting it edits, if any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Beat:
    id: str
    kind: str
    title: str
    sentence: str                     # contains {slot} placeholders
    slots: list[Slot] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    stats: list[dict[str, Any]] = field(default_factory=list)
    focus: list[float] | None = None  # x-range worth zooming to
    # "measured": the sentence states numbers computed by a procedure.
    # "read": it also asserts a classification the software made by a rule
    # ("this is a transition", "this stretch is quadratic").
    tier: str = "measured"
    findings: list[str] = field(default_factory=list)   # ledger ids it draws on

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "sentence": self.sentence,
            "tier": self.tier,
            "findings": self.findings,
            "slots": [s.to_dict() for s in self.slots],
            "annotations": [a.to_dict() for a in self.annotations],
            "stats": self.stats,
        }
        if self.focus:
            d["focus"] = self.focus
        return d


# ------------------------------------------------------------------ language
def _unit_pair(y_unit: str | None, x_unit: str | None) -> str | None:
    if y_unit and x_unit:
        return f"{y_unit}/{x_unit}"
    if y_unit:
        return f"{y_unit} per unit"
    return None


def _round_to(value: float, digits: int = 3) -> int:
    """How many decimals to show so a number reads cleanly."""
    if value == 0 or not np.isfinite(value):
        return 2
    mag = int(np.floor(np.log10(abs(value))))
    return int(np.clip(digits - 1 - mag, 0, 6))


class _Words:
    """Fills the fixed parts of a sentence, leaving only the numbers as slots.

    Labels and units never change while a reading is on screen, so they are
    resolved here; what stays in braces is exactly what the reader can take
    hold of and alter.
    """

    def __init__(self, x_label: str, y_label: str, x_unit: str | None,
                 y_unit: str | None, slope_unit: str | None):
        self.map = {
            "x_label": x_label,
            "y_label": y_label,
            "x_unit": x_unit or "",
            "y_unit": y_unit or "",
            "slope_unit": slope_unit or "",
        }

    def __call__(self, sentence: str, **extra: str) -> str:
        for k, v in {**self.map, **extra}.items():
            sentence = sentence.replace("{" + k + "}", v)
        # A missing unit leaves a double space and a space before punctuation.
        return " ".join(sentence.split()).replace(" ,", ",").replace(" .", ".")


def regime_phrase(seg: Segment, y_unit: str | None, x_unit: str | None) -> str:
    if seg.model == "constant":
        return "holds steady"
    if seg.model == "linear":
        return "rises in a straight line" if seg.slope_start > 0 else "falls in a straight line"
    return "curves upward" if seg.curvature > 0 else "curves over"


# ------------------------------------------------------------------ findings
@dataclass
class Reading:
    x_label: str
    y_label: str
    x_unit: str | None
    y_unit: str | None
    shape: str
    headline: str
    beats: list[Beat]
    quantities: list[dict[str, Any]]
    transition: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "shape": self.shape,
            "headline": self.headline,
            "beats": [b.to_dict() for b in self.beats],
            "quantities": self.quantities,
            "transition": self.transition,
        }


def _transition_quantities(b: Boundary, x: np.ndarray, y: np.ndarray,
                           x_unit: str | None, y_unit: str | None) -> dict[str, Any]:
    peak_i = int(np.argmax(np.abs(y)))
    return {
        "x": b.x,
        "x_lo": b.x_lo,
        "x_hi": b.x_hi,
        "width": b.width,
        "onset_value": b.y_before,
        "final_value": b.y_after,
        "step": b.gap,
        "step_sigma": b.gap_sigma,
        "sharpness": b.sharpness,
        "peak_value": float(y[peak_i]),
        "peak_x": float(x[peak_i]),
        "onset_vs_peak": float(b.y_before - y[peak_i]),
        "x_unit": x_unit,
        "y_unit": y_unit,
    }


def read_curve(
    x: np.ndarray,
    y: np.ndarray,
    seg: Segmentation,
    x_label: str = "x",
    y_label: str = "y",
    x_unit: str | None = None,
    y_unit: str | None = None,
    smoothing_points: int | None = None,
) -> Reading:
    """Describe a segmented curve as a sequence of narration beats."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    beats: list[Beat] = []
    quantities: list[dict[str, Any]] = []
    slope_unit = _unit_pair(y_unit, x_unit)
    words = _Words(x_label, y_label, x_unit, y_unit, slope_unit)

    x_span = [float(x[0]), float(x[-1])]
    transitions = [b for b in seg.boundaries if b.kind == "transition"]
    knees = [b for b in seg.boundaries if b.kind == "knee"]

    # --- 1. the curve itself ------------------------------------------------
    beats.append(Beat(
        id="overview",
        kind="overview",
        title="The measurement",
        tier="measured", findings=["noise"],
        sentence=words("{n} points of {y_label} against {x_label}, with a scatter of {noise} {y_unit} about the trend."),
        slots=[
            Slot("n", float(x.size), decimals=0),
            Slot("noise", seg.sigma, y_unit, _round_to(seg.sigma)),
        ],
        annotations=[Annotation(kind="curve", tone="accent")],
        stats=[
            {"label": "points", "value": int(x.size)},
            {"label": f"{x_label} range", "value": f"{x[0]:.3g} to {x[-1]:.3g}", "unit": x_unit},
            {"label": "noise", "value": round(float(seg.sigma), 6), "unit": y_unit},
        ],
    ))

    # --- 2. the shape -------------------------------------------------------
    n_reg = len(seg.segments)
    beats.append(Beat(
        id="shape",
        kind="shape",
        title="What the curve does",
        tier="read", findings=["shape"],
        sentence=words("Read at this level of detail, the curve has {k} regimes: {shape}.", shape=seg.shape),
        slots=[Slot("k", float(n_reg), decimals=0, control="n_segments",
                    minimum=1, maximum=8, step=1)],
        annotations=[
            Annotation(kind="region", tone="muted", x0=s.x0, x1=s.x1,
                       text=s.model if s.model != "constant" else "plateau")
            for s in seg.segments
        ],
        stats=[{"label": "regimes", "value": n_reg}, {"label": "shape", "value": seg.shape}],
    ))

    # --- 3. each regime -----------------------------------------------------
    for i, s in enumerate(seg.segments):
        ann = [Annotation(kind="region", tone="model", x0=s.x0, x1=s.x1),
               Annotation(kind="curve", tone="model", x0=s.x0, x1=s.x1)]
        stats = [
            {"label": "model", "value": s.model},
            {"label": f"{x_label} from", "value": round(s.x0, 4), "unit": x_unit},
            {"label": f"{x_label} to", "value": round(s.x1, 4), "unit": x_unit},
            {"label": "scatter about the fit", "value": round(s.sigma, 6), "unit": y_unit},
        ]
        if s.degree == 1:
            ann.append(Annotation(kind="slope", tone="model",
                                  x0=s.x0, x1=s.x1, y0=s.y_start, y1=s.y_end,
                                  text=f"{s.slope_start:.4g}"))
            sentence = "Between {x0} and {x1} it {phrase}, at {slope} {slope_unit}."
            slots = [Slot("x0", s.x0, x_unit, _round_to(s.x0)),
                     Slot("x1", s.x1, x_unit, _round_to(s.x1)),
                     Slot("slope", s.slope_start, slope_unit, _round_to(s.slope_start))]
            stats.append({"label": "slope", "value": round(s.slope_start, 6), "unit": slope_unit})
            quantities.append({"name": f"slope of regime {i + 1}", "value": s.slope_start,
                               "unit": slope_unit, "from": [s.x0, s.x1]})
        elif s.degree == 0:
            sentence = "Between {x0} and {x1} it {phrase} at {level} {y_unit}."
            slots = [Slot("x0", s.x0, x_unit, _round_to(s.x0)),
                     Slot("x1", s.x1, x_unit, _round_to(s.x1)),
                     Slot("level", s.y_start, y_unit, _round_to(s.y_start))]
            ann.append(Annotation(kind="hline", tone="model", y0=s.y_start, x0=s.x0, x1=s.x1))
            stats.append({"label": "level", "value": round(s.y_start, 6), "unit": y_unit})
            quantities.append({"name": f"plateau level (regime {i + 1})", "value": s.y_start,
                               "unit": y_unit, "from": [s.x0, s.x1]})
        else:
            sentence = ("Between {x0} and {x1} it {phrase}: the slope changes from "
                        "{s0} to {s1} {slope_unit}.")
            slots = [Slot("x0", s.x0, x_unit, _round_to(s.x0)),
                     Slot("x1", s.x1, x_unit, _round_to(s.x1)),
                     Slot("s0", s.slope_start, None, _round_to(s.slope_start)),
                     Slot("s1", s.slope_end, slope_unit, _round_to(s.slope_end))]
            stats.append({"label": "curvature", "value": round(s.curvature, 6)})
            quantities.append({"name": f"curvature of regime {i + 1}", "value": s.curvature,
                               "unit": None, "from": [s.x0, s.x1]})
        beats.append(Beat(
            id=f"regime-{i}", kind="regime",
            title=f"Regime {i + 1} of {n_reg} — {s.model}",
            tier="read",
            findings=[f"regime.{i}.model", f"regime.{i}.extent",
                      f"regime.{i}.level" if s.degree == 0 else f"regime.{i}.slope"]
                     + ([f"regime.{i}.curvature"] if s.degree == 2 else []),
            sentence=words(sentence, phrase=regime_phrase(s, y_unit, x_unit)),
            slots=slots, annotations=ann, stats=stats, focus=[s.x0, s.x1],
        ))

    # --- 4. the transitions -------------------------------------------------
    for j, b in enumerate(transitions):
        q = _transition_quantities(b, x, y, x_unit, y_unit)
        pad = max(b.width * 4, (x[-1] - x[0]) * 0.08)
        beats.append(Beat(
            id=f"transition-{j}", kind="transition",
            title="The transition" if len(transitions) == 1 else f"Transition {j + 1}",
            tier="read",
            findings=[f"transition.{j}.classified", f"transition.{j}.position",
                      f"transition.{j}.onset", f"transition.{j}.after",
                      f"transition.{j}.step", f"transition.{j}.width"],
            sentence=words("The curve leaves its regime at {onset_x} {x_unit}, where {y_label} is "
                           "{onset_y} {y_unit}, and settles at {final_y} {y_unit} — a step of {step} "
                           "{y_unit}, {sigma} times the scatter."),
            slots=[
                Slot("onset_x", b.x_lo, x_unit, _round_to(b.x_lo)),
                Slot("onset_y", b.y_before, y_unit, _round_to(b.y_before)),
                Slot("final_y", b.y_after, y_unit, _round_to(b.y_after)),
                Slot("step", b.gap, y_unit, _round_to(b.gap)),
                Slot("sigma", b.gap_sigma, None, 1),
            ],
            annotations=[
                Annotation(kind="region", tone="accent", x0=b.x_lo, x1=b.x_hi, text="transition"),
                Annotation(kind="circle", tone="accent", x0=b.x_lo, y0=b.y_before,
                           text="onset", detail=f"{b.y_before:.4g}"),
                Annotation(kind="circle", tone="model", x0=b.x_hi, y0=b.y_after,
                           text="after", detail=f"{b.y_after:.4g}"),
                Annotation(kind="bracket", tone="accent", x0=b.x_lo, x1=b.x_hi,
                           y0=b.y_before, y1=b.y_after,
                           text=f"step {b.gap:.3g}" + (f" {y_unit}" if y_unit else "")),
            ],
            stats=[
                {"label": f"onset {x_label}", "value": round(b.x_lo, 5), "unit": x_unit},
                {"label": f"onset {y_label}", "value": round(b.y_before, 5), "unit": y_unit},
                {"label": "after", "value": round(b.y_after, 5), "unit": y_unit},
                {"label": "step", "value": round(b.gap, 5), "unit": y_unit},
                {"label": "width", "value": round(b.width, 5), "unit": x_unit},
                {"label": "significance", "value": f"{b.gap_sigma:.1f} sigma"},
            ],
            focus=[max(x[0], b.x_lo - pad), min(x[-1], b.x_hi + pad)],
        ))
        quantities.append({"name": "transition point", "value": b.x_lo, "unit": x_unit,
                           "from": [b.x_lo, b.x_hi]})
        quantities.append({"name": f"{y_label} at transition", "value": b.y_before,
                           "unit": y_unit, "from": [b.x_lo, b.x_hi]})

        # Onset against extremum — the distinction the product is built on.
        if abs(q["onset_vs_peak"]) > 0.25 * seg.sigma:
            beats.append(Beat(
                id=f"onset-vs-peak-{j}", kind="caution",
                title="Onset, not peak",
                tier="measured",
                findings=["extremum.value", "extremum.position", f"transition.{j}.onset"],
                sentence=words("The largest {y_label} reached is {peak} {y_unit} at {peak_x} {x_unit}, "
                               "which is {diff} {y_unit} away from the onset. The extremum moves "
                               "with the noise and with smoothing; the onset does not."),
                slots=[
                    Slot("peak", q["peak_value"], y_unit, _round_to(q["peak_value"])),
                    Slot("peak_x", q["peak_x"], x_unit, _round_to(q["peak_x"])),
                    Slot("diff", abs(q["onset_vs_peak"]), y_unit, _round_to(q["onset_vs_peak"])),
                ],
                annotations=[
                    Annotation(kind="circle", tone="muted", x0=q["peak_x"], y0=q["peak_value"],
                               text="extremum"),
                    Annotation(kind="circle", tone="accent", x0=b.x_lo, y0=b.y_before,
                               text="onset"),
                ],
                stats=[
                    {"label": "extremum", "value": round(q["peak_value"], 5), "unit": y_unit},
                    {"label": "onset", "value": round(b.y_before, 5), "unit": y_unit},
                    {"label": "difference", "value": round(q["onset_vs_peak"], 5), "unit": y_unit},
                ],
                focus=[max(x[0], min(q["peak_x"], b.x_lo) - pad),
                       min(x[-1], max(q["peak_x"], b.x_hi) + pad)],
            ))

    # --- 5. the knees -------------------------------------------------------
    for j, b in enumerate(knees):
        beats.append(Beat(
            id=f"knee-{j}", kind="knee",
            title="A change of slope",
            tier="read",
            findings=[f"knee.{j}.classified", f"knee.{j}.position",
                      f"knee.{j}.slope_before", f"knee.{j}.slope_after"],
            sentence=words("At {x} {x_unit} the curve bends without stepping: the slope goes "
                           "from {s0} to {s1} {slope_unit}."),
            slots=[
                Slot("x", b.x, x_unit, _round_to(b.x)),
                Slot("s0", b.slope_before, None, _round_to(b.slope_before)),
                Slot("s1", b.slope_after, slope_unit, _round_to(b.slope_after)),
            ],
            annotations=[
                Annotation(kind="vline", tone="model", x0=b.x, text=f"{b.x:.4g}"),
                Annotation(kind="circle", tone="model", x0=b.x, y0=b.y_before, text="bend"),
            ],
            stats=[
                {"label": x_label, "value": round(b.x, 5), "unit": x_unit},
                {"label": "slope before", "value": round(b.slope_before, 6), "unit": slope_unit},
                {"label": "slope after", "value": round(b.slope_after, 6), "unit": slope_unit},
            ],
            focus=[max(x[0], b.x - (x[-1] - x[0]) * 0.12),
                   min(x[-1], b.x + (x[-1] - x[0]) * 0.12)],
        ))
        quantities.append({"name": "knee position", "value": b.x, "unit": x_unit,
                           "from": [b.x, b.x]})

    # --- headline -----------------------------------------------------------
    if transitions:
        main = max(transitions, key=lambda b: abs(b.gap))
        headline = (f"{y_label} leaves its regime at {x_label} = {main.x_lo:.4g}"
                    f"{' ' + x_unit if x_unit else ''}, stepping by {main.gap:.3g}"
                    f"{' ' + y_unit if y_unit else ''}")
        transition_summary = _transition_quantities(main, x, y, x_unit, y_unit)
    elif knees:
        headline = f"no step, but the slope changes {len(knees)} time(s)"
        transition_summary = None
    else:
        headline = f"one regime throughout: {seg.segments[0].model}"
        transition_summary = None

    return Reading(
        x_label=x_label, y_label=y_label, x_unit=x_unit, y_unit=y_unit,
        shape=seg.shape, headline=headline, beats=beats,
        quantities=quantities, transition=transition_summary,
    )
