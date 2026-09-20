"""The ledger: every finding, and who is responsible for it.

Three kinds of statement appear on the page, and a reader is entitled to
know which is which:

* **measured** — a number computed from the samples by a stated procedure.
  "The onset is at 2.148 %." The software is responsible for the arithmetic
  and nothing more.
* **read** — a judgement the software made by a stated rule. "This is a
  transition." "This stretch is quadratic." The rule is given with the
  finding, so the reader can disagree with the rule rather than with the
  number.
* **concluded** — what the researcher makes of it. "This is the onset of the
  martensitic transformation." The software cannot know this, does not claim
  it, and keeps it apart from the two tiers above wherever the findings
  travel.

Every finding carries its tier, its method or rule, the evidence it rests on,
where it sits on the chart, and — once they have been computed — its
bootstrap interval and how far it moves across the settings sweep. The beats
are a way of walking through the ledger; the ledger is the thing itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .measure import measure
from .segment import JUMP_SIGMA, Segmentation


@dataclass
class Finding:
    id: str
    label: str
    tier: str                            # "measured" | "read"
    value: float | None
    unit: str | None
    method: str                          # how the number was obtained
    rule: str | None = None              # for tier "read": why it was called that
    text: str | None = None              # a non-numeric value ("quadratic")
    evidence: dict[str, Any] = field(default_factory=dict)
    anchor: dict[str, float] = field(default_factory=dict)
    group: str = ""                      # transition.0, regime.2, ...
    interval: dict[str, Any] | None = None
    sensitivity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "id", "label", "tier", "value", "unit", "method", "rule", "text",
            "evidence", "anchor", "group", "interval", "sensitivity")}


def _pair(y_unit: str | None, x_unit: str | None) -> str | None:
    if y_unit and x_unit:
        return f"{y_unit}/{x_unit}"
    return f"{y_unit} per unit" if y_unit else None


def build_ledger(
    x: np.ndarray,
    y: np.ndarray,
    seg: Segmentation,
    x_label: str = "x",
    y_label: str = "y",
    x_unit: str | None = None,
    y_unit: str | None = None,
    uncertainty: dict[str, dict[str, Any]] | None = None,
    stability: dict[str, dict[str, Any]] | None = None,
    settings: dict[str, Any] | None = None,
) -> list[Finding]:
    """Everything the reading asserts, in order down the curve."""
    m = measure(x, y, seg)
    slope_u = _pair(y_unit, x_unit)
    curv_u = f"{slope_u}/{x_unit}" if slope_u and x_unit else None
    n = x.size
    settings = settings or {}
    out: list[Finding] = []

    def add(f: Finding) -> None:
        # The bootstrap takes the noise estimate as an input and generates
        # its replicates from it, so an interval on the noise from the same
        # bootstrap would be circular — and, because resampled residuals are
        # whiter than the originals, biased upward. It is not attached.
        if uncertainty and f.id in uncertainty and f.id != "noise":
            f.interval = uncertainty[f.id]
        if stability and f.id in stability:
            f.sensitivity = stability[f.id]
        out.append(f)

    # --- the noise ---------------------------------------------------------
    add(Finding(
        id="noise", group="curve", label="scatter about the trend", tier="measured",
        value=m["noise"], unit=y_unit,
        method="median absolute deviation of the second differences, scaled to a "
               "standard deviation; robust to the steps and bends that are the signal. "
               "No bootstrap interval: the bootstrap takes this value as its input",
        evidence={"n_samples": int(n)},
        anchor={"x0": float(x[0]), "x1": float(x[-1])},
    ))

    # --- the shape ---------------------------------------------------------
    add(Finding(
        id="shape", group="curve", label="shape of the curve", tier="read",
        value=None, unit=None, text=seg.shape,
        method="regimes chosen by penalised least squares; transitions by the step detector",
        rule=("a regime's model is the lowest degree (0, 1 or 2) whose residual, plus a "
              "penalty per coefficient, is smallest; the number of regimes "
              + ("was fixed by the reader" if seg.chosen_by == "requested"
                 else "minimises the same penalised cost across the curve")),
        evidence={"n_regimes": len(seg.segments), "chosen_by": seg.chosen_by,
                  "penalty_scale": settings.get("penalty_scale", 1.0)},
        anchor={"x0": float(x[0]), "x1": float(x[-1])},
    ))

    # --- transitions -------------------------------------------------------
    trans = [b for b in seg.boundaries if b.kind == "transition"]
    for k, b in enumerate(trans):
        g = f"transition.{k}"
        add(Finding(
            id=f"{g}.classified", group=g, label="called a transition", tier="read",
            value=None, unit=None, text="transition",
            method="locally detrended step statistic over a ladder of window widths",
            rule=(f"the detrended step stands {b.gap_sigma:.1f} times above the scatter "
                  f"(threshold {settings.get('event_threshold', 6.0):g}); it does not grow "
                  f"with the window width, and the straight lines either side do not "
                  f"cross within its span — so it is a discontinuity, not a bend"),
            evidence={"gap_sigma": b.gap_sigma, "sharpness": b.sharpness,
                      "samples": [int(b.index), int(b.index)], "width": b.width},
            anchor={"x0": b.x_lo, "x1": b.x_hi},
        ))
        add(Finding(
            id=f"{g}.position", group=g, label=f"transition position ({x_label})", tier="measured",
            value=b.x_lo, unit=x_unit,
            method="the onset: the last sample before a run of samples departs from the "
                   "regime on the left; the run length guards against a single noisy sample",
            evidence={"x_lo": b.x_lo, "x_hi": b.x_hi, "width": b.width},
            anchor={"x": b.x_lo, "y": b.y_before},
        ))
        add(Finding(
            id=f"{g}.onset", group=g, label=f"{y_label} at the onset", tier="measured",
            value=b.y_before, unit=y_unit,
            method="the regime on the left, evaluated at the onset — not the largest "
                   "sample, which is a property of the noise",
            evidence={"x": b.x_lo},
            anchor={"x": b.x_lo, "y": b.y_before},
        ))
        add(Finding(
            id=f"{g}.after", group=g, label=f"{y_label} once settled", tier="measured",
            value=b.y_after, unit=y_unit,
            method="the regime on the right, evaluated where the curve rejoins it",
            evidence={"x": b.x_hi},
            anchor={"x": b.x_hi, "y": b.y_after},
        ))
        add(Finding(
            id=f"{g}.step", group=g, label="size of the step", tier="measured",
            value=b.gap, unit=y_unit,
            method="disagreement between the two regimes at one place, the middle of the "
                   "transition — so the background's own rise across it is not counted",
            evidence={"x_mid": 0.5 * (b.x_lo + b.x_hi), "gap_sigma": b.gap_sigma},
            anchor={"x0": b.x_lo, "x1": b.x_hi, "y0": b.y_before, "y1": b.y_after},
        ))
        add(Finding(
            id=f"{g}.width", group=g, label="width of the transition", tier="measured",
            value=b.width, unit=x_unit,
            method="from the last sample on the old regime to the first run of samples on "
                   "the new one",
            evidence={"x_lo": b.x_lo, "x_hi": b.x_hi},
            anchor={"x0": b.x_lo, "x1": b.x_hi},
        ))

    # --- knees -------------------------------------------------------------
    knees = [b for b in seg.boundaries if b.kind in ("knee", "break", "step")]
    for k, b in enumerate(knees):
        g = f"knee.{k}"
        add(Finding(
            id=f"{g}.classified", group=g, label="called a change of slope", tier="read",
            value=None, unit=None, text=b.kind,
            method="a boundary placed by the partitioner inside a smooth stretch",
            rule=(f"the two fits meet here with a gap of {b.gap_sigma:.1f} times the scatter, "
                  f"below the {JUMP_SIGMA:g} that would make it a step; the slope changes by a "
                  f"factor of {b.slope_ratio:.2g}" if np.isfinite(b.slope_ratio) else
                  f"the two fits meet here with a gap of {b.gap_sigma:.1f} times the scatter"),
            evidence={"gap_sigma": b.gap_sigma, "slope_ratio": b.slope_ratio},
            anchor={"x": b.x, "y": b.y_before},
        ))
        add(Finding(
            id=f"{g}.position", group=g, label=f"bend position ({x_label})", tier="measured",
            value=b.x, unit=x_unit,
            method="the breakpoint between two regimes, on the partitioner's candidate grid",
            evidence={"sample": int(b.index)},
            anchor={"x": b.x, "y": b.y_before},
        ))
        add(Finding(
            id=f"{g}.slope_before", group=g, label="slope before the bend", tier="measured",
            value=b.slope_before, unit=slope_u,
            method="derivative of the regime on the left at the bend",
            anchor={"x": b.x, "y": b.y_before},
        ))
        add(Finding(
            id=f"{g}.slope_after", group=g, label="slope after the bend", tier="measured",
            value=b.slope_after, unit=slope_u,
            method="derivative of the regime on the right at the bend",
            anchor={"x": b.x, "y": b.y_after},
        ))

    # --- regimes -----------------------------------------------------------
    for i, s in enumerate(seg.segments):
        g = f"regime.{i}"
        add(Finding(
            id=f"{g}.model", group=g, label=f"regime {i + 1} model", tier="read",
            value=None, unit=None, text=s.model,
            method="least squares of degree 0, 1 and 2, compared by residual plus a "
                   "penalty per coefficient",
            rule=(f"{s.model} had the lowest penalised cost over {s.n} samples "
                  f"(residual scatter {s.sigma:.4g}, R² {s.r2:.3f})"),
            evidence={"n": s.n, "sigma": s.sigma, "r2": s.r2, "degree": s.degree},
            anchor={"x0": s.x0, "x1": s.x1},
        ))
        add(Finding(
            id=f"{g}.extent", group=g, label=f"regime {i + 1} runs from – to", tier="read",
            value=None, unit=x_unit, text=f"{s.x0:.4g} – {s.x1:.4g}",
            method="the boundaries the partitioner placed",
            rule="a boundary is where splitting the stretch lowers the penalised cost",
            evidence={"x0": s.x0, "x1": s.x1, "samples": [int(s.i0), int(s.i1)]},
            anchor={"x0": s.x0, "x1": s.x1},
        ))
        if s.degree == 0:
            add(Finding(
                id=f"{g}.level", group=g, label=f"regime {i + 1} level", tier="measured",
                value=m.get(f"{g}.level"), unit=y_unit,
                method="mean of the fitted constant over the stretch",
                anchor={"x0": s.x0, "x1": s.x1, "y": s.y_start},
            ))
        else:
            add(Finding(
                id=f"{g}.slope", group=g, label=f"regime {i + 1} slope", tier="measured",
                value=m.get(f"{g}.slope"), unit=slope_u,
                method="slope of a straight line fitted to the regime over its stretch — "
                       "for a curved regime, its average slope",
                anchor={"x0": s.x0, "x1": s.x1, "y0": s.y_start, "y1": s.y_end},
            ))
        if s.degree == 2:
            add(Finding(
                id=f"{g}.curvature", group=g, label=f"regime {i + 1} curvature", tier="measured",
                value=m.get(f"{g}.curvature"), unit=curv_u,
                method="second derivative of the fitted parabola",
                anchor={"x0": s.x0, "x1": s.x1},
            ))

    # --- the extremum, for the record --------------------------------------
    add(Finding(
        id="extremum.value", group="extremum", label=f"largest |{y_label}| in the samples",
        tier="measured", value=m["extremum.value"], unit=y_unit,
        method="the single sample farthest from zero — reported so it can be compared "
               "with the onset, not used as the transition value",
        evidence={"position": m["extremum.position"]},
        anchor={"x": m["extremum.position"], "y": m["extremum.value"]},
    ))
    add(Finding(
        id="extremum.position", group="extremum", label="where the extremum sits",
        tier="measured", value=m["extremum.position"], unit=x_unit,
        method="the x of the single sample farthest from zero",
        anchor={"x": m["extremum.position"], "y": m["extremum.value"]},
    ))
    return out


def ledger_summary(findings: list[Finding]) -> dict[str, Any]:
    tiers = {"measured": 0, "read": 0}
    with_ci = 0
    holds = 0
    swept = 0
    for f in findings:
        tiers[f.tier] = tiers.get(f.tier, 0) + 1
        if f.interval and f.interval.get("ci"):
            with_ci += 1
        if f.sensitivity:
            swept += 1
            if f.sensitivity.get("holds"):
                holds += 1
    return {"n": len(findings), "tiers": tiers, "with_interval": with_ci,
            "swept": swept, "holds": holds}
