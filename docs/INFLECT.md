# Version 4 — Inflect: curve analysis for any table of numbers

Status: **implemented** (2026-09-20). Analysis: `backend/tini/inflect/`;
API `/api/inflect/*`; page `#inflect`.

Versions 1–3 are one paper, one dataset, one model. This version is the part
that generalises: given any table a researcher uploads, work out what the
columns mean, where the curve changes regime, and what each regime is worth —
without the user writing code or choosing a parameter by hand.

## 1. What it does

1. **Reads the file** whatever its dialect. Comment prefix, header line,
   separator, column names and units are inferred from the bytes and reported
   with reasons.
2. **Decides what to plot.** Every column is measured and ranked for two jobs:
   the controlled variable and the response worth reading.
3. **Finds the transitions** — the places where the curve steps.
4. **Splits what is left into regimes** — constant, linear or quadratic —
   by exact dynamic programming.
5. **Says it in words**, as a sequence of narration beats with the numbers
   in them and the annotations to draw.
6. **Compares every file in the set**: where the transition sits, how much it
   moves, and whether anything does not belong.

## 2. Transitions before regimes

The order matters, and it is the main design decision.

A transition is an *event* with a finite width, not another polynomial
regime. Letting the partitioner discover the drop spends four or five pieces
describing the inside of one transition and reports nine regimes where a
reader can see three. So transitions are located first and held out of the
fitting.

## 3. Finding a step on a background that is already moving

The obvious detector — threshold the derivative — cannot work here. The
sample data's energy rises quadratically, so the curve's own rate of change
grows by a factor of forty before the transition, and the spread that a
threshold is measured against grows with it. Measured on the sample set, the
peak rate at the transition sits at 0.79–1.29 times the threshold: a coin
toss.

The detector used instead is a **locally detrended step statistic**. A
straight line is fitted to the `w` points before each sample and to the `w`
points after it; both are extrapolated to the sample and subtracted. On a
smooth curve — however steeply it rises or curves — the two agree and the
statistic vanishes to first order. Only a genuine discontinuity survives.

On the same curves this separates at about ten sigma instead of one.

Three refinements were needed to make it reliable:

* **Scale.** A wide window sees a step through less noise; a narrow one tells
  two nearby steps apart. A ladder of widths is tested and the best answer
  kept at each sample, so both are available.
* **Bends are not steps.** A sharp knee produces an apparent step in
  proportion to the measuring window. The lines either side are therefore
  tested for *where they cross*: at a bend they cross at the bend, at a
  discontinuity they cross far away or never. This is geometric and does not
  care how sharp the bend is.
* **The onset must not wander.** Walking out from the event until one sample
  agrees with the regime behind it puts the onset wherever the noise allows;
  a run of agreeing samples is required instead.

## 4. Regimes

Within each smooth stretch, the optimal partition into `k` pieces is found by
dynamic programming over candidate breakpoints. The residual sum of squares
of a degree 0, 1 or 2 fit on any interval comes from prefix moments in O(1),
so no greedy shortcut is needed. A budget of pieces is then shared between
the stretches optimally.

Cost is the residual sum of squares in units of the noise variance plus a
penalty per fitted coefficient, so a straight line beats a parabola unless
the parabola earns its coefficient. The reader may fix the number of regimes
instead, and that is the one number the interface exposes.

Two numerical points, both of which produced wrong answers before they were
fixed:

* The closed-form normal equations for a quadratic lose their significant
  digits on a short interval far from the origin, and return a residual of
  zero — a free segment, which the partitioner then collects. Central
  moments about each interval's own mean fix it.
* `y` is centred once globally: an energy column sits at −20080 and its
  square at 4×10⁸, so the residual would otherwise be a difference of two
  nearly equal large numbers.

## 5. Onset, not peak

The value reported at a transition is the response at its **onset** — the
last state before the curve leaves its regime — not the largest value
reached.

On a noisy trace the extremum is a property of the noise, and it moves with
however much smoothing was applied. On sample file 00005 the largest stress
reached is −0.725 GPa at 1.84 % strain, which is a single noisy sample; the
onset is −0.566 GPa at 2.15 %. A peak-based reading is wrong by 0.16 GPa and
by 0.3 % strain, and would move again with a different smoothing width.

The interface says so, on the chart, with both points circled.

## 6. Comparing a set

Each curve is reduced to the handful of numbers worth comparing: where the
transition is, the value and size of its step, the range covered, the initial
slope and the noise. Statistics are robust throughout — medians and median
absolute deviations — so that the outliers being looked for do not set the
scale they are judged against.

An outlier is reported with the measurement that makes it one, in words,
because a file that differs is either a discovery or a mistake and the reader
can only tell which if they are told what differs.

Two traps, both hit during validation:

* A feature taken from the *first fitted segment* manufactured fifteen
  outliers out of nothing, because whether the partitioner opened with a
  plateau is a property of the description, not of the curve.
* With near-identical repeats the median absolute deviation is zero, and
  falling back to the standard deviation lets the outlier set the scale it is
  judged against. A relative floor replaces it: nothing counts as unusual
  unless it is also worth mentioning.

## 7. Results on the sample set

296 compression runs of titanium nickel, 21 columns, 1000 rows each.

| | |
| --- | --- |
| columns chosen, unprompted | `-pxx` against `strain` |
| files read | 296 of 296, no failures |
| with a transition | 295 (stress), 279 (energy) |
| transition at | 2.20 ± 0.11 % strain |
| value at the transition | −0.559 ± 0.05 GPa |
| size of the step | 0.597 ± 0.09 GPa |
| dominant shape | run → transition → run (295 of 296) |
| time | 14 ms per curve; 5.8 s for the set |

**Outliers found: 00065, 00177, 00185** — "its range covered is 3.57 against
a typical 0.915 — 65 spreads above the rest". These are the three files an
independent characterisation had already flagged: each suffers an early
relaxation event that drops it into a deeper energy minimum, after which it
is three times stiffer and sustains five times the stress. The tool found
them without being told to look.

## 8. The interface

The chart is a stage, not a dashboard. The analysis plays over it as a
sequence of beats, each one drawing its own annotations — a shaded transition
zone, circled onset and settled points, a bracket for the step, a slope
triangle on a linear regime — and the camera eases to whatever the current
beat is about.

There is no sidebar and no panel of controls. A setting the reader can change
appears as the number it already is, inside the sentence that explains what
it does ("the curve has **4** regimes"), and is altered by dragging it or
typing over it. A control is easiest to understand when it sits inside its
own explanation.

## 9. How sure is it

Two questions a referee asks first, answered on the page and in the numbers.

**The error bar.** A moving-block bootstrap of the residuals: the fitted
regimes are kept, the residuals are resampled in blocks whose length is read
off their own autocorrelation, and the *entire* reading — detection and
partitioning — is re-run on each replicate. The interval on a transition's
position therefore includes the chance the detector would have placed it
elsewhere, not merely the scatter of a fit whose breakpoint was held fixed.
95 % percentile intervals are reported for the position, the value at the
onset, the step, and the slope of every regime; the position's interval is
drawn on the chart as a band.

Two mistakes were made and corrected in building it, and both are worth
recording because each produced an interval that looked perfectly plausible:

* Joining the two sides of a transition with a straight line, and resampling
  against that, hands every replicate a gentler transition than the one
  measured; the detector then misses it about half the time and the interval
  describes a curve nobody recorded.
* Pinning the transition to the observed samples fixes that but freezes the
  very stretch that decides where the onset is, and the interval on the
  position came back the same width whether the step stood three hundred
  times above the noise or four.

The transition's *shape* is now kept — lightly smoothed, so its noise is not
counted twice — and resampled noise is laid over the whole curve. Detection
survives at 99–100 %, and the interval widens with the noise as it must.

**The sweep.** The transition is re-derived under sixteen settings — six
detector thresholds, five levels of detail, five fixed regime counts — and
shown as a strip: a column of marks at one height is a result that belongs to
the data, not to the knobs. On the sample set: *all sixteen settings find the
transition and put it in the same place.*

## 10. Every curve at once

All curves on one pair of axes, drawn faintly enough that a hundred of them
read as a distribution, with the median and the interquartile ribbon over
them. The switch that matters shifts every curve so that its own transition
sits at zero: the scatter between repeats disappears and the shape that
every repeat shares emerges from underneath it. On the sample set the three
outliers are visible as separate traces leaving the frame — two populations,
seen rather than tabulated.

The vertical range holds 99.2 % of the drawn values rather than all of them,
so a handful of curves from another population cannot compress the rest into
a fifth of the frame; those curves are named on their own further down.

Every file is also shown as a thumbnail, so which ones differ is visible
before any of them is opened.

## 11. Taking it away

The figure downloads as SVG — the chart on screen is already plain SVG, so
what leaves is the thing being looked at, with the design tokens resolved
because a stylesheet does not travel with the file. Results download as CSV,
one row per file. A methods paragraph is written from the settings actually
used, including the sweep and the bootstrap, and ends with the set's
statistics and any files flagged as a different population.

## 12. The ledger: every finding, and who is responsible for it

Three kinds of statement appear on the page, and a reader is entitled to
know which is which.

* **measured** — a number from a stated procedure. "The onset is at 2.148 %."
  The software is responsible for the arithmetic and nothing more.
* **read** — a judgement the software made by a stated rule. "This is a
  transition." "This stretch is quadratic." The rule is given with the
  finding, so the reader can disagree with the rule rather than with the
  number.
* **concluded** — what the researcher makes of it. "This is the onset of the
  martensitic transformation." The software cannot know this, does not claim
  it, and keeps it apart from the two tiers above wherever the findings
  travel — a different voice on the page, its own heading in the record.

Every quantity is a first-class finding with its tier, its method or rule,
the evidence it rests on, where it sits on the chart, its bootstrap interval,
and how far it moves across the sixteen-setting sweep. Every beat names the
findings it draws on and carries the tier of what it asserts. The ledger is
the thing itself; the beats are a way of walking through it.

Two points of honesty in the intervals:

* The noise estimate carries no bootstrap interval. The bootstrap takes it
  as input and generates replicates from it; an interval from the same
  bootstrap would be circular — and, because resampled residuals are whiter
  than the originals, biased upward. The ledger says so.
* An estimator whose replicates all sit to one side of it (a transition's
  width on a clean step is at its resolution floor and can only widen) is
  reported as *shifted*, with the shift, not corrected away.

On the sample file: 18 findings, 12 measured and 6 read; 11 carry intervals;
11 of the 12 swept hold across every setting. Changing the regime count from
2 to 3 reports "4 of 24 findings moved, 1 no longer found, 7 new — the
transition position did not move."

## 13. The record

A result that can be handed to someone else and checked: a hash of the data
as read, a hash of the analysis code, every setting, the seed, the findings,
and the researcher's conclusions under their own heading. **Verify**
recomputes the ledger from the record's own settings and reports finding by
finding whether the numbers still agree, and whether the data and code on
disk are the ones the record names. A tampered record — one value moved, the
data hash altered — is caught and named.

On the sample file: reproduced, 18 of 18 findings to one part in 10⁹.

## 14. Limitations

* Regimes are polynomials up to degree two. An exponential or a power law is
  described as "curves over", not named. Fitting named models on a chosen
  range is not implemented.
* One response against one controlled variable at a time.
* The ensemble compares files that share a column layout; a set of files with
  different columns is read but not compared.
* Transitions closer together than the narrowest test window are reported as
  one. The window ladder starts at n/60 samples.
