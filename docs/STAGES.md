# Stage record — Version 1

Dates are 2026-09-17. Times are for an ASUS Zephyrus G16 (RTX 4060, 16 GB),
CPU only.

## Stage 1 — inspect the paper, task and dataset

* Paper read in full; the equations used are transcribed in
  `docs/PHYSICS.md` with their status (verified / derived / prototype).
* Dataset: 100 files × 997 rows × 3 columns, one shared time grid
  (Δt = 0.64 ps), 3.9 MB. Units established from the data itself
  (`docs/PHYSICS.md §1`), not assumed. Rate identified as the paper's (v).
* Handwritten task: V1 = the cubic U(x) and the forward relation between
  potential and force; inverse problem and Smoluchowski deferred.
* Design reference: the Keystone project (Razorpay_Hackathon/frontend) —
  tokens, display face, panel structure reused.

## Stage 2 — architecture and UI design

* Backend: FastAPI, one package `tini` with one module per concern, six
  resumable stages, atomic writes, fingerprinted checkpoints.
* Frontend: Vite + React + TS, plain CSS with the Keystone token set,
  Recharts for charts, KaTeX for notation. One scrolling page, six numbered
  panels, one status line. No dark mode, no sidebar, no KPI tiles.
* Kept out on purpose: Tailwind (the Keystone look is plain CSS), Plotly
  (3 MB for no gain), any worker/parallel processing (4 MB of data).

## Stage 3 — data inspection pipeline

* `inspect` and `validate` stages; per-file and cross-file checks.
* Tested: `test_validate_trace_flags_bad_input` (shape, NaN, non-monotonic
  time), and on the real data: 100/100 valid, 0 warnings, shared grid.

## Stage 4 — force-trace processing

* `preprocess` (per-file .npy checkpoints, then stacked .npz) and
  `statistics` (ensemble average, ḟ with window sensitivity, prototype peak
  picker with three readings of the peak).
* Tested: known-slope recovery, moving-average invariants, synthetic peak
  detection, per-file resume after a simulated crash
  (`test_pipeline_resumes_preprocess_per_file`), partial invalidation when a
  setting changes, cancel keeps checkpoints.
* Real data: ḟ = 0.0943 ± 0.0013 pN/ps vs the paper's 0.0945.

## Stage 5 — cubic potential explorer

* `potential.py`: paper cubic, tilt, f_c, barrier(f); derived F(ξ) and
  stationary points. 11 tests over both Table 1 parameter sets check the
  boundary values, stationary points, f_c as the extreme landscape force,
  analytic vs numerical derivative, barrier(f) against direct evaluation, and
  the vanishing barrier at f_c.
* Frontend evaluates the same formulas locally for instant sliders and
  compares against the backend on every (debounced) change — shows "exact".

## Stage 6 — connect backend and frontend

* Vite proxies `/api` → `:8000`. Progress polled at 1 s while running, 5 s
  otherwise. A finished run bumps a version counter so every panel refetches.
* Verified in the browser: all six panels render with real data; the ḟ
  window, smoothing width and landscape sliders round-trip through the API;
  selecting a file in the table opens it in panel 02; the tilt time in panel
  05 uses the ḟ chosen in panel 03 (checked: 19.67 pN / 0.0943 = 208.5 ps).
* Responsive: no horizontal overflow at 375 px.

## Stage 7 — test, validate, optimise, document

* 28 backend tests pass (`python -m pytest`, ~3 s). `tsc -b` clean.
* Full pipeline on 100 files: 1.9 s cold, 0.8 s warm. Memory: trivial.
* Fixes made during review: percent not restored after a server restart;
  Δt printed with float noise; axis ticks with float noise; uppercase CSS
  mangling Greek letters in titles; auto ticks like −10.22 replaced by
  nice ticks; code changes now invalidate the stages they affect.

## Unresolved / for later

1. The "post-transition" behaviour (force to ≈ −45 pN, then reload) is
   recorded but not interpreted. It is outside the paper's model, which
   stops at the transition.
2. The peak picker is a placeholder. Before Version 2 fits p(f|ḟ), the
   picker's bias against smoothing width should be characterised (the
   sensitivity is visible in panel 04 but not tabulated).
3. `t‡` is picked on the smoothed trace, which shifts the peak slightly
   earlier than the sharp drop; the raw-peak time within the smoothing
   window is also stored (`t_raw_peak_ps`) for comparison.
4. Only rate (v) is present. The force spectrum ⟨f⟩(ḟ) needs the other
   rates, which are not in this dataset.
5. The display font (Against) is a personal-use demo licence, carried over
   from Keystone; swap it before any public use.
6. A `.prev` copy is kept for every file but there is no rotation beyond one
   generation.


---

# Version 2 — inverse problem (2026-09-17)

## Stage 1 — scientific analysis
* `docs/INVERSE_PROBLEM.md`: the paper's two inverse formulations, known and
  unknown quantities for a single-rate dataset, identifiability argument, a
  typo in the paper's ω(ψ), the residual mass ψ(f_c), numerical and
  validation plan. Three readings of "the inverse problem" laid out; the
  user confirmed (A): MLE of Eq. (6) with Eq. (8) evaluated forward only.

## Stage 2 — mathematical baseline
* `tini/inverse/model.py`: Eqs. 3–8 in log-space dimensionless form with
  overflow guards, exact moments by quadrature, analytic inverse-CDF
  sampler. 22 tests on identities and limiting cases.

## Stage 3 — data and inference pipeline
* `tini/inverse/{data,density,likelihood,fit,uncertainty,diagnostics,
  synthetic,runs}.py`; `/api/inverse/*`; `python -m tini inverse`.
* Every run is a directory of per-phase JSON files; resuming a run_id skips
  completed phases (tested); a new run never touches an old one (tested);
  cancel keeps completed phases (tested). Code hash and dataset fingerprint
  recorded per run.
* Full run with defaults: 19 s, single-threaded.

## Stage 4 — validation
* Synthetic recovery at three designs (§10 of the spec); paper-design
  recovery test; Eq. (8) vs Fig. 3 readings; LR tests of Table 1 against
  the fit; sensitivity to both V1 prototype choices.

## Stage 5–6 — frontend and integration
* `#inverse` route, seven sections A–G. Cards removed on both pages; sections
  separated by whitespace and a hairline; prose cut to one sentence per
  section; readouts limited to what a reader would quote. Measured
  quantities in terracotta, inferred in slate, the paper's in light slate.
* Verified in the browser: run started from the API completes in the
  background with live progress; latest run loads; previous runs reopen.

## Stage 7 — review
* 61 tests pass; `tsc -b` and `vite build` clean.
* Open items: the profile scan for U‡ is open above within a 3× span — a
  wider span would find the crossing but the answer (weakly identified) is
  the same; bootstrap draws that hit the U‡ bound are reported as a fraction
  rather than clipped; no multi-rate data exist to attempt the paper's
  method (B).


---

# Version 3 — forward Smoluchowski model (2026-09-17)

## Stage 1 — audit
* η's unit: the paper's Kramers prefactor fixes it as pN·ps/Å; V2's label
  `eta_per_ps` corrected to `eta_pN_ps_per_A` (value unchanged).
* The forward model has no parameter beyond V2's: η follows from Γ₀.
* Time scales: well relaxation 0.1 ps vs escape at 30–10 000 ps → implicit
  integration required (explicit limit 2×10⁻⁴ ps).
* The absorbing-boundary placement is the sensitive modelling choice
  (Kramers vs recrossing); both placements implemented.

## Stage 2 — specification
* `docs/FORWARD_MODEL.md` §1–6: equation, units, domain, boundaries, initial
  state, the five observables kept distinct, scheme choice, validation plan.

## Stage 3 — solver
* Finite volume, Scharfetter–Gummel fluxes, backward Euler (θ-scheme for
  study), exact discrete conservation (absorbed mass = boundary flux), M-matrix
  positivity, analytic exact-MFPT benchmark. 1.4 s per 250 × 20 000 run.

## Stage 4 — validation
* 18 tests: identities to machine precision, exact MFPT to 0.1 %, grid order
  2.00, time order 1.00, Kramers factor 0.965 and recrossing factor 1.79,
  bitwise reproducibility. One benchmark corrected (well-confined initial
  density).

## Stage 5 — backend integration
* `tini/forward/runs.py`: resumable per-phase runs, parameter sources
  (paper / V2 latest or specific run / custom), force rate, temperature,
  solver settings; `/api/forward/*`; `tini forward`. 7 tests. 86 total.
* Atomic writer now retries `os.replace` on Windows sharing violations
  (OneDrive/reader contention seen once during a run).

## Stage 6 — forward predictions
* Table in FORWARD_MODEL.md §8: PDE vs Eq. 6 within 0.2–0.3 pN for every
  parameter set; spectrum gap grows with rate; Table 1 parameters rejected by
  the sample (KS D 0.21–0.24), consistent with V2.

## Stage 7–8 — frontend and end-to-end
* `#forward` route, sections A–G; run from the UI, live progress, run
  history. Two frontend bugs found in the browser and fixed at the source
  (snapshot stride mismatch in `observables.json`; overlaid line x-key).
* API run verified: background thread, progress, latest pointer, 5 runs listed.

## Stage 9–10 — documentation, review
* 86 tests, `tsc -b`, `vite build` clean. Committed.
* Open: moving absorbing boundary; second-order time stepping with
  positivity (e.g. TR-BDF2); no multi-rate data to fit the PDE directly.

## Review pass (user feedback, 2026-09-17)
* Forward route could blank: a section that threw during render took the
  whole page with it. Each route is now wrapped in an error boundary, and
  the density panel guards against snapshot arrays of unequal length (the
  cause of the original crash was a stride mismatch in `observables.json`,
  fixed at the source and regenerated for stored runs).
* Symbols: `f` with a combining dot, angle brackets and subscript-x were set
  in the system text font, which lacks the glyphs. All such symbols are now
  typeset with KaTeX in text (segmented controls accept rich labels) and
  spelled out in SVG axis labels.
* The "prototype assumption" tag no longer sits mid-sentence.
* Section 1.6 records now point to Versions 2 and 3 instead of "not in v1".
* Sections numbered as chapters: 1.1–1.6, 2.1–2.7, 3.1–3.7.

## Final validation (2026-09-17)
* Independent re-derivations: Eq. (5) against direct quadrature of
  Eq. (3)+(4) (10⁻¹²); Eq. (4) shown to be exactly Kramers' rate for the
  tilted cubic (ratio 1.0000); the E₁ identity behind Eq. (8) checked by
  quadrature (11.4543 both ways).
* Independent method: overdamped Langevin simulation (40 000 particles)
  reproduces the PDE's mean to 0.006 pN (MC error 0.014) and its CDF to
  KS = 0.004; Eq. (6) differs by 0.2 pN / KS 0.035. Added as validation I
  and as a test (3 000 particles, 2 s).
* Stress: backward Euler positive and conservative at Δt = 1.7 ps;
  Crank–Nicolson goes negative — the default is confirmed.
* **Bug found and fixed**: runs stopped exactly at f_c and reported the
  density still in transit as "residual mass ψ(f_c)". At the fastest paper
  rate that was 7 % (50 % for Table 1 col. 2); continuing the ramp shows all
  of it is absorbed just past f_c. Runs now go to 1.3 f_c; the quantity is
  reported as "absorbed after f_c", distinct from "never absorbed" (≈ 0).
  All stored forward runs regenerated; docs and UI updated. At the measured
  rate nothing changes (10⁻¹³).
* Routes: all three pages scanned for NaN/undefined/Infinity/error
  boundaries — none; 5 forward runs open without error.
* Documentation numbers re-extracted from the stored runs and compared line
  by line with INVERSE_PROBLEM.md §10 and FORWARD_MODEL.md §7–8; three V3
  entries updated for the longer run.
* 88 tests, `tsc -b`, `vite build` clean.

---

# Version 4 — Inflect: curve analysis for any table (2026-09-20)

Generalises the project from one paper's dataset to anything a researcher
uploads. Specification and results: `docs/INFLECT.md`.

## Stage 1 — the analysis core
* `ingest` reads a numeric table without being told its dialect; `columns`
  ranks every column for the two roles that matter, using the trend relative
  to the noise rather than the raw range; `segment` finds transitions first
  and then partitions the smooth stretches exactly.
* Two findings drove the design. Thresholding the derivative cannot find a
  transition on a quadratic background, because the ramp's own steepening
  inflates the spread the threshold is measured against — a locally
  detrended step statistic separates them at ten sigma instead of one. And a
  sharp bend fakes a step in proportion to the measuring window, so the lines
  either side are tested for where they cross.
* 296 curves, 0 failures, transition at 2.298 +/- 0.062 % strain, 14 ms each.
* 41 tests.

## Stage 2 — reading and comparing
* `transitions` turns a segmentation into narration beats — sentence,
  numbers, annotations — so the interpretation is testable in Python.
* `ensemble` compares a set: robust statistics, grouping by structure, and
  outliers reported with the measurement that makes them one.
* Four measurement problems found and fixed: an onset decided by a single
  in-tolerance sample; a step measured between the two edges of a transition
  rather than at one place; an outlier feature taken from the first fitted
  segment; and a median absolute deviation of zero letting the outlier set
  its own scale.
* The outlier search returns exactly the three files an independent
  characterisation had flagged. 29 tests.

## Stage 3 — sessions and the API
* A session is an upload plus everything derived from it, stored atomically
  and resumable. Files are either uploaded or referenced in place, so the
  107 MB sample set opens without going through a form.
* Ten endpoints under `/api/inflect`. A curve is decimated for drawing by
  keeping each bucket's extremes, so spikes survive the thinning.
* Fixed: NaN is not valid JSON; non-finite values become null on the way out.
* 12 tests. 158 total.

## Stage 4 — the interface
* `#inflect`: a stage rather than a dashboard. One chart holds the middle and
  the analysis is drawn onto it beat by beat, with the camera easing to
  whatever is being discussed.
* No sidebar and no control panel. A setting appears as the number it already
  is, inside the sentence that explains it, and is dragged or typed over.
* Verified in the browser end to end: upload, the inferred-column findings,
  every beat and its annotations, the live number re-deriving the curve, the
  296-file comparison and its three outliers. No console errors.
* Open: named model fitting on a chosen range; per-curve uncertainty on a
  transition's position; export.

## Stage 5 — the instrument
* `uncertainty`: moving-block bootstrap of the whole reading, and a stability
  sweep across sixteen settings. Two wrong null models found and replaced —
  one flattened the transition (detection fell to 49 %), one froze it (the
  position's interval stopped responding to noise).
* `chorus`: every curve on one axis, optionally aligned on its own
  transition, with median and IQR; thumbnails for every file. Built in the
  same pass as the comparison.
* Five endpoints: uncertainty, stability, chorus, results.csv, methods.
* Page: a "How sure is this?" beat with the interval drawn as a band, the
  sweep as a strip and a per-regime slope table; the chorus as the hero of
  the set view; a thumbnail grid in place of the chips; export of the figure
  as SVG, the results as CSV, and an auto-written methods paragraph.
* Verified in the browser: interval 2.148 % [2.10, 2.24]; all sixteen sweep
  settings agree; chorus and alignment; 296 thumbnails with the three
  outliers marked; methods paragraph 1849 characters ending with the
  flagged files. No console errors. 21 tests added, 179 pass.

## Stage 6 — the ledger and the record
* `measure`: every quantity of a reading, keyed, matched to a reference so
  that the bootstrap and the sweep re-measure the same features. Both now
  cover every finding, not only the transition.
* `findings`: the ledger — tier (measured by a procedure / read by a rule),
  method or rule in words, evidence, chart anchor, interval, sensitivity.
  The noise carries no bootstrap interval (circular); a shifted estimator is
  flagged, not fudged.
* `record`: data hash, code hash, settings, seed, findings and the
  researcher's conclusions; `verify` recomputes and compares finding by
  finding. A tampered record is caught.
* Notes: the researcher's conclusions, stored by finding id, exported under
  their own heading, never among the findings.
* Page: "Every finding" table with tier badges and filters; each row opens
  into method, evidence, the replicates, the sweep, and a place to write a
  conclusion; "show" draws the finding on the chart; each beat states its
  tier and shows conclusions in the display voice; a delta line reports what
  a change of setting moved; "Reproduce it" builds, downloads and verifies
  the record.
* Verified in the browser: 18 findings, note kept and shown on the beat,
  record built and verified 18/18, delta "4 of 24 moved … the transition
  position did not move". No console errors. 22 tests added, 201 pass.
