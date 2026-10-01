# TiNi — force traces and the cubic landscape

**Version 1: the basic physics model.** One hundred molecular-dynamics
force–time traces of a titanium–nickel single crystal under constant-rate
tension (rate (v) of Maitra & Gururajan, Sci. Rep. 15:3008, 2025), loaded,
validated, averaged, and set beside the paper's cubic free-energy landscape
and its tilt under force.

**Version 2: the inverse problem.** Maximum-likelihood inference of
(U‡, ξ‡, Γ₀) from the critical-force sample (paper Eqs. 1, 6), with profile
likelihoods, bootstrap, synthetic recovery and a comparison with the paper's
Table 1. Open `#inverse` in the app, or `python -m tini inverse`. The
formulation, its limits and the results are in `docs/INVERSE_PROBLEM.md`.

**Version 3: the forward Smoluchowski model.** The paper's Fokker–Planck
equation integrated on the cubic landscape under the force ramp (finite
volume, Scharfetter–Gummel, backward Euler), with conservation and positivity
by construction, an exact first-passage benchmark, grid/time convergence, and
the critical-force density compared with Eq. (6) and the measured sample.
Open `#forward`, or `python -m tini forward`. See `docs/FORWARD_MODEL.md`.

**Version 4: Inflect — curve analysis for any table of numbers.** The part
that generalises. Drop in any table and it is read back to you: the dialect
and columns are inferred and explained, the transitions are found before the
regimes are fitted, and each stretch of the curve is described in words with
the numbers to hand. Across a set it reports where the transition sits, how
much it moves, and which files do not belong — on the 296-run sample set it
singles out exactly the three that came from a different population. The
value reported at a transition is the one at its onset, not the largest
reached, because the extremum is a property of the noise. Every number
carries a bootstrap interval and a sixteen-setting stability sweep; every
curve in a set can be seen at once, aligned on its own transition; and the
figure, the results and a methods paragraph leave the page as SVG, CSV and
text. Every finding is a first-class object with its tier — measured by a
stated procedure, read by a stated rule, or concluded by the researcher —
its method, evidence, interval and sensitivity; the columns to plot are
suggested but chosen by the reader, every one shown with its own shape; and a
self-verifying record
names the data and code by hash and recomputes the findings on demand. Open
`#inflect`. See `docs/INFLECT.md`.

## Layout

```
backend/tini/        FastAPI + NumPy pipeline
  config.py          settings, physical constants, paper Table 1 (with provenance)
  dataset.py         discovery, inspection, per-file loading
  validation.py      per-file and cross-file checks
  statistics.py      ensemble average, force rate, prototype peak picker
  potential.py       cubic U₀(ξ), F = −dU₀/dξ, tilt, f_c, barrier(f)
  charts.py          static PNG record of every chart (matplotlib)
  pipeline.py        six resumable stages with fingerprinted checkpoints
  progress.py        progress.json: stage, files, %, elapsed, ETA, last checkpoint
  io_atomic.py       write-to-temp + os.replace; previous copy kept as .prev
  api.py             the endpoints
backend/tini/inverse/ Version 2: analytic model, likelihood, fit, uncertainty, diagnostics, runs
backend/tini/forward/ Version 3: Smoluchowski solver, validation battery, runs
backend/tini/inflect/ Version 4: dialect-agnostic ingest, column intelligence, segmentation
backend/tests/       210 tests: physics identities, solver benchmarks, validation, resume, inference recovery, curve analysis
frontend/            Vite + React + TypeScript, Recharts, KaTeX (light theme only)
data/raw/Force_traces/   the 100 .dat files (3.9 MB)
artifacts/           everything the pipeline writes (see below)
config/settings.json processing limits and defaults
docs/PHYSICS.md      every equation, verified vs derived vs prototype
docs/STAGES.md       what each stage did, how it was tested, open issues
```

## Run it

Backend (Python 3.11+; NumPy, SciPy, pandas, matplotlib, FastAPI, uvicorn):

```bash
cd backend && pip install -r requirements.txt && python -m tini serve
```

Frontend (Node 18+):

```bash
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>. The status line under the masthead has
**Process dataset** (first run), **Resume** (after an interruption) and
**Redo all** (ignore checkpoints). The API is at <http://localhost:8000/docs>.

Without the UI:

```bash
cd backend && python -m tini run              # resume / run all stages
cd backend && python -m tini run --max-files 10   # small subset first
cd backend && python -m tini run --force      # redo everything
cd backend && python -m tini status
cd backend && python -m pytest
```

## Resource use

The whole dataset is 3.9 MB on disk and 2.4 MB as float64. A full run is
about 2 s on the laptop; everything is sequential and single-threaded on
purpose. No GPU, no training, no iterative optimisation anywhere in V1.
Charts are drawn from at most 12 overlaid traces and the API strides samples
on request.

## Checkpoints

`artifacts/` after a run:

| file | what |
| --- | --- |
| `inspection.json` | file list, sizes, header, grid, extension rate, memory estimate, paper-rate match |
| `validation.json` | per-file issues and stats; cross-file grid check |
| `traces/NNNNN.npy`, `completed_files.json` | per-file binary conversion — the resumable unit |
| `traces.npz`, `files.csv` | stacked (100, 997, 3) array and a per-file table |
| `average_curves.csv` | ⟨f⟩, s.d., s.e.m., median, percentiles, ⟨Lₓ⟩ vs t |
| `force_rate.json` | ḟ, its s.e., R², the window, and the window sensitivity |
| `events.json`, `events.csv` | prototype transition events per trace + histograms |
| `potential.json`, `potential_curves.csv` | both Table 1 landscapes: f_c, tilts, curves |
| `charts/*.png` | static record of every chart |
| `checkpoints.json` | per stage: fingerprint, outputs, duration, completed time |
| `progress.json` | live status for the UI; survives restarts |
| `config_used.json` | the settings of the last run |
| `errors.log` | append-only tracebacks |

A stage's fingerprint covers the settings it depends on, the dataset manifest
(names, sizes, mtimes) and the hash of its own source modules. Changing any
of them re-runs that stage and the ones after it that share the change;
nothing else is touched. Every file is written to a temp name and renamed
into place; the previous good copy is kept as `*.prev`.

## Design

The frontend follows the Keystone project's language: numbered panels, a
display serif for titles, one terracotta accent for measured quantities and a
slate for the model, and nothing on the page that is not a chart, a number
or an explanation. Light theme only.
