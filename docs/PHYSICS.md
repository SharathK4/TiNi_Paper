# Physics notes — Version 1

Source: A. Maitra & M. P. Gururajan, *Extracting rates and activation free
energies of martensitic transitions using nanomechanical force statistics*,
Sci. Rep. **15**:3008 (2025), doi 10.1038/s41598-025-87324-w. A copy is in
`docs/Maitra_SciRep.pdf`.

Every equation used in the code is listed here with its status:

| status | meaning |
| --- | --- |
| **verified** | appears in the paper; transcribed and checked |
| **derived** | follows from a verified equation by elementary calculus; checked numerically in `backend/tests/test_potential.py` |
| **prototype** | a choice the paper leaves unspecified; configurable and reported with its sensitivity |

## 1. The data

`data/raw/Force_traces/property.T300.NNNNN.dat`, 100 files, 997 rows each.
Header (identical in every file):

```
# The original file is property.T300.00001.dat. Note Time is shifted to start from 0 to substract the equilibration time
#x in the header below correponds column number 1
#time(1)  | x(2) | Force per atom (3)
```

| column | symbol | unit | how the unit was established |
| --- | --- | --- | --- |
| 1 | t | ps | Δt = 0.64 on a 0 → 637.44 grid; the paper's MD timestep is 1 fs and the extension rate below only matches in ps |
| 2 | Lₓ | Å | 60.199 → 61.698; the paper's box is 60 Å along x; final strain 2.49 % matches the paper's 2.5 % cap |
| 3 | f | pN | force per atom, f = (Aₓₓ/n)σₓₓ (Methods); the mean-trace slope reproduces the paper's ḟ in pN/ps |

Rate identification (`inspection.json → paper_rate_match`): dLₓ/dt =
2.3515×10⁻³ Å/ps from a linear fit of column 2, which is the paper's rate
**(v)** 2.352×10⁻³ Å/ps to 0.02 %. The paper used 400 traces at this rate;
this dataset is 100 of them.

**Important.** Column 2 is the simulation-box length, not the reaction
coordinate ξ of the landscape model. They are never used interchangeably.
Lₓ is what was imposed; ξ is an internal order parameter that is not observed.

## 2. Force rate — verified equation, prototype window

Paper Eq. (2): ⟨f(t)⟩ ≈ ḟ t, with ḟ "computed from the gradient of the
averaged f–t trace in a small time interval near t = 0".

Implementation (`statistics.force_rate`): ordinary least squares of the
column-wise mean over traces on t ∈ [0, W]. **W is a prototype assumption**
(default 100 ps, `Settings.force_rate_window_ps`). Sensitivity from the data:

| W (ps) | ḟ (pN/ps) | R² |
| --- | --- | --- |
| 25 | 0.0762 | — |
| 50 | 0.0925 | — |
| 75 | 0.0959 | — |
| 100 | **0.0943 ± 0.0013** | 0.973 |
| 150 | 0.0918 | — |
| 200 | 0.0394 | — |

Flat between 50 and 150 ps; the collapse at 200 ps is the first transitions
bending the mean over. Paper's value for rate (v): 0.09452 pN/ps.

## 3. Transition events — verified definition, prototype picker

Paper (Methods): t‡ is "the transition time at which peak force occurs
(before the transition)", and f‡ = ḟ · t‡.

The paper does not say how the peak is located in a trace whose thermal
noise (s.d. ≈ 12 pN) exceeds the signal at any single sample. **The picker
is a prototype**: centred moving average of width `peak_smoothing_points`
(default 15 samples = 9.6 ps), then argmax. Three numbers are reported for
every trace so the choice is visible:

* `f_crit_from_rate_pN` = ḟ · t‡ — the paper's definition
* `f_peak_smoothed_pN` — what the picker saw
* `f_peak_raw_pN` — the largest single sample (noise-inflated; do not use)

Result with defaults: ⟨t‡⟩ = 196 ± 32 ps, ⟨f‡⟩ = 18.5 ± 3.0 pN (n = 100).

After the peak the force falls to about −45 pN and then reloads: the twinned
lattice is shorter than the box it is in, so it is briefly in compression.
The "trough" fields record this; nothing in V1 interprets it further.

## 4. The cubic landscape — verified

Paper, Results ("For analytical modelling, consider a free energy function"):

    U₀(ξ) = U‡/2 + (3U‡/2ξ‡)(ξ − ξ‡/2) − (2U‡/ξ‡³)(ξ − ξ‡/2)³

with U₀(0) = 0 (austenite minimum) and U₀(ξ‡) = U‡ (transition state).

Tilt under the imposed force (paper, text around Eq. 2):

    U(ξ, t) ≈ U₀(ξ) − ⟨f(t)⟩ ξ

Maximal turnover force (paper, below Eq. 4):    f_c ≡ 3U‡/(2ξ‡)

Barrier change under force (paper, Eq. 4):
ΔU‡(f) = U‡(1 − (1 − f/f_c)^{3/2}), so the remaining barrier is
U‡(f) = U‡(1 − f/f_c)^{3/2}.

Table 1 parameters (both columns are carried in `config.PAPER_PARAMETER_SETS`):

| | fit of p(f\|ḟ), Eq. 6 (MLE) | fit of ⟨f⟩(ḟ), Eq. 8 (LSQ) |
| --- | --- | --- |
| U‡ (pN·Å) | 638.14 (9.62) | 659 (38.6) |
| ξ‡ (Å) | 29.2 (0.45) | 36.1 (3.6) |
| Γ₀ (1/ps) | 3.96×10⁻⁷ | 6.12×10⁻⁸ |
| → f_c (pN) | 32.78 | 27.38 |

k_BT = 1.380649×10⁻²³ J/K × 300 K = 41.42 pN·Å, so U‡ ≈ 15.4 k_BT.

## 5. Landscape force — derived

    F(ξ) = −dU₀/dξ = −3U‡/(2ξ‡) + (6U‡/ξ‡³)(ξ − ξ‡/2)²

F(0) = F(ξ‡) = 0; the most negative value −f_c sits at the inflection ξ‡/2,
which is why the barrier vanishes exactly when the applied force reaches f_c.

Stationary points of the tilted profile, from F(ξ) + f = 0:

    ξ_min, ξ_TS = ξ‡/2 ∓ (ξ‡/2)√(1 − f/f_c),   0 ≤ f ≤ f_c

Substituting back reproduces U‡(f) = U‡(1 − f/f_c)^{3/2} exactly, which is the
numerical check in `test_barrier_under_force_matches_direct_evaluation`.

## 6. What V1 deliberately does not do

* No fit of U‡, ξ‡, Γ₀ to this dataset (the inverse problem, Eq. 6 / Eq. 8
  and the MLE/regression of the paper) — Version 2.
* No Smoluchowski / Kramers dynamics, survival probability ψ(f), or the
  PDF p(f|ḟ) — Version 3.
* No claim that the data validate the paper's parameters. The only comparisons
  made are (a) ḟ against the paper's tabulated ḟ for rate (v), and (b) the
  observation that every measured f‡ lies below f_c from Table 1.

A hand cross-check that is *not* implemented: with the Table 1 (MLE) values
and ḟ = 0.0943, the paper's Eq. (8) gives ⟨f‡⟩ ≈ 19.5 pN; the prototype
picker here gives 18.5 (ḟ·t‡) to 20.4 (smoothed peak). Consistent, but this
is Version 2 material and not shown in the app.
