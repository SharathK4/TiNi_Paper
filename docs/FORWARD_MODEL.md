# Version 3 — forward Smoluchowski model

Status: **implemented and validated** (Stages 2–8, 2026-09-17). §7 holds the
validation results, §8 the forward predictions. Solver:
`backend/tini/forward/solver.py`; validation: `validation.py`; runs:
`runs.py`; API `/api/forward/*`; page `#forward`.

Source: Maitra & Gururajan, Sci. Rep. 15:3008 (2025), *Results — Quantitative
description of phase transition linked force distribution*: the
Einstein–Smoluchowski equation, the survival probability Ψ(t), the absorbing
boundary at the barrier, and the reduction to Eq. (3) under steady flux.
Refs. 40–42 (Risken; Hänggi–Talkner–Borkovec; Kramers) for the standard
results used in validation.

---

## 1. The equation

The paper (page 6) writes the state probability density P(ξ, t) as governed by

    ∂ₜP(ξ,t) = −∂ξ [ −(k_BT/η) ∂ξP(ξ,t) + (−∂ξU(ξ,t)/η) P(ξ,t) ]        (S)

"for the case of overdamped thermal activation". Rearranged, with the flux
J made explicit:

    ∂ₜP = −∂ξ J,      J(ξ,t) = −D [ ∂ξP + β (∂ξU) P ],      D = k_BT/η,  β = 1/k_BT

This is the overdamped Fokker–Planck (Smoluchowski) equation with the
Einstein relation between diffusion coefficient and friction. **Verified**:
it is the paper's equation, term for term.

| symbol | meaning | unit | status |
| --- | --- | --- | --- |
| ξ | order parameter (landscape coordinate); **not** the box length Lₓ of V1 | Å | paper |
| t | time | ps | paper |
| P(ξ,t) | probability density of the order parameter | 1/Å | paper |
| U(ξ,t) = U₀(ξ) − f(t)·ξ | tilted landscape (V1, paper Eq. 2 context) | pN·Å | paper |
| U₀(ξ) | cubic landscape of V1, parameters U‡, ξ‡ | pN·Å | paper |
| f(t) = ḟ t | applied force ramp (paper Eq. 2) | pN | paper |
| η | friction coefficient (force per velocity) | pN·ps/Å | see §2 |
| D = k_BT/η | diffusion coefficient | Å²/ps | Einstein relation, paper |
| k_BT | 41.42 pN·Å at 300 K (V1) | pN·Å | paper |

All quantities are kept in these physical units; the only dimensionless
combination used inside the solver is u(ξ,t) = βU(ξ,t).

## 2. The friction coefficient — where η comes from

The paper gives the zero-force rate prefactor

    Γ₀ = [1/(2πη)] · (6U‡/ξ‡²) · e^{−U‡/k_BT}

which is Kramers' high-friction escape rate, Γ = √(κ_min κ_max)/(2πη) e^{−βU‡},
for the cubic (|κ_min| = |κ_max| = 6U‡/ξ‡²). Dimensional analysis of this
formula fixes η as a friction coefficient, pN·ps/Å; the paper's phrase
"η is a friction constant in units of inverse time" is inconsistent with its
own formula and is not used. (V2 reported the derived value under the key
`eta_per_ps`; the value was right, the label wrong. It is now
`eta_pN_ps_per_A`.)

Consequently the forward model has **no free parameter beyond V2's**:

    η = (6U‡/ξ‡²) e^{−U‡/k_BT} / (2π Γ₀),      D = k_BT/η

For Table 1 column 1: η = 0.368 pN·ps/Å, D = 113 Å²/ps, well relaxation time
η/κ = 0.08 ps, diffusion time ξ‡²/D = 7.6 ps. For the V2 MLE: η = 0.695,
D = 60 Å²/ps.

This mapping is itself the paper's model (Kramers), so a forward run at V2's
parameters is a test of the paper's *chain of approximations*, not an
independent measurement of η.

## 3. Domain, boundary conditions, initial condition

**Domain.** ξ ∈ [ξ_L, ξ_A]. The cubic rises steeply left of the well:
U₀(−ξ‡/2) = U‡, U₀(−ξ‡) = 5U‡, and the tilt −fξ only raises it further for
ξ < 0. Default ξ_L = −0.6 ξ‡ (U₀ ≈ 1.5 U‡ ≈ 23 k_BT above the well); the
Boltzmann weight there is < 10⁻⁹ and the sensitivity to ξ_L is checked.

**Left boundary, reflecting**: J(ξ_L, t) = 0.

**Right boundary, absorbing**: P(ξ_A, t) = 0. The paper: "the boundary at
barrier position is absorbing", i.e. ξ_A = ξ‡ (the zero-force transition
state), and Ψ(t) = ∫_{−∞}^{ξ‡} P dξ.

*Two consequences, both handled explicitly:*

1. Under force the transition state moves left, ξ_TS(f) = ξ‡/2 + (ξ‡/2)√(1−f/f_c)
   (V1), so for f > 0 the region ξ_TS(f) < ξ < ξ‡ is downhill. Density that
   crosses the moving barrier top slides to ξ_A and is absorbed a short
   transit later; it cannot return uphill against a strong drift, so the
   survival probability is only delayed, not changed in mass.
2. A density absorbed *at* a barrier top escapes at up to **twice** the Kramers
   rate (1.8× measured at a 9 k_BT barrier, §7): Kramers' flux-over-barrier counts only the half that does not
   recross (Hänggi et al. 1990 §IV). The paper combines the barrier-top
   absorbing boundary with the Kramers rate; the two are inconsistent by a
   factor 2 in Γ, i.e. ln 2 ≈ 0.69 in ln Γ₀. V2's Γ₀ interval spans two
   decades, so the fit cannot see this; the forward model can, and shows
   it by offering both boundary placements:

   * `barrier_top`: ξ_A = ξ‡ (the paper's statement, literally)
   * `beyond_barrier` (default for comparison with Eq. 6): ξ_A = ξ‡ + m,
     with m chosen so that U₀(ξ_A) is at least 4 k_BT below the barrier top,
     where recrossing is negligible and the escape rate is Kramers'.

**Initial condition.** The paper's traces start from equilibrium at zero
force, so P(ξ,0) ∝ e^{−βU₀(ξ)} on the well, ξ ≤ ξ_TS(0) = ξ‡, and zero
beyond it (the downhill region past the barrier that a `beyond_barrier`
domain also contains is not part of the austenite state). With P = 0
imposed at ξ_A this is not exactly stationary, but at U‡ ≈ 15 k_BT the
difference is < 10⁻⁶ in mass.

## 4. Observables — five different things, kept apart

| quantity | definition | how obtained | unit |
| --- | --- | --- | --- |
| P(ξ,t) | probability density of the order parameter | the PDE solution | 1/Å |
| Ψ(t) | survival probability, ∫ P(ξ,t) dξ over the domain | quadrature of the solution | — |
| ρ(t) = −dΨ/dt | first-passage-time density = flux into the absorbing boundary | the boundary flux J(ξ_A,t), computed exactly by the scheme | 1/ps |
| p(f \| ḟ) | critical-force density | ρ(t)/ḟ at f = ḟt — the paper's mapping t‡ → f‡ = ḟt‡ (Methods, Eq. 2) | 1/pN |
| ⟨f‡⟩, s.d. | moments of p(f) plus the mass ψ(f_c) that has not escaped when the barrier vanishes, placed at f_c as in V2 | quadrature | pN |

The experimentally measured statistic is the sample {f‡ᵢ = ḟ t‡ᵢ} of V1 and
its histogram; the quantity that can be compared with it is p(f | ḟ). V2's
analytic p(f | ḟ) (Eq. 6) is the *Kramers/steady-flux approximation* to the
PDE's p(f | ḟ); the forward model computes the latter without that
approximation.

The run continues to t_end = 1.3 f_c/ḟ, past the point where the barrier
vanishes. Density that crosses after f_c — transit through the vanished
barrier, which takes a finite time — is absorbed at f > f_c and reported as
``mass_after_fc``; Eq. (6) places the same mass as a point at f_c. Mass still
in the domain at t_end (``residual_mass``) is ≈ 0 for every case examined.
(An earlier version stopped at f_c and reported the transit mass as
"residual"; at the fastest paper rate that was 7 % — see §7.)

## 5. Numerical scheme — and why

The problem is a 1-D drift–diffusion equation with (a) a stiff separation
of time scales — well relaxation 0.1 ps against escape at 30–10 000 ps — and
(b) a drift that is strong relative to diffusion near the barrier
(|U′|/η up to ~90 Å/ps against D ≈ 100 Å²/ps). Requirements: exact
conservation of P plus absorbed mass, non-negativity without clipping, the
discrete Boltzmann distribution as an exact steady state, and unconditional
stability so that Δt is set by accuracy, not by the 10⁻⁴ ps explicit limit.

**Finite volume on a uniform grid with Scharfetter–Gummel (exponentially
fitted) fluxes, implicit (backward) Euler in time.**

* Cells i = 1…N of width h, centres ξᵢ; P̄ᵢ is the cell average.
* Interface flux with u = βU evaluated at cell centres and Δuᵢ = uᵢ₊₁ − uᵢ:

      J_{i+½} = (D/h) [ B(Δuᵢ) P̄ᵢ − B(−Δuᵢ) P̄ᵢ₊₁ ],      B(x) = x/(eˣ − 1)

  This is the exact flux for a piecewise-linear u between centres. It
  reduces to central differencing for Δu → 0, is upwinded for |Δu| ≫ 1, and
  vanishes identically for P̄ ∝ e^{−u} — so the discrete equilibrium is the
  Boltzmann distribution, exactly, at any h.
* Reflecting: J_{½} = 0. Absorbing: a ghost value P = 0 at ξ_A gives
  J_{N+½} = (D/h′) B(Δu) P̄_N with h′ the distance from ξ_N to ξ_A; this
  flux is the first-passage-time density ρ(t).
* Time: (I − Δt A(tⁿ⁺¹)) Pⁿ⁺¹ = Pⁿ with A tridiagonal, A having positive
  off-diagonals and non-positive diagonal (an M-matrix). Backward Euler with
  an M-matrix preserves non-negativity for any Δt, is L-stable (the 0.1 ps
  well relaxation is damped, not resolved), and conserves mass exactly:
  Σ P̄ⁿ⁺¹ h + Δt J_{N+½}ⁿ⁺¹ = Σ P̄ⁿ h.
* Order: second in h (SG), first in Δt. Accuracy in time is controlled by
  Δt and demonstrated by refinement (§7), not assumed. A Crank–Nicolson
  option exists for the convergence study only; it is not the default
  because it does not guarantee positivity.
* Cost: N ≈ 250, N_t ≈ 2·10⁴ tridiagonal solves ≈ 1 s per run on CPU.

Rejected: explicit schemes (10⁷ steps at slow rates); method of lines with
a generic stiff ODE solver (works, but hides the conservation and
positivity properties inside an adaptive integrator).

## 6. Validation plan (executed in Stage 4, results in §7)

| test | statement | benchmark |
| --- | --- | --- |
| A conservation | Ψ(t) + ∫₀ᵗ ρ dt′ = 1 | exact identity of the scheme; report the maximum deviation |
| B positivity | min P̄ ≥ 0 for all t | exact property; report the minimum |
| C1 reflecting | with both ends reflecting and f = 0, P stays Boltzmann | ‖P(t) − P(0)‖₁ → machine precision |
| C2 absorbing | at fixed f, the mean first-passage time to ξ_A | the **exact** 1-D formula τ = (1/D) ∫_{ξ₀}^{ξ_A} e^{u(y)} ∫_{ξ_L}^{y} e^{−u(z)} dz dy (Risken §5.2.2), averaged over the initial density |
| C3 free diffusion | U = 0, reflecting far walls: variance grows as 2Dt | analytic |
| D grid | ⟨f‡⟩ and ‖p‖₁-difference at h, h/2, h/4 | observed order → 2 |
| E time | the same at Δt, Δt/2, Δt/4 | observed order → 1; Richardson estimate of the error |
| F Kramers | fixed-force escape rate vs Γ(f) of Eq. (4), for both boundary placements | the factor 2 of §3, and the O(k_BT/U‡) corrections to Kramers |
| G reproducibility | identical arrays on rerun | bitwise |
| H physics | p ≥ 0; ∫p df + ψ(f_c) = 1; ⟨f‡⟩ increases with ḟ; agreement with Eq. 6/8 within the stated tolerance | V2 forward model |

There is no closed-form solution for the ramped problem itself; the evidence
for it is C2 (exact at fixed force), D–E (convergence) and F–H (limits).

## 7. Validation results (Table 1 col. 1 parameters unless stated)

| test | result | metric |
| --- | --- | --- |
| A conservation | Ψ + absorbed = 1 at every step | max deviation 2×10⁻¹³ (250 cells × 20 000 steps) |
| B positivity | no negative cell, no clipping | min P̄ = 7×10⁻³⁸ /Å |
| C1 equilibrium | Boltzmann density stationary, both walls reflecting, 50 ps | L1 drift 6×10⁻¹⁵ |
| C2 exact MFPT | fixed f = 0.6 f_c (3.9 k_BT barrier), from the well-confined Boltzmann start | 43.63 vs 43.59 ps exact, 0.10 %; also 0.09 % at the ξ‡ placement, 0.10 % at 0.5 f_c (5.5 k_BT), 0.12 % at 0.7 f_c (2.5 k_BT) |
| C3 diffusion | U ≡ 0, variance = 2Dt | relative error 1.4×10⁻⁸ |
| D grid | h → h/2 → h/4 → h/8, ⟨f‡⟩ change 0.0028, 0.0007, 0.0002 pN | observed order 2.00 |
| E time | Δt → Δt/2 → Δt/4 → Δt/8, ⟨f‡⟩ change −0.0017, −0.0009, −0.0004 pN | observed order 1.00; Richardson error at 20 000 steps ≤ 6×10⁻⁴ pN |
| F Kramers, f = 0.3 f_c (9.0 k_BT) | Γ_PDE/Γ_Eq.4 | absorbing beyond: **0.965**; at ξ‡: 1.21; on the transition state: **1.79** (recrossing, → 2) |
| F Kramers, f = 0.6 f_c (3.9 k_BT) | Γ_PDE/Γ_Eq.4 | beyond 0.92; at ξ‡ 0.97; on the TS 1.65 |
| G reproducibility | two runs | bitwise identical |
| H physics | p ≥ 0; ∫p df + residual = 1 − 10⁻¹³; ⟨f‡⟩ increases with ḟ | true |
| **I Langevin** | overdamped Euler–Maruyama, 40 000 particles, Δt = 0.004 ps, same η, D, ramp, boundaries, initial density; no shared code | MC ⟨f‡⟩ 19.890 ± 0.014 vs PDE 19.884 (Eq. 6: 19.689); KS(MC, PDE) = 0.004 below the 0.007 noise level; KS(MC, Eq. 6) = 0.035 |
| stress | backward Euler at Δt = 1.7 ps (20× the well relaxation time) | still positive, conservative, ⟨f‡⟩ moves 0.04 pN; Crank–Nicolson at Δt = 0.7 ps gives min P = −5×10⁻⁸ |
| identities | Eq. (5) closed form vs direct quadrature of Eq. (3)+(4); Eq. (4) vs Kramers with the tilted cubic's actual barrier and curvatures | 10⁻¹²; ratio 1.0000 at f/f_c = 0.2, 0.4, 0.6 (Eq. 4 *is* Kramers for this landscape) |

Reading of F: the paper's rate law Eq. (4) is what the PDE gives when the
density is absorbed *beyond* the barrier; absorbing on the transition state
itself escapes faster by the recrossing factor. The paper's literal
"absorbing at the barrier position" (ξ‡) is nevertheless within 3 % of
Kramers at the forces where transitions actually occur, because by then the
transition state has moved inside ξ‡ and the boundary is already downhill.

A benchmark note: the first version of the C2 test initialised with the
Boltzmann density over the whole domain, which for a domain extending past
the barrier puts most of the mass beyond it; the test was wrong, not the
solver. The physical initial state is the well, and both the PDE and the
exact formula now use the well-confined density.

## 8. Forward predictions at ḟ = 0.0943 pN/ps (rate (v))

| parameters | boundary | PDE ⟨f‡⟩, s.d. | Eq. 6 ⟨f‡⟩, s.d. | L1(p_PDE, p_Eq6) | max ΔCDF | sample ln L: PDE / Eq. 6 | KS D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2 MLE (539, 25.6, 2.5×10⁻⁶) | beyond | 18.76, 3.15 | 18.52, 3.10 | 0.074 | 0.037 | −251.3 / −250.9 | 0.095 (descriptive: fitted to this sample) |
| V2 MLE | at ξ‡ | 18.46, 3.24 | 18.52, 3.10 | 0.040 | 0.012 | −251.0 / −250.9 | 0.063 (descriptive) |
| Table 1 col. 1 | beyond | 19.89, 2.82 | 19.69, 2.78 | 0.067 | 0.033 | −261.9 / −259.1 | 0.212 (valid; 5 % critical 0.136) |
| Table 1 col. 2 | beyond | 20.18, 2.69 | 19.92, 2.72 | 0.104 | 0.050 | −267.9 / −263.1 | 0.241 (valid) |

Force spectrum (V2 MLE, beyond): PDE − Eq. 6 exact mean = +0.09, +0.11,
+0.13, +0.17, +0.24, +0.35, +0.51 pN at rates (i)–(vii). The fraction
absorbed *after* f_c is < 10⁻¹² up to rate (v), 10⁻⁴ at (vi) and 7 % at
(vii); for Table 1 column 2 (lowest f_c = 27.4 pN) it is 50 % at (vii) and
the PDE mean exceeds Eq. (6)'s by 2.0 pN there. This is where the paper's
closed form stops being a good approximation: when loading outruns escape,
Eq. (6) collapses the late transitions onto f_c.

**What this establishes.**

1. The paper's closed form, Eq. (6), is the Kramers/steady-flux
   approximation to its own Smoluchowski equation, and that approximation is
   good to 0.2–0.3 pN in the mean (≈ 1 %), ≤ 5 % in the CDF, for every
   parameter set examined. The discrepancy grows with loading rate — the
   transit past the barrier that a steady-flux picture neglects — and with
   it the residual mass at f_c.
2. With Version 2's parameters the PDE describes the measured sample as well
   as Eq. (6) does (Δ ln L = 0.4 at n = 100). This is expected — those
   parameters were fitted to this sample through Eq. (6) — and it is a test
   of the approximation chain, not of the physics.
3. With the paper's Table 1 parameters the PDE, like Eq. (6), is rejected by
   this sample (KS D = 0.21–0.24 against a critical 0.136). As in Version 2,
   the gap is the size of the Version 1 peak-picker bias; the forward model
   does not change that conclusion and cannot be used to claim agreement.
4. Nothing here identifies η independently: it is set by Γ₀ through the
   paper's Kramers formula, so the forward model has no free parameter of
   its own. Different (U‡, ξ‡, Γ₀) along Version 2's ridge give nearly the
   same p(f | ḟ) here too — the non-identifiability of V2 carries over.
5. The PDE is independently confirmed by a Langevin simulation (test I):
   the 0.2 pN gap between the PDE and Eq. (6) is physics — the steady-flux
   approximation — not discretisation.

## 9. Limitations

* One reaction coordinate, overdamped, with the paper's cubic; no
  inertial term, no coordinate-dependent friction, no coupling to the box.
* The absorbing boundary is fixed in ξ; a moving boundary on ξ_TS(f) is not
  implemented (its fixed-force effect is quantified in F).
* Backward Euler is first order in time; the Richardson estimate (4×10⁻⁴ pN
  at the default 20 000 steps) is reported with every run, and the
  Crank–Nicolson option exists for study only because it does not preserve
  positivity.
* Temperature other than 300 K is supported by holding η fixed and
  rescaling Γ₀ by Kramers — an assumption beyond the paper, labelled as such
  in the run record.
