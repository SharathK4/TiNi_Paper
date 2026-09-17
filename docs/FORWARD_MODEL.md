# Version 3 — forward Smoluchowski model

Status: specification (Stage 2), then implementation notes and validation
results appended as the stages complete.

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
2. A density absorbed *at* a barrier top escapes at **twice** the Kramers
   rate: Kramers' flux-over-barrier counts only the half that does not
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
force, so P(ξ,0) = e^{−βU₀(ξ)} / ∫ e^{−βU₀} dξ over the domain. (With P = 0
imposed at ξ_A this is not exactly stationary, but at U‡ ≈ 15 k_BT the
difference is < 10⁻⁶ in mass.)

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

The run stops at t_end = f_c/ḟ (barrier vanished). Mass still in the domain
at t_end is reported as ψ(f_c), exactly as V2 does for Eq. (6).

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

## 7. Results

(appended after Stage 4 and Stage 6)
