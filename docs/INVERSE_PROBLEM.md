# Version 2 — inverse problem specification

Status: **formulation for review** (Stage 1). Nothing in this document is
implemented yet except the forward model of §3, which every interpretation
below needs.

Source: Maitra & Gururajan, Sci. Rep. 15:3008 (2025), sections *Methods —
Estimation of parameterised landscape* (Eq. 1), *Results — Quantitative
description of phase transition linked force distribution* (Eqs. 2–6), and
*Nonlinear relationship between critical force and imposed loading rate*
(Eqs. 7–8), Table 1, Figs. 2–3.

---

## 1. What the paper calls the inverse problem

The paper has one landscape model with three parameters,

    w = { U‡, ξ‡, Γ0 }

| symbol | meaning | unit |
| --- | --- | --- |
| U‡ | free-energy barrier at zero force (TS minus austenite minimum) | pN·Å |
| ξ‡ | distance of the transition state from the austenite minimum | Å |
| Γ0 | intrinsic (zero-force) transition-rate prefactor | 1/ps |

and **two independent ways of estimating w from force statistics**:

**(A) Maximum likelihood on the critical-force distribution.** The data are
the critical forces {f‡ᵢⱼ}, i = 1…S traces at each force rate ḟⱼ, j = 1…Nᵣ.
The model is the PDF p(f | ḟ; w), Eq. (6). The likelihood is Eq. (1),

    L(w | {fᵢⱼ}) = ∏ⱼ ∏ᵢ p(fᵢⱼ | ḟⱼ ; w)

maximised with Nelder–Mead. In the paper Nᵣ = 4 rates × S = 400 = 1600
events, giving Table 1 column 1: U‡ = 638.14 (9.62), ξ‡ = 29.2 (0.45),
Γ0 = 3.96×10⁻⁷ (6.75×10⁻⁸).

**(B) Least squares on the force spectrum.** The data are the sample means
⟨f‡⟩(ḟⱼ) at the seven rates; the model is Eq. (8). Conjugate-gradient
least squares gives Table 1 column 2: U‡ = 659 (38.6), ξ‡ = 36.1 (3.6),
Γ0 = 6.12×10⁻⁸ (6×10⁻⁸).

Both are "given the observed force statistics, find the cubic landscape
(U‡, ξ‡) and the rate prefactor Γ0" — the arrow in the handwritten note,
*given ⟨F⟩ → U(x), cubic polynomial*.

## 2. Known and unknown quantities for **this** dataset

| quantity | status | value / source |
| --- | --- | --- |
| k_BT | known | 41.42 pN·Å (T = 300 K, Methods) |
| ḟ | known, measured in V1 | 0.0943 ± 0.0013 pN/ps (100 ps window — a V1 prototype assumption; sensitivity 0.092–0.096 over 50–150 ps) |
| {f‡ᵢ} | measured in V1 | S = 100 values, f‡ = ḟ·t‡, with t‡ from the V1 prototype peak picker |
| Nᵣ | fixed by the data | **1** (rate (v) only) |
| U‡, ξ‡, Γ0 | unknown | to be inferred |
| f_c = 3U‡/2ξ‡, βU‡, μ0 = Γ0 k_BT/(ḟ ξ‡), η | derived from w | reported, not fitted |

## 3. The forward model (verified equations, in the form to be coded)

With β = 1/k_BT, f_c = 3U‡/(2ξ‡), and the fractional barrier reduction

    Δ(f) ≡ 1 − (1 − f/f_c)^{3/2},    0 ≤ f ≤ f_c        (from Eq. 4)

| eq. | statement | status |
| --- | --- | --- |
| (3) | −∂_f Ψ = Γ(f) Ψ / ḟ | verified |
| (4) | Γ(f) = Γ0 (1 − f/f_c)^{1/2} · exp[βU‡ Δ(f)] | verified |
| (5) | ψ(f) = exp{ μ0 − μ(f)(1 − f/f_c)^{−1/2} },  μ(f) = Γ(f) k_BT/(ḟ ξ‡) | verified |
| (5′) | ψ(f) = exp{ μ0 [1 − exp(βU‡ Δ(f))] } | derived: identical to (5), since μ(f)(1−f/f_c)^{−1/2} = μ0 e^{βU‡Δ}; also obtained by integrating (3)–(4) in closed form (substitution u = (1−f/f_c)^{3/2}) |
| (6) | p(f \| ḟ) = [Γ(f)/ḟ] ψ(f) | verified; = −dψ/df |
| (7) | f(ω) = f_c[1 − (1 − ω)^{2/3}],  ω = (βU‡)^{−1} ln(1 − μ0^{−1} ln ψ) | verified up to a typo, see below |
| (8) | ⟨f⟩ ≅ f_c[1 − {1 − e^{μ0}E₁(μ0)/(βU‡)}^{2/3}] | verified; first-order series approximation, not exact |

**Typo in the paper.** Below Eq. (7) the paper prints ω(ψ) with "μ0 ln ψ".
Inverting Eq. (5′) gives ln(1 − μ0^{−1} ln ψ), and only that form
reproduces Eq. (8) through ∫₀^∞ e^{−s} ln(1 + s/μ0) ds = e^{μ0}E₁(μ0).
The code uses μ0^{−1}; Eq. (8) is therefore consistent with Eqs. (5)–(6).

**Residual mass at f_c.** ∫₀^{f_c} p df = 1 − ψ(f_c). The model puts the
remaining probability ψ(f_c) = exp{μ0(1 − e^{βU‡})} on "no thermal
transition before the barrier vanishes" (deterministic transition at f_c).
For Table 1 column 1 at our ḟ this is 2×10⁻¹³; for column 2 it is 2.4×10⁻³.
The likelihood uses the density for every observation (all f‡ < f_c) and
reports ψ(f_c) as a diagnostic; parameter sets with f_c ≤ max f‡ have zero
likelihood.

**Exact vs approximate mean.** The exact mean is ∫₀^{f_c} f p df + f_c ψ(f_c)
by quadrature. At our ḟ with Table 1 column 1: Eq. (8) gives 19.55 pN, the
exact integral 19.69 pN (s.d. 2.78). The difference is the neglected
f″ term of the paper's series; it will be reported.

**Small-force limit** (text above Eq. 3): Γ(f) ∝ exp(fξ‡/k_BT) for f ≪ U‡/ξ‡.
From (4): d ln Γ/df |₀ = βξ‡ − 1/(2f_c). Test case.

**Dimensionless form.** The PDF depends on w only through
(f_c, βU‡, μ0) ↔ (U‡, ξ‡, Γ0), a bijection once k_BT and ḟ are fixed:

    ln p(f) = ln(3μ0 βU‡ / 2f_c) + ½ ln(1 − f/f_c) + βU‡ Δ(f) + μ0[1 − e^{βU‡ Δ(f)}]

This is the form evaluated numerically (log-space; the product
μ0·e^{βU‡Δ} is computed as exp(ln μ0 + βU‡Δ) with an overflow guard).

## 4. Three readings of "the inverse problem" — the ambiguity

**(A) MLE of Eq. (6) on our 100 critical forces at one rate.**
Feasible. This is exactly the paper's method 1 restricted to Nᵣ = 1.
Limitation: with a single rate the three parameters are only weakly
separable — see §5. The fit must be reported with profile likelihoods and
bootstrap intervals, not just point estimates.

**(B) Least squares of Eq. (8) on ⟨f‡⟩(ḟ).**
Not feasible as an inverse problem: we have one (ḟ, ⟨f‡⟩) point and three
unknowns. What *is* possible is the **forward** direction of the
handwritten note (*given U(x) → ⟨F⟩*): evaluate Eq. (8) and the exact mean
with the paper's Table 1 parameters at our ḟ and compare with the sample
mean. That is a consistency check, not an inference, and will be labelled
as such.

**(C) Reconstruct U(x) from the mean force–displacement curve ⟨f⟩(Lₓ).**
A literal reading of "given ⟨F(x)⟩, find U(x)". **Not supported by the
paper**: Lₓ is the imposed box length, ξ is an unobserved order parameter,
and the paper never relates ⟨f(t)⟩ or ⟨f⟩(Lₓ) to U₀(ξ); −∫⟨f⟩ dLₓ is the
elastic work done on the box, not the landscape. V1 already warns against
conflating the two. **Recommended: do not implement.**

**Recommendation: implement (A) as the inverse problem, with (B)'s forward
evaluation as the paper comparison, and document why (C) is excluded.**

## 5. Identifiability with a single rate (the honest part)

Write a = βU‡, m = μ0. In the regime the paper's parameters occupy
(a ≈ 15, m ≈ 6×10⁻⁶, m·eᵃ ≈ 30), the density is concentrated where
m·e^{aΔ(f)} ~ 1, i.e. at Δ(f*) ≈ ln(1/m)/a, with a width set by
1/[a Δ′(f*)] (Gumbel-like). A single-rate sample therefore pins down
essentially **two** combinations — the location and the width of the peak —
and the third (mostly the trade-off between a and ln m along
ln(1/m)/a = const) is carried only by the skew of the distribution, which
100 samples barely resolve. The paper avoids this by using four rates:
m ∝ 1/ḟ moves the location with rate while a does not.

Consequences for V2:

1. The unconstrained 3-parameter fit will be run, but its Hessian will be
   near-singular along one direction. This is a property of the problem,
   not a bug, and will be shown (profile likelihood, bootstrap scatter).
2. **Constrained fits** will be run alongside: fix one parameter at its
   Table 1 value (ξ‡, or U‡) and infer the other two. These are
   conditional estimates and will be labelled as such.
3. **Synthetic recovery tests**: sample the model at the paper's parameters
   for S = 100 (our size), S = 400, and 4 rates × 400 (the paper's design),
   refit, and report how well each design recovers the truth. This turns
   the identifiability statement into numbers.

## 6. Additional assumptions not settled by the paper or the data

| assumption | where it enters | how it is handled |
| --- | --- | --- |
| ḟ fit window (V1) | scales every f‡ | refit at 50/100/150 ps; report parameter drift |
| peak picker smoothing (V1) | defines t‡ | refit at 7/15/31 samples; report drift |
| f‡ = ḟ·t‡ rather than the measured peak | paper's definition | primary; the smoothed-peak values are refit as a secondary set |
| independence of the 100 events | Eq. (1) | independent MD realisations (Methods) — accepted |
| absorbing boundary / steady flux (Eqs. 3–5) | model | paper's assumption — accepted, listed |
| ⟨f(t)⟩ = ḟt exactly (Eq. 2) | maps t‡ → f‡ | V1 shows linearity to R² = 0.97 on the window — accepted |

## 7. Numerical approach

* **Objective**: negative log-likelihood NLL(θ), θ = (ln U‡, ln ξ‡, ln Γ0),
  vectorised over the sample; multi-rate groups supported (Σⱼ over rates)
  so the synthetic 4-rate test uses the same code.
* **Bounds** (safe, generous): U‡ ∈ [100, 3000] pN·Å (2.4–72 k_BT),
  ξ‡ ∈ [2, 200] Å, Γ0 ∈ [10⁻¹⁴, 10²] ps⁻¹; f_c > max f‡ enforced by a
  finite penalty.
* **Optimiser**: Nelder–Mead (paper's choice) from a small deterministic
  multi-start set (Table 1 values ± factors), `maxiter`, `xatol`, `fatol`
  configurable; history (every accepted simplex best) recorded. Then an
  L-BFGS-B polish with bounds. Convergence status stored.
* **Uncertainty**: (i) observed-information s.e. from a finite-difference
  Hessian, with the condition number reported; (ii) profile likelihood on
  each parameter → likelihood-ratio 95 % intervals (χ²₁ = 3.84);
  (iii) non-parametric bootstrap, B configurable (default 200), seeded.
* **Model–data comparison**: histogram (Freedman–Diaconis) with Poisson
  bars; Gaussian KDE (Silverman) as a second estimator; fitted PDF; empirical
  vs model CDF; KS statistic and p-value; Pearson χ² on merged bins;
  per-bin residuals; log-likelihood and AIC; ψ(f_c).
* **Cost**: one NLL is ~100 exponentials; a full fit ≈ 10³ evaluations;
  bootstrap 200 fits and 3 profiles × 25 points are seconds on CPU. No
  parallelism, no GPU.
* **Storage**: `artifacts/inverse/<run_id>/` with config, dataset
  fingerprint, code hash, bounds, starts, history, result, uncertainty,
  residuals, diagnostics; `latest.json` pointer; previous runs never
  overwritten.

## 8. Validation strategy

1. Forward model: ∫p + ψ(f_c) = 1; p = −dψ/df numerically; ψ(0) = 1,
   monotone; Eq. (8) vs exact mean; small-force slope; ⟨f⟩ increases with ḟ
   (the paper's headline result); sampler's empirical CDF matches 1 − ψ.
2. Inference on synthetic data with known parameters (§5.3).
3. Real data: fitted parameters vs Table 1; NLL of our data under Table 1
   parameters vs the fit (likelihood-ratio); predicted ⟨f‡⟩ vs sample mean
   (18.5 ± 3.0 by ḟ·t‡; Fig. 3 reads ≈ 19.5 at this rate).

## 9. Implementation plan

| stage | deliverable |
| --- | --- |
| 2 | `tini/inverse/model.py` (forward model + sampler) and tests |
| 3 | `tini/inverse/{data,density,fit,uncertainty,diagnostics,runs}.py`; pipeline stage `inverse`; API |
| 4 | synthetic-recovery tests; paper comparison; sensitivity to V1 assumptions |
| 5–6 | inverse-problem page in the frontend (sections A–G of the brief) |
| 7 | review, stability, docs, commit |
