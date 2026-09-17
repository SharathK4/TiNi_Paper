"""Forward model of the critical-force distribution (paper Eqs. 3–8).

Everything here is the paper's model evaluated numerically; nothing is
fitted. See docs/INVERSE_PROBLEM.md §3 for the derivation of the forms used.

Parameters (``LandscapeKinetics``):
    U_barrier   U‡   free-energy barrier at zero force        [pN·Å]
    xi_barrier  ξ‡   transition-state distance                [Å]
    gamma0      Γ0   intrinsic rate prefactor                 [1/ps]
Conditions:
    force_rate  ḟ    imposed force rate                       [pN/ps]
    kBT              thermal energy (41.42 pN·Å at 300 K)     [pN·Å]

Dimensionless groups used throughout:
    f_c = 3U‡/(2ξ‡)                 maximal turnover force        (paper)
    a   = U‡/k_BT                   barrier in thermal units
    μ0  = Γ0 k_BT / (ḟ ξ‡)          paper's μ0 (below Eq. 6)
    Δ(f) = 1 − (1 − f/f_c)^{3/2}   fractional barrier reduction  (Eq. 4)

Verified forms:
    Γ(f)  = Γ0 (1 − f/f_c)^{1/2} exp[a Δ(f)]                    (Eq. 4)
    ψ(f)  = exp{ μ0 [1 − exp(a Δ(f))] }                          (Eq. 5, rewritten)
    p(f)  = Γ(f) ψ(f) / ḟ                                        (Eq. 6)
    ⟨f⟩  ≅ f_c [1 − (1 − e^{μ0} E1(μ0) / a)^{2/3}]              (Eq. 8, approximate)

Numerical care: μ0·exp(aΔ) is formed as exp(ln μ0 + aΔ) so that tiny μ0
and huge exp(aΔ) never meet as separate floats; the exponent is clipped at
``_EXP_MAX`` and any clipped evaluation is flagged in ``ModelEval.overflow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import exp1

from ..config import K_B_T_PN_ANGSTROM

_EXP_MAX = 700.0     # exp(700) ≈ 1e304, just under float64 overflow
_EPS = 1e-300


@dataclass(frozen=True)
class LandscapeKinetics:
    U_barrier: float
    xi_barrier: float
    gamma0: float
    kBT: float = K_B_T_PN_ANGSTROM

    def __post_init__(self) -> None:
        if not (self.U_barrier > 0 and self.xi_barrier > 0 and self.gamma0 > 0 and self.kBT > 0):
            raise ValueError("U‡, ξ‡, Γ0 and k_BT must be positive")

    # ------------------------------------------------------------ groups
    @property
    def f_c(self) -> float:
        return 3.0 * self.U_barrier / (2.0 * self.xi_barrier)

    @property
    def a(self) -> float:
        """βU‡ = U‡ / k_BT."""
        return self.U_barrier / self.kBT

    def mu0(self, force_rate: float) -> float:
        return self.gamma0 * self.kBT / (force_rate * self.xi_barrier)

    def eta(self) -> float:
        """Friction constant from the paper's Γ0 = (1/2πη)(6U‡/ξ‡²) e^{−βU‡}  [1/ps]. Derived, not fitted."""
        return (6.0 * self.U_barrier / self.xi_barrier**2) * np.exp(-self.a) / (2.0 * np.pi * self.gamma0)

    def to_dict(self, force_rate: float | None = None) -> dict[str, Any]:
        d = {
            "U_barrier_pN_A": self.U_barrier,
            "xi_barrier_A": self.xi_barrier,
            "gamma0_per_ps": self.gamma0,
            "kBT_pN_A": self.kBT,
            "f_c_pN": self.f_c,
            "U_barrier_kBT": self.a,
            "eta_per_ps": float(self.eta()),
        }
        if force_rate is not None:
            d["mu0"] = self.mu0(force_rate)
            d["mu0_times_exp_a"] = float(np.exp(np.log(self.mu0(force_rate)) + self.a)) if np.log(self.mu0(force_rate)) + self.a < _EXP_MAX else float("inf")
        return d

    # ------------------------------------------------------------ pieces
    def delta(self, f: np.ndarray) -> np.ndarray:
        """Δ(f) = 1 − (1 − f/f_c)^{3/2} on 0 ≤ f ≤ f_c (clipped outside)."""
        r = np.clip(1.0 - np.asarray(f, dtype=float) / self.f_c, 0.0, None)
        return 1.0 - r**1.5

    def rate(self, f: np.ndarray) -> np.ndarray:
        """Γ(f), Eq. (4)  [1/ps]."""
        f = np.asarray(f, dtype=float)
        r = np.clip(1.0 - f / self.f_c, 0.0, None)
        return self.gamma0 * np.sqrt(r) * np.exp(np.minimum(self.a * self.delta(f), _EXP_MAX))

    def log_survival(self, f: np.ndarray, force_rate: float) -> np.ndarray:
        """ln ψ(f) = μ0 [1 − exp(aΔ)], computed as μ0 − exp(ln μ0 + aΔ)."""
        mu0 = self.mu0(force_rate)
        x = np.log(mu0) + self.a * self.delta(f)
        return mu0 - np.exp(np.minimum(x, _EXP_MAX))

    def survival(self, f: np.ndarray, force_rate: float) -> np.ndarray:
        """ψ(f), Eq. (5): probability that the austenite state persists to force f."""
        return np.exp(self.log_survival(f, force_rate))

    def cdf(self, f: np.ndarray, force_rate: float) -> np.ndarray:
        """P(f‡ ≤ f) = 1 − ψ(f) for f < f_c; jumps to 1 at f_c (residual mass)."""
        f = np.asarray(f, dtype=float)
        out = 1.0 - self.survival(f, force_rate)
        return np.where(f >= self.f_c, 1.0, out)

    def residual_mass(self, force_rate: float) -> float:
        """ψ(f_c): probability of no thermal transition before the barrier vanishes."""
        return float(np.exp(self.log_survival(np.array([self.f_c]), force_rate))[0])

    def log_pdf(self, f: np.ndarray, force_rate: float) -> np.ndarray:
        """ln p(f | ḟ), Eq. (6), in log space. −inf outside [0, f_c)."""
        f = np.asarray(f, dtype=float)
        mu0 = self.mu0(force_rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 1.0 - f / self.f_c
            d = self.delta(f)
            x = np.log(mu0) + self.a * d
            ln_gamma = np.log(self.gamma0) + 0.5 * np.log(r) + self.a * d
            lp = ln_gamma - np.log(force_rate) + mu0 - np.exp(np.minimum(x, _EXP_MAX))
        lp = np.where((f >= 0) & (f < self.f_c), lp, -np.inf)
        return lp

    def pdf(self, f: np.ndarray, force_rate: float) -> np.ndarray:
        return np.exp(self.log_pdf(f, force_rate))

    # ------------------------------------------------------------ moments
    def mean_eq8(self, force_rate: float) -> float:
        """Paper Eq. (8): first-order series approximation to ⟨f‡⟩."""
        mu0 = self.mu0(force_rate)
        # e^{μ0} E1(μ0) via scipy.special.exp1; stable for μ0 down to ~1e-300
        s = float(np.exp(mu0) * exp1(mu0)) if mu0 < 700 else 1.0 / mu0
        inner = 1.0 - s / self.a
        if inner <= 0:
            return self.f_c  # series outside its validity: barrier reduction saturates
        return self.f_c * (1.0 - inner ** (2.0 / 3.0))

    def moments_exact(self, force_rate: float, n: int = 20001) -> dict[str, float]:
        """⟨f⟩ and s.d. by quadrature of Eq. (6) on [0, f_c) plus the point mass f_c·ψ(f_c)."""
        f = np.linspace(0.0, self.f_c, n, endpoint=False)
        p = self.pdf(f, force_rate)
        rm = self.residual_mass(force_rate)
        Z = float(np.trapezoid(p, f)) + rm
        m = float(np.trapezoid(f * p, f)) + self.f_c * rm
        v = float(np.trapezoid((f - m) ** 2 * p, f)) + (self.f_c - m) ** 2 * rm
        return {"mean": m, "sd": float(np.sqrt(max(v, 0.0))), "norm": Z, "residual_mass": rm}

    def mode(self, force_rate: float, n: int = 20001) -> float:
        f = np.linspace(0.0, self.f_c, n, endpoint=False)
        return float(f[int(np.argmax(self.log_pdf(f, force_rate)))])

    # ------------------------------------------------------------ sampling
    def quantile(self, u: np.ndarray, force_rate: float) -> np.ndarray:
        """Inverse CDF via the paper's ω mapping (Eq. 7 with the corrected μ0⁻¹):

            ψ = 1 − u,  ω = ln(1 − ln ψ / μ0) / a,  f = f_c [1 − (1 − ω)^{2/3}]

        Values with ω ≥ 1 belong to the residual mass and return f_c.
        """
        u = np.asarray(u, dtype=float)
        mu0 = self.mu0(force_rate)
        ln_psi = np.log1p(-u)
        omega = np.log1p(-ln_psi / mu0) / self.a
        r = np.clip(1.0 - omega, 0.0, None)
        return self.f_c * (1.0 - r ** (2.0 / 3.0))

    def sample(self, n: int, force_rate: float, seed: int | None = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.quantile(rng.random(n), force_rate)


def small_force_log_slope(land: LandscapeKinetics) -> float:
    """d ln Γ/df at f = 0: βξ‡ − 1/(2f_c). The paper's Γ ∝ exp(fξ‡/k_BT) to leading order."""
    return land.xi_barrier / land.kBT - 1.0 / (2.0 * land.f_c)
