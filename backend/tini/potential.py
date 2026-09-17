"""The cubic free-energy landscape and its force.

Everything in this module is either taken verbatim from Maitra & Gururajan,
Sci. Rep. 15:3008 (2025), or derived from it by elementary calculus and
checked numerically in ``tests/test_potential.py``. No fitting happens here
(that is the inverse problem, Version 2).

Verified from the paper
-----------------------
Cubic profile (Results, "Quantitative description ..."):

    U0(ξ) = U‡/2 + (3U‡ / 2ξ‡) (ξ − ξ‡/2) − (2U‡ / ξ‡³) (ξ − ξ‡/2)³

    U‡  free-energy barrier at zero force   [pN·Å]
    ξ‡  distance of the transition state from the austenite minimum  [Å]

Tilt under an imposed force (Eq. 2 and surrounding text):

    U(ξ, t) ≈ U0(ξ) − ⟨f(t)⟩ ξ,     ⟨f(t)⟩ ≈ ḟ t

Maximal turnover force (text below Eq. 4):    f_c ≡ 3U‡ / (2ξ‡)

Barrier reduction under force (Eq. 4):        ΔU‡(f) = U‡ (1 − (1 − f/f_c)^{3/2})
so the remaining barrier is                   U‡(f) = U‡ (1 − f/f_c)^{3/2}

Derived here (checked numerically)
----------------------------------
Landscape force, F(ξ) = −dU0/dξ:

    F(ξ) = −(3U‡ / 2ξ‡) + (6U‡ / ξ‡³) (ξ − ξ‡/2)²

so F(0) = F(ξ‡) = 0 (the minimum and the barrier top are stationary points)
and the most negative landscape force, −f_c, sits at the inflection ξ‡/2.
That is what makes f_c the force at which the tilted barrier vanishes.

Stationary points of the tilted profile U0(ξ) − f ξ:

    ξ_min(f), ξ_TS(f) = ξ‡/2 ∓ (ξ‡/2) √(1 − f/f_c)       for 0 ≤ f ≤ f_c

The austenite minimum moves *right* under force (this is the "m0 → mt" shift
the paper uses to explain why the mean critical force grows with rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS


@dataclass(frozen=True)
class Landscape:
    U_barrier: float   # U‡  [pN·Å]
    xi_barrier: float  # ξ‡  [Å]

    def __post_init__(self) -> None:
        if self.U_barrier <= 0 or self.xi_barrier <= 0:
            raise ValueError("U‡ and ξ‡ must be positive")

    # ---------------------------------------------------------- paper forms
    @property
    def f_c(self) -> float:
        """Maximal force to turn over the austenite state, f_c = 3U‡/(2ξ‡) [pN]."""
        return 3.0 * self.U_barrier / (2.0 * self.xi_barrier)

    def U0(self, xi: np.ndarray) -> np.ndarray:
        U, x = self.U_barrier, self.xi_barrier
        s = xi - x / 2.0
        return U / 2.0 + (3.0 * U / (2.0 * x)) * s - (2.0 * U / x**3) * s**3

    def U_tilted(self, xi: np.ndarray, f: float) -> np.ndarray:
        """U(ξ, t) = U0(ξ) − f ξ with f = ⟨f(t)⟩."""
        return self.U0(xi) - f * xi

    def barrier_under_force(self, f: float) -> float:
        """U‡(f) = U‡ (1 − f/f_c)^{3/2}; zero at and beyond f_c (Eq. 4 rearranged)."""
        r = 1.0 - f / self.f_c
        return float(self.U_barrier * max(r, 0.0) ** 1.5)

    # --------------------------------------------------------- derived forms
    def force(self, xi: np.ndarray) -> np.ndarray:
        """Landscape force F(ξ) = −dU0/dξ  [pN]."""
        U, x = self.U_barrier, self.xi_barrier
        s = xi - x / 2.0
        return -(3.0 * U / (2.0 * x)) + (6.0 * U / x**3) * s**2

    def force_tilted(self, xi: np.ndarray, f: float) -> np.ndarray:
        """−d/dξ [U0(ξ) − f ξ] = F(ξ) + f."""
        return self.force(xi) + f

    def stationary_points(self, f: float) -> dict[str, float | None]:
        r = 1.0 - f / self.f_c
        if r < 0:
            return {"xi_min": None, "xi_ts": None}
        h = (self.xi_barrier / 2.0) * np.sqrt(r)
        return {"xi_min": float(self.xi_barrier / 2.0 - h), "xi_ts": float(self.xi_barrier / 2.0 + h)}

    # --------------------------------------------------------------- output
    def to_dict(self) -> dict[str, Any]:
        return {
            "U_barrier_pN_A": self.U_barrier,
            "xi_barrier_A": self.xi_barrier,
            "f_c_pN": self.f_c,
            "U_barrier_kBT": self.U_barrier / K_B_T_PN_ANGSTROM,
            "kBT_pN_A": K_B_T_PN_ANGSTROM,
            "curvature_at_min_pN_per_A": 6.0 * self.U_barrier / self.xi_barrier**2,
        }


def landscape_from_paper(key: str) -> Landscape:
    p = PAPER_PARAMETER_SETS[key]
    return Landscape(p["U_barrier_pN_A"], p["xi_barrier_A"])


def xi_grid(land: Landscape, n: int = 241, pad: float = 0.35) -> np.ndarray:
    """Grid spanning a little beyond [0, ξ‡] so both stationary points are visible."""
    lo, hi = -pad * land.xi_barrier, (1.0 + pad) * land.xi_barrier
    return np.linspace(lo, hi, n)


def numerical_force(xi: np.ndarray, U: np.ndarray) -> np.ndarray:
    """−dU/dξ by central differences; used to cross-check the analytic force."""
    return -np.gradient(U, xi)


def potential_curves(
    land: Landscape,
    n_points: int = 241,
    tilt_forces: list[float] | None = None,
    force_rate: float | None = None,
) -> dict[str, Any]:
    """Everything the explorer needs for one parameter pair.

    ``tilt_forces`` are absolute forces (pN). If ``force_rate`` (pN/ps) is
    given, the time at which the mean ramp reaches each force is reported,
    t = f/ḟ (Eq. 2), so the tilt can be read as "the landscape at time t".
    """
    xi = xi_grid(land, n_points)
    U0 = land.U0(xi)
    F = land.force(xi)
    F_num = numerical_force(xi, U0)
    # max abs discrepancy in the interior (central differences are exact for
    # a cubic only up to O(h²)); reported so the UI can show the check.
    interior = slice(2, -2)
    force_check = float(np.max(np.abs(F[interior] - F_num[interior])))

    tilts = []
    for f in (tilt_forces or []):
        sp = land.stationary_points(f)
        tilts.append(
            {
                "f_pN": float(f),
                "f_over_fc": float(f / land.f_c),
                "t_ps": (float(f / force_rate) if force_rate else None),
                "U": land.U_tilted(xi, f),
                "barrier_pN_A": land.barrier_under_force(f),
                "barrier_kBT": land.barrier_under_force(f) / K_B_T_PN_ANGSTROM,
                **sp,
            }
        )
    return {
        "parameters": land.to_dict(),
        "xi_A": xi,
        "U0_pN_A": U0,
        "U0_kBT": U0 / K_B_T_PN_ANGSTROM,
        "F_pN": F,
        "F_numerical_pN": F_num,
        "force_check_max_abs_pN": force_check,
        "tilts": tilts,
        "inflection_xi_A": land.xi_barrier / 2.0,
    }
