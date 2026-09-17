"""Negative log-likelihood of the critical-force model (paper Eq. 1 with Eq. 6).

    NLL(w) = − Σⱼ Σᵢ ln p(fᵢⱼ | ḟⱼ ; w),   w = (U‡, ξ‡, Γ0)

Groups (fⱼ, ḟⱼ) allow several force rates in one likelihood — the paper's
design — even though the real dataset has a single rate.

Optimisation works in θ = ln w of the *free* parameters. Fixed parameters
(for the constrained fits) are held at their given values. Out-of-bounds
or f_c ≤ max f‡ evaluations return a large finite penalty that grows with
the violation, so a simplex can walk back into the feasible region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import K_B_T_PN_ANGSTROM
from .model import LandscapeKinetics

PARAM_NAMES = ("U_barrier", "xi_barrier", "gamma0")
PARAM_UNITS = {"U_barrier": "pN·Å", "xi_barrier": "Å", "gamma0": "1/ps"}

_PENALTY = 1e6


@dataclass(frozen=True)
class Bounds:
    U_barrier: tuple[float, float] = (100.0, 3000.0)     # 2.4 – 72 k_BT
    xi_barrier: tuple[float, float] = (2.0, 200.0)
    gamma0: tuple[float, float] = (1e-14, 1e2)

    def as_dict(self) -> dict[str, list[float]]:
        return {k: list(getattr(self, k)) for k in PARAM_NAMES}

    def log_bounds(self, free: list[str]) -> list[tuple[float, float]]:
        return [(np.log(getattr(self, k)[0]), np.log(getattr(self, k)[1])) for k in free]


@dataclass
class Likelihood:
    groups: list[tuple[np.ndarray, float]]          # [(forces, force_rate), ...]
    bounds: Bounds = field(default_factory=Bounds)
    fixed: dict[str, float] = field(default_factory=dict)
    kBT: float = K_B_T_PN_ANGSTROM
    n_eval: int = 0

    def __post_init__(self) -> None:
        for k in self.fixed:
            if k not in PARAM_NAMES:
                raise ValueError(f"unknown parameter {k}")
        self.free = [k for k in PARAM_NAMES if k not in self.fixed]
        self.f_max = max(float(np.max(f)) for f, _ in self.groups)
        self.n_obs = sum(int(f.size) for f, _ in self.groups)

    # ------------------------------------------------------------ mapping
    def theta_to_params(self, theta: np.ndarray) -> dict[str, float]:
        p = dict(self.fixed)
        for k, v in zip(self.free, np.asarray(theta, dtype=float)):
            p[k] = float(np.exp(v))
        return p

    def params_to_theta(self, params: dict[str, float]) -> np.ndarray:
        return np.array([np.log(params[k]) for k in self.free])

    # ------------------------------------------------------------ objective
    def nll_params(self, params: dict[str, float]) -> float:
        """Unpenalised NLL; +inf if any observation lies outside the support."""
        land = LandscapeKinetics(params["U_barrier"], params["xi_barrier"], params["gamma0"], self.kBT)
        total = 0.0
        for f, fdot in self.groups:
            lp = land.log_pdf(f, fdot)
            if not np.all(np.isfinite(lp)):
                return float("inf")
            total -= float(lp.sum())
        return total

    def nll(self, theta: np.ndarray) -> float:
        self.n_eval += 1
        theta = np.asarray(theta, dtype=float)
        pen = 0.0
        for k, v in zip(self.free, theta):
            lo, hi = np.log(getattr(self.bounds, k))
            if v < lo:
                pen += (lo - v) ** 2
            elif v > hi:
                pen += (v - hi) ** 2
        if pen > 0:
            return _PENALTY * (1.0 + pen)
        params = self.theta_to_params(theta)
        f_c = 3.0 * params["U_barrier"] / (2.0 * params["xi_barrier"])
        if f_c <= self.f_max * 1.0001:
            return _PENALTY * (1.0 + (self.f_max / f_c - 1.0))
        val = self.nll_params(params)
        if not np.isfinite(val):
            return _PENALTY
        return val

    # ------------------------------------------------------------ reporting
    def describe(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_groups": len(self.groups),
            "force_rates": [float(r) for _, r in self.groups],
            "free": self.free,
            "fixed": self.fixed,
            "bounds": self.bounds.as_dict(),
            "kBT_pN_A": self.kBT,
            "f_max_pN": self.f_max,
        }
