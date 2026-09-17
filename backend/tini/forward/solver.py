"""Forward Smoluchowski solver (paper Eq. S, docs/FORWARD_MODEL.md §1–5).

    ∂ₜP = ∂ξ D [ ∂ξP + β (∂ξU) P ],   U(ξ,t) = U₀(ξ) − ḟ t ξ,   D = k_BT/η

Finite volume on a uniform grid, Scharfetter–Gummel interface fluxes,
backward Euler in time (θ-scheme available for the convergence study).
Reflecting wall on the left, absorbing boundary on the right; the absorbed
mass is accumulated from the exact discrete boundary flux, so

    Σᵢ P̄ᵢ h + absorbed = 1

holds to rounding at every step. Nothing is clipped or renormalised.

Units: ξ in Å, t in ps, U in pN·Å, P in 1/Å, D in Å²/ps, η in pN·ps/Å.
The only dimensionless field inside is u = U/k_BT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.linalg import solve_banded

from ..config import K_B_T_PN_ANGSTROM
from ..potential import Landscape

BOUNDARIES = ("barrier_top", "beyond_barrier")


# ------------------------------------------------------------------ parameters
@dataclass(frozen=True)
class ForwardParams:
    U_barrier: float          # U‡  [pN·Å]
    xi_barrier: float         # ξ‡  [Å]
    gamma0: float             # Γ0  [1/ps]
    kBT: float = K_B_T_PN_ANGSTROM

    def __post_init__(self) -> None:
        if not (self.U_barrier > 0 and self.xi_barrier > 0 and self.gamma0 > 0 and self.kBT > 0):
            raise ValueError("U‡, ξ‡, Γ0, k_BT must be positive")

    @property
    def f_c(self) -> float:
        return 3.0 * self.U_barrier / (2.0 * self.xi_barrier)

    @property
    def curvature(self) -> float:
        """|U₀''| at the well and at the barrier top, 6U‡/ξ‡² [pN/Å]."""
        return 6.0 * self.U_barrier / self.xi_barrier**2

    @property
    def eta(self) -> float:
        """Friction coefficient from the paper's Kramers prefactor [pN·ps/Å] (§2)."""
        return self.curvature * np.exp(-self.U_barrier / self.kBT) / (2.0 * np.pi * self.gamma0)

    @property
    def D(self) -> float:
        """Diffusion coefficient k_BT/η [Å²/ps]."""
        return self.kBT / self.eta

    def to_dict(self) -> dict[str, float]:
        return {
            "U_barrier_pN_A": self.U_barrier, "xi_barrier_A": self.xi_barrier, "gamma0_per_ps": self.gamma0, "kBT_pN_A": self.kBT,
            "f_c_pN": self.f_c, "U_barrier_kBT": self.U_barrier / self.kBT, "curvature_pN_per_A": self.curvature,
            "eta_pN_ps_per_A": float(self.eta), "D_A2_per_ps": float(self.D),
            "tau_well_ps": float(self.eta / self.curvature), "tau_diffusion_ps": float(self.xi_barrier**2 / self.D),
        }


@dataclass(frozen=True)
class SolverSettings:
    n_cells: int = 250
    n_steps: int = 20000
    theta: float = 1.0                  # 1 = backward Euler (default, positive); 0.5 = Crank–Nicolson (study only)
    boundary: str = "beyond_barrier"
    margin_kBT: float = 4.0             # beyond_barrier: U₀ at ξ_A is this far below the barrier top
    xi_left_factor: float = 0.6         # ξ_L = −xi_left_factor · ξ‡
    t_end_factor: float = 1.3           # t_end = t_end_factor · f_c/ḟ: the ramp continues past f_c so transit through the vanished barrier completes
    n_snapshots: int = 80
    xi_A_override: float | None = None  # validation only: place the absorbing boundary here

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        if not (10 <= self.n_cells <= 20000 and 10 <= self.n_steps <= 5_000_000 and 0.5 <= self.theta <= 1.0):
            raise ValueError("solver settings outside safe limits")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ------------------------------------------------------------------ pieces
def bernoulli(x: np.ndarray) -> np.ndarray:
    """B(x) = x / (eˣ − 1), the Scharfetter–Gummel weight; B(0) = 1, stable for all x."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-10
    big_neg = x < -700.0
    mid = ~(small | big_neg)
    out[small] = 1.0 - x[small] / 2.0
    out[big_neg] = -x[big_neg]
    out[mid] = x[mid] / np.expm1(x[mid])
    return out


@dataclass
class Grid:
    xi_L: float
    xi_A: float
    n: int

    def __post_init__(self) -> None:
        self.h = (self.xi_A - self.xi_L) / self.n
        # cell centres; the absorbing boundary sits on the right face of the last cell
        self.centers = self.xi_L + (np.arange(self.n) + 0.5) * self.h

    def to_dict(self) -> dict[str, Any]:
        return {"xi_L_A": self.xi_L, "xi_A_A": self.xi_A, "n_cells": self.n, "h_A": self.h}


def absorbing_position(params: ForwardParams, land: Landscape, settings: SolverSettings) -> tuple[float, dict[str, Any]]:
    if settings.xi_A_override is not None:
        xa = float(settings.xi_A_override)
        return xa, {"boundary": "override", "xi_A_A": xa, "U0_at_xi_A_pN_A": float(land.U0(np.array([xa]))[0])}
    if settings.boundary == "barrier_top":
        return params.xi_barrier, {"boundary": "barrier_top", "xi_A_A": params.xi_barrier, "U0_at_xi_A_pN_A": params.U_barrier}
    # beyond the barrier: first ξ > ξ‡ where U₀ has dropped by margin_kBT·k_BT
    target = params.U_barrier - settings.margin_kBT * params.kBT
    xs = np.linspace(params.xi_barrier, 2.0 * params.xi_barrier, 4001)
    U = land.U0(xs)
    idx = int(np.argmax(U <= target))
    xa = float(xs[idx]) if U[idx] <= target else float(xs[-1])
    return xa, {"boundary": "beyond_barrier", "xi_A_A": xa, "U0_at_xi_A_pN_A": float(land.U0(np.array([xa]))[0]), "margin_kBT": settings.margin_kBT}


def make_grid(params: ForwardParams, settings: SolverSettings) -> tuple[Grid, dict[str, Any]]:
    land = Landscape(params.U_barrier, params.xi_barrier)
    xa, info = absorbing_position(params, land, settings)
    g = Grid(-settings.xi_left_factor * params.xi_barrier, xa, settings.n_cells)
    info["xi_L_A"] = g.xi_L
    info["U0_at_xi_L_kBT"] = float(land.U0(np.array([g.xi_L]))[0] / params.kBT)
    return g, info


PotentialFn = Callable[[np.ndarray, float], np.ndarray]   # (ξ, f) → U(ξ, f) in pN·Å


def cubic_potential(params: ForwardParams) -> PotentialFn:
    land = Landscape(params.U_barrier, params.xi_barrier)
    return lambda x, f: land.U0(x) - f * x


def transition_state(params: ForwardParams, force: float) -> float:
    """ξ_TS(f) = ξ‡/2 (1 + √(1 − f/f_c)) for f ≤ f_c (V1, derived); ξ‡/2 beyond."""
    r = max(0.0, 1.0 - force / params.f_c)
    return params.xi_barrier / 2.0 * (1.0 + np.sqrt(r))


def equilibrium(params: ForwardParams, grid: Grid, force: float = 0.0, potential: PotentialFn | None = None, well_only: bool = True) -> np.ndarray:
    """Boltzmann density e^{−βU} normalised on the grid (cell averages ≈ centre
    values). With ``well_only`` the density is confined to ξ ≤ ξ_TS(force):
    the physical initial state is the austenite well, not the downhill
    region beyond the barrier that a domain past ξ‡ also contains."""
    pot = potential or cubic_potential(params)
    u = pot(grid.centers, force) / params.kBT
    mask = grid.centers <= transition_state(params, force) if (well_only and potential is None) else np.ones(grid.n, bool)
    w = np.where(mask, np.exp(-(u - u[mask].min())), 0.0)
    return w / (w.sum() * grid.h)


def build_operator(params: ForwardParams, grid: Grid, force: float, absorbing: bool = True, potential: PotentialFn | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Tridiagonal A (lower, diag, upper) of dP/dt = A P, and the boundary-flux
    coefficient c such that J_A = c · P_N (the first-passage flux)."""
    pot = potential or cubic_potential(params)
    x = grid.centers
    h, D = grid.h, params.D
    u = pot(x, force) / params.kBT
    du = np.diff(u)                        # Δu at the N−1 interior interfaces
    Bp = bernoulli(du)                     # B(+Δu)
    Bm = bernoulli(-du)                    # B(−Δu)
    k = D / h**2
    lower = np.zeros(grid.n)
    upper = np.zeros(grid.n)
    diag = np.zeros(grid.n)
    # interior interfaces i+½, i = 0..N−2: J = k h [B(Δu) P_i − B(−Δu) P_{i+1}]
    diag[:-1] -= k * Bp
    upper[:-1] += k * Bm
    diag[1:] -= k * Bm
    lower[1:] += k * Bp
    c = 0.0
    if absorbing:
        # ghost P = 0 at the right face, half a cell from the last centre
        u_A = pot(np.array([grid.xi_A]), force)[0] / params.kBT
        cA = (D / (h / 2.0)) * bernoulli(np.array([u_A - u[-1]]))[0]
        diag[-1] -= cA / h
        c = cA
    return lower, diag, upper, c


# ------------------------------------------------------------------ solution
@dataclass
class ForwardSolution:
    t: np.ndarray                 # step times (n_steps+1)
    survival: np.ndarray          # Ψ(t)
    absorbed: np.ndarray          # cumulative absorbed mass
    flux: np.ndarray              # ρ(t) = boundary flux [1/ps], θ-weighted per step
    snapshot_t: np.ndarray
    snapshots: np.ndarray         # (n_snapshots, n_cells)
    grid: Grid
    params: ForwardParams
    settings: SolverSettings
    force_rate: float
    min_density: float
    conservation_error_max: float
    boundary_info: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def force(self) -> np.ndarray:
        return self.force_rate * self.t

    @property
    def residual_mass(self) -> float:
        return float(self.survival[-1])


def solve(
    params: ForwardParams,
    force_rate: float,
    settings: SolverSettings | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    force_override: Callable[[float], float] | None = None,
    t_end: float | None = None,
    initial: np.ndarray | None = None,
    absorbing: bool = True,
    potential: PotentialFn | None = None,
) -> ForwardSolution:
    """Integrate from equilibrium at f = 0 to t_end (default f_c/ḟ).

    ``force_override(t)`` replaces ḟ·t (used by the fixed-force validation
    runs); ``absorbing=False`` makes the right wall reflecting (test C1)."""
    import time as _time

    settings = settings or SolverSettings()
    if force_rate <= 0 and force_override is None:
        raise ValueError("force_rate must be positive")
    grid, binfo = make_grid(params, settings)
    T = t_end if t_end is not None else settings.t_end_factor * params.f_c / force_rate
    N, Nt, th = grid.n, settings.n_steps, settings.theta
    dt = T / Nt
    t = np.linspace(0.0, T, Nt + 1)
    f_of_t = force_override or (lambda tt: force_rate * tt)

    P = initial.copy() if initial is not None else equilibrium(params, grid, 0.0, potential)
    P = P / (P.sum() * grid.h)
    surv = np.empty(Nt + 1)
    absd = np.zeros(Nt + 1)
    flux = np.zeros(Nt + 1)
    surv[0] = P.sum() * grid.h
    snap_idx = np.unique(np.linspace(0, Nt, settings.n_snapshots).astype(int))
    snaps = np.empty((snap_idx.size, N))
    snaps[0] = P
    si = 1
    min_density = float(P.min())
    cons_err = 0.0
    t0 = _time.time()

    lo_n, di_n, up_n, c_n = build_operator(params, grid, f_of_t(t[0]), absorbing, potential)
    flux[0] = c_n * P[-1]
    ab = np.zeros((3, N))
    for n in range(Nt):
        if cancel and n % 200 == 0 and cancel():
            raise InterruptedError("cancelled")
        lo, di, up, c = build_operator(params, grid, f_of_t(t[n + 1]), absorbing, potential)
        # (I − θ dt A^{n+1}) P^{n+1} = (I + (1−θ) dt A^n) P^n
        if th < 1.0:
            rhs = P + (1 - th) * dt * (di_n * P + np.append(up_n[:-1] * P[1:], 0.0) + np.append(0.0, lo_n[1:] * P[:-1]))
        else:
            rhs = P
        ab[0, 1:] = -th * dt * up[:-1]
        ab[1, :] = 1.0 - th * dt * di
        ab[2, :-1] = -th * dt * lo[1:]
        P_new = solve_banded((1, 1), ab, rhs, overwrite_ab=False, overwrite_b=False, check_finite=False)
        J = th * c * P_new[-1] + (1 - th) * c_n * P[-1]
        absd[n + 1] = absd[n] + dt * J
        flux[n + 1] = c * P_new[-1]
        P = P_new
        surv[n + 1] = P.sum() * grid.h
        cons_err = max(cons_err, abs(surv[n + 1] + absd[n + 1] - 1.0))
        min_density = min(min_density, float(P.min()))
        lo_n, di_n, up_n, c_n = lo, di, up, c
        if si < snap_idx.size and n + 1 == snap_idx[si]:
            snaps[si] = P
            si += 1
        if progress and (n % 500 == 0 or n == Nt - 1):
            progress(n + 1, Nt)
    return ForwardSolution(
        t=t, survival=surv, absorbed=absd, flux=flux, snapshot_t=t[snap_idx], snapshots=snaps, grid=grid, params=params,
        settings=settings, force_rate=force_rate if force_override is None else 0.0, min_density=min_density,
        conservation_error_max=cons_err, boundary_info=binfo, duration_s=round(_time.time() - t0, 3),
    )


# ------------------------------------------------------------------ observables
def force_density(sol: ForwardSolution) -> dict[str, Any]:
    """p(f | ḟ) = ρ(t)/ḟ at f = ḟ t on the whole run, its moments and CDF.

    The PDE lets density be absorbed after the barrier has vanished (f > f_c):
    that transit mass is reported as ``mass_after_fc``. Whatever is still in
    the domain at the end of the run (``residual_mass``, ≈ 0 with the default
    t_end = 1.3 f_c/ḟ) is placed at the final force for the moments."""
    if sol.force_rate <= 0:
        raise ValueError("force density needs a ramped run")
    f = sol.force
    p = sol.flux / sol.force_rate
    cdf = sol.absorbed.copy()
    rm = sol.residual_mass
    fc = sol.params.f_c
    f_end = float(f[-1])
    Z = float(np.trapezoid(p, f)) + rm
    mean = float(np.trapezoid(f * p, f)) + f_end * rm
    var = float(np.trapezoid((f - mean) ** 2 * p, f)) + (f_end - mean) ** 2 * rm
    after = float(sol.absorbed[-1] - np.interp(fc, f, sol.absorbed)) if f_end > fc else 0.0
    return {
        "f_pN": f, "pdf": p, "cdf": cdf, "residual_mass": rm, "mass_after_fc": after, "norm": Z,
        "mean_pN": mean, "sd_pN": float(np.sqrt(max(var, 0.0))), "mode_pN": float(f[int(np.argmax(p))]),
    }


def mean_first_passage_exact(params: ForwardParams, grid: Grid, force: float, n_quad: int = 20001) -> float:
    """Exact MFPT of 1-D overdamped diffusion from the Boltzmann-distributed
    start to the absorbing point, reflecting at ξ_L (Risken §5.2.2):
        τ(x0) = (1/D) ∫_{x0}^{ξ_A} e^{u(y)} ∫_{ξ_L}^{y} e^{−u(z)} dz dy,  τ̄ = ∫ P0(x0) τ(x0) dx0."""
    land = Landscape(params.U_barrier, params.xi_barrier)
    x = np.linspace(grid.xi_L, grid.xi_A, n_quad)
    u = (land.U0(x) - force * x) / params.kBT
    well = x <= transition_state(params, force)
    u0 = u[well].min()
    dx = x[1] - x[0]
    inner = np.concatenate([[0.0], np.cumsum(0.5 * (np.exp(-(u[1:] - u0)) + np.exp(-(u[:-1] - u0))) * dx)])
    integrand = np.exp(u - u0) * inner
    # τ(x0) = (1/D) ∫_{x0}^{ξ_A} integrand dy  → cumulative from the right
    tail = np.concatenate([np.cumsum((0.5 * (integrand[1:] + integrand[:-1]) * dx)[::-1])[::-1], [0.0]])
    tau = tail / params.D
    P0 = np.where(well, np.exp(-(u - u0)), 0.0)
    P0 /= np.trapezoid(P0, x)
    return float(np.trapezoid(P0 * tau, x))
