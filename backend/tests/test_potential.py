"""Checks on the cubic landscape: the paper's stated properties must hold
exactly, and every derived formula must agree with a numerical evaluation."""

import numpy as np
import pytest

from tini.config import K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS
from tini.potential import Landscape, landscape_from_paper, numerical_force, potential_curves, xi_grid


@pytest.fixture(params=list(PAPER_PARAMETER_SETS))
def land(request):
    return landscape_from_paper(request.param)


def test_kBT_constant():
    # 1.380649e-23 J/K * 300 K = 4.1419e-21 J = 41.42 pN.A
    assert abs(K_B_T_PN_ANGSTROM - 41.419) < 0.01


def test_U0_boundary_values(land):
    # U0(0) = 0 (austenite minimum), U0(xi‡) = U‡ (barrier top)
    assert abs(land.U0(np.array([0.0]))[0]) < 1e-9
    assert abs(land.U0(np.array([land.xi_barrier]))[0] - land.U_barrier) < 1e-9


def test_stationary_points_at_zero_force(land):
    # dU0/dxi = 0 at xi = 0 and xi = xi‡  <=>  F = 0 there
    F = land.force(np.array([0.0, land.xi_barrier]))
    assert np.allclose(F, 0.0, atol=1e-9)
    # xi = 0 is a minimum, xi‡ a maximum
    h = 1e-3
    assert land.U0(np.array([h]))[0] > 0 and land.U0(np.array([-h]))[0] > 0
    top = land.U0(np.array([land.xi_barrier - h, land.xi_barrier + h]))
    assert (top < land.U_barrier).all()


def test_fc_is_the_extreme_landscape_force(land):
    xi = xi_grid(land, 2001)
    F = land.force(xi)
    assert abs(F.min() + land.f_c) < 1e-6
    assert abs(xi[F.argmin()] - land.xi_barrier / 2) < 0.05


def test_analytic_force_matches_numerical(land):
    xi = xi_grid(land, 4001)
    F = land.force(xi)
    Fn = numerical_force(xi, land.U0(xi))
    assert np.max(np.abs(F[2:-2] - Fn[2:-2])) < 1e-3 * land.f_c


def test_barrier_under_force_matches_direct_evaluation(land):
    xi = np.linspace(-land.xi_barrier, 2 * land.xi_barrier, 200001)
    for frac in (0.0, 0.2, 0.5, 0.8, 0.95):
        f = frac * land.f_c
        U = land.U_tilted(xi, f)
        sp = land.stationary_points(f)
        i_min = np.argmin(np.abs(xi - sp["xi_min"]))
        i_ts = np.argmin(np.abs(xi - sp["xi_ts"]))
        direct = U[i_ts] - U[i_min]
        assert abs(direct - land.barrier_under_force(f)) < 1e-3 * land.U_barrier
        # the tilted force must vanish at both stationary points
        assert np.allclose(land.force_tilted(np.array([sp["xi_min"], sp["xi_ts"]]), f), 0.0, atol=1e-6)


def test_barrier_vanishes_at_fc(land):
    assert land.barrier_under_force(land.f_c) == 0.0
    assert land.barrier_under_force(1.2 * land.f_c) == 0.0
    sp = land.stationary_points(land.f_c)
    assert abs(sp["xi_min"] - sp["xi_ts"]) < 1e-9


def test_minimum_moves_right_under_force(land):
    a = land.stationary_points(0.0)["xi_min"]
    b = land.stationary_points(0.5 * land.f_c)["xi_min"]
    assert a == pytest.approx(0.0) and b > a


def test_paper_values():
    # Table 1 (PDF/MLE column): f_c = 3*638.14/(2*29.2) = 32.78 pN
    l = landscape_from_paper("pdf_mle")
    assert abs(l.f_c - 32.78) < 0.01
    assert abs(l.U_barrier / K_B_T_PN_ANGSTROM - 15.41) < 0.01


def test_potential_curves_payload():
    l = Landscape(600.0, 30.0)
    c = potential_curves(l, 101, [0.0, 0.5 * l.f_c], force_rate=0.1)
    assert c["xi_A"].shape == (101,)
    assert c["tilts"][1]["t_ps"] == pytest.approx(0.5 * l.f_c / 0.1)
    assert c["force_check_max_abs_pN"] < 0.05 * l.f_c


def test_rejects_nonpositive():
    with pytest.raises(ValueError):
        Landscape(0.0, 1.0)
