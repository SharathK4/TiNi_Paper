"""Forward model (paper Eqs. 3–8): identities, limiting cases, sampler."""

import numpy as np
import pytest

from tini.config import K_B_T_PN_ANGSTROM, PAPER_PARAMETER_SETS
from tini.inverse.model import LandscapeKinetics, small_force_log_slope

FDOT = 0.0943  # pN/ps, rate (v) from V1


@pytest.fixture(params=list(PAPER_PARAMETER_SETS))
def land(request):
    p = PAPER_PARAMETER_SETS[request.param]
    return LandscapeKinetics(p["U_barrier_pN_A"], p["xi_barrier_A"], p["gamma0_per_ps"])


def test_groups_match_hand_values():
    l = LandscapeKinetics(638.14, 29.2, 3.96e-7)
    assert l.f_c == pytest.approx(32.78, abs=0.01)
    assert l.a == pytest.approx(15.41, abs=0.01)
    assert l.mu0(FDOT) == pytest.approx(5.954e-6, rel=1e-3)


def test_survival_starts_at_one_and_decreases(land):
    f = np.linspace(0, land.f_c, 2000, endpoint=False)
    psi = land.survival(f, FDOT)
    assert psi[0] == pytest.approx(1.0)
    assert np.all(np.diff(psi) <= 1e-12)
    assert 0.0 <= land.residual_mass(FDOT) < 0.01


def test_pdf_is_minus_dpsi_df(land):
    f = np.linspace(0.05 * land.f_c, 0.95 * land.f_c, 4001)
    psi = land.survival(f, FDOT)
    num = -np.gradient(psi, f)
    ana = land.pdf(f, FDOT)
    assert np.max(np.abs(num - ana)[5:-5]) < 2e-3 * ana.max()


def test_pdf_normalises_with_residual_mass(land):
    m = land.moments_exact(FDOT)
    assert m["norm"] == pytest.approx(1.0, abs=1e-4)


def test_pdf_zero_outside_support(land):
    assert land.pdf(np.array([-1.0, land.f_c, land.f_c + 1]), FDOT).tolist() == [0.0, 0.0, 0.0]


def test_eq8_approximates_exact_mean(land):
    ex = land.moments_exact(FDOT)["mean"]
    ap = land.mean_eq8(FDOT)
    # the paper drops the f'' term; agreement to ~1 % is what the series promises
    assert abs(ap - ex) / ex < 0.02


def test_paper_parameters_predict_our_rate():
    # Table 1 (MLE column) at our ḟ: ⟨f‡⟩ ≈ 19.7 pN, s.d. ≈ 2.8 — the values quoted in docs
    l = LandscapeKinetics(638.14, 29.2, 3.96e-7)
    m = l.moments_exact(FDOT)
    assert m["mean"] == pytest.approx(19.69, abs=0.05)
    assert m["sd"] == pytest.approx(2.78, abs=0.05)
    assert l.mean_eq8(FDOT) == pytest.approx(19.55, abs=0.05)


def test_mean_increases_with_force_rate(land):
    # the paper's headline: faster loading → higher critical force
    rates = [3.246e-3, 1.482e-2, 9.452e-2, 9.242e-1]
    means = [land.moments_exact(r)["mean"] for r in rates]
    assert all(np.diff(means) > 0)
    assert means[-1] < land.f_c


def test_small_force_limit(land):
    f = np.array([0.0, 1e-4])
    slope = np.diff(np.log(land.rate(f, )))[0] / 1e-4
    assert slope == pytest.approx(small_force_log_slope(land), rel=1e-3)
    # and it is the paper's Bell-like exp(f ξ‡/k_BT) up to the prefactor correction
    # the prefactor term 1/(2f_c) is ~2 % of βξ‡ here
    assert abs(slope - land.xi_barrier / K_B_T_PN_ANGSTROM) < 0.03 * land.xi_barrier / K_B_T_PN_ANGSTROM


def test_quantile_inverts_cdf(land):
    u = np.linspace(0.01, 0.99, 99)
    f = land.quantile(u, FDOT)
    assert np.allclose(land.cdf(f, FDOT), u, atol=1e-9)


def test_sampler_matches_cdf(land):
    x = land.sample(20000, FDOT, seed=1)
    x = np.sort(x)
    emp = np.arange(1, len(x) + 1) / len(x)
    ks = np.max(np.abs(emp - land.cdf(x, FDOT)))
    assert ks < 0.015  # KS critical value at n=20000 is ~0.0096 (5 %); allow ties at f_c
    assert abs(x.mean() - land.moments_exact(FDOT)["mean"]) < 0.1


def test_log_space_is_stable_at_extreme_parameters():
    # tiny Γ0 and a huge barrier: exp(aΔ) alone would overflow; the log form must not
    l = LandscapeKinetics(2500.0, 10.0, 1e-14)
    lp = l.log_pdf(np.linspace(0, l.f_c, 100, endpoint=False), FDOT)
    assert np.all(np.isfinite(lp))
    assert l.moments_exact(FDOT)["norm"] == pytest.approx(1.0, abs=1e-3)


def test_rejects_nonpositive():
    with pytest.raises(ValueError):
        LandscapeKinetics(1.0, 1.0, 0.0)
