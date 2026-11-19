import math

import numpy as np
import pandas as pd
import pytest

from wbr.benchmark import ar_forecast_moments, choose_ar_order, conditional_mean, conditional_variance, empirical_curve


def test_ar1_forecast_moments_match_closed_form():
    ah, var = ar_forecast_moments(np.array([0.8]), 1.0, 3)
    assert ah[0] == pytest.approx(0.8 ** 3)
    assert var == pytest.approx(1 + 0.8 ** 2 + 0.8 ** 4)


def test_ar2_moments_match_wold_representation():
    phi, sigma2, h = np.array([1.05, -0.12]), 0.7, 12
    psi = [1.0, phi[0]]
    for _ in range(2, h):
        psi.append(phi[0] * psi[-1] + phi[1] * psi[-2])
    _, var = ar_forecast_moments(phi, sigma2, h)
    assert var == pytest.approx(sigma2 * sum(p * p for p in psi[:h]))


def _simulate_ar2(n, phi, sigma, mu, seed=1):
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(2, n):
        z[t] = phi[0] * z[t - 1] + phi[1] * z[t - 2] + rng.normal(0, sigma)
    return mu + z


def test_bic_selects_the_true_order_and_recovers_parameters():
    v = _simulate_ar2(60_000, (1.0, -0.15), 0.8, 7.0)
    idx = pd.date_range("2016-01-01", periods=len(v), freq="h")
    order, (mu, phi, sigma2, _), _, _ = choose_ar_order(idx, v, 5)
    assert order == 2
    assert phi == pytest.approx([1.0, -0.15], abs=0.01)
    assert sigma2 == pytest.approx(0.64, rel=0.02)


@pytest.mark.parametrize("h", [1, 12, 48])
def test_quadrature_matches_monte_carlo(h):
    rng = np.random.default_rng(2)
    v = rng.gamma(4.0, 1.7, 200_000)
    p = 2050 / (1 + np.exp(-(v - 8.5) / 1.4)) + rng.normal(0, 60, v.size)
    m, s2 = empirical_curve(v, np.clip(p, 0, 2050), 2050.0, 40, 20)
    ar = {"order": 2, "mu": 6.8, "phi": np.array([1.02, -0.08]), "sigma2": 0.75}
    hist = np.array([[9.3, 9.0, 8.7, 8.5, 8.4]])
    var_q = conditional_variance(ar, m, s2, hist, h)[0]
    mean_q, _ = conditional_mean(ar, m, hist, h)
    ah, var_u = ar_forecast_moments(ar["phi"], ar["sigma2"], h)
    mu_h = ar["mu"] + (hist[0, :2] - ar["mu"]) @ ah
    draws = rng.normal(mu_h, math.sqrt(var_u), 2_000_000)
    assert var_q == pytest.approx(np.var(m(draws)) + np.mean(s2(draws)), rel=5e-3)
    assert mean_q[0] == pytest.approx(np.mean(m(draws)), rel=2e-3)
