"""
Tests for the GARCH module.

The README claims the MLE estimator recovers known parameters better than the
fixed defaults. That claim is now enforced here rather than only asserted in
prose.
"""

import numpy as np
import pandas as pd
import pytest

from risklib.market.garch import (
    fit_garch11_mle,
    garch11_filter,
    garch11_forecast_next,
    variance_path,
)


def simulate_garch(n=3000, omega=2e-6, alpha=0.08, beta=0.91, seed=0):
    """Simulate a GARCH(1,1) series with known parameters."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    sig2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(sig2) * rng.normal()
        sig2 = omega + alpha * r[t] ** 2 + beta * sig2
    return pd.Series(r, index=pd.date_range("2015-01-01", periods=n, freq="B"))


def test_variance_path_matches_manual_recursion():
    r = np.array([0.01, -0.02, 0.015, -0.005])
    omega, a, b = 1e-6, 0.05, 0.94
    sig2 = variance_path(r, omega, a, b, init_var=1e-4)

    expected = 1e-4
    assert sig2[0] == pytest.approx(expected)
    for t in range(1, len(r)):
        expected = omega + a * r[t - 1] ** 2 + b * expected
        assert sig2[t] == pytest.approx(expected, rel=1e-12)


def test_variance_path_stays_positive():
    """The floor must hold even for absurd parameter proposals."""
    r = np.zeros(50)
    assert (variance_path(r, 0.0, 0.0, 0.0, init_var=0.0) > 0).all()


def test_mle_recovers_known_parameters():
    """Estimates should land near the true values on simulated data."""
    r = simulate_garch(n=3000, omega=2e-6, alpha=0.08, beta=0.91, seed=1)
    fit = fit_garch11_mle(r, n_restarts=5)

    assert fit["converged"]
    assert fit["alpha_g"] == pytest.approx(0.08, abs=0.04)
    assert fit["beta_g"] == pytest.approx(0.91, abs=0.06)
    assert fit["persistence"] < 1.0


def test_mle_beats_fixed_defaults_on_alpha():
    """
    The claim in the model documentation: MLE estimates alpha more accurately
    than the fixed 0.05 default when the true value is 0.08.
    """
    true_alpha = 0.08
    r = simulate_garch(n=3000, alpha=true_alpha, beta=0.91, seed=2)
    fit = fit_garch11_mle(r, n_restarts=5)

    err_mle = abs(fit["alpha_g"] - true_alpha) / true_alpha
    err_fixed = abs(0.05 - true_alpha) / true_alpha
    assert err_mle < err_fixed


def test_mle_enforces_stationarity():
    """The parameter transform makes alpha + beta < 1 structurally impossible to break."""
    for seed in range(4):
        fit = fit_garch11_mle(simulate_garch(n=1200, seed=seed), n_restarts=3)
        assert 0 < fit["alpha_g"] < 1
        assert 0 <= fit["beta_g"] < 1
        assert fit["alpha_g"] + fit["beta_g"] < 1.0
        assert fit["omega"] > 0


def test_mle_rejects_short_series():
    with pytest.raises(ValueError, match="observations"):
        fit_garch11_mle(pd.Series(np.random.default_rng(0).normal(0, 0.01, 30)))


def test_loglikelihood_includes_normalising_constant():
    """
    REGRESSION. The 2*pi constant was previously dropped, leaving the reported
    log-likelihood, AIC and BIC offset by 0.5*n*log(2*pi) versus any external
    package. Verified against a direct Gaussian evaluation.
    """
    r = simulate_garch(n=800, seed=5)
    fit = fit_garch11_mle(r, n_restarts=3)

    sig2 = variance_path(r.values, fit["omega"], fit["alpha_g"], fit["beta_g"])
    manual = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sig2) + r.values ** 2 / sig2)
    assert fit["log_likelihood"] == pytest.approx(manual, rel=1e-8)

    k = 3
    assert fit["aic"] == pytest.approx(2 * k - 2 * manual, rel=1e-8)


def test_filter_fixed_mode_uses_variance_targeting():
    """omega = (1 - a - b) * sample variance, and the source is tagged 'fixed'."""
    r = simulate_garch(n=600, seed=3)
    sigma, info = garch11_filter(r, fit=False, alpha_g=0.05, beta_g=0.94)

    assert info["source"] == "fixed"
    assert info["omega"] == pytest.approx((1 - 0.05 - 0.94) * r.var(ddof=1), rel=1e-9)
    assert len(sigma) == len(r)
    assert (sigma > 0).all()


def test_filter_mle_mode_is_tagged_and_differs():
    """The source tag is what makes 'estimated vs assumed' auditable downstream."""
    r = simulate_garch(n=1200, seed=4)
    s_fixed, i_fixed = garch11_filter(r, fit=False)
    s_mle, i_mle = garch11_filter(r, fit=True)

    assert i_fixed["source"] == "fixed"
    assert i_mle["source"] == "MLE"
    assert not np.allclose(s_fixed.values, s_mle.values)


def test_filter_falls_back_rather_than_raising_on_short_series():
    """A stress panel should not vanish because an optimiser had a bad day."""
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 20))
    sigma, info = garch11_filter(r, fit=True)
    assert info["source"] == "fixed"
    assert "fallback_reason" in info
    assert len(sigma) == len(r)


def test_forecast_matches_recursion():
    assert garch11_forecast_next(0.02, 0.01, 1e-6, 0.05, 0.94) == pytest.approx(
        np.sqrt(1e-6 + 0.05 * 0.02 ** 2 + 0.94 * 0.01 ** 2), rel=1e-12
    )


def test_filter_responds_to_volatility_clustering():
    """A calm period followed by a shock must raise the conditional volatility."""
    r = pd.Series(np.concatenate([
        np.random.default_rng(0).normal(0, 0.005, 300),
        np.random.default_rng(1).normal(0, 0.04, 100),
    ]))
    sigma, _ = garch11_filter(r, fit=False)
    assert sigma.iloc[-1] > 3 * sigma.iloc[290]
