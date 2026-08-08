"""
Tests for the market risk measurement layer.

Property-based rather than golden-value: these assertions hold for ANY input,
so they test the contract rather than a frozen number, and they do not break
when a default changes.

Includes regression tests for two bugs found in the pre-refactor code:
  - MarketRiskConfig.fit_garch was accepted but never forwarded, so the MLE
    path was unreachable and the flag silently did nothing.
  - MarketRiskConfig had no Monte Carlo fields, so n_sims / seed / shrinkage
    were always the function defaults regardless of configuration.
"""

import numpy as np
import pytest

from risklib.market.market_risk_model import (
    METHODS,
    MarketRiskConfig,
    MarketRiskModel,
    cov_shrink,
    es_historical,
    es_parametric,
    var_historical,
    var_parametric,
)


# ---------------------------------------------------------------------------
# Loss-convention invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_loss_convention_holds_for_every_method(returns_3, weights_3, method):
    """VaR >= 0 and ES >= VaR, for all four methodologies."""
    cfg = MarketRiskConfig(alpha=0.99, method=method, exposure=1_000_000)
    m = MarketRiskModel(returns_3, weights_3, cfg).fit()
    assert m.compute_var() >= 0
    assert m.compute_es() >= m.compute_var()


@pytest.mark.parametrize("method", METHODS)
def test_var_monotonic_in_alpha(returns_3, weights_3, method):
    """A stricter confidence level cannot produce a smaller loss estimate."""
    v = []
    for a in (0.95, 0.975, 0.99):
        cfg = MarketRiskConfig(alpha=a, method=method, exposure=1_000_000)
        v.append(MarketRiskModel(returns_3, weights_3, cfg).fit().compute_var())
    assert v[0] <= v[1] <= v[2]


def test_var_scales_linearly_with_exposure(returns_3, weights_3):
    """VaR is homogeneous of degree 1 in exposure — catches scaling/sign errors."""
    base = var_parametric(returns_3, weights_3, alpha=0.99, exposure=1_000_000)
    doubled = var_parametric(returns_3, weights_3, alpha=0.99, exposure=2_000_000)
    assert doubled == pytest.approx(2.0 * base, rel=1e-12)


def test_es_strictly_exceeds_var_on_continuous_data(returns_3, weights_3):
    v = var_parametric(returns_3, weights_3, alpha=0.95, exposure=1e6)
    e = es_parametric(returns_3, weights_3, alpha=0.95, exposure=1e6)
    assert e > v

    vh = var_historical(returns_3, weights_3, alpha=0.95, exposure=1e6)
    eh = es_historical(returns_3, weights_3, alpha=0.95, exposure=1e6)
    assert eh > vh


def test_parametric_matches_closed_form(returns_3, weights_3):
    """Cross-check the estimator against the formula computed independently."""
    from statistics import NormalDist
    mu = returns_3.mean().values
    cov = returns_3.cov().values
    mu_p = float(weights_3 @ mu)
    sigma_p = float(np.sqrt(weights_3 @ cov @ weights_3))
    expected = (-mu_p + NormalDist().inv_cdf(0.99) * sigma_p) * 1e6
    assert var_parametric(returns_3, weights_3, alpha=0.99, exposure=1e6) == \
        pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        MarketRiskConfig(method="magic")


@pytest.mark.parametrize("kwargs", [
    {"alpha": 1.5}, {"alpha": 0.0}, {"horizon_days": 0},
    {"exposure": -1}, {"window": 1}, {"shrink_lambda": 2.0},
])
def test_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MarketRiskConfig(**kwargs)


def test_config_enforces_garch_stationarity():
    """alpha + beta >= 1 is a non-stationary GARCH and must be refused."""
    with pytest.raises(ValueError, match="stationarity"):
        MarketRiskConfig(alpha_g=0.10, beta_g=0.95)


def test_model_rejects_weight_length_mismatch(returns_3):
    with pytest.raises(ValueError, match="does not match"):
        MarketRiskModel(returns_3, np.array([0.5, 0.5]), MarketRiskConfig())


def test_compute_before_fit_raises(returns_3, weights_3):
    m = MarketRiskModel(returns_3, weights_3, MarketRiskConfig())
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        m.compute_var()


# ---------------------------------------------------------------------------
# Regression: config values must actually reach the computation
# ---------------------------------------------------------------------------

def test_fit_garch_flag_changes_the_result(returns_3, weights_3):
    """
    REGRESSION. `fit_garch=True` was accepted by the config but never forwarded
    to fhs_var_es_next, so the MLE path was dead code and the flag was inert
    while the documentation described it as a working feature.
    """
    fixed = MarketRiskModel(
        returns_3, weights_3,
        MarketRiskConfig(alpha=0.99, method="fhs", exposure=1e6, fit_garch=False),
    ).fit()
    mle = MarketRiskModel(
        returns_3, weights_3,
        MarketRiskConfig(alpha=0.99, method="fhs", exposure=1e6, fit_garch=True),
    ).fit()

    assert fixed.fit_info_["source"] == "fixed"
    assert mle.fit_info_["source"] == "MLE"
    assert mle.compute_var() != fixed.compute_var()


def test_assumptions_reflect_the_garch_mode(returns_3, weights_3):
    """An MLE-estimated filter must not be reported as fixed-parameter."""
    fixed = MarketRiskModel(
        returns_3, weights_3,
        MarketRiskConfig(method="fhs", fit_garch=False),
    ).fit()
    mle = MarketRiskModel(
        returns_3, weights_3,
        MarketRiskConfig(method="fhs", fit_garch=True),
    ).fit()

    assert any("fixed parameters" in a for a in fixed.assumptions())
    assert any("MLE" in a for a in mle.assumptions())


def test_monte_carlo_seed_is_honoured(returns_3, weights_3):
    """REGRESSION. Config seed was ignored; MC always ran with the default 42."""
    def run(seed):
        cfg = MarketRiskConfig(alpha=0.99, method="monte_carlo",
                               exposure=1e6, n_sims=20_000, seed=seed)
        return MarketRiskModel(returns_3, weights_3, cfg).fit().compute_var()

    assert run(1) != run(2)
    assert run(1) == run(1)          # and it is reproducible


def test_monte_carlo_n_sims_is_honoured(returns_3, weights_3):
    """REGRESSION. Config n_sims was ignored; MC always ran 100,000 paths."""
    def run(n):
        cfg = MarketRiskConfig(alpha=0.99, method="monte_carlo",
                               exposure=1e6, n_sims=n, seed=42)
        return MarketRiskModel(returns_3, weights_3, cfg).fit().compute_var()

    assert run(5_000) != run(80_000)


def test_monte_carlo_converges_to_parametric(returns_3, weights_3):
    """On normal data, MC with many paths should approach the closed form."""
    cfg = MarketRiskConfig(alpha=0.99, method="monte_carlo", exposure=1e6,
                           n_sims=400_000, seed=42, shrink_lambda=0.0)
    mc = MarketRiskModel(returns_3, weights_3, cfg).fit().compute_var()
    par = var_parametric(returns_3, weights_3, alpha=0.99, exposure=1e6)
    assert mc == pytest.approx(par, rel=0.03)


# ---------------------------------------------------------------------------
# Covariance shrinkage
# ---------------------------------------------------------------------------

def test_shrinkage_preserves_variances(correlated_returns):
    """Diagonal is untouched; only correlations are pulled in."""
    cov = correlated_returns.cov().values
    shrunk = cov_shrink(cov, lam=0.25)
    np.testing.assert_allclose(np.diag(cov), np.diag(shrunk), rtol=1e-12)
    off = ~np.eye(len(cov), dtype=bool)
    np.testing.assert_allclose(shrunk[off], 0.75 * cov[off], rtol=1e-12)


def test_shrinkage_keeps_matrix_positive_semidefinite(correlated_returns):
    cov = correlated_returns.cov().values
    assert np.linalg.eigvalsh(cov_shrink(cov, lam=0.10)).min() > 0


# ---------------------------------------------------------------------------
# Summary contract
# ---------------------------------------------------------------------------

def test_summary_carries_config_and_assumptions(returns_3, weights_3):
    """A risk number without its assumptions is not a deliverable."""
    cfg = MarketRiskConfig(alpha=0.975, method="parametric",
                           horizon_days=10, exposure=5e6)
    s = MarketRiskModel(returns_3, weights_3, cfg).fit().summary()

    assert s["VaR"] > 0 and s["ES"] >= s["VaR"]
    assert s["alpha"] == 0.975 and s["method"] == "parametric"
    assert s["config"]["horizon_days"] == 10
    assert "normality of portfolio returns" in s["assumptions"]
    assert any("square-root-of-time" in a for a in s["assumptions"])
