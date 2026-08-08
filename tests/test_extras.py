"""
Tests for attribution and risk budgeting.

The Euler summation identity (component VaR summing exactly to portfolio VaR)
is claimed in the README; this file is where that claim is actually enforced.
"""

import numpy as np
import pytest

from risklib.market.extras import (
    erc_weights,
    erc_weights_from_cov,
    incremental_var,
    var_parametric_normal_parts,
)
from risklib.market.market_risk_model import var_parametric


def test_component_var_sums_to_portfolio_var(correlated_returns):
    """
    Euler's theorem: VaR is homogeneous of degree 1 in w, so
    sum_i w_i * dVaR/dw_i = VaR exactly — not approximately.
    """
    w = np.array([0.25, 0.25, 0.50])
    parts = var_parametric_normal_parts(correlated_returns, w, alpha=0.95, exposure=1e6)
    assert parts["cVaR"].sum() == pytest.approx(parts["VaR"], rel=1e-10)


def test_percentage_contributions_sum_to_one(correlated_returns):
    w = np.array([0.4, 0.4, 0.2])
    parts = var_parametric_normal_parts(correlated_returns, w, alpha=0.99, exposure=1e6)
    assert parts["pContrib"].sum() == pytest.approx(1.0, rel=1e-10)


def test_decomposition_var_matches_standalone_estimator(correlated_returns):
    """The decomposition must not disagree with the plain estimator."""
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    parts = var_parametric_normal_parts(correlated_returns, w, alpha=0.99,
                                        horizon_days=1, exposure=1e6)
    direct = var_parametric(correlated_returns, w, alpha=0.99,
                            horizon_days=1, exposure=1e6)
    assert parts["VaR"] == pytest.approx(direct, rel=1e-10)


def test_riskier_asset_contributes_more_than_its_weight(correlated_returns):
    """TECH has the highest variance, so equal weighting must over-contribute."""
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    parts = var_parametric_normal_parts(correlated_returns, w, alpha=0.95, exposure=1e6)
    tech = list(correlated_returns.columns).index("TECH")
    bond = list(correlated_returns.columns).index("BOND")
    assert parts["pContrib"][tech] > 1 / 3
    assert parts["pContrib"][bond] < 1 / 3


def test_incremental_var_is_a_true_recomputation(correlated_returns):
    """
    REGRESSION. The old `incremental_var_normal` returned component VaR under
    the name incremental VaR. True iVaR removes the position and renormalises,
    so on a material weight the two must differ.
    """
    w = np.array([0.25, 0.25, 0.50])
    res = incremental_var(correlated_returns, w, alpha=0.95, exposure=1e6)
    assert not np.allclose(res["iVaR"], res["cVaR"], rtol=1e-3)


def test_incremental_var_matches_manual_removal(correlated_returns):
    """iVaR_i = VaR(full) - VaR(without i, renormalised)."""
    w = np.array([0.25, 0.25, 0.50])
    res = incremental_var(correlated_returns, w, alpha=0.95, exposure=1e6)

    w_ex = np.array([0.0, 0.25, 0.50])
    w_ex = w_ex / w_ex.sum()
    manual = res["VaR"] - var_parametric(correlated_returns, w_ex, alpha=0.95, exposure=1e6)
    assert res["iVaR"][0] == pytest.approx(manual, rel=1e-10)


def test_erc_equalises_risk_contributions(correlated_returns):
    """The objective itself: risk contributions to variance become equal."""
    w_erc, info = erc_weights(correlated_returns, tol=1e-10, max_iter=20_000)
    cov = correlated_returns.cov().values
    rc = w_erc * (cov @ w_erc)
    assert info["converged"]
    assert rc.std() / rc.mean() < 1e-4
    assert w_erc.sum() == pytest.approx(1.0, rel=1e-12)


def test_erc_underweights_the_riskiest_asset(correlated_returns):
    """Equalising risk means the high-vol asset gets a smaller weight."""
    w_erc, _ = erc_weights(correlated_returns, tol=1e-10)
    cols = list(correlated_returns.columns)
    assert w_erc[cols.index("TECH")] < w_erc[cols.index("BOND")]


def test_erc_respects_weight_bounds(correlated_returns):
    w_erc, _ = erc_weights(correlated_returns, min_w=0.2, max_w=0.5, tol=1e-10)
    assert w_erc.min() >= 0.2 - 1e-9
    assert w_erc.max() <= 0.5 + 1e-9


def test_erc_info_well_defined_with_zero_iterations():
    """REGRESSION: info referenced loop variables that could be unbound."""
    cov = np.diag([0.01, 0.02, 0.03])
    w, info = erc_weights_from_cov(cov, max_iter=0)
    assert info["iter"] == 0
    assert info["converged"] is False
    assert np.isfinite(info["sigma"])
    assert len(info["RC"]) == 3


def test_erc_on_identity_covariance_is_equal_weight():
    """Uncorrelated, equal-variance assets => equal weights."""
    w, _ = erc_weights_from_cov(np.eye(4) * 0.01, tol=1e-12)
    np.testing.assert_allclose(w, np.ones(4) / 4, atol=1e-6)
