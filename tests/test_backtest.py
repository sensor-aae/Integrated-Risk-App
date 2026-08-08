"""
Tests for the validation layer.

This file was previously empty — the backtesting module, which is the
centrepiece of the project, had no coverage at all.

The most important test here is `test_christoffersen_rejects_clustering`: it
constructs an exception series with the correct TOTAL count but deliberately
clustered in time, confirms Kupiec cannot see the problem, and confirms
Christoffersen does. That contrast is the entire argument for implementing the
independence test.
"""

import numpy as np
import pandas as pd
import pytest

from risklib.market.backtest import (
    backtest_var_fhs,
    backtest_var_historical,
    christoffersen_independence,
    joint_coverage_test,
    kupiec_pof,
)


# ---------------------------------------------------------------------------
# Kupiec
# ---------------------------------------------------------------------------

def test_kupiec_perfect_coverage_gives_zero_statistic():
    """Observed rate exactly equal to (1-alpha) => LR = 0, p = 1."""
    LR, p = kupiec_pof(exceedances=50, T=1000, alpha=0.95)
    assert LR == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_kupiec_rejects_gross_overshoot():
    """15% exceptions against a 5% target must be rejected decisively."""
    LR, p = kupiec_pof(exceedances=150, T=1000, alpha=0.95)
    assert LR > 10.0
    assert p < 0.01


def test_kupiec_rejects_too_few_exceptions():
    """An over-conservative model (0.5% vs 5% target) is also mis-specified."""
    _, p = kupiec_pof(exceedances=5, T=1000, alpha=0.95)
    assert p < 0.01


def test_kupiec_handles_zero_exceptions():
    """Zero exceptions must not produce log(0); the clamp handles it."""
    LR, p = kupiec_pof(exceedances=0, T=500, alpha=0.95)
    assert np.isfinite(LR) and np.isfinite(p)
    assert LR > 0


def test_kupiec_requires_positive_sample():
    with pytest.raises(ValueError):
        kupiec_pof(exceedances=0, T=0, alpha=0.95)


# ---------------------------------------------------------------------------
# Christoffersen independence — the core test
# ---------------------------------------------------------------------------

def test_christoffersen_accepts_independent_exceptions():
    """Randomly scattered exceptions should not be flagged as clustered."""
    rng = np.random.default_rng(42)
    exc = pd.Series(rng.binomial(1, 0.05, size=2000))
    LR, p, tr = christoffersen_independence(exc)
    assert p > 0.05
    assert tr["n00"] + tr["n01"] + tr["n10"] + tr["n11"] == len(exc) - 1


def test_christoffersen_rejects_clustering():
    """
    The headline test. Build a series with the CORRECT total exception count
    but all exceptions bunched together. Kupiec sees nothing wrong; the
    independence test must catch it.
    """
    T, n_exc = 1000, 50
    clustered = np.zeros(T, dtype=int)
    clustered[300:300 + n_exc] = 1          # one contiguous block
    exc = pd.Series(clustered)

    # Kupiec is blind: the count is exactly on target.
    _, p_kupiec = kupiec_pof(n_exc, T, alpha=0.95)
    assert p_kupiec > 0.9, "count is on target, so Kupiec must pass"

    # Christoffersen sees it.
    LR_ind, p_ind, tr = christoffersen_independence(exc)
    assert p_ind < 0.001, "clustered exceptions must be rejected"
    assert tr["pi_11"] > tr["pi_01"], "clustering means P(exc|exc) > P(exc|no exc)"


def test_christoffersen_statistic_is_non_negative():
    """LR is non-negative in theory; the guard must hold in degenerate cases."""
    for series in (np.zeros(100, dtype=int),
                   np.ones(100, dtype=int),
                   np.array([1] + [0] * 99)):
        LR, p, _ = christoffersen_independence(pd.Series(series))
        assert LR >= 0.0
        assert 0.0 <= p <= 1.0


def test_christoffersen_handles_degenerate_input():
    """Empty and single-element series must return neutral results, not raise."""
    for s in (pd.Series(dtype=int), pd.Series([1])):
        LR, p, tr = christoffersen_independence(s)
        assert LR == 0.0 and p == 1.0
        assert tr["n11"] == 0


# ---------------------------------------------------------------------------
# Joint conditional coverage
# ---------------------------------------------------------------------------

def test_joint_statistic_is_sum_of_components():
    """LR_cc = LR_uc + LR_ind by construction."""
    rng = np.random.default_rng(3)
    exc = pd.Series(rng.binomial(1, 0.05, size=1000))
    res = joint_coverage_test(int(exc.sum()), len(exc), 0.95, exc)
    assert res["joint_LR"] == pytest.approx(
        res["kupiec_LR"] + res["christoffersen_LR"], rel=1e-12
    )


def test_joint_pvalue_uses_chi2_two_dof():
    """For Chi^2(2) the survival function is exactly exp(-x/2)."""
    exc = pd.Series(np.zeros(100, dtype=int))
    res = joint_coverage_test(0, 100, 0.95, exc)
    assert res["joint_pvalue"] == pytest.approx(np.exp(-res["joint_LR"] / 2.0), rel=1e-12)


def test_joint_test_catches_clustering_that_kupiec_misses():
    """End-to-end: correct count, clustered timing => joint test rejects."""
    T, n_exc = 1000, 50
    clustered = np.zeros(T, dtype=int)
    clustered[100:100 + n_exc] = 1
    exc = pd.Series(clustered)

    res = joint_coverage_test(n_exc, T, 0.95, exc)
    assert res["kupiec_pvalue"] > 0.9
    assert res["joint_pvalue"] < 0.01


# ---------------------------------------------------------------------------
# Rolling backtests — out-of-sample discipline
# ---------------------------------------------------------------------------

def test_historical_backtest_has_no_lookahead(returns_3, weights_3):
    """
    The threshold at t must depend only on data through t-1. Verified directly:
    recompute the quantile from the trailing window ending at t-1 and compare.
    """
    window = 100
    bt = backtest_var_historical(returns_3, weights_3, alpha=0.95, window=window)

    r_p = bt["r_p"]
    t = window + 25
    expected = r_p.iloc[t - window:t].quantile(0.05)
    assert bt["VaR_threshold"].iloc[t] == pytest.approx(expected, rel=1e-12)


def test_historical_backtest_burn_in_excluded(returns_3, weights_3):
    """T must count only days with a defined threshold."""
    window = 250
    bt = backtest_var_historical(returns_3, weights_3, alpha=0.95, window=window)
    assert bt["T"] == len(returns_3) - window
    assert bt["VaR_threshold"].isna().sum() == window


def test_backtest_exception_count_matches_series(returns_3, weights_3):
    """The reported count must equal the flags actually raised in-sample."""
    bt = backtest_var_historical(returns_3, weights_3, alpha=0.95, window=100)
    mask = bt["VaR_threshold"].notna()
    manual = int((bt["r_p"][mask] < bt["VaR_threshold"][mask]).sum())
    assert bt["exceedances"] == manual


def test_backtest_hit_rate_near_target_on_iid_data(returns_3, weights_3):
    """On clean iid normal data a 95% VaR should exceed roughly 5% of the time."""
    bt = backtest_var_historical(returns_3, weights_3, alpha=0.95, window=250)
    assert 0.02 < bt["hit_rate"] < 0.09
    assert bt["kupiec_pvalue"] > 0.01


def test_backtest_higher_alpha_yields_fewer_exceptions(returns_3, weights_3):
    """Monotonicity: a stricter confidence level cannot be breached more often."""
    bt95 = backtest_var_historical(returns_3, weights_3, alpha=0.95, window=250)
    bt99 = backtest_var_historical(returns_3, weights_3, alpha=0.99, window=250)
    assert bt99["exceedances"] <= bt95["exceedances"]


def test_backtest_returns_all_three_tests(returns_3, weights_3):
    """The dict contract the app relies on."""
    bt = backtest_var_historical(returns_3, weights_3, alpha=0.95, window=250)
    for key in ("kupiec_LR", "kupiec_pvalue",
                "christoffersen_LR", "christoffersen_pvalue",
                "joint_LR", "joint_pvalue", "transitions",
                "r_p", "VaR_threshold", "exceptions", "T", "exceedances", "hit_rate"):
        assert key in bt, f"missing key: {key}"


# ---------------------------------------------------------------------------
# FHS backtest
# ---------------------------------------------------------------------------

def test_fhs_backtest_runs_and_is_calibrated(returns_3, weights_3):
    bt = backtest_var_fhs(returns_3, weights_3, alpha=0.95, window=250)
    assert bt["T"] > 0
    assert 0.01 < bt["hit_rate"] < 0.12


def test_fhs_backtest_sigma_is_causal(returns_3, weights_3):
    """
    sigma_t is the forecast formed at t-1, so truncating the sample after t must
    not change it. This is the regression test for the in-sample long-run
    variance that the previous implementation used.
    """
    window = 250
    full = backtest_var_fhs(returns_3, weights_3, alpha=0.95, window=window)
    cut = 600
    truncated = backtest_var_fhs(returns_3.iloc[:cut], weights_3, alpha=0.95, window=window)

    common = truncated["sigma"].index
    np.testing.assert_allclose(
        full["sigma"].loc[common].values,
        truncated["sigma"].values,
        rtol=1e-10,
        err_msg="sigma changed when future data was removed => look-ahead leakage",
    )


def test_fhs_backtest_short_sample_returns_empty_result(returns_3, weights_3):
    bt = backtest_var_fhs(returns_3.iloc[:100], weights_3, alpha=0.99, window=250)
    assert bt["T"] == 0
    assert bt["exceedances"] == 0
