"""
Contract tests between app/app.py and risklib.

The app indexes into the dicts that risklib returns. If a key is renamed or
dropped, the app raises a KeyError at runtime — in front of the user, on a
deployed instance, with no test having caught it. These tests pin the shape of
every payload the UI reads, and walk the same call sequence the app performs
against the shipped demo data.

They are also the regression net for the import-shadowing accident in the
previous version, where the app imported `backtest_var_historical` from two
different modules and only got the three-test version because of line ordering.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklib.credit import compute_el_table, summarize_el
from risklib.data import load_prices, to_returns
from risklib.market import (
    DEFAULT_DURATIONS,
    MarketRiskConfig,
    MarketRiskModel,
    backtest_var_fhs,
    backtest_var_historical,
    erc_weights,
    incremental_var,
    scenario_corr_bump_mc,
    scenario_covariance_scale,
    scenario_equities_shock,
    scenario_historical_replay,
    scenario_rates_bp,
    scenario_single_name,
    var_parametric_normal_parts,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def market():
    returns = to_returns(load_prices(DATA / "market_data.csv"), method="log")
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    return returns, weights


# ---------------------------------------------------------------------------
# Demo data must actually load
# ---------------------------------------------------------------------------

def test_demo_market_data_loads(market):
    returns, _ = market
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert returns.index.is_monotonic_increasing
    assert len(returns) > 250
    assert returns.notna().all().all()


def test_demo_credit_data_loads():
    df = pd.read_csv(DATA / "credit_example.csv")
    out, seg = compute_el_table(df)
    assert seg is not None
    assert out["EL"].sum() > 0


# ---------------------------------------------------------------------------
# Keys the app reads from summary()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo", "fhs"])
def test_summary_exposes_keys_the_app_reads(market, method):
    returns, weights = market
    cfg = MarketRiskConfig(alpha=0.99, method=method, horizon_days=1, exposure=1e6,
                           window=250, n_sims=20_000, seed=42, shrink_lambda=0.01)
    s = MarketRiskModel(returns, weights, cfg).fit().summary()

    for key in ("VaR", "ES", "assumptions", "config"):
        assert key in s
    assert isinstance(s["assumptions"], list) and s["assumptions"]
    assert s["VaR"] > 0 and s["ES"] >= s["VaR"]


def test_fhs_summary_exposes_garch_fit_info(market):
    """The app renders omega / alpha / beta / persistence from fit_info."""
    returns, weights = market
    for flag in (False, True):
        cfg = MarketRiskConfig(method="fhs", alpha=0.99, exposure=1e6, fit_garch=flag)
        s = MarketRiskModel(returns, weights, cfg).fit().summary()
        fi = s["fit_info"]
        for key in ("source", "omega", "alpha_g", "beta_g", "persistence"):
            assert key in fi
        assert fi["source"] == ("MLE" if flag else "fixed")


# ---------------------------------------------------------------------------
# Keys the app reads from the backtests
# ---------------------------------------------------------------------------

APP_BACKTEST_KEYS = (
    "r_p", "VaR_threshold", "exceptions", "window", "alpha", "T", "exceedances",
    "hit_rate", "kupiec_LR", "kupiec_pvalue", "christoffersen_LR",
    "christoffersen_pvalue", "joint_LR", "joint_pvalue", "transitions", "method",
)


def test_historical_backtest_payload(market):
    returns, weights = market
    bt = backtest_var_historical(returns, weights, alpha=0.99, window=250)
    for key in APP_BACKTEST_KEYS:
        assert key in bt, f"app reads bt[{key!r}] but it is missing"
    for key in ("n00", "n01", "n10", "n11", "pi_01", "pi_11"):
        assert key in bt["transitions"]


def test_fhs_backtest_payload_matches_historical(market):
    """
    Both backtests must expose the SAME keys — the app renders them through one
    code path and switches only the source.
    """
    returns, weights = market
    bt = backtest_var_fhs(returns, weights, alpha=0.99, window=250)
    for key in APP_BACKTEST_KEYS:
        assert key in bt, f"app reads bt[{key!r}] but it is missing"


def test_backtest_series_align_for_plotting(market):
    """The chart zips these three series together; they must share an index."""
    returns, weights = market
    for bt in (backtest_var_historical(returns, weights, alpha=0.99, window=250),
               backtest_var_fhs(returns, weights, alpha=0.99, window=250)):
        assert bt["r_p"].index.equals(bt["VaR_threshold"].index)
        assert bt["r_p"].index.equals(bt["exceptions"].index)


# ---------------------------------------------------------------------------
# Attribution payloads
# ---------------------------------------------------------------------------

def test_decomposition_payload(market):
    returns, weights = market
    parts = var_parametric_normal_parts(returns, weights, alpha=0.99,
                                        horizon_days=1, exposure=1e6)
    for key in ("VaR", "mu_p", "sigma_p", "mVaR", "cVaR", "pContrib", "w"):
        assert key in parts
    n = returns.shape[1]
    for key in ("mVaR", "cVaR", "pContrib", "w"):
        assert len(parts[key]) == n


def test_incremental_payload(market):
    returns, weights = market
    inc = incremental_var(returns, weights, alpha=0.99, horizon_days=1, exposure=1e6)
    for key in ("VaR", "iVaR", "cVaR"):
        assert key in inc
    assert len(inc["iVaR"]) == returns.shape[1]


def test_erc_payload(market):
    returns, weights = market
    w_erc, info = erc_weights(returns, horizon_days=1, init=weights, step=0.5)
    for key in ("iter", "rc_dispersion", "converged"):
        assert key in info
    assert w_erc.sum() == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Scenario payloads
# ---------------------------------------------------------------------------

def test_single_name_scenario_payload(market):
    returns, weights = market
    res = scenario_single_name(returns, weights, {returns.columns[0]: -0.15}, exposure=1e6)
    for key in ("loss", "shock_vector", "base"):
        assert key in res
    assert len(res["shock_vector"]) == returns.shape[1]


def test_rate_scenario_payload(market):
    returns, weights = market
    durations = {c: DEFAULT_DURATIONS.get(c, 0.0) for c in returns.columns}
    res = scenario_rates_bp(returns, weights, durations=durations, bp=200.0, exposure=1e6)
    for key in ("loss", "durations_used", "bp"):
        assert key in res
    assert set(res["durations_used"]) == set(returns.columns)


def test_covariance_and_correlation_scenario_payloads(market):
    returns, weights = market
    cov = scenario_covariance_scale(returns, weights, scale=2.0, alpha=0.99,
                                    exposure=1e6, n_sims=20_000, seed=42)
    corr = scenario_corr_bump_mc(returns, weights, alpha=0.99, exposure=1e6,
                                 corr_bump_pct=50.0, n_sims=20_000, seed=42)
    for res in (cov, corr):
        assert res["base"]["VaR"] > 0 and res["stressed"]["VaR"] > 0
        assert "delta_VaR" in res
    assert "vol_multiplier" in cov
    assert "psd_adjusted" in corr


def test_historical_replay_payload(market):
    returns, weights = market
    res = scenario_historical_replay(
        returns, weights,
        str(returns.index.min().date()), str(returns.index.max().date()),
        alpha=0.99, exposure=1e6, n_sims=20_000, seed=42)
    for key in ("VaR", "ES", "n_obs"):
        assert key in res


def test_equity_scenario_returns_scalar(market):
    returns, weights = market
    loss = scenario_equities_shock(returns, weights, list(returns.columns),
                                   shock=-0.20, exposure=1e6)
    assert isinstance(loss, float)
    assert loss == pytest.approx(0.20 * 1e6, rel=1e-9)


# ---------------------------------------------------------------------------
# Credit payloads
# ---------------------------------------------------------------------------

def test_credit_summary_payload():
    df = pd.read_csv(DATA / "credit_example.csv")
    out, seg = compute_el_table(df)
    grp, totals = summarize_el(out, seg)

    for key in ("total_EAD", "total_EL", "EL_pct_of_EAD", "facilities"):
        assert key in totals.index
    for col in ("total_EAD", "total_EL", "avg_PD", "avg_LGD", "EL_pct_of_EAD"):
        assert col in grp.columns
    assert "data_quality" in out.attrs


# ---------------------------------------------------------------------------
# Full app call sequence
# ---------------------------------------------------------------------------

def test_full_app_sequence_runs_without_error(market):
    """Walk the app's happy path end to end for every method."""
    returns, weights = market

    for method in ("historical", "parametric", "monte_carlo", "fhs"):
        cfg = MarketRiskConfig(alpha=0.99, method=method, horizon_days=1, exposure=1e6,
                               window=250, n_sims=20_000, seed=42)
        summary = MarketRiskModel(returns, weights, cfg).fit().summary()

        bt = backtest_var_historical(returns, weights, alpha=0.99, window=250)
        parts = var_parametric_normal_parts(returns, weights, alpha=0.99, exposure=1e6)

        # The report builder reads exactly these.
        assert summary["VaR"] > 0
        assert bt["T"] > 0
        assert parts["cVaR"].sum() == pytest.approx(parts["VaR"], rel=1e-9)
