"""Tests for data ingestion and the scenario library."""

import io

import numpy as np
import pandas as pd
import pytest

from risklib.data import clean_prices, load_prices, to_returns
from risklib.market.scenarios import (
    scenario_corr_bump_mc,
    scenario_covariance_scale,
    scenario_equities_shock,
    scenario_rates_bp,
    scenario_single_name,
    shock_vector,
)

CSV = """Date,SPY,TLT
2024-01-02,470.10,95.20
2024-01-03,468.50,95.80
2024-01-04,471.20,95.10
2024-01-05,473.00,94.70
"""


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def test_load_prices_parses_dates_and_numerics():
    df = load_prices(io.StringIO(CSV))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["SPY", "TLT"]
    assert df.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()


def test_load_prices_sorts_by_date():
    """A newest-first export would otherwise invert every rolling window."""
    reversed_csv = "Date,SPY\n2024-01-05,473.0\n2024-01-02,470.1\n"
    df = load_prices(io.StringIO(reversed_csv))
    assert df.index.is_monotonic_increasing


def test_load_prices_strips_currency_formatting():
    """Excel exports carry '$1,234.56' as object dtype, which poisons .cov()."""
    df = load_prices(io.StringIO('Date,SPY\n2024-01-02,"$1,234.56"\n2024-01-03,"$1,240.00"\n'))
    assert df["SPY"].iloc[0] == pytest.approx(1234.56)


def test_load_prices_drops_all_nan_columns_without_error():
    """
    REGRESSION. The all-NaN drop previously sat INSIDE the column loop, mutating
    the frame while iterating its own stale column index.
    """
    csv = "Date,SPY,JUNK,TLT\n2024-01-02,470.1,,95.2\n2024-01-03,468.5,,95.8\n"
    df = load_prices(io.StringIO(csv))
    assert list(df.columns) == ["SPY", "TLT"]


def test_load_prices_rejects_unparseable_dates():
    with pytest.raises(ValueError, match="No parseable dates"):
        load_prices(io.StringIO("Date,SPY\nnot-a-date,470\nalso-bad,471\n"))


def test_clean_prices_removes_duplicate_dates():
    idx = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx)
    cleaned = clean_prices(prices)
    assert len(cleaned) == 2
    assert cleaned["A"].iloc[0] == 100.0     # first occurrence kept


def test_clean_prices_nulls_non_positive_prices():
    """A zero price would emit -inf into log returns and NaN the covariance matrix."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"A": [100.0, 0.0, 102.0]}, index=idx)
    cleaned = clean_prices(prices)
    assert cleaned["A"].isna().iloc[1]
    assert cleaned["A"].notna().sum() == 2


def test_clean_prices_drops_columns_with_too_few_observations():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"GOOD": [100.0, 101.0, 102.0],
                           "SPARSE": [100.0, -5.0, np.nan]}, index=idx)
    assert list(clean_prices(prices).columns) == ["GOOD"]


def test_log_returns_are_time_additive():
    df = load_prices(io.StringIO(CSV))
    r = to_returns(df, "log")
    total = np.log(df["SPY"].iloc[-1] / df["SPY"].iloc[0])
    assert r["SPY"].sum() == pytest.approx(total, rel=1e-12)


def test_returns_drop_only_the_first_row():
    df = load_prices(io.StringIO(CSV))
    assert len(to_returns(df, "log")) == len(df) - 1


def test_returns_reject_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        to_returns(load_prices(io.StringIO(CSV)), method="geometric")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@pytest.fixture
def rets():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=600, freq="B")
    return pd.DataFrame(rng.normal(0, 0.01, (600, 3)), index=idx,
                        columns=["SPY", "QQQ", "TLT"])


def test_shock_vector_holds_unshocked_instruments_flat(rets):
    """
    REGRESSION. The two former scenario modules disagreed: one held unshocked
    names flat, the other carried their last observed return. Flat is now the
    single convention.
    """
    v = shock_vector(rets, {"SPY": -0.10})
    assert v["SPY"] == -0.10
    assert v["QQQ"] == 0.0 and v["TLT"] == 0.0


def test_shock_vector_last_mode_is_opt_in(rets):
    v = shock_vector(rets, {"SPY": -0.10}, base="last")
    assert v["QQQ"] == pytest.approx(rets["QQQ"].iloc[-1])


def test_shock_vector_rejects_unknown_ticker(rets):
    with pytest.raises(KeyError):
        shock_vector(rets, {"NVDA": -0.10})


def test_single_name_shock_loss_is_weight_times_shock(rets):
    w = np.array([0.5, 0.3, 0.2])
    res = scenario_single_name(rets, w, {"SPY": -0.20}, exposure=1_000_000)
    assert res["loss"] == pytest.approx(0.20 * 0.5 * 1_000_000, rel=1e-12)


def test_equities_shock_leaves_bonds_untouched(rets):
    w = np.array([0.4, 0.4, 0.2])
    loss = scenario_equities_shock(rets, w, ["SPY", "QQQ"], shock=-0.20, exposure=1e6)
    assert loss == pytest.approx(0.20 * 0.8 * 1e6, rel=1e-12)


def test_rate_shock_does_not_hit_equities(rets):
    """
    REGRESSION. default_duration was 7.0 applied to EVERY column, so a +200bp
    scenario knocked ~14% off SPY. Only instruments with a duration should move.
    """
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    res = scenario_rates_bp(rets, w, bp=200.0, exposure=1e6)
    assert res["durations_used"]["SPY"] == 0.0
    assert res["durations_used"]["QQQ"] == 0.0
    assert res["durations_used"]["TLT"] == 18.0

    # Loss comes from TLT alone: 18 * 0.02 * (1/3) * 1e6
    assert res["loss"] == pytest.approx(18.0 * 0.02 * (1 / 3) * 1e6, rel=1e-9)


def test_rate_shock_honours_explicit_duration_override(rets):
    w = np.array([0.0, 0.0, 1.0])
    res = scenario_rates_bp(rets, w, durations={"TLT": 10.0}, bp=100.0, exposure=1e6)
    assert res["loss"] == pytest.approx(10.0 * 0.01 * 1e6, rel=1e-9)


def test_covariance_scaling_increases_var(rets):
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    res = scenario_covariance_scale(rets, w, scale=4.0, alpha=0.99,
                                    exposure=1e6, n_sims=40_000, seed=1)
    assert res["stressed"]["VaR"] > res["base"]["VaR"]
    assert res["vol_multiplier"] == pytest.approx(2.0)


def test_correlation_bump_preserves_psd(rets):
    """
    REGRESSION. Element-wise correlation bumping can push the matrix out of the
    PSD cone and break the Cholesky factorisation. The projection must handle it.
    """
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    res = scenario_corr_bump_mc(rets, w, alpha=0.99, exposure=1e6,
                                corr_bump_pct=500.0, n_sims=20_000, seed=1)
    assert np.isfinite(res["stressed"]["VaR"])
    assert res["psd_adjusted"] in (True, False)


def test_correlation_bump_uses_same_seed_for_base_and_stress(rets):
    """The difference must be a covariance effect, not Monte Carlo noise."""
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    a = scenario_corr_bump_mc(rets, w, corr_bump_pct=0.0, n_sims=20_000, seed=5)
    assert a["base"]["VaR"] == pytest.approx(a["stressed"]["VaR"], rel=1e-9)
