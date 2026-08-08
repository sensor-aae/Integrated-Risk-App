"""
risklib.market
==============
Market risk measurement, validation, attribution and scenario analysis.

Layout
------
  market_risk_model : VaR/ES estimators, MarketRiskConfig, MarketRiskModel
  backtest          : Kupiec / Christoffersen / joint CC tests, rolling backtests
  garch             : the single GARCH(1,1) variance recursion (fixed + MLE)
  extras            : Euler decomposition, incremental VaR, ERC budgeting
  scenarios         : stress testing and the deterministic scenario library
"""

from .market_risk_model import (
    METHODS,
    MarketRiskConfig,
    MarketRiskModel,
    cov_shrink,
    es_historical,
    es_parametric,
    fhs_var_es_next,
    mc_portfolio_loss_from_mu_cov,
    normalize_weights,
    portfolio_returns,
    var_es_monte_carlo,
    var_historical,
    var_parametric,
)
from .backtest import (
    backtest_var_fhs,
    backtest_var_historical,
    christoffersen_independence,
    joint_coverage_test,
    kupiec_pof,
)
from .garch import (
    fit_garch11_mle,
    garch11_filter,
    garch11_forecast_next,
    variance_path,
)
from .extras import (
    erc_weights,
    erc_weights_from_cov,
    incremental_var,
    var_parametric_normal_parts,
)
from .scenarios import (
    DEFAULT_DURATIONS,
    historical_window_mu_cov,
    scale_covariance,
    scenario_corr_bump_mc,
    scenario_covariance_scale,
    scenario_equities_shock,
    scenario_historical_replay,
    scenario_rates_bp,
    scenario_single_name,
    shock_vector,
)

__all__ = [
    # measurement
    "MarketRiskConfig", "MarketRiskModel", "METHODS",
    "var_parametric", "es_parametric", "var_historical", "es_historical",
    "var_es_monte_carlo", "mc_portfolio_loss_from_mu_cov", "fhs_var_es_next",
    "portfolio_returns", "normalize_weights", "cov_shrink",
    # validation
    "kupiec_pof", "christoffersen_independence", "joint_coverage_test",
    "backtest_var_historical", "backtest_var_fhs",
    # volatility
    "garch11_filter", "garch11_forecast_next", "fit_garch11_mle", "variance_path",
    # attribution
    "var_parametric_normal_parts", "incremental_var",
    "erc_weights", "erc_weights_from_cov",
    # scenarios
    "shock_vector", "scenario_single_name", "scenario_equities_shock",
    "scenario_rates_bp", "scale_covariance", "scenario_covariance_scale",
    "scenario_corr_bump_mc", "historical_window_mu_cov",
    "scenario_historical_replay", "DEFAULT_DURATIONS",
]
