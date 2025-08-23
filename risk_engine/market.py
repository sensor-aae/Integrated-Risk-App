from __future__ import annotations
import numpy as np
import pandas as pd
from statistics import NormalDist
from math import sqrt, erfc
from typing import Tuple

# -------- Core helpers --------
def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Project multivariate return matrix onto portfolio weights."""
    return pd.Series(returns.values @ weights, index=returns.index, name="r_p")

# Convenience for Normal
_N = NormalDist()

def _z(alpha: float) -> float:
    """Standard Normal quantile Φ^{-1}(alpha)."""
    return _N.inv_cdf(alpha)

def _phi(x: float) -> float:
    """Standard Normal pdf φ(x)."""
    return _N.pdf(x)

# -------- Parametric (Normal) VaR/ES --------
def var_parametric(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.95,
                   horizon_days: int = 1, exposure: float = 1.0) -> float:
    mu = returns.mean().values
    cov = returns.cov().values
    mu_p = float(weights @ mu) * horizon_days
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * sqrt(horizon_days)
    z = _z(alpha)
    loss_var = (-mu_p + z * sigma_p) * exposure   # report as +ve loss
    return float(loss_var)

def es_parametric(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.95,
                  horizon_days: int = 1, exposure: float = 1.0) -> float:
    mu = returns.mean().values
    cov = returns.cov().values
    mu_p = float(weights @ mu) * horizon_days
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * sqrt(horizon_days)
    z = _z(alpha)
    es = (-mu_p + sigma_p * (_phi(z) / (1 - alpha))) * exposure
    return float(es)

# -------- Historical VaR/ES --------
def var_historical(returns: pd.DataFrame, weights: np.ndarray,
                   alpha: float = 0.95, horizon_days: int = 1,
                   exposure: float = 1.0, sqrt_time: bool = True) -> float:
    r_p = portfolio_returns(returns, weights)
    if sqrt_time and horizon_days > 1:
        r_p = r_p * sqrt(horizon_days)
    q = r_p.quantile(1 - alpha)      # lower tail (likely negative)
    return float(-q * exposure)      # report loss as +ve

def es_historical(returns: pd.DataFrame, weights: np.ndarray,
                  alpha: float = 0.95, horizon_days: int = 1,
                  exposure: float = 1.0, sqrt_time: bool = True) -> float:
    r_p = portfolio_returns(returns, weights)
    if sqrt_time and horizon_days > 1:
        r_p = r_p * sqrt(horizon_days)
    var_loss = -r_p.quantile(1 - alpha)
    tail = r_p[r_p <= -var_loss]
    es = -tail.mean() if len(tail) else var_loss
    return float(es * exposure)

# -------- Kupiec POF + rolling backtest --------
def kupiec_pof(exceedances: int, T: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Kupiec Proportion-of-Failures test. H0: hit rate == (1 - alpha).
    Returns (LR statistic ~ Chi^2(1), p-value).
    For χ² with 1 dof: CDF(x) = erf( sqrt(x/2) ), so p = 1 - CDF = erfc( sqrt(x/2) ).
    """
    p = 1 - alpha
    x = exceedances
    if T <= 0:
        raise ValueError("T must be > 0 for Kupiec test.")
    pi_hat = x / T

    # log-likelihoods
    # handle edge cases for stability
    eps = 1e-12
    pi_hat = min(max(pi_hat, eps), 1 - eps)
    p = min(max(p, eps), 1 - eps)

    ll0 = (T - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    LR = float(-2 * (ll0 - ll1))

    # p-value using χ²(1): p = 1 - F(LR) = erfc(sqrt(LR/2))
    pval = float(erfc(sqrt(LR / 2.0)))
    return LR, pval

def backtest_var_historical(returns: pd.DataFrame, weights: np.ndarray,
                            alpha: float = 0.95, window: int = 250) -> dict:
    """
    Rolling historical VaR backtest (1-day horizon).
    - Computes rolling (t-1)-based VaR_t from the last `window` obs.
    - Flags exceptions when r_p,t < VaR_t threshold (returns are negative).
    Returns dict with series and Kupiec stats.
    """
    r_p = portfolio_returns(returns, weights)
    q = r_p.rolling(window).quantile(1 - alpha).shift(1)
    exceptions = (r_p < q).astype(int)

    mask = q.notna()
    T = int(mask.sum())
    x = int(exceptions[mask].sum())
    LR, pval = kupiec_pof(x, T, alpha=alpha)

    return {
        "r_p": r_p,
        "VaR_threshold": q,
        "exceptions": exceptions,
        "window": window,
        "alpha": alpha,
        "T": T,
        "exceedances": x,
        "hit_rate": (x / T) if T > 0 else np.nan,
        "kupiec_LR": LR,
        "kupiec_pvalue": pval,
    }
