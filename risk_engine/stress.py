from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def apply_single_name_shocks(
    returns: pd.DataFrame,
    shocks: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply one-day shocks (in return space) to specified tickers.
    shocks: dict like {"AAPL": -0.10, "MSFT": -0.05}
    Non-specified tickers unchanged. Returns a *one-row* DataFrame you can dot with weights.
    """
    cols = returns.columns.tolist()
    last = returns.iloc[[-1]].copy()
    for k, v in shocks.items():
        if k in cols:
            last[k] = float(v)
    return last

def scale_covariance(cov: np.ndarray, scale: float) -> np.ndarray:
    """
    Uniformly scale covariance matrix by 'scale' factor.
    scale > 1 increases volatility, < 1 reduces it.
    """
    return cov * float(scale)

def historical_window_mu_cov(
    returns: pd.DataFrame,
    start: str,
    end: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean vector and covariance from a historical sub-window.
    Dates are inclusive; expects returns to have a DatetimeIndex.
    """
    sub = returns.loc[start:end]
    if len(sub) < 5:
        raise ValueError("Selected window too short for μ/Σ.")
    mu = sub.mean().values
    cov = sub.cov().values
    return mu, cov
