from __future__ import annotations
import numpy as np
import pandas as pd
from math import sqrt, erfc
from typing import Tuple


def kupiec_pof(exceedances: int, T: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Kupiec Proportion-of-Failures test. H0: hit rate == (1 - alpha).
    Returns (LR statistic ~ Chi^2(1), p-value).
    p-value via χ²(1): p = erfc(sqrt(LR/2)).
    """
    p = 1 - alpha
    x = exceedances
    if T <= 0:
        raise ValueError("T must be > 0 for Kupiec test.")

    eps = 1e-12
    pi_hat = min(max(x / T, eps), 1 - eps)
    p = min(max(p, eps), 1 - eps)

    ll0 = (T - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    LR = float(-2 * (ll0 - ll1))
    pval = float(erfc(sqrt(LR / 2.0)))
    return LR, pval


def backtest_var_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    window: int = 250
) -> dict:
    """
    Rolling historical VaR backtest (1-day).
    VaR threshold q_t computed from trailing window and shifted by 1.
    Exception if realized r_p,t < q_t.
    """
    r_p = pd.Series(returns.values @ weights, index=returns.index, name="r_p")
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
        "alpha": alpha,
        "window": window,
        "T": T,
        "exceedances": x,
        "hit_rate": (x / T) if T > 0 else np.nan,
        "kupiec_LR": LR,
        "kupiec_pvalue": pval,
    }
