"""
risklib/market/backtest.py
==========================
Backtesting framework for VaR models.

Implements:
  - Kupiec (1995) Proportion-of-Failures (POF) test — unconditional coverage
  - Christoffersen (1998) Independence test        — serial independence of exceptions
  - Christoffersen (1998) Joint Conditional Coverage test — LR_cc = LR_uc + LR_ind

References:
  Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models.
    Journal of Derivatives, 3(2), 73–84.
  Christoffersen, P. (1998). Evaluating Interval Forecasts.
    International Economic Review, 39(4), 841–862.

Regulatory context:
  Basel III/IV backtesting requirements (FRTB) test unconditional coverage at 99%.
  The joint conditional coverage test additionally detects exception clustering —
  a known failure mode of static VaR models during volatility regime shifts
  (e.g. March 2020, 2022 rate shock). Kupiec alone cannot detect this.

Design:
  - All functions are pure (no side effects, no global state)
  - backtest_var_historical() is the primary entry point; returns a flat dict
    containing raw series, exception counts, and all three test statistics
  - joint_coverage_test() can be called independently on any exception series
"""

from __future__ import annotations

import math
from math import erfc, log, sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Project multivariate return matrix onto portfolio weights."""
    return pd.Series(returns.values @ weights, index=returns.index, name="r_p")


def _clamp(x: float, eps: float = 1e-12) -> float:
    return min(max(float(x), eps), 1.0 - eps)


# ---------------------------------------------------------------------------
# Test 1 — Kupiec (1995) Proportion-of-Failures
# ---------------------------------------------------------------------------

def kupiec_pof(exceedances: int, T: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Kupiec (1995) Proportion-of-Failures (POF) test.

    H0: observed exception rate == (1 - alpha)  [unconditional coverage]
    LR_uc ~ Chi^2(1) under H0.

    Parameters
    ----------
    exceedances : number of VaR exceptions observed
    T           : total out-of-sample observations
    alpha       : VaR confidence level (e.g. 0.95)

    Returns
    -------
    (LR_uc, p_value)
        LR_uc  : likelihood-ratio statistic
        p_value: p-value; reject H0 (model mis-specified) if p < 0.05
    """
    if T <= 0:
        raise ValueError("T must be > 0.")

    p      = _clamp(1.0 - alpha)
    pi_hat = _clamp(exceedances / T)
    x      = exceedances

    ll0 = (T - x) * log(1.0 - p)      + x * log(p)
    ll1 = (T - x) * log(1.0 - pi_hat) + x * log(pi_hat)

    LR_uc = float(-2.0 * (ll0 - ll1))
    pval  = float(erfc(sqrt(LR_uc / 2.0)))   # Chi^2(1) survival
    return LR_uc, pval


# ---------------------------------------------------------------------------
# Test 2 — Christoffersen (1998) Independence
# ---------------------------------------------------------------------------

def christoffersen_independence(
    exceptions: pd.Series,
) -> Tuple[float, float, Dict]:
    """
    Christoffersen (1998) independence test.

    Tests H0: VaR exceptions are serially independent — i.e., knowing that
    an exception occurred yesterday conveys no information about today.
    Equivalently: the transition probability from exception to exception
    (pi_11) equals the transition probability from no-exception to exception
    (pi_01).

    H0: pi_01 == pi_11   [independence / no clustering]
    LR_ind ~ Chi^2(1) under H0.

    Transition counts from the binary indicator series I_t ∈ {0, 1}:
      n_ij = #{t : I_{t-1}=i, I_t=j}

    Parameters
    ----------
    exceptions : pd.Series of 0/1 exception indicators (aligned to OOS window)

    Returns
    -------
    (LR_ind, p_value, transitions_dict)
        LR_ind   : independence LR statistic
        p_value  : p-value; reject H0 (exceptions cluster) if p < 0.05
        transitions_dict: n00, n01, n10, n11, pi_01, pi_11, pi_hat
    """
    I = exceptions.dropna().values.astype(int)

    # Transition counts
    n00 = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
    n01 = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
    n10 = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
    n11 = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))

    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0, 1.0, {"n00": 0, "n01": 0, "n10": 0, "n11": 0,
                           "pi_01": 0.0, "pi_11": 0.0, "pi_hat": 0.0}

    # Conditional transition probabilities
    pi_01  = _clamp(n01 / (n00 + n01)) if (n00 + n01) > 0 else 1e-12
    pi_11  = _clamp(n11 / (n10 + n11)) if (n10 + n11) > 0 else 1e-12
    pi_hat = _clamp((n01 + n11) / total)

    # Log-likelihoods
    # H1: Markov(1) — transition probs depend on previous state
    ll_H1 = (n00 * log(1.0 - pi_01) + n01 * log(pi_01) +
             n10 * log(1.0 - pi_11) + n11 * log(pi_11))
    # H0: iid — single unconditional exception probability
    ll_H0 = ((n00 + n10) * log(1.0 - pi_hat) +
             (n01 + n11) * log(pi_hat))

    LR_ind = max(float(-2.0 * (ll_H0 - ll_H1)), 0.0)  # numerical guard
    pval   = float(erfc(sqrt(LR_ind / 2.0)))           # Chi^2(1) survival

    transitions = {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": float(pi_01),
        "pi_11": float(pi_11),
        "pi_hat": float(pi_hat),
    }
    return LR_ind, pval, transitions


# ---------------------------------------------------------------------------
# Test 3 — Joint Conditional Coverage (Christoffersen 1998)
# ---------------------------------------------------------------------------

def joint_coverage_test(
    exceedances: int,
    T: int,
    alpha: float,
    exceptions_series: pd.Series,
) -> Dict:
    """
    Christoffersen (1998) joint conditional coverage test.

    Combines unconditional coverage (Kupiec) and serial independence
    (Christoffersen) into a single joint test:

        LR_cc = LR_uc + LR_ind  ~  Chi^2(2) under H0

    H0: exception frequency is correct AND exceptions are independent.

    For Chi^2(2): p-value = exp(-LR_cc / 2)  [exact closed form]

    A model that passes Kupiec but fails the independence test produces
    correctly-sized exceptions on average but clusters them in time —
    understating risk during stressed periods and overstating it in calm ones.
    This is the key failure mode the independence test is designed to detect.

    Parameters
    ----------
    exceedances       : number of VaR exceptions (int)
    T                 : total OOS observations (int)
    alpha             : VaR confidence level
    exceptions_series : pd.Series of 0/1 exception indicators

    Returns
    -------
    dict with keys:
      kupiec_LR, kupiec_pvalue
      christoffersen_LR, christoffersen_pvalue
      joint_LR, joint_pvalue
      transitions  (sub-dict with n00/n01/n10/n11/pi_01/pi_11/pi_hat)
    """
    LR_uc,  p_uc  = kupiec_pof(exceedances, T, alpha)
    LR_ind, p_ind, transitions = christoffersen_independence(exceptions_series)

    LR_cc = LR_uc + LR_ind
    p_cc  = float(math.exp(-LR_cc / 2.0))   # Chi^2(2) exact

    return {
        "kupiec_LR":              LR_uc,
        "kupiec_pvalue":          p_uc,
        "christoffersen_LR":      LR_ind,
        "christoffersen_pvalue":  p_ind,
        "joint_LR":               LR_cc,
        "joint_pvalue":           p_cc,
        "transitions":            transitions,
    }


# ---------------------------------------------------------------------------
# Primary entry point — rolling historical VaR backtest
# ---------------------------------------------------------------------------

def backtest_var_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    window: int = 250,
) -> Dict:
    """
    Rolling 1-day historical VaR backtest with full statistical diagnostics.

    Procedure:
      1. Compute portfolio returns r_p = returns @ weights
      2. At each t, estimate VaR from the trailing `window` observations
         ending at t-1 (strict out-of-sample: no look-ahead)
      3. Flag exception if r_p,t < VaR_threshold_t
      4. Run Kupiec POF, Christoffersen independence, and joint CC tests

    Parameters
    ----------
    returns : pd.DataFrame of asset log-returns (rows = dates, cols = assets)
    weights : np.ndarray of portfolio weights (must sum to 1)
    alpha   : VaR confidence level (default 0.95)
    window  : rolling estimation window in trading days (default 250)

    Returns
    -------
    dict containing:
      r_p, VaR_threshold, exceptions   — time series (pd.Series)
      window, alpha, T, exceedances, hit_rate
      kupiec_LR, kupiec_pvalue
      christoffersen_LR, christoffersen_pvalue
      joint_LR, joint_pvalue
      transitions                       — Markov transition counts and probs
    """
    r_p        = _portfolio_returns(returns, weights)
    q          = r_p.rolling(window).quantile(1.0 - alpha).shift(1)
    exceptions = (r_p < q).astype(int)

    mask = q.notna()
    T = int(mask.sum())
    x = int(exceptions[mask].sum())

    tests = joint_coverage_test(x, T, alpha, exceptions[mask])

    return {
        "r_p":           r_p,
        "VaR_threshold": q,
        "exceptions":    exceptions,
        "window":        window,
        "alpha":         alpha,
        "T":             T,
        "exceedances":   x,
        "hit_rate":      float(x / T) if T > 0 else float("nan"),
        **tests,
    }
