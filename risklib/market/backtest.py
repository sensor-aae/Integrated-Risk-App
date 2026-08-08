"""
risklib/market/backtest.py
==========================
VaR backtesting: out-of-sample exception generation and coverage tests.

Tests implemented
-----------------
  Kupiec (1995) Proportion-of-Failures       — unconditional coverage
  Christoffersen (1998) Independence         — serial independence of exceptions
  Christoffersen (1998) Conditional Coverage — LR_cc = LR_uc + LR_ind ~ Chi^2(2)

Backtests implemented
---------------------
  backtest_var_historical : rolling historical-simulation VaR
  backtest_var_fhs        : rolling Filtered Historical Simulation VaR

Out-of-sample discipline
------------------------
Both backtests estimate the threshold at time t using information available
strictly through t-1. This is the single most important property of the module:
`rolling(window).quantile()` at row t includes row t itself — the very day being
predicted — so every threshold is `.shift(1)`-ed. Without that, every result in
the repository would be meaningless, and it is the first thing a reviewer checks.

Regulatory context
------------------
Basel III/IV (FRTB) backtesting tests unconditional coverage at 99%. The joint
conditional coverage test additionally detects exception clustering, a known
failure mode of static VaR models during volatility regime shifts (March 2020,
the 2022 rate shock). Kupiec alone cannot see it.

References
----------
  Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement
    Models. Journal of Derivatives, 3(2), 73-84.
  Christoffersen, P. (1998). Evaluating Interval Forecasts.
    International Economic Review, 39(4), 841-862.
"""

from __future__ import annotations

import math
from math import erfc, log, sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .garch import VAR_FLOOR, omega_from_variance_target
from .market_risk_model import portfolio_returns

__all__ = [
    "kupiec_pof",
    "christoffersen_independence",
    "joint_coverage_test",
    "backtest_var_historical",
    "backtest_var_fhs",
]


def _clamp(x: float, eps: float = 1e-12) -> float:
    """Keep probabilities strictly inside (0, 1) so log() is finite."""
    return min(max(float(x), eps), 1.0 - eps)


# ---------------------------------------------------------------------------
# Test 1 — Kupiec (1995) Proportion-of-Failures
# ---------------------------------------------------------------------------

def kupiec_pof(exceedances: int, T: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Kupiec Proportion-of-Failures test — unconditional coverage.

    Exceptions are modelled as i.i.d. Bernoulli. `ll0` is the log-likelihood
    under the null rate (1-alpha), `ll1` under the MLE rate x/T. The likelihood
    ratio is asymptotically Chi^2(1).

        H0: observed exception rate == (1 - alpha)

    The p-value uses the exact Chi^2(1) survival function,
    P(X > x) = 2*(1 - Phi(sqrt(x))) = erfc(sqrt(x/2)) — no SciPy, no table.

    Limitation: this test sees only the COUNT. A model producing exactly 5%
    exceptions, all in one week, passes. That is what the independence test
    below is for.

    Returns
    -------
    (LR_uc, p_value) — reject H0 (mis-specified coverage) if p < 0.05
    """
    if T <= 0:
        raise ValueError("T must be > 0.")

    p = _clamp(1.0 - alpha)
    pi_hat = _clamp(exceedances / T)
    x = exceedances

    ll0 = (T - x) * log(1.0 - p) + x * log(p)
    ll1 = (T - x) * log(1.0 - pi_hat) + x * log(pi_hat)

    LR_uc = max(float(-2.0 * (ll0 - ll1)), 0.0)
    return LR_uc, float(erfc(sqrt(LR_uc / 2.0)))


# ---------------------------------------------------------------------------
# Test 2 — Christoffersen (1998) Independence
# ---------------------------------------------------------------------------

def christoffersen_independence(exceptions: pd.Series) -> Tuple[float, float, Dict]:
    """
    Christoffersen independence test — do exceptions cluster in time?

    The exception indicator series is treated as a first-order Markov chain
    with transition counts n_ij = #{t : I_{t-1}=i, I_t=j}. Under H1 today's
    exception probability may depend on yesterday's state; under H0 there is a
    single unconditional probability.

        H0: pi_01 == pi_11   (no clustering)
        LR_ind ~ Chi^2(1)

    Why this is the test that matters: a VaR model that clusters its exceptions
    is systematically understating risk in stressed regimes and overstating it
    in calm ones — precisely the failure mode of a static historical-simulation
    model in March 2020. Kupiec is blind to it.

    Returns
    -------
    (LR_ind, p_value, transitions)
    """
    ind = pd.Series(exceptions).dropna().values.astype(int)

    if len(ind) < 2:
        return 0.0, 1.0, {"n00": 0, "n01": 0, "n10": 0, "n11": 0,
                          "pi_01": 0.0, "pi_11": 0.0, "pi_hat": 0.0}

    prev, curr = ind[:-1], ind[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    total = n00 + n01 + n10 + n11

    if total == 0:
        return 0.0, 1.0, {"n00": 0, "n01": 0, "n10": 0, "n11": 0,
                          "pi_01": 0.0, "pi_11": 0.0, "pi_hat": 0.0}

    pi_01 = _clamp(n01 / (n00 + n01)) if (n00 + n01) > 0 else _clamp(0.0)
    pi_11 = _clamp(n11 / (n10 + n11)) if (n10 + n11) > 0 else _clamp(0.0)
    pi_hat = _clamp((n01 + n11) / total)

    # H1: Markov(1) — transition probabilities depend on the previous state
    ll_H1 = (n00 * log(1.0 - pi_01) + n01 * log(pi_01) +
             n10 * log(1.0 - pi_11) + n11 * log(pi_11))
    # H0: i.i.d. — one unconditional exception probability
    ll_H0 = ((n00 + n10) * log(1.0 - pi_hat) +
             (n01 + n11) * log(pi_hat))

    # Non-negative in theory; floating point in degenerate cases (n11 = 0) can
    # drift slightly below zero.
    LR_ind = max(float(-2.0 * (ll_H0 - ll_H1)), 0.0)

    transitions = {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": float(pi_01), "pi_11": float(pi_11), "pi_hat": float(pi_hat),
    }
    return LR_ind, float(erfc(sqrt(LR_ind / 2.0))), transitions


# ---------------------------------------------------------------------------
# Test 3 — Joint Conditional Coverage
# ---------------------------------------------------------------------------

def joint_coverage_test(exceedances: int, T: int, alpha: float,
                        exceptions_series: pd.Series) -> Dict:
    """
    Christoffersen joint conditional coverage:  LR_cc = LR_uc + LR_ind ~ Chi^2(2).

    The additivity is the standard Christoffersen decomposition — correct
    conditional coverage means correct frequency AND independence, and the two
    LR statistics are asymptotically independent. The Chi^2(2) survival
    function is exactly exp(-x/2).

    A model that passes Kupiec but fails independence produces correctly-sized
    exceptions on average while clustering them in time.
    """
    LR_uc, p_uc = kupiec_pof(exceedances, T, alpha)
    LR_ind, p_ind, transitions = christoffersen_independence(exceptions_series)
    LR_cc = LR_uc + LR_ind

    return {
        "kupiec_LR": LR_uc,
        "kupiec_pvalue": p_uc,
        "christoffersen_LR": LR_ind,
        "christoffersen_pvalue": p_ind,
        "joint_LR": LR_cc,
        "joint_pvalue": float(math.exp(-LR_cc / 2.0)),
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Backtest 1 — rolling historical simulation
# ---------------------------------------------------------------------------

def backtest_var_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    window: int = 250,
) -> Dict:
    """
    Rolling one-day historical VaR backtest with full diagnostics.

      1. r_p = R @ w
      2. threshold at t = (1-alpha) quantile of the trailing `window`
         observations ending at t-1   [.shift(1) — no look-ahead]
      3. exception if r_p,t < threshold_t
      4. Kupiec, Christoffersen and joint conditional coverage

    `mask = q.notna()` drops the burn-in period so T counts only genuine
    out-of-sample days, and the exception series passed to the independence
    test is a contiguous block — transitions must not be computed across a gap.
    """
    r_p = portfolio_returns(returns, weights)
    q = r_p.rolling(window).quantile(1.0 - alpha).shift(1)
    exceptions = (r_p < q).astype(int)

    mask = q.notna()
    T = int(mask.sum())
    x = int(exceptions[mask].sum())

    tests = joint_coverage_test(x, T, alpha, exceptions[mask])

    return {
        "method": "historical",
        "r_p": r_p,
        "VaR_threshold": q,
        "exceptions": exceptions,
        "window": window,
        "alpha": alpha,
        "T": T,
        "exceedances": x,
        "hit_rate": float(x / T) if T > 0 else float("nan"),
        **tests,
    }


# ---------------------------------------------------------------------------
# Backtest 2 — rolling Filtered Historical Simulation
# ---------------------------------------------------------------------------

def backtest_var_fhs(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    window: int = 250,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
) -> Dict:
    """
    Rolling one-day FHS VaR backtest.

    Out-of-sample construction
    --------------------------
    The GARCH recursion sigma^2_t = omega_t + a*r^2_{t-1} + b*sigma^2_{t-1} is
    already causal in r and sigma. The remaining leakage risk is `omega`, which
    under variance targeting depends on the long-run variance. An earlier
    implementation computed that from the FULL sample, so every threshold
    embedded information about the future.

    Here the long-run variance is an EXPANDING-window estimate lagged by one
    day, so omega_t uses only returns through t-1:

        lrv_t   = Var(r_1 .. r_{t-1})
        omega_t = (1 - a - b) * lrv_t
        sig2_t  = omega_t + a*r^2_{t-1} + b*sig2_{t-1}

    sigma_t is therefore exactly the one-step-ahead forecast formed at t-1. The
    standardised-residual quantile is likewise a trailing rolling quantile,
    shifted by one.

    Note: alpha_g and beta_g remain fixed here. Re-estimating GARCH by MLE at
    every step is the correct extension but costs a full optimisation per day;
    it is deliberately out of scope for this function.
    """
    r_p = portfolio_returns(returns, weights).dropna()
    n = len(r_p)

    empty = pd.Series(index=r_p.index, dtype=float)
    if n <= window:
        return {
            "method": "fhs", "r_p": r_p, "VaR_threshold": empty,
            "exceptions": pd.Series(index=r_p.index, dtype=int),
            "window": window, "alpha": alpha, "T": 0, "exceedances": 0,
            "hit_rate": float("nan"),
            **joint_coverage_test(0, 1, alpha, pd.Series(dtype=int)),
        }

    r_vals = r_p.values.astype(float)

    # Expanding, lagged long-run variance -> causal omega_t
    lrv = r_p.expanding(min_periods=2).var(ddof=1).shift(1)
    lrv = lrv.bfill().fillna(float(np.var(r_vals[:window], ddof=1)))
    lrv_vals = np.maximum(lrv.values.astype(float), VAR_FLOOR)

    sig2 = np.empty(n, dtype=float)
    sig2[0] = max(lrv_vals[0], VAR_FLOOR)
    for t in range(1, n):
        omega_t = omega_from_variance_target(lrv_vals[t], alpha_g, beta_g)
        sig2[t] = max(VAR_FLOOR, omega_t + alpha_g * r_vals[t - 1] ** 2 + beta_g * sig2[t - 1])

    # sigma_t is the forecast made at t-1, so no further shift is required.
    sigma = pd.Series(np.sqrt(sig2), index=r_p.index, name="sigma_garch")

    # Standardised residuals; their trailing quantile is lagged one day.
    z = r_p / sigma.replace(0.0, np.nan).bfill().ffill()
    qz = z.rolling(window).quantile(1.0 - alpha).shift(1)

    var_thresh = qz * sigma            # negative return level
    exceptions = ((r_p < var_thresh) & var_thresh.notna()).astype(int)

    mask = var_thresh.notna()
    T = int(mask.sum())
    x = int(exceptions[mask].sum())
    tests = joint_coverage_test(x, T, alpha, exceptions[mask]) if T > 0 else \
        joint_coverage_test(0, 1, alpha, pd.Series(dtype=int))

    return {
        "method": "fhs",
        "r_p": r_p,
        "VaR_threshold": var_thresh,
        "sigma": sigma,
        "exceptions": exceptions,
        "window": window,
        "alpha": alpha,
        "alpha_g": alpha_g,
        "beta_g": beta_g,
        "T": T,
        "exceedances": x,
        "hit_rate": float(x / T) if T > 0 else float("nan"),
        **tests,
    }
