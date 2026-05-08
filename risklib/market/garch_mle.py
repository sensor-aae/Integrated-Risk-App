"""
risklib/market/garch_mle.py
============================
GARCH(1,1) Maximum Likelihood Estimation.

Provides MLE-based parameter estimation as an upgrade over the fixed-parameter
GARCH filter in market.py. This module closes the documented model limitation:

  "The GARCH(1,1) filter uses fixed parameters (α=0.05, β=0.94) rather than
   MLE-estimated parameters. This simplification may misspecify the volatility
   process for individual assets, particularly during regime shifts."

Model
-----
  sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

  Stationarity condition: alpha + beta < 1 (enforced via parameter transform)
  Gaussian log-likelihood maximised via scipy.optimize (L-BFGS-B)

Design
------
  - garch11_filter_mle() is a drop-in replacement for garch11_filter() in market.py
    with an additional `fit=True` flag. When fit=True, parameters are estimated
    by MLE from the data; when fit=False, behaviour is identical to the original.
  - No new runtime dependencies (scipy is already in requirements)
  - MarketRiskConfig gains an optional `fit_garch` flag; default=False preserves
    existing behaviour so no existing code breaks

References
----------
  Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
    Journal of Econometrics, 31(3), 307–327.
  Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates
    of the Variance of United Kingdom Inflation. Econometrica, 50(4), 987–1007.
"""

from __future__ import annotations

from math import log, sqrt
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Internal: variance path and negative log-likelihood
# ---------------------------------------------------------------------------

def _variance_path(
    r: np.ndarray,
    omega: float,
    alpha_g: float,
    beta_g: float,
) -> np.ndarray:
    """
    Compute conditional variance series sigma^2_t for GARCH(1,1).
    Initialised at the sample variance of r.
    """
    n = len(r)
    sig2 = np.empty(n, dtype=float)
    sig2[0] = max(1e-18, float(np.var(r, ddof=1)))
    r2 = r ** 2
    for t in range(1, n):
        sig2[t] = max(1e-18, omega + alpha_g * r2[t - 1] + beta_g * sig2[t - 1])
    return sig2


def _neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    """
    Negative Gaussian log-likelihood for GARCH(1,1).
    Parameters are in unconstrained space; transformations enforce constraints:
      omega   = exp(p0)                             > 0
      alpha_g = sigmoid(p1)                         ∈ (0,1)
      beta_g  = (1 - alpha_g - eps) * sigmoid(p2)  ensures alpha + beta < 1
    """
    p0, p1, p2 = params
    omega   = float(np.exp(p0))
    alpha_g = float(1.0 / (1.0 + np.exp(-p1)))
    beta_g  = float((1.0 - alpha_g - 1e-6) / (1.0 + np.exp(-p2)))

    sig2 = _variance_path(r, omega, alpha_g, beta_g)

    # Gaussian NLL: 0.5 * sum[log(sig2_t) + r_t^2 / sig2_t]
    nll = 0.5 * float(np.sum(np.log(sig2) + r ** 2 / sig2))
    return nll if np.isfinite(nll) else 1e18


# ---------------------------------------------------------------------------
# Public: MLE estimation
# ---------------------------------------------------------------------------

def fit_garch11_mle(
    r: pd.Series,
    n_restarts: int = 3,
    max_iter: int = 2000,
) -> Dict:
    """
    Fit GARCH(1,1) by Maximum Likelihood Estimation.

    Uses L-BFGS-B with multiple random restarts to avoid local minima.
    Parameters are estimated in unconstrained space and transformed to
    enforce stationarity (alpha + beta < 1).

    Parameters
    ----------
    r          : pd.Series of asset or portfolio returns
    n_restarts : number of random restarts (best result kept)
    max_iter   : maximum optimizer iterations per restart

    Returns
    -------
    dict with keys:
      omega, alpha_g, beta_g  — MLE parameter estimates
      persistence             — alpha + beta  (< 1 for stationarity)
      long_run_vol            — annualised long-run volatility = sqrt(omega / (1-alpha-beta)) * sqrt(252)
      log_likelihood          — maximised log-likelihood
      aic                     — Akaike information criterion  (3 free parameters)
      bic                     — Bayesian information criterion
      converged               — bool, True if best restart converged
      n_obs                   — number of observations used
      source                  — "MLE"
    """
    r_arr = r.dropna().values.astype(float)
    n = len(r_arr)
    if n < 50:
        raise ValueError(
            f"Need ≥50 observations for GARCH MLE; got {n}. "
            "Use fixed parameters (fit=False) for short series."
        )

    best_nll = np.inf
    best_res = None
    rng      = np.random.default_rng(42)
    var_est  = float(np.var(r_arr, ddof=1))

    # Deterministic starting points + random restarts
    start_configs = [
        [log(var_est * 0.05), 0.0,  0.0],
        [log(var_est * 0.10), 0.5,  1.5],
        [log(var_est * 0.02), -1.0, 2.5],
    ]
    for _ in range(max(0, n_restarts - 3)):
        start_configs.append(rng.normal(0.0, 1.0, 3).tolist())

    for x0 in start_configs[:n_restarts]:
        try:
            res = minimize(
                _neg_loglik,
                x0=x0,
                args=(r_arr,),
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is None:
        raise RuntimeError("GARCH MLE: optimisation failed on all restarts.")

    # Recover constrained estimates
    p0, p1, p2 = best_res.x
    omega   = float(np.exp(p0))
    alpha_g = float(1.0 / (1.0 + np.exp(-p1)))
    beta_g  = float((1.0 - alpha_g - 1e-6) / (1.0 + np.exp(-p2)))
    persist = alpha_g + beta_g

    log_lik  = float(-best_nll)
    k        = 3
    aic      = float(2 * k - 2 * log_lik)
    bic      = float(k * log(n) - 2 * log_lik)
    lrv_denom = max(1.0 - persist, 1e-12)
    long_run_vol = float(sqrt(omega / lrv_denom) * sqrt(252))

    return {
        "omega":          omega,
        "alpha_g":        alpha_g,
        "beta_g":         beta_g,
        "persistence":    persist,
        "long_run_vol":   long_run_vol,
        "log_likelihood": log_lik,
        "aic":            aic,
        "bic":            bic,
        "converged":      bool(best_res.success),
        "n_obs":          n,
        "source":         "MLE",
    }


# ---------------------------------------------------------------------------
# Public: volatility filter (drop-in replacement for market.garch11_filter)
# ---------------------------------------------------------------------------

def garch11_filter_mle(
    r: pd.Series,
    fit: bool = True,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
    n_restarts: int = 3,
) -> Tuple[pd.Series, Dict]:
    """
    GARCH(1,1) conditional volatility filter with optional MLE estimation.

    Drop-in replacement for garch11_filter() in market.py, adding the `fit`
    flag. When fit=True, omega/alpha/beta are estimated from the data.
    When fit=False, the function is behaviourally identical to the original.

    Parameters
    ----------
    r          : pd.Series of returns
    fit        : if True, estimate parameters via MLE (recommended)
                 if False, use the provided alpha_g / beta_g (legacy behaviour)
    alpha_g    : ARCH parameter — used only when fit=False (default 0.05)
    beta_g     : GARCH parameter — used only when fit=False (default 0.94)
    n_restarts : MLE random restarts (used only when fit=True)

    Returns
    -------
    (sigma, fit_info)
      sigma    : pd.Series of conditional standard deviations (same index as r)
      fit_info : dict with parameter estimates and model diagnostics
                 Always contains: omega, alpha_g, beta_g, persistence, source
                 MLE additionally: log_likelihood, aic, bic, converged, n_obs
    """
    r_clean = r.dropna()

    if fit:
        fit_info = fit_garch11_mle(r_clean, n_restarts=n_restarts)
        omega_   = fit_info["omega"]
        alpha_g_ = fit_info["alpha_g"]
        beta_g_  = fit_info["beta_g"]
    else:
        long_run_var = max(float(r_clean.var(ddof=1)), 1e-18)
        omega_   = max(1e-18, (1.0 - alpha_g - beta_g) * long_run_var)
        alpha_g_ = alpha_g
        beta_g_  = beta_g
        fit_info = {
            "omega":       omega_,
            "alpha_g":     alpha_g_,
            "beta_g":      beta_g_,
            "persistence": alpha_g + beta_g,
            "long_run_vol": float(sqrt(long_run_var) * sqrt(252)),
            "source":      "fixed",
        }

    sig2  = _variance_path(r_clean.values, omega_, alpha_g_, beta_g_)
    sigma = pd.Series(np.sqrt(sig2), index=r_clean.index, name="sigma_garch")
    return sigma, fit_info


# ---------------------------------------------------------------------------
# Convenience: one-step-ahead sigma forecast
# ---------------------------------------------------------------------------

def garch11_forecast_next(
    r_last: float,
    sigma_last: float,
    omega: float,
    alpha_g: float,
    beta_g: float,
) -> float:
    """
    One-step-ahead conditional standard deviation forecast.
    sigma^2_{t+1} = omega + alpha * r_t^2 + beta * sigma_t^2
    """
    sig2_next = omega + alpha_g * (r_last ** 2) + beta_g * (sigma_last ** 2)
    return float(sqrt(max(sig2_next, 1e-18)))
