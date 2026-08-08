"""
risklib/market/garch.py
=======================
GARCH(1,1) conditional volatility: filtering, forecasting, and MLE estimation.

Model
-----
    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Stationarity: alpha + beta < 1
    Unconditional (long-run) variance: omega / (1 - alpha - beta)

Two parameterisation modes
--------------------------
1. **Fixed parameters (variance targeting).** alpha and beta are supplied
   (defaults 0.05 / 0.94, the RiskMetrics-adjacent convention), and omega is
   pinned so the model's unconditional variance equals the sample variance:
       omega = (1 - alpha - beta) * var(r)
   Cheap, stable, and reproducible — but it is an assumption, not an estimate.

2. **Maximum likelihood.** omega, alpha, beta are estimated from the data by
   maximising the Gaussian conditional log-likelihood.

Design note
-----------
There is exactly ONE implementation of the variance recursion in this codebase
(`variance_path`). The fixed-parameter and MLE paths differ only in how they
obtain (omega, alpha, beta) before calling it. Prior to this module the same
recursion existed in three places with subtly different initialisation.

References
----------
  Bollerslev, T. (1986). Generalized Autoregressive Conditional
    Heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
  Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with
    Estimates of the Variance of United Kingdom Inflation.
    Econometrica, 50(4), 987-1007.
"""

from __future__ import annotations

from math import log, pi, sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

__all__ = [
    "variance_path",
    "garch11_filter",
    "garch11_forecast_next",
    "fit_garch11_mle",
    "TRADING_DAYS",
    "VAR_FLOOR",
]

TRADING_DAYS = 252
VAR_FLOOR = 1e-18  # keeps sigma^2 strictly positive through the recursion


# ---------------------------------------------------------------------------
# Core recursion — the single implementation
# ---------------------------------------------------------------------------

def variance_path(
    r: np.ndarray,
    omega: float,
    alpha_g: float,
    beta_g: float,
    init_var: float | None = None,
) -> np.ndarray:
    """
    Conditional variance series sigma^2_t for GARCH(1,1).

    Parameters
    ----------
    r        : 1-D array of returns
    omega    : constant term (> 0)
    alpha_g  : ARCH coefficient
    beta_g   : GARCH coefficient
    init_var : sigma^2_0; defaults to the sample variance of r

    Notes
    -----
    Each step is floored at VAR_FLOOR so that a poor parameter proposal during
    optimisation produces a large likelihood penalty rather than log(0) or a
    division by zero.
    """
    n = len(r)
    if n == 0:
        return np.empty(0, dtype=float)

    if init_var is None:
        init_var = float(np.var(r, ddof=1)) if n > 1 else float(np.mean(r ** 2))

    sig2 = np.empty(n, dtype=float)
    sig2[0] = max(VAR_FLOOR, float(init_var))
    r2 = np.asarray(r, dtype=float) ** 2
    for t in range(1, n):
        sig2[t] = max(VAR_FLOOR, omega + alpha_g * r2[t - 1] + beta_g * sig2[t - 1])
    return sig2


def omega_from_variance_target(long_run_var: float, alpha_g: float, beta_g: float) -> float:
    """omega implied by variance targeting: omega = (1 - alpha - beta) * sigma^2_LR."""
    return max(VAR_FLOOR, (1.0 - alpha_g - beta_g) * float(long_run_var))


# ---------------------------------------------------------------------------
# MLE
# ---------------------------------------------------------------------------

def _unpack(params: np.ndarray) -> Tuple[float, float, float]:
    """
    Map unconstrained R^3 to the admissible GARCH parameter space.

        omega   = exp(p0)                            > 0
        alpha   = sigmoid(p1)                        in (0, 1)
        beta    = (1 - alpha - eps) * sigmoid(p2)    => alpha + beta < 1

    Constraints are enforced structurally rather than handed to the optimiser.
    L-BFGS-B can then roam freely over R^3, which avoids the boundary-stalling
    that constrained solvers exhibit on the coupled stationarity inequality.
    """
    p0, p1, p2 = params
    omega = float(np.exp(np.clip(p0, -700.0, 100.0)))
    alpha_g = float(expit(p1))
    beta_g = float((1.0 - alpha_g - 1e-6) * expit(p2))
    return omega, alpha_g, beta_g


def _neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    """
    Negative Gaussian conditional log-likelihood, including the 2*pi constant.

    The constant is retained so that the reported log-likelihood, AIC and BIC
    are directly comparable to values from `arch`, `statsmodels` or R. (An
    earlier version dropped it, which left every reported IC offset by
    +0.5 * n * log(2*pi).)
    """
    omega, alpha_g, beta_g = _unpack(params)
    sig2 = variance_path(r, omega, alpha_g, beta_g)
    nll = 0.5 * float(np.sum(np.log(2.0 * pi) + np.log(sig2) + r ** 2 / sig2))
    return nll if np.isfinite(nll) else 1e18


def fit_garch11_mle(
    r: pd.Series | np.ndarray,
    n_restarts: int = 5,
    max_iter: int = 2000,
    min_obs: int = 50,
) -> Dict:
    """
    Fit GARCH(1,1) by maximum likelihood (L-BFGS-B, multiple restarts).

    Restarts matter: the GARCH likelihood is close to flat in the
    (alpha + beta) direction near persistence = 1, so a single start from a
    poor point can converge to a local optimum *and report success*. The best
    NLL across restarts is kept.

    Returns
    -------
    dict with omega, alpha_g, beta_g, persistence, long_run_vol (annualised),
    log_likelihood, aic, bic, converged, n_obs, source="MLE".
    """
    r_arr = np.asarray(pd.Series(r).dropna().values, dtype=float)
    n = len(r_arr)
    if n < min_obs:
        raise ValueError(
            f"Need >= {min_obs} observations for GARCH MLE; got {n}. "
            "Use fixed parameters (fit=False) for short series."
        )

    var_est = max(float(np.var(r_arr, ddof=1)), VAR_FLOOR)
    rng = np.random.default_rng(42)

    # Deterministic starts spanning low / mid / high persistence, then randoms.
    starts = [
        [log(var_est * 0.05), 0.0, 0.0],
        [log(var_est * 0.10), 0.5, 1.5],
        [log(var_est * 0.02), -1.0, 2.5],
    ]
    while len(starts) < n_restarts:
        starts.append([log(var_est) + rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)])
    starts = starts[:n_restarts]

    best_nll, best_res = np.inf, None
    for x0 in starts:
        try:
            res = minimize(
                _neg_loglik,
                x0=np.asarray(x0, dtype=float),
                args=(r_arr,),
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll, best_res = float(res.fun), res
        except Exception:
            continue

    if best_res is None:
        raise RuntimeError("GARCH MLE: optimisation failed on all restarts.")

    omega, alpha_g, beta_g = _unpack(best_res.x)
    persist = alpha_g + beta_g
    log_lik = float(-best_nll)
    k = 3

    return {
        "omega": omega,
        "alpha_g": alpha_g,
        "beta_g": beta_g,
        "persistence": persist,
        "long_run_vol": float(sqrt(omega / max(1.0 - persist, 1e-12)) * sqrt(TRADING_DAYS)),
        "log_likelihood": log_lik,
        "aic": float(2 * k - 2 * log_lik),
        "bic": float(k * log(n) - 2 * log_lik),
        "converged": bool(best_res.success),
        "n_obs": n,
        "source": "MLE",
    }


# ---------------------------------------------------------------------------
# Public filter — the single entry point used by the risk models
# ---------------------------------------------------------------------------

def garch11_filter(
    r: pd.Series,
    fit: bool = False,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
    n_restarts: int = 5,
) -> Tuple[pd.Series, Dict]:
    """
    Conditional volatility filter, with optional MLE estimation.

    Parameters
    ----------
    r          : returns series
    fit        : True  -> estimate (omega, alpha, beta) by MLE
                 False -> variance targeting on the supplied alpha_g / beta_g
    alpha_g    : ARCH parameter, used only when fit=False
    beta_g     : GARCH parameter, used only when fit=False
    n_restarts : MLE restarts, used only when fit=True

    Returns
    -------
    (sigma, info)
      sigma : conditional standard deviations, indexed like r
      info  : parameter dict; `info["source"]` is "MLE" or "fixed", so any
              downstream consumer can tell whether the numbers were estimated
              or assumed. This tag is what makes the choice auditable.

    If fit=True fails (non-convergence, too few observations), the function
    falls back to the fixed-parameter path and records the reason in
    info["fallback_reason"] rather than raising — a stress panel should not
    disappear because an optimiser had a bad day.
    """
    r_clean = pd.Series(r).dropna()
    long_run_var = max(float(r_clean.var(ddof=1)), VAR_FLOOR) if len(r_clean) > 1 else VAR_FLOOR
    fallback_reason = None

    if fit:
        try:
            info = fit_garch11_mle(r_clean, n_restarts=n_restarts)
            omega_, alpha_, beta_ = info["omega"], info["alpha_g"], info["beta_g"]
        except (ValueError, RuntimeError) as exc:
            fallback_reason = str(exc)
            fit = False

    if not fit:
        omega_ = omega_from_variance_target(long_run_var, alpha_g, beta_g)
        alpha_, beta_ = alpha_g, beta_g
        info = {
            "omega": omega_,
            "alpha_g": alpha_,
            "beta_g": beta_,
            "persistence": alpha_ + beta_,
            "long_run_vol": float(sqrt(long_run_var) * sqrt(TRADING_DAYS)),
            "source": "fixed",
        }
        if fallback_reason:
            info["fallback_reason"] = fallback_reason

    sig2 = variance_path(r_clean.values, omega_, alpha_, beta_, init_var=long_run_var)
    sigma = pd.Series(np.sqrt(sig2), index=r_clean.index, name="sigma_garch")
    return sigma, info


def garch11_forecast_next(
    r_last: float,
    sigma_last: float,
    omega: float,
    alpha_g: float,
    beta_g: float,
) -> float:
    """One-step-ahead conditional standard deviation: sigma^2_{t+1} = w + a*r_t^2 + b*sigma_t^2."""
    sig2_next = omega + alpha_g * (r_last ** 2) + beta_g * (sigma_last ** 2)
    return float(sqrt(max(sig2_next, VAR_FLOOR)))
