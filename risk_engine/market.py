from __future__ import annotations
import numpy as np
import pandas as pd
from statistics import NormalDist
from math import sqrt, erfc
from typing import Tuple


# -------- Core helpers --------
def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1 (safe if sum is 0)."""
    s = float(np.sum(w))
    return (w / s) if s != 0 else w

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


def _cov_shrink(cov: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """
    Tiny shrinkage toward diagonal to avoid near-singulaer coveriences.
    cov_s = (1-lam) * cov + lam * diag(diag(cov))
    
    """
    d = np.diag(np.diag(cov))
    return (1-lam) * cov + lam * d

def var_es_monte_carlo(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.95, horizon_days: int = 1, exposure: float = 1.0 , n_sims: int = 100_000, seed: int | None = 42 , shrink_lambda: float = 0.01) -> tuple[float, float]:
    """
    Monte-Carlo VaR & ES via multivariate normal using sample μ, Σ.
    Returns  (VaR, ES) as +ve loss amounts.
    
    Notes:
    - Adds light covariance shrinkage for stability.
    - Scales μ by horizon_days, Σ by horizon_days.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days
    cov = _cov_shrink(cov, lam = shrink_lambda)
    
    sims = rng.multivariate_normal(mu, cov, size=int(n_sims), method="cholesky")
    port = sims @ weights  #simulates portfolio returns for the horizon
    
    # VaR threshold is the (1 - alpha) lower-tail quantile of returns
    q = np.quantile(port, 1 - alpha)
    var_loss = float(-q * exposure)
    
    # ES = mean of returns <= quantile; report as +ve loss
    tail = port[port <= q]
    es_loss = float(-tail.mean() * exposure) if tail.size else var_loss
    return var_loss, es_loss
def mc_portfolio_loss_from_mu_cov(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    exposure: float = 1.0,
    n_sims: int = 50_000,
    seed: int | None = 42,
) -> tuple[float, float]:
    """
    Monte Carlo VaR & ES (losses) for a given μ, Σ (already at the chosen horizon).
    """
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu, cov, size=int(n_sims), method="cholesky")
    port = sims @ weights
    q = np.quantile(port, 1 - alpha)
    var_loss = float(-q * exposure)
    tail = port[port <= q]
    es_loss = float(-tail.mean() * exposure) if tail.size else var_loss
    return var_loss, es_loss

# ---------- GARCH(1,1) "lite" (fixed params) ----------
def garch11_filter(
    r: pd.Series,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
    long_run_var: float | None = None,
    init_var: float | None = None,
) -> pd.Series:
    """
    Compute conditional variance series sigma_t^2 via fixed-parameter GARCH(1,1).
    r: returns series (pd.Series)
    alpha_g, beta_g: GARCH parameters (alpha+beta < 1 recommended)
    long_run_var: if None, uses sample variance of r
    init_var: initial variance; if None, uses long_run_var
    Returns sigma (standard deviation) series aligned with r index.
    """
    r = r.dropna()
    if long_run_var is None:
        long_run_var = float(r.var(ddof=1)) if len(r) > 1 else float((r**2).mean())
    if init_var is None:
        init_var = long_run_var
    omega = max(1e-18, (1.0 - alpha_g - beta_g) * long_run_var)

    sig2 = np.empty(len(r), dtype=float)
    sig2[0] = max(1e-18, init_var)
    r2 = r.values**2
    for t in range(1, len(r)):
        sig2[t] = omega + alpha_g * r2[t-1] + beta_g * sig2[t-1]
    sigma = np.sqrt(sig2)
    return pd.Series(sigma, index=r.index, name="sigma_garch")

def garch11_forecast_sigma_next(
    r_last: float,
    sigma_last: float,
    alpha_g: float,
    beta_g: float,
    long_run_var: float
) -> float:
    """One-step-ahead sigma forecast under fixed-parameter GARCH(1,1)."""
    omega = max(1e-18, (1.0 - alpha_g - beta_g) * long_run_var)
    sig2_next = omega + alpha_g * (r_last**2) + beta_g * (sigma_last**2)
    return float(np.sqrt(max(sig2_next, 1e-18)))

def fhs_var_es_next(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    exposure: float = 1.0,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
) -> Tuple[float, float]:
    """
    Filtered Historical Simulation VaR/ES for t+1 (1-day horizon).
    1) Build portfolio return series r_t
    2) Fit GARCH-lite sigma_t
    3) Standardize z_t = r_t / sigma_t
    4) qz = quantile(z, 1-alpha); z-tail mean for ES
    5) Forecast sigma_{t+1}; VaR = -qz * sigma_{t+1} * exposure; ES similar
    """
    r_p = pd.Series(returns.values @ weights, index=returns.index, name="r_p").dropna()
    if len(r_p) < 50:
        # not enough history for a stable filter
        q = r_p.quantile(1 - alpha)
        var_loss = float(-q * exposure)
        # ES fallback
        tail = r_p[r_p <= q]
        es_loss = float(-tail.mean() * exposure) if len(tail) else var_loss
        return var_loss, es_loss

    long_var = float(r_p.var(ddof=1))
    sigma = garch11_filter(r_p, alpha_g=alpha_g, beta_g=beta_g, long_run_var=long_var)
    # Avoid div-by-zero
    sigma_safe = sigma.replace(0.0, np.nan).fillna(method="bfill").fillna(method="ffill")
    z = (r_p / sigma_safe).dropna()

    qz = z.quantile(1 - alpha)
    tail_z = z[z <= qz]
    z_es = tail_z.mean() if len(tail_z) else qz

    # Forecast next sigma
    sigma_next = garch11_forecast_sigma_next(float(r_p.iloc[-1]), float(sigma.iloc[-1]),
                                             alpha_g, beta_g, long_run_var=long_var)

    var_loss = float(-qz * sigma_next * exposure)
    es_loss = float(-z_es * sigma_next * exposure)
    return var_loss, es_loss

def backtest_fhs_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    window_min: int = 50,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
) -> dict:
    """
    Produce a rolling series of FHS VaR thresholds (1-day ahead) and exceptions.
    For each t, fit sigma up to t-1, get z-quantile from past z, forecast sigma_t, compare r_t.
    """
    r_p = pd.Series(returns.values @ weights, index=returns.index, name="r_p").dropna()
    if len(r_p) <= window_min:
        return {
            "r_p": r_p,
            "VaR_FHS": pd.Series(index=r_p.index, dtype=float),
            "exceptions": pd.Series(index=r_p.index, dtype=int),
            "T": 0, "exceedances": 0, "hit_rate": np.nan,
        }

    long_var = float(r_p.var(ddof=1))
    sigma = garch11_filter(r_p, alpha_g=alpha_g, beta_g=beta_g, long_run_var=long_var)
    sigma_safe = sigma.replace(0.0, np.nan).fillna(method="bfill").fillna(method="ffill")

    # Build z up to t-1; rolling quantile of z with at least window_min obs
    z = (r_p / sigma_safe).dropna()
    qz_series = z.rolling(window_min).quantile(1 - alpha).shift(1)

    # One-step-ahead sigma using GARCH recursion
    # sigma_next at t is computed from (t-1) info:
    r_shift = r_p.shift(1)
    sig_shift = sigma_safe.shift(1)
    omega = max(1e-18, (1.0 - alpha_g - beta_g) * long_var)
    sig2_next = omega + alpha_g * (r_shift**2) + beta_g * (sig_shift**2)
    sigma_next = (sig2_next**0.5).replace([np.inf, -np.inf], np.nan)

    # FHS VaR threshold at t (as negative return level)
    var_thresh = -qz_series * sigma_next

    # Exceptions when realized r_p < var_thresh
    exceptions = ((r_p < var_thresh) & var_thresh.notna()).astype(int)

    mask = var_thresh.notna()
    T = int(mask.sum())
    x = int(exceptions[mask].sum())
    hit_rate = (x / T) if T > 0 else np.nan

    return {
        "r_p": r_p,
        "VaR_FHS": var_thresh,
        "exceptions": exceptions,
        "alpha": alpha,
        "window_min": window_min,
        "T": T,
        "exceedances": x,
        "hit_rate": hit_rate,
    }
    
    

def normalize_weights(w: np.ndarray) -> np.ndarray:
    s = float(np.sum(w))
    return (w / s) if s != 0 else w

def var_parametric_normal_parts(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
) -> dict:
    """Parametric VaR (Normal) + marginal & component contributions (Euler)."""
    w = normalize_weights(np.asarray(weights, dtype=float))
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days
    sigma_p = float(np.sqrt(w @ cov @ w))
    mu_p = float(w @ mu)
    z = _N.inv_cdf(alpha)
    var_loss = (-mu_p + z * sigma_p) * exposure
    Sigma_w = cov @ w
    mvar = (-mu + z * (Sigma_w / (sigma_p if sigma_p > 0 else 1e-18))) * exposure
    cvar = w * mvar
    pcontrib = (cvar / var_loss) if var_loss != 0 else np.zeros_like(cvar)
    return {"VaR": float(var_loss), "mu_p": mu_p, "sigma_p": sigma_p,
            "mVaR": mvar, "cVaR": cvar, "pContrib": pcontrib, "cov": cov, "w": w}

def incremental_var_normal(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
) -> dict:
    """
    Incremental VaR (iVaR) and Component VaR under Normal approx.
    iVaR_i ≈ VaR(w) - VaR(w with a tiny reduction on asset i) ≈ w_i * mVaR_i
    Returns dict with portfolio VaR, mVaR (per asset), cVaR (per asset).
    """
    w = np.asarray(weights, dtype=float)
    s = float(w.sum())
    if s == 0:
        return {"VaR": 0.0, "mVaR": np.zeros_like(w), "cVaR": np.zeros_like(w)}

    w = w / s
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days

    sigma_p = float(np.sqrt(w @ cov @ w))
    mu_p = float(w @ mu)
    z = _N.inv_cdf(alpha)

    var_loss = (-mu_p + z * sigma_p) * exposure

    Sigma_w = cov @ w
    # ∂VaR/∂w_i = (-μ_i + z * (Σw)_i / σ_p) * exposure
    mvar = (-mu + z * (Sigma_w / (sigma_p if sigma_p > 0 else 1e-18))) * exposure
    cvar = w * mvar  # iVaR approximation via Euler allocation

    return {"VaR": float(var_loss), "mVaR": mvar, "cVaR": cvar}

# ---------- Risk Parity (Equal Risk Contribution) ----------


def erc_weights_from_cov(
    cov: np.ndarray,
    init: np.ndarray | None = None,
    min_w: float = 0.0,
    max_w: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 5000,
    step: float = 0.5,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """
    SciPy-free multiplicative updates to find Equal-Risk-Contribution (ERC) weights.
    Minimizes dispersion of risk contributions RC_i = w_i * (Σ w)_i.
    - 'step' in (0,1] damps updates for stability.
    - Box constraints via clipping [min_w, max_w], then renormalize to sum=1.
    Returns: (weights, info_dict)
    """
    n = cov.shape[0]
    if init is None:
        w = np.ones(n) / n
    else:
        w = _normalize_weights(np.asarray(init, dtype=float))
        if w.shape[0] != n:
            raise ValueError("init length must match covariance dimension")

    for it in range(1, max_iter + 1):
        Sigma_w = cov @ w
        RC = w * Sigma_w                      # risk contributions to variance
        sigma2 = float(w @ Sigma_w)
        target = sigma2 / n                   # target RC per asset

        # multiplicative update toward equal RC
        update = (target / (RC + eps)) ** step
        w_new = w * update

        # project to bounds + renormalize
        w_new = np.clip(w_new, min_w, max_w)
        if w_new.sum() == 0:
            w_new = np.ones(n) / n
        w_new = _normalize_weights(w_new)

        # convergence check (RC dispersion)
        Sigma_w_new = cov @ w_new
        RC_new = w_new * Sigma_w_new
        disp = float(np.linalg.norm(RC_new / (RC_new.mean() + eps) - 1.0, ord=np.inf))
        w_move = float(np.linalg.norm(w_new - w, 1))

        w = w_new
        if disp < tol or w_move < 1e-10:
            break

    info = {
        "iter": it,
        "rc_dispersion": disp,
        "sigma": float(np.sqrt(sigma2)),
        "RC": RC_new,
    }
    return w, info

def erc_weights(
    returns: pd.DataFrame,
    horizon_days: int = 1,
    init: np.ndarray | None = None,
    min_w: float = 0.0,
    max_w: float = 1.0,
    **kw,
) -> tuple[np.ndarray, dict]:
    """
    Convenience wrapper: compute Σ from returns, scale to horizon, then ERC.
    """
    cov = returns.cov().values * horizon_days
    return erc_weights_from_cov(cov, init=init, min_w=min_w, max_w=max_w, **kw)
