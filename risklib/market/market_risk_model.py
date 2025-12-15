from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class MarketRiskConfig:
    alpha: float = 0.99
    window: int = 250
    method: str = "historical"  # historical | parametric | monte_carlo | fhs
    horizon_days: int = 1
    exposure: float = 1.0


# Convenience for Normal
_N = NormalDist()


def _z(alpha: float) -> float:
    return _N.inv_cdf(alpha)


def _phi(x: float) -> float:
    return _N.pdf(x)


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return pd.Series(returns.values @ weights, index=returns.index, name="r_p")


def var_parametric(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
) -> float:
    mu = returns.mean().values
    cov = returns.cov().values
    mu_p = float(weights @ mu) * horizon_days
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * sqrt(horizon_days)
    z = _z(alpha)
    loss_var = (-mu_p + z * sigma_p) * exposure
    return float(loss_var)


def es_parametric(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
) -> float:
    mu = returns.mean().values
    cov = returns.cov().values
    mu_p = float(weights @ mu) * horizon_days
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * sqrt(horizon_days)
    z = _z(alpha)
    es = (-mu_p + sigma_p * (_phi(z) / (1 - alpha))) * exposure
    return float(es)


def var_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
    sqrt_time: bool = True,
) -> float:
    r_p = portfolio_returns(returns, weights)
    if sqrt_time and horizon_days > 1:
        r_p = r_p * sqrt(horizon_days)
    q = r_p.quantile(1 - alpha)
    return float(-q * exposure)


def es_historical(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
    sqrt_time: bool = True,
) -> float:
    r_p = portfolio_returns(returns, weights)
    if sqrt_time and horizon_days > 1:
        r_p = r_p * sqrt(horizon_days)
    var_loss = -r_p.quantile(1 - alpha)
    tail = r_p[r_p <= -var_loss]
    es = -tail.mean() if len(tail) else var_loss
    return float(es * exposure)


def _cov_shrink(cov: np.ndarray, lam: float = 0.01) -> np.ndarray:
    d = np.diag(np.diag(cov))
    return (1 - lam) * cov + lam * d


def var_es_monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
    n_sims: int = 100_000,
    seed: int | None = 42,
    shrink_lambda: float = 0.01,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days
    cov = _cov_shrink(cov, lam=shrink_lambda)

    sims = rng.multivariate_normal(mu, cov, size=int(n_sims), method="cholesky")
    port = sims @ weights

    q = np.quantile(port, 1 - alpha)
    var_loss = float(-q * exposure)

    tail = port[port <= q]
    es_loss = float(-tail.mean() * exposure) if tail.size else var_loss
    return var_loss, es_loss


def garch11_filter(
    r: pd.Series,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
    long_run_var: float | None = None,
    init_var: float | None = None,
) -> pd.Series:
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
        sig2[t] = omega + alpha_g * r2[t - 1] + beta_g * sig2[t - 1]
    sigma = np.sqrt(sig2)
    return pd.Series(sigma, index=r.index, name="sigma_garch")


def garch11_forecast_sigma_next(
    r_last: float,
    sigma_last: float,
    alpha_g: float,
    beta_g: float,
    long_run_var: float,
) -> float:
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
    r_p = pd.Series(returns.values @ weights, index=returns.index, name="r_p").dropna()
    if len(r_p) < 50:
        q = r_p.quantile(1 - alpha)
        var_loss = float(-q * exposure)
        tail = r_p[r_p <= q]
        es_loss = float(-tail.mean() * exposure) if len(tail) else var_loss
        return var_loss, es_loss

    long_var = float(r_p.var(ddof=1))
    sigma = garch11_filter(r_p, alpha_g=alpha_g, beta_g=beta_g, long_run_var=long_var)
    sigma_safe = sigma.replace(0.0, np.nan).bfill().ffill()
    z = (r_p / sigma_safe).dropna()

    qz = z.quantile(1 - alpha)
    tail_z = z[z <= qz]
    z_es = tail_z.mean() if len(tail_z) else qz

    sigma_next = garch11_forecast_sigma_next(
        float(r_p.iloc[-1]), float(sigma.iloc[-1]), alpha_g, beta_g, long_run_var=long_var
    )

    var_loss = float(-qz * sigma_next * exposure)
    es_loss = float(-z_es * sigma_next * exposure)
    return var_loss, es_loss


class MarketRiskModel:
    """
    Validation-ready Market Risk engine.
    Outputs are positive loss numbers: VaR >= 0, ES >= VaR.
    """

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray, config: MarketRiskConfig):
        self.returns = returns.dropna()
        self.weights = np.asarray(weights, dtype=float)
        self.config = config
        self.var_ = None
        self.es_ = None

    def fit(self) -> None:
        a = self.config.alpha
        h = self.config.horizon_days
        e = self.config.exposure
        m = self.config.method

        if m == "historical":
            self.var_ = var_historical(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
            self.es_ = es_historical(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
        elif m == "parametric":
            self.var_ = var_parametric(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
            self.es_ = es_parametric(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
        elif m == "monte_carlo":
            self.var_, self.es_ = var_es_monte_carlo(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
        elif m == "fhs":
            self.var_, self.es_ = fhs_var_es_next(self.returns, self.weights, alpha=a, exposure=e)
        else:
            raise ValueError(f"Unknown method: {m}")

        # Enforce loss convention invariants (model risk sanity)
        self.var_ = float(self.var_)
        self.es_ = float(self.es_)
        if self.var_ < 0:
            raise ValueError("VaR must be non-negative under loss convention.")
        if self.es_ < self.var_:
            raise ValueError("ES must be >= VaR under loss convention.")

    def compute_var(self) -> float:
        if self.var_ is None:
            raise RuntimeError("fit() must be called before compute_var().")
        return self.var_

    def compute_es(self) -> float:
        if self.es_ is None:
            raise RuntimeError("fit() must be called before compute_es().")
        return self.es_

    def assumptions(self) -> list:
        base = ["iid returns", "stationarity within rolling window"]
        if self.config.method == "parametric":
            base.append("normality")
        if self.config.method in ("monte_carlo",):
            base.append("multivariate normal joint distribution")
        if self.config.method in ("fhs",):
            base.append("fixed-parameter GARCH(1,1) volatility filter")
        return base

    def summary(self) -> Dict[str, Any]:
        return {
            "VaR": self.var_,
            "ES": self.es_,
            "alpha": self.config.alpha,
            "window": self.config.window,
            "method": self.config.method,
            "horizon_days": self.config.horizon_days,
            "exposure": self.config.exposure,
            "assumptions": self.assumptions(),
        }
