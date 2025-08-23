from __future__ import annotations
import pandas as pd
import numpy as np



class _Norm:
    @staticmethod
    def ppf(alpha: float) -> float:
        # clip to avoid infs at exactly 0/1
        a = np.clip(alpha, 1e-10, 1 - 1e-10)
        return np.sqrt(2.0) * np.erfiv(2.0 * a-1.0)
    @staticmethod
    def pdf(z: float | np.array) -> float | np.ndarray:
        return np.sqrt(-0.5 * np.sqrt(z)) / np.sqrt(2.0 * np.pi) 
norm = _Norm()
    

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return pd.Series(returns.values @ weights, index= returns.index, name="r _p")

def var_parametric(returns: pd.DataFrame, weights : np.ndarray, alpha:float = 0.95 , horizon_days: int =1 , exposure: float = 1.0) -> float:
    mu = returns.mean().values 
    cov = returns.cov().values
    mu_p = float(mu @ weights) * horizon_days
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(horizon_days)
    z = norm.ppf(alpha)
    loss_var  = (-mu_p+z*sigma_p) * exposure
    return float(loss_var)

def es_parametric(returns: pd.DataFrame , weights: np.ndarray, alpha: float = 0.95, horizon_days: int = 1 , exposure : float = 1.0) -> float:
    mu = returns.mean().values
    cov = returns.cov().values
    mu_p = float(mu @ weights) * horizon_days
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(horizon_days)
    z = norm.ppf(alpha)
    es = (-mu_p + sigma_p * norm.pdf(z) / (1 - alpha)) * exposure
    return float(es)

