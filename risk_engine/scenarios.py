from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple
from .market import mc_portfolio_loss_from_mu_cov

# Basic helpers
def _portfolio_loss_from_vector(returns: pd.DataFrame, weights: np.ndarray, shock_vector: Dict[str, float], exposure: float = 1.0) -> float:
    cols = returns.columns.tolist()
    v = np.zeros(len(cols), dtype=float)
    for i, c in enumerate(cols):
        if c in shock_vector:
            v[i] = float(shock_vector[c])
    port_ret = float(v @ weights)
    return -port_ret * float(exposure)

# 1) Equities -X% one-day
def scenario_equities_shock(
    returns: pd.DataFrame, weights: np.ndarray,
    equities: Iterable[str], shock: float = -0.05, exposure: float = 1.0
) -> float:
    shock_map = {e: shock for e in equities if e in returns.columns}
    return _portfolio_loss_from_vector(returns, weights, shock_map, exposure=exposure)

# 2) Rates +bp using duration approximation: ΔP/P ≈ -D * Δy
_DEFAULT_DUR = {
    "TLT": 18.0, "IEF": 7.5, "AGG": 6.5, "BND": 6.5, "EDV": 24.0, "ZROZ": 27.0
}
def scenario_rates_bp(
    returns: pd.DataFrame, weights: np.ndarray,
    durations: Dict[str, float] | None = None, bp: float = 200.0, exposure: float = 1.0,
    default_duration: float = 7.0
) -> float:
    dur = dict(_DEFAULT_DUR)
    if durations:
        dur.update({k: float(v) for k, v in durations.items()})
    dy = float(bp) / 10_000.0
    shock_map = {}
    for c in returns.columns:
        D = dur.get(c, default_duration)
        shock_map[c] = -D * dy
    return _portfolio_loss_from_vector(returns, weights, shock_map, exposure=exposure)

# 3) Correlations +X% (MC VaR/ES recompute with bumped correlations, vols constant)
def scenario_corr_bump_mc(
    returns: pd.DataFrame, weights: np.ndarray,
    alpha: float = 0.99, horizon_days: int = 1, exposure: float = 1.0,
    corr_bump_pct: float = 50.0, n_sims: int = 50_000, seed: int = 7
) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Multiply off-diagonal correlations by (1 + corr_bump_pct/100), clip to [-0.99, 0.99],
    keep vols same; recompute MC VaR/ES. Returns ((VaR_base, ES_base), (VaR_stress, ES_stress)).
    """
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days
    sig = np.sqrt(np.diag(cov))
    sig[sig == 0] = 1e-12
    R = cov / np.outer(sig, sig)
    bump = 1.0 + corr_bump_pct / 100.0
    Rb = R.copy()
    n = Rb.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                Rb[i, j] = np.clip(R[i, j] * bump, -0.99, 0.99)
    cov_bumped = np.outer(sig, sig) * Rb

    # base
    base = mc_portfolio_loss_from_mu_cov(mu, cov, weights, alpha=alpha, exposure=exposure, n_sims=int(n_sims), seed=int(seed))
    # stressed
    stressed = mc_portfolio_loss_from_mu_cov(mu, cov_bumped, weights, alpha=alpha, exposure=exposure, n_sims=int(n_sims), seed=int(seed))
    return base, stressed
