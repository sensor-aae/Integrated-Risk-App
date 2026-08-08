"""
risklib/market/scenarios.py
===========================
Stress testing and deterministic scenario analysis.

Merged from the former `risk_engine/stress.py` and `risk_engine/scenarios.py`,
which had drifted into two incompatible conventions for the same question.

Convention for unshocked instruments
------------------------------------
Instruments not named in a shock are held FLAT (zero return). This is the
standard "everything else unchanged" reading of a stress scenario, and it is
now applied uniformly.

The previous `apply_single_name_shocks` instead carried each unshocked
instrument's LAST OBSERVED RETURN into the scenario, so a single-name stress
was contaminated by whatever the rest of the book happened to do on the final
day of the sample. That behaviour is still reachable via `base="last"`, but it
is no longer the default and the choice is now explicit and reported.

Scenario types
--------------
  single-name shocks       — direct moves in return space
  rate shocks              — duration approximation, dP/P ~ -D * dy
  covariance scaling       — uniform volatility stress
  correlation bumps        — diversification-breakdown stress
  historical window replay — re-run today's book through a past regime
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .market_risk_model import mc_portfolio_loss_from_mu_cov

__all__ = [
    "shock_vector",
    "portfolio_loss_from_shocks",
    "scenario_single_name",
    "scenario_equities_shock",
    "scenario_rates_bp",
    "scale_covariance",
    "scenario_covariance_scale",
    "scenario_corr_bump_mc",
    "historical_window_mu_cov",
    "scenario_historical_replay",
    "DEFAULT_DURATIONS",
]

# Modified duration, years. Used only for instruments explicitly identified as
# rate-sensitive; everything else defaults to zero duration (see scenario_rates_bp).
DEFAULT_DURATIONS: Dict[str, float] = {
    "TLT": 18.0, "EDV": 24.0, "ZROZ": 27.0, "IEF": 7.5,
    "AGG": 6.5, "BND": 6.5, "SHY": 1.9, "TIP": 6.8, "LQD": 8.4, "HYG": 3.5,
}


# ---------------------------------------------------------------------------
# Core shock application
# ---------------------------------------------------------------------------

def shock_vector(
    returns: pd.DataFrame,
    shocks: Dict[str, float],
    base: str = "flat",
) -> pd.Series:
    """
    Build a one-period shock vector aligned to `returns.columns`.

    Parameters
    ----------
    base : "flat" -> unshocked instruments return 0.0   (default, standard)
           "last" -> unshocked instruments keep their last observed return

    Returns a Series indexed by ticker, so downstream code cannot silently
    misalign it against the weight vector.
    """
    if base not in ("flat", "last"):
        raise ValueError(f"base must be 'flat' or 'last'; got {base!r}.")

    if base == "last":
        v = returns.iloc[-1].astype(float).copy()
    else:
        v = pd.Series(0.0, index=returns.columns, dtype=float)

    unknown = set(shocks) - set(returns.columns)
    if unknown:
        raise KeyError(f"Shocked tickers not present in returns: {sorted(unknown)}")

    for k, val in shocks.items():
        v[k] = float(val)
    return v


def portfolio_loss_from_shocks(
    returns: pd.DataFrame,
    weights: np.ndarray,
    shocks: Dict[str, float],
    exposure: float = 1.0,
    base: str = "flat",
) -> float:
    """Portfolio loss (positive) implied by a shock vector."""
    v = shock_vector(returns, shocks, base=base)
    port_ret = float(v.values @ np.asarray(weights, dtype=float))
    return float(-port_ret * exposure)


def scenario_single_name(
    returns: pd.DataFrame,
    weights: np.ndarray,
    shocks: Dict[str, float],
    exposure: float = 1.0,
    base: str = "flat",
) -> Dict:
    """
    Single-name (or multi-name) shock in return space.

    Returns the loss plus the shock vector actually applied and the base
    convention used, so the scenario is self-documenting in an exported report.
    """
    v = shock_vector(returns, shocks, base=base)
    port_ret = float(v.values @ np.asarray(weights, dtype=float))
    return {
        "loss": float(-port_ret * exposure),
        "portfolio_return": port_ret,
        "shock_vector": v,
        "base": base,
    }


def scenario_equities_shock(
    returns: pd.DataFrame,
    weights: np.ndarray,
    equities: Iterable[str],
    shock: float = -0.05,
    exposure: float = 1.0,
) -> float:
    """Uniform shock applied to a named set of equity instruments; others flat."""
    present = [e for e in equities if e in returns.columns]
    return portfolio_loss_from_shocks(
        returns, weights, {e: shock for e in present}, exposure=exposure, base="flat"
    )


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def scenario_rates_bp(
    returns: pd.DataFrame,
    weights: np.ndarray,
    durations: Dict[str, float] | None = None,
    bp: float = 200.0,
    exposure: float = 1.0,
    default_duration: float = 0.0,
    use_default_map: bool = True,
) -> Dict:
    """
    Parallel interest rate shock via the first-order duration approximation:

        dP/P ~ -D * dy

    `default_duration` is 0.0. This is a deliberate correction: the previous
    implementation defaulted every unmapped instrument to 7.0 years, so a +200bp
    scenario knocked roughly 14% off SPY. A parallel rate shock does not do that
    to equities. Instruments now contribute only if a duration is supplied for
    them, explicitly or through DEFAULT_DURATIONS.

    Limitation: the approximation is linear and ignores convexity. At 200bp on
    an 18-year-duration bond the omitted second-order term is material and the
    linear estimate OVERSTATES the loss. Stated rather than silently absorbed.

    Returns the loss plus the duration map actually used.
    """
    dur: Dict[str, float] = dict(DEFAULT_DURATIONS) if use_default_map else {}
    if durations:
        dur.update({k: float(v) for k, v in durations.items()})

    dy = float(bp) / 10_000.0
    applied = {c: dur.get(c, default_duration) for c in returns.columns}
    shocks = {c: -D * dy for c, D in applied.items() if D != 0.0}

    return {
        "loss": portfolio_loss_from_shocks(returns, weights, shocks,
                                           exposure=exposure, base="flat"),
        "durations_used": applied,
        "bp": bp,
        "dy": dy,
    }


# ---------------------------------------------------------------------------
# Covariance stresses
# ---------------------------------------------------------------------------

def scale_covariance(cov: np.ndarray, scale: float) -> np.ndarray:
    """
    Uniformly scale a covariance matrix.

    Multiplying Sigma by k multiplies every VOLATILITY by sqrt(k) and leaves
    all CORRELATIONS unchanged. So "covariance x2" means "volatility x1.41,
    same correlation structure" — worth stating explicitly, because a reader
    will otherwise assume x2 doubles the vol.
    """
    return np.asarray(cov, dtype=float) * float(scale)


def scenario_covariance_scale(
    returns: pd.DataFrame,
    weights: np.ndarray,
    scale: float = 2.0,
    alpha: float = 0.99,
    horizon_days: int = 1,
    exposure: float = 1.0,
    n_sims: int = 50_000,
    seed: int = 7,
) -> Dict:
    """Re-run Monte Carlo VaR/ES with a uniformly scaled covariance matrix."""
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days

    base = mc_portfolio_loss_from_mu_cov(mu, cov, weights, alpha=alpha,
                                         exposure=exposure, n_sims=n_sims, seed=seed)
    stressed = mc_portfolio_loss_from_mu_cov(mu, scale_covariance(cov, scale), weights,
                                             alpha=alpha, exposure=exposure,
                                             n_sims=n_sims, seed=seed)
    return {
        "base": {"VaR": base[0], "ES": base[1]},
        "stressed": {"VaR": stressed[0], "ES": stressed[1]},
        "delta_VaR": stressed[0] - base[0],
        "vol_multiplier": float(np.sqrt(scale)),
        "cov_scale": scale,
    }


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-12) -> Tuple[np.ndarray, bool]:
    """
    Project a symmetric matrix onto the PSD cone by clipping eigenvalues.

    Returns (projected, was_adjusted).
    """
    sym = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= floor:
        return sym, False
    vals_clipped = np.clip(vals, floor, None)
    return vecs @ np.diag(vals_clipped) @ vecs.T, True


def scenario_corr_bump_mc(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    horizon_days: int = 1,
    exposure: float = 1.0,
    corr_bump_pct: float = 50.0,
    n_sims: int = 50_000,
    seed: int = 7,
) -> Dict:
    """
    Correlation-breakdown stress: multiply off-diagonal correlations by
    (1 + corr_bump_pct/100), clip to [-0.99, 0.99], hold volatilities constant,
    and re-simulate.

    Holding vols constant isolates the diversification channel. In a crisis
    correlations converge toward 1 and a "diversified" portfolio turns out to
    be one bet; this scenario measures exactly that, uncontaminated by a
    simultaneous volatility move.

    The same seed is used for base and stressed runs, so the difference is a
    pure covariance effect rather than Monte Carlo noise.

    Element-wise correlation bumping does NOT preserve positive semi-definiteness.
    The bumped matrix is therefore projected back onto the PSD cone via
    eigenvalue clipping, and `psd_adjusted` reports whether that was necessary —
    a large bump that requires adjustment is a signal the scenario is straining
    the correlation structure.
    """
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days

    sig = np.sqrt(np.diag(cov))
    sig[sig == 0] = 1e-12
    corr = cov / np.outer(sig, sig)

    bump = 1.0 + corr_bump_pct / 100.0
    corr_b = np.clip(corr * bump, -0.99, 0.99)
    np.fill_diagonal(corr_b, 1.0)

    corr_b, psd_adjusted = _nearest_psd(corr_b)
    np.fill_diagonal(corr_b, 1.0)
    cov_bumped = np.outer(sig, sig) * corr_b

    base = mc_portfolio_loss_from_mu_cov(mu, cov, weights, alpha=alpha,
                                         exposure=exposure, n_sims=n_sims, seed=seed)
    stressed = mc_portfolio_loss_from_mu_cov(mu, cov_bumped, weights, alpha=alpha,
                                             exposure=exposure, n_sims=n_sims, seed=seed)
    return {
        "base": {"VaR": base[0], "ES": base[1]},
        "stressed": {"VaR": stressed[0], "ES": stressed[1]},
        "delta_VaR": stressed[0] - base[0],
        "corr_bump_pct": corr_bump_pct,
        "psd_adjusted": psd_adjusted,
    }


# ---------------------------------------------------------------------------
# Historical replay
# ---------------------------------------------------------------------------

def historical_window_mu_cov(
    returns: pd.DataFrame,
    start: str,
    end: str,
    min_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance from a historical sub-window (inclusive dates)."""
    sub = returns.loc[start:end]
    if len(sub) < min_obs:
        raise ValueError(
            f"Selected window has {len(sub)} observations; need at least {min_obs} "
            "to estimate mu and Sigma."
        )
    return sub.mean().values, sub.cov().values


def scenario_historical_replay(
    returns: pd.DataFrame,
    weights: np.ndarray,
    start: str,
    end: str,
    alpha: float = 0.99,
    horizon_days: int = 1,
    exposure: float = 1.0,
    n_sims: int = 50_000,
    seed: int = 11,
) -> Dict:
    """
    Re-run today's portfolio through a past statistical regime: keep the current
    weights, adopt the selected window's mu and Sigma.
    """
    mu_w, cov_w = historical_window_mu_cov(returns, start, end)
    var_h, es_h = mc_portfolio_loss_from_mu_cov(
        mu_w * horizon_days, cov_w * horizon_days, weights,
        alpha=alpha, exposure=exposure, n_sims=n_sims, seed=seed,
    )
    return {
        "VaR": var_h,
        "ES": es_h,
        "start": str(start),
        "end": str(end),
        "n_obs": int(len(returns.loc[start:end])),
    }
