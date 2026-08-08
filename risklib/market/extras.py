"""
risklib/market/extras.py
========================
Risk attribution and allocation, under the Normal (parametric) approximation.

  var_parametric_normal_parts : marginal + component VaR (Euler allocation)
  incremental_var             : true incremental VaR (position-removal)
  erc_weights                 : Equal Risk Contribution budgeting

These answer "where is the risk coming from?" rather than "how much is there?",
which is why they live beside the measurement layer rather than inside it. The
migration plan scoped them out of V1; they are retained because portfolio-level
attribution is the output a risk committee actually acts on.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .market_risk_model import normalize_weights

__all__ = [
    "var_parametric_normal_parts",
    "incremental_var",
    "erc_weights_from_cov",
    "erc_weights",
]

_N = NormalDist()


# ---------------------------------------------------------------------------
# Euler decomposition
# ---------------------------------------------------------------------------

def var_parametric_normal_parts(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
) -> Dict:
    """
    Parametric VaR with marginal and component contributions (Euler allocation).

        VaR(w)  = (-w'mu + z * sqrt(w'Sigma w)) * E
        mVaR_i  = dVaR/dw_i = (-mu_i + z * (Sigma w)_i / sigma_p) * E
        cVaR_i  = w_i * mVaR_i

    The decomposition is EXACT, not approximate. VaR(w) is homogeneous of
    degree 1 in w, so Euler's theorem gives sum_i w_i * dVaR/dw_i = VaR(w).
    Directly: sum_i w_i * z*(Sigma w)_i / sigma_p = z * (w'Sigma w) / sigma_p
    = z * sigma_p. Component VaRs therefore sum to portfolio VaR to machine
    precision.

    Why it matters: this is what turns "portfolio VaR is $10,621" into "QQQ is
    47% of your risk on a 25% weight" — the number a risk committee acts on.

    Returns
    -------
    dict with VaR, mu_p, sigma_p, mVaR, cVaR, pContrib, cov, w
    """
    w = normalize_weights(np.asarray(weights, dtype=float))
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days

    sigma_p = float(np.sqrt(w @ cov @ w))
    mu_p = float(w @ mu)
    z = _N.inv_cdf(alpha)

    var_loss = (-mu_p + z * sigma_p) * exposure
    sigma_w = cov @ w
    mvar = (-mu + z * (sigma_w / (sigma_p if sigma_p > 0 else 1e-18))) * exposure
    cvar = w * mvar
    pcontrib = (cvar / var_loss) if var_loss != 0 else np.zeros_like(cvar)

    return {
        "VaR": float(var_loss),
        "mu_p": mu_p,
        "sigma_p": sigma_p,
        "mVaR": mvar,
        "cVaR": cvar,
        "pContrib": pcontrib,
        "cov": cov,
        "w": w,
    }


# ---------------------------------------------------------------------------
# Incremental VaR
# ---------------------------------------------------------------------------

def incremental_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 1,
    exposure: float = 1.0,
) -> Dict:
    """
    TRUE incremental VaR: the change in portfolio VaR from removing a position
    entirely and renormalising the remainder.

        iVaR_i = VaR(w) - VaR(w with position i removed, rest renormalised)

    This is a discrete recomputation, not a derivative. A previous version of
    this codebase returned component VaR under the name "incremental VaR"; the
    two converge only for small positions. On a 25% weight they differ
    materially, so both are now returned side by side and can be compared.

    Returns
    -------
    dict with VaR, iVaR (true), cVaR (Euler component), mVaR (marginal)
    """
    w = normalize_weights(np.asarray(weights, dtype=float))
    n = len(w)

    base = var_parametric_normal_parts(returns, w, alpha, horizon_days, exposure)
    var_full = base["VaR"]

    ivar = np.zeros(n, dtype=float)
    for i in range(n):
        w_ex = w.copy()
        w_ex[i] = 0.0
        if w_ex.sum() <= 0:
            # Removing the only funded position leaves nothing to measure.
            ivar[i] = var_full
            continue
        w_ex = normalize_weights(w_ex)
        var_ex = var_parametric_normal_parts(returns, w_ex, alpha, horizon_days, exposure)["VaR"]
        ivar[i] = var_full - var_ex

    return {
        "VaR": float(var_full),
        "iVaR": ivar,
        "cVaR": base["cVaR"],
        "mVaR": base["mVaR"],
    }


# ---------------------------------------------------------------------------
# Equal Risk Contribution
# ---------------------------------------------------------------------------

def _project_capped_simplex(
    v: np.ndarray,
    min_w: float,
    max_w: float,
    tol: float = 1e-14,
    max_bisect: int = 200,
) -> np.ndarray:
    """
    Project v onto {w : sum(w) = 1, min_w <= w_i <= max_w}.

    Clipping and then renormalising does NOT respect the box: renormalising
    scales every weight up, which can push a clipped weight back above max_w.
    (This is exactly what the previous implementation did, and it produced
    weights of 0.508 under a stated 0.50 cap.)

    The correct projection solves for a single shift theta such that
        w_i = clip(v_i - theta, min_w, max_w)
    sums to 1. sum(w(theta)) is continuous and non-increasing in theta, so a
    bisection converges reliably.
    """
    n = len(v)
    if n * min_w > 1.0 + 1e-12 or n * max_w < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible bounds: {n} assets with min_w={min_w}, max_w={max_w} "
            "cannot produce weights summing to 1."
        )

    lo, hi = float(v.min() - max_w - 1.0), float(v.max() - min_w + 1.0)
    for _ in range(max_bisect):
        theta = 0.5 * (lo + hi)
        s = float(np.clip(v - theta, min_w, max_w).sum())
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = theta
        else:
            hi = theta
    return np.clip(v - theta, min_w, max_w)


def erc_weights_from_cov(
    cov: np.ndarray,
    init: np.ndarray | None = None,
    min_w: float = 0.0,
    max_w: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 5000,
    step: float = 0.5,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict]:
    """
    Equal Risk Contribution weights by damped multiplicative updates.

        RC_i    = w_i * (Sigma w)_i          (contribution to VARIANCE)
        target  = w'Sigma w / n
        w_i    <- w_i * (target / RC_i)^step

    If asset i contributes more than its 1/n share then target/RC_i < 1 and its
    weight shrinks; and conversely. `step` in (0, 1] damps the update — at
    step=1 the iteration can overshoot and oscillate on ill-conditioned
    covariance matrices.

    Implemented directly rather than via scipy.optimize: this is a constrained
    problem where a generic solver needs careful bounds and starting points,
    whereas the multiplicative update is three lines and converges in tens of
    iterations. Convergence is measured on RISK-CONTRIBUTION DISPERSION, which
    is the objective itself rather than a proxy.

    Note: contributions are equalised in VARIANCE space, computed from Sigma
    alone. Percentage-of-VaR contributions additionally include the drift term
    (-mu_i), so they will be close to but not exactly equal.

    Returns
    -------
    (weights, info) with info = iter, rc_dispersion, sigma, RC, converged
    """
    n = cov.shape[0]
    if init is None:
        w = np.ones(n) / n
    else:
        w = normalize_weights(np.asarray(init, dtype=float))
        if w.shape[0] != n:
            raise ValueError("init length must match the covariance dimension.")
        if w.sum() <= 0:
            w = np.ones(n) / n

    # Initialised before the loop so `info` is always well-defined, even if
    # max_iter is 0 or the first iteration breaks immediately.
    sigma_w = cov @ w
    rc = w * sigma_w
    sigma2 = float(w @ sigma_w)
    disp = float("inf")
    it = 0
    converged = False

    for it in range(1, max_iter + 1):
        sigma_w = cov @ w
        rc = w * sigma_w
        sigma2 = float(w @ sigma_w)
        target = sigma2 / n

        w_new = w * (target / (rc + eps)) ** step
        if not np.isfinite(w_new).all() or w_new.sum() <= 0:
            w_new = np.ones(n) / n
        # Exact projection onto the box-constrained simplex: respects min_w and
        # max_w AND sums to 1. Clipping then renormalising satisfies neither.
        w_new = _project_capped_simplex(normalize_weights(w_new), min_w, max_w)

        rc_new = w_new * (cov @ w_new)
        disp = float(np.linalg.norm(rc_new / (rc_new.mean() + eps) - 1.0, ord=np.inf))
        w_move = float(np.linalg.norm(w_new - w, 1))

        w = w_new
        rc = rc_new
        if disp < tol or w_move < 1e-10:
            converged = True
            break

    return w, {
        "iter": it,
        "rc_dispersion": disp,
        "sigma": float(np.sqrt(max(sigma2, 0.0))),
        "RC": rc,
        "converged": converged,
    }


def erc_weights(
    returns: pd.DataFrame,
    horizon_days: int = 1,
    init: np.ndarray | None = None,
    min_w: float = 0.0,
    max_w: float = 1.0,
    **kw,
) -> Tuple[np.ndarray, Dict]:
    """Convenience wrapper: build Sigma from returns, scale to horizon, run ERC."""
    cov = returns.cov().values * horizon_days
    return erc_weights_from_cov(cov, init=init, min_w=min_w, max_w=max_w, **kw)
