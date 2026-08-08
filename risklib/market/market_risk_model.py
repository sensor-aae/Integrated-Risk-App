"""
risklib/market/market_risk_model.py
==================================
Market risk measurement: VaR and Expected Shortfall under four methodologies.

Loss convention (non-negotiable)
--------------------------------
Every quantity leaving this module is a POSITIVE LOSS amount:
    VaR >= 0
    ES  >= VaR
`MarketRiskModel.fit()` enforces both as runtime invariants and raises if
either is violated. Sign conventions are the classic silent-failure mode in
risk code — one estimator returns a signed return, another returns a loss,
someone patches it with abs(), and a confidence level is quietly wrong in one
branch forever. Encoding the convention as an assertion turns a methodology
error into an exception at fit time instead of a plausible-looking number in a
report.

Methods
-------
  historical   : empirical (1-alpha) quantile of realised portfolio returns
  parametric   : closed-form Normal
  monte_carlo  : multivariate Normal simulation with covariance shrinkage
  fhs          : Filtered Historical Simulation (GARCH-standardised residuals)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from statistics import NormalDist
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .garch import garch11_filter, garch11_forecast_next

__all__ = [
    "MarketRiskConfig",
    "MarketRiskModel",
    "portfolio_returns",
    "normalize_weights",
    "var_parametric",
    "es_parametric",
    "var_historical",
    "es_historical",
    "var_es_monte_carlo",
    "mc_portfolio_loss_from_mu_cov",
    "fhs_var_es_next",
    "cov_shrink",
    "METHODS",
]

METHODS = ("historical", "parametric", "monte_carlo", "fhs")

# stdlib Normal: keeps the core VaR math free of any SciPy dependency
_N = NormalDist()


def _z(alpha: float) -> float:
    """Standard Normal quantile, Phi^{-1}(alpha)."""
    return _N.inv_cdf(alpha)


def _phi(x: float) -> float:
    """Standard Normal pdf, phi(x)."""
    return _N.pdf(x)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MarketRiskConfig:
    """
    Complete specification of a market risk calculation.

    Every knob that changes a number lives here. That is deliberate: the config
    is one serialisable object you can log, diff and attach to a result, which
    is the difference between "we ran a VaR" and "we ran *this* VaR".

    Core
    ----
    alpha        : confidence level (e.g. 0.99)
    window       : rolling window for backtesting, in trading days
    method       : one of METHODS
    horizon_days : holding period; mu scales by h, sigma by sqrt(h)
    exposure     : portfolio notional; results scale linearly

    Monte Carlo
    -----------
    n_sims        : simulation count
    seed          : RNG seed (fixed by default so results are reproducible)
    shrink_lambda : covariance shrinkage toward the diagonal

    FHS / GARCH
    -----------
    fit_garch : True  -> estimate (omega, alpha, beta) by MLE
                False -> variance targeting on alpha_g / beta_g below
    alpha_g   : ARCH parameter used when fit_garch=False
    beta_g    : GARCH parameter used when fit_garch=False
    """

    alpha: float = 0.99
    window: int = 250
    method: str = "historical"
    horizon_days: int = 1
    exposure: float = 1.0

    n_sims: int = 100_000
    seed: int | None = 42
    shrink_lambda: float = 0.01

    fit_garch: bool = False
    alpha_g: float = 0.05
    beta_g: float = 0.94

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}. Expected one of {METHODS}.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}.")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1; got {self.horizon_days}.")
        if self.exposure < 0:
            raise ValueError(f"exposure must be non-negative; got {self.exposure}.")
        if self.window < 2:
            raise ValueError(f"window must be >= 2; got {self.window}.")
        if not 0.0 <= self.shrink_lambda <= 1.0:
            raise ValueError(f"shrink_lambda must be in [0, 1]; got {self.shrink_lambda}.")
        if self.alpha_g + self.beta_g >= 1.0:
            raise ValueError(
                f"GARCH parameters must satisfy alpha + beta < 1 for stationarity; "
                f"got {self.alpha_g} + {self.beta_g} = {self.alpha_g + self.beta_g}."
            )


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Scale weights to sum to 1. Returns unchanged if the sum is zero."""
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    return (w / s) if s != 0 else w


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Project the multivariate return matrix onto portfolio weights: r_p = R @ w."""
    return pd.Series(returns.values @ np.asarray(weights, dtype=float),
                     index=returns.index, name="r_p")


def _mu_cov(returns: pd.DataFrame, horizon_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizon-scaled mean vector and covariance matrix.

    Under i.i.d. returns the h-day return is a sum of h daily returns, so means
    add and variances add: mu * h and Sigma * h (hence sigma * sqrt(h)). This is
    the square-root-of-time assumption, and it fails whenever volatility is
    autocorrelated — which it always is. Documented as a known limitation.
    """
    mu = returns.mean().values * horizon_days
    cov = returns.cov().values * horizon_days
    return mu, cov


def cov_shrink(cov: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """
    Shrink a covariance matrix toward its diagonal.

        cov_s = (1 - lam) * cov + lam * diag(diag(cov))

    On the diagonal this is (1-lam)*s_ii + lam*s_ii = s_ii, so VARIANCES ARE
    UNCHANGED. Off the diagonal it is (1-lam)*s_ij, so every correlation is
    multiplied by (1 - lam).

    Why: the sample covariance is near-singular when the asset count is large
    relative to the sample, or when two series are nearly collinear (SPY/QQQ at
    rho = 0.86 is already close). A near-singular Sigma makes the Cholesky
    factorisation fail or return garbage. Shrinking correlations pulls the
    matrix away from the boundary of the PSD cone at negligible cost to the
    risk estimate.
    """
    d = np.diag(np.diag(cov))
    return (1.0 - lam) * cov + lam * d


# ---------------------------------------------------------------------------
# Parametric (Normal)
# ---------------------------------------------------------------------------

def var_parametric(returns, weights, alpha=0.95, horizon_days=1, exposure=1.0) -> float:
    """
    Closed-form Normal VaR.

    With R ~ N(mu_p, sigma_p^2) and loss L = -R, the alpha-quantile of L is
        VaR = -mu_p + z_alpha * sigma_p
    """
    w = np.asarray(weights, dtype=float)
    mu, cov = _mu_cov(returns, horizon_days)
    mu_p = float(w @ mu)
    sigma_p = float(np.sqrt(w @ cov @ w))
    return float((-mu_p + _z(alpha) * sigma_p) * exposure)


def es_parametric(returns, weights, alpha=0.95, horizon_days=1, exposure=1.0) -> float:
    """
    Closed-form Normal Expected Shortfall.

        ES = E[L | L > VaR] = -mu_p + sigma_p * phi(z_alpha) / (1 - alpha)

    ES is reported alongside VaR because VaR gives the threshold but says
    nothing about severity beyond it, and is not subadditive. ES is coherent
    and is the measure FRTB moved to.
    """
    w = np.asarray(weights, dtype=float)
    mu, cov = _mu_cov(returns, horizon_days)
    mu_p = float(w @ mu)
    sigma_p = float(np.sqrt(w @ cov @ w))
    z = _z(alpha)
    return float((-mu_p + sigma_p * (_phi(z) / (1.0 - alpha))) * exposure)


# ---------------------------------------------------------------------------
# Historical simulation
# ---------------------------------------------------------------------------

def _scaled_portfolio_returns(returns, weights, horizon_days, sqrt_time=True) -> pd.Series:
    r_p = portfolio_returns(returns, weights)
    if sqrt_time and horizon_days > 1:
        r_p = r_p * sqrt(horizon_days)
    return r_p


def var_historical(returns, weights, alpha=0.95, horizon_days=1,
                   exposure=1.0, sqrt_time=True) -> float:
    """
    Empirical (1-alpha) quantile of realised portfolio returns, sign-flipped.

    Makes no distributional assumption: fat tails, skew and jumps are captured
    to the extent they appear in the sample. That is also the weakness — it
    cannot produce a loss it has never observed.
    """
    r_p = _scaled_portfolio_returns(returns, weights, horizon_days, sqrt_time)
    return float(-r_p.quantile(1.0 - alpha) * exposure)


def es_historical(returns, weights, alpha=0.95, horizon_days=1,
                  exposure=1.0, sqrt_time=True) -> float:
    """
    Mean of observations at or beyond the VaR threshold.

    Falls back to VaR itself when the tail is empty (very high alpha on a short
    sample), which keeps the ES >= VaR invariant intact instead of returning NaN.
    """
    r_p = _scaled_portfolio_returns(returns, weights, horizon_days, sqrt_time)
    q = r_p.quantile(1.0 - alpha)
    tail = r_p[r_p <= q]
    es = -tail.mean() if len(tail) else -q
    return float(es * exposure)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def mc_portfolio_loss_from_mu_cov(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    exposure: float = 1.0,
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> Tuple[float, float]:
    """
    THE Monte Carlo core. Simulates multivariate Normal returns for a given
    (mu, Sigma) already scaled to the target horizon, and returns (VaR, ES) as
    positive losses.

    Factored out from `var_es_monte_carlo` because every stress scenario works
    by modifying mu or Sigma and re-simulating — covariance scaling, historical
    window replay, correlation bumping. One simulator, many scenarios.

    An explicit seed makes results reproducible, which is non-negotiable if a
    validator has to reproduce your numbers. `method="cholesky"` is materially
    faster than the default SVD path and is exact when Sigma is positive
    definite; if it is not, we fall back to the eigenvalue-based path rather
    than raising.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    try:
        sims = rng.multivariate_normal(mu, cov, size=int(n_sims), method="cholesky")
    except np.linalg.LinAlgError:
        sims = rng.multivariate_normal(mu, cov, size=int(n_sims), method="eigh")

    port = sims @ w
    q = float(np.quantile(port, 1.0 - alpha))
    var_loss = float(-q * exposure)
    tail = port[port <= q]
    es_loss = float(-tail.mean() * exposure) if tail.size else var_loss
    return var_loss, es_loss


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
    """Monte Carlo VaR/ES estimated from the sample mu and Sigma."""
    mu, cov = _mu_cov(returns, horizon_days)
    cov = cov_shrink(cov, lam=shrink_lambda)
    return mc_portfolio_loss_from_mu_cov(
        mu, cov, weights, alpha=alpha, exposure=exposure, n_sims=n_sims, seed=seed
    )


# ---------------------------------------------------------------------------
# Filtered Historical Simulation
# ---------------------------------------------------------------------------

def fhs_var_es_next(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    exposure: float = 1.0,
    alpha_g: float = 0.05,
    beta_g: float = 0.94,
    fit_garch: bool = False,
    min_obs: int = 50,
    return_info: bool = False,
):
    """
    Filtered Historical Simulation VaR/ES for t+1 (one-day horizon).

    Procedure
    ---------
      1. r_p = R @ w
      2. filter to obtain sigma_t
      3. standardise: z_t = r_t / sigma_t
      4. q_z = empirical (1-alpha) quantile of z; tail mean of z for ES
      5. forecast sigma_{t+1} and rescale:
             VaR = -q_z * sigma_{t+1} * exposure

    Why this beats both parents. Historical simulation keeps the empirical tail
    shape but treats a calm 2017 day and a panicked March-2020 day as equally
    informative. Parametric responds to current volatility but forces Normal
    tails. FHS keeps the empirical (fat, skewed) tail shape *of the
    standardised residuals* while letting the scale track today's volatility.

    Below `min_obs` observations the GARCH filter is noise, so the function
    degrades to plain historical simulation.
    """
    r_p = portfolio_returns(returns, weights).dropna()

    if len(r_p) < min_obs:
        q = r_p.quantile(1.0 - alpha)
        var_loss = float(-q * exposure)
        tail = r_p[r_p <= q]
        es_loss = float(-tail.mean() * exposure) if len(tail) else var_loss
        info = {"source": "insufficient_history", "n_obs": len(r_p)}
        return (var_loss, es_loss, info) if return_info else (var_loss, es_loss)

    sigma, info = garch11_filter(r_p, fit=fit_garch, alpha_g=alpha_g, beta_g=beta_g)

    # Guard against a degenerate sigma before dividing.
    sigma_safe = sigma.replace(0.0, np.nan).bfill().ffill()
    z = (r_p / sigma_safe).dropna()

    qz = float(z.quantile(1.0 - alpha))
    tail_z = z[z <= qz]
    z_es = float(tail_z.mean()) if len(tail_z) else qz

    sigma_next = garch11_forecast_next(
        float(r_p.iloc[-1]), float(sigma.iloc[-1]),
        info["omega"], info["alpha_g"], info["beta_g"],
    )

    var_loss = float(-qz * sigma_next * exposure)
    es_loss = float(-z_es * sigma_next * exposure)
    info = {**info, "sigma_next": sigma_next, "q_z": qz, "z_es": z_es}
    return (var_loss, es_loss, info) if return_info else (var_loss, es_loss)


# ---------------------------------------------------------------------------
# Model object
# ---------------------------------------------------------------------------

class MarketRiskModel:
    """
    Validation-ready market risk engine.

    Usage
    -----
        cfg   = MarketRiskConfig(alpha=0.99, method="fhs", exposure=1_000_000)
        model = MarketRiskModel(returns, weights, cfg)
        model.fit()
        model.compute_var(), model.compute_es(), model.summary()
    """

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray, config: MarketRiskConfig):
        self.returns = returns.dropna()
        self.weights = np.asarray(weights, dtype=float)
        self.config = config
        self.var_: float | None = None
        self.es_: float | None = None
        self.fit_info_: Dict[str, Any] = {}

        if self.returns.empty:
            raise ValueError("`returns` is empty after dropping NaN rows.")
        if self.weights.shape[0] != self.returns.shape[1]:
            raise ValueError(
                f"weights length ({self.weights.shape[0]}) does not match the number "
                f"of assets in returns ({self.returns.shape[1]})."
            )

    def fit(self) -> "MarketRiskModel":
        cfg = self.config
        a, h, e, m = cfg.alpha, cfg.horizon_days, cfg.exposure, cfg.method

        if m == "historical":
            self.var_ = var_historical(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
            self.es_ = es_historical(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)

        elif m == "parametric":
            self.var_ = var_parametric(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)
            self.es_ = es_parametric(self.returns, self.weights, alpha=a, horizon_days=h, exposure=e)

        elif m == "monte_carlo":
            # Config values are threaded through — previously these were the
            # function defaults regardless of what the caller configured.
            self.var_, self.es_ = var_es_monte_carlo(
                self.returns, self.weights, alpha=a, horizon_days=h, exposure=e,
                n_sims=cfg.n_sims, seed=cfg.seed, shrink_lambda=cfg.shrink_lambda,
            )

        elif m == "fhs":
            # FHS is a one-step-ahead forecast: horizon_days does not apply.
            self.var_, self.es_, self.fit_info_ = fhs_var_es_next(
                self.returns, self.weights, alpha=a, exposure=e,
                alpha_g=cfg.alpha_g, beta_g=cfg.beta_g,
                fit_garch=cfg.fit_garch, return_info=True,
            )
        else:
            raise ValueError(f"Unknown method: {m}")

        self.var_ = float(self.var_)
        self.es_ = float(self.es_)

        # Loss-convention invariants
        if self.var_ < 0:
            raise ValueError(
                f"VaR must be non-negative under the loss convention; got {self.var_:.6g} "
                f"(method={m}, alpha={a})."
            )
        if self.es_ < self.var_:
            raise ValueError(
                f"ES must be >= VaR under the loss convention; got ES={self.es_:.6g} "
                f"< VaR={self.var_:.6g} (method={m}, alpha={a})."
            )
        return self

    def compute_var(self) -> float:
        if self.var_ is None:
            raise RuntimeError("fit() must be called before compute_var().")
        return self.var_

    def compute_es(self) -> float:
        if self.es_ is None:
            raise RuntimeError("fit() must be called before compute_es().")
        return self.es_

    def assumptions(self) -> list:
        """
        Assumptions actually in force for this configuration.

        Branches on fit_garch so an MLE-estimated filter is not reported as a
        fixed-parameter one.
        """
        base = ["iid returns", "stationarity within estimation window"]
        m = self.config.method

        if self.config.horizon_days > 1 and m in ("historical", "parametric", "monte_carlo"):
            base.append("square-root-of-time horizon scaling")
        if m == "parametric":
            base.append("normality of portfolio returns")
        if m == "monte_carlo":
            base += ["multivariate normal joint distribution",
                     f"covariance shrinkage lambda={self.config.shrink_lambda}"]
        if m == "fhs":
            if self.config.fit_garch:
                base.append("GARCH(1,1) volatility filter, parameters estimated by MLE")
                if self.fit_info_.get("source") == "fixed":
                    base.append("MLE requested but fell back to fixed parameters")
            else:
                base.append(
                    f"GARCH(1,1) volatility filter, fixed parameters "
                    f"(alpha={self.config.alpha_g}, beta={self.config.beta_g})"
                )
            base.append("zero conditional mean")
        return base

    def summary(self) -> Dict[str, Any]:
        """
        VaR, ES, the full configuration, and the assumptions in force.

        The assumptions travel with the result rather than living only in a
        README someone forgot to update. A risk number without its assumptions
        is not a deliverable.
        """
        out: Dict[str, Any] = {
            "VaR": self.var_,
            "ES": self.es_,
            "config": asdict(self.config),
            "assumptions": self.assumptions(),
            "n_obs": int(len(self.returns)),
            "n_assets": int(self.returns.shape[1]),
        }
        # Flattened for backwards compatibility with existing callers.
        out.update({k: getattr(self.config, k)
                    for k in ("alpha", "window", "method", "horizon_days", "exposure")})
        if self.fit_info_:
            out["fit_info"] = self.fit_info_
        return out
