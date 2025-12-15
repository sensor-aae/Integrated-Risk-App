from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd


@dataclass
class MarketRiskConfig:
    alpha: float = 0.99
    window: int = 250
    method: str = "historical"  # historical | parametric | monte_carlo | fhs


class MarketRiskModel:
    """
    Market Risk Model for VaR / ES estimation.

    Loss convention:
        - Losses are positive
        - VaR and ES are reported as positive numbers
    """

    def __init__(self, returns: pd.Series, config: MarketRiskConfig):
        self.returns = returns.dropna()
        self.config = config

        # Enforce loss convention
        self.losses = -self.returns

        self.var_ = None
        self.es_ = None

    # ---------- public API ----------

    def fit(self) -> None:
        if self.config.method == "historical":
            self._fit_historical()
        elif self.config.method == "parametric":
            self._fit_parametric()
        elif self.config.method == "monte_carlo":
            self._fit_monte_carlo()
        elif self.config.method == "fhs":
            self._fit_filtered_historical()
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def compute_var(self) -> float:
        if self.var_ is None:
            raise RuntimeError("Model must be fitted before computing VaR")
        return self.var_

    def compute_es(self) -> float:
        if self.es_ is None:
            raise RuntimeError("Model must be fitted before computing ES")
        return self.es_

    def summary(self) -> Dict[str, Any]:
        return {
            "VaR": self.var_,
            "ES": self.es_,
            "alpha": self.config.alpha,
            "window": self.config.window,
            "method": self.config.method,
            "assumptions": self.assumptions(),
        }

    def assumptions(self) -> list:
        assumptions = ["iid returns", "stationarity within rolling window"]
        if self.config.method == "parametric":
            assumptions.append("normality")
        return assumptions

    # ---------- model implementations ----------

    def _fit_historical(self):
        window_losses = self.losses[-self.config.window :]
        self.var_ = np.quantile(window_losses, self.config.alpha)
        self.es_ = window_losses[window_losses >= self.var_].mean()

    def _fit_parametric(self):
        from scipy.stats import norm

        mu = self.losses.mean()
        sigma = self.losses.std(ddof=1)

        z = norm.ppf(self.config.alpha)
        self.var_ = mu + z * sigma
        self.es_ = mu + sigma * norm.pdf(z) / (1 - self.config.alpha)

    def _fit_monte_carlo(self):
        simulated = np.random.normal(
            self.losses.mean(),
            self.losses.std(ddof=1),
            size=100_000,
        )
        self.var_ = np.quantile(simulated, self.config.alpha)
        self.es_ = simulated[simulated >= self.var_].mean()

    def _fit_filtered_historical(self):
        # Placeholder: plug in existing GARCH-lite logic here
        filtered_losses = self.losses
        self.var_ = np.quantile(filtered_losses, self.config.alpha)
        self.es_ = filtered_losses[filtered_losses >= self.var_].mean()
