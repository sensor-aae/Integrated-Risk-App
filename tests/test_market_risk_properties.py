import numpy as np
import pandas as pd
from risklib.market.market_risk_model import MarketRiskModel, MarketRiskConfig


def test_es_ge_var():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(1000, 3)), columns=["A", "B", "C"])
    w = np.array([0.4, 0.3, 0.3])

    m = MarketRiskModel(returns, w, MarketRiskConfig(alpha=0.99, method="historical"))
    m.fit()

    assert m.compute_var() >= 0
    assert m.compute_es() >= m.compute_var()


def test_var_monotonicity():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(1000, 3)), columns=["A", "B", "C"])
    w = np.array([0.5, 0.2, 0.3])

    m95 = MarketRiskModel(returns, w, MarketRiskConfig(alpha=0.95, method="historical"))
    m99 = MarketRiskModel(returns, w, MarketRiskConfig(alpha=0.99, method="historical"))
    m95.fit()
    m99.fit()

    assert m99.compute_var() >= m95.compute_var()
