import numpy as np, pandas as pd
from risk_engine.market import var_parametric, backtest_var_historical

def fake_returns(n=300, m=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    R = pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=dates, columns=[f"A{i}" for i in range(m)])
    return R

def test_var_scales_with_exposure():
    R = fake_returns()
    w = np.ones(R.shape[1]) / R.shape[1]
    v1 = var_parametric(R, w, alpha=0.99, exposure=1_000_000)
    v2 = var_parametric(R, w, alpha=0.99, exposure=2_000_000)
    assert 1.9 < (v2 / v1) < 2.1

def test_backtest_shapes():
    R = fake_returns()
    w = np.ones(R.shape[1]) / R.shape[1]
    bt = backtest_var_historical(R, w, alpha=0.95, window=100)
    assert "T" in bt and bt["T"] > 0
