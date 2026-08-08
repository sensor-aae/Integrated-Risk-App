"""Shared fixtures. Puts the repo root on sys.path so `import risklib` works under bare pytest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def returns_3():
    """1,000 days of well-behaved iid normal returns across 3 assets."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    return pd.DataFrame(rng.normal(0, 0.01, size=(1000, 3)),
                        index=idx, columns=["A", "B", "C"])


@pytest.fixture
def weights_3():
    return np.array([0.4, 0.3, 0.3])


@pytest.fixture
def correlated_returns():
    """3 assets with a known correlation structure, for decomposition tests."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=1500, freq="B")
    cov = np.array([[0.0004, 0.00030, -0.00005],
                    [0.00030, 0.0009, -0.00008],
                    [-0.00005, -0.00008, 0.0002]])
    data = rng.multivariate_normal([0.0003, 0.0004, 0.0001], cov, size=1500)
    return pd.DataFrame(data, index=idx, columns=["EQ", "TECH", "BOND"])
