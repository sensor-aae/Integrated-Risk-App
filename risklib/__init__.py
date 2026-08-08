"""
risklib
=======
Validation-grade market and credit risk engine.

`risklib` is the single source of truth for all modelling, estimation and
validation logic. The Streamlit application in `app/` imports from here and
computes nothing itself.

    risklib.data    : price ingestion and return construction
    risklib.market  : VaR/ES, backtesting, GARCH, attribution, scenarios
    risklib.credit  : Expected Loss
"""

__version__ = "0.2.0"

from . import credit, data, market

__all__ = ["data", "market", "credit", "__version__"]
