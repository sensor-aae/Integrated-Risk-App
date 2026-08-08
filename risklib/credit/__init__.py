"""
risklib.credit
==============
Credit Expected Loss: validate -> shock -> compute -> summarise.
"""

from .credit_risk_model import (
    apply_credit_shocks,
    compute_el_table,
    summarize_el,
    validate_and_standardize,
)

__all__ = [
    "validate_and_standardize",
    "apply_credit_shocks",
    "compute_el_table",
    "summarize_el",
]
