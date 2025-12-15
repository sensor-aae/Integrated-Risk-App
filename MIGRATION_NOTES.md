# Migration Plan (Recruiter-Efficient): From "App With Features" → "Model Risk Validation Sandbox"

## Objective (North Star)
Turn the repo into a **validation-ready risk engine** where:
- the **engine** is the source of truth (VaR/ES/EL),
- the **backtesting module** validates model behavior,
- the **Streamlit app** only displays outputs (no math in the UI).

This migration is intentionally minimal: move only what increases credibility for **pensions / model validation / risk boutiques**.

---

## Step B — Decide What Stays vs What Moves (One-Time)

### ✅ MOVES into `risklib/market/market_risk_model.py` (Risk Measurement Layer)
Move all functions that answer: **"What is the risk number?"**

- Historical VaR / ES logic
- Parametric (Normal) VaR / ES logic
- Monte Carlo VaR / ES logic
- Filtered Historical (GARCH-lite / FHS) VaR / ES logic
- Supporting utilities required by those methods (only if used by the above)

**Rule:** VaR and ES must be reported as **positive loss numbers** (single loss convention).

---

### ✅ MOVES into `risklib/market/backtest.py` (Validation Layer)
Move all functions that answer: **"Did the model work?"**

- Kupiec POF test
- Rolling backtest loop (build VaR series over time)
- Exception counting (losses > VaR)
- Backtest summary outputs (exceptions, expected, p-values)

**Rule:** The app should call backtest functions and display tables/plots—**not implement backtesting itself**.

---

### ✅ STAYS (for now) — but must not recompute risk
These can remain in place while we stabilize the engine:

- Streamlit UI layout and charting (Plotly / Streamlit elements)
- Data loading (e.g., yfinance, CSV demos)

**Rule:** UI/plots may visualize VaR/ES, but must **never recompute** VaR/ES internally.

---

### ❌ REMOVE or DEPRECATE (Out of scope for the North Star)
These reduce clarity and create "doing too much" signals.

- Duplicate VaR implementations with inconsistent conventions
- Any `abs()` band-aids used to fix sign issues
- Any risk calculations performed inside `app/app.py`
- Extra features not required for validation sandbox V1 (e.g., ERC weights, marginal VaR decomposition)

**Rule:** If it’s not part of "measure → validate → document", it moves to `extras.py` or is marked deprecated.

---

## Definition of Done (What a model-risk reviewer expects)
- One loss convention (losses positive)
- ES ≥ VaR (tested)
- 99% VaR ≥ 95% VaR (tested)
- Backtest returns exceptions, expected exceptions, Kupiec p-value
- `app/app.py` calls the engine; it does not compute risk

# Migration Map: risk_engine/market.py → risklib/

## Move to `risklib/market/market_risk_model.py` (Measurement)
- portfolio_returns
- var_parametric
- es_parametric
- var_historical
- es_historical
- _cov_shrink
- var_es_monte_carlo
- garch11_filter
- garch11_forecast_sigma_next
- fhs_var_es_next

## Move to `risklib/market/backtest.py` (Validation)
- kupiec_pof
- backtest_var_historical
- backtest_fhs_var
- (optional) backtest_var utility wrapper

## Move to `risklib/market/extras.py` (Out of Scope for V1)
- var_parametric_normal_parts
- incremental_var_normal
- erc_weights_from_cov
- erc_weights
- normalize_weights / _normalize_weights (keep one)
