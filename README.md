# Integrated Risk App (Python)
 
**Validation-Grade Market & Credit Risk Engine**
 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://integrated-risk-app.onrender.com/)
 
---
 
## 🎯 Project Objective
 
This project is a **model-risk–oriented risk analytics engine** designed to **measure, validate, and document** market and credit risk models in a manner consistent with institutional risk management and model validation practices.
 
The objective is **not** to build a trading system or dashboard-centric application, but to demonstrate:
 
- Sound quantitative risk methodology
- Explicit assumptions and documented loss conventions
- Clear separation between **model logic**, **configuration**, and **presentation**
- Standard validation and backtesting diagnostics consistent with SR 11-7 and OSFI E-23 guidance
This repository functions as a **model risk validation sandbox** and work sample for roles in market risk, model validation, and risk analytics.
 
---
 
## 📌 Scope
 
### Market Risk
- Value-at-Risk (VaR) and Expected Shortfall (ES)
- Four methodologies: Historical Simulation, Parametric (Normal), Monte Carlo, Filtered Historical Simulation (GARCH-lite)
- Multi-confidence-level analysis (95%, 97.5%, 99%)
- Rolling-window estimation
- Out-of-sample backtesting: **Kupiec POF (unconditional coverage)**
- VaR decomposition: marginal, component, and incremental VaR (Euler allocation)
- Equal Risk Contribution (ERC) risk budgeting
### Credit Risk
- Expected Loss (EL) framework: PD × LGD × EAD
- Portfolio-level aggregation and segment-level decomposition
- Scenario shock capability (PD multiplier, additive basis points, LGD stress)
### Stress & Scenario Analysis
- Single-name equity shocks
- Interest rate shocks via duration approximation
- Covariance scaling (volatility + correlation stress)
- Historical window replay
---
 
## 📊 Validation Results
 
> Results generated using a 4-asset portfolio: **SPY 25% | QQQ 25% | TLT 25% | GLD 25%**
> **Exposure: $1,000,000 | Data: Jan 2020 – Dec 2024 | 1,258 daily observations**
> Parameters calibrated to observed market behaviour (SPY ~19% vol, QQQ ~24%, TLT ~15%, GLD ~13%).
 
### Point Risk Measures — 1-Day Horizon
 
| Method | VaR @ 95% | ES @ 95% | VaR @ 99% | ES @ 99% |
|---|---:|---:|---:|---:|
| Historical Simulation | $10,305 | $13,223 | $15,516 | $16,926 |
| Parametric (Normal) | $10,621 | $13,375 | $15,113 | $17,347 |
| Monte Carlo (100k sims) | $10,575 | $13,312 | $15,061 | $17,231 |
| Filtered Hist. (GARCH-lite) | $12,481 | $16,072 | $18,077 | $20,628 |
 
> **Interpretation:** The FHS/GARCH model produces materially higher estimates than static methods, reflecting its sensitivity to recent volatility clustering. The parametric and historical methods converge closely at 95%, diverging at 99% where distributional tail assumptions matter more. ES consistently exceeds VaR as required under the loss convention invariant enforced by the model object.
 
---
 
### Backtesting — Rolling Historical VaR (250-day window)
 
| α | OOS (T) | Exceedances | Hit Rate | Expected Rate | Kupiec LR | p-value | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 95.0% | 1,008 | 59 | 5.85% | 5.00% | 1.468 | 0.2257 | ✅ PASS |
| 97.5% | 1,008 | 27 | 2.68% | 2.50% | 0.129 | 0.7196 | ✅ PASS |
| 99.0% | 1,008 | 15 | 1.49% | 1.00% | 2.109 | 0.1464 | ✅ PASS |
 
> **Interpretation:** The Kupiec Proportion-of-Failures test evaluates H₀: observed exception rate equals (1 − α). All three confidence levels pass at the 5% significance level (p > 0.05), indicating the model is not statistically over- or under-estimating risk. Note: Kupiec tests unconditional coverage only; exception clustering (independence) is not assessed here — see **Known Limitations** below.
 
---
 
### VaR Decomposition — Component VaR (Normal, α=95%)
 
| Asset | Weight | Component VaR | % of Portfolio VaR |
|---|---:|---:|---:|
| SPY | 25.0% | $3,913 | 36.8% |
| QQQ | 25.0% | $5,005 | 47.1% |
| TLT | 25.0% | $409 | 3.9% |
| GLD | 25.0% | $1,294 | 12.2% |
| **Total** | **100%** | **$10,621** | **100.0%** |
 
> **Interpretation:** Despite equal weighting, QQQ contributes ~47% of total VaR due to its higher volatility (~23% annualised) and strong co-movement with SPY (ρ = 0.86). TLT's negative correlation with equities (ρ = −0.30 with SPY) reduces its risk contribution to under 4% of the portfolio total — a meaningful diversification effect. Component VaR sums to portfolio VaR under the Euler allocation.
 
---
 
### Stress Scenarios
 
| Scenario | Description | Portfolio Impact |
|---|---|---:|
| Equity shock | SPY + QQQ each −20%, TLT/GLD flat | **−$100,000** |
| Rate shock | +200bp parallel shift, TLT duration 18y | **−$90,000** |
| Vol/correlation stress | Covariance matrix ×2 | VaR: $10,575 → $15,246 (**+$4,671**) |
 
---
 
### Portfolio Summary Statistics
 
| Metric | Value |
|---|---|
| Annualised Return | 5.57% |
| Annualised Volatility | 10.46% |
| Sharpe Ratio (RF = 0) | 0.53 |
| SPY annualised vol | 18.55% |
| QQQ annualised vol | 23.02% |
| TLT annualised vol | 15.18% |
| GLD annualised vol | 12.94% |
 
### Asset Correlation Matrix
 
|  | SPY | QQQ | TLT | GLD |
|---|---:|---:|---:|---:|
| SPY | 1.000 | 0.859 | −0.302 | 0.072 |
| QQQ | 0.859 | 1.000 | −0.251 | 0.038 |
| TLT | −0.302 | −0.251 | 1.000 | 0.089 |
| GLD | 0.072 | 0.038 | 0.089 | 1.000 |
 
---
 
## ⚠️ Known Model Limitations
 
This section documents known weaknesses, as required under institutional model risk standards (SR 11-7 / OSFI E-23).
 
**1. Unconditional coverage only**
The backtesting framework implements Kupiec POF, which tests whether the *frequency* of exceptions is correct but does not test for *clustering* of exceptions. The Christoffersen conditional coverage test (which would detect periods of sustained volatility such as March 2020) is not yet implemented. This is a material gap for regulatory-grade validation.
 
**2. IID and stationarity assumptions**
All methods assume i.i.d. returns within the rolling window and stationarity of the return distribution. These assumptions are violated during volatility regime changes. The GARCH-lite filter partially addresses this for the FHS method only.
 
**3. GARCH parameter estimation**
The GARCH(1,1) filter uses fixed parameters (α = 0.05, β = 0.94) rather than MLE-estimated parameters. This simplification avoids optimisation complexity but may misspecify the volatility process for individual assets, particularly during regime shifts.
 
**4. Multivariate normality (Monte Carlo)**
The Monte Carlo method assumes a multivariate normal distribution for joint asset returns. Empirical return distributions exhibit excess kurtosis and negative skewness, meaning tail losses are likely underestimated at high confidence levels (99%+).
 
**5. 1-day horizon scaling**
Multi-day VaR is approximated via square-root-of-time scaling (√h). This assumption holds only if returns are i.i.d. normal — it underestimates risk when volatility is autocorrelated.
 
**6. Credit model scope**
The credit EL framework computes point-in-time Expected Loss using user-supplied PD/LGD/EAD inputs. It does not estimate PD from historical default data (e.g. via logistic regression or scorecard), does not model loss distributions (only expected values), and does not compute Unexpected Loss or Economic Capital.
 
---
 
## 🧱 Repository Structure
 
```
risklib/
  market/
    market_risk_model.py     # MarketRiskModel class + MarketRiskConfig
    market.py                # Risk primitives: VaR, ES, backtest, GARCH, ERC
  credit/
    credit_risk_model.py     # EL pipeline: validate → shock → compute → summarize
risk_engine/                 # Thin wrappers used by Streamlit app
app/
  app.py                     # Streamlit UI — presentation only, no risk logic
docs/
  model_report.md            # Full model methodology and validation notes
tests/                       # Unit tests for model invariants
notebooks/                   # Exploratory analysis
```
 
### Design Principles
 
- **`risklib/` is the source of truth** — All modelling, estimation, and validation logic lives here
- **`app/` is presentation-only** — The UI calls `risklib` and visualizes outputs; it computes nothing directly
- **Loss-based convention enforced** — All outputs are positive loss amounts; the `MarketRiskModel` raises if VaR < 0 or ES < VaR
---
 
## 🔍 Market Risk Model
 
```python
from risklib.market.market_risk_model import MarketRiskModel, MarketRiskConfig
 
cfg = MarketRiskConfig(
    alpha=0.99,
    method="historical",     # historical | parametric | monte_carlo | fhs
    horizon_days=1,
    exposure=1_000_000,
)
 
model = MarketRiskModel(returns, weights, cfg)
model.fit()
 
var = model.compute_var()    # e.g. 15,516
es  = model.compute_es()     # e.g. 16,926
summary = model.summary()    # includes assumptions, config metadata
```
 
The `MarketRiskModel` enforces two invariants at `fit()` time:
- `VaR >= 0` (loss convention)
- `ES >= VaR` (coherence requirement)
A `ValueError` is raised if either condition is violated, surfacing methodology errors early.
 
---
 
## 🧪 Validation & Testing
 
Unit tests verify model invariants independently of data:
 
- VaR monotonicity across confidence levels (VaR₉₉ > VaR₉₅)
- ES ≥ VaR under consistent loss convention
- Correct exception counting in rolling backtests
- Credit EL aggregation consistency (sum of facility EL = portfolio EL)
- Component VaR sums to portfolio VaR under Euler allocation
Backtesting is conducted **out-of-sample** using a trailing window to prevent look-ahead bias. The VaR threshold at time *t* is estimated from returns up to *t−1* only.
 
---
 
## 🧠 Methodology
 
| Model | Formula | Notes |
|---|---|---|
| VaR (Historical) | −Q₁₋ₐ(r_p) × exposure | Empirical quantile of portfolio returns |
| VaR (Parametric) | (−μ_p + z_α × σ_p) × exposure | Assumes normality |
| ES (Parametric) | (−μ_p + σ_p × φ(z_α)/(1−α)) × exposure | Closed-form under normality |
| VaR (Monte Carlo) | Empirical quantile of 100k simulated paths | Multivariate normal with covariance shrinkage |
| VaR (FHS) | −q_z × σ_{t+1} × exposure | GARCH-standardised residuals, one-step-ahead forecast |
| Expected Loss | PD × LGD × EAD | Per-facility; aggregated to portfolio/segment level |
 
---
 
## 🚫 Out of Scope (By Design)
 
This project does **not** attempt to be:
- A trading or portfolio optimisation system
- A real-time production risk engine
- A regulatory-approved model
Deferred extensions: factor models, ALM, CVA, optimisation, MLE-estimated GARCH, Christoffersen test.
 
---
 
## 🖥 Application Interface
 
A live **Streamlit app** is deployed at **[integrated-risk-app.onrender.com](https://integrated-risk-app.onrender.com/)**.
 
The UI allows a user to:
- Upload a prices CSV or exposures CSV
- Select method, confidence level, horizon, and exposure
- View VaR/ES point estimates, backtest chart with exception markers, Kupiec results
- Run stress scenarios and what-if weight analysis
- Export a Markdown risk report and CSV decompositions
All modelling logic remains in `risklib/`. The app is a viewer only.
 
---
 
## ⚙️ Tech Stack
 
| Layer | Libraries |
|---|---|
| Risk engine | NumPy, Pandas, SciPy, Statsmodels |
| Visualisation | Plotly |
| UI | Streamlit |
| Data (demo) | yfinance |
| Tests | pytest |
 
---
 
## ⚡ Quickstart
 
```bash
git clone https://github.com/sensor-aae/Integrated-Risk-App.git
cd Integrated-Risk-App
 
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
 
pip install -r requirements.txt
streamlit run app/app.py
```
 
---
 
## 📄 Documentation
 
A full **Model Risk Report** (`docs/model_report.md`) covers:
- Methodology and theoretical basis
- Assumptions and their implications
- Validation results and test statistics
- Known limitations and areas for improvement
This mirrors institutional model documentation standards.
 
---
 
## ⭐ Why This Project Exists
 
This repository is designed as a **work sample** for roles in:
 
- Market Risk
- Model Risk / Model Validation
- Credit Risk Analytics
- Pension & Institutional Investment Risk
- Risk Consulting
It reflects how quantitative risk models are **built, tested, challenged, and reviewed** — not just how they are computed.
 
---
 
## ⚠️ Disclaimer
 
This project is for educational and demonstrative purposes only. It is **not** intended for production use or investment decision-making. All results shown are generated from simulated data calibrated to approximate market conditions; they do not constitute forecasts.
 
