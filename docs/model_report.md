# Model Risk Report  
## Integrated Risk App — Market & Credit Risk

---

## 1. Model Purpose & Intended Use

### Purpose

The purpose of this model is to **measure and validate portfolio-level market and credit risk** using standard quantitative risk methodologies commonly employed by institutional investors, pensions, and risk management teams.

The model is designed as a **validation sandbox**, emphasizing:
- Transparent assumptions
- Reproducible results
- Clear separation between model logic and presentation
- Standard backtesting diagnostics

### Intended Use

This model is intended for:
- Risk measurement and monitoring
- Model validation exercises
- Educational and demonstrative analysis

### Non-Intended Use

The model is **not** intended for:
- Trading or portfolio optimization
- Real-time risk management
- Regulatory capital calculation
- Production deployment

---

## 2. Data Description

### Market Data

- Asset returns are computed from historical price data
- Data frequency: daily
- Data source: publicly available market data (via `yfinance`, demo use only)

Returns are computed as simple or log returns and aggregated into a portfolio using fixed user-defined weights.

### Credit Data

Credit risk inputs are provided as tabular datasets containing:
- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)

These inputs are assumed to be **exogenous** and are not estimated dynamically by the model.

---

## 3. Methodology Overview

### 3.1 Market Risk Measures

All market risk measures follow a **loss-based convention**, where reported values represent **positive losses**.

#### Value at Risk (VaR)

VaR at confidence level α is defined as the loss threshold exceeded with probability:

\[
P(L > \text{VaR}_\alpha) = 1 - \alpha
\]

#### Expected Shortfall (ES)

Expected Shortfall is defined as the **average loss conditional on exceeding VaR**:

\[
\text{ES}_\alpha = \mathbb{E}[L \mid L > \text{VaR}_\alpha]
\]

---

### 3.2 Market Risk Methodologies

The following methodologies are implemented:

#### Historical Simulation
- Empirical quantiles of historical portfolio returns
- No distributional assumptions
- Assumes stationarity of historical returns

#### Parametric (Normal)
- Portfolio returns assumed normally distributed
- Mean and covariance estimated from historical data
- Closed-form VaR and ES expressions

#### Monte Carlo Simulation
- Multivariate normal simulation of asset returns
- Mean and covariance estimated from data
- Light covariance shrinkage applied for numerical stability
- Portfolio losses simulated to estimate VaR and ES

#### Filtered Historical Simulation (GARCH-lite)
- Portfolio returns filtered using fixed-parameter GARCH(1,1)
- Standardized residuals used for tail estimation
- One-step-ahead volatility forecast applied
- Captures time-varying volatility while retaining empirical tails

---

### 3.3 Credit Risk Methodology

Credit risk is measured using the **Expected Loss (EL)** framework:

\[
\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}
\]

Expected Loss is:
- Computed at the facility level
- Aggregated to portfolio level
- Decomposed by segment where applicable

No default correlation or portfolio credit model (e.g. Vasicek) is assumed.

---

## 4. Backtesting & Validation

### 4.1 Market Risk Backtesting

Market risk models are evaluated using **rolling out-of-sample backtests**.

- 1-day VaR horizon
- Rolling estimation window
- VaR forecast computed using information available up to time *t−1*
- Exceptions recorded when realized return breaches the VaR threshold

---

### 4.2 Kupiec Proportion-of-Failures (POF) Test

The Kupiec POF test evaluates **unconditional coverage** of the VaR model.

#### Null Hypothesis

\[
H_0: \pi = 1 - \alpha
\]

Where:
- π is the observed exception rate
- α is the VaR confidence level

#### Test Statistic

The likelihood ratio statistic follows an asymptotic χ²(1) distribution.

- High LR statistic / low p-value → reject model coverage
- Low LR statistic → model consistent with expected exception rate

---

## 5. Model Assumptions & Limitations

### Key Assumptions

- Historical returns are representative of future risk
- Portfolio weights are static over the risk horizon
- Normality assumptions apply where specified
- Credit risk inputs (PD, LGD, EAD) are externally provided

### Limitations

- No dynamic correlation modeling
- No intraday or high-frequency data
- No regulatory capital framework (e.g. Basel) implemented
- GARCH parameters are fixed rather than estimated

These limitations are **intentional** to preserve clarity and interpretability.

---

## 6. Model Governance Notes

- Model logic is isolated in `risklib/`
- Configuration objects explicitly capture modeling assumptions
- UI layer does not modify or implement risk calculations
- Backtesting is performed out-of-sample
- Results are reproducible and exportable

This structure mirrors common **model risk governance principles**, including:
- Transparency
- Auditability
- Separation of concerns
- Clear documentation of assumptions

---

## 7. Conclusion

This model provides a **validation-focused implementation** of standard market and credit risk methodologies.

The emphasis is on:
- Correct methodology
- Proper validation
- Interpretability
- Governance-aligned design

The model is suitable as a **demonstration artifact** for risk analytics, model validation, and institutional risk roles.

---

**End of Report**
