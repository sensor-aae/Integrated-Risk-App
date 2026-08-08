# Integrated Risk App

**Validation-grade market and credit risk engine**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-129-brightgreen)

---

## What this is

A model-risk–oriented risk analytics engine that **measures, validates and
documents** market and credit risk models the way an institutional risk function
would.

It is not a trading system and not a dashboard. The organising question is not
"what is the number?" but "**is the number any good, and how would you know?**"

Three things follow from that:

- **Every published figure is reproducible.** The results in
  [`docs/validation_results.md`](docs/validation_results.md) are generated from
  the data committed in `data/` by `scripts/generate_validation_results.py`, and
  CI fails if the committed results drift from what the code produces.
- **Conventions are enforced, not documented.** VaR ≥ 0 and ES ≥ VaR are runtime
  invariants that raise, not notes in a README.
- **Assumptions travel with results.** `summary()` returns the risk numbers, the
  full configuration, and the assumptions actually in force for that
  configuration.

---

## Scope

**Market risk** — VaR and Expected Shortfall by four methods (historical
simulation, parametric Normal, Monte Carlo, Filtered Historical Simulation with
a GARCH(1,1) filter); multi-α analysis; rolling out-of-sample backtesting with
Kupiec POF, Christoffersen independence, and joint conditional coverage; Euler
VaR decomposition (marginal and component); true incremental VaR; Equal Risk
Contribution budgeting.

**Credit risk** — Expected Loss (PD × LGD × EAD) with portfolio and segment
aggregation, multiplicative and additive scenario shocks, and data-quality
reporting on out-of-range inputs.

**Stress and scenarios** — single-name shocks, parallel rate shocks via duration
approximation, covariance scaling, correlation-breakdown stress with PSD
projection, and historical window replay.

---

## Quickstart

```bash
git clone https://github.com/sensor-aae/Integrated-Risk-App.git
cd Integrated-Risk-App

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/app.py
```

Upload `data/market_data.csv` for market risk and `data/credit_example.csv` for
credit. To reproduce every published number:

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/generate_validation_results.py
```

---

## Using the engine directly

```python
import numpy as np
from risklib.data import load_prices, to_returns
from risklib.market import MarketRiskConfig, MarketRiskModel, backtest_var_historical

returns = to_returns(load_prices("data/market_data.csv"), method="log")
weights = np.ones(returns.shape[1]) / returns.shape[1]

cfg = MarketRiskConfig(
    alpha=0.99,
    method="fhs",          # historical | parametric | monte_carlo | fhs
    horizon_days=1,
    exposure=1_000_000,
    fit_garch=True,        # estimate omega, alpha, beta by MLE
)

model = MarketRiskModel(returns, weights, cfg).fit()

model.compute_var()        # positive loss amount
model.compute_es()
model.summary()            # VaR, ES, full config, assumptions in force
model.fit_info_["source"]  # "MLE" or "fixed" — estimated vs assumed

bt = backtest_var_historical(returns, weights, alpha=0.99, window=250)
bt["kupiec_pvalue"], bt["christoffersen_pvalue"], bt["joint_pvalue"]
```

`MarketRiskConfig` carries every parameter that changes a number — confidence
level, horizon, exposure, backtest window, Monte Carlo paths and seed,
covariance shrinkage, and the GARCH settings. It is one serialisable object you
can log, diff and attach to a result, which is the difference between "we ran a
VaR" and "we ran *this* VaR".

---

## Repository structure

```
risklib/                          # single source of truth for all modelling
  data.py                         # ingestion: prices -> returns
  market/
    market_risk_model.py          # estimators, MarketRiskConfig, MarketRiskModel
    backtest.py                   # Kupiec + Christoffersen + joint CC; both backtests
    garch.py                      # one GARCH(1,1) recursion: fixed and MLE
    extras.py                     # Euler decomposition, incremental VaR, ERC
    scenarios.py                  # stress testing and scenario library
  credit/
    credit_risk_model.py          # EL pipeline: validate -> shock -> compute -> summarise

app/app.py                        # Streamlit interface — presentation only
scripts/
  generate_validation_results.py  # regenerates docs/validation_results.md
docs/
  model_report.md                 # methodology, assumptions, limitations
  validation_results.md           # GENERATED — do not edit by hand
tests/                            # 129 tests
data/                             # demo market and credit data
```

**Design rules**

- `risklib/` holds all modelling, estimation and validation logic
- `app/` computes nothing — it collects inputs, calls `risklib`, and visualises
- Losses are positive throughout, enforced at `fit()` time

---

## Validation results

Full tables — point measures, GARCH parameter comparison, backtests across three
confidence levels, Euler decomposition, ERC, stress scenarios, and credit EL —
are in **[`docs/validation_results.md`](docs/validation_results.md)**, generated
from the shipped demo portfolio (AAPL / TLT / MSFT, equal-weighted, $1,000,000
exposure, 1,257 daily observations from January 2020 to December 2024).

A few results worth calling out:

**All three tests pass at every confidence level**, for both the historical and
FHS backtests. The independence p-values are the interesting ones — they confirm
exceptions are distributed across the sample rather than concentrated in stress
periods, which Kupiec alone cannot detect.

**Estimated GARCH parameters differ materially from the convention.** The fixed
defaults (α = 0.05, β = 0.94) imply persistence of 0.9900; MLE estimates 0.9665
on this sample. The assumed parameters overstate volatility persistence, which
is exactly the misspecification the MLE path exists to address.

**Component VaR sums to portfolio VaR to 3.6 × 10⁻¹²** — the Euler identity is
exact, not approximate, because VaR is homogeneous of degree 1 in the weights.

**ERC substantially rebalances the book.** Under equal weighting the two equities
contribute 93% of portfolio VaR between them; ERC moves the bond from 33% to 54%
of the portfolio and cuts VaR by roughly 20%.

---

## Validation and testing

129 tests. The suite asserts **model invariants** rather than frozen numbers, so
the assertions hold for any input and do not break when a default changes.

The test worth reading is `test_christoffersen_rejects_clustering`. It builds an
exception series with the **correct total count** but bunched into one contiguous
block, then asserts that Kupiec passes it with p > 0.9 and Christoffersen rejects
it at p < 0.001. That contrast is the entire argument for implementing the
independence test.

Also covered: no-look-ahead verification (the threshold at *t* is recomputed by
hand from *t−1* and compared), GARCH MLE parameter recovery on simulated data,
the Euler summation identity, credit EL aggregation consistency, and a
contract-test file pinning every dictionary key the Streamlit app reads — so a
rename in the engine fails in CI rather than in front of a user.

CI runs the suite on Python 3.11 and 3.12, verifies the package installs cleanly,
and re-runs the validation script to confirm the published results still
reproduce.

---

## Methodology summary

| Model | Formula | Notes |
|---|---|---|
| VaR (historical) | −Q₁₋α(r_p) × E | Empirical quantile; no distributional assumption |
| VaR (parametric) | (−μ_p + z_α σ_p) × E | Closed form under normality |
| ES (parametric) | (−μ_p + σ_p φ(z_α)/(1−α)) × E | Closed-form tail expectation |
| VaR (Monte Carlo) | Empirical quantile of simulated paths | Multivariate normal, covariance shrinkage |
| VaR (FHS) | −q_z × σ_{t+1} × E | GARCH-standardised residuals, one-step-ahead |
| Component VaR | w_i · ∂VaR/∂w_i | Euler allocation; sums exactly to portfolio VaR |
| Expected Loss | PD × LGD × EAD | Per facility, aggregated to segment and portfolio |

Full derivations, assumptions and limitations: **[`docs/model_report.md`](docs/model_report.md)**.

---

## Known limitations

Documented in full in [`docs/model_report.md`](docs/model_report.md#6-assumptions-and-limitations),
with resolved items retained as an audit trail. The open ones:

- i.i.d. and stationarity assumptions, violated across volatility regime changes
- Multivariate normality in Monte Carlo understates tails at high confidence
- √h horizon scaling understates risk when volatility is autocorrelated
- The rolling FHS backtest retains fixed GARCH parameters
- Zero conditional mean in the GARCH filter
- Duration approximation ignores convexity
- Credit model computes Expected Loss only — no loss distribution, no economic capital
- Results derive from a single demo portfolio and demonstrate reproducibility,
  not general model performance

---

## Out of scope by design

Trading and portfolio optimisation, real-time production risk, regulatory capital
calculation. Deferred: factor models, ALM, CVA, copula simulation, PD estimation
from default history, multi-step GARCH forecasting.

---

## Tech stack

| Layer | Libraries |
|---|---|
| Engine | NumPy, pandas, SciPy |
| Interface | Streamlit, Plotly |
| Tests | pytest |

The core VaR and backtesting math depends on neither SciPy nor statsmodels —
Normal quantiles come from `statistics.NormalDist` and χ² p-values from exact
closed forms (`erfc(√(x/2))` for one degree of freedom, `exp(−x/2)` for two).
SciPy is required only for GARCH maximum likelihood.

---

## Disclaimer

Educational and demonstrative. Not intended for production use or investment
decision-making, and not a regulatory-approved model.
