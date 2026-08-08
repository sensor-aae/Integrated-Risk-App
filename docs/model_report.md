# Model Risk Report
## Integrated Risk App — Market & Credit Risk

**Version 0.2 · Model owner: Amanda Achiangia**

This document consolidates what were previously two divergent reports
(`model_report.md` and `model_risk_report.md`), which had drifted apart and
described limitations that no longer held.

Empirical results are **not** reproduced here. They live in
[`validation_results.md`](validation_results.md), which is generated directly
from the code by `scripts/generate_validation_results.py` and verified by CI.
Numbers in a document that is maintained by hand go stale; numbers generated
from the engine cannot.

---

## 1. Purpose and intended use

### Purpose

Measure and validate portfolio-level market and credit risk using standard
quantitative methodologies, with the emphasis on **validation** rather than
production risk reporting: transparent assumptions, reproducible results, a
clean separation between model logic and presentation, and standard backtesting
diagnostics.

### Intended use

- Risk measurement and monitoring on a defined portfolio
- Model validation exercises and methodology comparison
- Educational and demonstrative analysis

### Non-intended use

- Trading, hedging, or portfolio optimisation decisions
- Real-time or production risk management
- Regulatory capital calculation
- Any use where the outputs inform an actual financial commitment

---

## 2. Governing conventions

### 2.1 Loss convention

Every quantity leaving the engine is a **positive loss amount**:

- returns below zero represent losses
- VaR ≥ 0
- ES ≥ VaR

`MarketRiskModel.fit()` enforces both inequalities as runtime invariants and
raises `ValueError` if either fails.

This is deliberate. Sign conventions are the classic silent-failure mode in risk
code: one estimator returns a signed return, another a positive loss, the
mismatch gets patched with `abs()`, and one confidence level is quietly wrong
forever. Encoding the convention as an assertion converts a methodology error
into an exception at fit time rather than a plausible-looking number in a report.

### 2.2 Configuration as specification

Every parameter that changes a number lives on `MarketRiskConfig` — confidence
level, horizon, exposure, backtest window, Monte Carlo paths and seed,
covariance shrinkage, and the GARCH settings. The config is a single
serialisable object that can be logged, diffed and attached to a result.

`summary()` returns the risk numbers together with the full config and the
assumptions actually in force for that configuration. A risk number without its
assumptions is not a deliverable.

### 2.3 Estimated versus assumed

Wherever a parameter can be either assumed or estimated, the result carries a
`source` tag (`"fixed"` or `"MLE"`). Any downstream consumer — including the
report exporter — can therefore tell which it received. If MLE is requested but
fails, the engine falls back to fixed parameters and records
`fallback_reason` rather than silently substituting.

---

## 3. Data

### 3.1 Market data

Daily prices, one column per instrument, converted to log returns by default.
Log returns are time-additive, which is the property that makes square-root-of-
time scaling and the i.i.d. assumption internally coherent. Simple returns are
available because portfolio aggregation (**w · r**) is strictly correct for them
and only approximate for log returns; the trade-off is exposed to the user
rather than decided silently.

Ingestion guards, each protecting a specific downstream failure:

| Guard | Failure prevented |
|---|---|
| Sort by date | A newest-first export runs every rolling window backwards through time and inverts the backtest |
| Drop duplicate dates | Breaks date slicing and double-counts observations in the window |
| Null non-positive prices | `log(P_t/P_{t-1})` is undefined; a single zero emits `-inf` into the covariance matrix |
| Strip currency formatting | Excel exports carry `"$1,234.56"` as text, which silently poisons `.cov()` and `.mean()` |
| Require ≥ 2 observations per column | Cannot produce even one return |

### 3.2 Credit data

PD, LGD and EAD supplied per facility, with an optional segment/rating column
detected automatically. Inputs are **exogenous** — the model does not estimate
PD from default history.

PD and LGD are accepted as either decimals or percentages: values above 1 and at
or below 100 are interpreted as percentages and divided by 100. A PD of exactly
1.0 is treated as certain default, not as 1%. Out-of-range values are clamped
to valid ranges, and the count of clamped rows is reported as a data-quality
finding rather than absorbed silently.

---

## 4. Methodology

### 4.1 Market risk measures

VaR at confidence α is the loss threshold exceeded with probability (1 − α):

$$P(L > \text{VaR}_\alpha) = 1 - \alpha$$

Expected Shortfall is the mean loss conditional on exceeding it:

$$\text{ES}_\alpha = \mathbb{E}[L \mid L > \text{VaR}_\alpha]$$

ES is reported alongside VaR throughout because VaR gives the threshold but says
nothing about severity beyond it, and is not subadditive. ES is coherent and is
the measure FRTB moved to.

| Method | Formula | Notes |
|---|---|---|
| Historical | −Q₁₋α(r_p) × E | Empirical quantile; no distributional assumption |
| Parametric | (−μ_p + z_α σ_p) × E | Closed form under normality |
| ES parametric | (−μ_p + σ_p φ(z_α)/(1−α)) × E | Closed-form tail expectation |
| Monte Carlo | Empirical quantile of simulated paths | Multivariate normal, covariance shrinkage |
| FHS | −q_z × σ_{t+1} × E | GARCH-standardised residuals, one-step-ahead |
| Expected Loss | PD × LGD × EAD | Per facility, aggregated |

### 4.2 Covariance shrinkage

$$\Sigma_s = (1-\lambda)\Sigma + \lambda \,\text{diag}(\Sigma)$$

On the diagonal this leaves variances **unchanged**; off the diagonal it
multiplies every correlation by (1 − λ). The purpose is numerical: the sample
covariance is near-singular when the instrument count is large relative to the
sample or when two series are nearly collinear, and a near-singular Σ makes the
Cholesky factorisation fail or return garbage. Shrinkage pulls the matrix away
from the boundary of the PSD cone at negligible cost to the risk estimate.

### 4.3 Filtered Historical Simulation

1. Project to portfolio returns, r_p = **R w**
2. Filter to obtain the conditional volatility path σ_t
3. Standardise: z_t = r_t / σ_t
4. Take the empirical (1 − α) quantile of z, and its tail mean for ES
5. Forecast σ_{t+1} and rescale: VaR = −q_z · σ_{t+1} · E

FHS improves on both parents. Historical simulation preserves the empirical tail
shape but treats a calm day and a panicked day as equally informative. Parametric
responds to current volatility but forces Normal tails. FHS keeps the empirical —
fat, skewed — tail shape of the *standardised residuals* while letting the scale
track today's volatility.

Below 50 observations the filter is noise, and the method degrades explicitly to
plain historical simulation.

### 4.4 GARCH(1,1)

$$\sigma^2_t = \omega + \alpha r^2_{t-1} + \beta \sigma^2_{t-1}$$

**Fixed mode (variance targeting).** α and β are supplied (defaults 0.05 / 0.94),
and ω is pinned so the model's unconditional variance ω/(1−α−β) equals the sample
variance. Cheap and reproducible, but an assumption.

**MLE mode.** ω, α and β are estimated by maximising the Gaussian conditional
log-likelihood via L-BFGS-B with multiple restarts. Restarts matter: the
likelihood is nearly flat in the (α + β) direction near persistence 1, so a
single start from a poor point can converge to a local optimum *and report
success*.

Constraints are enforced **structurally** rather than handed to the optimiser:

| Constraint | Transform |
|---|---|
| ω > 0 | ω = exp(p₀) |
| α ∈ (0,1) | α = sigmoid(p₁) |
| α + β < 1 | β = (1 − α − ε) · sigmoid(p₂) |

The optimiser roams freely over ℝ³ while stationarity is impossible to violate.
This avoids the boundary-stalling that constrained solvers exhibit on the coupled
stationarity inequality.

The reported log-likelihood **includes** the 2π normalising constant, so the
log-likelihood, AIC and BIC are directly comparable to values from `arch`,
`statsmodels` or R.

### 4.5 Attribution

**Euler allocation.** VaR is homogeneous of degree 1 in the weights, so Euler's
theorem applies *with equality*:

$$\text{mVaR}_i = \frac{\partial \text{VaR}}{\partial w_i} = \left(-\mu_i + z_\alpha \frac{(\Sigma w)_i}{\sigma_p}\right) E, \qquad \text{cVaR}_i = w_i \cdot \text{mVaR}_i, \qquad \sum_i \text{cVaR}_i = \text{VaR}$$

Component VaR sums to portfolio VaR to machine precision, not approximately.
This is what converts a portfolio-level number into "instrument X is 47% of your
risk on a 33% weight" — the figure a risk committee acts on.

**Incremental VaR** is a true recomputation: portfolio VaR minus the VaR of the
portfolio with that position removed and the remainder renormalised. It converges
to component VaR only for small positions; on a material weight the two differ,
so both are reported side by side.

**Equal Risk Contribution** equalises RC_i = w_i (Σw)_i by damped multiplicative
updates, with convergence measured on risk-contribution dispersion — the
objective itself rather than a proxy. Weights are projected onto the
box-constrained simplex exactly (bisection on a single shift), because clipping
and then renormalising scales clipped weights back above the cap and satisfies
neither constraint.

Note that ERC equalises contributions to **variance**. Percentage-of-VaR
contributions additionally carry the drift term (−μ_i), so they will be close to
but not exactly equal.

---

## 5. Validation approach

### 5.1 Out-of-sample discipline

Every backtest threshold at time *t* uses information available strictly through
*t − 1*.

This is not incidental. `rolling(window).quantile()` at row *t* includes row *t*
itself — the very day being predicted — so every threshold is shifted by one
period. Without that shift, every backtest result would be meaningless, and it
is the first property a reviewer checks.

The FHS backtest additionally derives ω from an **expanding, lagged** variance
estimate, so the volatility level is also free of look-ahead. A regression test
truncates the sample and asserts the σ path is unchanged.

### 5.2 The three tests

**Kupiec POF** (unconditional coverage). H₀: the exception rate equals (1 − α).
Exceptions are modelled as i.i.d. Bernoulli; the likelihood ratio is
asymptotically χ²(1). *Limitation: it sees only the count.* A model producing
exactly 5% exceptions, all in one week, passes.

**Christoffersen independence.** H₀: π₀₁ = π₁₁ — exceptions are serially
independent. The indicator series is treated as a first-order Markov chain.
This is the test that matters: a model clustering its exceptions is
systematically understating risk in stressed regimes and overstating it in calm
ones, which is exactly the failure mode of a static historical-simulation model
in March 2020. Kupiec is blind to it.

**Joint conditional coverage.** LR_cc = LR_uc + LR_ind ~ χ²(2). The additivity
is the standard Christoffersen decomposition; the two statistics are
asymptotically independent.

χ² p-values use exact closed forms — erfc(√(x/2)) for one degree of freedom and
exp(−x/2) for two — so the validation layer carries no SciPy dependency.

### 5.3 Test coverage

The test suite asserts model invariants independently of any dataset:

- VaR ≥ 0 and ES ≥ VaR, for all four methods
- VaR monotone in α, for all four methods
- VaR homogeneous of degree 1 in exposure
- Component VaR sums to portfolio VaR under Euler allocation
- Facility EL sums to portfolio EL; grouped subtotals reconcile to the total
- Backtest exception counts match the flags actually raised
- The threshold at *t* equals the trailing quantity computed by hand from *t−1*
- Christoffersen rejects a correctly-sized but clustered exception series that
  Kupiec passes
- GARCH MLE recovers known parameters from simulated data, and beats the fixed
  defaults on the same series
- FHS σ is unchanged when future observations are removed

A separate contract-test file pins every dictionary key the Streamlit app reads,
so a rename in the engine fails in CI rather than in front of a user.

---

## 6. Assumptions and limitations

Limitations are stated as standing properties of the model. Items resolved since
the previous version are retained with their resolution, as an audit trail.

### 6.1 Open limitations

**L1 — i.i.d. and stationarity.** All methods assume returns are i.i.d. within
the estimation window and that the return distribution is stationary. Both are
violated during volatility regime changes. The GARCH filter partially addresses
this, for the FHS method only.

**L2 — Multivariate normality (Monte Carlo).** Joint returns are assumed
multivariate normal. Empirical returns exhibit excess kurtosis and negative
skewness, so tail losses are likely understated at high confidence levels. A
copula or multivariate-*t* specification is the natural extension.

**L3 — Square-root-of-time scaling.** Multi-day VaR is approximated by √h
scaling, valid only under i.i.d. returns. It understates risk when volatility is
autocorrelated. Applies to the historical, parametric and Monte Carlo methods;
FHS is a one-step-ahead forecast and does not scale.

**L4 — Fixed GARCH parameters in the rolling FHS backtest.** Point estimates can
use MLE, but the rolling backtest retains fixed α and β. Re-estimating at every
step requires a full optimisation per day and is deliberately out of scope.

**L5 — Zero conditional mean in the GARCH filter.** Defensible for daily equity
returns, but it is an assumption, and it is now reported as one.

**L6 — Duration approximation ignores convexity.** ΔP/P ≈ −D·Δy is first-order.
At 200bp on a long-duration instrument the omitted second-order term is material,
and the linear estimate **overstates** the loss.

**L7 — Credit model scope.** Point-in-time Expected Loss from user-supplied
inputs. It does not estimate PD from default history, does not model the loss
distribution, and computes neither Unexpected Loss nor economic capital.

**L8 — Correlation stress and PSD.** Element-wise correlation bumping does not
preserve positive semi-definiteness. The bumped matrix is projected back onto
the PSD cone by eigenvalue clipping, and the projection is flagged in the output.
A large bump requiring adjustment signals that the scenario is straining the
correlation structure.

**L9 — Single demo dataset.** Published results derive from one three-instrument
portfolio over one sample period. They demonstrate that the machinery works and
is reproducible; they are not evidence of general model performance.

### 6.2 Resolved

**R1 — Unconditional coverage only.** *Resolved.* `backtest.py` implements the
full Christoffersen suite: independence and joint conditional coverage alongside
Kupiec. All three are returned by both backtests and displayed in the app.

**R2 — Fixed GARCH parameters.** *Resolved for point estimates.* `garch.py`
provides MLE estimation with stationarity enforced by parameter transformation.
Enabled via `fit_garch=True`; the default remains `False` so prior results stay
reproducible. See L4 for the remaining scope limit.

**R3 — `fit_garch` had no effect.** *Resolved.* The flag was accepted by the
config but never forwarded to the estimator, so the MLE path was unreachable
while the documentation described it as working. It is now threaded through
`fit()` and covered by a regression test.

**R4 — Monte Carlo settings were not honoured.** *Resolved.* Simulation count,
seed and shrinkage were collected in the interface but never reached the
computation. They are now config fields, threaded through and part of the cache
key.

**R5 — Look-ahead in the FHS backtest.** *Resolved.* The long-run variance
driving ω was computed from the full sample; it is now an expanding, lagged
estimate. Covered by a truncation test.

**R6 — Portfolio EL/EAD was equal-weighted.** *Resolved.* It was the mean of
per-facility ratios, which let a small facility move the portfolio figure as much
as a large one and disagreed with every grouped subtotal beside it. It is now
exposure-weighted.

**R7 — Rate shocks moved equities.** *Resolved.* The default duration was 7
years applied to every instrument, so a +200bp scenario knocked roughly 14% off
an equity position. The default is now 0; only instruments with an explicit or
recognised duration respond.

**R8 — Duplicate implementations.** *Resolved.* Twelve core functions existed in
two directories with divergent signatures. `risklib` is now the single source of
truth and `risk_engine` has been removed.

---

## 7. Change log

| Version | Change |
|---|---|
| 0.2 | Consolidated to a single `risklib` package; removed duplicate implementations; fixed R3–R7; unified GARCH into one recursion; added true incremental VaR; exact box-constrained ERC projection; generated, CI-verified validation results; test suite from 5 to 129 tests |
| 0.1 | Initial engine, Kupiec backtest, Streamlit interface |

---

## 8. Deferred extensions

Factor models, ALM, CVA, portfolio optimisation, multi-step GARCH forecasting,
copula and multivariate-*t* simulation, PD estimation from default history, loss
distributions and economic capital, bootstrap confidence intervals for VaR, and
window-sensitivity analysis.
