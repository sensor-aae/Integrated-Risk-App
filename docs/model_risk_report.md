# Model Risk Report (V1) — Market Risk VaR/ES

## 1. Purpose
Demonstrate a validation-ready risk engine that:
1) computes VaR/ES under clear assumptions and conventions, and  
2) evaluates reliability using standard model risk diagnostics.

## 2. Loss Convention (Non-negotiable)
All risk measures are reported as **positive loss amounts**:
- returns < 0 represent losses
- VaR, ES ≥ 0
- ES ≥ VaR

## 3. Models Implemented
### 3.1 Historical Simulation (HS)
- VaR computed as the empirical *(1 − α)* quantile of portfolio returns
- ES computed as the conditional mean of returns beyond the VaR threshold

### 3.2 Parametric Normal
- assumes portfolio returns are approximately normal
- VaR/ES derived from μ and σ using Φ^{-1}(α) and φ(z)

### 3.3 Monte Carlo (MVN)
- simulates multivariate normal returns using sample μ, Σ
- includes light covariance shrinkage toward diagonal for stability
- VaR/ES computed from simulated portfolio return distribution

### 3.4 Filtered Historical Simulation (GARCH-lite)
- fixed-parameter GARCH(1,1) volatility filter produces σ_t
- standardized residuals z_t = r_t / σ_t
- VaR/ES computed from z distribution and scaled by σ_{t+1}

## 4. Validation Approach
### 4.1 Rolling Backtest (1-day VaR)
- estimate VaR_t from trailing window using data up to t−1
- exception occurs if realized r_t falls below VaR threshold
- output: exception series and backtest summary

### 4.2 Kupiec POF (Unconditional Coverage)
- H₀: exception probability equals (1 − α)
- returns LR statistic and p-value
- small p-values indicate rejection of correct coverage

## 5. Known Limitations (Explicit)
- HS assumes returns are iid/stationary within window
- Normal and MVN assume elliptical tails and may understate tail risk
- GARCH-lite uses fixed parameters (not MLE-estimated)
- Backtest currently targets unconditional coverage only (no independence test in V1)

## 6. Next Validation Extensions (Planned)
- Christoffersen independence / conditional coverage
- stress testing of window sensitivity and α calibration
- bootstrap confidence intervals for VaR estimates
