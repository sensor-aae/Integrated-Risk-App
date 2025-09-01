# Integrated Risk App (Python)

Interactive Streamlit app for **market and credit risk**:

![App Screenshot](docs/screenshot.png)

Interactive Streamlit app for market & credit risk:
- VaR/ES: Historical, Parametric (Normal), Monte Carlo, GARCH-lite (Filtered Historical)
- Backtesting: rolling VaR with Kupiec POF test + multi-alpha calibration
- Analytics: correlation heatmap, VaR decomposition, what-if weights
- Stress & Scenarios: equity shocks, rates (duration), correlation bumps, historical window replay
- Credit: Expected Loss (batch)
- Risk Budgeting: Equal Risk Contribution (ERC)

**Live Demo:** <https://integrated-risk-app.onrender.com/>

## ⚡ Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```
Demo CSVs are included in data/ so you can click around immediately.

## 📊 How it works

VaR = loss threshold at confidence level α.

ES = average loss beyond VaR.

Kupiec test = checks if exceedances match expectation.

FHS (GARCH-lite) = filters volatility for better tail calibration.

ERC weights = equalize each asset’s risk contribution.

## ⚠️ Disclaimer
For educational use only. Not investment advice.



