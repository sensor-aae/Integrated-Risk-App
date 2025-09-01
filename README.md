# Integrated Risk App (Python)

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
        
