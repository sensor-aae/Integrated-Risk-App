from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import numpy as np
from risk_engine.data import load_prices, to_returns
from risk_engine.market import (
    var_parametric, es_parametric,
    var_historical, es_historical,
    backtest_var_historical,
    var_es_monte_carlo,
    mc_portfolio_loss_from_mu_cov
)
import plotly.graph_objects as go
from risk_engine.stress import (
    apply_single_name_shocks, scale_covariance, historical_window_mu_cov
)




st.set_page_config(page_title="Risk App", layout="centered")
st.title("Minimal Risk App")

# ---- Sidebar: data + config ----
st.sidebar.header("1) Data")
file = st.file_uploader("Upload a CSV (Date + Column of prices)" , type=["csv"])
date_col = st.sidebar.text_input("Date column (optional)", value="")
ret_method = st.sidebar.selectbox("Return type", ["log", "simple"], index=0)

st.sidebar.header("2) Portfolio")
weights_mode = st.sidebar.selectbox("Weights", ["Equal", "Manual by column name"])
alpha = st.slider("Confidence (alpha)", 0.8, 0.999, 0.95 , 0.05)
horizon = st.number_input("Horizon (days)" , 1,30 ,1)
exposure = st.number_input("Exposure ", 0.0 , 1e12,1_000_000.0, step = 1000.0)

st.sidebar.header("3) Method")
method_choice = st.sidebar.radio("Method", ["Historical", "Parametric (Normal)", "Monte Carlo"], index=0)

if method_choice == "Monte Carlo":
    n_sims = st.sidebar.number_input("MC simulations", min_value=5_000, value=50_000, step=5_000)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    shrink = st.sidebar.slider("Covariance shrinkage (λ)", 0.0, 0.2, 0.01, 0.005)


st.sidebar.header("4) Backtest")
bt_window = st.sidebar.number_input("Rolling window (days)", min_value=50, value=250, step=10)


if not file:
    st.info("Upload a CSV to compute VaR.")
    st.stop()

prices = load_prices(file, data_col= date_col if date_col else None)
returns = to_returns(prices, method=("log" if ret_method == "log" else "simple"))

st.subheader("Data preview")
st.dataframe(prices.tail())


# ---- Weights handling ----
if weights_mode == "Equal":
    weights = np.ones(len(returns.columns)) / len(returns.columns)
else:
    st.write("Enter weights like: TICKER=WEIGHT, TICKER=WEIGHT")
    manual = st.text_input("Example: AAPL=0.5, MSFT=0.3, TLT=0.2", value="")
    wdict = {}
    if manual.strip():
        for p in manual.split(","):
            p = p.strip()
            if "=" in p:
                k, v = p.split("=")
                try:
                    wdict[k.strip()] = float(v)
                except:
                    pass
    # align to returns columns
    weights = np.array([wdict.get(c, 0.0) for c in returns.columns], dtype=float)
    s = weights.sum()
    if s != 0:
        weights = weights / s

# ---- Point estimates (VaR/ES) ----
st.subheader("Point Risk Measures")
if method_choice == "Historical":
    var_val = var_historical(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
    es_val  = es_historical(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
elif method_choice == "Parametric (Normal)":
    var_val = var_parametric(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
    es_val  = es_parametric(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
else:  # Monte Carlo
    var_val, es_val = var_es_monte_carlo(
        returns, weights, alpha=alpha, horizon_days=horizon,
        exposure=exposure, n_sims=int(n_sims), seed=int(seed), shrink_lambda=float(shrink)
    )
    
if method_choice != "Monte Carlo":
    st.subheader("Backtest — Rolling Historical VaR (1d)")
    bt = backtest_var_historical(returns, weights, alpha=alpha, window=int(bt_window))
    # ... (existing chart + stats)


c1, c2 = st.columns(2)
c1.metric(f"VaR @ {int(alpha*100)}%, {horizon}d", f"{var_val:,.0f}")
c2.metric("Expected Shortfall (ES)", f"{es_val:,.0f}")

st.markdown("---")
st.header("Stress Testing")

tabs = st.tabs(["Single‑name shock", "Covariance scaling", "Historical window replay"])

# ========= Tab 1: Single-name shock =========
with tabs[0]:
    st.write("Inject one-day shocks to chosen tickers (in return space).")
    tickers = list(returns.columns)
    picked = st.multiselect("Choose tickers to shock", options=tickers, default=tickers[:1])

    shock_pairs = {}
    cols = st.columns(min(3, max(1, len(picked))))
    for i, t in enumerate(picked):
        with cols[i % len(cols)]:
            shock = st.number_input(f"{t} shock (e.g., -0.10 for -10%)", value=0.0, step=0.01, format="%.4f")
            shock_pairs[t] = shock

    if st.button("Run single‑name shock"):
        # Build a one-row return vector with shocks, then dot with weights
        shock_row = apply_single_name_shocks(returns, shock_pairs)
        # portfolio loss = -ret * exposure
        port_ret = float(shock_row.values @ weights)
        loss = -port_ret * exposure
        st.metric("Shocked one‑day loss", f"{loss:,.0f}")
        st.caption("This is a deterministic one‑day loss from specified instantaneous return shocks.")

# ========= Tab 2: Covariance scaling =========
with tabs[1]:
    st.write("Scale market covariance (volatility) and re-compute MC VaR/ES.")
    scale = st.slider("Covariance scale (×)", 0.5, 3.0, 1.5, 0.1)
    sims = st.number_input("MC simulations", min_value=10_000, value=50_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    if st.button("Run covariance scaling stress"):
        mu = returns.mean().values * horizon
        cov = returns.cov().values * horizon
        cov_s = scale_covariance(cov, scale)
        var_s, es_s = mc_portfolio_loss_from_mu_cov(mu, cov_s, weights, alpha=alpha, exposure=exposure, n_sims=int(sims), seed=int(seed))
        c1, c2 = st.columns(2)
        c1.metric(f"VaR stressed (×{scale:.1f})", f"{var_s:,.0f}")
        c2.metric("ES stressed", f"{es_s:,.0f}")

# ========= Tab 3: Historical window replay =========
with tabs[2]:
    st.write("Replay a historical period’s mean/covariance for your current portfolio (MC VaR/ES).")
    st.caption("Pick dates in your data’s range; we’ll compute μ/Σ on that window and simulate the horizon.")
    min_d, max_d = returns.index.min().date(), returns.index.max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("Window start", min_d, min_value=min_d, max_value=max_d)
    end   = c2.date_input("Window end", max_d, min_value=min_d, max_value=max_d)
    sims2 = st.number_input("MC simulations", min_value=10_000, value=50_000, step=10_000, key="sims_hist")
    seed2 = st.number_input("Random seed", min_value=0, value=11, step=1, key="seed_hist")

    if st.button("Run historical replay"):
        try:
            mu_win, cov_win = historical_window_mu_cov(returns, str(start), str(end))
            # Scale μ/Σ to chosen horizon
            mu_h = mu_win * horizon
            cov_h = cov_win * horizon
            var_h, es_h = mc_portfolio_loss_from_mu_cov(mu_h, cov_h, weights, alpha=alpha, exposure=exposure, n_sims=int(sims2), seed=int(seed2))
            c1, c2 = st.columns(2)
            c1.metric("VaR (historical window)", f"{var_h:,.0f}")
            c2.metric("ES (historical window)", f"{es_h:,.0f}")
        except Exception as e:
            st.error(f"Error: {e}")


st.caption("VaR/ES are reported as **loss amounts** (positive numbers).")

# ---- Backtest (Historical VaR, 1-day horizon) ----
st.subheader("Backtest — Rolling Historical VaR (1d)")
bt = backtest_var_historical(returns, weights, alpha=alpha, window=int(bt_window))

left, right = st.columns([2,1])

with left:
    fig = go.Figure()
    # Portfolio returns line
    fig.add_trace(go.Scatter(
        x=bt["r_p"].index, y=bt["r_p"].values,
        mode="lines", name="Portfolio returns"
    ))
    # VaR threshold line (this is a negative return threshold)
    fig.add_trace(go.Scatter(
        x=bt["VaR_threshold"].index, y=bt["VaR_threshold"].values,
        mode="lines", name=f"VaR threshold ({int(alpha*100)}%)"
    ))
    # Exceptions as markers
    exc_mask = (bt["exceptions"] == 1) & bt["VaR_threshold"].notna()
    fig.add_trace(go.Scatter(
        x=bt["r_p"].index[exc_mask], y=bt["r_p"].values[exc_mask],
        mode="markers", name="Exceptions"
    ))
    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Return")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("**Kupiec POF (1‑day horizon)**")
    st.write(f"Window (days): **{bt['window']}**")
    st.write(f"Out‑of‑sample points (T): **{bt['T']}**")
    st.write(f"Exceedances (x): **{bt['exceedances']}**")
    st.write(f"Hit rate (x/T): **{bt['hit_rate']:.4f}**")
    st.write(f"LR statistic: **{bt['kupiec_LR']:.3f}**")
    st.write(f"p‑value: **{bt['kupiec_pvalue']:.4f}**")
    st.caption("H₀: Actual exception rate equals (1 − α). Large LR / small p rejects H₀.")
