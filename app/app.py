from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import numpy as np
from risk_engine.data import load_prices, to_returns
from risk_engine.market import (
    var_parametric, es_parametric,
    var_historical, es_historical,
    backtest_var_historical
)
import plotly.graph_objects as go


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
method_choice = st.sidebar.radio("Method", ["Historical", "Parametric (Normal)"], index=0)

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
else:
    var_val = var_parametric(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
    es_val  = es_parametric(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)

c1, c2 = st.columns(2)
c1.metric(f"VaR @ {int(alpha*100)}%, {horizon}d", f"{var_val:,.0f}")
c2.metric("Expected Shortfall (ES)", f"{es_val:,.0f}")

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
