from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import numpy as np
from risk_engine.data import load_prices, to_returns
from risk_engine.market import var_parametric

st.set_page_config(page_title="Risk App", layout="centered")
st.title("Minimal Risk App")

file = st.file_uploader("Upload a CSV (Date + Column of prices)" , type=["csv"])
alpha = st.slider("Confidence (alpha)", 0.8, 0.999, 0.95 , 0.05)
horizon = st.number_input("Horizon (days)" , 1,30 ,1)
exposure = st.number_input("Exposure ", 0.0 , 1e12,1_000_000.0, step = 1000.0)

if not file:
    st.info("Upload a CSV to compute VaR.")
    st.stop()

try:
    prices = load_prices(file)
    st.write("Preview:" , prices.head())
    rets = to_returns(prices , method ="log")
    st.write("Returns preview:", rets.head())
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
    
# Equal weights to start 
w = np.ones(len(rets.columns)) / len(rets.columns)

from risk_engine.market import var_parametric
    
var_val = var_parametric(rets, w, alpha = alpha, horizon_days = int(horizon) , exposure= exposure)
st.metric(f"Parametric VaR @ {int(alpha*100)}%,{horizon}d",f"{var_val:,.0f}")

    