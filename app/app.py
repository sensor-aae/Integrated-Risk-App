# app/app.py
"""
UI RULE:
- This file must not compute VaR/ES/EL directly.
- It may only call risklib models/backtests and visualize the outputs.
- If risk math is needed, implement it in risklib/ and import it here.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from risk_engine.data import load_prices, to_returns
from risk_engine.market import (
    var_parametric, es_parametric,
    var_historical, es_historical,
    backtest_var_historical,
    var_es_monte_carlo,
    mc_portfolio_loss_from_mu_cov,
    fhs_var_es_next, backtest_fhs_var,
    incremental_var_normal, var_parametric_normal_parts,
    erc_weights
)
from risk_engine.stress import (
    apply_single_name_shocks, scale_covariance, historical_window_mu_cov
)
from risk_engine.credit import compute_el_table, summarize_el
from risk_engine.scenarios import scenario_equities_shock, scenario_rates_bp, scenario_corr_bump_mc
from risklib.market.backtest import backtest_var_historical



@st.cache_data(show_spinner=False)
def _cov_and_mu(returns: pd.DataFrame, horizon: int):
    mu = returns.mean().values * horizon
    cov = returns.cov().values * horizon
    return mu, cov

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Risk App", layout="wide")
st.title("Integrated Risk App (Python)")

# ---------------- SIDEBAR (unchanged controls) ----------------
st.sidebar.header("1) Portfolio")
ret_method = st.sidebar.selectbox("Return type", ["log", "simple"], index=0)
weights_mode = st.sidebar.selectbox("Weights", ["Equal", "Manual by column name"])
alpha = st.sidebar.slider("Confidence (α)", 0.80, 0.999, 0.95, 0.001)
horizon = st.sidebar.number_input("Horizon (days)", 1, 30, 1)
exposure = st.sidebar.number_input("Exposure", 0.0, 1e12, 1_000_000.0, step=1000.0)

st.sidebar.header("2) Method")
method_choice = st.sidebar.radio(
    "Method",
    ["Historical", "Parametric (Normal)", "Monte Carlo", "Filtered Historical (GARCH-lite)"],
    index=0,
    key="method_choice"
)

# Method-specific controls (used only when market data is present)
n_sims = seed = shrink = None
alpha_g = beta_g = None
if method_choice == "Monte Carlo":
    n_sims = st.sidebar.number_input("MC simulations", min_value=5_000, value=50_000, step=5_000)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    shrink = st.sidebar.slider("Covariance shrinkage (λ)", 0.0, 0.2, 0.01, 0.005)
if method_choice == "Filtered Historical (GARCH-lite)":
    alpha_g = st.sidebar.number_input("GARCH α (ARCH)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    beta_g  = st.sidebar.number_input("GARCH β (GARCH)", min_value=0.0, max_value=0.999, value=0.94, step=0.01)

st.sidebar.header("3) Backtest")
bt_window = st.sidebar.number_input("Rolling window (days)", min_value=50, value=250, step=10)

# ---------------- MARKET DATA INPUT (moved from sidebar) ----------------
st.markdown("### Market data")
st.caption("Upload a prices CSV (Date column + one column per ticker). This powers the market risk, backtests, analytics, scenarios, ERC, and report features.")
file = st.file_uploader("Upload prices CSV", type=["csv"], key="prices_csv_main")
date_col = st.text_input("Date column name (optional)", value="")

# non-blocking load
has_market = False
prices = None
returns = None
weights = None

if file is not None:
    try:
        prices = load_prices(file, data_col=(date_col or None))
        returns = to_returns(prices, method=("log" if ret_method == "log" else "simple"))
        has_market = True
    except Exception as e:
        st.error(f"Error loading market data: {e}")
else:
    st.info("No market CSV uploaded yet. You can still use the **Credit — Expected Loss (Batch)** section below.")

# ---------------- MARKET: DATA PREVIEW + WEIGHTS ----------------
if has_market:
    st.subheader("Market data preview")
    st.dataframe(prices.tail())

    # Weights
    if weights_mode == "Equal":
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    else:
        st.write("Enter weights like: TICKER=WEIGHT, TICKER=WEIGHT")
        manual = st.text_input("Example: AAPL=1.0, MSFT=0.0, TLT=0.0", value="")
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
        weights = np.array([wdict.get(c, 0.0) for c in returns.columns], dtype=float)
        s = weights.sum()
        if s != 0:
            weights = weights / s

# ---------------- MARKET: POINT MEASURES ----------------
if has_market and returns is not None and not returns.empty:
    st.subheader("Point Risk Measures")

    method_map = {
        "Historical": "historical",
        "Parametric (Normal)": "parametric",
        "Monte Carlo": "monte_carlo",
        "GARCH-lite": "fhs",
    }

    cfg = MarketRiskConfig(
        alpha=alpha,
        method=method_map.get(method_choice, "historical"),
        horizon_days=horizon,
        exposure=exposure,
    )

    model = MarketRiskModel(returns, weights, cfg)
    model.fit()

    var_val = model.compute_var()
    es_val = model.compute_es()

    c1, c2 = st.columns(2)
    c1.metric(f"VaR @ {int(alpha*100)}%, {horizon}d", f"{var_val:,.0f}")
    c2.metric("Expected Shortfall (ES)", f"{es_val:,.0f}")
    st.caption("VaR/ES are reported as loss amounts (positive numbers).")

else:
    st.info("Load market data to compute risk measures.")

# ---------------- MARKET: BACKTEST (Historical VaR) ----------------
if has_market and returns is not None and not returns.empty:
    st.subheader("Backtest — Rolling Historical VaR (1d)")

    available = len(returns)
    if available <= bt_window:
        st.warning(
            f"Not enough data for backtest: have {available} return rows, window is {bt_window}."
        )
    else:
        bt = backtest_var_historical(
            returns=returns,
            weights=weights,
            alpha=alpha,
            window=int(bt_window)
        )

        left, right = st.columns([2, 1])

        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bt["r_p"].index, y=bt["r_p"].values,
                mode="lines", name="Portfolio returns"
            ))
            fig.add_trace(go.Scatter(
                x=bt["VaR_threshold"].index, y=bt["VaR_threshold"].values,
                mode="lines", name=f"VaR threshold ({int(alpha*100)}%)"
            ))

            exc_mask = (bt["exceptions"] == 1) & bt["VaR_threshold"].notna()
            fig.add_trace(go.Scatter(
                x=bt["r_p"].index[exc_mask],
                y=bt["r_p"].values[exc_mask],
                mode="markers", name="Exceptions"
            ))

            fig.update_layout(
                height=420,
                xaxis_title="Date",
                yaxis_title="Return"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("**Kupiec POF (1-day)**")
            st.write(f"Window: **{bt['window']}**")
            st.write(f"OOS points (T): **{bt['T']}**")
            st.write(f"Exceedances (x): **{bt['exceedances']}**")
            st.write(f"Hit rate (x/T): **{bt['hit_rate']:.4f}**")
            st.write(f"LR statistic: **{bt['kupiec_LR']:.3f}**")
            st.write(f"p-value: **{bt['kupiec_pvalue']:.4f}**")
            st.caption("H₀: Actual exception rate equals (1 − α).")

        ex_df = pd.DataFrame({
            "date": bt["r_p"].index,
            "return": bt["r_p"].values,
            "VaR_threshold": bt["VaR_threshold"].values,
            "exception": bt["exceptions"].values
        })

        st.download_button(
            "Download backtest series (CSV)",
            data=ex_df.to_csv(index=False).encode("utf-8"),
            file_name=f"backtest_{int(alpha*100)}pct_window{bt_window}.csv",
            mime="text/csv"
        )

# ---------------- MARKET: STRESS TESTING ----------------
if has_market:
    st.markdown("---")
    st.header("Stress Testing")

    tabs = st.tabs(["Single-name shock", "Covariance scaling", "Historical window replay"])

    with tabs[0]:
        st.write("Inject one-day shocks to chosen tickers (in return space).")
        tickers = list(returns.columns)
        picked = st.multiselect("Choose tickers to shock", options=tickers, default=tickers[:1])
        shock_pairs = {}
        cols = st.columns(min(3, max(1, len(picked))))
        for i, t in enumerate(picked):
            with cols[i % len(cols)]:
                shock = st.number_input(f"{t} shock (e.g., -0.10 = -10%)", value=0.0, step=0.01, format="%.4f")
                shock_pairs[t] = shock
        if st.button("Run single-name shock"):
            shock_row = apply_single_name_shocks(returns, shock_pairs)
            port_ret = float(shock_row.values @ weights)
            loss = -port_ret * exposure
            st.metric("Shocked one-day loss", f"{loss:,.0f}")

    with tabs[1]:
        st.write("Scale market covariance (volatility) and re-compute MC VaR/ES.")
        scale = st.slider("Covariance scale (×)", 0.5, 3.0, 1.5, 0.1)
        sims = st.number_input("MC simulations", min_value=10_000, value=50_000, step=10_000)
        seed2 = st.number_input("Random seed", min_value=0, value=7, step=1)
        if st.button("Run covariance scaling stress"):
            mu = returns.mean().values * horizon
            cov = returns.cov().values * horizon
            cov_s = scale_covariance(cov, scale)
            var_s, es_s = mc_portfolio_loss_from_mu_cov(mu, cov_s, weights, alpha=alpha,
                                                        exposure=exposure, n_sims=int(sims), seed=int(seed2))
            c1, c2 = st.columns(2)
            c1.metric(f"VaR stressed (×{scale:.1f})", f"{var_s:,.0f}")
            c2.metric("ES stressed", f"{es_s:,.0f}")

    with tabs[2]:
        st.write("Replay a historical period’s mean/covariance for your current portfolio (MC VaR/ES).")
        st.caption("Choose dates within your uploaded data.")
        min_d, max_d = returns.index.min().date(), returns.index.max().date()
        c1, c2 = st.columns(2)
        start = c1.date_input("Window start", min_d, min_value=min_d, max_value=max_d)
        end   = c2.date_input("Window end", max_d, min_value=min_d, max_value=max_d)
        sims2 = st.number_input("MC simulations", min_value=10_000, value=50_000, step=10_000, key="sims_hist")
        seed_hist = st.number_input("Random seed", min_value=0, value=11, step=1, key="seed_hist")
        if st.button("Run historical replay"):
            try:
                mu_win, cov_win = historical_window_mu_cov(returns, str(start), str(end))
                mu_h, cov_h = mu_win * horizon, cov_win * horizon
                var_h, es_h = mc_portfolio_loss_from_mu_cov(mu_h, cov_h, weights, alpha=alpha,
                                                            exposure=exposure, n_sims=int(sims2), seed=int(seed_hist))
                c1, c2 = st.columns(2)
                c1.metric("VaR (historical window)", f"{var_h:,.0f}")
                c2.metric("ES (historical window)", f"{es_h:,.0f}")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- CREDIT EL (ALWAYS AVAILABLE; below market uploader) ----------------
st.markdown("---")
st.header("Credit — Expected Loss (Batch)")
st.write("Upload a CSV with columns for **PD**, **LGD**, **EAD** (case-insensitive). Optional grouping columns (Segment/Rating/etc.) are auto-detected.")
credit_file = st.file_uploader("Upload exposures CSV", type=["csv"], key="credit_csv")

with st.expander("Scenario shocks"):
    c1, c2, c3 = st.columns(3)
    with c1:
        pd_mult = st.number_input("PD multiplier (×)", min_value=0.0, value=1.0, step=0.05)
        pd_add_bps = st.number_input("PD additive (basis points)", min_value=-5000, max_value=5000, value=0, step=25)
    with c2:
        lgd_mult = st.number_input("LGD multiplier (×)", min_value=0.0, value=1.0, step=0.05)
        lgd_add_pct = st.number_input("LGD additive (percentage points)", min_value=-100, max_value=100, value=0, step=1)
    with c3:
        ead_mult = st.number_input("EAD multiplier (×)", min_value=0.0, value=1.0, step=0.05)

if credit_file is not None:
    try:
        cdf = pd.read_csv(credit_file)
        df_el, seg_col = compute_el_table(
            cdf, pd_mult=float(pd_mult), pd_add_bps=float(pd_add_bps),
            lgd_mult=float(lgd_mult), lgd_add_pct=float(lgd_add_pct),
            ead_mult=float(ead_mult),
        )
        st.subheader("Per-facility results")
        st.dataframe(df_el)

        grp, totals = summarize_el(df_el, seg_col)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total EAD", f"{totals['total_EAD']:,.0f}")
        c2.metric("Total EL", f"{totals['total_EL']:,.0f}")
        c3.metric("EL / EAD (avg)", f"{totals['EL_pct_of_EAD']*100:,.2f}%")

        if not grp.empty:
            st.subheader(f"Grouped summary by **{seg_col}**")
            st.dataframe(grp)

        st.download_button(
            "Download detailed EL (CSV)",
            data=df_el.to_csv(index=False).encode("utf-8"),
            file_name="credit_el_detailed.csv",
            mime="text/csv",
        )
        if not grp.empty:
            st.download_button(
                "Download grouped summary (CSV)",
                data=grp.to_csv(index=False).encode("utf-8"),
                file_name="credit_el_grouped.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing credit file: {e}")
else:
    st.info("For credit EL, upload an exposures CSV (PD, LGD, EAD). This does not require market data.")

# ---------------- CALIBRATION (MARKET) ----------------
if has_market:
    st.markdown("---")
    st.header("Calibration")

    import math
    def _kupiec_pval(x: int, T: int, alpha_: float) -> float:
        if T <= 0:
            return float("nan")
        p = 1 - alpha_
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        pi_hat = min(max(x / T, eps), 1 - eps)
        ll0 = (T - x) * np.log(1 - p) + x * np.log(p)
        ll1 = (T - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
        LR = -2 * (ll0 - ll1)
        return float(math.erfc(math.sqrt(LR / 2.0)))

    with st.expander("Run multi-alpha calibration"):
        alphas_to_test = st.multiselect(
            "Confidence levels",
            options=[0.95, 0.975, 0.99, 0.995],
            default=[0.95, 0.975, 0.99, 0.995]
        )
        window_cal = st.number_input("Backtest window (days)", min_value=50, value=int(bt_window), step=10, key="calib_bt_window")
        include_fhs = st.checkbox("Include GARCH-lite (FHS) backtest comparison",
                                  value=(method_choice == "Filtered Historical (GARCH-lite)"))

        if st.button("Run calibration"):
            rows = []
            for a in alphas_to_test:
                if len(returns) > window_cal:
                    bt_hist = backtest_var_historical(returns, weights, alpha=a, window=int(window_cal))
                    T_h = int(bt_hist["T"]); x_h = int(bt_hist["exceedances"])
                    hit_h = float(bt_hist["hit_rate"]) if T_h > 0 else float("nan")
                    p_h = _kupiec_pval(x_h, T_h, a) if T_h > 0 else float("nan")
                else:
                    T_h = x_h = float("nan"); hit_h = p_h = float("nan")

                row = {
                    "alpha": a, "expected_exceed_%": (1 - a) * 100.0,
                    "Hist_T": T_h, "Hist_exceed": x_h,
                    "Hist_hit_%": hit_h * 100.0 if not np.isnan(hit_h) else np.nan,
                    "Hist_Kupiec_p": p_h,
                }

                if include_fhs:
                    bt_fhs = backtest_fhs_var(
                        returns, weights, alpha=a, window_min=int(window_cal),
                        alpha_g=float(alpha_g if alpha_g is not None else 0.05),
                        beta_g=float(beta_g if beta_g is not None else 0.94)
                    )
                    T_f = int(bt_fhs["T"]); x_f = int(bt_fhs["exceedances"])
                    hit_f = float(bt_fhs["hit_rate"]) if T_f > 0 else float("nan")
                    p_f = _kupiec_pval(x_f, T_f, a) if T_f > 0 else float("nan")
                    row.update({
                        "FHS_T": T_f, "FHS_exceed": x_f,
                        "FHS_hit_%": hit_f * 100.0 if not np.isnan(hit_f) else np.nan,
                        "FHS_Kupiec_p": p_f,
                    })

                rows.append(row)

            calib_df = pd.DataFrame(rows)
            st.subheader("Calibration table")
            st.dataframe(calib_df)

            try:
                figc = go.Figure()
                figc.add_trace(go.Bar(
                    x=[f"{int(a*100)}%" for a in calib_df["alpha"]],
                    y=calib_df["expected_exceed_%"], name="Expected exceed %"
                ))
                figc.add_trace(go.Bar(
                    x=[f"{int(a*100)}%" for a in calib_df["alpha"]],
                    y=calib_df["Hist_hit_%"], name="Historical hit %"
                ))
                if "FHS_hit_%" in calib_df.columns:
                    figc.add_trace(go.Bar(
                        x=[f"{int(a*100)}%" for a in calib_df["alpha"]],
                        y=calib_df["FHS_hit_%"], name="FHS hit %"
                    ))
                figc.update_layout(barmode="group", yaxis_title="Percent", xaxis_title="Alpha", height=360)
                st.plotly_chart(figc, use_container_width=True)
            except Exception:
                pass

            st.download_button(
                "Download calibration CSV",
                data=calib_df.to_csv(index=False).encode("utf-8"),
                file_name=f"calibration_window{int(window_cal)}.csv",
                mime="text/csv",
            )

# ---------------- MARKET: ANALYTICS ----------------
if has_market:
    st.markdown("---")
    st.header("Analytics")

    tabs_a = st.tabs(["Correlation heatmap", "VaR decomposition (Normal)"])

    with tabs_a[0]:
        st.write("Correlation matrix of asset returns (last 250 days by default).")
        lookback = st.number_input("Lookback (days)", min_value=50, value=250, step=10, key="corr_lookback")
        r_slice = returns.tail(int(lookback)) if len(returns) >= lookback else returns
        corr = r_slice.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlation heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tabs_a[1]:
        st.write("Component & marginal VaR under a Normal approximation (Euler allocation).")
        parts = var_parametric_normal_parts(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
        tickers = list(returns.columns)
        df_parts = pd.DataFrame({
            "Ticker": tickers,
            "Weight": (weights / weights.sum()) if weights.sum() != 0 else weights,
            "mVaR": parts["mVaR"], "cVaR": parts["cVaR"],
            "Percent of VaR": parts["pContrib"]
        })
        df_parts["Percent of VaR"] = (df_parts["Percent of VaR"] * 100).round(2)
        st.dataframe(df_parts)

        try:
            fig = px.bar(df_parts, x="Ticker", y="cVaR", title="Component VaR (money)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio VaR (Normal)", f"{parts['VaR']:,.0f}")
        c2.metric("μ (portfolio, horizon)", f"{parts['mu_p']:.6f}")
        c3.metric("σ (portfolio, horizon)", f"{parts['sigma_p']:.6f}")
        st.caption("Component VaR sums (≈) to portfolio VaR. Marginal VaR is the sensitivity to a small weight increase.")

# ---------------- MARKET: WHAT-IF WEIGHTS ----------------
if has_market:
    st.markdown("---")
    st.header("What-if: tweak weights")

    tickers = list(returns.columns)
    cols = st.columns(min(4, max(2, len(tickers))))
    new_w = []
    for i, t in enumerate(tickers):
        with cols[i % len(cols)]:
            v = st.slider(f"{t} weight", min_value=0.0, max_value=1.0,
                          value=float(weights[i]) if weights.sum() > 0 else 0.0,
                          step=0.01, key=f"w_{t}")
            new_w.append(v)
    new_w = np.array(new_w, dtype=float)
    if new_w.sum() > 0:
        new_w = new_w / new_w.sum()
    else:
        st.warning("All weights are zero; cannot compute.")
        new_w = weights

    curr = var_parametric_normal_parts(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
    what = var_parametric_normal_parts(returns, new_w,   alpha=alpha, horizon_days=horizon, exposure=exposure)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current VaR (Normal)", f"{curr['VaR']:,.0f}")
    c2.metric("What-if VaR (Normal)", f"{what['VaR']:,.0f}")
    c3.metric("Δ VaR", f"{(what['VaR'] - curr['VaR']):,.0f}")

    df_curr = pd.DataFrame({
        "Ticker": tickers,
        "Weight": (weights / weights.sum()) if weights.sum() != 0 else weights,
        "cVaR (curr)": curr["cVaR"],
        "%VaR (curr)": (curr["pContrib"] * 100.0),
    })
    df_what = pd.DataFrame({
        "Ticker": tickers,
        "Weight (what-if)": new_w,
        "cVaR (what-if)": what["cVaR"],
        "%VaR (what-if)": (what["pContrib"] * 100.0),
    })
    df_join = df_curr.merge(df_what, on="Ticker")
    st.dataframe(df_join)

    inc = incremental_var_normal(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
    df_inc = pd.DataFrame({"Ticker": tickers, "iVaR ≈ cVaR (curr)": inc["cVaR"]})
    st.caption("Incremental VaR ≈ component VaR under the Normal Euler allocation.")
    st.dataframe(df_inc)

    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="cVaR (curr)", x=tickers, y=df_join["cVaR (curr)"]))
        fig.add_trace(go.Bar(name="cVaR (what-if)", x=tickers, y=df_join["cVaR (what-if)"]))
        fig.update_layout(barmode="group", title="Component VaR (money)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ---------------- MARKET: RISK BUDGETING (ERC) ----------------
if has_market:
    st.markdown("---")
    st.header("Risk budgeting — Equal Risk Contribution (ERC)")

    with st.expander("Compute ERC weights"):
        erc_min = st.number_input("Min weight", 0.0, 1.0, 0.0, 0.01)
        erc_max = st.number_input("Max weight", 0.0, 1.0, 1.0, 0.01)
        erc_step = st.slider("Update damping (step)", 0.1, 1.0, 0.5, 0.1)
        erc_tol = st.number_input("Tolerance", 1e-12, 1e-2, 1e-8, format="%.1e")
        init_from_current = st.checkbox("Initialize from current weights", value=True)

        if st.button("Run ERC"):
            init = weights if init_from_current and weights.sum() > 0 else None
            w_erc, info = erc_weights(returns, horizon_days=horizon, init=init,
                                      min_w=float(erc_min), max_w=float(erc_max),
                                      step=float(erc_step), tol=float(erc_tol))
            parts_curr = var_parametric_normal_parts(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
            parts_erc  = var_parametric_normal_parts(returns, w_erc,   alpha=alpha, horizon_days=horizon, exposure=exposure)

            tickers = list(returns.columns)
            df_erc = pd.DataFrame({
                "Ticker": tickers,
                "Weight (current)": (weights / weights.sum()) if weights.sum() != 0 else weights,
                "Weight (ERC)": w_erc,
                "%VaR (current)": (parts_curr["pContrib"] * 100.0),
                "%VaR (ERC)": (parts_erc["pContrib"] * 100.0),
            })
            c1, c2 = st.columns(2)
            c1.metric("Portfolio VaR (current, Normal)", f"{parts_curr['VaR']:,.0f}")
            c2.metric("Portfolio VaR (ERC, Normal)", f"{parts_erc['VaR']:,.0f}")
            st.dataframe(df_erc)

            st.download_button(
                "Download ERC weights (CSV)",
                data=df_erc[["Ticker","Weight (ERC)"]].to_csv(index=False).encode("utf-8"),
                file_name="erc_weights.csv", mime="text/csv"
            )
            st.caption(f"Converged in {info['iter']} iters; RC dispersion={info['rc_dispersion']:.2e}")

# ---------------- MARKET: SCENARIO LIBRARY ----------------
if has_market:
    st.markdown("---")
    st.header("Scenario library")

    tabs_s = st.tabs(["Equities −X%", "Rates +bp (duration)", "Correlations +X% (MC)"])

    with tabs_s[0]:
        eqs = st.multiselect("Equity tickers", options=list(returns.columns),
                             default=[c for c in returns.columns if c not in ["TLT","IEF","AGG","BND","EDV","ZROZ"]])
        eq_shock = st.number_input("Equity shock (return, e.g., -0.05 = -5%)",
                                   min_value=-1.0, max_value=1.0, value=-0.05, step=0.01)
        if st.button("Run equities shock"):
            loss = scenario_equities_shock(returns, weights, eqs, shock=float(eq_shock), exposure=exposure)
            st.metric("Scenario P&L (loss +ve)", f"{loss:,.0f}")

    with tabs_s[1]:
        st.write("Duration approximation: ΔP/P ≈ −D × Δy.")
        bp = st.number_input("Rate move (bp)", min_value=-1000, max_value=1000, value=200, step=25)
        st.caption("Override durations (defaults provided for common ETFs):")
        dur_inputs = {}
        cols = st.columns(min(4, max(2, len(returns.columns))))
        for i, c in enumerate(returns.columns):
            with cols[i % len(cols)]:
                dur_inputs[c] = st.number_input(f"{c} duration", min_value=0.0,
                                                value=18.0 if c == "TLT" else 7.0, step=0.5)
        if st.button("Run rates shock"):
            loss = scenario_rates_bp(returns, weights, durations=dur_inputs, bp=float(bp), exposure=exposure)
            st.metric("Scenario P&L (loss +ve)", f"{loss:,.0f}")

    with tabs_s[2]:
        a_corr = st.number_input("Alpha for VaR/ES", min_value=0.8, max_value=0.999,
                                 value=float(alpha), step=0.001, format="%.3f")
        corr_bump = st.slider("Correlation bump (%)", min_value=0, max_value=200, value=50, step=5)
        sims3 = st.number_input("MC simulations", min_value=10_000, value=50_000, step=10_000, key="sims_corr")
        seed3 = st.number_input("Random seed", min_value=0, value=13, step=1, key="seed_corr")
        if st.button("Run correlation bump"):
            base, stressed = scenario_corr_bump_mc(returns, weights, alpha=float(a_corr), horizon_days=horizon,
                                                   exposure=exposure, corr_bump_pct=float(corr_bump),
                                                   n_sims=int(sims3), seed=int(seed3))
            (var_b, es_b), (var_s, es_s) = base, stressed
            c1, c2 = st.columns(2)
            c1.metric("VaR (base)", f"{var_b:,.0f}")
            c2.metric("VaR (corr bumped)", f"{var_s:,.0f}")
            st.caption(f"ΔVaR = {var_s - var_b:,.0f}; ES base {es_b:,.0f} → stressed {es_s:,.0f}")

# ---------------- REPORT EXPORT (MARKET) ----------------
if has_market:
    st.markdown("---")
    st.header("Report export")

    summary = []
    summary.append("# Risk Report\n")
    summary.append(f"**Method:** {method_choice}\n")
    summary.append(f"**Alpha:** {alpha:.3f}   **Horizon (days):** {horizon}   **Exposure:** {exposure:,.0f}\n")
    summary.append("## Current portfolio\n")
    tickers = list(returns.columns)
    w_norm = (weights / weights.sum()) if weights.sum() != 0 else weights
    summary.append("| Ticker | Weight |")
    summary.append("|---|---:|")
    for t, wv in zip(tickers, w_norm):
        summary.append(f"| {t} | {wv:.4f} |")
    summary.append("\n## Point risk measures")
    summary.append(f"- VaR: **{var_val:,.0f}**")
    summary.append(f"- ES : **{es_val:,.0f}**")

    try:
        summary.append("\n## Backtest (Historical VaR)")
        summary.append(f"- Window: {bt['window']}  |  OOS T: {bt['T']}  |  Exceedances: {bt['exceedances']}  |  Hit rate: {bt['hit_rate']:.4f}  |  Kupiec p: {bt['kupiec_pvalue']:.4f}")
    except Exception:
        pass

    report_md = "\n".join(summary)

    st.download_button(
        "Download report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="risk_report.md",
        mime="text/markdown"
    )

    try:
        parts_now = var_parametric_normal_parts(returns, weights, alpha=alpha, horizon_days=horizon, exposure=exposure)
        df_contrib = pd.DataFrame({
            "Ticker": tickers, "Weight": w_norm,
            "Component VaR": parts_now["cVaR"],
            "Percent of VaR": parts_now["pContrib"]
        })
        st.download_button(
            "Download VaR decomposition (CSV)",
            data=df_contrib.to_csv(index=False).encode("utf-8"),
            file_name="var_decomposition.csv",
            mime="text/csv"
        )
    except Exception:
        pass

st.caption("Tip: PNG exports of charts require `kaleido` (optional).")


